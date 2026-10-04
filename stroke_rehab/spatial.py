"""Causal common spatial patterns (CSP) from streaming covariance statistics.

Continuous causal filtering keeps its state between chunks. An entire trial
is represented by three 16x16 covariance matrices, not buffered EEG epochs.
Labels only enter the fold-local spatial filter fit, never signal extraction.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.signal import sosfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .features import BANDS
from .streaming import CausalPowerStream


@dataclass(frozen=True)
class CovarianceEvent:
    onset_sample: int
    decision_sample: int
    covariance: np.ndarray  # (bands, channels, channels), sample-mean centered.


class CausalCovarianceStream(CausalPowerStream):
    """Accumulate centered covariance for exactly one post-instruction window.

    The active trial uses O(bands * channels^2) state. Before the window ends,
    process returns None, irrespective of chunk boundaries or trigger label.
    """

    def process(self, chunk, *, onsets=()):
        x = np.asarray(chunk, dtype=np.float64)
        if (x.ndim != 2 or x.shape[1] != self.channels or not len(x)
                or not np.isfinite(x).all()):
            raise ValueError("Expected a nonempty finite (samples, channels) EEG chunk")
        beginning, ending = self.sample, self.sample + len(x)
        onsets = tuple(onsets)
        if len(onsets) > 1:
            raise ValueError("At most one cue onset per chunk; use smaller chunks")
        if onsets:
            onset = onsets[0]
            if (not isinstance(onset, (int, np.integer)) or onset < beginning
                    or onset >= ending or onset <= self._last_onset or self._trial is not None):
                raise ValueError("Out-of-order, future, duplicate or overlapping trial onset")
            self._last_onset = int(onset)
            self._trial = dict(onset=int(onset), count=0,
                               mean=np.zeros((len(BANDS), self.channels)),
                               scatter=np.zeros((len(BANDS), self.channels, self.channels)))
        filtered = []
        for band, sos in enumerate(self.sos):
            y, self.zi[band] = sosfilt(sos, x, axis=0, zi=self.zi[band])
            filtered.append(y)
        event = None
        if self._trial is not None:
            trial = self._trial
            left = max(beginning, trial["onset"] + self.start_offset)
            right = min(ending, trial["onset"] + self.stop_offset)
            if right > left:
                values = np.stack([band[left - beginning:right - beginning]
                                   for band in filtered])
                n = values.shape[1]
                chunk_mean = values.mean(axis=1)
                centered = values - chunk_mean[:, None, :]
                chunk_scatter = np.einsum("bti,btj->bij", centered, centered)
                old_n = trial["count"]
                total = old_n + n
                delta = chunk_mean - trial["mean"]
                trial["scatter"] += (chunk_scatter +
                                     np.einsum("bi,bj->bij", delta, delta) *
                                     (old_n * n / total))
                trial["mean"] += delta * (n / total)
                trial["count"] = total
            if ending >= trial["onset"] + self.stop_offset:
                if trial["count"] != self.stop_offset - self.start_offset:
                    raise ValueError("Incomplete trial: cue or EEG samples were skipped")
                # Projected variance is unaffected by the n vs n-1 factor.
                covariance = trial["scatter"] / trial["count"]
                event = CovarianceEvent(trial["onset"], ending, covariance)
                self._trial = None
        self.sample = ending
        return event


def replay_covariances(recording, *, chunk_samples=64, window=(2.5, 3.5)):
    """Replay training EEG; return covariances and EEG-sample decision offsets.

    Triggers must be sorted, unique and observed during their arriving chunk.
    A missing or overlapping trial is an error, not an imputed observation.
    """
    if not isinstance(chunk_samples, int) or chunk_samples < 1:
        raise ValueError("chunk_samples must be a positive integer")
    stream = CausalCovarianceStream(recording.fs, recording.signal.shape[1], window)
    onsets = np.asarray(recording.onsets)
    if (onsets.ndim != 1 or len(onsets) == 0 or np.any(onsets < 0)
            or np.any(onsets[1:] <= onsets[:-1])):
        raise ValueError("Trial onsets must be sorted, unique and nonnegative")
    pointer, events = 0, []
    for left in range(0, len(recording.signal), chunk_samples):
        right = min(left + chunk_samples, len(recording.signal))
        cues = []
        while pointer < len(onsets) and onsets[pointer] < right:
            if onsets[pointer] < left:
                raise ValueError("Late trial onset")
            cues.append(int(onsets[pointer]))
            pointer += 1
        event = stream.process(recording.signal[left:right], onsets=cues)
        if event is not None:
            events.append(event)
    if (pointer != len(onsets) or len(events) != len(onsets)
            or stream._trial is not None
            or [e.onset_sample for e in events] != list(onsets)):
        raise ValueError("Missing or out-of-order trial decision in the replay")
    return np.stack([e.covariance for e in events]), np.array(
        [e.decision_sample - e.onset_sample for e in events])


class CovarianceCSP(BaseEstimator, TransformerMixin):
    """Fold-local ridge CSP trained on trace-normalized causal covariances."""

    def __init__(self, n_pairs=2, ridge=0.01):
        self.n_pairs = n_pairs
        self.ridge = ridge

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if (X.ndim != 4 or X.shape[2] != X.shape[3] or
                X.shape[0] != len(y) or sorted(np.unique(y).tolist()) != [-1, 1] or
                not np.isfinite(X).all()):
            raise ValueError("Expected finite (trial, band, channel, channel) and +/-1 labels")
        channels = X.shape[-1]
        if (not isinstance(self.n_pairs, int) or not 0 < self.n_pairs <= channels // 2
                or not 0 < self.ridge < 1):
            raise ValueError("Invalid CSP regularization or filter count")
        self.filters_ = []
        self.n_bands_ = X.shape[1]
        self.n_channels_ = channels
        for band in range(X.shape[1]):
            cov = X[:, band]
            scale = np.trace(cov, axis1=-2, axis2=-1)
            if np.any(scale <= 1e-20):
                raise ValueError("Zero-variance covariance in training data")
            normalized = cov / scale[:, None, None]
            a = normalized[y == 1].mean(axis=0)
            b = normalized[y == -1].mean(axis=0)
            reg = self.ridge / channels * np.eye(channels)
            _, vectors = eigh(a + reg, a + b + 2 * reg, check_finite=True)
            keep = np.r_[np.arange(self.n_pairs),
                         np.arange(channels - self.n_pairs, channels)]
            self.filters_.append(vectors[:, keep])
        return self

    def transform(self, X):
        check_is_fitted(self, "filters_")
        X = np.asarray(X, dtype=np.float64)
        if (X.ndim != 4 or X.shape[1:] !=
                (self.n_bands_, self.n_channels_, self.n_channels_)
                or not np.isfinite(X).all()):
            raise ValueError("Unexpected covariance shape or nonfinite values")
        output = []
        for band, filters in enumerate(self.filters_):
            variance = np.einsum("cf,ncd,df->nf", filters, X[:, band], filters)
            variance = np.maximum(variance, 1e-30)
            variance /= variance.sum(axis=1, keepdims=True)
            output.append(np.log(variance))
        return np.concatenate(output, axis=1)


@dataclass(frozen=True)
class SpatialDecision:
    onset_sample: int
    decision_sample: int
    label: int  # Prediction, not the known trial label.


class CausalSpatialDecoder:
    """Inference-only, run-local spatial decoder with a fitted sklearn model.

    Construct a *new* decoder per continuous run; the filter starts at zero.
    Incoming EEG and onset samples never include future observations. No
    reference to labels or recording data is stored in this object.
    """

    def __init__(self, fitted_estimator, *, candidate="causal_csp_1pair",
                 fs=256, channels=16, window=(2.5, 3.5)):
        check_is_fitted(fitted_estimator)
        if candidate not in ("causal_power", "causal_csp_1pair", "causal_csp_2pairs"):
            raise ValueError("Unknown candidate")
        self.model = fitted_estimator
        self.candidate = candidate
        self.stream = CausalCovarianceStream(fs, channels, window)

    def process(self, chunk, *, onsets=()):
        event = self.stream.process(chunk, onsets=onsets)
        if event is None:
            return None
        features = event.covariance[None, :]
        if self.candidate == "causal_power":
            power = np.diagonal(features, axis1=-2, axis2=-1)
            features = np.log(np.maximum(power, 1e-12)).reshape(1, -1)
        label = int(self.model.predict(features)[0])
        if label not in (-1, 1):
            raise ValueError("Expected -1 (right) or +1 (left) prediction")
        return SpatialDecision(event.onset_sample, event.decision_sample, label)

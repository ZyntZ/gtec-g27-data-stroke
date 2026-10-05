"""Stateful, causal EEG feature extraction with trial-wise decision timestamps.

The only run-level state is the three Butterworth filter states. Per-trial
statistics contain 48 means/variances; no whole-epoch EEG buffer is retained.
The stream does not know the labels and never examines future signal samples.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt

from .features import BANDS


@dataclass(frozen=True)
class FeatureEvent:
    onset_sample: int
    decision_sample: int  # End of the chunk producing the feature, not a feedback timestamp.
    features: np.ndarray  # band-major log population variances, (3 * channels,).


class CausalPowerStream:
    """Build log-band-variance features from a strictly ordered EEG chunk stream.

    A caller reports a trigger onset only inside the current chunk. Signal
    samples are zero-state filtered continuously, including pre-cue samples.
    A decision becomes available after the configured EEG window has arrived.
    A pre-instruction window is valid for negative controls, not for decoding
    an instructed movement. A chunk contains at most one new cue.
    """

    def __init__(self, fs=256, channels=16, window=(2.5, 3.5)):
        if not isinstance(fs, int) or fs <= 2 * max(hi for _, hi in BANDS):
            raise ValueError("Invalid or insufficient sampling rate")
        if not isinstance(channels, int) or channels < 1:
            raise ValueError("Expected positive channel count")
        start, stop = (round(t * fs) for t in window)
        if not (0 <= start < stop <= 8 * fs):
            raise ValueError("Window must be nonnegative and inside the trial")
        self.fs = fs
        self.channels = channels
        self.start_offset = start
        self.stop_offset = stop
        self.sos = [butter(4, (lo, hi), fs=fs, btype="bandpass", output="sos")
                    for lo, hi in BANDS]
        # Zero initial conditions are appropriate at the beginning of each run.
        self.zi = [np.zeros((len(sos), 2, channels), dtype=np.float64) for sos in self.sos]
        self.sample = 0
        self._trial = None
        self._last_onset = -1

    def process(self, chunk, *, onsets=()):
        """Return zero or one completed trial feature without future-sample access."""
        x = np.asarray(chunk, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.channels or not len(x) or not np.isfinite(x).all():
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
                               m2=np.zeros((len(BANDS), self.channels)))
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
                selected = np.stack([band[left - beginning:right - beginning]
                                     for band in filtered])
                n = selected.shape[1]
                new_mean = selected.mean(axis=1)
                new_m2 = np.sum((selected - new_mean[:, None, :]) ** 2, axis=1)
                old_n = trial["count"]
                total = old_n + n
                delta = new_mean - trial["mean"]
                trial["m2"] += new_m2 + delta ** 2 * (old_n * n / total)
                trial["mean"] += delta * (n / total)
                trial["count"] = total
            if ending >= trial["onset"] + self.stop_offset:
                if trial["count"] != self.stop_offset - self.start_offset:
                    raise ValueError("Incomplete trial: cue or EEG samples were skipped")
                variance = trial["m2"] / trial["count"]
                event = FeatureEvent(trial["onset"], ending,
                                     np.log(np.maximum(variance, 1e-12)).ravel())
                self._trial = None
        self.sample = ending
        return event


def replay_features(recording, *, chunk_samples=64, window=(2.5, 3.5), timing=False):
    """Replay one recording; return features and first-availability latencies.

    Trial onsets are supplied only when their sample enters the current chunk.
    `timing` additionally measures chunk processing wall-clock time; runtime
    figures are machine-specific and should never be mixed with EEG time.
    """
    if not isinstance(chunk_samples, int) or chunk_samples < 1:
        raise ValueError("chunk_samples must be a positive integer")
    from time import perf_counter

    stream = CausalPowerStream(recording.fs, recording.signal.shape[1], window)
    onsets = np.asarray(recording.onsets)
    if (onsets.ndim != 1 or len(onsets) == 0 or np.any(onsets < 0)
            or np.any(onsets[1:] <= onsets[:-1])):
        raise ValueError("Trial onsets must be sorted, unique and nonnegative")
    pointer, events, times = 0, [], []
    for left in range(0, len(recording.signal), chunk_samples):
        right = min(left + chunk_samples, len(recording.signal))
        cues = []
        while pointer < len(onsets) and onsets[pointer] < right:
            if onsets[pointer] < left:
                raise ValueError("Late trial onset")
            cues.append(int(onsets[pointer]))
            pointer += 1
        start = perf_counter() if timing else None
        event = stream.process(recording.signal[left:right], onsets=cues)
        if timing:
            times.append(perf_counter() - start)
        if event is not None:
            events.append(event)
    if pointer != len(onsets) or len(events) != len(onsets) or stream._trial is not None:
        raise ValueError("Missing a trial decision; incomplete EEG stream")
    if [e.onset_sample for e in events] != list(onsets):
        raise ValueError("Trial decisions were not emitted in trigger order")
    return np.stack([e.features for e in events]), np.array(
        [e.decision_sample - e.onset_sample for e in events]), np.asarray(times)


@dataclass(frozen=True)
class DecisionEvent:
    onset_sample: int
    decision_sample: int
    label: int  # Predicted left (+1) or right (-1), never the known trial label.


class CausalTrialDecoder:
    """Inference-only wrapper around an already fitted, training-only estimator.

    Signal and filter state are initialized for every run. Do not pass an
    estimator fit on the held-out recording or reuse this object across runs.
    """

    def __init__(self, fitted_estimator, *, fs=256, channels=16, window=(2.5, 3.5)):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(fitted_estimator)
        self.model = fitted_estimator
        self.stream = CausalPowerStream(fs=fs, channels=channels, window=window)

    def process(self, chunk, *, onsets=()):
        event = self.stream.process(chunk, onsets=onsets)
        if event is None:
            return None
        predicted = int(self.model.predict(event.features[None, :])[0])
        if predicted not in (-1, 1):
            raise ValueError("Decoder must predict -1 (right) or +1 (left)")
        return DecisionEvent(event.onset_sample, event.decision_sample, predicted)


def replay_decisions(recording, fitted_estimator, *, chunk_samples=64, window=(2.5, 3.5)):
    """Simulate EEG arrival; caller must supply a separately fitted estimator.

    Returns decisions and per-chunk processing times; never computes a score
    against labels. An acquisition system must supply real trigger onsets.
    """
    from time import perf_counter

    if not isinstance(chunk_samples, int) or chunk_samples < 1:
        raise ValueError("chunk_samples must be a positive integer")
    decoder = CausalTrialDecoder(fitted_estimator, fs=recording.fs,
                                 channels=recording.signal.shape[1], window=window)
    onsets = np.asarray(recording.onsets)
    if (onsets.ndim != 1 or len(onsets) == 0 or np.any(onsets < 0)
            or np.any(onsets[1:] <= onsets[:-1])):
        raise ValueError("Trial onsets must be sorted, unique and nonnegative")
    pointer, decisions, times = 0, [], []
    for left in range(0, len(recording.signal), chunk_samples):
        right = min(left + chunk_samples, len(recording.signal))
        cues = []
        while pointer < len(onsets) and onsets[pointer] < right:
            cues.append(int(onsets[pointer]))
            pointer += 1
        start = perf_counter()
        event = decoder.process(recording.signal[left:right], onsets=cues)
        times.append(perf_counter() - start)
        if event is not None:
            decisions.append(event)
    if pointer != len(onsets) or len(decisions) != len(onsets):
        raise ValueError("Missing a trial decision in the replay")
    if [event.onset_sample for event in decisions] != list(onsets):
        raise ValueError("Out-of-order decision")
    return decisions, np.asarray(times)

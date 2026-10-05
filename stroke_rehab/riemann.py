"""Filter-bank Riemannian tangent-space decoder with label-free online recentering.

Each trial is summarized by one channel covariance matrix per frequency band.
"Recentering" divides out the run's average covariance, so slow changes in
signal scale between the training and the test run do not move the features.
On new trials the average is updated one trial at a time, without labels.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BANDS = tuple((float(low), float(low + 4)) for low in range(4, 32, 4))
TRIAL_SECONDS = 8


def filtered_trials(recording, bands=BANDS, *, causal=False):
    """Band-pass the continuous run, then cut trials -> trial x band x time x channel.

    `causal=True` uses a one-pass filter, so a sample never contains later EEG.
    Its state starts at the first sample's steady state to avoid a start-up
    transient from large DC offsets.
    """
    fs = recording.fs
    signal = recording.signal
    length = TRIAL_SECONDS * fs
    output = np.empty((len(recording.onsets), len(bands), length, signal.shape[1]))
    for b, band in enumerate(bands):
        sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
        if causal:
            state = sosfilt_zi(sos)[:, :, None] * signal[0]
            filtered, _ = sosfilt(sos, signal, axis=0, zi=state)
        else:
            filtered = sosfiltfilt(sos, signal, axis=0)
        for t, onset in enumerate(recording.onsets):
            output[t, b] = filtered[onset:onset + length]
    return output


def window_covariances(trials, fs, window):
    """Shrunk (OAS) covariance per trial and band -> trial x band x channel x channel."""
    lo, hi = round(window[0] * fs), round(window[1] * fs)
    if not 0 <= lo < hi <= trials.shape[2]:
        raise ValueError("Window outside the eight-second trial")
    return np.array([[oas_from_covariance(np.cov(band[lo:hi].T, bias=True), hi - lo)
                      for band in trial] for trial in trials])


def _apply(matrix, function):
    values, vectors = eigh(matrix)
    return (vectors * function(values)) @ vectors.T


def inverse_sqrt(matrix):
    return _apply(matrix, lambda v: 1.0 / np.sqrt(v))


def geodesic(a, b, step):
    """Point a fraction `step` of the way from a to b on the SPD manifold."""
    root, inverse_root = _apply(a, np.sqrt), inverse_sqrt(a)
    return root @ _apply(inverse_root @ b @ inverse_root, lambda v: v ** step) @ root


def riemann_mean(covariances, *, tolerance=1e-8, max_iterations=50):
    mean = np.mean(covariances, axis=0)
    for _ in range(max_iterations):
        root, inverse_root = _apply(mean, np.sqrt), inverse_sqrt(mean)
        step = np.mean([_apply(inverse_root @ c @ inverse_root, np.log)
                        for c in covariances], axis=0)
        mean = root @ _apply(step, np.exp) @ root
        if np.linalg.norm(step) < tolerance:
            break
    return mean


def tangent_vector(covariance, reference):
    """Whiten by `reference`, take the matrix logarithm, keep the upper triangle."""
    w = inverse_sqrt(reference)
    log = _apply(w @ covariance @ w.T, np.log)
    upper = np.triu_indices(len(log))
    weights = np.where(upper[0] == upper[1], 1.0, np.sqrt(2.0))
    return log[upper] * weights


class RecenteredTangentSpace(BaseEstimator, ClassifierMixin):
    """Input: trial x band x channel x channel covariances, in recording order.

    Each band has a reference covariance. It starts at the training mean
    (weighted as `prior_trials` trials) and moves toward every incoming trial
    before that trial is classified. No labels and no later trials are used.
    `adapt=False` keeps the training mean.

    The adaptation state is explicit: `references_` and `n_seen_`.
    - `predict(X)` is stateless. It scores X as one fresh chronological stream
      from the training reference and leaves the stored state untouched, so
      repeated calls and cross-validation give the same answer.
    - `predict_next(trial)` and `predict_stream(X)` are stateful. They advance
      the stored state, so feeding a run trial by trial, in chunks, or all at
      once gives identical decisions. Call `reset()` before a new run.
    """

    def __init__(self, C=0.01, prior_trials=8, adapt=True):
        self.C = C
        self.prior_trials = prior_trials
        self.adapt = adapt

    def fit(self, X, y):
        X = np.asarray(X)
        if X.ndim != 4 or X.shape[0] != len(y) or len(np.unique(y)) != 2:
            raise ValueError("Expected trial x band x channel x channel and two labels")
        self.classes_ = np.unique(y)
        self.means_ = [riemann_mean(X[:, b]) for b in range(X.shape[1])]
        features = np.array([np.concatenate([tangent_vector(c, m)
                             for c, m in zip(trial, self.means_)]) for trial in X])
        self.model_ = make_pipeline(
            StandardScaler(), LogisticRegression(C=self.C, max_iter=3000)).fit(features, y)
        return self.reset()

    def reset(self):
        """Return the adaptation state to the training reference (start of a run)."""
        self.references_ = [m.copy() for m in self.means_]
        self.n_seen_ = 0
        return self

    def _check(self, X, ndim):
        X = np.asarray(X)
        if X.ndim != ndim or X.shape[-3] != len(self.means_):
            raise ValueError("Covariance input does not match fitted bands")
        return X

    def _step(self, references, n_seen, trial):
        """One trial: updated references, updated count, feature vector."""
        if self.adapt:
            step = 1.0 / (self.prior_trials + n_seen + 1)
            references = [geodesic(m, c, step) for m, c in zip(references, trial)]
        features = np.concatenate([tangent_vector(c, m)
                                   for c, m in zip(trial, references)])
        return references, n_seen + 1, features

    def _fresh_features(self, X):
        references, n_seen, rows = self.means_, 0, []
        for trial in self._check(X, 4):
            references, n_seen, features = self._step(references, n_seen, trial)
            rows.append(features)
        return np.array(rows)

    def predict(self, X):
        return self.model_.predict(self._fresh_features(X))

    def decision_function(self, X):
        return self.model_.decision_function(self._fresh_features(X))

    def _advance(self, trial):
        self.references_, self.n_seen_, features = self._step(
            self.references_, self.n_seen_, self._check(trial, 3))
        return features[None, :]

    def predict_next(self, trial):
        """Classify one band x channel x channel trial and advance the state."""
        return self.model_.predict(self._advance(trial))[0]

    def decision_next(self, trial):
        """Signed score for one trial (positive favours `classes_[1]`); advances the state."""
        return float(self.model_.decision_function(self._advance(trial))[0])

    def predict_stream(self, X):
        """Classify the next trials of the current run, continuing from the stored state."""
        return np.array([self.predict_next(trial) for trial in self._check(X, 4)])


def oas_from_covariance(covariance, n_samples):
    """OAS shrinkage of a mean-centred sample covariance (divided by n_samples).

    Same estimator as `sklearn.covariance.oas`, computed from the covariance
    alone so a stream can keep running sums instead of the EEG samples.
    """
    n_features = len(covariance)
    mu = np.trace(covariance) / n_features
    alpha = np.mean(covariance ** 2)
    denominator = (n_samples + 1.0) * (alpha - mu ** 2 / n_features)
    shrinkage = 1.0 if denominator == 0 else min((alpha + mu ** 2) / denominator, 1.0)
    shrunk = (1.0 - shrinkage) * covariance
    shrunk.flat[::n_features + 1] += shrinkage * mu
    return shrunk

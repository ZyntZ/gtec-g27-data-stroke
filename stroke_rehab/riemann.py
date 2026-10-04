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
from sklearn.covariance import oas
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
    return np.array([[oas(band[lo:hi])[0] for band in trial] for trial in trials])


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

    `predict` treats its input as one chronological stream: each band's
    reference starts at the training mean (weighted as `prior_trials` trials)
    and moves toward every incoming trial before that trial is classified. No
    labels and no later trials are used. `adapt=False` keeps the training mean.
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
        return self

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim != 4 or X.shape[1] != len(self.means_):
            raise ValueError("Covariance input does not match fitted bands")
        references = [m.copy() for m in self.means_]
        features = []
        for i, trial in enumerate(X):
            if self.adapt:
                step = 1.0 / (self.prior_trials + i + 1)
                references = [geodesic(m, c, step) for m, c in zip(references, trial)]
            features.append(np.concatenate([tangent_vector(c, m)
                                            for c, m in zip(trial, references)]))
        return self.model_.predict(np.array(features))

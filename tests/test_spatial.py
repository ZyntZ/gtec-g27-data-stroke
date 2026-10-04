"""Synthetic tests for causal spatial filtering and fold-local CSP."""
import numpy as np
import pytest
from scipy.signal import sosfilt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import NotFittedError

from stroke_rehab.data import Recording
from stroke_rehab.spatial import (
    CausalCovarianceStream, CausalSpatialDecoder, CovarianceCSP, replay_covariances)


def recording():
    rng = np.random.default_rng(33)
    return Recording(rng.normal(size=(2800, 16)), np.array([1], dtype=np.int8),
                     np.array([73]), 256)


def test_streaming_covariance_matches_continuous_causal_filter_and_chunks():
    rec = recording()
    first, latency = replay_covariances(rec, chunk_samples=1)
    for size in (2, 31, 64, 511):
        cov, offsets = replay_covariances(rec, chunk_samples=size)
        np.testing.assert_allclose(first, cov, atol=1e-11, rtol=1e-11)
        assert 3.5 <= offsets[0] / rec.fs < 3.5 + size / rec.fs
    stream = CausalCovarianceStream()
    lo = rec.onsets[0] + stream.start_offset
    hi = rec.onsets[0] + stream.stop_offset
    reference = np.stack([
        np.cov(sosfilt(sos, rec.signal, axis=0)[lo:hi], rowvar=False, bias=True)
        for sos in stream.sos])
    np.testing.assert_allclose(first[0], reference, atol=1e-11, rtol=1e-11)


def test_no_label_or_future_sample_leakage():
    rec = recording()
    X, latency = replay_covariances(rec)
    changed = rec.signal.copy()
    changed[rec.onsets[0] + 896:] *= 1e5
    after, other_latency = replay_covariances(Recording(
        changed, -rec.labels, rec.onsets, rec.fs))
    np.testing.assert_array_equal(X, after)
    np.testing.assert_array_equal(latency, other_latency)


def test_zero_covariance_and_wrong_input_rejected():
    X = np.zeros((6, 3, 4, 4))
    with pytest.raises(ValueError, match="Zero-variance"):
        CovarianceCSP(n_pairs=1).fit(X, [1, 1, 1, -1, -1, -1])
    with pytest.raises(ValueError, match="Invalid CSP"):
        CovarianceCSP(n_pairs=3).fit(np.tile(np.eye(4), (6, 3, 1, 1)),
                                      [1, 1, 1, -1, -1, -1])
    with pytest.raises(NotFittedError):
        CovarianceCSP().transform(np.ones((1, 3, 16, 16)))
    rec = recording()
    with pytest.raises(ValueError, match="Missing"):
        replay_covariances(Recording(rec.signal[:500], rec.labels, rec.onsets, rec.fs))


def test_csp_recovers_known_covariance_axis_and_decoder_timing():
    rng = np.random.default_rng(47)
    n, channels = 40, 16
    X = np.stack([np.diag([10.0 if i % 2 == 0 else 0.1,
                           0.1 if i % 2 == 0 else 10.0] + [1.0] * (channels - 2))
                  + np.diag(rng.uniform(0, 0.08, channels))
                  for i in range(n)])[:, None, :, :]
    labels = np.tile([1, -1], n // 2)
    classifier = make_pipeline(CovarianceCSP(n_pairs=1), StandardScaler(),
                               LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    classifier.fit(X, labels)
    assert np.mean(classifier.predict(X) == labels) == 1.0
    rec = recording()
    cov, _ = replay_covariances(rec)
    classifier = make_pipeline(CovarianceCSP(n_pairs=1), StandardScaler(),
                               LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    augmented = np.concatenate([cov + np.diag(rng.uniform(0, 0.1, 16))[None, None] * (i + 1)
                                for i in range(6)])
    classifier.fit(augmented, np.array([-1, 1, -1, 1, -1, 1]))
    stream = CausalSpatialDecoder(classifier, candidate="causal_csp_1pair")
    seen = []
    for left in range(0, len(rec.signal), 64):
        right = min(left + 64, len(rec.signal))
        cue = [73] if left <= 73 < right else []
        result = stream.process(rec.signal[left:right], onsets=cue)
        if result is not None:
            seen.append(result)
    assert len(seen) == 1
    assert seen[0].label == classifier.predict(cov)[0]
    assert 3.5 <= (seen[0].decision_sample - 73) / 256 < 3.75


def test_unfitted_decoder_fails():
    with pytest.raises(NotFittedError):
        CausalSpatialDecoder(CovarianceCSP())

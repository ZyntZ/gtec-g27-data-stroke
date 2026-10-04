"""Regression tests for training-only drift and instruction-time controls."""
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from stroke_rehab.data import Recording
from stroke_rehab.diagnostics import (out_of_fold_score, pre_cue_features,
                                      temporal_folds)


def test_temporal_purge_respects_order_and_partition():
    y = np.tile([1, -1], 20)
    folds = temporal_folds(y, n_splits=5, purge=2)
    validated = np.concatenate([valid for _, valid in folds])
    np.testing.assert_array_equal(validated, np.arange(40))
    for fit, valid in folds:
        assert len(np.intersect1d(fit, valid)) == 0
        assert np.min(np.abs(fit[:, None] - valid[None, :])) > 2
        assert np.array_equal(np.unique(y[fit]), [-1, 1])
        assert np.array_equal(np.unique(y[valid]), [-1, 1])


def test_temporal_fold_rejects_single_class_validation():
    with pytest.raises(ValueError, match="Both classes"):
        temporal_folds(np.r_[np.ones(8), -np.ones(8)], n_splits=2)


def test_causal_pre_cue_is_unchanged_by_future_samples():
    rng = np.random.default_rng(3)
    signal = rng.normal(size=(3000, 16))
    onset = 100
    rec = Recording(signal, np.array([1]), np.array([onset]), 256)
    before = pre_cue_features(rec)
    altered = signal.copy()
    altered[onset + 2 * 256:onset + 3 * 256] += 1e5
    after = pre_cue_features(Recording(altered, rec.labels, rec.onsets, rec.fs))
    np.testing.assert_array_equal(before, after)
    assert before.shape == (1, 48)


def test_invalid_future_window_rejected():
    rec = Recording(np.zeros((400, 16)), np.array([1]), np.array([0]), 256)
    with pytest.raises(ValueError, match="extends"):
        pre_cue_features(Recording(rec.signal, rec.labels, np.array([200]), rec.fs))


def test_out_of_fold_rejects_missing_predictions():
    X = np.arange(40).reshape(-1, 1).astype(float)
    y = np.tile([1, -1], 20)
    with pytest.raises(ValueError, match="Every trial"):
        out_of_fold_score(X, y, DummyClassifier(strategy="most_frequent"),
                          temporal_folds(y)[:-1])

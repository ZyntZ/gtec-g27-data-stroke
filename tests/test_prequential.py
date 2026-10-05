"""Strict chronological scoring and cue-window controls on synthetic EEG."""
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import sosfilt

from stroke_rehab.data import Recording
from stroke_rehab.prequential import (
    label_only_predictions, prequential_predictions, run_prequential, score_predictions)
from stroke_rehab.streaming import CausalPowerStream, replay_features


def test_preinstruction_window_causal_chunk_invariant():
    fs = 256
    rng = np.random.default_rng(9)
    eeg = rng.normal(size=(8 * fs, 16))
    onsets = np.array([73])
    rec = Recording(eeg, np.array([1]), onsets, fs)
    first, latency, _ = replay_features(rec, window=(0.5, 1.5), chunk_samples=31)
    other, _, _ = replay_features(rec, window=(0.5, 1.5), chunk_samples=512)
    np.testing.assert_allclose(first, other, rtol=0, atol=1e-9)
    assert 1.5 <= latency[0] / fs < 1.5 + 31 / fs
    stream = CausalPowerStream(window=(0.5, 1.5))
    left, right = onsets[0] + 128, onsets[0] + 384
    expected = np.concatenate([
        np.log(np.maximum(np.var(sosfilt(sos, eeg, axis=0)[left:right], axis=0), 1e-12))
        for sos in stream.sos])
    np.testing.assert_allclose(first[0], expected, atol=1e-9, rtol=0)
    corrupted = eeg.copy()
    corrupted[right:] *= 1000
    changed, _, _ = replay_features(
        Recording(corrupted, rec.labels, onsets, fs), window=(0.5, 1.5))
    np.testing.assert_allclose(first, changed, rtol=0, atol=1e-12)


def test_predictions_ignore_all_future_labels_and_features():
    rng = np.random.default_rng(27)
    x = rng.normal(size=(50, 12))
    y = np.tile(np.array([-1, 1]), 25)
    original = prequential_predictions(x, y, start=10)
    newer_x, newer_y = x.copy(), y.copy()
    newer_x[25:] *= 1000
    newer_y[25:] *= -1
    changed = prequential_predictions(newer_x, newer_y, start=10)
    np.testing.assert_array_equal(original[:15], changed[:15])
    previous = label_only_predictions(y, start=10)
    np.testing.assert_array_equal(previous["previous_label"], y[9:-1])
    assert len(previous["prefix_majority"]) == 40
    with pytest.raises(ValueError, match="lacks a class"):
        prequential_predictions(x, np.r_[np.ones(10), y[10:]], start=10)


def test_prequential_only_opens_training_files_and_exports_aggregates(monkeypatch, tmp_path):
    import stroke_rehab.prequential as module
    rng = np.random.default_rng(4)
    labels = np.tile(np.array([-1, 1]), 40)
    rec = Recording(np.zeros((100, 16)), labels, np.arange(80), 256)
    seen = []

    def fake_read(path):
        seen.append(Path(path).name)
        assert str(path).endswith("_training.mat")
        return rec

    def fake_replay(recording, *, window, chunk_samples):
        assert window in module.WINDOWS.values()
        assert chunk_samples == 512
        return rng.normal(size=(80, 8)), np.full(80, int(window[1] * 256)), np.array([])

    monkeypatch.setattr(module, "read_recording", fake_read)
    monkeypatch.setattr(module, "replay_features", fake_replay)
    output = tmp_path / "results.csv"
    rows = run_prequential(tmp_path, output)
    assert len(rows) == 24 and len(seen) == 6
    assert all(r["trials"] == 60 for r in rows)
    contents = output.read_text()
    assert "true_label" not in contents and "predicted_label" not in contents
    assert "P3_post_test.mat" not in seen


def test_validation_class_missing_rejected():
    with pytest.raises(ValueError, match="both classes"):
        score_predictions("P1", "pre", "method", np.ones(10), np.ones(20), start=10)

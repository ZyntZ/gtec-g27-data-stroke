"""Synthetic checks for temporal integrity of the visual-cue control."""

import numpy as np
import pytest

from stroke_rehab.cue_audit import audit_recording, voltage_features
from stroke_rehab.data import Recording


def example_recording():
    fs = 256
    rng = np.random.default_rng(59)
    labels = np.tile([1, -1], 40).astype(np.int8)
    onsets = np.arange(80) * 3 * fs
    signal = rng.standard_normal((80 * 3 * fs, 16)) * 0.05
    # An artificial class-coded visual cue, not a motor-intent response.
    for onset, label in zip(onsets, labels):
        signal[onset + round(2.05 * fs):onset + round(2.4 * fs), 0] += label
    return Recording(signal, labels, onsets, fs)


def test_cue_features_falsify_intent_claim_without_pre_cue_leakage():
    recording = example_recording()
    features = voltage_features(recording)
    assert set(features) == {"pre_instruction", "early_cue"}
    assert all(value.shape == (80, 80) for value in features.values())
    scores = {row["condition"]: row for row in audit_recording(recording)}
    assert scores["early_cue"]["correct"] >= 75
    assert scores["pre_instruction"]["correct"] < 60
    assert scores["early_cue"]["purge_trials"] == 1
    assert scores["early_cue"]["balanced_accuracy"] >= 0.93


def test_causal_features_ignore_future_samples():
    recording = example_recording()
    reference = voltage_features(recording)
    modified = recording.signal.copy()
    modified[recording.onsets[-1] + round(2.55 * recording.fs):] += 1e5
    changed = voltage_features(Recording(modified, recording.labels,
                                         recording.onsets, recording.fs))
    for key in reference:
        np.testing.assert_array_equal(reference[key], changed[key])


def test_invalid_sampling_and_window_bounds():
    recording = example_recording()
    with pytest.raises(ValueError, match="Nyquist"):
        voltage_features(Recording(recording.signal, recording.labels,
                                   recording.onsets, 50))
    with pytest.raises(ValueError, match="extends beyond"):
        voltage_features(Recording(recording.signal[:recording.onsets[-1] + 500],
                                   recording.labels, recording.onsets, recording.fs))


def test_runner_reads_training_only(monkeypatch, tmp_path):
    from stroke_rehab import cue_audit
    recording = example_recording()
    seen = []

    def fake_read(path):
        seen.append(path.name)
        assert path.name.endswith("_training.mat")
        return recording

    monkeypatch.setattr(cue_audit, "read_recording", fake_read)
    output = tmp_path / "audit.csv"
    rows = cue_audit.run_cue_audit(tmp_path, output)
    assert len(rows) == 12
    assert len(seen) == 6
    assert all(not name.endswith("_test.mat") for name in seen)
    assert output.read_text().count("\n") == 13

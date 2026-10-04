"""Synthetic negative-control and permutation invariants."""
from pathlib import Path

import numpy as np
import pytest

from stroke_rehab.data import Recording
from stroke_rehab import controls


def fake_recording():
    rng = np.random.default_rng(23)
    onsets = np.arange(6) * 2500 + 17
    return Recording(rng.normal(size=(len(onsets) * 2500, 16)),
                     np.tile([1, -1], 3).astype(np.int8), onsets, 256)


def test_no_future_sample_in_pre_instruction_covariance():
    recording = fake_recording()
    original = controls.pre_instruction_covariances(recording)
    changed = recording.signal.copy()
    final_onset = recording.onsets[-1]
    changed[final_onset + int(1.75 * 256):] += 1e5
    again = controls.pre_instruction_covariances(
        Recording(changed, -recording.labels, recording.onsets, recording.fs))
    np.testing.assert_array_equal(original, again)
    assert original.shape == (len(recording.labels), 3, 16, 16)


def test_bad_negative_control_timing_rejected():
    recording = fake_recording()
    with pytest.raises(ValueError, match="before the 2-second cue"):
        controls.pre_instruction_covariances(recording, window=(0.5, 2.0))


def test_label_permutations_preserve_block_counts_and_seed():
    labels = np.tile([-1, 1, -1, 1, 1], 8)
    a = controls.blocked_permutation(labels, 5, np.random.default_rng(27))
    b = controls.blocked_permutation(labels, 5, np.random.default_rng(27))
    np.testing.assert_array_equal(a, b)
    for block in np.array_split(np.arange(len(labels)), 5):
        np.testing.assert_array_equal(np.sort(labels[block]), np.sort(a[block]))
    np.testing.assert_array_equal(labels, np.tile([-1, 1, -1, 1, 1], 8))
    with pytest.raises(ValueError, match="binary"):
        controls.blocked_permutation(np.zeros(80), 5, np.random.default_rng(1))


def test_permutation_tail_p_values_have_finite_resolution(monkeypatch):
    y = np.tile([-1, 1], 10)
    X = np.tile(np.eye(4), (len(y), 3, 1, 1))
    monkeypatch.setattr(controls, "fixed_csp_predictions", lambda _, labels, __: labels)
    score, null, upper, lower, two_sided = controls.permutation_audit(
        X, y, permutations=9, blocks=2)
    assert score == len(y) and np.all(null == len(y))
    assert upper == lower == two_sided == 1.0
    with pytest.raises(ValueError, match="positive integer"):
        controls.permutation_audit(X, y, permutations=0, blocks=2)


def test_cli_audit_reads_only_training_files(monkeypatch, tmp_path):
    rng = np.random.default_rng(9)
    paths = []

    def read_training(path):
        paths.append(Path(path).name)
        assert path.name.endswith("_training.mat")
        return fake_recording()

    def pre(_):
        return np.tile(np.eye(4), (6, 3, 1, 1))

    def post(recording, **kwargs):
        return pre(recording), np.zeros(6)

    def permutations(X, y, **kwargs):
        return 3, np.array([2, 3, 4]), 0.75, 0.75, 1.0

    monkeypatch.setattr(controls, "read_recording", read_training)
    monkeypatch.setattr(controls, "pre_instruction_covariances", pre)
    monkeypatch.setattr(controls, "replay_covariances", post)
    monkeypatch.setattr(controls, "permutation_audit", permutations)
    rows = controls.run_controls(tmp_path, tmp_path / "controls.csv", permutations=3)
    assert len(paths) == 6 and len(rows) == 12
    assert all(name.endswith("_training.mat") for name in paths)
    assert (tmp_path / "controls.csv").read_text().count("\n") == 13

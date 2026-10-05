"""Causal, chronological and training-only invariants for the mechanism audit."""
from pathlib import Path

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stroke_rehab.data import Recording
from stroke_rehab.mechanism_audit import (
    prequential_csp_predictions, run_mechanism_audit, score_recording)
from stroke_rehab.spatial import CovarianceCSP, replay_covariances
from stroke_rehab.cue_audit import voltage_features


def covariance_trials(n=48):
    rng = np.random.default_rng(209)
    x = np.empty((n, 3, 16, 16))
    y = np.tile([-1, 1], n // 2)
    for i in range(n):
        for b in range(3):
            a = rng.normal(size=(16, 16)) * 0.08
            x[i, b] = a @ a.T + np.eye(16)
            x[i, b, 0 if y[i] == 1 else 1, 0 if y[i] == 1 else 1] += 0.7
    return x, y


def test_csp_fits_only_earlier_labels_and_features():
    x, y = covariance_trials()
    preds = prequential_csp_predictions(x, y, start=10)
    model = make_pipeline(CovarianceCSP(n_pairs=1, ridge=0.01),
                          StandardScaler(),
                          LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    assert preds[0] == model.fit(x[:10], y[:10]).predict(x[10:11])[0]
    changed_x, changed_y = x.copy(), y.copy()
    changed_x[25:] *= 100
    changed_y[25:] *= -1
    changed = prequential_csp_predictions(changed_x, changed_y, start=10)
    np.testing.assert_array_equal(changed[:15], preds[:15])
    with pytest.raises(ValueError, match="lacks a class"):
        prequential_csp_predictions(x, np.r_[np.ones(10), y[10:]], start=10)
    with pytest.raises(ValueError, match="covariance"):
        prequential_csp_predictions(x[:, 0], y, start=10)


def test_causal_covariance_and_cue_voltage_ignore_future_eeg():
    rng = np.random.default_rng(10)
    fs = 256
    onsets = np.array([71, 71 + 2500], dtype=int)
    signal = rng.normal(size=(onsets[-1] + 2200, 16))
    rec = Recording(signal, np.array([1, -1]), onsets, fs)
    original, offsets = replay_covariances(rec, window=(2.5, 3.5), chunk_samples=64)
    other, _ = replay_covariances(rec, window=(2.5, 3.5), chunk_samples=512)
    np.testing.assert_allclose(original, other, atol=1e-10, rtol=0)
    assert np.all(offsets >= 3.5 * fs)
    changed = signal.copy()
    changed[onsets[0] + 896:onsets[1]] *= 1e4
    changed_rec = Recording(changed, rec.labels, onsets, fs)
    changed_cov, _ = replay_covariances(changed_rec, window=(2.5, 3.5))
    np.testing.assert_allclose(original[0], changed_cov[0], atol=1e-10, rtol=0)
    voltages = voltage_features(rec)
    changed_voltage = voltage_features(changed_rec)
    np.testing.assert_allclose(voltages["early_cue"][0],
                               changed_voltage["early_cue"][0], atol=1e-10, rtol=0)
    changed_rec = Recording(signal, -rec.labels, onsets, fs)
    np.testing.assert_array_equal(voltage_features(rec)["early_cue"],
                                  voltage_features(changed_rec)["early_cue"])


def test_score_recording_uses_same_trials_and_class_counts(monkeypatch):
    import stroke_rehab.mechanism_audit as audit
    rng = np.random.default_rng(27)
    y = np.tile([-1, 1], 40)
    rec = Recording(np.zeros((8, 16)), y, np.arange(80), 256)

    def fake_power(recording, *, window, chunk_samples):
        return rng.normal(size=(80, 8)), np.full(80, round(window[1] * 256)), np.array([])

    def fake_spatial(recording, *, window, chunk_samples):
        return np.tile(np.eye(16), (80, 3, 1, 1)), np.full(80, round(window[1] * 256))

    monkeypatch.setattr(audit, "replay_features", fake_power)
    monkeypatch.setattr(audit, "replay_covariances", fake_spatial)
    monkeypatch.setattr(audit, "prequential_csp_predictions", lambda x, y, start: y[start:].copy())
    monkeypatch.setattr(audit, "voltage_features", lambda _: {
        "pre_instruction": rng.normal(size=(80, 8)),
        "early_cue": rng.normal(size=(80, 8))})
    rows = score_recording(rec, "P1", "pre")
    assert len(rows) == 8
    assert {row["first_validation_trial_1based"] for row in rows} == {21}
    assert all(row["correct_left"] + row["correct_right"] == row["correct"]
               for row in rows)
    assert next(row for row in rows if row["method"] == "early_postcue_csp")["correct"] == 60
    assert next(row for row in rows if row["method"] == "early_cue_voltage")["feature_stop_s"] == 2.54
    assert next(row for row in rows if row["method"] == "early_cue_voltage")["max_replayed_decision_s"] == ""


def test_command_uses_only_training_recordings(monkeypatch, tmp_path):
    import stroke_rehab.mechanism_audit as audit
    opened = []

    def fake_read(path):
        opened.append(Path(path).name)
        assert Path(path).name.endswith("_training.mat")
        Path(path).write_bytes(b"fake training file")
        return object()

    def fake_score(rec, patient, stage, *, warmup, chunk_samples):
        assert warmup == 20 and chunk_samples == 64
        return [{"patient": patient, "session": stage, "correct": 1}]

    monkeypatch.setattr(audit, "read_recording", fake_read)
    monkeypatch.setattr(audit, "score_recording", fake_score)
    rows = run_mechanism_audit(tmp_path, tmp_path / "out.csv")
    assert len(rows) == 6 and len(opened) == 6
    assert all(name.endswith("_training.mat") for name in opened)
    assert (tmp_path / "out.csv").read_text().count("\n") == 7
    import json
    provenance = json.loads((tmp_path / "out_provenance.json").read_text())
    assert sorted(provenance["input_sha256"]) == sorted(opened)
    assert provenance["first_scored_trial_1based"] == 21

"""Data-free tests for chronology, label access and the fixed validation tail."""
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from stroke_rehab import forward_calibration as forward
from stroke_rehab.forward_plot import plot_forward_calibration


def synthetic_recording():
    return SimpleNamespace(labels=np.tile([-1, 1], 40),
                           onsets=np.arange(80) * 2048, fs=256)


def test_same_unseen_tail_for_every_prefix_and_candidate(monkeypatch):
    calls = []

    class Spy(BaseEstimator):
        def fit(self, x, labels):
            calls.append(("fit", np.asarray(x).tolist(), np.asarray(labels).tolist()))
            return self

        def predict(self, x):
            calls.append(("predict", np.asarray(x).tolist()))
            return np.ones(len(x), dtype=int)

    monkeypatch.setattr(forward, "_inputs", lambda recording, causal: (
        {name: np.arange(80) for name in forward.MODELS} if causal else pytest.fail("Noncausal filter")))
    monkeypatch.setattr(forward, "_design", lambda name, x, fs, window: x)
    monkeypatch.setattr(forward, "_estimator", lambda name: Spy())
    rows = forward.score_recording(synthetic_recording(), "P1", "pre",
                                   budgets=(10, 20, 40, 60), stops=(2.0, 3.5))
    assert len(rows) == 2 * 2 * 4 * 20
    assert set(r["validation_trial_1based"] for r in rows) == set(range(61, 81))
    assert {r["calibration_trials"] for r in rows} == {10, 20, 40, 60}
    assert all(r["window_end_s"] == r["decision_time_s"] for r in rows)
    assert len(calls) == 32
    assert all(c[1] == list(range(len(c[1]))) and
               c[2] == synthetic_recording().labels[:len(c[1])].tolist()
               for c in calls if c[0] == "fit")
    assert all(c[1] == list(range(60, 80)) for c in calls if c[0] == "predict")
    summary = forward.summarize(rows)
    assert len(summary) == 16
    assert all(s["correct"] == 10 and s["trials"] == 20 and
               s["balanced_accuracy"] == .5 for s in summary)


def test_reject_overlaps_windows_and_single_class(monkeypatch):
    for bad in [(60, 61), (10, 10), (0, 20)]:
        with pytest.raises(ValueError, match="Budgets"):
            forward.validate_design(bad, (3.5,))
    for bad in [(9.0,), (3.5, 3.5), (0.5,)]:
        with pytest.raises(ValueError, match="Decision"):
            forward.validate_design((20,), bad)
    rec = synthetic_recording()
    rec.labels[:10] = 1
    with pytest.raises(ValueError, match="lacks a class"):
        forward.score_recording(rec, "P1", "pre", budgets=(10,), stops=(3.5,))


def test_run_never_loads_test_recordings(monkeypatch, tmp_path):
    rec = synthetic_recording()
    paths = []

    def fake_load(path):
        paths.append(Path(path).name)
        assert Path(path).name.endswith("_training.mat")
        return rec

    def fake_score(rec, patient, stage, **kwargs):
        return [dict(patient=patient, session=stage, model="mock",
                     decision_time_s=3.5, calibration_trials=10,
                     validation_trial_1based=i, window_end_s=3.5,
                     true_label=(-1 if i % 2 else 1), predicted_label=1,
                     correct=(i % 2 == 0)) for i in range(61, 81)]

    monkeypatch.setattr(forward, "read_recording", fake_load)
    monkeypatch.setattr(forward, "_hash", lambda path: "verified-digest")
    monkeypatch.setattr(forward, "score_recording", fake_score)
    # A small mock run: bypass the normal count check by matching expected length.
    # The public run's expected-length guard is tested separately with this fixture.
    with pytest.raises(AssertionError, match="Unexpected number"):
        forward.run_forward_calibration(tmp_path, tmp_path / "out", budgets=(10,), stops=(3.5,))
    assert len(paths) == 6
    assert set(paths) == {f"P{p}_{s}_training.mat" for p in (1, 2, 3)
                          for s in ("pre", "post")}


def test_plot_requires_unique_rows(tmp_path):
    csv_path = tmp_path / "table.csv"
    fieldnames = ("patient", "session", "model", "decision_time_s",
                  "calibration_trials", "accuracy", "balanced_accuracy")
    rows = [dict(patient="P1", session="pre", model=name, decision_time_s=t,
                 calibration_trials=n, accuracy="0.5", balanced_accuracy="0.5")
            for name in forward.MODELS for t in (2.0, 3.5, 4.25, 6.5)
            for n in (10, 20, 40, 60)]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    target = plot_forward_calibration(csv_path, tmp_path / "figure.png")
    assert target.stat().st_size > 10000
    with csv_path.open("a", newline="") as stream:
        csv.DictWriter(stream, fieldnames=fieldnames).writerow(rows[0])
    with pytest.raises(ValueError, match="Duplicate"):
        plot_forward_calibration(csv_path, target)


def test_plot_rejects_incomplete_time_grid(tmp_path):
    csv_path = tmp_path / "three_times.csv"
    fields = ("patient", "session", "model", "decision_time_s", "calibration_trials",
              "accuracy", "balanced_accuracy")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for t in (2.0, 3.5, 4.25):
            writer.writerow(dict(patient="P1", session="pre", model=forward.MODELS[0],
                                 decision_time_s=t, calibration_trials=10,
                                 accuracy=.5, balanced_accuracy=.5))
    with pytest.raises(ValueError, match="four configured"):
        plot_forward_calibration(csv_path, tmp_path / "unused.png")

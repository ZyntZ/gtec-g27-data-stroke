"""Chronology, label isolation and fixed-time policy tests without recordings."""
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from stroke_rehab import early_selection as early


def _recording():
    return SimpleNamespace(labels=np.tile([-1, 1], 40), onsets=np.arange(80) * 2048, fs=256)


def _spies(monkeypatch):
    calls = []

    class Spy(BaseEstimator):
        def __init__(self, name=None):
            self.name = name

        def fit(self, x, labels):
            calls.append(("fit", self.name, list(x), list(labels)))
            self.fit_stop_ = len(x)
            return self

        def predict(self, x):
            calls.append(("predict", self.name, self.fit_stop_, list(x)))
            # Riemannian wins known inner folds; the tail favors CSP instead.
            labels = np.where(np.asarray(x) % 2 == 0, -1, 1)
            return labels if ((self.name == early.MODELS[1]) == (max(x) < 60)) else -labels

    monkeypatch.setattr(early, "_inputs", lambda rec, causal: (
        {name: np.arange(80) for name in early.MODELS} if causal else pytest.fail("noncausal")))
    windows = []
    monkeypatch.setattr(early, "_design", lambda name, x, fs, window: (windows.append(window) or x))
    monkeypatch.setattr(early, "_estimator", lambda name: Spy(name))
    return calls, windows


def test_selection_uses_earlier_folds_even_when_later_scores_reverse(monkeypatch):
    calls, windows = _spies(monkeypatch)
    predictions, choices, inner = early.score_recording(_recording(), "P2", "pre")
    assert len(predictions) == 320 and len(choices) == 4
    assert set(windows) == {(2.5, 3.5)}
    assert choices[0]["chosen_model"] == early.MODELS[0]
    assert "fallback" in choices[0]["selection_reason"]
    assert all(c["chosen_model"] == early.MODELS[1] for c in choices[1:])
    assert all(c["inner_scores"][early.MODELS[1]] == 1.0 for c in choices[1:])
    assert all(row["validation_trial_1based"] <= row["calibration_trials"] for row in inner)
    assert all(row["fit_stop_trial_1based"] < row["validation_trial_1based"] for row in inner)
    for call in calls:
        if call[0] == "fit":
            assert call[2] == list(range(len(call[2]))) and max(call[2]) < 60
            assert call[3] == _recording().labels[:len(call[2])].tolist()
        else:
            assert min(call[3]) >= call[2]
    for row in early.summarize(predictions):
        assert row["trials"] == 20
        if row["strategy"] == early.SELECTED and row["calibration_trials"] > 10:
            assert row["correct"] == 0  # Selection must not use the tempting tail score.


def test_later_labels_cannot_change_selection_or_predictions(monkeypatch):
    _spies(monkeypatch)
    rec = _recording()
    first, choices, inner = early.score_recording(rec, "P3", "post")
    rec.labels[60:] *= -1
    second, changed_choices, changed_inner = early.score_recording(rec, "P3", "post")
    assert choices == changed_choices and inner == changed_inner
    assert [r["predicted_label"] for r in first] == [r["predicted_label"] for r in second]
    assert [r["correct"] for r in first] != [r["correct"] for r in second]


def test_unused_prefix_labels_do_not_affect_smaller_budget_choice(monkeypatch):
    _spies(monkeypatch)
    rec = _recording()
    _, choices, _ = early.score_recording(rec, "P2", "post", budgets=(20,))
    rec.labels[20:60] *= -1
    _, changed, _ = early.score_recording(rec, "P2", "post", budgets=(20,))
    assert choices == changed


def test_tie_and_missing_inner_class_use_declared_fallback():
    chosen, scores, reason = early.choose_model({name: [-1, 1] for name in early.MODELS}, [-1, 1])
    assert chosen == early.MODELS[0] and scores[early.MODELS[0]] == 1.0
    chosen, scores, reason = early.choose_model({name: [] for name in early.MODELS}, [])
    assert chosen == early.MODELS[0] and scores == {} and "fallback" in reason
    with pytest.raises(ValueError, match="Both classes"):
        early.balanced_accuracy([1, 1], [1, 1])


def test_loading_policy_only_has_six_training_paths(monkeypatch, tmp_path):
    loaded = []
    monkeypatch.setattr(early, "read_recording", lambda path: (loaded.append(path.name) or _recording()))
    monkeypatch.setattr(early, "_hash", lambda path: "0" * 64)
    monkeypatch.setattr(early, "score_recording", lambda *args, **kwargs: ([], [], []))
    design = dict(budgets=[10], decision_time_s=3.5, window_s=1.0,
                  validation_start_0based=60, validation_stop_0based=80,
                  inner_initial_trials=10, inner_step_trials=10,
                  selection_metric="pooled_inner_balanced_accuracy", tie_and_no_inner_fold=early.MODELS[0])
    with pytest.raises(AssertionError, match="Incomplete"):
        early.run(tmp_path, tmp_path / "output", design)
    assert loaded == [f"P{p}_{s}_training.mat" for p in (1, 2, 3) for s in ("pre", "post")]


def test_invalid_budget_and_tail_are_rejected(monkeypatch):
    _spies(monkeypatch)
    with pytest.raises(ValueError, match="Budgets"):
        early.score_recording(_recording(), "P1", "pre", budgets=(61,))
    rec = _recording()
    rec.labels[60:] = 1
    with pytest.raises(ValueError, match="tail"):
        early.score_recording(rec, "P1", "pre")

"""Time selection guards: never use the public test labels to select time."""
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from stroke_rehab import locked_time


def test_peak_is_train_only_and_ties_choose_earliest(monkeypatch):
    labels = np.tile([1, -1], 40).astype(np.int8)
    train = SimpleNamespace(labels=labels, fs=256)
    # Two discriminative endpoints tie; the middle endpoint has no signal.
    features = {3.25: labels.astype(float), 3.5: np.zeros(80),
                3.75: labels.astype(float)}
    monkeypatch.setattr(locked_time, "filtered_trials", lambda *args, **kw: object())
    monkeypatch.setattr(locked_time, "_design", lambda name, trials, fs, window:
                        np.c_[features[window[1]], np.arange(80) * 0.001])
    monkeypatch.setattr(locked_time, "_estimator", lambda name:
                        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    rows, chosen = locked_time.choose_times(
        train, stops=(3.25, 3.5, 3.75), early_stops=(3.25, 3.5), n_splits=5)
    assert chosen == {"train_selected_full": 3.25,
                      "train_selected_early": 3.25, "fixed_3p5": 3.5}
    assert len(rows) == 3
    assert rows[0]["train_oof_correct"] == 80
    assert all(row["train_trials"] == 80 for row in rows)


def test_rejects_invalid_endpoint_grid_before_reading_data():
    train = SimpleNamespace(labels=np.tile([1, -1], 40), fs=256)
    for stops, early in [((3.5, 3.25), (3.25,)), ((3.5, 3.5), (3.5,)),
                         ((3.25, 3.5, 8.25), (3.25,)),
                         ((2.5, 3.5), (3.5,)), ((3.25, 3.5), (3.75,))]:
        with pytest.raises(ValueError, match="endpoints"):
            locked_time.choose_times(train, stops=stops, early_stops=early)


def test_locked_test_scoring_does_not_select_again(monkeypatch):
    labels = np.tile([1, -1], 40).astype(np.int8)
    train = SimpleNamespace(labels=labels, fs=256)
    test = SimpleNamespace(labels=labels.copy(), fs=256)
    calls = []
    class FakeModel:
        def fit(self, data, truth):
            assert np.array_equal(truth, train.labels)
            return self
        def predict(self, data):
            return labels.copy()
    monkeypatch.setattr(locked_time, "filtered_trials", lambda *args, **kw: object())
    def design(name, trials, fs, window):
        calls.append(window[1])
        return np.ones((80, 2))
    monkeypatch.setattr(locked_time, "_design", design)
    monkeypatch.setattr(locked_time, "_estimator", lambda name: FakeModel())
    selected = {"train_selected_full": 5.75, "fixed_3p5": 3.5}
    scores = locked_time.evaluate_locked(train, test, selected)
    assert calls == [5.75, 5.75, 3.5, 3.5]
    assert [row["test_correct"] for row in scores.values()] == [80, 80]
    # Test label mutation changes only scoring, never the locked fit endpoints.
    reversed_test = SimpleNamespace(labels=-labels, fs=256)
    assert [row["test_correct"] for row in
            locked_time.evaluate_locked(train, reversed_test, selected).values()] == [0, 0]
    assert calls[-4:] == calls[:4]

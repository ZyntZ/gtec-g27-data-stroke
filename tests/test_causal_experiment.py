"""Only explicitly named training files may be read for causal CV."""
from pathlib import Path
import numpy as np

from stroke_rehab.data import Recording
from stroke_rehab import causal_experiment


def test_causal_experiment_does_not_open_any_test_run(monkeypatch, tmp_path):
    opened = []
    rng = np.random.default_rng(27)

    def read_training(path):
        opened.append(Path(path).name)
        assert Path(path).name.endswith("_training.mat")
        return Recording(rng.normal(size=(320, 16)),
                         np.tile([1, -1], 40).astype(np.int8),
                         np.arange(80, dtype=int), 256)

    def features(rec, **kwargs):
        X = rng.normal(size=(80, 48))
        return X, np.full(80, 896), np.array([])

    def decisions(rec, estimator, **kwargs):
        from stroke_rehab.streaming import DecisionEvent
        return [DecisionEvent(int(onset), int(onset + 896), 1)
                for onset in rec.onsets], np.full(16, 1e-3)

    monkeypatch.setattr(causal_experiment, "read_recording", read_training)
    monkeypatch.setattr(causal_experiment, "replay_features", features)
    monkeypatch.setattr(causal_experiment, "replay_decisions", decisions)
    result = causal_experiment.run_causal_training(tmp_path, tmp_path / "cv.csv")
    assert len(opened) == len(result) == 6
    assert all(name.endswith("_training.mat") for name in opened)
    assert (tmp_path / "cv.csv").read_text().count("\n") == 7

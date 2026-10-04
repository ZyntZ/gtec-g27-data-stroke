"""Chronological nested-selection tests with no organizer test EEG."""
from pathlib import Path
import json
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from stroke_rehab.data import Recording
from stroke_rehab import nested
from stroke_rehab.spatial import CausalSpatialDecoder


class Spy(BaseEstimator, ClassifierMixin):
    """Assert that every held-out trial was absent from the fitted fold."""

    def fit(self, X, y):
        self.trained_ = set(X[:, 0].astype(int))
        self.classes_ = np.array([-1, 1])
        return self

    def predict(self, X):
        assert not self.trained_.intersection(X[:, 0].astype(int))
        return np.ones(len(X), dtype=np.int8)


def test_no_inner_or_outer_transform_fit_on_validation(monkeypatch):
    n = 60
    y = np.tile([1, -1], n // 2)
    X = np.arange(n, dtype=float)[:, None]
    cov = np.broadcast_to(np.eye(4), (n, 1, 4, 4)).copy()
    monkeypatch.setattr(nested, "candidates", lambda: {name: Spy() for name in nested.CANDIDATES})
    monkeypatch.setattr(nested, "input_for", lambda _, __: X)
    rows, predicted, candidate_predictions = nested.nested_run(
        cov, y, outer_folds=5, inner_folds=3, purge=1)
    assert len(rows) == 5 and len(predicted) == n
    assert candidate_predictions.shape == (3, n)


def test_nested_report_shape_and_purge():
    rng = np.random.default_rng(13)
    y = np.tile([1, -1], 20)
    cov = np.zeros((len(y), 3, 4, 4))
    for i in range(len(y)):
        base = np.diag([2 if y[i] == 1 else 0.3,
                        0.3 if y[i] == 1 else 2, 1, 1])
        cov[i] = np.stack([base + np.diag(rng.uniform(0, 0.2, 4))
                           for _ in range(3)])
    folds, picked, all_preds = nested.nested_run(cov, y, outer_folds=5, inner_folds=3)
    assert len(folds) == 5
    assert picked.shape == y.shape and all_preds.shape == (3, len(y))
    assert sum(f["n_validation"] for f in folds) == len(y)
    assert all(f["selected"] in nested.CANDIDATES for f in folds)
    assert np.all(np.isin(picked, (-1, 1)))


def test_calibration_requires_training_suffix(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="training"):
        nested.calibrate_training_run(tmp_path / "P1_pre_test.mat")


def test_export_calibrates_only_training_files(monkeypatch, tmp_path):
    opened = []
    source = tmp_path / "source"
    source.mkdir()
    for patient in ("P1", "P2", "P3"):
        for stage in ("pre", "post"):
            (source / f"{patient}_{stage}_training.mat").write_bytes(b"synthetic")

    def calibrate(path, **kwargs):
        opened.append(Path(path).name)
        rng = np.random.default_rng(23)
        model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"))
        model.fit(rng.normal(size=(8, 48)), np.tile([-1, 1], 4))
        return "causal_power", model, {"causal_power": 0.5}

    monkeypatch.setattr(nested, "calibrate_training_run", calibrate)
    target = tmp_path / "models"
    metadata = nested.export_calibrations(source, target)
    assert len(metadata) == 6 and len(opened) == 6
    assert all(f.endswith("_training.mat") for f in opened)
    assert len(list(target.glob("*.joblib"))) == len(list(target.glob("*.json"))) == 6
    import joblib
    fitted = joblib.load(target / "P1_pre.joblib")  # A trusted, locally generated file.
    assert isinstance(CausalSpatialDecoder(fitted, candidate="causal_power"), CausalSpatialDecoder)
    stored = json.loads((target / "P1_pre.json").read_text())
    assert stored["candidate"] == "causal_power"
    assert len(stored["training_sha256"]) == 64


def test_latency_audit_rejects_window_selection_and_never_opens_test(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="increasing"):
        nested.run_latency_audit(tmp_path, tmp_path / "a.csv", stops=(4.5, 3.5))
    requested = []
    rng = np.random.default_rng(19)

    def training_only(path):
        requested.append(Path(path).name)
        assert path.name.endswith("_training.mat")
        return Recording(np.zeros((320, 16)), np.tile([1, -1], 40),
                         np.arange(80), 256)

    def dummy_cov(rec, **kwargs):
        matrices = np.tile(np.eye(4), (80, 3, 1, 1))
        matrices += rng.normal(0, 0.005, matrices.shape)
        matrices = (matrices + matrices.transpose(0, 1, 3, 2)) / 2
        return matrices, np.full(80, 900)

    monkeypatch.setattr(nested, "read_recording", training_only)
    monkeypatch.setattr(nested, "replay_covariances", dummy_cov)
    rows = nested.run_latency_audit(tmp_path, tmp_path / "audit.csv",
                                    stops=(3.5, 4.5), folds=5)
    assert len(rows) == 12
    assert len(requested) == 6 and all("_training.mat" in name for name in requested)
    assert all(row["trials"] == 80 for row in rows)

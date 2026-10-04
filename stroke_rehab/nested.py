"""Nested chronological cross-validation for *training-only* causal spatial EEG.

Outer folds provide an unbiased-within-this-run estimate of the predeclared
selection rule. Inner folds choose among three fixed candidate decoders.
No held-out organizer test recording or test label is read by this module.
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import read_recording
from .diagnostics import temporal_folds
from .spatial import CovarianceCSP, replay_covariances


CANDIDATES = ("causal_power", "causal_csp_1pair", "causal_csp_2pairs")


def candidates():
    """Fixed, auditable candidates; power baseline wins exact inner-CV ties."""
    return {
        "causal_power": make_pipeline(
            StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        "causal_csp_1pair": make_pipeline(
            CovarianceCSP(n_pairs=1, ridge=0.01), StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        "causal_csp_2pairs": make_pipeline(
            CovarianceCSP(n_pairs=2, ridge=0.01), StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    }


def input_for(name, covariances):
    """Causal channel log-power uses only diagonals of the same covariance."""
    if name == "causal_power":
        power = np.diagonal(covariances, axis1=-2, axis2=-1)
        return np.log(np.maximum(power, 1e-12)).reshape(len(covariances), -1)
    return covariances


def _predict(model, X, labels, train_indices, validation_indices):
    """Never fit a transformation on a validation or outer-test trial."""
    return clone(model).fit(X[train_indices], labels[train_indices]).predict(
        X[validation_indices])


def nested_run(covariances, labels, *, outer_folds=5, inner_folds=3, purge=1):
    """Return fold rows and all out-of-fold labels in their original order.

    Inner folds are chronological *within the outer-training subset*; they
    never contain an outer-validation index. The fit side of each temporal
    fold excludes its direct neighbours. Outer accuracy is never used to
    select between candidate models, filters or feature windows.
    """
    covariances = np.asarray(covariances)
    labels = np.asarray(labels)
    if (covariances.ndim != 4 or len(covariances) != len(labels)
            or not np.isfinite(covariances).all()):
        raise ValueError("Invalid covariance/label array")
    X = {name: input_for(name, covariances) for name in CANDIDATES}
    models = candidates()
    rows = []
    outer_predictions = np.zeros((len(CANDIDATES), len(labels)), dtype=np.int8)
    predictions = np.zeros(len(labels), dtype=np.int8)
    seen = np.zeros(len(labels), dtype=bool)
    for outer_id, (train, validation) in enumerate(
            temporal_folds(labels, n_splits=outer_folds, purge=purge)):
        local = temporal_folds(labels[train], n_splits=inner_folds, purge=purge)
        inner = {}
        for name in CANDIDATES:
            inner_pred = np.zeros(len(train), dtype=np.int8)
            visited = np.zeros(len(train), dtype=bool)
            for fit, check in local:
                if np.intersect1d(train[fit], validation).size:
                    raise RuntimeError("Outer-validation trial entered an inner fit")
                inner_pred[check] = _predict(models[name], X[name][train],
                                             labels[train], fit, check)
                visited[check] = True
            if not visited.all():
                raise RuntimeError("Inner validation did not cover every training trial")
            inner[name] = float(np.mean(inner_pred == labels[train]))
            outer_predictions[CANDIDATES.index(name), validation] = _predict(
                models[name], X[name], labels, train, validation)
        chosen = max(CANDIDATES, key=inner.__getitem__)
        predictions[validation] = outer_predictions[CANDIDATES.index(chosen), validation]
        seen[validation] = True
        rows.append(dict(fold=outer_id + 1, first_trial=int(validation[0]),
                         last_trial=int(validation[-1]), n_train=len(train),
                         n_validation=len(validation), selected=chosen,
                         inner_power=inner["causal_power"],
                         inner_csp_1pair=inner["causal_csp_1pair"],
                         inner_csp_2pairs=inner["causal_csp_2pairs"],
                         outer_selected_correct=int(np.sum(predictions[validation] == labels[validation])),
                         outer_power_correct=int(np.sum(outer_predictions[0, validation] == labels[validation])),
                         outer_csp_1pair_correct=int(np.sum(outer_predictions[1, validation] == labels[validation])),
                         outer_csp_2pairs_correct=int(np.sum(outer_predictions[2, validation] == labels[validation]))))
    if not seen.all():
        raise RuntimeError("Missing outer-fold predictions")
    return rows, predictions, outer_predictions


def run_nested_training(data_root, output_dir, *, chunk_samples=64,
                        window=(2.5, 3.5), outer_folds=5, inner_folds=3, purge=1):
    """Write per-fold and per-session reports for six organizer training MATs."""
    detail, summary = [], []
    for patient in ("P1", "P2", "P3"):
        for session in ("pre", "post"):
            path = Path(data_root) / f"{patient}_{session}_training.mat"
            recording = read_recording(path)
            cov, offsets = replay_covariances(recording, chunk_samples=chunk_samples,
                                              window=window)
            folds, selected, candidate_pred = nested_run(
                cov, recording.labels, outer_folds=outer_folds,
                inner_folds=inner_folds, purge=purge)
            for row in folds:
                detail.append(dict(patient=patient, session=session, **row))
            y = recording.labels
            row = dict(patient=patient, session=session, train_trials=len(y),
                       nested_correct=int(np.sum(selected == y)),
                       fixed_power_correct=int(np.sum(candidate_pred[0] == y)),
                       fixed_csp_1pair_correct=int(np.sum(candidate_pred[1] == y)),
                       fixed_csp_2pairs_correct=int(np.sum(candidate_pred[2] == y)),
                       selection_power=sum(f["selected"] == "causal_power" for f in folds),
                       selection_csp_1pair=sum(f["selected"] == "causal_csp_1pair" for f in folds),
                       selection_csp_2pairs=sum(f["selected"] == "causal_csp_2pairs" for f in folds),
                       window_start_s=window[0], window_stop_s=window[1],
                       chunk_samples=chunk_samples,
                       latest_decision_s=float(offsets.max() / recording.fs))
            summary.append(row)
            print(f"{patient} {session}: nested {row['nested_correct']}/80; "
                  f"power {row['fixed_power_correct']}/80; "
                  f"CSP2 {row['fixed_csp_2pairs_correct']}/80", flush=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("causal_nested_folds.csv", detail),
                       ("causal_nested_summary.csv", summary)):
        with (output_dir / name).open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary, detail


def calibrate_training_run(path, *, chunk_samples=64, window=(2.5, 3.5),
                           n_splits=3, purge=1):
    """Select a decoder on one training run, then fit it on all its trials.

    This never reads a test MAT and does not report an in-sample score.
    A new CausalSpatialDecoder can consume the fitted estimator afterwards.
    """
    path = Path(path)
    if not path.name.endswith("_training.mat"):
        raise ValueError("Calibration requires a training MAT file")
    recording = read_recording(path)
    cov, _ = replay_covariances(recording, chunk_samples=chunk_samples, window=window)
    labels = recording.labels
    folds = temporal_folds(labels, n_splits=n_splits, purge=purge)
    scores = {}
    for name, model in candidates().items():
        X = input_for(name, cov)
        predicted = np.zeros(len(labels), dtype=np.int8)
        for fit, validation in folds:
            predicted[validation] = _predict(model, X, labels, fit, validation)
        scores[name] = float(np.mean(predicted == labels))
    name = max(CANDIDATES, key=scores.__getitem__)
    fitted = candidates()[name].fit(input_for(name, cov), labels)
    return name, fitted, scores


def export_calibrations(data_root, destination, *, chunk_samples=64,
                        window=(2.5, 3.5), n_splits=3, purge=1):
    """Persist fitted *training-only* pipelines locally, never in Git.

    Joblib uses pickle; load a model only from a trusted local source. No
    training EEG arrays, labels or test MAT files are serialized.
    """
    import hashlib
    import json
    import joblib
    import sklearn

    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    metadata = []
    for patient in ("P1", "P2", "P3"):
        for session in ("pre", "post"):
            path = Path(data_root) / f"{patient}_{session}_training.mat"
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            name, fitted, scores = calibrate_training_run(
                path, chunk_samples=chunk_samples, window=window,
                n_splits=n_splits, purge=purge)
            target = output / f"{patient}_{session}.joblib"
            joblib.dump(fitted, target)
            item = dict(patient=patient, session=session, candidate=name,
                        training_sha256=digest, window_seconds=list(window),
                        chunk_samples=chunk_samples, fs_hz=256, n_channels=16,
                        selection_folds=n_splits, purge_trials=purge,
                        selection_train_cv=scores, sklearn_version=sklearn.__version__)
            (output / f"{patient}_{session}.json").write_text(
                json.dumps(item, indent=2) + "\n")
            metadata.append(item)
            print(f"{patient} {session}: calibrated {name} from training only", flush=True)
    return metadata


def run_latency_audit(data_root, output_file, *, window_starts=2.5,
                      stops=(3.5, 4.5, 5.5, 6.5), chunk_samples=64,
                      folds=5, purge=1):
    """Training-only latency sensitivity for *fixed CSP-1*; never select a window.

    The later windows can contain contingent feedback and are therefore not
    compared as equivalent motor-intent endpoints. Do not use this audit to
    pick a window and quote the same outer folds as unbiased validation.
    """
    if (len(stops) < 2 or sorted(set(stops)) != list(stops)
            or any(stop <= window_starts or stop > 8 for stop in stops)):
        raise ValueError("Expected increasing post-cue window endpoints <=8s")
    rows = []
    for patient in ("P1", "P2", "P3"):
        for session in ("pre", "post"):
            recording = read_recording(Path(data_root) / f"{patient}_{session}_training.mat")
            y = recording.labels
            cv = temporal_folds(y, n_splits=folds, purge=purge)
            for stop in stops:
                cov, offsets = replay_covariances(recording, chunk_samples=chunk_samples,
                                                   window=(window_starts, stop))
                model = candidates()["causal_csp_1pair"]
                predictions = np.zeros(len(y), dtype=np.int8)
                for fit, validation in cv:
                    predictions[validation] = _predict(model, cov, y, fit, validation)
                correct = int(np.sum(predictions == y))
                rows.append(dict(patient=patient, session=session,
                                 start_s=window_starts, stop_s=stop,
                                 correct=correct, trials=len(y), accuracy=correct / len(y),
                                 max_decision_s=float(offsets.max() / recording.fs)))
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows

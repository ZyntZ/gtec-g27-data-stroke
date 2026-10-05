"""Matched, training-only chronological stress test for cue and feedback confounds.

The CSP decoder is fixed to one spatial pair per band, 8-30 Hz and a one-second
window. Pre-instruction CSP and bandpower are counterfactual signal controls;
short cue-locked voltage is a visual-cue challenge, not another motor decoder.
All predictions use only earlier labeled trials, and no test recording is opened.
"""
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .cue_audit import (BIN_CENTERS, CONDITIONS, HALF_BIN_S, voltage_features)
from .data import read_recording
from .features import BANDS
from .prequential import (label_only_predictions, prequential_predictions,
                          score_predictions)
from .spatial import CovarianceCSP, replay_covariances
from .streaming import replay_features

POWER_WINDOWS = {"pre_instruction_power": (0.5, 1.5),
                 "early_postcue_power": (2.5, 3.5)}
SPATIAL_WINDOWS = {"pre_instruction_csp": (0.5, 1.5),
                   "early_postcue_csp": (2.5, 3.5)}
VOLTAGE_NAMES = {"pre_instruction": "pre_instruction_voltage",
                 "early_cue": "early_cue_voltage"}
VOLTAGE_STOPS_S = {
    translated: round(CONDITIONS[source][0] + max(BIN_CENTERS) + HALF_BIN_S, 2)
    for source, translated in VOLTAGE_NAMES.items()
}


def prequential_csp_predictions(covariances, labels, *, start=20):
    """Refit CSP, scaler and LDA on each strictly preceding trial prefix.

    The fit uses only trial indices < i. Target i is never used for selecting
    spatial filters, shrinkage or scaling. Filter-bank covariances may have
    been extracted from the continuous signal by a causal run-wide filter.
    """
    x, y = np.asarray(covariances), np.asarray(labels)
    if (x.ndim != 4 or x.shape[0] != len(y) or x.shape[2] != x.shape[3]
            or not np.isfinite(x).all() or not np.isin(y, (-1, 1)).all()
            or not isinstance(start, int) or not 2 <= start < len(y)):
        raise ValueError("Expected finite covariance features and binary labels")
    predictions = np.empty(len(y) - start, dtype=np.int8)
    for i in range(start, len(y)):
        if len(np.unique(y[:i])) != 2:
            raise ValueError("Calibration prefix lacks a class")
        model = make_pipeline(
            CovarianceCSP(n_pairs=1, ridge=0.01), StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
        model.fit(x[:i], y[:i])
        predictions[i - start] = model.predict(x[i:i + 1])[0]
    return predictions


def score_recording(recording, patient, stage, *, warmup=20,
                    chunk_samples=64):
    """Apply eight fixed controls to identically numbered later training trials."""
    if not isinstance(chunk_samples, int) or chunk_samples < 1:
        raise ValueError("chunk_samples must be positive")
    y = recording.labels
    if not isinstance(warmup, int) or not 2 <= warmup < len(y):
        raise ValueError("Invalid calibration warmup")
    rows = []

    def record(method, predictions, feature_stop_s, note, delivery_s=""):
        """Keep signal-window endpoint separate from replay delivery time."""
        row = score_predictions(patient, stage, method, predictions, y, start=warmup)
        selected = y[warmup:]
        predictions = np.asarray(predictions)
        row.update(correct_left=int(np.sum((selected == 1) & (predictions == 1))),
                   correct_right=int(np.sum((selected == -1) & (predictions == -1))),
                   feature_stop_s=feature_stop_s,
                   max_replayed_decision_s=delivery_s, feature_family=note)
        rows.append(row)

    for method, window in POWER_WINDOWS.items():
        x, offsets, _ = replay_features(recording, window=window,
                                        chunk_samples=chunk_samples)
        if (len(x) != len(y) or np.any(offsets < round(window[1] * recording.fs))
                or np.any(offsets >= round(window[1] * recording.fs) + chunk_samples)):
            raise RuntimeError("Missing, premature or delayed power decision")
        record(method, prequential_predictions(x, y, start=warmup),
               window[1], "causal_bandpower", float(offsets.max() / recording.fs))

    for method, window in SPATIAL_WINDOWS.items():
        x, offsets = replay_covariances(recording, window=window,
                                        chunk_samples=chunk_samples)
        if (len(x) != len(y) or np.any(offsets < round(window[1] * recording.fs))
                or np.any(offsets >= round(window[1] * recording.fs) + chunk_samples)):
            raise RuntimeError("Missing, premature or delayed spatial decision")
        record(method, prequential_csp_predictions(x, y, start=warmup),
               window[1], "causal_csp", float(offsets.max() / recording.fs))

    voltages = voltage_features(recording)
    for original, method in VOLTAGE_NAMES.items():
        record(method, prequential_predictions(voltages[original], y,
                                               start=warmup),
               VOLTAGE_STOPS_S[method], "causal_cue_voltage")
    for method, predictions in label_only_predictions(y, start=warmup).items():
        record(method, predictions, "", "label_only")
    return rows


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_mechanism_audit(data_root, output_file, *, warmup=20,
                        chunk_samples=64):
    """Read only organizer training runs and export per-session aggregate counts."""
    rows, hashes = [], {}
    for patient in ("P1", "P2", "P3"):
        for stage in ("pre", "post"):
            source = Path(data_root) / f"{patient}_{stage}_training.mat"
            recording = read_recording(source)
            hashes[source.name] = _sha256(source)
            rows.extend(score_recording(recording, patient, stage,
                                        warmup=warmup, chunk_samples=chunk_samples))
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "scope": "Retrospective training runs only; exploratory, already-exposed data",
        "input_sha256": hashes,
        "source_sha256": {name: _sha256(Path(__file__).with_name(name))
                          for name in ("mechanism_audit.py", "prequential.py",
                                       "streaming.py", "spatial.py", "cue_audit.py",
                                       "data.py", "features.py")},
        "python": platform.python_version(),
        "packages": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "scikit_learn": sklearn.__version__},
        "first_scored_trial_1based": warmup + 1,
        "calibration_label_access": "Only labels of strictly preceding trials",
        "chunk_samples": chunk_samples,
        "power_and_csp_windows_s": {**POWER_WINDOWS, **SPATIAL_WINDOWS},
        "cue_voltage_feature_stops_s": VOLTAGE_STOPS_S,
        "csp_pairs_per_band": 1,
        "csp_ridge": 0.01,
        "csp_bands_hz": [list(band) for band in BANDS],
        "voltage_status": "Causal offline extraction only; no measured delivery latency",
        "interpretation": "No measured per-trial feedback times, new patient, or clinical outcome"
    }
    path.with_name(path.stem + "_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return rows

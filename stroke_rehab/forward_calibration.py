"""Training-only forward calibration audit for fixed causal decision windows.

No test recording is opened. Every budget uses the same *later* trial IDs 61–80
as its validation set; trial order, including online unlabeled adaptation, is
preserved. This is a retrospective analysis of available training data, not a
prospective treatment or a newly blinded clinical test.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.base import clone

from .comparison import MODELS, _design, _inputs, _estimator
from .data import read_recording

DEFAULT_BUDGETS = (10, 20, 40, 60)
DEFAULT_STOPS = (2.0, 3.5, 4.25, 6.5)
TRIAL_SECONDS = 8.0


def validate_design(budgets, stops, *, validation_start=60, validation_stop=80, window_s=1.0):
    """Reject overlaps, future windows, duplicate settings and label-empty fits."""
    if (not 2 <= validation_start < validation_stop <= 80 or
            not budgets or len(set(budgets)) != len(budgets) or
            not all(isinstance(n, int) and 2 <= n <= validation_start for n in budgets)):
        raise ValueError("Budgets must be unique training prefixes before the validation tail")
    if (not stops or len(set(stops)) != len(stops) or
            not 0 < window_s <= TRIAL_SECONDS or
            not all(window_s <= float(t) <= TRIAL_SECONDS for t in stops)):
        raise ValueError("Decision times must be unique and inside a trial")


def training_files(data_root):
    """Yield ONLY training file paths: no listing or loading test recordings."""
    root = Path(data_root)
    for patient in ("P1", "P2", "P3"):
        for stage in ("pre", "post"):
            yield patient, stage, root / f"{patient}_{stage}_training.mat"


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def score_recording(recording, patient, stage, *, budgets=DEFAULT_BUDGETS,
                    stops=DEFAULT_STOPS, validation_start=60,
                    validation_stop=80, window_s=1.0):
    """Return prediction rows for a single chronological training recording.

    Calibration uses only the first `budget` labels. The target is the SAME
    [validation_start, validation_stop) tail for every candidate and budget.
    Causal band-pass filters run in chronological order; the Riemannian model
    updates its target reference only using the current and earlier target EEG.
    """
    budgets, stops = tuple(budgets), tuple(stops)
    validate_design(budgets, stops, validation_start=validation_start,
                    validation_stop=validation_stop, window_s=window_s)
    if (recording.fs != 256 or len(recording.labels) != 80 or
            len(recording.onsets) != 80):
        raise ValueError("Expected 80 trials at 256 Hz; no trials may be dropped")
    if not np.isin(recording.labels, [-1, 1]).all():
        raise ValueError("Only the documented left/right labels are allowed")
    for n in budgets:
        if len(set(recording.labels[:n])) != 2:
            raise ValueError(f"Calibration prefix {n} lacks a class")
    inputs = _inputs(recording, causal=True)
    rows = []
    for stop in stops:
        window = (float(stop) - window_s, float(stop))
        for model_name in MODELS:
            x = _design(model_name, inputs[model_name], recording.fs, window)
            for n in budgets:
                fitted = clone(_estimator(model_name)).fit(x[:n], recording.labels[:n])
                prediction = fitted.predict(x[validation_start:validation_stop])
                if len(prediction) != validation_stop - validation_start:
                    raise ValueError("Model returned a wrong number of predictions")
                for i, answer in enumerate(prediction, start=validation_start):
                    rows.append({"patient": patient, "session": stage,
                                 "model": model_name, "decision_time_s": float(stop),
                                 "window_start_s": window[0], "window_end_s": window[1],
                                 "calibration_trials": n,
                                 "validation_trial_1based": i + 1,
                                 "true_label": int(recording.labels[i]),
                                 "predicted_label": int(answer),
                                 "correct": int(answer == recording.labels[i])})
    return rows


def summarize(rows):
    """Aggregate fixed trial identities and report accuracy + balanced accuracy."""
    grouped = {}
    for row in rows:
        key = (row["patient"], row["session"], row["model"],
               row["decision_time_s"], row["calibration_trials"])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (patient, stage, model, stop, n), group in sorted(grouped.items()):
        correct = sum(row["correct"] for row in group)
        by_class = {str(label): [r for r in group if r["true_label"] == label]
                    for label in (-1, 1)}
        if not all(by_class.values()):
            raise ValueError("Validation tail must contain both classes")
        summary.append(dict(patient=patient, session=stage, model=model,
                            decision_time_s=stop, calibration_trials=n,
                            validation_first_trial_1based=min(r["validation_trial_1based"] for r in group),
                            validation_last_trial_1based=max(r["validation_trial_1based"] for r in group),
                            correct=correct, trials=len(group),
                            accuracy=correct / len(group),
                            balanced_accuracy=sum(sum(r["correct"] for r in subset) / len(subset)
                                                  for subset in by_class.values()) / 2,
                            left_trials=len(by_class["1"]), right_trials=len(by_class["-1"])))
    return summary


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_forward_calibration(data_root, output_dir, *, budgets=DEFAULT_BUDGETS,
                            stops=DEFAULT_STOPS, validation_start=60,
                            validation_stop=80, window_s=1.0):
    """Run the frozen training-only design; save derived data and provenance."""
    budgets, stops = tuple(budgets), tuple(stops)
    validate_design(budgets, stops, validation_start=validation_start,
                    validation_stop=validation_stop, window_s=window_s)
    root, out = Path(data_root), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    predictions, hashes = [], {}
    for patient, stage, path in training_files(root):
        recording = read_recording(path)
        hashes[path.name] = _hash(path)
        predictions.extend(score_recording(recording, patient, stage,
                                          budgets=budgets, stops=stops,
                                          validation_start=validation_start,
                                          validation_stop=validation_stop,
                                          window_s=window_s))
    expected = 6 * len(MODELS) * len(stops) * len(budgets) * (validation_stop - validation_start)
    if len(predictions) != expected:
        raise AssertionError("Unexpected number of validation decisions")
    summary = summarize(predictions)
    _write_csv(out / "forward_calibration_predictions.csv", predictions)
    _write_csv(out / "forward_calibration_summary.csv", summary)
    metadata = {"scope": "Training runs only; already-exposed dataset; exploratory",
                "design": "Prefix calibration, fixed later validation tail; no test files",
                "input_sha256": hashes,
                "source_sha256": {name: _hash(Path(__file__).with_name(name))
                                  for name in ("forward_calibration.py", "comparison.py",
                                               "riemann.py", "models.py")},
                "runtime": {"python": platform.python_version(),
                            "numpy": np.__version__, "scipy": scipy.__version__,
                            "scikit_learn": sklearn.__version__},
                "model_order": MODELS,
                "budget_prefixes": budgets, "decision_time_s": stops,
                "one_second_window": window_s,
                "validation_trial_1based": [validation_start + 1, validation_stop],
                "calibration_label_access": "Only labels strictly before the validation tail",
                "online_adaptation": "Target covariance reference updates without labels, in trial order",
                "feedback": "Onset per trial unknown; early decisions are not verified feedback-free",
                "interpretation": "Three participants / six correlated sessions; no clinical benefit inference"}
    (out / "forward_calibration_provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return predictions, summary

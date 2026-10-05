"""Training-run-only prequential benchmark with matched pre/post cue controls.

Freeze the three prediction rules before scoring. For trial i, fit only trials
< i, predict i, then release its label. No test recording, future trial label,
or future EEG sample is needed for the decision. Training recordings already
contain feedback, so this remains retrospective, not clinical validation.
"""
from pathlib import Path
import csv

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import read_recording
from .streaming import replay_features

WINDOWS = {"pre_instruction_power": (0.5, 1.5),
           "post_instruction_power": (2.5, 3.5)}
BASELINES = ("previous_label", "prefix_majority")


def prequential_predictions(features, labels, *, start=20):
    """Predict each trial from a separately fitted, strictly earlier prefix.

    `features` are rows in chronological order; one model is retrained for each
    trial. Models never see later examples even without their labels. Returns
    predictions aligned to labels[start:].
    """
    x, y = np.asarray(features), np.asarray(labels)
    if (x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or
            not np.isfinite(x).all() or not np.isin(y, (-1, 1)).all() or
            not isinstance(start, int) or not 2 <= start < len(y)):
        raise ValueError("Need finite features, binary labels and a valid warmup")
    predictions = np.empty(len(y) - start, dtype=np.int8)
    for i in range(start, len(y)):
        if len(np.unique(y[:i])) != 2:
            raise ValueError("Calibration prefix lacks a class")
        model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"))
        model.fit(x[:i], y[:i])
        predictions[i - start] = model.predict(x[i:i + 1])[0]
    return predictions


def label_only_predictions(labels, *, start=20):
    """Use the preceding label or majority of preceding labels; ties are -1."""
    y = np.asarray(labels)
    if (y.ndim != 1 or not np.isin(y, (-1, 1)).all() or
            not isinstance(start, int) or not 1 <= start < len(y)):
        raise ValueError("Invalid label-only baseline input")
    previous = y[start - 1:-1].copy()
    majority = np.array([1 if np.sum(y[:i]) > 0 else -1
                         for i in range(start, len(y))], dtype=np.int8)
    return dict(zip(BASELINES, (previous, majority)))


def score_predictions(patient, stage, method, predictions, labels, *, start):
    """Summarize both recall classes without pretending sessions are people."""
    y = np.asarray(labels)[start:]
    pred = np.asarray(predictions)
    if len(y) != len(pred) or not np.isin(y, (-1, 1)).all() or not np.isin(pred, (-1, 1)).all():
        raise ValueError("Predictions and validation labels must be aligned and binary")
    counts = {label: int(np.count_nonzero(y == label)) for label in (-1, 1)}
    if min(counts.values()) == 0:
        raise ValueError("Validation needs both classes")
    correct = int(np.count_nonzero(pred == y))
    balanced = sum(np.mean(pred[y == label] == label) for label in (-1, 1)) / 2
    return dict(patient=patient, session=stage, method=method,
                first_validation_trial_1based=start + 1,
                last_validation_trial_1based=len(labels),
                correct=correct, trials=len(y), accuracy=correct / len(y),
                balanced_accuracy=float(balanced),
                left_trials=counts[1], right_trials=counts[-1])


def run_prequential(data_root, output_file, *, warmup=20, chunk_samples=512):
    """Run frozen, matched-window training-only predictions; write summary only.

    Deliberately do not export per-trial labels or raw data. Both EEG conditions
    use the same causal streaming filter bank and 1 s duration. The label-only
    rules expose any trial-order predictability without using any EEG.
    """
    if not isinstance(chunk_samples, int) or chunk_samples < 1:
        raise ValueError("chunk_samples must be a positive integer")
    rows = []
    for patient in ("P1", "P2", "P3"):
        for stage in ("pre", "post"):
            path = Path(data_root) / f"{patient}_{stage}_training.mat"
            recording = read_recording(path)
            y = recording.labels
            for method, window in WINDOWS.items():
                x, offsets, _ = replay_features(recording,
                    window=window, chunk_samples=chunk_samples)
                if len(x) != len(y) or np.any(offsets < round(window[1] * recording.fs)):
                    raise RuntimeError("Incomplete or premature feature event")
                predictions = prequential_predictions(x, y, start=warmup)
                rows.append(score_predictions(patient, stage, method,
                                               predictions, y, start=warmup))
            for method, predictions in label_only_predictions(y, start=warmup).items():
                rows.append(score_predictions(patient, stage, method,
                                               predictions, y, start=warmup))
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows

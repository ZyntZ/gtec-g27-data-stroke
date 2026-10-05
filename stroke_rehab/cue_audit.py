"""Training-only cue-locked voltage falsification control.

A class-specific visual instruction may be readable from early event-related
potentials (ERPs). This audit tests whether cue identity can be decoded without
relying on sustained 8-30 Hz motor imagery; it cannot prove a neural source.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import read_recording
from .diagnostics import out_of_fold_score, temporal_folds

# Centers span 1.1-1.5 s (pre-instruction) and 2.1-2.5 s (post-instruction).
# Both use five 80 ms means, 16 channels, and a 400 ms earlier baseline.
BIN_CENTERS = (0.1, 0.2, 0.3, 0.4, 0.5)
HALF_BIN_S = 0.04
CONDITIONS = {
    "pre_instruction": (1.0, (0.5, 0.9)),
    "early_cue": (2.0, (1.5, 1.9)),
}


def voltage_features(recording):
    """Return both fixed feature matrices using one continuous causal low-pass.

    All sample windows are within the EEG already received by 2.54 seconds
    after the trigger. No test recording or class label is used here.
    """
    fs = recording.fs
    if fs <= 60:
        raise ValueError("Sampling rate must exceed the 30 Hz low-pass Nyquist limit")
    signal = np.asarray(recording.signal, dtype=np.float64)
    onsets = np.asarray(recording.onsets)
    if (signal.ndim != 2 or onsets.ndim != 1 or not len(onsets) or
            not np.isfinite(signal).all()):
        raise ValueError("Expected finite EEG and at least one onset")
    sos = butter(3, 30, fs=fs, btype="lowpass", output="sos")
    filtered = sosfilt(sos, signal, axis=0)
    result = {}
    for name, (anchor, baseline) in CONDITIONS.items():
        base_lo, base_hi = (round(fs * v) for v in baseline)
        base = []
        columns = [[] for _ in BIN_CENTERS]
        for onset in onsets:
            onset = int(onset)
            if onset < 0 or onset + round(fs * (anchor + .54)) > len(signal):
                raise ValueError("Cue voltage window extends beyond recording")
            base.append(filtered[onset + base_lo:onset + base_hi].mean(axis=0))
            for i, relative in enumerate(BIN_CENTERS):
                lo = onset + round(fs * (anchor + relative - HALF_BIN_S))
                hi = onset + round(fs * (anchor + relative + HALF_BIN_S))
                if lo < 0 or hi <= lo:
                    raise ValueError("Empty voltage bin")
                columns[i].append(filtered[lo:hi].mean(axis=0))
        reference = np.stack(base)
        result[name] = np.concatenate([np.stack(col) - reference
                                        for col in columns], axis=1)
    return result


def audit_recording(recording, *, folds=5, purge=1):
    """Fit and score the same fixed linear classifier on purged temporal folds."""
    labels = np.asarray(recording.labels)
    if len(labels) != len(recording.onsets):
        raise ValueError("Each onset must have one label")
    splits = temporal_folds(labels, n_splits=folds, purge=purge)
    estimator = make_pipeline(StandardScaler(),
                              LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    rows = []
    for condition, X in voltage_features(recording).items():
        correct, balanced = out_of_fold_score(X, labels, estimator, splits)
        rows.append(dict(condition=condition, correct=correct, trials=len(labels),
                         accuracy=correct / len(labels), balanced_accuracy=balanced,
                         features=X.shape[1], folds=folds, purge_trials=purge))
    return rows


def run_cue_audit(data_root, output_file, *, folds=5, purge=1):
    """Read only six training runs and export descriptive, out-of-fold scores."""
    rows = []
    for patient in ("P1", "P2", "P3"):
        for session in ("pre", "post"):
            path = Path(data_root) / f"{patient}_{session}_training.mat"
            for row in audit_recording(read_recording(path), folds=folds, purge=purge):
                rows.append(dict(patient=patient, session=session, **row))
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows

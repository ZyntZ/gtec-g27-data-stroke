"""Training-run-only diagnostics for temporal drift and pre-instruction leakage.

These scores are diagnostic checks, not alternative test-set leaderboards. In
particular a decodable pre-instruction interval is not evidence of motor imagery.
"""
from pathlib import Path
import csv

import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import read_recording, session_paths
from .features import BANDS, extract
from .models import candidates

PRE_CUE = (0.5, 1.75)  # Entire interval precedes instruction at +2 seconds.


def pre_cue_features(recording):
    """Causal bandpower of continuous data, ending before instruction onset.

    Filtering is causal so no future post-cue EEG can reach the negative control.
    Filters have state across trials; this check can still detect carry-over,
    anticipation or experimental cues other than motor imagery.
    """
    fs = recording.fs
    if fs <= 2 * max(high for _, high in BANDS):
        raise ValueError("Sampling rate below filter-bank Nyquist requirement")
    left, right = (round(t * fs) for t in PRE_CUE)
    if np.any(recording.onsets + right > len(recording.signal)):
        raise ValueError("A pre-cue window extends beyond the recording")
    columns = []
    for low, high in BANDS:
        sos = butter(4, (low, high), btype="bandpass", fs=fs, output="sos")
        filtered = sosfilt(sos, recording.signal, axis=0)
        epochs = np.stack([filtered[onset + left:onset + right]
                           for onset in recording.onsets])
        columns.append(np.log(np.maximum(np.var(epochs, axis=1), 1e-12)))
    return np.concatenate(columns, axis=1)


def temporal_folds(y, n_splits=5, purge=1):
    """Contiguous, chronological validation blocks with neighbouring trials purged.

    Each training partition must retain both classes; no label-dependent fold
    reordering is allowed, since that would mask temporal drift.
    """
    y = np.asarray(y)
    if y.ndim != 1 or n_splits < 2 or purge < 0 or n_splits > len(y):
        raise ValueError("Invalid temporal fold settings")
    folds = []
    for validation in np.array_split(np.arange(len(y)), n_splits):
        start = max(0, int(validation[0]) - purge)
        stop = min(len(y), int(validation[-1]) + purge + 1)
        fit = np.r_[0:start, stop:len(y)]
        if len(np.unique(y[fit])) != 2 or len(np.unique(y[validation])) != 2:
            raise ValueError("Both classes required in each fit and validation block")
        folds.append((fit, validation))
    return folds


def out_of_fold_score(X, y, model, folds):
    """Refit the complete estimator in every fold; return accuracy and counts."""
    y = np.asarray(y)
    predictions = np.empty_like(y)
    seen = np.zeros(len(y), dtype=bool)
    for fit, validation in folds:
        if np.intersect1d(fit, validation).size or seen[validation].any():
            raise ValueError("Overlapping or repeated validation indices")
        predictions[validation] = clone(model).fit(X[fit], y[fit]).predict(X[validation])
        seen[validation] = True
    if not seen.all():
        raise ValueError("Every trial must receive one validation prediction")
    return int(np.sum(y == predictions)), float(balanced_accuracy_score(y, predictions))


def run_diagnostics(data_root, output_file, *, seed=27, n_splits=5, purge=1):
    """Audit each training run only; never load or inspect test-run EEG or labels."""
    rows = []
    for patient, stage, train_path, _ in session_paths(data_root):
        recording = read_recording(train_path)
        y = recording.labels
        random_folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=seed).split(np.zeros(len(y)), y))
        blocked_folds = temporal_folds(y, n_splits=n_splits, purge=purge)
        inputs = {"pre_cue_causal_power": pre_cue_features(recording),
                  "post_cue_offline_csp": extract(recording).epochs}
        estimators = {
            "pre_cue_causal_power": make_pipeline(
                StandardScaler(),
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            "post_cue_offline_csp": candidates()["filterbank_csp_shrinkage_lda"],
        }
        for feature, X in inputs.items():
            for split, folds in (("random", random_folds), ("blocked_purged", blocked_folds)):
                correct, balanced = out_of_fold_score(X, y, estimators[feature], folds)
                rows.append(dict(patient=patient, session=stage, feature=feature,
                                 split=split, correct=correct, trials=len(y),
                                 accuracy=correct / len(y), balanced_accuracy=balanced,
                                 seed=seed, folds=n_splits, purge_trials=(purge if split == "blocked_purged" else 0)))
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows

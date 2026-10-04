"""Predeclared time-window sensitivity audit (not used for model selection)."""
import csv
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from .data import read_recording, session_paths
from .features import CSP_WINDOW, extract
from .models import candidates

# Restrict attention to an earlier post-cue second versus a later interval.
# These are diagnostic time windows, NOT an optimized result or online scores.
INTERVALS = {"early_2p5_to_3p5s": (2.5, 3.5),
             "late_4p5_to_6p5s": (4.5, 6.5)}


def run_audit(data_root, output_file, *, seed=27):
    rows = []
    for patient, stage, train_path, test_path in session_paths(data_root):
        train, test = read_recording(train_path), read_recording(test_path)
        train_features, test_features = extract(train), extract(test)
        for interval, (start, stop) in INTERVALS.items():
            lo = round((start - CSP_WINDOW[0]) * train.fs)
            hi = round((stop - CSP_WINDOW[0]) * train.fs)
            x_train = train_features.epochs[..., lo:hi]
            x_test = test_features.epochs[..., lo:hi]
            model = candidates()["filterbank_csp_shrinkage_lda"]
            predictions = np.empty(len(train.labels), dtype=np.int8)
            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for fit, validation in folds.split(x_train, train.labels):
                fitted = clone(model).fit(x_train[fit], train.labels[fit])
                predictions[validation] = fitted.predict(x_train[validation])
            cv = accuracy_score(train.labels, predictions)
            fitted = clone(model).fit(x_train, train.labels)
            heldout = fitted.predict(x_test)
            rows.append({"patient": patient, "session": stage, "interval": interval,
                         "train_cv_accuracy": cv,
                         "test_accuracy": accuracy_score(test.labels, heldout),
                         "correct_test_trials": int((test.labels == heldout).sum()),
                         "test_trials": len(test.labels)})
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows

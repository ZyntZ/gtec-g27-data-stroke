"""Filter-bank CSP and the Riemannian decoder under one scoring contract.

Same loader (80 trials per run), same training-run folds, same windows, one
decision per test trial. `run_comparison` scores the offline windows used by
`run` and `audit`. `run_decision_time` slides a one-second window through the
trial with CAUSAL filters, so a decision at time t contains no EEG after t.
"""
from pathlib import Path
import csv

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from . import features, riemann
from .data import read_recording, session_paths
from .models import candidates

WINDOWS = {"main_2p5_to_6p5s": (2.5, 6.5),
           "early_2p5_to_3p5s": (2.5, 3.5),
           "late_4p5_to_6p5s": (4.5, 6.5)}
CUE_S = 2.0
# Feedback phase per DatasetInformation.pdf. Per-trial stimulation timestamps
# are not in the data, so this is the protocol's nominal start, not a measurement.
NOMINAL_FEEDBACK_S = 3.5
MODELS = ("filterbank_csp_shrinkage_lda", "filterbank_riemann_recentered")


def _inputs(recording, *, causal):
    """Band-filtered trials for both models: CSP on the team's three bands."""
    return {MODELS[0]: riemann.filtered_trials(recording, features.BANDS, causal=causal),
            MODELS[1]: riemann.filtered_trials(recording, riemann.BANDS, causal=causal)}


def _design(name, trials, fs, window):
    if name == MODELS[0]:
        lo, hi = round(window[0] * fs), round(window[1] * fs)
        return trials[:, :, lo:hi].transpose(0, 1, 3, 2)
    return riemann.window_covariances(trials, fs, window)


def _estimator(name):
    return candidates()[name] if name == MODELS[0] else riemann.RecenteredTangentSpace()


def _write(rows, output_file):
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_comparison(data_root, output_file, *, seed=27, n_splits=5):
    """Offline (zero-phase) windows: training-run CV and held-out test counts."""
    rows = []
    for patient, stage, train_path, test_path in session_paths(data_root):
        train, test = read_recording(train_path), read_recording(test_path)
        folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                     .split(np.zeros(len(train.labels)), train.labels))
        train_trials, test_trials = _inputs(train, causal=False), _inputs(test, causal=False)
        for interval, window in WINDOWS.items():
            for name in MODELS:
                x_train = _design(name, train_trials[name], train.fs, window)
                x_test = _design(name, test_trials[name], test.fs, window)
                cv = np.empty(len(train.labels), dtype=np.int8)
                for fit, validation in folds:
                    fitted = clone(_estimator(name)).fit(x_train[fit], train.labels[fit])
                    cv[validation] = fitted.predict(x_train[validation])
                predicted = _estimator(name).fit(x_train, train.labels).predict(x_test)
                rows.append(dict(
                    patient=patient, session=stage, interval=interval, model=name,
                    train_cv_correct=int((cv == train.labels).sum()),
                    train_trials=len(train.labels),
                    correct_test_trials=int((predicted == test.labels).sum()),
                    test_trials=len(test.labels)))
                print(f"{patient} {stage} {interval} {name}: CV "
                      f"{rows[-1]['train_cv_correct']}/80, test "
                      f"{rows[-1]['correct_test_trials']}/80", flush=True)
    return _write(rows, output_file)


def run_decision_time(data_root, output_file, *, window_s=1.0, step_s=0.25):
    """Causal one-second window ending at each decision time; train run -> test run."""
    stops = np.arange(window_s, riemann.TRIAL_SECONDS + 1e-9, step_s)
    rows = []
    for patient, stage, train_path, test_path in session_paths(data_root):
        train, test = read_recording(train_path), read_recording(test_path)
        train_trials, test_trials = _inputs(train, causal=True), _inputs(test, causal=True)
        for name in MODELS:
            for stop in stops:
                window = (stop - window_s, stop)
                fitted = _estimator(name).fit(
                    _design(name, train_trials[name], train.fs, window), train.labels)
                predicted = fitted.predict(_design(name, test_trials[name], test.fs, window))
                rows.append(dict(patient=patient, session=stage, model=name,
                                 decision_time_s=float(stop), window_s=window_s,
                                 correct_test_trials=int((predicted == test.labels).sum()),
                                 test_trials=len(test.labels)))
        print(f"{patient} {stage}: {len(stops)} decision times scored", flush=True)
    return _write(rows, output_file)

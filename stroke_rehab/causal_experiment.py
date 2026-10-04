"""Fixed causal decoder evaluation, confined to organizer training runs.

For each patient/session, replay raw EEG without future-sample access and
score a small regularized model with contiguous purged trial folds. No test
MAT path is read, and no classifier/window choice is made using test labels.
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import read_recording
from .diagnostics import out_of_fold_score, temporal_folds
from .streaming import replay_decisions, replay_features


def run_causal_training(data_root, output_file, *, chunk_samples=64,
                        window=(2.5, 3.5), n_splits=5, purge=1):
    """Write one *training-only* out-of-fold score per patient/session."""
    rows = []
    for patient in ("P1", "P2", "P3"):
        for stage in ("pre", "post"):
            path = Path(data_root) / f"{patient}_{stage}_training.mat"
            recording = read_recording(path)
            X, decision_offsets, _ = replay_features(
                recording, chunk_samples=chunk_samples, window=window)
            y = recording.labels
            model = make_pipeline(
                StandardScaler(),
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
            correct, balanced = out_of_fold_score(
                X, y, model, temporal_folds(y, n_splits=n_splits, purge=purge))
            # Full-training fit is used solely to time a complete replay, not to
            # score in-sample predictions or select a model.
            model.fit(X, y)
            decisions, times = replay_decisions(
                recording, model, chunk_samples=chunk_samples, window=window)
            if not np.array_equal(
                    [d.decision_sample - d.onset_sample for d in decisions], decision_offsets):
                raise RuntimeError("Feature and classifier replays disagree on timing")
            rows.append(dict(patient=patient, session=stage, train_trials=len(y),
                             correct=correct, accuracy=correct / len(y),
                             balanced_accuracy=balanced, chunk_samples=chunk_samples,
                             window_start_s=window[0], window_stop_s=window[1],
                             n_splits=n_splits, purge_trials=purge,
                             min_decision_s=float(decision_offsets.min() / recording.fs),
                             max_decision_s=float(decision_offsets.max() / recording.fs)))
            decision_chunks = [(d.decision_sample - 1) // chunk_samples for d in decisions]
            print(f"{patient} {stage}: train-only blocked CV {correct}/{len(y)}; "
                  f"EEG available by {decision_offsets.max() / recording.fs:.3f}s "
                  f"after trigger; decision-chunk CPU p95 "
                  f"{np.percentile(times[decision_chunks], 95) * 1e3:.2f}ms",
                  flush=True)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows

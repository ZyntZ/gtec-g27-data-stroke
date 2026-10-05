"""Training-only selection of a fixed causal decision time per session.

The held-out test run is read only after the time has been selected. Its labels
are already publicly exposed, so resulting scores remain exploratory, not blind.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone

from .comparison import MODELS, _design, _estimator
from .data import read_recording, session_paths
from .diagnostics import temporal_folds
from .features import BANDS
from .riemann import filtered_trials

MODEL = MODELS[0]
WINDOW_S = 1.0
STOPS_S = tuple(round(i / 4, 2) for i in range(13, 33))  # 3.25...8.00 s
EARLY_STOPS_S = (3.25, 3.5)
FIXED_S = 3.5


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def choose_times(train, *, stops=STOPS_S, early_stops=EARLY_STOPS_S,
                 n_splits=5, purge=1):
    """Return train-only blocked out-of-fold (OOF) scores and locked times.

    All endpoints use the same folds and causal, continuously filtered signal.
    The entire CSP -> standardization -> LDA pipeline is fitted inside each
    fold, never on its validation trials. Time selection itself is NOT nested;
    its maximum OOF score is optimistically selected and not a performance
    estimate. Ties go to the earliest time.
    """
    stops = tuple(float(t) for t in stops)
    early_stops = tuple(float(t) for t in early_stops)
    if (not stops or not early_stops or tuple(sorted(set(stops))) != stops
            or not set(early_stops) <= set(stops) or FIXED_S not in stops
            or stops[0] < 3.25 or stops[-1] > 8.0):
        raise ValueError("Expected unique ordered post-instruction endpoints within 3.25..8 s")
    y = np.asarray(train.labels)
    if len(y) != 80 or train.fs != 256 or not np.isin(y, (-1, 1)).all():
        raise ValueError("Expected an organizer training recording")
    folds = temporal_folds(y, n_splits=n_splits, purge=purge)
    epochs = filtered_trials(train, BANDS, causal=True)
    rows = []
    for stop in stops:
        x = _design(MODEL, epochs, train.fs, (stop - WINDOW_S, stop))
        predictions = np.zeros(len(y), dtype=np.int8)
        seen = np.zeros(len(y), dtype=bool)
        for fit, validation in folds:
            if seen[validation].any() or np.intersect1d(fit, validation).size:
                raise AssertionError("Fold partitions overlap")
            predictions[validation] = clone(_estimator(MODEL)).fit(
                x[fit], y[fit]).predict(x[validation])
            seen[validation] = True
        if not seen.all() or not np.isin(predictions, (-1, 1)).all():
            raise AssertionError("OOF predictions must cover every trial")
        correct = predictions == y
        left, right = y == 1, y == -1
        rows.append(dict(decision_time_s=stop, train_oof_correct=int(correct.sum()),
                         train_trials=len(y), train_oof_accuracy=float(correct.mean()),
                         train_oof_balanced_accuracy=float(
                             (correct[left].mean() + correct[right].mean()) / 2)))
    # Match the organizer's accuracy objective; use deterministic earliest ties.
    full = max(rows, key=lambda row: row["train_oof_correct"])["decision_time_s"]
    early = max((row for row in rows if row["decision_time_s"] in early_stops),
                key=lambda row: row["train_oof_correct"])["decision_time_s"]
    return rows, {"train_selected_full": full, "train_selected_early": early,
                  "fixed_3p5": FIXED_S}


def evaluate_locked(train, test, selections):
    """Fit once on full calibration run for each *already locked* endpoint."""
    if train.fs != test.fs or len(test.labels) != 80:
        raise ValueError("Expected a matching organizer test recording")
    # Do not compute an accuracy-by-time test grid or choose based on test labels.
    train_epochs = filtered_trials(train, BANDS, causal=True)
    test_epochs = filtered_trials(test, BANDS, causal=True)
    scores = {}
    for strategy, stop in selections.items():
        interval = (stop - WINDOW_S, stop)
        fitted = _estimator(MODEL).fit(_design(MODEL, train_epochs, train.fs, interval),
                                       train.labels)
        predicted = fitted.predict(_design(MODEL, test_epochs, test.fs, interval))
        truth = np.asarray(test.labels)
        left, right = truth == 1, truth == -1
        if not np.isin(predicted, (-1, 1)).all() or not left.any() or not right.any():
            raise ValueError("Invalid prediction or test label")
        correct = predicted == truth
        scores[strategy] = dict(strategy=strategy, decision_time_s=stop,
                                test_correct=int(correct.sum()), test_trials=len(truth),
                                test_accuracy=float(correct.mean()),
                                left_correct=int(correct[left].sum()), left_trials=int(left.sum()),
                                right_correct=int(correct[right].sum()),
                                right_trials=int(right.sum()))
    return scores


def run(data_root, output_dir):
    """Reproduce six exploratory sessions; export only aggregates and hashes."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selection_rows, scores, hashes = [], [], {}
    for patient, session, train_path, test_path in session_paths(data_root):
        train = read_recording(train_path)
        hashes[train_path.name] = _hash(train_path)
        oof, chosen = choose_times(train)
        selection_rows += [dict(patient=patient, session=session, **row) for row in oof]
        # Only now load the test run, after both selected times are frozen.
        test = read_recording(test_path)
        hashes[test_path.name] = _hash(test_path)
        for row in evaluate_locked(train, test, chosen).values():
            scores.append(dict(patient=patient, session=session, model=MODEL, **row))
        print(f"{patient} {session}: full +{chosen['train_selected_full']} s, "
              f"early +{chosen['train_selected_early']} s", flush=True)
    _write(out / "training_time_cv.csv", selection_rows)
    _write(out / "locked_test_scores.csv", scores)
    summary = []
    for strategy in ("train_selected_full", "train_selected_early", "fixed_3p5"):
        subset = [r for r in scores if r["strategy"] == strategy]
        n = sum(r["test_trials"] for r in subset)
        correct = sum(r["test_correct"] for r in subset)
        summary.append(dict(strategy=strategy, sessions=len(subset),
                            test_correct=correct, test_trials=n, test_accuracy=correct / n))
    _write(out / "locked_summary.csv", summary)
    source = Path(__file__).resolve().parent
    provenance = dict(scope="Exploratory reused organizer tests; not blind",
                      contract="One full-training CSP model for each preselected session endpoint",
                      model=MODEL, bands_hz=list(map(list, BANDS)), window_s=WINDOW_S,
                      candidates_s=list(STOPS_S), early_candidates_s=list(EARLY_STOPS_S),
                      folds="five contiguous blocks; one adjacent trial purged from each fit",
                      training_selection="maximum OOF correct count; earliest tie; not nested",
                      test_label_access="after training-only time selection; used only to score predictions",
                      feedback="Time of FES/visual feedback per trial unknown; late scores may reflect it",
                      input_sha256=hashes,
                      source_sha256={name: _hash(source / name) for name in
                                     ("locked_time.py", "comparison.py", "diagnostics.py",
                                      "riemann.py", "models.py", "features.py", "data.py")},
                      output_sha256={name: _hash(out / name) for name in
                                     ("training_time_cv.csv", "locked_test_scores.csv",
                                      "locked_summary.csv")})
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n",
                                         encoding="utf-8")
    return selection_rows, scores, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/stroke-rehab"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/locked_time"))
    args = parser.parse_args(argv)
    _, _, summary = run(args.data_dir, args.output_dir)
    for row in summary:
        print(f"{row['strategy']}: {row['test_correct']}/{row['test_trials']} "
              f"({row['test_accuracy']:.1%})")


if __name__ == "__main__":
    main()

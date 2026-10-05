"""Lock a decision time using training tails, then audit the exposed test curve.

This analysis reuses already-inspected recordings. It is not blind validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

MODEL = "filterbank_csp_shrinkage_lda"
SESSIONS = {(patient, stage) for patient in ("P1", "P2", "P3")
            for stage in ("pre", "post")}
CALIBRATION = 60
TAIL = tuple(range(61, 81))
FIXED = 3.5


def _rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {path}")
        return list(reader)


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _int(row, name):
    value = int(row[name])
    if str(value) != row[name].strip():
        raise ValueError(f"Noncanonical integer in {name}: {row[name]}")
    return value


def _time(row):
    value = float(row["decision_time_s"])
    if not (0 < value <= 8 and value * 4 == int(value * 4)):
        raise ValueError("Expected a finite quarter-second decision time")
    return value


def _training(path):
    by_session = defaultdict(lambda: defaultdict(dict))
    truth = {}
    found = 0
    for row in _rows(path):
        if row["model"] != MODEL or _int(row, "calibration_trials") != CALIBRATION:
            continue
        found += 1
        session = row["patient"], row["session"]
        if session not in SESSIONS:
            raise ValueError(f"Unexpected training session: {session}")
        time = _time(row)
        trial = _int(row, "validation_trial_1based")
        label, prediction, correct = (_int(row, field) for field in
                                      ("true_label", "predicted_label", "correct"))
        if trial not in TAIL or label not in (-1, 1) or prediction not in (-1, 1) or correct != int(label == prediction):
            raise ValueError(f"Invalid training tail row: {session}, {time}, {trial}")
        key = session, trial
        if key in truth and truth[key] != label:
            raise ValueError(f"Conflicting training truth: {key}")
        truth[key] = label
        if trial in by_session[session][time]:
            raise ValueError(f"Duplicate training trial: {session}, {time}, {trial}")
        by_session[session][time][trial] = correct
    if not found or set(by_session) != SESSIONS:
        raise ValueError("Missing training sessions")
    grids = {frozenset(times) for times in (set(curves) for curves in by_session.values())}
    if len(grids) != 1:
        raise ValueError("Training sessions need the same time grid")
    for session, curves in by_session.items():
        for time, trials in curves.items():
            if set(trials) != set(TAIL) or {truth[session, i] for i in TAIL} != {-1, 1}:
                raise ValueError(f"Incomplete or single-class training tail: {session}, {time}")
    return {session: {time: sum(trials.values()) for time, trials in curves.items()}
            for session, curves in by_session.items()}


def _test(path):
    by_session = defaultdict(dict)
    for row in _rows(path):
        if row["model"] != MODEL:
            raise ValueError("Unexpected test model")
        session = row["patient"], row["session"]
        if session not in SESSIONS:
            raise ValueError(f"Unexpected test session: {session}")
        time = _time(row)
        correct, trials = _int(row, "correct"), _int(row, "trials")
        left, right = _int(row, "left_trials"), _int(row, "right_trials")
        left_correct = _int(row, "left_correct")
        right_correct = _int(row, "right_correct")
        if (trials != 80 or left + right != trials or min(left, right) <= 0
                or not 0 <= correct <= trials or not 0 <= left_correct <= left
                or not 0 <= right_correct <= right or left_correct + right_correct != correct
                or abs(float(row["accuracy"]) - correct / trials) > 1e-10
                or abs(float(row["window_start_s"]) - (time - 1.0)) > 1e-10):
            raise ValueError(f"Inconsistent test timecourse: {session}, {time}")
        if time in by_session[session]:
            raise ValueError(f"Duplicate test time: {session}, {time}")
        by_session[session][time] = correct
    if set(by_session) != SESSIONS or len({frozenset(c) for c in by_session.values()}) != 1:
        raise ValueError("Incomplete or unmatched test-session time grids")
    return dict(by_session)


def _choose(training, sessions, candidates):
    counts = {time: sum(training[session][time] for session in sessions)
              for time in candidates}
    return min(candidates, key=lambda time: (-counts[time], time))


def audit(training_csv, test_csv):
    """Return training-locked and descriptive test-label-selected scores.

    No fitting is performed: test counts are read only after time is selected.
    The individual test-trial prediction matrix was not saved, so paired
    uncertainty and a prospective test claim are not available here.
    """
    training, test = _training(training_csv), _test(test_csv)
    candidates = sorted(set.intersection(*(set(training[s]) & set(test[s]) for s in SESSIONS)))
    if FIXED not in candidates or len(candidates) < 2:
        raise ValueError("No common fixed time or insufficient candidate times")
    ordered = sorted(SESSIONS)
    chosen = _choose(training, ordered, candidates)
    time_counts = {str(time): sum(training[s][time] for s in ordered) for time in candidates}
    held_out = []
    for patient in ("P1", "P2", "P3"):
        train_sessions = [s for s in ordered if s[0] != patient]
        selected = _choose(training, train_sessions, candidates)
        held_out.append({"patient": patient, "selected_time_s": selected,
                         "training_correct_on_other_patients": sum(training[s][selected] for s in train_sessions),
                         "exploratory_test_correct": sum(test[s][selected] for s in ordered if s[0] == patient),
                         "test_trials": 160})
    per_session = [{"patient": p, "session": stage,
                    "training_tail_correct_at_lock": training[p, stage][chosen],
                    "training_tail_trials": 20,
                    "test_correct_at_lock": test[p, stage][chosen],
                    "test_correct_at_fixed_early": test[p, stage][FIXED],
                    "test_posthoc_peak_correct": max(test[p, stage].values())}
                   for p, stage in ordered]
    locked = sum(r["test_correct_at_lock"] for r in per_session)
    fixed = sum(r["test_correct_at_fixed_early"] for r in per_session)
    oracle_session = sum(r["test_posthoc_peak_correct"] for r in per_session)
    test_pooled_time = min(test[ordered[0]], key=lambda t: (-sum(test[s][t] for s in ordered), t))
    return {
        "scope": "Exploratory reanalysis of already-exposed training and test CSVs; NOT a blind or clinical result",
        "selection": "Pool trials 61-80 across six training runs after 60 calibration labels; maximize training accuracy; earliest time breaks ties",
        "model": MODEL, "candidate_times_s": candidates, "training_correct_of_120": time_counts,
        "locked_time_s": chosen, "training_correct_at_lock": time_counts[str(chosen)],
        "training_trials": 120, "test_correct_at_lock": locked, "test_trials": 480,
        "test_accuracy_at_lock": locked / 480, "test_correct_at_fixed_early": fixed,
        "test_accuracy_at_fixed_early": fixed / 480,
        "test_pooled_posthoc_peak_time_s": test_pooled_time,
        "test_pooled_posthoc_peak_correct": sum(test[s][test_pooled_time] for s in ordered),
        "test_sum_of_session_posthoc_peaks_correct": oracle_session,
        "per_session": per_session, "leave_one_patient_out_selection": held_out,
        "caveats": ["The test timecourse has been inspected before: no independent confirmatory evaluation.",
                    "The 60-trial models choose time, but the published test curve uses 80-trial fits.",
                    "The three participants, not 480 trials, are the independent population units.",
                    "Late windows may include visual or electrical feedback; improved classification is not evidence of motor recovery.",
                    "Aggregate test timecourses cannot support paired trial-level uncertainty estimates."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=Path("results/forward_calibration/forward_calibration_predictions.csv"))
    parser.add_argument("--test", type=Path, default=Path("results/organizer_metric/organizer_metric_timecourse.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = audit(args.training, args.test)
    summary["input_sha256"] = {"training_csv": _sha256(args.training), "test_csv": _sha256(args.test)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Training-locked time: {summary['locked_time_s']} s; exposed test: "
          f"{summary['test_correct_at_lock']}/{summary['test_trials']} (NOT blind).")


if __name__ == "__main__":
    main()

"""Validate a matched-tail jury figure without requiring patient recordings."""
import csv
import hashlib
from pathlib import Path

import pytest

from analysis.jury_tradeoff import SESSIONS, TIMES, TRIALS, read_same_tail, render


def make_fixture(path):
    fields = ("patient", "session", "model", "calibration_trials",
              "decision_time_s", "validation_trial_1based", "true_label",
              "predicted_label", "correct")
    rows = []
    for patient, stage in SESSIONS:
        for t in TIMES:
            for trial in TRIALS:
                label = 1 if trial % 2 else -1
                prediction = label if t != 2.0 else -label
                rows.append(dict(patient=patient, session=stage,
                                 model="filterbank_csp_shrinkage_lda",
                                 calibration_trials=60, decision_time_s=t,
                                 validation_trial_1based=trial, true_label=label,
                                 predicted_label=prediction,
                                 correct=int(label == prediction)))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows, fields


def write_rows(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_matches_trials_and_exports_counts(tmp_path):
    source = tmp_path / "predictions.csv"
    make_fixture(source)
    counts = read_same_tail(source)
    assert counts[("P1", "pre", 2.0)] == 0
    assert counts[("P1", "pre", 3.5)] == 20
    rows = render(counts, source, tmp_path / "figures")
    assert {(r["decision_time_s"], r["correct"])
            for r in rows if r["patient"] == "pooled"} == {
                (2.0, 0), (3.5, 120), (4.25, 120), (6.5, 120)}
    for name in ("same_tail_counts.csv", "same_tail_tradeoff.png", "same_tail_tradeoff.svg"):
        assert (tmp_path / "figures" / name).stat().st_size > 100


@pytest.mark.parametrize("fault, expected", [
    ("missing", "Incomplete"), ("duplicate", "Duplicate"),
    ("wrong_label", "Labels differ"), ("false_correct", "Invalid labels")])
def test_corrupt_or_unmatched_trials_are_rejected(tmp_path, fault, expected):
    source = tmp_path / "predictions.csv"
    rows, fields = make_fixture(source)
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[-1].copy())
    elif fault == "wrong_label":
        rows[-1]["true_label"] = -rows[-1]["true_label"]
        rows[-1]["correct"] = int(rows[-1]["true_label"] == rows[-1]["predicted_label"])
    else:
        rows[-1]["correct"] = 1 - rows[-1]["correct"]
    write_rows(source, rows, fields)
    with pytest.raises(ValueError, match=expected):
        read_same_tail(source)


def test_committed_figures_match_saved_matched_tail():
    repo = Path(__file__).resolve().parents[1]
    source = repo / "results/forward_calibration/forward_calibration_predictions.csv"
    output = repo / "results/jury_tradeoff/same_tail_counts.csv"
    counts = read_same_tail(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with output.open(newline="", encoding="utf-8") as stream:
        actual = list(csv.DictReader(stream))
    assert len(actual) == len(SESSIONS) * len(TIMES) + len(TIMES)
    assert {row["source_sha256"] for row in actual} == {digest}
    for row in actual:
        t = float(row["decision_time_s"])
        expected = (sum(counts[(*pair, t)] for pair in SESSIONS)
                    if row["patient"] == "pooled"
                    else counts[(row["patient"], row["session"], t)])
        assert int(row["correct"]) == expected
        assert int(row["trials"]) == (120 if row["patient"] == "pooled" else 20)
    assert [int(row["correct"]) for row in actual[-4:]] == [55, 82, 99, 99]

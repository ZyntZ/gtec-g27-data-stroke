"""Data-free regression tests for the training-locked decision-time audit."""
import csv
import json
from pathlib import Path

import pytest

from analysis.training_locked_endpoint import _choose, audit, main

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "results/forward_calibration/forward_calibration_predictions.csv"
TEST = ROOT / "results/organizer_metric/organizer_metric_timecourse.csv"


def _mutated_csv(source, destination, change):
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    change(rows)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def test_training_lock_matches_published_curves():
    result = audit(TRAIN, TEST)
    assert result["training_correct_of_120"] == {"3.5": 82, "4.25": 99, "6.5": 99}
    assert result["locked_time_s"] == 4.25
    assert (result["test_correct_at_lock"], result["test_correct_at_fixed_early"]) == (407, 358)
    assert (result["test_pooled_posthoc_peak_correct"],
            result["test_sum_of_session_posthoc_peaks_correct"]) == (407, 427)
    assert [row["selected_time_s"] for row in result["leave_one_patient_out_selection"]] == [6.5, 4.25, 4.25]


def test_selection_tie_is_broken_without_test_labels():
    sample = {("P1", "pre"): {3.5: 9, 4.25: 10, 6.5: 10}}
    assert _choose(sample, [("P1", "pre")], [3.5, 4.25, 6.5]) == 4.25


def test_conflicting_training_labels_rejected(tmp_path):
    def change(rows):
        for row in rows:
            if (row["patient"], row["session"], row["model"], row["calibration_trials"],
                    row["validation_trial_1based"], row["decision_time_s"]) == (
                    "P1", "pre", "filterbank_csp_shrinkage_lda", "60", "61", "4.25"):
                row["true_label"] = str(-int(row["true_label"]))
                row["correct"] = str(int(row["true_label"] == row["predicted_label"]))
                return
        raise AssertionError("Target row not found")
    bad = _mutated_csv(TRAIN, tmp_path / "train.csv", change)
    with pytest.raises(ValueError, match="Conflicting training truth"):
        audit(bad, TEST)


def test_missing_and_duplicate_training_trial_rejected(tmp_path):
    def missing(rows):
        rows[:] = [r for r in rows if not (r["patient"] == "P1" and r["session"] == "pre"
                   and r["model"] == "filterbank_csp_shrinkage_lda"
                   and r["calibration_trials"] == "60"
                   and r["decision_time_s"] == "4.25" and r["validation_trial_1based"] == "61")]
    bad = _mutated_csv(TRAIN, tmp_path / "missing.csv", missing)
    with pytest.raises(ValueError, match="Incomplete"):
        audit(bad, TEST)
    bad = _mutated_csv(TRAIN, tmp_path / "duplicate.csv", lambda rows: rows.append(rows[0].copy()))
    # The first row is outside the 60-label selection and should not affect it.
    assert audit(bad, TEST)["locked_time_s"] == 4.25
    def duplicate_selected(rows):
        target = next(r for r in rows if r["calibration_trials"] == "60"
                      and r["model"] == "filterbank_csp_shrinkage_lda")
        rows.append(target.copy())
    bad = _mutated_csv(TRAIN, tmp_path / "duplicate_selected.csv", duplicate_selected)
    with pytest.raises(ValueError, match="Duplicate training trial"):
        audit(bad, TEST)


def test_test_curve_corruption_rejected(tmp_path):
    def duplicate(rows):
        rows.append(rows[0].copy())
    bad = _mutated_csv(TEST, tmp_path / "duplicate.csv", duplicate)
    with pytest.raises(ValueError, match="Duplicate test time"):
        audit(TRAIN, bad)
    def impossible(rows):
        rows[0]["correct"] = "81"
    bad = _mutated_csv(TEST, tmp_path / "bad_count.csv", impossible)
    with pytest.raises(ValueError, match="Inconsistent test timecourse"):
        audit(TRAIN, bad)


def test_cli_writes_hashes_and_does_not_copy_raw_trials(tmp_path):
    output = tmp_path / "summary.json"
    main(["--training", str(TRAIN), "--test", str(TEST), "--output", str(output)])
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["input_sha256"]["training_csv"] == "3772a68d367e9eb4ed5d281609523bef80f5c65b5793391a88e77cac2eea7dcc"
    assert "true_label" not in result and len(result["per_session"]) == 6

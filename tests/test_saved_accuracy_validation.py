"""Published result checks must work without the private covariance cache."""
import csv
import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("saved_validation", ROOT / "analysis/accuracy_calibration/validate.py")
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


@pytest.fixture
def published_results(tmp_path):
    source = ROOT / "results/accuracy_calibration"
    for path in source.glob("*.csv"):
        shutil.copyfile(path, tmp_path / path.name)
    shutil.copyfile(source / "selection_training_only.json", tmp_path / "selection_training_only.json")
    return tmp_path


def mutate(path, change):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    change(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_published_csvs_validate_without_cache_or_claiming_tests_ran(published_results):
    assert not (published_results / "training_covariances.npz").exists()
    receipt = VALIDATOR.validate_results(published_results)
    assert receipt["status"] == "pass"
    assert all(receipt["checks"].values())
    assert "tests" not in receipt
    assert receipt["test_execution"].startswith("Not run")


@pytest.mark.parametrize("change,failed_check", [
    (lambda rows: rows[0].update(trial=rows[1]["trial"]), "oof_trial_coverage"),
    (lambda rows: rows[0].update(p_left="nan"), "oof_labels_and_probabilities_valid"),
    (lambda rows: rows[0].update(p_left="1.1"), "oof_labels_and_probabilities_valid"),
    (lambda rows: rows[0].update(p_left="0.9"), "oof_metrics_reproduce"),
])
def test_rejects_corrupt_predictions(published_results, change, failed_check):
    mutate(published_results / "training_oof_predictions.csv", change)
    receipt = VALIDATOR.validate_results(published_results)
    assert receipt["status"] == "failed"
    assert not receipt["checks"][failed_check]


def test_rejects_balanced_but_contradictory_family_labels(published_results):
    def change(rows):
        group = [r for r in rows if r["session"] == "P1_pre" and r["family"] == "Riemann"]
        left = next(r for r in group if r["label"] == "1")
        right = next(r for r in group if r["label"] == "-1")
        left["label"], right["label"] = right["label"], left["label"]
    mutate(published_results / "training_oof_predictions.csv", change)
    receipt = VALIDATOR.validate_results(published_results)
    assert receipt["checks"]["oof_labels_and_probabilities_valid"]
    assert not receipt["checks"]["oof_family_labels_identical"]
    assert receipt["status"] == "failed"


@pytest.mark.parametrize("bad_indices", ["0,0,2,3,4,5,6,7,8,9", "80,1,2,3,4,5,6,7,8,9"])
def test_rejects_duplicate_or_out_of_range_calibration_members(published_results, bad_indices):
    def change(rows):
        next(r for r in rows if r["n_calibration"] == "10")["indices"] = bad_indices
    mutate(published_results / "calibration_subsets.csv", change)
    receipt = VALIDATOR.validate_results(published_results)
    assert receipt["status"] == "failed"
    assert not receipt["checks"]["exact_balanced_subset_counts"]


def test_rejects_changed_training_metric_and_curve_summary(published_results):
    mutate(published_results / "nested_training_cv.csv", lambda rows: rows[0].update(brier="0.01"))
    mutate(published_results / "calibration_curves.csv", lambda rows: rows[0].update(accuracy="0.9"))
    receipt = VALIDATOR.validate_results(published_results)
    assert receipt["status"] == "failed"
    assert not receipt["checks"]["oof_metrics_reproduce"]
    assert not receipt["checks"]["calibration_curve_arithmetic"]


def test_rejects_consistent_but_impossible_calibration_scores(published_results):
    def change_subsets(rows):
        for row in rows:
            if row["session"] == "P1_pre" and row["n_calibration"] == "10":
                row.update(correct="88", accuracy="1.1", brier="-0.1")
    def change_curve(rows):
        row = next(r for r in rows if r["session"] == "P1_pre" and r["n_calibration"] == "10")
        row.update(accuracy="1.1", minimum="1.1", maximum="1.1", subset_sd="0", brier="-0.1")
    mutate(published_results / "calibration_subsets.csv", change_subsets)
    mutate(published_results / "calibration_curves.csv", change_curve)
    receipt = VALIDATOR.validate_results(published_results)
    assert receipt["status"] == "failed"
    assert not receipt["checks"]["metric_domains_valid"]


def test_rejects_setting_only_mutation_at_full_calibration(published_results):
    def change(rows):
        next(r for r in rows if r["session"] == "P1_pre" and r["n_calibration"] == "80")["chosen_setting"] = "('FBCSP', 2)"
    mutate(published_results / "calibration_subsets.csv", change)
    receipt = VALIDATOR.validate_results(published_results)
    assert receipt["status"] == "failed"
    assert not receipt["checks"]["full_calibration_subset_matches_frozen_model"]

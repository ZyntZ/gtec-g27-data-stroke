"""Validate published CSV arithmetic without recordings, model fitting or caches.

This command does not run pytest. Test execution has its own dated receipt.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/accuracy_calibration"
SESSIONS = tuple(f"{p}_{s}" for p in ("P1", "P2", "P3") for s in ("pre", "post"))
FAMILIES = ("CSP-baseline", "FBCSP", "Riemann", "TV-CSP", "selected")


def read(out, name):
    with (out / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-10)


def bounded(value, lower=0, upper=1):
    value = float(value)
    return math.isfinite(value) and lower <= value <= upper


def oof_metrics(rows):
    """Recalculate scores using P(left) >= .5 as the tie rule."""
    labels = [int(r["label"]) for r in rows]
    probabilities = [float(r["p_left"]) for r in rows]
    correct = [int((1 if p >= .5 else -1) == y) for y, p in zip(labels, probabilities)]
    recalls = [sum(ok for ok, y in zip(correct, labels) if y == lab) / labels.count(lab)
               for lab in (-1, 1)]
    clipped = [min(1 - sys.float_info.epsilon, max(sys.float_info.epsilon, p)) for p in probabilities]
    return dict(correct=sum(correct), trials=len(rows), accuracy=sum(correct) / len(rows),
                balanced_accuracy=sum(recalls) / 2,
                brier=sum((p - (y == 1)) ** 2 for y, p in zip(labels, probabilities)) / len(rows),
                log_loss=-sum(math.log(p if y == 1 else 1 - p)
                              for y, p in zip(labels, clipped)) / len(rows))


def validate_results(out=OUT):
    out = Path(out)
    legacy = read(out, "baseline_reproduction.csv")
    train = read(out, "nested_training_cv.csv")
    test = read(out, "exploratory_test_accuracy.csv")
    curves = read(out, "calibration_curves.csv")
    subsets = read(out, "calibration_subsets.csv")
    reliability = read(out, "probability_reliability.csv")
    oof = read(out, "training_oof_predictions.csv")
    transfer = read(out, "pre_post_transfer.csv")
    selection = json.loads((out / "selection_training_only.json").read_text(encoding="utf-8"))
    checks = {}
    expected = {(s, f) for s in SESSIONS for f in FAMILIES}
    checks["metric_domains_valid"] = all(
        int(r["trials"]) > 0 and 0 <= int(r["correct"]) <= int(r["trials"])
        and all(bounded(r[key]) for key in ("accuracy", "balanced_accuracy", "brier") if key in r)
        and ("log_loss" not in r or bounded(r["log_loss"], upper=math.inf))
        for r in legacy + train + test + subsets + transfer) and all(
        all(bounded(r[key]) for key in ("accuracy", "minimum", "maximum", "brier"))
        and bounded(r["subset_sd"], upper=math.inf)
        and float(r["minimum"]) <= float(r["accuracy"]) <= float(r["maximum"]) for r in curves)
    checks["expected_sessions_and_families"] = (
        set(selection) == set(SESSIONS) and len(train) == len(test) == len(expected)
        and all({(r["session"], r["family"]) for r in rows} == expected for rows in (train, test, oof)))
    checks["team_baseline_six_sessions_match"] = (
        len(legacy) == 6 and {r["session"] for r in legacy} == set(SESSIONS)
        and all(r["matches_previous"] == "True" and close(r["accuracy"], r["previous_accuracy"])
                and int(r["train_trials"]) == int(r["trials"]) == 80
                and close(r["accuracy"], int(r["correct"]) / 80) for r in legacy))
    checks["team_baseline_433_of_480"] = sum(int(r["correct"]) for r in legacy) == 433 and sum(int(r["trials"]) for r in legacy) == 480
    checks["all_new_train_and_test_counts_80"] = (
        all(int(r["trials"]) == 80 and 0 <= int(r["correct"]) <= 80
            and close(r["accuracy"], int(r["correct"]) / 80) for r in train + test)
        and all(int(r["train_trials"]) == 80 for r in test))
    groups = {(s, f): [r for r in oof if r["session"] == s and r["family"] == f] for s, f in expected}
    checks["oof_trial_coverage"] = all(sorted(int(r["trial"]) for r in rows) == list(range(80)) for rows in groups.values())
    checks["oof_labels_and_probabilities_valid"] = all(
        sorted(int(r["label"]) for r in rows) == [-1] * 40 + [1] * 40
        and all(math.isfinite(float(r["p_left"])) and 0 <= float(r["p_left"]) <= 1 for r in rows)
        for rows in groups.values())
    label_maps = {key: {int(r["trial"]): int(r["label"]) for r in rows} for key, rows in groups.items()}
    checks["oof_family_labels_identical"] = all(label_maps[s, f] == label_maps[s, "CSP-baseline"] for s, f in expected)
    oof_valid = all(checks[name] for name in ("expected_sessions_and_families", "oof_trial_coverage",
                                            "oof_labels_and_probabilities_valid", "oof_family_labels_identical"))
    checks["oof_metrics_reproduce"] = oof_valid and all(
        all(close(r[key], value) for key, value in oof_metrics(groups[r["session"], r["family"]]).items()) for r in train)

    expected_subset_keys = {(s, n, rep) for s in SESSIONS for n in (0, 10, 20, 40, 80)
                            for rep in (range(5) if n in (10, 20, 40) else range(1))}
    subset_keys = [(r["session"], int(r["n_calibration"]), int(r["repeat"])) for r in subsets]
    checks["subset_memberships_complete"] = len(subset_keys) == len(expected_subset_keys) and set(subset_keys) == expected_subset_keys
    indices = {key: [int(v) for v in r["indices"].split(",") if v] for key, r in zip(subset_keys, subsets)}
    checks["exact_balanced_subset_counts"] = oof_valid and checks["subset_memberships_complete"] and all(
        (not idx and int(r["fit_trials"]) == 320) if key[1] == 0 else (
            int(r["fit_trials"]) == key[1] == len(idx) == len(set(idx))
            and all(0 <= trial < 80 for trial in idx)
            and sum(label_maps[key[0], "CSP-baseline"][trial] == 1 for trial in idx) == key[1] // 2)
        for key, r in zip(subset_keys, subsets) for idx in [indices[key]])
    checks["nested_subsets"] = checks["subset_memberships_complete"] and all(
        set(indices[s, a, rep]) <= set(indices[s, b, rep])
        for s in SESSIONS for rep in range(5) for a, b in ((10, 20), (20, 40)))

    expected_curve_keys = {(s, n) for s in SESSIONS for n in (0, 10, 20, 40, 80)}
    checks["all_requested_calibration_budgets"] = len(curves) == len(expected_curve_keys) and {(r["session"], int(r["n_calibration"])) for r in curves} == expected_curve_keys
    curve_match = checks["all_requested_calibration_budgets"] and checks["subset_memberships_complete"]
    if curve_match:
        for row in curves:
            members = [r for r in subsets if r["session"] == row["session"] and r["n_calibration"] == row["n_calibration"]]
            values = [float(r["accuracy"]) for r in members]
            mean = sum(values) / len(values)
            sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** .5 if len(values) > 1 else 0.
            curve_match &= (int(row["repeats"]) == len(members) and int(row["test_trials"]) == 80
                and all(int(r["trials"]) == 80 and close(r["accuracy"], int(r["correct"]) / 80) for r in members)
                and all(close(row[key], value) for key, value in {
                    "accuracy": mean, "minimum": min(values), "maximum": max(values), "subset_sd": sd,
                    "brier": sum(float(r["brier"]) for r in members) / len(members)}.items()))
    checks["calibration_curve_arithmetic"] = bool(curve_match)
    selected = {r["session"]: r for r in test if r["family"] == "selected"}
    checks["full_calibration_endpoint_matches_selected"] = checks["expected_sessions_and_families"] and checks["all_requested_calibration_budgets"] and all(
        close(r["accuracy"], selected[r["session"]]["accuracy"]) for r in curves if int(r["n_calibration"]) == 80)
    checks["full_calibration_subset_matches_frozen_model"] = checks["expected_sessions_and_families"] and checks["subset_memberships_complete"] and all(
        list(ast.literal_eval(r["chosen_setting"])) == selection[r["session"]]["chosen"]
        and ast.literal_eval(r["chosen_setting"]) == ast.literal_eval(selected[r["session"]]["chosen_setting"])
        and all(close(r[key], selected[r["session"]][key]) for key in (
            "correct", "trials", "accuracy", "balanced_accuracy", "brier", "log_loss"))
        for r in subsets if int(r["n_calibration"]) == 80)
    checks["full_choice_matches_training_receipt"] = checks["expected_sessions_and_families"] and all(
        list(ast.literal_eval(r["chosen_setting"])) == selection[r["session"]]["families"][r["family"]]
        and (r["family"] != "selected" or list(ast.literal_eval(r["chosen_setting"])) == selection[r["session"]]["chosen"]) for r in test)
    checks["reliability_bins_account_for_80_trials"] = (
        len(reliability) == 30 and {r["session"] for r in reliability} == set(SESSIONS)
        and all(len([r for r in reliability if r["session"] == s]) == 5
                and sum(int(r["n"]) for r in reliability if r["session"] == s) == 80 for s in SESSIONS)
        and all(int(r["n"]) >= 0 for r in reliability))
    checks["pre_post_transfer_only_pre_training_labels"] = (
        len(transfer) == 3 and {r["patient"] for r in transfer} == {"P1", "P2", "P3"}
        and all(int(r["source_trials"]) == int(r["trials"]) == 80
                and int(r["source_test_labels_used"]) == int(r["target_calibration_labels"]) == 0
                and close(r["accuracy"], int(r["correct"]) / 80) for r in transfer))
    sources = sorted(out.glob("*.csv")) + [out / "selection_training_only.json"]
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(checks.values()) else "failed", "checks": checks,
        "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "test_execution": "Not run by this validator; see ../overnight/FINAL_CHECKS.json for separate dated test evidence",
        "scope": "Published CSV arithmetic and consistency only; no recording hashes rechecked, model fitting, test prediction replay or scientific validation",
        "pooled_correct": {f: sum(int(r["correct"]) for r in test if r["family"] == f) for f in FAMILIES},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT, help="Directory of published result CSVs")
    args = parser.parse_args()
    receipt = validate_results(args.output)
    (args.output / "validation.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "checks": receipt["checks"]}, indent=2))
    if receipt["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

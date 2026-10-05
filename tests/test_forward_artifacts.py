"""Check committed training-only receipts without downloading EEG recordings."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from stroke_rehab.forward_calibration import summarize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "forward_calibration"


def test_frozen_audit_receipts_are_self_consistent():
    with (OUT / "forward_calibration_predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with (OUT / "forward_calibration_summary.csv").open(newline="", encoding="utf-8") as stream:
        summary = list(csv.DictReader(stream))
    provenance = json.loads((OUT / "forward_calibration_provenance.json").read_text())
    parity = json.loads((OUT / "stream_parity.json").read_text())
    config = json.loads((ROOT / "configs" / "forward_calibration.json").read_text())
    assert len(rows) == 3840 and len(summary) == 192
    assert provenance["budget_prefixes"] == config["budgets"]
    assert provenance["decision_time_s"] == config["decision_times_s"]
    assert len(provenance["input_sha256"]) == 6
    assert len(provenance["source_sha256"]) == 4
    assert parity["historical_source_sha256"] == provenance["source_sha256"]
    assert parity["input_sha256"] == provenance["input_sha256"]
    assert parity["prediction_rows_matched"] == len(rows)
    assert parity["summary_rows_matched"] == len(summary)
    assert parity["source_sha256"].keys() == provenance["source_sha256"].keys()
    for source, digest in parity["source_sha256"].items():
        assert hashlib.sha256((ROOT / "stroke_rehab" / source).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == digest
    assert set(parity["additional_source_sha256_lf"]) >= {
        "stroke_rehab/riemann_stream.py", "stroke_rehab/features.py", "stroke_rehab/data.py"}
    for source, digest in parity["additional_source_sha256_lf"].items():
        assert hashlib.sha256((ROOT / source).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == digest
    assert set(parity["output_sha256_lf"]) == {
        "forward_calibration_predictions.csv", "forward_calibration_summary.csv"}
    for output, digest in parity["output_sha256_lf"].items():
        assert hashlib.sha256((OUT / output).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == digest
    assert len(parity["stream_checks"]) == 12
    assert {(r["patient"], r["session"], r["adapt"]) for r in parity["stream_checks"]} == {
        (patient, session, adapt) for patient in ("P1", "P2", "P3")
        for session in ("pre", "post") for adapt in (False, True)}
    for check in parity["stream_checks"]:
        assert check["trials"] == 20
        assert all(check[key] for key in (
            "batch_single_partition_labels_match", "final_references_match", "decision_scores_match"))
        if check["adapt"]:
            assert check["raw_training_replay_labels_match"] and check["raw_training_trials"] == 80
    for source, digest in provenance["source_sha256"].items():
        assert len(digest) == 64
        # Git may check text out with CRLF on Windows. The original receipt
        # identifies historical LF source, not a requirement to freeze code.
        current = hashlib.sha256((ROOT / "stroke_rehab" / source).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if current != digest:
            assert parity["source_sha256"][source] == current
            assert parity["historical_source_sha256"] == provenance["source_sha256"]
            assert parity["prediction_rows_matched"] == 3840
            assert parity["summary_rows_matched"] == 192
            assert parity["input_sha256"] == provenance["input_sha256"]
    assert all(n.endswith("_training.mat") and len(digest) == 64
               for n, digest in provenance["input_sha256"].items())
    assert all(r["validation_trial_1based"] in map(str, range(61, 81))
               for r in rows)
    assert not any("test" in path for path in provenance["input_sha256"])
    assert all(int(r["true_label"]) in (-1, 1) and int(r["predicted_label"]) in (-1, 1)
               and int(r["correct"]) == int(r["true_label"] == r["predicted_label"])
               for r in rows)
    keys = [(r["patient"], r["session"], r["model"],
             r["decision_time_s"], r["calibration_trials"],
             r["validation_trial_1based"]) for r in rows]
    assert len(set(keys)) == len(rows)
    converted = [{**r, "decision_time_s": float(r["decision_time_s"]),
                  "calibration_trials": int(r["calibration_trials"]),
                  "validation_trial_1based": int(r["validation_trial_1based"]),
                  "correct": int(r["correct"]), "true_label": int(r["true_label"])}
                 for r in rows]
    recomputed = summarize(converted)
    index = {(r["patient"], r["session"], r["model"],
              float(r["decision_time_s"]), int(r["calibration_trials"])): r
             for r in summary}
    assert len(index) == len(recomputed)
    for r in recomputed:
        key = (r["patient"], r["session"], r["model"],
               r["decision_time_s"], r["calibration_trials"])
        original = index[key]
        assert int(original["correct"]) == r["correct"]
        assert int(original["trials"]) == r["trials"] == 20
        assert np.isclose(float(original["balanced_accuracy"]), r["balanced_accuracy"])

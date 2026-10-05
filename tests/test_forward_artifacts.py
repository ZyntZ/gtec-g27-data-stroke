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
    config = json.loads((ROOT / "configs" / "forward_calibration.json").read_text())
    assert len(rows) == 3840 and len(summary) == 192
    assert provenance["budget_prefixes"] == config["budgets"]
    assert provenance["decision_time_s"] == config["decision_times_s"]
    assert len(provenance["input_sha256"]) == 6
    assert len(provenance["source_sha256"]) == 4
    for source, digest in provenance["source_sha256"].items():
        assert len(digest) == 64
        assert hashlib.sha256((ROOT / "stroke_rehab" / source).read_bytes()).hexdigest() == digest
    assert all(n.endswith("_training.mat") and len(digest) == 64
               for n, digest in provenance["input_sha256"].items())
    assert all(r["validation_trial_1based"] in map(str, range(61, 81))
               for r in rows)
    assert not any("test" in path for path in provenance["input_sha256"])
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

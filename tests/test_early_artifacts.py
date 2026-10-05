"""Reconstruct the committed fixed policy without needing raw EEG."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from stroke_rehab import early_selection as early

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/early_selection"


def _rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ints = {"calibration_trials", "validation_trial_1based", "fit_stop_trial_1based",
            "true_label", "predicted_label", "correct"}
    return [{k: int(v) if k in ints else float(v) if k.endswith("_s") else v
             for k, v in row.items()} for row in rows]


def test_saved_receipts_choices_and_main_trial_parity():
    rows = _rows(OUT / "early_predictions.csv")
    inner = _rows(OUT / "early_inner_predictions.csv")
    summary = _rows(OUT / "early_summary.csv")
    choices = json.loads((OUT / "early_choices.json").read_text())
    receipt = json.loads((OUT / "early_provenance.json").read_text())
    design = json.loads((ROOT / "configs/early_selection.json").read_text())
    assert len(rows) == 1920 and len(summary) == 96 and len(choices) == 24
    assert receipt["design"] == design
    assert receipt["design_sha256"] == hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    assert len(receipt["source_commit"]) == 40
    assert set(receipt["source_sha256"]) >= {
        "stroke_rehab/features.py", "stroke_rehab/forward_calibration.py", "stroke_rehab/early_selection.py"}
    for path, digest in receipt["source_sha256"].items():
        assert early._source_hash(ROOT / path) == digest
    assert set(receipt["output_sha256"]) == {
        "early_predictions.csv", "early_summary.csv", "early_inner_predictions.csv", "early_choices.json"}
    for path, digest in receipt["output_sha256"].items():
        assert early._hash(OUT / path) == digest
    assert len(receipt["input_sha256"]) == 6
    assert all(n.endswith("_training.mat") and len(h) == 64 for n, h in receipt["input_sha256"].items())
    key = lambda r: (r["patient"], r["session"], r["calibration_trials"])
    trialkey = lambda r: (*key(r), r["validation_trial_1based"])
    main = _rows(ROOT / "results/forward_calibration/forward_calibration_predictions.csv")
    expected = {(*trialkey(r), r["model"]): r for r in main if r["decision_time_s"] == 3.5}
    family = {(*trialkey(r), r["strategy"]): r for r in rows if r["strategy"] in early.MODELS}
    assert len(expected) == len(family) == 960 and family.keys() == expected.keys()
    assert len({(*trialkey(r), r["strategy"]) for r in rows}) == len(rows)
    for k, r in family.items():
        assert (r["true_label"], r["predicted_label"], r["correct"]) == (
            expected[k]["true_label"], expected[k]["predicted_label"], expected[k]["correct"])
        assert r["window_start_s"] == 2.5 and r["window_end_s"] == 3.5
    for choice in choices:
        k = key(choice)
        pooled = {name: sorted((r for r in inner if key(r) == k and r["model"] == name),
                              key=lambda r: r["validation_trial_1based"]) for name in early.MODELS}
        labels = [r["true_label"] for r in pooled[early.MODELS[0]]]
        assert labels == [r["true_label"] for r in pooled[early.MODELS[1]]]
        assert len(labels) == choice["inner_trials"]
        for name, block in pooled.items():
            assert len({r["validation_trial_1based"] for r in block}) == len(block)
            assert all(r["fit_stop_trial_1based"] < r["validation_trial_1based"] <= k[2] for r in block)
        chosen, scores, reason = early.choose_model(
            {name: [r["predicted_label"] for r in block] for name, block in pooled.items()}, labels)
        assert (chosen, scores, reason) == (choice["chosen_model"], choice["inner_scores"], choice["selection_reason"])
        majority = 1 if choice["calibration_left"] > choice["calibration_right"] else -1
        assert choice["majority_label"] == majority
        assert choice["calibration_left"] + choice["calibration_right"] == k[2]
        for r in (r for r in rows if key(r) == k):
            assert r["chosen_model"] == chosen and r["correct"] == int(r["true_label"] == r["predicted_label"])
            assert 61 <= r["validation_trial_1based"] <= 80
            if r["strategy"] == early.SELECTED:
                assert r["predicted_label"] == family[(*trialkey(r), chosen)]["predicted_label"]
            if r["strategy"] == early.MAJORITY:
                assert r["predicted_label"] == majority
    recomputed = early.summarize(rows)
    assert len(recomputed) == len(summary)
    for actual, saved in zip(recomputed, summary, strict=True):
        for name, value in actual.items():
            if isinstance(value, (int, float)):
                assert np.isclose(value, float(saved[name]), rtol=0, atol=1e-14)
            else:
                assert value == saved[name]

"""Select an early decoder using only chronological folds inside calibration.

The fixed +3.5 s endpoint is a declared design choice, not a test-selected time.
Available training tails have already been inspected: this is retrospective
procedural validation, not new independent evidence of improvement.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.base import clone

from .comparison import MODELS, _design, _estimator, _inputs
from .data import read_recording
from .forward_calibration import training_files, validate_design

SELECTED = "training_selected"
MAJORITY = "calibration_majority"
STRATEGIES = (*MODELS, SELECTED, MAJORITY)


def balanced_accuracy(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.shape != predictions.shape or not np.isin(labels, (-1, 1)).all():
        raise ValueError("Labels and predictions must be matching binary vectors")
    if not np.isin(predictions, (-1, 1)).all():
        raise ValueError("Invalid predicted label")
    if set(labels.tolist()) != {-1, 1}:
        raise ValueError("Both classes are required for balanced accuracy")
    return float(np.mean([np.mean(predictions[labels == label] == label)
                          for label in (-1, 1)]))


def choose_model(inner_predictions, labels):
    """Deterministic pooled-inner score; ties/no usable folds choose fixed CSP."""
    labels = np.asarray(labels)
    if set(labels.tolist()) != {-1, 1}:
        return MODELS[0], {}, "fixed_fallback_no_two_class_inner_validation"
    scores = {name: balanced_accuracy(labels, inner_predictions[name]) for name in MODELS}
    # MODELS order makes the tie explicit and stable.
    chosen = max(MODELS, key=lambda name: scores[name])
    return chosen, scores, "chronological_inner_balanced_accuracy"


def score_recording(recording, patient, stage, *, budgets=(10, 20, 40, 60),
                    decision_time_s=3.5, window_s=1.0, validation_start=60,
                    validation_stop=80, inner_initial_trials=10, inner_step_trials=10):
    validate_design(tuple(budgets), (decision_time_s,), validation_start=validation_start,
                    validation_stop=validation_stop, window_s=window_s)
    if not (isinstance(inner_initial_trials, int) and inner_initial_trials >= 2 and
            isinstance(inner_step_trials, int) and inner_step_trials >= 1):
        raise ValueError("Invalid chronological inner design")
    if recording.fs != 256 or len(recording.labels) != 80 or len(recording.onsets) != 80:
        raise ValueError("Expected all 80 trials at 256 Hz")
    labels = np.asarray(recording.labels)
    if not np.isin(labels, (-1, 1)).all():
        raise ValueError("Invalid labels")
    if set(labels[validation_start:validation_stop].tolist()) != {-1, 1}:
        raise ValueError("Validation tail lacks a class")
    for n in budgets:
        if set(labels[:n].tolist()) != {-1, 1}:
            raise ValueError(f"Calibration prefix {n} lacks a class")
    window = (decision_time_s - window_s, decision_time_s)
    inputs = _inputs(recording, causal=True)
    designs = {name: _design(name, inputs[name], recording.fs, window) for name in MODELS}
    cache, predictions, selections, inner_rows = {}, [], [], []
    for n in budgets:
        fold_records, pooled_ids = [], []
        pooled_predictions = {name: [] for name in MODELS}
        for fit_stop in range(inner_initial_trials, n, inner_step_trials):
            stop = min(fit_stop + inner_step_trials, n)
            usable = set(labels[:fit_stop].tolist()) == {-1, 1}
            fold_records.append(dict(fit_trial_1based=[1, fit_stop],
                                     validation_trial_1based=[fit_stop + 1, stop],
                                     usable=usable,
                                     reason="two_class_fit" if usable else "single_class_fit_skipped"))
            if not usable:
                continue
            pooled_ids.extend(range(fit_stop, stop))
            for name in MODELS:
                key = (name, fit_stop, stop)
                if key not in cache:
                    fitted = clone(_estimator(name)).fit(designs[name][:fit_stop], labels[:fit_stop])
                    # Each validation block starts a fresh target-reference stream.
                    cache[key] = fitted.predict(designs[name][fit_stop:stop])
                answer = cache[key]
                pooled_predictions[name].extend(map(int, answer))
                for trial, pred in zip(range(fit_stop, stop), answer, strict=True):
                    inner_rows.append(dict(patient=patient, session=stage, calibration_trials=n,
                                           model=name, fit_stop_trial_1based=fit_stop,
                                           validation_trial_1based=trial + 1,
                                           true_label=int(labels[trial]), predicted_label=int(pred)))
        chosen, scores, reason = choose_model(pooled_predictions, labels[pooled_ids])
        majority_label = 1 if np.sum(labels[:n] == 1) > np.sum(labels[:n] == -1) else -1
        selections.append(dict(patient=patient, session=stage, calibration_trials=n,
                               chosen_model=chosen, selection_reason=reason,
                               inner_scores=scores, inner_trials=len(pooled_ids),
                               inner_folds=fold_records,
                               calibration_left=int(np.sum(labels[:n] == 1)),
                               calibration_right=int(np.sum(labels[:n] == -1)),
                               majority_label=majority_label,
                               decision_time_s=decision_time_s,
                               target_reference_reset_trial_1based=validation_start + 1))
        tail_predictions = {}
        for name in MODELS:
            fitted = clone(_estimator(name)).fit(designs[name][:n], labels[:n])
            tail_predictions[name] = fitted.predict(designs[name][validation_start:validation_stop])
        tail_predictions[SELECTED] = tail_predictions[chosen]
        tail_predictions[MAJORITY] = np.full(validation_stop - validation_start, majority_label)
        for strategy in STRATEGIES:
            for trial, pred in zip(range(validation_start, validation_stop),
                                   tail_predictions[strategy], strict=True):
                predictions.append(dict(patient=patient, session=stage, strategy=strategy,
                                        calibration_trials=n, chosen_model=chosen,
                                        decision_time_s=decision_time_s,
                                        window_start_s=window[0], window_end_s=window[1],
                                        validation_trial_1based=trial + 1,
                                        true_label=int(labels[trial]), predicted_label=int(pred),
                                        correct=int(pred == labels[trial])))
    return predictions, selections, inner_rows


def summarize(predictions):
    grouped = {}
    for row in predictions:
        key = (row["patient"], row["session"], row["strategy"], row["calibration_trials"])
        grouped.setdefault(key, []).append(row)
    output = []
    for (patient, stage, strategy, n), rows in sorted(grouped.items()):
        labels = np.array([r["true_label"] for r in rows])
        answers = np.array([r["predicted_label"] for r in rows])
        correct = int(np.sum(labels == answers))
        output.append(dict(patient=patient, session=stage, strategy=strategy,
                           calibration_trials=n, chosen_model=rows[0]["chosen_model"],
                           decision_time_s=rows[0]["decision_time_s"], correct=correct,
                           trials=len(rows), accuracy=correct / len(rows),
                           balanced_accuracy=balanced_accuracy(labels, answers),
                           left_recall=float(np.mean(answers[labels == 1] == 1)),
                           right_recall=float(np.mean(answers[labels == -1] == -1)),
                           left_trials=int(np.sum(labels == 1)), right_trials=int(np.sum(labels == -1))))
    return output


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(data_root, output_dir, design):
    """Load six training files only; retain every model, choice and failure cell."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    kwargs = {k: design[k] for k in ("budgets", "decision_time_s", "window_s",
                                  "inner_initial_trials", "inner_step_trials")}
    kwargs.update(validation_start=design["validation_start_0based"],
                  validation_stop=design["validation_stop_0based"])
    if design["selection_metric"] != "pooled_inner_balanced_accuracy" or \
            design["tie_and_no_inner_fold"] != MODELS[0]:
        raise ValueError("Unsupported selection rule")
    predictions, choices, inner, hashes = [], [], [], {}
    for patient, stage, path in training_files(data_root):
        print(f"{patient} {stage}: fixed +{design['decision_time_s']} s chronological selection", flush=True)
        recording = read_recording(path)
        hashes[path.name] = _hash(path)
        a, b, c = score_recording(recording, patient, stage, **kwargs)
        predictions.extend(a)
        choices.extend(b)
        inner.extend(c)
    expected = 6 * len(design["budgets"]) * len(STRATEGIES) * (kwargs["validation_stop"] - kwargs["validation_start"])
    if len(predictions) != expected:
        raise AssertionError("Incomplete prediction grid")
    summary = summarize(predictions)
    _write(out / "early_predictions.csv", predictions)
    _write(out / "early_summary.csv", summary)
    _write(out / "early_inner_predictions.csv", inner)
    (out / "early_choices.json").write_text(json.dumps(choices, indent=2) + "\n", encoding="utf-8")
    source_root = Path(__file__).resolve().parents[1]
    metadata = dict(scope="Retrospective training-only selection; exposed tails; no new independent validation",
                    design=design, design_sha256=hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest(),
                    input_sha256=hashes,
                    source_sha256={str(p.relative_to(source_root)).replace('\\', '/'): _hash(p)
                                   for p in (Path(__file__), source_root / "stroke_rehab/comparison.py",
                                             source_root / "stroke_rehab/riemann.py", source_root / "stroke_rehab/models.py",
                                             source_root / "stroke_rehab/data.py")},
                    output_sha256={p.name: _hash(p) for p in out.glob("early_*.csv")},
                    runtime=dict(python=platform.python_version(), numpy=np.__version__,
                                 scipy=scipy.__version__, sklearn=sklearn.__version__),
                    label_access="Fit and inner selection use only trial IDs within the specified prefix",
                    adaptation="Riemannian reference resets at the start of each validation block; skipped gap EEG is not used for reference adaptation",
                    feedback="Early post-cue; per-trial FES onset unavailable; not certified feedback-free",
                    latency="EEG cutoff at +3.5 s; hardware/transport and delivered latency not measured")
    (out / "early_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return predictions, summary, choices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/early_selection"))
    parser.add_argument("--config", type=Path, default=Path("configs/early_selection.json"))
    args = parser.parse_args()
    design = json.loads(args.config.read_text(encoding="utf-8"))
    run(args.data_dir, args.output_dir, design)


if __name__ == "__main__":
    main()

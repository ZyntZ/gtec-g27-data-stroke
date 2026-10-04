"""Within-session evaluation: train-run CV decides the pipeline; test once."""
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from .data import read_recording, session_paths
from .features import extract
from .models import candidate_input, candidates


@dataclass(frozen=True)
class Result:
    patient: str
    session: str
    selected_model: str
    train_trials: int
    test_trials: int
    train_cv_accuracy: float
    train_cv_balanced_accuracy: float
    test_accuracy: float
    test_balanced_accuracy: float
    correct_test_trials: int


def evaluate_session(train_path, test_path, *, seed=27, n_splits=5):
    """Select by training-only stratified CV, then refit on all training trials.

    Return selected model and CV scores for all candidates. Test labels are
    accessed only after selection and model fitting; no cross-session leakage.
    """
    train = read_recording(train_path)
    test = read_recording(test_path)
    if train.fs != test.fs:
        raise ValueError("Training and test sample rates differ")
    train_features = extract(train)
    folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=seed).split(
        np.zeros(len(train.labels)), train.labels))
    model_scores = {}
    for name, model in candidates().items():
        X = candidate_input(name, train_features)
        predictions = np.empty(len(train.labels), dtype=np.int8)
        for fit_idx, validation_idx in folds:
            # A new estimator is made per fold to prevent fit-state carryover.
            from sklearn.base import clone
            fitted = clone(model).fit(X[fit_idx], train.labels[fit_idx])
            predictions[validation_idx] = fitted.predict(X[validation_idx])
        model_scores[name] = {
            "train_cv_accuracy": float(accuracy_score(train.labels, predictions)),
            "train_cv_balanced_accuracy": float(
                balanced_accuracy_score(train.labels, predictions)),
        }
    # First candidate wins exact ties: a simpler raw-power baseline.
    selected = max(model_scores, key=lambda k: model_scores[k]["train_cv_accuracy"])
    fitted = candidates()[selected].fit(candidate_input(selected, train_features), train.labels)
    test_features = extract(test)
    predicted = fitted.predict(candidate_input(selected, test_features))
    p, stage, _ = Path(train_path).stem.split("_")
    scores = model_scores[selected]
    result = Result(
        patient=p, session=stage, selected_model=selected,
        train_trials=len(train.labels), test_trials=len(test.labels),
        train_cv_accuracy=scores["train_cv_accuracy"],
        train_cv_balanced_accuracy=scores["train_cv_balanced_accuracy"],
        test_accuracy=float(accuracy_score(test.labels, predicted)),
        test_balanced_accuracy=float(balanced_accuracy_score(test.labels, predicted)),
        correct_test_trials=int((predicted == test.labels).sum()))
    return result, model_scores


def evaluate_all(data_root, output_root, *, seed=27, n_splits=5):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results, selection = [], {}
    for patient, stage, train_path, test_path in session_paths(data_root):
        result, scores = evaluate_session(train_path, test_path,
                                           seed=seed, n_splits=n_splits)
        results.append(asdict(result))
        selection[f"{patient}_{stage}"] = scores
        print(f"{patient} {stage}: {result.selected_model} | "
              f"CV {result.train_cv_accuracy:.3f} | "
              f"test {result.correct_test_trials}/{result.test_trials} "
              f"({result.test_accuracy:.3f})", flush=True)
    with (output_root / "session_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (output_root / "train_cv_candidates.json").write_text(
        json.dumps({"seed": seed, "folds": n_splits,
                    "scores": selection}, indent=2) + "\n")
    return results

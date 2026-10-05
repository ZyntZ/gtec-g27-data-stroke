"""Unit tests for the organizer's max-over-time-of-average-trial contract."""
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stroke_rehab import organizer_metric as metric


def test_peak_is_max_of_trial_mean_not_per_trial_oracle():
    correctness = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.int8)
    assert metric.peak_of_mean(correctness, [3.5, 4.0]) == (0, 2, 0.5)
    assert np.max(correctness, axis=0).mean() == 1.0
    with pytest.raises(ValueError, match="binary"):
        metric.peak_of_mean(correctness.astype(float) / 2, [3.5, 4.0])
    with pytest.raises(ValueError, match="binary"):
        metric.peak_of_mean(correctness, [4.0, 3.5])
    with pytest.raises(ValueError, match="binary"):
        metric.peak_of_mean(np.ones((2, 4, 1)), [3.5, 4.0])


def test_fit_receives_only_training_labels_and_all_times_share_test_trial_ids(monkeypatch):
    y_train = np.r_[np.ones(40, dtype=np.int8), -np.ones(40, dtype=np.int8)]
    y_test = -y_train
    train = SimpleNamespace(signal=np.zeros((1, 16)), onsets=np.arange(80),
                            labels=y_train, fs=256)
    test = SimpleNamespace(signal=np.zeros((1, 16)), onsets=np.arange(80),
                           labels=y_test, fs=256)
    seen = []

    def fake_trials(recording, bands, *, causal):
        assert causal and bands == metric.BANDS
        return np.full((80, 3, 2, 16), 0 if recording is train else 1)

    class FakeEstimator:
        def fit(self, x, labels):
            assert len(x) == 80 and np.array_equal(labels, y_train)
            seen.append("fit")
            return self

        def predict(self, x):
            assert seen[-1] == "fit" and len(x) == 80
            seen.append("predict")
            return np.ones(len(x), dtype=np.int8)

    monkeypatch.setattr(metric, "filtered_trials", fake_trials)
    monkeypatch.setattr(metric, "_design", lambda name, trials, fs, window: trials)
    monkeypatch.setattr(metric, "_estimator", lambda name: FakeEstimator())
    rows, summary = metric.score_recording(train, test, "P1", "pre",
                                           stops=(3.5, 3.75), fixed_s=3.5)
    assert seen == ["fit", "predict", "fit", "predict"]
    assert len(rows) == 2 and all(row["correct"] == 40 for row in rows)
    assert summary["peak_time_s"] == 3.5 and summary["peak_minus_fixed_correct"] == 0
    assert all(row["left_correct"] == 40 and row["right_correct"] == 0 for row in rows)
    with pytest.raises(ValueError, match="Endpoints"):
        metric.score_recording(train, test, "P1", "pre", stops=(3.5,), fixed_s=4.0)


def test_pooled_peak_differs_from_sum_of_six_session_peaks():
    rows, summaries = [], []
    for index, (patient, stage) in enumerate(
            (('P1', 'pre'), ('P1', 'post'), ('P2', 'pre'),
             ('P2', 'post'), ('P3', 'pre'), ('P3', 'post'))):
        counts = (70, 45) if index < 3 else (45, 70)
        for stop, count in zip((3.5, 4.0), counts):
            rows.append(dict(patient=patient, session=stage,
                             decision_time_s=stop, correct=count, trials=80))
        summaries.append(dict(patient=patient, session=stage, fixed_time_s=3.5,
                              fixed_correct=counts[0], peak_correct=70, test_trials=80))
    pooled = metric.aggregate_session_scores(rows, summaries)
    assert pooled['fixed_correct'] == pooled['pooled_peak_correct'] == 345
    assert pooled['sum_of_session_peaks_correct'] == 420
    assert pooled['pooled_peak_time_s'] == 3.5  # deterministic earliest tie
    with pytest.raises(ValueError, match="same ordered time grid"):
        metric.aggregate_session_scores(rows[:-1], summaries)


def test_reference_rejects_duplicate_or_changed_historical_rows(tmp_path):
    source = tmp_path / 'decision_time.csv'
    base = dict(patient='P1', session='pre', model=metric.MODEL,
                decision_time_s='3.5', window_s='1.0',
                correct_test_trials='40', test_trials='80')
    def write(rows):
        with source.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(base))
            writer.writeheader()
            writer.writerows(rows)
    new = [dict(patient='P1', session='pre', model=metric.MODEL,
                decision_time_s=3.5, correct=40, trials=80)]
    write([base])
    metric._check_reference(new, source)
    write([base, base])
    with pytest.raises(ValueError, match='Duplicate'):
        metric._check_reference(new, source)
    write([{**base, 'correct_test_trials': '41'}])
    with pytest.raises(ValueError, match='mismatch'):
        metric._check_reference(new, source)


def test_saved_aggregate_and_source_receipt():
    # Run this check without private organizer recordings.
    root = Path(__file__).resolve().parents[1]
    out = root / 'results' / 'organizer_metric'
    import json
    import hashlib
    summary = list(csv.DictReader((out / 'organizer_metric_summary.csv').open()))
    times = list(csv.DictReader((out / 'organizer_metric_timecourse.csv').open()))
    aggregate = list(csv.DictReader((out / 'organizer_metric_aggregate.csv').open()))
    receipt = json.loads((out / 'organizer_metric_provenance.json').read_text())
    assert len(summary) == 6 and len(times) == 138 and len(aggregate) == 1
    assert int(aggregate[0]['pooled_peak_correct']) <= int(aggregate[0]['sum_of_session_peaks_correct'])
    assert len(receipt['input_sha256']) == 12 and receipt['historical_timecourse_parity']
    for name, digest in receipt['source_sha256'].items():
        assert hashlib.sha256((root / 'stroke_rehab' / name).read_bytes()).hexdigest() == digest
    for name, digest in receipt['output_sha256'].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest

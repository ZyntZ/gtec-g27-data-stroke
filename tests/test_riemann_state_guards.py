"""A rejected covariance or classifier error must not poison later trials."""
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from stroke_rehab.riemann import RecenteredTangentSpace


@pytest.fixture
def covariances():
    rng = np.random.default_rng(42)
    a = rng.normal(size=(12, 2, 3, 3))
    return a @ a.swapaxes(-1, -2) + np.eye(3), np.tile([-1, 1], 6)


@pytest.mark.parametrize('mode', ['nan', 'inf', 'asymmetric', 'non_spd', 'shape', 'empty'])
def test_bad_batch_is_atomic(covariances, mode):
    x, y = covariances
    model = RecenteredTangentSpace().fit(x[:8], y[:8])
    model.predict_stream(x[8:9])
    before = [r.copy() for r in model.references_], model.n_seen_
    bad = x[9:].copy()
    if mode in ('nan', 'inf'): bad[-1, 0, 0, 0] = float(mode)
    elif mode == 'asymmetric': bad[-1, 0, 0, 1] += 1
    elif mode == 'non_spd': bad[-1, 0] = -np.eye(3)
    elif mode == 'shape': bad = bad[:, :, :2, :2]
    else: bad = bad[:0]
    with pytest.raises(ValueError): model.predict_stream(bad)
    assert model.n_seen_ == before[1]
    for a, b in zip(before[0], model.references_): np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('method', ['predict_next', 'decision_next', 'predict_stream'])
def test_classifier_error_rolls_back_state(covariances, monkeypatch, method):
    x, y = covariances
    model = RecenteredTangentSpace().fit(x[:8], y[:8])
    before = [r.copy() for r in model.references_]
    calls = 0
    original = model.model_.predict
    def fail_later(a):
        nonlocal calls
        calls += 1
        if calls == (2 if method == 'predict_stream' else 1): raise RuntimeError('classifier failure')
        return original(a)
    monkeypatch.setattr(model.model_, 'decision_function' if method == 'decision_next' else 'predict', fail_later)
    with pytest.raises(RuntimeError): getattr(model, method)(x[8:] if method == 'predict_stream' else x[8])
    assert model.n_seen_ == 0
    for a, b in zip(before, model.references_): np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('prior', [-1, float('nan'), float('inf')])
def test_invalid_prior_rejected_before_fitting(covariances, prior):
    x, y = covariances
    with pytest.raises(ValueError, match='prior_trials'): RecenteredTangentSpace(prior_trials=prior).fit(x, y)


def test_unfitted_prediction_has_clear_error(covariances):
    x, _ = covariances
    with pytest.raises(NotFittedError): RecenteredTangentSpace().predict(x)

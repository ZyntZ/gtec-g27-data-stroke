import numpy as np
import pytest
from sklearn.base import clone

from stroke_rehab.models import FilterBankCSP, candidates


def test_csp_transform_shape_and_finite_values():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((20, 3, 16, 160)).astype('float32')
    labels = np.tile([1, -1], 10)
    X[labels == 1, :, :2] *= 3.0
    model = FilterBankCSP()
    transformed = model.fit_transform(X, labels)
    assert transformed.shape == (20, 12)
    assert np.isfinite(transformed).all()
    assert np.allclose(transformed, clone(model).fit_transform(X, labels))


def test_csp_requires_two_classes():
    with pytest.raises(ValueError, match="two labels"):
        FilterBankCSP().fit(np.zeros((8, 3, 16, 100)), np.ones(8))


def test_candidates_are_unfitted_until_train():
    for candidate in candidates().values():
        assert not hasattr(candidate[-1], 'classes_')

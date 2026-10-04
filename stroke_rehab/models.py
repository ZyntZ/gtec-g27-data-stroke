"""Two deliberately small, auditable candidate models; fit on training trials only."""

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Regularized common spatial patterns (CSP), trained inside each CV fold.

    Per-trial covariance matrices are normalized by trace to reduce the
    influence of gross amplitude differences. Four filters per frequency
    band (two at each end of the generalized eigenvalue spectrum).
    """

    def __init__(self, n_pairs=2, ridge=0.01):
        self.n_pairs = n_pairs
        self.ridge = ridge

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 4 or X.shape[0] != len(y) or len(np.unique(y)) != 2:
            raise ValueError("Expected trial x band x channel x time and two labels")
        _, bands, channels, _ = X.shape
        if not 0 < self.n_pairs <= channels // 2 or not 0 < self.ridge < 1:
            raise ValueError("Invalid CSP settings")
        self.filters_ = []
        for band in range(bands):
            xb = X[:, band].astype(np.float64)
            xb -= xb.mean(axis=-1, keepdims=True)
            cov = xb @ xb.transpose(0, 2, 1) / (xb.shape[-1] - 1)
            cov /= np.maximum(np.trace(cov, axis1=1, axis2=2)[:, None, None], 1e-30)
            left = cov[y == 1].mean(axis=0)
            right = cov[y == -1].mean(axis=0)
            # Common covariance scale is one (trace sum == 2).
            regularizer = self.ridge / channels * np.eye(channels)
            eigenvalues, eigenvectors = eigh(left + regularizer,
                                             left + right + 2 * regularizer)
            order = np.r_[np.arange(self.n_pairs),
                          np.arange(channels - self.n_pairs, channels)]
            self.filters_.append(eigenvectors[:, order])
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim != 4 or X.shape[1] != len(self.filters_):
            raise ValueError("CSP input shape does not match fitted bands")
        output = []
        for band, filt in enumerate(self.filters_):
            projected = np.einsum('nct,cf->nft', X[:, band], filt)
            var = np.var(projected, axis=-1, dtype=np.float64)
            # Relative variances mitigate global gain shifts across trials.
            var /= np.maximum(var.sum(axis=1, keepdims=True), 1e-30)
            output.append(np.log(np.maximum(var, 1e-30)))
        return np.concatenate(output, axis=1)


def candidates():
    """Fixed candidates, fixed order for deterministic CV tie breaking."""
    return {
        "bandpower_shrinkage_lda": make_pipeline(
            StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        "filterbank_csp_shrinkage_lda": make_pipeline(
            FilterBankCSP(), StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    }


def candidate_input(name, features):
    return features.epochs if name.startswith("filterbank_csp") else features.power

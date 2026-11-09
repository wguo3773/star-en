"""Fold-local imputation and scaling, without feature screening."""
import numpy as np


def matrix(X):
    a = np.asarray(X, dtype=float)
    if a.ndim != 2 or min(a.shape) < 1 or np.any(np.isinf(a)):
        raise ValueError("X must be a nonempty numeric matrix; NaN is allowed, infinity is not")
    return a


class Preprocessor:
    def fit(self, X):
        self.medians_ = np.array([np.median(v[~np.isnan(v)]) if np.any(~np.isnan(v))
                                  else 0.0 for v in X.T])
        filled = np.where(np.isnan(X), self.medians_, X)
        self.means_ = np.mean(filled, axis=0)
        self.scales_ = np.std(filled, axis=0)
        self.scales_[self.scales_ < 1e-12] = 1.0
        return self

    def transform(self, X):
        return (np.where(np.isnan(X), self.medians_, X) - self.means_) / self.scales_

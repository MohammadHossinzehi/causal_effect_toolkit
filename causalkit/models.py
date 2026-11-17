"""Small, dependable nuisance models: ridge OLS and logistic regression.

Both add an intercept column themselves. The logistic model is fitted with
Newton-Raphson (IRLS) plus a tiny ridge penalty, which keeps it well defined
under quasi separation, a common situation for propensity models when one
covariate strongly predicts treatment.
"""

from __future__ import annotations

import numpy as np


def _design(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return np.column_stack([np.ones(len(X)), X])


class LinearRegression:
    def __init__(self, ridge: float = 0.0):
        self.ridge = float(ridge)
        self.coef_: np.ndarray | None = None

    def fit(self, X, y) -> "LinearRegression":
        A = _design(X)
        y = np.asarray(y, dtype=float)
        P = self.ridge * np.eye(A.shape[1])
        P[0, 0] = 0.0  # never shrink the intercept
        self.coef_ = np.linalg.solve(A.T @ A + P, A.T @ y) if self.ridge else np.linalg.lstsq(A, y, rcond=None)[0]
        return self

    def predict(self, X) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model is not fitted")
        return _design(X) @ self.coef_


class LogisticRegression:
    def __init__(self, ridge: float = 1e-6, max_iter: int = 100, tol: float = 1e-10):
        self.ridge = float(ridge)
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: np.ndarray | None = None
        self.n_iter_ = 0

    def fit(self, X, t) -> "LogisticRegression":
        A = _design(X)
        t = np.asarray(t, dtype=float)
        if not np.all((t == 0) | (t == 1)):
            raise ValueError("treatment must be binary 0/1")
        w = np.zeros(A.shape[1])
        P = self.ridge * np.eye(A.shape[1])
        P[0, 0] = 0.0
        for i in range(1, self.max_iter + 1):
            eta = np.clip(A @ w, -35, 35)
            p = 1.0 / (1.0 + np.exp(-eta))
            grad = A.T @ (t - p) - P @ w
            H = (A * (p * (1 - p))[:, None]).T @ A + P
            step = np.linalg.solve(H + 1e-12 * np.eye(len(w)), grad)
            w = w + step
            self.n_iter_ = i
            if np.max(np.abs(step)) < self.tol:
                break
        self.coef_ = w
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model is not fitted")
        eta = np.clip(_design(X) @ self.coef_, -35, 35)
        return 1.0 / (1.0 + np.exp(-eta))

"""Treatment effect estimators for a binary treatment under no unmeasured confounding.

Every estimator returns an ``Effect`` with a point estimate, a standard error
and a 95% normal interval. Standard errors come from each estimator's own
theory rather than one generic bootstrap:

* difference in means: Neyman's conservative variance
* g-computation: nonparametric bootstrap (the plug in has no simple closed form)
* IPW (Hajek): the empirical influence function, treating the propensity as
  known, which is conservative for the ATE (Lunceford and Davidian 2004)
* AIPW: the efficient influence function with K fold cross fitting
* matching: the Abadie and Imbens (2006) variance, because the bootstrap is
  known to be invalid for nearest neighbour matching with replacement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .models import LinearRegression, LogisticRegression

Z95 = 1.959963984540054


@dataclass
class Effect:
    method: str
    estimand: str
    estimate: float
    se: float
    n: int
    extra: dict = field(default_factory=dict)

    @property
    def ci(self) -> tuple:
        return (self.estimate - Z95 * self.se, self.estimate + Z95 * self.se)

    def covers(self, truth: float) -> bool:
        lo, hi = self.ci
        return lo <= truth <= hi

    def __str__(self) -> str:
        lo, hi = self.ci
        return f"{self.method:<22} {self.estimand}  {self.estimate:+.4f}  (se {self.se:.4f}, 95% CI {lo:+.4f} to {hi:+.4f})"


def _arrays(X, t, y):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if not (len(X) == len(t) == len(y)):
        raise ValueError("X, t and y must have the same number of rows")
    if not np.all((t == 0) | (t == 1)):
        raise ValueError("treatment must be binary 0/1")
    if t.sum() < 2 or (1 - t).sum() < 2:
        raise ValueError("need at least two treated and two control units")
    return X, t, y


def _check_estimand(estimand: str) -> str:
    e = estimand.upper()
    if e not in ("ATE", "ATT"):
        raise ValueError("estimand must be 'ATE' or 'ATT'")
    return e


def propensity_scores(X, t, ridge: float = 1e-6) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return LogisticRegression(ridge=ridge).fit(X, t).predict_proba(X)


# ---------------------------------------------------------------------------
def difference_in_means(t, y) -> Effect:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    y1, y0 = y[t == 1], y[t == 0]
    est = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return Effect("difference in means", "ATE", float(est), float(se), len(y))


def g_computation(X, t, y, estimand: str = "ATE", n_boot: int = 200, seed: Optional[int] = 0) -> Effect:
    """Outcome regression per arm (a T learner), then average the imputed contrasts."""
    X, t, y = _arrays(X, t, y)
    estimand = _check_estimand(estimand)

    def point(Xb, tb, yb):
        m1 = LinearRegression().fit(Xb[tb == 1], yb[tb == 1])
        m0 = LinearRegression().fit(Xb[tb == 0], yb[tb == 0])
        rows = Xb if estimand == "ATE" else Xb[tb == 1]
        return float(np.mean(m1.predict(rows) - m0.predict(rows)))

    est = point(X, t, y)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(y)
    while len(boots) < n_boot:
        idx = rng.integers(0, n, n)
        tb = t[idx]
        if tb.sum() <= X.shape[1] + 1 or (1 - tb).sum() <= X.shape[1] + 1:
            continue
        boots.append(point(X[idx], tb, y[idx]))
    se = float(np.std(boots, ddof=1)) if n_boot > 1 else float("nan")
    return Effect("g-computation", estimand, est, se, n, {"n_boot": n_boot})


def ipw(X, t, y, estimand: str = "ATE", trim: float = 0.0, ps: Optional[np.ndarray] = None) -> Effect:
    """Normalised (Hajek) inverse probability weighting.

    ``trim`` drops units whose propensity lies outside [trim, 1 - trim]
    (Crump et al. style), which changes the target population, so the
    number of retained units is reported in ``extra``.
    """
    X, t, y = _arrays(X, t, y)
    estimand = _check_estimand(estimand)
    e = propensity_scores(X, t) if ps is None else np.asarray(ps, dtype=float)
    keep = (e >= trim) & (e <= 1 - trim)
    X, t, y, e = X[keep], t[keep], y[keep], e[keep]
    n = len(y)
    if estimand == "ATE":
        w1, w0 = t / e, (1 - t) / (1 - e)
    else:
        w1, w0 = t, (1 - t) * e / (1 - e)
    mu1 = np.sum(w1 * y) / np.sum(w1)
    mu0 = np.sum(w0 * y) / np.sum(w0)
    est = mu1 - mu0
    # influence function of the Hajek ratio, scaled to the mean weight
    if1 = w1 * (y - mu1) / np.mean(w1)
    if0 = w0 * (y - mu0) / np.mean(w0)
    se = np.std(if1 - if0, ddof=1) / np.sqrt(n)
    return Effect("IPW (Hajek)", estimand, float(est), float(se), n, {"kept": int(n), "trim": trim})


def aipw(X, t, y, estimand: str = "ATE", folds: int = 5, seed: Optional[int] = 0, clip: float = 1e-3,
         ps_features=None) -> Effect:
    """Augmented IPW (doubly robust) with K fold cross fitting.

    Consistent if either the propensity model or the outcome model is right,
    and semiparametrically efficient when both are. ``ps_features`` lets the
    propensity model use a different design matrix from the outcome model,
    which is how the double robustness tests misspecify one model at a time.
    """
    X, t, y = _arrays(X, t, y)
    W = X if ps_features is None else np.asarray(ps_features, dtype=float).reshape(len(y), -1)
    estimand = _check_estimand(estimand)
    n = len(y)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(n) % max(1, folds)
    e = np.empty(n)
    m1 = np.empty(n)
    m0 = np.empty(n)
    for k in range(max(1, folds)):
        test = fold == k
        train = ~test if folds > 1 else test
        Xtr, ttr, ytr = X[train], t[train], y[train]
        e[test] = LogisticRegression().fit(W[train], ttr).predict_proba(W[test])
        m1[test] = LinearRegression().fit(Xtr[ttr == 1], ytr[ttr == 1]).predict(X[test])
        m0[test] = LinearRegression().fit(Xtr[ttr == 0], ytr[ttr == 0]).predict(X[test])
    e = np.clip(e, clip, 1 - clip)
    if estimand == "ATE":
        phi = m1 - m0 + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
        est = phi.mean()
        psi = phi - est
    else:
        p = t.mean()
        phi = (t * (y - m0) - (1 - t) * e / (1 - e) * (y - m0)) / p
        est = phi.mean()
        psi = phi - est * t / p
    se = np.std(psi, ddof=1) / np.sqrt(n)
    return Effect("AIPW (cross-fit)", estimand, float(est), float(se), n, {"folds": folds})


def match_att(X, t, y, caliper: Optional[float] = None, on: str = "logit_ps") -> Effect:
    """1:1 nearest neighbour matching with replacement for the ATT.

    ``on`` is ``"logit_ps"`` (match on the estimated propensity logit) or
    ``"mahalanobis"`` (match on the covariates directly). ``caliper`` is in
    standard deviations of the matching score; treated units without a
    control inside the caliper are dropped and counted in ``extra``.
    """
    X, t, y = _arrays(X, t, y)
    if on == "logit_ps":
        e = np.clip(propensity_scores(X, t), 1e-9, 1 - 1e-9)
        S = np.log(e / (1 - e))[:, None]
    elif on == "mahalanobis":
        cov = np.atleast_2d(np.cov(X, rowvar=False))
        L = np.linalg.cholesky(np.linalg.inv(cov + 1e-9 * np.eye(len(cov))))
        S = X @ L
    else:
        raise ValueError("on must be 'logit_ps' or 'mahalanobis'")
    tr = np.flatnonzero(t == 1)
    co = np.flatnonzero(t == 0)
    D = np.sqrt(((S[tr][:, None, :] - S[co][None, :, :]) ** 2).sum(-1))
    j = D.argmin(1)
    dist = D[np.arange(len(tr)), j]
    ok = np.ones(len(tr), bool)
    if caliper is not None:
        scale = np.std(S[:, 0]) if S.shape[1] == 1 else 1.0
        ok = dist <= caliper * scale
    if ok.sum() < 2:
        raise ValueError("fewer than two treated units matched; widen the caliper")
    tr_ok = tr[ok]
    matched = co[j[ok]]
    diffs = y[tr_ok] - y[matched]
    est = diffs.mean()
    n1 = len(tr_ok)
    # Abadie-Imbens: K_j = times control j is used; sigma^2 from matching each
    # control to its nearest other control.
    K = np.bincount(np.searchsorted(co, matched), minlength=len(co)).astype(float)
    Dc = np.sqrt(((S[co][:, None, :] - S[co][None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(Dc, np.inf)
    nn = Dc.argmin(1)
    sigma2 = 0.5 * (y[co] - y[co][nn]) ** 2
    var = (np.sum((diffs - est) ** 2) + np.sum(K * (K - 1) * sigma2)) / n1**2
    return Effect(
        f"matching ({on})",
        "ATT",
        float(est),
        float(np.sqrt(var)),
        int(n1 + len(np.unique(matched))),
        {"matched_treated": int(n1), "dropped_treated": int((~ok).sum()), "distinct_controls": int(len(np.unique(matched)))},
    )

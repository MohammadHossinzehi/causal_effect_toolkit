"""Balance and overlap diagnostics, plus E-value sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


def standardized_mean_differences(X, t, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Absolute standardised mean difference per covariate.

    The denominator always uses the unweighted pooled SD so that values
    before and after weighting are on the same scale (Austin 2009). The
    common rule of thumb treats |SMD| < 0.1 as balanced.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    t = np.asarray(t, dtype=float)
    w = np.ones(len(t)) if weights is None else np.asarray(weights, dtype=float)
    a, b = t == 1, t == 0
    m1 = np.average(X[a], axis=0, weights=w[a])
    m0 = np.average(X[b], axis=0, weights=w[b])
    sd = np.sqrt((X[a].var(axis=0, ddof=1) + X[b].var(axis=0, ddof=1)) / 2)
    sd = np.where(sd > 0, sd, 1.0)
    return np.abs(m1 - m0) / sd


def ipw_weights(ps, t, estimand: str = "ATE") -> np.ndarray:
    ps = np.asarray(ps, dtype=float)
    t = np.asarray(t, dtype=float)
    if estimand.upper() == "ATE":
        return t / ps + (1 - t) / (1 - ps)
    if estimand.upper() == "ATT":
        return t + (1 - t) * ps / (1 - ps)
    raise ValueError("estimand must be 'ATE' or 'ATT'")


def effective_sample_size(w) -> float:
    """Kish's effective sample size, (sum w)^2 / sum w^2."""
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / np.sum(w**2))


@dataclass
class OverlapReport:
    treated_range: tuple
    control_range: tuple
    common_support: tuple
    outside_support: int
    extreme: int  # scores outside [0.05, 0.95]

    def __str__(self) -> str:
        return (
            f"propensity range treated {self.treated_range[0]:.3f} to {self.treated_range[1]:.3f}, "
            f"control {self.control_range[0]:.3f} to {self.control_range[1]:.3f}; "
            f"{self.outside_support} units outside common support, {self.extreme} with extreme scores"
        )


def overlap(ps, t) -> OverlapReport:
    ps = np.asarray(ps, dtype=float)
    t = np.asarray(t, dtype=float)
    p1, p0 = ps[t == 1], ps[t == 0]
    lo, hi = max(p1.min(), p0.min()), min(p1.max(), p0.max())
    return OverlapReport(
        (float(p1.min()), float(p1.max())),
        (float(p0.min()), float(p0.max())),
        (float(lo), float(hi)),
        int(np.sum((ps < lo) | (ps > hi))),
        int(np.sum((ps < 0.05) | (ps > 0.95))),
    )


def balance_table(X, t, ps, names: Optional[Sequence[str]] = None, estimand: str = "ATE") -> List[tuple]:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    names = list(names) if names is not None else [f"x{i}" for i in range(X.shape[1])]
    before = standardized_mean_differences(X, t)
    after = standardized_mean_differences(X, t, ipw_weights(ps, t, estimand))
    return [(n, float(b), float(a)) for n, b, a in zip(names, before, after)]


# --------------------------------------------------------------------------
def e_value_rr(rr: float, ci_limit: Optional[float] = None) -> tuple:
    """E-value for a risk ratio (VanderWeele and Ding 2017).

    The minimum strength of association, on the risk ratio scale, that an
    unmeasured confounder would need with both treatment and outcome to
    fully explain away the observed estimate. Returns ``(point, ci)``; the
    CI E-value uses the limit closest to 1 and is 1 if the CI crosses 1.
    """
    if rr <= 0:
        raise ValueError("risk ratio must be positive")

    def ev(r: float) -> float:
        r = 1 / r if r < 1 else r
        return float(r + np.sqrt(r * (r - 1)))

    point = ev(rr)
    if ci_limit is None:
        return point, None
    if (rr >= 1 and ci_limit <= 1) or (rr < 1 and ci_limit >= 1):
        return point, 1.0
    return point, ev(ci_limit)


def e_value_smd(d: float, se: Optional[float] = None) -> tuple:
    """E-value for a standardised mean difference via RR ~ exp(0.91 d)."""
    rr = float(np.exp(0.91 * abs(d)))
    limit = None if se is None else float(np.exp(0.91 * (abs(d) - 1.959963984540054 * se)))
    return e_value_rr(rr, limit)

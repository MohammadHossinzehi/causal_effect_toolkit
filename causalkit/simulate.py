"""Synthetic data with known ground truth, for benchmarks and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .dag import DAG


@dataclass
class Simulation:
    X: np.ndarray
    t: np.ndarray
    y: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    ps: np.ndarray

    @property
    def true_ate(self) -> float:
        return float(np.mean(self.y1 - self.y0))

    @property
    def true_att(self) -> float:
        return float(np.mean((self.y1 - self.y0)[self.t == 1]))


def confounded(n: int = 2000, seed: Optional[int] = 0, effect: float = 2.0, heterogeneity: float = 1.0,
               confounding: float = 1.0, noise: float = 1.0) -> Simulation:
    """Three confounders drive both treatment and outcome.

    Units with large x0 are both likelier to be treated and have higher
    baseline outcomes, so the naive difference in means is biased upward.
    The effect varies with x1, which makes the ATT differ from the ATE.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    logit = confounding * (0.9 * X[:, 0] + 0.6 * X[:, 1] - 0.4 * X[:, 2])
    ps = 1 / (1 + np.exp(-logit))
    t = (rng.random(n) < ps).astype(float)
    base = 1.0 + 2.0 * X[:, 0] + 1.0 * X[:, 1] - 0.5 * X[:, 2]
    tau = effect + heterogeneity * X[:, 1]
    eps = noise * rng.normal(size=n)
    y0 = base + eps
    y1 = base + tau + eps
    y = np.where(t == 1, y1, y0)
    return Simulation(X, t, y, y0, y1, ps)


def linear_sem(dag: DAG, n: int, coefs: Dict[tuple, float], binary: tuple = (), seed: Optional[int] = 0,
               noise: float = 1.0, intervene: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
    """Sample a linear structural equation model over ``dag``.

    Nodes listed in ``binary`` are thresholded through a logistic link.
    ``intervene`` implements Pearl's do() operator by replacing a node's
    equation with a constant, so the true causal effect can be read off
    by simulation rather than trusted from algebra.
    """
    rng = np.random.default_rng(seed)
    intervene = intervene or {}
    data: Dict[str, np.ndarray] = {}
    for v in dag.topological_order():
        if v in intervene:
            data[v] = np.full(n, float(intervene[v]))
            continue
        lin = noise * rng.normal(size=n)
        for p in dag.parents(v):
            lin = lin + coefs.get((p, v), 0.0) * data[p]
        if v in binary:
            data[v] = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(float)
        else:
            data[v] = lin
    return data

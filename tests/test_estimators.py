import numpy as np
import pytest

from causalkit import (
    LinearRegression,
    LogisticRegression,
    aipw,
    confounded,
    difference_in_means,
    g_computation,
    ipw,
    match_att,
)


def test_difference_in_means_is_fine_in_a_randomised_trial():
    rng = np.random.default_rng(0)
    n = 20_000
    t = (rng.random(n) < 0.5).astype(float)
    y = 1 + 3 * t + rng.normal(size=n)
    r = difference_in_means(t, y)
    assert r.covers(3.0)
    assert abs(r.se - np.sqrt(2 / (n / 2))) < 0.003


def test_naive_estimate_is_biased_under_confounding():
    sim = confounded(n=5000, seed=1)
    assert difference_in_means(sim.t, sim.y).estimate - sim.true_ate > 1.5


@pytest.mark.parametrize("estimand", ["ATE", "ATT"])
def test_adjusted_estimators_recover_truth(estimand):
    sim = confounded(n=8000, seed=2)
    truth = sim.true_ate if estimand == "ATE" else sim.true_att
    for f in (g_computation, ipw, aipw):
        r = f(sim.X, sim.t, sim.y, estimand=estimand)
        assert r.estimand == estimand
        assert abs(r.estimate - truth) < 4 * r.se + 0.02, (f.__name__, r, truth)


def test_att_and_ate_differ_under_heterogeneity():
    sim = confounded(n=8000, seed=3, heterogeneity=1.5)
    a = aipw(sim.X, sim.t, sim.y, estimand="ATE").estimate
    b = aipw(sim.X, sim.t, sim.y, estimand="ATT").estimate
    assert b - a > 0.3


def test_matching_recovers_att_both_metrics():
    sim = confounded(n=3000, seed=4)
    for on in ("logit_ps", "mahalanobis"):
        r = match_att(sim.X, sim.t, sim.y, on=on)
        assert abs(r.estimate - sim.true_att) < 4 * r.se, (on, r, sim.true_att)
        assert r.extra["matched_treated"] == int(sim.t.sum())


def test_caliper_drops_poorly_matched_units():
    sim = confounded(n=2000, seed=5, confounding=2.5)
    loose = match_att(sim.X, sim.t, sim.y)
    tight = match_att(sim.X, sim.t, sim.y, caliper=0.01)
    assert tight.extra["dropped_treated"] > loose.extra["dropped_treated"] == 0


def _nonlinear_outcome(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    ps = 1 / (1 + np.exp(-(1.2 * X[:, 0] - 0.5 * X[:, 1])))
    t = (rng.random(n) < ps).astype(float)
    y = 2 * np.exp(X[:, 0]) + X[:, 1] ** 2 + 1.0 * t + rng.normal(size=n)
    return X, t, y


def test_double_robustness_outcome_model_wrong():
    """Linear outcome models are wrong here; the logistic propensity is right."""
    X, t, y = _nonlinear_outcome(40_000, 6)
    g = g_computation(X, t, y, n_boot=20)
    dr = aipw(X, t, y)
    assert abs(g.estimate - 1.0) > max(0.15, 5 * g.se)
    assert abs(dr.estimate - 1.0) < 4 * dr.se + 0.02


def test_double_robustness_propensity_model_wrong():
    """Propensity depends on x0 squared, which a linear logit cannot express."""
    rng = np.random.default_rng(7)
    n = 40_000
    X = rng.normal(size=(n, 2))
    ps = 1 / (1 + np.exp(-(1.5 * X[:, 0] ** 2 - 1.5 + 0.5 * X[:, 1])))
    t = (rng.random(n) < ps).astype(float)
    y = 2 * X[:, 0] ** 2 + X[:, 1] + 1.0 * t + rng.normal(size=n)
    # outcome features are right (x0^2, x1); propensity features are raw x, so wrong
    feats = np.column_stack([X[:, 0] ** 2, X[:, 1]])
    ps_wrong = LogisticRegression().fit(X, t).predict_proba(X)
    bad = ipw(X, t, y, ps=ps_wrong)
    dr = aipw(feats, t, y, ps_features=X)
    assert abs(bad.estimate - 1.0) > 0.3
    assert abs(dr.estimate - 1.0) < 4 * dr.se + 0.02


def test_aipw_confidence_interval_coverage():
    hits = 0
    reps = 120
    for s in range(reps):
        sim = confounded(n=800, seed=100 + s)
        hits += aipw(sim.X, sim.t, sim.y, folds=2, seed=s).covers(sim.true_ate)
    assert 0.88 <= hits / reps <= 1.0


def test_ipw_trimming_reports_kept_units():
    sim = confounded(n=3000, seed=8, confounding=2.0)
    r = ipw(sim.X, sim.t, sim.y, trim=0.05)
    assert r.extra["kept"] < 3000


def test_input_validation():
    X = np.zeros((4, 1))
    with pytest.raises(ValueError):
        aipw(X, [0, 1, 2, 1], [1, 2, 3, 4])
    with pytest.raises(ValueError):
        ipw(X, [0, 0, 0, 1], [1, 2, 3, 4])
    with pytest.raises(ValueError):
        ipw(np.zeros((5, 1)), [0, 1, 0, 1, 0], [1, 2, 3, 4, 5], estimand="CATE")


def test_logistic_regression_recovers_coefficients():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(50_000, 2))
    p = 1 / (1 + np.exp(-(-0.5 + 1.0 * X[:, 0] - 2.0 * X[:, 1])))
    t = (rng.random(len(p)) < p).astype(float)
    m = LogisticRegression().fit(X, t)
    assert np.allclose(m.coef_, [-0.5, 1.0, -2.0], atol=0.05)
    assert m.n_iter_ < 20


def test_logistic_survives_separation():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    t = np.array([0, 0, 1, 1.0])
    p = LogisticRegression(ridge=1e-3).fit(X, t).predict_proba(X)
    assert np.all(np.isfinite(p)) and p[0] < 0.1 and p[-1] > 0.9


def test_linear_regression_matches_numpy():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(200, 3))
    y = X @ [1.0, -2.0, 0.5] + 3 + rng.normal(size=200)
    m = LinearRegression().fit(X, y)
    ref = np.linalg.lstsq(np.column_stack([np.ones(200), X]), y, rcond=None)[0]
    assert np.allclose(m.coef_, ref)

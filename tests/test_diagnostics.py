import numpy as np
import pytest

from causalkit import (
    balance_table,
    confounded,
    e_value_rr,
    e_value_smd,
    effective_sample_size,
    ipw_weights,
    overlap,
    propensity_scores,
    standardized_mean_differences,
)
from causalkit.__main__ import main


def test_e_value_matches_published_example():
    # VanderWeele and Ding (2017): RR 3.9 with lower limit 1.8 -> 7.26 and 3.0
    pt, ci = e_value_rr(3.9, 1.8)
    assert pt == pytest.approx(7.26, abs=0.01)
    assert ci == pytest.approx(3.0, abs=0.01)


def test_e_value_protective_and_null_crossing():
    pt, ci = e_value_rr(0.5, 1.2)
    assert pt == pytest.approx(2 + np.sqrt(2))
    assert ci == 1.0
    assert e_value_rr(1.0)[0] == 1.0
    with pytest.raises(ValueError):
        e_value_rr(0.0)


def test_e_value_smd_is_monotone():
    assert e_value_smd(0.2)[0] < e_value_smd(0.5)[0] < e_value_smd(1.0)[0]


def test_weighting_balances_covariates():
    sim = confounded(n=6000, seed=11)
    ps = propensity_scores(sim.X, sim.t)
    before = standardized_mean_differences(sim.X, sim.t)
    after = standardized_mean_differences(sim.X, sim.t, ipw_weights(ps, sim.t))
    assert before.max() > 0.3
    assert after.max() < 0.08
    rows = balance_table(sim.X, sim.t, ps, ["a", "b", "c"], estimand="ATT")
    assert [r[0] for r in rows] == ["a", "b", "c"]
    assert all(r[2] < r[1] for r in rows)


def test_effective_sample_size():
    assert effective_sample_size(np.ones(50)) == pytest.approx(50)
    assert effective_sample_size([1, 0, 0, 0]) == pytest.approx(1)


def test_overlap_report():
    t = np.array([1, 1, 0, 0, 0.0])
    ps = np.array([0.6, 0.99, 0.01, 0.3, 0.7])
    r = overlap(ps, t)
    assert r.common_support == (0.6, 0.7)
    assert r.outside_support == 3
    assert r.extreme == 2
    assert "outside common support" in str(r)


def test_ipw_weights_rejects_bad_estimand():
    with pytest.raises(ValueError):
        ipw_weights([0.5], [1], "ATC")


def test_cli_smoke(capsys, tmp_path):
    assert main(["demo", "--n", "1500"]) == 0
    assert "AIPW" in capsys.readouterr().out
    assert main(["adjust", "Z -> X; Z -> Y; X -> Y", "X", "Y"]) == 0
    assert "{Z}" in capsys.readouterr().out
    assert main(["adjust", "U -> X; U -> Y; X -> Y", "X", "Y", "--latent", "U"]) == 1
    capsys.readouterr()
    assert main(["dsep", "A -> B -> C", "A", "C", "--given", "B"]) == 0
    assert "_||_" in capsys.readouterr().out
    sim = confounded(n=500, seed=0)
    f = tmp_path / "d.csv"
    lines = ["x0,x1,x2,t,y"] + [",".join(f"{v:.6f}" for v in (*row, ti, yi)) for row, ti, yi in zip(sim.X, sim.t, sim.y)]
    f.write_text("\n".join(lines))
    assert main(["estimate", str(f), "--treatment", "t", "--outcome", "y"]) == 0
    assert "matching" in capsys.readouterr().out

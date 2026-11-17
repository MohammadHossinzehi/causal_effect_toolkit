"""Command line interface.

    python -m causalkit demo [--n 4000] [--seed 0]
    python -m causalkit adjust "Z -> X; Z -> Y; X -> Y" X Y [--latent U]
    python -m causalkit dsep "A -> B -> C" A C --given B
    python -m causalkit estimate data.csv --treatment t --outcome y
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from . import DAG, aipw, balance_table, confounded, difference_in_means, e_value_smd, g_computation, ipw
from . import match_att, overlap, propensity_scores


def _report(X, t, y, names, truth=None) -> None:
    ps = propensity_scores(X, t)
    print("Overlap:", overlap(ps, t))
    print("\nBalance (|SMD| before -> after ATE weighting)")
    for name, b, a in balance_table(X, t, ps, names):
        flag = "ok" if a < 0.1 else "CHECK"
        print(f"  {name:<12} {b:6.3f} -> {a:6.3f}  {flag}")
    print()
    results = [
        difference_in_means(t, y),
        g_computation(X, t, y),
        ipw(X, t, y),
        aipw(X, t, y),
        g_computation(X, t, y, estimand="ATT"),
        ipw(X, t, y, estimand="ATT"),
        aipw(X, t, y, estimand="ATT"),
        match_att(X, t, y),
    ]
    for r in results:
        line = str(r)
        if truth is not None:
            tv = truth[r.estimand]
            line += f"   bias {r.estimate - tv:+.3f}  {'covers' if r.covers(tv) else 'MISSES'}"
        print(line)
    best = results[3]
    pt, ci = e_value_smd(best.estimate / np.std(y), best.se / np.std(y))
    print(f"\nE-value (AIPW ATE): {pt:.2f} for the estimate, {ci:.2f} for the CI limit")


def cmd_demo(a) -> int:
    sim = confounded(n=a.n, seed=a.seed)
    print(f"Simulated confounded study: n={a.n}, true ATE {sim.true_ate:+.4f}, true ATT {sim.true_att:+.4f}\n")
    _report(sim.X, sim.t, sim.y, ["x0", "x1", "x2"], {"ATE": sim.true_ate, "ATT": sim.true_att})
    return 0


def cmd_adjust(a) -> int:
    g = DAG.parse(a.graph)
    sets = g.adjustment_sets(a.treatment, a.outcome, exclude=a.latent or [])
    if not sets:
        print(f"No backdoor adjustment set identifies {a.treatment} -> {a.outcome}.")
        return 1
    print(f"Minimal backdoor adjustment sets for {a.treatment} -> {a.outcome}:")
    for s in sets:
        print("  {" + ", ".join(sorted(s)) + "}")
    return 0


def cmd_dsep(a) -> int:
    g = DAG.parse(a.graph)
    sep = g.d_separated(a.x.split(","), a.y.split(","), a.given.split(",") if a.given else [])
    print(f"{a.x} {'_||_' if sep else 'not _||_'} {a.y} | {{{a.given or ''}}}")
    return 0


def cmd_estimate(a) -> int:
    with open(a.csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("empty file", file=sys.stderr)
        return 1
    covs = a.covariates.split(",") if a.covariates else [c for c in rows[0] if c not in (a.treatment, a.outcome)]
    X = np.array([[float(r[c]) for c in covs] for r in rows])
    t = np.array([float(r[a.treatment]) for r in rows])
    y = np.array([float(r[a.outcome]) for r in rows])
    _report(X, t, y, covs)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="causalkit")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run every estimator on simulated confounded data")
    d.add_argument("--n", type=int, default=4000)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=cmd_demo)
    ad = sub.add_parser("adjust", help="find minimal backdoor adjustment sets")
    ad.add_argument("graph")
    ad.add_argument("treatment")
    ad.add_argument("outcome")
    ad.add_argument("--latent", nargs="*", help="unobserved nodes that cannot be adjusted for")
    ad.set_defaults(fn=cmd_adjust)
    ds = sub.add_parser("dsep", help="test d-separation")
    ds.add_argument("graph")
    ds.add_argument("x")
    ds.add_argument("y")
    ds.add_argument("--given", default="")
    ds.set_defaults(fn=cmd_dsep)
    es = sub.add_parser("estimate", help="estimate effects from a CSV file")
    es.add_argument("csv")
    es.add_argument("--treatment", required=True)
    es.add_argument("--outcome", required=True)
    es.add_argument("--covariates", default="")
    es.set_defaults(fn=cmd_estimate)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

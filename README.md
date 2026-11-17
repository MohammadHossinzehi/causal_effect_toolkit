# causalkit

Causal effect estimation and identification from first principles, in plain NumPy.

Correlation tells you that treated patients did better. Causal inference asks whether they did better *because* of the treatment, or because the people who got treated were healthier to begin with. causalkit covers the two halves of that question:

1. **Identification.** Given a causal DAG, which variables do you need to adjust for? causalkit implements d separation (twice, with two independent algorithms), Pearl's backdoor criterion and a search for every minimal valid adjustment set, including the cases where you must *not* adjust (colliders, M bias, mediators) and where no backdoor set exists at all (front door graphs with a latent confounder).
2. **Estimation.** Given observational data and an adjustment set, how big is the effect? causalkit ships five estimators for the ATE and ATT of a binary treatment, each with a standard error derived from that estimator's own theory, plus balance diagnostics and E value sensitivity analysis.

The only runtime dependency is NumPy. The logistic and linear nuisance models are written here too.

## Quick start

```bash
pip install numpy pytest
python -m causalkit demo            # every estimator on simulated confounded data
python -m pytest -q                 # 38 tests, about two seconds
```

Or install it as a package with a `causalkit` command:

```bash
pip install -e ".[test]"
causalkit demo --n 4000
```

The demo simulates a study where the units most likely to be treated also have the highest baseline outcome, then shows how each estimator does against the known truth:

```
Simulated confounded study: n=4000, true ATE +2.0071, true ATT +2.2411

Balance (|SMD| before -> after ATE weighting)
  x0            0.790 ->  0.027  ok
  x1            0.478 ->  0.033  ok
  x2            0.276 ->  0.011  ok

difference in means    ATE  +4.3089  (se 0.0784, 95% CI +4.1551 to +4.4626)   bias +2.302  MISSES
g-computation          ATE  +1.9946  (se 0.0389, 95% CI +1.9183 to +2.0708)   bias -0.013  covers
IPW (Hajek)            ATE  +2.1219  (se 0.1161, 95% CI +1.8943 to +2.3495)   bias +0.115  covers
AIPW (cross-fit)       ATE  +2.0015  (se 0.0423, 95% CI +1.9186 to +2.0845)   bias -0.006  covers
g-computation          ATT  +2.2540  (se 0.0430, 95% CI +2.1697 to +2.3383)   bias +0.013  covers
IPW (Hajek)            ATT  +2.3113  (se 0.0985, 95% CI +2.1183 to +2.5044)   bias +0.070  covers
AIPW (cross-fit)       ATT  +2.2497  (se 0.0451, 95% CI +2.1613 to +2.3380)   bias +0.009  covers
matching (logit_ps)    ATT  +2.2921  (se 0.0565, 95% CI +2.1814 to +2.4028)   bias +0.051  covers

E-value (AIPW ATE): 2.88 for the estimate, 2.80 for the CI limit
```

The naive comparison more than doubles the true effect. Every adjusted estimator recovers it.

## Command line

```bash
# Which variables should I adjust for? U is unobserved.
causalkit adjust "U -> X; U -> Z; Z -> Y; X -> Y; X -> M; M -> Y; X -> C; Y -> C" X Y --latent U
#   Minimal backdoor adjustment sets for X -> Y:
#     {Z}

# Is A independent of C given B?  (a collider, so conditioning opens the path)
causalkit dsep "A -> B; C -> B" A C --given B
#   A not _||_ C | {B}

# Run the full report on your own data (every non treatment, non outcome column is a covariate by default)
causalkit estimate study.csv --treatment treated --outcome income --covariates age,education,prior_income
```

## Library

```python
import causalkit as ck

g = ck.DAG.parse("U1 -> X; U1 -> M; U2 -> M; U2 -> Y; X -> Y")   # the M bias graph
g.satisfies_backdoor("X", "Y", set())      # True: adjusting for nothing is fine
g.satisfies_backdoor("X", "Y", {"M"})      # False: M is a collider, adjusting opens a path
g.adjustment_sets("X", "Y")                # [frozenset()]

sim = ck.confounded(n=5000, seed=1)
r = ck.aipw(sim.X, sim.t, sim.y, estimand="ATE", folds=5)
print(r, r.ci, r.covers(sim.true_ate))

ps = ck.propensity_scores(sim.X, sim.t)
print(ck.overlap(ps, sim.t))
for name, before, after in ck.balance_table(sim.X, sim.t, ps):
    print(name, round(before, 3), round(after, 3))

ck.e_value_rr(3.9, 1.8)                    # (7.26, 3.0), the VanderWeele and Ding example
```

## What is in the box

| Module | Contents |
| --- | --- |
| `dag.py` | DAG with cycle rejection, parser, ancestors and descendants, topological order, Bayes ball d separation, moral graph d separation, backdoor criterion, minimal adjustment set enumeration with latent variables |
| `estimators.py` | difference in means, g computation (per arm outcome regression), Hajek IPW with optional trimming, cross fitted AIPW, nearest neighbour matching on the propensity logit or Mahalanobis distance with calipers |
| `models.py` | ridge OLS and Newton (IRLS) logistic regression with a small ridge to survive separation |
| `diagnostics.py` | standardised mean differences before and after weighting, overlap report, Kish effective sample size, E values for risk ratios and standardised differences |
| `simulate.py` | a confounded benchmark with heterogeneous effects, and a linear SEM sampler with a real `do()` operator |

## Design decisions

**Each estimator has its own variance.** A generic bootstrap would have been less code, but it is the wrong tool in at least one case: Abadie and Imbens (2008) showed the bootstrap is invalid for nearest neighbour matching with replacement. So matching uses the Abadie and Imbens (2006) variance, which accounts for controls reused K times by adding a K(K minus 1) sigma squared term, with each control's conditional variance estimated by matching it to its nearest other control. AIPW uses its efficient influence function. Hajek IPW uses the influence function of the ratio estimator, treating the propensity as known; Lunceford and Davidian show this is conservative for the ATE. g computation, which has no tidy closed form, is the one estimator that bootstraps.

**AIPW is cross fitted.** Nuisance models are fitted on K minus 1 folds and evaluated on the held out fold. With linear models this barely matters, but it is the step that keeps the influence function variance honest once someone swaps in a flexible learner, so the plumbing is there from day one. The `ps_features` argument lets the propensity model see a different design matrix from the outcome model, which is also how the tests misspecify one model at a time.

**The ATT version of AIPW is not just the ATE formula restricted to the treated.** It uses the score `[t(y − m0) − (1 − t) e/(1 − e) (y − m0)] / p` and its influence function subtracts `t·τ/p`, because the estimand itself depends on the treated fraction.

**Two d separation algorithms.** The production path is the linear time reachability procedure (Koller and Friedman, Algorithm 3.1) that walks (node, direction) states. The reference is the moralised ancestral graph test from Lauritzen et al. They share no code, so agreement is real evidence.

**Backdoor checks reduce to d separation.** Condition 2 of the criterion (block every path that enters the treatment through an arrowhead) is exactly d separation in the graph with the treatment's outgoing edges deleted. Reusing the tested d separation code rather than enumerating paths keeps the logic small and the search over adjustment sets fast enough for graphs with a couple of dozen nodes.

**SMDs use the unweighted pooled SD in both columns**, so "before" and "after" weighting are on the same scale (Austin 2009). Normalising by the weighted SD can make imbalance look better than it is.

## Testing

The 38 tests lean on ground truth rather than hand checked numbers:

* **Oracle agreement.** Bayes ball and the moral graph test are compared on every (x, y, Z) query over 60 random DAGs, more than five thousand queries, plus 200 set valued queries on larger graphs.
* **Interventional truth.** A linear SEM is sampled under `do(X = 0)` and `do(X = 1)` to get the causal effect by simulation. Regression on the backdoor set found by the DAG code matches it; regression with no adjustment, or with a collider added, is shown to be biased. This ties the identification half of the library to the estimation half.
* **Double robustness, both directions.** With a nonlinear outcome and a correct propensity model, g computation is visibly biased while AIPW is not. With a propensity that depends on x0 squared and a correct outcome model, IPW is biased while AIPW is not.
* **Interval coverage.** AIPW is run on 120 independent simulated studies and its 95% intervals must cover the true ATE at least 88% of the time.
* **Textbook graphs.** Chains, forks, colliders and collider descendants, M bias, a mediator, and the front door graph (where the code must report that no backdoor set exists).
* **Published numbers.** The E value example from VanderWeele and Ding (RR 3.9, lower limit 1.8, giving 7.26 and 3.0).
* **Nuisance models.** Logistic regression recovers known coefficients and stays finite under perfect separation; OLS matches `numpy.linalg.lstsq`.
* **CLI smoke tests** for every subcommand, including the CSV path.

## Limitations

The estimators assume a binary treatment, no interference between units, and that the covariates you pass are a valid adjustment set; causalkit can tell you which set that is if you can draw the DAG, but no software can tell you the DAG is right, which is what the E value is for. The adjustment set search is exhaustive over subsets, so it is meant for graphs of tens of nodes, not thousands. Matching builds a dense distance matrix and is quadratic in memory.

## References

* Pearl, *Causality* (2009), chapter 3: the backdoor criterion.
* Koller and Friedman, *Probabilistic Graphical Models* (2009), Algorithm 3.1.
* Lauritzen, Dawid, Larsen and Leimer, "Independence properties of directed Markov fields", *Networks* (1990).
* Robins, Rotnitzky and Zhao (1994); Chernozhukov et al., "Double/debiased machine learning" (2018).
* Abadie and Imbens, "Large sample properties of matching estimators" (2006), and "On the failure of the bootstrap for matching estimators" (2008).
* Lunceford and Davidian, "Stratification and weighting via the propensity score" (2004).
* Austin, "Balance diagnostics for comparing the distribution of baseline covariates" (2009).
* VanderWeele and Ding, "Sensitivity analysis in observational research: introducing the E value" (2017).

## License

MIT

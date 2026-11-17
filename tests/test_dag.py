import random
from itertools import combinations

import numpy as np
import pytest

from causalkit import DAG, CycleError, linear_sem


def test_parse_chains_and_isolated_nodes():
    g = DAG.parse("A -> B -> C; D")
    assert g.edges == [("A", "B"), ("B", "C")]
    assert g.nodes == ["A", "B", "C", "D"]
    assert g.ancestors("C") == {"A", "B"}
    assert g.descendants("A") == {"B", "C"}


def test_cycles_rejected():
    with pytest.raises(CycleError):
        DAG.parse("A -> B -> C -> A")
    with pytest.raises(CycleError):
        DAG.parse("A -> A")


def test_malformed_spec():
    with pytest.raises(ValueError):
        DAG.parse("A -> ")


def test_topological_order_respects_edges():
    g = DAG.parse("C -> A; B -> A; D -> C")
    order = g.topological_order()
    for u, v in g.edges:
        assert order.index(u) < order.index(v)


@pytest.mark.parametrize("method", ["d_separated", "d_separated_moral"])
def test_three_canonical_structures(method):
    chain = DAG.parse("A -> B -> C")
    fork = DAG.parse("B -> A; B -> C")
    collider = DAG.parse("A -> B; C -> B; B -> D")
    for g in (chain, fork):
        f = getattr(g, method)
        assert not f({"A"}, {"C"})
        assert f({"A"}, {"C"}, {"B"})
    f = getattr(collider, method)
    assert f({"A"}, {"C"})
    assert not f({"A"}, {"C"}, {"B"})
    assert not f({"A"}, {"C"}, {"D"})  # conditioning on a descendant opens the collider


def test_unknown_node_raises():
    with pytest.raises(KeyError):
        DAG.parse("A -> B").d_separated({"A"}, {"Q"})


def random_dag(rng, n, p):
    names = [f"v{i}" for i in range(n)]
    g = DAG(nodes=names)
    for i, j in combinations(range(n), 2):
        if rng.random() < p:
            g.add_edge(names[i], names[j])
    return g


def test_bayes_ball_agrees_with_moral_graph_on_random_dags():
    rng = random.Random(7)
    checked = 0
    for _ in range(60):
        n = rng.randint(3, 7)
        g = random_dag(rng, n, rng.choice([0.2, 0.35, 0.5]))
        nodes = g.nodes
        for x, y in combinations(nodes, 2):
            rest = [v for v in nodes if v not in (x, y)]
            for k in range(len(rest) + 1):
                for z in combinations(rest, k):
                    a = g.d_separated({x}, {y}, set(z))
                    b = g.d_separated_moral({x}, {y}, set(z))
                    assert a == b, (g, x, y, z)
                    checked += 1
    assert checked > 5000


def test_set_valued_queries_agree():
    rng = random.Random(3)
    for _ in range(200):
        g = random_dag(rng, 8, 0.3)
        nodes = g.nodes
        rng.shuffle(nodes)
        X, Y, Z = set(nodes[:2]), set(nodes[2:4]), set(nodes[4 : 4 + rng.randint(0, 3)])
        assert g.d_separated(X, Y, Z) == g.d_separated_moral(X, Y, Z)


def test_backdoor_confounding_triangle():
    g = DAG.parse("Z -> X; Z -> Y; X -> Y")
    assert not g.satisfies_backdoor("X", "Y", set())
    assert g.satisfies_backdoor("X", "Y", {"Z"})
    assert g.adjustment_sets("X", "Y") == [frozenset({"Z"})]


def test_m_bias_adjusting_the_collider_is_wrong():
    g = DAG.parse("U1 -> X; U1 -> M; U2 -> M; U2 -> Y; X -> Y")
    assert g.satisfies_backdoor("X", "Y", set())
    assert not g.satisfies_backdoor("X", "Y", {"M"})
    assert g.satisfies_backdoor("X", "Y", {"M", "U1"})
    assert g.adjustment_sets("X", "Y") == [frozenset()]


def test_descendants_of_treatment_never_allowed():
    g = DAG.parse("Z -> X; Z -> Y; X -> M -> Y")
    assert not g.satisfies_backdoor("X", "Y", {"Z", "M"})


def test_front_door_graph_has_no_backdoor_set_when_u_is_latent():
    g = DAG.parse("U -> X; U -> Y; X -> M -> Y")
    assert g.adjustment_sets("X", "Y", exclude=["U"]) == []
    assert g.adjustment_sets("X", "Y") == [frozenset({"U"})]


def test_minimal_sets_are_minimal_and_valid():
    g = DAG.parse("A -> X; B -> X; A -> Y; B -> Y; A -> B; C -> Y; X -> Y")
    sets = g.adjustment_sets("X", "Y")
    for s in sets:
        assert g.satisfies_backdoor("X", "Y", s)
        for v in s:
            assert not g.satisfies_backdoor("X", "Y", s - {v})
    assert frozenset({"A", "B"}) in sets


def test_adjustment_recovers_interventional_effect_in_a_linear_sem():
    """Regression on a backdoor set should match the slope under do(X)."""
    g = DAG.parse("Z -> X; Z -> Y; X -> Y; X -> C; Y -> C")
    coefs = {("Z", "X"): 1.5, ("Z", "Y"): 2.0, ("X", "Y"): 0.7, ("X", "C"): 1.0, ("Y", "C"): 1.0}
    d = linear_sem(g, 50_000, coefs, seed=1)
    lo = linear_sem(g, 50_000, coefs, seed=2, intervene={"X": 0.0})["Y"].mean()
    hi = linear_sem(g, 50_000, coefs, seed=2, intervene={"X": 1.0})["Y"].mean()
    truth = hi - lo

    def slope(cols):
        A = np.column_stack([np.ones(len(d["X"])), d["X"]] + [d[c] for c in cols])
        return np.linalg.lstsq(A, d["Y"], rcond=None)[0][1]

    assert g.satisfies_backdoor("X", "Y", {"Z"})
    assert abs(slope(["Z"]) - truth) < 0.02
    assert abs(slope([]) - truth) > 0.5  # confounded
    assert not g.satisfies_backdoor("X", "Y", {"Z", "C"})
    assert abs(slope(["Z", "C"]) - truth) > 0.2  # collider bias

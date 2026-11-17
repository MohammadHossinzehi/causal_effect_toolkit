"""Causal DAGs: d-separation, the backdoor criterion and adjustment set search.

Two independent d-separation algorithms live here on purpose:

* ``d_separated`` uses the linear time "reachable" procedure (Bayes ball,
  Koller and Friedman, Algorithm 3.1). It walks (node, direction) states.
* ``d_separated_moral`` uses the classic moralised ancestral graph test
  (Lauritzen et al. 1990): restrict to ancestors of X, Y, Z, marry parents,
  drop directions, delete Z, and check graph connectivity.

The test suite checks that both agree on every query over random DAGs, which
is a far stronger guarantee than a handful of hand written examples.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, FrozenSet, Iterable, List, Optional, Set


class CycleError(ValueError):
    """Raised when an edge would create a directed cycle."""


class DAG:
    def __init__(self, edges: Iterable[tuple] = (), nodes: Iterable[str] = ()):
        self._parents: Dict[str, Set[str]] = {}
        self._children: Dict[str, Set[str]] = {}
        for n in nodes:
            self.add_node(n)
        for u, v in edges:
            self.add_edge(u, v)

    # construction --------------------------------------------------------
    @classmethod
    def parse(cls, spec: str) -> "DAG":
        """Parse ``"Z -> X; Z -> Y; X -> Y"``. Chains like ``A -> B -> C`` work,
        and a bare name declares an isolated node."""
        g = cls()
        for stmt in spec.replace("\n", ";").split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            parts = [p.strip() for p in stmt.split("->")]
            if any(not p for p in parts):
                raise ValueError(f"malformed statement: {stmt!r}")
            for p in parts:
                g.add_node(p)
            for u, v in zip(parts, parts[1:]):
                g.add_edge(u, v)
        return g

    def add_node(self, n: str) -> None:
        self._parents.setdefault(n, set())
        self._children.setdefault(n, set())

    def add_edge(self, u: str, v: str) -> None:
        if u == v:
            raise CycleError(f"self loop on {u!r}")
        self.add_node(u)
        self.add_node(v)
        if u in self.descendants(v) or u == v:
            raise CycleError(f"edge {u}->{v} creates a cycle")
        self._parents[v].add(u)
        self._children[u].add(v)

    # queries ---------------------------------------------------------------
    @property
    def nodes(self) -> List[str]:
        return sorted(self._parents)

    @property
    def edges(self) -> List[tuple]:
        return sorted((u, v) for v, ps in self._parents.items() for u in ps)

    def parents(self, n: str) -> Set[str]:
        return set(self._parents[n])

    def children(self, n: str) -> Set[str]:
        return set(self._children[n])

    def _closure(self, start: Iterable[str], step: Dict[str, Set[str]]) -> Set[str]:
        seen: Set[str] = set()
        stack = list(start)
        while stack:
            n = stack.pop()
            for m in step[n]:
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        return seen

    def ancestors(self, n: str) -> Set[str]:
        """Strict ancestors of ``n``."""
        return self._closure([n], self._parents)

    def descendants(self, n: str) -> Set[str]:
        """Strict descendants of ``n``."""
        return self._closure([n], self._children)

    def topological_order(self) -> List[str]:
        indeg = {n: len(p) for n, p in self._parents.items()}
        ready = sorted(n for n, d in indeg.items() if d == 0)
        order: List[str] = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for c in sorted(self._children[n]):
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
            ready.sort()
        return order

    # d-separation ------------------------------------------------------------
    def _check(self, *sets: Set[str]) -> None:
        for s in sets:
            missing = set(s) - set(self._parents)
            if missing:
                raise KeyError(f"unknown nodes: {sorted(missing)}")

    def d_separated(self, xs: Iterable[str], ys: Iterable[str], zs: Iterable[str] = ()) -> bool:
        """True iff every path between ``xs`` and ``ys`` is blocked by ``zs``."""
        X, Y, Z = set(xs), set(ys), set(zs)
        self._check(X, Y, Z)
        if X & Y:
            return False
        X -= Z
        Y -= Z
        if not X or not Y:
            return True
        # Phase 1: Z together with all its ancestors (colliders open if they
        # or a descendant are observed).
        anc_z = set(Z)
        for z in Z:
            anc_z |= self.ancestors(z)
        # Phase 2: traverse active trails. "up" means we arrived from a child
        # (moving against an edge), "down" means we arrived from a parent.
        visited = set()
        frontier = [(x, "up") for x in X]
        while frontier:
            node, d = frontier.pop()
            if (node, d) in visited:
                continue
            visited.add((node, d))
            if node not in Z and node in Y:
                return False
            if d == "up" and node not in Z:
                frontier.extend((p, "up") for p in self._parents[node])
                frontier.extend((c, "down") for c in self._children[node])
            elif d == "down":
                if node not in Z:
                    frontier.extend((c, "down") for c in self._children[node])
                if node in anc_z:
                    frontier.extend((p, "up") for p in self._parents[node])
        return True

    def d_separated_moral(self, xs: Iterable[str], ys: Iterable[str], zs: Iterable[str] = ()) -> bool:
        """Reference implementation via the moralised ancestral graph."""
        X, Y, Z = set(xs), set(ys), set(zs)
        self._check(X, Y, Z)
        if X & Y:
            return False
        X -= Z
        Y -= Z
        if not X or not Y:
            return True
        keep = X | Y | Z
        for n in list(keep):
            keep |= self.ancestors(n)
        adj: Dict[str, Set[str]] = {n: set() for n in keep}
        for v in keep:
            ps = self._parents[v] & keep
            for p in ps:
                adj[p].add(v)
                adj[v].add(p)
            for a, b in combinations(sorted(ps), 2):
                adj[a].add(b)
                adj[b].add(a)
        seen = set(X)
        stack = list(X)
        while stack:
            n = stack.pop()
            for m in adj[n]:
                if m in Z or m in seen:
                    continue
                if m in Y:
                    return False
                seen.add(m)
                stack.append(m)
        return True

    # identification ------------------------------------------------------
    def satisfies_backdoor(self, treatment: str, outcome: str, adjustment: Iterable[str]) -> bool:
        """Pearl's backdoor criterion for a single treatment and outcome.

        1. no element of the set is a descendant of the treatment, and
        2. the set blocks every path from treatment to outcome that starts
           with an arrow into the treatment.

        Condition 2 is checked as d-separation in the graph with the
        treatment's outgoing edges removed.
        """
        S = set(adjustment)
        self._check({treatment, outcome}, S)
        if treatment in S or outcome in S:
            return False
        if S & self.descendants(treatment):
            return False
        g = DAG(nodes=self.nodes)
        for u, v in self.edges:
            if u != treatment:
                g.add_edge(u, v)
        return g.d_separated({treatment}, {outcome}, S)

    def adjustment_sets(
        self,
        treatment: str,
        outcome: str,
        max_size: Optional[int] = None,
        minimal: bool = True,
        exclude: Iterable[str] = (),
    ) -> List[FrozenSet[str]]:
        """Enumerate valid backdoor adjustment sets by increasing size.

        ``exclude`` lists unobserved variables that may not be adjusted for.
        With ``minimal=True`` only inclusion minimal sets are returned, which
        is usually what an analyst wants to look at.
        """
        banned = set(exclude) | {treatment, outcome} | self.descendants(treatment)
        candidates = [n for n in self.nodes if n not in banned]
        limit = len(candidates) if max_size is None else max_size
        found: List[FrozenSet[str]] = []
        for k in range(limit + 1):
            for combo in combinations(candidates, k):
                s = frozenset(combo)
                if minimal and any(f <= s for f in found):
                    continue
                if self.satisfies_backdoor(treatment, outcome, s):
                    found.append(s)
        return found

    def __repr__(self) -> str:
        body = "; ".join(f"{u} -> {v}" for u, v in self.edges)
        return f"DAG({body!r})"

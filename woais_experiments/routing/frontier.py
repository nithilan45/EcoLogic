"""Query-independent (cost, quality) Pareto frontier and hull interpolation.

A static policy is a point in the convex hull of the N model means
`(mean_cost_m, mean_quality_m)`. With two objectives the efficient boundary
is a chain of at most N vertices (a subset of the pure models). Any frontier
point is a mixture of **at most two** adjacent hull vertices (Carathéodory).

No fitting: the hull is combinatorial. Collinear interior vertices are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class Mixture:
    """A query-independent mix: expected cost/quality equal the convex combination."""

    weights: dict[str, float]
    cost: float
    quality: float
    kind: str
    feasible: bool = True
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "cost": self.cost,
            "quality": self.quality,
            "kind": self.kind,
            "feasible": self.feasible,
            "note": self.note,
        }


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _clean_weights(names: Sequence[str], raw: dict[str, float], eps: float = EPS) -> dict[str, float]:
    w = {n: float(raw.get(n, 0.0)) for n in names}
    s = sum(max(0.0, v) for v in w.values())
    if s <= eps:
        raise ValueError("mixture weights sum to 0")
    w = {n: max(0.0, v) / s for n, v in w.items()}
    return {n: v for n, v in w.items() if v > eps}


class StaticMarket:
    """N models with unconditional mean cost and mean quality.

    For query-independent policies, `mean_cost` must be the panel mean over
    *all* queries (not the routed-to-m subset). Then expected cost equals
    realized cost.
    """

    def __init__(
        self,
        names: Sequence[str],
        mean_cost: Sequence[float],
        mean_quality: Sequence[float],
    ):
        if len(names) == 0:
            raise ValueError("need at least one model")
        if len(names) != len(mean_cost) or len(names) != len(mean_quality):
            raise ValueError("names, mean_cost, mean_quality length mismatch")
        if len(set(names)) != len(names):
            raise ValueError("model names must be unique")
        self.names = tuple(str(n) for n in names)
        self.mean_cost = np.asarray(mean_cost, dtype=float).reshape(-1)
        self.mean_quality = np.asarray(mean_quality, dtype=float).reshape(-1)
        if np.any(~np.isfinite(self.mean_cost)) or np.any(~np.isfinite(self.mean_quality)):
            raise ValueError("non-finite mean cost or quality")

    def __len__(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def mixture(self, weights: dict[str, float], *, kind: str = "mixture") -> Mixture:
        w = _clean_weights(self.names, weights)
        full = {n: w.get(n, 0.0) for n in self.names}
        cost = float(sum(full[n] * self.mean_cost[i] for i, n in enumerate(self.names)))
        qual = float(sum(full[n] * self.mean_quality[i] for i, n in enumerate(self.names)))
        return Mixture(weights=w, cost=cost, quality=qual, kind=kind)

    def pure(self, name: str) -> Mixture:
        return self.mixture({name: 1.0}, kind=f"always_{name}")


def pairwise_pareto_filter(market: StaticMarket, *, eps: float = EPS) -> list[int]:
    """Indices not strictly dominated by any *single* other model."""
    keep = []
    c, q = market.mean_cost, market.mean_quality
    for i in range(len(market)):
        dominated = False
        for j in range(len(market)):
            if i == j:
                continue
            cheaper_eq = c[j] <= c[i] + eps
            better_eq = q[j] >= q[i] - eps
            strict = (c[j] < c[i] - eps) or (q[j] > q[i] + eps)
            if cheaper_eq and better_eq and strict:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def upper_hull_indices(market: StaticMarket, *, eps: float = EPS) -> list[int]:
    """Left-to-right upper hull of mixture-undominated models (increasing cost).

    A model below the chord of two others is dropped: a query-independent mix
    of those two strictly beats it. Slopes along the hull are non-increasing.
    """
    n = len(market)
    order = sorted(range(n), key=lambda i: (market.mean_cost[i], -market.mean_quality[i], market.names[i]))
    # Staircase: strictly improving quality as cost increases.
    stair: list[int] = []
    best_q = -np.inf
    for i in order:
        if market.mean_quality[i] > best_q + eps:
            stair.append(i)
            best_q = market.mean_quality[i]
        elif abs(market.mean_quality[i] - best_q) <= eps and stair:
            # same quality, later in sort ⇒ higher or equal cost: skip (worse or duplicate)
            continue
    if len(stair) <= 2:
        return stair
    hull: list[int] = []
    pts = np.stack([market.mean_cost, market.mean_quality], axis=1)
    for i in stair:
        while len(hull) >= 2 and _cross(pts[hull[-2]], pts[hull[-1]], pts[i]) >= -eps:
            hull.pop()
        hull.append(i)
    return hull


def hull_vertices(market: StaticMarket, *, eps: float = EPS) -> list[Mixture]:
    return [market.pure(market.names[i]) for i in upper_hull_indices(market, eps=eps)]


def interpolate_at_cost(
    market: StaticMarket,
    target_cost: float,
    *,
    mode: str = "equal",
    eps: float = EPS,
) -> Mixture:
    """Max-quality static policy at expected cost `target_cost`.

    mode='equal': exact cost (upper envelope of the convex hull).
    mode='at_most': expected cost ≤ target (running max of that envelope).
    """
    if mode not in {"equal", "at_most"}:
        raise ValueError("mode must be 'equal' or 'at_most'")
    idx = upper_hull_indices(market, eps=eps)
    verts = [market.pure(market.names[i]) for i in idx]
    c_lo, c_hi = verts[0].cost, verts[-1].cost
    q_best = max(verts, key=lambda v: (v.quality, -v.cost))

    if mode == "at_most":
        if target_cost < c_lo - eps:
            return Mixture(
                weights=dict(verts[0].weights), cost=verts[0].cost, quality=verts[0].quality,
                kind="budget_infeasible", feasible=False,
                note="target cost is below the cheapest hull vertex",
            )
        # highest quality among hull vertices with cost <= target, else interpolate
        # on the edge that crosses the budget (same as equal if envelope increases).
        if target_cost >= q_best.cost - eps:
            return Mixture(
                weights=dict(q_best.weights), cost=q_best.cost, quality=q_best.quality,
                kind="budget_best_model", feasible=True,
                note="extra budget is not spent; a cheaper model already maximises quality",
            )
        return interpolate_at_cost(market, target_cost, mode="equal", eps=eps)

    # exact cost
    if target_cost < c_lo - eps or target_cost > c_hi + eps:
        return Mixture(
            weights=dict(verts[0].weights) if target_cost < c_lo else dict(verts[-1].weights),
            cost=float(target_cost),
            quality=float("nan"),
            kind="cost_out_of_range",
            feasible=False,
            note=f"target cost {target_cost} outside hull span [{c_lo}, {c_hi}]",
        )
    if abs(target_cost - c_lo) <= eps:
        return verts[0]
    if abs(target_cost - c_hi) <= eps:
        return verts[-1]
    for a, b in zip(verts, verts[1:]):
        lo, hi = (a.cost, b.cost) if a.cost <= b.cost else (b.cost, a.cost)
        if target_cost < lo - eps or target_cost > hi + eps:
            continue
        span = b.cost - a.cost
        if abs(span) <= eps:
            return a if a.quality >= b.quality else b
        t = (target_cost - a.cost) / span
        t = min(1.0, max(0.0, t))
        w: dict[str, float] = {}
        for n, p in a.weights.items():
            w[n] = w.get(n, 0.0) + (1.0 - t) * p
        for n, p in b.weights.items():
            w[n] = w.get(n, 0.0) + t * p
        mix = market.mixture(w, kind=f"hull_edge:{a.kind}+{b.kind}")
        # numerically pin cost
        return Mixture(
            weights=mix.weights, cost=float(target_cost), quality=mix.quality,
            kind=mix.kind, feasible=True,
            note=f"interpolation t={t:.6f} between hull vertices",
        )
    return Mixture(
        weights=dict(verts[0].weights), cost=float(target_cost), quality=float("nan"),
        kind="cost_out_of_range", feasible=False, note="no hull edge contains target cost",
    )


def interpolate_at_quality(
    market: StaticMarket,
    target_quality: float,
    *,
    eps: float = EPS,
) -> Mixture:
    """Minimum expected cost among static policies with quality ≥ target.

    On an increasing hull edge this is exact-quality interpolation.
    """
    idx = upper_hull_indices(market, eps=eps)
    verts = [market.pure(market.names[i]) for i in idx]
    q_lo, q_hi = verts[0].quality, verts[-1].quality
    if target_quality > q_hi + eps:
        return Mixture(
            weights=dict(verts[-1].weights), cost=verts[-1].cost, quality=verts[-1].quality,
            kind="quality_infeasible", feasible=False,
            note="target quality exceeds every static policy",
        )
    if target_quality <= q_lo + eps:
        return Mixture(
            weights=dict(verts[0].weights), cost=verts[0].cost, quality=verts[0].quality,
            kind="quality_met_by_cheapest_hull", feasible=True,
        )
    for a, b in zip(verts, verts[1:]):
        if target_quality > b.quality + eps:
            continue
        if target_quality <= a.quality + eps:
            return a
        span = b.quality - a.quality
        if abs(span) <= eps:
            return a if a.cost <= b.cost else b
        t = (target_quality - a.quality) / span
        t = min(1.0, max(0.0, t))
        w: dict[str, float] = {}
        for n, p in a.weights.items():
            w[n] = w.get(n, 0.0) + (1.0 - t) * p
        for n, p in b.weights.items():
            w[n] = w.get(n, 0.0) + t * p
        mix = market.mixture(w, kind=f"hull_edge:{a.kind}+{b.kind}")
        return Mixture(
            weights=mix.weights, cost=mix.cost, quality=float(target_quality),
            kind=mix.kind, feasible=True,
            note=f"interpolation t={t:.6f} between hull vertices",
        )
    return verts[-1]

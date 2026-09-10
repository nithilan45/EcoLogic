"""Catalog of query-independent routing baselines. No ML.

Policies
--------
1. always-cheapest model (ties: higher quality, then name)
2. always-most-expensive model (ties: lower quality, then name)
3. always-each-model
4. random (50-50) mixture between every pair
5. general simplex mixture (caller supplies weights; uniform 1/N included)
6. minimum-cost static policy at a target quality (hull interpolation)
7. maximum-quality static policy under a target expected cost

A query-dependent router's realized cost is matched in `cost_matching.py`.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from woais_experiments.routing.cost_matching import compare_router_to_static, router_means
from woais_experiments.routing.frontier import (
    EPS,
    Mixture,
    StaticMarket,
    hull_vertices,
    interpolate_at_cost,
    interpolate_at_quality,
    pairwise_pareto_filter,
    upper_hull_indices,
)


def market_from_means(
    names: Sequence[str],
    mean_cost: Sequence[float],
    mean_quality: Sequence[float],
) -> StaticMarket:
    return StaticMarket(names, mean_cost, mean_quality)


def market_from_panel(
    names: Sequence[str],
    cost: np.ndarray,
    quality: np.ndarray,
) -> StaticMarket:
    """Unconditional means over queries (axis 0). `cost`/`quality` shape (n, M)."""
    cost = np.asarray(cost, dtype=float)
    quality = np.asarray(quality, dtype=float)
    if cost.shape != quality.shape or cost.ndim != 2:
        raise ValueError("cost and quality must be 2-D arrays of the same shape")
    if cost.shape[1] != len(names):
        raise ValueError("column count must match names")
    return StaticMarket(names, cost.mean(axis=0), quality.mean(axis=0))


def always_cheapest(market: StaticMarket) -> Mixture:
    i = min(
        range(len(market)),
        key=lambda j: (market.mean_cost[j], -market.mean_quality[j], market.names[j]),
    )
    mix = market.pure(market.names[i])
    return Mixture(mix.weights, mix.cost, mix.quality, kind="always_cheapest", feasible=True)


def always_most_expensive(market: StaticMarket) -> Mixture:
    i = max(
        range(len(market)),
        key=lambda j: (market.mean_cost[j], -market.mean_quality[j], market.names[j]),
    )
    mix = market.pure(market.names[i])
    return Mixture(mix.weights, mix.cost, mix.quality, kind="always_most_expensive", feasible=True)


def always_each_model(market: StaticMarket) -> dict[str, Mixture]:
    return {n: market.pure(n) for n in market.names}


def pairwise_random_mixtures(market: StaticMarket, *, f: float = 0.5) -> list[Mixture]:
    """Query-independent Bernoulli mix between every unordered pair.

    `f` is the weight on the *second* name in lexicographic pair order.
    """
    if not 0.0 <= f <= 1.0:
        raise ValueError("f must be in [0, 1]")
    out = []
    for a, b in combinations(market.names, 2):
        w = {a: 1.0 - f, b: f}
        out.append(market.mixture(w, kind=f"pairwise[{a},{b}]_f={f:g}"))
    return out


def pairwise_segments(market: StaticMarket, *, n_grid: int = 5) -> list[Mixture]:
    """Grid along every pair segment, including endpoints."""
    if n_grid < 2:
        raise ValueError("n_grid must be >= 2")
    out = []
    fs = np.linspace(0.0, 1.0, n_grid)
    for a, b in combinations(market.names, 2):
        for f in fs:
            out.append(market.mixture({a: 1.0 - float(f), b: float(f)},
                                      kind=f"pairwise[{a},{b}]_f={float(f):g}"))
    return out


def uniform_mixture(market: StaticMarket) -> Mixture:
    w = {n: 1.0 / len(market) for n in market.names}
    return market.mixture(w, kind="uniform_1/N")


def general_mixture(market: StaticMarket, weights: Mapping[str, float]) -> Mixture:
    return market.mixture(dict(weights), kind="general_mixture")


def min_cost_for_quality(market: StaticMarket, target_quality: float, *, eps: float = EPS) -> Mixture:
    return interpolate_at_quality(market, target_quality, eps=eps)


def max_quality_for_cost(
    market: StaticMarket,
    target_cost: float,
    *,
    mode: str = "equal",
    eps: float = EPS,
) -> Mixture:
    return interpolate_at_cost(market, target_cost, mode=mode, eps=eps)


def catalog(market: StaticMarket, *, pair_f: float = 0.5) -> dict[str, Any]:
    """All named static baselines plus hull vertices."""
    pairs = pairwise_random_mixtures(market, f=pair_f)
    return {
        "n_models": len(market),
        "models": list(market.names),
        "mean_cost": {n: float(market.mean_cost[i]) for i, n in enumerate(market.names)},
        "mean_quality": {n: float(market.mean_quality[i]) for i, n in enumerate(market.names)},
        "always_cheapest": always_cheapest(market).as_dict(),
        "always_most_expensive": always_most_expensive(market).as_dict(),
        "always_each": {n: m.as_dict() for n, m in always_each_model(market).items()},
        "pairwise_random": [m.as_dict() for m in pairs],
        "uniform_mixture": uniform_mixture(market).as_dict(),
        "hull_vertices": [m.as_dict() for m in hull_vertices(market)],
        "hull_model_names": [market.names[i] for i in upper_hull_indices(market)],
        "pairwise_undominated": [market.names[i] for i in pairwise_pareto_filter(market)],
    }


def evaluate_assignment_against_static(
    names: Sequence[str],
    cost: np.ndarray,
    quality: np.ndarray,
    assignment: Mapping[Any, str],
    query_ids: Sequence[Any] | None = None,
    *,
    eps: float = EPS,
) -> dict[str, Any]:
    """Build the market from a panel, score a router, return the full report."""
    market = market_from_panel(names, cost, quality)
    rq, rc = router_means(assignment, names, cost, quality, query_ids=query_ids)
    report = compare_router_to_static(market, rq, rc, eps=eps)
    report["catalog"] = catalog(market)
    report["min_cost_at_router_quality"] = min_cost_for_quality(market, rq).as_dict()
    report["max_quality_at_router_cost"] = max_quality_for_cost(market, rc, mode="equal").as_dict()
    report["max_quality_at_router_budget"] = max_quality_for_cost(market, rc, mode="at_most").as_dict()
    return report

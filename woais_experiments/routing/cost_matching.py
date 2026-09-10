"""Match a query-dependent router to the static hull at equal cost / equal quality.

Router cost must be **realized** per-query cost, not mix × unconditional mean.
Static expected cost equals realized cost (query-independent assignment).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from woais_experiments.routing.frontier import (
    EPS,
    Mixture,
    StaticMarket,
    interpolate_at_cost,
    interpolate_at_quality,
    upper_hull_indices,
)


def router_means(
    assignment: Mapping[Any, str],
    names: Sequence[str],
    cost: np.ndarray,
    quality: np.ndarray,
    query_ids: Sequence[Any] | None = None,
) -> tuple[float, float]:
    """Realized mean quality and cost of a (possibly query-dependent) assignment.

    Panel rows must align with `query_ids` (or, if omitted, `list(assignment)`).
    """
    names = tuple(names)
    col = {n: i for i, n in enumerate(names)}
    ids = list(query_ids) if query_ids is not None else list(assignment)
    n = len(ids)
    if n == 0:
        raise ValueError("empty assignment")
    cost = np.asarray(cost, dtype=float)
    quality = np.asarray(quality, dtype=float)
    if cost.shape != quality.shape or cost.ndim != 2 or cost.shape[0] != n:
        raise ValueError("cost/quality must be shape (n_queries, n_models) aligned with query ids")
    q = 0.0
    c = 0.0
    for i, qid in enumerate(ids):
        m = assignment[qid]
        if m not in col:
            raise KeyError(f"assignment model {m!r} is not in {names}")
        j = col[m]
        q += float(quality[i, j])
        c += float(cost[i, j])
    return q / n, c / n


def compare_router_to_static(
    market: StaticMarket,
    router_quality: float,
    router_realized_cost: float,
    *,
    eps: float = EPS,
) -> dict[str, Any]:
    """Score a router against cost-matched and quality-matched static policies."""
    same_cost = interpolate_at_cost(market, router_realized_cost, mode="equal", eps=eps)
    at_budget = interpolate_at_cost(market, router_realized_cost, mode="at_most", eps=eps)
    same_quality = interpolate_at_quality(market, router_quality, eps=eps)

    q_span = float(np.max(market.mean_quality) - np.min(market.mean_quality))
    denom = q_span if q_span > eps else max(abs(same_cost.quality), eps)

    quality_advantage = None
    relative_regret = None
    if same_cost.feasible:
        quality_advantage = float(router_quality - same_cost.quality)
        relative_regret = float((same_cost.quality - router_quality) / denom)

    cost_savings = None
    if same_quality.feasible:
        cost_savings = float(same_quality.cost - router_realized_cost)

    hull = [market.names[i] for i in upper_hull_indices(market, eps=eps)]
    return {
        "router_quality": float(router_quality),
        "router_realized_cost": float(router_realized_cost),
        "static_quality_at_same_cost": same_cost.quality if same_cost.feasible else None,
        "static_mix_at_same_cost": same_cost.weights if same_cost.feasible else None,
        "static_cost_matched": same_cost.cost if same_cost.feasible else None,
        "same_cost_feasible": same_cost.feasible,
        "same_cost_note": same_cost.note,
        "quality_advantage": quality_advantage,
        "static_quality_at_budget": at_budget.quality if at_budget.feasible else None,
        "static_mix_at_budget": at_budget.weights if at_budget.feasible else None,
        "router_cost_at_same_quality": float(router_realized_cost),
        "static_cost_at_same_quality": same_quality.cost if same_quality.feasible else None,
        "static_mix_at_same_quality": same_quality.weights if same_quality.feasible else None,
        "same_quality_feasible": same_quality.feasible,
        "cost_savings": cost_savings,
        "relative_regret": relative_regret,
        "relative_regret_denominator": denom,
        "hull_models": hull,
        "quality_advantage_definition": "router_quality - static_quality_at_same_cost",
        "cost_savings_definition": "static_cost_at_same_quality - router_realized_cost (>0 if router is cheaper)",
        "relative_regret_definition": "(static_quality_at_same_cost - router_quality) / (max_q - min_q)",
    }


def match_cost(
    market: StaticMarket,
    router_realized_cost: float,
    *,
    mode: str = "equal",
    eps: float = EPS,
) -> Mixture:
    return interpolate_at_cost(market, router_realized_cost, mode=mode, eps=eps)


def match_quality(
    market: StaticMarket,
    router_quality: float,
    *,
    eps: float = EPS,
) -> Mixture:
    return interpolate_at_quality(market, router_quality, eps=eps)

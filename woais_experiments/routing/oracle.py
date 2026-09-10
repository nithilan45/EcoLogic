"""Budgeted oracle routing vs static hull vs a learned/query-dependent router.

The oracle is the best *assignment* (one model per query) under a total cost
budget — the integer MCKP. The static frontier is query-independent (a mix of
unconditional means) at the same *expected* total cost. The learned router is
a fixed assignment; it is feasible at budget B iff its realized cost is ≤ B.

Gaps at a budget (when the router is feasible):

    router_vs_static_gap = router_quality − static_frontier_quality
    oracle_vs_router_gap = oracle_quality − router_quality
    fraction_of_available_routing_value_captured
        = (router_quality − static_quality) / (oracle_quality − static_quality)

No ML fitting. Frozen Stage 1–10 trees are not modified.
"""

from __future__ import annotations

import csv
from io import StringIO
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from woais_experiments.frozen import ItemMatrix, write_result
from woais_experiments.routing.frontier import EPS, interpolate_at_cost
from woais_experiments.routing.mckp import (
    MCKPInstance,
    Solution,
    cheapest_assignment,
    default_budgets,
    lp_relaxation,
    solve,
    unconstrained_oracle,
)
from woais_experiments.routing.static_baselines import market_from_panel

SWEEP_FIELDS = (
    "budget",
    "static_frontier_quality",
    "learned_router_quality",
    "oracle_quality",
    "router_feasible",
    "oracle_feasible",
    "static_feasible",
    "oracle_cost",
    "router_realized_cost",
    "static_expected_cost",
    "oracle_method",
    "oracle_lp_quality",
    "router_vs_static_gap",
    "oracle_vs_router_gap",
    "fraction_of_available_routing_value_captured",
)


def instance_from_arrays(
    cost: np.ndarray,
    quality: np.ndarray,
    names: Sequence[str] | None = None,
    query_ids: Sequence[Any] | None = None,
) -> MCKPInstance:
    return MCKPInstance(
        cost,
        quality,
        names=tuple(names) if names is not None else (),
        query_ids=tuple(query_ids) if query_ids is not None else (),
    )


def instance_from_item_matrix(
    matrix: ItemMatrix,
    cost_of: Callable[[int, str], float],
    *,
    tiers: tuple[int, ...] = (1, 2, 3),
) -> MCKPInstance:
    n = matrix.n
    m = len(tiers)
    cost = np.zeros((n, m), dtype=float)
    quality = np.zeros((n, m), dtype=float)
    for i, qid in enumerate(matrix.item_ids):
        for k, t in enumerate(tiers):
            cost[i, k] = float(cost_of(t, qid))
            quality[i, k] = 1.0 if matrix.correct[(t, qid)] else 0.0
    names = tuple(matrix.model_of.get(t, f"t{t}") for t in tiers)
    return MCKPInstance(cost, quality, names=names, query_ids=tuple(matrix.item_ids))


def choice_from_tiers(
    assignment: Mapping[Any, int],
    query_ids: Sequence[Any],
    *,
    tiers: tuple[int, ...] = (1, 2, 3),
) -> np.ndarray:
    col = {t: k for k, t in enumerate(tiers)}
    out = np.empty(len(query_ids), dtype=int)
    for i, qid in enumerate(query_ids):
        t = int(assignment[qid])
        if t not in col:
            raise KeyError(f"tier {t} is not in {tiers}")
        out[i] = col[t]
    return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def captured_fraction(
    router_quality: float,
    static_quality: float,
    oracle_quality: float,
    *,
    eps: float = EPS,
) -> float | None:
    available = float(oracle_quality) - float(static_quality)
    if abs(available) <= eps:
        return None
    return float((router_quality - static_quality) / available)


def _static_at_total_budget(inst: MCKPInstance, budget: float, *, eps: float = EPS):
    market = market_from_panel(inst.names, inst.cost, inst.quality)
    per_query = float(budget) / inst.n
    return interpolate_at_cost(market, per_query, mode="at_most", eps=eps)


def evaluate_at_budget(
    inst: MCKPInstance,
    budget: float,
    router_choice: np.ndarray | None = None,
    *,
    method: str = "auto",
    eps: float = EPS,
    scale: float | None = None,
) -> dict[str, Any]:
    """Static hull, learned router, and MCKP oracle at total cost ``budget``."""
    static = _static_at_total_budget(inst, budget, eps=eps)
    oracle = solve(inst, budget, method=method, eps=eps, scale=scale)
    lp = lp_relaxation(inst, budget, eps=eps)

    router_sol: Solution | None = None
    router_feasible = False
    if router_choice is not None:
        router_sol = inst.evaluate(router_choice)
        router_sol.method = "router"
        router_feasible = router_sol.total_cost <= float(budget) + eps

    static_q = static.quality if static.feasible else None
    oracle_q = oracle.mean_quality if oracle.feasible else None
    router_q = router_sol.mean_quality if router_sol is not None else None

    r_vs_s = o_vs_r = frac = None
    if router_feasible and static_q is not None and oracle_q is not None and router_q is not None:
        r_vs_s = float(router_q - static_q)
        o_vs_r = float(oracle_q - router_q)
        frac = captured_fraction(router_q, static_q, oracle_q, eps=eps)

    return {
        "budget": float(budget),
        "static_frontier_quality": static_q,
        "learned_router_quality": router_q,
        "oracle_quality": oracle_q,
        "router_feasible": router_feasible,
        "oracle_feasible": oracle.feasible,
        "static_feasible": static.feasible,
        "oracle_cost": oracle.total_cost if oracle.feasible else None,
        "router_realized_cost": router_sol.total_cost if router_sol is not None else None,
        "static_expected_cost": (static.cost * inst.n) if static.feasible else None,
        "oracle_method": oracle.method,
        "oracle_note": oracle.note,
        "oracle_lp_quality": lp.mean_quality if lp.feasible else None,
        "oracle_mix": oracle.as_dict().get("model_mix"),
        "static_mix": static.weights if static.feasible else None,
        "router_vs_static_gap": r_vs_s,
        "oracle_vs_router_gap": o_vs_r,
        "fraction_of_available_routing_value_captured": frac,
    }


def budget_sweep(
    inst: MCKPInstance,
    budgets: Sequence[float] | None = None,
    router_choice: np.ndarray | None = None,
    *,
    method: str = "auto",
    n_grid: int = 21,
    eps: float = EPS,
    scale: float | None = None,
) -> dict[str, Any]:
    extra = []
    router_sol = None
    if router_choice is not None:
        router_sol = inst.evaluate(router_choice)
        extra.append(router_sol.total_cost)
    if budgets is None:
        budgets = default_budgets(inst, n_grid=n_grid, extra=extra)
    rows = [
        evaluate_at_budget(inst, float(b), router_choice, method=method, eps=eps, scale=scale)
        for b in budgets
    ]
    at_router = None
    if router_sol is not None:
        at_router = evaluate_at_budget(
            inst, router_sol.total_cost, router_choice, method=method, eps=eps, scale=scale
        )
    cheap = cheapest_assignment(inst)
    unconstrained = unconstrained_oracle(inst)
    return {
        "n_queries": inst.n,
        "n_models": inst.m,
        "models": list(inst.names),
        "min_total_cost": cheap.total_cost,
        "max_total_cost": inst.max_total_cost(),
        "unconstrained_oracle": unconstrained.as_dict(),
        "cheapest_assignment": cheap.as_dict(),
        "router": None if router_sol is None else {
            "quality": router_sol.mean_quality,
            "realized_cost": router_sol.total_cost,
            "mix": {
                inst.names[k]: int(np.sum(router_choice == k))
                for k in range(inst.m)
            },
        },
        "gaps_at_router_cost": at_router,
        "method": method,
        "sweep": rows,
    }


def sweep_to_csv(payload: Mapping[str, Any]) -> str:
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(SWEEP_FIELDS), extrasaction="ignore")
    writer.writeheader()
    for row in payload["sweep"]:
        writer.writerow({k: row.get(k) for k in SWEEP_FIELDS})
    return buf.getvalue()


def save_budget_sweep(
    payload: Mapping[str, Any],
    relpath: str = "routing/oracle_budget_sweep",
) -> dict[str, str]:
    """Write JSON + CSV under woais_experiments/results/."""
    json_path = write_result(f"{relpath}.json", _jsonable(dict(payload)))
    csv_path = write_result(f"{relpath}.csv", sweep_to_csv(payload))
    return {"json": str(json_path), "csv": str(csv_path)}


def sweep_item_matrix(
    matrix: ItemMatrix,
    cost_of: Callable[[int, str], float],
    router_tiers: Mapping[Any, int],
    *,
    method: str = "approx",
    n_grid: int = 21,
    relpath: str | None = "routing/oracle_budget_sweep",
    tiers: tuple[int, ...] = (1, 2, 3),
) -> dict[str, Any]:
    inst = instance_from_item_matrix(matrix, cost_of, tiers=tiers)
    choice = choice_from_tiers(router_tiers, matrix.item_ids, tiers=tiers)
    payload = budget_sweep(inst, router_choice=choice, method=method, n_grid=n_grid)
    payload["cost_axis"] = getattr(cost_of, "__name__", "cost_of")
    if relpath:
        payload["saved"] = save_budget_sweep(payload, relpath=relpath)
    return payload

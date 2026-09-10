"""Naive vs realized accounting and baseline comparisons for an external router."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from woais_experiments.external_routing.panel import ExternalRouterPanel
from woais_experiments.external_routing.reconstruct_assignments import (
    EXTERNAL_LEARNED_ROUTER,
    ORACLE_ASSIGNMENT,
    as_side_index,
    oracle_route,
    route_by_score,
)
from woais_experiments.routing.cost_matching import compare_router_to_static
from woais_experiments.routing.static_baselines import market_from_panel
from woais_experiments.statistics.paired_tests import paired_comparison

NEGLIGIBLE_REL = 0.01
NEGLIGIBLE_ABS = 1e-6
POLICY_ORDER = (
    "always_cheap",
    "always_strong",
    "random_mixture",
    "cost_matched_static",
    "external_learned_router",
    "oracle_assignment",
)


def two_model_quota(
    ids: Sequence[str],
    p_strong: float,
    *,
    cheap: str,
    strong: str,
) -> dict[str, str]:
    ordered = sorted(str(i) for i in ids)
    n = len(ordered)
    n_s = int(round(float(np.clip(p_strong, 0.0, 1.0)) * n))
    n_s = min(max(n_s, 0), n)
    return {qid: (strong if i < n_s else cheap) for i, qid in enumerate(ordered)}


def naive_realized(
    side: np.ndarray,
    cheap_cost: np.ndarray,
    strong_cost: np.ndarray,
) -> dict[str, float]:
    n = int(side.size)
    f = float(side.mean()) if n else float("nan")
    m0, m1 = float(cheap_cost.mean()), float(strong_cost.mean())
    naive = (1.0 - f) * m0 + f * m1
    realized = float(np.where(side == 1, strong_cost, cheap_cost).mean())
    abs_err = float(naive - realized)
    rel = abs_err / realized if realized else float("nan")
    return {
        "routing_rate": f,
        "naive_cost": naive,
        "realized_cost": realized,
        "absolute_accounting_error": abs_err,
        "relative_accounting_error": rel,
        "cheap_mean_cost": m0,
        "strong_mean_cost": m1,
    }


def _quality(side: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    return np.where(side == 1, q1, q0).astype(float)


def _cost(side: np.ndarray, c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    return np.where(side == 1, c1, c0).astype(float)


def _wide_market(panel: ExternalRouterPanel):
    cost = np.column_stack([panel.cheap_cost, panel.strong_cost])
    quality = np.column_stack([panel.cheap_quality, panel.strong_quality])
    names = (panel.cheap_model, panel.strong_model)
    return names, cost, quality, market_from_panel(names, cost, quality)


def evaluate_threshold(
    panel: ExternalRouterPanel,
    threshold: float,
    *,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    assign = route_by_score(
        panel.scores, threshold, cheap=panel.cheap_model, strong=panel.strong_model
    )
    side = as_side_index(assign, cheap=panel.cheap_model, strong=panel.strong_model)
    acc = naive_realized(side, panel.cheap_cost, panel.strong_cost)
    q = _quality(side, panel.cheap_quality, panel.strong_quality)
    c = _cost(side, panel.cheap_cost, panel.strong_cost)
    names, cost, quality, market = _wide_market(panel)
    vs_realized = compare_router_to_static(market, float(q.mean()), acc["realized_cost"])
    vs_naive = compare_router_to_static(market, float(q.mean()), acc["naive_cost"])

    oracle = oracle_route(
        panel.cheap_quality,
        panel.strong_quality,
        panel.cheap_cost,
        panel.strong_cost,
        cheap=panel.cheap_model,
        strong=panel.strong_model,
    )
    oracle_side = as_side_index(oracle, cheap=panel.cheap_model, strong=panel.strong_model)
    q_or = _quality(oracle_side, panel.cheap_quality, panel.strong_quality)
    c_or = _cost(oracle_side, panel.cheap_cost, panel.strong_cost)

    q_cheap = panel.cheap_quality
    q_strong = panel.strong_quality
    f = acc["routing_rate"]
    random_q = (1.0 - f) * float(q_cheap.mean()) + f * float(q_strong.mean())
    random_c = (1.0 - f) * acc["cheap_mean_cost"] + f * acc["strong_mean_cost"]

    f_match = vs_realized.get("static_mix_at_same_cost") or {}
    p_strong_match = float(f_match.get(panel.strong_model, f) or 0.0)
    quota = two_model_quota(
        panel.query_ids, p_strong_match, cheap=panel.cheap_model, strong=panel.strong_model
    )
    quota_side = as_side_index(
        [quota[i] for i in panel.query_ids], cheap=panel.cheap_model, strong=panel.strong_model
    )
    q_static = _quality(quota_side, panel.cheap_quality, panel.strong_quality)
    c_static = _cost(quota_side, panel.cheap_cost, panel.strong_cost)

    paired_vs_static = paired_comparison(
        q, q_static, query_ids=panel.query_ids, name_a="external_learned_router",
        name_b="cost_matched_static", n_boot=n_boot, n_perm=n_perm, seed=seed,
    )
    paired_vs_cheap = paired_comparison(
        q, q_cheap, query_ids=panel.query_ids, name_a="external_learned_router",
        name_b="always_cheap", n_boot=n_boot, n_perm=n_perm, seed=seed + 1,
    )

    adv_real = vs_realized.get("quality_advantage")
    adv_naive = vs_naive.get("quality_advantage")
    save_real = vs_realized.get("cost_savings")
    save_naive = vs_naive.get("cost_savings")

    oracle_cost = float(c_or.mean())
    cost_regret_naive = acc["naive_cost"] - oracle_cost
    cost_regret_realized = acc["realized_cost"] - oracle_cost

    policies = {
        "always_cheap": {
            "kind": "baseline",
            "quality": float(q_cheap.mean()),
            "realized_cost": acc["cheap_mean_cost"],
            "naive_cost": acc["cheap_mean_cost"],
        },
        "always_strong": {
            "kind": "baseline",
            "quality": float(q_strong.mean()),
            "realized_cost": acc["strong_mean_cost"],
            "naive_cost": acc["strong_mean_cost"],
        },
        "random_mixture": {
            "kind": "baseline",
            "quality": float(random_q),
            "realized_cost": float(random_c),
            "naive_cost": float(random_c),
            "note": "query-independent mix at the external router's strong call fraction; naive=realized",
        },
        "cost_matched_static": {
            "kind": "baseline",
            "quality": float(q_static.mean()),
            "realized_cost": float(c_static.mean()),
            "naive_cost": float(c_static.mean()),
        },
        "external_learned_router": {
            "kind": EXTERNAL_LEARNED_ROUTER,
            "quality": float(q.mean()),
            "realized_cost": acc["realized_cost"],
            "naive_cost": acc["naive_cost"],
        },
        "oracle_assignment": {
            "kind": ORACLE_ASSIGNMENT,
            "quality": float(q_or.mean()),
            "realized_cost": oracle_cost,
            "naive_cost": oracle_cost,
            "note": "hindsight upper bound; not a deployable router",
        },
    }

    return {
        "threshold": float(threshold),
        "assignment_kind": EXTERNAL_LEARNED_ROUTER,
        "n": panel.n,
        "routing_rate": acc["routing_rate"],
        "quality": float(q.mean()),
        "naive_cost": acc["naive_cost"],
        "realized_cost": acc["realized_cost"],
        "absolute_accounting_error": acc["absolute_accounting_error"],
        "relative_accounting_error": acc["relative_accounting_error"],
        "quality_advantage_vs_cost_matched_static": adv_real,
        "quality_advantage_vs_cost_matched_static_naive_cost": adv_naive,
        "cost_savings_at_matched_quality": save_real,
        "cost_savings_at_matched_quality_naive_cost": save_naive,
        "oracle_quality": float(q_or.mean()),
        "oracle_gap": float(q_or.mean() - q.mean()),
        "cost_regret_naive_vs_oracle": cost_regret_naive,
        "cost_regret_realized_vs_oracle": cost_regret_realized,
        "always_cheap_quality": float(q_cheap.mean()),
        "always_strong_quality": float(q_strong.mean()),
        "random_mixture_quality": float(random_q),
        "random_mixture_cost": float(random_c),
        "cost_matched_static_quality": float(q_static.mean()),
        "cost_matched_static_cost": float(c_static.mean()),
        "policies": policies,
        "vs_static_realized": vs_realized,
        "vs_static_naive_cost": vs_naive,
        "paired_vs_cost_matched_static": _compact_paired(paired_vs_static),
        "paired_vs_always_cheap": _compact_paired(paired_vs_cheap),
        "mean_effect": (paired_vs_static.get("mean_paired_difference") or {}).get("estimate"),
        "ci_lo": (paired_vs_static.get("mean_paired_difference") or {}).get("ci_lo"),
        "ci_hi": (paired_vs_static.get("mean_paired_difference") or {}).get("ci_hi"),
        "p_raw": (paired_vs_static.get("mean_paired_difference") or {}).get("p_raw"),
        "effect_size_cohens_dz": (paired_vs_static.get("cohens_dz") or {}).get("estimate"),
        "effect_size_cliffs_delta": (paired_vs_static.get("cliffs_delta") or {}).get("estimate"),
        "bootstrap_unit": "query",
        "n_boot": int(n_boot),
        "n_perm": int(n_perm),
    }


def _compact_paired(block: Mapping[str, Any]) -> dict[str, Any]:
    mean = block.get("mean_paired_difference") or {}
    dz = block.get("cohens_dz") or {}
    cliff = block.get("cliffs_delta") or {}
    return {
        "estimate": mean.get("estimate"),
        "ci_lo": mean.get("ci_lo"),
        "ci_hi": mean.get("ci_hi"),
        "p_raw": mean.get("p_raw"),
        "n": mean.get("n"),
        "cohens_dz": dz.get("estimate") if isinstance(dz, dict) else None,
        "cliffs_delta": cliff.get("estimate") if isinstance(cliff, dict) else None,
    }


def _rank_key(policy: Mapping[str, Any], *, cost_field: str) -> tuple[float, float]:
    q = float(policy["quality"])
    c = float(policy[cost_field])
    return (q, -c)


def rank_policies(policies: Mapping[str, Mapping[str, Any]], *, cost_field: str) -> list[str]:
    return sorted(POLICY_ORDER, key=lambda n: _rank_key(policies[n], cost_field=cost_field), reverse=True)


def classify_threshold(row: Mapping[str, Any]) -> dict[str, Any]:
    naive = float(row["naive_cost"])
    realized = float(row["realized_cost"])
    abs_err = float(row["absolute_accounting_error"])
    rel = row.get("relative_accounting_error")
    adv_n = row.get("quality_advantage_vs_cost_matched_static_naive_cost")
    adv_r = row.get("quality_advantage_vs_cost_matched_static")
    r_n = float(row["cost_regret_naive_vs_oracle"])
    r_r = float(row["cost_regret_realized_vs_oracle"])
    ranks_naive = rank_policies(row["policies"], cost_field="naive_cost")
    ranks_real = rank_policies(row["policies"], cost_field="realized_cost")
    negligible = (abs(abs_err) < NEGLIGIBLE_ABS) or (
        rel is not None and math.isfinite(float(rel)) and abs(float(rel)) < NEGLIGIBLE_REL
    )
    naive_wins = adv_n is not None and float(adv_n) > 0
    exact_loses = adv_r is None or float(adv_r) <= 0
    return {
        "threshold": row["threshold"],
        "accounting_sign_flip": bool(r_n * r_r < 0),
        "ranking_flip": ranks_naive != ranks_real,
        "naive_wins_exact_loses": bool(naive_wins and exact_loses),
        "negligible_accounting_difference": bool(negligible),
        "rank_naive_cost": ranks_naive,
        "rank_realized_cost": ranks_real,
        "router_rank_naive": ranks_naive.index("external_learned_router"),
        "router_rank_realized": ranks_real.index("external_learned_router"),
        "naive_advantage": adv_n,
        "realized_advantage": adv_r,
        "absolute_accounting_error": abs_err,
        "relative_accounting_error": rel,
    }


def query_rows_for_threshold(panel: ExternalRouterPanel, threshold: float) -> list[dict[str, Any]]:
    from woais_experiments.external_routing.reconstruct_assignments import panel_row

    assign = route_by_score(
        panel.scores, threshold, cheap=panel.cheap_model, strong=panel.strong_model
    )
    side = as_side_index(assign, cheap=panel.cheap_model, strong=panel.strong_model)
    q = _quality(side, panel.cheap_quality, panel.strong_quality)
    c = _cost(side, panel.cheap_cost, panel.strong_cost)
    rows = []
    for i, qid in enumerate(panel.query_ids):
        rows.append(
            panel_row(
                query_id=qid,
                router_score=float(panel.scores[i]),
                router_assignment=str(assign[i]),
                cheap_model=panel.cheap_model,
                strong_model=panel.strong_model,
                cheap_quality=float(panel.cheap_quality[i]),
                strong_quality=float(panel.strong_quality[i]),
                cheap_input_tokens=float(panel.cheap_input_tokens[i]),
                cheap_output_tokens=float(panel.cheap_output_tokens[i]),
                strong_input_tokens=float(panel.strong_input_tokens[i]),
                strong_output_tokens=float(panel.strong_output_tokens[i]),
                selected_model=str(assign[i]),
                selected_quality=float(q[i]),
                selected_realized_cost=float(c[i]),
                assignment_kind_value=EXTERNAL_LEARNED_ROUTER,
            )
        )
        rows[-1]["threshold"] = float(threshold)
    return rows

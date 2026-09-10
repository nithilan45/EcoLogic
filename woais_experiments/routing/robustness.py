"""Systematic robustness of the Stage 1–2 routing conclusion.

Primary claim
-------------
On frozen Stage 1–2, the EcoLogic keyword router is **dominated** by the
quality-matched static hull on realized USD (Always-Tier-2 is the hull vertex
under committed prices). This module re-evaluates that claim under plausible
changes to prices, overhead, latency penalties, quality cuts, budgets, RNG
seeds, bootstrap resamples, tail queries, and named static mixtures.

Prices are only **scaled or reassigned** from `configs/models.json`. Missing
latency is not invented: stored `latency_s` is used as-is. Significance is a
paired sign-flip p-value, never CI overlap.

CSV columns (tidy, one scenario per row)
----------------------------------------
setting, parameter, parameter_value, router_advantage, cost_difference,
quality_difference, significance, conclusion_changed

``conclusion_changed`` is true when ``status`` differs from the baseline hull
status. Bootstrap rows are included so a flip *rate* is visible; they do not
by themselves overturn the point-estimate claim.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from io import StringIO
from typing import Any, Mapping, Sequence

import numpy as np

from woais_experiments.accounting.breakeven import (
    STATUS_EPS,
    analytic_usd_threshold,
    classify_status,
)
from woais_experiments.accounting.costs import (
    TIERS,
    energy_j,
    paper_energy_rates,
    subst_energy_rates,
)
from woais_experiments.accounting.per_query_cost import (
    PriceTable,
    TokenPrices,
    load_price_table,
)
from woais_experiments.frozen import ItemMatrix, write_result
from woais_experiments.routing.frontier import interpolate_at_cost
from woais_experiments.routing.policies import random_tiers
from woais_experiments.routing.static_baselines import (
    always_cheapest,
    always_most_expensive,
    catalog,
    market_from_panel,
    uniform_mixture,
)
from woais_experiments.statistics.bootstrap import DEFAULT_SEED, SIGNIFICANCE_NOTE
from woais_experiments.statistics.effect_sizes import cohens_dz
from woais_experiments.statistics.paired_tests import benjamini_hochberg, paired_permutation_test

SCHEMA_VERSION = "1.0"
PRIMARY_STATUS = "dominated"
PRIMARY_CLAIM = (
    "EcoLogic keyword routing is dominated by the quality-matched static hull "
    "on realized USD (Always-Tier-2 under committed prices)."
)

CSV_FIELDS = (
    "setting",
    "parameter",
    "parameter_value",
    "router_advantage",
    "cost_difference",
    "quality_difference",
    "significance",
    "conclusion_changed",
)

# Extra columns: useful, but the eight above are the contract.
CSV_FIELDS_EXTENDED = CSV_FIELDS + (
    "status",
    "n",
    "p_raw",
    "p_adjusted",
    "effect_size",
    "router_quality",
    "router_cost",
    "claim_reversed",
)

OUTPUT_MULTS = (0.5, 1.0, 2.0, 5.0, 10.0)
OVERHEAD_USD = (0.0, 1e-5, 1e-4, 1e-3, 1e-2)
LATENCY_PENALTY_USD_PER_S = (0.0, 1e-5, 1e-4, 1e-3, 1e-2)
QUALITY_FLOORS = (0.0, 1.0)
# Pre-registered 2.5 pp grid on [0.80, 0.95]. Do not pin interior points to
# this evaluation set's EcoLogic / Always-T2 accuracies.
QUALITY_TARGETS = tuple(round(0.80 + i * 0.025, 3) for i in range(7))
BUDGET_MULTS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
SEED_GRID = (20260909, 20260905, 1, 42, 12345)
TAIL_FRACS = (0.01, 0.05, 0.10)
DEFAULT_N_PERM = 199
DEFAULT_N_BOOTSTRAP = 50
ALPHA = 0.05

# Named USD scale factors applied to committed $/1M rates (not invented rates).
MODEL_SCALE_GRID: tuple[tuple[str, dict[int, float]], ...] = (
    ("committed", {}),
    ("t1_x0.5", {1: 0.5}),
    ("t1_x2", {1: 2.0}),
    ("t2_x0.5", {2: 0.5}),
    ("t2_x2", {2: 2.0}),
    ("t2_x10", {2: 10.0}),
    ("t2_x50", {2: 50.0}),
    ("t3_x0.5", {3: 0.5}),
    ("t3_x2", {3: 2.0}),
)


@dataclass
class RobustnessPanel:
    """Query × model measurements. Cost axes are filled by the caller."""

    query_ids: tuple[str, ...]
    names: tuple[str, ...]
    tiers: tuple[int, ...]
    quality: np.ndarray
    prompt_tokens: np.ndarray
    completion_tokens: np.ndarray
    stored_usd: np.ndarray
    latency_s: np.ndarray
    energy_paper: np.ndarray
    energy_subst: np.ndarray
    choice: np.ndarray
    t2_index: int

    @property
    def n(self) -> int:
        return int(self.quality.shape[0])

    @property
    def m(self) -> int:
        return int(self.quality.shape[1])

    def take(self, idx: np.ndarray) -> "RobustnessPanel":
        idx = np.asarray(idx, dtype=int)
        return RobustnessPanel(
            query_ids=tuple(self.query_ids[i] for i in idx),
            names=self.names,
            tiers=self.tiers,
            quality=self.quality[idx],
            prompt_tokens=self.prompt_tokens[idx],
            completion_tokens=self.completion_tokens[idx],
            stored_usd=self.stored_usd[idx],
            latency_s=self.latency_s[idx],
            energy_paper=self.energy_paper[idx],
            energy_subst=self.energy_subst[idx],
            choice=self.choice[idx],
            t2_index=self.t2_index,
        )


def panel_from_item_matrix(
    matrix: ItemMatrix,
    assignment_tiers: Mapping[Any, int],
    *,
    tiers: tuple[int, ...] = TIERS,
) -> RobustnessPanel:
    names = tuple(matrix.model_of[t] for t in tiers)
    n = matrix.n
    m = len(tiers)
    quality = np.zeros((n, m), dtype=float)
    prompt = np.zeros((n, m), dtype=float)
    completion = np.zeros((n, m), dtype=float)
    stored = np.zeros((n, m), dtype=float)
    latency = np.zeros((n, m), dtype=float)
    e_paper = np.zeros((n, m), dtype=float)
    e_subst = np.zeros((n, m), dtype=float)
    paper = paper_energy_rates()
    subst = subst_energy_rates()
    ids = tuple(matrix.item_ids)
    col = {int(t): k for k, t in enumerate(tiers)}
    choice = np.empty(n, dtype=int)
    for i, qid in enumerate(ids):
        t_hat = int(assignment_tiers[qid])
        if t_hat not in col:
            raise KeyError(f"assignment tier {t_hat} is not in {tiers}")
        choice[i] = col[t_hat]
        for k, t in enumerate(tiers):
            quality[i, k] = 1.0 if matrix.correct[(t, qid)] else 0.0
            prompt[i, k] = float(matrix.prompt_tokens[(t, qid)])
            completion[i, k] = float(matrix.completion_tokens[(t, qid)])
            stored[i, k] = float(matrix.usd[(t, qid)])
            ls = matrix.latency_s.get((t, qid))
            latency[i, k] = float("nan") if ls is None else float(ls)
            tok = float(matrix.tokens[(t, qid)])
            e_paper[i, k] = energy_j(tok, paper[t])
            e_subst[i, k] = energy_j(tok, subst[t])
    t2_index = col.get(2, 1 if m > 1 else 0)
    return RobustnessPanel(
        query_ids=ids,
        names=names,
        tiers=tuple(int(t) for t in tiers),
        quality=quality,
        prompt_tokens=prompt,
        completion_tokens=completion,
        stored_usd=stored,
        latency_s=latency,
        energy_paper=e_paper,
        energy_subst=e_subst,
        choice=choice,
        t2_index=int(t2_index),
    )


def panel_from_arrays(
    *,
    names: Sequence[str],
    quality: np.ndarray,
    cost: np.ndarray,
    choice: np.ndarray,
    query_ids: Sequence[Any] | None = None,
    t2_index: int = 1,
    latency_s: np.ndarray | None = None,
    completion_tokens: np.ndarray | None = None,
) -> RobustnessPanel:
    """Minimal panel for unit tests. ``cost`` is stored as ``stored_usd``."""
    quality = np.asarray(quality, dtype=float)
    cost = np.asarray(cost, dtype=float)
    choice = np.asarray(choice, dtype=int)
    n, m = quality.shape
    if cost.shape != (n, m) or choice.shape != (n,):
        raise ValueError("quality, cost, choice shape mismatch")
    ids = tuple(str(i) if query_ids is None else str(query_ids[i]) for i in range(n))
    z = np.zeros((n, m), dtype=float)
    lat = z.copy() if latency_s is None else np.asarray(latency_s, dtype=float)
    ct = z.copy() if completion_tokens is None else np.asarray(completion_tokens, dtype=float)
    return RobustnessPanel(
        query_ids=ids,
        names=tuple(names),
        tiers=tuple(range(1, m + 1)),
        quality=quality,
        prompt_tokens=z.copy(),
        completion_tokens=ct,
        stored_usd=cost,
        latency_s=lat,
        energy_paper=z.copy(),
        energy_subst=z.copy(),
        choice=choice,
        t2_index=int(t2_index),
    )


def _scale_for_tier(model_scale: Mapping[int, float] | None, tier: int) -> float:
    if not model_scale:
        return 1.0
    return float(model_scale.get(int(tier), 1.0))


def usd_cost_panel(
    panel: RobustnessPanel,
    prices: PriceTable,
    *,
    output_mult: float = 1.0,
    model_scale: Mapping[int, float] | None = None,
    table: PriceTable | None = None,
) -> np.ndarray:
    """USD from committed (or reassigned) $/1M rates × stored tokens."""
    src = table if table is not None else prices
    n, m = panel.quality.shape
    out = np.zeros((n, m), dtype=float)
    for k, (name, tier) in enumerate(zip(panel.names, panel.tiers)):
        p = src.require(name)
        scale = _scale_for_tier(model_scale, tier)
        in_rate = p.input_usd_per_million * scale / 1_000_000.0
        out_rate = p.output_usd_per_million * scale * float(output_mult) / 1_000_000.0
        out[:, k] = panel.prompt_tokens[:, k] * in_rate + panel.completion_tokens[:, k] * out_rate
    return out


def swap_t1_t2_table(prices: PriceTable, panel: RobustnessPanel) -> PriceTable:
    """Reassign committed T1/T2 rates; do not invent numbers."""
    if len(panel.names) < 2:
        raise ValueError("need at least two models to swap T1/T2 rates")
    n1, n2 = panel.names[0], panel.names[1]
    p1, p2 = prices.require(n1), prices.require(n2)
    rows: dict[str, TokenPrices] = {}
    for name in prices.models():
        src = prices.require(name)
        rows[name] = TokenPrices(name, src.input_usd_per_million, src.output_usd_per_million)
    rows[n1] = TokenPrices(n1, p2.input_usd_per_million, p2.output_usd_per_million)
    rows[n2] = TokenPrices(n2, p1.input_usd_per_million, p1.output_usd_per_million)
    return PriceTable(rows)


def apply_latency_penalty(cost: np.ndarray, latency_s: np.ndarray, usd_per_s: float) -> np.ndarray:
    """Add λ × measured latency. Non-finite latency is not imputed as 0."""
    if usd_per_s == 0.0:
        return np.asarray(cost, dtype=float)
    lat = np.asarray(latency_s, dtype=float)
    if not np.all(np.isfinite(lat)):
        raise ValueError("latency penalty requires finite latency on every cell")
    return np.asarray(cost, dtype=float) + lat * float(usd_per_s)


def pareto_status(quality_gap: float, cost_gap: float, *, eps: float = STATUS_EPS) -> str:
    """Router minus comparator. Positive quality / negative cost is a win."""
    better_q = quality_gap > eps
    worse_q = quality_gap < -eps
    cheaper = cost_gap < -eps
    costlier = cost_gap > eps
    if better_q and cheaper:
        return "beneficial"
    if worse_q and costlier:
        return "dominated"
    if abs(quality_gap) <= eps and abs(cost_gap) <= eps:
        return "neutral"
    return "tradeoff"


def format_significance(
    p_raw: float | None,
    *,
    p_adjusted: float | None = None,
    effect_size: float | None = None,
    alpha: float = ALPHA,
) -> str:
    p = p_adjusted if p_adjusted is not None else p_raw
    if p is None or not math.isfinite(p):
        return "n/a"
    if p_adjusted is not None:
        label = "p_adj<0.05" if p < alpha else "p_adj>=0.05"
    else:
        label = "p<0.05" if p < alpha else "p>=0.05"
    bits = [f"{p:.6g} ({label})"]
    if p_raw is not None and p_adjusted is not None and math.isfinite(p_raw):
        bits.append(f"p_raw={p_raw:.6g}")
    if effect_size is not None and math.isfinite(effect_size):
        bits.append(f"dz={effect_size:.4g}")
    return "; ".join(bits)


def drop_highest_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    """Keep all but the highest ``fraction`` of values. Keep at least one row."""
    n = int(values.size)
    mask = np.ones(n, dtype=bool)
    if fraction <= 0.0 or n == 0:
        return mask
    n_drop = int(math.floor(n * float(fraction) + 1e-12))
    if n_drop <= 0:
        return mask
    if n_drop >= n:
        n_drop = n - 1
    order = np.argsort(values, kind="mergesort")
    mask[order[-n_drop:]] = False
    return mask


def _selected(panel: np.ndarray, choice: np.ndarray) -> np.ndarray:
    rows = np.arange(panel.shape[0])
    return panel[rows, choice]


def _quality_perm_p(
    q_router: np.ndarray,
    q_t2: np.ndarray,
    *,
    n_perm: int,
    seed: int,
) -> float | None:
    if n_perm <= 0 or q_router.size == 0:
        return None
    got = paired_permutation_test(
        q_router, q_t2, "mean", n_perm=int(n_perm), seed=int(seed)
    )
    p = got.get("p_raw")
    if p is None or not np.isfinite(p):
        return None
    return float(p)


def _quality_effect(q_router: np.ndarray, q_t2: np.ndarray) -> float | None:
    est = cohens_dz(q_router, q_t2).get("estimate")
    if est is None or not np.isfinite(est):
        return None
    return float(est)


def score_hull(
    names: Sequence[str],
    quality: np.ndarray,
    cost: np.ndarray,
    choice: np.ndarray,
    *,
    overhead_usd: float = 0.0,
    t2_index: int = 1,
    n_perm: int = 0,
    seed: int = DEFAULT_SEED,
    eps: float = STATUS_EPS,
    budget: float | None = None,
) -> dict[str, Any]:
    """Score a router against the quality-matched static hull (and optional budget)."""
    names = tuple(names)
    quality = np.asarray(quality, dtype=float)
    cost = np.asarray(cost, dtype=float)
    choice = np.asarray(choice, dtype=int)
    n = quality.shape[0]
    if n == 0:
        raise ValueError("empty panel")
    q_sel = _selected(quality, choice)
    c_sel = _selected(cost, choice) + float(overhead_usd)
    q_r = float(q_sel.mean())
    c_r = float(c_sel.mean())
    market = market_from_panel(names, cost, quality)
    usd = analytic_usd_threshold(market, c_r, q_r, eps=eps)
    raw = usd["raw_router_savings"]
    status = classify_status(raw, eps=eps)
    router_advantage = usd["quality_advantage_at_router_cost"]
    static_c = usd["static_cost_at_same_quality"]
    if raw is not None and isinstance(raw, float) and math.isinf(raw):
        cost_difference = -raw
    elif static_c is None:
        cost_difference = None
    else:
        cost_difference = float(c_r) - float(static_c)

    t2 = int(t2_index) if 0 <= t2_index < quality.shape[1] else 0
    q_t2 = quality[:, t2]
    quality_difference = q_r - float(q_t2.mean())
    p_raw = _quality_perm_p(q_sel, q_t2, n_perm=n_perm, seed=seed)
    effect = _quality_effect(q_sel, q_t2)

    note = usd.get("note") or ""
    if budget is not None:
        b = float(budget)
        feasible = c_r <= b + eps
        at_b = interpolate_at_cost(market, b, mode="at_most", eps=eps)
        if not feasible:
            status = "infeasible"
            router_advantage = None
            cost_difference = c_r - b
            note = "router realized cost exceeds the stated per-query budget"
        elif at_b.feasible:
            router_advantage = q_r - float(at_b.quality)
            cost_difference = c_r - b
            if router_advantage > eps:
                status = "beneficial"
            elif router_advantage < -eps:
                status = "dominated"
            else:
                status = "neutral"
            note = at_b.note or "static max quality at budget (at_most)"
        else:
            status = "infeasible"
            router_advantage = None
            cost_difference = c_r - b
            note = at_b.note or "static mix infeasible at budget"

    return {
        "n": n,
        "status": status,
        "router_advantage": router_advantage,
        "cost_difference": cost_difference,
        "quality_difference": quality_difference,
        "p_raw": p_raw,
        "effect_size": effect,
        "significance": format_significance(p_raw, effect_size=effect),
        "router_quality": q_r,
        "router_cost": c_r,
        "static_cost_at_same_quality": static_c,
        "static_quality_at_router_cost": usd.get("static_quality_at_router_cost"),
        "raw_router_savings": raw,
        "t2_quality": float(q_t2.mean()),
        "t2_cost": float(cost[:, t2].mean()),
        "catalog_hull": catalog(market)["hull_model_names"],
        "note": note,
        "significance_note": SIGNIFICANCE_NOTE,
        "p_raw_estimand": "paired_quality_vs_always_t2",
        "primary_claim_tested_by_p_raw": False,
    }


def score_named_mix(
    names: Sequence[str],
    quality: np.ndarray,
    cost: np.ndarray,
    choice: np.ndarray,
    weights: Mapping[str, float],
    *,
    overhead_usd: float = 0.0,
    t2_index: int = 1,
    n_perm: int = 0,
    seed: int = DEFAULT_SEED,
    eps: float = STATUS_EPS,
    mix_kind: str = "named_mix",
) -> dict[str, Any]:
    names = tuple(names)
    quality = np.asarray(quality, dtype=float)
    cost = np.asarray(cost, dtype=float)
    choice = np.asarray(choice, dtype=int)
    q_sel = _selected(quality, choice)
    c_sel = _selected(cost, choice) + float(overhead_usd)
    q_r = float(q_sel.mean())
    c_r = float(c_sel.mean())
    market = market_from_panel(names, cost, quality)
    mix = market.mixture(dict(weights), kind=mix_kind)
    dq = q_r - float(mix.quality)
    dc = c_r - float(mix.cost)
    t2 = int(t2_index) if 0 <= t2_index < quality.shape[1] else 0
    q_t2 = quality[:, t2]
    p_raw = _quality_perm_p(q_sel, q_t2, n_perm=n_perm, seed=seed)
    effect = _quality_effect(q_sel, q_t2)
    return {
        "n": int(quality.shape[0]),
        "status": pareto_status(dq, dc, eps=eps),
        "router_advantage": dq,
        "cost_difference": dc,
        "quality_difference": q_r - float(q_t2.mean()),
        "p_raw": p_raw,
        "effect_size": effect,
        "significance": format_significance(p_raw, effect_size=effect),
        "router_quality": q_r,
        "router_cost": c_r,
        "mix_quality": float(mix.quality),
        "mix_cost": float(mix.cost),
        "mix_weights": dict(mix.weights),
        "note": mix.kind,
        "significance_note": SIGNIFICANCE_NOTE,
    }


def score_quality_target(
    names: Sequence[str],
    quality: np.ndarray,
    cost: np.ndarray,
    choice: np.ndarray,
    target: float,
    *,
    overhead_usd: float = 0.0,
    t2_index: int = 1,
    n_perm: int = 0,
    seed: int = DEFAULT_SEED,
    eps: float = STATUS_EPS,
) -> dict[str, Any]:
    """Does the router meet quality ``target`` at lower cost than the static hull?"""
    names = tuple(names)
    quality = np.asarray(quality, dtype=float)
    cost = np.asarray(cost, dtype=float)
    choice = np.asarray(choice, dtype=int)
    q_sel = _selected(quality, choice)
    c_sel = _selected(cost, choice) + float(overhead_usd)
    q_r = float(q_sel.mean())
    c_r = float(c_sel.mean())
    market = market_from_panel(names, cost, quality)
    usd = analytic_usd_threshold(market, c_r, float(target), eps=eps)
    t2 = int(t2_index) if 0 <= t2_index < quality.shape[1] else 0
    p_raw = _quality_perm_p(q_sel, quality[:, t2], n_perm=n_perm, seed=seed)
    effect = _quality_effect(q_sel, quality[:, t2])
    meets = q_r + eps >= float(target)
    if not meets:
        status = "dominated"
        router_advantage = q_r - float(target)
        static_c = usd["static_cost_at_same_quality"]
        cost_difference = None if static_c is None else c_r - float(static_c)
        note = "router mean quality is below the stated threshold"
    else:
        raw = usd["raw_router_savings"]
        status = classify_status(raw, eps=eps)
        if raw is not None and isinstance(raw, float) and math.isinf(raw):
            cost_difference = -raw
        elif usd["static_cost_at_same_quality"] is None:
            cost_difference = None
        else:
            cost_difference = c_r - float(usd["static_cost_at_same_quality"])
        router_advantage = q_r - float(target)
        note = usd.get("note") or "router meets quality target; cost vs min-cost static"
    return {
        "n": int(quality.shape[0]),
        "status": status,
        "router_advantage": router_advantage,
        "cost_difference": cost_difference,
        "quality_difference": q_r - float(quality[:, t2].mean()),
        "p_raw": p_raw,
        "effect_size": effect,
        "significance": format_significance(p_raw, effect_size=effect),
        "router_quality": q_r,
        "router_cost": c_r,
        "note": note,
        "significance_note": SIGNIFICANCE_NOTE,
    }


def _row(
    setting: str,
    parameter: str,
    parameter_value: Any,
    scored: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "setting": setting,
        "parameter": parameter,
        "parameter_value": parameter_value,
        "router_advantage": scored.get("router_advantage"),
        "cost_difference": scored.get("cost_difference"),
        "quality_difference": scored.get("quality_difference"),
        "significance": scored.get("significance") or "n/a",
        "status": scored.get("status"),
        "n": scored.get("n"),
        "p_raw": scored.get("p_raw"),
        "p_adjusted": scored.get("p_adjusted"),
        "effect_size": scored.get("effect_size"),
        "router_quality": scored.get("router_quality"),
        "router_cost": scored.get("router_cost"),
        "note": scored.get("note"),
        "exclusion_basis": scored.get("exclusion_basis"),
        "p_raw_estimand": scored.get("p_raw_estimand") or "paired_quality_vs_always_t2",
        "primary_claim_tested_by_p_raw": scored.get("primary_claim_tested_by_p_raw", False),
        "conclusion_changed": None,
    }


def attach_bh(rows: list[dict[str, Any]], *, alpha: float = ALPHA) -> list[dict[str, Any]]:
    """BH-adjust structural (non-bootstrap) permutation p-values as one family."""
    idx = [
        i for i, row in enumerate(rows)
        if row.get("setting") != "bootstrap_resample" and row.get("p_raw") is not None
    ]
    bh = benjamini_hochberg([rows[i]["p_raw"] for i in idx], alpha=alpha)
    for j, i in enumerate(idx):
        rows[i]["p_adjusted"] = bh["p_adjusted"][j]
        rows[i]["significance"] = format_significance(
            rows[i].get("p_raw"),
            p_adjusted=rows[i]["p_adjusted"],
            effect_size=rows[i].get("effect_size"),
        )
    for row in rows:
        if row.get("setting") == "bootstrap_resample":
            row["p_adjusted"] = None
            row["significance"] = "n/a (bootstrap sensitivity; not a test)"
    return rows


def attach_conclusion_flags(
    rows: list[dict[str, Any]],
    baseline_status: str,
) -> list[dict[str, Any]]:
    for row in rows:
        status = row.get("status")
        changed = status is not None and status != baseline_status
        row["conclusion_changed"] = bool(changed)
        row["claim_reversed"] = bool(
            status == "beneficial" and baseline_status != "beneficial"
        )
    return rows


def _committed_usd(panel: RobustnessPanel, prices: PriceTable) -> np.ndarray:
    return usd_cost_panel(panel, prices)


def run_sweep(
    panel: RobustnessPanel,
    *,
    prices: PriceTable | None = None,
    n_perm: int = DEFAULT_N_PERM,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    include_bootstrap: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate every robustness axis. Point estimates use the full panel."""
    prices = prices if prices is not None else load_price_table()
    rng = np.random.default_rng(int(seed))
    cost0 = _committed_usd(panel, prices)
    base = score_hull(
        panel.names, panel.quality, cost0, panel.choice,
        t2_index=panel.t2_index, n_perm=n_perm, seed=seed,
    )
    rows: list[dict[str, Any]] = [
        _row("baseline", "none", "committed_usd", base),
    ]

    # 1. model pricing assumptions
    for label, scales in MODEL_SCALE_GRID:
        if label == "committed":
            continue
        cost = usd_cost_panel(panel, prices, model_scale=scales)
        scored = score_hull(
            panel.names, panel.quality, cost, panel.choice,
            t2_index=panel.t2_index, n_perm=n_perm, seed=seed,
        )
        rows.append(_row("model_pricing", "usd_scale", label, scored))
    stored = score_hull(
        panel.names, panel.quality, panel.stored_usd, panel.choice,
        t2_index=panel.t2_index, n_perm=n_perm, seed=seed,
    )
    rows.append(_row("model_pricing", "usd_source", "stored_usd", stored))
    swap = usd_cost_panel(panel, prices, table=swap_t1_t2_table(prices, panel))
    rows.append(_row(
        "model_pricing", "usd_source", "swap_t1_t2_rates",
        score_hull(
            panel.names, panel.quality, swap, panel.choice,
            t2_index=panel.t2_index, n_perm=n_perm, seed=seed,
        ),
    ))
    rows.append(_row(
        "model_pricing", "cost_axis", "energy_paper",
        score_hull(
            panel.names, panel.quality, panel.energy_paper, panel.choice,
            t2_index=panel.t2_index, n_perm=n_perm, seed=seed,
        ),
    ))
    rows.append(_row(
        "model_pricing", "cost_axis", "energy_subst",
        score_hull(
            panel.names, panel.quality, panel.energy_subst, panel.choice,
            t2_index=panel.t2_index, n_perm=n_perm, seed=seed,
        ),
    ))

    # 2. output-token price multipliers
    for mult in OUTPUT_MULTS:
        cost = usd_cost_panel(panel, prices, output_mult=mult)
        rows.append(_row(
            "output_token_multiplier", "output_mult", mult,
            score_hull(
                panel.names, panel.quality, cost, panel.choice,
                t2_index=panel.t2_index, n_perm=n_perm, seed=seed,
            ),
        ))

    # 3. router overhead (USD / query, router only)
    for h in OVERHEAD_USD:
        rows.append(_row(
            "router_overhead", "overhead_usd", h,
            score_hull(
                panel.names, panel.quality, cost0, panel.choice,
                overhead_usd=h, t2_index=panel.t2_index, n_perm=n_perm, seed=seed,
            ),
        ))

    # 4. latency penalties (USD per second of measured latency, all models)
    lat_ok = np.isfinite(panel.latency_s).all(axis=1)
    if int(lat_ok.sum()) == 0:
        pass
    else:
        lat_panel = panel if bool(np.all(lat_ok)) else panel.take(np.flatnonzero(lat_ok))
        lat_cost0 = _committed_usd(lat_panel, prices)
        for lam in LATENCY_PENALTY_USD_PER_S:
            cost = apply_latency_penalty(lat_cost0, lat_panel.latency_s, lam)
            rows.append(_row(
                "latency_penalty", "usd_per_second", lam,
                score_hull(
                    lat_panel.names, lat_panel.quality, cost, lat_panel.choice,
                    t2_index=lat_panel.t2_index, n_perm=n_perm, seed=seed,
                ),
            ))

    # 5. quality thresholds
    max_q = panel.quality.max(axis=1)
    for floor in QUALITY_FLOORS:
        keep = max_q >= float(floor) - 1e-15
        if int(keep.sum()) == 0:
            continue
        sub = panel.take(np.flatnonzero(keep))
        cost = _committed_usd(sub, prices)
        rows.append(_row(
            "quality_threshold", "min_max_quality", floor,
            score_hull(
                sub.names, sub.quality, cost, sub.choice,
                t2_index=sub.t2_index, n_perm=n_perm, seed=seed,
            ),
        ))
    for tau in QUALITY_TARGETS:
        rows.append(_row(
            "quality_threshold", "target_quality", tau,
            score_quality_target(
                panel.names, panel.quality, cost0, panel.choice, tau,
                t2_index=panel.t2_index, n_perm=n_perm, seed=seed,
            ),
        ))

    # 6. routing budgets (multiples of realized router mean USD)
    c_r = float(_selected(cost0, panel.choice).mean())
    for mlt in BUDGET_MULTS:
        rows.append(_row(
            "routing_budget", "budget_mult_of_router_mean", mlt,
            score_hull(
                panel.names, panel.quality, cost0, panel.choice,
                t2_index=panel.t2_index, n_perm=n_perm, seed=seed,
                budget=mlt * c_r,
            ),
        ))

    # 7. random seed variation (permutation RNG; point status is seed-invariant)
    for s in SEED_GRID:
        rows.append(_row(
            "random_seed", "permutation_seed", s,
            score_hull(
                panel.names, panel.quality, cost0, panel.choice,
                t2_index=panel.t2_index, n_perm=n_perm, seed=int(s),
            ),
        ))
    items = list(panel.query_ids)
    for s in SEED_GRID:
        rand = random_tiers(items, seed=int(s))
        col = {t: k for k, t in enumerate(panel.tiers)}
        other = np.array([col[int(rand[qid])] for qid in panel.query_ids], dtype=int)
        q_r = float(_selected(panel.quality, panel.choice).mean())
        c_r_row = float(_selected(cost0, panel.choice).mean())
        q_o = float(_selected(panel.quality, other).mean())
        c_o = float(_selected(cost0, other).mean())
        scored = {
            "n": panel.n,
            "status": pareto_status(q_r - q_o, c_r_row - c_o),
            "router_advantage": q_r - q_o,
            "cost_difference": c_r_row - c_o,
            "quality_difference": q_r - float(panel.quality[:, panel.t2_index].mean()),
            "p_raw": _quality_perm_p(
                _selected(panel.quality, panel.choice),
                panel.quality[:, panel.t2_index],
                n_perm=n_perm, seed=seed,
            ),
            "effect_size": _quality_effect(
                _selected(panel.quality, panel.choice),
                panel.quality[:, panel.t2_index],
            ),
            "router_quality": q_r,
            "router_cost": c_r_row,
            "note": f"pareto vs random_tiers seed={s}",
        }
        scored["significance"] = format_significance(
            scored["p_raw"], effect_size=scored.get("effect_size")
        )
        rows.append(_row("random_seed", "random_policy_seed", s, scored))

    # 8. bootstrap resamples
    if include_bootstrap and n_bootstrap > 0:
        for b in range(int(n_bootstrap)):
            idx = rng.integers(0, panel.n, size=panel.n)
            boot = panel.take(idx)
            cost = _committed_usd(boot, prices)
            rows.append(_row(
                "bootstrap_resample", "replicate", b,
                score_hull(
                    boot.names, boot.quality, cost, boot.choice,
                    t2_index=boot.t2_index, n_perm=0, seed=seed,
                ),
            ))

    # 9. drop highest-cost queries (EcoLogic realized USD — policy-conditioned)
    eco_cost = _selected(cost0, panel.choice)
    for frac in TAIL_FRACS:
        keep = drop_highest_mask(eco_cost, frac)
        sub = panel.take(np.flatnonzero(keep))
        cost = _committed_usd(sub, prices)
        scored = score_hull(
            sub.names, sub.quality, cost, sub.choice,
            t2_index=sub.t2_index, n_perm=n_perm, seed=seed,
        )
        scored["exclusion_basis"] = "ecologic_realized_usd"
        scored["note"] = (
            (scored.get("note") or "")
            + "; tail drop conditioned on EcoLogic realized USD, not a policy-neutral sample"
        ).strip("; ")
        rows.append(_row("drop_highest_cost", "fraction", frac, scored))

    # 10. drop longest-output queries (EcoLogic completion tokens — policy-conditioned)
    eco_out = _selected(panel.completion_tokens, panel.choice)
    for frac in TAIL_FRACS:
        keep = drop_highest_mask(eco_out, frac)
        sub = panel.take(np.flatnonzero(keep))
        cost = _committed_usd(sub, prices)
        scored = score_hull(
            sub.names, sub.quality, cost, sub.choice,
            t2_index=sub.t2_index, n_perm=n_perm, seed=seed,
        )
        scored["exclusion_basis"] = "ecologic_completion_tokens"
        scored["note"] = (
            (scored.get("note") or "")
            + "; tail drop conditioned on EcoLogic completion tokens"
        ).strip("; ")
        rows.append(_row("drop_longest_output", "fraction", frac, scored))

    # 11. named static baseline mixtures
    market = market_from_panel(panel.names, cost0, panel.quality)
    named: list[tuple[str, Mapping[str, float]]] = []
    for i, name in enumerate(panel.names):
        named.append((f"always_{name}", {name: 1.0}))
        named.append((f"always_t{panel.tiers[i]}", {name: 1.0}))
    named.append(("uniform", {n: 1.0 / len(panel.names) for n in panel.names}))
    named.append(("always_cheapest", always_cheapest(market).weights))
    named.append(("always_most_expensive", always_most_expensive(market).weights))
    named.append(("uniform_mixture", uniform_mixture(market).weights))
    if len(panel.names) >= 2:
        a, b = panel.names[0], panel.names[1]
        named.append((f"pairwise_50_{a[:12]}_{b[:12]}", {a: 0.5, b: 0.5}))
    if len(panel.names) >= 3:
        b, c = panel.names[1], panel.names[2]
        a = panel.names[0]
        named.append((f"pairwise_50_t2_t3", {b: 0.5, c: 0.5}))
        named.append((f"pairwise_50_t1_t3", {a: 0.5, c: 0.5}))
    seen: set[str] = set()
    for label, weights in named:
        if label in seen:
            continue
        seen.add(label)
        rows.append(_row(
            "static_baseline_mixture", "mixture", label,
            score_named_mix(
                panel.names, panel.quality, cost0, panel.choice, weights,
                t2_index=panel.t2_index, n_perm=n_perm, seed=seed, mix_kind=label,
            ),
        ))
    rows.append(_row(
        "static_baseline_mixture", "mixture", "cost_matched_hull",
        score_hull(
            panel.names, panel.quality, cost0, panel.choice,
            t2_index=panel.t2_index, n_perm=n_perm, seed=seed,
        ),
    ))

    attach_bh(rows)
    attach_conclusion_flags(rows, str(base["status"]))
    return rows


HULL_SETTINGS = {
    "baseline",
    "model_pricing",
    "output_token_multiplier",
    "router_overhead",
    "latency_penalty",
    "quality_threshold",
    "routing_budget",
    "drop_highest_cost",
    "drop_longest_output",
}


def _sig_tag(value: Any) -> str:
    text = "" if value is None else str(value)
    if "p_adj<0.05" in text or ("p<0.05" in text and "p_adj" not in text):
        return "p<0.05"
    if "p_adj>=0.05" in text or "p>=0.05" in text:
        return "p>=0.05"
    return "n/a"


def _is_hull_row(row: Mapping[str, Any]) -> bool:
    setting = str(row.get("setting"))
    parameter = str(row.get("parameter"))
    value = str(row.get("parameter_value"))
    if setting in HULL_SETTINGS:
        return True
    if setting == "random_seed" and parameter == "permutation_seed":
        return True
    if setting == "static_baseline_mixture" and value == "cost_matched_hull":
        return True
    if setting == "bootstrap_resample":
        return True
    return False


def summarize(rows: Sequence[Mapping[str, Any]], *, baseline_status: str) -> dict[str, Any]:
    structural = [r for r in rows if r.get("setting") != "bootstrap_resample"]
    boots = [r for r in rows if r.get("setting") == "bootstrap_resample"]
    hull_structural = [r for r in structural if _is_hull_row(r)]
    alt_structural = [r for r in structural if not _is_hull_row(r)]
    flips = [r for r in structural if r.get("conclusion_changed")]
    hull_flips = [r for r in hull_structural if r.get("conclusion_changed")]
    alt_flips = [r for r in alt_structural if r.get("conclusion_changed")]
    hull_sig_changed = [
        r for r in hull_structural
        if r.get("setting") != "baseline"
        and _sig_tag(r.get("significance")) != "n/a"
        and _sig_tag(r.get("significance")) != _sig_tag(rows[0].get("significance") if rows else None)
    ]
    hull_reversed = [r for r in hull_structural if r.get("claim_reversed")]
    boot_flips = [r for r in boots if r.get("conclusion_changed")]
    boot_reversed = [r for r in boots if r.get("claim_reversed")]
    by_setting: dict[str, int] = {}
    for r in flips:
        by_setting[str(r["setting"])] = by_setting.get(str(r["setting"]), 0) + 1
    return {
        "primary_claim": PRIMARY_CLAIM,
        "baseline_status": baseline_status,
        "n_rows": len(rows),
        "n_structural": len(structural),
        "n_bootstrap": len(boots),
        "n_structural_flips": len(flips),
        "n_hull_flips": len(hull_flips),
        "n_alternative_comparator_flips": len(alt_flips),
        "structural_flip_rate": (len(flips) / len(structural)) if structural else 0.0,
        "hull_flip_rate": (len(hull_flips) / len(hull_structural)) if hull_structural else 0.0,
        "n_bootstrap_flips": len(boot_flips),
        "bootstrap_flip_rate": (len(boot_flips) / len(boots)) if boots else None,
        "flips_by_setting": by_setting,
        "n_hull_significance_changed": len(hull_sig_changed),
        "hull_significance_changed_rows": [
            {
                "setting": r["setting"],
                "parameter": r["parameter"],
                "parameter_value": r["parameter_value"],
                "significance": r.get("significance"),
                "status": r.get("status"),
            }
            for r in hull_sig_changed
        ],
        "n_hull_claim_reversed": len(hull_reversed),
        "n_bootstrap_claim_reversed": len(boot_reversed),
        "bootstrap_claim_reversed_rate": (
            (len(boot_reversed) / len(boots)) if boots else None
        ),
        "primary_conclusion_survives_hull_perturbations": len(hull_flips) == 0,
        "primary_conclusion_survives_structural": len(flips) == 0,
        "primary_conclusion_survives_became_beneficial_only": len(hull_reversed) == 0,
        "survival_definition": (
            "survives_hull_perturbations is false if any hull-family status differs "
            "from baseline (including infeasible and tradeoff). "
            "survives_became_beneficial_only is the old one-sided flag and does not "
            "falsify a domination claim. p_raw tests quality vs Always-T2, not USD hull domination."
        ),
        "flip_rows": [
            {
                "setting": r["setting"],
                "parameter": r["parameter"],
                "parameter_value": r["parameter_value"],
                "status": r.get("status"),
                "router_advantage": r.get("router_advantage"),
                "cost_difference": r.get("cost_difference"),
                "quality_difference": r.get("quality_difference"),
            }
            for r in hull_flips
        ],
        "alternative_comparator_flip_rows": [
            {
                "setting": r["setting"],
                "parameter": r["parameter"],
                "parameter_value": r["parameter_value"],
                "status": r.get("status"),
            }
            for r in alt_flips
        ],
        "significance_note": SIGNIFICANCE_NOTE,
    }


def _fmt_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (np.bool_,)):
        return "true" if bool(value) else "false"
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if math.isnan(v):
            return ""
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    return value


def rows_to_csv(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str] = CSV_FIELDS,
) -> str:
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _fmt_cell(row.get(k)) for k in fields})
    return buf.getvalue()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        if math.isnan(obj):
            return None
        return "Infinity" if obj > 0 else "-Infinity"
    return obj


def save_robustness(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    relpath: str = "routing/robustness",
) -> dict[str, str]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "primary_claim": PRIMARY_CLAIM,
        "summary": dict(summary),
        "rows": list(rows),
    }
    json_path = write_result(f"{relpath}.json", _jsonable(payload))
    csv_path = write_result(f"{relpath}.csv", rows_to_csv(rows, CSV_FIELDS))
    wide = write_result(f"{relpath}_extended.csv", rows_to_csv(rows, CSV_FIELDS_EXTENDED))
    return {"json": str(json_path), "csv": str(csv_path), "extended_csv": str(wide)}


def run_stage12(
    matrix: ItemMatrix,
    assignment_tiers: Mapping[Any, int],
    *,
    n_perm: int = DEFAULT_N_PERM,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    relpath: str = "routing/robustness",
) -> dict[str, Any]:
    panel = panel_from_item_matrix(matrix, assignment_tiers)
    rows = run_sweep(panel, n_perm=n_perm, n_bootstrap=n_bootstrap, seed=seed)
    baseline_status = str(rows[0]["status"])
    summary = summarize(rows, baseline_status=baseline_status)
    saved = save_robustness(rows, summary, relpath=relpath)
    return {
        "schema_version": SCHEMA_VERSION,
        "n": panel.n,
        "baseline_status": baseline_status,
        "summary": summary,
        "n_rows": len(rows),
        "saved": saved,
        "primary_claim": PRIMARY_CLAIM,
        "n_perm": int(n_perm),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }


def main() -> None:
    from woais_experiments.frozen import load_stage12_matrix, load_stage12_routing
    from woais_experiments.routing.policies import build_stage12_policies

    matrix = load_stage12_matrix()
    routing = load_stage12_routing()
    policies = build_stage12_policies(matrix, routing)
    payload = run_stage12(matrix, policies["ecologic"])
    print(
        f"baseline={payload['baseline_status']} "
        f"hull_flips={payload['summary']['n_hull_flips']} "
        f"claim_reversed={payload['summary']['n_hull_claim_reversed']} "
        f"bootstrap_flip_rate={payload['summary']['bootstrap_flip_rate']} "
        f"csv={payload['saved']['csv']}"
    )


if __name__ == "__main__":
    main()

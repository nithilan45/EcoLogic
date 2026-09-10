"""Break-even router overhead vs the best cost-matched static policy.

A query-dependent router with inference cost ``C_r`` and quality ``Q_r`` is
compared to the cheapest query-independent mix that matches ``Q_r`` (the dual
of the max-quality mix at ``C_r`` on an increasing hull). Router overhead ``H``
is paid only by the router.

    raw_router_savings = C_static(Q_r) − C_r     (USD / ms / tokens)
    net_savings        = raw_router_savings − H
    break_even_overhead = raw_router_savings     (max H with net ≥ 0)

If ``Q_r`` exceeds every static hull vertex, static cannot match quality and
the dollar threshold is +∞: extra billed overhead never lets a query-independent
mix catch the router on quality.

Thresholds are closed-form (hull interpolation / differences of means) and
checked by inverting the cost envelope numerically. Bootstrap CIs resample
queries, then recompute unconditional means, the hull, and the thresholds.

No prices are invented: USD/token conversion uses a caller-supplied rate or
the committed router-overhead spec. Unmeasured latency overhead is not inferred
from tokens; it is treated as 0 with that noted.
"""

from __future__ import annotations

import csv
import math
from io import StringIO
from typing import Any, Mapping, Sequence

import numpy as np

from woais_experiments.accounting.per_query_cost import (
    RouterOverhead,
    load_router_overhead,
)
from woais_experiments.frozen import ItemMatrix, write_result
from woais_experiments.latency.analyze_latency import (
    DEFAULT_LEVEL,
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    percentile,
)
from woais_experiments.routing.frontier import (
    EPS,
    Mixture,
    StaticMarket,
    interpolate_at_cost,
    interpolate_at_quality,
    upper_hull_indices,
)
from woais_experiments.routing.static_baselines import market_from_panel

SCHEMA_VERSION = "1.0"
STATUS_EPS = 1e-12
AXES = ("usd", "latency_ms", "tokens")
UNITS = {"usd": "USD/query", "latency_ms": "ms/query", "tokens": "tokens/query"}
TABLE_FIELDS = (
    "router",
    "dataset",
    "budget",
    "axis",
    "unit",
    "raw_router_savings",
    "router_overhead",
    "net_savings",
    "break_even_overhead",
    "percentage_of_break_even_used",
    "status",
    "routing_is_beneficial",
    "routing_is_neutral",
    "routing_is_dominated",
    "status_ci",
    "break_even_lo",
    "break_even_hi",
    "net_lo",
    "net_hi",
    "numeric_verified",
    "numeric_abs_err",
)

CONDITIONS = {
    "beneficial": (
        "net_savings > 0: after overhead, the router is still strictly cheaper "
        "(faster, fewer tokens) than the min-cost static mix at the same quality; "
        "or router quality is strictly above the static hull (dollar axis: +∞)."
    ),
    "neutral": (
        "|net_savings| ≤ eps: overhead exhausts the raw gap, or the assignment "
        "is itself the cost-matched static policy (zero gap, zero overhead)."
    ),
    "dominated": (
        "net_savings < 0: raw savings are already negative at H = 0, or overhead "
        "exceeds the break-even threshold."
    ),
}

STAGE12_POLICY_OVERHEAD = {
    "ecologic": "ecologic_keyword",
    "ecologic_wrapped": "ecologic_keyword",
}


def mix_expectation(
    weights: Mapping[str, float],
    names: Sequence[str],
    values: Sequence[float] | np.ndarray,
) -> float | None:
    """Expected value of a query-independent mix. None if any used mean is non-finite."""
    vals = np.asarray(values, dtype=float)
    if vals.shape != (len(names),):
        raise ValueError("values must be aligned with names")
    total = 0.0
    wsum = 0.0
    for i, name in enumerate(names):
        w = float(weights.get(name, 0.0))
        if w <= 0.0:
            continue
        v = float(vals[i])
        if not math.isfinite(v):
            return None
        total += w * v
        wsum += w
    if wsum <= EPS:
        return None
    return total


def selected_mean(panel: np.ndarray, choice: np.ndarray) -> float | None:
    """Mean of the selected column per row. Non-finite cells are dropped."""
    panel = np.asarray(panel, dtype=float)
    choice = np.asarray(choice, dtype=int)
    n = panel.shape[0]
    if n == 0 or choice.shape != (n,):
        raise ValueError("panel rows must align with choice")
    sel = panel[np.arange(n), choice]
    finite = sel[np.isfinite(sel)]
    if finite.size == 0:
        return None
    return float(finite.mean())


def col_means(panel: np.ndarray) -> np.ndarray:
    panel = np.asarray(panel, dtype=float)
    return np.nanmean(panel, axis=0)


def _as_choice(
    assignment: Mapping[Any, Any] | np.ndarray,
    names: Sequence[str],
    query_ids: Sequence[Any],
) -> np.ndarray:
    names = tuple(names)
    col = {n: i for i, n in enumerate(names)}
    if isinstance(assignment, np.ndarray) and assignment.ndim == 1:
        out = np.asarray(assignment, dtype=int)
        if out.shape[0] != len(query_ids):
            raise ValueError("assignment length must match query_ids")
        if np.any(out < 0) or np.any(out >= len(names)):
            raise ValueError("assignment indices out of range")
        return out
    out = np.empty(len(query_ids), dtype=int)
    for i, qid in enumerate(query_ids):
        raw = assignment[qid]
        if isinstance(raw, str):
            if raw not in col:
                raise KeyError(f"assignment model {raw!r} is not in {names}")
            out[i] = col[raw]
        else:
            idx = int(raw)
            if idx not in range(len(names)):
                raise ValueError(f"assignment index {idx} out of range for {names}")
            out[i] = idx
    return out


def hull_quality_span(market: StaticMarket, *, eps: float = EPS) -> tuple[float, float]:
    idx = upper_hull_indices(market, eps=eps)
    qs = [float(market.mean_quality[i]) for i in idx]
    return min(qs), max(qs)


def quality_matched_static(
    market: StaticMarket,
    router_quality: float,
    *,
    eps: float = EPS,
) -> Mixture:
    """Min-cost static mix at router quality, or the best hull vertex if above the hull."""
    mix = interpolate_at_quality(market, router_quality, eps=eps)
    if mix.feasible:
        return mix
    idx = upper_hull_indices(market, eps=eps)
    best = max(idx, key=lambda i: (market.mean_quality[i], -market.mean_cost[i], -i))
    top = market.pure(market.names[best])
    return Mixture(
        weights=dict(top.weights),
        cost=top.cost,
        quality=top.quality,
        kind="quality_above_hull_best_static",
        feasible=False,
        note="router quality exceeds every static hull vertex",
    )


def analytic_usd_threshold(
    market: StaticMarket,
    router_cost: float,
    router_quality: float,
    *,
    eps: float = EPS,
) -> dict[str, Any]:
    """Closed-form dollar break-even: ``C_static(Q_r) − C_r``, or +∞ above the hull."""
    matched = interpolate_at_quality(market, router_quality, eps=eps)
    at_cost = interpolate_at_cost(market, router_cost, mode="at_most", eps=eps)
    q_lo, q_hi = hull_quality_span(market, eps=eps)
    above = router_quality > q_hi + eps or not matched.feasible
    if above:
        raw = math.inf
        note = matched.note or "router quality exceeds every static hull vertex"
        static_cost = None
        weights = quality_matched_static(market, router_quality, eps=eps).weights
    else:
        raw = float(matched.cost) - float(router_cost)
        note = matched.note
        static_cost = float(matched.cost)
        weights = dict(matched.weights)
    quality_advantage = None
    static_q = None
    if at_cost.feasible:
        static_q = float(at_cost.quality)
        quality_advantage = float(router_quality) - static_q
    return {
        "raw_router_savings": raw,
        "break_even_overhead": raw,
        "same_quality_feasible": not above,
        "static_cost_at_same_quality": static_cost,
        "static_mix_at_same_quality": weights,
        "static_quality_at_router_cost": static_q,
        "quality_advantage_at_router_cost": quality_advantage,
        "hull_quality_lo": q_lo,
        "hull_quality_hi": q_hi,
        "note": note,
    }


def numeric_min_cost_for_quality(
    market: StaticMarket,
    target_quality: float,
    *,
    eps: float = EPS,
    n_bisect: int = 80,
) -> float:
    """Invert the equal-cost envelope by bisection (does not call interpolate_at_quality)."""
    idx = upper_hull_indices(market, eps=eps)
    c_lo = float(market.mean_cost[idx[0]])
    c_hi = float(market.mean_cost[idx[-1]])
    q_lo = float(market.mean_quality[idx[0]])
    q_hi = float(market.mean_quality[idx[-1]])
    if target_quality > q_hi + eps:
        return math.inf
    if target_quality <= q_lo + eps:
        return c_lo
    lo, hi = c_lo, c_hi
    for _ in range(int(n_bisect)):
        mid = 0.5 * (lo + hi)
        mix = interpolate_at_cost(market, mid, mode="equal", eps=eps)
        if not mix.feasible or not math.isfinite(mix.quality):
            hi = mid
            continue
        if mix.quality < target_quality - eps:
            lo = mid
        else:
            hi = mid
    return float(hi)


def numeric_usd_threshold(
    market: StaticMarket,
    router_cost: float,
    router_quality: float,
    *,
    eps: float = EPS,
    n_bisect: int = 80,
) -> float:
    c_star = numeric_min_cost_for_quality(
        market, router_quality, eps=eps, n_bisect=n_bisect
    )
    if math.isinf(c_star):
        return math.inf
    return float(c_star) - float(router_cost)


def _net(raw: float | None, overhead: float | None) -> float | None:
    if raw is None:
        return None
    h = 0.0 if overhead is None else float(overhead)
    if math.isinf(raw):
        return raw if raw > 0 else raw - h
    return float(raw) - h


def classify_status(net_savings: float | None, *, eps: float = STATUS_EPS) -> str:
    if net_savings is None:
        return "dominated"
    if math.isnan(net_savings):
        return "dominated"
    if math.isinf(net_savings):
        return "beneficial" if net_savings > 0 else "dominated"
    if net_savings > eps:
        return "beneficial"
    if net_savings < -eps:
        return "dominated"
    return "neutral"


def classify_status_ci(
    lo: float | None,
    hi: float | None,
    *,
    eps: float = STATUS_EPS,
) -> str | None:
    if lo is None or hi is None:
        return None
    if any(isinstance(v, float) and math.isnan(v) for v in (lo, hi)):
        return None
    if (math.isinf(lo) and lo > 0) or (math.isfinite(lo) and lo > eps):
        return "beneficial"
    if (math.isinf(hi) and hi < 0) or (math.isfinite(hi) and hi < -eps):
        return "dominated"
    return "neutral"


def percentage_of_break_even_used(
    overhead: float | None,
    break_even: float | None,
    *,
    eps: float = STATUS_EPS,
) -> float | None:
    """``100 * H / H*`` when ``H* > 0``. 0 when ``H* = +∞``. None if already past."""
    if break_even is None:
        return None
    h = 0.0 if overhead is None else float(overhead)
    h_star = float(break_even)
    if math.isnan(h_star) or math.isnan(h):
        return None
    if math.isinf(h_star) and h_star > 0:
        return 0.0
    if math.isinf(h_star) and h_star < 0:
        return None
    if abs(h_star) <= eps:
        if abs(h) <= eps:
            return 0.0
        return math.inf
    if h_star < 0:
        return None
    return 100.0 * h / h_star


def priced_token_break_even(
    usd_threshold: float,
    usd_per_million_tokens: float | None,
) -> float | None:
    """Token count that prices to the dollar threshold. Free tokens → +∞."""
    if usd_per_million_tokens is None:
        return None
    rate = float(usd_per_million_tokens)
    if rate < 0:
        raise ValueError("usd_per_million_tokens must be ≥ 0")
    if rate == 0.0:
        return math.inf
    if math.isinf(usd_threshold):
        return math.inf if usd_threshold > 0 else float("-inf")
    return float(usd_threshold) * 1_000_000.0 / rate


def verify_usd_threshold(
    market: StaticMarket,
    router_cost: float,
    router_quality: float,
    analytic: float,
    *,
    eps: float = EPS,
    n_bisect: int = 80,
    n_scan: int = 401,
) -> dict[str, Any]:
    """Compare hull interpolation to envelope inversion and a sign scan on net savings."""
    numeric = numeric_usd_threshold(
        market, router_cost, router_quality, eps=eps, n_bisect=n_bisect
    )
    abs_err = None
    if math.isfinite(analytic) and math.isfinite(numeric):
        abs_err = abs(float(analytic) - float(numeric))
    elif math.isinf(analytic) and math.isinf(numeric) and (analytic > 0) == (numeric > 0):
        abs_err = 0.0

    matched = interpolate_at_quality(market, router_quality, eps=eps)
    scan_ok = True
    scan_note = ""
    if matched.feasible and math.isfinite(analytic):
        c_s = float(matched.cost)
        span = max(abs(analytic), 1e-6)
        hs = np.linspace(analytic - span, analytic + span, int(n_scan))
        for h in hs:
            net = c_s - float(router_cost) - float(h)
            if h < analytic - 1e-9 and net <= 0:
                scan_ok = False
                scan_note = f"expected positive net at H={h} < H*"
                break
            if h > analytic + 1e-9 and net >= 0:
                scan_ok = False
                scan_note = f"expected negative net at H={h} > H*"
                break
        net_at = c_s - float(router_cost) - float(analytic)
        if abs(net_at) > 1e-9:
            scan_ok = False
            scan_note = f"net at analytic H* is {net_at}"
    elif math.isinf(analytic) and analytic > 0:
        scan_note = "quality above hull; every finite H remains beneficial on quality"
    else:
        scan_note = "static quality infeasible or non-finite analytic threshold"

    verified = abs_err is not None and abs_err <= 1e-8 and scan_ok
    return {
        "analytic": analytic,
        "numeric": numeric,
        "abs_err": abs_err,
        "scan_ok": scan_ok,
        "scan_note": scan_note,
        "verified": verified,
        "method": "bisection on interpolate_at_cost(equal) vs interpolate_at_quality",
    }


def verify_mean_gap(
    raw: float | None,
    router_mean: float | None,
    static_mean: float | None,
    *,
    n_scan: int = 21,
) -> dict[str, Any]:
    """Latency/token gap is a difference of means; confirm the sign flip at H*."""
    if raw is None or router_mean is None or static_mean is None:
        return {
            "analytic": raw,
            "numeric": None,
            "abs_err": None,
            "scan_ok": False,
            "scan_note": "missing means",
            "verified": False,
            "method": "static_mean - router_mean",
        }
    numeric = float(static_mean) - float(router_mean)
    abs_err = abs(float(raw) - numeric)
    scan_ok = True
    span = max(abs(numeric), 1e-6)
    for h in np.linspace(numeric - span, numeric + span, int(n_scan)):
        net = numeric - float(h)
        if h < numeric - 1e-9 and net <= 0:
            scan_ok = False
            break
        if h > numeric + 1e-9 and net >= 0:
            scan_ok = False
            break
    return {
        "analytic": raw,
        "numeric": numeric,
        "abs_err": abs_err,
        "scan_ok": scan_ok,
        "scan_note": "" if scan_ok else "sign scan failed",
        "verified": abs_err <= 1e-12 and scan_ok,
        "method": "static_mean - router_mean with sign scan",
    }


def _ci_payload(
    samples: Sequence[float],
    point: float | None,
    *,
    n_boot: int,
    seed: int,
    level: float,
) -> dict[str, Any]:
    arr = np.asarray(list(samples), dtype=float)
    n_finite = int(np.isfinite(arr).sum()) if arr.size else 0
    n_pos_inf = int(np.isposinf(arr).sum()) if arr.size else 0
    n_neg_inf = int(np.isneginf(arr).sum()) if arr.size else 0
    payload = {
        "point": point,
        "lo": point,
        "hi": point,
        "n_boot": int(n_boot),
        "n_finite": n_finite,
        "n_plus_inf": n_pos_inf,
        "n_minus_inf": n_neg_inf,
        "level": float(level),
        "seed": int(seed),
    }
    if arr.size == 0 or n_boot <= 0:
        return payload
    if n_pos_inf == arr.size:
        payload["lo"] = math.inf
        payload["hi"] = math.inf
        return payload
    if n_neg_inf == arr.size:
        payload["lo"] = float("-inf")
        payload["hi"] = float("-inf")
        return payload
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return payload
    lo_q = 100.0 * (1.0 - level) / 2.0
    hi_q = 100.0 * (1.0 + level) / 2.0
    lo = percentile(finite, lo_q)
    hi = percentile(finite, hi_q)
    payload["lo"] = float("-inf") if n_neg_inf else lo
    payload["hi"] = math.inf if n_pos_inf else hi
    return payload


def _number_kind(value: float | None) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float) and math.isnan(value):
        return "missing"
    if isinstance(value, float) and math.isinf(value):
        return "plus_infinity" if value > 0 else "minus_infinity"
    return "finite"


def _axis_row(
    *,
    router: str,
    dataset: str,
    budget: float,
    axis: str,
    raw: float | None,
    overhead: float | None,
    verify: Mapping[str, Any] | None,
    extra: Mapping[str, Any] | None = None,
    eps: float = STATUS_EPS,
) -> dict[str, Any]:
    net = _net(raw, overhead)
    status = None if raw is None else classify_status(net, eps=eps)
    row = {
        "router": router,
        "dataset": dataset,
        "budget": float(budget),
        "axis": axis,
        "unit": UNITS[axis],
        "raw_router_savings": raw,
        "router_overhead": overhead,
        "net_savings": net,
        "break_even_overhead": raw,
        "percentage_of_break_even_used": percentage_of_break_even_used(
            overhead, raw, eps=eps
        ),
        "raw_router_savings_kind": _number_kind(raw),
        "break_even_kind": _number_kind(raw),
        "status": status,
        "routing_is_beneficial": status == "beneficial",
        "routing_is_neutral": status == "neutral",
        "routing_is_dominated": status == "dominated",
        "numeric_verified": None if verify is None else bool(verify.get("verified")),
        "numeric_abs_err": None if verify is None else verify.get("abs_err"),
        "numeric": None if verify is None else dict(verify),
    }
    if extra:
        row.update(extra)
    return row


def _point_metrics(
    names: Sequence[str],
    cost: np.ndarray,
    quality: np.ndarray,
    choice: np.ndarray,
    *,
    latency: np.ndarray | None,
    tokens: np.ndarray | None,
    overhead_usd: float,
    overhead_latency_ms: float | None,
    overhead_tokens: float,
    token_usd_per_million: float | None,
    router: str,
    dataset: str,
    budget: float | None,
    eps: float,
    verify: bool,
) -> dict[str, Any]:
    names = tuple(names)
    cost = np.asarray(cost, dtype=float)
    quality = np.asarray(quality, dtype=float)
    choice = np.asarray(choice, dtype=int)
    n, m = cost.shape
    if quality.shape != (n, m) or choice.shape != (n,) or m != len(names):
        raise ValueError("cost, quality, choice, names shape mismatch")
    market = market_from_panel(names, cost, quality)
    q_r = selected_mean(quality, choice)
    c_r = selected_mean(cost, choice)
    if q_r is None or c_r is None:
        raise ValueError("router quality/cost mean is undefined")
    usd = analytic_usd_threshold(market, c_r, q_r, eps=eps)
    op_budget = float(c_r) if budget is None else float(budget)
    mix = quality_matched_static(market, q_r, eps=eps)

    usd_verify = None
    if verify:
        usd_verify = verify_usd_threshold(
            market, c_r, q_r, float(usd["break_even_overhead"]), eps=eps
        )

    lat_panel = None if latency is None else np.asarray(latency, dtype=float)
    tok_panel = None if tokens is None else np.asarray(tokens, dtype=float)
    lat_means = None if lat_panel is None else col_means(lat_panel)
    tok_means = None if tok_panel is None else col_means(tok_panel)
    l_r = None if lat_panel is None else selected_mean(lat_panel, choice)
    t_r = None if tok_panel is None else selected_mean(tok_panel, choice)
    l_s = None if lat_means is None else mix_expectation(mix.weights, names, lat_means)
    t_s = None if tok_means is None else mix_expectation(mix.weights, names, tok_means)
    lat_raw = None if l_r is None or l_s is None else float(l_s) - float(l_r)
    tok_raw = None if t_r is None or t_s is None else float(t_s) - float(t_r)

    lat_verify = verify_mean_gap(lat_raw, l_r, l_s) if verify else None
    tok_verify = verify_mean_gap(tok_raw, t_r, t_s) if verify else None

    priced = priced_token_break_even(float(usd["break_even_overhead"]), token_usd_per_million)

    extra_usd = {
        "router_quality": q_r,
        "router_realized_cost": c_r,
        "same_quality_feasible": usd["same_quality_feasible"],
        "static_cost_at_same_quality": usd["static_cost_at_same_quality"],
        "static_mix_at_same_quality": usd["static_mix_at_same_quality"],
        "quality_advantage_at_router_cost": usd["quality_advantage_at_router_cost"],
        "static_quality_at_router_cost": usd["static_quality_at_router_cost"],
        "hull_quality_lo": usd["hull_quality_lo"],
        "hull_quality_hi": usd["hull_quality_hi"],
        "note": usd["note"],
    }
    extra_lat = {
        "router_mean": l_r,
        "static_mean": l_s,
        "static_mix_at_same_quality": mix.weights,
        "overhead_unmeasured": overhead_latency_ms is None,
        "note": (
            "router latency overhead unmeasured; not inferred from tokens; "
            "treated as 0 for net savings"
            if overhead_latency_ms is None
            else mix.note
        ),
    }
    extra_tok = {
        "router_mean": t_r,
        "static_mean": t_s,
        "static_mix_at_same_quality": mix.weights,
        "priced_token_break_even": priced,
        "token_usd_per_million": token_usd_per_million,
        "note": mix.note,
    }

    lat_oh = 0.0 if overhead_latency_ms is None else float(overhead_latency_ms)
    rows = [
        _axis_row(
            router=router, dataset=dataset, budget=op_budget, axis="usd",
            raw=usd["raw_router_savings"], overhead=float(overhead_usd),
            verify=usd_verify, extra=extra_usd, eps=eps,
        ),
        _axis_row(
            router=router, dataset=dataset, budget=op_budget, axis="latency_ms",
            raw=lat_raw, overhead=lat_oh, verify=lat_verify, extra=extra_lat, eps=eps,
        ),
        _axis_row(
            router=router, dataset=dataset, budget=op_budget, axis="tokens",
            raw=tok_raw, overhead=float(overhead_tokens), verify=tok_verify,
            extra=extra_tok, eps=eps,
        ),
    ]
    return {
        "router": router,
        "dataset": dataset,
        "budget": op_budget,
        "n": n,
        "router_quality": q_r,
        "router_realized_cost": c_r,
        "rows": rows,
        "usd": rows[0],
        "latency_ms": rows[1],
        "tokens": rows[2],
    }


def analyze_assignment(
    names: Sequence[str],
    cost: np.ndarray,
    quality: np.ndarray,
    assignment: Mapping[Any, Any] | np.ndarray,
    *,
    query_ids: Sequence[Any] | None = None,
    latency: np.ndarray | None = None,
    tokens: np.ndarray | None = None,
    overhead_usd: float = 0.0,
    overhead_latency_ms: float | None = None,
    overhead_tokens: float = 0.0,
    token_usd_per_million: float | None = None,
    router: str = "router",
    dataset: str = "panel",
    budget: float | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
    eps: float = EPS,
    verify: bool = True,
) -> dict[str, Any]:
    """Break-even on USD, latency, and tokens for one assignment.

    ``assignment`` is model names, column indices, or a length-n integer array.
    """
    names = tuple(names)
    cost = np.asarray(cost, dtype=float)
    quality = np.asarray(quality, dtype=float)
    if cost.shape != quality.shape or cost.ndim != 2:
        raise ValueError("cost and quality must be 2-D arrays of the same shape")
    n = cost.shape[0]
    ids = list(range(n)) if query_ids is None else list(query_ids)
    if len(ids) != n:
        raise ValueError("query_ids length must match panel rows")
    choice = _as_choice(assignment, names, ids)
    point = _point_metrics(
        names, cost, quality, choice,
        latency=latency, tokens=tokens,
        overhead_usd=overhead_usd,
        overhead_latency_ms=overhead_latency_ms,
        overhead_tokens=overhead_tokens,
        token_usd_per_million=token_usd_per_million,
        router=router, dataset=dataset, budget=budget, eps=eps, verify=verify,
    )

    boot_usd: list[float] = []
    boot_net_usd: list[float] = []
    boot_lat: list[float] = []
    boot_net_lat: list[float] = []
    boot_tok: list[float] = []
    boot_net_tok: list[float] = []
    if n_boot > 0 and n > 1:
        rng = np.random.default_rng(seed)
        lat_oh = 0.0 if overhead_latency_ms is None else float(overhead_latency_ms)
        for _ in range(int(n_boot)):
            idx = rng.choice(n, size=n, replace=True)
            lat_s = None if latency is None else np.asarray(latency, dtype=float)[idx]
            tok_s = None if tokens is None else np.asarray(tokens, dtype=float)[idx]
            sample = _point_metrics(
                names, cost[idx], quality[idx], choice[idx],
                latency=lat_s, tokens=tok_s,
                overhead_usd=overhead_usd,
                overhead_latency_ms=overhead_latency_ms,
                overhead_tokens=overhead_tokens,
                token_usd_per_million=token_usd_per_million,
                router=router, dataset=dataset, budget=budget, eps=eps, verify=False,
            )
            boot_usd.append(float(sample["usd"]["break_even_overhead"]))
            boot_net_usd.append(float(sample["usd"]["net_savings"]))
            if sample["latency_ms"]["break_even_overhead"] is not None:
                boot_lat.append(float(sample["latency_ms"]["break_even_overhead"]))
                boot_net_lat.append(float(_net(sample["latency_ms"]["raw_router_savings"], lat_oh)))
            if sample["tokens"]["break_even_overhead"] is not None:
                boot_tok.append(float(sample["tokens"]["break_even_overhead"]))
                boot_net_tok.append(float(_net(sample["tokens"]["raw_router_savings"], overhead_tokens)))

    def _attach(row: dict[str, Any], be_s: list[float], net_s: list[float]) -> None:
        be_ci = _ci_payload(be_s, row["break_even_overhead"], n_boot=n_boot, seed=seed, level=level)
        net_ci = _ci_payload(net_s, row["net_savings"], n_boot=n_boot, seed=seed, level=level)
        raw_ci = _ci_payload(be_s, row["raw_router_savings"], n_boot=n_boot, seed=seed, level=level)
        row["ci"] = {
            "break_even_overhead": be_ci,
            "net_savings": net_ci,
            "raw_router_savings": raw_ci,
        }
        row["break_even_lo"] = be_ci["lo"]
        row["break_even_hi"] = be_ci["hi"]
        row["net_lo"] = net_ci["lo"]
        row["net_hi"] = net_ci["hi"]
        row["status_ci"] = classify_status_ci(net_ci["lo"], net_ci["hi"])
        row["routing_is_beneficial_ci"] = row["status_ci"] == "beneficial"
        row["routing_is_neutral_ci"] = row["status_ci"] == "neutral"
        row["routing_is_dominated_ci"] = row["status_ci"] == "dominated"

    _attach(point["usd"], boot_usd, boot_net_usd)
    _attach(point["latency_ms"], boot_lat, boot_net_lat)
    _attach(point["tokens"], boot_tok, boot_net_tok)
    point["n_boot"] = int(n_boot)
    point["seed"] = int(seed)
    point["level"] = float(level)
    point["conditions"] = dict(CONDITIONS)
    return point


def panels_from_item_matrix(
    matrix: ItemMatrix,
    *,
    tiers: tuple[int, ...] = (1, 2, 3),
) -> dict[str, Any]:
    names = tuple(matrix.model_of[t] for t in tiers)
    n = matrix.n
    m = len(tiers)
    cost = np.zeros((n, m), dtype=float)
    quality = np.zeros((n, m), dtype=float)
    latency = np.zeros((n, m), dtype=float)
    tokens = np.zeros((n, m), dtype=float)
    for i, qid in enumerate(matrix.item_ids):
        for k, t in enumerate(tiers):
            cost[i, k] = float(matrix.usd[(t, qid)])
            quality[i, k] = 1.0 if matrix.correct[(t, qid)] else 0.0
            ls = matrix.latency_s.get((t, qid))
            latency[i, k] = float("nan") if ls is None else float(ls) * 1000.0
            tokens[i, k] = float(matrix.tokens[(t, qid)])
    return {
        "names": names,
        "query_ids": list(matrix.item_ids),
        "cost": cost,
        "quality": quality,
        "latency": latency,
        "tokens": tokens,
        "tiers": tiers,
    }


def choice_from_tiers(
    assignment: Mapping[Any, int],
    query_ids: Sequence[Any],
    *,
    tiers: tuple[int, ...] = (1, 2, 3),
) -> np.ndarray:
    col = {int(t): k for k, t in enumerate(tiers)}
    out = np.empty(len(query_ids), dtype=int)
    for i, qid in enumerate(query_ids):
        t = int(assignment[qid])
        if t not in col:
            raise KeyError(f"tier {t} is not in {tiers}")
        out[i] = col[t]
    return out


def analyze_item_matrix(
    matrix: ItemMatrix,
    assignment_tiers: Mapping[Any, int],
    *,
    router: str,
    dataset: str = "stage12",
    overhead: RouterOverhead | None = None,
    overhead_latency_ms: float | None = None,
    token_usd_per_million: float | None = None,
    budget: float | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
    verify: bool = True,
    tiers: tuple[int, ...] = (1, 2, 3),
) -> dict[str, Any]:
    panels = panels_from_item_matrix(matrix, tiers=tiers)
    choice = choice_from_tiers(assignment_tiers, panels["query_ids"], tiers=tiers)
    oh = overhead if overhead is not None else load_router_overhead("none")
    return analyze_assignment(
        panels["names"],
        panels["cost"],
        panels["quality"],
        choice,
        query_ids=panels["query_ids"],
        latency=panels["latency"],
        tokens=panels["tokens"],
        overhead_usd=oh.cost_usd,
        overhead_latency_ms=overhead_latency_ms,
        overhead_tokens=oh.tokens,
        token_usd_per_million=token_usd_per_million,
        router=router,
        dataset=dataset,
        budget=budget,
        n_boot=n_boot,
        seed=seed,
        level=level,
        verify=verify,
    )


def table_from_policies(
    matrix: ItemMatrix,
    policies: Mapping[str, Mapping[Any, int]],
    *,
    dataset: str = "stage12",
    overhead_by_router: Mapping[str, RouterOverhead | str] | None = None,
    overhead_latency_ms: float | None = None,
    token_usd_per_million: float | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
    verify: bool = True,
    skip: Sequence[str] = ("ecologic_wrapped", "oracle_energy"),
) -> dict[str, Any]:
    """One break-even block per named assignment on a shared panel."""
    skip_set = set(skip)
    blocks = []
    rows = []
    for name, assign in policies.items():
        if name in skip_set:
            continue
        spec = None
        if overhead_by_router and name in overhead_by_router:
            spec = overhead_by_router[name]
        elif name in STAGE12_POLICY_OVERHEAD:
            spec = STAGE12_POLICY_OVERHEAD[name]
        if isinstance(spec, RouterOverhead):
            oh = spec
        elif isinstance(spec, str):
            oh = load_router_overhead(spec)
        else:
            oh = load_router_overhead("none")
        block = analyze_item_matrix(
            matrix,
            assign,
            router=name,
            dataset=dataset,
            overhead=oh,
            overhead_latency_ms=overhead_latency_ms,
            token_usd_per_million=token_usd_per_million,
            n_boot=n_boot,
            seed=seed,
            level=level,
            verify=verify,
        )
        blocks.append(block)
        rows.extend(block["rows"])
    counts = {axis: {"beneficial": 0, "neutral": 0, "dominated": 0} for axis in AXES}
    for row in rows:
        if row["status"] in counts[row["axis"]]:
            counts[row["axis"]][row["status"]] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "n_queries": matrix.n,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "level": float(level),
        "conditions": dict(CONDITIONS),
        "status_counts": counts,
        "routers": [b["router"] for b in blocks],
        "blocks": blocks,
        "rows": rows,
    }


def _fmt_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    return value


def rows_to_csv(rows: Sequence[Mapping[str, Any]], fields: Sequence[str] = TABLE_FIELDS) -> str:
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
        return None
    return obj


def save_tables(
    payload: Mapping[str, Any],
    *,
    relpath: str = "accounting/breakeven/stage12",
) -> dict[str, str]:
    json_path = write_result(f"{relpath}.json", _jsonable(dict(payload)))
    csv_path = write_result(f"{relpath}.csv", rows_to_csv(payload["rows"]))
    written = {"json": str(json_path), "csv": str(csv_path)}
    for axis in AXES:
        axis_rows = [r for r in payload["rows"] if r["axis"] == axis]
        p = write_result(f"{relpath}_{axis}.csv", rows_to_csv(axis_rows))
        written[axis] = str(p)
    return written


def run_stage12(
    matrix: ItemMatrix,
    policies: Mapping[str, Mapping[Any, int]],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
    relpath: str = "accounting/breakeven/stage12",
) -> dict[str, Any]:
    payload = table_from_policies(
        matrix, policies, dataset="stage12", n_boot=n_boot, seed=seed, level=level
    )
    payload["saved"] = save_tables(payload, relpath=relpath)
    payload["overhead_notes"] = {
        "usd": "ecologic_keyword cost_usd from configs/models.json (0.0); other policies use none.",
        "tokens": "ecologic_keyword tokens from config (0); unpriced local NLP.",
        "latency_ms": (
            "router_decision_ms was not recorded in frozen EcoLogic logs; "
            "latency overhead is treated as 0 and is not inferred from tokens."
        ),
    }
    return payload

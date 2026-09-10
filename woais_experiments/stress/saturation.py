"""Saturation detector for a *tested* environment. Not a universal capacity limit."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from woais_experiments.latency.analyze_latency import percentile
from woais_experiments.stress.failure_analysis import summarize_failures
from woais_experiments.stress.records import occupancy_at_arrivals

DEFAULT_SLA_P95_S = 2.0
DEFAULT_FAILURE_RATE = 0.05
DEFAULT_QUEUE_SLOPE = 0.05
DEFAULT_QUEUE_R2 = 0.30


def occupancy_growth(
    arrivals_s: Sequence[float],
    occupancy: Sequence[int],
    *,
    min_slope: float = DEFAULT_QUEUE_SLOPE,
    min_r2: float = DEFAULT_QUEUE_R2,
) -> dict[str, Any]:
    x = np.asarray(list(arrivals_s), dtype=float)
    y = np.asarray(list(occupancy), dtype=float)
    if x.size < 4 or y.size != x.size:
        return {
            "slope_per_s": None,
            "r2": None,
            "persistent_growth": False,
            "n": int(x.size),
            "note": "need at least 4 arrivals to estimate queue growth",
        }
    if np.allclose(x, x[0]):
        return {
            "slope_per_s": None,
            "r2": None,
            "persistent_growth": False,
            "n": int(x.size),
            "note": "zero duration; cannot estimate slope",
        }
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(y.mean())) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 1e-18 else 0.0
    persistent = bool(slope > float(min_slope) and r2 >= float(min_r2) and y[-1] > y[0] + 1e-9)
    return {
        "slope_per_s": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "persistent_growth": persistent,
        "n": int(x.size),
        "occupancy_first": float(y[0]),
        "occupancy_last": float(y[-1]),
        "occupancy_mean": float(y.mean()),
        "occupancy_max": float(y.max()),
        "thresholds": {"min_slope": float(min_slope), "min_r2": float(min_r2)},
    }


def detect_saturation(
    rows: Sequence[Mapping[str, Any]],
    *,
    sla_p95_s: float = DEFAULT_SLA_P95_S,
    failure_rate_threshold: float = DEFAULT_FAILURE_RATE,
    min_slope: float = DEFAULT_QUEUE_SLOPE,
    min_r2: float = DEFAULT_QUEUE_R2,
    p95_latency_s: float | None = None,
) -> dict[str, Any]:
    """Flag saturation for this log only. Do not generalize beyond the test setup."""
    fails = summarize_failures(rows)
    e2e = []
    arrivals = []
    completes = []
    for row in rows:
        arrivals.append(float(row.get("scheduled_arrival_s") or 0.0))
        e2e_s = _e2e_s(row)
        if e2e_s is not None:
            e2e.append(e2e_s)
            completes.append(float(row["scheduled_arrival_s"]) + e2e_s)
        else:
            completes.append(row.get("complete_s"))
    p95 = p95_latency_s if p95_latency_s is not None else percentile(e2e, 95)
    occ = occupancy_at_arrivals(arrivals, completes) if arrivals else []
    growth = occupancy_growth(arrivals, occ, min_slope=min_slope, min_r2=min_r2)
    p95_breach = p95 is not None and float(p95) > float(sla_p95_s)
    fail_breach = fails["failure_rate"] is not None and float(fails["failure_rate"]) > float(
        failure_rate_threshold
    )
    queue_breach = bool(growth.get("persistent_growth"))
    saturated = bool(p95_breach or fail_breach or queue_breach)
    reasons = []
    if p95_breach:
        reasons.append("p95_exceeds_sla")
    if fail_breach:
        reasons.append("failure_rate_exceeds_threshold")
    if queue_breach:
        reasons.append("queue_grows_persistently")
    return {
        "saturated": saturated,
        "reasons": reasons,
        "p95_latency_s": p95,
        "sla_p95_s": float(sla_p95_s),
        "p95_exceeds_sla": p95_breach,
        "failure_rate": fails["failure_rate"],
        "failure_rate_threshold": float(failure_rate_threshold),
        "failure_rate_exceeds_threshold": fail_breach,
        "queue_growth": growth,
        "environment_only": True,
        "note": (
            "Saturation is reported only for this tested environment, load profile, "
            "policy, and backend. It is not a universal EcoLogic capacity limit."
        ),
    }


def saturation_point(
    cells: Sequence[Mapping[str, Any]],
    *,
    profile: str,
    policy: str,
) -> dict[str, Any]:
    """Smallest tested multiplier that is saturated, for one profile×policy."""
    matched = [
        c
        for c in cells
        if str(c.get("profile")) == profile and str(c.get("policy")) == policy
    ]
    matched = sorted(matched, key=lambda c: float(c.get("multiplier") or 0.0))
    first = None
    for cell in matched:
        sat = cell.get("saturation") or {}
        if sat.get("saturated"):
            first = cell
            break
    return {
        "profile": profile,
        "policy": policy,
        "saturation_multiplier": None if first is None else float(first["multiplier"]),
        "n_levels_tested": len(matched),
        "unsaturated_multipliers": [
            float(c["multiplier"])
            for c in matched
            if not (c.get("saturation") or {}).get("saturated")
        ],
        "environment_only": True,
        "note": (
            "No saturation among tested levels"
            if first is None
            else f"first saturated level in this environment: {first['multiplier']}x"
        ),
    }


def max_sustainable_throughput(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Highest observed throughput among unsaturated cells in this environment."""
    uns = [
        c
        for c in cells
        if not (c.get("saturation") or {}).get("saturated")
        and c.get("throughput_per_s") is not None
    ]
    if not uns:
        return {
            "max_sustainable_throughput_per_s": None,
            "cell": None,
            "environment_only": True,
            "note": "every tested cell was saturated, or throughput was unobserved",
        }
    best = max(uns, key=lambda c: float(c["throughput_per_s"]))
    return {
        "max_sustainable_throughput_per_s": float(best["throughput_per_s"]),
        "profile": best.get("profile"),
        "policy": best.get("policy"),
        "multiplier": best.get("multiplier"),
        "environment_only": True,
        "note": "Maximum among unsaturated cells in this run; not a universal limit.",
    }


def _e2e_s(row: Mapping[str, Any]) -> float | None:
    for key in ("end_to_end_s", "sojourn_s"):
        v = row.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    v = row.get("end_to_end_ms")
    if v is None:
        return None
    try:
        return float(v) / 1000.0
    except (TypeError, ValueError):
        return None

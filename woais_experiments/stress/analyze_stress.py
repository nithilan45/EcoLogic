"""Aggregate stress-test logs. MEASURED and SIMULATED files are never mixed."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from woais_experiments.frozen import to_jsonable, write_result
from woais_experiments.latency.analyze_latency import percentile
from woais_experiments.routing.frontier import StaticMarket, pairwise_pareto_filter
from woais_experiments.runner.gitinfo import git_snapshot
from woais_experiments.stress import MEASURED, SIMULATED
from woais_experiments.stress.failure_analysis import classify_row, summarize_failures
from woais_experiments.stress.records import (
    MixedSourceError,
    assert_homogeneous,
    load_jsonl,
    occupancy_at_arrivals,
    results_prefix,
)
from woais_experiments.stress.saturation import (
    detect_saturation,
    max_sustainable_throughput,
    occupancy_growth,
    saturation_point,
)

POLICY_ORDER = ("always_cheap", "always_strong", "ecologic", "cost_matched_static")


def _finite(xs: Sequence[Any]) -> list[float]:
    out: list[float] = []
    for x in xs:
        if x is None:
            continue
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if v == v and abs(v) != float("inf"):
            out.append(v)
    return out


def _mean(xs: Sequence[Any]) -> float | None:
    vals = _finite(xs)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _e2e_s(row: Mapping[str, Any]) -> float | None:
    for key in ("end_to_end_s", "sojourn_s"):
        if row.get(key) is not None:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                pass
    if row.get("end_to_end_ms") is not None:
        try:
            return float(row["end_to_end_ms"]) / 1000.0
        except (TypeError, ValueError):
            return None
    return None


def _cost(row: Mapping[str, Any]) -> float | None:
    for key in ("realized_provider_cost", "cost_usd", "inference_cost", "total_cost_usd"):
        if row.get(key) is not None:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return None


def _quality(row: Mapping[str, Any]) -> float | None:
    v = row.get("quality")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _strong(row: Mapping[str, Any], strong_model: str | None) -> bool | None:
    model = row.get("selected_model") or row.get("model")
    if model is None or strong_model is None:
        tier = row.get("selected_tier")
        if tier is None:
            return None
        try:
            return int(tier) == 3
        except (TypeError, ValueError):
            return None
    return str(model) == str(strong_model)


def cell_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    measurement_type: str,
    sla: Mapping[str, Any],
    capacity: int,
    strong_model: str | None,
    profile: str,
    multiplier: float,
    policy: str,
) -> dict[str, Any]:
    assert_homogeneous(rows, measurement_type)
    issued = [
        r
        for r in rows
        if not (r.get("error_type") == "cost_cap_stop" and r.get("dispatch_s") is None and r.get("complete_s") is None)
    ]
    arrivals = [float(r.get("scheduled_arrival_s") or 0.0) for r in issued]
    e2e = [_e2e_s(r) for r in issued]
    completes: list[float | None] = []
    for r, lat in zip(issued, e2e):
        if r.get("complete_s") is not None:
            try:
                completes.append(float(r["complete_s"]))
                continue
            except (TypeError, ValueError):
                pass
        if lat is None:
            completes.append(None)
        else:
            completes.append(float(r.get("scheduled_arrival_s") or 0.0) + lat)
    occ = occupancy_at_arrivals(arrivals, completes) if arrivals else []
    completed = [
        r
        for r in issued
        if r.get("complete_s") is not None or _e2e_s(r) is not None
    ]
    fails = summarize_failures(rows)
    e2e_all = _finite([_e2e_s(r) for r in completed])
    e2e_ok = _finite([
        _e2e_s(r) for r in completed if classify_row(r) == "ok"
    ])
    qwait = _finite([r.get("queue_wait_s") for r in rows])
    svc = _finite([r.get("service_time_s") for r in rows])
    rdec = _finite([
        r.get("router_decision_s")
        if r.get("router_decision_s") is not None
        else (None if r.get("router_decision_ms") is None else float(r["router_decision_ms"]) / 1000.0)
        for r in rows
    ])
    costs = _finite([_cost(r) for r in rows])
    n_costed = len(costs)
    n_missing_cost = len(rows) - n_costed
    total_cost = float(sum(costs)) if costs else 0.0
    quals = _finite([_quality(r) for r in rows])
    strong_flags = [_strong(r, strong_model) for r in rows]
    strong_known = [1.0 if x else 0.0 for x in strong_flags if x is not None]
    t0 = min(arrivals) if arrivals else 0.0
    t_end_candidates = [c for c in completes if c is not None]
    t_end = max(t_end_candidates) if t_end_candidates else t0
    horizon = max(0.0, t_end - t0)
    n_completed = len(completed)
    offered = None
    if len(arrivals) > 1:
        span = max(arrivals) - min(arrivals)
        offered = ((len(arrivals) - 1) / span) if span > 0 else None
    throughput = (n_completed / horizon) if horizon > 0 else None
    busy = float(sum(svc)) if svc else 0.0
    util = None
    if horizon > 0 and capacity > 0:
        util = min(1.0, busy / (horizon * int(capacity)))
    mean_occ = float(sum(occ) / len(occ)) if occ else None
    if util is None and mean_occ is not None and capacity > 0:
        util = min(1.0, mean_occ / float(capacity))
    growth = occupancy_growth(
        arrivals,
        occ,
        min_slope=float(sla.get("queue_growth_min_slope", 0.05)),
        min_r2=float(sla.get("queue_growth_min_r2", 0.30)),
    )
    p50 = percentile(e2e_ok, 50)
    p95 = percentile(e2e_ok, 95)
    p99 = percentile(e2e_ok, 99)
    p95_all = percentile(e2e_all, 95)
    sat = detect_saturation(
        rows,
        sla_p95_s=float(sla.get("p95_latency_s", 2.0)),
        failure_rate_threshold=float(sla.get("failure_rate", 0.05)),
        min_slope=float(sla.get("queue_growth_min_slope", 0.05)),
        min_r2=float(sla.get("queue_growth_min_r2", 0.30)),
        p95_latency_s=p95,
    )
    return {
        "measurement_type": measurement_type,
        "profile": profile,
        "multiplier": float(multiplier),
        "policy": policy,
        "n_scheduled": len(rows),
        "n_completed": n_completed,
        "arrival_rate_per_s": offered,
        "throughput_per_s": throughput,
        "horizon_s": horizon,
        "queue_depth_mean": mean_occ,
        "queue_depth_max": (max(occ) if occ else None),
        "queue_wait_s_mean": _mean(qwait),
        "queue_wait_s_p95": percentile(qwait, 95),
        "service_time_s_mean": _mean(svc),
        "router_decision_s_mean": _mean(rdec),
        "end_to_end_s_mean": _mean(e2e_ok),
        "p50_latency_s": p50,
        "p95_latency_s": p95,
        "p99_latency_s": p99,
        "latency_population": "successful_requests",
        "p95_latency_s_including_failures": p95_all,
        "n_latency_success": len(e2e_ok),
        "n_latency_including_failures": len(e2e_all),
        "failure_rate": fails["failure_rate"],
        "timeout_rate": fails["timeout_rate"],
        "retry_rate": fails["retry_rate"],
        "cost_per_query": (total_cost / n_costed) if n_costed else None,
        "cost_per_query_denominator": "n_requests_with_finite_cost",
        "cost_per_scheduled_missing_as_zero": (total_cost / len(rows)) if rows else None,
        "n_costed": n_costed,
        "n_missing_cost": n_missing_cost,
        "cost_per_successful_query": (total_cost / len(e2e_ok)) if e2e_ok else None,
        "total_cost": total_cost,
        "quality_mean": _mean(quals),
        "strong_model_fraction": _mean(strong_known),
        "utilization": util,
        "capacity": int(capacity),
        "failures": fails,
        "queue_growth": growth,
        "saturation": sat,
        "environment_only": True,
    }


def _inflation(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline == 0:
        return None
    return float(value) / float(baseline)


def attach_inflation(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[float, dict[str, Any]]] = defaultdict(dict)
    for cell in cells:
        by_key[(cell["profile"], cell["policy"])][float(cell["multiplier"])] = cell
    for (_profile, _policy), levels in by_key.items():
        base = levels.get(1.0) or levels.get(1)
        if base is None:
            for cell in levels.values():
                cell["latency_inflation_vs_1x"] = None
                cell["cost_inflation_vs_1x"] = None
            continue
        for cell in levels.values():
            cell["latency_inflation_vs_1x"] = _inflation(cell.get("p95_latency_s"), base.get("p95_latency_s"))
            cell["cost_inflation_vs_1x"] = _inflation(cell.get("cost_per_query"), base.get("cost_per_query"))
    return cells


def pareto_under_load(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Whether EcoLogic is undominated at each (profile, multiplier).

    Quality axis: mean quality when observed, otherwise −p95 latency (lower
    latency is treated as higher quality). Cost is minimized. This is for the
    tested environment only.
    """
    groups: dict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells:
        groups[(str(cell["profile"]), float(cell["multiplier"]))].append(cell)
    reports = []
    for (profile, mult), group in sorted(groups.items()):
        by_name = {str(c["policy"]): c for c in group}
        names = [p for p in POLICY_ORDER if p in by_name]
        if len(names) < 2:
            reports.append({
                "profile": profile,
                "multiplier": mult,
                "ecologic_pareto_efficient": None,
                "note": "need at least two policies to compare",
            })
            continue
        costs = []
        quals = []
        used_latency_proxy = False
        missing_cost = False
        for name in names:
            c = by_name[name]
            raw_cost = c.get("cost_per_query")
            if raw_cost is None:
                missing_cost = True
                break
            try:
                costs.append(float(raw_cost))
            except (TypeError, ValueError):
                missing_cost = True
                break
            q = c.get("quality_mean")
            if q is None:
                p95 = c.get("p95_latency_s")
                quals.append(0.0 if p95 is None else -float(p95))
                used_latency_proxy = True
            else:
                quals.append(float(q))
        if missing_cost:
            reports.append({
                "profile": profile,
                "multiplier": mult,
                "policies": names,
                "undominated": [],
                "ecologic_pareto_efficient": None,
                "quality_axis": None,
                "cost_axis": "cost_per_query",
                "environment_only": True,
                "note": (
                    "Refusing Pareto ranking: at least one policy has missing "
                    "cost_per_query. Missing cost is not treated as $0."
                ),
            })
            continue
        market = StaticMarket(names, costs, quals)
        keep = pairwise_pareto_filter(market)
        undominated = [names[i] for i in keep]
        eco = "ecologic" in undominated
        reports.append({
            "profile": profile,
            "multiplier": mult,
            "policies": names,
            "undominated": undominated,
            "ecologic_pareto_efficient": eco,
            "quality_axis": "negative_p95_latency" if used_latency_proxy else "mean_quality",
            "cost_axis": "cost_per_query",
            "points": [
                {
                    "policy": n,
                    "cost_per_query": by_name[n].get("cost_per_query"),
                    "quality_mean": by_name[n].get("quality_mean"),
                    "p95_latency_s": by_name[n].get("p95_latency_s"),
                }
                for n in names
            ],
            "environment_only": True,
            "note": (
                "EcoLogic is Pareto-efficient under this load if it is not strictly "
                "dominated in (cost, quality). Quality is mean correctness when "
                "logged, otherwise −p95 latency. Not a universal claim."
            ),
        })
    return reports


def analyze_cells(
    cells: list[dict[str, Any]],
    *,
    measurement_type: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    types = {c.get("measurement_type") for c in cells}
    if types and types != {measurement_type}:
        raise MixedSourceError(
            f"analysis cells mixed {types}; expected only {measurement_type!r}"
        )
    attach_inflation(cells)
    sat_points = []
    policies = sorted({str(c["policy"]) for c in cells})
    profiles = sorted({str(c["profile"]) for c in cells})
    for profile in profiles:
        for policy in policies:
            sat_points.append(saturation_point(cells, profile=profile, policy=policy))
    sustain = max_sustainable_throughput(cells)
    pareto = pareto_under_load(cells)
    eco_flags = [p["ecologic_pareto_efficient"] for p in pareto if p["ecologic_pareto_efficient"] is not None]
    return {
        "measurement_type": measurement_type,
        "n_cells": len(cells),
        "cells": cells,
        "inflation_note": "latency_inflation_vs_1x and cost_inflation_vs_1x are ratios to the 1x cell of the same profile and policy.",
        "saturation_points": sat_points,
        "max_sustainable_throughput": sustain,
        "pareto_under_load": pareto,
        "ecologic_pareto_efficient_all_tested_loads": (all(eco_flags) if eco_flags else None),
        "environment_only": True,
        "disclaimer": (
            "Saturation, sustainable throughput, and Pareto-efficiency are for this "
            "tested environment only. They are not universal EcoLogic limits."
        ),
        "git": git_snapshot(),
        **dict(extra or {}),
    }


def load_suite_jsonl(root: Path, *, expected: str) -> list[dict[str, Any]]:
    req_dir = Path(root) / "requests"
    if not req_dir.is_dir():
        raise FileNotFoundError(f"no request logs under {req_dir}")
    rows: list[dict[str, Any]] = []
    for path in sorted(req_dir.glob("*.jsonl")):
        rows.extend(load_jsonl(path, expected=expected))
    assert_homogeneous(rows, expected)
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze EcoLogic stress logs (MEASURED or SIMULATED, never mixed).")
    p.add_argument("--mode", required=True, choices=("measured", "simulated"))
    p.add_argument("--root", default=None, help="Directory containing requests/*.jsonl")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mt = MEASURED if args.mode == "measured" else SIMULATED
    root = Path(args.root) if args.root else None
    if root is None:
        from woais_experiments.paths import get_results_root

        root = get_results_root() / results_prefix(mt)
    rows = load_suite_jsonl(root, expected=mt)
    # Analyzer entry without cell grouping is not the primary path; stress_test writes analysis.
    write_result(
        f"{results_prefix(mt)}/raw_concat_meta.json",
        {
            "measurement_type": mt,
            "n_records": len(rows),
            "root": str(root),
        },
    )
    print(json.dumps({"measurement_type": mt, "n_records": len(rows), "root": str(root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

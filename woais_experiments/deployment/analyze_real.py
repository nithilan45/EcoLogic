"""Summaries for MEASURED_REAL_DEPLOYMENT logs. Energy is never labeled measured."""

from __future__ import annotations

import csv
from collections import defaultdict
from io import StringIO
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from woais_experiments.deployment.records import (
    ALLOWED_V2_MEASUREMENT_TYPES,
    MEASUREMENT_TYPE_DRY_RUN,
    SIMULATOR_KEYS,
)
from woais_experiments.frozen import write_result
from woais_experiments.latency.analyze_latency import bootstrap_ci, summarize
from woais_experiments.paths import public_relpath

RESULTS_PREFIX = "deployment_real_v2"
DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 20260909
DEFAULT_LEVEL = 0.95
FILE_PROVENANCE_KEYS = (
    "measurement_type",
    "cloud_backend",
    "region",
    "instance_configuration",
    "timestamp",
    "git_commit",
    "model_identifiers",
    "pricing_config_hash",
    "query_set_hash",
)


class MixedSourceError(ValueError):
    pass


def assert_real_record(row: Mapping[str, Any]) -> None:
    if not isinstance(row, dict):
        raise TypeError("record must be a dict")
    mt = row.get("measurement_type")
    if mt not in ALLOWED_V2_MEASUREMENT_TYPES:
        raise MixedSourceError(
            "deployment_real_v2 requires measurement_type in "
            f"{sorted(ALLOWED_V2_MEASUREMENT_TYPES)}, got {mt!r}"
        )
    overlap = SIMULATOR_KEYS.intersection(row)
    if overlap:
        raise MixedSourceError(f"refusing simulator fields {sorted(overlap)}")
    if row.get("energy_j") is not None or row.get("measured_energy") is not None:
        raise MixedSourceError("refusing energy fields labeled as measured")


def _suite_measurement_type(rows: Sequence[Mapping[str, Any]]) -> str:
    types = {r.get("measurement_type") for r in rows if r.get("measurement_type")}
    if not types:
        return MEASUREMENT_TYPE_DRY_RUN
    if types - ALLOWED_V2_MEASUREMENT_TYPES:
        raise MixedSourceError(
            f"unsupported deployment_real_v2 measurement_type values: {sorted(types)}"
        )
    if len(types) != 1:
        raise MixedSourceError(
            f"refusing mixed measurement_type values in one analysis: {sorted(types)}"
        )
    return str(next(iter(types)))


def _is_failure(row: Mapping[str, Any]) -> bool:
    if row.get("error_type") or row.get("error"):
        return True
    try:
        return int(row.get("http_status") or 0) >= 400
    except (TypeError, ValueError):
        return True


def _finite(xs: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for x in xs:
        if x is None:
            continue
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            out.append(v)
    return out


def _ci(samples: Sequence[float], point: float | None, stat: str, *, n_boot: int, seed: int, level: float) -> dict[str, Any]:
    if point is None or not samples:
        return {"available": False, "point": point, "reason": "no samples"}
    payload = bootstrap_ci(samples, stat, n_boot=n_boot, seed=seed, level=level)
    payload["available"] = payload.get("point") is not None
    return payload


def summarize_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> dict[str, Any]:
    n = len(rows)
    fails = [1.0 if _is_failure(r) else 0.0 for r in rows]
    retries = [1.0 if int(r.get("retry_count") or 0) > 0 else 0.0 for r in rows]
    success = [r for r in rows if not _is_failure(r)]
    lats_ok = _finite(r.get("end_to_end_ms") for r in success)
    router = _finite(r.get("router_decision_ms") for r in rows)
    costs_ok = _finite(_cost_of(r) for r in success)
    costs_all = _finite(_cost_of(r) for r in rows)
    inn = _finite(r.get("input_tokens") for r in success)
    outt = _finite(r.get("output_tokens") for r in success)
    walls = _finite(r.get("phase_wall_s") for r in rows)
    phase_wall = float(walls[0]) if walls else None
    throughput = (n / phase_wall) if phase_wall and phase_wall > 0 else None
    lat = summarize(lats_ok, n_boot=n_boot, seed=seed, level=level, ci_stats=("mean", "p50", "p95", "p99"))
    cost_mean = float(np.mean(costs_ok)) if costs_ok else None
    router_overhead = float(np.mean(router)) if router else None
    cold = sum(1 for r in rows if r.get("cold_start_observed") is True)
    success_rate = (1.0 - float(np.mean(fails))) if n else None
    kinds = sorted({str(r.get("environment_kind")) for r in rows if r.get("environment_kind")})
    serverless_flags = {bool(r.get("serverless")) for r in rows if r.get("serverless") is not None}
    return {
        "measurement_type": _suite_measurement_type(rows) if rows else MEASUREMENT_TYPE_DRY_RUN,
        "policy": rows[0].get("setup_name") or rows[0].get("policy") if rows else None,
        "setup": rows[0].get("setup") if rows else None,
        "workload": rows[0].get("workload") if rows else None,
        "n": n,
        "success_rate": success_rate,
        "success_rate_ci": _ci([1.0 - x for x in fails], success_rate, "mean", n_boot=n_boot, seed=seed, level=level),
        "throughput": throughput,
        "throughput_req_s": throughput,
        "p50_latency": lat["p50"],
        "p95_latency": lat["p95"],
        "p99_latency": lat["p99"],
        "mean_latency": lat["mean"],
        "p50_latency_ms": lat["p50"],
        "p95_latency_ms": lat["p95"],
        "p99_latency_ms": lat["p99"],
        "mean_latency_ms": lat["mean"],
        "latency_ci": lat["ci"],
        "latency_population": "successful_requests",
        "router_overhead": router_overhead,
        "router_overhead_ms": router_overhead,
        "router_overhead_ci": _ci(router, router_overhead, "mean", n_boot=n_boot, seed=seed, level=level) if router else {"available": False, "point": None},
        "input_tokens": float(np.sum(inn)) if inn else None,
        "output_tokens": float(np.sum(outt)) if outt else None,
        "mean_input_tokens": float(np.mean(inn)) if inn else None,
        "mean_output_tokens": float(np.mean(outt)) if outt else None,
        "realized_cost_per_query": cost_mean,
        "realized_cost_per_query_ci": _ci(costs_ok, cost_mean, "mean", n_boot=n_boot, seed=seed, level=level) if costs_ok else {"available": False, "point": None},
        "total_realized_cost": float(np.sum(costs_all)) if costs_all else None,
        "cold_start_count": int(cold),
        "retry_rate": float(np.mean(retries)) if n else None,
        "environment_kinds": kinds,
        "serverless": True if serverless_flags == {True} else False if serverless_flags == {False} else None,
        "cloud_measured": bool(rows[0].get("cloud_measured")) if rows else None,
        "generation_mode": rows[0].get("generation_mode") if rows else None,
        "backend": rows[0].get("backend") if rows else None,
        "energy": {
            "measured": False,
            "status": "not_measured",
            "note": "Estimated energy is not reported as measured energy.",
        },
        "phase_wall_s": phase_wall,
    }


def _cost_of(row: Mapping[str, Any]) -> float | None:
    for key in ("provider_cost", "realized_provider_cost"):
        v = row.get(key)
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(x):
            return x
    return None


def _group_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("setup"),
        row.get("setup_name") or row.get("policy"),
        row.get("workload"),
        row.get("generation_mode"),
        row.get("backend"),
    )


def _pareto_status(cost_a: float | None, lat_a: float | None, cost_b: float | None, lat_b: float | None) -> str:
    """Lower cost and lower latency are better. Quality is not graded in this run."""
    if cost_a is None or lat_a is None or cost_b is None or lat_b is None:
        return "incomparable_missing_metrics"
    a_better_cost = cost_a < cost_b
    a_better_lat = lat_a < lat_b
    b_better_cost = cost_b < cost_a
    b_better_lat = lat_b < lat_a
    a_not_worse = cost_a <= cost_b and lat_a <= lat_b
    b_not_worse = cost_b <= cost_a and lat_b <= lat_a
    if a_not_worse and (a_better_cost or a_better_lat):
        return "ecologic_dominates"
    if b_not_worse and (b_better_cost or b_better_lat):
        return "static_dominates"
    if abs(cost_a - cost_b) < 1e-18 and abs(lat_a - lat_b) < 1e-18:
        return "neutral"
    return "tradeoff"


def compare_ecologic_to_static(
    groups: Sequence[Mapping[str, Any]],
    *,
    mix: Mapping[str, Any] | None = None,
    measurement_type: str | None = None,
) -> dict[str, Any]:
    by_wl: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for g in groups:
        pol = str(g.get("policy") or "")
        wl = str(g.get("workload") or "")
        by_wl[wl][pol] = g
    mix = dict(mix or {})
    feasible = mix.get("cost_match_feasible")
    rows = []
    for wl, pols in sorted(by_wl.items()):
        eco = pols.get("ecologic")
        static = pols.get("cost_matched_static") or pols.get("static_mixture")
        if not eco or not static:
            continue
        c_e, c_s = eco.get("realized_cost_per_query"), static.get("realized_cost_per_query")
        l_e, l_s = eco.get("p95_latency_ms"), static.get("p95_latency_ms")
        lat_over = None
        if eco.get("mean_latency_ms") is not None and static.get("mean_latency_ms") is not None:
            lat_over = float(eco["mean_latency_ms"]) - float(static["mean_latency_ms"])
        router_ms = eco.get("router_overhead_ms")
        if feasible is False:
            rows.append({
                "workload": wl,
                "ecologic_cost_per_query": c_e,
                "static_cost_per_query": c_s,
                "cost_savings_vs_static": None,
                "cost_match_feasible": False,
                "comparator_role": "clamped_static_not_cost_matched",
                "ecologic_p95_ms": l_e,
                "static_p95_ms": l_s,
                "mean_latency_overhead_ms": lat_over,
                "router_decision_ms": router_ms,
                "dollar_break_even_routing_overhead_usd": None,
                "dollar_break_even_note": (
                    "Static mix was clamped because EcoLogic estimated cost is "
                    "outside [cheap, strong]. This is not a cost-matched comparison."
                ),
                "pareto_status_cost_vs_p95_latency": "not_a_cost_matched_comparison",
                "quality_graded_in_this_run": False,
            })
            continue
        savings = None if c_e is None or c_s is None else float(c_s) - float(c_e)
        break_even = None
        break_even_note = None
        if savings is None:
            break_even_note = "missing costs"
        elif savings <= 0:
            break_even = 0.0
            break_even_note = (
                "EcoLogic realized cost is not below cost-matched static; "
                "a positive routing fee cannot be justified on cost"
            )
        else:
            break_even = float(savings)
            break_even_note = (
                "Per-query routing fee (USD) that would erase EcoLogic's measured "
                "cost savings vs cost-matched static. Keyword router overhead in "
                "models.json is $0; this is a break-even on measured provider USD."
            )
        rows.append({
            "workload": wl,
            "ecologic_cost_per_query": c_e,
            "static_cost_per_query": c_s,
            "cost_savings_vs_static": savings,
            "cost_match_feasible": True if feasible is True else None,
            "comparator_role": "cost_matched_static",
            "ecologic_p95_ms": l_e,
            "static_p95_ms": l_s,
            "mean_latency_overhead_ms": lat_over,
            "router_decision_ms": router_ms,
            "dollar_break_even_routing_overhead_usd": break_even,
            "dollar_break_even_note": break_even_note,
            "pareto_status_cost_vs_p95_latency": _pareto_status(c_e, l_e, c_s, l_s),
            "quality_graded_in_this_run": False,
        })
    burst = next((r for r in rows if r["workload"] == "burst"), None)
    seq = next((r for r in rows if r["workload"] == "sequential_warm"), None)
    ranking_changed = None
    if burst and seq:
        ranking_changed = (
            burst["pareto_status_cost_vs_p95_latency"] != seq["pareto_status_cost_vs_p95_latency"]
        )
    mt = measurement_type
    if mt is None:
        mts = {g.get("measurement_type") for g in groups if g.get("measurement_type")}
        mt = next(iter(mts)) if len(mts) == 1 else None
    return {
        "measurement_type": mt,
        "cost_match_feasible": feasible,
        "workloads": rows,
        "conclusions_change_under_burst": ranking_changed,
        "quality_not_measured": True,
        "energy_not_measured": True,
    }


def analyze_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
    extra: Mapping[str, Any] | None = None,
    static_mix: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    for row in rows:
        assert_real_record(row)
    mt = _suite_measurement_type(rows)
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)
    groups = [summarize_group(v, n_boot=n_boot, seed=seed, level=level) for _, v in sorted(grouped.items())]
    payload: dict[str, Any] = {
        "measurement_type": mt,
        "n_records": len(rows),
        "n_groups": len(groups),
        "bootstrap": {"n_boot": int(n_boot), "seed": int(seed), "level": float(level), "method": "percentile"},
        "groups": groups,
        "comparison_ecologic_vs_cost_matched_static": compare_ecologic_to_static(
            groups, mix=static_mix, measurement_type=mt,
        ),
        "energy": {
            "measured": False,
            "status": "not_measured",
            "note": "Estimated energy is not reported as measured energy.",
        },
        "note": (
            "Suite measurement_type matches execution mode. DRY_RUN_LOCAL is not "
            "MEASURED_REAL_DEPLOYMENT. local_stub / local_paid are not serverless. "
            "cold_start_observed is process/container init only."
        ),
    }
    if extra:
        payload["run"] = dict(extra)
        payload["run"]["measurement_type"] = mt
    return payload


def summary_csv(payload: Mapping[str, Any], *, provenance: Mapping[str, Any] | None = None) -> str:
    fields = [
        "measurement_type", "setup", "policy", "workload", "n", "success_rate",
        "throughput", "p50_latency", "p95_latency", "p99_latency", "mean_latency",
        "router_overhead", "input_tokens", "output_tokens",
        "realized_cost_per_query", "total_realized_cost", "cold_start_count", "retry_rate",
        "generation_mode", "backend", "serverless",
    ]
    buf = StringIO()
    suite = payload.get("measurement_type") or MEASUREMENT_TYPE_DRY_RUN
    buf.write(f"# measurement_type={suite}\n")
    buf.write("# energy=not_measured (estimated energy is not reported as measured energy)\n")
    if provenance:
        buf.write(f"# cloud_backend={provenance.get('cloud_backend')}\n")
        buf.write(f"# region={provenance.get('region')}\n")
        buf.write(f"# timestamp={provenance.get('timestamp')}\n")
        buf.write(f"# git_commit={provenance.get('git_commit')}\n")
        buf.write(f"# pricing_config_hash={provenance.get('pricing_config_hash')}\n")
        buf.write(f"# query_set_hash={provenance.get('query_set_hash')}\n")
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for g in payload.get("groups") or []:
        row = {k: g.get(k) for k in fields}
        row["measurement_type"] = g.get("measurement_type") or suite
        w.writerow(row)
    return buf.getvalue()


def attach_file_provenance(payload: Mapping[str, Any] | None, provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Copy required file-level provenance onto a JSON artifact. Does not invent energy."""
    out = dict(payload or {})
    for key in FILE_PROVENANCE_KEYS:
        if key == "timestamp" and out.get("timestamp") and key not in provenance:
            continue
        if key in provenance:
            out[key] = provenance[key]
    if provenance.get("measurement_type"):
        out["measurement_type"] = provenance["measurement_type"]
    out.setdefault("energy", {
        "measured": False,
        "status": "not_measured",
        "note": "Estimated energy is not reported as measured energy.",
    })
    return out


def write_analysis(
    payload: Mapping[str, Any],
    *,
    prefix: str = RESULTS_PREFIX,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    payload = dict(payload)
    if provenance and provenance.get("measurement_type"):
        payload["measurement_type"] = provenance["measurement_type"]
    if provenance:
        payload = attach_file_provenance(payload, provenance)
    json_path = write_result(f"{prefix}/summary.json", payload, clobber=True)
    csv_path = write_result(f"{prefix}/summary.csv", summary_csv(payload, provenance=provenance), clobber=True)
    comparison = payload.get("comparison_ecologic_vs_cost_matched_static") or {}
    if provenance:
        comparison = attach_file_provenance(comparison if isinstance(comparison, Mapping) else {}, provenance)
    cmp_path = write_result(f"{prefix}/comparison.json", comparison, clobber=True)
    return {
        "summary_json": public_relpath(json_path),
        "summary_csv": public_relpath(csv_path),
        "comparison_json": public_relpath(cmp_path),
    }

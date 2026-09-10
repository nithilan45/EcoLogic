"""Summaries of *measured* deployment runs. Refuses simulator records."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from woais_experiments.deployment.records import (
    MEASUREMENT_TYPE,
    REQUIRED_REQUEST_FIELDS,
    assert_measured_record,
    missing_required_fields,
)
from woais_experiments.frozen import write_result
from woais_experiments.paths import public_relpath
from woais_experiments.latency.analyze_latency import bootstrap_ci, summarize
from woais_experiments.runner.gitinfo import git_snapshot

DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 20260909
DEFAULT_LEVEL = 0.95
RESULTS_PREFIX = "deployment_real"


class MixedSourceError(ValueError):
    pass


def _is_failure(row: Mapping[str, Any]) -> bool:
    if row.get("error_type"):
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


def load_request_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = Path(path).read_text(encoding="utf-8")
    if path.suffix == ".json" and text.lstrip().startswith("{"):
        payload = json.loads(text)
        if isinstance(payload, dict) and "records" in payload:
            raw_rows = payload["records"]
        elif isinstance(payload, list):
            raw_rows = payload
        else:
            raise MixedSourceError(f"unrecognized JSON layout in {path}")
        for row in raw_rows:
            assert_measured_record(row)
            rows.append(row)
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        assert_measured_record(row)
        missing = missing_required_fields(row)
        if missing:
            raise MixedSourceError(f"record missing required fields {missing}")
        rows.append(row)
    if not rows:
        raise MixedSourceError(f"no MEASURED records in {path}")
    return rows


def _group_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("setup"),
        row.get("setup_name") or row.get("policy"),
        row.get("concurrency"),
        row.get("generation_mode"),
        row.get("backend"),
    )


def summarize_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> dict[str, Any]:
    n = len(rows)
    fails = [1.0 if _is_failure(r) else 0.0 for r in rows]
    failure_rate = float(np.mean(fails)) if n else None
    success = [r for r in rows if not _is_failure(r)]
    lats_all = _finite(r.get("end_to_end_ms") for r in rows)
    lats_ok = _finite(r.get("end_to_end_ms") for r in success)
    router = _finite(r.get("router_decision_ms") for r in rows)
    costs_all = _finite(r.get("realized_provider_cost") for r in rows)
    costs_ok = _finite(r.get("realized_provider_cost") for r in success)
    walls = _finite(r.get("phase_wall_s") for r in rows)
    phase_wall = float(walls[0]) if walls else None
    throughput = None
    if phase_wall is not None and phase_wall > 0:
        throughput = n / phase_wall

    lat_sum = summarize(lats_ok, n_boot=n_boot, seed=seed, level=level, ci_stats=("mean", "p50", "p90", "p95", "p99"))
    lat_all = summarize(lats_all, n_boot=n_boot, seed=seed, level=level, ci_stats=("mean", "p50", "p95", "p99"))
    cost_mean = float(np.mean(costs_ok)) if costs_ok else None
    total_cost = float(np.sum(costs_all)) if costs_all else None
    kinds = sorted({str(r.get("environment_kind")) for r in rows if r.get("environment_kind")})
    cloud_flags = {bool(r.get("cloud_measured")) for r in rows if r.get("cloud_measured") is not None}

    def _ci(samples: Sequence[float], point: float | None, stat: str) -> dict[str, Any]:
        if point is None or not samples:
            return {"available": False, "point": point, "reason": "no samples"}
        payload = bootstrap_ci(samples, stat, n_boot=n_boot, seed=seed, level=level)
        payload["available"] = payload.get("point") is not None
        return payload

    router_overhead = float(np.mean(router)) if router else None
    out = {
        "measurement_type": MEASUREMENT_TYPE,
        "n": n,
        "n_failed": int(sum(fails)),
        "failure_rate": failure_rate,
        "failure_rate_ci": _ci(fails, failure_rate, "mean"),
        "throughput_req_s": throughput,
        "throughput_method": "n_over_phase_wall_clock" if throughput is not None else None,
        "throughput_ci": {
            "available": False,
            "reason": "single measured phase wall-clock; not resampled",
            "point": throughput,
        },
        "p50_latency_ms": lat_sum["p50"],
        "p90_latency_ms": lat_sum["p90"],
        "p95_latency_ms": lat_sum["p95"],
        "p99_latency_ms": lat_sum["p99"],
        "mean_latency_ms": lat_sum["mean"],
        "latency_ci": lat_sum["ci"],
        "latency_population": "successful_requests",
        "n_latency_success": len(lats_ok),
        "n_latency_including_failures": len(lats_all),
        "p95_latency_ms_including_failures": lat_all["p95"],
        "mean_latency_ms_including_failures": lat_all["mean"],
        "router_overhead_ms": router_overhead,
        "router_overhead_ci": _ci(router, router_overhead, "mean") if router else {"available": False, "point": None, "reason": "no router_decision_ms (direct or static mix)"},
        "router_overhead_definition": "mean observed router_decision_ms; not inferred from residual latency",
        "realized_cost_per_query": cost_mean,
        "realized_cost_per_successful_query": cost_mean,
        "realized_cost_per_query_ci": _ci(costs_ok, cost_mean, "mean") if costs_ok else {"available": False, "point": None, "reason": "no costs"},
        "total_realized_cost": total_cost,
        "cost_denominator": "successful_requests_with_a_cost",
        "environment_kinds": kinds,
        "cloud_measured": True if cloud_flags == {True} else False if cloud_flags == {False} else None,
        "not_a_cloud_measurement": (cloud_flags == {False}) or ("local_stub" in kinds) or ("in_process_lambda_stub" in kinds),
        "phase_wall_s": phase_wall,
        "cost_basis_values": sorted({str(r.get("cost_basis")) for r in rows if r.get("cost_basis")}),
        "generation_mode": rows[0].get("generation_mode") if rows else None,
        "backend": rows[0].get("backend") if rows else None,
        "setup": rows[0].get("setup") if rows else None,
        "setup_name": rows[0].get("setup_name") or rows[0].get("policy") if rows else None,
        "concurrency": rows[0].get("concurrency") if rows else None,
        "cold_n": sum(1 for r in rows if r.get("lifecycle_state") == "cold"),
        "warm_n": sum(1 for r in rows if r.get("lifecycle_state") == "warm"),
        "lifecycle_bases": sorted({str(r.get("lifecycle_basis")) for r in rows if r.get("lifecycle_basis")}),
    }
    return out


def analyze_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    for row in rows:
        assert_measured_record(row)
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    modes = set()
    for row in rows:
        grouped[_group_key(row)].append(row)
        modes.add(row.get("generation_mode"))
    setups = [summarize_group(v, n_boot=n_boot, seed=seed, level=level) for _, v in sorted(grouped.items())]
    payload: dict[str, Any] = {
        "measurement_type": MEASUREMENT_TYPE,
        "n_records": len(rows),
        "n_groups": len(setups),
        "generation_modes": sorted(str(m) for m in modes if m is not None),
        "bootstrap": {"n_boot": int(n_boot), "seed": int(seed), "level": float(level), "method": "percentile"},
        "git": git_snapshot(),
        "groups": setups,
        "note": (
            "Deployment logs tagged MEASURED are not DES outputs. "
            "local_stub / in_process_lambda_stub / dry_run are not cloud measurements. "
            "Latency percentiles are success-conditioned unless _including_failures."
        ),
    }
    if extra:
        payload["run"] = dict(extra)
        payload["run"]["measurement_type"] = MEASUREMENT_TYPE
    return payload


def summary_csv(payload: Mapping[str, Any]) -> str:
    fields = [
        "measurement_type",
        "setup",
        "setup_name",
        "concurrency",
        "generation_mode",
        "backend",
        "n",
        "throughput_req_s",
        "p50_latency_ms",
        "p90_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "mean_latency_ms",
        "failure_rate",
        "router_overhead_ms",
        "realized_cost_per_query",
        "total_realized_cost",
        "cold_n",
        "warm_n",
    ]
    buf = StringIO()
    buf.write(f"# measurement_type={MEASUREMENT_TYPE}\n")
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for g in payload.get("groups") or []:
        row = {k: g.get(k) for k in fields}
        row["measurement_type"] = MEASUREMENT_TYPE
        w.writerow(row)
    return buf.getvalue()


def write_analysis(
    payload: Mapping[str, Any],
    *,
    prefix: str = RESULTS_PREFIX,
) -> dict[str, str]:
    payload = dict(payload)
    payload["measurement_type"] = MEASUREMENT_TYPE
    json_path = write_result(f"{prefix}/summary.json", payload)
    csv_path = write_result(f"{prefix}/summary.csv", summary_csv(payload))
    return {"summary_json": public_relpath(json_path), "summary_csv": public_relpath(csv_path)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze MEASURED EcoLogic deployment logs")
    p.add_argument("requests_jsonl")
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--prefix", default=RESULTS_PREFIX)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_request_records(Path(args.requests_jsonl))
    payload = analyze_records(rows, n_boot=args.n_boot, seed=args.seed)
    paths = write_analysis(payload, prefix=args.prefix)
    print(json.dumps({"measurement_type": MEASUREMENT_TYPE, "paths": paths, "n": payload["n_records"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

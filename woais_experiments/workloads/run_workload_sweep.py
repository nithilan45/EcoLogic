"""Run SIMULATED sensitivity sweeps from YAML. Not a cloud deployment.

Sweeps are one-factor-at-a-time around the YAML baseline. Factor values are
exactly those listed in the config; this script does not add extra points.
"""

from __future__ import annotations

import copy
import csv
from io import StringIO
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from woais_experiments.frozen import write_result
from woais_experiments.paths import CONFIGS
from woais_experiments.workloads.arrival_processes import arrivals_from_config
from woais_experiments.workloads.simulator import (
    DISCLAIMER,
    LABEL,
    Catalog,
    catalog_from_item_matrix,
    catalog_synthetic,
    simulate,
    validate_config,
    _nested_get,
    _policy_cfg,
)

DEFAULT_CONFIG = CONFIGS / "workload_simulator.yaml"

SWEEP_CSV_FIELDS = (
    "SIMULATED",
    "policy",
    "policy_type",
    "factor",
    "factor_value",
    "arrivals_process",
    "n_completed",
    "throughput_per_s",
    "queue_delay_mean_s",
    "end_to_end_mean_s",
    "p50_latency_s",
    "p95_latency_s",
    "p99_latency_s",
    "cold_start_count",
    "cold_start_rate",
    "requests_per_instance_mean",
    "cost_per_request",
    "total_realized_inference_cost",
    "sla_violation_rate",
)


def load_yaml_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else DEFAULT_CONFIG
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise TypeError("workload YAML must be a mapping")
    validate_config(cfg)
    if "shared_arrivals_across_policies" not in cfg or cfg["shared_arrivals_across_policies"] is None:
        raise KeyError("shared_arrivals_across_policies must be specified in YAML")
    if "sweep" not in cfg or cfg["sweep"] is None:
        raise KeyError("sweep must be specified in YAML")
    return cfg


def _set_dotted(cfg: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Any = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            raise KeyError(f"cannot set {path}: missing {p}")
        cur = cur[p]
    cur[parts[-1]] = value


def _get_dotted(cfg: Mapping[str, Any], path: str) -> Any:
    return _nested_get(cfg, path)


def _build_catalog(cfg: Mapping[str, Any]) -> Catalog:
    source = str(_nested_get(cfg, "catalog.source")).lower()
    if source == "synthetic":
        return catalog_synthetic(cfg)
    if source == "empirical_stage12":
        from woais_experiments.accounting.costs import cost_fn_usd
        from woais_experiments.frozen import load_stage12_matrix, load_stage12_routing
        from woais_experiments.routing.policies import keyword_from_routing, oracle

        matrix = load_stage12_matrix()
        routing = load_stage12_routing()
        eco = keyword_from_routing(routing, matrix.item_ids, "raw")
        ora = oracle(matrix, cost_fn_usd(matrix))
        return catalog_from_item_matrix(cfg, matrix, routing, ora, eco)
    raise ValueError(f"unknown catalog.source {source!r}")


def policy_runnable(catalog: Catalog, cfg: Mapping[str, Any], name: str) -> tuple[bool, str]:
    p = _policy_cfg(cfg, name)
    kind = str(p["type"])
    if kind in {"always_cheap", "always_strong"}:
        return True, ""
    if kind == "cost_matched_static":
        if p.get("f_strong") is not None:
            return True, ""
        target = p.get("match_to")
        if target in catalog.assignments:
            return True, ""
        return False, f"SIMULATED skip: cost-matched needs assignments[{target!r}]"
    if kind in catalog.assignments or name in catalog.assignments:
        return True, ""
    return False, f"SIMULATED skip: no assignment map for policy {name!r} type {kind!r}"


def _run_policies(
    cfg: dict[str, Any],
    catalog: Catalog,
    arrivals: np.ndarray,
    queries: list[str],
    *,
    factor: str,
    factor_value: Any,
) -> list[dict[str, Any]]:
    shared = bool(cfg["shared_arrivals_across_policies"])
    rows = []
    seed = int(_nested_get(cfg, "seed"))
    for i, name in enumerate(_nested_get(cfg, "policy_order")):
        ok, reason = policy_runnable(catalog, cfg, name)
        if not ok:
            rows.append(
                {
                    "SIMULATED": True,
                    "label": LABEL,
                    "disclaimer": DISCLAIMER,
                    "policy": name,
                    "skipped": True,
                    "skip_reason": reason,
                    "factor": factor,
                    "factor_value": factor_value,
                }
            )
            continue
        rng = np.random.default_rng(seed + 17 * i)
        if shared:
            a, q = arrivals, queries
        else:
            a = arrivals_from_config(_nested_get(cfg, "arrivals"), int(cfg["n_requests"]), rng)
            q = [str(x) for x in rng.choice(catalog.query_ids, size=int(cfg["n_requests"]), replace=True)]
        out = simulate(cfg, catalog, name, rng=rng, arrivals_s=a, query_seq=q)
        out["factor"] = factor
        out["factor_value"] = factor_value
        out.pop("per_request", None)
        rows.append(out)
    return rows


def _baseline_value(cfg: Mapping[str, Any], path: str) -> Any:
    try:
        return _get_dotted(cfg, path)
    except KeyError:
        return None


def _values_equal(a: Any, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) <= 1e-12
    except (TypeError, ValueError):
        return a == b


def expand_ofat(cfg: dict[str, Any]) -> list[tuple[str, Any, dict[str, Any]]]:
    sweep = _nested_get(cfg, "sweep")
    mode = str(_nested_get(sweep, "mode")).lower()
    if mode != "ofat":
        raise ValueError(f"sweep.mode {mode!r} is not implemented; YAML must use ofat")
    factors = _nested_get(sweep, "factors")
    burst_proc = str(_nested_get(sweep, "burstiness_sets_process"))
    snapshots: list[tuple[str, Any, dict[str, Any]]] = [("baseline", None, copy.deepcopy(cfg))]
    for path, values in factors.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"sweep.factors.{path} must be a non-empty list")
        base = _baseline_value(cfg, path)
        for val in values:
            if base is not None and _values_equal(val, base):
                continue
            snapped = copy.deepcopy(cfg)
            _set_dotted(snapped, path, val)
            if path.startswith("arrivals.on_off."):
                snapped["arrivals"]["process"] = burst_proc
            snapshots.append((str(path), val, snapped))
    return snapshots


def run_sweeps(
    cfg: Mapping[str, Any] | None = None,
    catalog: Catalog | None = None,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    if cfg is None:
        cfg = load_yaml_config(config_path)
    else:
        validate_config(cfg)
    cfg = copy.deepcopy(dict(cfg))
    built = catalog or _build_catalog(cfg)
    snapshots = expand_ofat(cfg)
    all_rows: list[dict[str, Any]] = []
    for factor, value, snapped in snapshots:
        rng0 = np.random.default_rng(int(snapped["seed"]))
        n = int(snapped["n_requests"])
        arrivals = arrivals_from_config(snapped["arrivals"], n, rng0)
        queries = [str(x) for x in rng0.choice(built.query_ids, size=n, replace=True)]
        all_rows.extend(
            _run_policies(snapped, built, arrivals, queries, factor=factor, factor_value=value)
        )
    payload = {
        "SIMULATED": True,
        "label": LABEL,
        "disclaimer": DISCLAIMER,
        "n_runs": len(all_rows),
        "n_snapshots": len(snapshots),
        "catalog_note": built.service_time_note,
        "runs": all_rows,
    }
    return payload


def _csv_row(run: Mapping[str, Any]) -> dict[str, Any]:
    m = run.get("metrics") or {}
    q = m.get("queue_delay_s") or {}
    rpi = m.get("requests_per_instance") or {}
    return {
        "SIMULATED": True,
        "policy": run.get("policy"),
        "policy_type": run.get("policy_type"),
        "factor": run.get("factor"),
        "factor_value": run.get("factor_value"),
        "arrivals_process": run.get("arrivals_process"),
        "n_completed": run.get("n_completed"),
        "throughput_per_s": m.get("throughput_per_s"),
        "queue_delay_mean_s": q.get("mean"),
        "end_to_end_mean_s": (m.get("end_to_end_latency_s") or {}).get("mean"),
        "p50_latency_s": m.get("p50_latency_s"),
        "p95_latency_s": m.get("p95_latency_s"),
        "p99_latency_s": m.get("p99_latency_s"),
        "cold_start_count": m.get("cold_start_count"),
        "cold_start_rate": m.get("cold_start_rate"),
        "requests_per_instance_mean": rpi.get("mean"),
        "cost_per_request": m.get("cost_per_request"),
        "total_realized_inference_cost": m.get("total_realized_inference_cost"),
        "sla_violation_rate": m.get("sla_violation_rate"),
    }


def sweep_to_csv(payload: Mapping[str, Any]) -> str:
    buf = StringIO()
    wr = csv.DictWriter(buf, fieldnames=list(SWEEP_CSV_FIELDS))
    wr.writeheader()
    for run in payload["runs"]:
        wr.writerow(_csv_row(run))
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
    if isinstance(obj, float) and not np.isfinite(obj):
        if np.isnan(obj):
            return None
        return "Infinity" if obj > 0 else "-Infinity"
    return obj


def run_and_save(
    cfg: Mapping[str, Any] | None = None,
    catalog: Catalog | None = None,
    *,
    config_path: str | Path | None = None,
    relpath: str = "workloads/simulator_sweep",
) -> dict[str, Any]:
    payload = run_sweeps(cfg, catalog, config_path=config_path)
    write_result(f"{relpath}.json", _jsonable(payload))
    write_result(f"{relpath}.csv", sweep_to_csv(payload))
    payload["saved"] = {"json": f"{relpath}.json", "csv": f"{relpath}.csv"}
    return payload


def main() -> None:
    payload = run_and_save()
    print(f"SIMULATED {payload['n_runs']} runs; wrote {payload['saved']}")


if __name__ == "__main__":
    main()

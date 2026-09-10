"""CLI orchestrator for EcoLogic systems stress tests.

Modes are exclusive:
  --mode simulated  → measurement_type=SIMULATED, never calls a provider API
  --mode measured   → measurement_type=MEASURED, local stub unless --allow-api

Paid measured runs require both --allow-api and --max-cost-usd.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from woais_experiments.deployment.app import start_server, stop_server
from woais_experiments.deployment.client import DeploymentClient
from woais_experiments.deployment.config import load_deployment_config
from woais_experiments.deployment.infer import build_context
from woais_experiments.deployment.static_mix import cheap_strong_slugs, cost_matched_assignments
from woais_experiments.frozen import to_jsonable, write_result
from woais_experiments.paths import PACKAGE, assert_inside_results, get_results_root
from woais_experiments.runner.gitinfo import git_snapshot
from woais_experiments.stress import MEASURED, POLICY_SPECS, SIMULATED
from woais_experiments.stress.analyze_stress import analyze_cells, cell_metrics
from woais_experiments.stress.load_generator import CostCapExceeded, CostGuard, run_open_loop
from woais_experiments.stress.records import (
    MixedSourceError,
    cell_jsonl_relpath,
    cell_key,
    load_jsonl,
    occupancy_at_arrivals,
    results_prefix,
    write_jsonl,
)
from woais_experiments.stress.workload_profiles import (
    LOAD_MULTIPLIERS,
    PROFILE_NAMES,
    arrivals_for_profile,
    safe_multipliers,
)
from woais_experiments.workloads.simulator import catalog_synthetic, simulate, validate_config

DEFAULT_CONFIG = PACKAGE / "stress" / "stress_config.yaml"


def load_stress_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_CONFIG
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"stress config {p} must be a mapping")
    raw["_config_path"] = str(p.resolve())
    return raw


def _csv(text: str | None) -> list[str]:
    if not text:
        return []
    return [p.strip() for p in str(text).split(",") if p.strip()]


def _profiles(cfg: Mapping[str, Any], requested: Sequence[str] | None) -> list[str]:
    if requested:
        names = [str(x) for x in requested]
    else:
        raw = cfg.get("profiles") or "all"
        if raw == "all" or raw is None:
            names = list(PROFILE_NAMES)
        elif isinstance(raw, str):
            names = _csv(raw)
        else:
            names = [str(x) for x in raw]
    unknown = [n for n in names if n not in PROFILE_NAMES]
    if unknown:
        raise ValueError(f"unknown profiles {unknown}; known: {PROFILE_NAMES}")
    return names


def _multipliers(cfg: Mapping[str, Any], requested: Sequence[float] | None) -> list[float]:
    if requested:
        return [float(x) for x in requested]
    raw = cfg.get("multipliers") or list(LOAD_MULTIPLIERS)
    return [float(x) for x in raw]


def _n_requests(cfg: Mapping[str, Any], mode: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    if mode == "measured":
        return int(cfg.get("n_requests_measured") or cfg.get("n_requests") or 8)
    return int(cfg.get("n_requests") or 16)


def _hash_assign(qids: Sequence[str], cheap: str, strong: str) -> dict[str, str]:
    out = {}
    for q in qids:
        h = int(hashlib.sha256(str(q).encode("utf-8")).hexdigest()[:8], 16)
        out[str(q)] = strong if h % 2 == 0 else cheap
    return out


def _quality_map(sim_cfg: Mapping[str, Any]) -> dict[str, float]:
    models = ((sim_cfg.get("catalog") or {}).get("models") or {})
    out: dict[str, float] = {}
    for name, spec in models.items():
        if isinstance(spec, Mapping) and spec.get("quality") is not None:
            out[str(name)] = float(spec["quality"])
    return out


def build_sim_cfg(stress_cfg: Mapping[str, Any], *, n: int, seed: int) -> dict[str, Any]:
    sim = dict(stress_cfg.get("simulator") or {})
    if not sim:
        raise ValueError("stress config missing simulator: block")
    catalog = dict(sim.get("catalog") or {})
    models = dict(catalog.get("models") or {})
    catalog["models"] = models
    catalog.setdefault("source", "synthetic")
    catalog.setdefault("cheap_model", "cheap")
    catalog.setdefault("strong_model", "strong")
    qids = [str(q) for q in (catalog.get("query_ids") or [f"q{i}" for i in range(n)])]
    catalog["query_ids"] = qids
    assigns = dict(catalog.get("assignments") or {})
    eco = assigns.get("ecologic") or _hash_assign(qids, str(catalog["cheap_model"]), str(catalog["strong_model"]))
    assigns["ecologic"] = {str(k): str(v) for k, v in eco.items()}
    catalog["assignments"] = assigns
    sim["catalog"] = catalog
    sim["simulated"] = True
    sim["seed"] = int(seed)
    sim["n_requests"] = int(n)
    sim.setdefault("label", "SIMULATED")
    sim.setdefault(
        "arrivals",
        {
            "process": "constant",
            "start_s": 0.0,
            "interarrival_s": 1.0,
            "rate_per_s": 1.0,
            "timestamps_s": [0.0],
            "on_off": {
                "lambda_on_per_s": 1.0,
                "lambda_off_per_s": 0.0,
                "mean_on_s": 1.0,
                "mean_off_s": 1.0,
                "initial_state": "ON",
            },
        },
    )
    sim.setdefault("policy_order", [p["simulated"] for p in POLICY_SPECS])
    validate_config(sim)
    return sim


def _wait_healthy(client: DeploymentClient, *, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    last: Exception | None = None
    while time.time() < deadline:
        try:
            payload = client.health()
            if payload.get("ok"):
                return
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(0.05)
    raise RuntimeError(f"stress measured server failed health check: {last}")


def _prompts(dep_cfg: Mapping[str, Any]) -> list[dict[str, str]]:
    items = list(dep_cfg.get("prompts") or [])
    if not items:
        raise SystemExit("deployment config has no prompts")
    out = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            out.append({"id": f"p{i}", "text": item})
        else:
            out.append({"id": str(item.get("id") or f"p{i}"), "text": str(item["text"])})
    return out


def _state_path(measurement_type: str) -> Path:
    return assert_inside_results(get_results_root() / results_prefix(measurement_type) / "state.json")


def _load_state(measurement_type: str) -> dict[str, Any]:
    path = _state_path(measurement_type)
    if not path.exists():
        return {"measurement_type": measurement_type, "cells": {}, "stopped": False}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("measurement_type") != measurement_type:
        raise MixedSourceError(
            f"state file {path} has measurement_type={raw.get('measurement_type')!r}"
        )
    raw.setdefault("cells", {})
    return raw


def _save_state(state: Mapping[str, Any]) -> Path:
    mt = str(state["measurement_type"])
    path = _state_path(mt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(dict(state)), indent=2) + "\n", encoding="utf-8")
    return path


def _pad_replay(timestamps: Sequence[float], n: int) -> list[float]:
    ts = [float(x) for x in timestamps]
    if not ts:
        raise ValueError("replay profile requires replay_timestamps_s")
    if len(ts) >= n:
        return ts
    gap = (ts[1] - ts[0]) if len(ts) > 1 else 1.0
    period = (ts[-1] - ts[0]) + max(gap, 1e-6)
    out = list(ts)
    k = 1
    while len(out) < n:
        out.extend([x + k * period for x in ts])
        k += 1
    return out[:n]


def simulated_rows(
    des_out: Mapping[str, Any],
    *,
    profile: str,
    multiplier: float,
    policy: str,
    quality_of: Mapping[str, float],
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pr in des_out.get("per_request") or []:
        e2e = pr.get("end_to_end_s")
        arrival = float(pr["arrival_s"])
        inf = pr.get("inference_cost")
        router_c = pr.get("router_cost_usd")
        if inf is None:
            cost = None
        else:
            cost = float(inf) + float(router_c or 0.0)
        model = str(pr["model"])
        rows.append({
            "measurement_type": SIMULATED,
            "suite": "stress_simulated",
            "request_id": f"{profile}-{float(multiplier):g}-{policy}-{pr['rid']}",
            "policy": policy,
            "stress_profile": profile,
            "load_multiplier": float(multiplier),
            "scheduled_arrival_s": arrival,
            "complete_s": None if e2e is None else arrival + float(e2e),
            "end_to_end_s": e2e,
            "sojourn_s": e2e,
            "end_to_end_ms": None if e2e is None else float(e2e) * 1000.0,
            "queue_wait_s": pr.get("queue_delay_s"),
            "service_time_s": pr.get("service_s"),
            "router_decision_s": pr.get("router_overhead_s"),
            "router_decision_ms": None if pr.get("router_overhead_s") is None else float(pr["router_overhead_s"]) * 1000.0,
            "selected_model": model,
            "model": model,
            "cost_usd": cost,
            "inference_cost": pr.get("inference_cost"),
            "realized_provider_cost": pr.get("inference_cost"),
            "quality": quality_of.get(model),
            "http_status": 200,
            "error_type": None,
            "retry_count": 0,
            "des_wait_s": pr.get("queue_delay_s"),
            "des_sojourn_s": e2e,
            "simulated_sojourn_s": e2e,
            "cold_penalty_s": pr.get("cold_delay_s"),
            "simulator": True,
            "simulation_seed": int(seed),
            "modeled_cold_start": bool(pr.get("cold_start")),
        })
    occ = occupancy_at_arrivals(
        [r["scheduled_arrival_s"] for r in rows],
        [r["complete_s"] for r in rows],
    )
    for row, depth in zip(rows, occ):
        row["queue_depth_at_arrival"] = int(depth)
    return rows


def _run_simulated_cell(
    *,
    sim_cfg: dict[str, Any],
    catalog,
    profile: str,
    multiplier: float,
    policy: str,
    arrivals: np.ndarray,
    query_seq: list[str],
    seed: int,
    quality_of: Mapping[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    des = simulate(
        sim_cfg,
        catalog,
        policy,
        rng=rng,
        arrivals_s=arrivals,
        query_seq=query_seq,
    )
    rows = simulated_rows(
        des,
        profile=profile,
        multiplier=multiplier,
        policy=policy,
        quality_of=quality_of,
        seed=seed,
    )
    return rows, des


def _measured_payloads(
    *,
    prompts: Sequence[Mapping[str, str]],
    n: int,
    profile: str,
    multiplier: float,
    policy_name: str,
    endpoint_policy: str,
    assignments: Mapping[str, str] | None,
) -> list[dict[str, Any]]:
    out = []
    for i in range(n):
        item = prompts[i % len(prompts)]
        pid = str(item["id"])
        forced = None
        if endpoint_policy == "static_mixture" and assignments is not None:
            forced = assignments[pid]
        out.append({
            "request_id": f"{profile}-{float(multiplier):g}-{policy_name}-{i}",
            "prompt": str(item["text"]),
            "prompt_id": pid,
            "policy": endpoint_policy,
            "canonical_policy": policy_name,
            "forced_model": forced,
            "stress_profile": profile,
            "load_multiplier": float(multiplier),
        })
    return out


def _run_measured_cell(
    *,
    client: DeploymentClient,
    prompts: Sequence[Mapping[str, str]],
    n: int,
    profile: str,
    multiplier: float,
    policy_name: str,
    endpoint_policy: str,
    assignments: Mapping[str, str] | None,
    arrivals: np.ndarray,
    max_workers: int,
    cost_guard: CostGuard,
    extra_base: Mapping[str, Any],
) -> dict[str, Any]:
    payloads = _measured_payloads(
        prompts=prompts,
        n=n,
        profile=profile,
        multiplier=multiplier,
        policy_name=policy_name,
        endpoint_policy=endpoint_policy,
        assignments=assignments,
    )

    def submit(i: int, payload: Mapping[str, Any]) -> dict[str, Any]:
        rec = client.infer(
            str(payload["prompt"]),
            policy=str(payload["policy"]),
            request_id=str(payload["request_id"]),
            forced_model=payload.get("forced_model"),
            extra={
                **dict(extra_base),
                "prompt_id": payload["prompt_id"],
                "stress_profile": payload["stress_profile"],
                "load_multiplier": payload["load_multiplier"],
            },
        )
        row = rec.to_dict()
        row["measurement_type"] = MEASURED
        row["suite"] = "stress_measured"
        row["policy"] = policy_name
        row["endpoint_policy"] = endpoint_policy
        row["stress_profile"] = profile
        row["load_multiplier"] = float(multiplier)
        row["prompt_id"] = payload["prompt_id"]
        return row

    return run_open_loop(
        list(arrivals),
        payloads,
        submit,
        max_workers=max_workers,
        cost_guard=cost_guard,
    )


def plan_cells(
    profiles: Sequence[str],
    multipliers: Sequence[float],
    policies: Sequence[str],
) -> list[tuple[str, float, str]]:
    return [(p, float(m), pol) for p in profiles for m in multipliers for pol in policies]


def run_suite(
    cfg: Mapping[str, Any],
    *,
    mode: str,
    allow_api: bool,
    dry_run: bool,
    max_cost_usd: float | None,
    resume: bool,
    profiles: Sequence[str],
    multipliers: Sequence[float],
    n_requests: int,
    max_multiplier: float | None,
    deployment_config: str | Path | None = None,
    base_url: str | None = None,
    start_server_local: bool = True,
) -> dict[str, Any]:
    mode = str(mode).lower()
    if mode not in {"measured", "simulated"}:
        raise ValueError("mode must be measured or simulated")
    if mode == "simulated" and allow_api:
        raise ValueError("simulated stress tests never call APIs; omit --allow-api")
    if allow_api and dry_run:
        raise ValueError("--allow-api and --dry-run are mutually exclusive")
    paid = bool(allow_api) and mode == "measured" and not dry_run
    if paid and max_cost_usd is None:
        raise ValueError("paid measured runs require --max-cost-usd")
    if mode == "measured" and not allow_api:
        dry_run = True
        paid = False

    mt = MEASURED if mode == "measured" else SIMULATED
    seed = int(cfg.get("seed") or 20260909)
    rates = cfg.get("rates_1x") or {}
    replay = [float(x) for x in (cfg.get("replay_timestamps_s") or [])]
    sla = dict(cfg.get("sla") or {})
    estimate = (cfg.get("cost") or {}).get("estimate_usd_per_query")
    safe = safe_multipliers(mode=mode, paid=paid, max_multiplier=max_multiplier, requested=list(multipliers))
    policy_names = [str(p) for p in (cfg.get("policies") or [s["name"] for s in POLICY_SPECS])]
    cells_plan = plan_cells(profiles, safe, policy_names)
    guard = CostGuard(max_cost_usd, estimate_usd_per_query=estimate, paid=paid)
    state = _load_state(mt) if resume else {"measurement_type": mt, "cells": {}, "stopped": False}
    if state.get("measurement_type") != mt:
        raise MixedSourceError("resume state measurement_type does not match --mode")
    prefix = results_prefix(mt)

    sim_cfg = None
    catalog = None
    quality_of: dict[str, float] = {}
    capacity = int(cfg.get("max_concurrency") or 8)
    strong_model = None
    client = None
    httpd = None
    prompts: list[dict[str, str]] = []
    assignments: dict[str, str] | None = None
    extra_base: dict[str, Any] = {}

    if mode == "simulated":
        sim_cfg = build_sim_cfg(cfg, n=n_requests, seed=seed)
        catalog = catalog_synthetic(sim_cfg)
        quality_of = _quality_map(sim_cfg)
        strong_model = catalog.strong_model
        srv = sim_cfg["serverless"]
        capacity = int(srv["max_instances"]) * int(srv["max_concurrency_per_instance"])
    else:
        dep = load_deployment_config(deployment_config)
        if paid:
            capacity = min(capacity, int(cfg.get("max_concurrency_paid") or 2))
        ctx = build_context(dep, allow_api=paid, dry_run=not paid, backend="local")
        prompts = _prompts(dep)
        cheap_slug, strong_slug = cheap_strong_slugs(ctx.models, ctx.roles)
        strong_model = strong_slug
        stub_out = int((dep.get("stub") or {}).get("output_tokens", 16))
        mix = cost_matched_assignments(
            prompts,
            prices=ctx.prices,
            models=ctx.models,
            classify_fn=ctx.classify_fn,
            cheap_slug=cheap_slug,
            strong_slug=strong_slug,
            output_tokens=stub_out,
            seed=seed,
            chars_per_token=float((dep.get("stub") or {}).get("chars_per_token", 4.0)),
        )
        assignments = dict(mix["assignments"])
        extra_base = {
            "generation_mode": ctx.generation_mode,
            "backend": "local",
            "paid_api": paid,
        }
        timeout = float(cfg.get("timeout_s") or (dep.get("client") or {}).get("timeout_s") or 30.0)
        retries = int((dep.get("retries") or {}).get("client_max_retries", 2))
        url = base_url
        if start_server_local and not url:
            listen = dep.get("listen") or {}
            host = str(listen.get("host") or "127.0.0.1")
            if host in {"0.0.0.0", "::"}:
                host = "127.0.0.1"
            httpd, _thread = start_server(ctx, host=host, port=0)
            port = int(httpd.server_address[1])
            url = f"http://{host}:{port}"
        if url is None:
            raise RuntimeError("measured mode needs a local server or --base-url")
        client = DeploymentClient(url, timeout_s=timeout, max_retries=retries)
        if httpd is not None:
            _wait_healthy(client)

    def _is_done(profile: str, multiplier: float, policy: str) -> bool:
        if not resume:
            return False
        prev_status = str(state.get("cells", {}).get(cell_key(profile, multiplier, policy), {}).get("status") or "")
        dest = get_results_root() / cell_jsonl_relpath(mt, profile, multiplier, policy)
        return prev_status in {"complete", "cost_cap_stop"} and dest.exists()

    pending = [cell for cell in cells_plan if not _is_done(*cell)]
    metrics_cells: list[dict[str, Any]] = []
    stopped = False
    stop_reason = None
    pending_index = 0
    try:
        for profile, multiplier, policy in cells_plan:
            key = cell_key(profile, multiplier, policy)
            relpath = cell_jsonl_relpath(mt, profile, multiplier, policy)
            dest = get_results_root() / relpath
            if _is_done(profile, multiplier, policy):
                rows = load_jsonl(dest, expected=mt)
                metrics_cells.append(
                    cell_metrics(
                        rows,
                        measurement_type=mt,
                        sla=sla,
                        capacity=capacity,
                        strong_model=strong_model,
                        profile=profile,
                        multiplier=multiplier,
                        policy=policy,
                    )
                )
                continue
            try:
                guard.check_ahead((len(pending) - pending_index) * n_requests)
            except CostCapExceeded as exc:
                stopped = True
                stop_reason = exc.as_dict()
                state.setdefault("cells", {})[key] = {
                    "status": "cost_cap_stop",
                    "jsonl": relpath,
                    "n_records": 0,
                }
                _save_state(state)
                break
            pending_index += 1
            if dest.exists():
                dest.unlink()
            state.setdefault("cells", {})[key] = {"status": "in_progress", "jsonl": relpath}
            _save_state(state)
            spec = arrivals_for_profile(
                profile,
                n_requests,
                multiplier=multiplier,
                seed=seed,
                timestamps_s=_pad_replay(replay, n_requests) if profile == "replay" else None,
                rates_1x=rates,
            )
            arrivals = spec["arrivals_s"]
            rows: list[dict[str, Any]] = []
            status = "complete"
            try:
                if mode == "simulated":
                    qids = list(catalog.query_ids)
                    rng = np.random.default_rng(int(spec["seed"]))
                    query_seq = [str(qids[int(rng.integers(0, len(qids)))]) for _ in range(n_requests)]
                    rows, _des = _run_simulated_cell(
                        sim_cfg=sim_cfg,
                        catalog=catalog,
                        profile=profile,
                        multiplier=multiplier,
                        policy=policy,
                        arrivals=arrivals,
                        query_seq=query_seq,
                        seed=int(spec["seed"]),
                        quality_of=quality_of,
                    )
                    write_jsonl(relpath, rows, clobber=True)
                    guard.add_planned(len(rows))
                    for row in rows:
                        guard.record(row.get("cost_usd"), consume_planned=True)
                    status = "complete"
                else:
                    endpoint = next(s["measured"] for s in POLICY_SPECS if s["name"] == policy)
                    result = _run_measured_cell(
                        client=client,
                        prompts=prompts,
                        n=n_requests,
                        profile=profile,
                        multiplier=multiplier,
                        policy_name=policy,
                        endpoint_policy=endpoint,
                        assignments=assignments,
                        arrivals=arrivals,
                        max_workers=capacity,
                        cost_guard=guard,
                        extra_base=extra_base,
                    )
                    rows = result["records"]
                    write_jsonl(relpath, rows, clobber=True)
                    status = "cost_cap_stop" if result.get("stopped_on_cost_cap") else "complete"
                    if result.get("stopped_on_cost_cap"):
                        stopped = True
                        stop_reason = result.get("cost_cap_exceeded")
            except CostCapExceeded as exc:
                stopped = True
                stop_reason = exc.as_dict()
                status = "cost_cap_stop"
                while guard.remaining_planned > 0:
                    try:
                        guard.record(None, consume_planned=True)
                    except CostCapExceeded:
                        if guard.remaining_planned <= 0:
                            break
            state["cells"][key] = {
                "status": status,
                "jsonl": relpath,
                "n_records": len(rows),
                "seed": int(spec["seed"]),
            }
            _save_state(state)
            if rows:
                metrics_cells.append(
                    cell_metrics(
                        rows,
                        measurement_type=mt,
                        sla=sla,
                        capacity=capacity,
                        strong_model=strong_model,
                        profile=profile,
                        multiplier=multiplier,
                        policy=policy,
                    )
                )
            if stopped:
                break
    finally:
        if httpd is not None:
            stop_server(httpd)

    analysis = analyze_cells(
        metrics_cells,
        measurement_type=mt,
        extra={
            "mode": mode,
            "paid_api": paid,
            "dry_run": bool(dry_run) if mode == "measured" else False,
            "n_requests": n_requests,
            "profiles": list(profiles),
            "multipliers_requested": [float(x) for x in multipliers],
            "multipliers_executed": safe,
            "stopped_on_cost_cap": stopped,
            "stop_reason": stop_reason,
            "cost": guard.snapshot(),
            "capacity": capacity,
            "seed": seed,
        },
    )
    write_result(f"{prefix}/analysis.json", analysis, clobber=True)
    write_result(
        f"{prefix}/run.json",
        {
            "measurement_type": mt,
            "mode": mode,
            "paid_api": paid,
            "n_cells": len(metrics_cells),
            "stopped_on_cost_cap": stopped,
            "git": git_snapshot(),
            "cost": guard.snapshot(),
        },
        clobber=True,
    )
    state["stopped"] = stopped
    state["cost"] = guard.snapshot()
    _save_state(state)
    return {
        "measurement_type": mt,
        "analysis": analysis,
        "n_cells": len(metrics_cells),
        "stopped_on_cost_cap": stopped,
        "prefix": prefix,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EcoLogic systems stress test. MEASURED and SIMULATED modes never mix.",
        epilog=(
            "Simulated: python3.13 -m woais_experiments.stress.stress_test --mode simulated\n"
            "Measured dry-run: python3.13 -m woais_experiments.stress.stress_test --mode measured\n"
            "Paid: add --allow-api --max-cost-usd <limit>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", required=True, choices=("measured", "simulated"))
    p.add_argument("--config", default=None)
    p.add_argument("--deployment-config", default=None)
    p.add_argument("--allow-api", action="store_true", help="Permit paid provider HTTP (measured only)")
    p.add_argument("--dry-run", action="store_true", help="Measured stub only (default if no --allow-api)")
    p.add_argument("--max-cost-usd", type=float, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--n-requests", type=int, default=None)
    p.add_argument("--profiles", default=None, help="Comma-separated profile names")
    p.add_argument("--multipliers", default=None, help="Comma-separated load multipliers")
    p.add_argument("--max-multiplier", type=float, default=None)
    p.add_argument("--base-url", default=None, help="Existing MEASURED endpoint; skip local server")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.allow_api and args.mode == "simulated":
        print("error: simulated stress tests never call APIs; omit --allow-api", file=sys.stderr)
        return 2
    if args.allow_api and args.dry_run:
        print("error: --allow-api and --dry-run are mutually exclusive", file=sys.stderr)
        return 2
    if args.allow_api and args.max_cost_usd is None:
        print("error: paid runs require --max-cost-usd", file=sys.stderr)
        return 2
    dry_run = bool(args.dry_run) or not args.allow_api
    cfg = load_stress_config(args.config)
    profiles = _profiles(cfg, _csv(args.profiles) or None)
    if args.multipliers:
        multipliers = [float(x) for x in _csv(args.multipliers)]
    else:
        multipliers = _multipliers(cfg, None)
    n = _n_requests(cfg, args.mode, args.n_requests)
    try:
        result = run_suite(
            cfg,
            mode=str(args.mode),
            allow_api=bool(args.allow_api),
            dry_run=dry_run,
            max_cost_usd=args.max_cost_usd,
            resume=bool(args.resume),
            profiles=profiles,
            multipliers=multipliers,
            n_requests=n,
            max_multiplier=args.max_multiplier,
            deployment_config=args.deployment_config,
            base_url=args.base_url,
        )
    except (ValueError, MixedSourceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except CostCapExceeded as exc:
        print(json.dumps(exc.as_dict(), indent=2))
        return 3
    print(json.dumps({
        "measurement_type": result["measurement_type"],
        "n_cells": result["n_cells"],
        "stopped_on_cost_cap": result["stopped_on_cost_cap"],
        "prefix": result["prefix"],
        "ecologic_pareto_efficient_all_tested_loads": result["analysis"].get(
            "ecologic_pareto_efficient_all_tested_loads"
        ),
        "max_sustainable_throughput": result["analysis"].get("max_sustainable_throughput"),
        "disclaimer": result["analysis"].get("disclaimer"),
    }, indent=2, default=str))
    return 3 if result["stopped_on_cost_cap"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

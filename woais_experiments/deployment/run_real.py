"""MEASURED_REAL_DEPLOYMENT benchmark. Does not provision Cloud Run or Lambda.

Paid execution requires ALL of: --allow-api --allow-cloud --max-cost-usd.
There is no default spend limit. This module never runs paid calls on import.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from woais_experiments.accounting.per_query_cost import load_models_config
from woais_experiments.deployment.analyze_real import (
    RESULTS_PREFIX,
    analyze_records,
    attach_file_provenance,
    write_analysis,
)
from woais_experiments.deployment.app import start_server, stop_server
from woais_experiments.deployment.backends import print_all_commands
from woais_experiments.deployment.budget import (
    abort_if_over_budget,
    check_approach,
    make_guard,
    worst_case_budget,
)
from woais_experiments.deployment.client import DeploymentClient
from woais_experiments.deployment.config import load_deployment_config
from woais_experiments.deployment.infer import build_context, handle_inference
from woais_experiments.deployment.lifecycle import (
    keep_lifecycle_if_same_process,
    platform_region,
)
from woais_experiments.deployment.query_set import (
    TARGET_N,
    as_prompts,
    executed_slice,
    select_query_set,
)
from woais_experiments.deployment.records import (
    RequestRecord,
    utc_timestamp,
    v2_measurement_type,
)
from woais_experiments.deployment.static_mix import cheap_strong_slugs, cost_matched_assignments
from woais_experiments.deployment.workloads import Workload, as_dicts, default_workloads
from woais_experiments.frozen import write_result
from woais_experiments.paths import public_relpath
from woais_experiments.runner.gitinfo import git_snapshot
from woais_experiments.stress.load_generator import CostCapExceeded

SETUPS = (
    ("A", "always_cheap", "direct_cheap"),
    ("B", "always_strong", "direct_strong"),
    ("C", "ecologic", "ecologic"),
    ("D", "cost_matched_static", "static_mixture"),
)

REAL_COMMAND = (
    "python run_woais.py deployment-real --allow-api --allow-cloud "
    "--max-cost-usd REPLACE_WITH_LIMIT --backend cloudrun "
    "--base-url https://YOUR_CLOUD_RUN_URL"
)

LOCAL_NOT_SERVERLESS = (
    "Local and loopback results are not labeled serverless. "
    "MEASURED_REAL_DEPLOYMENT serverless labels require Cloud Run or Lambda "
    "with --allow-api --allow-cloud --max-cost-usd and a non-loopback --base-url."
)


class SafetyError(ValueError):
    pass


def loopback_url(url: str | None) -> bool:
    if not url:
        return True
    host = (urlparse(url).hostname or "").lower()
    return host in {"", "localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"}


def pricing_config_hash(cfg: Mapping[str, Any]) -> str:
    blob = {
        "models_json": load_models_config(),
        "price_alias": cfg.get("price_alias"),
        "prices_usd_per_million": cfg.get("prices_usd_per_million"),
        "router_overhead_name": cfg.get("router_overhead_name"),
    }
    raw = json.dumps(blob, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def provenance(
    *,
    cfg: Mapping[str, Any],
    backend: str,
    serverless: bool,
    region: str | None,
    query_set_hash: str,
    generation_mode: str,
    extra: Mapping[str, Any] | None = None,
    measurement_type: str | None = None,
) -> dict[str, Any]:
    git = git_snapshot()
    models = cfg.get("models") or {}
    payload = {
        "measurement_type": measurement_type or v2_measurement_type(paid=False, serverless=False),
        "cloud_backend": backend if serverless else None,
        "region": region,
        "instance_configuration": {
            "source": "not_provisioned_by_this_tool",
            "backend": backend,
            "serverless": bool(serverless),
            "note": (
                "This runner does not deploy Cloud Run or Lambda and does not "
                "invent CPU/RAM. Recorded fields come from flags and platform env."
            ),
        },
        "timestamp": utc_timestamp(),
        "git_commit": git.get("commit"),
        "git": git,
        "model_identifiers": {
            str(k): {"name": (v or {}).get("name"), "provider": (v or {}).get("provider")}
            for k, v in models.items()
            if isinstance(v, Mapping)
        },
        "pricing_config_hash": pricing_config_hash(cfg),
        "query_set_hash": query_set_hash,
        "generation_mode": generation_mode,
        "energy": {
            "measured": False,
            "status": "not_measured",
            "note": "Estimated energy is not reported as measured energy.",
        },
    }
    if extra:
        payload.update(dict(extra))
    return payload


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
    raise RuntimeError(f"deployment server failed health check: {last}")


def _stamp_record(
    rec: RequestRecord,
    *,
    setup: str,
    setup_name: str,
    policy: str,
    workload: str,
    concurrency: int,
    backend: str,
    generation_mode: str,
    serverless: bool,
    prov: Mapping[str, Any],
    wall: float | None = None,
    measurement_type: str | None = None,
) -> dict[str, Any]:
    mt = str(
        measurement_type
        or prov.get("measurement_type")
        or rec.measurement_type
        or v2_measurement_type(paid=False, serverless=bool(serverless))
    )
    rec.setup = setup
    rec.setup_name = setup_name
    rec.policy = policy
    rec.workload = workload
    rec.concurrency = concurrency
    rec.backend = backend
    rec.generation_mode = generation_mode
    rec.serverless = bool(serverless)
    rec.measurement_type = mt
    rec.cloud_backend = backend if serverless else None
    if wall is not None:
        rec.extra["phase_wall_s"] = wall
    row = rec.to_dict()
    row["measurement_type"] = mt
    row["error"] = row.get("error") if "error" in row else row.get("error_type")
    if row.get("provider_cost") is None:
        row["provider_cost"] = row.get("realized_provider_cost")
    if row.get("provider_latency_ms") is None:
        row["provider_latency_ms"] = row.get("provider_request_ms")
    row["query_id"] = row.get("query_id") or row.get("prompt_id")
    row["energy_measured"] = False
    row["energy_status"] = "not_measured"
    row["energy_note"] = "not measured; estimated energy is not reported as measured energy"
    row["git_commit"] = prov.get("git_commit")
    row["pricing_config_hash"] = prov.get("pricing_config_hash")
    row["query_set_hash"] = prov.get("query_set_hash")
    row["run_timestamp"] = prov.get("timestamp")
    row["instance_configuration"] = prov.get("instance_configuration")
    row["cloud_backend"] = prov.get("cloud_backend")
    if row.get("region") is None:
        row["region"] = prov.get("region")
    return row


def _slice_prompts(prompts: Sequence[Mapping[str, Any]], n: int | None) -> list[Mapping[str, Any]]:
    items = list(prompts)
    if n is None:
        return items
    return items[: max(1, min(int(n), len(items)))]


def _run_http(
    client: DeploymentClient,
    prompts: Sequence[Mapping[str, Any]],
    *,
    setup: str,
    setup_name: str,
    policy: str,
    workload: Workload,
    assignments: Mapping[str, str] | None,
    extra_base: Mapping[str, Any],
    prov: Mapping[str, Any],
    guard,
    paid: bool,
) -> tuple[list[dict[str, Any]], float]:
    batch = _slice_prompts(prompts, workload.n_requests)

    def one(item: Mapping[str, Any]) -> RequestRecord:
        if paid:
            check_approach(guard, extra_requests=1)
        forced = None
        if policy == "static_mixture" and assignments is not None:
            forced = assignments[str(item["id"])]
        extra = dict(extra_base)
        extra.update({
            "setup": setup,
            "setup_name": setup_name,
            "concurrency": workload.concurrency,
            "prompt_id": item["id"],
            "query_id": item["id"],
            "workload": workload.name,
            "arrival_timestamp": utc_timestamp(),
            "measurement_type": extra_base.get("measurement_type"),
        })
        rec = client.infer(
            str(item["text"]),
            policy=policy,
            forced_model=forced,
            extra=extra,
        )
        if paid or guard.limit is not None:
            guard.record(rec.realized_provider_cost)
        return rec

    if paid:
        check_approach(guard, extra_requests=max(1, len(batch)))
    t0 = time.perf_counter()
    records: list[RequestRecord]
    conc = max(1, int(workload.concurrency))
    if conc <= 1:
        records = [one(p) for p in batch]
    else:
        records = []
        with ThreadPoolExecutor(max_workers=conc) as pool:
            futs = [pool.submit(one, p) for p in batch]
            for fut in as_completed(futs):
                records.append(fut.result())
        records.sort(key=lambda r: str(r.query_id or r.prompt_id or r.request_id))
    wall = time.perf_counter() - t0
    rows = [
        _stamp_record(
            rec,
            setup=setup,
            setup_name=setup_name,
            policy=policy,
            workload=workload.name,
            concurrency=conc,
            backend=str(extra_base.get("backend") or "local"),
            generation_mode=str(extra_base.get("generation_mode") or "dry_run"),
            serverless=bool(extra_base.get("serverless")),
            prov=prov,
            wall=wall,
            measurement_type=extra_base.get("measurement_type"),
        )
        for rec in records
    ]
    return rows, wall


def _run_inprocess(
    ctx,
    prompts: Sequence[Mapping[str, Any]],
    *,
    setup: str,
    setup_name: str,
    policy: str,
    workload: Workload,
    assignments: Mapping[str, str] | None,
    prov: Mapping[str, Any],
    guard,
    paid: bool,
) -> tuple[list[dict[str, Any]], float]:
    batch = _slice_prompts(prompts, workload.n_requests)

    def one(item: Mapping[str, Any]) -> RequestRecord:
        if paid:
            check_approach(guard, extra_requests=1)
        body: dict[str, Any] = {
            "prompt": item["text"],
            "policy": policy,
            "setup": setup,
            "setup_name": setup_name,
            "concurrency": workload.concurrency,
            "prompt_id": item["id"],
            "query_id": item["id"],
            "workload": workload.name,
            "arrival_timestamp": utc_timestamp(),
            "measurement_type": ctx.measurement_type,
        }
        if policy == "static_mixture" and assignments is not None:
            body["forced_model"] = assignments[str(item["id"])]
        _status, rec = handle_inference(body, ctx=ctx)
        if paid or guard.limit is not None:
            guard.record(rec.realized_provider_cost)
        return rec

    if paid:
        check_approach(guard, extra_requests=max(1, len(batch)))
    t0 = time.perf_counter()
    conc = max(1, int(workload.concurrency))
    if conc <= 1:
        records = [one(p) for p in batch]
    else:
        records = []
        with ThreadPoolExecutor(max_workers=conc) as pool:
            futs = [pool.submit(one, p) for p in batch]
            for fut in as_completed(futs):
                records.append(fut.result())
        records.sort(key=lambda r: str(r.query_id or r.prompt_id or r.request_id))
    wall = time.perf_counter() - t0
    rows = [
        _stamp_record(
            rec,
            setup=setup,
            setup_name=setup_name,
            policy=policy,
            workload=workload.name,
            concurrency=conc,
            backend=ctx.backend,
            generation_mode=ctx.generation_mode,
            serverless=ctx.serverless_labeled,
            prov=prov,
            wall=wall,
            measurement_type=ctx.measurement_type,
        )
        for rec in records
    ]
    return rows, wall


def enforce_safety(
    *,
    allow_api: bool,
    allow_cloud: bool,
    max_cost_usd: float | None,
    dry_run: bool,
    backend: str,
    base_url: str | None,
) -> tuple[bool, bool]:
    """Return (paid, serverless). Raises SafetyError if flags are incomplete."""
    if allow_api and dry_run:
        raise SafetyError("--allow-api and --dry-run are mutually exclusive")
    paid = bool(allow_api) and not bool(dry_run)
    if paid:
        missing = []
        if not allow_cloud:
            missing.append("--allow-cloud")
        if max_cost_usd is None:
            missing.append("--max-cost-usd")
        if missing:
            raise SafetyError(
                "REAL API execution requires all three flags: "
                "--allow-api --allow-cloud --max-cost-usd <value>. Missing: "
                + ", ".join(missing)
                + ". There is no default spend limit."
            )
        if float(max_cost_usd) <= 0:
            raise SafetyError("--max-cost-usd must be > 0; there is no default spend limit")
    if backend in {"docker", "cloudrun"} and not base_url:
        raise SafetyError(
            f"--base-url is required for backend={backend} "
            "(this tool does not provision containers or cloud services)"
        )
    if paid and backend in {"cloudrun", "lambda"} and not base_url:
        raise SafetyError(
            f"backend={backend} paid runs require --base-url of an already-deployed service; "
            "this tool does not provision Cloud Run or Lambda"
        )
    if backend == "lambda" and allow_cloud and not base_url:
        raise SafetyError(
            "in-process Lambda is not a serverless measurement; pass --base-url of a Function URL"
        )
    serverless = bool(
        paid
        and allow_cloud
        and backend in {"cloudrun", "lambda"}
        and base_url
        and not loopback_url(base_url)
    )
    return paid, serverless


def persist_preflight(
    *,
    query_set: Mapping[str, Any],
    budget: Mapping[str, Any],
    prefix: str,
    prov: Mapping[str, Any],
) -> dict[str, str]:
    """Write IDs, hashes, and the cost cap estimate BEFORE any inference."""
    qs = attach_file_provenance(query_set, prov)
    qs["frozen_before_benchmark"] = True
    bud = attach_file_provenance(budget, prov)
    return {
        "query_set_json": public_relpath(write_result(f"{prefix}/query_set.json", qs, clobber=True)),
        "budget_json": public_relpath(write_result(f"{prefix}/budget.json", bud, clobber=True)),
    }


def persist(result: Mapping[str, Any], *, prefix: str = RESULTS_PREFIX) -> dict[str, str]:
    prov = result["provenance"]
    mix = attach_file_provenance(result["mix"], prov)
    qs = attach_file_provenance(result["query_set"], prov)
    bud = attach_file_provenance(result["budget"], prov)
    run = attach_file_provenance({
        "generation_mode": result["generation_mode"],
        "serverless": result["serverless"],
        "paid": result["paid"],
        "n_records": len(result["records"]),
        "stopped": result.get("stopped"),
        "real_command": REAL_COMMAND,
        "git": git_snapshot(),
        "note": LOCAL_NOT_SERVERLESS,
    }, prov)
    paths = {
        "requests_jsonl": public_relpath(write_result(f"{prefix}/requests.jsonl", result["jsonl"], clobber=True)),
        "static_mix_json": public_relpath(write_result(f"{prefix}/static_mix.json", mix, clobber=True)),
        "query_set_json": public_relpath(write_result(f"{prefix}/query_set.json", qs, clobber=True)),
        "budget_json": public_relpath(write_result(f"{prefix}/budget.json", bud, clobber=True)),
        "provenance_json": public_relpath(write_result(f"{prefix}/provenance.json", prov, clobber=True)),
        "run_json": public_relpath(write_result(f"{prefix}/run.json", run, clobber=True)),
    }
    if result["records"]:
        paths.update(write_analysis(result["summary"], prefix=prefix, provenance=prov))
    else:
        paths["summary_json"] = public_relpath(
            write_result(
                f"{prefix}/summary.json",
                attach_file_provenance(result["summary"], prov),
                clobber=True,
            )
        )
    return paths


def run_real(
    cfg: Mapping[str, Any],
    *,
    allow_api: bool,
    allow_cloud: bool,
    max_cost_usd: float | None,
    dry_run: bool,
    backend: str,
    base_url: str | None,
    n_queries: int = TARGET_N,
    max_queries: int | None = None,
    include_concurrency_8: bool = True,
    idle_s: float | None = None,
    start_server_local: bool = True,
    prefix: str = RESULTS_PREFIX,
    write_preflight: bool = True,
) -> dict[str, Any]:
    paid, serverless = enforce_safety(
        allow_api=allow_api,
        allow_cloud=allow_cloud,
        max_cost_usd=max_cost_usd,
        dry_run=dry_run or not allow_api,
        backend=backend,
        base_url=base_url,
    )
    dry_run = not paid
    allow_api_eff = bool(paid)
    if paid and not serverless:
        print(f"warning: {LOCAL_NOT_SERVERLESS}", file=sys.stderr)

    query_set = select_query_set(n=int(n_queries), allow_small=int(n_queries) < 50)
    executed_items = executed_slice(query_set, max_queries=max_queries)
    prompts = as_prompts(executed_items)
    query_set["executed_ids"] = [p["id"] for p in prompts]
    query_set["executed_n"] = len(prompts)
    query_set["executed_hash"] = hashlib.sha256(
        "\n".join(query_set["executed_ids"]).encode("utf-8")
    ).hexdigest()

    wl_cfg = cfg.get("workloads") or {}
    if idle_s is not None:
        idle = float(idle_s)
    elif dry_run:
        idle = float(wl_cfg.get("idle_gap_s_dry_run", 0.05))
    else:
        idle = float(wl_cfg.get("idle_gap_s_real", 20.0))
    workloads = default_workloads(
        n_queries=len(prompts),
        include_concurrency_8=include_concurrency_8,
        idle_s=idle,
    )

    mt = v2_measurement_type(paid=paid, serverless=serverless)
    ctx = build_context(
        cfg,
        allow_api=allow_api_eff,
        dry_run=dry_run,
        backend=backend,
        allow_cloud=bool(allow_cloud) and backend in {"cloudrun", "lambda"} and not loopback_url(base_url),
        measurement_type=mt,
    )
    cheap_slug, strong_slug = cheap_strong_slugs(ctx.models, ctx.roles)
    stub_out = int((cfg.get("stub") or {}).get("output_tokens", 16))
    max_out = int((cfg.get("paid") or {}).get("max_tokens", 64)) if paid else stub_out
    mix = cost_matched_assignments(
        prompts,
        prices=ctx.prices,
        models=ctx.models,
        classify_fn=ctx.classify_fn,
        cheap_slug=cheap_slug,
        strong_slug=strong_slug,
        output_tokens=stub_out,
        seed=int((cfg.get("seeds") or {}).get("static_mix", 20260909)),
        chars_per_token=float((cfg.get("stub") or {}).get("chars_per_token", 4.0)),
    )
    mix["measurement_type"] = mt
    assignments: dict[str, str] = dict(mix["assignments"])

    n_planned = sum(len(_slice_prompts(prompts, w.n_requests)) for w in workloads) * len(SETUPS)
    estimate = worst_case_budget(
        prompts,
        prices=ctx.prices,
        strong_model=strong_slug,
        n_policies=len(SETUPS),
        n_workloads=len(workloads),
        max_output_tokens=max_out,
        max_retries=int(ctx.max_retries),
        chars_per_token=float((cfg.get("stub") or {}).get("chars_per_token", 4.0)),
    )
    estimate["n_planned_requests"] = n_planned
    estimate["max_cost_usd"] = max_cost_usd
    prov = provenance(
        cfg=cfg,
        backend=backend,
        serverless=serverless,
        region=platform_region(),
        query_set_hash=str(query_set["query_set_hash"]),
        generation_mode=ctx.generation_mode,
        measurement_type=mt,
        extra={
            "allow_api": allow_api_eff,
            "allow_cloud": bool(allow_cloud),
            "max_cost_usd": max_cost_usd,
            "dry_run": dry_run,
            "n_prompts_frozen": query_set["n"],
            "n_prompts_executed": len(prompts),
            "workloads": as_dicts(workloads),
        },
    )
    if write_preflight:
        persist_preflight(query_set=query_set, budget=estimate, prefix=prefix, prov=prov)
    if paid:
        abort_if_over_budget(estimate, float(max_cost_usd))
    unit_est = float(estimate["unit_worst_usd"])
    guard = make_guard(
        paid=paid,
        max_cost_usd=float(max_cost_usd) if paid else None,
        estimate_usd_per_query=unit_est,
        n_planned=n_planned,
    )

    httpd = None
    client: DeploymentClient | None = None
    url = base_url
    listen = cfg.get("listen") or {}
    host = str(listen.get("host") or "127.0.0.1")
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    port = int(listen.get("port") or 8765)
    use_inprocess = backend == "lambda" and not url
    if backend == "local" and start_server_local and not url:
        httpd, _thread = start_server(ctx, host=host, port=port)
        port = int(httpd.server_address[1])
        url = f"http://{host}:{port}"
    if url:
        timeout = float((cfg.get("client") or {}).get("timeout_s", 30.0))
        retries = int((cfg.get("retries") or {}).get("client_max_retries", 2))
        client = DeploymentClient(url, timeout_s=timeout, max_retries=retries)
        if httpd is not None:
            _wait_healthy(client)

    for spec in ctx.models.values():
        if isinstance(spec, dict) and spec.get("name"):
            ctx.prices.resolve(str(spec["name"]))

    extra_base = {
        "generation_mode": ctx.generation_mode,
        "backend": backend,
        "serverless": serverless,
        "measurement_type": mt,
        "cloud_backend": backend if serverless else None,
    }
    all_rows: list[dict[str, Any]] = []
    phases: list[dict[str, Any]] = []
    stopped: dict[str, Any] | None = None
    try:
        for setup, setup_name, policy in SETUPS:
            for wl in workloads:
                if wl.idle_s > 0:
                    time.sleep(float(wl.idle_s))
                    if backend == "local" and httpd is not None and wl.kind == "idle_gap":
                        # Restart the local HTTP listener after an idle gap so
                        # connection setup is observed. Same OS pid is not a
                        # platform cold start.
                        stop_server(httpd)
                        ctx.lifecycle = keep_lifecycle_if_same_process(ctx.lifecycle)
                        httpd, _thread = start_server(ctx, host=host, port=0)
                        port = int(httpd.server_address[1])
                        url = f"http://{host}:{port}"
                        client = DeploymentClient(
                            url,
                            timeout_s=float((cfg.get("client") or {}).get("timeout_s", 30.0)),
                            max_retries=int((cfg.get("retries") or {}).get("client_max_retries", 2)),
                        )
                        _wait_healthy(client)
                try:
                    if use_inprocess:
                        recs, wall = _run_inprocess(
                            ctx, prompts,
                            setup=setup, setup_name=setup_name, policy=policy,
                            workload=wl, assignments=assignments, prov=prov,
                            guard=guard, paid=paid,
                        )
                    else:
                        if client is None:
                            raise RuntimeError("HTTP client is not configured")
                        recs, wall = _run_http(
                            client, prompts,
                            setup=setup, setup_name=setup_name, policy=policy,
                            workload=wl, assignments=assignments,
                            extra_base=extra_base, prov=prov, guard=guard, paid=paid,
                        )
                except CostCapExceeded as exc:
                    stopped = exc.as_dict()
                    break
                all_rows.extend(recs)
                phases.append({
                    "measurement_type": mt,
                    "setup": setup,
                    "setup_name": setup_name,
                    "policy": policy,
                    "workload": wl.name,
                    "concurrency": wl.concurrency,
                    "n": len(recs),
                    "phase_wall_s": wall,
                    "idle_s_before": wl.idle_s,
                    "serverless": serverless,
                })
            if stopped:
                break
    finally:
        if httpd is not None:
            stop_server(httpd)

    n_boot = int((cfg.get("bootstrap") or {}).get("n_boot", 2000))
    seed = int((cfg.get("bootstrap") or {}).get("seed", 20260909))
    region = prov.get("region")
    if all_rows:
        region = all_rows[0].get("region") or region
    prov = provenance(
        cfg=cfg,
        backend=backend,
        serverless=serverless,
        region=region,
        query_set_hash=str(query_set["query_set_hash"]),
        generation_mode=ctx.generation_mode,
        measurement_type=mt,
        extra={
            "allow_api": allow_api_eff,
            "allow_cloud": bool(allow_cloud),
            "max_cost_usd": max_cost_usd,
            "dry_run": dry_run,
            "base_url": None if (not url or loopback_url(url) or not serverless) else url,
            "n_prompts_frozen": query_set["n"],
            "n_prompts_executed": len(prompts),
            "workloads": as_dicts(workloads),
            "stopped": stopped,
            "cost_guard": guard.snapshot(),
            "preflight_written_before_inference": bool(write_preflight),
        },
    )
    summary = analyze_records(
        all_rows,
        n_boot=n_boot,
        seed=seed,
        extra=prov,
        static_mix=mix,
    ) if all_rows else {
        "measurement_type": mt,
        "n_records": 0,
        "groups": [],
        "run": prov,
        "note": "no records (stopped before first success or empty run)",
        "energy": {"measured": False, "status": "not_measured"},
    }
    summary["phases"] = phases
    summary["static_mix"] = mix
    summary["query_set"] = {
        "n": query_set["n"],
        "query_set_hash": query_set["query_set_hash"],
        "executed_n": query_set["executed_n"],
        "executed_ids": query_set["executed_ids"],
        "frozen_before_benchmark": True,
    }
    summary["budget"] = estimate
    summary["provenance"] = prov
    summary["measurement_type"] = mt
    jsonl = "\n".join(json.dumps(r, default=str) for r in all_rows) + ("\n" if all_rows else "")
    return {
        "records": all_rows,
        "summary": summary,
        "mix": mix,
        "query_set": query_set,
        "jsonl": jsonl,
        "generation_mode": ctx.generation_mode,
        "serverless": serverless,
        "paid": paid,
        "budget": estimate,
        "provenance": prov,
        "stopped": stopped,
        "real_command": REAL_COMMAND,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "EcoLogic MEASURED_REAL_DEPLOYMENT benchmark. "
            "No paid APIs unless --allow-api --allow-cloud --max-cost-usd are all set. "
            "There is no default spend limit."
        ),
        epilog="This tool never provisions Cloud Run or Lambda.",
    )
    p.add_argument("--config", default=None)
    p.add_argument("--backend", default="local", choices=("local", "docker", "cloudrun", "lambda"))
    p.add_argument("--base-url", default=None)
    p.add_argument("--allow-api", action="store_true")
    p.add_argument("--allow-cloud", action="store_true")
    p.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help="Required for paid runs. No default; omitting this flag forbids spend.",
    )
    p.add_argument("--dry-run", action="store_true", help="Stub provider; default if safety flags are incomplete")
    p.add_argument("--n-queries", type=int, default=TARGET_N, help="Frozen subset size (official: 50–150)")
    p.add_argument("--max-queries", type=int, default=None, help="Execute only the first K frozen IDs")
    p.add_argument("--skip-concurrency-8", action="store_true")
    p.add_argument("--idle-s", type=float, default=None)
    p.add_argument("--print-commands", action="store_true")
    p.add_argument("--prefix", default=RESULTS_PREFIX)
    return p


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "print_commands", False):
        print(print_all_commands())
        print("\n# deployment-real paid command (this tool does not run it automatically):")
        print(REAL_COMMAND)
        print("# Required env (never commit values): TOGETHER_API_KEY OPENAI_API_KEY")
        return 0
    dry_run = bool(getattr(args, "dry_run", False)) or not bool(getattr(args, "allow_api", False))
    cfg = load_deployment_config(getattr(args, "config", None))
    try:
        result = run_real(
            cfg,
            allow_api=bool(getattr(args, "allow_api", False)),
            allow_cloud=bool(getattr(args, "allow_cloud", False)),
            max_cost_usd=getattr(args, "max_cost_usd", None),
            dry_run=dry_run,
            backend=str(getattr(args, "backend", None) or "local"),
            base_url=getattr(args, "base_url", None),
            n_queries=int(getattr(args, "n_queries", None) or TARGET_N),
            max_queries=getattr(args, "max_queries", None),
            include_concurrency_8=not bool(getattr(args, "skip_concurrency_8", False)),
            idle_s=getattr(args, "idle_s", None),
            prefix=str(getattr(args, "prefix", None) or RESULTS_PREFIX),
        )
    except (SafetyError, CostCapExceeded) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    paths = persist(result, prefix=str(getattr(args, "prefix", None) or RESULTS_PREFIX))
    print(json.dumps({
        "measurement_type": result["summary"].get("measurement_type") or result["provenance"].get("measurement_type"),
        "generation_mode": result["generation_mode"],
        "serverless": result["serverless"],
        "paid": result["paid"],
        "n_records": len(result["records"]),
        "query_set_hash": result["query_set"]["query_set_hash"],
        "query_set_n_frozen": result["query_set"]["n"],
        "paths": paths,
        "real_benchmark_command": REAL_COMMAND,
        "required_env": ["TOGETHER_API_KEY", "OPENAI_API_KEY"],
        "note": LOCAL_NOT_SERVERLESS,
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_cli(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

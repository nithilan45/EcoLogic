"""Run the EcoLogic deployment benchmark (setups A–D, concurrency sweep).

No paid API calls unless ``--allow-api``. Default is dry-run (stub provider).
Does not provision Docker, Cloud Run, or Lambda.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Mapping, Sequence

from woais_experiments.deployment.analyze_deployment import (
    RESULTS_PREFIX,
    analyze_records,
    write_analysis,
)
from woais_experiments.deployment.app import start_server, stop_server
from woais_experiments.deployment.backends import print_all_commands, require_base_url
from woais_experiments.deployment.client import DeploymentClient
from woais_experiments.deployment.config import load_deployment_config
from woais_experiments.deployment.infer import build_context, handle_inference
from woais_experiments.deployment.records import MEASUREMENT_TYPE, RequestRecord
from woais_experiments.deployment.static_mix import cheap_strong_slugs, cost_matched_assignments
from woais_experiments.frozen import write_result
from woais_experiments.paths import public_relpath
from woais_experiments.runner.gitinfo import git_snapshot

SETUPS = (
    ("A", "direct_cheap", "direct_cheap"),
    ("B", "direct_strong", "direct_strong"),
    ("C", "ecologic", "ecologic"),
    ("D", "cost_matched_static", "static_mixture"),
)


def _prompts(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = list(cfg.get("prompts") or [])
    if not items:
        raise SystemExit("deployment config has no prompts")
    out = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            out.append({"id": f"p{i}", "text": item})
        else:
            out.append({"id": str(item.get("id") or f"p{i}"), "text": str(item["text"])})
    return out


def _concurrencies(cfg: Mapping[str, Any], *, allow_api: bool, include_16: bool | None) -> list[int]:
    wl = cfg.get("workloads") or {}
    values = [int(x) for x in (wl.get("concurrency") or [1, 2, 4, 8, 16])]
    max_c = int(wl.get("max_concurrency") or 16)
    safe16 = bool(wl.get("concurrency_16_safe", True))
    skip16_paid = bool(wl.get("skip_concurrency_16_if_allow_api", True))
    out: list[int] = []
    for c in values:
        if c > max_c:
            continue
        if c >= 16:
            if include_16 is False:
                continue
            if include_16 is None and (not safe16 or (allow_api and skip16_paid)):
                continue
        out.append(c)
    return out or [1]


def _idle_gap_s(cfg: Mapping[str, Any]) -> float:
    return float((cfg.get("workloads") or {}).get("idle_gap_s", 2.0))


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


def _run_phase_http(
    client: DeploymentClient,
    prompts: Sequence[Mapping[str, Any]],
    *,
    setup: str,
    setup_name: str,
    policy: str,
    concurrency: int,
    assignments: Mapping[str, str] | None,
    extra_base: Mapping[str, Any],
) -> tuple[list[RequestRecord], float]:
    def one(item: Mapping[str, Any]) -> RequestRecord:
        forced = None
        if policy == "static_mixture" and assignments is not None:
            forced = assignments[str(item["id"])]
        extra = dict(extra_base)
        extra.update({
            "setup": setup,
            "setup_name": setup_name,
            "concurrency": concurrency,
            "prompt_id": item["id"],
        })
        return client.infer(
            str(item["text"]),
            policy=policy,
            forced_model=forced,
            extra=extra,
        )

    t0 = time.perf_counter()
    records: list[RequestRecord]
    if concurrency <= 1:
        records = [one(p) for p in prompts]
    else:
        records = []
        with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
            futs = [pool.submit(one, p) for p in prompts]
            for fut in as_completed(futs):
                records.append(fut.result())
        records.sort(key=lambda r: str(r.prompt_id or r.request_id))
    wall = time.perf_counter() - t0
    for rec in records:
        rec.extra["phase_wall_s"] = wall
        rec.concurrency = concurrency
        rec.setup = setup
        rec.setup_name = setup_name
    return records, wall


def _run_phase_lambda(
    ctx,
    prompts: Sequence[Mapping[str, Any]],
    *,
    setup: str,
    setup_name: str,
    policy: str,
    concurrency: int,
    assignments: Mapping[str, str] | None,
) -> tuple[list[RequestRecord], float]:
    def one(item: Mapping[str, Any]) -> RequestRecord:
        body: dict[str, Any] = {
            "prompt": item["text"],
            "policy": policy,
            "setup": setup,
            "setup_name": setup_name,
            "concurrency": concurrency,
            "prompt_id": item["id"],
        }
        if policy == "static_mixture" and assignments is not None:
            body["forced_model"] = assignments[str(item["id"])]
        _status, rec = handle_inference(body, ctx=ctx)
        rec.setup = setup
        rec.setup_name = setup_name
        rec.concurrency = concurrency
        rec.prompt_id = str(item["id"])
        rec.backend = "lambda"
        rec.extra["lambda_in_process"] = True
        rec.extra["handler"] = "handle_inference"
        return rec

    t0 = time.perf_counter()
    if concurrency <= 1:
        records = [one(p) for p in prompts]
    else:
        records = []
        with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
            futs = [pool.submit(one, p) for p in prompts]
            for fut in as_completed(futs):
                records.append(fut.result())
        records.sort(key=lambda r: str(r.prompt_id or r.request_id))
    wall = time.perf_counter() - t0
    for rec in records:
        rec.extra["phase_wall_s"] = wall
        rec.concurrency = concurrency
        rec.setup = setup
        rec.setup_name = setup_name
        rec.backend = "lambda"
    return records, wall


def _records_to_jsonl(records: Sequence[RequestRecord]) -> str:
    lines = []
    for rec in records:
        row = rec.to_dict()
        row["measurement_type"] = MEASUREMENT_TYPE
        if rec.extra.get("phase_wall_s") is not None:
            row["phase_wall_s"] = rec.extra["phase_wall_s"]
        lines.append(json.dumps(row, default=str))
    return "\n".join(lines) + "\n"


def run_benchmark(
    cfg: Mapping[str, Any],
    *,
    allow_api: bool,
    dry_run: bool,
    backend: str,
    base_url: str | None,
    include_16: bool | None = None,
    restart_between_phases: bool = False,
    start_server_local: bool = True,
) -> dict[str, Any]:
    if allow_api and dry_run:
        raise ValueError("--allow-api and --dry-run are mutually exclusive")
    dry_run = bool(dry_run or not allow_api)
    allow_api = bool(allow_api) and not dry_run
    prompts = _prompts(cfg)
    concs = _concurrencies(cfg, allow_api=allow_api, include_16=include_16)
    idle = _idle_gap_s(cfg)
    ctx = build_context(cfg, allow_api=allow_api, dry_run=dry_run, backend=backend)
    cheap_slug, strong_slug = cheap_strong_slugs(ctx.models, ctx.roles)
    stub_out = int((cfg.get("stub") or {}).get("output_tokens", 16))
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
    mix["measurement_type"] = MEASUREMENT_TYPE
    assignments: dict[str, str] = dict(mix["assignments"])

    httpd = None
    client: DeploymentClient | None = None
    url = base_url
    listen = cfg.get("listen") or {}
    host = str(listen.get("host") or "127.0.0.1")
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    port = int(listen.get("port") or 8765)

    use_lambda_inprocess = backend == "lambda" and not url
    if backend in {"docker", "cloudrun"}:
        require_base_url(backend, url)
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

    # Fail closed if a serving slug has no price (never invent a rate).
    for spec in ctx.models.values():
        if isinstance(spec, dict) and spec.get("name"):
            ctx.prices.resolve(str(spec["name"]))

    all_records: list[RequestRecord] = []
    phases: list[dict[str, Any]] = []
    first_phase = True
    try:
        for setup, setup_name, policy in SETUPS:
            for conc in concs:
                if not first_phase and idle > 0:
                    time.sleep(idle)
                first_phase = False
                extra_base = {
                    "generation_mode": ctx.generation_mode,
                    "backend": backend,
                }
                if use_lambda_inprocess:
                    recs, wall = _run_phase_lambda(
                        ctx, prompts,
                        setup=setup, setup_name=setup_name, policy=policy,
                        concurrency=conc, assignments=assignments,
                    )
                else:
                    if client is None:
                        raise RuntimeError("HTTP client is not configured")
                    recs, wall = _run_phase_http(
                        client, prompts,
                        setup=setup, setup_name=setup_name, policy=policy,
                        concurrency=conc, assignments=assignments,
                        extra_base=extra_base,
                    )
                for rec in recs:
                    rec.generation_mode = ctx.generation_mode
                    rec.backend = backend
                    rec.paid_api = bool(allow_api)
                all_records.extend(recs)
                phases.append({
                    "measurement_type": MEASUREMENT_TYPE,
                    "setup": setup,
                    "setup_name": setup_name,
                    "policy": policy,
                    "concurrency": conc,
                    "n": len(recs),
                    "phase_wall_s": wall,
                    "idle_gap_s_before": 0.0 if len(phases) == 0 else idle,
                    "restarted_process": False,
                })
                last_setup = setup == SETUPS[-1][0] and conc == concs[-1]
                if restart_between_phases and backend == "local" and httpd is not None and not last_setup:
                    stop_server(httpd)
                    from woais_experiments.deployment.lifecycle import ProcessLifecycle
                    ctx.lifecycle = ProcessLifecycle()
                    httpd, _thread = start_server(ctx, host=host, port=0)
                    port = int(httpd.server_address[1])
                    url = f"http://{host}:{port}"
                    client = DeploymentClient(
                        url,
                        timeout_s=float((cfg.get("client") or {}).get("timeout_s", 30.0)),
                        max_retries=int((cfg.get("retries") or {}).get("client_max_retries", 2)),
                    )
                    _wait_healthy(client)
                    phases[-1]["restarted_process"] = True
    finally:
        if httpd is not None:
            stop_server(httpd)

    rows = []
    for rec in all_records:
        row = rec.to_dict()
        row["measurement_type"] = MEASUREMENT_TYPE
        if rec.extra.get("phase_wall_s") is not None:
            row["phase_wall_s"] = rec.extra["phase_wall_s"]
        rows.append(row)

    n_boot = int((cfg.get("bootstrap") or {}).get("n_boot", 2000))
    seed = int((cfg.get("bootstrap") or {}).get("seed", 20260909))
    summary = analyze_records(
        rows,
        n_boot=n_boot,
        seed=seed,
        extra={
            "allow_api": allow_api,
            "dry_run": dry_run,
            "backend": backend,
            "base_url": url,
            "generation_mode": ctx.generation_mode,
            "n_prompts": len(prompts),
            "concurrencies": concs,
            "idle_gap_s": idle,
            "git": git_snapshot(),
        },
    )
    summary["phases"] = phases
    summary["static_mix"] = mix
    summary["measurement_type"] = MEASUREMENT_TYPE
    return {
        "records": rows,
        "summary": summary,
        "mix": mix,
        "jsonl": _records_to_jsonl(all_records),
        "generation_mode": ctx.generation_mode,
    }


def persist(result: Mapping[str, Any], *, prefix: str = RESULTS_PREFIX) -> dict[str, str]:
    paths = {
        "requests_jsonl": public_relpath(write_result(f"{prefix}/requests.jsonl", result["jsonl"])),
        "static_mix_json": public_relpath(write_result(f"{prefix}/static_mix.json", result["mix"])),
        "run_json": public_relpath(write_result(f"{prefix}/run.json", {
            "measurement_type": MEASUREMENT_TYPE,
            "generation_mode": result["generation_mode"],
            "n_records": len(result["records"]),
            "git": git_snapshot(),
        })),
    }
    paths.update(write_analysis(result["summary"], prefix=prefix))
    return paths


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EcoLogic real deployment benchmark (MEASURED). No paid APIs unless --allow-api.",
        epilog="This tool never provisions Cloud Run, Lambda, or Docker registries.",
    )
    p.add_argument("--config", default=None)
    p.add_argument("--backend", default="local", choices=("local", "docker", "cloudrun", "lambda"))
    p.add_argument("--base-url", default=None, help="Existing endpoint. Required for docker/cloudrun.")
    p.add_argument("--allow-api", action="store_true", help="Permit paid provider HTTP")
    p.add_argument("--dry-run", action="store_true", help="Validate + stub generation (default if no --allow-api)")
    p.add_argument("--include-concurrency-16", action="store_true")
    p.add_argument("--skip-concurrency-16", action="store_true")
    p.add_argument("--restart-between-phases", action="store_true", help="Local only: new process so the next phase starts cold")
    p.add_argument("--print-commands", action="store_true")
    p.add_argument("--prefix", default=RESULTS_PREFIX)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.print_commands:
        print(print_all_commands())
        return 0
    if args.allow_api and args.dry_run:
        print("error: --allow-api and --dry-run are mutually exclusive", file=sys.stderr)
        return 2
    dry_run = not args.allow_api
    include_16 = None
    if args.include_concurrency_16:
        include_16 = True
    if args.skip_concurrency_16:
        include_16 = False
    cfg = load_deployment_config(args.config)
    result = run_benchmark(
        cfg,
        allow_api=bool(args.allow_api),
        dry_run=dry_run,
        backend=str(args.backend),
        base_url=args.base_url,
        include_16=include_16,
        restart_between_phases=bool(args.restart_between_phases),
    )
    paths = persist(result, prefix=args.prefix)
    print(json.dumps({
        "measurement_type": MEASUREMENT_TYPE,
        "generation_mode": result["generation_mode"],
        "n_records": len(result["records"]),
        "paths": paths,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

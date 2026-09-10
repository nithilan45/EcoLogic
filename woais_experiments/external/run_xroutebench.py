"""Run the EcoLogic audit stack on xRouteBench execution panels.

Downloads the public Hugging Face dataset (cached locally). Does not call paid
model APIs. Interrupted runs resume from ``woais_experiments/results/xroutebench/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import deque
from dataclasses import dataclass, replace
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from woais_experiments.external.adapter import WidePanel, assignment_dict, to_wide
from woais_experiments.external.run_external_audit import (
    ORACLE_N_GRID_DEFAULT,
    _jsonable,
    bootstrap_router_vs_static,
    naive_vs_realized,
    pick_oracle_method,
    quality_only_audit,
    static_report,
)
from woais_experiments.external.schema import CANONICAL_COLUMNS, MAX_MODELS, MIN_MODELS, SchemaError
from woais_experiments.external.xroutebench import (
    DEFAULT_REPO,
    MIN_QUERIES,
    CandidatePrices,
    cache_dir,
    discover_catalog,
    filter_records_by_task,
    load_hf_split,
    load_price_table,
    routing_configs,
    routing_panel_from_records,
    task_name_counts,
)
from woais_experiments.frozen import write_result
from woais_experiments.latency.analyze_latency import DEFAULT_LEVEL, DEFAULT_SEED
from woais_experiments.paths import RESULTS, public_relpath
from woais_experiments.routing.mckp import MCKPInstance, unconstrained_oracle
from woais_experiments.routing.oracle import budget_sweep, sweep_to_csv
from woais_experiments.routing.static_baselines import evaluate_assignment_against_static

PER_QUERY_CSV_MAX_N = 4000
MAX_TASK_SUBSETS = 20
RESULT_SUBDIR_DEFAULT = "xroutebench"


def slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._-")
    return (s or "unnamed")[:120]


def combo_key(config: str, split: str, subset: str) -> str:
    return f"{slug(config)}__{slug(split)}__{slug(subset)}"


def pick_n_boot(n_queries: int) -> int:
    if n_queries <= 500:
        return 2000
    if n_queries <= 2000:
        return 400
    return 200


def relative_accounting_error(naive: float, realized: float) -> dict[str, Any]:
    payload = {
        "naive_aggregate_cost": naive,
        "exact_realized_cost": realized,
        "relative_accounting_error": None,
        "definition": "(naive_aggregate_cost - exact_realized_cost) / exact_realized_cost",
        "available": False,
    }
    if realized is None or not np.isfinite(realized) or realized == 0.0:
        payload["reason"] = "realized cost is zero or missing; relative error not defined"
        return payload
    payload["available"] = True
    payload["relative_accounting_error"] = float((naive - realized) / realized)
    return payload


def slim_catalog(report: Mapping[str, Any] | None) -> Any:
    if not report:
        return report
    out = dict(report)
    cat = out.get("catalog")
    if isinstance(cat, dict) and int(cat.get("n_models") or 0) > 12:
        cat = dict(cat)
        pairs = cat.get("pairwise_random") or []
        n_pair = len(pairs) if isinstance(pairs, list) else pairs
        cat["pairwise_random"] = f"omitted ({n_pair} pairs); m={cat['n_models']} > 12"
        out["catalog"] = cat
    return out


def combo_is_complete(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    if not payload.get("schema_version"):
        return False
    return payload.get("status") in {"ok", "skipped"}


def load_json_if_complete(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return blob if combo_is_complete(blob) else None


@dataclass(frozen=True)
class Job:
    config: str
    split: str
    subset: str = "all"

    @property
    def key(self) -> str:
        return combo_key(self.config, self.split, self.subset)

    @property
    def dataset(self) -> str:
        if self.subset == "all":
            return f"xroutebench/{self.config}/{self.split}"
        return f"xroutebench/{self.config}/{self.split}/{self.subset}"


def plan_base_jobs(catalog: Sequence[Mapping[str, Any]]) -> list[Job]:
    jobs: list[Job] = []
    for cfg in routing_configs(catalog):
        splits = list(cfg.get("splits") or ("train", "test"))
        for split in splits:
            if str(split).lower() not in {"train", "test"}:
                continue
            jobs.append(Job(config=str(cfg["name"]), split=str(split), subset="all"))
    return jobs


def subset_jobs_from_meta(meta: Mapping[str, Any], job: Job) -> list[Job]:
    counts = meta.get("task_name_query_counts") or {}
    names = sorted(
        k for k, n in counts.items() if k != "_missing_task_name" and int(n) >= MIN_QUERIES
    )
    if len(names) <= 1:
        return []
    if len(names) > MAX_TASK_SUBSETS:
        names = names[:MAX_TASK_SUBSETS]
    return [Job(config=job.config, split=job.split, subset=name) for name in names]


def _router_missing_payload() -> dict[str, Any]:
    return {
        "available": False,
        "status": "skipped",
        "reason": (
            "published xRouteBench execution tables have no router_score / "
            "router_assignment; a performance-argmax policy is the unconstrained "
            "oracle, not a learned router"
        ),
    }


def audit_execution_wide(
    wide: WidePanel,
    *,
    n_boot: int,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
    n_grid: int = ORACLE_N_GRID_DEFAULT,
    oracle_method: str | None = None,
) -> dict[str, Any]:
    if not (MIN_MODELS <= wide.m <= MAX_MODELS):
        raise SchemaError(f"model count {wide.m} is outside [{MIN_MODELS}, {MAX_MODELS}]")
    inst = MCKPInstance(wide.cost, wide.quality, names=wide.names, query_ids=wide.query_ids)
    method = oracle_method or pick_oracle_method(inst.n, inst.m)
    oracle = unconstrained_oracle(inst)
    oracle_assign = inst.named_assignment(oracle.choice)
    nv_oracle = naive_vs_realized(wide.cost, oracle.choice, wide.names)
    rel_oracle = relative_accounting_error(
        nv_oracle["naive_inference_mean"], nv_oracle["realized_inference_mean"]
    )
    cm_oracle = slim_catalog(
        evaluate_assignment_against_static(
            wide.names, wide.cost, wide.quality, oracle_assign, query_ids=wide.query_ids
        )
    )
    has_router = wide.assignment is not None
    if has_router:
        nv_router = naive_vs_realized(wide.cost, wide.assignment, wide.names)
        rel_router = relative_accounting_error(
            nv_router["naive_inference_mean"], nv_router["realized_inference_mean"]
        )
        cm_router = slim_catalog(
            evaluate_assignment_against_static(
                wide.names,
                wide.cost,
                wide.quality,
                assignment_dict(wide),
                query_ids=wide.query_ids,
            )
        )
        router_frontier = budget_sweep(
            inst, router_choice=wide.assignment, method=method, n_grid=n_grid
        )
        boot_router = bootstrap_router_vs_static(wide, n_boot=n_boot, seed=seed, level=level)
    else:
        nv_router = _router_missing_payload()
        rel_router = _router_missing_payload()
        cm_router = _router_missing_payload()
        router_frontier = _router_missing_payload()
        boot_router = _router_missing_payload()

    oracle_frontier = budget_sweep(inst, router_choice=None, method=method, n_grid=n_grid)
    oracle_frontier["method_selected"] = method
    wide_oracle = replace(wide, assignment=np.asarray(oracle.choice, dtype=int))
    boot_oracle = bootstrap_router_vs_static(wide_oracle, n_boot=n_boot, seed=seed, level=level)
    static = static_report(replace(wide, assignment=None))

    return {
        "schema_version": "1.0",
        "usd_available": True,
        "dataset": wide.dataset,
        "n_queries": wide.n,
        "n_models": wide.m,
        "models": list(wide.names),
        "canonical_columns": list(CANONICAL_COLUMNS),
        "notes": list(wide.notes),
        "has_router_assignment": has_router,
        "naive_aggregate_cost": rel_oracle.get("naive_aggregate_cost"),
        "exact_realized_cost": rel_oracle.get("exact_realized_cost"),
        "relative_accounting_error": rel_oracle.get("relative_accounting_error"),
        "accounting": {
            "policy": "unconstrained_oracle",
            "naive_vs_realized": nv_oracle,
            "relative_accounting_error": rel_oracle,
            "router_naive_vs_realized": nv_router,
            "router_relative_accounting_error": rel_router,
        },
        "cost_matched_static": {
            "oracle": cm_oracle,
            "router": cm_router,
        },
        "static_baselines": slim_catalog(static),
        "oracle_frontier": oracle_frontier,
        "router_frontier": router_frontier,
        "bootstrap": {
            "oracle_vs_static": boot_oracle,
            "router_vs_static": boot_router,
        },
    }


def audit_xroute_panel(
    panel,
    *,
    n_boot: int | None = None,
    seed: int = DEFAULT_SEED,
    n_grid: int = ORACLE_N_GRID_DEFAULT,
    require_cost: bool = True,
) -> dict[str, Any]:
    try:
        wide = to_wide(panel, require_cost=True, require_quality=True)
    except SchemaError:
        wide = to_wide(panel, require_cost=False, require_quality=True)
        payload = quality_only_audit(wide)
        payload["source"] = panel.source
        payload["long_n_rows"] = panel.n_rows
        payload["naive_aggregate_cost"] = None
        payload["exact_realized_cost"] = None
        payload["relative_accounting_error"] = None
        payload["cost_matched_static"] = {
            "oracle": {"available": False, "reason": "realized_cost missing"},
            "router": _router_missing_payload(),
        }
        payload["oracle_frontier"] = {"available": False, "reason": "realized_cost missing"}
        payload["router_frontier"] = _router_missing_payload()
        return payload
    boot = pick_n_boot(wide.n) if n_boot is None else int(n_boot)
    payload = audit_execution_wide(wide, n_boot=boot, seed=seed, n_grid=n_grid)
    payload["source"] = panel.source
    payload["long_n_rows"] = panel.n_rows
    payload["n_boot"] = boot
    return payload


def save_combo(payload: Mapping[str, Any], *, rel_stem: str) -> dict[str, str]:
    slim = dict(payload)
    per_query = slim.pop("per_query", None)
    written = {"json": public_relpath(write_result(f"{rel_stem}.json", _jsonable(slim)))}
    n = int(payload.get("n_queries") or 0)
    if per_query and n <= PER_QUERY_CSV_MAX_N:
        buf = StringIO()
        fields = list(per_query[0].keys())
        w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in per_query:
            w.writerow({k: "" if row.get(k) is None else row.get(k) for k in fields})
        written["per_query_csv"] = public_relpath(write_result(f"{rel_stem}_per_query.csv", buf.getvalue()))
    oracle = payload.get("oracle_frontier") or {}
    if isinstance(oracle, dict) and oracle.get("sweep"):
        written["oracle_csv"] = public_relpath(
            write_result(f"{rel_stem}_oracle.csv", sweep_to_csv(oracle))
        )
    return written


def _index_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    acc = payload.get("accounting") or {}
    rel = acc.get("relative_accounting_error") or {}
    cm = ((payload.get("cost_matched_static") or {}).get("oracle")) or {}
    ora = payload.get("oracle_frontier") or {}
    uo = ora.get("unconstrained_oracle") if isinstance(ora, dict) else None
    router = payload.get("router_frontier") or {}
    return {
        "key": payload.get("combo_key"),
        "config": payload.get("config"),
        "split": payload.get("split"),
        "subset": payload.get("subset"),
        "dataset_repo": payload.get("dataset_repo"),
        "status": payload.get("status"),
        "reason": payload.get("reason"),
        "n_queries": payload.get("n_queries"),
        "n_models": payload.get("n_models"),
        "usd_available": payload.get("usd_available"),
        "query_id_scheme": payload.get("query_id_scheme"),
        "naive_aggregate_cost": payload.get("naive_aggregate_cost"),
        "exact_realized_cost": payload.get("exact_realized_cost"),
        "relative_accounting_error": payload.get("relative_accounting_error")
        if payload.get("relative_accounting_error") is not None
        else rel.get("relative_accounting_error"),
        "oracle_mean_quality": None if not isinstance(uo, dict) else uo.get("mean_quality"),
        "static_quality_at_oracle_cost": cm.get("static_quality_at_same_cost") if isinstance(cm, dict) else None,
        "oracle_quality_advantage": cm.get("quality_advantage") if isinstance(cm, dict) else None,
        "router_frontier": "available"
        if isinstance(router, dict) and router.get("available") is not False and router.get("sweep")
        else "skipped",
        "resume_skipped": bool(payload.get("resume_skipped")),
    }


def write_index(rows: Sequence[Mapping[str, Any]], *, subdir: str) -> dict[str, str]:
    index = {
        "schema_version": "1.0",
        "dataset_repo": rows[0].get("dataset_repo") if rows else DEFAULT_REPO,
        "n_combos": len(rows),
        "paid_model_apis": False,
        "downloaded": True,
        "combos": list(rows),
    }
    buf = StringIO()
    fields = [
        "key",
        "config",
        "split",
        "subset",
        "status",
        "n_queries",
        "n_models",
        "usd_available",
        "naive_aggregate_cost",
        "exact_realized_cost",
        "relative_accounting_error",
        "oracle_mean_quality",
        "static_quality_at_oracle_cost",
        "oracle_quality_advantage",
        "router_frontier",
        "reason",
    ]
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for row in rows:
        w.writerow({k: "" if row.get(k) is None else row.get(k) for k in fields})
    return {
        "index_json": public_relpath(write_result(f"{subdir}/index.json", _jsonable(index))),
        "index_csv": public_relpath(write_result(f"{subdir}/index.csv", buf.getvalue())),
    }


LoadSplitFn = Callable[..., list[dict[str, Any]]]


def run_combo(
    job: Job,
    records: Sequence[Mapping[str, Any]],
    prices: Mapping[str, CandidatePrices],
    *,
    repo: str = DEFAULT_REPO,
    n_boot: int | None = None,
    seed: int = DEFAULT_SEED,
    n_grid: int = ORACLE_N_GRID_DEFAULT,
    min_queries: int = MIN_QUERIES,
) -> dict[str, Any]:
    source = f"huggingface:{repo}/{job.config}/{job.split}"
    subset_records = filter_records_by_task(records, job.subset)
    if not subset_records:
        return {
            "schema_version": "1.0",
            "status": "skipped",
            "reason": f"no rows for subset {job.subset!r}",
            "combo_key": job.key,
            "config": job.config,
            "split": job.split,
            "subset": job.subset,
            "dataset": job.dataset,
            "usd_available": False,
            "naive_aggregate_cost": None,
            "exact_realized_cost": None,
            "relative_accounting_error": None,
            "router_frontier": _router_missing_payload(),
            "oracle_frontier": {"available": False, "reason": "empty subset"},
            "cost_matched_static": {"oracle": {"available": False}, "router": _router_missing_payload()},
            "task_name_query_counts": {},
        }
    panel, info = routing_panel_from_records(
        subset_records,
        dataset=job.dataset,
        prices=prices,
        source=source,
        require_cost=True,
        min_queries=min_queries,
    )
    base = {
        "schema_version": "1.0",
        "combo_key": job.key,
        "config": job.config,
        "split": job.split,
        "subset": job.subset,
        "dataset": job.dataset,
        "dataset_repo": repo,
        "paid_model_apis": False,
        "downloaded": True,
        "query_id_scheme": (info.get("map") or {}).get("query_id_scheme"),
        "missing_fields": info.get("coverage"),
        "selection_notes": info.get("selection_notes"),
        "source_models": info.get("source_models"),
        "n_source_queries": info.get("n_source_queries"),
        "n_source_models": info.get("n_source_models"),
        "task_name_query_counts": info.get("task_name_query_counts") or task_name_counts(subset_records),
    }
    if panel is None:
        qpanel, qinfo = routing_panel_from_records(
            subset_records,
            dataset=job.dataset,
            prices=prices,
            source=source,
            require_cost=False,
            min_queries=min_queries,
        )
        if qpanel is None:
            return {
                **base,
                "status": "skipped",
                "reason": info.get("reason") or qinfo.get("reason"),
                "usd_available": False,
                "naive_aggregate_cost": None,
                "exact_realized_cost": None,
                "relative_accounting_error": None,
                "router_frontier": _router_missing_payload(),
                "oracle_frontier": {"available": False, "reason": info.get("reason")},
                "cost_matched_static": {
                    "oracle": {"available": False, "reason": info.get("reason")},
                    "router": _router_missing_payload(),
                },
            }
        payload = audit_xroute_panel(qpanel, n_boot=n_boot, seed=seed, n_grid=n_grid)
        payload.update(base)
        payload["status"] = "ok"
        payload["usd_available"] = False
        payload["reason"] = "quality-only: realized_cost incomplete; USD modules skipped"
        return payload
    payload = audit_xroute_panel(panel, n_boot=n_boot, seed=seed, n_grid=n_grid)
    payload.update(base)
    payload["status"] = "ok"
    return payload


def run_audit(
    *,
    repo: str = DEFAULT_REPO,
    cache: Path | str | None = None,
    result_subdir: str = RESULT_SUBDIR_DEFAULT,
    resume: bool = True,
    n_boot: int | None = None,
    seed: int = DEFAULT_SEED,
    n_grid: int = ORACLE_N_GRID_DEFAULT,
    catalog: Sequence[Mapping[str, Any]] | None = None,
    prices: Mapping[str, CandidatePrices] | None = None,
    load_split: LoadSplitFn | None = None,
    jobs: Sequence[Job] | None = None,
    min_queries: int = MIN_QUERIES,
    max_jobs: int | None = None,
) -> dict[str, Any]:
    dest = cache_dir(cache)
    cat = list(catalog) if catalog is not None else discover_catalog(repo, cache=dest)
    write_result(f"{result_subdir}/catalog.json", _jsonable({"repo": repo, "configs": cat}))
    price_meta: dict[str, Any]
    if prices is None:
        price_table, price_meta = load_price_table(
            repo, catalog=cat, cache=dest, load_split=load_split
        )
    else:
        price_table = dict(prices)
        price_meta = {
            "available": bool(price_table),
            "n_models": len(price_table),
            "models": sorted(price_table),
            "note": "injected price table",
        }
    write_result(f"{result_subdir}/prices.json", _jsonable(price_meta))

    combo_dir = RESULTS / result_subdir / "combos"
    combo_dir.mkdir(parents=True, exist_ok=True)
    meta_dir_rel = f"{result_subdir}/meta"

    queue: deque[Job] = deque(jobs if jobs is not None else plan_base_jobs(cat))
    # Restore subset jobs from previous meta so resume does not drop them.
    if jobs is None:
        extra: list[Job] = []
        for job in list(queue):
            meta_path = RESULTS / meta_dir_rel / f"{combo_key(job.config, job.split, 'all')}_tasks.json"
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                extra.extend(subset_jobs_from_meta(meta, job))
        for job in extra:
            queue.append(job)

    seen: set[str] = set()
    summaries: list[dict[str, Any]] = []
    n_run = 0
    loader = load_split or load_hf_split
    record_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    while queue:
        if max_jobs is not None and n_run >= max_jobs:
            break
        job = queue.popleft()
        if job.key in seen:
            continue
        seen.add(job.key)
        rel_stem = f"{result_subdir}/combos/{job.key}"
        existing = load_json_if_complete(RESULTS / f"{rel_stem}.json") if resume else None
        if existing is not None:
            print(f"[xroutebench] resume skip {job.key}", flush=True)
            existing = dict(existing)
            existing["resume_skipped"] = True
            summaries.append(_index_row(existing))
            if job.subset == "all":
                for sub in subset_jobs_from_meta(existing, job):
                    if sub.key not in seen:
                        queue.append(sub)
            continue

        cache_key = (job.config, job.split)
        if cache_key not in record_cache:
            print(f"[xroutebench] load {job.config}/{job.split}", flush=True)
            record_cache[cache_key] = loader(repo, job.config, job.split, cache=dest)
        records = record_cache[cache_key]
        print(f"[xroutebench] audit {job.key} n_rows={len(records)}", flush=True)
        try:
            payload = run_combo(
                job,
                records,
                price_table,
                repo=repo,
                n_boot=n_boot,
                seed=seed,
                n_grid=n_grid,
                min_queries=min_queries,
            )
        except SchemaError as exc:
            payload = {
                "schema_version": "1.0",
                "status": "skipped",
                "reason": str(exc),
                "combo_key": job.key,
                "config": job.config,
                "split": job.split,
                "subset": job.subset,
                "dataset": job.dataset,
                "dataset_repo": repo,
                "usd_available": False,
                "naive_aggregate_cost": None,
                "exact_realized_cost": None,
                "relative_accounting_error": None,
                "router_frontier": _router_missing_payload(),
                "oracle_frontier": {"available": False, "reason": str(exc)},
                "cost_matched_static": {
                    "oracle": {"available": False, "reason": str(exc)},
                    "router": _router_missing_payload(),
                },
                "paid_model_apis": False,
            }
        payload["saved"] = save_combo(payload, rel_stem=rel_stem)
        n_run += 1
        summaries.append(_index_row(payload))

        if job.subset == "all":
            meta = {
                "config": job.config,
                "split": job.split,
                "task_name_query_counts": payload.get("task_name_query_counts") or task_name_counts(records),
            }
            write_result(
                f"{meta_dir_rel}/{combo_key(job.config, job.split, 'all')}_tasks.json",
                _jsonable(meta),
            )
            for sub in subset_jobs_from_meta(meta, job):
                if sub.key not in seen:
                    queue.append(sub)

        state = {
            "schema_version": "1.0",
            "n_completed_files": len(summaries),
            "n_audited_this_process": n_run,
            "keys": [row.get("key") for row in summaries],
        }
        write_result(f"{result_subdir}/state.json", _jsonable(state))

    written = write_index(summaries, subdir=result_subdir)
    return {
        "schema_version": "1.0",
        "repo": repo,
        "cache_dir": str(dest),
        "result_subdir": result_subdir,
        "n_combos": len(summaries),
        "n_audited_this_process": n_run,
        "paid_model_apis": False,
        "saved": written,
        "combos": summaries,
        "price_meta": price_meta,
        "n_catalog_configs": len(cat),
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="xRouteBench EcoLogic audit (HF download, no model APIs)")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--result-subdir", default=RESULT_SUBDIR_DEFAULT)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--n-boot", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-grid", type=int, default=ORACLE_N_GRID_DEFAULT)
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = run_audit(
        repo=args.repo,
        cache=args.cache_dir,
        result_subdir=args.result_subdir,
        resume=not args.no_resume,
        n_boot=args.n_boot,
        seed=args.seed,
        n_grid=args.n_grid,
        max_jobs=args.max_jobs,
    )
    print(
        json.dumps(
            {
                "saved": payload.get("saved"),
                "n_combos": payload.get("n_combos"),
                "n_audited_this_process": payload.get("n_audited_this_process"),
            },
            indent=2,
        )
    )
    return payload


if __name__ == "__main__":
    main()

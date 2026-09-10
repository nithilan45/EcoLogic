"""Run the WOAIS audit stack on a canonical external routing panel.

Per-query accounting, naive vs realized cost, static baselines, cost-matching,
oracle / MCKP frontier, and bootstrap significance. Works for 2–50 models.
Does not download datasets or fit routers.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from woais_experiments.external.adapter import (
    LongPanel,
    WidePanel,
    assignment_dict,
    discover_sources,
    load_source,
    normalize_records,
    to_wide,
)
from woais_experiments.external.schema import CANONICAL_COLUMNS, MAX_MODELS, MIN_MODELS, SchemaError
from woais_experiments.frozen import write_result
from woais_experiments.paths import public_relpath
from woais_experiments.latency.analyze_latency import (
    DEFAULT_LEVEL,
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    percentile,
)
from woais_experiments.routing.cost_matching import compare_router_to_static, router_means
from woais_experiments.routing.mckp import MCKPInstance
from woais_experiments.routing.oracle import budget_sweep
from woais_experiments.routing.static_baselines import catalog, evaluate_assignment_against_static, market_from_panel
from woais_experiments.statistics.inference import wilson_dict
from woais_experiments.statistics.regret import population_cov

ORACLE_N_GRID_DEFAULT = 9


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
        if math.isnan(obj):
            return None
        return "Infinity" if obj > 0 else "-Infinity"
    return obj


def pick_oracle_method(n: int, m: int) -> str:
    if n <= 10 and m <= 6:
        return "auto"
    return "approx"


def naive_vs_realized(cost: np.ndarray, choice: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    """Mix × unconditional mean vs realized per-query cost. No prices required."""
    n, m = cost.shape
    means = cost.mean(axis=0)
    mix = np.bincount(choice, minlength=m).astype(float) / n
    naive = float(mix @ means)
    realized = float(cost[np.arange(n), choice].mean())
    cov_sum = 0.0
    per_model = {}
    for j, name in enumerate(names):
        ind = (choice == j).astype(float)
        cov = population_cov(ind, cost[:, j])
        cov_sum += cov
        routed = ind > 0
        cond = float(cost[routed, j].mean()) if routed.any() else None
        per_model[name] = {
            "fraction": float(mix[j]),
            "unconditional_mean": float(means[j]),
            "conditional_mean": cond,
            "cov_assignment_cost": cov,
        }
    residual = abs(realized - (naive + cov_sum))
    return {
        "n": n,
        "realized_inference_mean": realized,
        "naive_inference_mean": naive,
        "realized_minus_naive": realized - naive,
        "covariance_sum": cov_sum,
        "identity": "realized = naive + sum_m Cov(1{t=m}, c_m)",
        "abs_residual": residual,
        "reconciles": bool(residual <= 1e-12),
        "per_model": per_model,
        "mix_fractions": {names[j]: float(mix[j]) for j in range(m) if mix[j] > 0},
    }


def per_query_rows(wide: WidePanel) -> list[dict[str, Any]]:
    assign = wide.assignment
    rows = []
    for i, qid in enumerate(wide.query_ids):
        j = int(assign[i]) if assign is not None else None
        rows.append(
            {
                "query_id": qid,
                "dataset": wide.dataset,
                "selected_model": None if j is None else wide.names[j],
                "quality": None if j is None else float(wide.quality[i, j]),
                "realized_cost": None if j is None else float(wide.cost[i, j]),
                "input_tokens": None
                if j is None or not np.isfinite(wide.input_tokens[i, j])
                else int(wide.input_tokens[i, j]),
                "output_tokens": None
                if j is None or not np.isfinite(wide.output_tokens[i, j])
                else int(wide.output_tokens[i, j]),
                "latency_ms": None
                if j is None or not np.isfinite(wide.latency_ms[i, j])
                else float(wide.latency_ms[i, j]),
            }
        )
    return rows


def _ci_from_samples(
    samples: Sequence[float],
    point: float | None,
    *,
    n_boot: int,
    seed: int,
    level: float,
) -> dict[str, Any]:
    arr = np.asarray(list(samples), dtype=float)
    finite = arr[np.isfinite(arr)]
    payload = {
        "point": point,
        "lo": point,
        "hi": point,
        "n_boot": int(n_boot),
        "n_finite": int(finite.size),
        "level": float(level),
        "seed": int(seed),
    }
    if finite.size == 0 or n_boot <= 0:
        return payload
    lo_q = 100.0 * (1.0 - level) / 2.0
    hi_q = 100.0 * (1.0 + level) / 2.0
    payload["lo"] = percentile(finite, lo_q)
    payload["hi"] = percentile(finite, hi_q)
    return payload


def bootstrap_router_vs_static(
    wide: WidePanel,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> dict[str, Any]:
    """Resample queries; recompute hull + router vs cost-matched static."""
    if wide.assignment is None:
        return {"available": False, "reason": "no router_assignment"}
    names = wide.names
    assign_map = assignment_dict(wide)
    report0 = evaluate_assignment_against_static(
        names, wide.cost, wide.quality, assign_map, query_ids=wide.query_ids
    )
    qa0 = report0.get("quality_advantage")
    cs0 = report0.get("cost_savings")
    if n_boot <= 0 or wide.n <= 1:
        return {
            "available": True,
            "quality_advantage": _ci_from_samples([], qa0, n_boot=n_boot, seed=seed, level=level),
            "cost_savings": _ci_from_samples([], cs0, n_boot=n_boot, seed=seed, level=level),
            "bootstrap_share_quality_advantage_le_zero": None,
            "p_quality_advantage_le_zero": None,
            "effect_size": qa0,
            "significant_05_quality": None,
            "ci_is_not_a_hypothesis_test": True,
        }
    rng = np.random.default_rng(seed)
    qa_s: list[float] = []
    cs_s: list[float] = []
    n = wide.n
    for _ in range(int(n_boot)):
        idx = rng.choice(n, size=n, replace=True)
        qids = [wide.query_ids[i] for i in idx]
        sub_assign = {qids[k]: assign_map[wide.query_ids[idx[k]]] for k in range(n)}
        try:
            rq, rc = router_means(sub_assign, names, wide.cost[idx], wide.quality[idx], query_ids=qids)
            market = market_from_panel(names, wide.cost[idx], wide.quality[idx])
            cmp_ = compare_router_to_static(market, rq, rc)
        except (ValueError, KeyError):
            continue
        if cmp_.get("quality_advantage") is not None:
            qa_s.append(float(cmp_["quality_advantage"]))
        if cmp_.get("cost_savings") is not None:
            cs_s.append(float(cmp_["cost_savings"]))
    p_le0 = None
    if qa_s:
        p_le0 = float(sum(1 for x in qa_s if x <= 0.0) / len(qa_s))
    qa_ci = _ci_from_samples(qa_s, qa0, n_boot=n_boot, seed=seed, level=level)
    return {
        "available": True,
        "quality_advantage": qa_ci,
        "cost_savings": _ci_from_samples(cs_s, cs0, n_boot=n_boot, seed=seed, level=level),
        "bootstrap_share_quality_advantage_le_zero": p_le0,
        "p_quality_advantage_le_zero": p_le0,
        "effect_size": qa0,
        "significant_05_quality": None,
        "ci_is_not_a_hypothesis_test": True,
        "n_boot_requested": int(n_boot),
        "n_boot_quality": len(qa_s),
        "n_boot_cost": len(cs_s),
        "n_boot_dropped": int(n_boot) - len(qa_s),
        "definition": (
            "Percentile bootstrap over queries. Market means and the hull are "
            "recomputed on each resample. The CI is an interval estimate, not a "
            "significance test (see statistics.bootstrap.SIGNIFICANCE_NOTE). "
            "bootstrap_share_quality_advantage_le_zero is the share of *successful* "
            "resamples with advantage <= 0; failed resamples are excluded from "
            "that share (not a p-value)."
        ),
    }


def static_report(wide: WidePanel) -> dict[str, Any]:
    market = market_from_panel(wide.names, wide.cost, wide.quality)
    cat = catalog(market)
    if wide.m > 12:
        n_pair = len(cat.get("pairwise_random") or [])
        cat = dict(cat)
        cat["pairwise_random"] = f"omitted ({n_pair} pairs); m={wide.m} > 12"
    out: dict[str, Any] = {"catalog": cat, "n_models": wide.m, "models": list(wide.names)}
    if wide.assignment is not None:
        assign = assignment_dict(wide)
        out["router_vs_static"] = evaluate_assignment_against_static(
            wide.names, wide.cost, wide.quality, assign, query_ids=wide.query_ids
        )
        k = int(np.sum(wide.quality[np.arange(wide.n), wide.assignment] >= 0.5))
        # Wilson on mean quality only if quality is 0/1-like
        qsel = wide.quality[np.arange(wide.n), wide.assignment]
        if np.all(np.isin(np.round(qsel, 12), (0.0, 1.0))):
            out["router_wilson"] = wilson_dict(int(np.round(qsel.sum())), wide.n)
        else:
            out["router_mean_quality"] = float(qsel.mean())
            _ = k
    return out


def oracle_report(
    wide: WidePanel,
    *,
    n_grid: int = ORACLE_N_GRID_DEFAULT,
    method: str | None = None,
) -> dict[str, Any]:
    inst = MCKPInstance(wide.cost, wide.quality, names=wide.names, query_ids=wide.query_ids)
    method = method or pick_oracle_method(inst.n, inst.m)
    choice = wide.assignment
    payload = budget_sweep(inst, router_choice=choice, method=method, n_grid=n_grid)
    payload["method_selected"] = method
    return payload


def audit_wide(
    wide: WidePanel,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
    n_grid: int = ORACLE_N_GRID_DEFAULT,
    oracle_method: str | None = None,
) -> dict[str, Any]:
    if not (MIN_MODELS <= wide.m <= MAX_MODELS):
        raise SchemaError(f"model count {wide.m} is outside [{MIN_MODELS}, {MAX_MODELS}]")
    acc: dict[str, Any] = {"per_query": per_query_rows(wide)}
    if wide.assignment is not None:
        acc["naive_vs_realized"] = naive_vs_realized(wide.cost, wide.assignment, wide.names)
    else:
        acc["naive_vs_realized"] = {"available": False, "reason": "no router_assignment"}
    static = static_report(wide)
    oracle = oracle_report(wide, n_grid=n_grid, method=oracle_method)
    boot = bootstrap_router_vs_static(wide, n_boot=n_boot, seed=seed, level=level)
    return {
        "schema_version": "1.0",
        "dataset": wide.dataset,
        "n_queries": wide.n,
        "n_models": wide.m,
        "models": list(wide.names),
        "canonical_columns": list(CANONICAL_COLUMNS),
        "notes": list(wide.notes),
        "has_router_assignment": wide.assignment is not None,
        "accounting": {
            "per_query_n": len(acc["per_query"]),
            "naive_vs_realized": acc["naive_vs_realized"],
        },
        "static_baselines": static,
        "cost_matching": static.get("router_vs_static"),
        "oracle_frontier": oracle,
        "bootstrap": boot,
        "per_query": acc["per_query"],
    }


def quality_only_audit(wide: WidePanel) -> dict[str, Any]:
    """When costs are missing: model mean quality only. No hull / oracle / USD."""
    means = {wide.names[j]: float(np.nanmean(wide.quality[:, j])) for j in range(wide.m)}
    router = None
    if wide.assignment is not None:
        qsel = wide.quality[np.arange(wide.n), wide.assignment]
        router = {"mean_quality": float(qsel.mean()), "n": wide.n}
    return {
        "schema_version": "1.0",
        "dataset": wide.dataset,
        "usd_available": False,
        "n_queries": wide.n,
        "n_models": wide.m,
        "models": list(wide.names),
        "canonical_columns": list(CANONICAL_COLUMNS),
        "notes": list(wide.notes),
        "has_router_assignment": wide.assignment is not None,
        "mean_quality_by_model": means,
        "router": router,
        "skipped": [
            "naive accounting (needs realized_cost)",
            "cost-matching (needs realized_cost)",
            "oracle frontier (needs realized_cost)",
            "bootstrap vs cost-matched static",
        ],
        "reason": "realized_cost missing for at least one model; quality-only report",
    }


def audit_panel(
    panel: LongPanel,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    require_cost: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        wide = to_wide(panel, require_cost=True, require_quality=True)
    except SchemaError:
        wide = to_wide(panel, require_cost=False, require_quality=True)
        payload = quality_only_audit(wide)
        payload["source"] = panel.source
        payload["long_n_rows"] = panel.n_rows
        return payload
    payload = audit_wide(wide, n_boot=n_boot, seed=seed, **kwargs)
    payload["usd_available"] = True
    payload["source"] = panel.source
    payload["long_n_rows"] = panel.n_rows
    return payload


def audit_summary_source(blob: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "kind": "summary_only",
        "dataset": blob.get("name", "summary"),
        "panel_available": False,
        "n_models": blob.get("n_models"),
        "models": blob.get("models"),
        "modules_run": [],
        "skipped": [
            "per-query accounting",
            "naive accounting",
            "static baselines (hull)",
            "cost-matching",
            "oracle frontier",
            "bootstrap",
        ],
        "reason": blob.get("note"),
        "committed": blob.get("summary"),
        "release_audit": blob.get("generalization"),
    }


def save_audit(payload: Mapping[str, Any], *, relpath: str) -> dict[str, str]:
    json_path = write_result(f"{relpath}.json", _jsonable(dict(payload)))
    written = {"json": public_relpath(json_path)}
    long_rows = payload.get("per_query")
    if long_rows:
        import csv
        from io import StringIO

        buf = StringIO()
        fields = list(long_rows[0].keys())
        w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in long_rows:
            w.writerow({k: "" if row.get(k) is None else row.get(k) for k in fields})
        written["per_query_csv"] = public_relpath(write_result(f"{relpath}_per_query.csv", buf.getvalue()))
    oracle = payload.get("oracle_frontier") or {}
    if oracle.get("sweep"):
        from woais_experiments.routing.oracle import sweep_to_csv

        written["oracle_csv"] = public_relpath(
            write_result(f"{relpath}_oracle.csv", sweep_to_csv(oracle))
        )
    return written


def run_discovered(
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    n_grid: int = ORACLE_N_GRID_DEFAULT,
    relpath: str = "external/audit/index",
) -> dict[str, Any]:
    discovered = discover_sources()
    audits: dict[str, Any] = {}
    saved: dict[str, Any] = {}
    for src in discovered:
        if not src.available:
            audits[src.name] = {
                "available": False,
                "kind": src.kind,
                "note": src.note,
                "path": src.path,
            }
            continue
        loaded = load_source(src.name)
        if isinstance(loaded, LongPanel):
            payload = audit_panel(loaded, n_boot=n_boot, seed=seed, n_grid=n_grid)
            payload["available"] = True
            payload["kind"] = "panel"
            stem = f"external/audit/{src.name}"
            payload["saved"] = save_audit(payload, relpath=stem)
            saved[src.name] = payload["saved"]
            # drop bulky per-query from the index blob
            slim = dict(payload)
            slim.pop("per_query", None)
            audits[src.name] = slim
        else:
            payload = audit_summary_source(loaded)
            payload["available"] = True
            payload["kind"] = "summary"
            stem = f"external/audit/{src.name}"
            payload["saved"] = save_audit(payload, relpath=stem)
            saved[src.name] = payload["saved"]
            audits[src.name] = payload
    index = {
        "schema_version": "1.0",
        "downloaded": False,
        "min_models": MIN_MODELS,
        "max_models": MAX_MODELS,
        "n_boot": n_boot,
        "seed": seed,
        "discovered": [src.__dict__ for src in discovered],
        "audits": audits,
        "saved": saved,
        "note": (
            "No remote fetches. RouteLLM per-query GSM8K is audited only if a local "
            "CSV already exists; otherwise the committed Stage 9 summary is recorded."
        ),
    }
    index["saved_index"] = public_relpath(write_result(f"{relpath}.json", _jsonable(index)))
    return index


def run_records(
    records: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    **kwargs: Any,
) -> dict[str, Any]:
    panel = normalize_records(records, dataset=dataset, source="caller")
    return audit_panel(panel, **kwargs)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="External routing benchmark audit (no downloads)")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-grid", type=int, default=ORACLE_N_GRID_DEFAULT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = run_discovered(n_boot=args.n_boot, seed=args.seed, n_grid=args.n_grid)
    print(json.dumps(_jsonable({"saved": payload.get("saved"), "discovered": payload.get("discovered")}), indent=2))
    return payload


if __name__ == "__main__":
    main()

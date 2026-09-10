"""Evaluate a frozen threshold grid on an external router. No post-hoc retuning."""

from __future__ import annotations

import argparse
import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, Sequence

from woais_experiments.external_routing.audit_router import (
    classify_threshold,
    evaluate_threshold,
    query_rows_for_threshold,
)
from woais_experiments.external_routing.panel import ExternalRouterPanel, load_generic_csv
from woais_experiments.external_routing.routellm_adapter import (
    historical_stage9_pointer,
    inventory,
    load_routellm_panel,
)
from woais_experiments.frozen import write_result
from woais_experiments.paths import PACKAGE
from woais_experiments.statistics.bootstrap import DEFAULT_N_BOOT, DEFAULT_SEED
from woais_experiments.statistics.paired_tests import DEFAULT_N_PERM

GRID_PATH = PACKAGE / "external_routing" / "threshold_grid.json"
RESULTS_PREFIX = "external_router"
THRESHOLD_CSV = f"{RESULTS_PREFIX}/threshold_results.csv"
QUERY_CSV = f"{RESULTS_PREFIX}/query_level.csv"
ACCOUNTING_FLIPS = f"{RESULTS_PREFIX}/accounting_flips.csv"
RANKING_FLIPS = f"{RESULTS_PREFIX}/ranking_flips.csv"
SUMMARY = f"{RESULTS_PREFIX}/summary.json"
GRID_COPY = f"{RESULTS_PREFIX}/threshold_grid.json"

THRESHOLD_FIELDS = (
    "threshold",
    "assignment_kind",
    "n",
    "routing_rate",
    "quality",
    "naive_cost",
    "realized_cost",
    "absolute_accounting_error",
    "relative_accounting_error",
    "quality_advantage_vs_cost_matched_static",
    "quality_advantage_vs_cost_matched_static_naive_cost",
    "cost_savings_at_matched_quality",
    "oracle_gap",
    "cost_regret_naive_vs_oracle",
    "cost_regret_realized_vs_oracle",
    "mean_effect",
    "ci_lo",
    "ci_hi",
    "p_raw",
    "effect_size_cohens_dz",
    "effect_size_cliffs_delta",
    "bootstrap_unit",
    "n_boot",
    "n_perm",
    "always_cheap_quality",
    "always_strong_quality",
    "random_mixture_quality",
    "cost_matched_static_quality",
    "oracle_quality",
)

FLIP_FIELDS = (
    "threshold",
    "accounting_sign_flip",
    "ranking_flip",
    "naive_wins_exact_loses",
    "negligible_accounting_difference",
    "router_rank_naive",
    "router_rank_realized",
    "naive_advantage",
    "realized_advantage",
    "absolute_accounting_error",
    "relative_accounting_error",
    "rank_naive_cost",
    "rank_realized_cost",
)

QUERY_FIELDS = (
    "query_id",
    "threshold",
    "router_score",
    "router_assignment",
    "assignment_kind",
    "cheap_model",
    "strong_model",
    "cheap_quality",
    "strong_quality",
    "cheap_input_tokens",
    "cheap_output_tokens",
    "strong_input_tokens",
    "strong_output_tokens",
    "selected_model",
    "selected_quality",
    "selected_realized_cost",
)


def load_grid(path: Path = GRID_PATH) -> dict[str, Any]:
    grid = json.loads(path.read_text(encoding="utf-8"))
    if not grid.get("frozen_before_quality_cost"):
        raise ValueError("threshold grid is not marked frozen_before_quality_cost")
    if not grid.get("thresholds"):
        raise ValueError("empty threshold grid")
    return grid


def _csv(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    buf = StringIO()
    w = csv.DictWriter(buf, fieldnames=list(fields), extrasaction="ignore")
    w.writeheader()
    for row in rows:
        out = {}
        for k in fields:
            v = row.get(k)
            if isinstance(v, (list, tuple)):
                v = "|".join(str(x) for x in v)
            elif isinstance(v, bool):
                v = int(v)
            out[k] = "" if v is None else v
        w.writerow(out)
    return buf.getvalue()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return _jsonable(obj.item())
        except Exception:
            return str(obj)
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def run_sweep(
    panel: ExternalRouterPanel,
    grid: Mapping[str, Any],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = DEFAULT_SEED,
    write: bool = True,
) -> dict[str, Any]:
    # Score range is recorded before any quality/cost aggregation. Grid is not changed.
    score_range = panel.score_range()
    frozen = dict(grid)
    frozen["score_range_observed"] = score_range
    frozen["score_range_note"] = (
        "Observed after scoring, before quality/cost. The linspace grid was not retuned."
    )
    if write:
        write_result(GRID_COPY, frozen, clobber=True)

    rows = []
    query_rows: list[dict[str, Any]] = []
    for tau in grid["thresholds"]:
        rec = evaluate_threshold(panel, float(tau), n_boot=n_boot, n_perm=n_perm, seed=seed)
        rows.append(rec)
        query_rows.extend(query_rows_for_threshold(panel, float(tau)))

    flags = [classify_threshold(r) for r in rows]
    ranking_flips = [f for f in flags if f["ranking_flip"]]
    negligible = [f for f in flags if f["negligible_accounting_difference"]]

    # Representative operating point is the predefined midpoint 0.5, not a cherry-pick.
    mid = min(rows, key=lambda r: abs(float(r["threshold"]) - 0.5))
    n_pos = sum(
        1
        for r in rows
        if r.get("quality_advantage_vs_cost_matched_static") is not None
        and float(r["quality_advantage_vs_cost_matched_static"]) > 0
    )
    summary = {
        "protocol": "external_router_v1",
        "primary_question": (
            "Does exact per-query accounting materially alter the apparent efficiency "
            "of an independently developed routing policy?"
        ),
        "secondary_question": (
            "Does that router beat the best query-independent policy at the same realized cost?"
        ),
        "router_name": panel.router_name,
        "router_kind": "EXTERNAL_LEARNED_ROUTER",
        "oracle_kind": "ORACLE_ASSIGNMENT",
        "dataset": panel.dataset,
        "n": panel.n,
        "cheap_model": panel.cheap_model,
        "strong_model": panel.strong_model,
        "score_provenance": panel.score_provenance,
        "notes": panel.notes,
        "source": panel.source,
        "threshold_grid": frozen,
        "n_thresholds": len(rows),
        "n_accounting_sign_flips": sum(1 for f in flags if f["accounting_sign_flip"]),
        "n_ranking_flips": len(ranking_flips),
        "n_naive_wins_exact_loses": sum(1 for f in flags if f["naive_wins_exact_loses"]),
        "n_negligible_accounting_difference": len(negligible),
        "n_thresholds_positive_realized_advantage": n_pos,
        "representative_threshold_0.5": {
            "threshold": mid["threshold"],
            "quality": mid["quality"],
            "naive_cost": mid["naive_cost"],
            "realized_cost": mid["realized_cost"],
            "absolute_accounting_error": mid["absolute_accounting_error"],
            "quality_advantage_vs_cost_matched_static": mid["quality_advantage_vs_cost_matched_static"],
            "mean_effect": mid["mean_effect"],
            "ci_lo": mid["ci_lo"],
            "ci_hi": mid["ci_hi"],
            "p_raw": mid["p_raw"],
            "effect_size_cohens_dz": mid["effect_size_cohens_dz"],
        },
        "claim_accounting_material": bool(
            sum(1 for f in flags if f["accounting_sign_flip"] or f["naive_wins_exact_loses"] or f["ranking_flip"])
            and not all(f["negligible_accounting_difference"] for f in flags)
        ),
        "claim_beats_cost_matched_static": _beats_claim(rows),
        "inventory": inventory(),
        "historical_stage9_aggregates": historical_stage9_pointer(),
        "null_results_preserved": True,
        "flag_tables_include_all_thresholds": True,
    }
    if write:
        write_result(THRESHOLD_CSV, _csv(rows, THRESHOLD_FIELDS), clobber=True)
        write_result(QUERY_CSV, _csv(query_rows, QUERY_FIELDS), clobber=True)
        write_result(ACCOUNTING_FLIPS, _csv(flags, FLIP_FIELDS), clobber=True)
        write_result(RANKING_FLIPS, _csv(flags, FLIP_FIELDS), clobber=True)
        write_result(SUMMARY, _jsonable(summary), clobber=True)
        summary["written"] = {
            "threshold_results.csv": THRESHOLD_CSV,
            "query_level.csv": QUERY_CSV,
            "accounting_flips.csv": ACCOUNTING_FLIPS,
            "ranking_flips.csv": RANKING_FLIPS,
            "summary.json": SUMMARY,
            "threshold_grid.json": GRID_COPY,
        }
    summary["rows"] = rows
    summary["flags"] = flags
    return summary


def _beats_claim(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Do not claim a win unless the predefined midpoint CI is entirely above 0."""
    mid = min(rows, key=lambda r: abs(float(r["threshold"]) - 0.5))
    lo, p = mid.get("ci_lo"), mid.get("p_raw")
    adv = mid.get("quality_advantage_vs_cost_matched_static")
    win = lo is not None and float(lo) > 0 and p is not None and float(p) < 0.05
    return {
        "threshold": mid["threshold"],
        "supported": bool(win),
        "quality_advantage": adv,
        "ci_lo": lo,
        "ci_hi": mid.get("ci_hi"),
        "p_raw": p,
        "statement": (
            "held-out-style evidence supports a quality advantage vs cost-matched static"
            if win
            else "does not support a claim that the external router beats cost-matched static"
        ),
    }


def write_incomplete_artifacts(reason: str, *, gsm8k_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Write the required result files without inventing scores, tokens, or routes."""
    grid = load_grid()
    write_result(GRID_COPY, dict(grid), clobber=True)
    write_result(THRESHOLD_CSV, _csv([], THRESHOLD_FIELDS), clobber=True)
    write_result(ACCOUNTING_FLIPS, _csv([], FLIP_FIELDS), clobber=True)
    write_result(RANKING_FLIPS, _csv([], FLIP_FIELDS), clobber=True)
    query_rows = []
    if gsm8k_rows:
        from woais_experiments.external_routing.routellm_adapter import STRONG, WEAK

        for r in gsm8k_rows:
            query_rows.append({
                "query_id": r["query_id"],
                "threshold": "",
                "router_score": "",
                "router_assignment": "",
                "assignment_kind": "",
                "cheap_model": WEAK,
                "strong_model": STRONG,
                "cheap_quality": r["cheap_correct"],
                "strong_quality": r["strong_correct"],
                "cheap_input_tokens": "",
                "cheap_output_tokens": "",
                "strong_input_tokens": "",
                "strong_output_tokens": "",
                "selected_model": "",
                "selected_quality": "",
                "selected_realized_cost": "",
            })
    write_result(QUERY_CSV, _csv(query_rows, QUERY_FIELDS), clobber=True)
    payload = {
        "available": False,
        "query_level_assignments_available": False,
        "reason": reason,
        "n_queries_with_released_correctness": None if gsm8k_rows is None else len(gsm8k_rows),
        "fields_present_from_release": [
            "query/prompt",
            "cheap_quality",
            "strong_quality",
            "response_text",
        ],
        "fields_missing": [
            "router_score (needs local BERT checkpoint forward pass)",
            "token counts (not published; need tiktoken + Mixtral tokenizer)",
            "realized_cost",
            "router_assignment",
        ],
        "threshold_grid_frozen": grid,
        "router_kind": "EXTERNAL_LEARNED_ROUTER",
        "oracle_kind": "ORACLE_ASSIGNMENT",
        "note": (
            "Refusing to invent RouteLLM scores, token counts, costs, or assignments. "
            "Committed Stage 9 JSON is aggregate-only and is not written into "
            "threshold_results.csv as if it were this query-level sweep."
        ),
        "inventory": inventory(),
        "historical_stage9_aggregates": historical_stage9_pointer(),
        "primary_question": (
            "Does exact per-query accounting materially alter the apparent efficiency "
            "of an independently developed routing policy?"
        ),
        "secondary_question": (
            "Does that router beat the best query-independent policy at the same realized cost?"
        ),
        "primary_answer": "unanswered — query-level scores and token counts are not on disk",
        "secondary_answer": "unanswered — no external-router assignment could be reconstructed without scores",
        "null_results_preserved": True,
        "written": {
            "threshold_results.csv": THRESHOLD_CSV,
            "query_level.csv": QUERY_CSV,
            "accounting_flips.csv": ACCOUNTING_FLIPS,
            "ranking_flips.csv": RANKING_FLIPS,
            "summary.json": SUMMARY,
            "threshold_grid.json": GRID_COPY,
        },
    }
    write_result(SUMMARY, _jsonable(payload), clobber=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="External router threshold sweep (RouteLLM by default).")
    p.add_argument("--download", action="store_true", help="Fetch public RouteLLM GSM8K eval files")
    p.add_argument("--panel-csv", default=None, help="Generic external-router CSV instead of RouteLLM")
    p.add_argument("--router-name", default="generic_external_router")
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--inventory-only", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.inventory_only:
        print(json.dumps(inventory(), indent=2))
        return 0
    grid = load_grid()
    if args.panel_csv:
        panel = load_generic_csv(args.panel_csv, router_name=str(args.router_name))
    else:
        gsm8k_rows = None
        try:
            if args.download:
                from woais_experiments.external_routing.routellm_adapter import download_gsm8k

                download_gsm8k()
            from woais_experiments.external_routing.routellm_adapter import load_gsm8k_rows

            try:
                gsm8k_rows = load_gsm8k_rows()
            except FileNotFoundError:
                gsm8k_rows = None
            panel = load_routellm_panel(download=False)
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            payload = write_incomplete_artifacts(str(exc), gsm8k_rows=gsm8k_rows)
            print(json.dumps(_jsonable(payload), indent=2))
            return 1
    summary = run_sweep(
        panel, grid, n_boot=int(args.n_boot), n_perm=int(args.n_perm), seed=int(args.seed)
    )
    print(f"TRAIN/VAL not used; EXTERNAL n={summary['n']}")
    print(f"thresholds\t{summary['n_thresholds']}")
    print(f"accounting_sign_flips\t{summary['n_accounting_sign_flips']}")
    print(f"ranking_flips\t{summary['n_ranking_flips']}")
    print(f"naive_wins_exact_loses\t{summary['n_naive_wins_exact_loses']}")
    print(f"claim_accounting_material\t{summary['claim_accounting_material']}")
    print(f"claim_beats_static\t{summary['claim_beats_cost_matched_static']['statement']}")
    mid = summary["representative_threshold_0.5"]
    print(
        "midpoint_0.5\t"
        f"quality={mid['quality']:.6g}\t"
        f"naive={mid['naive_cost']:.6g}\t"
        f"realized={mid['realized_cost']:.6g}\t"
        f"adv={mid['quality_advantage_vs_cost_matched_static']}"
    )
    print(f"written\t{summary.get('written')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

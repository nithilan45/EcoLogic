"""Regenerate headline aggregates from raw query-level files. Does not overwrite them."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from woais_experiments.accounting.costs import cost_fn_usd
from woais_experiments.external.routellm import summarize_s9
from woais_experiments.frozen import sha256_file
from woais_experiments.paths import CONFIGS, RESULTS, ROOT, repo_rel
from woais_experiments.routing.policies import (
    accuracy_optimal_static_mixture,
    build_stage12_policies,
    evaluate_policies,
    regret_for_policy,
    two_model_matched_cost,
)

TOL = 1e-8
SCRIPT = "woais_experiments/reproducibility/reproduce_tables.py"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hash(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return sha256_file(path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _close(a: Any, b: Any, *, tol: float = TOL) -> bool:
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        if a is None or b is None:
            return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            # stored tables may have extra provenance keys; require all recomputed keys
            return all(k in b and _close(a[k], b[k], tol=tol) for k in a)
        return all(_close(a[k], b[k], tol=tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_close(x, y, tol=tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) is bool(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, bool) or isinstance(b, bool):
            return a == b
        return abs(float(a) - float(b)) <= tol
    return a == b


def _record(
    *,
    result_name: str,
    raw_inputs: Sequence[str],
    script_used: str,
    config_used: str | None,
    stored: Path,
    traced: bool,
    matches: bool | None,
    detail: str,
) -> dict[str, Any]:
    hashes = {}
    for rel in raw_inputs:
        p = ROOT / rel if not Path(rel).is_absolute() else Path(rel)
        hashes[rel] = _hash(p)
    return {
        "result_name": result_name,
        "raw_inputs": list(raw_inputs),
        "input_hashes": hashes,
        "script_used": script_used,
        "config_used": config_used,
        "output_hash": _hash(stored) if stored.exists() else None,
        "timestamp": _utc(),
        "traced_to_raw": traced,
        "matches_recomputed": matches,
        "detail": detail,
        "stored_path": repo_rel(stored) if stored.exists() else None,
    }


def _recompute_accounting() -> dict[str, Any]:
    from woais_experiments.frozen import load_stage12_matrix, load_stage12_routing

    matrix = load_stage12_matrix()
    routing = load_stage12_routing()
    exp = json.loads((CONFIGS / "experiment.json").read_text())
    policies = build_stage12_policies(matrix, routing, seed=int(exp.get("random_policy_seed", 20260905)))
    usd = cost_fn_usd(matrix)
    usd_table = evaluate_policies(
        matrix, policies, usd, vs="ecologic",
        cost_kind="usd", cost_unit="USD",
        cost_provenance="stored_call_usd",
    )
    eco = usd_table["policies"]["ecologic"]["cost"]
    t2 = usd_table["policies"]["always_t2"]["cost"]
    anchors = exp["published_stage12"]
    return {
        "n": matrix.n,
        "ecologic_usd": eco,
        "always_t2_usd": t2,
        "cost_ratio_ecologic_over_t2": eco / t2,
        "ecologic_correct": usd_table["policies"]["ecologic"]["correct"],
        "always_t2_correct": usd_table["policies"]["always_t2"]["correct"],
        "matches_published_usd": abs(eco - anchors["ecologic_usd"]) < 1e-8,
        "always_t2_dominates_ecologic": (
            usd_table["policies"]["always_t2"]["accuracy"]
            > usd_table["policies"]["ecologic"]["accuracy"]
            and t2 < eco
        ),
        "axis_usd_policies": {
            name: {
                "cost": usd_table["policies"][name]["cost"],
                "correct": usd_table["policies"][name]["correct"],
                "accuracy": usd_table["policies"][name]["accuracy"],
            }
            for name in ("ecologic", "always_t1", "always_t2", "frontier", "random")
        },
    }


def _recompute_static(accounting: Mapping[str, Any]) -> dict[str, Any]:
    from woais_experiments.frozen import load_stage12_matrix, load_stage12_routing

    matrix = load_stage12_matrix()
    routing = load_stage12_routing()
    exp = json.loads((CONFIGS / "experiment.json").read_text())
    policies = build_stage12_policies(matrix, routing, seed=int(exp.get("random_policy_seed", 20260905)))
    usd = cost_fn_usd(matrix)
    means_usd = {t: sum(matrix.usd[(t, i)] for i in matrix.item_ids) / matrix.n for t in (1, 2, 3)}
    accs = {}
    for t in (1, 2, 3):
        k = sum(matrix.correct[(t, i)] for i in matrix.item_ids)
        accs[t] = k / matrix.n
    eco_cost = accounting["axis_usd_policies"]["ecologic"]["cost"] / matrix.n
    opt = accuracy_optimal_static_mixture(means_usd, accs, eco_cost)
    mix = two_model_matched_cost(eco_cost, means_usd[2], means_usd[3], accs[2], accs[3])
    regret = regret_for_policy(matrix, policies["ecologic"], policies["oracle_usd"], usd)
    return {
        "accuracy_optimal_static_at_ecologic_budget": opt,
        "matched_cost_mix_t2_t3": mix,
        "regret_ecologic_vs_oracle_usd": {
            "R_true": regret["R_true"],
            "reconciles": regret["reconciles"],
        },
    }


def _trace_accounting(stored: Path) -> dict[str, Any]:
    raw = [
        "raw_results/graded.jsonl",
        "raw_results/routing.json",
        "woais_experiments/configs/experiment.json",
        "woais_experiments/configs/models.json",
    ]
    if not stored.exists():
        return _record(
            result_name="accounting/stage12.json",
            raw_inputs=raw,
            script_used=SCRIPT,
            config_used="woais_experiments/configs/experiment.json",
            stored=stored,
            traced=False,
            matches=None,
            detail="stored aggregate missing; cannot trace a headline table",
        )
    recomputed = _recompute_accounting()
    blob = _load_json(stored)
    stored_repro = blob.get("reproduction_vs_published") or {}
    stored_axis = ((blob.get("axis_usd") or {}).get("policies") or {})
    pairs = [
        (recomputed["n"], stored_repro.get("n")),
        (recomputed["ecologic_usd"], stored_repro.get("ecologic_usd")),
        (recomputed["always_t2_usd"], stored_repro.get("always_t2_usd")),
        (recomputed["cost_ratio_ecologic_over_t2"], stored_repro.get("cost_ratio_ecologic_over_t2")),
        (recomputed["ecologic_correct"], stored_repro.get("ecologic_correct")),
        (recomputed["always_t2_correct"], stored_repro.get("always_t2_correct")),
        (recomputed["always_t2_dominates_ecologic"], stored_repro.get("always_t2_dominates_ecologic")),
    ]
    for name in ("ecologic", "always_t1", "always_t2", "frontier", "random"):
        got = recomputed["axis_usd_policies"][name]
        exp = stored_axis.get(name) or {}
        pairs.extend([
            (got["cost"], exp.get("cost")),
            (got["correct"], exp.get("correct")),
            (got["accuracy"], exp.get("accuracy")),
        ])
    ok = all(_close(a, b) for a, b in pairs)
    return _record(
        result_name="accounting/stage12.json",
        raw_inputs=raw,
        script_used=SCRIPT,
        config_used="woais_experiments/configs/experiment.json",
        stored=stored,
        traced=True,
        matches=ok,
        detail="ok" if ok else "recomputed Stage 1–2 USD table disagrees with stored aggregate",
    )


def _trace_static(stored: Path) -> dict[str, Any]:
    raw = [
        "raw_results/graded.jsonl",
        "raw_results/routing.json",
        "woais_experiments/configs/experiment.json",
        "woais_experiments/results/accounting/stage12.json",
    ]
    if not stored.exists():
        return _record(
            result_name="routing/stage12_policies.json",
            raw_inputs=raw,
            script_used=SCRIPT,
            config_used="woais_experiments/configs/experiment.json",
            stored=stored,
            traced=False,
            matches=None,
            detail="stored static-baseline table missing",
        )
    recomputed_acc = _recompute_accounting()
    recomputed = _recompute_static(recomputed_acc)
    blob = _load_json(stored)
    opt = recomputed["accuracy_optimal_static_at_ecologic_budget"]
    stored_opt = blob.get("accuracy_optimal_static_at_ecologic_budget") or {}
    mix = recomputed["matched_cost_mix_t2_t3"]
    stored_mix = blob.get("matched_cost_mix_t2_t3") or {}
    regret = recomputed["regret_ecologic_vs_oracle_usd"]
    stored_regret = blob.get("regret_ecologic_vs_oracle_usd") or {}
    pairs = [
        (opt.get("accuracy"), stored_opt.get("accuracy")),
        (opt.get("cost"), stored_opt.get("cost")),
        (opt.get("kind"), stored_opt.get("kind")),
        (opt.get("feasible"), stored_opt.get("feasible")),
        (mix.get("f_strong"), stored_mix.get("f_strong")),
        (mix.get("accuracy"), stored_mix.get("accuracy")),
        (mix.get("expected_cost"), stored_mix.get("expected_cost")),
        (regret.get("R_true"), stored_regret.get("R_true")),
        (regret.get("reconciles"), stored_regret.get("reconciles")),
    ]
    ok = all(_close(a, b) for a, b in pairs)
    return _record(
        result_name="routing/stage12_policies.json",
        raw_inputs=raw,
        script_used=SCRIPT,
        config_used="woais_experiments/configs/experiment.json",
        stored=stored,
        traced=True,
        matches=ok,
        detail="ok" if ok else "recomputed static baselines disagree with stored table",
    )


def _trace_routellm(stored: Path) -> dict[str, Any]:
    raw = ["stage7_10/s9_static_baselines.json"]
    if not stored.exists():
        return _record(
            result_name="external/routellm_tables.json",
            raw_inputs=raw,
            script_used=SCRIPT,
            config_used="stage7_10/s9_static_baselines.json",
            stored=stored,
            traced=False,
            matches=None,
            detail="stored RouteLLM table missing",
        )
    s9 = summarize_s9()
    blob = _load_json(stored)
    stored_s9 = blob.get("s9_static_baselines") or {}
    keys = (
        "n_items",
        "mean_edge_matched_cost_pp",
        "n_interior",
        "published_mean_edge_matched_cost_pp",
    )
    recomputed = {k: s9.get(k) for k in keys}
    headline = {k: stored_s9.get(k) for k in keys}
    ok = _close(recomputed, headline)
    return _record(
        result_name="external/routellm_tables.json",
        raw_inputs=raw,
        script_used=SCRIPT,
        config_used="stage7_10/s9_static_baselines.json",
        stored=stored,
        traced=True,
        matches=ok,
        detail="ok" if ok else "recomputed RouteLLM s9 summary disagrees with stored table",
    )


def _trace_per_query_aggregate(per_query: Path, aggregate: Path, name: str) -> dict[str, Any]:
    raw = [repo_rel(per_query)]
    if not per_query.exists() or not aggregate.exists():
        return _record(
            result_name=name,
            raw_inputs=raw,
            script_used=SCRIPT,
            config_used=None,
            stored=aggregate,
            traced=False,
            matches=None,
            detail="per-query or aggregate file missing; headline cannot be traced",
        )
    rows = _load_json(per_query)
    if not isinstance(rows, list):
        return _record(
            result_name=name,
            raw_inputs=raw,
            script_used=SCRIPT,
            config_used=None,
            stored=aggregate,
            traced=False,
            matches=False,
            detail="per-query file is not a list of query records",
        )
    total = math.fsum(float(r["realized_cost"]) for r in rows)
    blob = _load_json(aggregate)
    stored_total = blob.get("realized_total")
    ok = stored_total is not None and abs(float(stored_total) - total) <= TOL
    n_ok = int(blob.get("n") or 0) == len(rows)
    return _record(
        result_name=name,
        raw_inputs=raw + [repo_rel(aggregate)],
        script_used=SCRIPT,
        config_used=None,
        stored=aggregate,
        traced=True,
        matches=bool(ok and n_ok),
        detail="ok" if ok and n_ok else "aggregate realized_total/n does not match per-query sum",
    )


def reconstruct(*, results_root: Path | None = None) -> dict[str, Any]:
    """Compare stored headline tables to recomputation. Does not write those tables."""
    root = Path(results_root) if results_root is not None else RESULTS
    records = []
    records.append(_trace_accounting(root / "accounting" / "stage12.json"))
    records.append(_trace_static(root / "routing" / "stage12_policies.json"))
    records.append(_trace_routellm(root / "external" / "routellm_tables.json"))
    records.append(
        _trace_per_query_aggregate(
            root / "accounting" / "framework" / "stage12_ecologic_per_query.json",
            root / "accounting" / "framework" / "stage12_ecologic_aggregate.json",
            "accounting/framework/stage12_ecologic_aggregate.json",
        )
    )
    records.append(
        _trace_per_query_aggregate(
            root / "accounting" / "framework" / "stage12_always_t2_per_query.json",
            root / "accounting" / "framework" / "stage12_always_t2_aggregate.json",
            "accounting/framework/stage12_always_t2_aggregate.json",
        )
    )
    failed = [r for r in records if not r.get("traced_to_raw") or r.get("matches_recomputed") is False]
    return {
        "ok": len(failed) == 0,
        "n_tables": len(records),
        "n_failed": len(failed),
        "records": records,
        "note": (
            "Stored experimental tables were not modified. "
            "A table fails if it cannot be traced to raw query-level files "
            "or if recomputation disagrees beyond tolerance."
        ),
    }


def smoke_external_public() -> dict[str, Any]:
    """RouteLLM s9 JSON is committed public data — no download, no API."""
    path = ROOT / "stage7_10" / "s9_static_baselines.json"
    if not path.exists():
        return {"ok": False, "available": False, "detail": "stage7_10/s9_static_baselines.json missing"}
    s9 = summarize_s9()
    return {
        "ok": True,
        "available": True,
        "n_items": s9.get("n_items"),
        "mean_edge_matched_cost_pp": s9.get("mean_edge_matched_cost_pp"),
        "source": s9.get("source"),
        "detail": "re-tabulated committed RouteLLM GSM8K table; no download",
    }

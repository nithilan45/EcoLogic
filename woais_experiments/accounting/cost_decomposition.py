"""Naive vs realized cost: covariance decomposition.

Let c_m(i) be the inference cost of query i on model m, t(i) the routed model,
f_m = P(t = m), and μ_m = E[c_m] (unconditional, over the full query set).

    naive     = Σ_m f_m μ_m
    realized  = E[c_{t(X)}(X)] = Σ_m f_m E[c_m | t = m]

The accounting error is exactly the assignment–cost covariance:

    realized − naive = Σ_m Cov(1{t = m}, c_m)
                     = Σ_m f_m (E[c_m | t = m] − μ_m)

Two ingredients are necessary for a nonzero error:

1. Query-dependent routing: 1{t = m} varies with i.
2. Within-model cost heterogeneity: Var(c_m) > 0 so that E[c_m | t = m] can
   differ from μ_m.

If either is absent the identity still holds and both sides are zero.
Router overhead is added after this inference-cost identity; a constant
per-query overhead does not change the difference.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from woais_experiments.accounting.aggregate_cost import (
    CostPanel,
    aggregate,
    records_from_assignment,
    unconditional_model_means,
)
from woais_experiments.accounting.per_query_cost import (
    PER_QUERY_CSV_FIELDS,
    PriceTable,
    QueryCostRecord,
    RouterOverhead,
    SCHEMA_VERSION,
)
from woais_experiments.paths import RESULTS, assert_inside_results
from woais_experiments.statistics.regret import population_cov


def _models_in_play(
    records: Sequence[QueryCostRecord],
    panel: CostPanel | None,
) -> list[str]:
    seen: list[str] = []
    if panel is not None:
        for m in panel.models:
            if m not in seen:
                seen.append(m)
    for r in records:
        if r.selected_model not in seen:
            seen.append(r.selected_model)
    return seen


def decompose_naive_vs_realized(
    records: Sequence[QueryCostRecord],
    panel: CostPanel,
    prices: PriceTable,
    *,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Numerically verify realized = naive + Σ_m Cov(1{t=m}, c_m)."""
    n = len(records)
    if n == 0:
        raise ValueError("no queries")
    selected = {r.query_id: r.selected_model for r in records}
    if set(selected) != set(panel.query_ids):
        raise ValueError("assignment query ids must match the cost panel")

    models = _models_in_play(records, panel)
    means = unconditional_model_means(panel, prices, models)
    agg = aggregate(records, panel=panel, prices=prices, average_model_cost=means)

    qids = list(panel.query_ids)
    t_hat = np.array([selected[q] for q in qids])
    per_model: dict[str, Any] = {}
    cov_sum = 0.0
    for m in models:
        c_m = np.array([panel.inference_cost(m, q, prices) for q in qids], dtype=float)
        ind = (t_hat == m).astype(float)
        cov = population_cov(ind, c_m)
        cov_sum += cov
        routed = ind > 0
        f = float(ind.mean())
        cond = float(c_m[routed].mean()) if routed.any() else None
        var = float(np.mean((c_m - c_m.mean()) ** 2))
        cv = (math.sqrt(var) / float(c_m.mean())) if float(c_m.mean()) else 0.0
        per_model[m] = {
            "fraction": f,
            "unconditional_mean": means[m],
            "conditional_mean": cond,
            "mean_shift": None if cond is None else cond - means[m],
            "cov_assignment_cost": cov,
            "within_model_variance": var,
            "within_model_cv": cv,
            "naive_contrib": f * means[m],
            "realized_contrib": None if cond is None else f * cond,
        }

    realized_inf_mean = agg["realized_inference_mean"]
    naive_mean = agg["naive_inference_mean"]
    identity_rhs = naive_mean + cov_sum
    residual = abs(realized_inf_mean - identity_rhs)
    return {
        "schema_version": SCHEMA_VERSION,
        "n": n,
        "models": models,
        "mix_fractions": agg["mix_fractions"],
        "unconditional_model_means": means,
        "realized_inference_mean": realized_inf_mean,
        "realized_inference_total": agg["realized_inference_total"],
        "naive_inference_mean": naive_mean,
        "naive_inference_total": agg["naive_inference_total"],
        "router_overhead_mean": agg["router_overhead_mean"],
        "realized_mean": agg["realized_mean"],
        "naive_mean_with_overhead": agg["naive_mean_with_overhead"],
        "covariance_sum": cov_sum,
        "realized_minus_naive": realized_inf_mean - naive_mean,
        "identity": "realized_inference = naive + sum_m Cov(1{t=m}, c_m)",
        "identity_rhs": identity_rhs,
        "abs_residual": residual,
        "reconciles": bool(residual <= atol),
        "atol": atol,
        "naive_underestimates_realized": bool(naive_mean < realized_inf_mean - atol),
        "naive_overestimates_realized": bool(naive_mean > realized_inf_mean + atol),
        "per_model": per_model,
        "heterogeneity_needed": any(
            per_model[m]["within_model_variance"] > atol for m in models
        ),
        "n_models_selected": len(agg["mix_fractions"]),
    }


def compare_policies(
    name_a: str,
    decomp_a: Mapping[str, Any],
    name_b: str,
    decomp_b: Mapping[str, Any],
    *,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Sign-flip detector: naive ranking disagrees with realized ranking."""
    na, ra = decomp_a["naive_inference_mean"], decomp_a["realized_inference_mean"]
    nb, rb = decomp_b["naive_inference_mean"], decomp_b["realized_inference_mean"]
    naive_a_cheaper = na < nb - atol
    realized_a_cheaper = ra < rb - atol
    return {
        "policy_a": name_a,
        "policy_b": name_b,
        "naive_a": na,
        "naive_b": nb,
        "realized_a": ra,
        "realized_b": rb,
        "naive_prefers": name_a if naive_a_cheaper else (name_b if nb < na - atol else "tie"),
        "realized_prefers": name_a if realized_a_cheaper else (
            name_b if rb < ra - atol else "tie"
        ),
        "sign_flip": bool(
            abs(na - nb) > atol
            and abs(ra - rb) > atol
            and (na - nb) * (ra - rb) < 0
        ),
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def write_tables(
    *,
    label: str,
    records: Sequence[QueryCostRecord],
    dest_dir: Path,
    panel: CostPanel | None = None,
    prices: PriceTable | None = None,
    extra: Mapping[str, Any] | None = None,
    atol: float = 1e-12,
) -> dict[str, Path]:
    """Write per-query CSV plus aggregate / decomposition JSON.

    `dest_dir` may be a tempfile (tests) or a path under
    `woais_experiments/results/` (committed artifacts). Paths inside the
    frozen Stage 1–10 trees are refused via `assert_inside_results` when the
    destination resolves under `results/`.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        dest_dir.resolve().relative_to(RESULTS.resolve())
        assert_inside_results(dest_dir / "probe")
    except ValueError:
        pass

    per_query_csv = dest_dir / f"{label}_per_query.csv"
    with per_query_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PER_QUERY_CSV_FIELDS)
        w.writeheader()
        for r in records:
            row = r.to_row()
            w.writerow({k: row[k] for k in PER_QUERY_CSV_FIELDS})

    per_query_json = dest_dir / f"{label}_per_query.json"
    per_query_json.write_text(json.dumps([r.to_row() for r in records], indent=2) + "\n")

    agg = aggregate(records, panel=panel, prices=prices)
    agg_path = dest_dir / f"{label}_aggregate.json"
    payload = {"schema_version": SCHEMA_VERSION, "label": label, **agg}
    if extra:
        payload["extra"] = dict(extra)
    agg_path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")

    written = {
        "per_query_csv": per_query_csv,
        "per_query_json": per_query_json,
        "aggregate_json": agg_path,
    }
    if panel is not None and prices is not None:
        decomp = decompose_naive_vs_realized(records, panel, prices, atol=atol)
        if extra:
            decomp = {**decomp, "extra": dict(extra)}
        decomp_path = dest_dir / f"{label}_decomposition.json"
        decomp_path.write_text(json.dumps(_jsonable(decomp), indent=2) + "\n")
        written["decomposition_json"] = decomp_path
    return written


def bundle_from_assignment(
    panel: CostPanel,
    assignment: Mapping[str, str],
    prices: PriceTable,
    overhead: RouterOverhead | None = None,
    *,
    atol: float = 1e-12,
) -> tuple[list[QueryCostRecord], dict[str, Any]]:
    records = records_from_assignment(panel, assignment, prices, overhead=overhead)
    decomp = decompose_naive_vs_realized(records, panel, prices, atol=atol)
    return records, decomp


def config_cheap_expensive(prices: PriceTable) -> tuple[str, str]:
    """Two slugs that already have committed USD/1M rates in models.json."""
    cheap = "openai/gpt-oss-20b"
    expensive = "gpt-4o"
    prices.require(cheap)
    prices.require(expensive)
    return cheap, expensive


def _panel(
    query_ids: list[str],
    models: list[str],
    inputs: dict[tuple[str, str], int],
    outputs: dict[tuple[str, str], int],
) -> CostPanel:
    return CostPanel(
        query_ids=query_ids,
        models=models,
        input_tokens=inputs,
        output_tokens=outputs,
        latency_ms={(m, q): 0.0 for m in models for q in query_ids},
    )


def homogeneous_token_panel(prices: PriceTable, n: int = 8) -> CostPanel:
    """Zero within-model cost variance: naive equals realized for any mix."""
    cheap, expensive = config_cheap_expensive(prices)
    qids = [f"h{i}" for i in range(n)]
    models = [cheap, expensive]
    inp, out = {}, {}
    for q in qids:
        inp[(cheap, q)] = 80
        out[(cheap, q)] = 20
        inp[(expensive, q)] = 80
        out[(expensive, q)] = 20
    return _panel(qids, models, inp, out)


def heterogeneous_token_panel(prices: PriceTable) -> CostPanel:
    """Three short + three long cheap-model completions; expensive model stable."""
    cheap, expensive = config_cheap_expensive(prices)
    qids = [f"q{i}" for i in range(6)]
    models = [cheap, expensive]
    inp, out = {}, {}
    for i, q in enumerate(qids):
        inp[(cheap, q)] = 100
        inp[(expensive, q)] = 100
        out[(cheap, q)] = 10 if i < 3 else 10_000
        out[(expensive, q)] = 100
    return _panel(qids, models, inp, out)


def export_framework_tables(
    dest_dir: Path,
    *,
    prices: PriceTable | None = None,
    overhead: RouterOverhead | None = None,
    stage12: tuple[CostPanel, Mapping[str, str], Mapping[str, str]] | None = None,
) -> dict[str, dict[str, Path]]:
    """Write standardized CSV/JSON for the synthetic cases and optional Stage 1–2."""
    from woais_experiments.accounting.per_query_cost import load_price_table, load_router_overhead

    prices = prices or load_price_table()
    overhead = overhead or load_router_overhead("none")
    cheap, expensive = config_cheap_expensive(prices)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, dict[str, Path]] = {}

    homo = homogeneous_token_panel(prices)
    mix_assign = {q: (cheap if i % 2 == 0 else expensive) for i, q in enumerate(homo.query_ids)}
    rec, _ = bundle_from_assignment(homo, mix_assign, prices, overhead)
    written["naive_equals_realized_zero_variance"] = write_tables(
        label="naive_equals_realized_zero_variance",
        records=rec, dest_dir=dest_dir, panel=homo, prices=prices,
        extra={"case": "homogeneous tokens, mixed assignment"},
    )

    hetero = heterogeneous_token_panel(prices)
    always_cheap = {q: cheap for q in hetero.query_ids}
    rec, _ = bundle_from_assignment(hetero, always_cheap, prices, overhead)
    written["naive_equals_realized_static"] = write_tables(
        label="naive_equals_realized_static",
        records=rec, dest_dir=dest_dir, panel=hetero, prices=prices,
        extra={"case": "static always-cheap, heterogeneous tokens"},
    )

    # Underestimate: route the long cheap-model items to cheap, plus extras.
    under_assign = {
        "q0": cheap, "q1": cheap, "q2": expensive,
        "q3": cheap, "q4": cheap, "q5": cheap,
    }
    rec, _ = bundle_from_assignment(hetero, under_assign, prices, overhead)
    written["naive_underestimates"] = write_tables(
        label="naive_underestimates",
        records=rec, dest_dir=dest_dir, panel=hetero, prices=prices,
        extra={"case": "router keeps long cheap-model completions on the cheap model"},
    )

    over_assign = {
        "q0": cheap, "q1": cheap, "q2": cheap,
        "q3": expensive, "q4": expensive, "q5": expensive,
    }
    rec, _ = bundle_from_assignment(hetero, over_assign, prices, overhead)
    written["naive_overestimates"] = write_tables(
        label="naive_overestimates",
        records=rec, dest_dir=dest_dir, panel=hetero, prices=prices,
        extra={"case": "router sends only short cheap-model completions to cheap"},
    )

    rec_a, decomp_a = bundle_from_assignment(hetero, under_assign, prices, overhead)
    rec_b, decomp_b = bundle_from_assignment(hetero, over_assign, prices, overhead)
    flip = compare_policies("length_biased", decomp_a, "short_to_cheap", decomp_b)
    written["sign_flip_length_biased"] = write_tables(
        label="sign_flip_length_biased",
        records=rec_a, dest_dir=dest_dir, panel=hetero, prices=prices,
        extra={"case": "policy A (length-biased)", "comparison": flip},
    )
    written["sign_flip_short_to_cheap"] = write_tables(
        label="sign_flip_short_to_cheap",
        records=rec_b, dest_dir=dest_dir, panel=hetero, prices=prices,
        extra={"case": "policy B (short to cheap)", "comparison": flip},
    )
    (dest_dir / "sign_flip_comparison.json").write_text(
        json.dumps(_jsonable(flip), indent=2) + "\n"
    )

    if stage12 is not None:
        panel, eco_assign, t2_assign = stage12
        rec, _ = bundle_from_assignment(panel, eco_assign, prices, overhead)
        written["stage12_ecologic"] = write_tables(
            label="stage12_ecologic",
            records=rec, dest_dir=dest_dir, panel=panel, prices=prices,
            extra={"case": "frozen Stage 1–2 EcoLogic keyword routing"},
        )
        rec, _ = bundle_from_assignment(panel, t2_assign, prices, overhead)
        written["stage12_always_t2"] = write_tables(
            label="stage12_always_t2",
            records=rec, dest_dir=dest_dir, panel=panel, prices=prices,
            extra={"case": "frozen Stage 1–2 always Tier 2"},
        )
    return written

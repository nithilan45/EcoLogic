"""Exact aggregates of per-query realized cost, and the mix × mean naive estimator.

The naive estimator used in the routing literature is

    naive = Σ_m  f_m * μ_m

where f_m is the fraction of queries routed to model m and μ_m is that model's
*unconditional* mean inference cost (the mean over the full query set, not over
the queries it actually received). For a static or query-independent policy
this equals realized cost. For a query-dependent router it does not.

Realized aggregate cost is the sum of per-query `realized_cost_i` (math.fsum),
never a mix of averages.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from woais_experiments.accounting.per_query_cost import (
    PriceTable,
    QueryCostRecord,
    RouterOverhead,
    make_query,
)
from woais_experiments.frozen import ItemMatrix


@dataclass
class CostPanel:
    """Full query × model token panel, so unconditional model means exist."""

    query_ids: list[str]
    models: list[str]
    input_tokens: dict[tuple[str, str], int]
    output_tokens: dict[tuple[str, str], int]
    latency_ms: dict[tuple[str, str], float | None] = field(default_factory=dict)

    def inference_cost(self, model: str, query_id: str, prices: PriceTable) -> float:
        return prices.inference_cost(
            model,
            self.input_tokens[(model, query_id)],
            self.output_tokens[(model, query_id)],
        )


def records_from_assignment(
    panel: CostPanel,
    assignment: Mapping[str, str],
    prices: PriceTable,
    overhead: RouterOverhead | None = None,
) -> list[QueryCostRecord]:
    rows: list[QueryCostRecord] = []
    for qid in panel.query_ids:
        model = assignment[qid]
        lat = panel.latency_ms.get((model, qid))
        rows.append(
            make_query(
                qid,
                model,
                panel.input_tokens[(model, qid)],
                panel.output_tokens[(model, qid)],
                prices=prices,
                overhead=overhead,
                latency_ms=lat,
            )
        )
    return rows


def panel_from_item_matrix(
    matrix: ItemMatrix,
    assignment_tiers: Mapping[str, int],
) -> tuple[CostPanel, dict[str, str]]:
    """Stage 1–2 / Stage 7 complete three-tier panel. Models from stored slugs."""
    models_by_tier = dict(matrix.model_of)
    if set(models_by_tier) != {1, 2, 3}:
        raise ValueError(f"expected tiers 1–3 in matrix.model_of, got {sorted(models_by_tier)}")
    models = [models_by_tier[t] for t in (1, 2, 3)]
    inp: dict[tuple[str, str], int] = {}
    out: dict[tuple[str, str], int] = {}
    lat: dict[tuple[str, str], float | None] = {}
    for t, model in models_by_tier.items():
        for qid in matrix.item_ids:
            inp[(model, qid)] = int(matrix.prompt_tokens[(t, qid)])
            out[(model, qid)] = int(matrix.completion_tokens[(t, qid)])
            ls = matrix.latency_s.get((t, qid))
            lat[(model, qid)] = None if ls is None else float(ls) * 1000.0
    assignment = {
        qid: models_by_tier[int(assignment_tiers[qid])] for qid in matrix.item_ids
    }
    panel = CostPanel(
        query_ids=list(matrix.item_ids),
        models=models,
        input_tokens=inp,
        output_tokens=out,
        latency_ms=lat,
    )
    return panel, assignment


def realized_total(records: Sequence[QueryCostRecord]) -> float:
    """Exact sum of per-query realized costs (math.fsum)."""
    return math.fsum(r.realized_cost() for r in records)


def realized_inference_total(records: Sequence[QueryCostRecord]) -> float:
    return math.fsum(r.inference_cost() for r in records)


def overhead_total(records: Sequence[QueryCostRecord]) -> float:
    return math.fsum(r.router_overhead_cost for r in records)


def mix_fractions(records: Sequence[QueryCostRecord]) -> dict[str, float]:
    n = len(records)
    if n == 0:
        return {}
    counts = Counter(r.selected_model for r in records)
    return {m: counts[m] / n for m in sorted(counts)}


def unconditional_model_means(
    panel: CostPanel,
    prices: PriceTable,
    models: Iterable[str] | None = None,
) -> dict[str, float]:
    """μ_m = mean_i c_m(i) over the full panel, not over routed-to-m queries."""
    models = list(models) if models is not None else list(panel.models)
    n = len(panel.query_ids)
    if n == 0:
        return {m: 0.0 for m in models}
    out = {}
    for m in models:
        out[m] = math.fsum(
            panel.inference_cost(m, qid, prices) for qid in panel.query_ids
        ) / n
    return out


def naive_mean(fractions: Mapping[str, float], average_model_cost: Mapping[str, float]) -> float:
    """Σ_m f_m μ_m. Missing models in `fractions` contribute 0."""
    return math.fsum(fractions.get(m, 0.0) * average_model_cost[m] for m in average_model_cost)


def aggregate(
    records: Sequence[QueryCostRecord],
    *,
    panel: CostPanel | None = None,
    prices: PriceTable | None = None,
    average_model_cost: Mapping[str, float] | None = None,
) -> dict:
    """Realized totals plus naive mix×mean, when unconditional means are available."""
    n = len(records)
    real_inf = realized_inference_total(records)
    real = realized_total(records)
    oh = overhead_total(records)
    mix = mix_fractions(records)
    means = dict(average_model_cost) if average_model_cost is not None else None
    if means is None and panel is not None and prices is not None:
        models = list(dict.fromkeys([*panel.models, *[r.selected_model for r in records]]))
        means = unconditional_model_means(panel, prices, models)
    payload: dict = {
        "n": n,
        "realized_inference_total": real_inf,
        "realized_inference_mean": (real_inf / n) if n else 0.0,
        "router_overhead_total": oh,
        "router_overhead_mean": (oh / n) if n else 0.0,
        "realized_total": real,
        "realized_mean": (real / n) if n else 0.0,
        "mix_fractions": mix,
        "unconditional_model_means": means,
        "naive_inference_mean": None,
        "naive_inference_total": None,
        "naive_mean_with_overhead": None,
        "naive_minus_realized_inference_mean": None,
    }
    if means is not None:
        naive = naive_mean(mix, means)
        payload["naive_inference_mean"] = naive
        payload["naive_inference_total"] = naive * n
        payload["naive_mean_with_overhead"] = naive + payload["router_overhead_mean"]
        payload["naive_minus_realized_inference_mean"] = naive - payload["realized_inference_mean"]
    return payload

"""Cheap vs strong vs routed latency, with measured router overhead only.

Direct cheap / direct strong are query-independent always-one-model policies.
``router_plus_selected`` uses the model the router actually chose. End-to-end
for each is the recorded span for that (query, model) pair.

Router overhead is ``router_decision_ms`` and that quantity as a percentage of
the routed call's ``end_to_end_ms``. It is left missing when the router span
was not recorded (frozen EcoLogic logs).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from woais_experiments.frozen import ItemMatrix, write_result
from woais_experiments.latency.analyze_latency import (
    DEFAULT_LEVEL,
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    _jsonable,
    rows_to_csv,
    summarize,
)
from woais_experiments.latency.timing import seconds_to_ms

FRONTIER_FIELDS = (
    "query_id",
    "cheap_end_to_end_ms",
    "strong_end_to_end_ms",
    "router_selected_end_to_end_ms",
    "router_selected_tier",
    "router_selected_model",
    "router_decision_ms",
    "router_overhead_ms",
    "router_overhead_pct",
    "router_minus_cheap_ms",
    "router_minus_strong_ms",
    "source",
)


def overhead_ms(router_decision_ms: float | None) -> float | None:
    """Absolute overhead is the measured router span, not a residual."""
    if router_decision_ms is None:
        return None
    return float(router_decision_ms)


def overhead_pct(router_decision_ms: float | None, end_to_end_ms: float | None) -> float | None:
    if router_decision_ms is None or end_to_end_ms is None:
        return None
    if end_to_end_ms <= 0:
        return None
    return 100.0 * float(router_decision_ms) / float(end_to_end_ms)


def frontier_row(
    query_id: str,
    cheap_ms: float | None,
    strong_ms: float | None,
    router_ms: float | None,
    *,
    selected_tier: int | None = None,
    selected_model: str | None = None,
    router_decision_ms: float | None = None,
    source: str = "legacy_latency_s",
) -> dict[str, Any]:
    oh = overhead_ms(router_decision_ms)
    return {
        "query_id": query_id,
        "cheap_end_to_end_ms": None if cheap_ms is None else float(cheap_ms),
        "strong_end_to_end_ms": None if strong_ms is None else float(strong_ms),
        "router_selected_end_to_end_ms": None if router_ms is None else float(router_ms),
        "router_selected_tier": selected_tier,
        "router_selected_model": selected_model,
        "router_decision_ms": None if router_decision_ms is None else float(router_decision_ms),
        "router_overhead_ms": oh,
        "router_overhead_pct": overhead_pct(router_decision_ms, router_ms),
        "router_minus_cheap_ms": (
            None if router_ms is None or cheap_ms is None else float(router_ms) - float(cheap_ms)
        ),
        "router_minus_strong_ms": (
            None if router_ms is None or strong_ms is None else float(router_ms) - float(strong_ms)
        ),
        "source": source,
    }


def compare_direct_and_routed(
    query_ids: Sequence[Any],
    cheap_ms: Sequence[float | None],
    strong_ms: Sequence[float | None],
    router_ms: Sequence[float | None],
    *,
    selected_tier: Sequence[int | None] | None = None,
    selected_model: Sequence[str | None] | None = None,
    router_decision_ms: Sequence[float | None] | None = None,
    source: str | Sequence[str] = "legacy_latency_s",
    cheap_name: str = "cheap",
    strong_name: str = "strong",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> dict[str, Any]:
    n = len(query_ids)
    if not (len(cheap_ms) == len(strong_ms) == len(router_ms) == n):
        raise ValueError("all series must have one value per query")

    def _get(seq, i, default=None):
        if seq is None:
            return default
        return seq[i]

    src_is_str = isinstance(source, str)
    rows = []
    for i, qid in enumerate(query_ids):
        rows.append(
            frontier_row(
                str(qid),
                cheap_ms[i],
                strong_ms[i],
                router_ms[i],
                selected_tier=_get(selected_tier, i),
                selected_model=_get(selected_model, i),
                router_decision_ms=_get(router_decision_ms, i),
                source=source if src_is_str else source[i],
            )
        )

    def col(key: str) -> list[float | None]:
        return [r[key] for r in rows]

    overhead_measured = sum(r["router_overhead_ms"] is not None for r in rows)
    return {
        "n_queries": n,
        "cheap_name": cheap_name,
        "strong_name": strong_name,
        "honesty": {
            "legacy": (
                "When source is legacy_latency_s, end-to-end is 1000 * latency_s "
                "for that (query, model). TTFT/queue/generation are unavailable. "
                "Router overhead is missing unless a router_decision_ms span was measured."
            ),
            "overhead": (
                "router_overhead_ms is the measured router_decision span, not "
                "end_to_end minus the selected model's API time (that residual "
                "would mix clock error with unmeasured queueing)."
            ),
        },
        "policies": {
            "direct_cheap": summarize(col("cheap_end_to_end_ms"), n_boot=n_boot, seed=seed, level=level),
            "direct_strong": summarize(col("strong_end_to_end_ms"), n_boot=n_boot, seed=seed, level=level),
            "router_plus_selected": summarize(
                col("router_selected_end_to_end_ms"), n_boot=n_boot, seed=seed, level=level
            ),
        },
        "router_overhead": {
            "n_measured": overhead_measured,
            "n_missing": n - overhead_measured,
            "absolute_ms": summarize(col("router_overhead_ms"), n_boot=n_boot, seed=seed, level=level),
            "percent_of_end_to_end": summarize(
                col("router_overhead_pct"), n_boot=n_boot, seed=seed, level=level
            ),
        },
        "deltas": {
            "router_minus_cheap_ms": summarize(
                col("router_minus_cheap_ms"), n_boot=n_boot, seed=seed, level=level
            ),
            "router_minus_strong_ms": summarize(
                col("router_minus_strong_ms"), n_boot=n_boot, seed=seed, level=level
            ),
        },
        "per_query": rows,
    }


def from_item_matrix(
    matrix: ItemMatrix,
    router_assign: Mapping[str, int],
    *,
    cheap_tier: int = 2,
    strong_tier: int = 3,
    router_decision_ms: Mapping[str, float] | Sequence[float | None] | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> dict[str, Any]:
    """Legacy complete panel: each cell is ``latency_s`` (seconds) → milliseconds."""
    ids = list(matrix.item_ids)
    cheap = [seconds_to_ms(matrix.latency_s[(cheap_tier, i)]) for i in ids]
    strong = [seconds_to_ms(matrix.latency_s[(strong_tier, i)]) for i in ids]
    router = [seconds_to_ms(matrix.latency_s[(int(router_assign[i]), i)]) for i in ids]
    tiers = [int(router_assign[i]) for i in ids]
    models = [matrix.model_of.get(t) for t in tiers]
    decisions: list[float | None] | None
    if router_decision_ms is None:
        decisions = None
    elif isinstance(router_decision_ms, Mapping):
        decisions = [router_decision_ms.get(i) for i in ids]
    else:
        decisions = list(router_decision_ms)
        if len(decisions) != len(ids):
            raise ValueError("router_decision_ms length must match n_queries")
    return compare_direct_and_routed(
        ids,
        cheap,
        strong,
        router,
        selected_tier=tiers,
        selected_model=models,
        router_decision_ms=decisions,
        source="legacy_latency_s",
        cheap_name=matrix.model_of.get(cheap_tier, f"t{cheap_tier}"),
        strong_name=matrix.model_of.get(strong_tier, f"t{strong_tier}"),
        n_boot=n_boot,
        seed=seed,
        level=level,
    )


def save_frontier(
    payload: Mapping[str, Any],
    relpath: str = "latency/frontier",
) -> dict[str, str]:
    rows = payload["per_query"]
    summary = {k: v for k, v in payload.items() if k != "per_query"}
    json_path = write_result(f"{relpath}.json", _jsonable(dict(payload)))
    csv_path = write_result(f"{relpath}_per_query.csv", rows_to_csv(rows, FRONTIER_FIELDS))
    agg_path = write_result(f"{relpath}_aggregates.json", _jsonable(summary))
    return {"json": str(json_path), "csv": str(csv_path), "aggregates": str(agg_path)}

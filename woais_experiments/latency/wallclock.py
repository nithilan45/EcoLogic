"""Empirical wall-clock latency from stored latency_s. Not TTFT."""

from __future__ import annotations

from collections import defaultdict

from woais_experiments.accounting.costs import TIERS, _percentile
from woais_experiments.frozen import ItemMatrix


def _summarize(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "p50": _percentile(xs, 50),
        "p90": _percentile(xs, 90),
        "p95": _percentile(xs, 95),
        "p99": _percentile(xs, 99),
        "min": min(xs),
        "max": max(xs),
    }


def latency_by_tier_benchmark(matrix: ItemMatrix) -> dict:
    buckets: dict[tuple[int, str], list[float]] = defaultdict(list)
    for t in TIERS:
        for i in matrix.item_ids:
            b = matrix.bench_of.get(i, "unknown")
            buckets[(t, b)].append(matrix.latency_s[(t, i)])
    out: dict[str, dict] = {}
    for (t, b), xs in sorted(buckets.items()):
        out[f"t{t}/{b}"] = _summarize(xs)
    by_tier = {}
    for t in TIERS:
        xs = [matrix.latency_s[(t, i)] for i in matrix.item_ids]
        by_tier[str(t)] = _summarize(xs)
    return {"by_tier": by_tier, "by_tier_benchmark": out}


def policy_latency(matrix: ItemMatrix, assign: dict[str, int]) -> dict:
    xs = [matrix.latency_s[(assign[i], i)] for i in matrix.item_ids]
    return _summarize(xs)


def latency_per_output_token(matrix: ItemMatrix) -> dict:
    out = {}
    for t in TIERS:
        ratios = []
        for i in matrix.item_ids:
            ct = matrix.completion_tokens[(t, i)]
            if ct > 0:
                ratios.append(matrix.latency_s[(t, i)] / ct)
        out[str(t)] = _summarize(ratios)
    return out

"""Token-count dispersion: the quantity that makes mean-cost accounting biased."""

from __future__ import annotations

import math
from collections import defaultdict

from woais_experiments.frozen import ItemMatrix

TIERS = (1, 2, 3)


def _cv(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    if mean == 0:
        return 0.0
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    return math.sqrt(var) / mean


def token_dispersion(matrix: ItemMatrix) -> dict:
    by_tier: dict[int, dict[str, list[float]]] = {t: defaultdict(list) for t in TIERS}
    for t in TIERS:
        for i in matrix.item_ids:
            by_tier[t]["total"].append(float(matrix.tokens[(t, i)]))
            by_tier[t]["prompt"].append(float(matrix.prompt_tokens[(t, i)]))
            by_tier[t]["completion"].append(float(matrix.completion_tokens[(t, i)]))
            by_tier[t]["usd"].append(float(matrix.usd[(t, i)]))
    out = {}
    for t in TIERS:
        rec = {}
        for name, xs in by_tier[t].items():
            rec[name] = {
                "n": len(xs),
                "mean": sum(xs) / len(xs),
                "cv": _cv(xs),
                "min": min(xs),
                "max": max(xs),
            }
        out[str(t)] = rec
    return out


def corr_tokens_vs_latency(matrix: ItemMatrix) -> dict:
    """Pearson correlation of completion tokens with wall-clock latency, per tier."""
    out = {}
    for t in TIERS:
        x = [float(matrix.completion_tokens[(t, i)]) for i in matrix.item_ids]
        y = [float(matrix.latency_s[(t, i)]) for i in matrix.item_ids]
        out[str(t)] = {"pearson_r": _pearson(x, y), "n": len(x)}
    return out


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)

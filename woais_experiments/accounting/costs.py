"""USD and modelled-energy accounting from stored token/usage fields."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Callable, Iterable

from woais_experiments.frozen import ItemMatrix
from woais_experiments.paths import CONFIGS
from woais_experiments.statistics.inference import wilson_dict

TIERS = (1, 2, 3)


def _models_config() -> dict:
    return json.loads((CONFIGS / "models.json").read_text())


def paper_energy_rates() -> dict[int, float]:
    """Committed paper J/1k rates from `configs/models.json` only."""
    cfg = _models_config()["eval_models"]
    return {int(t): float(cfg[str(t)]["paper_energy_per_1k"]) for t in TIERS}


def subst_energy_rates() -> dict[int, float]:
    cfg = _models_config()["eval_models"]
    return {int(t): float(cfg[str(t)]["subst_energy_per_1k"]) for t in TIERS}


def usd_rates_per_million() -> dict[str, dict[str, float]]:
    return dict(_models_config()["usd_per_million"])


def usd_from_tokens(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """USD from committed $/1M input and output rates in `configs/models.json`."""
    rates = usd_rates_per_million().get(model)
    if not rates:
        return None
    return prompt_tokens / 1_000_000 * rates["input"] + completion_tokens / 1_000_000 * rates["output"]


def energy_j(total_tokens: float, rate_per_1k: float) -> float:
    return total_tokens / 1000.0 * rate_per_1k


def cost_fn_usd(matrix: ItemMatrix) -> Callable[[int, str], float]:
    return lambda t, i: matrix.usd[(t, i)]


def cost_fn_energy(matrix: ItemMatrix, rates: dict[int, float] | None = None) -> Callable[[int, str], float]:
    rates = rates or paper_energy_rates()
    return lambda t, i: energy_j(matrix.tokens[(t, i)], rates[t])


def evaluate_assignment(
    matrix: ItemMatrix,
    assign: dict[str, int],
    cost_of: Callable[[int, str], float],
    *,
    name: str,
    overhead_per_query: float = 0.0,
    cost_kind: str = "usd",
    cost_unit: str = "USD",
    cost_provenance: str = "stored_call_usd",
) -> dict:
    items = matrix.item_ids
    outcomes = [bool(matrix.correct[(assign[i], i)]) for i in items]
    k = sum(outcomes)
    n = len(items)
    acc = wilson_dict(k, n)
    inference_cost = sum(cost_of(assign[i], i) for i in items)
    total_cost = inference_cost + float(overhead_per_query) * n
    total_tokens = sum(matrix.tokens[(assign[i], i)] for i in items)
    lat = [matrix.latency_s[(assign[i], i)] for i in items]
    mix: dict[int, int] = defaultdict(int)
    for i in items:
        mix[assign[i]] += 1
    per_bench = {}
    benches = sorted({matrix.bench_of.get(i, "unknown") for i in items})
    for b in benches:
        sub = [i for i in items if matrix.bench_of.get(i) == b]
        kk = sum(matrix.correct[(assign[i], i)] for i in sub)
        per_bench[b] = {"k": kk, "n": len(sub), "acc": kk / len(sub) if sub else 0.0}
    return {
        "name": name,
        "accuracy": acc["p"],
        "ci": [acc["ci_lo"], acc["ci_hi"]],
        "correct": k,
        "n": n,
        "cost": total_cost,
        "inference_cost": inference_cost,
        "router_overhead_per_query": float(overhead_per_query),
        "cost_per_item": total_cost / n if n else 0.0,
        "cost_kind": cost_kind,
        "cost_unit": cost_unit,
        "cost_provenance": cost_provenance,
        "role": (
            "hindsight_oracle"
            if str(name).startswith("oracle")
            else "deployable_or_baseline"
        ),
        "tokens": total_tokens,
        "latency_mean_s": sum(lat) / n if n else 0.0,
        "latency_p50_s": _percentile(lat, 50),
        "latency_p95_s": _percentile(lat, 95),
        "latency_p99_s": _percentile(lat, 99),
        "tier_mix": {str(t): mix[t] for t in TIERS},
        "per_benchmark": per_bench,
        "outcomes": outcomes,
    }


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    idx = min(len(ys) - 1, max(0, round((q / 100.0) * (len(ys) - 1))))
    return ys[idx]


def naive_policy_cost(mix_frac: dict[int, float], mean_cost: dict[int, float]) -> float:
    return sum(mix_frac[t] * mean_cost[t] for t in mix_frac)


def true_policy_cost(assign: dict[str, int], cost_of: Callable[[int, str], float], items: Iterable[str]) -> float:
    items = list(items)
    return sum(cost_of(assign[i], i) for i in items) / len(items)


def mean_cost_by_tier(matrix: ItemMatrix, cost_of: Callable[[int, str], float]) -> dict[int, float]:
    n = matrix.n
    return {t: sum(cost_of(t, i) for i in matrix.item_ids) / n for t in TIERS}


def drop_outcomes(stats: dict) -> dict:
    out = dict(stats)
    out.pop("outcomes", None)
    return out

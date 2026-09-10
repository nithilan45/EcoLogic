"""Frozen academic item sets. Read-only; never rebuilds benchmark_items.json."""

from __future__ import annotations

from collections import Counter, defaultdict

from woais_experiments.frozen import ItemMatrix, load_json
from woais_experiments.paths import ROOT


def stage12_item_mix(matrix: ItemMatrix) -> dict:
    counts = Counter(matrix.bench_of[i] for i in matrix.item_ids)
    n = matrix.n
    return {
        "n": n,
        "counts": dict(counts),
        "fractions": {k: v / n for k, v in counts.items()},
        "source": "raw_results/benchmark_items.json (read-only; not rebuilt)",
    }


def resample_by_benchmark(
    matrix: ItemMatrix,
    weights: dict[str, float],
    n: int,
    seed: int,
) -> list[str]:
    """Draw `n` item ids with replacement according to benchmark weights.

    Items are drawn uniformly within each benchmark. This is a *synthetic mix*
    of already-measured queries, not a new workload and not an arrival process.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    by_b: dict[str, list[str]] = defaultdict(list)
    for i in matrix.item_ids:
        by_b[matrix.bench_of[i]].append(i)
    names = list(weights)
    p = np.array([weights[b] for b in names], dtype=float)
    p = p / p.sum()
    benches = rng.choice(names, size=n, p=p)
    out = []
    for b in benches:
        pool = by_b[b]
        out.append(str(rng.choice(pool)))
    return out


def load_pool_metadata() -> dict:
    """Pointers at frozen pools; does not open the large generation files."""
    return {
        "stage12_items": str(ROOT / "raw_results" / "benchmark_items.json"),
        "router_v2_train": str(ROOT / "router_v2" / "train_pool.json"),
        "router_v2_calibration": str(ROOT / "router_v2" / "calibration_pool.json"),
        "s7_train": str(ROOT / "stage7_10" / "s7_train_pool.json"),
        "s7_calibration": str(ROOT / "stage7_10" / "s7_calibration_pool.json"),
        "s7_test": str(ROOT / "stage7_10" / "s7_test_set.json"),
    }


def pool_sizes() -> dict:
    out = {}
    mapping = {
        "stage12": ROOT / "raw_results" / "benchmark_items.json",
        "v2_train": ROOT / "router_v2" / "train_pool.json",
        "v2_calibration": ROOT / "router_v2" / "calibration_pool.json",
        "s7_train": ROOT / "stage7_10" / "s7_train_pool.json",
        "s7_calibration": ROOT / "stage7_10" / "s7_calibration_pool.json",
        "s7_test": ROOT / "stage7_10" / "s7_test_set.json",
    }
    for name, path in mapping.items():
        blob = load_json(path)
        items = blob["items"] if isinstance(blob, dict) and "items" in blob else blob
        out[name] = {"n": len(items), "path": path.relative_to(ROOT).as_posix()}
    return out

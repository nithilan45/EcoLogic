"""Shared offline robustness stats for the three papers.

Reads nothing itself. Callers pass already-loaded item tables.
No writes to frozen trees. No API calls.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from itertools import product
from typing import Callable, Iterable


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) * (0.5**n) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail)


def mcnemar_counts(eco_ok: list[bool], other_ok: list[bool]) -> tuple[int, int, float]:
    b = c = 0
    for x, y in zip(eco_ok, other_ok):
        if x and not y:
            b += 1
        elif y and not x:
            c += 1
    return b, c, mcnemar_exact(b, c)


def cohens_g(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 0.0
    return abs(b / n - 0.5)


def holm(pvalues: list[tuple[str, float]]) -> dict[str, float]:
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i][1])
    adj: dict[str, float] = {}
    running = 0.0
    for rank, idx in enumerate(order):
        name, raw = pvalues[idx]
        a = min(1.0, raw * (m - rank))
        a = max(a, running)
        running = a
        adj[name] = a
    return adj


def leave_one_benchmark(
    items: list[str],
    bench_of: dict[str, str],
    eco_ok: dict[str, bool],
    t2_ok: dict[str, bool],
    eco_cost: dict[str, float],
    t2_cost: dict[str, float],
) -> list[dict]:
    benches = sorted({bench_of[i] for i in items})
    rows = []
    subsets = [("all", items)]
    for drop in benches:
        keep = [i for i in items if bench_of[i] != drop]
        subsets.append((f"drop_{drop}", keep))
    for bname in benches:
        keep = [i for i in items if bench_of[i] == bname]
        subsets.append((f"only_{bname}", keep))
    for name, keep in subsets:
        n = len(keep)
        e_ok = [eco_ok[i] for i in keep]
        t_ok = [t2_ok[i] for i in keep]
        b, c, p = mcnemar_counts(e_ok, t_ok)
        e_acc = sum(e_ok) / n
        t_acc = sum(t_ok) / n
        e_c = sum(eco_cost[i] for i in keep)
        t_c = sum(t2_cost[i] for i in keep)
        rows.append(
            {
                "subset": name,
                "n": n,
                "eco_acc": e_acc,
                "t2_acc": t_acc,
                "eco_cost": e_c,
                "t2_cost": t_c,
                "t2_dominates_acc_and_cost": (t_acc >= e_acc and t_c <= e_c)
                and (t_acc > e_acc or t_c < e_c),
                "mcnemar_b": b,
                "mcnemar_c": c,
                "mcnemar_p": p,
                "cohens_g": cohens_g(b, c),
            }
        )
    return rows


def mix_reweight_grid(
    items: list[str],
    bench_of: dict[str, str],
    per_item: dict[str, dict[str, float]],
    policies: Iterable[str],
    step: float = 0.2,
) -> list[dict]:
    """Weighted per-query means under a simplex of benchmark weights.

    per_item[iid][policy] is a scalar (USD, J, or seconds).
    """
    benches = ("humaneval", "mmlu", "gsm8k")
    by_b: dict[str, list[str]] = {b: [i for i in items if bench_of[i] == b] for b in benches}
    means: dict[str, dict[str, float]] = {}
    for pol in policies:
        means[pol] = {
            b: sum(per_item[i][pol] for i in by_b[b]) / len(by_b[b]) for b in benches
        }
    rows = []
    ticks = []
    k = int(round(1.0 / step))
    for a, b, c in product(range(k + 1), repeat=3):
        if a + b + c != k:
            continue
        w = (a / k, b / k, c / k)
        ticks.append(w)
    for w_he, w_mm, w_gs in ticks:
        w = {"humaneval": w_he, "mmlu": w_mm, "gsm8k": w_gs}
        rec = {"w_humaneval": w_he, "w_mmlu": w_mm, "w_gsm8k": w_gs}
        for pol in policies:
            rec[pol] = sum(w[b] * means[pol][b] for b in benches)
        rec["t2_cheaper_than_eco"] = rec.get("always_t2", 0) < rec.get("ecologic", 1e99)
        rows.append(rec)
    return rows


def bootstrap_t2_dominates(
    items: list[str],
    eco_ok: dict[str, bool],
    t2_ok: dict[str, bool],
    eco_cost: dict[str, float],
    t2_cost: dict[str, float],
    n_boot: int = 10000,
    seed: int = 20260905,
    extra_cost: dict[str, dict[str, float]] | None = None,
    extra_better: Callable[[float, float], bool] | None = None,
) -> dict:
    rng = random.Random(seed)
    n = len(items)
    dom = 0
    extra_dom = 0
    acc_d = []
    cost_d = []
    for _ in range(n_boot):
        samp = [items[rng.randrange(n)] for _ in range(n)]
        e_c = sum(eco_ok[i] for i in samp)
        t_c = sum(t2_ok[i] for i in samp)
        e_u = sum(eco_cost[i] for i in samp)
        t_u = sum(t2_cost[i] for i in samp)
        if t_c >= e_c and t_u <= e_u and (t_c > e_c or t_u < e_u):
            dom += 1
        if extra_cost is not None and extra_better is not None:
            e_x = sum(extra_cost["ecologic"][i] for i in samp)
            t_x = sum(extra_cost["always_t2"][i] for i in samp)
            if extra_better(t_x, e_x) and t_c >= e_c:
                extra_dom += 1
        acc_d.append((t_c - e_c) / n)
        cost_d.append(t_u - e_u)
    acc_d.sort()
    cost_d.sort()
    def pct(xs, q):
        return xs[min(len(xs) - 1, max(0, round((q / 100.0) * (len(xs) - 1))))]
    out = {
        "n_boot": n_boot,
        "frac_t2_dominates_quality_and_cost": dom / n_boot,
        "acc_gap_t2_minus_eco_p50": pct(acc_d, 50),
        "acc_gap_t2_minus_eco_p025": pct(acc_d, 2.5),
        "acc_gap_t2_minus_eco_p975": pct(acc_d, 97.5),
        "cost_gap_t2_minus_eco_p50": pct(cost_d, 50),
        "cost_gap_t2_minus_eco_p025": pct(cost_d, 2.5),
        "cost_gap_t2_minus_eco_p975": pct(cost_d, 97.5),
    }
    if extra_cost is not None:
        out["frac_t2_dominates_quality_and_extra"] = extra_dom / n_boot
    return out


def oracle_confusion(
    items: list[str],
    eco_tier: dict[str, int],
    oracle_tier: dict[str, int],
    eco_ok: dict[str, bool],
    t2_ok: dict[str, bool],
) -> dict:
    agree = sum(1 for i in items if eco_tier[i] == oracle_tier[i])
    counts = defaultdict(int)
    for i in items:
        counts[(eco_tier[i], oracle_tier[i])] += 1
    disagree = [i for i in items if eco_tier[i] != oracle_tier[i]]
    disagree_t2_right_eco_wrong = sum(
        1 for i in disagree if t2_ok[i] and not eco_ok[i]
    )
    return {
        "n": len(items),
        "agreement": agree / len(items),
        "n_agree": agree,
        "n_disagree": len(disagree),
        "disagree_t2_right_eco_wrong": disagree_t2_right_eco_wrong,
        "cells": {f"eco{i}_ora{j}": counts[(i, j)] for i in (1, 2, 3) for j in (1, 2, 3)},
    }

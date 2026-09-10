"""Routing policies as functions of a frozen ItemMatrix. No API calls."""

from __future__ import annotations

import random
from typing import Callable

from woais_experiments.accounting.costs import (
    TIERS,
    cost_fn_energy,
    cost_fn_usd,
    drop_outcomes,
    evaluate_assignment,
    mean_cost_by_tier,
    naive_policy_cost,
    paper_energy_rates,
    true_policy_cost,
)
from woais_experiments.frozen import ItemMatrix
from woais_experiments.statistics.inference import matched_cost_fraction, mcnemar, mixture_accuracy
from woais_experiments.statistics.regret import decompose_regret

import numpy as np

RANDOM_SEED = 20260905  # same as benchmark/analyze.py


def always_tier(items: list[str], tier: int) -> dict[str, int]:
    return {i: tier for i in items}


def random_tiers(items: list[str], seed: int = RANDOM_SEED, choices: tuple[int, ...] = TIERS) -> dict[str, int]:
    rng = random.Random(seed)
    return {i: rng.choice(list(choices)) for i in items}


def keyword_from_routing(routing: dict, items: list[str], key: str = "raw") -> dict[str, int]:
    return {i: int(routing[i][key]["tier"]) for i in items}


def oracle(
    matrix: ItemMatrix,
    cost_of: Callable[[int, str], float],
) -> dict[str, int]:
    assign = {}
    for i in matrix.item_ids:
        cands = [(cost_of(t, i), t) for t in TIERS if matrix.correct[(t, i)]]
        if cands:
            assign[i] = min(cands)[1]
        else:
            assign[i] = min((cost_of(t, i), t) for t in TIERS)[1]
    return assign


def cost_ordered_cascade(
    p1: np.ndarray,
    p2: np.ndarray,
    tau: float,
    order: tuple[int, ...] = (2, 1),
) -> np.ndarray:
    """Identical rule to router_v2.calibrate.route, kept local to avoid training imports."""
    p = {1: p1, 2: p2}
    t = np.full(len(p1), 3, dtype=int)
    for tier in reversed(order):
        t[p[tier] >= tau] = tier
    return t


def accuracy_optimal_static_mixture(
    mean_cost: dict[int, float],
    mean_acc: dict[int, float],
    budget: float,
) -> dict:
    """Max expected accuracy of a query-independent mix with E[cost] ≤ budget."""
    models = list(mean_cost)
    candidates: list[dict] = []

    for t in models:
        if mean_cost[t] <= budget + 1e-15:
            mix = {m: 0.0 for m in models}
            mix[t] = 1.0
            candidates.append({
                "kind": "pure",
                "mix": mix,
                "cost": mean_cost[t],
                "accuracy": mean_acc[t],
            })

    for i, a in enumerate(models):
        for b in models[i + 1:]:
            ca, cb = mean_cost[a], mean_cost[b]
            if abs(ca - cb) < 1e-18:
                continue
            # sit on the budget hyperplane if it falls between the two costs
            lo, hi = (ca, cb) if ca < cb else (cb, ca)
            if budget < lo - 1e-15 or budget > hi + 1e-15:
                continue
            pb = (budget - ca) / (cb - ca)
            pa = 1.0 - pb
            if pa < -1e-12 or pb < -1e-12:
                continue
            pa, pb = max(0.0, pa), max(0.0, pb)
            mix = {m: 0.0 for m in models}
            mix[a], mix[b] = pa, pb
            cost = pa * ca + pb * cb
            acc = pa * mean_acc[a] + pb * mean_acc[b]
            candidates.append({
                "kind": f"mix_{a}_{b}",
                "mix": mix,
                "cost": cost,
                "accuracy": acc,
            })

    if not candidates:
        t = min(models, key=lambda m: mean_cost[m])
        mix = {m: 0.0 for m in models}
        mix[t] = 1.0
        return {"feasible": False, "mix": mix, "cost": mean_cost[t], "accuracy": mean_acc[t]}

    best = max(candidates, key=lambda c: (c["accuracy"], -c["cost"]))
    best["feasible"] = True
    best["n_candidates"] = len(candidates)
    return best


def two_model_matched_cost(
    router_cost: float,
    mean_weak: float,
    mean_strong: float,
    acc_weak: float,
    acc_strong: float,
) -> dict:
    f = matched_cost_fraction(router_cost, mean_weak, mean_strong)
    return {
        "f_strong": f,
        "accuracy": mixture_accuracy(f, acc_weak, acc_strong),
        "expected_cost": f * mean_strong + (1.0 - f) * mean_weak,
    }


def build_stage12_policies(
    matrix: ItemMatrix,
    routing: dict,
    *,
    seed: int = RANDOM_SEED,
) -> dict[str, dict[str, int]]:
    items = matrix.item_ids
    usd = cost_fn_usd(matrix)
    energy = cost_fn_energy(matrix, paper_energy_rates())
    return {
        "ecologic": keyword_from_routing(routing, items, "raw"),
        "ecologic_wrapped": keyword_from_routing(routing, items, "wrapped"),
        "always_t1": always_tier(items, 1),
        "always_t2": always_tier(items, 2),
        "frontier": always_tier(items, 3),
        "random": random_tiers(items, seed=seed),
        "oracle_usd": oracle(matrix, usd),
        "oracle_energy": oracle(matrix, energy),
    }


def evaluate_policies(
    matrix: ItemMatrix,
    policies: dict[str, dict[str, int]],
    cost_of,
    *,
    vs: str = "ecologic",
) -> dict:
    stats = {}
    for name, assign in policies.items():
        stats[name] = evaluate_assignment(matrix, assign, cost_of, name=name)
    ref = stats[vs]["outcomes"]
    tests = {}
    for name, s in stats.items():
        if name == vs:
            continue
        tests[name] = mcnemar(ref, s["outcomes"])
    cleaned = {k: drop_outcomes(v) for k, v in stats.items()}
    return {"policies": cleaned, "mcnemar_vs": tests}


def cost_accounting_for_policy(
    matrix: ItemMatrix,
    assign: dict[str, int],
    cost_of,
) -> dict:
    """Naive (tier-mean × mix) vs true (per-item) mean cost of one assignment."""
    items = matrix.item_ids
    n = len(items)
    mix = {t: sum(1 for i in items if assign[i] == t) / n for t in TIERS}
    means = mean_cost_by_tier(matrix, cost_of)
    naive = naive_policy_cost(mix, means)
    true = true_policy_cost(assign, cost_of, items)
    return {
        "mix": mix,
        "per_tier_mean": means,
        "naive_mean_cost": naive,
        "true_mean_cost": true,
        "naive_minus_true": naive - true,
        "relative_bias": (naive - true) / true if true else None,
    }


def regret_for_policy(
    matrix: ItemMatrix,
    t_hat: dict[str, int],
    t_star: dict[str, int],
    cost_of,
) -> dict:
    items = matrix.item_ids
    costs = np.array([[cost_of(t, i) for t in TIERS] for i in items], dtype=float)
    star = np.array([t_star[i] for i in items])
    hat = np.array([t_hat[i] for i in items])
    return decompose_regret(star, hat, costs, labels=list(TIERS))

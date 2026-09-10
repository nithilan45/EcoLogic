"""Exact confusion-matrix regret correction (Stage 8 identity).

    R_true = R_naive + Σ_{i≠j} Cov(1{t*=i, t̂=j}, e_j(X) − e_i(X))

This is a library. It does not import or run stage7_10/regret_correction.py,
which writes into the frozen tree.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def population_cov(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(a * b) - np.mean(a) * np.mean(b))


def decompose_regret(
    t_star: np.ndarray,
    t_hat: np.ndarray,
    costs: np.ndarray,
    labels: list[int] | None = None,
) -> dict[str, Any]:
    """Decompose per-item regret into naive (tier-mean) and covariance terms.

    Parameters
    ----------
    t_star, t_hat : shape (n,) integer model ids
    costs : shape (n, m) per-item cost of each model, columns aligned with `labels`
    labels : model ids for columns of `costs`; default unique sorted ids
    """
    t_star = np.asarray(t_star)
    t_hat = np.asarray(t_hat)
    costs = np.asarray(costs, dtype=float)
    n, m = costs.shape
    if labels is None:
        labels = [int(x) for x in sorted(set(t_star) | set(t_hat))]
        if len(labels) != m:
            labels = list(range(1, m + 1))
    col = {lab: i for i, lab in enumerate(labels)}

    star_k = np.array([col[int(t)] for t in t_star])
    hat_k = np.array([col[int(t)] for t in t_hat])
    r_true = float(np.mean(costs[np.arange(n), hat_k] - costs[np.arange(n), star_k]))

    e_mean = {lab: float(costs[:, col[lab]].mean()) for lab in labels}
    r_naive = 0.0
    cells: dict[tuple[int, int], float] = {}
    for i_star in labels:
        for j_hat in labels:
            pr = float(np.mean((t_star == i_star) & (t_hat == j_hat)))
            if pr > 0:
                cells[(i_star, j_hat)] = pr
            if i_star != j_hat and pr > 0:
                r_naive += pr * (e_mean[j_hat] - e_mean[i_star])

    correction = 0.0
    per_cell = {}
    for (i_star, j_hat), pr in cells.items():
        if i_star == j_hat:
            continue
        d = costs[:, col[j_hat]] - costs[:, col[i_star]]
        ind = ((t_star == i_star) & (t_hat == j_hat)).astype(float)
        c = population_cov(ind, d)
        correction += c
        per_cell[f"t*={i_star},that={j_hat}"] = {
            "P_cell": pr,
            "E_D_uncond": float(d.mean()),
            "E_D_given_cell": float(d[ind > 0].mean()) if ind.sum() else None,
            "cov": c,
            "naive_contrib": pr * (e_mean[j_hat] - e_mean[i_star]),
        }

    reconciled = r_naive + correction
    resid = abs(reconciled - r_true)
    return {
        "n": n,
        "labels": labels,
        "per_model_mean": e_mean,
        "R_naive": r_naive,
        "correction": correction,
        "R_naive_plus_correction": reconciled,
        "R_true": r_true,
        "abs_residual": resid,
        "reconciles": bool(resid < 1e-9),
        "cells": per_cell,
    }


def synthetic_length_biased_example(n: int = 400, seed: int = 0) -> dict[str, Any]:
    """Router sends short items to the cheap model: naive understates cost."""
    rng = np.random.default_rng(seed)
    length = rng.lognormal(mean=2.0, sigma=0.8, size=n)
    cheap = 0.01 * length
    expensive = 0.10 * length + 1.0
    costs = np.stack([cheap, expensive], axis=1)
    # Oracle: cheap if we define "correct" always; regret vs oracle=all-cheap
    t_star = np.zeros(n, dtype=int)
    # Escalate the longest half
    t_hat = (length >= np.median(length)).astype(int)
    return decompose_regret(t_star, t_hat, costs, labels=[0, 1])


def synthetic_query_independent_example(n: int = 400, seed: int = 1) -> dict[str, Any]:
    """Random assignment: correction term should vanish in the large-n limit."""
    rng = np.random.default_rng(seed)
    length = rng.lognormal(mean=2.0, sigma=0.8, size=n)
    cheap = 0.01 * length
    expensive = 0.10 * length + 1.0
    costs = np.stack([cheap, expensive], axis=1)
    t_star = np.zeros(n, dtype=int)
    t_hat = rng.integers(0, 2, size=n)
    return decompose_regret(t_star, t_hat, costs, labels=[0, 1])

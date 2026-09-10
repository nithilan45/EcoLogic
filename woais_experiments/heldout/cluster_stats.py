"""Cluster (group) bootstrap and permutation. Independent unit = group."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from woais_experiments.statistics.bootstrap import DEFAULT_LEVEL, estimate_payload, percentile_ci
from woais_experiments.statistics.effect_sizes import cliffs_delta, cohens_dz
from woais_experiments.statistics.paired_tests import _p_from_counts, _two_sided_count


def group_members(
    ids: Sequence[str],
    group_of: Mapping[str, str],
) -> tuple[list[str], list[np.ndarray]]:
    order: dict[str, list[int]] = {}
    gids: list[str] = []
    for i, qid in enumerate(ids):
        g = str(group_of[qid])
        if g not in order:
            order[g] = []
            gids.append(g)
        order[g].append(i)
    members = [np.asarray(order[g], dtype=int) for g in gids]
    return gids, members


def _group_sums(values: np.ndarray, members: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    sums = np.asarray([float(values[idx].sum()) for idx in members], dtype=float)
    ns = np.asarray([float(idx.size) for idx in members], dtype=float)
    return sums, ns


def cluster_mean_ci(
    values: Sequence[float],
    *,
    ids: Sequence[str],
    group_of: Mapping[str, str],
    n_boot: int,
    seed: int,
    level: float = DEFAULT_LEVEL,
    name: str = "cluster_mean",
) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    gids, members = group_members(ids, group_of)
    g = len(gids)
    n = int(x.size)
    point = float(x.mean()) if n else None
    if n == 0 or g == 0:
        return estimate_payload(
            name=name, n=0, estimate=None, level=level, n_boot=n_boot,
            available=False, reason="empty", method="cluster_percentile_bootstrap", seed=seed,
        )
    if g <= 1 or n_boot <= 0:
        return estimate_payload(
            name=name, n=n, estimate=point, ci_lo=point, ci_hi=point, level=level,
            n_boot=n_boot, available=point is not None, method="cluster_percentile_bootstrap",
            seed=seed, reason="degenerate cluster bootstrap",
        )
    sums, ns = _group_sums(x, members)
    rng = np.random.default_rng(int(seed))
    draw = rng.integers(0, g, size=(int(n_boot), g), endpoint=False)
    boot = sums[draw].sum(axis=1) / np.maximum(ns[draw].sum(axis=1), 1.0)
    ci = percentile_ci(boot.tolist(), point=point, level=level, n_boot=n_boot, seed=seed)
    payload = estimate_payload(
        name=name, n=n, estimate=ci["estimate"], ci_lo=ci["ci_lo"], ci_hi=ci["ci_hi"],
        level=level, n_boot=n_boot, available=ci["available"], reason=ci.get("reason"),
        n_valid_boot=ci["n_valid_boot"], method="cluster_percentile_bootstrap", seed=seed,
        pairing="group",
    )
    payload["n_groups"] = g
    payload["bootstrap_unit"] = "group"
    return payload


def cluster_paired_mean(
    a: Sequence[float],
    b: Sequence[float],
    *,
    ids: Sequence[str],
    group_of: Mapping[str, str],
    n_boot: int,
    n_perm: int,
    seed: int,
    level: float = DEFAULT_LEVEL,
    name: str = "cluster_paired_mean_difference",
) -> dict[str, Any]:
    """Paired a-b. Bootstrap and sign-flip the group, the independent unit."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.shape != bb.shape:
        raise ValueError("paired series length mismatch")
    d = aa - bb
    gids, members = group_members(ids, group_of)
    g = len(gids)
    n = int(d.size)
    point = float(d.mean()) if n else None
    dz = cohens_dz(aa, bb)
    cliff = cliffs_delta(aa, bb, paired=True)
    if n == 0 or g == 0:
        return {
            "name": name, "n": 0, "n_groups": 0, "estimate": None, "available": False,
            "bootstrap_unit": "group", "effect_size": {"cohens_dz": dz, "cliffs_delta": cliff},
        }
    sums, ns = _group_sums(d, members)
    rng = np.random.default_rng(int(seed))
    if g > 1 and n_boot > 0:
        draw = rng.integers(0, g, size=(int(n_boot), g), endpoint=False)
        boot = sums[draw].sum(axis=1) / np.maximum(ns[draw].sum(axis=1), 1.0)
        ci = percentile_ci(boot.tolist(), point=point, level=level, n_boot=n_boot, seed=seed)
    else:
        ci = {"estimate": point, "ci_lo": point, "ci_hi": point, "available": True, "n_valid_boot": 0}

    perm: dict[str, Any]
    if g == 0 or n_perm <= 0:
        perm = {"p_raw": None, "available": False, "method": None}
    elif g <= 16:
        # Exact sign patterns on groups; query-weighted mean uses group sums.
        n_draw = 1 << g
        total_n = float(ns.sum())
        obs = float(sums.sum() / total_n)
        extreme = 0
        for mask in range(n_draw):
            signed = 0.0
            for i in range(g):
                signed += sums[i] if (mask >> i) & 1 else -sums[i]
            null = signed / total_n
            if abs(null) >= abs(obs) - 1e-15:
                extreme += 1
        perm = {
            "p_raw": _p_from_counts(extreme, n_draw, monte_carlo=False),
            "available": True,
            "method": "exact_group_sign_flip",
            "n_perm": n_draw,
            "n_extreme": extreme,
        }
    else:
        total_n = float(ns.sum())
        obs = float(sums.sum() / total_n)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_perm), g))
        nulls = (signs * sums).sum(axis=1) / total_n
        extreme = _two_sided_count(obs, nulls)
        perm = {
            "p_raw": _p_from_counts(extreme, int(n_perm), monte_carlo=True),
            "available": True,
            "method": "monte_carlo_group_sign_flip",
            "n_perm": int(n_perm),
            "n_extreme": extreme,
        }

    payload = estimate_payload(
        name=name, n=n, estimate=ci["estimate"], ci_lo=ci.get("ci_lo"), ci_hi=ci.get("ci_hi"),
        level=level, n_boot=n_boot, available=bool(ci.get("available")),
        n_valid_boot=ci.get("n_valid_boot"), method="cluster_percentile_bootstrap", seed=seed,
        difference="a - b", pairing="group",
    )
    payload["n_groups"] = g
    payload["bootstrap_unit"] = "group"
    payload["permutation_unit"] = "group"
    payload["p_raw"] = perm.get("p_raw")
    payload["permutation"] = perm
    payload["effect_size"] = {
        "cohens_dz": dz,
        "cliffs_delta": cliff,
        "note": "Effect sizes are query-level point estimates; inference unit is the group.",
    }
    return payload

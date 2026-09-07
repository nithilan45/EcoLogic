"""Stage 12 — the router-free predictability ceiling, from k=3 replicate generations.

This is the stage that answers "you tested a weak router" without testing another
router. Proposition 3 of `theory.md` caps the gain of EVERY router at
sqrt(beta(1-beta)) * sd(delta), where delta(x) = eta_l(x) - eta_s(x) is the
difference of the two models' SUCCESS PROBABILITIES on x. Proposition 4 identifies
sd(delta) from repeated generations of the same prompt.

Stage 7 generated k = 3 replicates for every (item, tier):
  - pool: 5,000 items x 3 tiers x 3 replicates at temperature 0.7 (45,000 calls)
  - test:   364 items x 3 tiers x 3 replicates at temperature 0.0  (3,276 calls)

Both are analysed. The temperature-0 set is the one that matches how routers are
deployed; the temperature-0.7 set has 13.7x the items.

Also computes, for each tier pair, how much of the ORACLE HEADROOM that a
single-generation evaluation reports is an artifact of generation noise -- the
quantity that makes published oracle numbers unattainable in principle.

Outputs: s12_ceiling.json, s12_ceiling.md
"""

from __future__ import annotations

import json
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
S710 = os.path.join(os.path.dirname(HERE), "stage7_10")
sys.path.insert(0, HERE)
from decomp import (  # noqa: E402
    bayes_auc_beta_moment_match,
    bayes_auc_bound,
    binary_cells,
    ceiling_from_sd,
    oracle_frontier_at,
    sd_delta_from_replicates,
    static_frontier_at,
    variance_components,
)

SEED = 20260907
BETAS = np.round(np.arange(0.1, 0.95, 0.1), 2)
HEADLINE_BETA = 0.5
N_BOOT = 2000
TIERS = [1, 2, 3]
TIER_NAME = {1: "Qwen/Qwen3.5-9B", 2: "openai/gpt-oss-20b", 3: "gpt-4o"}

# Held-out per-tier AUCs measured in earlier stages, for the Proposition 5 check.
# Sources: stage7_10/SUMMARY.md (Stage 5 -> Stage 7 head AUCs) and s7_ceiling.md.
MEASURED_AUC = {
    "stage5_tier1": 0.656, "stage5_tier2": 0.701,
    "stage7_tier1": 0.665, "stage7_tier2": 0.702,
    "s7c_logistic_mean": 0.6816, "s7c_gbm_mean": 0.6547,
    "s7c_rf_mean": 0.6703, "s7c_knn_mean": 0.6521,
}


def load_reps(path):
    """-> (item_ids, reps[item, tier, k] of 0/1, cost[item, tier, k] in USD)."""
    d = pd.read_csv(path)
    d["correct"] = d["correct"].fillna(False).astype(bool).astype(float)
    d = d[d.sample_idx.isin([0, 1, 2])]
    items = sorted(d.item_id.unique())
    ii = {v: i for i, v in enumerate(items)}
    n, k = len(items), 3
    reps = np.full((n, len(TIERS), k), np.nan)
    cost = np.full((n, len(TIERS), k), np.nan)
    for ti, t in enumerate(TIERS):
        sub = d[d.tier == t]
        r = sub.item_id.map(ii).to_numpy()
        s = sub.sample_idx.to_numpy(int)
        reps[r, ti, s] = sub.correct.to_numpy()
        cost[r, ti, s] = sub.usd.to_numpy()
    keep = ~np.isnan(reps).any(axis=(1, 2)) & ~np.isnan(cost).any(axis=(1, 2))
    return [items[i] for i in np.flatnonzero(keep)], reps[keep], cost[keep]


def analyse(reps, cost, label, benchmarks=None):
    """The full Stage 12 table for one replicate set."""
    n, T, k = reps.shape
    eta_hat = reps.mean(axis=2)              # (n, T) noisy estimate of eta
    gam_hat = cost.mean(axis=2)              # (n, T) noisy estimate of gamma
    out = {"label": label, "n_items": int(n), "k": int(k), "tiers": TIERS,
           "tier_names": TIER_NAME}

    # ---- per-tier variance components and the implied Bayes AUC ------------
    per_tier = {}
    for ti, t in enumerate(TIERS):
        vc = variance_components(reps[:, ti, :])
        auc_bound = bayes_auc_bound(vc["sd_between"], vc["mean"])
        mm = bayes_auc_beta_moment_match(vc["mean"], vc["sd_between"])
        naive_sd = float(np.sqrt(vc["var_eta_hat"]))
        per_tier[f"tier{t}"] = {
            "model": TIER_NAME[t], **vc,
            "flip_rate": float(np.mean((reps[:, ti, :].min(axis=1) !=
                                        reps[:, ti, :].max(axis=1)))),
            "auc_star_upper_bound": auc_bound,
            "auc_star_beta_moment_match": mm.get("auc_star"),
            "auc_star_beta_feasible": mm.get("feasible"),
            "auc_bound_if_noise_ignored": bayes_auc_bound(naive_sd, vc["mean"]),
        }
    out["per_tier"] = per_tier

    # ---- per-pair ceiling vs headroom -------------------------------------
    rng = np.random.default_rng(SEED)
    pairs = {}
    for ia, ib in combinations(range(T), 2):
        lo, hi = (ia, ib) if gam_hat[:, ia].mean() <= gam_hat[:, ib].mean() else (ib, ia)
        key = f"tier{TIERS[lo]}_vs_tier{TIERS[hi]}"
        sd = sd_delta_from_replicates(reps[:, lo, :], reps[:, hi, :])

        # single-draw view: exactly what a one-generation dataset would contain
        u1 = reps[:, :, 0][:, [lo, hi]]
        c1 = cost[:, :, 0][:, [lo, hi]]
        # replicate-mean view: the best available estimate of (eta, gamma)
        um = eta_hat[:, [lo, hi]]
        cm = gam_hat[:, [lo, hi]]

        rec = {"cheap_tier": TIERS[lo], "exp_tier": TIERS[hi],
               "cheap_model": TIER_NAME[TIERS[lo]], "exp_model": TIER_NAME[TIERS[hi]],
               **{f"sd_{a}": b for a, b in sd.items()},
               "cells_single_draw": binary_cells(u1[:, 0], u1[:, 1]),
               "betas": {}}

        for beta in BETAS:
            b_single = float((1 - beta) * c1[:, 0].mean() + beta * c1[:, 1].mean())
            S1 = static_frontier_at(c1.mean(axis=0), u1.mean(axis=0), b_single)
            A1, _, _ = oracle_frontier_at(u1, c1, b_single)
            b_mean = float((1 - beta) * cm[:, 0].mean() + beta * cm[:, 1].mean())
            Sm = static_frontier_at(cm.mean(axis=0), um.mean(axis=0), b_mean)
            Am, _, _ = oracle_frontier_at(um, cm, b_mean)

            # split-replicate: choose with replicate 0, pay/score on replicates 1-2.
            # An unbiased estimate of the value of a policy given ONE noisy look --
            # not a bound on A+, but the natural empirical reference point.
            sel_u = reps[:, :, 0][:, [lo, hi]]
            ev_u = reps[:, :, 1:].mean(axis=2)[:, [lo, hi]]
            ev_c = cost[:, :, 1:].mean(axis=2)[:, [lo, hi]]
            from decomp import router_value_at
            split_v, _ = router_value_at(ev_u, ev_c, sel_u, b_mean)
            S_ev = static_frontier_at(ev_c.mean(axis=0), ev_u.mean(axis=0), b_mean)

            ceil_true = ceiling_from_sd(sd["sd_delta"], beta)
            ceil_naive = ceiling_from_sd(sd["sd_delta_observed"], beta)
            rec["betas"][f"{beta:.1f}"] = {
                "kappa_single_draw": float(A1 - S1),
                "kappa_replicate_mean": float(Am - Sm),
                "ceiling_any_router": ceil_true,
                "ceiling_if_noise_ignored": ceil_naive,
                "ceiling_over_kappa_single": (float(ceil_true / (A1 - S1))
                                              if A1 - S1 > 1e-12 else float("nan")),
                "split_replicate_gain": float(split_v - S_ev),
            }

        # bootstrap the headline beta
        bh = f"{HEADLINE_BETA:.1f}"
        boots = {"sd_delta": [], "ceiling": [], "kappa_single": [], "ratio": []}
        for _ in range(N_BOOT):
            idx = rng.integers(0, reps.shape[0], reps.shape[0])
            s_b = sd_delta_from_replicates(reps[idx][:, lo, :], reps[idx][:, hi, :])
            u1b, c1b = reps[idx][:, :, 0][:, [lo, hi]], cost[idx][:, :, 0][:, [lo, hi]]
            bb = float((1 - HEADLINE_BETA) * c1b[:, 0].mean()
                       + HEADLINE_BETA * c1b[:, 1].mean())
            Sb = static_frontier_at(c1b.mean(axis=0), u1b.mean(axis=0), bb)
            Ab, _, _ = oracle_frontier_at(u1b, c1b, bb)
            cb = ceiling_from_sd(s_b["sd_delta"], HEADLINE_BETA)
            boots["sd_delta"].append(s_b["sd_delta"])
            boots["ceiling"].append(cb)
            boots["kappa_single"].append(Ab - Sb)
            boots["ratio"].append(cb / (Ab - Sb) if Ab - Sb > 1e-12 else np.nan)
        rec["bootstrap_beta_0.5"] = {
            key2: {"lo": float(np.nanpercentile(v, 2.5)),
                   "hi": float(np.nanpercentile(v, 97.5))}
            for key2, v in boots.items()}
        rec["n_clipped_bootstrap_sd"] = int(np.sum(np.asarray(boots["sd_delta"]) <= 0))
        pairs[key] = rec
    out["pairs"] = pairs

    if benchmarks is not None:
        by_bm = {}
        for bm in sorted(set(benchmarks)):
            sel = np.asarray(benchmarks) == bm
            if sel.sum() < 50:
                continue
            sub = {}
            for ia, ib in combinations(range(T), 2):
                lo, hi = (ia, ib) if gam_hat[:, ia].mean() <= gam_hat[:, ib].mean() else (ib, ia)
                sd = sd_delta_from_replicates(reps[sel][:, lo, :], reps[sel][:, hi, :])
                u1 = reps[sel][:, :, 0][:, [lo, hi]]
                c1 = cost[sel][:, :, 0][:, [lo, hi]]
                bb = float(0.5 * c1[:, 0].mean() + 0.5 * c1[:, 1].mean())
                S1 = static_frontier_at(c1.mean(axis=0), u1.mean(axis=0), bb)
                A1, _, _ = oracle_frontier_at(u1, c1, bb)
                ct = ceiling_from_sd(sd["sd_delta"], 0.5)
                sub[f"tier{TIERS[lo]}_vs_tier{TIERS[hi]}"] = {
                    "n": int(sel.sum()), "sd_delta": sd["sd_delta"],
                    "sd_delta_observed": sd["sd_delta_observed"],
                    "noise_share": sd["noise_share"],
                    "kappa_single_draw": float(A1 - S1), "ceiling_any_router": ct,
                    "ceiling_over_kappa": (float(ct / (A1 - S1))
                                           if A1 - S1 > 1e-12 else float("nan")),
                }
            by_bm[bm] = sub
        out["by_benchmark"] = by_bm
    return out


def main():
    res = {"seed": SEED, "n_boot": N_BOOT, "headline_beta": HEADLINE_BETA,
           "measured_auc_earlier_stages": MEASURED_AUC, "sets": {}}

    for label, path, temp in [
        ("pool_temp0.7", os.path.join(S710, "s7_pool_samples.csv.gz"), 0.7),
        ("test_temp0.0", os.path.join(S710, "s7_test_samples.csv.gz"), 0.0),
    ]:
        print(f"--- {label} ---", flush=True)
        items, reps, cost = load_reps(path)
        bm = [i.split("/")[0] for i in items]
        r = analyse(reps, cost, label, benchmarks=bm)
        r["temperature"] = temp
        res["sets"][label] = r
        for key, rec in r["pairs"].items():
            b = rec["betas"][f"{HEADLINE_BETA:.1f}"]
            print(f"  {key:22s} sd_delta {rec['sd_sd_delta']:.4f} "
                  f"(observed {rec['sd_sd_delta_observed']:.4f}, "
                  f"noise {100*rec['sd_noise_share']:.0f}%)  "
                  f"ceiling {100*b['ceiling_any_router']:.2f} pp  "
                  f"kappa_single {100*b['kappa_single_draw']:.2f} pp  "
                  f"ratio {b['ceiling_over_kappa_single']:.3f}", flush=True)

    # ---- H3 verdict, on the pre-registered rule ---------------------------
    ratios = []
    for label, r in res["sets"].items():
        for key, rec in r["pairs"].items():
            v = rec["betas"][f"{HEADLINE_BETA:.1f}"]["ceiling_over_kappa_single"]
            if np.isfinite(v):
                ratios.append({"set": label, "pair": key, "ratio": v})
    med = float(np.median([x["ratio"] for x in ratios]))
    res["H3"] = {
        "rule": "ceiling on any router <= 50% of single-draw kappa, at beta=0.5",
        "median_ceiling_over_kappa": med,
        "all": ratios,
        "verdict": "SUPPORTED" if med <= 0.5 else "NOT SUPPORTED",
    }

    # ---- Proposition 5 consistency check ----------------------------------
    preds = []
    for label, r in res["sets"].items():
        for tk, rec in r["per_tier"].items():
            preds.append({"set": label, "tier": tk,
                          "auc_star_bound": rec["auc_star_upper_bound"],
                          "auc_star_point": rec["auc_star_beta_moment_match"]})
    plateau = float(np.mean([MEASURED_AUC[k] for k in
                             ("stage7_tier1", "stage7_tier2", "s7c_logistic_mean")]))
    pool_t12 = [p for p in preds if p["set"] == "pool_temp0.7" and p["tier"] in ("tier1", "tier2")]
    mean_bound = float(np.mean([p["auc_star_bound"] for p in pool_t12]))
    mean_point = float(np.mean([p["auc_star_point"] for p in pool_t12
                                if p["auc_star_point"] is not None]))
    res["P5_check"] = {
        "rule": "replicate-implied AUC* within 0.05 of the measured 0.65-0.68 plateau",
        "measured_plateau_mean": plateau,
        "implied_auc_star_upper_bound_tiers12_pool": mean_bound,
        "implied_auc_star_point_tiers12_pool": mean_point,
        "abs_diff_point_vs_plateau": abs(mean_point - plateau),
        "verdict": "CONSISTENT" if abs(mean_point - plateau) <= 0.05 else "INCONSISTENT",
        "all": preds,
    }

    with open(os.path.join(HERE, "s12_ceiling.json"), "w") as f:
        json.dump(res, f, indent=1)

    print("\n=== H3 ===")
    print(f"median ceiling / single-draw kappa = {med:.3f} -> {res['H3']['verdict']}")
    print("=== Proposition 5 ===")
    print(f"measured plateau {plateau:.4f}; replicate-implied AUC* point estimate "
          f"{mean_point:.4f} (upper bound {mean_bound:.4f}) -> {res['P5_check']['verdict']}")


if __name__ == "__main__":
    main()

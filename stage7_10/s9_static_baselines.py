"""Stage 9b — do static policies beat a *published* router, at matched cost?

Stage 9 validated the Stage 8 correction term on RouteLLM's data. It did not test
the other headline claim -- that routers must be measured against static
single-tier policies rather than against always-frontier -- which so far exists
only on our own 364-item audit. That makes the claim a case study. This script
tests it on someone else's router, models and data.

Two comparisons, on RouteLLM's released GSM8K responses:

  (a) MATCHED CALL FRACTION. The router vs. random assignment sending the same
      fraction of queries to the strong model. This is essentially RouteLLM's own
      APGR baseline, and the router should win here.

  (b) MATCHED COST. The router vs. random assignment whose *expected dollar cost*
      equals the router's *actual* dollar cost. This is the comparison the naive
      accounting cannot see, and it is strictly the fairer one.

(a) and (b) differ for a reason that is exactly the Stage 8 mechanism, and it is
worth stating precisely because it is the point of this script:

    For random routing, assignment is independent of the item, so
    E[cost] = f*E[c_strong] + (1-f)*E[c_weak] holds EXACTLY -- the naive
    per-model-mean formula is unbiased for random.

    For a real router, assignment correlates with per-item length, so the same
    formula is biased (Stage 8/9: it understated true cost at every non-degenerate
    operating point here).

So the standard "router beats random at equal call fraction" comparison is tilted
in the router's favour: it charges the router its per-model mean while the
baseline's mean is exact. Matching on realised cost removes that tilt.

No new API calls: RouteLLM's own released response text, their own decontamination
list, their own released BERT checkpoint, their own threshold grid.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from external_check import (PRICES, RL, STRONG, WEAK, add_costs, bert_win_rates,
                            load_benchmark)

OUT = Path(__file__).resolve().parent


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact two-sided McNemar on paired boolean correctness vectors."""
    from scipy.stats import binomtest
    b_only = int((a & ~b).sum())
    c_only = int((~a & b).sum())
    n = b_only + c_only
    p = 1.0 if n == 0 else float(binomtest(b_only, n, 0.5).pvalue)
    return {"a_only_correct": b_only, "b_only_correct": c_only, "p_exact": p}


MC_DRAWS = 20000


def mc_pvalue(t_hat: np.ndarray, f_matched: float, ok_w: np.ndarray,
              ok_s: np.ndarray, router_acc: float, seed: int = 20260907) -> dict:
    """Sampling distribution of the matched-cost query-independent mixture.

    The mixture's accuracy is usually quoted as its expectation f*acc_s +
    (1-f)*acc_w, but a realised random assignment is a draw, so a point
    comparison against the expectation is not a test. Draw MC_DRAWS assignments
    that send each item to the strong model independently with probability
    f_matched, and report the one-sided fraction that match or beat the router.
    """
    rng = np.random.default_rng(seed)
    n = len(ok_w)
    accs = np.empty(MC_DRAWS)
    for d in range(MC_DRAWS):
        pick = rng.random(n) < f_matched
        accs[d] = np.where(pick, ok_s, ok_w).mean()
    return {
        "mc_baseline_mean_pct": float(100 * accs.mean()),
        "mc_baseline_sd_pp": float(100 * accs.std()),
        "mc_p_value": float((accs >= router_acc).mean()),
    }


def main() -> None:
    if not RL.exists():
        raise SystemExit("RouteLLM repo not cloned at /tmp/routellm_chk")

    df = add_costs(load_benchmark("gsm8k"))
    wr = bert_win_rates(df["prompt"].astype(str).tolist(), Path("/tmp/rl_wr_gsm8k.npy"))

    c_w = df["cost_weak"].to_numpy()
    c_s = df["cost_strong"].to_numpy()
    ok_w = df[WEAK].to_numpy().astype(bool)
    ok_s = df[STRONG].to_numpy().astype(bool)
    n = len(df)

    acc_w, acc_s = float(ok_w.mean()), float(ok_s.mean())
    mc_w, mc_s = float(c_w.mean()), float(c_s.mean())

    # ---- static policies. For a two-model pool the static family is: always-weak,
    # always-strong, and any query-independent mixture of the two. The mixture line
    # is the static frontier a router has to beat to justify reading the query.
    statics = {
        "always_weak": {"accuracy": acc_w, "cost_per_item": mc_w,
                        "n_correct": int(ok_w.sum())},
        "always_strong": {"accuracy": acc_s, "cost_per_item": mc_s,
                          "n_correct": int(ok_s.sum())},
    }
    for k, v in statics.items():
        lo, hi = wilson(v["n_correct"], n)
        v["wilson"] = [lo, hi]

    # Sanity: the naive per-model-mean formula is EXACT for query-independent
    # assignment. Verify empirically on a random draw rather than asserting it.
    rng = np.random.default_rng(20260907)
    f_probe = 0.5
    draw = rng.random(n) < f_probe
    exact_cost = float(np.where(draw, c_s, c_w).mean())
    naive_cost = float(draw.mean() * mc_s + (1 - draw.mean()) * mc_w)
    naive_exact_for_random_residual = abs(exact_cost - naive_cost)

    def random_acc_at_cost(cost: float) -> tuple[float, float]:
        """Expected accuracy of the query-independent mixture with this cost."""
        f = (cost - mc_w) / (mc_s - mc_w)
        f = float(np.clip(f, 0.0, 1.0))
        return f * acc_s + (1 - f) * acc_w, f

    _, thresholds = pd.qcut(wr, 10, retbins=True, duplicates="drop")
    rows = []
    for thr in thresholds:
        t_hat = wr >= thr
        f_router = float(t_hat.mean())
        router_ok = np.where(t_hat, ok_s, ok_w)
        router_acc = float(router_ok.mean())
        router_cost = float(np.where(t_hat, c_s, c_w).mean())

        # (a) matched call fraction: random sending the same fraction to strong.
        rand_acc_frac = f_router * acc_s + (1 - f_router) * acc_w
        rand_cost_frac = f_router * mc_s + (1 - f_router) * mc_w

        # (b) matched cost: random whose expected cost equals the router's actual.
        rand_acc_cost, f_matched = random_acc_at_cost(router_cost)

        rows.append({
            "threshold": float(thr),
            "strong_pct": 100 * f_router,
            "router_accuracy_pct": 100 * router_acc,
            "router_cost_per_item_usd": router_cost,
            # (a)
            "random_same_fraction_accuracy_pct": 100 * rand_acc_frac,
            "router_minus_random_same_fraction_pp": 100 * (router_acc - rand_acc_frac),
            "random_same_fraction_cost_per_item_usd": rand_cost_frac,
            # (b)
            "random_matched_cost_fraction": f_matched,
            "random_matched_cost_accuracy_pct": 100 * rand_acc_cost,
            "router_minus_random_matched_cost_pp": 100 * (router_acc - rand_acc_cost),
            # how much of the router's apparent edge is the accounting tilt
            "advantage_lost_to_cost_matching_pp":
                100 * ((router_acc - rand_acc_frac) - (router_acc - rand_acc_cost)),
            # Monte-Carlo p-value: the matched-cost mixture is an *expectation*, so
            # comparing a point to it is not a test. Draw the baseline's sampling
            # distribution and ask where the router falls.
            **mc_pvalue(t_hat, f_matched, ok_w, ok_s, router_acc),
        })

    # For a two-model pool, "dominates always-weak on both axes" is vacuous:
    # always-weak is the cheapest policy that exists, so nothing can undercut it.
    # The meaningful static bar is the query-independent MIXTURE line at matched
    # cost, i.e. router_minus_random_matched_cost_pp > 0.
    beats_mix = [r for r in rows if r["router_minus_random_matched_cost_pp"] > 0
                 and 0.5 < r["strong_pct"] < 99.5]
    interior = [r for r in rows if 0.5 < r["strong_pct"] < 99.5]
    sig_beats_mix = [r for r in beats_mix if r["mc_p_value"] < 0.05]

    # No single operating point may clear 0.05, yet a consistent sign across the
    # sweep is itself evidence. Operating points share items so they are not
    # independent; this is a descriptive consistency check, not an exact test.
    from scipy.stats import binomtest
    sign_test = binomtest(len(beats_mix), len(interior), 0.5)

    # Paired significance at the operating point closest to 50% strong, which is
    # the mid-sweep point a cost-quality paper is most likely to quote.
    mid = min(rows, key=lambda r: abs(r["strong_pct"] - 50))
    t_hat_mid = wr >= mid["threshold"]
    router_ok_mid = np.where(t_hat_mid, ok_s, ok_w)
    mcn_vs_weak = mcnemar_exact(router_ok_mid.astype(bool), ok_w)
    mcn_vs_strong = mcnemar_exact(router_ok_mid.astype(bool), ok_s)

    worst = max(rows, key=lambda r: r["advantage_lost_to_cost_matching_pp"])

    result = {
        "source": "RouteLLM released GSM8K responses (Ong et al., ICLR 2025)",
        "repo_commit_note": "github.com/lm-sys/RouteLLM, per stage7_10/external_generalization.md",
        "n_items": n,
        "weak_model": WEAK, "strong_model": STRONG, "prices_usd_per_1m": PRICES,
        "accuracy": {"always_weak_pct": 100 * acc_w, "always_strong_pct": 100 * acc_s},
        "mean_cost_per_item_usd": {"weak": mc_w, "strong": mc_s},
        "statics": statics,
        "naive_formula_exact_for_random_residual_usd": naive_exact_for_random_residual,
        "sweep": rows,
        "mc_draws": MC_DRAWS,
        "n_interior_points": len(interior),
        "n_interior_beating_matched_cost_mixture": len(beats_mix),
        "n_interior_beating_matched_cost_mixture_p05": len(sig_beats_mix),
        "sign_test_across_sweep": {
            "n_positive": len(beats_mix), "n_interior": len(interior),
            "p_two_sided": float(sign_test.pvalue),
            "note": ("operating points share items and are not independent, so this "
                     "is a descriptive consistency check, not an exact test"),
        },
        "mean_router_edge_matched_cost_pp": float(np.mean(
            [r["router_minus_random_matched_cost_pp"] for r in interior])),
        "max_advantage_lost_to_cost_matching_pp":
            worst["advantage_lost_to_cost_matching_pp"],
        "max_advantage_lost_at_strong_pct": worst["strong_pct"],
        "mid_point": {
            "strong_pct": mid["strong_pct"],
            "router_accuracy_pct": mid["router_accuracy_pct"],
            "mcnemar_router_vs_always_weak": mcn_vs_weak,
            "mcnemar_router_vs_always_strong": mcn_vs_strong,
        },
    }

    with open(OUT / "s9_static_baselines.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"n={n}  always-weak {100*acc_w:.2f}% @ ${mc_w:.6f}   "
          f"always-strong {100*acc_s:.2f}% @ ${mc_s:.6f}")
    print(f"naive-formula-exact-for-random residual: "
          f"${naive_exact_for_random_residual:.2e} (confirms the asymmetry)")
    print()
    print(f"{'strong%':>8} {'router%':>8} {'$/item':>10} "
          f"{'vs rand(frac)':>14} {'vs rand($)':>11} {'edge lost':>10} {'mc p':>8}")
    for r in rows:
        print(f"{r['strong_pct']:8.1f} {r['router_accuracy_pct']:8.2f} "
              f"{r['router_cost_per_item_usd']:10.6f} "
              f"{r['router_minus_random_same_fraction_pp']:+14.2f} "
              f"{r['router_minus_random_matched_cost_pp']:+11.2f} "
              f"{r['advantage_lost_to_cost_matching_pp']:10.2f} "
              f"{r['mc_p_value']:8.3f}")
    print()
    print(f"interior operating points beating the matched-cost static mixture: "
          f"{len(beats_mix)}/{len(interior)}  "
          f"(at p<0.05: {len(sig_beats_mix)}/{len(interior)})")
    print(f"sign test across interior sweep: {len(beats_mix)}/{len(interior)} positive, "
          f"p={sign_test.pvalue:.3g}; mean edge "
          f"{np.mean([r['router_minus_random_matched_cost_pp'] for r in interior]):+.2f} pp")
    print(f"largest edge lost to cost-matching: "
          f"{worst['advantage_lost_to_cost_matching_pp']:.2f} pp "
          f"at {worst['strong_pct']:.0f}% strong")
    print(f"mid point ({mid['strong_pct']:.0f}% strong): "
          f"vs always-weak p={mcn_vs_weak['p_exact']:.3g}, "
          f"vs always-strong p={mcn_vs_strong['p_exact']:.3g}")
    print("wrote stage7_10/s9_static_baselines.json")


if __name__ == "__main__":
    main()

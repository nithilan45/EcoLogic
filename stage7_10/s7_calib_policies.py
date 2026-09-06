"""CALIBRATION-split policy comparison for Stage 7.

This is *not* the pre-registered one-shot evaluation -- that runs on the frozen
test set in s7_final.py and is still pending Tier 3 generation. This exists so
the part of Stage 7 whose data is complete (the 1,500-item CALIBRATION split)
can be reported with the same statistics as the Stage 5 table, rather than
reported only as a pair of gap numbers.

No policy or hyperparameter is selected here. The router, its variant and its
threshold were all fixed earlier: variant by TRAIN cross-validation only
(s7_fit.py), threshold by the pre-registered budget rule (s7_calibrate.py).

Usage: python3 stage7_10/s7_calib_policies.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in ("benchmark", "router_v2", "stage7_10"):
    sys.path.insert(0, str(ROOT / p))
from analyze import mcnemar, wilson  # noqa: E402
from s7_data import PAPER_RATES, load_split, pool_correct_and_tokens  # noqa: E402
from s7_fit import make_xy  # noqa: E402
from train_router import predict  # noqa: E402

OUT = ROOT / "stage7_10"
SEED = 20260906


def main() -> None:
    items = load_split("calibration")
    correct, tokens = pool_correct_and_tokens()
    n = len(items)
    ids = [it["item_id"] for it in items]

    def e_of(tier: int, i: int) -> float:
        return tokens[(tier, ids[i])] / 1000 * PAPER_RATES[tier]

    with open(OUT / "s7_router_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    chosen = json.loads((OUT / "s7_chosen_threshold.json").read_text())
    tau = chosen["chosen"]["tau"]
    order = chosen["candidate_order"]

    _, X, _ = make_xy(items, correct)
    probs = predict(bundle["model"], X)

    # fixed router: first candidate tier whose P(correct) clears tau, else Tier 3
    router_tier = []
    for i in range(n):
        pick = 3
        for t in order:
            if probs[t][i] >= tau:
                pick = t
                break
        router_tier.append(pick)

    rng = np.random.default_rng(SEED)
    rand_tier = rng.integers(1, 4, size=n)

    oracle_tier = []
    for i in range(n):
        cands = [t for t in (1, 2, 3) if correct[(t, ids[i])]]
        pool = cands or [1, 2, 3]
        oracle_tier.append(min(pool, key=lambda t: e_of(t, i)))

    policies = {
        "Always Tier 1": [1] * n,
        "Always Tier 2": [2] * n,
        "Always Tier 3 (frontier)": [3] * n,
        "Random": list(rand_tier),
        "Oracle": oracle_tier,
        "Learned router (Stage 7)": router_tier,
    }

    vec = {name: np.array([correct[(int(t), ids[i])] for i, t in enumerate(tiers)], dtype=bool)
           for name, tiers in policies.items()}
    en = {name: float(sum(e_of(int(t), i) for i, t in enumerate(tiers)))
          for name, tiers in policies.items()}
    frontier_e = en["Always Tier 3 (frontier)"]

    rows = []
    for name, tiers in policies.items():
        k = int(vec[name].sum())
        _, lo, hi = wilson(k, n)
        mix = {t: int(sum(1 for x in tiers if x == t)) for t in (1, 2, 3)}
        rows.append({"policy": name, "n_correct": k, "accuracy": k / n,
                     "wilson_lo": lo, "wilson_hi": hi,
                     "energy_J": en[name], "energy_pct_of_frontier": 100 * en[name] / frontier_e,
                     "energy_J_per_item": en[name] / n, "tier_mix": mix})

    ref = "Learned router (Stage 7)"
    tests = {}
    for name in policies:
        if name == ref:
            continue
        m = mcnemar(list(vec[ref]), list(vec[name]))
        tests[name] = {"router_only_correct": m["b"], "other_only_correct": m["c"],
                       "p_exact": m["p_value"],
                       "acc_diff_pp": 100 * (vec[ref].mean() - vec[name].mean()),
                       "energy_diff_pct": 100 * (en[ref] - en[name]) / en[name]}

    gaps = json.loads((OUT / "s7_mckp_gaps.json").read_text())
    out = {"note": ("CALIBRATION split, not the frozen test set. Reported because Stage 7's "
                    "pool generation is complete while the frozen test set is still missing "
                    "Tier 3. The pre-registered S1/S2 verdict is decided on the frozen test "
                    "set only, and is not decided here."),
           "n_items": n, "tau": tau, "candidate_order": order,
           "rows": rows, "mcnemar_vs_learned_router": tests,
           "calibration_gap_pp": gaps["calibration_gap_pp"],
           "discreteness_gap_pp": gaps["discreteness_gap_at_router_energy_pp"]}
    (OUT / "s7_calib_policies.json").write_text(json.dumps(out, indent=2))

    print(f"CALIBRATION policy comparison (n={n}, tau={tau:.4f})\n")
    print(f"{'policy':<26} {'acc':>7} {'95% CI':>16} {'energy J':>11} {'%front':>7} {'mix 1/2/3':>16}")
    for r in rows:
        m = r["tier_mix"]
        print(f"{r['policy']:<26} {r['accuracy']:>7.4f} "
              f"[{r['wilson_lo']:.4f},{r['wilson_hi']:.4f}] {r['energy_J']:>11,.1f} "
              f"{r['energy_pct_of_frontier']:>6.1f}% {m[1]:>5}/{m[2]}/{m[3]}")
    print(f"\nMcNemar exact vs {ref}:")
    for name, t in tests.items():
        print(f"  vs {name:<26} diff={t['acc_diff_pp']:+6.2f}pp  b={t['router_only_correct']:>3} "
              f"c={t['other_only_correct']:>3}  p={t['p_exact']:.4g}  dE={t['energy_diff_pct']:+7.1f}%")
    print(f"\ncalibration gap = {out['calibration_gap_pp']:+.2f} pp   "
          f"discreteness gap = {out['discreteness_gap_pp']:+.2f} pp")
    print("wrote stage7_10/s7_calib_policies.json")


if __name__ == "__main__":
    main()

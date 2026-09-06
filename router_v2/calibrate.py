"""Stage 3 - threshold calibration on the CALIBRATION split only.

Routing rule (one continuous knob, exactly one threshold tau):

    P(Tier 1 correct) >= tau        -> Tier 1
    else P(Tier 2 correct) >= tau   -> Tier 2
    else                            -> Tier 3

tau is swept over 50 values. The threshold carried forward is chosen by the
rule fixed in PREREGISTRATION.md before any of this was run:

    highest CALIBRATION accuracy among thresholds whose CALIBRATION energy is
    <= 10% of always-frontier energy on the same split; ties -> lower energy.

Writes router_v2/threshold_sweep.csv, threshold_sweep.png, chosen_threshold.json.
"""

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "router_v2"))
sys.path.insert(0, str(ROOT / "benchmark"))
from api import MODELS  # noqa: E402
from train_router import load_correct, load_split, make_xy, predict  # noqa: E402

OUT = ROOT / "router_v2"
TIERS = [1, 2, 3]
PAPER_RATES = {t: MODELS[t]["paper_energy_per_1k"] for t in TIERS}

ENERGY_BUDGET_FRAC = 0.10  # X from the pre-registration
N_GRID = 50


def load_tokens() -> dict:
    tok = {}
    with open(OUT / "pool_graded.jsonl") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                tok[(r["tier"], r["item_id"])] = int(r.get("total_tokens") or 0)
    return tok


def route(p1: np.ndarray, p2: np.ndarray, tau: float) -> np.ndarray:
    t = np.full(len(p1), 3, dtype=int)
    t[p2 >= tau] = 2
    t[p1 >= tau] = 1
    return t


def evaluate(tiers, items, correct, tokens):
    acc = np.mean([correct[(int(t), it["item_id"])] for t, it in zip(tiers, items)])
    energy = sum(tokens[(int(t), it["item_id"])] / 1000 * PAPER_RATES[int(t)]
                 for t, it in zip(tiers, items))
    mix = {t: int(np.sum(tiers == t)) for t in TIERS}
    return float(acc), float(energy), mix


def main():
    with open(OUT / "router_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model, variant = bundle["model"], bundle["variant"]

    correct = load_correct()
    tokens = load_tokens()
    cal_items, Xca, yca = make_xy(load_split("calibration"), correct)
    p = predict(model, Xca)
    p1, p2 = p[1], p[2]

    e_frontier = sum(tokens[(3, it["item_id"])] / 1000 * PAPER_RATES[3] for it in cal_items)
    budget = ENERGY_BUDGET_FRAC * e_frontier
    print(f"variant={variant}  CALIBRATION n={len(cal_items)}")
    print(f"always-frontier calibration energy = {e_frontier:,.1f} J; "
          f"pre-registered budget = {ENERGY_BUDGET_FRAC:.0%} = {budget:,.1f} J")

    grid = np.linspace(0.0, 1.0, N_GRID)
    rows = []
    for tau in grid:
        tiers = route(p1, p2, tau)
        acc, energy, mix = evaluate(tiers, cal_items, correct, tokens)
        rows.append({
            "tau": round(float(tau), 6),
            "accuracy": acc,
            "energy_J": energy,
            "energy_frac_of_frontier": energy / e_frontier,
            "n_tier1": mix[1], "n_tier2": mix[2], "n_tier3": mix[3],
            "within_budget": energy <= budget,
        })

    with open(OUT / "threshold_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    eligible = [r for r in rows if r["within_budget"]]
    if eligible:
        best = max(eligible, key=lambda r: (r["accuracy"], -r["energy_J"]))
        rule_note = "pre-registered rule applied as written"
    else:
        best = min(rows, key=lambda r: r["energy_J"])
        rule_note = ("NO threshold met the <=10% energy constraint; fell back to the "
                     "lowest-energy threshold, as the pre-registration specifies")
    print(f"\nchosen tau = {best['tau']:.4f}  acc={best['accuracy']:.4f}  "
          f"energy={best['energy_J']:,.1f} J ({best['energy_frac_of_frontier']:.1%} of frontier)  "
          f"mix={best['n_tier1']}/{best['n_tier2']}/{best['n_tier3']}")
    print(f"  {rule_note}; {len(eligible)}/{len(rows)} thresholds were within budget")

    with open(OUT / "chosen_threshold.json", "w") as f:
        json.dump({
            "variant": variant,
            "rule": ("highest CALIBRATION accuracy among thresholds with CALIBRATION energy "
                     f"<= {ENERGY_BUDGET_FRAC:.0%} of always-frontier energy; "
                     "ties broken toward lower energy"),
            "rule_note": rule_note,
            "energy_budget_frac": ENERGY_BUDGET_FRAC,
            "calibration_frontier_energy_J": e_frontier,
            "calibration_budget_J": budget,
            "n_eligible_thresholds": len(eligible),
            "chosen": best,
        }, f, indent=2)

    # ---- chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    taus = [r["tau"] for r in rows]
    ax[0].plot(taus, [r["accuracy"] for r in rows], "o-", ms=3, color="#1f77b4",
               label="accuracy")
    ax[0].axvline(best["tau"], ls="--", color="crimson",
                  label=f"chosen tau={best['tau']:.3f}")
    ax[0].set_xlabel("threshold tau"); ax[0].set_ylabel("CALIBRATION accuracy")
    ax[0].set_title("Accuracy vs threshold")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)

    ax2 = ax[1]
    ax2.plot([r["energy_J"] for r in rows], [r["accuracy"] for r in rows], "o-", ms=3,
             color="#2ca02c", label="router threshold sweep")
    ax2.axvline(budget, ls=":", color="k", label=f"pre-registered budget ({ENERGY_BUDGET_FRAC:.0%})")
    ax2.plot([best["energy_J"]], [best["accuracy"]], "*", ms=16, color="crimson",
             label="chosen threshold")
    ax2.set_xscale("log")
    ax2.set_xlabel("CALIBRATION energy (J, log scale)"); ax2.set_ylabel("accuracy")
    ax2.set_title("Realized (energy, accuracy) curve")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig.suptitle(f"Stage 3 threshold calibration — variant {variant} (CALIBRATION split only)")
    fig.tight_layout()
    fig.savefig(OUT / "threshold_sweep.png", dpi=150)
    print("wrote router_v2/threshold_sweep.csv, threshold_sweep.png, chosen_threshold.json")


if __name__ == "__main__":
    main()

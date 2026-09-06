"""Stage 3 - threshold calibration on the CALIBRATION split only.

Routing rule (one continuous knob, exactly one threshold tau): consider the
candidate tiers in ascending order of expected energy and take the first whose
predicted P(correct) clears tau; if none clears it, escalate to Tier 3.

The candidate ordering is derived from TRAIN-split mean energy per tier only.
On this pool that ordering is Tier 2 < Tier 1 < Tier 3: Tier 2 is cheaper than
Tier 1 despite a 3x higher per-token rate, because Tier 1 emits roughly 6x more
tokens. So the rule is

    P(Tier 2 correct) >= tau        -> Tier 2
    else P(Tier 1 correct) >= tau   -> Tier 1
    else                            -> Tier 3

The naive tier-index ordering (Tier 1 checked first) is swept too and reported
as an ablation. It is structurally unable to reach the pre-registered energy
budget, because its cheapest reachable policy is always-Tier-1, which already
costs more than always-Tier-2. Routing-rule *structure* was not pre-registered;
the threshold-selection rule below was, and is applied unchanged.

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
from train_router import load_correct, load_split, make_xy, open_graded, predict  # noqa: E402

OUT = ROOT / "router_v2"
TIERS = [1, 2, 3]
PAPER_RATES = {t: MODELS[t]["paper_energy_per_1k"] for t in TIERS}

ENERGY_BUDGET_FRAC = 0.10  # X from the pre-registration
N_GRID = 50


def load_tokens() -> dict:
    tok = {}
    with open_graded() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                tok[(r["tier"], r["item_id"])] = int(r.get("total_tokens") or 0)
    return tok


def route(p1: np.ndarray, p2: np.ndarray, tau: float,
          order: tuple[int, ...] = (2, 1)) -> np.ndarray:
    """Take the first tier in `order` whose P(correct) clears tau, else Tier 3.

    `order` is cheapest-expected-energy-first. Assigning in reverse means the
    earlier (cheaper) candidates overwrite the later ones, so the first
    qualifying tier in `order` wins.
    """
    p = {1: p1, 2: p2}
    t = np.full(len(p1), 3, dtype=int)
    for tier in reversed(order):
        t[p[tier] >= tau] = tier
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
    train_items, _, _ = make_xy(load_split("train"), correct)
    cal_items, Xca, yca = make_xy(load_split("calibration"), correct)
    p = predict(model, Xca)
    p1, p2 = p[1], p[2]

    # candidate ordering comes from TRAIN energy only, never calibration or test
    train_e = {t: float(np.mean([tokens[(t, it["item_id"])] / 1000 * PAPER_RATES[t]
                                 for it in train_items])) for t in TIERS}
    cost_order = tuple(sorted([1, 2], key=lambda t: train_e[t]))
    print(f"variant={variant}  CALIBRATION n={len(cal_items)}")
    print("TRAIN mean energy per item: " +
          ", ".join(f"tier {t} {train_e[t]:.4f} J" for t in TIERS))
    print(f"=> cost-ordered candidates: {cost_order} then Tier 3 fallback")

    e_frontier = sum(tokens[(3, it["item_id"])] / 1000 * PAPER_RATES[3] for it in cal_items)
    budget = ENERGY_BUDGET_FRAC * e_frontier
    print(f"always-frontier calibration energy = {e_frontier:,.1f} J; "
          f"pre-registered budget = {ENERGY_BUDGET_FRAC:.0%} = {budget:,.1f} J")

    grid = np.linspace(0.0, 1.0, N_GRID)
    orders = {"cost_ordered": cost_order, "index_ordered": (1, 2)}
    all_rows = []
    for rule_name, order in orders.items():
        for tau in grid:
            tiers = route(p1, p2, tau, order)
            acc, energy, mix = evaluate(tiers, cal_items, correct, tokens)
            all_rows.append({
                "rule": rule_name,
                "tau": round(float(tau), 6),
                "accuracy": acc,
                "energy_J": energy,
                "energy_frac_of_frontier": energy / e_frontier,
                "n_tier1": mix[1], "n_tier2": mix[2], "n_tier3": mix[3],
                "within_budget": energy <= budget,
            })

    with open(OUT / "threshold_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)

    rows = [r for r in all_rows if r["rule"] == "cost_ordered"]
    abl = [r for r in all_rows if r["rule"] == "index_ordered"]
    print(f"\nablation (index-ordered): min energy {min(r['energy_J'] for r in abl):,.1f} J, "
          f"{sum(r['within_budget'] for r in abl)}/{len(abl)} thresholds within budget")

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
            "routing_rule": "cost_ordered",
            "candidate_order": list(cost_order),
            "train_mean_energy_per_item_J": train_e,
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

    ax[0].plot([r["tau"] for r in abl], [r["accuracy"] for r in abl], "s-", ms=2.5,
               color="#999999", alpha=0.8, label="ablation: index-ordered")
    ax2 = ax[1]
    ax2.plot([r["energy_J"] for r in abl], [r["accuracy"] for r in abl], "s-", ms=2.5,
             color="#999999", alpha=0.8, label="ablation: index-ordered")
    ax2.plot([r["energy_J"] for r in rows], [r["accuracy"] for r in rows], "o-", ms=3,
             color="#2ca02c", label="router threshold sweep (cost-ordered)")
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

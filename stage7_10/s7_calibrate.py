"""Stage 7 - threshold sweep and MCKP/LP frontier on the NEW calibration split.

Identical procedure and identical pre-registered rules to Stages 3 and 4; the
pure functions (routing rule, LP/MILP frontier solvers, interpolation) are
imported from router_v2 so they cannot drift. Only the data changed.

Threshold rule, unchanged from the Stage 3 pre-registration and restated in
prereg_stage7.md before any of this ran:

    highest CALIBRATION accuracy among thresholds whose CALIBRATION energy is
    <= 10% of always-frontier energy on the same split; ties -> lower energy.

The routing rule is the cost-ordered cascade, with the candidate order derived
from TRAIN mean energy only. Unlike Stage 3, this is fixed in advance.

Writes s7_threshold_sweep.csv/.png, s7_chosen_threshold.json,
s7_mckp_frontier.csv/.png, s7_mckp_gaps.json.
"""

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in ("benchmark", "router_v2", "stage7_10", "backend"):
    sys.path.insert(0, str(ROOT / p))
from calibrate import route  # noqa: E402
from mckp import build_matrices, interp_at, lp_frontier, milp_frontier  # noqa: E402
from s7_data import PAPER_RATES, TIERS, load_split, pool_correct_and_tokens  # noqa: E402
from s7_fit import make_xy  # noqa: E402
from train_router import predict  # noqa: E402

OUT = ROOT / "stage7_10"
ENERGY_BUDGET_FRAC = 0.10
N_GRID = 50
N_BUDGETS = 30


def evaluate(tiers, items, correct, tokens):
    acc = np.mean([correct[(int(t), it["item_id"])] for t, it in zip(tiers, items)])
    energy = sum(tokens[(int(t), it["item_id"])] / 1000 * PAPER_RATES[int(t)]
                 for t, it in zip(tiers, items))
    mix = {t: int(np.sum(tiers == t)) for t in TIERS}
    return float(acc), float(energy), mix


def main():
    with open(OUT / "s7_router_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model, variant = bundle["model"], bundle["variant"]

    correct, tokens = pool_correct_and_tokens()
    train_items, _, _ = make_xy(load_split("train"), correct)
    cal_items, Xca, _ = make_xy(load_split("calibration"), correct)
    p = predict(model, Xca)
    p1, p2 = p[1], p[2]

    train_e = {t: float(np.mean([tokens[(t, it["item_id"])] / 1000 * PAPER_RATES[t]
                                 for it in train_items])) for t in TIERS}
    cost_order = tuple(sorted([1, 2], key=lambda t: train_e[t]))
    print(f"variant={variant}  CALIBRATION n={len(cal_items)}")
    print("TRAIN mean energy per item: " +
          ", ".join(f"tier {t} {train_e[t]:.4f} J" for t in TIERS))
    print(f"=> cost-ordered candidates: {cost_order} then Tier 3 fallback")

    e_frontier = sum(tokens[(3, it["item_id"])] / 1000 * PAPER_RATES[3]
                     for it in cal_items)
    budget = ENERGY_BUDGET_FRAC * e_frontier
    print(f"always-frontier calibration energy = {e_frontier:,.1f} J; "
          f"budget = {ENERGY_BUDGET_FRAC:.0%} = {budget:,.1f} J")

    grid = np.linspace(0.0, 1.0, N_GRID)
    all_rows = []
    for rule_name, order in (("cost_ordered", cost_order), ("index_ordered", (1, 2))):
        for tau in grid:
            tiers = route(p1, p2, tau, order)
            acc, energy, mix = evaluate(tiers, cal_items, correct, tokens)
            all_rows.append({
                "rule": rule_name, "tau": round(float(tau), 6), "accuracy": acc,
                "energy_J": energy, "energy_frac_of_frontier": energy / e_frontier,
                "n_tier1": mix[1], "n_tier2": mix[2], "n_tier3": mix[3],
                "within_budget": energy <= budget,
            })
    with open(OUT / "s7_threshold_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)

    rows = [r for r in all_rows if r["rule"] == "cost_ordered"]
    abl = [r for r in all_rows if r["rule"] == "index_ordered"]
    eligible = [r for r in rows if r["within_budget"]]
    if eligible:
        best = max(eligible, key=lambda r: (r["accuracy"], -r["energy_J"]))
        rule_note = "pre-registered rule applied as written"
    else:
        best = min(rows, key=lambda r: r["energy_J"])
        rule_note = ("NO threshold met the <=10% energy constraint; fell back to the "
                     "lowest-energy threshold, as the pre-registration specifies")
    print(f"\nchosen tau = {best['tau']:.4f}  acc={best['accuracy']:.4f}  "
          f"energy={best['energy_J']:,.1f} J "
          f"({best['energy_frac_of_frontier']:.1%} of frontier)  "
          f"mix={best['n_tier1']}/{best['n_tier2']}/{best['n_tier3']}")
    print(f"  {rule_note}; {len(eligible)}/{len(rows)} thresholds within budget")

    with open(OUT / "s7_chosen_threshold.json", "w") as f:
        json.dump({
            "variant": variant, "routing_rule": "cost_ordered",
            "candidate_order": list(cost_order),
            "train_mean_energy_per_item_J": train_e,
            "rule": ("highest CALIBRATION accuracy among thresholds with CALIBRATION "
                     f"energy <= {ENERGY_BUDGET_FRAC:.0%} of always-frontier energy; "
                     "ties toward lower energy"),
            "rule_note": rule_note, "energy_budget_frac": ENERGY_BUDGET_FRAC,
            "calibration_frontier_energy_J": e_frontier,
            "calibration_budget_J": budget,
            "n_eligible_thresholds": len(eligible), "chosen": best,
        }, f, indent=2)

    # ------------------------------------------------------------ MCKP frontier
    tau, order = best["tau"], cost_order
    v, w = build_matrices(cal_items, correct, tokens)
    n = len(cal_items)
    e_by_tier = {t: float(w[:, k].sum()) for k, t in enumerate(TIERS)}
    acc_by_tier = {t: float(v[:, k].mean()) for k, t in enumerate(TIERS)}

    oracle_idx = np.where(v.sum(axis=1) > 0,
                          np.argmax(np.where(v > 0, -w, -np.inf), axis=1),
                          np.argmin(w, axis=1))
    oracle_acc = float(v[np.arange(n), oracle_idx].mean())
    oracle_energy = float(w[np.arange(n), oracle_idx].sum())
    oracle_tier = np.array([TIERS[k] for k in oracle_idx])

    router_tiers = route(p1, p2, tau, order)
    router_k = np.array([TIERS.index(int(t)) for t in router_tiers])
    router_acc = float(v[np.arange(n), router_k].mean())
    router_energy = float(w[np.arange(n), router_k].sum())

    from main import classify_prompt_local_nlp
    kw_tiers = np.array([int(classify_prompt_local_nlp(it["raw_query"]).recommended_tier)
                         for it in cal_items])
    kw_k = np.array([TIERS.index(int(t)) for t in kw_tiers])
    kw_acc = float(v[np.arange(n), kw_k].mean())
    kw_energy = float(w[np.arange(n), kw_k].sum())

    lo, hi = float(w.min(axis=1).sum()), float(w.max(axis=1).sum())
    budgets = np.unique(np.concatenate([
        np.geomspace(lo, hi, N_BUDGETS),
        [router_energy, oracle_energy, kw_energy, e_by_tier[1], e_by_tier[2]],
    ]))
    budgets = budgets[(budgets >= lo - 1e-9) & (budgets <= hi + 1e-9)]
    print(f"\nsolving LP frontier at {len(budgets)} budgets (n={n}) ...", flush=True)
    lp_acc = lp_frontier(v, w, budgets)
    print(f"solving integer MCKP frontier at {len(budgets)} budgets ...", flush=True)
    ip_acc = milp_frontier(v, w, budgets)

    with open(OUT / "s7_mckp_frontier.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["budget_J", "lp_relaxed_accuracy", "integer_mckp_accuracy"])
        for B, a, b in zip(budgets, lp_acc, ip_acc):
            wr.writerow([f"{B:.6f}", f"{a:.6f}", f"{b:.6f}"])

    lp_at_router = interp_at(budgets, lp_acc, router_energy)
    ip_at_router = interp_at(budgets, ip_acc, router_energy)
    lp_at_oracle = interp_at(budgets, lp_acc, oracle_energy)
    calibration_gap = lp_at_router - router_acc
    disc_at_router = lp_at_router - ip_at_router
    disc_at_oracle = lp_at_oracle - oracle_acc
    ok = ~np.isnan(lp_acc) & ~np.isnan(ip_acc)
    disc_curve = lp_acc[ok] - ip_acc[ok]

    with open(ROOT / "router_v2" / "mckp_gaps.json") as f:
        s6 = json.load(f)

    gaps = {
        "variant": variant, "tau": tau, "n_calibration_items": n,
        "label_rule": "majority of k=3 @ temp 0.7",
        "acc_by_tier": acc_by_tier, "energy_by_tier_J": e_by_tier,
        "router": {"accuracy": router_acc, "energy_J": router_energy,
                   "tier_mix": {str(t): int((router_tiers == t).sum()) for t in TIERS}},
        "oracle": {"accuracy": oracle_acc, "energy_J": oracle_energy,
                   "tier_mix": {str(t): int((oracle_tier == t).sum()) for t in TIERS}},
        "ecologic_keyword_on_calibration": {
            "accuracy": kw_acc, "energy_J": kw_energy,
            "tier_mix": {str(t): int((kw_tiers == t).sum()) for t in TIERS}},
        "lp_frontier_accuracy_at_router_energy": lp_at_router,
        "integer_frontier_accuracy_at_router_energy": ip_at_router,
        "calibration_gap_pp": calibration_gap * 100,
        "discreteness_gap_at_router_energy_pp": disc_at_router * 100,
        "discreteness_gap_at_oracle_energy_pp": disc_at_oracle * 100,
        "discreteness_gap_curve_mean_pp": float(disc_curve.mean() * 100),
        "discreteness_gap_curve_max_pp": float(disc_curve.max() * 100),
        "stage6_reference": {
            "calibration_gap_pp": s6["calibration_gap_pp"],
            "discreteness_gap_at_router_energy_pp":
                s6["discreteness_gap_at_router_energy_pp"],
            "n_calibration_items": s6["n_calibration_items"]},
    }
    with open(OUT / "s7_mckp_gaps.json", "w") as f:
        json.dump(gaps, f, indent=2)

    print(f"\n  router  acc={router_acc:.4f} energy={router_energy:,.1f} J")
    print(f"  LP @ router energy      = {lp_at_router:.4f}")
    print(f"  integer @ router energy = {ip_at_router:.4f}")
    print(f"  CALIBRATION GAP  = {calibration_gap * 100:+.2f} pp "
          f"(Stage 6: {s6['calibration_gap_pp']:+.2f} pp)")
    print(f"  discreteness gap = {disc_at_router * 100:+.2f} pp "
          f"(Stage 6: {s6['discreteness_gap_at_router_energy_pp']:+.2f} pp)")

    # ---- chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.plot(budgets, lp_acc * 100, "-", lw=2, color="#1f77b4", label="LP-relaxed MCKP frontier")
    ax.plot(budgets, ip_acc * 100, "--", lw=1.6, color="#7f7f7f",
            label="integer MCKP frontier (discrete optimum)")
    ax.plot([r["energy_J"] for r in rows], [r["accuracy"] * 100 for r in rows], "o-",
            ms=3.5, color="#2ca02c", label="learned router threshold sweep")
    ax.plot([oracle_energy], [oracle_acc * 100], "D", ms=9, color="#d62728",
            label="oracle discrete policy")
    ax.plot([kw_energy], [kw_acc * 100], "s", ms=9, color="#9467bd",
            label="EcoLogic keyword classifier")
    ax.plot([router_energy], [router_acc * 100], "*", ms=18, color="crimson",
            label=f"router @ chosen tau={tau:.3f}")
    ax.annotate("", xy=(router_energy, lp_at_router * 100),
                xytext=(router_energy, router_acc * 100),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.3))
    ax.text(router_energy * 1.15, (router_acc + lp_at_router) / 2 * 100,
            f"calibration gap\n{calibration_gap * 100:.2f} pp", fontsize=8.5, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("energy budget (J, log scale)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"Stage 7 MCKP frontier — new CALIBRATION split "
                 f"(n={n}, variant {variant}, k=3 majority labels)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "s7_mckp_frontier.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(7.4, 4.8))
    ax2.plot([r["energy_J"] for r in abl], [r["accuracy"] * 100 for r in abl], "s-",
             ms=2.5, color="#999999", label="ablation: index-ordered")
    ax2.plot([r["energy_J"] for r in rows], [r["accuracy"] * 100 for r in rows], "o-",
             ms=3, color="#2ca02c", label="cost-ordered (used)")
    ax2.axvline(budget, ls=":", color="k",
                label=f"pre-registered budget ({ENERGY_BUDGET_FRAC:.0%})")
    ax2.plot([best["energy_J"]], [best["accuracy"] * 100], "*", ms=16, color="crimson",
             label="chosen threshold")
    ax2.set_xscale("log")
    ax2.set_xlabel("CALIBRATION energy (J, log scale)")
    ax2.set_ylabel("accuracy (%)")
    ax2.set_title("Stage 7 threshold calibration")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(OUT / "s7_threshold_sweep.png", dpi=150)
    print("wrote s7_threshold_sweep.csv/.png, s7_chosen_threshold.json, "
          "s7_mckp_frontier.csv/.png, s7_mckp_gaps.json")


if __name__ == "__main__":
    main()

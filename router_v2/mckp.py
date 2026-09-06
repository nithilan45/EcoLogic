"""Stage 4 - LP-relaxed MCKP frontier, gaps, and energy-regret decomposition.

Formulation. Each CALIBRATION item is one multiple-choice class g; the three
tiers are its incompatible choices. With v_{g,t} = 1 if tier t answered item g
correctly and w_{g,t} = that response's energy in joules:

    maximize    sum_{g,t} v_{g,t} x_{g,t}
    subject to  sum_{g,t} w_{g,t} x_{g,t} <= B          (energy budget)
                sum_t x_{g,t} = 1  for every g          (exactly one tier)
                x_{g,t} >= 0

x binary is the multiple-choice knapsack problem (Kellerer, Pferschy &
Pisinger, *Knapsack Problems*, Springer 2004, Ch. 11). Relaxing x to [0,1]
gives the LP bound, solved here with scipy.optimize.linprog; the integer
optimum is solved with scipy.optimize.milp for the discreteness comparison.

Two gaps, deliberately kept separate:

  calibration gap - LP frontier accuracy minus the ROUTER's realized accuracy
                    at matched energy. Measures how well-calibrated the learned
                    probabilities are; a property of the router.
  discreteness gap - LP frontier accuracy minus the DISCRETE optimum accuracy
                    at matched energy. The cost of being forced to pick one
                    tier per item instead of a fractional mixture; a property
                    of the problem, not of the router.

Writes router_v2/mckp_frontier_v2.png, mckp_frontier_v2.csv,
regret_verification_v2.md, mckp_gaps.json.
"""

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, linprog, milp, Bounds
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "router_v2"))
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT / "backend"))
from api import MODELS  # noqa: E402
from calibrate import load_tokens, route  # noqa: E402
from train_router import load_correct, load_split, make_xy, predict  # noqa: E402

OUT = ROOT / "router_v2"
TIERS = [1, 2, 3]
PAPER_RATES = {t: MODELS[t]["paper_energy_per_1k"] for t in TIERS}
N_BUDGETS = 36


def build_matrices(items, correct, tokens):
    n = len(items)
    v = np.zeros((n, 3))
    w = np.zeros((n, 3))
    for i, it in enumerate(items):
        for k, t in enumerate(TIERS):
            v[i, k] = float(correct[(t, it["item_id"])])
            w[i, k] = tokens[(t, it["item_id"])] / 1000 * PAPER_RATES[t]
    return v, w


def _constraints(n, w):
    # one-hot per item
    rows, cols = [], []
    for i in range(n):
        for k in range(3):
            rows.append(i)
            cols.append(i * 3 + k)
    A_eq = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, 3 * n))
    return A_eq, w.reshape(-1)


def lp_frontier(v, w, budgets):
    n = v.shape[0]
    A_eq, wflat = _constraints(n, w)
    c = -v.reshape(-1)
    out = []
    for B in budgets:
        res = linprog(c, A_ub=csr_matrix(wflat.reshape(1, -1)), b_ub=[B],
                      A_eq=A_eq, b_eq=np.ones(n), bounds=(0, 1), method="highs")
        out.append((-res.fun / n) if res.success else np.nan)
    return np.array(out)


def milp_frontier(v, w, budgets):
    n = v.shape[0]
    A_eq, wflat = _constraints(n, w)
    c = -v.reshape(-1)
    out = []
    for B in budgets:
        cons = [
            LinearConstraint(csr_matrix(wflat.reshape(1, -1)), -np.inf, B),
            LinearConstraint(A_eq, 1, 1),
        ]
        res = milp(c=c, constraints=cons, integrality=np.ones(3 * n),
                   bounds=Bounds(0, 1))
        out.append((-res.fun / n) if res.success else np.nan)
    return np.array(out)


def interp_at(x_curve, y_curve, x0):
    """Monotone-increasing interpolation of a frontier at energy x0."""
    ok = ~np.isnan(y_curve)
    xs, ys = np.asarray(x_curve)[ok], np.asarray(y_curve)[ok]
    order = np.argsort(xs)
    xs, ys = xs[order], np.maximum.accumulate(ys[order])
    return float(np.interp(x0, xs, ys))


def main():
    with open(OUT / "router_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model, variant = bundle["model"], bundle["variant"]
    with open(OUT / "chosen_threshold.json") as f:
        chosen = json.load(f)
    tau = chosen["chosen"]["tau"]

    correct = load_correct()
    tokens = load_tokens()
    items, X, y = make_xy(load_split("calibration"), correct)
    n = len(items)
    v, w = build_matrices(items, correct, tokens)

    # ---- reference policies on CALIBRATION
    e_by_tier = {t: float(w[:, k].sum()) for k, t in enumerate(TIERS)}
    acc_by_tier = {t: float(v[:, k].mean()) for k, t in enumerate(TIERS)}

    oracle_idx = np.where(v.sum(axis=1) > 0,
                          np.argmax(np.where(v > 0, -w, -np.inf), axis=1),
                          np.argmin(w, axis=1))
    oracle_acc = float(v[np.arange(n), oracle_idx].mean())
    oracle_energy = float(w[np.arange(n), oracle_idx].sum())
    oracle_tier = np.array([TIERS[k] for k in oracle_idx])

    p = predict(model, X)
    router_tiers = route(p[1], p[2], tau)
    router_k = np.array([TIERS.index(t) for t in router_tiers])
    router_acc = float(v[np.arange(n), router_k].mean())
    router_energy = float(w[np.arange(n), router_k].sum())

    # EcoLogic keyword classifier, evaluated on CALIBRATION for an
    # apples-to-apples point (its frozen-test-set point appears in Stage 5)
    from main import classify_prompt_local_nlp
    kw_tiers = np.array([int(classify_prompt_local_nlp(it["raw_query"]).recommended_tier)
                         for it in items])
    kw_k = np.array([TIERS.index(t) for t in kw_tiers])
    kw_acc = float(v[np.arange(n), kw_k].mean())
    kw_energy = float(w[np.arange(n), kw_k].sum())

    # ---- frontiers
    lo, hi = float(w.min(axis=1).sum()), float(w.max(axis=1).sum())
    budgets = np.unique(np.concatenate([
        np.geomspace(lo, hi, N_BUDGETS),
        [router_energy, oracle_energy, kw_energy, e_by_tier[1], e_by_tier[2]],
    ]))
    budgets = budgets[(budgets >= lo - 1e-9) & (budgets <= hi + 1e-9)]
    print(f"solving LP frontier at {len(budgets)} budgets (n={n} items) ...")
    lp_acc = lp_frontier(v, w, budgets)
    print(f"solving integer MCKP frontier at {len(budgets)} budgets ...")
    ip_acc = milp_frontier(v, w, budgets)

    with open(OUT / "mckp_frontier_v2.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["budget_J", "lp_relaxed_accuracy", "integer_mckp_accuracy"])
        for B, a, b in zip(budgets, lp_acc, ip_acc):
            wr.writerow([f"{B:.6f}", f"{a:.6f}", f"{b:.6f}"])

    # ---- the two gaps, at matched energy
    lp_at_router = interp_at(budgets, lp_acc, router_energy)
    ip_at_router = interp_at(budgets, ip_acc, router_energy)
    lp_at_oracle = interp_at(budgets, lp_acc, oracle_energy)

    calibration_gap = lp_at_router - router_acc
    discreteness_gap_at_router = lp_at_router - ip_at_router
    discreteness_gap_at_oracle = lp_at_oracle - oracle_acc

    ok = ~np.isnan(lp_acc) & ~np.isnan(ip_acc)
    disc_curve = (lp_acc[ok] - ip_acc[ok])

    gaps = {
        "variant": variant, "tau": tau, "n_calibration_items": n,
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
        "discreteness_gap_at_router_energy_pp": discreteness_gap_at_router * 100,
        "discreteness_gap_at_oracle_energy_pp": discreteness_gap_at_oracle * 100,
        "discreteness_gap_curve_mean_pp": float(disc_curve.mean() * 100),
        "discreteness_gap_curve_max_pp": float(disc_curve.max() * 100),
    }

    # ---- energy regret, verified two ways
    # Per-tier constant energies make the confusion-matrix identity exact; the
    # per-item version is reported alongside because energy is item-dependent.
    e_const = {t: float(w[:, k].mean()) for k, t in enumerate(TIERS)}
    direct_const = float(np.mean([e_const[int(a)] - e_const[int(b)]
                                  for a, b in zip(router_tiers, oracle_tier)]))
    conf = {}
    for i_star in TIERS:
        for j_hat in TIERS:
            m = float(np.mean((oracle_tier == i_star) & (router_tiers == j_hat)))
            if m > 0:
                conf[(i_star, j_hat)] = m
    formula_const = sum(pr * (e_const[j] - e_const[i])
                        for (i, j), pr in conf.items() if i != j)
    per_item = float(np.mean(w[np.arange(n), router_k] - w[np.arange(n), oracle_idx]))

    gaps["regret"] = {
        "per_tier_mean_energy_J": e_const,
        "method_a_direct_paired_J_per_item": direct_const,
        "method_b_confusion_formula_J_per_item": formula_const,
        "abs_difference": abs(direct_const - formula_const),
        "match": abs(direct_const - formula_const) < 1e-9,
        "per_item_actual_energy_regret_J_per_item": per_item,
        "confusion_joint_probabilities": {f"t*={i},that={j}": pr for (i, j), pr in sorted(conf.items())},
    }
    with open(OUT / "mckp_gaps.json", "w") as f:
        json.dump(gaps, f, indent=2)

    # ---- chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = list(csv.DictReader(open(OUT / "threshold_sweep.csv")))
    sx = [float(r["energy_J"]) for r in sweep]
    sy = [float(r["accuracy"]) for r in sweep]
    order = np.argsort(sx)
    sx = np.array(sx)[order]; sy = np.array(sy)[order]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.plot(budgets, lp_acc, "-", lw=2.2, color="#1f77b4", label="LP-relaxed MCKP frontier")
    ax.plot(budgets, ip_acc, "--", lw=1.6, color="#17becf", label="integer MCKP optimum")
    ax.plot(sx, sy, "o-", ms=4, lw=1.4, color="#2ca02c",
            label=f"learned router {variant}, threshold sweep")
    ax.plot([router_energy], [router_acc], "*", ms=20, color="crimson", zorder=5,
            label=f"learned router at chosen tau={tau:.3f}")
    ax.plot([oracle_energy], [oracle_acc], "D", ms=9, color="#9467bd", zorder=5,
            label="oracle discrete policy")
    ax.plot([kw_energy], [kw_acc], "s", ms=9, color="#d62728", zorder=5,
            label="EcoLogic keyword classifier")
    for t, mk in zip(TIERS, ["v", "^", "P"]):
        ax.plot([e_by_tier[t]], [acc_by_tier[t]], mk, ms=8, color="gray", alpha=0.75,
                label=f"always Tier {t}")
    ax.set_xscale("log")
    ax.set_xlabel("energy on CALIBRATION split (J, log scale)")
    ax.set_ylabel("accuracy")
    ax.set_title("Stage 4 — MCKP frontier vs learned router (CALIBRATION split only)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "mckp_frontier_v2.png", dpi=150)

    # ---- regret write-up
    L = []
    L.append("# Energy-regret verification (Stage 4)\n")
    L.append("Energy regret is the extra energy the learned router spends per item relative "
             "to the oracle tier choice, on the CALIBRATION split "
             f"(n={n}, variant {variant}, threshold tau={tau:.4f}).\n")
    L.append("## Two independent computations\n")
    L.append("Both use per-tier mean energies as the constants `e_t`, which is what makes "
             "the confusion-matrix identity exact:\n")
    L.append("| Tier | mean energy per item (J) |")
    L.append("|---|---|")
    for t in TIERS:
        L.append(f"| {t} | {e_const[t]:.4f} |")
    L.append("")
    L.append("| Method | Formula | Result (J/item) |")
    L.append("|---|---|---|")
    L.append(f"| A — direct, paired samples | `mean_i( e[that_i] - e[t*_i] )` "
             f"| **{direct_const:.10f}** |")
    L.append(f"| B — confusion matrix | `sum_{{i!=j}} P(t*=i, that=j) * (e_j - e_i)` "
             f"| **{formula_const:.10f}** |")
    L.append("")
    L.append(f"**Absolute difference: {abs(direct_const - formula_const):.2e} — "
             f"{'they match' if gaps['regret']['match'] else 'THEY DO NOT MATCH'}.** "
             "The two are algebraically the same quantity, so agreement to floating-point "
             "precision is a correctness check on the confusion matrix and the pairing, "
             "not independent evidence about the router.\n")
    L.append(f"Using each item's **actual** energy instead of per-tier means gives "
             f"**{per_item:.4f} J/item**. This differs from the two figures above because "
             "energy is item-dependent (token counts vary per item), so the constant-`e_t` "
             "identity no longer holds. It is the more faithful number for real energy "
             "accounting; the constant version is the one the confusion-matrix formula "
             "applies to.\n")
    L.append("## Confusion matrix, P(oracle tier = i, router tier = j)\n")
    L.append("| oracle t* \\ router t^ | Tier 1 | Tier 2 | Tier 3 |")
    L.append("|---|---|---|---|")
    for i in TIERS:
        cells = [f"{conf.get((i, j), 0.0):.4f}" for j in TIERS]
        L.append(f"| Tier {i} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("Off-diagonal mass above the diagonal is false escalation (router picked a "
             "costlier tier than needed); below it is false de-escalation (router picked a "
             "cheaper tier than the oracle, which costs accuracy rather than energy).\n")
    with open(OUT / "regret_verification_v2.md", "w") as f:
        f.write("\n".join(L))

    print(f"\nCALIBRATION results (variant {variant}, tau={tau:.4f}):")
    print(f"  router           acc={router_acc:.4f} energy={router_energy:,.1f} J")
    print(f"  oracle           acc={oracle_acc:.4f} energy={oracle_energy:,.1f} J")
    print(f"  keyword (EcoLogic) acc={kw_acc:.4f} energy={kw_energy:,.1f} J")
    print(f"  always tier1/2/3 acc={acc_by_tier[1]:.3f}/{acc_by_tier[2]:.3f}/{acc_by_tier[3]:.3f}")
    print(f"  calibration gap                 = {calibration_gap * 100:+.2f} pp")
    print(f"  discreteness gap @ router energy = {discreteness_gap_at_router * 100:+.2f} pp")
    print(f"  discreteness gap @ oracle energy = {discreteness_gap_at_oracle * 100:+.2f} pp")
    print(f"  regret A={direct_const:.6f}  B={formula_const:.6f}  "
          f"match={gaps['regret']['match']}")
    print("wrote mckp_frontier_v2.png/.csv, mckp_gaps.json, regret_verification_v2.md")


if __name__ == "__main__":
    main()

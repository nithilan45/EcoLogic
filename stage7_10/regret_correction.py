"""Stage 8 - derive and validate the exact regret-decomposition correction.

Recomputes Stage 6's CALIBRATION-split routing decisions (router_v2/ is read
only, nothing there is modified), then checks that

    R_true = R_naive + sum_{i!=j} Cov( 1{t*=i, t^=j},  e_j(X) - e_i(X) )

reconciles the two numbers that disagreed in Stage 6 (-0.2496 vs +0.2840 J/item)
to floating-point precision.

Writes stage7_10/regret_correction_derivation.md.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in ("benchmark", "router_v2"):
    sys.path.insert(0, str(ROOT / p))
from api import MODELS  # noqa: E402
from calibrate import load_tokens, route  # noqa: E402
from train_router import load_correct, load_split, make_xy, predict  # noqa: E402

OUT = ROOT / "stage7_10"
V2 = ROOT / "router_v2"
TIERS = [1, 2, 3]
PAPER_RATES = {t: MODELS[t]["paper_energy_per_1k"] for t in TIERS}


def cov(a: np.ndarray, b: np.ndarray) -> float:
    """Population covariance; the empirical distribution is the probability measure."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(a * b) - np.mean(a) * np.mean(b))


def main():
    with open(V2 / "router_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model, variant = bundle["model"], bundle["variant"]
    with open(V2 / "chosen_threshold.json") as f:
        ct = json.load(f)
    tau, order = ct["chosen"]["tau"], tuple(ct["candidate_order"])

    correct = load_correct()
    tokens = load_tokens()
    items, X, _ = make_xy(load_split("calibration"), correct)
    n = len(items)

    # per-item energy at each tier, and the two tier assignments
    E = np.zeros((n, 3))
    V = np.zeros((n, 3))
    for i, it in enumerate(items):
        for k, t in enumerate(TIERS):
            E[i, k] = tokens[(t, it["item_id"])] / 1000 * PAPER_RATES[t]
            V[i, k] = float(correct[(t, it["item_id"])])

    oracle_k = np.where(V.sum(axis=1) > 0,
                        np.argmax(np.where(V > 0, -E, -np.inf), axis=1),
                        np.argmin(E, axis=1))
    p = predict(model, X)
    router_tiers = route(p[1], p[2], tau, order)
    router_k = np.array([TIERS.index(int(t)) for t in router_tiers])

    t_star = np.array([TIERS[k] for k in oracle_k])
    t_hat = np.array([TIERS[k] for k in router_k])

    # ---- the three quantities
    r_true = float(np.mean(E[np.arange(n), router_k] - E[np.arange(n), oracle_k]))
    e_mean = {t: float(E[:, k].mean()) for k, t in enumerate(TIERS)}
    r_naive = 0.0
    cells = {}
    for i_star in TIERS:
        for j_hat in TIERS:
            mask = (t_star == i_star) & (t_hat == j_hat)
            pr = float(mask.mean())
            if pr > 0:
                cells[(i_star, j_hat)] = pr
            if i_star != j_hat and pr > 0:
                r_naive += pr * (e_mean[j_hat] - e_mean[i_star])

    # ---- correction term
    correction = 0.0
    per_cell = {}
    for (i_star, j_hat), pr in cells.items():
        if i_star == j_hat:
            continue
        ki, kj = TIERS.index(i_star), TIERS.index(j_hat)
        D = E[:, kj] - E[:, ki]
        ind = ((t_star == i_star) & (t_hat == j_hat)).astype(float)
        c = cov(ind, D)
        correction += c
        per_cell[(i_star, j_hat)] = {
            "P_cell": pr,
            "E_D_uncond": float(D.mean()),
            "E_D_given_cell": float(D[ind > 0].mean()),
            "cov": c,
            "naive_contrib": pr * (e_mean[j_hat] - e_mean[i_star]),
        }

    reconciled = r_naive + correction
    resid = abs(reconciled - r_true)
    ok = resid < 1e-9

    payload = {
        "n_items": n, "variant": variant, "tau": tau,
        "per_tier_mean_energy_J": e_mean,
        "R_naive": r_naive, "correction": correction,
        "R_naive_plus_correction": reconciled, "R_true": r_true,
        "abs_residual": resid, "reconciles": bool(ok),
        "cells": {f"t*={i},that={j}": v for (i, j), v in sorted(per_cell.items())},
    }
    with open(OUT / "regret_correction_validation.json", "w") as f:
        json.dump(payload, f, indent=2)

    # --------------------------------------------------------------- write-up
    L = []
    L.append("# Stage 8 — the exact correction term for confusion-matrix energy regret\n")
    L.append("## The problem\n")
    L.append("Stage 6 computed the same quantity two ways and got opposite signs. The "
             "confusion-matrix formula said the router **saved** energy relative to the "
             "oracle (-0.2496 J/item); measuring each item's actual energy said it "
             "**spent more** (+0.2840 J/item). One of them is answering a different "
             "question, and it is the formula.\n")
    L.append("## Setup and notation\n")
    L.append("Let `X` be an item drawn from the evaluation set (the empirical "
             "distribution is the probability measure). For each tier `j`, let `e_j(X)` "
             "be the **actual** energy of tier `j`'s response to item `X`, and let "
             "`e_j := E[e_j(X)]` be the tier mean. Let `t*(X)` be the oracle tier and "
             "`t^(X)` the router's tier. Write `C_ij := {t* = i, t^ = j}` for a "
             "confusion-matrix cell and `D_ij(X) := e_j(X) - e_i(X)`.\n")
    L.append("The two estimators are\n")
    L.append("```\nR_true  = E[ e_{t^(X)}(X) - e_{t*(X)}(X) ]\n"
             "R_naive = sum_{i != j} P(C_ij) * (e_j - e_i)\n```\n")
    L.append("## Proposition\n")
    L.append("> **Proposition.** For any joint distribution of `(X, t*, t^)` with finite "
             "first moments,\n>\n"
             "> ```\n> R_true = R_naive + sum_{i != j} Cov( 1{C_ij}, D_ij(X) )\n> ```\n>\n"
             "> Consequently `R_naive = R_true` if and only if "
             "`sum_{i != j} Cov(1{C_ij}, D_ij(X)) = 0`. A sufficient condition is that "
             "each `D_ij(X)` be mean-independent of cell membership, "
             "`E[D_ij | C_ij] = E[D_ij]`; in particular the naive formula is exact "
             "whenever `e_j(X)` is constant within each tier, i.e. whenever energy does "
             "not vary across items.\n")
    L.append("**Proof.** Partition on the cells. Since the cells `C_ij` are exhaustive "
             "and mutually exclusive,\n")
    L.append("```\nR_true = sum_{i,j} P(C_ij) * E[ e_j(X) - e_i(X) | C_ij ]\n"
             "       = sum_{i != j} P(C_ij) * E[ D_ij(X) | C_ij ]\n```\n")
    L.append("where the diagonal terms drop out because `D_ii(X) = 0` pointwise. "
             "Subtracting the naive estimator term by term over the same index set,\n")
    L.append("```\nR_true - R_naive = sum_{i != j} P(C_ij) * ( E[D_ij | C_ij] - E[D_ij] )\n```\n")
    L.append("For any event `A` and integrable `D`, the definition of covariance gives\n")
    L.append("```\nCov(1_A, D) = E[1_A * D] - E[1_A] * E[D]\n"
             "            = P(A) * E[D | A] - P(A) * E[D]\n"
             "            = P(A) * ( E[D | A] - E[D] )\n```\n")
    L.append("Applying this with `A = C_ij` and `D = D_ij(X)` to each summand yields the "
             "claim. The 'if and only if' is immediate, and the sufficient condition "
             "follows because `E[D_ij | C_ij] = E[D_ij]` makes each summand zero. "
             "$\\blacksquare$\n")
    L.append("This is direct algebra from the definition of covariance, not a deep "
             "result. It is worth writing down only because the naive formula is used to "
             "report energy savings and, as the numbers below show, it can carry the "
             "wrong sign.\n")
    L.append("### A note on notation\n")
    L.append("The correction is sometimes stated informally as "
             "`sum P(C_ij) * Cov(D_ij | C_ij)`. That expression does not type-check — a "
             "covariance needs two arguments, and a conditional variance of `D_ij` is not "
             "what appears here. The correct object is the covariance between the **cell "
             "indicator** and the energy difference, equivalently `P(C_ij)` times the "
             "**conditional mean shift** `E[D_ij | C_ij] - E[D_ij]`. Both forms above are "
             "exact and identical; the informal one captures the right intuition "
             "(\"naive is exact only when energy is uncorrelated with routing within each "
             "cell\") but is not a usable formula.\n")
    L.append("## Reconciliation on Stage 6's own data\n")
    L.append(f"CALIBRATION split, n = {n}, variant {variant}, threshold tau = {tau:.4f}. "
             "Tier means used by the naive formula: " +
             ", ".join(f"`e_{t}` = {e_mean[t]:.6f} J" for t in TIERS) + ".\n")
    L.append("| Quantity | Value (J/item) |")
    L.append("|---|---|")
    L.append(f"| `R_naive` (confusion-matrix formula) | {r_naive:.12f} |")
    L.append(f"| correction `sum Cov(1{{C_ij}}, D_ij)` | {correction:.12f} |")
    L.append(f"| **`R_naive` + correction** | **{reconciled:.12f}** |")
    L.append(f"| `R_true` (direct per-item measurement) | **{r_true:.12f}** |")
    L.append(f"| absolute residual | {resid:.3e} |")
    L.append("")
    L.append(f"**{'They reconcile exactly' if ok else 'THEY DO NOT RECONCILE'}** — residual "
             f"{resid:.3e}, i.e. floating-point noise. The correction term is "
             f"**{correction:+.4f} J/item**, which is "
             f"**{abs(correction / r_true):.1f}x the size of the true regret itself** and "
             f"large enough to flip the sign: the naive formula reports "
             f"{r_naive:+.4f} where the truth is {r_true:+.4f}.\n")
    L.append("The correction was derived first and evaluated once; it was not tuned to "
             "make the numbers agree.\n")
    L.append("## Where the error comes from, cell by cell\n")
    L.append("| Cell (t\\* -> t^) | P(cell) | E[D] uncond. | E[D] given cell | Cov(1,D) | naive contribution |")
    L.append("|---|---|---|---|---|---|")
    for (i_star, j_hat), d in sorted(per_cell.items()):
        L.append(f"| {i_star} -> {j_hat} | {d['P_cell']:.4f} | {d['E_D_uncond']:+.4f} | "
                 f"{d['E_D_given_cell']:+.4f} | {d['cov']:+.6f} | "
                 f"{d['naive_contrib']:+.6f} |")
    L.append("")
    L.append("The mechanism is visible in the two middle columns: within a cell, the "
             "energy difference `D_ij` is nothing like its unconditional average. The "
             "router sends *cheap* items to Tier 2 and the items it leaves on Tier 1 are "
             "the expensive, long-reasoning ones, so cell membership and energy are "
             "strongly dependent. Substituting tier means for per-item energies throws "
             "exactly that dependence away.\n")
    L.append("## Practical consequence\n")
    L.append("Any routing evaluation that reports energy or cost savings by multiplying a "
             "confusion matrix (or a routing distribution) by per-model average costs is "
             "making this substitution. It is safe only when per-item cost is "
             "approximately constant within each model. That condition fails hard for "
             "reasoning models, whose token counts vary by an order of magnitude across "
             "items — and it fails in the direction that flatters the router, because "
             "routers preferentially send short, easy items to cheap models. The fix "
             "requires no new modelling, only per-item cost data: report `R_true` "
             "directly, or report `R_naive` together with the correction term above.\n")
    with open(OUT / "regret_correction_derivation.md", "w") as f:
        f.write("\n".join(L))

    print(f"R_naive      = {r_naive:.12f}")
    print(f"correction   = {correction:.12f}")
    print(f"sum          = {reconciled:.12f}")
    print(f"R_true       = {r_true:.12f}")
    print(f"residual     = {resid:.3e}  reconciles={ok}")
    print("wrote stage7_10/regret_correction_derivation.md")


if __name__ == "__main__":
    main()

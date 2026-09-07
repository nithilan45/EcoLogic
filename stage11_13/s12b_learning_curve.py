"""Stage 12b (exploratory, NOT pre-registered -- see DEVIATIONS.md D3).

Stage 12 refuted H3: item difficulty is highly reliable, so the binding term in
the decomposition is `eps`, the generalisation gap from prompt text to difficulty.
The first question about `eps` is "would more training data close it?". Stage 7
answered it with two points (1,200 -> 5,000 items moved held-out AUC by 0.009).
RouterBench has 36,494 items, so here it is answered with a curve.

Protocol: one fixed held-out set (20%, stratified by benchmark family). Training
subsets are drawn from the remaining 80% at a geometric grid of sizes, with
repeats at the small sizes. Reports mean held-out AUC over the 11 models and the
realised matched-cost gain at beta = 0.5, then fits a power law
    AUC(n) = A - B * n^(-c)
and reports the extrapolated asymptote A, which is the answer to the question.

Outputs: s12b_learning_curve.json, s12b_learning_curve.csv
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from decomp import oracle_frontier_at, router_value_at, static_frontier_at  # noqa: E402
from s11_routerbench import family_of, load  # noqa: E402

SEED = 20260907
SIZES = [250, 500, 1000, 2000, 4000, 8000, 16000, 29000]
REPEATS = {250: 4, 500: 4, 1000: 3, 2000: 3, 4000: 2, 8000: 2, 16000: 1, 29000: 1}
HEADLINE_BETA = 0.5


def main():
    shot = sys.argv[1] if len(sys.argv) > 1 else "0shot"
    df, models, _, _ = load(shot)
    util = df[models].to_numpy(float)
    cost = df[[f"{m}|total_cost" for m in models]].to_numpy(float)
    ybin = (util >= 0.5).astype(int)
    emb = np.load(os.path.join(HERE, f"s11_emb_{shot}.npy"))
    assert len(emb) == len(df)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=SEED,
                                      stratify=df.family.to_numpy())
    print(f"pool {len(tr_idx)} train / {len(te_idx)} held out", flush=True)

    U_te, C_te = util[te_idx], cost[te_idx]
    mc = C_te.mean(axis=0)
    budget = float(mc.min() + HEADLINE_BETA * (mc.max() - mc.min()))
    S = static_frontier_at(mc, U_te.mean(axis=0), budget)
    A_star, _, _ = oracle_frontier_at(U_te, C_te, budget)
    kappa = A_star - S
    print(f"held-out kappa at beta=0.5: {100*kappa:.2f} pp (S={100*S:.2f}, A*={100*A_star:.2f})",
          flush=True)

    rows = []
    rng = np.random.default_rng(SEED)
    for n in SIZES:
        if n > len(tr_idx):
            continue
        for rep in range(REPEATS.get(n, 1)):
            sub = rng.choice(tr_idx, n, replace=False)
            for name in ("minilm_logreg", "minilm_mlp"):
                P = np.zeros((len(te_idx), len(models)))
                if name == "minilm_logreg":
                    for m in range(len(models)):
                        y = ybin[sub, m]
                        if y.min() == y.max():
                            P[:, m] = float(y[0])
                            continue
                        clf = LogisticRegression(max_iter=3000)
                        clf.fit(emb[sub], y)
                        P[:, m] = clf.predict_proba(emb[te_idx])[:, 1]
                else:
                    sc = StandardScaler().fit(emb[sub])
                    mlp = MLPRegressor(hidden_layer_sizes=(256, 64), early_stopping=True,
                                       n_iter_no_change=10, validation_fraction=0.15,
                                       max_iter=400, random_state=SEED + rep)
                    mlp.fit(sc.transform(emb[sub]), util[sub])
                    P = np.clip(mlp.predict(sc.transform(emb[te_idx])), 0, 1)
                aucs = [roc_auc_score(ybin[te_idx, m], P[:, m]) for m in range(len(models))
                        if 0 < ybin[te_idx, m].mean() < 1]
                v, vc = router_value_at(U_te, C_te, P, budget)
                rows.append({"shot": shot, "router": name, "n_train": n, "rep": rep,
                             "mean_auc": float(np.mean(aucs)),
                             "gain_pp": float(100 * (v - S)),
                             "rho": float((v - S) / kappa) if kappa > 1e-12 else np.nan,
                             "router_cost": vc, "budget": budget})
                print(f"  n={n:6d} rep={rep} {name:14s} AUC={np.mean(aucs):.4f} "
                      f"gain={100*(v-S):+.2f}pp rho={(v-S)/kappa:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, f"s12b_learning_curve_{shot}.csv"), index=False)

    # ---- power-law extrapolation:  AUC(n) = A - B n^(-c) --------------------
    from scipy.optimize import curve_fit
    fits = {}
    for name, g in out.groupby("router"):
        agg = g.groupby("n_train")["mean_auc"].mean().reset_index()
        gg = g.groupby("n_train")["gain_pp"].mean().reset_index()
        x, y = agg.n_train.to_numpy(float), agg.mean_auc.to_numpy(float)
        rec = {"n": x.tolist(), "auc": y.tolist(), "gain_pp": gg.gain_pp.tolist()}
        try:
            p, _ = curve_fit(lambda t, A, B, c: A - B * t ** (-c), x, y,
                             p0=[0.75, 0.5, 0.3],
                             bounds=([0.5, 0.0, 0.01], [1.0, 10.0, 2.0]), maxfev=40000)
            rec.update({"asymptote_A": float(p[0]), "B": float(p[1]), "c": float(p[2]),
                        "auc_at_1e6": float(p[0] - p[1] * 1e6 ** (-p[2])),
                        "auc_at_1e9": float(p[0] - p[1] * 1e9 ** (-p[2])),
                        "doublings_to_gain_0.01_auc": None})
            # how many doublings from the largest observed n to add 0.01 AUC?
            n0 = float(x.max())
            cur = p[0] - p[1] * n0 ** (-p[2])
            tgt = cur + 0.01
            if tgt < p[0]:
                n_need = (p[1] / (p[0] - tgt)) ** (1.0 / p[2])
                rec["doublings_to_gain_0.01_auc"] = float(np.log2(n_need / n0))
                rec["n_needed_for_0.01_auc"] = float(n_need)
            else:
                rec["doublings_to_gain_0.01_auc"] = float("inf")
        except Exception as e:
            rec["fit_error"] = str(e)
        fits[name] = rec

    res = {"shot": shot, "seed": SEED, "n_held_out": len(te_idx),
           "n_train_pool": len(tr_idx), "kappa_pp": 100 * kappa,
           "S": S, "A_star": A_star, "sizes": SIZES, "repeats": REPEATS,
           "fits": fits,
           "note": "EXPLORATORY, not pre-registered; see DEVIATIONS.md D3"}
    with open(os.path.join(HERE, f"s12b_learning_curve_{shot}.json"), "w") as f:
        json.dump(res, f, indent=1)

    print("\n=== power-law extrapolation ===")
    for name, r in fits.items():
        if "asymptote_A" in r:
            print(f"{name:14s} asymptote A={r['asymptote_A']:.4f}  "
                  f"AUC(1e6)={r['auc_at_1e6']:.4f}  AUC(1e9)={r['auc_at_1e9']:.4f}  "
                  f"doublings for +0.01 AUC = {r['doublings_to_gain_0.01_auc']:.1f}")


if __name__ == "__main__":
    main()

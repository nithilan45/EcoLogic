"""Stage 13d: what actually drives Stage 13c's 9.7% -> 32.6%? (exploratory)

Stage 13c (`s13c_encoder_routerbench.py`) reports that unfreezing
`all-MiniLM-L6-v2` takes realised `rho` from 9.7% to 32.6% on RouterBench, and
DEVIATIONS.md D7 claims "the only difference between the rows is whether the
encoder was updated". That claim does not hold, because the two rows differ in
*two* ways at once:

  * the fine-tuned encoder minimises BCE against **graded** utilities in [0,1]
    with one joint 11-output head (`s13c_...py`: `Ytr = util[tr]`);
  * the frozen logistic reference is fit per model against **binarised**
    labels, `util >= 0.5` (`s13c_...py`: `y = ybin[tr, m]`).

21.1% of RouterBench score cells carry fractional partial credit, and a
matched-cost policy ranks items by the *difference* between two models'
predicted utilities. Binarising at 0.5 therefore discards exactly the
cross-model utility spread the policy consumes, so the comparison confounds
"unfroze the encoder" with "trained on graded rather than thresholded targets".

This isolates the two. Row A holds the encoder **frozen** and changes only the
target and loss to match the fine-tuned run: a linear head on the same cached
embeddings, `BCEWithLogitsLoss` on graded utilities, same epochs, batch, head
learning rate, schedule and inner-val epoch selection. Row B reproduces the
paper's frozen baseline. Row C is a second graded-target frozen control from a
different model class. Five splits, since Stage 13c reports one.

Result (see the JSON): a fully frozen linear probe on graded targets reaches
`rho` = 31.6% +/- 1.6%, against 10.5% +/- 1.8% for the same frozen embeddings
with binarised targets. Stage 13c's unfrozen encoder reached 32.6% on one
split. So the target encoding accounts for essentially the whole jump and
unfreezing accounts for about a point, inside the split-to-split spread.

Two consequences, pulling in opposite directions:

  * Stage 13c's *causal* reading -- that closing `eps` "looks more like a
    representation-learning problem" -- is **not supported** by that
    experiment. It is a supervision-target effect, not a representation effect.
  * the paper's AUC-vs-gain dissociation gets **stronger and cheaper**. Row A
    has 0.055 *lower* mean AUC than row B and 3x its realised gain, on the same
    frozen representation, the same items and the same scoring code, over five
    splits -- no fine-tuning, no GPU and no API spend required to show it.

Cost: $0, CPU only, about 45 s.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from decomp import oracle_frontier_at, router_value_at, static_frontier_at  # noqa: E402
from s11_routerbench import load  # noqa: E402

# Matched to s13c_encoder_routerbench.py so the rows stay comparable.
SEEDS = [20260907, 11, 202, 3033, 77]
HEADLINE_BETA = 0.5
INNER_VAL = 0.05
EPOCHS = 3
BATCH = 32
HEAD_LR = 1e-3


def main():
    shot = sys.argv[1] if len(sys.argv) > 1 else "0shot"

    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    df, models, _, _ = load(shot)
    util = df[models].to_numpy(float)
    cost = df[[f"{m}|total_cost" for m in models]].to_numpy(float)
    ybin = (util >= 0.5).astype(int)
    k = len(models)

    emb = np.load(os.path.join(HERE, f"s11_emb_{shot}.npy"))
    assert len(emb) == len(df), "embedding cache does not match the loaded frame"
    X = torch.tensor(emb, dtype=torch.float32)
    Y = torch.tensor(util, dtype=torch.float32)

    rows = []
    for seed in SEEDS:
        idx = np.arange(len(df))
        tr_all, te = train_test_split(idx, test_size=0.2, random_state=seed,
                                      stratify=df.family.to_numpy())
        rng = np.random.default_rng(seed)
        perm = rng.permutation(tr_all)
        cut = int(len(perm) * (1 - INNER_VAL))
        tr, va = perm[:cut], perm[cut:]

        U_te, C_te = util[te], cost[te]
        mc = C_te.mean(axis=0)
        budget = float(mc.min() + HEADLINE_BETA * (mc.max() - mc.min()))
        S = static_frontier_at(mc, U_te.mean(axis=0), budget)
        A_star, _, _ = oracle_frontier_at(U_te, C_te, budget)
        kappa = A_star - S

        def score_row(P):
            aucs = [roc_auc_score(ybin[te, m], P[:, m]) for m in range(k)
                    if 0 < ybin[te, m].mean() < 1]
            v, vc = router_value_at(U_te, C_te, P, budget)
            return {"mean_auc": float(np.mean(aucs)),
                    "gain_pp": float(100 * (v - S)),
                    "rho": float((v - S) / kappa)}

        # --- A: frozen encoder, graded soft targets, identical loss/head -----
        torch.manual_seed(seed)
        head = nn.Linear(X.shape[1], k)
        opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=0.01)
        lossf = nn.BCEWithLogitsLoss()
        n = len(tr)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=HEAD_LR,
            total_steps=EPOCHS * ((n + BATCH - 1) // BATCH) + 1, pct_start=0.1)
        best = {"val_auc": -1.0}
        for ep in range(1, EPOCHS + 1):
            head.train()
            p = rng.permutation(n)
            for i in range(0, n, BATCH):
                j = tr[p[i:i + BATCH]]
                opt.zero_grad()
                lossf(head(X[j]), Y[j]).backward()
                opt.step()
                sched.step()
            head.eval()
            with torch.no_grad():
                Pva = torch.sigmoid(head(X[va])).numpy()
            vauc = float(np.mean([roc_auc_score(ybin[va, m], Pva[:, m])
                                  for m in range(k)
                                  if 0 < ybin[va, m].mean() < 1]))
            if vauc > best["val_auc"]:
                with torch.no_grad():
                    Pte = torch.sigmoid(head(X[te])).numpy()
                best = {"epoch": ep, "val_auc": vauc, "held_out": score_row(Pte)}

        # --- B: the paper's frozen baseline, per-model, binarised targets ----
        Pl = np.zeros((len(te), k))
        for m in range(k):
            y = ybin[tr, m]
            if y.min() == y.max():
                Pl[:, m] = float(y[0])
                continue
            Pl[:, m] = LogisticRegression(max_iter=3000).fit(
                emb[tr], y).predict_proba(emb[te])[:, 1]

        # --- C: frozen, graded targets, different model class ----------------
        Pr = np.clip(Ridge(alpha=1.0).fit(emb[tr], util[tr]).predict(emb[te]), 0, 1)

        rows.append({
            "seed": seed, "kappa_pp": 100 * kappa,
            "n_train": len(tr), "n_inner_val": len(va), "n_held_out": len(te),
            "A_frozen_graded_bce": best,
            "B_frozen_logreg_binarised": score_row(Pl),
            "C_frozen_ridge_graded": score_row(Pr),
        })
        print(f"seed {seed}: kappa {100*kappa:5.2f}pp | "
              f"A frozen+graded AUC {best['held_out']['mean_auc']:.4f} "
              f"rho {best['held_out']['rho']:6.1%} | "
              f"B frozen+binary  AUC {rows[-1]['B_frozen_logreg_binarised']['mean_auc']:.4f} "
              f"rho {rows[-1]['B_frozen_logreg_binarised']['rho']:6.1%} | "
              f"C ridge+graded   AUC {rows[-1]['C_frozen_ridge_graded']['mean_auc']:.4f} "
              f"rho {rows[-1]['C_frozen_ridge_graded']['rho']:6.1%}", flush=True)

    def agg(key):
        # row A nests its scores under the selected epoch; B and C are flat.
        vals = [r[key]["held_out"] if "held_out" in r[key] else r[key]
                for r in rows]
        rho = [v["rho"] for v in vals]
        auc = [v["mean_auc"] for v in vals]
        gain = [v["gain_pp"] for v in vals]
        return {"rho_mean": float(np.mean(rho)), "rho_sd": float(np.std(rho)),
                "rho_min": float(min(rho)), "rho_max": float(max(rho)),
                "auc_mean": float(np.mean(auc)), "auc_sd": float(np.std(auc)),
                "gain_pp_mean": float(np.mean(gain))}

    summary = {
        "A_frozen_graded_bce": agg("A_frozen_graded_bce"),
        "B_frozen_logreg_binarised": agg("B_frozen_logreg_binarised"),
        "C_frozen_ridge_graded": agg("C_frozen_ridge_graded"),
    }
    # s13c's unfrozen encoder, for reference: one split only.
    s13c = json.load(open(os.path.join(
        HERE, f"s13c_encoder_routerbench_{shot}.json")))
    ref = s13c["encoder_finetuned"]["held_out"]

    out = {
        "what": "isolates graded-vs-binarised supervision from unfreezing",
        "pre_registered": False, "deviation": "DEVIATIONS.md D9",
        "shot": shot, "seeds": SEEDS, "beta": HEADLINE_BETA,
        "epochs": EPOCHS, "batch": BATCH, "head_lr": HEAD_LR,
        "per_seed": rows, "summary": summary,
        "s13c_unfrozen_reference_one_split": {
            "mean_auc": ref["mean_auc"], "gain_pp": ref["gain_pp"],
            "rho": ref["rho"],
        },
        "finding": (
            "A frozen linear probe on graded targets reaches rho = "
            f"{summary['A_frozen_graded_bce']['rho_mean']:.3f} "
            f"(sd {summary['A_frozen_graded_bce']['rho_sd']:.3f}, 5 splits) "
            f"against {summary['B_frozen_logreg_binarised']['rho_mean']:.3f} "
            "for the same frozen embeddings with binarised targets. s13c's "
            f"unfrozen encoder reached {ref['rho']:.3f} on one split, so the "
            "target encoding accounts for essentially all of the 9.7% -> 32.6% "
            "jump and unfreezing for about a point, inside the split spread. "
            "The AUC-vs-gain dissociation is correspondingly strengthened: row "
            "A has lower AUC than row B and ~3x its realised gain."
        ),
        "usd": 0.0,
    }
    p = os.path.join(HERE, f"s13d_target_encoding_ablation_{shot}.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\nwrote", p)
    print(json.dumps(summary, indent=1))
    print("\n" + out["finding"])


if __name__ == "__main__":
    main()

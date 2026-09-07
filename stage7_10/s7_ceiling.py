"""Stage 7c — is the ~0.67 AUC ceiling the task, or just logistic regression?

The central negative claim of this project is that per-item tier success is only
weakly predictable from the prompt, so a router cannot capture the oracle
headroom. That claim currently rests on two linear models (TF-IDF and MiniLM
embeddings, each with a logistic head). The obvious reviewer objection is that we
tested a weak router and then blamed the workload.

This script attacks our own claim with stronger learners on the identical data,
labels and splits: gradient boosting on the MiniLM embeddings, a random forest,
and a k-NN probe. If any of them lifts held-out AUC materially above the linear
result, the "structural limit" reading is wrong and the honest conclusion becomes
"we used an inadequate model class."

Selection discipline is unchanged: everything is fit on TRAIN, scored once on
CALIBRATION. The frozen test set is not touched, so the pre-registered one-shot
property of Stage 7 survives this script entirely.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in ("benchmark", "router_v2", "stage7_10"):
    sys.path.insert(0, str(ROOT / p))
from s7_data import load_split, pool_correct_and_tokens  # noqa: E402
from s7_features import CachedEmbeddingFeatures  # noqa: E402
from s7_fit import make_xy  # noqa: E402

from sklearn.ensemble import (HistGradientBoostingClassifier,  # noqa: E402
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

OUT = ROOT / "stage7_10"
SEED = 20260907
HEAD_TIERS = [1, 2]


def dense_embeddings(texts: list[str]) -> np.ndarray:
    """MiniLM vectors as a dense array (the R2 representation, unchanged)."""
    feat = CachedEmbeddingFeatures()
    return np.asarray(feat.transform(texts).todense(), dtype=float)


def models() -> dict:
    return {
        "logistic (R2 baseline)": lambda: LogisticRegression(
            max_iter=2000, C=1.0, random_state=SEED),
        "gradient boosting": lambda: HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=SEED),
        "random forest": lambda: RandomForestClassifier(
            n_estimators=600, min_samples_leaf=4, n_jobs=-1, random_state=SEED),
        "k-NN (k=50)": lambda: KNeighborsClassifier(n_neighbors=50, n_jobs=-1),
    }


def main() -> None:
    correct, _ = pool_correct_and_tokens()
    tr_items, Xtr_txt, ytr = make_xy(load_split("train"), correct)
    ca_items, Xca_txt, yca = make_xy(load_split("calibration"), correct)
    print(f"TRAIN {len(Xtr_txt)}, CALIBRATION {len(Xca_txt)} "
          f"(labels = majority of k=3 @ temp 0.7)")

    print("encoding with MiniLM (cached)...", flush=True)
    Etr = dense_embeddings(Xtr_txt)
    Eca = dense_embeddings(Xca_txt)
    print(f"  embedding dim {Etr.shape[1]}")

    results = {}
    for name, ctor in models().items():
        per_head = {}
        for t in HEAD_TIERS:
            clf = ctor()
            clf.fit(Etr, ytr[t])
            p = clf.predict_proba(Eca)[:, 1]
            # Train AUC too: a large train/calibration gap means the model has
            # capacity to spare and is memorising, which is itself evidence that
            # the ceiling is not a capacity problem.
            p_tr = clf.predict_proba(Etr)[:, 1]
            per_head[t] = {
                "calibration_auc": float(roc_auc_score(yca[t], p)),
                "train_auc": float(roc_auc_score(ytr[t], p_tr)),
                "calibration_base_rate": float(yca[t].mean()),
            }
        mean_cal = float(np.mean([per_head[t]["calibration_auc"] for t in HEAD_TIERS]))
        mean_tr = float(np.mean([per_head[t]["train_auc"] for t in HEAD_TIERS]))
        results[name] = {"per_head": per_head, "mean_calibration_auc": mean_cal,
                         "mean_train_auc": mean_tr}
        print(f"  {name:24s} CALIB mean AUC {mean_cal:.4f}  "
              f"(t1 {per_head[1]['calibration_auc']:.4f}, "
              f"t2 {per_head[2]['calibration_auc']:.4f})  "
              f"[TRAIN {mean_tr:.4f}]", flush=True)

    base = results["logistic (R2 baseline)"]["mean_calibration_auc"]
    best_name = max((k for k in results if k != "logistic (R2 baseline)"),
                    key=lambda k: results[k]["mean_calibration_auc"])
    best = results[best_name]["mean_calibration_auc"]
    lift = best - base

    verdict = ("ceiling is structural: no stronger learner beats the linear head by "
               ">= 0.02 AUC on held-out data"
               if lift < 0.02 else
               "ceiling is NOT structural: a stronger learner lifts held-out AUC "
               "materially, so the linear model class was the binding constraint")

    result = {
        "n_train": len(Xtr_txt), "n_calibration": len(Xca_txt),
        "representation": "sentence-transformers/all-MiniLM-L6-v2, 384-d, CPU",
        "labels": "majority of k=3 samples at temperature 0.7",
        "seed": SEED,
        "models": results,
        "linear_baseline_mean_calibration_auc": base,
        "best_nonlinear_model": best_name,
        "best_nonlinear_mean_calibration_auc": best,
        "auc_lift_over_linear": lift,
        "threshold_for_structural_claim": 0.02,
        "verdict": verdict,
        "note": ("Fit on TRAIN, scored once on CALIBRATION. The Stage 7 frozen test "
                 "set is not touched, so the pre-registered one-shot property holds."),
    }
    with open(OUT / "s7_ceiling.json", "w") as f:
        json.dump(result, f, indent=2)

    L = ["# Stage 7c — is the predictability ceiling structural or just linear?\n",
         "The project's central negative claim is that per-item tier success is only "
         "weakly predictable from the prompt. That claim rested on two linear models, "
         "so the obvious objection is that we tested a weak router and blamed the "
         "workload. Here the same representation, labels and splits are given to "
         "stronger learners.\n",
         f"TRAIN n = {len(Xtr_txt)}, CALIBRATION n = {len(Xca_txt)}. Fit on TRAIN, "
         "scored once on CALIBRATION. The frozen test set is untouched.\n",
         "| Model | Tier 1 AUC | Tier 2 AUC | Mean CALIB AUC | Mean TRAIN AUC |",
         "|---|---|---|---|---|"]
    for name, r in results.items():
        ph = r["per_head"]
        L.append(f"| {name} | {ph[1]['calibration_auc']:.4f} | "
                 f"{ph[2]['calibration_auc']:.4f} | "
                 f"**{r['mean_calibration_auc']:.4f}** | {r['mean_train_auc']:.4f} |")
    L += ["", f"Best non-linear model: **{best_name}**, mean CALIBRATION AUC "
              f"{best:.4f} versus the linear head's {base:.4f} — a lift of "
              f"**{lift:+.4f}**.", "",
          f"**Verdict: {verdict}.**", "",
          "The TRAIN column is the useful diagnostic. Where a model's train AUC runs "
          "far above its calibration AUC it has capacity to spare and is spending it "
          "on memorisation, which means the held-out ceiling is a property of the "
          "signal available in the prompt rather than of model capacity.", ""]
    with open(OUT / "s7_ceiling.md", "w") as f:
        f.write("\n".join(L))

    print()
    print(f"linear baseline {base:.4f} -> best non-linear {best:.4f} "
          f"({lift:+.4f})")
    print(f"VERDICT: {verdict}")
    print("wrote stage7_10/s7_ceiling.json, s7_ceiling.md")


if __name__ == "__main__":
    main()

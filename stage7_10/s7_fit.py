"""Stage 7 - refit both router variants on the new, larger, majority-voted labels.

Same procedure as Stage 2, same code paths (the featurizers, CV routine and
fitting helpers are imported from router_v2/train_router.py so they cannot
drift). The only differences are the training data and the labels: ~4.2x more
items, and per-tier correctness from a majority of k=3 samples rather than a
single sample.

Variant and hyperparameters are chosen by 5-fold cross-validation WITHIN TRAIN
only; CALIBRATION is scored once afterwards for both variants so the ablation
stays visible. The new frozen test set is not touched here.

Writes stage7_10/s7_model_comparison.md, s7_model_selection.json, s7_router_model.pkl.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in ("benchmark", "router_v2", "stage7_10"):
    sys.path.insert(0, str(ROOT / p))
from s7_data import load_split, pool_correct_and_tokens  # noqa: E402
from s7_features import build_r2_cached  # noqa: E402
from train_router import (N_FOLDS, SEED, build_r1, cv_score, eval_probs,  # noqa: E402
                          fit_final, predict)

OUT = ROOT / "stage7_10"
HEAD_TIERS = [1, 2]


def make_xy(items, correct):
    keep = [it for it in items if all((t, it["item_id"]) in correct for t in (1, 2, 3))]
    X = [it["raw_query"] for it in keep]
    y = {t: np.array([int(correct[(t, it["item_id"])]) for it in keep]) for t in HEAD_TIERS}
    return keep, X, y


def main():
    correct, _ = pool_correct_and_tokens()
    train_items, Xtr, ytr = make_xy(load_split("train"), correct)
    cal_items, Xca, yca = make_xy(load_split("calibration"), correct)
    print(f"TRAIN {len(Xtr)} items, CALIBRATION {len(Xca)} items "
          f"(labels = majority of k=3 @ temp 0.7)")
    for t in HEAD_TIERS:
        print(f"  tier {t} majority-correct base rate: train {ytr[t].mean():.3f} "
              f"calib {yca[t].mean():.3f}")

    r1_grid = [(cn, md, C) for cn in [(2, 4), (3, 5)] for md in [1, 2]
               for C in [0.1, 1.0, 10.0]]
    r1_results = []
    print("\nR1 grid (TF-IDF + hand), 5-fold CV within TRAIN:")
    for cn, md, C in r1_grid:
        s = cv_score(lambda cn=cn, md=md: build_r1(cn, md), C, Xtr, ytr)
        r1_results.append({"char_ngrams": list(cn), "min_df": md, "C": C, **s})
        print(f"  char{cn} min_df={md} C={C:<5} mean_auc={s['mean_auc']:.4f} "
              f"(t1 {s['per_head_auc'][1]:.4f}, t2 {s['per_head_auc'][2]:.4f})", flush=True)
    best_r1 = max(r1_results, key=lambda r: r["mean_auc"])

    r2_results = []
    print("\nR2 grid (MiniLM embeddings + hand), 5-fold CV within TRAIN:")
    for C in [0.1, 1.0, 10.0]:
        s = cv_score(build_r2_cached, C, Xtr, ytr)
        r2_results.append({"C": C, **s})
        print(f"  C={C:<5} mean_auc={s['mean_auc']:.4f} "
              f"(t1 {s['per_head_auc'][1]:.4f}, t2 {s['per_head_auc'][2]:.4f})", flush=True)
    best_r2 = max(r2_results, key=lambda r: r["mean_auc"])

    winner = "R1" if best_r1["mean_auc"] >= best_r2["mean_auc"] else "R2"
    print(f"\nTRAIN-CV winner: {winner} "
          f"(R1 {best_r1['mean_auc']:.4f} vs R2 {best_r2['mean_auc']:.4f})")

    m1 = fit_final(lambda: build_r1(tuple(best_r1["char_ngrams"]), best_r1["min_df"]),
                   best_r1["C"], Xtr, ytr)
    m2 = fit_final(build_r2_cached, best_r2["C"], Xtr, ytr)
    cal1 = eval_probs(predict(m1, Xca), yca)
    cal2 = eval_probs(predict(m2, Xca), yca)

    with open(OUT / "s7_router_model.pkl", "wb") as f:
        pickle.dump({"variant": winner, "model": m1 if winner == "R1" else m2,
                     "head_tiers": HEAD_TIERS, "seed": SEED}, f)

    # Stage 2's numbers, for the data-scaling comparison
    with open(ROOT / "router_v2" / "model_selection.json") as f:
        s2 = json.load(f)

    summary = {
        "winner": winner, "best_r1": best_r1, "best_r2": best_r2,
        "r1_grid": r1_results, "r2_grid": r2_results,
        "calibration_r1": cal1, "calibration_r2": cal2,
        "n_train": len(Xtr), "n_calibration": len(Xca),
        "label_rule": "majority of k=3 samples at temperature 0.7",
        "train_base_rates": {str(t): float(ytr[t].mean()) for t in HEAD_TIERS},
        "calibration_base_rates": {str(t): float(yca[t].mean()) for t in HEAD_TIERS},
        "stage2_reference": {"winner": s2["winner"], "n_train": s2["n_train"],
                             "best_r1_mean_auc": s2["best_r1"]["mean_auc"],
                             "best_r2_mean_auc": s2["best_r2"]["mean_auc"]},
    }
    with open(OUT / "s7_model_selection.json", "w") as f:
        json.dump(summary, f, indent=2)

    L = []
    L.append("# Stage 7 — R1 vs R2 ablation on the scaled, majority-voted pool\n")
    L.append(f"Same two variants and the same selection procedure as Stage 2, refit on "
             f"**{len(Xtr)} TRAIN items** (Stage 2: {s2['n_train']}) with per-tier labels "
             f"from a **majority of k=3 samples at temperature 0.7** instead of a single "
             f"temperature-0 sample. Selection is 5-fold cross-validation within TRAIN "
             f"only; CALIBRATION (n={len(Xca)}) is scored once, afterwards, for both "
             f"variants.\n")
    L.append("## TRAIN-CV selection metric: mean ROC-AUC across the two tier heads\n")
    L.append("| Variant | Best config | Tier 1 AUC | Tier 2 AUC | Mean AUC | Stage 2 mean AUC |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| R1 | char{tuple(best_r1['char_ngrams'])}, min_df={best_r1['min_df']}, "
             f"C={best_r1['C']} | {best_r1['per_head_auc'][1]:.4f} | "
             f"{best_r1['per_head_auc'][2]:.4f} | **{best_r1['mean_auc']:.4f}** | "
             f"{s2['best_r1']['mean_auc']:.4f} |")
    L.append(f"| R2 | C={best_r2['C']} | {best_r2['per_head_auc'][1]:.4f} | "
             f"{best_r2['per_head_auc'][2]:.4f} | **{best_r2['mean_auc']:.4f}** | "
             f"{s2['best_r2']['mean_auc']:.4f} |")
    L.append(f"\n**TRAIN-CV winner: {winner}.**\n")
    L.append("The Stage 2 column is the direct test of the data-scaling hypothesis at the "
             "level of the probability heads: if the calibration gap were caused by "
             "insufficient or noisy labels, these AUCs should be visibly higher than "
             "Stage 2's.\n")
    L.append("## Confirmation on CALIBRATION (scored once, both variants)\n")
    L.append("| Variant | Head | Base rate | AUC | Brier | Log loss |")
    L.append("|---|---|---|---|---|---|")
    for label, cal in (("R1", cal1), ("R2", cal2)):
        for t in HEAD_TIERS:
            c = cal[t]
            auc = f"{c['auc']:.4f}" if c["auc"] is not None else "n/a"
            L.append(f"| {label} | Tier {t} P(correct) | {c['base_rate']:.3f} | {auc} | "
                     f"{c['brier']:.4f} | {c['log_loss']:.4f} |")
    L.append("")
    L.append("## Full TRAIN-CV grids\n")
    L.append("| Variant | char n-grams | min_df | C | Tier 1 AUC | Tier 2 AUC | Mean AUC |")
    L.append("|---|---|---|---|---|---|---|")
    for r in r1_results:
        L.append(f"| R1 | {tuple(r['char_ngrams'])} | {r['min_df']} | {r['C']} | "
                 f"{r['per_head_auc'][1]:.4f} | {r['per_head_auc'][2]:.4f} | "
                 f"{r['mean_auc']:.4f} |")
    for r in r2_results:
        L.append(f"| R2 | (embeddings) | - | {r['C']} | {r['per_head_auc'][1]:.4f} | "
                 f"{r['per_head_auc'][2]:.4f} | {r['mean_auc']:.4f} |")
    L.append("")
    with open(OUT / "s7_model_comparison.md", "w") as f:
        f.write("\n".join(L))
    print("wrote stage7_10/s7_model_comparison.md, s7_model_selection.json, "
          "s7_router_model.pkl")


if __name__ == "__main__":
    main()

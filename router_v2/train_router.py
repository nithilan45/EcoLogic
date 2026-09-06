"""Stage 2 - fit the two learned-router variants.

R1  TF-IDF (word + char n-grams) + hand-engineered features -> per-tier
    logistic regression predicting P(tier t answers correctly | query).
    This adapts the core idea of RouteLLM (Ong et al., ICLR 2025) -- a
    predicted success/win probability driving a cost threshold -- to a
    locally-computed feature space. It is NOT a reproduction of their method:
    RouteLLM learns from human preference data with a hosted preference model,
    whereas this fits directly on observed per-tier correctness using features
    computed on CPU with zero marginal API cost.

R2  all-MiniLM-L6-v2 local sentence embeddings + the same hand features and the
    same per-tier logistic regression. Ablation against R1: does a richer but
    still local representation help?

Architecture and hyperparameters are selected by cross-validation WITHIN TRAIN
only. CALIBRATION is scored once afterwards, for both variants, so the ablation
is visible even though only the TRAIN-CV winner proceeds. The frozen test set is
not touched.

Writes router_v2/model_comparison.md and router_v2/router_model.pkl.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "router_v2"))
from features import Combine, EmbeddingFeatures, HandFeatures  # noqa: E402

OUT = ROOT / "router_v2"
HEAD_TIERS = [1, 2]
SEED = 771113
N_FOLDS = 5


def load_split(name: str) -> list[dict]:
    with open(OUT / f"{name}_pool.json") as f:
        return json.load(f)["items"]


def open_graded():
    """pool_graded.jsonl is stored gzipped in the repo; accept either form."""
    plain = OUT / "pool_graded.jsonl"
    if plain.exists():
        return open(plain)
    import gzip
    return gzip.open(plain.with_suffix(".jsonl.gz"), "rt")


def load_correct() -> dict:
    correct = {}
    with open_graded() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                correct[(r["tier"], r["item_id"])] = bool(r.get("correct"))
    return correct


def make_xy(items, correct):
    keep = [it for it in items if all((t, it["item_id"]) in correct for t in (1, 2, 3))]
    X = [it["raw_query"] for it in keep]
    y = {t: np.array([int(correct[(t, it["item_id"])]) for it in keep]) for t in HEAD_TIERS}
    return keep, X, y


def build_r1(char_ngrams, min_df):
    return Combine([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=min_df,
                                 sublinear_tf=True, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngrams, min_df=min_df,
                                 sublinear_tf=True)),
        ("hand", HandFeatures()),
    ])


def build_r2():
    return Combine([("emb", EmbeddingFeatures()), ("hand", HandFeatures())])


def cv_score(build_featurizer, C, X, y) -> dict:
    """Mean ROC-AUC over the two tier heads, cross-validated within TRAIN."""
    X = np.asarray(X, dtype=object)
    per_head = {}
    for t in HEAD_TIERS:
        yt = y[t]
        if len(np.unique(yt)) < 2:
            per_head[t] = float("nan")
            continue
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        oof = np.zeros(len(yt), dtype=float)
        for tr, va in skf.split(X, yt):
            fz = build_featurizer()
            Xtr = fz.fit(X[tr], yt[tr]).transform(X[tr])
            Xva = fz.transform(X[va])
            clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
            clf.fit(Xtr, yt[tr])
            oof[va] = clf.predict_proba(Xva)[:, 1]
        per_head[t] = float(roc_auc_score(yt, oof))
    vals = [v for v in per_head.values() if not np.isnan(v)]
    return {"per_head_auc": per_head, "mean_auc": float(np.mean(vals)) if vals else float("nan")}


def fit_final(build_featurizer, C, X, y):
    X = np.asarray(X, dtype=object)
    fz = build_featurizer()
    Xt = fz.fit(X, None).transform(X)
    heads = {}
    for t in HEAD_TIERS:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        clf.fit(Xt, y[t])
        heads[t] = clf
    return {"featurizer": fz, "heads": heads, "C": C}


def predict(model, X) -> dict:
    Xt = model["featurizer"].transform(np.asarray(X, dtype=object))
    return {t: model["heads"][t].predict_proba(Xt)[:, 1] for t in HEAD_TIERS}


def eval_probs(p, y) -> dict:
    out = {}
    for t in HEAD_TIERS:
        yt = y[t]
        out[t] = {
            "base_rate": float(yt.mean()),
            "auc": float(roc_auc_score(yt, p[t])) if len(np.unique(yt)) > 1 else None,
            "brier": float(brier_score_loss(yt, p[t])),
            "log_loss": float(log_loss(yt, np.clip(p[t], 1e-6, 1 - 1e-6), labels=[0, 1])),
        }
    return out


def main():
    correct = load_correct()
    train_items, Xtr, ytr = make_xy(load_split("train"), correct)
    cal_items, Xca, yca = make_xy(load_split("calibration"), correct)
    print(f"TRAIN {len(Xtr)} items, CALIBRATION {len(Xca)} items")
    for t in HEAD_TIERS:
        print(f"  tier {t} correct base rate: train {ytr[t].mean():.3f} "
              f"calib {yca[t].mean():.3f}")

    # ---------- hyperparameter search, TRAIN CV only ----------
    r1_grid = [(cn, md, C) for cn in [(2, 4), (3, 5)] for md in [1, 2] for C in [0.1, 1.0, 10.0]]
    r1_results = []
    print("\nR1 grid (TF-IDF + hand), 5-fold CV within TRAIN:")
    for cn, md, C in r1_grid:
        s = cv_score(lambda cn=cn, md=md: build_r1(cn, md), C, Xtr, ytr)
        r1_results.append({"char_ngrams": list(cn), "min_df": md, "C": C, **s})
        print(f"  char{cn} min_df={md} C={C:<5} mean_auc={s['mean_auc']:.4f} "
              f"(t1 {s['per_head_auc'][1]:.4f}, t2 {s['per_head_auc'][2]:.4f})")
    best_r1 = max(r1_results, key=lambda r: r["mean_auc"])

    r2_results = []
    print("\nR2 grid (MiniLM embeddings + hand), 5-fold CV within TRAIN:")
    for C in [0.1, 1.0, 10.0]:
        s = cv_score(build_r2, C, Xtr, ytr)
        r2_results.append({"C": C, **s})
        print(f"  C={C:<5} mean_auc={s['mean_auc']:.4f} "
              f"(t1 {s['per_head_auc'][1]:.4f}, t2 {s['per_head_auc'][2]:.4f})")
    best_r2 = max(r2_results, key=lambda r: r["mean_auc"])

    winner = "R1" if best_r1["mean_auc"] >= best_r2["mean_auc"] else "R2"
    print(f"\nTRAIN-CV winner: {winner} "
          f"(R1 {best_r1['mean_auc']:.4f} vs R2 {best_r2['mean_auc']:.4f})")

    # ---------- fit both on full TRAIN, score both once on CALIBRATION ----------
    m1 = fit_final(lambda: build_r1(tuple(best_r1["char_ngrams"]), best_r1["min_df"]),
                   best_r1["C"], Xtr, ytr)
    m2 = fit_final(build_r2, best_r2["C"], Xtr, ytr)
    cal1 = eval_probs(predict(m1, Xca), yca)
    cal2 = eval_probs(predict(m2, Xca), yca)

    chosen_model = m1 if winner == "R1" else m2
    with open(OUT / "router_model.pkl", "wb") as f:
        pickle.dump({"variant": winner, "model": chosen_model,
                     "head_tiers": HEAD_TIERS, "seed": SEED}, f)

    summary = {
        "winner": winner,
        "best_r1": best_r1, "best_r2": best_r2,
        "r1_grid": r1_results, "r2_grid": r2_results,
        "calibration_r1": cal1, "calibration_r2": cal2,
        "n_train": len(Xtr), "n_calibration": len(Xca),
        "train_base_rates": {str(t): float(ytr[t].mean()) for t in HEAD_TIERS},
        "calibration_base_rates": {str(t): float(yca[t].mean()) for t in HEAD_TIERS},
    }
    with open(OUT / "model_selection.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---------- ablation write-up ----------
    L = []
    L.append("# R1 vs R2 ablation (Stage 2)\n")
    L.append("Both variants predict `P(tier t answers correctly | query)` for Tier 1 and "
             "Tier 2 with a per-tier logistic regression, and both compute their features "
             "entirely on CPU with **zero marginal API cost**, preserving EcoLogic's "
             "constraint that routing must not itself call an LLM.\n")
    L.append("This adapts the core idea of **RouteLLM** (Ong et al., *RouteLLM: Learning to "
             "Route LLMs from Preference Data*, ICLR 2025) -- a predicted success probability "
             "driving a cost threshold -- to a local feature space. It is **not** a "
             "reproduction of their method: RouteLLM learns from human preference data via a "
             "hosted preference model, whereas these variants fit directly on observed "
             "per-tier correctness using locally-computed features.\n")
    L.append(f"- **R1**: TF-IDF word 1-2 grams + char_wb n-grams + 25 hand-engineered "
             f"features (length, code-syntax markers, question words, numeric content, "
             f"multiple-choice markers).")
    L.append(f"- **R2**: `all-MiniLM-L6-v2` sentence embeddings (384-d, CPU) + the same 25 "
             f"hand features.\n")
    L.append(f"Architecture and hyperparameters were chosen by {N_FOLDS}-fold "
             f"cross-validation **within TRAIN only** "
             f"(n={len(Xtr)}); CALIBRATION (n={len(Xca)}) was scored once, afterwards.\n")

    L.append("## Selection metric: mean ROC-AUC across the two tier heads (TRAIN CV)\n")
    L.append("| Variant | Best config | Tier 1 AUC | Tier 2 AUC | Mean AUC |")
    L.append("|---|---|---|---|---|")
    L.append(f"| R1 | char{tuple(best_r1['char_ngrams'])}, min_df={best_r1['min_df']}, "
             f"C={best_r1['C']} | {best_r1['per_head_auc'][1]:.4f} | "
             f"{best_r1['per_head_auc'][2]:.4f} | **{best_r1['mean_auc']:.4f}** |")
    L.append(f"| R2 | C={best_r2['C']} | {best_r2['per_head_auc'][1]:.4f} | "
             f"{best_r2['per_head_auc'][2]:.4f} | **{best_r2['mean_auc']:.4f}** |")
    L.append(f"\n**TRAIN-CV winner: {winner}.** This is the variant carried to Stage 3 and, "
             f"once, to the frozen test set.\n")

    L.append("## Confirmation on CALIBRATION (scored once, both variants)\n")
    L.append("| Variant | Head | Base rate | AUC | Brier | Log loss |")
    L.append("|---|---|---|---|---|---|")
    for label, cal in (("R1", cal1), ("R2", cal2)):
        for t in HEAD_TIERS:
            c = cal[t]
            auc = f"{c['auc']:.4f}" if c["auc"] is not None else "n/a"
            L.append(f"| {label} | Tier {t} P(correct) | {c['base_rate']:.3f} | {auc} "
                     f"| {c['brier']:.4f} | {c['log_loss']:.4f} |")
    L.append("")
    L.append("An AUC near 0.5 means the head cannot tell which queries that tier will get "
             "right. Read these numbers before reading any downstream routing result: the "
             "whole learned-router approach depends on these heads carrying signal.\n")

    L.append("## Full TRAIN-CV grids\n")
    L.append("| Variant | char n-grams | min_df | C | Tier 1 AUC | Tier 2 AUC | Mean AUC |")
    L.append("|---|---|---|---|---|---|---|")
    for r in r1_results:
        L.append(f"| R1 | {tuple(r['char_ngrams'])} | {r['min_df']} | {r['C']} | "
                 f"{r['per_head_auc'][1]:.4f} | {r['per_head_auc'][2]:.4f} | {r['mean_auc']:.4f} |")
    for r in r2_results:
        L.append(f"| R2 | (embeddings) | - | {r['C']} | {r['per_head_auc'][1]:.4f} | "
                 f"{r['per_head_auc'][2]:.4f} | {r['mean_auc']:.4f} |")
    L.append("")

    with open(OUT / "model_comparison.md", "w") as f:
        f.write("\n".join(L))
    print(f"\nwrote router_v2/model_comparison.md, model_selection.json, router_model.pkl")


if __name__ == "__main__":
    main()

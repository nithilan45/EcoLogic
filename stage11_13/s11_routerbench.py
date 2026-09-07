"""Stage 11 — the decomposition measured on RouterBench (11 models x 8 benchmark families).

Computes, for every model pair and every benchmark family:

  kappa(b)  = A*(b) - S(b)          complementarity: oracle over matched-cost static
  gain(b)   = A_router(b) - S(b)    what a fitted router actually realises
  rho_hat   = gain / kappa           the realised share of the headroom
  ceiling   = sqrt(b(1-b)) sd(D)    the LOOSE router-free cap available without replicates

Protocol is fixed by `prereg_stage11_13.md` sections 2.1-2.6. Nothing here is
tuned on the quantities it reports.

Outputs: s11_routerbench.json, s11_routerbench_pairs.csv, s11_routerbench.md
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from decomp import (  # noqa: E402
    benjamini_hochberg,
    binary_cells,
    ceiling_from_sd,
    holm_bonferroni,
    oracle_frontier_at,
    oracle_integer_at,
    router_value_at,
    static_frontier_at,
)

DATA = os.path.join(os.path.dirname(HERE), "external_data")
SEED = 20260907
BETAS = np.round(np.arange(0.1, 0.95, 0.1), 2)
HEADLINE_BETA = 0.5
N_BOOT = 2000
MIN_FAMILY_N = 100

# ---- family map, fixed in prereg 2.3 --------------------------------------
FRACTIONAL_EVALS = {"grade-school-math", "mtbench", "mtbench-math",
                    "mtbench-reference", "consensus_summary", "chinese_idioms"}
EXCLUDE_EVALS = {"test-match"}


def family_of(eval_name: str) -> str:
    e = eval_name
    if e.startswith("mmlu-"):
        return "mmlu"
    if e == "grade-school-math":
        return "gsm8k"
    if e in ("hellaswag", "winogrande", "mbpp"):
        return e
    if e == "arc-challenge":
        return "arc"
    if e.startswith("mtbench"):
        return "mtbench"
    if e.lower().startswith("chinese"):
        return "chinese"
    return "other"


# ---------------------------------------------------------------------------
def load(shot: str):
    path = os.path.join(DATA, f"routerbench_{shot}.pkl")
    df = pd.read_pickle(path)
    with open(path, "rb") as f:
        h = hashlib.sha256()
        while chunk := f.read(1 << 22):
            h.update(chunk)
    models = [c for c in df.columns if "|" not in c and c not in
              ("sample_id", "prompt", "eval_name", "oracle_model_to_route_to")]
    models = sorted(models)
    df = df[~df.eval_name.isin(EXCLUDE_EVALS)].reset_index(drop=True)
    df["family"] = df.eval_name.map(family_of)

    # The 5-shot release has 154 missing score cells spread over 28 arc-challenge
    # items (the 0-shot release has none). An item with a missing score for some
    # model has no defined utility vector, so it is dropped rather than imputed;
    # the count is returned so it can be reported. See DEVIATIONS.md D6.
    cols = models + [f"{m}|total_cost" for m in models]
    keep = df[cols].notna().all(axis=1).to_numpy()
    dropped = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    return df, models, h.hexdigest(), dropped


def flatten_prompt(p):
    if isinstance(p, str):
        try:
            v = ast.literal_eval(p)
        except (ValueError, SyntaxError):
            return p
    else:
        v = p
    if isinstance(v, (list, tuple)):
        return "\n".join(str(x) for x in v)
    return str(v)


# ---------------------------------------------------------------------------
def build_features(texts, cache_tag):
    """MiniLM embeddings (cached to disk) and a TF-IDF matrix."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack

    emb_path = os.path.join(HERE, f"s11_emb_{cache_tag}.npy")
    if os.path.exists(emb_path) and len(np.load(emb_path, mmap_mode="r")) == len(texts):
        emb = np.load(emb_path)
    else:
        from sentence_transformers import SentenceTransformer
        print(f"  embedding {len(texts)} prompts with all-MiniLM-L6-v2 ...", flush=True)
        m = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        m.max_seq_length = 256
        t0 = time.time()
        emb = m.encode(list(texts), batch_size=128, convert_to_numpy=True,
                       show_progress_bar=False, normalize_embeddings=True)
        print(f"  embedded in {time.time()-t0:.0f}s", flush=True)
        np.save(emb_path, emb)

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000,
                           sublinear_tf=True, strip_accents="unicode")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           max_features=200_000, sublinear_tf=True)
    tfidf = hstack([word.fit_transform(texts), char.fit_transform(texts)]).tocsr()
    return emb, tfidf


def fit_routers(emb, tfidf, util, families, seed=SEED):
    """5-fold out-of-fold predictions of per-model utility, for each router family.

    Folds are grouped by item (each item is in exactly one fold) and stratified by
    benchmark family, so no item is scored by a model that saw it. Hyperparameters
    are fixed in advance; nothing is selected on the reported quantities.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    n, k = util.shape
    ybin = (util >= 0.5).astype(int)
    preds = {name: np.full((n, k), np.nan) for name in
             ("tfidf_logreg", "minilm_logreg", "minilm_mlp", "minilm_gbm")}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (tr, te) in enumerate(skf.split(np.zeros(n), families)):
        t0 = time.time()
        # --- linear heads on both representations (Stage 5/7 R1 and R2) ---
        for name, X in (("tfidf_logreg", tfidf), ("minilm_logreg", emb)):
            Xtr, Xte = X[tr], X[te]
            for m in range(k):
                ytr = ybin[tr, m]
                if ytr.min() == ytr.max():
                    preds[name][te, m] = float(ytr[0])
                    continue
                clf = LogisticRegression(max_iter=3000, C=1.0)
                clf.fit(Xtr, ytr)
                preds[name][te, m] = clf.predict_proba(Xte)[:, 1]

        # --- MLP on frozen embeddings: the pre-registered answer to "linear head" ---
        sc = StandardScaler().fit(emb[tr])
        Etr, Ete = sc.transform(emb[tr]), sc.transform(emb[te])
        mlp = MLPRegressor(hidden_layer_sizes=(256, 64), activation="relu",
                           early_stopping=True, n_iter_no_change=10,
                           validation_fraction=0.15, max_iter=300,
                           random_state=seed, learning_rate_init=1e-3)
        mlp.fit(Etr, util[tr])
        preds["minilm_mlp"][te] = np.clip(mlp.predict(Ete), 0.0, 1.0)

        # --- boosted trees on frozen embeddings ---
        for m in range(k):
            g = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                              early_stopping=True, random_state=seed)
            g.fit(emb[tr], util[tr, m])
            preds["minilm_gbm"][te, m] = np.clip(g.predict(emb[te]), 0.0, 1.0)
        print(f"  fold {fold+1}/5 fitted in {time.time()-t0:.0f}s", flush=True)

    for name in preds:
        assert not np.isnan(preds[name]).any(), f"{name} has unfilled predictions"
    return preds


# ---------------------------------------------------------------------------
def pair_row(u, c, preds, beta, ia, ib, strictly_binary, with_integer=False):
    """One (family, pair, beta) row of the decomposition.

    Columns are reordered so index 0 is the cheaper model and index 1 the dearer,
    which is what `beta` (the fraction of traffic sent to the dearer model) refers to.
    """
    idx = [ia, ib] if c[:, ia].mean() <= c[:, ib].mean() else [ib, ia]
    uu, cc = u[:, idx], c[:, idx]
    mean_c, mean_u = cc.mean(axis=0), uu.mean(axis=0)
    budget = float((1 - beta) * mean_c[0] + beta * mean_c[1])

    S = static_frontier_at(mean_c, mean_u, budget)
    A_star, star_cost, _ = oracle_frontier_at(uu, cc, budget)
    kappa = A_star - S

    row = {
        "beta": float(beta), "budget": budget,
        "cheap_model_is_a": bool(idx[0] == ia),
        "mean_cost_cheap": float(mean_c[0]), "mean_cost_exp": float(mean_c[1]),
        "acc_cheap": float(mean_u[0]), "acc_exp": float(mean_u[1]),
        "S": float(S), "A_star": float(A_star), "A_star_cost": float(star_cost),
        "kappa": float(kappa),
    }
    if with_integer:
        A_int, _ = oracle_integer_at(uu, cc, budget)
        row["A_star_integer"] = float(A_int)
        row["discreteness_gap"] = float(A_star - A_int)

    D = uu[:, 1] - uu[:, 0]
    row["sd_D_observed"] = float(np.std(D))
    row["ceiling_loose"] = ceiling_from_sd(row["sd_D_observed"], beta)

    if strictly_binary:
        row.update({f"cell_{k2}": v for k2, v in binary_cells(uu[:, 0], uu[:, 1]).items()})

    for name, P in preds.items():
        pv, pc = router_value_at(uu, cc, P[:, idx], budget)
        row[f"router_{name}_acc"] = float(pv)
        row[f"router_{name}_cost"] = float(pc)
        row[f"router_{name}_gain"] = float(pv - S)
        row[f"router_{name}_rho"] = float((pv - S) / kappa) if kappa > 1e-12 else float("nan")
    return row


def main():
    shot = sys.argv[1] if len(sys.argv) > 1 else "0shot"
    print(f"=== Stage 11: RouterBench {shot} ===", flush=True)
    df, models, sha, dropped = load(shot)
    print(f"loaded {len(df)} items x {len(models)} models  sha256={sha[:16]}  "
          f"dropped {dropped} items with missing cells", flush=True)

    util = df[models].to_numpy(float)
    cost = df[[f"{m}|total_cost" for m in models]].to_numpy(float)
    texts = [flatten_prompt(p) for p in df.prompt]

    emb, tfidf = build_features(texts, shot)
    print(f"features: emb {emb.shape}, tfidf {tfidf.shape}", flush=True)
    preds = fit_routers(emb, tfidf, util, df.family.to_numpy())

    # global out-of-fold router quality, for the record
    from sklearn.metrics import roc_auc_score
    auc_table = {}
    ybin = (util >= 0.5).astype(int)
    for name, P in preds.items():
        aucs = [roc_auc_score(ybin[:, m], P[:, m]) for m in range(len(models))
                if 0 < ybin[:, m].mean() < 1]
        auc_table[name] = {"mean_auc": float(np.mean(aucs)),
                           "per_model": {models[m]: float(roc_auc_score(ybin[:, m], P[:, m]))
                                         for m in range(len(models))
                                         if 0 < ybin[:, m].mean() < 1}}
        print(f"  {name:15s} mean out-of-fold AUC over 11 models = {np.mean(aucs):.4f}", flush=True)

    families = [f for f in ["mmlu", "gsm8k", "hellaswag", "winogrande", "arc",
                            "mbpp", "mtbench", "chinese", "other"]
                if (df.family == f).sum() > 0]
    rows = []
    rng = np.random.default_rng(SEED)

    for fam in families:
        sel = (df.family == fam).to_numpy()
        n_f = int(sel.sum())
        strictly_binary = not bool(set(df.eval_name[sel]) & FRACTIONAL_EVALS)
        u_f, c_f = util[sel], cost[sel]
        p_f = {k2: v[sel] for k2, v in preds.items()}
        print(f"[{fam}] n={n_f} strictly_binary={strictly_binary}", flush=True)

        # full 11-model problem
        for beta in BETAS:
            mean_c, mean_u = c_f.mean(axis=0), u_f.mean(axis=0)
            budget = float(mean_c.min() + beta * (mean_c.max() - mean_c.min()))
            S = static_frontier_at(mean_c, mean_u, budget)
            A_star, sc_, _ = oracle_frontier_at(u_f, c_f, budget)
            r = {"shot": shot, "family": fam, "n": n_f, "scope": "all11",
                 "model_a": "ALL", "model_b": "ALL", "beta": float(beta),
                 "budget": budget, "S": float(S), "A_star": float(A_star),
                 "kappa": float(A_star - S), "strictly_binary": strictly_binary}
            for name, P in p_f.items():
                pv, pc = router_value_at(u_f, c_f, P, budget)
                r[f"router_{name}_acc"] = float(pv)
                r[f"router_{name}_gain"] = float(pv - S)
                r[f"router_{name}_rho"] = (float((pv - S) / (A_star - S))
                                           if A_star - S > 1e-12 else float("nan"))
            rows.append(r)

        # every unordered pair
        for ia, ib in combinations(range(len(models)), 2):
            for beta in BETAS:
                r = pair_row(u_f, c_f, p_f, beta, ia, ib, strictly_binary,
                             with_integer=bool(abs(beta - HEADLINE_BETA) < 1e-9))
                r.update({"shot": shot, "family": fam, "n": n_f, "scope": "pair",
                          "model_a": models[ia], "model_b": models[ib],
                          "strictly_binary": strictly_binary})
                rows.append(r)
        print(f"[{fam}] done", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, f"s11_routerbench_pairs_{shot}.csv.gz"),
               index=False, compression="gzip")

    # ---------------- headline aggregation + bootstrap + corrections ----------
    router_names = list(preds.keys())
    head = out[(out.scope == "pair") & (out.beta == HEADLINE_BETA) &
               (out.n >= MIN_FAMILY_N)].copy()
    best_router = max(router_names, key=lambda nm: head[f"router_{nm}_gain"].median())

    summary = {
        "shot": shot, "sha256": sha, "n_items": int(len(df)),
        "n_items_dropped_missing_cells": dropped, "n_models": len(models),
        "models": models, "seed": SEED, "n_boot": N_BOOT,
        "headline_beta": HEADLINE_BETA, "min_family_n": MIN_FAMILY_N,
        "family_sizes": {f: int((df.family == f).sum()) for f in families},
        "out_of_fold_auc": auc_table,
        "best_router_by_median_gain": best_router,
        "n_pairs_scored": int(len(head)),
    }

    def med_ci(vals):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        if not len(v):
            return {"median": float("nan"), "lo": float("nan"), "hi": float("nan")}
        bs = [float(np.median(rng.choice(v, len(v), replace=True))) for _ in range(1000)]
        return {"median": float(np.median(v)),
                "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}

    summary["H1_kappa_pp"] = med_ci(head["kappa"] * 100)
    summary["H2_best_router_gain_pp"] = med_ci(head[f"router_{best_router}_gain"] * 100)
    summary["H2_best_router_rho"] = med_ci(head[f"router_{best_router}_rho"])
    summary["per_router_median"] = {
        nm: {"gain_pp": med_ci(head[f"router_{nm}_gain"] * 100),
             "rho": med_ci(head[f"router_{nm}_rho"]),
             "n_pairs_positive_gain": int((head[f"router_{nm}_gain"] > 0).sum())}
        for nm in router_names}
    summary["H1_verdict"] = ("SUPPORTED" if summary["H1_kappa_pp"]["median"] >= 5.0
                             else "NOT SUPPORTED")
    summary["H2_verdict"] = ("SUPPORTED" if summary["H2_best_router_rho"]["median"] <= 0.30
                             else "NOT SUPPORTED")

    # per-pair significance of "router beats matched static", pooled over families,
    # with Holm-Bonferroni and BH across the 55 pairs (prereg 2.6)
    pairs = sorted({(a, b) for a, b in zip(head.model_a, head.model_b)})
    pvals, effects = [], []
    for (a, b) in pairs:
        sub = head[(head.model_a == a) & (head.model_b == b)]
        d = (sub[f"router_{best_router}_gain"]).to_numpy(float)
        d = d[np.isfinite(d)]
        if len(d) < 2:
            pvals.append(1.0)
            effects.append(float(np.mean(d)) if len(d) else float("nan"))
            continue
        bs = np.array([np.mean(rng.choice(d, len(d), replace=True)) for _ in range(N_BOOT)])
        p = 2.0 * min((bs <= 0).mean(), (bs >= 0).mean())
        pvals.append(float(min(1.0, max(p, 1.0 / N_BOOT))))
        effects.append(float(np.mean(d)))
    rej_h, adj_h = holm_bonferroni(pvals)
    rej_b, adj_b = benjamini_hochberg(pvals)
    summary["pairwise_tests"] = {
        "n_pairs": len(pairs), "method": "bootstrap over benchmark families, 2-sided",
        "n_raw_p_below_.05": int(sum(p < 0.05 for p in pvals)),
        "n_significant_holm": int(rej_h.sum()),
        "n_significant_bh": int(rej_b.sum()),
        "n_positive_effect": int(sum(e > 0 for e in effects)),
        "detail": [{"model_a": a, "model_b": b, "mean_gain_pp": 100 * e,
                    "p_raw": p, "p_holm": float(ah), "p_bh": float(ab)}
                   for (a, b), e, p, ah, ab in zip(pairs, effects, pvals, adj_h, adj_b)],
    }

    with open(os.path.join(HERE, f"s11_routerbench_{shot}.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print("\n=== HEADLINE ===")
    print(f"median kappa at beta=0.5      : {summary['H1_kappa_pp']['median']:.2f} pp "
          f"[{summary['H1_kappa_pp']['lo']:.2f}, {summary['H1_kappa_pp']['hi']:.2f}]  "
          f"-> H1 {summary['H1_verdict']}")
    print(f"best router ({best_router}) gain : {summary['H2_best_router_gain_pp']['median']:.2f} pp")
    print(f"median rho                    : {summary['H2_best_router_rho']['median']:.3f} "
          f"[{summary['H2_best_router_rho']['lo']:.3f}, {summary['H2_best_router_rho']['hi']:.3f}]  "
          f"-> H2 {summary['H2_verdict']}")
    print(f"pairs significant after Holm  : {summary['pairwise_tests']['n_significant_holm']}"
          f"/{len(pairs)}  (BH: {summary['pairwise_tests']['n_significant_bh']})")


if __name__ == "__main__":
    main()

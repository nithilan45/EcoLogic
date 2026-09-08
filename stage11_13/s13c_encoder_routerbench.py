"""Stage 13c (exploratory, NOT pre-registered -- see DEVIATIONS.md D7).

Stage 13b found that *unfreezing the encoder* is the first intervention in this
whole project that moves held-out router AUC by more than noise. That result is
on 2,975 training items of our own data, which is exactly the sample size a
reviewer would distrust. So it is repeated here on RouterBench, with 29,182
training items and 11 models instead of 3 tiers.

Protocol, chosen to make the comparison mechanical rather than rhetorical:

  * the **same** fixed 80/20 item split, seed and stratification as
    `s12b_learning_curve.py`, so the end-to-end encoder can be read straight off
    against the frozen-representation learning curve at its largest size;
  * the **same** held-out items are used to re-score the four Stage 11 frozen
    routers from their stored out-of-fold predictions, so every row of the
    output table is computed on identical items with identical code;
  * the same matched-cost policy and the same `beta = 0.5` operating point.

Reported: mean held-out AUC over the 11 models, realised matched-cost gain in
accuracy points, and realised `rho` -- the share of the held-out complementarity
`kappa` that the router actually captures.

Cost: $0, CPU only.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from decomp import oracle_frontier_at, router_value_at, static_frontier_at  # noqa: E402
from s11_routerbench import flatten_prompt, load  # noqa: E402
from s13b_encoder_finetune import Router, collate, predict  # noqa: E402

SEED = 20260907
MAX_LEN = 256
EPOCHS = 3
BATCH = 32
LR = 3e-5
HEAD_LR = 1e-3
INNER_VAL = 0.05
BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"
HEADLINE_BETA = 0.5


def main():
    shot = sys.argv[1] if len(sys.argv) > 1 else "0shot"
    df, models, _, _ = load(shot)
    util = df[models].to_numpy(float)
    cost = df[[f"{m}|total_cost" for m in models]].to_numpy(float)
    ybin = (util >= 0.5).astype(int)
    k = len(models)

    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from transformers import AutoTokenizer

    idx = np.arange(len(df))
    tr_all, te = train_test_split(idx, test_size=0.2, random_state=SEED,
                                  stratify=df.family.to_numpy())
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(tr_all)
    cut = int(len(perm) * (1 - INNER_VAL))
    tr, va = perm[:cut], perm[cut:]
    print(f"train {len(tr)} / inner-val {len(va)} / held out {len(te)}", flush=True)

    U_te, C_te = util[te], cost[te]
    mc = C_te.mean(axis=0)
    budget = float(mc.min() + HEADLINE_BETA * (mc.max() - mc.min()))
    S = static_frontier_at(mc, U_te.mean(axis=0), budget)
    A_star, _, _ = oracle_frontier_at(U_te, C_te, budget)
    kappa = A_star - S
    print(f"held-out kappa at beta=0.5: {100*kappa:.2f} pp", flush=True)

    def score_row(P):
        aucs = [roc_auc_score(ybin[te, m], P[:, m]) for m in range(k)
                if 0 < ybin[te, m].mean() < 1]
        v, vc = router_value_at(U_te, C_te, P, budget)
        return {"mean_auc": float(np.mean(aucs)),
                "gain_pp": float(100 * (v - S)),
                "rho": float((v - S) / kappa) if kappa > 1e-12 else float("nan"),
                "router_cost": float(vc)}

    # ---- the end-to-end fine-tuned encoder ---------------------------------
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    pad = tok.pad_token_id
    texts = [flatten_prompt(p) for p in df.prompt]
    print("tokenising ...", flush=True)
    ids = tok([texts[i] for i in idx], truncation=True, max_length=MAX_LEN)["input_ids"]
    Ytr = torch.tensor(util[tr], dtype=torch.float32)

    model = Router(BACKBONE, n_out=k)
    opt = torch.optim.AdamW(
        [{"params": model.enc.parameters(), "lr": LR},
         {"params": model.head.parameters(), "lr": HEAD_LR}], weight_decay=0.01)
    # Graded RouterBench scores are used directly as soft BCE targets, which is
    # the right estimator for E[u | x] on a [0,1] utility.
    lossf = nn.BCEWithLogitsLoss()
    n = len(tr)
    steps = EPOCHS * ((n + BATCH - 1) // BATCH) + 1
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR, HEAD_LR],
                                                total_steps=steps, pct_start=0.1)

    hist, best = [], {"val_auc": -1.0}
    for ep in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        p = rng.permutation(n)
        chunks = [p[i:i + BATCH * 16] for i in range(0, n, BATCH * 16)]
        order = np.concatenate([c[np.argsort([len(ids[tr[a]]) for a in c], kind="stable")]
                                for c in chunks])
        batches = [order[i:i + BATCH] for i in range(0, n, BATCH)]
        rng.shuffle(batches)
        tot = 0.0
        for bi, j in enumerate(batches):
            I, M = collate([ids[tr[a]] for a in j], pad)
            opt.zero_grad()
            loss = lossf(model(I, M), Ytr[j])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(j)
            if bi % 100 == 0:
                print(f"    epoch {ep} batch {bi}/{len(batches)} "
                      f"loss {tot/max(1,(bi+1)*BATCH):.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        Pva = predict(model, [ids[a] for a in va], pad)
        vaucs = [roc_auc_score(ybin[va, m], Pva[:, m]) for m in range(k)
                 if 0 < ybin[va, m].mean() < 1]
        vauc = float(np.mean(vaucs))
        hist.append({"epoch": ep, "train_loss": tot / n, "inner_val_auc": vauc,
                     "secs": time.time() - t0})
        print(f"  epoch {ep}/{EPOCHS} loss {tot/n:.4f} inner-val AUC {vauc:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if vauc > best["val_auc"]:
            Pte = predict(model, [ids[a] for a in te], pad)
            best = {"epoch": ep, "val_auc": vauc, "held_out": score_row(Pte)}
            np.save(os.path.join(HERE, f"s13c_pred_{shot}.npy"), Pte)
            print(f"    -> held out: {json.dumps(best['held_out'])}", flush=True)

    # ---- the frozen reference, refitted on the identical training items ----
    # Refitting here rather than reusing s12b's numbers removes the last
    # difference between the two rows: both see exactly the same 27,735 items.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    emb = np.load(os.path.join(HERE, f"s11_emb_{shot}.npy"))
    assert len(emb) == len(df), "embedding cache does not match the loaded frame"
    print("refitting frozen routers on the same training items ...", flush=True)
    Pl = np.zeros((len(te), k))
    for m in range(k):
        y = ybin[tr, m]
        if y.min() == y.max():
            Pl[:, m] = float(y[0])
            continue
        Pl[:, m] = LogisticRegression(max_iter=3000).fit(
            emb[tr], y).predict_proba(emb[te])[:, 1]
    sc = StandardScaler().fit(emb[tr])
    mlp = MLPRegressor(hidden_layer_sizes=(256, 64), early_stopping=True,
                       n_iter_no_change=10, validation_fraction=0.15,
                       max_iter=400, random_state=SEED)
    mlp.fit(sc.transform(emb[tr]), util[tr])
    Pm = np.clip(mlp.predict(sc.transform(emb[te])), 0, 1)
    frozen = {"frozen_minilm_logreg": score_row(Pl),
              "frozen_minilm_mlp": score_row(Pm)}

    out = {
        "what": "end-to-end fine-tuned encoder as a router, on RouterBench",
        "pre_registered": False, "deviation": "DEVIATIONS.md D7",
        "shot": shot, "seed": SEED, "backbone": BACKBONE, "max_len": MAX_LEN,
        "epochs": EPOCHS, "batch": BATCH, "lr_encoder": LR, "lr_head": HEAD_LR,
        "n_train": len(tr), "n_inner_val": len(va), "n_held_out": len(te),
        "n_models": k, "beta": HEADLINE_BETA,
        "held_out_S": S, "held_out_A_star": A_star, "held_out_kappa_pp": 100 * kappa,
        "encoder_finetuned": best, "history": hist,
        "frozen_routers_same_items": frozen,
        "usd": 0.0,
    }
    p = os.path.join(HERE, f"s13c_encoder_routerbench_{shot}.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\nwrote", p)
    print(json.dumps({"encoder": best.get("held_out"), "frozen": frozen}, indent=1))


if __name__ == "__main__":
    main()

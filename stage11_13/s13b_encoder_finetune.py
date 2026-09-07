"""Stage 13b — end-to-end fine-tuning of the text encoder itself.

Every router in Stages 5, 7, 7c and 11 puts a *fitted head* on a *frozen*
representation. That leaves one version of the "weak router" objection open:
maybe the MiniLM sentence embedding simply does not contain the information, and
a representation trained on the task would. This rung removes the objection by
unfreezing the encoder and training all of its weights on the routing labels.

Two backbones, both trained end-to-end with a 3-logit head (one per tier) and
per-tier binary cross-entropy:

  MiniLM-L6   sentence-transformers/all-MiniLM-L6-v2  (22M) -- the *same*
              backbone whose frozen output gives 0.6816 AUC, so the comparison
              isolates exactly the effect of unfreezing it
  MiniLM-L12  sentence-transformers/all-MiniLM-L12-v2 (33M) -- deeper, to
              separate "not enough depth" from "not enough signal"

Fitted on the Stage 7 TRAIN split (3,500 items) and scored once on the Stage 7
CALIBRATION split (1,500 items), which is the split every earlier rung reports,
so the comparison is apples to apples. The frozen Stage 7 test set is untouched.
Model selection (which epoch to report) uses a 15% split *carved out of TRAIN*,
never CALIBRATION.

Cost: $0. Runs on CPU. This rung is not pre-registered; see DEVIATIONS.md D7.
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
from s13_llm_router import load_split  # noqa: E402

SEED = 20260907
MAX_LEN = 256
EPOCHS = 6
BATCH = 16
LR = 2e-5
HEAD_LR = 1e-3
VAL_FRACTION = 0.15
BACKBONES = ["sentence-transformers/all-MiniLM-L6-v2",
             "sentence-transformers/all-MiniLM-L12-v2"]

# Stage 7c's best frozen-representation router on the same split, for reference.
FROZEN_BEST_AUC = 0.6816


class Router(nn.Module):
    """Encoder + mean-pooling + one logit per tier."""

    def __init__(self, name):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(name)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(self.enc.config.hidden_size, 3)

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.head(self.drop(pooled))


def encode(tok, items):
    b = tok([it["raw_query"] for it in items], truncation=True, max_length=MAX_LEN,
            padding="max_length", return_tensors="pt")
    y = torch.tensor([[float(it["y"][t]) for t in (1, 2, 3)] for it in items])
    return b["input_ids"], b["attention_mask"], y


def mean_auc(y, p):
    from sklearn.metrics import roc_auc_score
    aucs = [roc_auc_score(y[:, t], p[:, t]) for t in range(3)
            if 0 < y[:, t].mean() < 1]
    return float(np.mean(aucs)), [float(roc_auc_score(y[:, t], p[:, t]))
                                  for t in range(3)]


@torch.no_grad()
def predict(model, ids, mask, bs=64):
    model.eval()
    out = []
    for i in range(0, len(ids), bs):
        out.append(torch.sigmoid(model(ids[i:i + bs], mask[i:i + bs])).numpy())
    return np.concatenate(out)


def run(name, train, val, cal, tok_cache):
    from transformers import AutoTokenizer
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tok = AutoTokenizer.from_pretrained(name)
    key = name
    if key not in tok_cache:
        tok_cache[key] = (encode(tok, train), encode(tok, val), encode(tok, cal))
    (Itr, Mtr, Ytr), (Iv, Mv, Yv), (Ic, Mc, Yc) = tok_cache[key]

    model = Router(name)
    opt = torch.optim.AdamW(
        [{"params": model.enc.parameters(), "lr": LR},
         {"params": model.head.parameters(), "lr": HEAD_LR}], weight_decay=0.01)
    lossf = nn.BCEWithLogitsLoss()
    n = len(Itr)
    steps = EPOCHS * ((n + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR, HEAD_LR],
                                                total_steps=steps, pct_start=0.1)

    hist, best = [], {"val_auc": -1.0}
    for ep in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        order = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            j = order[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Itr[j], Mtr[j]), Ytr[j])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(j)
        va, _ = mean_auc(Yv.numpy(), predict(model, Iv, Mv))
        hist.append({"epoch": ep, "train_loss": tot / n, "val_auc": va,
                     "secs": time.time() - t0})
        print(f"  {name.split('/')[-1]} epoch {ep}/{EPOCHS} "
              f"loss {tot/n:.4f} inner-val AUC {va:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        # Epoch selection on the inner split carved out of TRAIN only.
        if va > best["val_auc"]:
            pc = predict(model, Ic, Mc)
            ca, per = mean_auc(Yc.numpy(), pc)
            best = {"epoch": ep, "val_auc": va, "cal_auc": ca, "cal_per_tier": per,
                    "cal_pred": pc}
    print(f"  -> selected epoch {best['epoch']}: CALIBRATION mean AUC "
          f"{best['cal_auc']:.4f} (per tier {['%.4f' % x for x in best['cal_per_tier']]})",
          flush=True)
    return best, hist


def main():
    tr = load_split("s7_train_pool.json")
    cal = load_split("s7_calibration_pool.json")
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(tr))
    cut = int(len(tr) * (1 - VAL_FRACTION))
    train = [tr[i] for i in idx[:cut]]
    val = [tr[i] for i in idx[cut:]]
    print(f"TRAIN {len(train)} / inner-val {len(val)} / CALIBRATION {len(cal)}",
          flush=True)

    results, cache = {}, {}
    for name in BACKBONES:
        best, hist = run(name, train, val, cal, cache)
        pred = best.pop("cal_pred")
        results[name] = {"selection": best, "history": hist}
        np.save(os.path.join(HERE, f"s13b_pred_{name.split('/')[-1]}.npy"), pred)

    best_name = max(results, key=lambda k: results[k]["selection"]["cal_auc"])
    best_auc = results[best_name]["selection"]["cal_auc"]
    out = {
        "what": "end-to-end fine-tuned text encoders as routers (Stage 13b)",
        "pre_registered": False,
        "deviation": "DEVIATIONS.md D7",
        "seed": SEED, "max_len": MAX_LEN, "epochs": EPOCHS, "batch": BATCH,
        "lr_encoder": LR, "lr_head": HEAD_LR, "val_fraction": VAL_FRACTION,
        "n_train": len(train), "n_inner_val": len(val), "n_calibration": len(cal),
        "results": results,
        "best_backbone": best_name,
        "best_cal_mean_auc": best_auc,
        "frozen_representation_best_auc": FROZEN_BEST_AUC,
        "delta_vs_frozen": best_auc - FROZEN_BEST_AUC,
        "C1_threshold_delta": 0.05,
        "C1_met": bool(best_auc - FROZEN_BEST_AUC >= 0.05),
        "usd": 0.0,
    }
    p = os.path.join(HERE, "s13b_encoder_finetune.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nbest {best_name}: {best_auc:.4f} vs frozen {FROZEN_BEST_AUC:.4f} "
          f"(delta {best_auc-FROZEN_BEST_AUC:+.4f}); C1_met={out['C1_met']}")
    print("wrote", p)


if __name__ == "__main__":
    main()

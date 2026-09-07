"""Stage 13 — the router-strength ladder: prompted LLM router, then a fine-tuned LLM router.

Given Stage 12's refutation of H3 (see DEVIATIONS.md D1), the binding term in the
decomposition is `eps` -- the generalisation gap from prompt text to item
difficulty -- not `rho`. So the question "would a much stronger router close it?"
is now the central empirical question rather than a robustness check.

Rungs, in the order pre-registered in `prereg_stage11_13.md` section 4:
  R-b  prompted LLM-as-router, zero-shot and few-shot, scored by the logprob of
       the "yes" token so the output is a continuous probability, not a label
  R-c  LoRA fine-tune of an open-weights model on the Stage 7 TRAIN labels

Both are fitted on TRAIN (3,500 items) where applicable and scored ONCE on the
Stage 7 CALIBRATION split (1,500 items) -- the same split Stage 7c used, so the
comparison against logistic regression / GBM / RF / k-NN is apples to apples. The
frozen Stage 7 test set is not touched.

HARD COST GATE: $25 of additional spend (prereg section 4.1). Every call's cost
is accumulated from the provider's own usage field and the run aborts at the gate.

Subcommands:  probe | prompted | build_ft | launch_ft | poll_ft | score_ft | report
"""

from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
S710 = os.path.join(os.path.dirname(HERE), "stage7_10")
STATE = os.path.join(HERE, "s13_state.json")
SPEND = os.path.join(HERE, "s13_spend.json")

COST_GATE_USD = 25.00
SEED = 20260907
API = "https://api.together.xyz"

PROMPTED_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
PROMPTED_PRICE_IN = 1.04 / 1e6      # USD per token, from /v1/models pricing
PROMPTED_PRICE_OUT = 1.04 / 1e6

# Ordered by preference. Each must (a) accept LoRA fine-tuning and (b) have a
# matching `*-lora` serverless inference target in /v1/models, otherwise scoring
# the fine-tune would need a dedicated hourly endpoint.
FT_BASE_CANDIDATES = ["google/gemma-3-27b-it", "Qwen/Qwen3.5-2B",
                      "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"]
FT_BATCH_SIZE = 8

TIER_DESC = {
    1: "Model A: a small 9B open-weights reasoning model (Qwen3.5-9B)",
    2: "Model B: a mid-size 20B open-weights model (gpt-oss-20b)",
    3: "Model C: a large frontier model (GPT-4o)",
}
MAX_QUERY_CHARS = 1200

_lock = threading.Lock()
_spend = {"usd": 0.0, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "by_stage": {}}


def key():
    k = os.environ.get("TOGETHER_API_KEY")
    if not k and os.path.exists("/tmp/together_key.env"):
        k = open("/tmp/together_key.env").read().split("=", 1)[1].strip()
    if not k:
        raise SystemExit("TOGETHER_API_KEY not set")
    return k


def save_spend():
    with open(SPEND, "w") as f:
        json.dump(_spend, f, indent=1)


def charge(stage, pt, ct, price_in, price_out):
    with _lock:
        usd = pt * price_in + ct * price_out
        _spend["usd"] += usd
        _spend["calls"] += 1
        _spend["prompt_tokens"] += pt
        _spend["completion_tokens"] += ct
        b = _spend["by_stage"].setdefault(stage, {"usd": 0.0, "calls": 0})
        b["usd"] += usd
        b["calls"] += 1
        if _spend["usd"] > COST_GATE_USD:
            raise SystemExit(f"COST GATE HIT: ${_spend['usd']:.2f} > ${COST_GATE_USD}")
        return usd


# ---------------------------------------------------------------------------
def load_split(name):
    d = json.load(open(os.path.join(S710, name)))
    labels = json.load(open(os.path.join(S710, "s7_pool_labels.json")))["labels"]
    out = []
    for it in d["items"]:
        lab = labels.get(it["item_id"])
        if lab is None:
            continue
        out.append({
            "item_id": it["item_id"],
            "benchmark": it["benchmark"],
            "raw_query": it["raw_query"],
            "y": {int(t): bool(v) for t, v in lab["majority_correct_by_tier"].items()},
            "no_tier_correct": lab["no_tier_correct"],
        })
    return out


def router_prompt(query, tier, shots=()):
    head = ("You are predicting whether a particular language model will answer a "
            "question correctly. Answer with a single word: yes or no.\n")
    body = ""
    for s in shots:
        body += (f"\n{TIER_DESC[s['tier']]}\nQuestion: {s['q'][:400]}\n"
                 f"Will this model answer correctly? Answer: {'yes' if s['y'] else 'no'}\n")
    body += (f"\n{TIER_DESC[tier]}\nQuestion: {query[:MAX_QUERY_CHARS]}\n"
             f"Will this model answer correctly? Answer:")
    return head + body


YES_TOKENS = {" yes", " Yes", " YES", "yes", "Yes", "YES"}
NO_TOKENS = {" no", " No", " NO", "no", "No", "NO"}


def score_one(sess, model, prompt, stage, price_in, price_out, retries=5):
    """-> P(yes) from the top-5 logprobs of the single generated token."""
    import math
    for attempt in range(retries):
        try:
            r = sess.post(f"{API}/v1/completions", timeout=120, json={
                "model": model, "prompt": prompt, "max_tokens": 1,
                "temperature": 0, "logprobs": 5})
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(30, 2 ** attempt) + random.random())
                continue
            r.raise_for_status()
            j = r.json()
            u = j.get("usage") or {}
            charge(stage, u.get("prompt_tokens", 0), u.get("completion_tokens", 0),
                   price_in, price_out)
            top = (j["choices"][0].get("logprobs") or {}).get("top_logprobs") or [{}]
            top = top[0] or {}
            py = sum(math.exp(v) for k2, v in top.items() if k2 in YES_TOKENS)
            pn = sum(math.exp(v) for k2, v in top.items() if k2 in NO_TOKENS)
            if py + pn <= 0:
                txt = (j["choices"][0].get("text") or "").strip().lower()
                return (1.0 if txt.startswith("y") else 0.0), True
            return py / (py + pn), False
        except SystemExit:
            raise
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(min(30, 2 ** attempt) + random.random())
    raise RuntimeError("unreachable")


def run_scoring(model, items, shots, stage, price_in, price_out, out_path, workers=8):
    """Score every (item, tier); resumable and keyed on (item_id, tier)."""
    done = {}
    if os.path.exists(out_path):
        for line in open(out_path):
            r = json.loads(line)
            done[(r["item_id"], r["tier"])] = r
    todo = [(it, t) for it in items for t in (1, 2, 3)
            if (it["item_id"], t) not in done]
    print(f"  {stage}: {len(done)} cached, {len(todo)} to score", flush=True)
    if not todo:
        return done

    sess = requests.Session()
    sess.headers.update({"Authorization": f"Bearer {key()}",
                         "Content-Type": "application/json"})
    fh = open(out_path, "a")
    n_done = [0]

    def work(pair):
        it, t = pair
        p = router_prompt(it["raw_query"], t, shots)
        py, fb = score_one(sess, model, p, stage, price_in, price_out)
        rec = {"item_id": it["item_id"], "tier": t, "p_yes": py,
               "fallback": fb, "y": it["y"][t], "benchmark": it["benchmark"]}
        with _lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_done[0] += 1
            if n_done[0] % 200 == 0:
                print(f"    {n_done[0]}/{len(todo)}  spend ${_spend['usd']:.2f}", flush=True)
                save_spend()
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for rec in ex.map(work, todo):
            done[(rec["item_id"], rec["tier"])] = rec
    fh.close()
    save_spend()
    return done


# ---------------------------------------------------------------------------
def cmd_probe():
    """Estimate the cost of each rung before spending anything."""
    cal = load_split("s7_calibration_pool.json")
    tr = load_split("s7_train_pool.json")
    print(f"TRAIN {len(tr)} items, CALIBRATION {len(cal)} items")
    def toks(items, shots):
        return sum(len(router_prompt(it["raw_query"], t, shots)) / 4.0
                   for it in items for t in (1, 2, 3))
    z = toks(cal, ())
    shots = build_shots(tr, 4)
    f = toks(cal, shots)
    print(f"zero-shot: ~{z/1e6:.2f}M prompt tokens -> ${z*PROMPTED_PRICE_IN:.2f}")
    print(f"4-shot   : ~{f/1e6:.2f}M prompt tokens -> ${f*PROMPTED_PRICE_IN:.2f}")
    ft = sum(len(router_prompt(it["raw_query"], t, ())) / 4.0 + 2
             for it in tr for t in (1, 2, 3))
    print(f"fine-tune: ~{ft/1e6:.2f}M tokens/epoch, 3 epochs -> {3*ft/1e6:.2f}M trained tokens")
    print(f"fine-tuned inference on CALIBRATION: ~{z/1e6:.2f}M tokens")
    print(f"cost gate: ${COST_GATE_USD}")


def build_shots(train_items, n, seed=SEED):
    """Few-shot exemplars: balanced yes/no per tier, drawn from TRAIN only."""
    rng = random.Random(seed)
    pool = [{"q": it["raw_query"], "tier": t, "y": it["y"][t]}
            for it in train_items for t in (1, 2, 3)]
    pos = [p for p in pool if p["y"]]
    neg = [p for p in pool if not p["y"]]
    rng.shuffle(pos)
    rng.shuffle(neg)
    out = []
    for i in range(n):
        out.append(pos[i] if i % 2 == 0 else neg[i])
    rng.shuffle(out)
    return out


def cmd_prompted(shots_n=0):
    cal = load_split("s7_calibration_pool.json")
    tr = load_split("s7_train_pool.json")
    shots = build_shots(tr, shots_n) if shots_n else ()
    stage = f"prompted_{shots_n}shot"
    out = os.path.join(HERE, f"s13_{stage}.jsonl")
    run_scoring(PROMPTED_MODEL, cal, shots, stage,
                PROMPTED_PRICE_IN, PROMPTED_PRICE_OUT, out)
    print(f"done {stage}; spend so far ${_spend['usd']:.4f}")


# ---------------------------------------------------------------------------
def cmd_build_ft():
    tr = load_split("s7_train_pool.json")
    path = os.path.join(HERE, "s13_ft_train.jsonl")
    n = 0
    with open(path, "w") as f:
        for it in tr:
            for t in (1, 2, 3):
                f.write(json.dumps({
                    "prompt": router_prompt(it["raw_query"], t, ()),
                    "completion": " yes" if it["y"][t] else " no"}) + "\n")
                n += 1
    print(f"wrote {n} examples to {path} ({os.path.getsize(path)/1e6:.1f} MB)")


def cmd_launch_ft():
    from together import Together
    c = Together(api_key=key())
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    path = os.path.join(HERE, "s13_ft_train.jsonl")

    if "file_id" not in st:
        up = c.files.upload(file=path, check=True)
        st["file_id"] = up.id
        json.dump(st, open(STATE, "w"), indent=1)
        print("uploaded", up.id)

    if "job_id" not in st:
        last = None
        for base in FT_BASE_CANDIDATES:
            try:
                job = c.fine_tuning.create(
                    training_file=st["file_id"], model=base, n_epochs=3,
                    lora=True, learning_rate=1e-4, batch_size=FT_BATCH_SIZE,
                    suffix="s13router", n_checkpoints=1)
                st.update({"job_id": job.id, "base_model": base})
                json.dump(st, open(STATE, "w"), indent=1)
                print("launched", job.id, "on", base)
                return
            except Exception as e:
                last = f"{base}: {e}"
                print("  rejected", last)
        raise SystemExit(f"no base model accepted the job; last error {last}")
    print("job already launched:", st["job_id"])


def cmd_poll_ft():
    from together import Together
    c = Together(api_key=key())
    st = json.load(open(STATE))
    j = c.fine_tuning.retrieve(st["job_id"])
    status = str(getattr(j, "status", "?"))
    print("status", status, "model", getattr(j, "output_name", None))
    st["status"] = status
    if getattr(j, "output_name", None):
        st["ft_model"] = j.output_name
    for f in ("total_price", "token_count", "trainingfile_numlines"):
        v = getattr(j, f, None)
        if v is not None:
            st[f] = v
    # Together reports fine-tune price in nano-dollars. Book it against the gate
    # once, so the training spend is not invisible in the ledger.
    price_usd = float(st.get("total_price", 0)) / 1e9
    st["total_price_usd"] = price_usd
    if price_usd > 0 and not st.get("price_booked"):
        with _lock:
            _spend["usd"] += price_usd
            _spend["by_stage"]["finetune_training"] = {"usd": price_usd, "calls": 0}
        st["price_booked"] = True
        save_spend()
        print(f"booked fine-tune training cost ${price_usd:.2f}; "
              f"total spend ${_spend['usd']:.2f} of ${COST_GATE_USD}")
        if _spend["usd"] > COST_GATE_USD:
            raise SystemExit(f"COST GATE HIT: ${_spend['usd']:.2f}")
    json.dump(st, open(STATE, "w"), indent=1)
    return status


def cmd_score_ft():
    st = json.load(open(STATE))
    model = st.get("ft_model")
    if not model:
        raise SystemExit("fine-tune not finished; no output model name in state")
    # LoRA price: charged at the base model's serverless rate.
    cal = load_split("s7_calibration_pool.json")
    out = os.path.join(HERE, "s13_finetuned.jsonl")
    price = st.get("infer_price_per_token", 0.36 / 1e6)
    run_scoring(model, cal, (), "finetuned", price, price, out, workers=4)
    print(f"done finetuned; spend so far ${_spend['usd']:.4f}")


# ---------------------------------------------------------------------------
def cmd_report():
    """Evaluate every rung against the pre-registered criteria C1/C2."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, HERE)
    from decomp import ceiling_from_sd, oracle_frontier_at, router_value_at, static_frontier_at

    cal = load_split("s7_calibration_pool.json")
    by_id = {it["item_id"]: it for it in cal}

    # per-item cost and utility for the three tiers, from Stage 7's own samples
    import pandas as pd
    d = pd.read_csv(os.path.join(S710, "s7_pool_samples.csv.gz"))
    d = d[d.item_id.isin(by_id)]
    d["correct"] = d["correct"].fillna(False).astype(bool).astype(float)
    piv_u = d.pivot_table(index="item_id", columns="tier", values="correct", aggfunc="mean")
    piv_c = d.pivot_table(index="item_id", columns="tier", values="usd", aggfunc="mean")
    ids = [i for i in piv_u.index if i in by_id]
    U = piv_u.loc[ids, [1, 2, 3]].to_numpy(float)
    C = piv_c.loc[ids, [1, 2, 3]].to_numpy(float)
    ypos = np.column_stack([[float(by_id[i]["y"][t]) for i in ids] for t in (1, 2, 3)])

    rungs = {}
    for stage, fn in [("prompted_0shot", "s13_prompted_0shot.jsonl"),
                      ("prompted_4shot", "s13_prompted_4shot.jsonl"),
                      ("finetuned", "s13_finetuned.jsonl")]:
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            rungs[stage] = {"status": "NOT RUN"}
            continue
        sc = {}
        for line in open(p):
            r = json.loads(line)
            sc[(r["item_id"], r["tier"])] = r["p_yes"]
        P = np.full((len(ids), 3), np.nan)
        for a, i in enumerate(ids):
            for b, t in enumerate((1, 2, 3)):
                if (i, t) in sc:
                    P[a, b] = sc[(i, t)]
        cov = float(np.mean(~np.isnan(P)))
        if cov < 0.999:
            rungs[stage] = {"status": f"PARTIAL coverage={cov:.3f}"}
            continue
        aucs = [float(roc_auc_score(ypos[:, b], P[:, b])) for b in range(3)]
        rec = {"status": "complete", "n_items": len(ids),
               "auc_by_tier": {f"tier{t}": aucs[b] for b, t in enumerate((1, 2, 3))},
               "mean_auc": float(np.mean(aucs)),
               "mean_auc_tiers12": float(np.mean(aucs[:2])), "betas": {}}
        for beta in (0.3, 0.5, 0.7):
            mc = C.mean(axis=0)
            budget = float(mc.min() + beta * (mc.max() - mc.min()))
            S = static_frontier_at(mc, U.mean(axis=0), budget)
            A, _, _ = oracle_frontier_at(U, C, budget)
            v, vc = router_value_at(U, C, P, budget)
            rec["betas"][f"{beta:.1f}"] = {
                "S": S, "A_star": A, "kappa": A - S, "router_acc": v,
                "router_cost": vc, "gain_pp": 100 * (v - S),
                "rho": (v - S) / (A - S) if A - S > 1e-12 else None}
        rungs[stage] = rec

    # classical rungs on the same split, for the C1/C2 comparison
    baseline = {"s7c_best_mean_auc": 0.6816, "s7c_best_model": "logistic regression"}
    res = {"cost_gate_usd": COST_GATE_USD,
           "spend": json.load(open(SPEND)) if os.path.exists(SPEND) else _spend,
           "state": json.load(open(STATE)) if os.path.exists(STATE) else {},
           "baseline_stage7c": baseline, "rungs": rungs}

    best = max((r for r in rungs.values() if r.get("status") == "complete"),
               key=lambda r: r["mean_auc"], default=None)
    if best:
        res["C1"] = {
            "rule": "mean held-out per-tier AUC exceeds Stage 7c best (0.6816) by >= 0.05",
            "best_llm_router_mean_auc": best["mean_auc"],
            "delta_vs_s7c": best["mean_auc"] - baseline["s7c_best_mean_auc"],
            "verdict": ("MET -> ceiling claim must be narrowed"
                        if best["mean_auc"] - baseline["s7c_best_mean_auc"] >= 0.05
                        else "NOT MET -> the plateau survives an LLM router")}
    with open(os.path.join(HERE, "s13_llm_router.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res.get("C1", {}), indent=1))
    for k2, v in rungs.items():
        print(f"{k2:16s} {v.get('status')}  mean_auc={v.get('mean_auc')}")


if __name__ == "__main__":
    if os.path.exists(SPEND):
        _spend.update(json.load(open(SPEND)))
    cmd = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if cmd == "probe":
        cmd_probe()
    elif cmd == "prompted":
        cmd_prompted(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    elif cmd == "build_ft":
        cmd_build_ft()
    elif cmd == "launch_ft":
        cmd_launch_ft()
    elif cmd == "poll_ft":
        cmd_poll_ft()
    elif cmd == "score_ft":
        cmd_score_ft()
    elif cmd == "report":
        cmd_report()
    else:
        raise SystemExit(f"unknown subcommand {cmd}")

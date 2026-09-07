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

# Second fine-tune, on a base that can actually be served. The gemma-3-27b run
# above trained fine but turned out to be unservable: serverless LoRA inference
# is unavailable account-wide, dedicated-endpoints v1 no longer accepts creates,
# and gemma-3-27b has no certified v2 config, so there is no route to its
# weights. `Qwen/Qwen3.5-9B` is both fine-tunable and has a certified v2 config
# on 1x H100, and it happens to be the Tier-1 model of the system under study,
# so the router costs no more to run than the cheapest tier it routes to.
# See DEVIATIONS.md D8.
FT2_BASE = "Qwen/Qwen3.5-9B"
FT2_CONFIG_ID = "cr_CeQCqcGQpVCeTctadrHjy"   # certified BF16 / 1x H100 profile
STATE2 = os.path.join(HERE, "s13_state_ft2.json")
V2_PROJECT_ID = "proj_CXT5sXEkTXThN2uNQ5ViU"
V2_CENTS_PER_MIN = 399 / 60.0                # 1x H100-80GB, $3.99/replica-hour
V2_MAX_MINUTES = 40                          # self-imposed cap: 40 min = $2.66

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


_deadline = [None]      # wall-clock stop for time-metered dedicated capacity


def score_one(sess, model, prompt, stage, price_in, price_out, retries=5):
    """-> P(yes) from the top-5 logprobs of the single generated token."""
    import math
    if _deadline[0] and time.time() > _deadline[0]:
        raise SystemExit("endpoint wall-clock deadline reached; stopping scoring")
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
    # Together exposes the servable name under different fields across SDK
    # versions; take the first that is populated.
    name = next((getattr(j, f, None) for f in
                 ("output_name", "api_model_object_name", "x_model_output_name",
                  "adapter_object_name")
                 if getattr(j, f, None)), None)
    print("status", status, "model", name)
    st["status"] = status
    if name:
        st["ft_model"] = name
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


# ---------------------------------------------------------------------------
# Scoring the fine-tune needs a DEDICATED endpoint: no `*-lora` serverless
# inference target is available on this account (all probed and recorded in
# s13_endpoint_probe.json), so the adapter cannot be called serverlessly.
# Dedicated capacity is billed by wall-clock, so this rung is metered by TIME
# rather than by tokens. `GET /v1/hardware?model=google/gemma-3-27b-it` offers
# 1x/2x/4x H100-80GB at 9/18/36 cents per minute; the smallest fits a 27B model
# in bf16 with a 400-token prompt and one generated token, so it is the one used.
ENDPOINT_HARDWARE = "1x_nvidia_h100_80gb_sxm"
ENDPOINT_CENTS_PER_MIN = 9
ENDPOINT_MAX_MINUTES = 45          # self-imposed cap: 45 min = $4.05


def _endpoint_charge(minutes, note=""):
    usd = minutes * ENDPOINT_CENTS_PER_MIN / 100.0
    with _lock:
        _spend["usd"] += usd
        b = _spend["by_stage"].setdefault("finetune_endpoint",
                                          {"usd": 0.0, "calls": 0, "minutes": 0.0})
        b["usd"] += usd
        b["minutes"] = b.get("minutes", 0.0) + minutes
        save_spend()
    print(f"  endpoint: +{minutes:.1f} min = ${usd:.2f} {note}; "
          f"total ${_spend['usd']:.2f} of ${COST_GATE_USD}", flush=True)
    return usd


def cmd_start_endpoint():
    from together import Together
    c = Together(api_key=key())
    st = json.load(open(STATE))
    if not st.get("ft_model"):
        raise SystemExit("fine-tune not finished; no output model name in state")
    projected = ENDPOINT_MAX_MINUTES * ENDPOINT_CENTS_PER_MIN / 100.0
    print(f"projection: up to {ENDPOINT_MAX_MINUTES} min at "
          f"{ENDPOINT_CENTS_PER_MIN}c/min = ${projected:.2f}; "
          f"already spent ${_spend['usd']:.2f} of ${COST_GATE_USD}")
    if _spend["usd"] + projected > COST_GATE_USD:
        raise SystemExit("projected endpoint cost would breach the gate; not starting")
    if st.get("endpoint_id"):
        print("endpoint already created:", st["endpoint_id"])
        return
    ep = c.endpoints.create(
        model=st["ft_model"], hardware=ENDPOINT_HARDWARE,
        autoscaling={"min_replicas": 1, "max_replicas": 1},
        inactive_timeout=5, state="STARTED",
        display_name="s13-router-eval")
    st["endpoint_id"] = ep.id
    st["endpoint_name"] = getattr(ep, "name", None)
    st["endpoint_started_at"] = time.time()
    json.dump(st, open(STATE, "w"), indent=1)
    print("created endpoint", ep.id, "name", st["endpoint_name"])


def cmd_stop_endpoint():
    from together import Together
    c = Together(api_key=key())
    st = json.load(open(STATE))
    eid = st.get("endpoint_id")
    if not eid:
        print("no endpoint to stop")
        return
    if st.get("endpoint_started_at") and not st.get("endpoint_billed"):
        mins = (time.time() - st["endpoint_started_at"]) / 60.0
        _endpoint_charge(mins, "(wall clock from create to stop)")
        st["endpoint_billed"] = True
        st["endpoint_minutes"] = mins
    try:
        c.endpoints.delete(eid)
        print("deleted endpoint", eid)
        st["endpoint_deleted"] = True
    except Exception as e:
        print("WARNING: could not delete endpoint:", e)
        print("Delete it manually at https://api.together.ai/endpoints")
    json.dump(st, open(STATE, "w"), indent=1)


def cmd_score_ft():
    """Start the endpoint, wait for it, score, and stop it -- always stop."""
    from together import Together
    c = Together(api_key=key())
    st = json.load(open(STATE))
    model = st.get("ft_model")
    if not model:
        raise SystemExit("fine-tune not finished; no output model name in state")
    if not st.get("endpoint_id"):
        cmd_start_endpoint()
        st = json.load(open(STATE))

    try:
        t0 = time.time()
        while True:
            ep = c.endpoints.retrieve(st["endpoint_id"])
            state = str(getattr(ep, "state", "?"))
            waited = (time.time() - t0) / 60.0
            print(f"  endpoint state={state} after {waited:.1f} min", flush=True)
            if state.upper() == "STARTED":
                break
            if waited > ENDPOINT_MAX_MINUTES:
                raise SystemExit(f"endpoint did not start within "
                                 f"{ENDPOINT_MAX_MINUTES} min; aborting")
            time.sleep(30)

        cal = load_split("s7_calibration_pool.json")
        out = os.path.join(HERE, "s13_finetuned.jsonl")
        # Dedicated capacity is billed by time, not tokens, so per-token price
        # is zero here and the endpoint minutes are charged separately. The
        # deadline is absolute from endpoint creation, so a slow endpoint cannot
        # walk past the cap; scoring is resumable if it does.
        _deadline[0] = st["endpoint_started_at"] + ENDPOINT_MAX_MINUTES * 60
        run_scoring(model, cal, (), "finetuned", 0.0, 0.0, out, workers=16)
    finally:
        _deadline[0] = None
        cmd_stop_endpoint()
    print(f"done finetuned; spend so far ${_spend['usd']:.4f}")


# ---------------------------------------------------------------------------
# R-c, second attempt: fine-tune a base that has a route to inference, and serve
# it through the Dedicated Endpoints v2 resource model (endpoint -> deployment ->
# traffic split). Billing is per replica-minute, so the run is metered by wall
# clock and the deployment is torn down in a `finally`.

def _v2():
    from together import Together
    return Together(api_key=key(), project_id=V2_PROJECT_ID).beta


def _v2_charge(minutes, note=""):
    usd = minutes * V2_CENTS_PER_MIN / 100.0
    with _lock:
        _spend["usd"] += usd
        b = _spend["by_stage"].setdefault("finetune2_deployment",
                                          {"usd": 0.0, "calls": 0, "minutes": 0.0})
        b["usd"] += usd
        b["minutes"] = b.get("minutes", 0.0) + minutes
        save_spend()
    print(f"  deployment: +{minutes:.1f} min = ${usd:.2f} {note}; "
          f"total ${_spend['usd']:.2f} of ${COST_GATE_USD}", flush=True)
    return usd


def _st2():
    return json.load(open(STATE2)) if os.path.exists(STATE2) else {}


def _save2(st):
    json.dump(st, open(STATE2, "w"), indent=1)


def cmd_launch_ft2():
    from together import Together
    c = Together(api_key=key())
    st = _st2()
    if "file_id" not in st:
        st["file_id"] = json.load(open(STATE))["file_id"]   # same training file
        _save2(st)
    if "job_id" in st:
        print("job already launched:", st["job_id"])
        return
    job = c.fine_tuning.create(
        training_file=st["file_id"], model=FT2_BASE, n_epochs=3, lora=True,
        learning_rate=1e-4, batch_size=FT_BATCH_SIZE, suffix="s13router9b",
        n_checkpoints=1)
    st.update({"job_id": job.id, "base_model": FT2_BASE})
    _save2(st)
    print("launched", job.id, "on", FT2_BASE)


def cmd_poll_ft2():
    from together import Together
    c = Together(api_key=key())
    st = _st2()
    j = c.fine_tuning.retrieve(st["job_id"])
    st["status"] = str(getattr(j, "status", "?"))
    name = next((getattr(j, f, None) for f in
                 ("output_name", "api_model_object_name", "x_model_output_name",
                  "adapter_object_name")
                 if getattr(j, f, None)), None)
    if name:
        st["ft_model"] = name
    for f in ("total_price", "token_count"):
        v = getattr(j, f, None)
        if v is not None:
            st[f] = v
    price_usd = float(st.get("total_price", 0)) / 1e9      # nano-dollars
    st["total_price_usd"] = price_usd
    if price_usd > 0 and not st.get("price_booked"):
        with _lock:
            _spend["usd"] += price_usd
            _spend["by_stage"]["finetune2_training"] = {"usd": price_usd, "calls": 0}
        st["price_booked"] = True
        save_spend()
        print(f"booked fine-tune-2 training ${price_usd:.2f}; "
              f"total ${_spend['usd']:.2f} of ${COST_GATE_USD}")
        if _spend["usd"] > COST_GATE_USD:
            raise SystemExit(f"COST GATE HIT: ${_spend['usd']:.2f}")
    _save2(st)
    print("status", st["status"], "model", st.get("ft_model"))
    return st["status"]


def _v2_model_ids(b, ft_name):
    """-> (merged_model_id, adapter_model_id) for a finished fine-tune."""
    merged = adapter = None
    for m in b.models.list():
        if m.name == ft_name and m.weights.type == "WEIGHTS_TYPE_DEFAULT":
            merged = m.id
        elif m.name == ft_name + "-adapter":
            adapter = m.id
    return merged, adapter


def cmd_deploy_ft2():
    b = _v2()
    st = _st2()
    if not st.get("ft_model"):
        raise SystemExit("fine-tune-2 not finished; no output model name in state")
    projected = V2_MAX_MINUTES * V2_CENTS_PER_MIN / 100.0
    print(f"projection: up to {V2_MAX_MINUTES} min at {V2_CENTS_PER_MIN:.2f}c/min "
          f"= ${projected:.2f}; already spent ${_spend['usd']:.2f} of ${COST_GATE_USD}")
    if _spend["usd"] + projected > COST_GATE_USD:
        raise SystemExit("projected deployment cost would breach the gate; not starting")

    merged, adapter = _v2_model_ids(b, st["ft_model"])
    st["v2_merged_model_id"], st["v2_adapter_model_id"] = merged, adapter
    if not merged:
        raise SystemExit(f"no v2 merged model found for {st['ft_model']}")
    if not st.get("endpoint_id"):
        ep = b.endpoints.create(name="s13-router-ft2")
        st["endpoint_id"] = ep.id
        st["endpoint_slug"] = getattr(ep, "name", None)
        _save2(st)
        print("endpoint", ep.id, st["endpoint_slug"])
    if not st.get("deployment_id"):
        dep = b.endpoints.deployments.create(
            st["endpoint_id"], name="ft2", model_id=merged,
            config_id=FT2_CONFIG_ID,
            autoscaling={"min_replicas": 1, "max_replicas": 1})
        st["deployment_id"] = dep.id
        st["deploy_started_at"] = time.time()
        _save2(st)
        print("deployment", dep.id)
        # A deployment receives no traffic until the endpoint routes to it.
        b.endpoints.update(st["endpoint_id"], update_mask="traffic_split",
                           traffic_split=[{"deployment_id": dep.id, "weight": 100}])
        print("traffic routed 100% to", dep.id)
    _save2(st)


def cmd_teardown_ft2():
    b = _v2()
    st = _st2()
    if st.get("deploy_started_at") and not st.get("deploy_billed"):
        mins = (time.time() - st["deploy_started_at"]) / 60.0
        _v2_charge(mins, "(wall clock from deployment create to teardown)")
        st["deploy_billed"] = True
        st["deploy_minutes"] = mins
    for step, fn in (
            ("scale to 0", lambda: b.endpoints.deployments.update(
                st["deployment_id"], endpoint_id=st["endpoint_id"],
                update_mask="autoscaling",
                autoscaling={"min_replicas": 0, "max_replicas": 0})),
            ("delete deployment", lambda: b.endpoints.deployments.delete(
                st["deployment_id"], endpoint_id=st["endpoint_id"])),
            ("delete endpoint", lambda: b.endpoints.delete(st["endpoint_id"]))):
        try:
            fn()
            print(step, "ok")
        except Exception as e:
            print(f"WARNING: {step} failed: {str(e)[:200]}")
            print("Check https://api.together.ai/endpoints -- GPUs bill until removed.")
    st["torn_down"] = True
    _save2(st)


def cmd_score_ft2():
    """Deploy, wait for READY, score the CALIBRATION split, always tear down."""
    b = _v2()
    if not _st2().get("deployment_id"):
        cmd_deploy_ft2()
    st = _st2()
    try:
        t0 = time.time()
        while True:
            d = b.endpoints.deployments.retrieve(
                st["deployment_id"], endpoint_id=st["endpoint_id"])
            state = str(getattr(d, "state", getattr(d, "status", "?")))
            waited = (time.time() - t0) / 60.0
            print(f"  deployment state={state} after {waited:.1f} min", flush=True)
            if "READY" in state.upper() or "RUNNING" in state.upper():
                break
            if waited > V2_MAX_MINUTES / 2:
                raise SystemExit(f"deployment not ready within {V2_MAX_MINUTES/2:.0f} "
                                 f"min; aborting")
            time.sleep(30)

        # v2 serves dedicated inference under <project_slug>/<endpoint_name> at
        # api-inference.together.ai; the adapter is baked into the merged model.
        served = f"{st.get('endpoint_slug')}"
        st["served_model"] = served
        _save2(st)
        global API
        API = "https://api-inference.together.ai"
        cal = load_split("s7_calibration_pool.json")
        out = os.path.join(HERE, "s13_finetuned.jsonl")
        _deadline[0] = st["deploy_started_at"] + V2_MAX_MINUTES * 60
        run_scoring(served, cal, (), "finetuned", 0.0, 0.0, out, workers=16)
    finally:
        _deadline[0] = None
        cmd_teardown_ft2()
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
    elif cmd == "launch_ft2":
        cmd_launch_ft2()
    elif cmd == "poll_ft2":
        cmd_poll_ft2()
    elif cmd == "deploy_ft2":
        cmd_deploy_ft2()
    elif cmd == "score_ft2":
        cmd_score_ft2()
    elif cmd == "teardown_ft2":
        cmd_teardown_ft2()
    elif cmd == "report":
        cmd_report()
    else:
        raise SystemExit(f"unknown subcommand {cmd}")

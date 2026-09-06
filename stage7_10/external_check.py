"""Stage 9 - external generalization check of the Stage 8 correction term.

Uses RouteLLM's own released artifacts (github.com/lm-sys/RouteLLM, Ong et al.,
ICLR 2025):
  - per-item correctness for both routed models on GSM8K and MMLU
  - the full response TEXT for both models, from which per-item token counts are
    reconstructed with the models' real tokenizers (cl100k_base for
    gpt-4-1106-preview, the Mixtral-8x7B-Instruct-v0.1 SentencePiece tokenizer)
  - RouteLLM's own released BERT router checkpoint (routellm/bert_gpt4_augmented),
    run locally, replicating routers.py::BERTRouter.calculate_strong_win_rate and
    controller.py's `win_rate >= threshold -> strong` rule
  - their decontamination list (contaminated_prompts.jsonl)

Nothing is simulated. Token counts are DERIVED from released response text; they
are not published by RouteLLM as numeric fields.

Writes stage7_10/external_generalization.json (the .md is written by hand from it).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RL = Path("/tmp/routellm_chk")
OUT = ROOT / "stage7_10"

WEAK = "mistralai/Mixtral-8x7B-Instruct-v0.1"
STRONG = "gpt-4-1106-preview"

# Published list prices at the time of Ong et al. (2025), $ per 1M tokens.
PRICES = {
    STRONG: {"in": 10.00, "out": 30.00},   # OpenAI gpt-4-1106-preview
    WEAK: {"in": 0.60, "out": 0.60},       # Together AI Mixtral-8x7B-Instruct
}


def cov(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.mean(a * b) - np.mean(a) * np.mean(b))


def audit_release() -> dict:
    """Which released files carry the columns needed for per-item cost?"""
    audit = {}
    gsm = pd.read_csv(RL / "routellm/evals/gsm8k/gsm8k_responses.csv")
    audit["gsm8k"] = {"columns": list(gsm.columns), "n_rows": int(len(gsm))}
    mm = pd.read_csv(RL / "routellm/evals/mmlu/responses/mmlu_anatomy.csv")
    audit["mmlu"] = {"columns": list(mm.columns),
                     "n_subject_files": len(list((RL / "routellm/evals/mmlu/responses")
                                                 .glob("mmlu_*.csv")))}
    with open(RL / "routellm/evals/mt_bench/judgements.jsonl") as f:
        audit["mt_bench"] = {"columns": list(json.loads(f.readline()).keys())}
    for k, v in audit.items():
        has_resp = any(c.endswith("_response") for c in v.get("columns", []))
        has_tok = any("token" in c.lower() or "cost" in c.lower()
                      for c in v.get("columns", []))
        v["has_response_text"] = has_resp
        v["has_token_or_cost_field"] = has_tok
        v["per_item_cost_reconstructable"] = has_resp
    return audit


def load_benchmark(name: str) -> pd.DataFrame:
    df = pd.read_csv(RL / "routellm/evals/gsm8k/gsm8k_responses.csv")
    contam = pd.read_json(RL / "routellm/evals/gsm8k/contaminated_prompts.jsonl",
                          lines=True)["eval_prompt"].tolist()
    n0 = len(df)
    df = df[~df["prompt"].isin(contam)].reset_index(drop=True)
    print(f"  {name}: {len(df)}/{n0} items after RouteLLM's own decontamination")
    return df


def add_costs(df: pd.DataFrame) -> pd.DataFrame:
    enc = tiktoken.encoding_for_model(STRONG)
    mx = AutoTokenizer.from_pretrained(WEAK)

    prompts = df["prompt"].astype(str).tolist()
    p_strong = np.array([len(x) for x in enc.encode_batch(prompts)], float)
    p_weak = np.array([len(mx.encode(x)) for x in prompts], float)

    def out_tokens(col, tokenizer_kind):
        texts = df[col].fillna("").astype(str).tolist()
        if tokenizer_kind == "tiktoken":
            return np.array([len(x) for x in enc.encode_batch(texts)], float)
        return np.array([len(mx.encode(x)) for x in texts], float)

    o_strong = out_tokens(f"{STRONG}_response", "tiktoken")
    o_weak = out_tokens(f"{WEAK}_response", "mixtral")

    df["cost_strong"] = (p_strong * PRICES[STRONG]["in"]
                         + o_strong * PRICES[STRONG]["out"]) / 1e6
    df["cost_weak"] = (p_weak * PRICES[WEAK]["in"]
                       + o_weak * PRICES[WEAK]["out"]) / 1e6
    df["tok_out_strong"] = o_strong
    df["tok_out_weak"] = o_weak
    return df


def bert_win_rates(prompts: list[str], cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)
    model = AutoModelForSequenceClassification.from_pretrained(
        "routellm/bert_gpt4_augmented", num_labels=3)
    tok = AutoTokenizer.from_pretrained("routellm/bert_gpt4_augmented")
    model.eval()
    out = []
    B = 32
    with torch.no_grad():
        for i in range(0, len(prompts), B):
            batch = prompts[i:i + B]
            inp = tok(batch, return_tensors="pt", padding=True, truncation=True)
            logits = model(**inp).logits.numpy()
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            sm = e / e.sum(axis=1, keepdims=True)
            out.extend(1.0 - sm[:, -2:].sum(axis=1))  # replicates BERTRouter
            if i % (B * 40) == 0:
                print(f"    router {i}/{len(prompts)}", flush=True)
    arr = np.array(out, float)
    np.save(cache, arr)
    return arr


def analyse(df: pd.DataFrame, wr: np.ndarray, label: str) -> dict:
    c_w = df["cost_weak"].to_numpy()
    c_s = df["cost_strong"].to_numpy()
    ok_w = df[WEAK].to_numpy().astype(bool)
    ok_s = df[STRONG].to_numpy().astype(bool)
    n = len(df)

    # oracle: cheapest model that answered correctly; if neither, the cheaper one
    # (weak is cheaper than strong for every item in this data, asserted below)
    assert (c_w < c_s).all(), "weak is not uniformly cheaper; oracle rule needs revisiting"
    t_star = np.where(ok_w, 0, np.where(ok_s, 1, 0))  # 0 = weak, 1 = strong

    mean_cost = {0: float(c_w.mean()), 1: float(c_s.mean())}
    per_item = {0: c_w, 1: c_s}

    rows = []
    # RouteLLM's own threshold grid: 10 equally-sized quantile bins
    _, thresholds = pd.qcut(wr, 10, retbins=True, duplicates="drop")
    for thr in thresholds:
        t_hat = (wr >= thr).astype(int)
        strong_pct = float(t_hat.mean() * 100)

        true_cost = float(np.mean(np.where(t_hat == 1, c_s, c_w)))
        naive_cost = float(t_hat.mean() * mean_cost[1] + (1 - t_hat.mean()) * mean_cost[0])

        r_true = float(np.mean(per_item[1] * (t_hat == 1) + per_item[0] * (t_hat == 0)
                               - (per_item[1] * (t_star == 1) + per_item[0] * (t_star == 0))))
        r_naive, corr = 0.0, 0.0
        for i_star in (0, 1):
            for j_hat in (0, 1):
                if i_star == j_hat:
                    continue
                mask = (t_star == i_star) & (t_hat == j_hat)
                pr = float(mask.mean())
                if pr == 0:
                    continue
                r_naive += pr * (mean_cost[j_hat] - mean_cost[i_star])
                D = per_item[j_hat] - per_item[i_star]
                corr += cov(mask.astype(float), D)
        rows.append({
            "threshold": float(thr), "strong_pct": strong_pct,
            "accuracy_pct": float(np.mean(np.where(t_hat == 1, ok_s, ok_w)) * 100),
            "true_cost_per_item_usd": true_cost,
            "naive_cost_per_item_usd": naive_cost,
            "cost_misestimate_pct": float((naive_cost - true_cost) / true_cost * 100)
            if true_cost else float("nan"),
            "regret_naive": r_naive, "regret_correction": corr,
            "regret_naive_plus_correction": r_naive + corr, "regret_true": r_true,
            "abs_residual": abs(r_naive + corr - r_true),
            "sign_flip": bool(r_naive * r_true < 0),
        })

    all_strong = float(c_s.mean())
    return {
        "benchmark": label, "n_items": n,
        "weak_model": WEAK, "strong_model": STRONG, "prices_usd_per_1m": PRICES,
        "weak_acc_pct": float(ok_w.mean() * 100), "strong_acc_pct": float(ok_s.mean() * 100),
        "mean_cost_per_item_usd": mean_cost,
        "within_model_cost_cv": {
            "weak": float(c_w.std() / c_w.mean()), "strong": float(c_s.std() / c_s.mean())},
        "output_token_stats": {
            "weak_mean": float(df["tok_out_weak"].mean()),
            "weak_p10_p90": [float(np.percentile(df["tok_out_weak"], 10)),
                             float(np.percentile(df["tok_out_weak"], 90))],
            "strong_mean": float(df["tok_out_strong"].mean()),
            "strong_p10_p90": [float(np.percentile(df["tok_out_strong"], 10)),
                               float(np.percentile(df["tok_out_strong"], 90))]},
        "all_strong_cost_per_item_usd": all_strong,
        "max_abs_residual": max(r["abs_residual"] for r in rows),
        "any_sign_flip": any(r["sign_flip"] for r in rows),
        "max_abs_cost_misestimate_pct": max(abs(r["cost_misestimate_pct"]) for r in rows),
        "sweep": rows,
    }


def main():
    if not RL.exists():
        raise SystemExit("RouteLLM repo not cloned at /tmp/routellm_chk")
    results = {"release_audit": audit_release()}
    print("release audit:")
    for k, v in results["release_audit"].items():
        print(f"  {k:9} response_text={v['has_response_text']} "
              f"token/cost_field={v['has_token_or_cost_field']} "
              f"-> per-item cost reconstructable: {v['per_item_cost_reconstructable']}")
    # Only GSM8K ships response text, so it is the only benchmark in the release
    # where per-item cost can be reconstructed at all.
    for name, label in (("gsm8k", "GSM8K"),):
        df = load_benchmark(name)
        df = add_costs(df)
        wr = bert_win_rates(df["prompt"].astype(str).tolist(),
                            Path(f"/tmp/rl_wr_{name}.npy"))
        results[name] = analyse(df, wr, label)
        r = results[name]
        print(f"  {label}: residual max {r['max_abs_residual']:.3e}, "
              f"sign flip anywhere: {r['any_sign_flip']}, "
              f"max naive cost misestimate {r['max_abs_cost_misestimate_pct']:.1f}%")
    with open(OUT / "external_generalization.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote stage7_10/external_generalization.json")


if __name__ == "__main__":
    main()

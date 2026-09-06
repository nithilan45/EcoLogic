"""Stage 10(a) - decompose accuracy variance into item-sampling and generation noise.

The Stage 7 frozen test set was generated k=3 times per item per tier at
**temperature 0**, so the replicates differ only through provider-side
nondeterminism (batching, kernel scheduling, MoE routing, fp reduction order).
That is a different quantity from Stage 7's temperature-0.7 training samples,
which deliberately sample the output distribution.

Model. For item i and replicate r let Y_ir be the graded outcome under a fixed
policy, and let p_i = E[Y_ir | i] be that item's success probability across
regenerations. Then for a single-replicate accuracy estimate,

    Var(Y) = Var(p_i)          +  E[p_i (1 - p_i)]
             \\_ between-item _/    \\_ within-item generation _/
           = p_bar (1 - p_bar)

Estimators from k replicates, with p_hat_i = mean_r Y_ir:

    within_hat  = mean_i [ k/(k-1) * p_hat_i (1 - p_hat_i) ]        (unbiased)
    between_hat = s^2(p_hat_i) - within_hat / k                     (unbiased)

Reported interval half-widths at 95%:

    sampling-only        1.96 * sqrt(between_hat / n)
      - uncertainty from having sampled these n items, with generation noise
        averaged away. This is the quantity a Wilson interval is usually *read*
        as, and it is the smaller of the two.
    sampling+generation  1.96 * sqrt((between_hat + within_hat) / n)
      - uncertainty of one single-generation evaluation run, which is what the
        headline table actually reports.
    k-replicate          1.96 * sqrt((between_hat + within_hat / k) / n)
      - what you get by averaging k generations, for reference.

Writes stage7_10/s7_generation_variance.json and s7_generation_variance.md.
"""

import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in ("benchmark", "router_v2", "stage7_10", "backend"):
    sys.path.insert(0, str(ROOT / p))
from analyze import wilson  # noqa: E402
from calibrate import route  # noqa: E402
from s7_data import PAPER_RATES, TIERS, load_split, test_rows  # noqa: E402
from train_router import predict  # noqa: E402

OUT = ROOT / "stage7_10"
V2 = ROOT / "router_v2"
K = 3
Z = 1.959963984540054


def decompose(Y: np.ndarray) -> dict:
    """Y is (n_items, k) of 0/1 outcomes."""
    n, k = Y.shape
    phat = Y.mean(axis=1)
    within = float(np.mean(k / (k - 1) * phat * (1 - phat)))
    between = float(np.var(phat, ddof=1) - within / k)
    between_clipped = max(between, 0.0)
    total = between_clipped + within
    p_bar = float(phat.mean())
    return {
        "n_items": n, "k": k, "mean_accuracy_over_k": p_bar,
        "within_item_generation_var": within,
        "between_item_var_raw": between,
        "between_item_var": between_clipped,
        "between_was_clipped_at_zero": between < 0,
        "total_single_draw_var": total,
        "bernoulli_check_p_bar_1_minus_p_bar": p_bar * (1 - p_bar),
        "within_share_of_total": (within / total) if total else 0.0,
        "half_width_sampling_only_pp": Z * np.sqrt(between_clipped / n) * 100,
        "half_width_sampling_plus_generation_pp": Z * np.sqrt(total / n) * 100,
        "half_width_k_replicate_pp": Z * np.sqrt((between_clipped + within / k) / n) * 100,
        "n_items_with_flip": int(np.sum((phat > 0) & (phat < 1))),
        "frac_items_with_flip": float(np.mean((phat > 0) & (phat < 1))),
    }


def main():
    items_meta = {it["item_id"]: it for it in load_split("test")}
    # outcomes[(tier, item)] = list of k outcomes, ordered by sample_idx
    by = defaultdict(dict)
    tok = defaultdict(dict)
    for r in test_rows():
        by[(r["tier"], r["item_id"])][r["sample_idx"]] = bool(r.get("correct"))
        tok[(r["tier"], r["item_id"])][r["sample_idx"]] = int(r.get("total_tokens") or 0)

    items = sorted(i for i in items_meta
                   if all(len(by.get((t, i), {})) >= K for t in TIERS))
    print(f"{len(items)}/{len(items_meta)} items have all 3 tiers x k={K} generations")
    if not items:
        raise SystemExit("no complete items; Stage 10(a) blocked")

    def outcomes(assign) -> np.ndarray:
        return np.array([[int(by[(assign[i], i)][s]) for s in range(K)] for i in items])

    # ---- policies whose per-item tier is deterministic given the prompt
    assign = {}
    from main import classify_prompt_local_nlp
    assign["ecologic"] = {i: int(classify_prompt_local_nlp(
        items_meta[i]["raw_query"]).recommended_tier) for i in items}
    for t in TIERS:
        assign[f"always_t{t}"] = {i: t for i in items}
    rng = random.Random(20260906)
    assign["random"] = {i: rng.choice(TIERS) for i in items}

    texts = [items_meta[i]["raw_query"] for i in items]
    for name, pkl, thr in (("learned_s7", OUT / "s7_router_model.pkl",
                            OUT / "s7_chosen_threshold.json"),
                           ("learned_s5", V2 / "router_model.pkl",
                            V2 / "chosen_threshold.json")):
        with open(pkl, "rb") as f:
            bundle = pickle.load(f)
        with open(thr) as f:
            ct = json.load(f)
        p = predict(bundle["model"], texts)
        tiers = route(p[1], p[2], ct["chosen"]["tau"], tuple(ct["candidate_order"]))
        assign[name] = {i: int(t) for i, t in zip(items, tiers)}

    results = {}
    for name, m in assign.items():
        Y = outcomes(m)
        d = decompose(Y)
        k0 = int(Y[:, 0].sum())
        acc0, lo, hi = wilson(k0, len(items))
        d["single_run_accuracy_sample0"] = acc0
        d["wilson_half_width_pp"] = (hi - lo) / 2 * 100
        results[name] = d

    # ---- per-tier view: raw provider nondeterminism, independent of any policy
    per_tier = {}
    for t in TIERS:
        Y = np.array([[int(by[(t, i)][s]) for s in range(K)] for i in items])
        d = decompose(Y)
        tt = np.array([[tok[(t, i)][s] for s in range(K)] for i in items], float)
        d["mean_tokens"] = float(tt.mean())
        d["mean_within_item_token_sd"] = float(tt.std(axis=1, ddof=1).mean())
        d["frac_items_identical_token_counts"] = float(
            np.mean(tt.std(axis=1) == 0))
        per_tier[str(t)] = d

    payload = {"k": K, "temperature": 0.0, "n_items": len(items),
               "policies": results, "per_tier": per_tier}
    with open(OUT / "s7_generation_variance.json", "w") as f:
        json.dump(payload, f, indent=2)

    L = []
    L.append("# Stage 10(a) — generation-variance decomposition\n")
    L.append(f"The Stage 7 frozen test set (n = {len(items)}) was regenerated **k = 3 times "
             f"per item per tier at temperature 0**. Replicates therefore differ only "
             f"through provider-side nondeterminism, not sampling temperature — which is a "
             f"distinct quantity from Stage 7's temperature-0.7 training samples.\n")
    L.append("## How much does a temperature-0 rerun actually change?\n")
    L.append("| Tier | Mean accuracy over k=3 | Items that flipped across replicates | "
             "Mean within-item token SD | Items with byte-identical token counts |")
    L.append("|---|---|---|---|---|")
    for t in TIERS:
        d = per_tier[str(t)]
        L.append(f"| Tier {t} | {d['mean_accuracy_over_k']:.1%} | "
                 f"{d['n_items_with_flip']} ({d['frac_items_with_flip']:.1%}) | "
                 f"{d['mean_within_item_token_sd']:.1f} tokens | "
                 f"{d['frac_items_identical_token_counts']:.1%} |")
    L.append("")
    L.append("A flipped item is one where the same tier, on the same prompt, at temperature "
             "0, graded correct on some replicates and incorrect on others. Temperature 0 "
             "is not determinism.\n")
    L.append("## Variance decomposition per policy\n")
    L.append("| Policy | Acc (k=3 mean) | between-item var | within-item var | "
             "within share | sampling-only ± | sampling+generation ± | Wilson ± (reported) |")
    L.append("|---|---|---|---|---|---|---|---|")
    order = ["ecologic", "always_t1", "always_t2", "always_t3", "random",
             "learned_s7", "learned_s5"]
    pretty = {"ecologic": "EcoLogic keyword", "always_t1": "Always Tier 1",
              "always_t2": "Always Tier 2", "always_t3": "Always-frontier",
              "random": "Random tier", "learned_s7": "**Learned router (Stage 7)**",
              "learned_s5": "Learned router (Stage 5)"}
    for name in order:
        d = results[name]
        L.append(f"| {pretty[name]} | {d['mean_accuracy_over_k']:.1%} | "
                 f"{d['between_item_var']:.5f} | {d['within_item_generation_var']:.5f} | "
                 f"{d['within_share_of_total']:.1%} | "
                 f"{d['half_width_sampling_only_pp']:.2f} pp | "
                 f"{d['half_width_sampling_plus_generation_pp']:.2f} pp | "
                 f"{d['wilson_half_width_pp']:.2f} pp |")
    L.append("")
    L.append("## Reading this table\n")
    L.append("**The Wilson intervals in every table of this project are not too narrow.** "
             "That is the first thing to establish, because it is the natural worry. A "
             "single-generation evaluation draws one `Y_i` per item, and that draw already "
             "contains the generation noise, so `p(1-p)/n` is the right total variance for "
             "it. The Wilson column matches the sampling+generation column closely, which is "
             "the arithmetic confirming this.\n")
    L.append("**What the decomposition does show** is that a Wilson interval is routinely "
             "*read* as the wrong thing. Readers treat it as the uncertainty in the "
             "benchmark's verdict about these models — as if rerunning generation would land "
             "inside the interval and only a different item sample would move it. The "
             "'within share' column says how much of the interval is instead pure "
             "regeneration noise that no amount of item sampling would reduce, and that "
             "rerunning the identical evaluation would reproduce differently.\n")
    L.append("**Consequence for tight comparisons.** Two policies whose accuracies differ by "
             "less than the sampling+generation half-width cannot be separated by a single "
             "run, however many items are used, unless the comparison is paired on identical "
             "generations — which is why the McNemar tests in this project are all paired on "
             "the same response set, and why the pre-registered verdict rests on a paired "
             "test rather than on overlapping intervals.\n")
    L.append("Negative raw between-item variance estimates (possible because the estimator "
             "is unbiased, not non-negative) are reported in the JSON and clipped to zero in "
             "the table; a clipped value means item difficulty is indistinguishable from "
             "pure generation noise at this sample size.\n")
    with open(OUT / "s7_generation_variance.md", "w") as f:
        f.write("\n".join(L))

    print("\nper-tier flip rates at temperature 0:")
    for t in TIERS:
        d = per_tier[str(t)]
        print(f"  tier {t}: {d['n_items_with_flip']}/{len(items)} items flipped "
              f"({d['frac_items_with_flip']:.1%}), token SD {d['mean_within_item_token_sd']:.1f}")
    print("\npolicy interval widths (pp):")
    for name in order:
        d = results[name]
        print(f"  {name:12} sampling-only {d['half_width_sampling_only_pp']:.2f}  "
              f"+generation {d['half_width_sampling_plus_generation_pp']:.2f}  "
              f"wilson {d['wilson_half_width_pp']:.2f}")
    print("wrote stage7_10/s7_generation_variance.json/.md")


if __name__ == "__main__":
    main()

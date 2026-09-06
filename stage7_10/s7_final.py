"""Stage 7 - ONE-SHOT evaluation on the new frozen 364-item test set.

Nothing tunable remains: featurizer, weights and threshold are loaded from disk
exactly as s7_fit.py and s7_calibrate.py left them. Wilson intervals, McNemar
exact tests and the energy-rate sensitivity sweep are the same code as the
previous reports (imported from benchmark/analyze.py).

Eight rows. Rows 1-6 are the Stage 5 policy set. Row 7 is the new Stage 7
router. Row 8 is **Stage 5's own router re-applied to this same test set**,
which makes the data-scaling effect a paired, item-for-item comparison instead
of a cross-test-set one; Stage 5's original number on its original test set is
quoted alongside for reference but is not paired with anything here.

Writes stage7_10/stage7_results.md and s7_final_results.json.
"""

import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in ("benchmark", "router_v2", "stage7_10", "backend"):
    sys.path.insert(0, str(ROOT / p))
from analyze import mcnemar, wilson  # noqa: E402
from calibrate import route  # noqa: E402
from s7_data import (BENCHMARKS, PAPER_RATES, TIERS, load_split,  # noqa: E402
                     test_correct_and_tokens)
from train_router import predict  # noqa: E402

OUT = ROOT / "stage7_10"
V2 = ROOT / "router_v2"

PRETTY = {
    "ecologic": "EcoLogic keyword router",
    "always_t1": "Always Tier 1",
    "always_t2": "Always Tier 2",
    "random": "Random tier",
    "oracle": "Oracle routing",
    "frontier": "Always-frontier (gpt-4o)",
    "learned_s7": "**Learned router — Stage 7 (scaled + k=3 labels)**",
    "learned_s5": "Learned router — Stage 5 model, re-applied here",
}
ROW_ORDER = ["ecologic", "always_t1", "always_t2", "random", "oracle", "frontier",
             "learned_s7", "learned_s5"]


def energy(tokens, tier, item, rates):
    return tokens[(tier, item)] / 1000.0 * rates[tier]


def apply_router(pkl_path, thr_path, texts):
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    with open(thr_path) as f:
        ct = json.load(f)
    p = predict(bundle["model"], texts)
    tiers = route(p[1], p[2], ct["chosen"]["tau"], tuple(ct["candidate_order"]))
    return [int(t) for t in tiers], bundle["variant"], ct["chosen"]["tau"]


def main():
    correct, tokens, bench_of, items = test_correct_and_tokens(sample_idx=0)
    item_meta = {it["item_id"]: it for it in load_split("test")}
    n = len(items)
    print(f"new frozen test set: {n} items complete across all 3 tiers "
          f"(of {len(item_meta)} requested)")

    assign = {}
    from main import classify_prompt_local_nlp
    assign["ecologic"] = {i: int(classify_prompt_local_nlp(
        item_meta[i]["raw_query"]).recommended_tier) for i in items}
    assign["frontier"] = {i: 3 for i in items}
    assign["always_t1"] = {i: 1 for i in items}
    assign["always_t2"] = {i: 2 for i in items}
    rng = random.Random(20260906)
    assign["random"] = {i: rng.choice(TIERS) for i in items}
    oracle = {}
    for i in items:
        cands = [(energy(tokens, t, i, PAPER_RATES), t) for t in TIERS if correct[(t, i)]]
        oracle[i] = (min(cands)[1] if cands
                     else min((energy(tokens, t, i, PAPER_RATES), t) for t in TIERS)[1])
    assign["oracle"] = oracle

    texts = [item_meta[i]["raw_query"] for i in items]
    t7, var7, tau7 = apply_router(OUT / "s7_router_model.pkl",
                                  OUT / "s7_chosen_threshold.json", texts)
    assign["learned_s7"] = dict(zip(items, t7))
    t5, var5, tau5 = apply_router(V2 / "router_model.pkl",
                                  V2 / "chosen_threshold.json", texts)
    assign["learned_s5"] = dict(zip(items, t5))
    print(f"Stage 7 router: variant {var7}, tau={tau7:.4f}")
    print(f"Stage 5 router: variant {var5}, tau={tau5:.4f} (re-applied, not re-tuned)")

    e_frontier = sum(energy(tokens, 3, i, PAPER_RATES) for i in items)

    def stats(name):
        m = assign[name]
        out = [correct[(m[i], i)] for i in items]
        k = sum(out)
        acc, lo, hi = wilson(k, n)
        e = sum(energy(tokens, m[i], i, PAPER_RATES) for i in items)
        per_b = {}
        for b in BENCHMARKS:
            sub = [i for i in items if bench_of[i] == b]
            kk = sum(correct[(m[i], i)] for i in sub)
            per_b[b] = {"correct": kk, "n": len(sub),
                        "acc": kk / len(sub) if sub else 0.0}
        return {"accuracy": acc, "ci": [lo, hi], "correct": k, "n": n,
                "energy_J": e, "energy_vs_frontier": e / e_frontier,
                "tokens": sum(tokens[(m[i], i)] for i in items),
                "per_benchmark": per_b,
                "tier_mix": {str(t): sum(1 for i in items if m[i] == t) for t in TIERS},
                "outcomes": out}

    pol = {name: stats(name) for name in ROW_ORDER}

    # ---- pre-registered verdict
    L_out, T2_out = pol["learned_s7"]["outcomes"], pol["always_t2"]["outcomes"]
    mc_t2 = mcnemar(L_out, T2_out)
    acc_l, acc_t2 = pol["learned_s7"]["accuracy"], pol["always_t2"]["accuracy"]
    e_l, e_t2 = pol["learned_s7"]["energy_J"], pol["always_t2"]["energy_J"]
    energy_red = (1 - e_l / e_t2) * 100
    s1 = (acc_l > acc_t2) and (mc_t2["p_value"] < 0.05)
    s2 = (mc_t2["p_value"] >= 0.05) and (energy_red >= 15.0)

    # third pre-registered outcome: did the gap to Always-Tier-2 narrow >= 1.0 pp
    # relative to Stage 5's 0.28 pp gap on its own test set?
    with open(V2 / "final_test_set_results.json") as f:
        s5 = json.load(f)
    s5_gap_pp = -s5["prereg_comparison_vs_always_t2"]["accuracy_delta_pp"]
    s7_gap_pp = (acc_t2 - acc_l) * 100
    narrowed_pp = s5_gap_pp - s7_gap_pp
    if s1:
        verdict, label = "S1", "hypothesis supported"
    elif s2:
        verdict, label = "S2", "hypothesis supported"
    elif narrowed_pp >= 1.0:
        verdict, label = "PARTIAL", "partial support, inconclusive"
    else:
        verdict, label = "NEITHER", "hypothesis not supported"

    mcs = {name: mcnemar(L_out, pol[name]["outcomes"])
           for name in ROW_ORDER if name != "learned_s7"}

    # ---- sensitivity, 27 combinations
    combos = []
    for m1 in (0.2, 1.0, 5.0):
        for m2 in (0.2, 1.0, 5.0):
            for m3 in (0.2, 1.0, 5.0):
                rates = {1: PAPER_RATES[1] * m1, 2: PAPER_RATES[2] * m2,
                         3: PAPER_RATES[3] * m3}
                el = sum(energy(tokens, assign["learned_s7"][i], i, rates) for i in items)
                ef = sum(energy(tokens, 3, i, rates) for i in items)
                e2 = sum(energy(tokens, 2, i, rates) for i in items)
                combos.append({"mult": [m1, m2, m3], "learned_J": el, "frontier_J": ef,
                               "always_t2_J": e2,
                               "savings_vs_frontier_pct": (1 - el / ef) * 100,
                               "savings_vs_always_t2_pct": (1 - el / e2) * 100})
    sav = [c["savings_vs_frontier_pct"] for c in combos]
    sav_t2 = [c["savings_vs_always_t2_pct"] for c in combos]
    flips = [c for c in combos if c["savings_vs_frontier_pct"] <= 0]
    flips_t2 = [c for c in combos if c["savings_vs_always_t2_pct"] <= 0]

    with open(OUT / "s7_mckp_gaps.json") as f:
        gaps = json.load(f)
    with open(OUT / "s7_model_selection.json") as f:
        msel = json.load(f)
    with open(OUT / "s7_pool_labels.json") as f:
        plab = json.load(f)
        plab.pop("labels", None)

    payload = {
        "verdict": verdict, "verdict_label": label,
        "s1_met": bool(s1), "s2_met": bool(s2),
        "gap_to_always_t2_pp": {"stage5": s5_gap_pp, "stage7": s7_gap_pp,
                                "narrowed_by_pp": narrowed_pp},
        "variant": var7, "tau": tau7, "n_items": n,
        "n_items_requested": len(item_meta),
        "prereg_comparison_vs_always_t2": {
            "learned_accuracy": acc_l, "always_t2_accuracy": acc_t2,
            "accuracy_delta_pp": (acc_l - acc_t2) * 100, "mcnemar": mc_t2,
            "learned_energy_J": e_l, "always_t2_energy_J": e_t2,
            "energy_reduction_vs_always_t2_pct": energy_red},
        "policies": {k: {kk: vv for kk, vv in v.items() if kk != "outcomes"}
                     for k, v in pol.items()},
        "mcnemar_learned_s7_vs": mcs,
        "stage5_original_test_set": {
            "learned_accuracy": s5["prereg_comparison_vs_always_t2"]["learned_accuracy"],
            "always_t2_accuracy": s5["prereg_comparison_vs_always_t2"]["always_t2_accuracy"],
            "verdict": s5["verdict"]},
        "calibration_gaps": gaps, "pool_label_summary": plab,
        "sensitivity": {"n_combinations": len(combos),
                        "savings_vs_frontier_min_pct": min(sav),
                        "savings_vs_frontier_max_pct": max(sav),
                        "n_flips_vs_frontier": len(flips),
                        "savings_vs_always_t2_min_pct": min(sav_t2),
                        "savings_vs_always_t2_max_pct": max(sav_t2),
                        "n_flips_vs_always_t2": len(flips_t2),
                        "combinations": combos},
    }
    with open(OUT / "s7_final_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    # ------------------------------------------------------------------ report
    n_pool = plab["n_items_labelled"]
    cg7 = gaps["calibration_gap_pp"]
    cg6 = gaps["stage6_reference"]["calibration_gap_pp"]
    L = []
    L.append("# Stage 7 — scaled, cleaned retest of the calibration-gap hypothesis\n")
    L.append("## Verdict\n")
    headline = {
        "S1": (f"**Hypothesis supported (S1).** Trained on {n_pool} majority-voted items, "
               f"the learned router beats Always-Tier-2 on the new frozen test set "
               f"({acc_l:.1%} vs {acc_t2:.1%}, McNemar exact p = {mc_t2['p_value']:.4g})."),
        "S2": (f"**Hypothesis supported (S2).** Trained on {n_pool} majority-voted items, "
               f"the learned router matches Always-Tier-2 ({acc_l:.1%} vs {acc_t2:.1%}, "
               f"McNemar exact p = {mc_t2['p_value']:.4g}) at {energy_red:.1f}% lower "
               f"energy."),
        "PARTIAL": (f"**Partial support, inconclusive.** Scaling the training pool "
                    f"{n_pool / 1200:.1f}x and replacing single-sample labels with k=3 "
                    f"majority votes narrowed the accuracy gap to Always-Tier-2 by "
                    f"{narrowed_pp:.2f} pp (Stage 5: {s5_gap_pp:.2f} pp behind, Stage 7: "
                    f"{s7_gap_pp:.2f} pp behind) without reaching significance "
                    f"(McNemar exact p = {mc_t2['p_value']:.4g}); this is reported as "
                    f"partial support per the pre-registration, not as a positive result."),
        "NEITHER": (f"**Hypothesis not supported.** Scaling the training pool "
                    f"{n_pool / 1200:.1f}x and replacing single-sample labels with k=3 "
                    f"majority votes did not let the learned router surpass static "
                    f"assignment: {acc_l:.1%} vs Always-Tier-2's {acc_t2:.1%} "
                    f"(McNemar exact p = {mc_t2['p_value']:.4g}), at {energy_red:+.1f}% "
                    f"energy relative to it. The calibration gap is therefore "
                    f"**structural** — a limit of the feature representation or of "
                    f"prompt-level predictability — not a shortage of training data."),
    }[verdict]
    L.append(headline + "\n")
    L.append(f"Pre-registered outcome label: **{label}**.\n")
    L.append("| Pre-registered criterion | Result |")
    L.append("|---|---|")
    L.append(f"| S1 — beats Always-Tier-2, McNemar p < 0.05 | {'MET' if s1 else 'not met'} |")
    L.append(f"| S2 — matches accuracy at >= 15% less energy | {'MET' if s2 else 'not met'} |")
    L.append(f"| Third outcome — gap to Always-Tier-2 narrows >= 1.0 pp | "
             f"narrowed {narrowed_pp:+.2f} pp "
             f"({'qualifies' if narrowed_pp >= 1.0 else 'does not qualify'}) |")
    L.append("")
    L.append("## What changed relative to Stage 5\n")
    L.append("| | Stage 5 | Stage 7 |")
    L.append("|---|---|---|")
    L.append(f"| Training pool size | 1,200 items | **{n_pool} items** "
             f"({n_pool / 1200:.2f}x) |")
    L.append(f"| Labels | single sample, temp 0 | **majority of k=3, temp 0.7** |")
    L.append(f"| TRAIN items | {msel['stage2_reference']['n_train']} | {msel['n_train']} |")
    L.append(f"| TRAIN-CV winner | {msel['stage2_reference']['winner']} "
             f"({max(msel['stage2_reference']['best_r1_mean_auc'], msel['stage2_reference']['best_r2_mean_auc']):.4f} mean AUC) "
             f"| {msel['winner']} "
             f"({max(msel['best_r1']['mean_auc'], msel['best_r2']['mean_auc']):.4f}) |")
    L.append(f"| Calibration gap to LP frontier | {cg6:+.2f} pp | **{cg7:+.2f} pp** |")
    L.append(f"| Discreteness gap | "
             f"{gaps['stage6_reference']['discreteness_gap_at_router_energy_pp']:+.2f} pp | "
             f"{gaps['discreteness_gap_at_router_energy_pp']:+.2f} pp |")
    L.append("")
    L.append("The calibration-gap row is the direct test. If more and cleaner data were the "
             "binding constraint, the gap should have shrunk materially.\n")

    L.append(f"## Eight-policy comparison, new frozen test set (n = {n})\n")
    L.append("| Policy | Accuracy [95% Wilson CI] | Tokens | Energy (J) | vs frontier | "
             "Tier mix (T1/T2/T3) | McNemar vs Stage 7 router |")
    L.append("|---|---|---|---|---|---|---|")
    for name in ROW_ORDER:
        s = pol[name]
        mx = s["tier_mix"]
        cell = "—" if name == "learned_s7" else (
            f"p={mcs[name]['p_value']:.4g} ({mcs[name]['b']}/{mcs[name]['c']})")
        L.append(f"| {PRETTY[name]} | {s['accuracy']:.1%} "
                 f"[{s['ci'][0]:.1%}, {s['ci'][1]:.1%}] ({s['correct']}/{s['n']}) | "
                 f"{s['tokens']:,} | {s['energy_J']:,.1f} | "
                 f"{s['energy_vs_frontier']:.3f}x | {mx['1']}/{mx['2']}/{mx['3']} | {cell} |")
    L.append("")
    L.append("McNemar columns are exact two-sided tests on the identical item set; `(b/c)` "
             "are discordant counts (Stage-7-router-only-correct / other-only-correct).\n")
    L.append(f"Row 8 is Stage 5's router — same weights, same threshold "
             f"(tau = {tau5:.4f}), no re-tuning — scored on **this** test set, so rows 7 "
             f"and 8 are a paired comparison isolating the effect of scaling the training "
             f"data. For reference, on its own original test set the Stage 5 router scored "
             f"{s5['prereg_comparison_vs_always_t2']['learned_accuracy']:.1%} against "
             f"Always-Tier-2's "
             f"{s5['prereg_comparison_vs_always_t2']['always_t2_accuracy']:.1%}; that "
             f"number is not paired with anything in this table.\n")

    L.append("### Per-benchmark accuracy by policy\n")
    L.append("| Policy | MBPP (%d) | MMLU (%d) | GSM8K (%d) |" % tuple(
        pol["oracle"]["per_benchmark"][b]["n"] for b in BENCHMARKS))
    L.append("|---|---|---|---|")
    for name in ROW_ORDER:
        pb = pol[name]["per_benchmark"]
        L.append(f"| {PRETTY[name]} | " + " | ".join(
            f"{pb[b]['acc']:.1%} ({pb[b]['correct']}/{pb[b]['n']})"
            for b in BENCHMARKS) + " |")
    L.append("")
    L.append("**The code column carries the pre-registered confound.** MBPP had only 410 "
             "unused items left for the training pool against 400 in Stage 1 — a 1.02x "
             "scale-up, versus 5.74x for MMLU and GSM8K. Any null result on code is "
             "therefore confounded with the inability to add code training data, exactly as "
             "flagged in `prereg_stage7.md` before the run.\n")

    L.append("## Pre-registered comparison in detail: Stage 7 router vs Always Tier 2\n")
    L.append("| Quantity | Value |")
    L.append("|---|---|")
    L.append(f"| Stage 7 router accuracy | {acc_l:.1%} ({pol['learned_s7']['correct']}/{n}) |")
    L.append(f"| Always-Tier-2 accuracy | {acc_t2:.1%} ({pol['always_t2']['correct']}/{n}) |")
    L.append(f"| Accuracy difference | {(acc_l - acc_t2) * 100:+.2f} pp |")
    L.append(f"| McNemar exact p | {mc_t2['p_value']:.4g} "
             f"(discordant {mc_t2['b']}/{mc_t2['c']}) |")
    L.append(f"| Stage 7 router energy | {e_l:,.1f} J |")
    L.append(f"| Always-Tier-2 energy | {e_t2:,.1f} J |")
    L.append(f"| Energy reduction vs Always-Tier-2 | {energy_red:+.1f}% "
             f"(>= 15% needed for S2) |")
    L.append(f"| **S1 met** | {bool(s1)} |")
    L.append(f"| **S2 met** | {bool(s2)} |")
    L.append("")

    L.append("## Label noise the k=3 majority vote actually removed\n")
    L.append("| Tier | Items with a 2-1 sample split | Share |")
    L.append("|---|---|---|")
    for t in ("1", "2", "3"):
        d = plab["sample_disagreement_by_tier"][t]
        tot = d["unanimous"] + d["split_2_1"]
        L.append(f"| Tier {t} | {d['split_2_1']:,} / {tot:,} | "
                 f"{d['split_2_1'] / tot:.1%} |" if tot else f"| Tier {t} | n/a | n/a |")
    L.append("")
    L.append(f"Items where no tier's majority grade was correct (label noise, flagged not "
             f"treated as ground truth): **{plab['n_no_tier_correct_flagged']:,}** of "
             f"{n_pool:,}. Oracle label mix: {plab['label_distribution']}.\n")
    L.append("This table quantifies what the label fix bought. A tier whose three samples "
             "disagree is an item whose single-sample Stage 1 label was a coin flip; the "
             "share of such items bounds how much label noise scaling could ever have "
             "removed.\n")

    L.append("## Energy-rate sensitivity for the Stage 7 router (27 combinations)\n")
    L.append(f"- Savings vs always-frontier range **{min(sav):.1f}% to {max(sav):.1f}%**; "
             f"combinations where that conclusion flips: **{len(flips)}/27**.")
    L.append(f"- Savings vs Always-Tier-2 range **{min(sav_t2):.1f}% to {max(sav_t2):.1f}%**; "
             f"combinations where the router uses *more* energy than Always-Tier-2: "
             f"**{len(flips_t2)}/27**.\n")
    L.append("| Tier1 x | Tier2 x | Tier3 x | Router J | Frontier J | vs frontier % | "
             "vs Always-T2 % |")
    L.append("|---|---|---|---|---|---|---|")
    for c in (sorted(combos, key=lambda c: c["savings_vs_frontier_pct"])[:5]
              + sorted(combos, key=lambda c: -c["savings_vs_frontier_pct"])[:3]):
        m = c["mult"]
        L.append(f"| {m[0]:g} | {m[1]:g} | {m[2]:g} | {c['learned_J']:,.0f} | "
                 f"{c['frontier_J']:,.0f} | {c['savings_vs_frontier_pct']:.1f} | "
                 f"{c['savings_vs_always_t2_pct']:.1f} |")
    L.append("\n(5 least-favourable then 3 most-favourable combinations.)\n")
    L.append("## Artifacts\n")
    L.append("- `prereg_stage7.md` — pre-registration, committed before any data\n"
             "- `s7_model_comparison.md` — R1 vs R2 ablation on the scaled pool\n"
             "- `s7_threshold_sweep.csv` / `.png` — CALIBRATION threshold sweep\n"
             "- `s7_mckp_frontier.csv` / `.png` — LP and integer frontiers with both gaps\n"
             "- `s7_final_results.json` — every number above, machine-readable\n"
             "- `s7_LIMITATIONS.md` — what this stage does and does not establish\n")

    with open(OUT / "stage7_results.md", "w") as f:
        f.write("\n".join(L))

    print(f"\nVERDICT: {verdict} ({label})")
    print(f"  stage7 router {acc_l:.4f} ({pol['learned_s7']['correct']}/{n}) "
          f"energy {e_l:,.1f} J")
    print(f"  always_t2     {acc_t2:.4f} ({pol['always_t2']['correct']}/{n}) "
          f"energy {e_t2:,.1f} J")
    print(f"  stage5 router {pol['learned_s5']['accuracy']:.4f} "
          f"({pol['learned_s5']['correct']}/{n}) energy "
          f"{pol['learned_s5']['energy_J']:,.1f} J")
    print(f"  McNemar vs t2 p={mc_t2['p_value']:.4g} b={mc_t2['b']} c={mc_t2['c']}")
    print(f"  gap to t2: stage5 {s5_gap_pp:+.2f} pp -> stage7 {s7_gap_pp:+.2f} pp "
          f"(narrowed {narrowed_pp:+.2f} pp)")
    print("wrote stage7_10/stage7_results.md, s7_final_results.json")


if __name__ == "__main__":
    main()

"""Stage 5 - one-shot evaluation on the frozen 364-item test set.

Nothing tunable remains: the featurizer, the model weights and the threshold are
all loaded from disk exactly as Stages 2 and 3 left them.

No new API calls are made. The already-collected per-tier generations in
raw_results/graded.jsonl are reused, so the learned router is scored on exactly
the same responses as every baseline in the previous report. raw_results/ is
opened read-only; all output goes to router_v2/.

The grading and policy pipeline is imported from benchmark/analyze.py rather
than reimplemented, so the six rows are directly comparable to the previous
report line for line.

Writes router_v2/final_test_set_results.md and final_test_set_results.json.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT / "router_v2"))
from analyze import (BENCHMARKS, PAPER_RATES, TIERS, build_policies,  # noqa: E402
                     energy, load, mcnemar, wilson)
from calibrate import route  # noqa: E402
from train_router import predict  # noqa: E402

OUT = ROOT / "router_v2"
RAW = ROOT / "raw_results"

PRETTY = {
    "ecologic": "EcoLogic keyword router (previous result)",
    "always_t1": "Always Tier 1",
    "always_t2": "Always Tier 2",
    "random": "Random tier",
    "oracle": "Oracle routing",
    "learned": "**Learned router (new)**",
    "frontier": "Always-frontier (gpt-4o)",
}
ROW_ORDER = ["ecologic", "always_t1", "always_t2", "random", "oracle", "learned"]


def main():
    with open(OUT / "router_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model, variant = bundle["model"], bundle["variant"]
    with open(OUT / "chosen_threshold.json") as f:
        _ct = json.load(f)
    tau = _ct["chosen"]["tau"]
    order = tuple(_ct["candidate_order"])

    correct, tokens, usd, items, bench_of, rows = load()
    with open(RAW / "routing.json") as f:
        routing = json.load(f)
    with open(RAW / "benchmark_items.json") as f:
        frozen_items = {it["item_id"]: it for it in json.load(f)["items"]}

    assign = build_policies(correct, tokens, items, routing, PAPER_RATES)

    # the learned router, applied once
    X = [frozen_items[i]["raw_query"] for i in items]
    p = predict(model, X)
    learned_tiers = route(p[1], p[2], tau, order)
    assign["learned"] = {i: int(t) for i, t in zip(items, learned_tiers)}

    n = len(items)
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
            per_b[b] = {"correct": kk, "n": len(sub), "acc": kk / len(sub)}
        mix = {str(t): sum(1 for i in items if m[i] == t) for t in TIERS}
        return {"accuracy": acc, "ci": [lo, hi], "correct": k, "n": n,
                "energy_J": e, "energy_vs_frontier": e / e_frontier,
                "tokens": sum(tokens[(m[i], i)] for i in items),
                "per_benchmark": per_b, "tier_mix": mix, "outcomes": out}

    pol = {name: stats(name) for name in ROW_ORDER + ["frontier"]}

    # ---- pre-registered verdict: learned router vs Always Tier 2
    L_out, T2_out = pol["learned"]["outcomes"], pol["always_t2"]["outcomes"]
    mc_t2 = mcnemar(L_out, T2_out)
    acc_l, acc_t2 = pol["learned"]["accuracy"], pol["always_t2"]["accuracy"]
    e_l, e_t2 = pol["learned"]["energy_J"], pol["always_t2"]["energy_J"]
    energy_red = (1 - e_l / e_t2) * 100

    s1 = (acc_l > acc_t2) and (mc_t2["p_value"] < 0.05)
    s2 = (mc_t2["p_value"] >= 0.05) and (energy_red >= 15.0)
    verdict = "S1" if s1 else ("S2" if s2 else "NEITHER")

    # ---- McNemar: learned vs every other row
    mcs = {}
    for name in ROW_ORDER + ["frontier"]:
        if name == "learned":
            continue
        mcs[name] = mcnemar(L_out, pol[name]["outcomes"])

    # ---- sensitivity for the learned router
    mults = [0.2, 1.0, 5.0]
    combos = []
    for m1 in mults:
        for m2 in mults:
            for m3 in mults:
                rates = {1: PAPER_RATES[1] * m1, 2: PAPER_RATES[2] * m2, 3: PAPER_RATES[3] * m3}
                el = sum(energy(tokens, assign["learned"][i], i, rates) for i in items)
                ef = sum(energy(tokens, 3, i, rates) for i in items)
                e2 = sum(energy(tokens, 2, i, rates) for i in items)
                combos.append({
                    "mult": [m1, m2, m3], "learned_J": el, "frontier_J": ef, "always_t2_J": e2,
                    "savings_vs_frontier_pct": (1 - el / ef) * 100,
                    "savings_vs_always_t2_pct": (1 - el / e2) * 100,
                })
    sav = [c["savings_vs_frontier_pct"] for c in combos]
    flips = [c for c in combos if c["savings_vs_frontier_pct"] <= 0]
    sav_t2 = [c["savings_vs_always_t2_pct"] for c in combos]
    flips_t2 = [c for c in combos if c["savings_vs_always_t2_pct"] <= 0]

    payload = {
        "variant": variant, "tau": tau, "n_items": n,
        "verdict": verdict, "s1_met": bool(s1), "s2_met": bool(s2),
        "prereg_comparison_vs_always_t2": {
            "learned_accuracy": acc_l, "always_t2_accuracy": acc_t2,
            "accuracy_delta_pp": (acc_l - acc_t2) * 100,
            "mcnemar": mc_t2,
            "learned_energy_J": e_l, "always_t2_energy_J": e_t2,
            "energy_reduction_vs_always_t2_pct": energy_red,
        },
        "policies": {k: {kk: vv for kk, vv in v.items() if kk != "outcomes"}
                     for k, v in pol.items()},
        "mcnemar_learned_vs": mcs,
        "sensitivity": {
            "n_combinations": len(combos),
            "savings_vs_frontier_min_pct": min(sav), "savings_vs_frontier_max_pct": max(sav),
            "n_flips_vs_frontier": len(flips), "flips_vs_frontier": flips,
            "savings_vs_always_t2_min_pct": min(sav_t2),
            "savings_vs_always_t2_max_pct": max(sav_t2),
            "n_flips_vs_always_t2": len(flips_t2),
            "combinations": combos,
        },
    }
    with open(OUT / "final_test_set_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    # ------------------------------------------------------------------ report
    L = []
    L.append("# Stage 5 — one-shot frozen test set results\n")
    verdict_line = {
        "S1": f"**S1 was met**: the learned router beats Always-Tier-2 on accuracy "
              f"({acc_l:.1%} vs {acc_t2:.1%}) with McNemar exact p = {mc_t2['p_value']:.4g} < 0.05.",
        "S2": f"**S2 was met**: the learned router matches Always-Tier-2 on accuracy "
              f"({acc_l:.1%} vs {acc_t2:.1%}, McNemar p = {mc_t2['p_value']:.4g} >= 0.05) "
              f"while using {energy_red:.1f}% less energy (>= 15% required).",
        "NEITHER": f"**Neither S1 nor S2 was met: a properly trained router still does not "
                   f"surpass static assignment on this workload.** The learned router scored "
                   f"{acc_l:.1%} against Always-Tier-2's {acc_t2:.1%} "
                   f"(McNemar exact p = {mc_t2['p_value']:.4g}) using "
                   f"{energy_red:+.1f}% energy relative to it "
                   f"(a >= 15% reduction was required for S2).",
    }[verdict]
    L.append("## Pre-registration verdict\n")
    L.append(verdict_line + "\n")
    L.append(f"Evaluated once, as pre-registered: variant **{variant}**, threshold "
             f"**tau = {tau:.4f}**, both fixed before the test set was touched. No new API "
             f"calls; the router selects among per-tier responses already collected for the "
             f"previous report, so all six rows below are scored on identical generations.\n")

    L.append("## Six-policy comparison, frozen 364-item test set\n")
    L.append("| Policy | Accuracy [95% Wilson CI] | Tokens | Energy (J) | vs frontier | Tier mix (T1/T2/T3) | McNemar vs learned |")
    L.append("|---|---|---|---|---|---|---|")
    for name in ROW_ORDER:
        s = pol[name]
        mx = s["tier_mix"]
        mcell = "—" if name == "learned" else (
            f"p={mcs[name]['p_value']:.4g} ({mcs[name]['b']}/{mcs[name]['c']})")
        L.append(f"| {PRETTY[name]} | {s['accuracy']:.1%} [{s['ci'][0]:.1%}, {s['ci'][1]:.1%}] "
                 f"({s['correct']}/{s['n']}) | {s['tokens']:,} | {s['energy_J']:,.1f} "
                 f"| {s['energy_vs_frontier']:.3f}x | {mx['1']}/{mx['2']}/{mx['3']} | {mcell} |")
    s = pol["frontier"]
    mx = s["tier_mix"]
    L.append(f"| {PRETTY['frontier']} *(reference)* | {s['accuracy']:.1%} "
             f"[{s['ci'][0]:.1%}, {s['ci'][1]:.1%}] ({s['correct']}/{s['n']}) | "
             f"{s['tokens']:,} | {s['energy_J']:,.1f} | 1.000x | {mx['1']}/{mx['2']}/{mx['3']} "
             f"| p={mcs['frontier']['p_value']:.4g} ({mcs['frontier']['b']}/{mcs['frontier']['c']}) |")
    L.append("")
    L.append("McNemar columns are exact two-sided tests on the identical item set; `(b/c)` "
             "are the discordant counts (learned-only-correct / other-only-correct).\n")

    L.append("### Per-benchmark accuracy by policy\n")
    L.append("| Policy | HumanEval (164) | MMLU (100) | GSM8K (100) |")
    L.append("|---|---|---|---|")
    for name in ROW_ORDER:
        pb = pol[name]["per_benchmark"]
        L.append(f"| {PRETTY[name]} | " + " | ".join(
            f"{pb[b]['acc']:.1%} ({pb[b]['correct']}/{pb[b]['n']})" for b in BENCHMARKS) + " |")
    L.append("")

    L.append("## Pre-registered comparison in detail: learned router vs Always Tier 2\n")
    L.append("| Quantity | Value |")
    L.append("|---|---|")
    L.append(f"| Learned router accuracy | {acc_l:.1%} ({pol['learned']['correct']}/{n}) |")
    L.append(f"| Always-Tier-2 accuracy | {acc_t2:.1%} ({pol['always_t2']['correct']}/{n}) |")
    L.append(f"| Accuracy difference | {(acc_l - acc_t2) * 100:+.1f} pp |")
    L.append(f"| McNemar exact p | {mc_t2['p_value']:.4g} (discordant {mc_t2['b']}/{mc_t2['c']}) |")
    L.append(f"| Learned router energy | {e_l:,.1f} J |")
    L.append(f"| Always-Tier-2 energy | {e_t2:,.1f} J |")
    L.append(f"| Energy reduction vs Always-Tier-2 | {energy_red:+.1f}% (>= 15% needed for S2) |")
    L.append(f"| **S1 met** | {s1} |")
    L.append(f"| **S2 met** | {s2} |")
    L.append("")

    L.append("## Energy-rate sensitivity for the learned router (27 combinations)\n")
    L.append(f"- Savings vs always-frontier range **{min(sav):.1f}% to {max(sav):.1f}%**; "
             f"combinations where that conclusion flips: **{len(flips)}/27**.")
    L.append(f"- Savings vs Always-Tier-2 range **{min(sav_t2):.1f}% to {max(sav_t2):.1f}%**; "
             f"combinations where the learned router uses *more* energy than Always-Tier-2: "
             f"**{len(flips_t2)}/27**.\n")
    L.append("| Tier1 x | Tier2 x | Tier3 x | Learned J | Frontier J | vs frontier % | vs Always-T2 % |")
    L.append("|---|---|---|---|---|---|---|")
    worst = sorted(combos, key=lambda c: c["savings_vs_frontier_pct"])[:5]
    best = sorted(combos, key=lambda c: -c["savings_vs_frontier_pct"])[:3]
    for c in worst + best:
        m = c["mult"]
        L.append(f"| {m[0]:g} | {m[1]:g} | {m[2]:g} | {c['learned_J']:,.0f} | "
                 f"{c['frontier_J']:,.0f} | {c['savings_vs_frontier_pct']:.1f} | "
                 f"{c['savings_vs_always_t2_pct']:.1f} |")
    L.append("\n(5 least-favourable then 3 most-favourable combinations.)\n")

    with open(OUT / "final_test_set_results.md", "w") as f:
        f.write("\n".join(L))

    print(f"VERDICT: {verdict}")
    print(f"learned {acc_l:.4f} ({pol['learned']['correct']}/{n}) energy {e_l:,.1f} J")
    print(f"always_t2 {acc_t2:.4f} ({pol['always_t2']['correct']}/{n}) energy {e_t2:,.1f} J")
    print(f"McNemar p={mc_t2['p_value']:.4g} b={mc_t2['b']} c={mc_t2['c']}")
    print(f"energy reduction vs always_t2 = {energy_red:+.1f}%")
    print("wrote router_v2/final_test_set_results.md/.json")


if __name__ == "__main__":
    main()

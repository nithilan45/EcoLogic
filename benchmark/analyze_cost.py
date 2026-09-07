"""Policy comparison on a MEASURED cost axis instead of a modelled energy axis.

`analyze.py` reports the policy table in joules, computed as token counts times
the audited system's own assumed J/1k-token rates. Those rates are an assumption
we inherited and never verified; no joule was measured anywhere in this project,
which is the single largest limitation of the energy framing.

Per-call dollar cost, by contrast, is *measured*: it comes from the providers'
own `usage` fields at their published prices, and is recorded per call in
`raw_results/graded.jsonl`. Re-running the identical policy comparison on that
axis therefore removes the rate assumption entirely.

This is additive. It does not modify `analyze.py`, `raw_results/analysis.json` or
`raw_results/tables.md`; the energy table stands as reported.

The interesting question this settles: is the finding that a static single-tier
policy dominates the router an artifact of the assumed energy rates, or does it
survive on a measured axis?
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from analyze import (BENCHMARKS, RAW, ROUTING_KEY, TIERS, load, mcnemar, wilson)

OUT = RAW


def build_policies_cost(correct, usd, items, routing) -> dict:
    """Same policy family as analyze.py, but the oracle minimises measured cost."""
    assign = {}
    assign["ecologic"] = {i: routing[i][ROUTING_KEY]["tier"] for i in items}
    assign["frontier"] = {i: 3 for i in items}
    assign["always_t1"] = {i: 1 for i in items}
    assign["always_t2"] = {i: 2 for i in items}

    # Same seed as analyze.py so the random policy is the identical assignment
    # and the two tables are directly comparable row for row.
    rng = random.Random(20260905)
    assign["random"] = {i: rng.choice(TIERS) for i in items}

    oracle = {}
    for i in items:
        cands = [(usd[(t, i)], t) for t in TIERS if correct[(t, i)]]
        oracle[i] = (min(cands) if cands
                     else min((usd[(t, i)], t) for t in TIERS))[1]
    assign["oracle"] = oracle
    return assign


def stats_cost(assign_map, correct, usd, items, bench_of) -> dict:
    k = sum(correct[(assign_map[i], i)] for i in items)
    n = len(items)
    p, lo, hi = wilson(k, n)
    total = sum(usd[(assign_map[i], i)] for i in items)
    per_bench = {}
    for b in BENCHMARKS:
        sub = [i for i in items if bench_of[i] == b]
        kk = sum(correct[(assign_map[i], i)] for i in sub)
        per_bench[b] = {"k": kk, "n": len(sub),
                        "acc": kk / len(sub) if sub else 0.0}
    dist = defaultdict(int)
    for i in items:
        dist[assign_map[i]] += 1
    return {
        "accuracy": p, "ci": [lo, hi], "correct": k, "n": n,
        "cost_usd": total, "cost_usd_per_item": total / n,
        "tier_mix": {t: dist[t] for t in TIERS},
        "per_benchmark": per_bench,
    }


PRETTY = {
    "ecologic": "EcoLogic routing (real classifier)",
    "frontier": "Always-frontier (gpt-4o)",
    "random": "Random tier",
    "oracle": "Oracle (cheapest correct)",
    "always_t1": "Always Tier 1",
    "always_t2": "Always Tier 2",
}


def main() -> None:
    correct, tokens, usd, items, bench_of, _ = load()
    with open(RAW / "routing.json") as f:
        routing = json.load(f)

    assign = build_policies_cost(correct, usd, items, routing)
    pol = {name: stats_cost(m, correct, usd, items, bench_of)
           for name, m in assign.items()}

    frontier_cost = pol["frontier"]["cost_usd"]
    for v in pol.values():
        v["cost_vs_frontier"] = v["cost_usd"] / frontier_cost

    # Paired tests against the router, on the identical item set.
    vec = {name: [correct[(m[i], i)] for i in items] for name, m in assign.items()}
    tests = {name: mcnemar(vec["ecologic"], vec[name])
             for name in assign if name != "ecologic"}

    order = ["ecologic", "always_t1", "always_t2", "random", "frontier", "oracle"]

    # Does the qualitative finding survive the change of axis? On the energy axis
    # Always-Tier-2 dominates EcoLogic on both accuracy and energy.
    eco, t2 = pol["ecologic"], pol["always_t2"]
    dominated = (t2["accuracy"] > eco["accuracy"]
                 and t2["cost_usd"] < eco["cost_usd"])

    result = {
        "axis": "measured USD from provider usage fields (no modelled rates)",
        "n_items": len(items),
        "total_measured_cost_usd_all_tiers": sum(usd.values()),
        "policies": {k: pol[k] for k in order},
        "mcnemar_vs_ecologic": tests,
        "always_t2_dominates_ecologic_on_measured_cost": bool(dominated),
        "accuracy_gap_t2_minus_ecologic_pp": 100 * (t2["accuracy"] - eco["accuracy"]),
        "cost_ratio_ecologic_over_t2": eco["cost_usd"] / t2["cost_usd"],
    }
    with open(OUT / "analysis_cost.json", "w") as f:
        json.dump(result, f, indent=2)

    lines = ["### Policy comparison on a MEASURED cost axis (n = "
             f"{len(items)} items)", "",
             "Dollar cost from the providers' own `usage` fields at published "
             "prices. No modelled energy rates are used anywhere in this table.",
             "",
             "| Policy | Accuracy [95% CI] | Cost (USD) | $/item | vs frontier | "
             "Tier mix (T1/T2/T3) | McNemar vs EcoLogic |", "|---|---|---|---|---|---|---|"]
    for name in order:
        v = pol[name]
        mc = ("—" if name == "ecologic" else
              f"p={tests[name]['p_value']:.4g} ({tests[name]['b']}/{tests[name]['c']})")
        lines.append(
            f"| {PRETTY[name]} | {v['accuracy']:.1%} [{v['ci'][0]:.1%}, "
            f"{v['ci'][1]:.1%}] ({v['correct']}/{v['n']}) | ${v['cost_usd']:.4f} | "
            f"${v['cost_usd_per_item']:.6f} | {v['cost_vs_frontier']:.3f}x | "
            f"{v['tier_mix'][1]}/{v['tier_mix'][2]}/{v['tier_mix'][3]} | {mc} |")
    lines += ["", f"**Always-Tier-2 dominates EcoLogic on the measured cost axis: "
                 f"{dominated}** "
                 f"(+{100*(t2['accuracy']-eco['accuracy']):.1f} pp accuracy at "
                 f"{eco['cost_usd']/t2['cost_usd']:.2f}x the router's cost).", ""]
    with open(OUT / "tables_cost.md", "w") as f:
        f.write("\n".join(lines))

    for name in order:
        v = pol[name]
        print(f"  {PRETTY[name]:36s} {v['accuracy']:6.1%}  ${v['cost_usd']:8.4f}  "
              f"{v['cost_vs_frontier']:6.3f}x frontier")
    print()
    print(f"Always-Tier-2 dominates EcoLogic on measured cost: {dominated}")
    print(f"  accuracy gap +{100*(t2['accuracy']-eco['accuracy']):.1f} pp, "
          f"EcoLogic costs {eco['cost_usd']/t2['cost_usd']:.2f}x Always-Tier-2")
    print("wrote raw_results/analysis_cost.json and raw_results/tables_cost.md")


if __name__ == "__main__":
    main()

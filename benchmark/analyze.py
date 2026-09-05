"""Policy comparison, statistics and energy accounting.

Reads raw_results/graded.jsonl + raw_results/routing.json.
Writes raw_results/analysis.json and raw_results/tables.md.

Policies compared on the identical item set:
  ecologic  - the real classifier's tier for each item
  frontier  - tier 3 (gpt-4o) on everything            [quality ceiling]
  random    - uniform random tier per item             [sanity floor]
  oracle    - lowest-energy tier that got it right     [efficiency ceiling]
"""

import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api import MODELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_results"

TIERS = [1, 2, 3]
BENCHMARKS = ["humaneval", "mmlu", "gsm8k"]
Z = 1.959963984540054  # 95%

PAPER_RATES = {t: MODELS[t]["paper_energy_per_1k"] for t in TIERS}
SUBST_RATES = {t: MODELS[t]["subst_energy_per_1k"] for t in TIERS}
ROUTING_KEY = "raw"


# ------------------------------------------------------------------ statistics

def wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denom
    half = Z / denom * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (p, max(0.0, center - half), min(1.0, center + half))


def fmt_acc(k: int, n: int) -> str:
    p, lo, hi = wilson(k, n)
    return f"{p:.1%} [{lo:.1%}, {hi:.1%}] ({k}/{n})"


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar on paired binary outcomes."""
    b_only = sum(1 for x, y in zip(a, b) if x and not y)
    c_only = sum(1 for x, y in zip(a, b) if y and not x)
    n = b_only + c_only
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0, "note": "no discordant pairs"}
    p = binomtest(b_only, n, 0.5, alternative="two-sided").pvalue
    return {"b": b_only, "c": c_only, "p_value": float(p), "note": ""}


# ----------------------------------------------------------------------- data

def load() -> tuple[dict, dict, dict, list[str]]:
    correct: dict[tuple[int, str], bool] = {}
    tokens: dict[tuple[int, str], int] = {}
    usd: dict[tuple[int, str], float] = {}
    rows = []
    with open(RAW / "graded.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    bench_of = {}
    for r in rows:
        key = (r["tier"], r["item_id"])
        correct[key] = bool(r.get("correct"))
        tokens[key] = int(r.get("total_tokens") or 0)
        usd[key] = float(r.get("usd") or 0.0)
        bench_of[r["item_id"]] = r["benchmark"]

    # only items with a successful response from all three tiers are comparable
    item_ids = sorted({i for (_, i) in correct}, key=lambda x: (bench_of[x], x))
    complete = [i for i in item_ids if all((t, i) in correct for t in TIERS)]
    return correct, tokens, usd, complete, bench_of, rows


def energy(tokens: dict, tier: int, item: str, rates: dict) -> float:
    return tokens[(tier, item)] / 1000.0 * rates[tier]


# -------------------------------------------------------------------- policies

def build_policies(correct, tokens, items, routing, rates) -> dict:
    assign = {}

    assign["ecologic"] = {i: routing[i][ROUTING_KEY]["tier"] for i in items}
    assign["frontier"] = {i: 3 for i in items}
    # static single-tier baselines: a router must beat these to be worth having
    assign["always_t1"] = {i: 1 for i in items}
    assign["always_t2"] = {i: 2 for i in items}

    rng = random.Random(20260905)
    assign["random"] = {i: rng.choice(TIERS) for i in items}

    oracle = {}
    for i in items:
        cands = [(energy(tokens, t, i, rates), t) for t in TIERS if correct[(t, i)]]
        if cands:
            oracle[i] = min(cands)[1]
        else:
            # nothing was right; the system still has to answer something, so
            # charge it the cheapest available response
            oracle[i] = min((energy(tokens, t, i, rates), t) for t in TIERS)[1]
    assign["oracle"] = oracle
    return assign


def policy_stats(assign_map, correct, tokens, items, rates, bench_of) -> dict:
    outcomes = [correct[(assign_map[i], i)] for i in items]
    k = sum(outcomes)
    n = len(items)
    p, lo, hi = wilson(k, n)
    e = sum(energy(tokens, assign_map[i], i, rates) for i in items)
    tok = sum(tokens[(assign_map[i], i)] for i in items)
    per_bench = {}
    for b in BENCHMARKS:
        sub = [i for i in items if bench_of[i] == b]
        kk = sum(correct[(assign_map[i], i)] for i in sub)
        per_bench[b] = {"k": kk, "n": len(sub), "acc": kk / len(sub) if sub else 0.0}
    dist = defaultdict(int)
    for i in items:
        dist[assign_map[i]] += 1
    return {
        "accuracy": p, "ci": [lo, hi], "correct": k, "n": n,
        "energy_J": e, "tokens": tok,
        "per_benchmark": per_bench,
        "tier_distribution": {str(t): dist[t] for t in TIERS},
        "outcomes": outcomes,
    }


# ------------------------------------------------------------------ main flow

def main():
    correct, tokens, usd, items, bench_of, rows = load()
    with open(RAW / "routing.json") as f:
        routing = json.load(f)

    out: dict = {"n_items_complete": len(items), "routing_key": ROUTING_KEY}
    lines: list[str] = []

    # ---- data completeness / failures
    all_ids = sorted({i for (_, i) in correct})
    missing = {}
    for i in all_ids:
        absent = [t for t in TIERS if (t, i) not in correct]
        if absent:
            missing[i] = absent
    errors = [r for r in rows if r.get("error")]
    truncated = defaultdict(int)
    ungradable = defaultdict(int)
    for r in rows:
        if r.get("truncated"):
            truncated[(r["tier"], r["benchmark"])] += 1
        if r.get("gradable") is False:
            ungradable[(r["tier"], r["benchmark"])] += 1
    out["failures"] = {
        "api_errors": len(errors),
        "api_error_detail": [
            {"tier": r["tier"], "item_id": r["item_id"], "error": r["error"]} for r in errors
        ][:50],
        "items_missing_a_tier": missing,
        "truncated_by_tier_benchmark": {f"t{k[0]}/{k[1]}": v for k, v in sorted(truncated.items())},
        "ungradable_by_tier_benchmark": {f"t{k[0]}/{k[1]}": v for k, v in sorted(ungradable.items())},
    }
    out["total_cost_usd"] = round(sum(usd.values()), 4)

    # ---- per-tier per-benchmark accuracy
    lines.append("### Per-tier accuracy by benchmark (95% Wilson CI)\n")
    lines.append("| Benchmark | n | Tier 1 (Qwen3.5-9B) | Tier 2 (gpt-oss-20b) | Tier 3 (gpt-4o) |")
    lines.append("|---|---|---|---|---|")
    tier_acc = {}
    for b in BENCHMARKS + ["ALL"]:
        sub = items if b == "ALL" else [i for i in items if bench_of[i] == b]
        cells = []
        for t in TIERS:
            k = sum(correct[(t, i)] for i in sub)
            tier_acc[(t, b)] = (k, len(sub))
            cells.append(fmt_acc(k, len(sub)))
        label = "**ALL (system-wide)**" if b == "ALL" else b
        lines.append(f"| {label} | {len(sub)} | " + " | ".join(cells) + " |")
    lines.append("")
    out["tier_accuracy"] = {
        f"tier{t}/{b}": {"correct": k, "n": n, "acc": (k / n if n else 0),
                         "ci": list(wilson(k, n)[1:])}
        for (t, b), (k, n) in tier_acc.items()
    }

    # ---- MMLU per subject
    subj_of = {}
    for r in rows:
        if r["benchmark"] == "mmlu":
            subj_of[r["item_id"]] = r["subject"]
    lines.append("### MMLU accuracy by subject\n")
    lines.append("| Subject | n | Tier 1 | Tier 2 | Tier 3 |")
    lines.append("|---|---|---|---|---|")
    subj_tbl = {}
    for s in sorted(set(subj_of.values())):
        sub = [i for i in items if subj_of.get(i) == s]
        cells, rec = [], {}
        for t in TIERS:
            k = sum(correct[(t, i)] for i in sub)
            cells.append(f"{k}/{len(sub)}")
            rec[f"tier{t}"] = {"correct": k, "n": len(sub)}
        subj_tbl[s] = rec
        lines.append(f"| {s} | {len(sub)} | " + " | ".join(cells) + " |")
    lines.append("")
    out["mmlu_by_subject"] = subj_tbl

    # ---- McNemar, paired tier vs tier
    lines.append("### McNemar exact test, paired tier-vs-tier on identical items\n")
    lines.append("| Benchmark | Pair | A>B | B>A | p-value | Significant at 0.05 |")
    lines.append("|---|---|---|---|---|---|")
    mc = {}
    for b in BENCHMARKS + ["ALL"]:
        sub = items if b == "ALL" else [i for i in items if bench_of[i] == b]
        for a, bb in ((1, 2), (1, 3), (2, 3)):
            va = [correct[(a, i)] for i in sub]
            vb = [correct[(bb, i)] for i in sub]
            r = mcnemar(va, vb)
            mc[f"{b}/t{a}_vs_t{bb}"] = r
            sig = "yes" if r["p_value"] < 0.05 else "no"
            label = "**ALL**" if b == "ALL" else b
            lines.append(
                f"| {label} | T{a} vs T{bb} | {r['b']} | {r['c']} | "
                f"{r['p_value']:.4g} | {sig} |"
            )
    lines.append("")
    out["mcnemar"] = mc

    # ---- four-policy comparison
    assign = build_policies(correct, tokens, items, routing, PAPER_RATES)
    pol = {name: policy_stats(m, correct, tokens, items, PAPER_RATES, bench_of)
           for name, m in assign.items()}

    # random policy: exact expectation + simulated spread
    rng = random.Random(7)
    sims_acc, sims_energy = [], []
    for _ in range(2000):
        a = 0
        e = 0.0
        for i in items:
            t = rng.choice(TIERS)
            a += correct[(t, i)]
            e += energy(tokens, t, i, PAPER_RATES)
        sims_acc.append(a / len(items))
        sims_energy.append(e)
    exp_acc = statistics.mean(tier_acc[(t, "ALL")][0] / len(items) for t in TIERS)
    exp_energy = statistics.mean(
        sum(energy(tokens, t, i, PAPER_RATES) for i in items) for t in TIERS
    )
    out["random_policy_expectation"] = {
        "exact_expected_accuracy": exp_acc,
        "exact_expected_energy_J": exp_energy,
        "sim_accuracy_mean": statistics.mean(sims_acc),
        "sim_accuracy_p2.5": sorted(sims_acc)[int(0.025 * len(sims_acc))],
        "sim_accuracy_p97.5": sorted(sims_acc)[int(0.975 * len(sims_acc)) - 1],
        "sim_energy_mean_J": statistics.mean(sims_energy),
    }

    lines.append("### Four-policy comparison (identical %d-item set)\n" % len(items))
    lines.append("| Policy | Accuracy [95% CI] | Total tokens | Energy (J) | Energy vs frontier | Tier mix (T1/T2/T3) |")
    lines.append("|---|---|---|---|---|---|")
    ef = pol["frontier"]["energy_J"]
    order = ["ecologic", "frontier", "random", "oracle", "always_t1", "always_t2"]
    pretty = {"ecologic": "(a) EcoLogic routing", "frontier": "(b) Always-frontier (gpt-4o)",
              "random": "(c) Random tier", "oracle": "(d) Oracle routing",
              "always_t1": "Always Tier 1 (baseline)", "always_t2": "Always Tier 2 (baseline)"}
    for name in order:
        s = pol[name]
        d = s["tier_distribution"]
        lines.append(
            f"| {pretty[name]} | {s['accuracy']:.1%} [{s['ci'][0]:.1%}, {s['ci'][1]:.1%}] "
            f"| {s['tokens']:,} | {s['energy_J']:,.1f} | {s['energy_J'] / ef:.3f}x "
            f"| {d['1']}/{d['2']}/{d['3']} |"
        )
    lines.append("")

    for name in order:
        pol[name].pop("outcomes", None)
    out["policies"] = pol

    # ---- oracle gap + quality gap
    eco, fro, orc = pol["ecologic"], pol["frontier"], pol["oracle"]
    eco_out = [correct[(assign["ecologic"][i], i)] for i in items]
    fro_out = [correct[(3, i)] for i in items]
    orc_out = [correct[(assign["oracle"][i], i)] for i in items]
    mc_eco_fro = mcnemar(eco_out, fro_out)
    mc_eco_orc = mcnemar(eco_out, orc_out)
    t1_out = [correct[(1, i)] for i in items]
    mc_eco_t1 = mcnemar(eco_out, t1_out)

    quality_gap_pp = (fro["accuracy"] - eco["accuracy"]) * 100
    energy_gap_abs = eco["energy_J"] - orc["energy_J"]
    energy_gap_pct = (energy_gap_abs / orc["energy_J"] * 100) if orc["energy_J"] else 0.0
    savings_vs_frontier = (1 - eco["energy_J"] / ef) * 100

    out["headline"] = {
        "ecologic_accuracy": eco["accuracy"],
        "frontier_accuracy": fro["accuracy"],
        "oracle_accuracy": orc["accuracy"],
        "random_accuracy": pol["random"]["accuracy"],
        "quality_gap_pp_vs_frontier": quality_gap_pp,
        "quality_gap_mcnemar": mc_eco_fro,
        "quality_gap_vs_oracle_mcnemar": mc_eco_orc,
        "vs_always_tier1_mcnemar": mc_eco_t1,
        "always_tier1_accuracy": pol["always_t1"]["accuracy"],
        "always_tier1_energy_J": pol["always_t1"]["energy_J"],
        "ecologic_energy_J": eco["energy_J"],
        "frontier_energy_J": ef,
        "oracle_energy_J": orc["energy_J"],
        "oracle_gap_energy_J": energy_gap_abs,
        "oracle_gap_energy_pct_over_oracle": energy_gap_pct,
        "ecologic_energy_savings_vs_frontier_pct": savings_vs_frontier,
    }

    lines.append("### Headline gaps\n")
    lines.append("| Quantity | Value |")
    lines.append("|---|---|")
    lines.append(f"| EcoLogic accuracy | {eco['accuracy']:.1%} |")
    lines.append(f"| Always-frontier accuracy | {fro['accuracy']:.1%} |")
    lines.append(f"| **Quality gap** (frontier - EcoLogic) | **{quality_gap_pp:+.1f} pp** "
                 f"(McNemar p={mc_eco_fro['p_value']:.4g}) |")
    lines.append(f"| EcoLogic energy | {eco['energy_J']:,.1f} J |")
    lines.append(f"| Oracle energy | {orc['energy_J']:,.1f} J |")
    lines.append(f"| **Oracle gap** (energy left on table) | **{energy_gap_abs:,.1f} J "
                 f"= {energy_gap_pct:+.1f}% over oracle** |")
    lines.append(f"| EcoLogic energy savings vs frontier | {savings_vs_frontier:.1f}% |")
    lines.append(f"| Oracle accuracy (efficiency ceiling) | {orc['accuracy']:.1%} |")
    lines.append(f"| Always-Tier-1 accuracy / energy | {pol['always_t1']['accuracy']:.1%} / "
                 f"{pol['always_t1']['energy_J']:,.1f} J |")
    lines.append(f"| EcoLogic vs Always-Tier-1 | McNemar p={mc_eco_t1['p_value']:.4g} "
                 f"(discordant {mc_eco_t1['b']}/{mc_eco_t1['c']}) |")
    lines.append("")

    # ---- energy accounting with substitute-model rates too
    lines.append("### Energy under both rate sets\n")
    lines.append("| Policy | Energy, paper rates (0.5/1.5/60 J per 1k) | Energy, substitute-model rates (1.1/2.0/60) |")
    lines.append("|---|---|---|")
    energy_alt = {}
    for name in order:
        e1 = sum(energy(tokens, assign[name][i], i, PAPER_RATES) for i in items)
        e2 = sum(energy(tokens, assign[name][i], i, SUBST_RATES) for i in items)
        energy_alt[name] = {"paper_J": e1, "substitute_J": e2}
        lines.append(f"| {pretty[name]} | {e1:,.1f} J | {e2:,.1f} J |")
    lines.append("")
    out["energy_both_rate_sets"] = energy_alt

    # ---- sensitivity: +/-5x on each tier's rate independently
    mults = [0.2, 1.0, 5.0]
    combos = []
    for m1 in mults:
        for m2 in mults:
            for m3 in mults:
                rates = {1: PAPER_RATES[1] * m1, 2: PAPER_RATES[2] * m2, 3: PAPER_RATES[3] * m3}
                e_eco = sum(energy(tokens, assign["ecologic"][i], i, rates) for i in items)
                e_fro = sum(energy(tokens, 3, i, rates) for i in items)
                e_orc_assign = build_policies(correct, tokens, items, routing, rates)["oracle"]
                e_orc = sum(energy(tokens, e_orc_assign[i], i, rates) for i in items)
                combos.append({
                    "mult": [m1, m2, m3],
                    "eco_J": e_eco, "frontier_J": e_fro, "oracle_J": e_orc,
                    "savings_vs_frontier_pct": (1 - e_eco / e_fro) * 100 if e_fro else 0.0,
                    "oracle_gap_pct": ((e_eco - e_orc) / e_orc * 100) if e_orc else 0.0,
                })
    sav = [c["savings_vs_frontier_pct"] for c in combos]
    flips = [c for c in combos if c["savings_vs_frontier_pct"] <= 0]
    out["sensitivity"] = {
        "n_combinations": len(combos),
        "savings_min_pct": min(sav),
        "savings_max_pct": max(sav),
        "n_flips": len(flips),
        "flip_combinations": flips,
        "combinations": combos,
    }
    lines.append("### Energy-rate sensitivity: each tier's rate independently x0.2, x1, x5\n")
    lines.append(f"- {len(combos)} rate combinations evaluated.")
    lines.append(f"- EcoLogic energy savings vs always-frontier ranges "
                 f"**{min(sav):.1f}% to {max(sav):.1f}%**.")
    lines.append(f"- Combinations where the savings conclusion flips (savings <= 0): "
                 f"**{len(flips)}**.")
    lines.append("")
    lines.append("| Tier1 x | Tier2 x | Tier3 x | EcoLogic J | Frontier J | Savings % | Oracle gap % |")
    lines.append("|---|---|---|---|---|---|---|")
    worst = sorted(combos, key=lambda c: c["savings_vs_frontier_pct"])[:5]
    best = sorted(combos, key=lambda c: -c["savings_vs_frontier_pct"])[:3]
    for c in worst + best:
        m = c["mult"]
        lines.append(f"| {m[0]:g} | {m[1]:g} | {m[2]:g} | {c['eco_J']:,.0f} | "
                     f"{c['frontier_J']:,.0f} | {c['savings_vs_frontier_pct']:.1f} | "
                     f"{c['oracle_gap_pct']:.1f} |")
    lines.append("")
    lines.append("(5 least-favourable combinations then 3 most-favourable.)\n")

    # ---- router quality: how often did the router pick a tier that was right
    route_raw = {i: routing[i]["raw"]["tier"] for i in items}
    route_wrapped = {i: routing[i]["wrapped"]["tier"] for i in items}
    agree = sum(1 for i in items if route_raw[i] == route_wrapped[i])
    out["router"] = {
        "raw_distribution": {str(t): sum(1 for i in items if route_raw[i] == t) for t in TIERS},
        "wrapped_distribution": {str(t): sum(1 for i in items if route_wrapped[i] == t) for t in TIERS},
        "raw_vs_wrapped_agreement": agree / len(items),
    }
    e_wrapped = sum(energy(tokens, route_wrapped[i], i, PAPER_RATES) for i in items)
    k_wrapped = sum(correct[(route_wrapped[i], i)] for i in items)
    out["router"]["wrapped_policy"] = {
        "accuracy": k_wrapped / len(items), "energy_J": e_wrapped
    }

    # how often EcoLogic's chosen tier was wrong but a cheaper/equal tier was right
    recoverable = sum(
        1 for i in items
        if not correct[(route_raw[i], i)] and any(correct[(t, i)] for t in TIERS)
    )
    out["router"]["misroutes_recoverable"] = recoverable
    lines.append("### Router behaviour\n")
    lines.append(f"- Raw-query tier mix: {out['router']['raw_distribution']}")
    lines.append(f"- Wrapped-prompt tier mix: {out['router']['wrapped_distribution']}")
    lines.append(f"- Raw vs wrapped routing agreement: {agree / len(items):.1%}")
    lines.append(f"- Items EcoLogic got wrong where some other tier was right: "
                 f"{recoverable}/{len(items)} ({recoverable / len(items):.1%})")
    lines.append("")

    with open(RAW / "analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(RAW / "tables.md", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {RAW / 'analysis.json'} and {RAW / 'tables.md'}")
    print(f"total measured API cost: ${out['total_cost_usd']}")


if __name__ == "__main__":
    main()

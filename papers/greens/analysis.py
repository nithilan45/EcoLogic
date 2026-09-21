#!/usr/bin/env python3
"""Reconstruct GREENS Paper B energy tables from committed artifacts.

Reads (never writes) frozen trees:
  raw_results/graded.jsonl, raw_results/routing.json, raw_results/tables_cost.md
Writes only under papers/greens/artifacts/.

Energy is tokens/1000 * assumed J-per-1k rates. No joule was metered.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panel_stats import bootstrap_t2_dominates, holm, leave_one_benchmark, mix_reweight_grid

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "raw_results"
OUT = Path(__file__).resolve().parent / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)

TIERS = (1, 2, 3)
BENCHMARKS = ("humaneval", "mmlu", "gsm8k")
PAPER_RATES = {1: 0.5, 2: 1.5, 3: 60.0}  # J per 1k tokens; assumed, not metered
SUBST_RATES = {1: 1.1, 2: 2.0, 3: 60.0}
MULTS = (0.2, 1.0, 5.0)  # ±5× around paper rates
Z = 1.959963984540054
RANDOM_SEED = 20260905
ROUTING_KEY = "raw"


def wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denom
    half = Z / denom * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (p, max(0.0, center - half), min(1.0, center + half))


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    b_only = sum(1 for x, y in zip(a, b) if x and not y)
    c_only = sum(1 for x, y in zip(a, b) if y and not x)
    n = b_only + c_only
    if n == 0:
        return {"eco_only": 0, "other_only": 0, "p_value": 1.0}
    p = float(binomtest(b_only, n, 0.5, alternative="two-sided").pvalue)
    return {"eco_only": b_only, "other_only": c_only, "p_value": p}


def energy_j(tokens: int, rate: float) -> float:
    return tokens / 1000.0 * rate


def load() -> tuple[dict, dict, dict, dict, list[str], dict]:
    correct: dict[tuple[int, str], bool] = {}
    tokens: dict[tuple[int, str], int] = {}
    usd: dict[tuple[int, str], float] = {}
    bench_of: dict[str, str] = {}
    with open(RAW / "graded.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("correct") is None or r.get("total_tokens") is None or r.get("usd") is None:
                continue
            key = (int(r["tier"]), r["item_id"])
            correct[key] = bool(r["correct"])
            tokens[key] = int(r["total_tokens"])
            usd[key] = float(r["usd"])
            bench_of[r["item_id"]] = r["benchmark"]

    item_ids = sorted({i for (_, i) in correct}, key=lambda x: (bench_of.get(x, ""), x))
    complete = [
        i
        for i in item_ids
        if all((t, i) in correct and (t, i) in tokens and (t, i) in usd for t in TIERS)
    ]
    routing = json.loads((RAW / "routing.json").read_text())
    return correct, tokens, usd, bench_of, complete, routing


def assignments(items, tokens, correct, routing, rates):
    eco = {i: routing[i][ROUTING_KEY]["tier"] for i in items}
    frontier = {i: 3 for i in items}
    always_t1 = {i: 1 for i in items}
    always_t2 = {i: 2 for i in items}
    rng = random.Random(RANDOM_SEED)
    rand = {i: rng.choice(TIERS) for i in items}
    oracle = {}
    for i in items:
        cands = [(energy_j(tokens[(t, i)], rates[t]), t) for t in TIERS if correct[(t, i)]]
        if cands:
            oracle[i] = min(cands)[1]
        else:
            oracle[i] = min((energy_j(tokens[(t, i)], rates[t]), t) for t in TIERS)[1]
    return {
        "ecologic": eco,
        "always_t1": always_t1,
        "always_t2": always_t2,
        "frontier": frontier,
        "random": rand,
        "oracle": oracle,
    }


def evaluate(assign, items, correct, tokens, usd, bench_of, rates):
    outcomes = [correct[(assign[i], i)] for i in items]
    k = sum(outcomes)
    n = len(items)
    p, lo, hi = wilson(k, n)
    e = sum(energy_j(tokens[(assign[i], i)], rates[assign[i]]) for i in items)
    tok = sum(tokens[(assign[i], i)] for i in items)
    dollars = sum(usd[(assign[i], i)] for i in items)
    mix = defaultdict(int)
    for i in items:
        mix[assign[i]] += 1
    per_bench = {}
    for b in BENCHMARKS:
        sub = [i for i in items if bench_of[i] == b]
        kk = sum(correct[(assign[i], i)] for i in sub)
        ee = sum(energy_j(tokens[(assign[i], i)], rates[assign[i]]) for i in sub)
        tt = sum(tokens[(assign[i], i)] for i in sub)
        dd = sum(usd[(assign[i], i)] for i in sub)
        per_bench[b] = {
            "n": len(sub),
            "correct": kk,
            "acc": kk / len(sub) if sub else 0.0,
            "energy_J": ee,
            "tokens": tt,
            "usd": dd,
        }
    energy_by_tier = {}
    for t in TIERS:
        sub = [i for i in items if assign[i] == t]
        energy_by_tier[str(t)] = {
            "n": len(sub),
            "energy_J": sum(energy_j(tokens[(t, i)], rates[t]) for i in sub),
            "tokens": sum(tokens[(t, i)] for i in sub),
        }
    return {
        "accuracy": p,
        "ci": [lo, hi],
        "correct": k,
        "n": n,
        "energy_J": e,
        "tokens": tok,
        "usd": dollars,
        "tier_mix": {str(t): mix[t] for t in TIERS},
        "per_benchmark": per_bench,
        "energy_by_tier": energy_by_tier,
        "outcomes": outcomes,
    }


def round_j(x: float) -> float:
    """Publication rounding used in EVALUATION.md / results_report.md."""
    return round(x)


def main() -> int:
    correct, tokens, usd, bench_of, items, routing = load()
    n = len(items)
    assert n == 364, n

    assign = assignments(items, tokens, correct, routing, PAPER_RATES)
    policies = {
        name: evaluate(m, items, correct, tokens, usd, bench_of, PAPER_RATES)
        for name, m in assign.items()
    }
    subst = {
        name: evaluate(m, items, correct, tokens, usd, bench_of, SUBST_RATES)
        for name, m in assign.items()
    }

    eco = policies["ecologic"]
    t2 = policies["always_t2"]
    t1 = policies["always_t1"]
    fro = policies["frontier"]
    orc = policies["oracle"]

    wrap_assign = {i: routing[i]["wrapped"]["tier"] for i in items}
    wrap_pol = evaluate(wrap_assign, items, correct, tokens, usd, bench_of, PAPER_RATES)
    wrap_agree = sum(1 for i in items if assign["ecologic"][i] == wrap_assign[i]) / n

    # Sanity against committed headline rounding (EVALUATION.md)
    assert abs(eco["energy_J"] - 688.953) < 1e-6, eco["energy_J"]
    assert abs(t2["energy_J"] - 393.8595) < 1e-6, t2["energy_J"]
    assert abs(t1["energy_J"] - 677.093) < 1e-6, t1["energy_J"]
    assert abs(fro["energy_J"] - 6065.88) < 1e-6, fro["energy_J"]
    assert abs(orc["energy_J"] - 385.5215) < 1e-6, orc["energy_J"]
    assert abs(wrap_agree - 0.5302197802197802) < 1e-9, wrap_agree
    assert abs(wrap_pol["energy_J"] - 3419.1) < 0.6, wrap_pol["energy_J"]

    eco_ok = {i: correct[(assign["ecologic"][i], i)] for i in items}
    t2_ok = {i: correct[(2, i)] for i in items}
    eco_j_i = {
        i: energy_j(tokens[(assign["ecologic"][i], i)], PAPER_RATES[assign["ecologic"][i]]) for i in items
    }
    t2_j_i = {i: energy_j(tokens[(2, i)], PAPER_RATES[2]) for i in items}
    per_j = {
        i: {
            "ecologic": eco_j_i[i],
            "always_t2": t2_j_i[i],
            "frontier": energy_j(tokens[(3, i)], PAPER_RATES[3]),
        }
        for i in items
    }
    per_tok = {
        i: {
            "ecologic": tokens[(assign["ecologic"][i], i)],
            "always_t2": tokens[(2, i)],
            "frontier": tokens[(3, i)],
        }
        for i in items
    }
    boot_j = bootstrap_t2_dominates(items, eco_ok, t2_ok, eco_j_i, t2_j_i)
    lob_j = leave_one_benchmark(items, bench_of, eco_ok, t2_ok, eco_j_i, t2_j_i)
    mix_j = mix_reweight_grid(items, bench_of, per_j, ["ecologic", "always_t2", "frontier"], step=0.2)
    mix_tok = mix_reweight_grid(items, bench_of, per_tok, ["ecologic", "always_t2", "frontier"], step=0.2)
    n_mix_t2_less_j = sum(1 for r in mix_j if r["t2_cheaper_than_eco"])
    n_mix_t2_less_tok = sum(1 for r in mix_tok if r["t2_cheaper_than_eco"])
    n_lob_dom = sum(1 for r in lob_j if r["t2_dominates_acc_and_cost"])
    j_per_correct = {
        "ecologic": eco["energy_J"] / eco["correct"],
        "always_t2": t2["energy_J"] / t2["correct"],
        "frontier": fro["energy_J"] / fro["correct"],
    }

    mc_t2 = mcnemar(eco["outcomes"], t2["outcomes"])
    mc_t1 = mcnemar(eco["outcomes"], t1["outcomes"])
    mc_fro = mcnemar(eco["outcomes"], fro["outcomes"])
    mc_orc = mcnemar(eco["outcomes"], orc["outcomes"])
    holm_p = holm(
        [
            ("always_t1", mc_t1["p_value"]),
            ("always_t2", mc_t2["p_value"]),
            ("frontier", mc_fro["p_value"]),
            ("oracle", mc_orc["p_value"]),
        ]
    )

    j_save_vs_frontier = (1 - eco["energy_J"] / fro["energy_J"]) * 100
    usd_save_vs_frontier = (1 - eco["usd"] / fro["usd"]) * 100
    residual_j = eco["energy_J"] / fro["energy_J"]
    residual_usd = eco["usd"] / fro["usd"]
    residual_ratio = residual_usd / residual_j  # ~4× reporting gap

    eco_over_t2 = eco["energy_J"] / t2["energy_J"]
    t2_over_eco = t2["energy_J"] / eco["energy_J"]
    acc_gap_pp = (t2["accuracy"] - eco["accuracy"]) * 100
    token_inflation = eco["tokens"] / fro["tokens"]

    # 27-cell ±5× factorial on the same token vectors
    combos = []
    for m1, m2, m3 in product(MULTS, repeat=3):
        rates = {1: PAPER_RATES[1] * m1, 2: PAPER_RATES[2] * m2, 3: PAPER_RATES[3] * m3}
        # oracle depends on rates; rebuild under this rate vector
        asg = assignments(items, tokens, correct, routing, rates)
        pol = {
            name: evaluate(m, items, correct, tokens, usd, bench_of, rates)
            for name, m in asg.items()
        }
        save_f = (1 - pol["ecologic"]["energy_J"] / pol["frontier"]["energy_J"]) * 100
        save_t2 = (1 - pol["ecologic"]["energy_J"] / pol["always_t2"]["energy_J"]) * 100
        combos.append(
            {
                "m1": m1,
                "m2": m2,
                "m3": m3,
                "eco_J": pol["ecologic"]["energy_J"],
                "t1_J": pol["always_t1"]["energy_J"],
                "t2_J": pol["always_t2"]["energy_J"],
                "frontier_J": pol["frontier"]["energy_J"],
                "oracle_J": pol["oracle"]["energy_J"],
                "savings_vs_frontier_pct": save_f,
                "savings_vs_t2_pct": save_t2,
                "eco_gt_t2": pol["ecologic"]["energy_J"] > pol["always_t2"]["energy_J"],
                "eco_gt_frontier": pol["ecologic"]["energy_J"] > pol["frontier"]["energy_J"],
            }
        )

    saves_f = [c["savings_vs_frontier_pct"] for c in combos]
    n_flip_frontier = sum(1 for c in combos if c["eco_gt_frontier"])
    n_eco_worse_t2 = sum(1 for c in combos if c["eco_gt_t2"])
    n_eco_better_t2 = 27 - n_eco_worse_t2
    t2_saves = [c["savings_vs_t2_pct"] for c in combos]

    # One-at-a-time tornado at paper rates (other multipliers = 1)
    tornado = []
    for which, label in ((1, "Tier 1 rate"), (2, "Tier 2 rate"), (3, "Tier 3 rate")):
        for m in (0.2, 5.0):
            rates = dict(PAPER_RATES)
            rates[which] *= m
            asg = assignments(items, tokens, correct, routing, rates)
            pol = {
                name: evaluate(mm, items, correct, tokens, usd, bench_of, rates)
                for name, mm in asg.items()
            }
            tornado.append(
                {
                    "factor": label,
                    "mult": m,
                    "eco_J": pol["ecologic"]["energy_J"],
                    "t2_J": pol["always_t2"]["energy_J"],
                    "frontier_J": pol["frontier"]["energy_J"],
                    "savings_vs_frontier_pct": (1 - pol["ecologic"]["energy_J"] / pol["frontier"]["energy_J"])
                    * 100,
                    "savings_vs_t2_pct": (1 - pol["ecologic"]["energy_J"] / pol["always_t2"]["energy_J"])
                    * 100,
                }
            )

    headline = {
        "n": n,
        "energy_model": "tokens/1000 * assumed J/1k; NOT metered",
        "paper_rates_J_per_1k": PAPER_RATES,
        "policies_rounded": {
            "ecologic": {
                "acc_pct": round(eco["accuracy"] * 100, 1),
                "energy_J": round_j(eco["energy_J"]),
                "tokens": eco["tokens"],
                "usd": round(eco["usd"], 4),
                "mix": eco["tier_mix"],
            },
            "always_t2": {
                "acc_pct": round(t2["accuracy"] * 100, 1),
                "energy_J": round_j(t2["energy_J"]),
                "tokens": t2["tokens"],
                "usd": round(t2["usd"], 4),
            },
            "always_t1": {
                "acc_pct": round(t1["accuracy"] * 100, 1),
                "energy_J": round_j(t1["energy_J"]),
                "tokens": t1["tokens"],
                "usd": round(t1["usd"], 4),
            },
            "frontier": {
                "acc_pct": round(fro["accuracy"] * 100, 1),
                "energy_J": round_j(fro["energy_J"]),
                "tokens": fro["tokens"],
                "usd": round(fro["usd"], 4),
            },
            "oracle": {
                "acc_pct": round(orc["accuracy"] * 100, 1),
                "energy_J": round_j(orc["energy_J"]),
                "tokens": orc["tokens"],
                "usd": round(orc["usd"], 4),
            },
        },
        "always_t2_acc_advantage_pp": round(acc_gap_pp, 1),
        "eco_energy_over_always_t2": round(eco_over_t2, 4),
        "always_t2_energy_over_eco": round(t2_over_eco, 4),
        "mcnemar_eco_vs_always_t2": mc_t2,
        "mcnemar_eco_vs_always_t1": mc_t1,
        "mcnemar_eco_vs_frontier": mc_fro,
        "mcnemar_eco_vs_oracle": mc_orc,
        "joules_vs_dollars_vs_frontier": {
            "modelled_J_saving_pct": round(j_save_vs_frontier, 1),
            "measured_usd_saving_pct": round(usd_save_vs_frontier, 1),
            "residual_cost_J_x": round(residual_j, 3),
            "residual_cost_usd_x": round(residual_usd, 3),
            "residual_ratio_usd_over_J": round(residual_ratio, 2),
        },
        "token_inflation_eco_over_frontier": round(token_inflation, 2),
        "sensitivity_27": {
            "n_combinations": 27,
            "savings_vs_frontier_min_pct": min(saves_f),
            "savings_vs_frontier_max_pct": max(saves_f),
            "n_sign_flips_vs_frontier": n_flip_frontier,
            "n_eco_uses_more_energy_than_always_t2": n_eco_worse_t2,
            "n_eco_uses_less_energy_than_always_t2": n_eco_better_t2,
            "savings_vs_t2_min_pct": min(t2_saves),
            "savings_vs_t2_max_pct": max(t2_saves),
        },
        "substitute_rates_energy_J": {k: subst[k]["energy_J"] for k in subst},
        "wrapped": {
            "agreement_raw": wrap_agree,
            "energy_J": wrap_pol["energy_J"],
            "energy_J_rounded": round_j(wrap_pol["energy_J"]),
            "acc_pct": round(wrap_pol["accuracy"] * 100, 1),
            "usd": round(wrap_pol["usd"], 4),
            "mix": wrap_pol["tier_mix"],
        },
        "robustness": {
            "bootstrap_frac_t2_dominates": boot_j["frac_t2_dominates_quality_and_cost"],
            "n_lob": len(lob_j),
            "n_lob_t2_dominates": n_lob_dom,
            "mix_n_cells": len(mix_j),
            "mix_n_t2_less_energy": n_mix_t2_less_j,
            "mix_n_t2_less_tokens": n_mix_t2_less_tok,
            "j_per_correct": j_per_correct,
            "holm": holm_p,
            "token_ratio_eco_over_t2": eco["tokens"] / t2["tokens"],
        },
    }

    # Strip outcomes before writing policy dump
    policies_out = {}
    for name, rec in policies.items():
        rec = dict(rec)
        rec.pop("outcomes", None)
        policies_out[name] = rec

    payload = {
        "headline": headline,
        "policies_paper_rates": policies_out,
        "sensitivity_combinations": combos,
        "tornado": tornado,
    }
    (OUT / "energy_tables.json").write_text(json.dumps(payload, indent=2))

    # CSVs for the paper tables
    with open(OUT / "policy_energy.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "accuracy",
                "ci_lo",
                "ci_hi",
                "correct",
                "n",
                "energy_J",
                "tokens",
                "usd",
                "vs_frontier_J",
                "t1",
                "t2",
                "t3",
            ],
        )
        w.writeheader()
        order = [
            ("ecologic", "EcoLogic"),
            ("always_t2", "Always-T2"),
            ("always_t1", "Always-T1"),
            ("frontier", "Always-frontier"),
            ("random", "Random"),
            ("oracle", "Oracle"),
        ]
        for key, label in order:
            s = policies[key]
            w.writerow(
                {
                    "policy": label,
                    "accuracy": s["accuracy"],
                    "ci_lo": s["ci"][0],
                    "ci_hi": s["ci"][1],
                    "correct": s["correct"],
                    "n": s["n"],
                    "energy_J": s["energy_J"],
                    "tokens": s["tokens"],
                    "usd": s["usd"],
                    "vs_frontier_J": s["energy_J"] / fro["energy_J"],
                    "t1": s["tier_mix"]["1"],
                    "t2": s["tier_mix"]["2"],
                    "t3": s["tier_mix"]["3"],
                }
            )

    with open(OUT / "per_benchmark_energy.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["policy", "benchmark", "n", "correct", "acc", "energy_J", "tokens", "usd"],
        )
        w.writeheader()
        for key, label in order:
            for b in BENCHMARKS:
                pb = policies[key]["per_benchmark"][b]
                w.writerow({"policy": label, "benchmark": b, **pb})

    with open(OUT / "sensitivity_27.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(combos[0].keys()))
        w.writeheader()
        w.writerows(combos)

    with open(OUT / "mix_reweight_energy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mix_j[0].keys()))
        w.writeheader()
        w.writerows(mix_j)
    with open(OUT / "leave_one_benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lob_j[0].keys()))
        w.writeheader()
        w.writerows(lob_j)

    _figures(policies, combos, tornado, headline)
    _write_numbers_snippet(headline, policies)

    print("Wrote", OUT)
    print(json.dumps(headline["policies_rounded"], indent=2))
    print("J vs USD:", headline["joules_vs_dollars_vs_frontier"])
    print("Sensitivity:", headline["sensitivity_27"])
    print("McNemar vs T2:", mc_t2)
    return 0


def _figures(policies, combos, tornado, headline) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )

    # --- Fig 1: energy–accuracy scatter ---
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    spec = [
        ("ecologic", "EcoLogic", "C3", "D", 90),
        ("always_t2", "Always-T2", "C2", "o", 80),
        ("always_t1", "Always-T1", "C0", "s", 70),
        ("frontier", "Always-frontier", "0.35", "^", 80),
        ("random", "Random", "C1", "P", 70),
        ("oracle", "Oracle", "C4", "*", 110),
    ]
    for key, label, color, marker, size in spec:
        s = policies[key]
        ax.scatter(
            s["energy_J"],
            s["accuracy"] * 100,
            c=color,
            marker=marker,
            s=size,
            zorder=3,
            edgecolors="k",
            linewidths=0.4,
            label=label,
        )
        ax.errorbar(
            s["energy_J"],
            s["accuracy"] * 100,
            yerr=[
                [(s["accuracy"] - s["ci"][0]) * 100],
                [(s["ci"][1] - s["accuracy"]) * 100],
            ],
            fmt="none",
            ecolor=color,
            elinewidth=0.8,
            capsize=2,
            zorder=2,
        )
    ax.set_xlabel("Modelled energy (J)  —  tokens × assumed J/1k; not metered")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xscale("log")
    ax.set_xlim(300, 9000)
    ax.set_ylim(84, 98)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.7)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_title("Static Always-T2 Pareto-dominates the advertised router")
    fig.savefig(OUT / "fig_energy_accuracy.pdf")
    fig.savefig(OUT / "fig_energy_accuracy.png")
    plt.close(fig)

    # --- Fig 2: sensitivity band + tornado-ish ranking vs T2 ---
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35))

    ax = axes[0]
    xs = list(range(len(combos)))
    ys_f = [c["savings_vs_frontier_pct"] for c in combos]
    ys_t = [c["savings_vs_t2_pct"] for c in combos]
    order_idx = sorted(range(len(combos)), key=lambda i: ys_f[i])
    ax.plot(
        xs,
        [ys_f[i] for i in order_idx],
        color="0.4",
        lw=1.2,
        label="vs always-frontier",
    )
    ax.plot(
        xs,
        [ys_t[i] for i in order_idx],
        color="C3",
        lw=1.4,
        label="vs Always-T2",
    )
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.fill_between(
        xs,
        [ys_f[i] for i in order_idx],
        0,
        where=[ys_f[i] < 0 for i in order_idx],
        color="0.75",
        alpha=0.5,
        interpolate=True,
    )
    ax.set_xlabel("Rate combinations (sorted by vs-frontier saving)")
    ax.set_ylabel("Modelled energy saving (%)")
    ax.set_xlim(0, 26)
    ax.legend(loc="lower right")
    ax.set_title("±5× factorial (27 cells)")
    ax.grid(True, ls=":", lw=0.4, alpha=0.7)

    ax = axes[1]
    # tornado: one-at-a-time effect on EcoLogic−T2 energy gap (%)
    labels = ["Tier 1 rate", "Tier 2 rate", "Tier 3 rate"]
    paper_gap = (1 - policies["ecologic"]["energy_J"] / policies["always_t2"]["energy_J"]) * 100
    lows, highs = [], []
    for lab in labels:
        rows = [t for t in tornado if t["factor"] == lab]
        by_m = {t["mult"]: t["savings_vs_t2_pct"] for t in rows}
        lows.append(by_m[0.2])
        highs.append(by_m[5.0])
    y = range(len(labels))[::-1]
    ax.axvline(paper_gap, color="C3", lw=1.0, ls="--", label="paper rates")
    ax.axvline(0, color="k", lw=0.7)
    for yi, lo, hi, lab in zip(y, lows, highs, labels):
        ax.plot([lo, hi], [yi, yi], color="0.25", lw=4, solid_capstyle="butt")
        ax.scatter([lo, hi], [yi, yi], c=["C0", "C1"], s=28, zorder=3, edgecolors="k", linewidths=0.3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("EcoLogic saving vs Always-T2 (%)")
    ax.set_title("One-at-a-time ×0.2 / ×5")
    ax.grid(True, axis="x", ls=":", lw=0.4, alpha=0.7)
    # legend proxies
    ax.scatter([], [], c="C0", s=28, edgecolors="k", linewidths=0.3, label="×0.2")
    ax.scatter([], [], c="C1", s=28, edgecolors="k", linewidths=0.3, label="×5")
    ax.legend(loc="lower left", fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT / "fig_sensitivity.pdf")
    fig.savefig(OUT / "fig_sensitivity.png")
    plt.close(fig)

    # --- Fig 3: per-benchmark stacked energy ---
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    names = ["EcoLogic", "Always-T2", "Always-T1", "Always-frontier"]
    keys = ["ecologic", "always_t2", "always_t1", "frontier"]
    benches = list(BENCHMARKS)
    colors = {"humaneval": "#4C78A8", "mmlu": "#F58518", "gsm8k": "#54A24B"}
    bottoms = [0.0] * len(keys)
    x = range(len(keys))
    for b in benches:
        heights = [policies[k]["per_benchmark"][b]["energy_J"] for k in keys]
        ax.bar(
            list(x),
            heights,
            bottom=bottoms,
            color=colors[b],
            edgecolor="k",
            linewidth=0.3,
            label=b,
        )
        bottoms = [bo + h for bo, h in zip(bottoms, heights)]
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Modelled energy (J)")
    ax.set_title("Per-benchmark energy mix (paper rates)")
    ax.legend(title=None, fontsize=8)
    ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.7)
    fig.savefig(OUT / "fig_per_benchmark.pdf")
    fig.savefig(OUT / "fig_per_benchmark.png")
    plt.close(fig)


def _write_numbers_snippet(headline, policies) -> None:
    lines = [
        "# Auto-extracted headline (see NUMBERS.md for provenance)",
        json.dumps(headline, indent=2),
        "",
        "# Per-benchmark EcoLogic energy",
        json.dumps(policies["ecologic"]["per_benchmark"], indent=2),
        "",
        "# EcoLogic energy by chosen tier",
        json.dumps(policies["ecologic"]["energy_by_tier"], indent=2),
    ]
    (OUT / "headline_dump.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Reconstruct ICAART Paper A tables from committed artifacts. Offline only."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panel_stats import (
    bootstrap_t2_dominates,
    cohens_g,
    holm,
    leave_one_benchmark,
    mix_reweight_grid,
    oracle_confusion,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)

Z95 = 1.959963984540054
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    CHECKS.append((name, bool(ok), detail))
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")


def dump_json(name: str, obj) -> None:
    path = OUT / name
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


def dump_csv(name: str, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path = OUT / name
    if not rows:
        path.write_text("")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path.relative_to(ROOT)}")


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / den
    half = (z / den) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def binom_pmf(n: int, k: int) -> float:
    return math.comb(n, k) * (0.5 ** n)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided McNemar exact (binomial n=b+c, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(binom_pmf(n, i) for i in range(0, k + 1))
    p = min(1.0, 2.0 * tail)
    return p


def sign_test_two_sided(n_pos: int, n: int) -> float:
    if n == 0:
        return 1.0
    k = min(n_pos, n - n_pos)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def load_json(rel: str):
    path = ROOT / rel
    with path.open() as f:
        return json.load(f), str(path)


def load_graded() -> dict[tuple[str, int], dict]:
    out = {}
    with (ROOT / "raw_results/graded.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["item_id"], int(rec["tier"]))
            # first occurrence per (item, tier); repeats exist for determinism
            if rec.get("repeat_index", 0) != 0:
                continue
            out[key] = rec
    return out


def load_routing_raw() -> dict[str, int]:
    data = json.loads((ROOT / "raw_results/routing.json").read_text())
    return {iid: int(v["raw"]["tier"]) for iid, v in data.items()}


def reconstruct_stage12():
    graded = load_graded()
    routing = load_routing_raw()
    stored, src = load_json("raw_results/analysis_cost.json")

    items = sorted({iid for iid, _t in graded})
    n = len(items)
    check("stage12_n", n == 364, f"n={n}")

    by_item = defaultdict(dict)
    for (iid, tier), rec in graded.items():
        by_item[iid][tier] = rec

    missing = [iid for iid in items if set(by_item[iid].keys()) != {1, 2, 3}]
    check("stage12_complete_tiers", len(missing) == 0, f"incomplete={len(missing)}")

    def policy_from_choice(choice_fn):
        correct = 0
        cost = 0.0
        mix = {1: 0, 2: 0, 3: 0}
        flags = []
        for iid in items:
            t = choice_fn(iid, by_item[iid])
            rec = by_item[iid][t]
            mix[t] += 1
            cost += float(rec["usd"])
            ok = bool(rec.get("correct"))
            correct += int(ok)
            flags.append(ok)
        return {
            "correct": correct,
            "n": n,
            "accuracy": correct / n,
            "cost_usd": cost,
            "mix": mix,
            "flags": flags,
        }

    eco = policy_from_choice(lambda iid, _c: routing[iid])
    t1 = policy_from_choice(lambda iid, _c: 1)
    t2 = policy_from_choice(lambda iid, _c: 2)
    t3 = policy_from_choice(lambda iid, _c: 3)

    def oracle_choice(iid, cells):
        correct_tiers = [t for t in (1, 2, 3) if cells[t].get("correct")]
        pool = correct_tiers or [1, 2, 3]
        return min(pool, key=lambda t: float(cells[t]["usd"]))

    ora = policy_from_choice(oracle_choice)

    def mcnemar_pair(a_flags, b_flags):
        b = c = 0
        for xa, xb in zip(a_flags, b_flags):
            if xa and not xb:
                b += 1
            elif xb and not xa:
                c += 1
        return b, c, mcnemar_exact(b, c)

    reconstructed = {
        "ecologic": eco,
        "always_t1": t1,
        "always_t2": t2,
        "frontier": t3,
        "oracle": ora,
    }

    rows = []
    for key, label in [
        ("ecologic", "EcoLogic routing (raw query)"),
        ("always_t1", "Always Tier 1"),
        ("always_t2", "Always Tier 2"),
        ("frontier", "Always-frontier (gpt-4o)"),
        ("oracle", "Oracle (cheapest correct)"),
    ]:
        pol = reconstructed[key]
        p, lo, hi = wilson(pol["correct"], pol["n"])
        stored_pol = stored["policies"][key]
        cost_ok = abs(pol["cost_usd"] - stored_pol["cost_usd"]) < 1e-8
        acc_ok = pol["correct"] == stored_pol["correct"]
        check(f"stage12_{key}_usd", cost_ok, f"{pol['cost_usd']:.8f} vs stored {stored_pol['cost_usd']:.8f}")
        check(f"stage12_{key}_correct", acc_ok, f"{pol['correct']} vs stored {stored_pol['correct']}")
        mcn = None
        if key != "ecologic":
            b, c, pval = mcnemar_pair(eco["flags"], pol["flags"])
            stored_m = stored["mcnemar_vs_ecologic"][key if key != "frontier" else "frontier"]
            # analysis_cost uses keys always_t1, always_t2, frontier, oracle, random
            sm_key = {"always_t1": "always_t1", "always_t2": "always_t2", "frontier": "frontier", "oracle": "oracle"}[key]
            sm = stored["mcnemar_vs_ecologic"][sm_key]
            check(
                f"stage12_mcnemar_{key}",
                abs(pval - sm["p_value"]) < 1e-9 and b == sm["b"] and c == sm["c"],
                f"b={b} c={c} p={pval:.10g} vs stored p={sm['p_value']:.10g}",
            )
            mcn = {"b": b, "c": c, "p_exact": pval}
        rows.append(
            {
                "policy": label,
                "key": key,
                "n": pol["n"],
                "correct": pol["correct"],
                "accuracy": p,
                "wilson_lo": lo,
                "wilson_hi": hi,
                "cost_usd": pol["cost_usd"],
                "usd_per_item": pol["cost_usd"] / pol["n"],
                "vs_frontier": pol["cost_usd"] / reconstructed["frontier"]["cost_usd"],
                "mix_t1": pol["mix"][1],
                "mix_t2": pol["mix"][2],
                "mix_t3": pol["mix"][3],
                "mcnemar_b": None if mcn is None else mcn["b"],
                "mcnemar_c": None if mcn is None else mcn["c"],
                "mcnemar_p": None if mcn is None else mcn["p_exact"],
                "in_sample_keyword": key == "ecologic",
            }
        )

    # Random policy is a stored seeded assignment; do not re-roll.
    rnd = stored["policies"]["random"]
    p, lo, hi = wilson(rnd["correct"], rnd["n"])
    sm = stored["mcnemar_vs_ecologic"]["random"]
    rows.insert(
        3,
        {
            "policy": "Random tier (stored seed)",
            "key": "random",
            "n": rnd["n"],
            "correct": rnd["correct"],
            "accuracy": rnd["accuracy"],
            "wilson_lo": lo,
            "wilson_hi": hi,
            "cost_usd": rnd["cost_usd"],
            "usd_per_item": rnd["cost_usd_per_item"],
            "vs_frontier": rnd["cost_vs_frontier"],
            "mix_t1": rnd["tier_mix"]["1"],
            "mix_t2": rnd["tier_mix"]["2"],
            "mix_t3": rnd["tier_mix"]["3"],
            "mcnemar_b": sm["b"],
            "mcnemar_c": sm["c"],
            "mcnemar_p": sm["p_value"],
            "in_sample_keyword": False,
        },
    )

    ratio = eco["cost_usd"] / t2["cost_usd"]
    gap_pp = (t2["accuracy"] - eco["accuracy"]) * 100
    check("stage12_ratio", abs(ratio - stored["cost_ratio_ecologic_over_t2"]) < 1e-9, f"ratio={ratio}")
    dump_csv("stage12_usd.csv", rows)
    summary = {
        "source_reconstructed": "raw_results/graded.jsonl + raw_results/routing.json (raw)",
        "source_stored_checksum": src,
        "n": n,
        "ecologic_acc_pct": round(eco["accuracy"] * 100, 1),
        "ecologic_correct": eco["correct"],
        "ecologic_usd": eco["cost_usd"],
        "always_t2_acc_pct": round(t2["accuracy"] * 100, 1),
        "always_t2_correct": t2["correct"],
        "always_t2_usd": t2["cost_usd"],
        "mcnemar_t2_p": next(r["mcnemar_p"] for r in rows if r["key"] == "always_t2"),
        "mcnemar_t2_b": next(r["mcnemar_b"] for r in rows if r["key"] == "always_t2"),
        "mcnemar_t2_c": next(r["mcnemar_c"] for r in rows if r["key"] == "always_t2"),
        "cost_ratio_eco_over_t2": ratio,
        "accuracy_gap_t2_minus_eco_pp": gap_pp,
        "headline_eco_usd_4dp": round(eco["cost_usd"], 4),
        "headline_t2_usd_4dp": round(t2["cost_usd"], 4),
        "in_sample_caveat": True,
    }
    dump_json("stage12_summary.json", summary)
    _fig_cost_accuracy(rows)
    robust = reconstruct_robustness(items, by_item, routing, eco, t2, ora, stored)
    summary["robustness"] = {k: robust[k] for k in robust if k != "lob_rows"}
    return summary, rows


def reconstruct_robustness(items, by_item, routing, eco, t2, ora, stored):
    bench_of = {}
    eco_ok, t2_ok = {}, {}
    eco_cost, t2_cost = {}, {}
    eco_tier, ora_tier = {}, {}
    per_usd = {}
    for iid in items:
        rec = by_item[iid][1]
        bench_of[iid] = rec.get("benchmark") or rec["item_id"].split("/")[0]
        te = routing[iid]
        eco_tier[iid] = te
        eco_ok[iid] = bool(by_item[iid][te].get("correct"))
        t2_ok[iid] = bool(by_item[iid][2].get("correct"))
        eco_cost[iid] = float(by_item[iid][te]["usd"])
        t2_cost[iid] = float(by_item[iid][2]["usd"])
        ora_t = None
        correct_tiers = [t for t in (1, 2, 3) if by_item[iid][t].get("correct")]
        pool = correct_tiers or [1, 2, 3]
        ora_t = min(pool, key=lambda t: float(by_item[iid][t]["usd"]))
        ora_tier[iid] = ora_t
        per_usd[iid] = {"ecologic": eco_cost[iid], "always_t2": t2_cost[iid], "frontier": float(by_item[iid][3]["usd"])}

    family = [
        ("always_t1", stored["mcnemar_vs_ecologic"]["always_t1"]["p_value"]),
        ("always_t2", stored["mcnemar_vs_ecologic"]["always_t2"]["p_value"]),
        ("frontier", stored["mcnemar_vs_ecologic"]["frontier"]["p_value"]),
        ("random", stored["mcnemar_vs_ecologic"]["random"]["p_value"]),
        ("oracle", stored["mcnemar_vs_ecologic"]["oracle"]["p_value"]),
    ]
    holm_adj = holm(family)
    b, c = stored["mcnemar_vs_ecologic"]["always_t2"]["b"], stored["mcnemar_vs_ecologic"]["always_t2"]["c"]
    boot = bootstrap_t2_dominates(items, eco_ok, t2_ok, eco_cost, t2_cost)
    lob = leave_one_benchmark(items, bench_of, eco_ok, t2_ok, eco_cost, t2_cost)
    mix = mix_reweight_grid(items, bench_of, per_usd, ["ecologic", "always_t2", "frontier"], step=0.2)
    n_mix_t2_cheaper = sum(1 for r in mix if r["t2_cheaper_than_eco"])
    conf = oracle_confusion(items, eco_tier, ora_tier, eco_ok, t2_ok)
    usd_per_correct_eco = eco["cost_usd"] / eco["correct"]
    usd_per_correct_t2 = t2["cost_usd"] / t2["correct"]
    check("boot_t2_dom_ge_95", boot["frac_t2_dominates_quality_and_cost"] >= 0.95, str(boot["frac_t2_dominates_quality_and_cost"]))
    check("lob_all_t2_dom", all(r["t2_dominates_acc_and_cost"] for r in lob), str([r["subset"] for r in lob if not r["t2_dominates_acc_and_cost"]]))
    check("mix_t2_always_cheaper", n_mix_t2_cheaper == len(mix), f"{n_mix_t2_cheaper}/{len(mix)}")
    dump_csv("leave_one_benchmark.csv", lob)
    dump_csv("mix_reweight_usd.csv", mix)
    dump_json(
        "robustness.json",
        {
            "holm": holm_adj,
            "cohens_g_vs_t2": cohens_g(b, c),
            "bootstrap": boot,
            "oracle_confusion": conf,
            "usd_per_correct": {"ecologic": usd_per_correct_eco, "always_t2": usd_per_correct_t2},
            "mix_n_cells": len(mix),
            "mix_n_t2_cheaper": n_mix_t2_cheaper,
            "n_lob": len(lob),
            "n_lob_t2_dominates": sum(1 for r in lob if r["t2_dominates_acc_and_cost"]),
        },
    )
    _fig_lob(lob)
    return {
        "holm": holm_adj,
        "cohens_g_vs_t2": cohens_g(b, c),
        "bootstrap": boot,
        "oracle_confusion": conf,
        "usd_per_correct_eco": usd_per_correct_eco,
        "usd_per_correct_t2": usd_per_correct_t2,
        "mix_n_cells": len(mix),
        "mix_n_t2_cheaper": n_mix_t2_cheaper,
        "n_lob_t2_dominates": sum(1 for r in lob if r["t2_dominates_acc_and_cost"]),
        "n_lob": len(lob),
        "lob_rows": lob,
    }


def _fig_lob(rows: list[dict]) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    show = [r for r in rows if r["subset"].startswith("only_") or r["subset"] == "all"]
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    xs = range(len(show))
    ax.bar([x - 0.18 for x in xs], [r["eco_acc"] * 100 for r in show], 0.35, label="EcoLogic", color="C3")
    ax.bar([x + 0.18 for x in xs], [r["t2_acc"] * 100 for r in show], 0.35, label="Always-T2", color="C2")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r["subset"].replace("only_", "").replace("all", "all three") for r in show], rotation=15)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(75, 100)
    ax.legend(framealpha=0.95)
    ax.grid(True, axis="y", ls=":", lw=0.4)
    fig.savefig(OUT / "fig_lob.pdf")
    plt.close(fig)


def _fig_cost_accuracy(rows: list[dict]) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    spec = {
        "ecologic": ("EcoLogic", "C3", "D", 90),
        "always_t2": ("Always-T2", "C2", "o", 90),
        "always_t1": ("Always-T1", "C0", "s", 70),
        "frontier": ("Always-frontier", "0.35", "^", 80),
        "random": ("Random", "C1", "P", 70),
        "oracle": ("Oracle", "C4", "*", 110),
    }
    for row in rows:
        key = row["key"]
        if key not in spec:
            continue
        label, color, marker, size = spec[key]
        ax.scatter(
            row["cost_usd"],
            row["accuracy"] * 100,
            c=color,
            marker=marker,
            s=size,
            zorder=3,
            edgecolors="k",
            linewidths=0.4,
            label=label,
        )
        ax.errorbar(
            row["cost_usd"],
            row["accuracy"] * 100,
            yerr=[
                [(row["accuracy"] - row["wilson_lo"]) * 100],
                [(row["wilson_hi"] - row["accuracy"]) * 100],
            ],
            fmt="none",
            ecolor=color,
            elinewidth=0.8,
            capsize=2,
            zorder=2,
        )
    ax.set_xlabel("Measured USD (panel total)")
    ax.set_ylabel("Accuracy (%, Wilson 95% CI)")
    ax.set_xlim(0, 0.72)
    ax.set_ylim(82, 99)
    ax.grid(True, ls=":", lw=0.4, alpha=0.7)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=8)
    fig.savefig(OUT / "fig_cost_accuracy.pdf")
    fig.savefig(OUT / "fig_cost_accuracy.png")
    plt.close(fig)


def reconstruct_routellm():
    s9, src = load_json("stage7_10/s9_static_baselines.json")
    audit, asrc = load_json("woais_experiments/results/external/audit/routellm_s9_committed.json")
    sweep = s9["sweep"]
    interior = [r for r in sweep if 0.5 < r["strong_pct"] < 99.5]
    check("routellm_n_interior", len(interior) == s9["n_interior_points"] == 9, f"n_int={len(interior)}")

    edges_cost = [r["router_minus_random_matched_cost_pp"] for r in interior]
    edges_frac = [r["router_minus_random_same_fraction_pp"] for r in interior]
    lost = [r["advantage_lost_to_cost_matching_pp"] for r in interior]
    n_win = sum(1 for e in edges_cost if e > 0)
    n_sig = sum(1 for r in interior if r["mc_p_value"] < 0.05)
    mean_cost = sum(edges_cost) / len(edges_cost)
    mean_frac = sum(edges_frac) / len(edges_frac)
    n_pos = sum(1 for e in edges_cost if e > 0)
    p_sign = sign_test_two_sided(n_pos, len(interior))

    check("routellm_mean_cost_edge", abs(mean_cost - s9["mean_router_edge_matched_cost_pp"]) < 1e-12, f"{mean_cost}")
    check("routellm_n_win", n_win == s9["n_interior_beating_matched_cost_mixture"], f"{n_win}")
    check("routellm_n_sig", n_sig == s9["n_interior_beating_matched_cost_mixture_p05"] == 0, f"{n_sig}")
    check("routellm_sign_p", abs(p_sign - s9["sign_test_across_sweep"]["p_two_sided"]) < 1e-12, f"p={p_sign}")
    check(
        "routellm_lost_range",
        min(x for x in lost if x > 0) >= 0.08 and max(lost) < 0.31,
        f"lost {min(lost):.4f}–{max(lost):.4f}",
    )
    committed = audit["committed"]
    check(
        "routellm_audit_mean",
        abs(committed["mean_edge_matched_cost_pp"] - mean_cost) < 1e-12,
        "audit JSON matches reconstructed mean",
    )

    rows = []
    for r in sweep:
        rows.append(
            {
                "threshold": r["threshold"],
                "strong_pct": r["strong_pct"],
                "router_acc_pct": r["router_accuracy_pct"],
                "router_usd_per_item": r["router_cost_per_item_usd"],
                "frac_match_acc_pct": r["random_same_fraction_accuracy_pct"],
                "edge_frac_pp": r["router_minus_random_same_fraction_pp"],
                "cost_match_acc_pct": r["random_matched_cost_accuracy_pct"],
                "edge_cost_pp": r["router_minus_random_matched_cost_pp"],
                "advantage_lost_pp": r["advantage_lost_to_cost_matching_pp"],
                "mc_p": r["mc_p_value"],
                "interior": 0.5 < r["strong_pct"] < 99.5,
            }
        )
    dump_csv("routellm_gsm8k_sweep.csv", rows)
    summary = {
        "source": src,
        "audit_source": asrc,
        "n_items": s9["n_items"],
        "n_interior": len(interior),
        "mean_edge_call_fraction_pp": mean_frac,
        "mean_edge_cost_matched_pp": mean_cost,
        "mean_edge_cost_matched_pp_2dp": round(mean_cost, 2),
        "n_win_cost_matched": n_win,
        "n_sig_p05": n_sig,
        "sign_test_p": p_sign,
        "sign_test_p_3dp": round(p_sign, 3),
        "lost_min_interior_pp": min(lost),
        "lost_max_interior_pp": max(lost),
        "always_weak_pct": s9["accuracy"]["always_weak_pct"],
        "always_strong_pct": s9["accuracy"]["always_strong_pct"],
        "naive_random_residual_usd": s9["naive_formula_exact_for_random_residual_usd"],
    }
    dump_json("routellm_summary.json", summary)
    power = _routellm_power(interior, mean_cost, s9["n_items"])
    dump_json("routellm_power.json", power)
    summary["power"] = power
    _fig_routellm(rows)
    return summary


def _z_from_two_sided_p(p: float) -> float:
    """|z| such that 2(1-Phi(|z|)) = p."""
    if p <= 0.0:
        return float("inf")
    if p >= 1.0:
        return 0.0
    from statistics import NormalDist

    return float(NormalDist().inv_cdf(1.0 - p / 2.0))


def _routellm_power(interior: list[dict], mean_edge_pp: float, n_items: int) -> dict:
    """Implied MC standard errors and n to detect the mean cost-matched edge at 80% power."""
    ses = []
    for r in interior:
        edge = abs(r["router_minus_random_matched_cost_pp"])
        p = float(r["mc_p_value"])
        z = _z_from_two_sided_p(p)
        if z > 0 and math.isfinite(z) and edge > 0:
            ses.append(edge / z)
    se_med = sorted(ses)[len(ses) // 2] if ses else None
    # two-sided α=0.05, 80% power → z_α/2 + z_β = 1.959964 + 0.841621 = 2.801585
    z_star = 1.959963984540054 + 0.841621233572914
    n80 = None
    if se_med and mean_edge_pp > 0:
        se_needed = mean_edge_pp / z_star
        n80 = int(math.ceil(n_items * (se_med / se_needed) ** 2))
    return {
        "n_items": n_items,
        "mean_cost_matched_edge_pp": mean_edge_pp,
        "n_implied_se": len(ses),
        "median_implied_mc_se_pp": se_med,
        "z_star_80pct_alpha05": z_star,
        "n_items_for_80pct_power_at_mean_edge": n80,
        "note": (
            "SE implied from Monte Carlo two-sided p and |edge| at each interior "
            "threshold; points share items so n80 is a lower bound on independent items."
        ),
    }


def _fig_routellm(rows: list[dict]) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
    interior = [r for r in rows if r["interior"]]
    xs = [r["strong_pct"] for r in interior]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.axhline(0.0, color="k", lw=0.7, ls="--")
    ax.plot(xs, [r["edge_frac_pp"] for r in interior], "o--", color="0.45", lw=1.1, ms=5, label="Call-fraction match")
    ax.plot(xs, [r["edge_cost_pp"] for r in interior], "s-", color="C3", lw=1.4, ms=5.5, label="Cost match (Eq. 1)")
    ax.set_xlabel("Strong-model call fraction (%)")
    ax.set_ylabel("Router edge vs query-independent mixture (pp)")
    ax.set_xlim(5, 95)
    ax.grid(True, ls=":", lw=0.4, alpha=0.7)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.savefig(OUT / "fig_routellm_edge.pdf")
    fig.savefig(OUT / "fig_routellm_edge.png")
    plt.close(fig)


def reconstruct_accounting():
    val, vsrc = load_json("stage7_10/regret_correction_validation.json")
    ext, esrc = load_json("stage7_10/external_generalization.json")
    eco, csrc = load_json("woais_experiments/results/accounting/framework/stage12_ecologic_decomposition.json")

    residual = abs(val["R_naive"] + val["correction"] - val["R_true"])
    check("s8_reconciles", val["reconciles"] and residual < 1e-12, f"residual={val['abs_residual']}")
    sign_flip_eco = (val["R_naive"] < 0) and (val["R_true"] > 0)
    check("s8_sign_flip", sign_flip_eco, f"naive={val['R_naive']} true={val['R_true']}")
    ratio = val["correction"] / abs(val["R_true"])
    check("s8_correction_vs_true", abs(ratio - 1.9) < 0.05, f"correction/|R_true|={ratio:.3f}")

    gsm = ext["gsm8k"]
    check("s9_identity_residual", gsm["max_abs_residual"] < 1e-15, f"{gsm['max_abs_residual']}")
    check("s9_any_sign_flip", gsm["any_sign_flip"] is True, "RouteLLM sign flip present")
    max_mis = gsm["max_abs_cost_misestimate_pct"]
    check("s9_max_misest", abs(max_mis - 6.305405742092525) < 1e-9, f"{max_mis}")
    flip_rows = [r for r in gsm["sweep"] if r["sign_flip"]]
    check("s9_one_flip", len(flip_rows) == 1, f"n_flip={len(flip_rows)}")

    naive_under = (eco["naive_inference_total"] / eco["realized_inference_total"]) - 1.0
    # naive understates => naive/realized - 1 is negative; understatement fraction:
    under_frac = 1.0 - eco["naive_inference_total"] / eco["realized_inference_total"]
    check("eco_usd_identity", eco["reconciles"] and eco["abs_residual"] < 1e-12, f"resid={eco['abs_residual']}")
    check("eco_naive_under", eco["naive_underestimates_realized"] is True, f"under={under_frac*100:.2f}%")

    dump_csv(
        "routellm_accounting_sweep.csv",
        [
            {
                "threshold": r["threshold"],
                "strong_pct": r["strong_pct"],
                "true_usd": r["true_cost_per_item_usd"],
                "naive_usd": r["naive_cost_per_item_usd"],
                "misestimate_pct": r["cost_misestimate_pct"],
                "R_naive": r["regret_naive"],
                "correction": r["regret_correction"],
                "R_true": r["regret_true"],
                "sign_flip": r["sign_flip"],
            }
            for r in gsm["sweep"]
        ],
    )
    summary = {
        "s8_source": vsrc,
        "s8_n": val["n_items"],
        "R_naive": val["R_naive"],
        "correction": val["correction"],
        "R_true": val["R_true"],
        "s8_residual": val["abs_residual"],
        "s8_sign_flip": True,
        "correction_over_abs_R_true": ratio,
        "eco_usd_source": csrc,
        "eco_realized_usd": eco["realized_inference_total"],
        "eco_naive_usd": eco["naive_inference_total"],
        "eco_understatement_frac": under_frac,
        "eco_understatement_pct": under_frac * 100,
        "routellm_source": esrc,
        "routellm_max_overstated_savings_pct": max_mis,
        "routellm_max_overstated_savings_1dp": round(max_mis, 1),
        "routellm_flip_strong_pct": flip_rows[0]["strong_pct"],
        "routellm_cv_weak": gsm["within_model_cost_cv"]["weak"],
        "routellm_cv_strong": gsm["within_model_cost_cv"]["strong"],
        "identity": "R_true = R_naive + sum_{i!=j} Cov(1{C_ij}, D_ij)",
    }
    dump_json("accounting_summary.json", summary)
    return summary


def reconstruct_heldout():
    path = ROOT / "woais_experiments/results/heldout/final_test_results.csv"
    rows = list(csv.DictReader(path.open()))
    cfg, csrc = load_json("woais_experiments/results/heldout/frozen_config.json")
    by = {r["method"]: r for r in rows}
    log_q = float(by["logistic"]["quality"])
    cheap_q = float(by["always_cheap"]["quality"])
    cms_q = float(by["cost_matched_static"]["quality"])
    log_c = float(by["logistic"]["realized_cost"])
    cms_c = float(by["cost_matched_static"]["realized_cost"])
    ident_q = log_q == cheap_q == cms_q
    ident_c = math.isclose(log_c, cms_c, rel_tol=0, abs_tol=1e-18)
    check("heldout_null_quality", ident_q, f"logistic={log_q} cheap={cheap_q} cms={cms_q}")
    check("heldout_null_cost", ident_c, f"logistic={log_c} cms={cms_c}")
    check("heldout_selected", by["logistic"]["selected"] == "1", "logistic selected")
    check("heldout_n", int(by["logistic"]["n"]) == 243, by["logistic"]["n"])
    adv = float(by["logistic"]["quality_advantage_vs_cost_matched_static"])
    check("heldout_adv_zero", adv == 0.0, f"adv={adv}")

    dump_csv("heldout_test.csv", rows)
    summary = {
        "source": str(path),
        "config_source": csrc,
        "n_test": 243,
        "n_train": cfg["n"]["train"],
        "n_val": cfg["n"]["val"],
        "selected_method": cfg["selected_method"],
        "logistic_quality": log_q,
        "logistic_quality_4dp": round(log_q, 4),
        "always_cheap_quality": cheap_q,
        "cost_matched_static_quality": cms_q,
        "logistic_usd": log_c,
        "identical_to_static": True,
        "verdict": "NULL",
        "heuristic_quality": float(by["ecologic_heuristic"]["quality"]),
        "oracle_quality": float(by["oracle"]["quality"]),
    }
    dump_json("heldout_summary.json", summary)
    return summary


def reconstruct_stage5():
    data, src = load_json("router_v2/final_test_set_results.json")
    pr = data["prereg_comparison_vs_always_t2"]
    check("stage5_n", data["n_items"] == 364, str(data["n_items"]))
    p = pr["mcnemar"]["p_value"]
    check("stage5_mcnemar_p1", abs(p - 1.0) < 1e-12, f"p={p}")
    summary = {
        "source": src,
        "learned_accuracy": pr["learned_accuracy"],
        "always_t2_accuracy": pr["always_t2_accuracy"],
        "learned_acc_pct_1dp": round(pr["learned_accuracy"] * 100, 1),
        "always_t2_acc_pct_1dp": round(pr["always_t2_accuracy"] * 100, 1),
        "mcnemar_p": p,
        "s1_met": data["s1_met"],
        "s2_met": data["s2_met"],
        "verdict": data["verdict"],
        "note": "Stage 5 one-shot on frozen n=364; S2 used modelled energy, not reported as a USD result here.",
    }
    dump_json("stage5_prereg.json", summary)
    return summary


def reconstruct_ceiling():
    data, src = load_json("stage7_10/s7_ceiling.json")
    rows = []
    for name, m in data["models"].items():
        rows.append(
            {
                "model": name,
                "mean_calib_auc": m["mean_calibration_auc"],
                "mean_train_auc": m["mean_train_auc"],
                "t1_calib_auc": m["per_head"]["1"]["calibration_auc"],
                "t2_calib_auc": m["per_head"]["2"]["calibration_auc"],
            }
        )
    dump_csv("ceiling_auc.csv", rows)
    log_auc = data["models"]["logistic (R2 baseline)"]["mean_calibration_auc"]
    check("ceiling_logistic_auc", abs(log_auc - 0.681642039359188) < 1e-12, f"{log_auc}")
    check("ceiling_lift_negative", data["auc_lift_over_linear"] < 0, str(data["auc_lift_over_linear"]))
    summary = {
        "source": src,
        "n_train": data["n_train"],
        "n_calibration": data["n_calibration"],
        "logistic_mean_calib_auc": log_auc,
        "logistic_mean_calib_auc_4dp": round(log_auc, 4),
        "gb_mean_calib_auc": data["models"]["gradient boosting"]["mean_calibration_auc"],
        "rf_mean_calib_auc": data["models"]["random forest"]["mean_calibration_auc"],
        "knn_mean_calib_auc": data["models"]["k-NN (k=50)"]["mean_calibration_auc"],
        "rf_mean_train_auc": data["models"]["random forest"]["mean_train_auc"],
        "best_nonlinear": data["best_nonlinear_model"],
        "auc_lift_over_linear": data["auc_lift_over_linear"],
        "verdict": data["verdict"],
    }
    dump_json("ceiling_summary.json", summary)
    return summary


def reconstruct_noise():
    data, src = load_json("stage7_10/s7_generation_variance.json")
    t1 = data["policies"]["always_t1"]
    t3 = data["policies"]["always_t3"]
    check("t1_flip_n", t1["n_items_with_flip"] == 72, str(t1["n_items_with_flip"]))
    check("t1_flip_frac", abs(t1["frac_items_with_flip"] - 72 / 364) < 1e-12, str(t1["frac_items_with_flip"]))
    check("t1_within_share", abs(t1["within_share_of_total"] - 0.4697114923799483) < 1e-12, str(t1["within_share_of_total"]))
    check("t3_flip_n", t3["n_items_with_flip"] == 5, str(t3["n_items_with_flip"]))
    rows = []
    for key, pol in data["policies"].items():
        rows.append(
            {
                "policy": key,
                "mean_acc_k3": pol["mean_accuracy_over_k"],
                "n_flip": pol["n_items_with_flip"],
                "frac_flip": pol["frac_items_with_flip"],
                "within_share": pol["within_share_of_total"],
                "wilson_half_width_pp": pol["wilson_half_width_pp"],
                "sampling_plus_gen_half_pp": pol["half_width_sampling_plus_generation_pp"],
            }
        )
    dump_csv("generation_variance.csv", rows)
    summary = {
        "source": src,
        "k": data["k"],
        "temperature": data["temperature"],
        "n_items": data["n_items"],
        "t1_n_flip": t1["n_items_with_flip"],
        "t1_flip_pct": t1["frac_items_with_flip"] * 100,
        "t1_flip_pct_1dp": round(t1["frac_items_with_flip"] * 100, 1),
        "t1_within_share": t1["within_share_of_total"],
        "t1_within_share_pct": t1["within_share_of_total"] * 100,
        "t1_within_share_pct_0dp": round(t1["within_share_of_total"] * 100),
        "t3_flip_pct": t3["frac_items_with_flip"] * 100,
        "t3_flip_pct_1dp": round(t3["frac_items_with_flip"] * 100, 1),
        "learned_stage7_sampling_plus_gen_pp": data["policies"]["learned_s7"][
            "half_width_sampling_plus_generation_pp"
        ]
        if "learned_s7" in data["policies"]
        else data["policies"].get("learned_router_s7", {}).get("half_width_sampling_plus_generation_pp"),
    }
    # policy key for learned router
    learned_keys = [k for k in data["policies"] if "learned" in k.lower() or k.startswith("s7")]
    summary["learned_policy_keys"] = learned_keys
    dump_json("generation_variance_summary.json", summary)
    return summary


def write_checks():
    path = OUT / "CHECKS.md"
    failed = [c for c in CHECKS if not c[1]]
    lines = ["# Reconstruction checks\n", f"passed: {sum(1 for c in CHECKS if c[1])}  failed: {len(failed)}\n"]
    for name, ok, detail in CHECKS:
        lines.append(f"- {'PASS' if ok else 'FAIL'} `{name}` — {detail}\n")
    path.write_text("".join(lines))
    dump_json(
        "checks.json",
        {"n": len(CHECKS), "n_failed": len(failed), "failed": [{"name": n, "detail": d} for n, ok, d in CHECKS if not ok]},
    )
    return failed


def main() -> int:
    print(f"ROOT={ROOT}")
    s12, _ = reconstruct_stage12()
    rl = reconstruct_routellm()
    acc = reconstruct_accounting()
    ho = reconstruct_heldout()
    s5 = reconstruct_stage5()
    ceil = reconstruct_ceiling()
    noise = reconstruct_noise()
    dump_json(
        "paper_numbers.json",
        {
            "stage12": s12,
            "robustness": s12.get("robustness"),
            "routellm": rl,
            "accounting": acc,
            "heldout": ho,
            "stage5": s5,
            "ceiling": ceil,
            "noise": noise,
        },
    )
    failed = write_checks()
    if failed:
        print("FAILED CHECKS:", failed, file=sys.stderr)
        return 1
    print("all reconstruction checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

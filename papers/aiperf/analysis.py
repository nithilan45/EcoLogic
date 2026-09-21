#!/usr/bin/env python3
"""Offline reconstruction for AIPerf 2027 Paper C.

Rebuilds per-tier latency distributions, per-query policy latency, USD–latency
Pareto points, and token–latency correlations from frozen Stage 1–2 artifacts.
Writes only under papers/aiperf/artifacts/. No API calls. No frozen-tree writes.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panel_stats import bootstrap_t2_dominates, leave_one_benchmark, mix_reweight_grid

REPO = Path(__file__).resolve().parents[2]
TIERS = (1, 2, 3)
WILSON_Z = 1.959963984540054
RANDOM_SEED = 20260905
GRADED = REPO / "raw_results" / "graded.jsonl"
ROUTING = REPO / "raw_results" / "routing.json"

OUT = Path(__file__).resolve().parent / "artifacts"
COMMITTED_WALL = REPO / "woais_experiments" / "results" / "latency" / "stage12_wallclock.json"
COMMITTED_FRONTIER_CSV = (
    REPO / "woais_experiments" / "results" / "latency" / "frontier_per_query.csv"
)
COMMITTED_FRONTIER_AGG = (
    REPO / "woais_experiments" / "results" / "latency" / "frontier_aggregates.json"
)
COMMITTED_USD = REPO / "woais_experiments" / "results" / "accounting" / "stage12.json"

POLICY_LABELS = {
    "ecologic": "EcoLogic (keyword)",
    "always_t1": "Always-T1",
    "always_t2": "Always-T2",
    "frontier": "Always-T3",
    "random": "Random",
    "oracle_usd": "Oracle (hindsight)",
}

DISPLAY_ORDER = [
    "ecologic",
    "always_t1",
    "always_t2",
    "frontier",
    "random",
    "oracle_usd",
]

DEPLOYABLE = {"ecologic", "always_t1", "always_t2", "frontier", "random"}


@dataclass
class Panel:
    item_ids: List[str]
    bench_of: Dict[str, str]
    correct: Dict[Tuple[int, str], bool]
    tokens: Dict[Tuple[int, str], int]
    prompt_tokens: Dict[Tuple[int, str], int]
    completion_tokens: Dict[Tuple[int, str], int]
    usd: Dict[Tuple[int, str], float]
    latency_s: Dict[Tuple[int, str], float]
    model_of: Dict[int, str] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.item_ids)


def _percentile(xs, q: float) -> float:
    """Nearest-rank, identical to woais_experiments.accounting.costs._percentile."""
    if not xs:
        return 0.0
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    idx = min(len(ys) - 1, max(0, round((q / 100.0) * (len(ys) - 1))))
    return ys[idx]


def wilson(k: int, n: int):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + WILSON_Z * WILSON_Z / n
    center = (p + WILSON_Z * WILSON_Z / (2 * n)) / denom
    half = WILSON_Z / denom * math.sqrt(p * (1 - p) / n + WILSON_Z * WILSON_Z / (4 * n * n))
    return (p, max(0.0, center - half), min(1.0, center + half))


def _binomtest_two_sided(k: int, n: int, p: float = 0.5) -> float:
    def pmf(i: int) -> float:
        return math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

    observed = pmf(k)
    return float(min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= observed + 1e-15)))


def mcnemar(a, b):
    b_only = sum(1 for x, y in zip(a, b) if x and not y)
    c_only = sum(1 for x, y in zip(a, b) if y and not x)
    n = b_only + c_only
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0}
    return {"b": b_only, "c": c_only, "p_value": _binomtest_two_sided(b_only, n, 0.5)}


def pearson(x, y) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def load_panel() -> Panel:
    correct, tokens, usd, lat = {}, {}, {}, {}
    pt, ct = {}, {}
    bench_of = {}
    model_of = {}
    with GRADED.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            t, i = int(r["tier"]), str(r["item_id"])
            key = (t, i)
            correct[key] = bool(r["correct"])
            tokens[key] = int(r["total_tokens"])
            pt[key] = int(r["prompt_tokens"])
            ct[key] = int(r["completion_tokens"])
            usd[key] = float(r["usd"])
            lat[key] = float(r["latency_s"])
            if r.get("benchmark"):
                bench_of[i] = r["benchmark"]
            if r.get("model"):
                model_of[t] = r["model"]
    item_ids = sorted({i for (_, i) in correct}, key=lambda x: (bench_of.get(x, ""), x))
    complete = [i for i in item_ids if all((t, i) in correct for t in TIERS)]
    return Panel(complete, bench_of, correct, tokens, pt, ct, usd, lat, model_of)


def build_policies(matrix: Panel, routing: dict, seed: int = RANDOM_SEED):
    items = matrix.item_ids
    rng = random.Random(seed)
    eco = {i: int(routing[i]["raw"]["tier"]) for i in items}
    wrapped = {i: int(routing[i]["wrapped"]["tier"]) for i in items}
    oracle = {}
    for i in items:
        cands = [(matrix.usd[(t, i)], t) for t in TIERS if matrix.correct[(t, i)]]
        if cands:
            oracle[i] = min(cands)[1]
        else:
            oracle[i] = min((matrix.usd[(t, i)], t) for t in TIERS)[1]
    return {
        "ecologic": eco,
        "ecologic_wrapped": wrapped,
        "always_t1": {i: 1 for i in items},
        "always_t2": {i: 2 for i in items},
        "frontier": {i: 3 for i in items},
        "random": {i: rng.choice(list(TIERS)) for i in items},
        "oracle_usd": oracle,
    }


def evaluate_assignment(matrix: Panel, assign: dict):
    items = matrix.item_ids
    outcomes = [bool(matrix.correct[(assign[i], i)]) for i in items]
    k = sum(outcomes)
    n = len(items)
    p, lo, hi = wilson(k, n)
    cost = sum(matrix.usd[(assign[i], i)] for i in items)
    mix = {t: sum(1 for i in items if assign[i] == t) for t in TIERS}
    return {
        "accuracy": p,
        "ci": [lo, hi],
        "correct": k,
        "n": n,
        "cost": cost,
        "outcomes": outcomes,
        "mix": mix,
    }


def latency_by_tier_benchmark(matrix: Panel) -> dict:
    buckets = defaultdict(list)
    for t in TIERS:
        for i in matrix.item_ids:
            b = matrix.bench_of.get(i, "unknown")
            buckets[(t, b)].append(matrix.latency_s[(t, i)])
    out = {}
    for (t, b), xs in sorted(buckets.items()):
        out["t{}/{}".format(t, b)] = _summarize(xs)
    by_tier = {}
    for t in TIERS:
        xs = [matrix.latency_s[(t, i)] for i in matrix.item_ids]
        by_tier[str(t)] = _summarize(xs)
    return {"by_tier": by_tier, "by_tier_benchmark": out}


def policy_latency(matrix: Panel, assign: dict) -> dict:
    xs = [matrix.latency_s[(assign[i], i)] for i in matrix.item_ids]
    return _summarize(xs)


def bootstrap_percentile(xs, q: float, n_boot: int = 5000, seed: int = RANDOM_SEED):
    rng = np.random.default_rng(seed)
    arr = np.asarray(xs, dtype=float)
    n = len(arr)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        samp = rng.choice(arr, size=n, replace=True)
        stats[b] = _percentile(samp.tolist(), q)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def slo_fractions(xs, thresholds):
    arr = np.asarray(xs, dtype=float)
    return {str(t): float(np.mean(arr <= t)) for t in thresholds}


def policy_latency_by_benchmark(matrix: Panel, assign: dict) -> dict:
    out = {}
    for b in ("humaneval", "mmlu", "gsm8k"):
        xs = [
            matrix.latency_s[(assign[i], i)]
            for i in matrix.item_ids
            if matrix.bench_of.get(i) == b
        ]
        out[b] = _summarize(xs)
    return out


def tail_mass(xs, frac: float = 0.10) -> dict:
    arr = np.sort(np.asarray(xs, dtype=float))
    n = len(arr)
    k = max(1, int(math.ceil(frac * n)))
    tail = arr[-k:]
    total = float(arr.sum())
    return {
        "frac_queries": k / n,
        "n_tail": k,
        "share_of_latency_mass": float(tail.sum() / total) if total else 0.0,
        "tail_min_s": float(tail[0]),
        "tail_mean_s": float(tail.mean()),
    }


def mg1_wait(mean_s: float, cv: float, lambda_qph: float) -> dict:
    """Pollaczek–Khinchine wait if measured RTT were exclusive service time.

    Honesty: HTTP RTT already includes provider queueing. This is a
    dedicated-replica what-if, not a prediction of the measured system.
    """
    lam = lambda_qph / 3600.0
    mu = 1.0 / mean_s if mean_s > 0 else float("inf")
    rho = lam * mean_s
    if rho >= 1.0 or rho < 0.0:
        return {
            "lambda_qph": lambda_qph,
            "rho": rho,
            "ew_s": None,
            "et_s": None,
            "stable": False,
            "honesty": "hypothetical dedicated replica; RTT treated as exclusive S",
        }
    c2 = cv * cv
    ew = rho * mean_s * (1.0 + c2) / (2.0 * (1.0 - rho))
    return {
        "lambda_qph": lambda_qph,
        "rho": rho,
        "ew_s": ew,
        "et_s": ew + mean_s,
        "stable": True,
        "honesty": "hypothetical dedicated replica; RTT treated as exclusive S",
    }


def per_benchmark_slo(matrix: Panel, policies: dict, names, thresholds) -> list[dict]:
    rows = []
    for name in names:
        assign = policies[name]
        for b in ("humaneval", "mmlu", "gsm8k"):
            xs = [
                matrix.latency_s[(assign[i], i)]
                for i in matrix.item_ids
                if matrix.bench_of.get(i) == b
            ]
            slo = slo_fractions(xs, thresholds)
            rows.append(
                {
                    "policy": name,
                    "benchmark": b,
                    "n": len(xs),
                    "mean_s": float(np.mean(xs)),
                    **{f"leq_{t}s": slo[str(t)] for t in thresholds},
                }
            )
    return rows


def completion_by_bench_tier(matrix: Panel) -> list[dict]:
    rows = []
    for t in TIERS:
        for b in ("humaneval", "mmlu", "gsm8k"):
            toks = [
                matrix.completion_tokens[(t, i)]
                for i in matrix.item_ids
                if matrix.bench_of.get(i) == b
            ]
            lats = [
                matrix.latency_s[(t, i)]
                for i in matrix.item_ids
                if matrix.bench_of.get(i) == b
            ]
            rows.append(
                {
                    "tier": t,
                    "benchmark": b,
                    "n": len(toks),
                    "completion_mean": float(np.mean(toks)),
                    "latency_mean_s": float(np.mean(lats)),
                    "pearson_r": pearson(toks, lats),
                }
            )
    return rows


def latency_per_output_token(matrix: Panel) -> dict:
    out = {}
    for t in TIERS:
        ratios = []
        for i in matrix.item_ids:
            ct = matrix.completion_tokens[(t, i)]
            if ct > 0:
                ratios.append(matrix.latency_s[(t, i)] / ct)
        out[str(t)] = _summarize(ratios)
    return out


def token_dispersion_completion(matrix: Panel) -> dict:
    out = {}
    for t in TIERS:
        xs = [float(matrix.completion_tokens[(t, i)]) for i in matrix.item_ids]
        mean = sum(xs) / len(xs)
        out[str(t)] = {"mean": mean, "n": len(xs)}
    return out


def corr_tokens_vs_latency(matrix: Panel) -> dict:
    out = {}
    for t in TIERS:
        x = [float(matrix.completion_tokens[(t, i)]) for i in matrix.item_ids]
        y = [float(matrix.latency_s[(t, i)]) for i in matrix.item_ids]
        out[str(t)] = {"pearson_r": pearson(x, y), "n": len(x)}
    return out


def _summarize(xs: list) -> dict:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "p50": _percentile(xs, 50),
        "p90": _percentile(xs, 90),
        "p95": _percentile(xs, 95),
        "p99": _percentile(xs, 99),
        "min": min(xs),
        "max": max(xs),
        "std": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0,
    }


def _rel_close(a: float, b: float, rtol: float = 1e-9, atol: float = 1e-9) -> bool:
    return abs(a - b) <= atol + rtol * abs(b)


def _empirical_cdf(xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys = np.sort(xs)
    n = ys.size
    p = np.arange(1, n + 1, dtype=float) / n
    return ys, p


def mix_weighted_mean(mix: dict[int, int], tier_means: dict[int, float], n: int) -> float:
    return sum(mix[t] / n * tier_means[t] for t in TIERS)


def pareto_front(
    points: list[dict],
    *,
    cost_key: str = "usd",
    lat_key: str = "mean_s",
    deployable_only: bool = True,
) -> list[str]:
    """Minimize cost and mean latency. Accuracy is not an axis here."""
    cands = [p for p in points if (p["name"] in DEPLOYABLE) or not deployable_only]
    names = []
    for p in cands:
        dominated = False
        for q in cands:
            if q["name"] == p["name"]:
                continue
            le_cost = q[cost_key] <= p[cost_key] + 1e-15
            le_lat = q[lat_key] <= p[lat_key] + 1e-15
            strict = (q[cost_key] < p[cost_key] - 1e-15) or (q[lat_key] < p[lat_key] - 1e-15)
            if le_cost and le_lat and strict:
                dominated = True
                break
        if not dominated:
            names.append(p["name"])
    return names


def dominates_3d(a: dict, b: dict) -> bool:
    """a dominates b if weakly better on USD, mean latency, and accuracy, strictly on one."""
    le_c = a["usd"] <= b["usd"] + 1e-15
    le_l = a["mean_s"] <= b["mean_s"] + 1e-15
    ge_a = a["accuracy"] + 1e-15 >= b["accuracy"]
    strict = (
        a["usd"] < b["usd"] - 1e-15
        or a["mean_s"] < b["mean_s"] - 1e-15
        or a["accuracy"] > b["accuracy"] + 1e-15
    )
    return le_c and le_l and ge_a and strict


def _fig_tokens_latency(matrix: Panel) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    colors = {1: "#1f4e79", 2: "#c45911", 3: "#548235"}
    labels = {1: "T1", 2: "T2", 3: "T3"}
    for t in TIERS:
        x = [matrix.completion_tokens[(t, i)] for i in matrix.item_ids]
        y = [matrix.latency_s[(t, i)] for i in matrix.item_ids]
        ax.scatter(x, y, s=8, alpha=0.45, c=colors[t], label=labels[t], edgecolors="none")
    ax.set_xlabel("Completion tokens")
    ax.set_ylabel("HTTP RTT (s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    fig.savefig(OUT / "fig3_tokens_latency.pdf", bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def write_cdf_csv(lat_by_tier: dict[int, np.ndarray], path: Path) -> None:
    """One row per (tier, order-statistic) for PGFPlots step CDFs."""
    rows: list[dict] = []
    for t in TIERS:
        x, p = _empirical_cdf(lat_by_tier[t])
        for xi, pi in zip(x, p):
            rows.append({"tier": t, "latency_s": float(xi), "cdf": float(pi)})
    write_csv(path, rows, ["tier", "latency_s", "cdf"])


def write_tokens_latency_csv(matrix, path: Path) -> None:
    rows = []
    for t in TIERS:
        for i in matrix.item_ids:
            rows.append(
                {
                    "tier": t,
                    "query_id": i,
                    "completion_tokens": matrix.completion_tokens[(t, i)],
                    "latency_s": matrix.latency_s[(t, i)],
                }
            )
    write_csv(path, rows, ["tier", "query_id", "completion_tokens", "latency_s"])


def write_svg_tier_cdf(lat_by_tier: dict[int, np.ndarray], path: Path) -> None:
    """Standalone SVG (no matplotlib). Log-x empirical CDFs."""
    w, h, l, r, t, b = 680, 420, 70, 24, 18, 56
    pw, ph = w - l - r, h - t - b
    xmin, xmax = 0.3, 300.0
    colors = {1: "#1f4e79", 2: "#c45911", 3: "#548235"}
    labels = {1: "T1 Qwen3.5-9B", 2: "T2 gpt-oss-20b", 3: "T3 gpt-4o"}

    def xmap(x: float) -> float:
        return l + pw * (math_log(x) - math_log(xmin)) / (math_log(xmax) - math_log(xmin))

    def ymap(p: float) -> float:
        return t + ph * (1.0 - p)

    def math_log(x: float) -> float:
        return float(np.log(x))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{w/2}" y="{h-12}" text-anchor="middle" font-size="13" font-family="Times,serif">'
        "HTTP round-trip latency (s), log scale</text>",
        f'<text x="16" y="{h/2}" text-anchor="middle" font-size="13" font-family="Times,serif" '
        f'transform="rotate(-90 16 {h/2})">Empirical CDF</text>',
        f'<rect x="{l}" y="{t}" width="{pw}" height="{ph}" fill="none" stroke="#333" stroke-width="1"/>',
    ]
    for tick in (0.5, 1, 2, 5, 10, 20, 50, 100, 200):
        xx = xmap(tick)
        parts.append(
            f'<line x1="{xx:.1f}" y1="{t}" x2="{xx:.1f}" y2="{t+ph}" stroke="#ddd" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xx:.1f}" y="{t+ph+16}" text-anchor="middle" font-size="11" '
            f'font-family="Times,serif">{tick:g}</text>'
        )
    for pval in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = ymap(pval)
        parts.append(
            f'<line x1="{l}" y1="{yy:.1f}" x2="{l+pw}" y2="{yy:.1f}" stroke="#eee" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{l-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11" '
            f'font-family="Times,serif">{pval:g}</text>'
        )
    legend_y = t + 18
    for t_id in TIERS:
        x, p = _empirical_cdf(lat_by_tier[t_id])
        d = [f"M {xmap(max(x[0], xmin)):.2f} {ymap(p[0]):.2f}"]
        for xi, pi in zip(x[1:], p[1:]):
            xx = xmap(min(max(xi, xmin), xmax))
            d.append(f"H {xx:.2f}")
            d.append(f"V {ymap(pi):.2f}")
        parts.append(
            f'<path d="{" ".join(d)}" fill="none" stroke="{colors[t_id]}" stroke-width="2.2"/>'
        )
        parts.append(
            f'<line x1="{l+pw-210}" y1="{legend_y}" x2="{l+pw-185}" y2="{legend_y}" '
            f'stroke="{colors[t_id]}" stroke-width="2.2"/>'
        )
        parts.append(
            f'<text x="{l+pw-178}" y="{legend_y+4}" font-size="12" font-family="Times,serif">'
            f"{labels[t_id]}</text>"
        )
        legend_y += 18
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def write_svg_cost_latency(points: list[dict], path: Path) -> None:
    w, h, l, r, t, b = 680, 420, 70, 18, 18, 56
    pw, ph = w - l - r, h - t - b
    xmin, xmax = 0.02, 0.75
    ymin, ymax = 0.8, 60.0
    colors = {
        "ecologic": "#c00000",
        "always_t1": "#7f7f7f",
        "always_t2": "#2e75b6",
        "frontier": "#548235",
        "random": "#ed7d31",
        "oracle_usd": "#7030a0",
    }
    short = {
        "ecologic": "EcoLogic",
        "always_t1": "Always-T1",
        "always_t2": "Always-T2",
        "frontier": "Always-T3",
        "random": "Random",
        "oracle_usd": "Oracle",
    }

    def xmap(x: float) -> float:
        return l + pw * (x - xmin) / (xmax - xmin)

    def ymap(y: float) -> float:
        return t + ph * (1.0 - (np.log(y) - np.log(ymin)) / (np.log(ymax) - np.log(ymin)))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{w/2}" y="{h-12}" text-anchor="middle" font-size="13" font-family="Times,serif">'
        "Measured USD (panel total). Marker area ~ accuracy.</text>",
        f'<text x="16" y="{h/2}" text-anchor="middle" font-size="13" font-family="Times,serif" '
        f'transform="rotate(-90 16 {h/2})">Mean HTTP RTT (s), log</text>',
        f'<rect x="{l}" y="{t}" width="{pw}" height="{ph}" fill="none" stroke="#333" stroke-width="1"/>',
    ]
    for xv in (0.05, 0.15, 0.30, 0.45, 0.60):
        xx = xmap(xv)
        parts.append(
            f'<line x1="{xx:.1f}" y1="{t}" x2="{xx:.1f}" y2="{t+ph}" stroke="#eee" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xx:.1f}" y="{t+ph+16}" text-anchor="middle" font-size="11" '
            f'font-family="Times,serif">{xv:.2f}</text>'
        )
    for yv in (1, 2, 5, 10, 20, 40):
        yy = ymap(yv)
        parts.append(
            f'<line x1="{l}" y1="{yy:.1f}" x2="{l+pw}" y2="{yy:.1f}" stroke="#eee" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{l-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11" '
            f'font-family="Times,serif">{yv:g}</text>'
        )
    for p in points:
        cx, cy = xmap(p["usd"]), ymap(p["mean_s"])
        rad = 6 + 22 * max(0.0, (p["accuracy"] - 0.86) / 0.11)
        if p["name"] == "oracle_usd":
            parts.append(
                f'<text x="{cx:.1f}" y="{cy+4:.1f}" text-anchor="middle" font-size="16" '
                f'fill="{colors[p["name"]]}">+</text>'
            )
        else:
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{rad:.1f}" fill="{colors[p["name"]]}" '
                f'stroke="#111" stroke-width="0.8" fill-opacity="0.85"/>'
            )
        parts.append(
            f'<text x="{cx+rad+4:.1f}" y="{cy-6:.1f}" font-size="12" font-family="Times,serif" '
            f'fill="{colors[p["name"]]}">{short[p["name"]]}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    matrix = load_panel()
    routing = json.loads(ROUTING.read_text())
    policies = build_policies(matrix, routing, seed=RANDOM_SEED)
    committed_wall = json.loads(COMMITTED_WALL.read_text())
    committed_usd = json.loads(COMMITTED_USD.read_text())
    committed_agg = json.loads(COMMITTED_FRONTIER_AGG.read_text())

    lat_by_tier = {
        t: np.array([matrix.latency_s[(t, i)] for i in matrix.item_ids], dtype=float)
        for t in TIERS
    }
    by_tier = latency_by_tier_benchmark(matrix)
    per_tok = latency_per_output_token(matrix)
    corr = corr_tokens_vs_latency(matrix)
    disp = token_dispersion_completion(matrix)
    usd_eval = {"policies": {}, "mcnemar_vs": {}}
    for name, assign in policies.items():
        usd_eval["policies"][name] = evaluate_assignment(matrix, assign)
    ref = usd_eval["policies"]["ecologic"]["outcomes"]
    for name, s in usd_eval["policies"].items():
        if name == "ecologic":
            continue
        usd_eval["mcnemar_vs"][name] = mcnemar(ref, s["outcomes"])

    checks: list[str] = []
    mismatches: list[str] = []

    # Cross-check per-tier summaries against committed wall-clock JSON.
    for t in TIERS:
        recon = by_tier["by_tier"][str(t)]
        gold = committed_wall["by_tier_benchmark"]["by_tier"][str(t)]
        for k in ("n", "mean", "p50", "p90", "p95", "p99", "min", "max"):
            if not _rel_close(float(recon[k]), float(gold[k])):
                mismatches.append(f"tier {t} {k}: {recon[k]} vs committed {gold[k]}")
    checks.append(f"per-tier wall-clock vs {COMMITTED_WALL.name}: {'OK' if not mismatches else 'FAIL'}")

    # Policy latency: PER-QUERY selected-tier latency_s (not mix-weighted).
    policy_lat: dict[str, dict] = {}
    per_query_rows: list[dict] = []
    for name in DISPLAY_ORDER:
        assign = policies[name]
        policy_lat[name] = policy_latency(matrix, assign)
        gold = committed_wall["policies"].get(name)
        if gold is not None:
            for k in ("n", "mean", "p50", "p90", "p95", "p99"):
                if not _rel_close(float(policy_lat[name][k]), float(gold[k])):
                    mismatches.append(
                        f"policy {name} {k}: {policy_lat[name][k]} vs committed {gold[k]}"
                    )
        for i in matrix.item_ids:
            t = assign[i]
            per_query_rows.append(
                {
                    "policy": name,
                    "query_id": i,
                    "tier": t,
                    "latency_s": matrix.latency_s[(t, i)],
                    "usd": matrix.usd[(t, i)],
                    "correct": int(matrix.correct[(t, i)]),
                    "completion_tokens": matrix.completion_tokens[(t, i)],
                }
            )

    # EcoLogic mix decomposition.
    eco_assign = policies["ecologic"]
    mix = {t: sum(1 for i in matrix.item_ids if eco_assign[i] == t) for t in TIERS}
    n = matrix.n
    tier_means = {t: float(np.mean(lat_by_tier[t])) for t in TIERS}
    naive_mean = mix_weighted_mean(mix, tier_means, n)
    realized_mean = policy_lat["ecologic"]["mean"]
    eco_lats = np.array(
        [matrix.latency_s[(eco_assign[i], i)] for i in matrix.item_ids], dtype=float
    )
    t2_lats = lat_by_tier[2]
    t3_lats = lat_by_tier[3]
    t1_mass = sum(
        matrix.latency_s[(1, i)] for i in matrix.item_ids if eco_assign[i] == 1
    )
    total_mass = float(eco_lats.sum())
    slower_than_t2 = int(np.sum(eco_lats > t2_lats))
    slower_than_t3 = int(np.sum(eco_lats > t3_lats))
    paired_delta_vs_t2 = eco_lats - t2_lats

    # Routed T1 subset vs unconditional T1.
    routed_t1_ids = [i for i in matrix.item_ids if eco_assign[i] == 1]
    routed_t1_lat = [matrix.latency_s[(1, i)] for i in routed_t1_ids]
    unrouted_t1_on_t2 = [
        matrix.latency_s[(1, i)] for i in matrix.item_ids if eco_assign[i] != 1
    ]

    mix_decomp = {
        "method": "per_query_selected_tier_latency_s",
        "n": n,
        "mix_counts": {str(t): mix[t] for t in TIERS},
        "mix_frac": {str(t): mix[t] / n for t in TIERS},
        "realized_mean_s": realized_mean,
        "mix_weighted_mean_s": naive_mean,
        "mix_weighted_minus_realized_s": naive_mean - realized_mean,
        "relative_bias_mix_weighted": (naive_mean - realized_mean) / realized_mean,
        "t1_share_of_latency_mass": t1_mass / total_mass,
        "n_slower_than_always_t2": slower_than_t2,
        "frac_slower_than_always_t2": slower_than_t2 / n,
        "n_slower_than_always_t3": slower_than_t3,
        "mean_paired_delta_vs_t2_s": float(paired_delta_vs_t2.mean()),
        "p50_paired_delta_vs_t2_s": float(np.median(paired_delta_vs_t2)),
        "routed_t1": _summarize(routed_t1_lat),
        "unconditional_t1": _summarize(list(lat_by_tier[1])),
        "items_not_sent_to_t1_their_t1_latency": _summarize(unrouted_t1_on_t2)
        if unrouted_t1_on_t2
        else {"n": 0},
        "honesty": (
            "latency_s is full HTTP round-trip. TTFT was not recorded. "
            "Router decision span was not recorded on this panel (n_measured=0)."
        ),
    }

    SLO_THRESHOLDS = (2, 5, 10, 30, 60, 120)
    slo_rows = []
    boot_p90 = {}
    bench_lat = {}
    goodput = {}
    for name in DISPLAY_ORDER:
        assign = policies[name]
        xs = [matrix.latency_s[(assign[i], i)] for i in matrix.item_ids]
        slo = slo_fractions(xs, SLO_THRESHOLDS)
        mean_s = policy_lat[name]["mean"]
        goodput[name] = {
            "mean_s": mean_s,
            "serial_queries_per_hour": 3600.0 / mean_s if mean_s > 0 else None,
        }
        boot_p90[name] = {
            "p90_s": policy_lat[name]["p90"],
            "p90_bootstrap_ci95": list(bootstrap_percentile(xs, 90)),
        }
        bench_lat[name] = policy_latency_by_benchmark(matrix, assign)
        slo_rows.append(
            {
                "name": name,
                "label": POLICY_LABELS[name],
                **{f"leq_{t}s": slo[str(t)] for t in SLO_THRESHOLDS},
                "serial_qph": goodput[name]["serial_queries_per_hour"],
                "p90_s": policy_lat[name]["p90"],
                "p90_ci_lo": boot_p90[name]["p90_bootstrap_ci95"][0],
                "p90_ci_hi": boot_p90[name]["p90_bootstrap_ci95"][1],
            }
        )
    n_faster_than_t2 = int(np.sum(eco_lats < t2_lats))
    n_tie_t2 = int(np.sum(eco_lats == t2_lats))
    n_discord_t2 = slower_than_t2 + n_faster_than_t2
    sign_p_lat_vs_t2 = _binomtest_two_sided(slower_than_t2, n_discord_t2) if n_discord_t2 else 1.0
    mix_decomp["n_faster_than_always_t2"] = n_faster_than_t2
    mix_decomp["n_tie_always_t2"] = n_tie_t2
    mix_decomp["sign_test_slower_than_t2_p"] = sign_p_lat_vs_t2

    eco_ok = {i: bool(matrix.correct[(eco_assign[i], i)]) for i in matrix.item_ids}
    t2_ok = {i: bool(matrix.correct[(2, i)]) for i in matrix.item_ids}
    eco_lat = {i: matrix.latency_s[(eco_assign[i], i)] for i in matrix.item_ids}
    t2_lat = {i: matrix.latency_s[(2, i)] for i in matrix.item_ids}
    per_lat = {
        i: {
            "ecologic": eco_lat[i],
            "always_t2": t2_lat[i],
            "frontier": matrix.latency_s[(3, i)],
        }
        for i in matrix.item_ids
    }
    boot_lat = bootstrap_t2_dominates(matrix.item_ids, eco_ok, t2_ok, eco_lat, t2_lat)
    lob_lat = leave_one_benchmark(matrix.item_ids, matrix.bench_of, eco_ok, t2_ok, eco_lat, t2_lat)
    mix_lat = mix_reweight_grid(
        matrix.item_ids, matrix.bench_of, per_lat, ["ecologic", "always_t2", "frontier"], step=0.2
    )
    n_mix_t2_faster = sum(1 for r in mix_lat if r["t2_cheaper_than_eco"])
    n_lob_dom = sum(1 for r in lob_lat if r["t2_dominates_acc_and_cost"])
    cv_t1 = float(np.std(lat_by_tier[1]) / np.mean(lat_by_tier[1]))
    cv_t2 = float(np.std(lat_by_tier[2]) / np.mean(lat_by_tier[2]))
    cv_t3 = float(np.std(lat_by_tier[3]) / np.mean(lat_by_tier[3]))
    robustness = {
        "bootstrap_frac_t2_dominates_acc_and_latency": boot_lat["frac_t2_dominates_quality_and_cost"],
        "n_lob": len(lob_lat),
        "n_lob_t2_dominates": n_lob_dom,
        "mix_n_cells": len(mix_lat),
        "mix_n_t2_faster": n_mix_t2_faster,
        "service_cv": {"t1": cv_t1, "t2": cv_t2, "t3": cv_t3},
    }
    write_csv(OUT / "mix_reweight_latency.csv", mix_lat, list(mix_lat[0].keys()))
    write_csv(OUT / "leave_one_benchmark.csv", lob_lat, list(lob_lat[0].keys()))

    tail_rows = []
    for name in DISPLAY_ORDER:
        xs = [matrix.latency_s[(policies[name][i], i)] for i in matrix.item_ids]
        rec = tail_mass(xs, 0.10)
        rec["name"] = name
        tail_rows.append(rec)
    write_csv(
        OUT / "tail_mass.csv",
        tail_rows,
        ["name", "frac_queries", "n_tail", "share_of_latency_mass", "tail_min_s", "tail_mean_s"],
    )

    offered = (10.0, 50.0, 90.0, 200.0)
    cvs = {"ecologic": None, "always_t1": cv_t1, "always_t2": cv_t2, "frontier": cv_t3, "random": None}
    eco_cv = float(np.std(eco_lats) / np.mean(eco_lats))
    rand_xs = np.array(
        [matrix.latency_s[(policies["random"][i], i)] for i in matrix.item_ids], dtype=float
    )
    rand_cv = float(np.std(rand_xs) / np.mean(rand_xs))
    cvs["ecologic"] = eco_cv
    cvs["random"] = rand_cv
    mg1_rows = []
    for name in ("ecologic", "always_t1", "always_t2", "frontier", "random"):
        mean_s = policy_lat[name]["mean"]
        cv = cvs[name]
        for lam in offered:
            rec = mg1_wait(mean_s, cv, lam)
            rec["name"] = name
            rec["cv"] = cv
            rec["mean_s"] = mean_s
            mg1_rows.append(rec)
    write_csv(
        OUT / "mg1_whatif.csv",
        mg1_rows,
        ["name", "lambda_qph", "mean_s", "cv", "rho", "ew_s", "et_s", "stable", "honesty"],
    )

    bench_slo = per_benchmark_slo(
        matrix, policies, ("ecologic", "always_t2", "frontier"), (2, 10, 30)
    )
    write_csv(
        OUT / "slo_by_benchmark.csv",
        bench_slo,
        ["policy", "benchmark", "n", "mean_s", "leq_2s", "leq_10s", "leq_30s"],
    )
    tok_rows = completion_by_bench_tier(matrix)
    write_csv(
        OUT / "tokens_by_bench_tier.csv",
        tok_rows,
        ["tier", "benchmark", "n", "completion_mean", "latency_mean_s", "pearson_r"],
    )
    write_csv(
        OUT / "slo_attainment.csv",
        slo_rows,
        [
            "name",
            "label",
            "leq_2s",
            "leq_5s",
            "leq_10s",
            "leq_30s",
            "leq_60s",
            "leq_120s",
            "serial_qph",
            "p90_s",
            "p90_ci_lo",
            "p90_ci_hi",
        ],
    )

    # Cross-check frontier CSV mix and selected latencies.
    frontier_rows = list(csv.DictReader(COMMITTED_FRONTIER_CSV.open()))
    if len(frontier_rows) != n:
        mismatches.append(f"frontier csv n={len(frontier_rows)} vs matrix n={n}")
    csv_mix = defaultdict(int)
    csv_selected_s = []
    for row in frontier_rows:
        csv_mix[int(row["router_selected_tier"])] += 1
        csv_selected_s.append(float(row["router_selected_end_to_end_ms"]) / 1000.0)
    if dict(csv_mix) != mix:
        mismatches.append(f"frontier csv mix {dict(csv_mix)} vs routing mix {mix}")
    if not _rel_close(float(np.mean(csv_selected_s)), realized_mean, rtol=1e-9, atol=1e-12):
        mismatches.append(
            f"frontier csv mean {np.mean(csv_selected_s)} vs policy mean {realized_mean}"
        )
    checks.append("frontier_per_query.csv mix+mean vs per-query policy: " + ("OK" if dict(csv_mix) == mix else "FAIL"))

    # USD cross-check.
    for name in ("ecologic", "always_t2", "always_t1", "frontier"):
        recon_usd = usd_eval["policies"][name]["cost"]
        gold_usd = committed_usd["axis_usd"]["policies"][name]["cost"]
        if not _rel_close(recon_usd, gold_usd, rtol=1e-12, atol=1e-12):
            mismatches.append(f"USD {name}: {recon_usd} vs {gold_usd}")

    points = []
    for name in DISPLAY_ORDER:
        pstat = usd_eval["policies"][name]
        lat = policy_lat[name]
        points.append(
            {
                "name": name,
                "label": POLICY_LABELS[name],
                "role": "hindsight_oracle" if name.startswith("oracle") else "deployable",
                "usd": pstat["cost"],
                "accuracy": pstat["accuracy"],
                "correct": pstat["correct"],
                "n": pstat["n"],
                "ci_lo": pstat["ci"][0],
                "ci_hi": pstat["ci"][1],
                "mean_s": lat["mean"],
                "p50_s": lat["p50"],
                "p90_s": lat["p90"],
                "p95_s": lat["p95"],
                "p99_s": lat["p99"],
                "min_s": lat["min"],
                "max_s": lat["max"],
                "mix_t1": sum(1 for i in matrix.item_ids if policies[name][i] == 1),
                "mix_t2": sum(1 for i in matrix.item_ids if policies[name][i] == 2),
                "mix_t3": sum(1 for i in matrix.item_ids if policies[name][i] == 3),
            }
        )

    by_name = {p["name"]: p for p in points}
    front = pareto_front(points, deployable_only=True)
    t2_dom_eco_3d = dominates_3d(by_name["always_t2"], by_name["ecologic"])
    t3_dom_eco_costlat = (
        by_name["frontier"]["mean_s"] < by_name["ecologic"]["mean_s"]
        and by_name["frontier"]["usd"] > by_name["ecologic"]["usd"]
    )
    eco_dominated_2d_by_t2 = (
        by_name["always_t2"]["usd"] < by_name["ecologic"]["usd"]
        and by_name["always_t2"]["mean_s"] < by_name["ecologic"]["mean_s"]
    )

    pareto = {
        "objective": "minimize USD and mean HTTP RTT; accuracy shown as bubble, not an axis of the 2D hull",
        "deployable_2d_front": front,
        "always_t2_dominates_ecologic_usd_and_latency": eco_dominated_2d_by_t2,
        "always_t2_dominates_ecologic_3d_usd_latency_accuracy": t2_dom_eco_3d,
        "always_t3_faster_costlier_than_ecologic": t3_dom_eco_costlat,
        "note": (
            "Always-T3 is the low-latency vertex; Always-T2 is the low-USD vertex. "
            "EcoLogic is strictly interior to Always-T2 on both 2D axes and on accuracy."
        ),
    }

    # Seconds-per-token vs committed.
    for t in TIERS:
        recon = per_tok[str(t)]["mean"]
        gold = committed_wall["seconds_per_output_token"][str(t)]["mean"]
        if not _rel_close(recon, gold):
            mismatches.append(f"s/token tier {t}: {recon} vs {gold}")

    # Router overhead honesty.
    overhead = {
        "n_measured": committed_agg["router_overhead"]["n_measured"],
        "n_missing": committed_agg["router_overhead"]["n_missing"],
        "source": str(COMMITTED_FRONTIER_AGG.relative_to(REPO)),
        "interpretation": (
            "Classifier wall-clock was not recorded on the frozen panel. "
            "Reported EcoLogic latency is the selected backend HTTP RTT only."
        ),
    }

    corr_committed = committed_usd["corr_tokens_latency"]
    for t in TIERS:
        if not _rel_close(corr[str(t)]["pearson_r"], corr_committed[str(t)]["pearson_r"]):
            mismatches.append(f"corr tier {t}")

    # Tables.
    write_csv(
        OUT / "tier_latency.csv",
        [
            {
                "tier": t,
                **{k: by_tier["by_tier"][str(t)][k] for k in ("n", "mean", "p50", "p90", "p95", "p99", "min", "max")},
                "s_per_output_token_mean": per_tok[str(t)]["mean"],
                "s_per_output_token_p50": per_tok[str(t)]["p50"],
                "completion_tokens_mean": disp[str(t)]["mean"],
                "pearson_r_completion_vs_latency": corr[str(t)]["pearson_r"],
            }
            for t in TIERS
        ],
        [
            "tier",
            "n",
            "mean",
            "p50",
            "p90",
            "p95",
            "p99",
            "min",
            "max",
            "s_per_output_token_mean",
            "s_per_output_token_p50",
            "completion_tokens_mean",
            "pearson_r_completion_vs_latency",
        ],
    )
    write_csv(
        OUT / "policy_cost_latency.csv",
        points,
        [
            "name",
            "label",
            "role",
            "usd",
            "accuracy",
            "correct",
            "n",
            "ci_lo",
            "ci_hi",
            "mean_s",
            "p50_s",
            "p90_s",
            "p95_s",
            "p99_s",
            "min_s",
            "max_s",
            "mix_t1",
            "mix_t2",
            "mix_t3",
        ],
    )
    write_csv(
        OUT / "per_query_policy_latency.csv",
        per_query_rows,
        [
            "policy",
            "query_id",
            "tier",
            "latency_s",
            "usd",
            "correct",
            "completion_tokens",
        ],
    )

    write_cdf_csv(lat_by_tier, OUT / "fig1_tier_latency_cdf.csv")
    for t in TIERS:
        x, p = _empirical_cdf(lat_by_tier[t])
        write_csv(
            OUT / f"fig1_cdf_t{t}.csv",
            [{"latency_s": float(xi), "cdf": float(pi)} for xi, pi in zip(x, p)],
            ["latency_s", "cdf"],
        )
    write_tokens_latency_csv(matrix, OUT / "fig3_tokens_vs_latency.csv")
    write_svg_tier_cdf(lat_by_tier, OUT / "fig1_tier_latency_cdf.svg")
    write_svg_cost_latency(points, OUT / "fig2_cost_latency_pareto.svg")
    _fig_tokens_latency(matrix)

    mcnemar_vs = {
        k: {
            "p_value": v["p_value"],
            "p_adjusted": v.get("p_adjusted"),
            "b": v.get("b"),
            "c": v.get("c"),
        }
        for k, v in usd_eval["mcnemar_vs"].items()
        if k in DEPLOYABLE or k == "always_t2"
    }

    summary = {
        "n": n,
        "policy_latency_method": "per_query_selected_tier",
        "not_method": "mix_weighted_tier_means",
        "honesty_latency": committed_wall["honesty"],
        "models": matrix.model_of,
        "by_tier": by_tier["by_tier"],
        "by_tier_benchmark": by_tier["by_tier_benchmark"],
        "seconds_per_output_token": per_tok,
        "corr_tokens_latency": corr,
        "token_dispersion_completion_mean": {
            str(t): disp[str(t)]["mean"] for t in TIERS
        },
        "policies": {p["name"]: p for p in points},
        "mix_decomp": mix_decomp,
        "robustness": robustness,
        "tail_mass_top10": {r["name"]: r for r in tail_rows},
        "mg1_whatif": mg1_rows,
        "slo_by_benchmark": bench_slo,
        "tokens_by_bench_tier": tok_rows,
        "policy_service_cv": cvs,
        "slo": {r["name"]: r for r in slo_rows},
        "p90_bootstrap": boot_p90,
        "serial_goodput_qph": goodput,
        "policy_latency_by_benchmark": bench_lat,
        "pareto": pareto,
        "router_overhead": overhead,
        "mcnemar_vs_ecologic": mcnemar_vs,
        "mismatches": mismatches,
        "n_mismatches": len(mismatches),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (OUT / "mix_decomposition.json").write_text(json.dumps(mix_decomp, indent=2) + "\n")
    (OUT / "pareto.json").write_text(json.dumps(pareto, indent=2) + "\n")

    check_text = [
        "AIPerf analysis reconstruction checks",
        f"n={n}",
        f"policy latency method: per-query selected-tier latency_s",
        f"EcoLogic mix T1/T2/T3: {mix[1]}/{mix[2]}/{mix[3]}",
        f"EcoLogic realized mean {realized_mean:.6f}s vs mix-weighted {naive_mean:.6f}s",
        f"Always-T2 3D-dominates EcoLogic: {t2_dom_eco_3d}",
        f"2D deployable Pareto front: {front}",
        f"router_decision measured: {overhead['n_measured']}/{n}",
        *checks,
        f"mismatches ({len(mismatches)}):",
        *(mismatches or ["none"]),
    ]
    (OUT / "CHECKS.txt").write_text("\n".join(check_text) + "\n")
    print("\n".join(check_text))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

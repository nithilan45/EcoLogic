"""Publication figures. PNGs are written under woais_experiments/results/figures/."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_mpl_dir = Path(tempfile.gettempdir()) / "woais_mplconfig"
_mpl_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from woais_experiments.frozen import write_result_bytes


def _save(fig, relpath: str) -> Path:
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return write_result_bytes(relpath, buf.getvalue())


def plot_accuracy_vs_cost(policies: dict, *, cost_key: str = "cost", title: str, relpath: str) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    names, xs, ys = [], [], []
    for name, s in policies.items():
        if name.startswith("oracle"):
            continue
        names.append(name)
        xs.append(s[cost_key])
        ys.append(100 * s["accuracy"])
    ax.scatter(xs, ys)
    for name, x, y in zip(names, xs, ys):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Cost on the frozen item set")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return _save(fig, relpath)


def plot_latency_cdf(matrix, relpath: str) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for t, label in ((1, "Tier 1"), (2, "Tier 2"), (3, "Tier 3")):
        xs = sorted(matrix.latency_s[(t, i)] for i in matrix.item_ids)
        ys = [(i + 1) / len(xs) for i in range(len(xs))]
        ax.plot(xs, ys, label=label)
    ax.set_xlabel("Wall-clock latency_s (s)")
    ax.set_ylabel("CDF")
    ax.set_title("Empirical service-time CDF by tier (Stage 1–2, frozen)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, relpath)


def plot_naive_vs_true(rows: dict, relpath: str) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    names = list(rows)
    naive = [rows[n]["naive_mean_cost"] for n in names]
    true = [rows[n]["true_mean_cost"] for n in names]
    idx = range(len(names))
    w = 0.35
    ax.bar([i - w / 2 for i in idx], naive, w, label="naive (tier-mean × mix)")
    ax.bar([i + w / 2 for i in idx], true, w, label="true (per-item)")
    ax.set_xticks(list(idx), names, rotation=25, ha="right")
    ax.set_ylabel("Mean cost")
    ax.set_title("Naive vs exact cost accounting")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, relpath)


def plot_sojourn_vs_load(series: dict, relpath: str) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, pts in series.items():
        xs = [p["load"] for p in pts]
        ys = [p["p95_sojourn_s"] for p in pts]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("Arrival rate / (n_servers × 1/E[S])  (offered load)")
    ax.set_ylabel("p95 sojourn time (s)")
    ax.set_title("Open-loop FCFS tails from empirical service times")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, relpath)

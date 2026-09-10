"""Paired percentile bootstrap confidence intervals.

Query indices are resampled with replacement; both series travel together.
Default ``n_boot=10_000`` is for final runs — tests and drafts should pass a
smaller value. CI overlap (with zero or with another interval) is not a
hypothesis test; use ``paired_tests`` for p-values.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from woais_experiments.statistics.effect_sizes import (
    cliffs_delta,
    cohens_dz,
    finite_pairs,
    mean_paired_difference,
    median_paired_difference,
)

DEFAULT_N_BOOT = 10_000
DEFAULT_SEED = 20260909
DEFAULT_LEVEL = 0.95
SIGNIFICANCE_NOTE = (
    "Do not reject or fail to reject from whether a CI overlaps zero or another "
    "CI. Report p_raw from a paired permutation or Wilcoxon test; when several "
    "comparisons share a family, use Benjamini–Hochberg p_adjusted."
)

StatFn = Callable[[np.ndarray, np.ndarray], float | None]


def _quantile(samples: np.ndarray, q: float) -> float | None:
    """Linear (Hyndman–Fan type 7) quantile. Empty → None."""
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.quantile(arr, q, method="linear"))


def attach_ci_fields(payload: dict[str, Any]) -> dict[str, Any]:
    est, lo, hi = payload.get("estimate"), payload.get("ci_lo"), payload.get("ci_hi")
    excludes = None
    if (
        est is not None
        and lo is not None
        and hi is not None
        and np.isfinite(est)
        and np.isfinite(lo)
        and np.isfinite(hi)
    ):
        excludes = not (float(lo) <= 0.0 <= float(hi))
    payload["ci_excludes_zero"] = excludes
    payload.setdefault("p_raw", None)
    payload.setdefault("p_adjusted", None)
    payload.setdefault("effect_size", None)
    payload["significance_note"] = SIGNIFICANCE_NOTE
    return payload


def estimate_payload(
    *,
    name: str,
    n: int,
    estimate: float | None,
    ci_lo: float | None = None,
    ci_hi: float | None = None,
    level: float = DEFAULT_LEVEL,
    p_raw: float | None = None,
    p_adjusted: float | None = None,
    effect_size: Any = None,
    n_boot: int | None = None,
    n_perm: int | None = None,
    available: bool = True,
    reason: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Canonical comparison output: n, estimate, 95% CI, p-values, effect size."""
    payload: dict[str, Any] = {
        "name": name,
        "n": int(n),
        "estimate": None if estimate is None or not np.isfinite(estimate) else float(estimate),
        "ci_lo": None if ci_lo is None or not np.isfinite(ci_lo) else float(ci_lo),
        "ci_hi": None if ci_hi is None or not np.isfinite(ci_hi) else float(ci_hi),
        "level": float(level),
        "p_raw": None if p_raw is None or not np.isfinite(p_raw) else float(p_raw),
        "p_adjusted": None if p_adjusted is None or not np.isfinite(p_adjusted) else float(p_adjusted),
        "effect_size": effect_size,
        "available": bool(available),
        "reason": reason,
        "pairing": "query-level",
    }
    if n_boot is not None:
        payload["n_boot"] = int(n_boot)
    if n_perm is not None:
        payload["n_perm"] = int(n_perm)
    payload.update(extra)
    return attach_ci_fields(payload)


def percentile_ci(
    samples: Sequence[float] | np.ndarray,
    *,
    point: float | None,
    level: float = DEFAULT_LEVEL,
    n_boot: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Percentile interval from already-drawn bootstrap replicates."""
    arr = np.asarray(list(samples), dtype=float)
    ok = arr[np.isfinite(arr)]
    alpha = 1.0 - float(level)
    lo = _quantile(ok, alpha / 2.0)
    hi = _quantile(ok, 1.0 - alpha / 2.0)
    finite_point = point is not None and np.isfinite(point)
    payload = {
        "n_valid_boot": int(ok.size),
        "estimate": float(point) if finite_point else None,
        "ci_lo": lo,
        "ci_hi": hi,
        "level": float(level),
        "method": "percentile_bootstrap",
        "available": bool(ok.size > 0 and finite_point),
    }
    if n_boot is not None:
        payload["n_boot"] = int(n_boot)
    if seed is not None:
        payload["seed"] = int(seed)
    if ok.size == 0:
        payload["reason"] = "no finite bootstrap replicates"
    return payload


def _chunk_rows(n_boot: int, n: int, *, max_elems: int = 2_000_000) -> int:
    if n <= 0:
        return max(1, n_boot)
    return max(1, min(int(n_boot), int(max_elems // n)))


def draw_paired_bootstrap_stats(
    d: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Resample paired differences; return bootstrap draws for core statistics."""
    n = int(d.size)
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[np.ndarray]] = {
        "mean": [],
        "median": [],
        "cohens_dz": [],
        "cliffs_delta": [],
    }
    remaining = int(n_boot)
    while remaining > 0:
        rows = min(remaining, _chunk_rows(remaining, n))
        idx = rng.integers(0, n, size=(rows, n), endpoint=False)
        star = d[idx]
        buckets["mean"].append(star.mean(axis=1))
        buckets["median"].append(np.median(star, axis=1))
        m = star.mean(axis=1)
        if n > 1:
            sd = star.std(axis=1, ddof=1)
        else:
            sd = np.full(rows, np.nan)
        dz = np.full(rows, np.nan)
        ok_sd = np.isfinite(sd) & (sd > 1e-15)
        np.divide(m, sd, out=dz, where=ok_sd)
        zero_var = np.isfinite(sd) & (sd <= 1e-15) & (np.abs(m) <= 1e-15)
        dz[zero_var] = 0.0
        buckets["cohens_dz"].append(dz)
        buckets["cliffs_delta"].append(
            np.mean(star > 0.0, axis=1) - np.mean(star < 0.0, axis=1)
        )
        remaining -= rows
    return {k: np.concatenate(v) if v else np.asarray([], dtype=float) for k, v in buckets.items()}


def paired_bootstrap(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    statistic: StatFn,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
    name: str = "paired_statistic",
) -> dict[str, Any]:
    """Percentile CI for a statistic of paired ``(a, b)`` by resampling queries."""
    aa, bb, meta = finite_pairs(a, b)
    n = meta["n"]
    if n == 0:
        return estimate_payload(
            name=name,
            n=0,
            estimate=None,
            level=level,
            n_boot=n_boot,
            available=False,
            reason="no finite pairs",
            n_dropped=meta["n_dropped"],
            n_input=meta["n_input"],
            method="percentile_bootstrap",
            seed=seed,
        )
    point = statistic(aa, bb)
    if n <= 1 or n_boot <= 0:
        return estimate_payload(
            name=name,
            n=n,
            estimate=None if point is None else float(point) if np.isfinite(point) else None,
            ci_lo=None if point is None else float(point) if np.isfinite(point) else None,
            ci_hi=None if point is None else float(point) if np.isfinite(point) else None,
            level=level,
            n_boot=n_boot,
            available=point is not None and np.isfinite(point),
            reason="degenerate bootstrap (n<=1 or n_boot<=0); CI collapsed to the point",
            n_dropped=meta["n_dropped"],
            n_valid_boot=0,
            method="percentile_bootstrap",
            seed=seed,
            difference="a - b",
        )

    rng = np.random.default_rng(seed)
    boots: list[float] = []
    n_nonfinite = 0
    remaining = int(n_boot)
    while remaining > 0:
        rows = min(remaining, _chunk_rows(remaining, n))
        idx = rng.integers(0, n, size=(rows, n), endpoint=False)
        for row in idx:
            val = statistic(aa[row], bb[row])
            if val is not None and np.isfinite(val):
                boots.append(float(val))
            else:
                n_nonfinite += 1
        remaining -= rows
    ci = percentile_ci(boots, point=None if point is None else float(point) if np.isfinite(point) else None,
                       level=level, n_boot=n_boot, seed=seed)
    payload = estimate_payload(
        name=name,
        n=n,
        estimate=ci["estimate"],
        ci_lo=ci["ci_lo"],
        ci_hi=ci["ci_hi"],
        level=level,
        n_boot=n_boot,
        available=ci["available"],
        reason=ci.get("reason"),
        n_dropped=meta["n_dropped"],
        n_valid_boot=ci["n_valid_boot"],
        method="percentile_bootstrap",
        seed=seed,
        difference="a - b",
    )
    payload["n_boot_requested"] = int(n_boot)
    payload["n_nonfinite_replicates"] = n_nonfinite
    if n_nonfinite:
        payload["reason"] = (
            (payload.get("reason") or "")
            + f"; dropped {n_nonfinite}/{n_boot} non-finite bootstrap replicates"
        ).strip("; ")
    return payload


def paired_bootstrap_mean(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    **kwargs: Any,
) -> dict[str, Any]:
    return paired_bootstrap(a, b, mean_paired_difference, name="mean_paired_difference", **kwargs)


def paired_bootstrap_median(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    **kwargs: Any,
) -> dict[str, Any]:
    return paired_bootstrap(a, b, median_paired_difference, name="median_paired_difference", **kwargs)


def paired_bootstrap_effects(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> dict[str, Any]:
    """Paired bootstrap CIs for mean, median, Cohen's dz, and paired Cliff's delta."""
    aa, bb, meta = finite_pairs(a, b)
    n = meta["n"]
    header = {
        "n": n,
        "n_dropped": meta["n_dropped"],
        "n_input": meta["n_input"],
        "pairing": "query-level",
        "difference": "a - b",
        "n_boot": int(n_boot),
        "seed": int(seed),
        "level": float(level),
        "significance_note": SIGNIFICANCE_NOTE,
    }
    dz_pt = cohens_dz(aa, bb) if n else {"estimate": None, "available": False, "reason": "no finite pairs"}
    cliff_pt = cliffs_delta(aa, bb, paired=True) if n else {"estimate": None, "available": False}

    def _one(name: str, estimate: float | None, samples: np.ndarray | None, extra_effect: Any, reason: str | None = None) -> dict[str, Any]:
        if n == 0:
            return estimate_payload(
                name=name, n=0, estimate=None, level=level, n_boot=n_boot,
                available=False, reason="no finite pairs", seed=seed,
                effect_size=extra_effect, method="percentile_bootstrap",
            )
        if n <= 1 or n_boot <= 0:
            est = None if estimate is None or not np.isfinite(estimate) else float(estimate)
            return estimate_payload(
                name=name, n=n, estimate=est, ci_lo=est, ci_hi=est, level=level,
                n_boot=n_boot, available=est is not None, seed=seed,
                effect_size=extra_effect, method="percentile_bootstrap",
                reason=reason or "degenerate bootstrap (n<=1 or n_boot<=0); CI collapsed to the point",
                n_valid_boot=0,
            )
        ci = percentile_ci(samples if samples is not None else [], point=estimate, level=level, n_boot=n_boot, seed=seed)
        if estimate is None:
            ci["available"] = False
            ci["ci_lo"] = None
            ci["ci_hi"] = None
            ci["reason"] = reason
        return estimate_payload(
            name=name, n=n, estimate=ci.get("estimate"), ci_lo=ci.get("ci_lo"),
            ci_hi=ci.get("ci_hi"), level=level, n_boot=n_boot,
            available=ci.get("available", False), reason=ci.get("reason") or reason,
            seed=seed, effect_size=extra_effect, method="percentile_bootstrap",
            n_valid_boot=ci.get("n_valid_boot"), n_dropped=meta["n_dropped"],
        )

    if n == 0 or n <= 1 or n_boot <= 0:
        mean_pt = None if n == 0 else float((aa - bb).mean())
        median_pt = None if n == 0 else float(np.median(aa - bb))
        return {
            **header,
            "mean_paired_difference": _one(
                "mean_paired_difference", mean_pt, None,
                {"cohens_dz": dz_pt.get("estimate"), "cliffs_delta": cliff_pt.get("estimate")},
            ),
            "median_paired_difference": _one(
                "median_paired_difference", median_pt, None,
                {"cliffs_delta": cliff_pt.get("estimate")},
            ),
            "cohens_dz": _one("cohens_dz", dz_pt.get("estimate"), None, {"cohens_dz": dz_pt.get("estimate")}, dz_pt.get("reason")),
            "cliffs_delta": _one("cliffs_delta", cliff_pt.get("estimate"), None, {"cliffs_delta": cliff_pt.get("estimate")}),
        }

    d = aa - bb
    draws = draw_paired_bootstrap_stats(d, n_boot=n_boot, seed=seed)
    mean_pt = float(d.mean())
    median_pt = float(np.median(d))
    return {
        **header,
        "mean_paired_difference": _one(
            "mean_paired_difference", mean_pt, draws["mean"],
            {"cohens_dz": dz_pt.get("estimate"), "cliffs_delta": cliff_pt.get("estimate")},
        ),
        "median_paired_difference": _one(
            "median_paired_difference", median_pt, draws["median"],
            {"cliffs_delta": cliff_pt.get("estimate")},
        ),
        "cohens_dz": _one(
            "cohens_dz", dz_pt.get("estimate"), draws["cohens_dz"],
            {"cohens_dz": dz_pt.get("estimate")}, dz_pt.get("reason"),
        ),
        "cliffs_delta": _one(
            "cliffs_delta", cliff_pt.get("estimate"), draws["cliffs_delta"],
            {"cliffs_delta": cliff_pt.get("estimate")},
        ),
    }

"""Summaries of latency spans: percentiles, MAD, bootstrap CIs.

Works on instrumented records and on legacy ``latency_s`` totals. Unobserved
spans stay missing; they are not filled from tokens or other proxies.
"""

from __future__ import annotations

import csv
from io import StringIO
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from woais_experiments.frozen import write_result
from woais_experiments.latency.timing import (
    SPAN_FIELDS,
    LatencyBreakdown,
    breakdown_from_legacy_total,
    breakdown_from_log_row,
)

DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 20260909
DEFAULT_LEVEL = 0.95
SUMMARY_STATS = ("mean", "std", "mad", "p50", "p90", "p95", "p99")


def _finite(xs: Iterable[float | None]) -> np.ndarray:
    out = []
    for x in xs:
        if x is None:
            continue
        v = float(x)
        if np.isfinite(v):
            out.append(v)
    return np.asarray(out, dtype=float)


def percentile(xs: Sequence[float] | np.ndarray, q: float) -> float | None:
    """Nearest-rank percentile, same index rule as ``accounting.costs._percentile``.

    ``idx = round((q/100) * (n-1))``. Empty input → None.
    """
    arr = np.asarray(list(xs), dtype=float)
    if arr.size == 0:
        return None
    ys = np.sort(arr)
    if ys.size == 1:
        return float(ys[0])
    idx = int(round((float(q) / 100.0) * (ys.size - 1)))
    idx = min(ys.size - 1, max(0, idx))
    return float(ys[idx])


def median(xs: Sequence[float] | np.ndarray) -> float | None:
    arr = _finite(xs)
    if arr.size == 0:
        return None
    return float(np.median(arr))


def mean(xs: Sequence[float] | np.ndarray) -> float | None:
    arr = _finite(xs)
    if arr.size == 0:
        return None
    return float(arr.mean())


def std(xs: Sequence[float] | np.ndarray, *, ddof: int = 1) -> float | None:
    arr = _finite(xs)
    if arr.size == 0:
        return None
    if arr.size == 1:
        return 0.0
    return float(arr.std(ddof=ddof))


def mad(xs: Sequence[float] | np.ndarray) -> float | None:
    """Median absolute deviation from the median (unscaled)."""
    arr = _finite(xs)
    if arr.size == 0:
        return None
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def _stat(xs: np.ndarray, name: str) -> float | None:
    if name == "mean":
        return mean(xs)
    if name == "std":
        return std(xs)
    if name == "mad":
        return mad(xs)
    if name.startswith("p"):
        return percentile(xs, float(name[1:]))
    raise ValueError(f"unknown statistic {name!r}")


def bootstrap_ci(
    xs: Sequence[float] | np.ndarray,
    stat: Callable[[np.ndarray], float | None] | str,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> dict[str, Any]:
    """Percentile bootstrap CI. Degenerate (point = lo = hi) when n≤1 or zero variance."""
    arr = _finite(xs)
    fn: Callable[[np.ndarray], float | None]
    if isinstance(stat, str):
        fn = lambda a, s=stat: _stat(a, s)
    else:
        fn = stat
    point = fn(arr)
    payload = {
        "point": point,
        "lo": point,
        "hi": point,
        "n": int(arr.size),
        "n_boot": int(n_boot),
        "level": float(level),
        "seed": int(seed),
    }
    if point is None or arr.size <= 1 or n_boot <= 0:
        return payload
    rng = np.random.default_rng(seed)
    lo_q = 100.0 * (1.0 - level) / 2.0
    hi_q = 100.0 * (1.0 + level) / 2.0
    boots = np.empty(n_boot, dtype=float)
    n = arr.size
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        val = fn(sample)
        boots[i] = float("nan") if val is None else float(val)
    ok = boots[np.isfinite(boots)]
    if ok.size == 0:
        return payload
    payload["lo"] = percentile(ok, lo_q)
    payload["hi"] = percentile(ok, hi_q)
    return payload


def summarize(
    xs: Sequence[float | None] | np.ndarray,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
    ci_stats: Sequence[str] = ("mean", "p50", "p95", "p99"),
) -> dict[str, Any]:
    vals = list(xs)
    arr = _finite(vals)
    n_missing = len(vals) - int(arr.size)
    out: dict[str, Any] = {
        "n": int(arr.size),
        "n_missing": n_missing,
        "mean": mean(arr),
        "std": std(arr),
        "mad": mad(arr),
        "p50": percentile(arr, 50),
        "p90": percentile(arr, 90),
        "p95": percentile(arr, 95),
        "p99": percentile(arr, 99),
        "min": float(arr.min()) if arr.size else None,
        "max": float(arr.max()) if arr.size else None,
        "ci": {},
    }
    for name in ci_stats:
        out["ci"][name] = bootstrap_ci(arr, name, n_boot=n_boot, seed=seed, level=level)
    return out


def field_series(records: Sequence[LatencyBreakdown], name: str) -> list[float | None]:
    if name not in SPAN_FIELDS:
        raise ValueError(f"not a latency span: {name}")
    return [getattr(r, name) for r in records]


def summarize_records(
    records: Sequence[LatencyBreakdown],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> dict[str, Any]:
    by_field = {}
    for name in SPAN_FIELDS:
        by_field[name] = summarize(
            field_series(records, name), n_boot=n_boot, seed=seed, level=level
        )
    sources = {}
    for r in records:
        sources[r.source] = sources.get(r.source, 0) + 1
    return {
        "n_records": len(records),
        "sources": sources,
        "by_span": by_field,
    }


def records_from_log_rows(rows: Iterable[Mapping[str, Any]]) -> list[LatencyBreakdown]:
    return [breakdown_from_log_row(r) for r in rows]


def records_from_latency_s(
    query_ids: Sequence[Any],
    latency_s: Sequence[float],
    *,
    model: str | None = None,
    policy: str | None = None,
    selected_tier: int | None = None,
) -> list[LatencyBreakdown]:
    if len(query_ids) != len(latency_s):
        raise ValueError("query_ids and latency_s length mismatch")
    return [
        breakdown_from_legacy_total(
            latency_s=float(lat),
            query_id=str(qid),
            model=model,
            policy=policy,
            selected_tier=selected_tier,
        )
        for qid, lat in zip(query_ids, latency_s)
    ]


def per_query_rows(records: Sequence[LatencyBreakdown]) -> list[dict[str, Any]]:
    return [r.as_dict() for r in records]


def rows_to_csv(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> str:
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for k in row:
                if k not in seen and k != "marks_ns":
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k) for k in fieldnames})
    return buf.getvalue()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        if np.isnan(obj):
            return None
        return "Infinity" if obj > 0 else "-Infinity"
    return obj


def save_latency_tables(
    records: Sequence[LatencyBreakdown],
    *,
    relpath: str = "latency/instrumented",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> dict[str, str]:
    rows = per_query_rows(records)
    summary = summarize_records(records, n_boot=n_boot, seed=seed, level=level)
    payload = {
        "honesty": {
            "clock": "Local spans use time.perf_counter_ns().",
            "legacy": (
                "Frozen EcoLogic logs expose only latency_s (client HTTP round-trip). "
                "TTFT, queue wait, and generation are not inferred."
            ),
            "provider": "Queue/TTFT/generation are copied only when the API returns them.",
        },
        "aggregates": summary,
        "per_query": rows,
    }
    json_path = write_result(f"{relpath}.json", _jsonable(payload))
    csv_fields = [
        "query_id",
        "model",
        "policy",
        "selected_tier",
        "source",
        *SPAN_FIELDS,
        "note",
    ]
    csv_path = write_result(f"{relpath}_per_query.csv", rows_to_csv(rows, csv_fields))
    agg_path = write_result(f"{relpath}_aggregates.json", _jsonable(summary))
    return {"json": str(json_path), "csv": str(csv_path), "aggregates": str(agg_path)}

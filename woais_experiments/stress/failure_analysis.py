"""Classify request failures for the stress suite. Does not invent error types."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

TAXONOMY = (
    "ok",
    "timeout",
    "http_429",
    "http_5xx",
    "http_4xx",
    "connection_error",
    "malformed_response",
    "cost_cap_stop",
    "other_error",
)


def classify_row(row: Mapping[str, Any]) -> str:
    if row.get("cost_cap_stopped") or row.get("error_type") == "cost_cap_stop":
        return "cost_cap_stop"
    err = row.get("error_type")
    try:
        status = int(row.get("http_status") or 0)
    except (TypeError, ValueError):
        status = 0
    if err == "timeout" or status in {408, 504}:
        return "timeout"
    if err == "http_429" or status == 429:
        return "http_429"
    if err in {"http_5xx", "provider_5xx"} or status >= 500:
        return "http_5xx"
    if err == "connection_error":
        return "connection_error"
    if err == "malformed_response":
        return "malformed_response"
    if err in {"http_4xx", "http_error"} or 400 <= status < 500:
        return "http_4xx"
    if err:
        return "other_error"
    if status and status >= 400:
        return "http_4xx" if status < 500 else "http_5xx"
    return "ok"


def is_failure(row: Mapping[str, Any]) -> bool:
    return classify_row(row) != "ok"


def is_timeout(row: Mapping[str, Any]) -> bool:
    return classify_row(row) == "timeout"


def retried(row: Mapping[str, Any]) -> bool:
    try:
        return int(row.get("retry_count") or 0) > 0
    except (TypeError, ValueError):
        return False


def summarize_failures(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    counts = Counter(classify_row(r) for r in rows)
    n_fail = sum(1 for r in rows if is_failure(r))
    n_timeout = sum(1 for r in rows if is_timeout(r))
    n_retry = sum(1 for r in rows if retried(r))
    n_cap = int(counts.get("cost_cap_stop") or 0)
    return {
        "n": n,
        "n_ok": int(counts.get("ok") or 0),
        "n_failed": n_fail,
        "failure_rate": (n_fail / n) if n else None,
        "timeout_rate": (n_timeout / n) if n else None,
        "retry_rate": (n_retry / n) if n else None,
        "cost_cap_stop_rate": (n_cap / n) if n else None,
        "counts": {k: int(counts.get(k) or 0) for k in TAXONOMY},
    }

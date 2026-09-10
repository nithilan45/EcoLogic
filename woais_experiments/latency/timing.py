"""High-resolution latency marks for *future* inference runs.

Local timestamps are ``time.perf_counter_ns()``. Spans that the provider API
does not expose are left ``None`` — they are never inferred from tokens, total
latency, or other proxies.

Legacy logs that store only ``latency_s`` populate ``end_to_end_ms`` and nothing
else.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator, Mapping

NS_PER_MS = 1_000_000.0
SCHEMA_VERSION = 1

# Canonical span names. Only fill a field when its defining marks (or provider
# key) were observed.
SPAN_FIELDS = (
    "router_decision_ms",
    "request_queue_ms",
    "provider_api_ms",
    "time_to_first_token_ms",
    "generation_ms",
    "end_to_end_ms",
)

# Keys we will copy from a provider payload *if present*. No aliases that
# require interpreting HTTP dates, token counts, or total latency.
PROVIDER_QUEUE_KEYS = (
    "queue_time_ms",
    "queued_ms",
    "request_queue_ms",
)
PROVIDER_TTFT_KEYS = (
    "time_to_first_token_ms",
    "ttft_ms",
)
PROVIDER_GENERATION_KEYS = (
    "generation_ms",
    "output_ms",
)

LEGACY_TOTAL_KEYS = ("latency_s", "latency_ms", "end_to_end_ms")


def now_ns() -> int:
    return time.perf_counter_ns()


def ns_to_ms(ns: int | float | None) -> float | None:
    if ns is None:
        return None
    return float(ns) / NS_PER_MS


def ms_to_ns(ms: int | float | None) -> int | None:
    if ms is None:
        return None
    return int(round(float(ms) * NS_PER_MS))


def seconds_to_ms(seconds: int | float | None) -> float | None:
    if seconds is None:
        return None
    return float(seconds) * 1000.0


def _first_present(payload: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for k in keys:
        if k not in payload or payload[k] is None:
            continue
        try:
            return float(payload[k])
        except (TypeError, ValueError):
            continue
    return None


def span_ms(start_ns: int | None, end_ns: int | None) -> float | None:
    """Elapsed milliseconds between two observed local timestamps."""
    if start_ns is None or end_ns is None:
        return None
    return (int(end_ns) - int(start_ns)) / NS_PER_MS


@dataclass
class LatencyBreakdown:
    """One inference attempt. Missing spans are None, never imputed."""

    query_id: str | None = None
    model: str | None = None
    policy: str | None = None
    selected_tier: int | None = None
    source: str = "instrumented"
    schema_version: int = SCHEMA_VERSION

    router_decision_ms: float | None = None
    request_queue_ms: float | None = None
    provider_api_ms: float | None = None
    time_to_first_token_ms: float | None = None
    generation_ms: float | None = None
    end_to_end_ms: float | None = None

    marks_ns: dict[str, int] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    note: str = ""

    def __post_init__(self) -> None:
        missing = [name for name in SPAN_FIELDS if getattr(self, name) is None]
        self.missing = missing

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["marks_ns"] = {k: int(v) for k, v in self.marks_ns.items()}
        return d


class InferenceTimer:
    """Named ``perf_counter_ns`` marks around router + provider I/O.

    Expected marks
    --------------
    router_start, router_end
        Local router / classifier. Optional for direct-model calls.
    request_start, response_end
        Client-side HTTP (or SDK) round trip. This *is* ``provider_api_ms``.
    first_token, last_token
        Only if the caller observed stream chunks. Never synthesised.

    ``request_queue_ms`` is not a local mark. Set it with
    :meth:`set_provider_queue_ms` when the API actually returns queue time.
    """

    def __init__(self) -> None:
        self.marks: dict[str, int] = {}
        self._provider_queue_ms: float | None = None
        self._provider_ttft_ms: float | None = None
        self._provider_generation_ms: float | None = None

    def mark(self, name: str) -> int:
        ns = now_ns()
        self.marks[name] = ns
        return ns

    def has(self, name: str) -> bool:
        return name in self.marks

    def span(self, start: str, end: str) -> float | None:
        if start not in self.marks or end not in self.marks:
            return None
        return span_ms(self.marks[start], self.marks[end])

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        self.mark(f"{name}_start")
        try:
            yield
        finally:
            self.mark(f"{name}_end")

    def note_first_token(self) -> int:
        """Record the first streamed chunk. No-op if already marked."""
        if "first_token" not in self.marks:
            return self.mark("first_token")
        return self.marks["first_token"]

    def note_last_token(self) -> int:
        return self.mark("last_token")

    def set_provider_queue_ms(self, value: float | None) -> None:
        self._provider_queue_ms = None if value is None else float(value)

    def absorb_provider_payload(self, payload: Mapping[str, Any] | None) -> None:
        """Copy queue / TTFT / generation only when the payload contains them."""
        if not payload:
            return
        q = _first_present(payload, PROVIDER_QUEUE_KEYS)
        if q is not None:
            self._provider_queue_ms = q
        ttft = _first_present(payload, PROVIDER_TTFT_KEYS)
        if ttft is not None:
            self._provider_ttft_ms = ttft
        gen = _first_present(payload, PROVIDER_GENERATION_KEYS)
        if gen is not None:
            self._provider_generation_ms = gen

    def to_breakdown(
        self,
        *,
        query_id: str | None = None,
        model: str | None = None,
        policy: str | None = None,
        selected_tier: int | None = None,
        note: str = "",
    ) -> LatencyBreakdown:
        router_ms = self.span("router_start", "router_end")
        api_ms = self.span("request_start", "response_end")
        ttft_local = self.span("request_start", "first_token")
        gen_end = "last_token" if "last_token" in self.marks else "response_end"
        generation_local = (
            self.span("first_token", gen_end) if "first_token" in self.marks else None
        )
        e2e_start = "router_start" if "router_start" in self.marks else "request_start"
        e2e_ms = self.span(e2e_start, "response_end")

        ttft = self._provider_ttft_ms if self._provider_ttft_ms is not None else ttft_local
        generation = (
            self._provider_generation_ms
            if self._provider_generation_ms is not None
            else generation_local
        )
        return LatencyBreakdown(
            query_id=query_id,
            model=model,
            policy=policy,
            selected_tier=selected_tier,
            source="instrumented",
            router_decision_ms=router_ms,
            request_queue_ms=self._provider_queue_ms,
            provider_api_ms=api_ms,
            time_to_first_token_ms=ttft,
            generation_ms=generation,
            end_to_end_ms=e2e_ms,
            marks_ns=dict(self.marks),
            note=note,
        )


def time_callable(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Run ``fn`` and return ``(result, elapsed_ms)`` from ``perf_counter_ns``."""
    t0 = now_ns()
    result = fn(*args, **kwargs)
    t1 = now_ns()
    return result, (t1 - t0) / NS_PER_MS


def breakdown_from_legacy_total(
    *,
    latency_s: float | None = None,
    latency_ms: float | None = None,
    query_id: str | None = None,
    model: str | None = None,
    policy: str | None = None,
    selected_tier: int | None = None,
) -> LatencyBreakdown:
    """Existing logs: only a total wall-clock is known.

    ``latency_s`` is the frozen EcoLogic field (full HTTP round-trip). It is
    mapped to ``end_to_end_ms`` and not copied into TTFT, queue, or generation.
    """
    e2e = latency_ms
    if e2e is None and latency_s is not None:
        e2e = seconds_to_ms(latency_s)
    return LatencyBreakdown(
        query_id=query_id,
        model=model,
        policy=policy,
        selected_tier=selected_tier,
        source="legacy_latency_s",
        end_to_end_ms=e2e,
        note="only total wall-clock was recorded; other spans were not exposed",
    )


def breakdown_from_log_row(row: Mapping[str, Any]) -> LatencyBreakdown:
    """Prefer instrumented fields; otherwise fall back to ``latency_s`` / ``latency_ms``.

    Token counts and HTTP metadata are ignored as latency sources.
    """
    present = {name: row[name] for name in SPAN_FIELDS if name in row and row[name] is not None}
    if present:
        kwargs = {name: (float(present[name]) if name in present else None) for name in SPAN_FIELDS}
        return LatencyBreakdown(
            query_id=_as_str(row.get("query_id") or row.get("item_id")),
            model=_as_str(row.get("model")),
            policy=_as_str(row.get("policy")),
            selected_tier=_as_int(row.get("tier") if row.get("selected_tier") is None else row.get("selected_tier")),
            source=str(row.get("source") or "instrumented"),
            marks_ns=dict(row["marks_ns"]) if isinstance(row.get("marks_ns"), dict) else {},
            note=str(row.get("note") or ""),
            **kwargs,
        )
    lat_ms = row.get("latency_ms")
    lat_s = row.get("latency_s")
    return breakdown_from_legacy_total(
        latency_s=None if lat_s is None else float(lat_s),
        latency_ms=None if lat_ms is None else float(lat_ms),
        query_id=_as_str(row.get("query_id") or row.get("item_id")),
        model=_as_str(row.get("model")),
        policy=_as_str(row.get("policy")),
        selected_tier=_as_int(row.get("tier") if row.get("selected_tier") is None else row.get("selected_tier")),
    )


def _as_str(v: Any) -> str | None:
    if v is None:
        return None
    return str(v)


def _as_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    return int(v)

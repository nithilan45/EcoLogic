"""Open-loop load generator for MEASURED endpoint tests.

Sleeps until each scheduled arrival, then submits into a bounded thread pool.
Never used for SIMULATED runs. Cost-cap stops the whole cell immediately.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Mapping, Sequence

from woais_experiments.stress import MEASURED

SubmitFn = Callable[[int, Mapping[str, Any]], dict[str, Any]]


class CostCapExceeded(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        realized_usd: float,
        projected_usd: float,
        limit_usd: float,
    ) -> None:
        super().__init__(message)
        self.realized_usd = float(realized_usd)
        self.projected_usd = float(projected_usd)
        self.limit_usd = float(limit_usd)

    def as_dict(self) -> dict[str, Any]:
        return {
            "error": "cost_cap_exceeded",
            "realized_usd": self.realized_usd,
            "projected_usd": self.projected_usd,
            "limit_usd": self.limit_usd,
            "message": str(self),
        }


class CostGuard:
    """Stop before realized or projected USD exceeds ``max_cost_usd``.

    Paid runs must construct this with a finite limit. Dry-run / simulated
    runs may pass ``max_cost_usd=None`` to disable the cap.
    """

    def __init__(
        self,
        max_cost_usd: float | None,
        *,
        estimate_usd_per_query: float | None,
        paid: bool,
    ) -> None:
        if paid and max_cost_usd is None:
            raise ValueError("paid runs require --max-cost-usd")
        if max_cost_usd is not None and float(max_cost_usd) < 0:
            raise ValueError("--max-cost-usd must be >= 0")
        self.limit = None if max_cost_usd is None else float(max_cost_usd)
        self.estimate = None if estimate_usd_per_query is None else float(estimate_usd_per_query)
        self.paid = bool(paid)
        self.realized = 0.0
        self.n_costed = 0
        self.remaining_planned = 0
        self._lock = threading.Lock()

    def add_planned(self, n: int) -> None:
        with self._lock:
            self.remaining_planned += int(n)

    def unit_cost(self) -> float:
        with self._lock:
            if self.n_costed > 0:
                return self.realized / self.n_costed
            if self.estimate is not None:
                return self.estimate
            return 0.0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "realized_usd": self.realized,
                "n_costed": self.n_costed,
                "remaining_planned": self.remaining_planned,
                "limit_usd": self.limit,
                "estimate_usd_per_query": self.estimate,
                "unit_cost_usd": (self.realized / self.n_costed) if self.n_costed else self.estimate,
            }

    def projected(self) -> float:
        with self._lock:
            unit = (self.realized / self.n_costed) if self.n_costed else (self.estimate or 0.0)
            return self.realized + max(0, self.remaining_planned) * unit

    def check_ahead(self, extra_requests: int) -> None:
        """Project ``extra_requests`` at the current unit cost (future cells)."""
        if self.limit is None:
            return
        extra = max(0, int(extra_requests))
        with self._lock:
            unit = (self.realized / self.n_costed) if self.n_costed else (self.estimate or 0.0)
            realized = self.realized
            projected = realized + extra * unit
            limit = self.limit
        if realized > limit or projected > limit:
            raise CostCapExceeded(
                f"cost cap ${limit:.6f} exceeded (realized=${realized:.6f}, projected=${projected:.6f})",
                realized_usd=realized,
                projected_usd=projected,
                limit_usd=limit,
            )

    def check(self, *, extra_cost: float = 0.0) -> None:
        if self.limit is None:
            return
        with self._lock:
            unit = (self.realized / self.n_costed) if self.n_costed else (self.estimate or 0.0)
            realized = self.realized + float(extra_cost)
            projected = realized + max(0, self.remaining_planned) * unit
            limit = self.limit
        if realized > limit or projected > limit:
            raise CostCapExceeded(
                f"cost cap ${limit:.6f} exceeded (realized=${realized:.6f}, projected=${projected:.6f})",
                realized_usd=realized,
                projected_usd=projected,
                limit_usd=limit,
            )

    def record(self, cost: float | None, *, consume_planned: bool = True) -> None:
        add = 0.0 if cost is None else float(cost)
        with self._lock:
            if add:
                self.realized += add
                self.n_costed += 1
            if consume_planned:
                self.remaining_planned = max(0, self.remaining_planned - 1)
            realized = self.realized
            limit = self.limit
        if limit is not None and realized > limit:
            raise CostCapExceeded(
                f"realized cost ${realized:.6f} exceeded cap ${limit:.6f}",
                realized_usd=realized,
                projected_usd=realized,
                limit_usd=limit,
            )


def _cost_of(row: Mapping[str, Any]) -> float | None:
    for key in ("realized_provider_cost", "cost_usd", "inference_cost"):
        v = row.get(key)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def run_open_loop(
    arrivals_s: Sequence[float],
    payloads: Sequence[Mapping[str, Any]],
    submit: SubmitFn,
    *,
    max_workers: int,
    cost_guard: CostGuard | None = None,
    origin: float | None = None,
) -> dict[str, Any]:
    """Issue one request per arrival time. Returns MEASURED request rows.

    ``submit(index, payload)`` must return a dict. This function adds stress
    timing fields and never writes SIMULATED keys.
    """
    if len(arrivals_s) != len(payloads):
        raise ValueError("arrivals and payloads length mismatch")
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    n = len(arrivals_s)
    arrivals = [float(t) for t in arrivals_s]
    stop = threading.Event()
    lock = threading.Lock()
    in_flight = 0
    cap_error: CostCapExceeded | None = None
    t0 = time.perf_counter() if origin is None else float(origin)
    rows: list[dict[str, Any] | None] = [None] * n
    futures: list[Future[None]] = []

    def worker(i: int, payload: Mapping[str, Any], queued: int) -> None:
        nonlocal in_flight, cap_error
        t_start = time.perf_counter() - t0
        row: dict[str, Any]
        try:
            row = dict(submit(i, payload))
        except Exception as exc:  # noqa: BLE001
            row = {
                "measurement_type": MEASURED,
                "request_id": str(payload.get("request_id") or f"stress-{i}"),
                "error_type": "connection_error",
                "http_status": 0,
                "retry_count": 0,
                "end_to_end_ms": (time.perf_counter() - t0 - t_start) * 1000.0,
                "submit_exception": type(exc).__name__,
            }
        t_done = time.perf_counter() - t0
        scheduled = arrivals[i]
        row.setdefault("measurement_type", MEASURED)
        if row.get("measurement_type") != MEASURED:
            raise ValueError("open-loop generator only records MEASURED rows")
        row["scheduled_arrival_s"] = scheduled
        row["dispatch_s"] = t_start
        row["complete_s"] = t_done
        row["queue_depth_at_arrival"] = int(queued)
        row["pool_queue_wait_s"] = max(0.0, t_start - scheduled)
        row["queue_wait_s"] = max(0.0, t_start - scheduled)
        sojourn = max(0.0, t_done - scheduled)
        row["sojourn_s"] = sojourn
        row["end_to_end_s"] = sojourn
        if row.get("end_to_end_ms") is None:
            row["end_to_end_ms"] = sojourn * 1000.0
        svc = row.get("provider_request_ms")
        if svc is None:
            svc = row.get("generation_ms")
        row["service_time_s"] = None if svc is None else float(svc) / 1000.0
        rdec = row.get("router_decision_ms")
        row["router_decision_s"] = None if rdec is None else float(rdec) / 1000.0
        row["suite"] = "stress_measured"
        if payload.get("stress_profile") is not None:
            row.setdefault("stress_profile", payload["stress_profile"])
        if payload.get("load_multiplier") is not None:
            row.setdefault("load_multiplier", payload["load_multiplier"])
        if payload.get("canonical_policy") is not None:
            row.setdefault("policy", payload["canonical_policy"])
        elif payload.get("policy") is not None:
            row.setdefault("policy", payload["policy"])
        cost = _cost_of(row)
        with lock:
            in_flight = max(0, in_flight - 1)
        if cost_guard is not None:
            try:
                cost_guard.record(cost, consume_planned=True)
            except CostCapExceeded as exc:
                row["cost_cap_stopped"] = True
                row["error_type"] = row.get("error_type") or "cost_cap_stop"
                stop.set()
                with lock:
                    if cap_error is None:
                        cap_error = exc
        rows[i] = row

    if cost_guard is not None:
        cost_guard.add_planned(n)
        try:
            cost_guard.check()
        except CostCapExceeded:
            for _ in range(n):
                try:
                    cost_guard.record(None, consume_planned=True)
                except CostCapExceeded:
                    pass
            raise

    with ThreadPoolExecutor(max_workers=int(max_workers)) as pool:
        for i, scheduled in enumerate(arrivals):
            if stop.is_set():
                break
            now = time.perf_counter() - t0
            delay = scheduled - now
            if delay > 0:
                time.sleep(delay)
            if stop.is_set():
                break
            if cost_guard is not None:
                try:
                    cost_guard.check()
                except CostCapExceeded as exc:
                    stop.set()
                    with lock:
                        cap_error = exc
                    break
            with lock:
                queued = in_flight
                in_flight += 1
            futures.append(pool.submit(worker, i, payloads[i], queued))
        for fut in futures:
            fut.result()

    with lock:
        leftover = 0
        if cost_guard is not None and cap_error is not None:
            leftover = cost_guard.remaining_planned
            # Unissued requests were still counted as planned.
            for _ in range(leftover):
                try:
                    cost_guard.record(None, consume_planned=True)
                except CostCapExceeded:
                    pass

    issued = [r for r in rows if r is not None]
    for i, row in enumerate(rows):
        if row is not None:
            continue
        # Unissued because the cap stopped the cell.
        rows[i] = {
            "measurement_type": MEASURED,
            "suite": "stress_measured",
            "request_id": str(payloads[i].get("request_id") or f"stress-{i}"),
            "policy": payloads[i].get("canonical_policy") or payloads[i].get("policy"),
            "stress_profile": payloads[i].get("stress_profile"),
            "load_multiplier": payloads[i].get("load_multiplier"),
            "scheduled_arrival_s": arrivals[i],
            "dispatch_s": None,
            "complete_s": None,
            "queue_depth_at_arrival": None,
            "queue_wait_s": None,
            "end_to_end_ms": None,
            "end_to_end_s": None,
            "sojourn_s": None,
            "http_status": 0,
            "error_type": "cost_cap_stop",
            "cost_cap_stopped": True,
            "retry_count": 0,
            "realized_provider_cost": None,
        }
    out_rows = [r for r in rows if r is not None]
    return {
        "measurement_type": MEASURED,
        "n_scheduled": n,
        "n_issued": len(issued),
        "n_logged": len(out_rows),
        "max_workers": int(max_workers),
        "cost_cap_exceeded": cap_error.as_dict() if cap_error is not None else None,
        "cost": None if cost_guard is None else cost_guard.snapshot(),
        "records": out_rows,
        "stopped_on_cost_cap": cap_error is not None,
    }

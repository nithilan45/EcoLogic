"""Named traffic profiles for deployment-real. Local tests use short idle gaps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Workload:
    name: str
    kind: str
    concurrency: int
    n_requests: int | None = None  # None = all executed queries
    idle_s: float = 0.0
    optional: bool = False
    note: str = ""


def default_workloads(
    *,
    n_queries: int,
    include_concurrency_8: bool = True,
    burst_n: int | None = None,
    idle_s: float = 20.0,
) -> list[Workload]:
    n = max(1, int(n_queries))
    burst = int(burst_n) if burst_n is not None else min(16, max(10, min(20, n)))
    burst = min(burst, n)
    idle_n = min(16, n)
    out = [
        Workload(
            name="sequential_warm",
            kind="sequential",
            concurrency=1,
            n_requests=n,
            note="one request at a time after process start (warm after first)",
        ),
        Workload(
            name="concurrency_4",
            kind="concurrent",
            concurrency=4,
            n_requests=n,
            note="four in-flight requests",
        ),
    ]
    if include_concurrency_8:
        out.append(
            Workload(
                name="concurrency_8",
                kind="concurrent",
                concurrency=8,
                n_requests=n,
                optional=True,
                note="eight in-flight requests; skipped when --skip-concurrency-8",
            )
        )
    out.append(
        Workload(
            name="burst",
            kind="burst",
            concurrency=max(burst, 1),
            n_requests=burst,
            note=f"simultaneous burst of {burst} requests from the frozen ID list",
        )
    )
    out.append(
        Workload(
            name="idle_then_batch",
            kind="idle_gap",
            concurrency=1,
            n_requests=idle_n,
            idle_s=float(idle_s),
            note=(
                "idle gap then a new batch. Cold start is labeled only if the "
                "process/container actually restarts; high latency is not a proxy."
            ),
        )
    )
    return out


def as_dicts(workloads: list[Workload]) -> list[dict[str, Any]]:
    return [
        {
            "name": w.name,
            "kind": w.kind,
            "concurrency": w.concurrency,
            "n_requests": w.n_requests,
            "idle_s": w.idle_s,
            "optional": w.optional,
            "note": w.note,
        }
        for w in workloads
    ]

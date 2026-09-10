"""Process/container lifecycle labels.

Cold vs warm is taken from the observed process (pid + request index since
import / start). Idle gaps do **not** relabel a live process as cold. Latency
is never used as a cold-start proxy.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

COLD = "cold"
WARM = "warm"
BASIS_FIRST = "first_request_in_process"
BASIS_SUBSEQUENT = "subsequent_request_same_process"


@dataclass(frozen=True)
class LifecycleObservation:
    lifecycle_state: str
    lifecycle_basis: str
    process_id: int
    request_index_in_process: int
    process_uptime_s: float
    process_start_monotonic: float
    init_type: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "lifecycle_state": self.lifecycle_state,
            "lifecycle_basis": self.lifecycle_basis,
            "process_id": self.process_id,
            "request_index_in_process": self.request_index_in_process,
            "process_uptime_s": self.process_uptime_s,
            "process_start_monotonic": self.process_start_monotonic,
            "init_type": self.init_type,
        }


class ProcessLifecycle:
    """One instance per process. Thread-safe request indexing."""

    def __init__(self) -> None:
        self.process_id = os.getpid()
        self.process_start_monotonic = time.monotonic()
        self._index = 0
        self._lock = threading.Lock()
        self.init_type = os.environ.get("AWS_LAMBDA_INITIALIZATION_TYPE")

    def observe(self) -> LifecycleObservation:
        with self._lock:
            self._index += 1
            index = self._index
        uptime = time.monotonic() - self.process_start_monotonic
        if index == 1:
            state, basis = COLD, BASIS_FIRST
        else:
            state, basis = WARM, BASIS_SUBSEQUENT
        return LifecycleObservation(
            lifecycle_state=state,
            lifecycle_basis=basis,
            process_id=self.process_id,
            request_index_in_process=index,
            process_uptime_s=uptime,
            process_start_monotonic=self.process_start_monotonic,
            init_type=self.init_type,
        )

    @property
    def request_count(self) -> int:
        with self._lock:
            return self._index


def platform_instance_id(*, process_id: int | None = None) -> str:
    """Container/process identity from env or pid. Never inferred from latency."""
    pid = int(process_id or os.getpid())
    lambda_stream = os.environ.get("AWS_LAMBDA_LOG_STREAM_NAME")
    if lambda_stream:
        return f"lambda:{lambda_stream}"
    k_rev = os.environ.get("K_REVISION")
    k_svc = os.environ.get("K_SERVICE")
    if k_rev or k_svc:
        host = socket.gethostname()
        return f"cloudrun:{k_svc or 'unknown'}:{k_rev or 'unknown'}:{host}:pid{pid}"
    return f"pid:{pid}"


def platform_region() -> str | None:
    return (
        os.environ.get("CLOUD_RUN_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or None
    )


def platform_cloud_backend() -> str | None:
    if os.environ.get("K_SERVICE") or os.environ.get("K_REVISION"):
        return "cloudrun"
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME") or os.environ.get("AWS_LAMBDA_RUNTIME_API"):
        return "lambda"
    return None


def cold_start_observed(obs: LifecycleObservation) -> bool:
    """True only from process/container init — never from high latency."""
    if obs.init_type and str(obs.init_type).lower() in {"on-demand", "cold"}:
        return obs.request_index_in_process == 1
    return obs.lifecycle_state == COLD and obs.lifecycle_basis == BASIS_FIRST and obs.request_index_in_process == 1


def keep_lifecycle_if_same_process(life: ProcessLifecycle) -> ProcessLifecycle:
    """Reuse the tracker unless the OS pid actually changed.

    Constructing a new ``ProcessLifecycle`` in the same Python process is not a
    platform cold start. Idle HTTP-server restarts must call this instead of
    ``ProcessLifecycle()``.
    """
    if int(life.process_id) == int(os.getpid()):
        return life
    return ProcessLifecycle()

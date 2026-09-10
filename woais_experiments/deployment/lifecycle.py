"""Process/container lifecycle labels.

Cold vs warm is taken from the observed process (pid + request index since
import / start). Idle gaps do **not** relabel a live process as cold. Latency
is never used as a cold-start proxy.
"""

from __future__ import annotations

import os
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

"""SIMULATED per-model serverless pool: warm instances, cold starts, concurrency.

Nothing here is a cloud measurement. Idle timeout, cold-start delay, max
instances, and per-instance concurrency are all required constructor arguments
— this module does not pick "reasonable" values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

LABEL = "SIMULATED"


@dataclass(frozen=True)
class ServerlessParams:
    max_instances: int
    max_concurrency_per_instance: int
    idle_timeout_s: float
    cold_start_delay_s: float
    first_request_cold: bool
    scale_to_zero: bool
    queue_discipline: str

    def __post_init__(self) -> None:
        if self.max_instances < 1:
            raise ValueError("max_instances must be >= 1")
        if self.max_concurrency_per_instance < 1:
            raise ValueError("max_concurrency_per_instance must be >= 1")
        if self.idle_timeout_s < 0:
            raise ValueError("idle_timeout_s must be >= 0")
        if self.cold_start_delay_s < 0:
            raise ValueError("cold_start_delay_s must be >= 0")
        if str(self.queue_discipline).lower() != "fifo":
            raise ValueError(
                f"queue_discipline={self.queue_discipline!r} is not implemented; YAML must use fifo"
            )


def params_from_mapping(raw: dict[str, Any]) -> ServerlessParams:
    needed = (
        "max_instances",
        "max_concurrency_per_instance",
        "idle_timeout_s",
        "cold_start_delay_s",
        "first_request_cold",
        "scale_to_zero",
        "queue_discipline",
    )
    missing = [k for k in needed if k not in raw or raw[k] is None]
    if missing:
        raise KeyError(f"serverless YAML missing {missing} (no implicit defaults)")
    return ServerlessParams(
        max_instances=int(raw["max_instances"]),
        max_concurrency_per_instance=int(raw["max_concurrency_per_instance"]),
        idle_timeout_s=float(raw["idle_timeout_s"]),
        cold_start_delay_s=float(raw["cold_start_delay_s"]),
        first_request_cold=bool(raw["first_request_cold"]),
        scale_to_zero=bool(raw["scale_to_zero"]),
        queue_discipline=str(raw["queue_discipline"]),
    )


@dataclass
class Instance:
    iid: int
    model: str
    in_flight: int = 0
    warm: bool = False
    creating: bool = False
    last_finish_s: float | None = None
    idle_seq: int = 0
    n_served: int = 0
    n_cold_starts: int = 0
    alive: bool = True


@dataclass
class Admit:
    """Result of trying to place one request on the pool."""

    queued: bool
    instance_id: int | None
    cold: bool
    t_ready: float | None  # when service may begin (after cold start if any)
    note: str = ""


@dataclass
class Pull:
    request_id: int
    instance_id: int
    cold: bool
    t_ready: float


class ModelPool:
    """One autoscaling pool for a single model."""

    def __init__(self, model: str, params: ServerlessParams):
        self.model = model
        self.params = params
        self.instances: list[Instance] = []
        self.retired: list[Instance] = []
        self.queue: list[int] = []
        self._next_iid = 0
        self.n_scale_outs = 0
        self.n_cold_starts = 0

    def n_alive(self) -> int:
        return sum(1 for i in self.instances if i.alive)

    def max_instances_used(self) -> int:
        return max((len(self.instances),), default=0)

    def requests_per_instance(self) -> list[int]:
        return [i.n_served for i in self.instances] + [i.n_served for i in self.retired]

    def _bump_idle(self, inst: Instance) -> None:
        inst.idle_seq += 1

    def _pick_warm_slot(self) -> Instance | None:
        p = self.params
        cands = [
            i
            for i in self.instances
            if i.alive and i.warm and not i.creating and i.in_flight < p.max_concurrency_per_instance
        ]
        if not cands:
            return None
        cands.sort(key=lambda i: (i.in_flight, i.iid))
        return cands[0]

    def _pick_cold_idle(self) -> Instance | None:
        for i in self.instances:
            if i.alive and (not i.warm) and (not i.creating) and i.in_flight == 0:
                return i
        return None

    def try_admit(self, now: float, request_id: int) -> Admit:
        p = self.params
        warm = self._pick_warm_slot()
        if warm is not None:
            warm.in_flight += 1
            self._bump_idle(warm)
            return Admit(queued=False, instance_id=warm.iid, cold=False, t_ready=now)

        cold = self._pick_cold_idle()
        if cold is not None:
            return self._begin_cold(cold, now)

        if self.n_alive() < p.max_instances:
            inst = Instance(iid=self._next_iid, model=self.model)
            self._next_iid += 1
            self.instances.append(inst)
            self.n_scale_outs += 1
            first_ever = self.n_scale_outs == 1
            pay_cold = (not first_ever) or p.first_request_cold
            if not pay_cold:
                inst.warm = True
                inst.creating = False
                inst.in_flight = 1
                return Admit(queued=False, instance_id=inst.iid, cold=False, t_ready=now, note="first_instance_warm")
            return self._begin_cold(inst, now)

        self.queue.append(request_id)
        return Admit(queued=True, instance_id=None, cold=False, t_ready=None)

    def _begin_cold(self, inst: Instance, now: float) -> Admit:
        inst.creating = True
        inst.warm = False
        inst.in_flight = 1
        inst.n_cold_starts += 1
        self.n_cold_starts += 1
        self._bump_idle(inst)
        ready = now + self.params.cold_start_delay_s
        return Admit(queued=False, instance_id=inst.iid, cold=True, t_ready=ready)

    def on_cold_ready(self, instance_id: int, now: float) -> None:
        inst = self._by_id(instance_id)
        inst.creating = False
        inst.warm = True

    def fill_all(self, instance_id: int, now: float) -> list[Pull]:
        out: list[Pull] = []
        while True:
            pulled = self._fill_from_queue(self._by_id(instance_id), now)
            if pulled is None:
                break
            out.append(pulled)
        return out

    def on_service_complete(self, instance_id: int, now: float) -> list[Pull]:
        inst = self._by_id(instance_id)
        if inst.in_flight <= 0:
            raise RuntimeError("service complete on idle instance")
        inst.in_flight -= 1
        inst.n_served += 1
        inst.last_finish_s = now
        return self.fill_all(instance_id, now)

    def idle_seq(self, instance_id: int) -> int:
        return self._by_id(instance_id).idle_seq

    def on_idle_expire(self, instance_id: int, seq: int, now: float) -> bool:
        """Return True if the instance actually went cold / scaled in."""
        inst = self._by_id(instance_id)
        if seq != inst.idle_seq or inst.in_flight != 0 or not inst.alive:
            return False
        inst.warm = False
        if self.params.scale_to_zero:
            inst.alive = False
            self.retired.append(inst)
        return True

    def _fill_from_queue(self, inst: Instance, now: float) -> Pull | None:
        if not self.queue:
            return None
        if not inst.alive or not inst.warm or inst.creating:
            return None
        if inst.in_flight >= self.params.max_concurrency_per_instance:
            return None
        rid = self.queue.pop(0)
        inst.in_flight += 1
        self._bump_idle(inst)
        return Pull(request_id=rid, instance_id=inst.iid, cold=False, t_ready=now)

    def _by_id(self, iid: int) -> Instance:
        for i in self.instances:
            if i.iid == iid:
                return i
        raise KeyError(f"instance {iid} not in pool {self.model}")

    def snapshot(self) -> dict[str, Any]:
        return {
            "SIMULATED": True,
            "model": self.model,
            "n_alive": self.n_alive(),
            "n_created": len(self.instances),
            "n_scale_outs": self.n_scale_outs,
            "n_cold_starts": self.n_cold_starts,
            "queue_depth": len(self.queue),
            "requests_per_instance": self.requests_per_instance(),
            "params": {
                "max_instances": self.params.max_instances,
                "max_concurrency_per_instance": self.params.max_concurrency_per_instance,
                "idle_timeout_s": self.params.idle_timeout_s,
                "cold_start_delay_s": self.params.cold_start_delay_s,
                "first_request_cold": self.params.first_request_cold,
                "scale_to_zero": self.params.scale_to_zero,
                "queue_discipline": self.params.queue_discipline,
            },
        }

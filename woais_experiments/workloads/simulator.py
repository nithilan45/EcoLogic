"""Discrete-event inference workload simulator. All outputs are SIMULATED.

Each request: arrive → (optional router queue + overhead) → model pool
(queue / cold start / service) → complete.

This is not a measured cloud deployment. Queueing, cold starts, and
concurrency are synthetic. Empirical ``latency_s`` used as service time already
embeds historical HTTP delay; simulated queueing is *additional*.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from woais_experiments.workloads.arrival_processes import arrivals_from_config
from woais_experiments.workloads.serverless_model import ModelPool, ServerlessParams, params_from_mapping

LABEL = "SIMULATED"
DISCLAIMER = (
    "SIMULATED discrete-event run, not a measured cloud deployment. "
    "Idle timeout, cold-start delay, concurrency, and router overhead are "
    "YAML parameters, not observations."
)


def _nested_get(cfg: Mapping[str, Any], path: str) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise KeyError(f"config.{path} must be specified in YAML (missing {part})")
        cur = cur[part]
    if cur is None:
        raise KeyError(f"config.{path} is null; YAML must set it explicitly")
    return cur


def validate_config(cfg: Mapping[str, Any]) -> None:
    if not bool(_nested_get(cfg, "simulated")):
        raise ValueError("config.simulated must be true: this module only produces SIMULATED output")
    for path in (
        "seed",
        "n_requests",
        "arrivals.process",
        "arrivals.start_s",
        "router.max_concurrency",
        "sla.max_e2e_s",
        "catalog.source",
        "catalog.cheap_model",
        "catalog.strong_model",
        "serverless.max_instances",
        "serverless.max_concurrency_per_instance",
        "serverless.idle_timeout_s",
        "serverless.cold_start_delay_s",
        "serverless.first_request_cold",
        "serverless.scale_to_zero",
        "serverless.queue_discipline",
        "policies",
        "policy_order",
    ):
        _nested_get(cfg, path)
    if int(_nested_get(cfg, "router.max_concurrency")) < 1:
        raise ValueError("router.max_concurrency must be >= 1")
    process = str(_nested_get(cfg, "arrivals.process")).lower()
    if process == "constant":
        _nested_get(cfg, "arrivals.interarrival_s")
    elif process == "poisson":
        _nested_get(cfg, "arrivals.rate_per_s")
    elif process in {"on_off", "bursty", "bursty_on_off"}:
        _nested_get(cfg, "arrivals.on_off.lambda_on_per_s")
        _nested_get(cfg, "arrivals.on_off.lambda_off_per_s")
        _nested_get(cfg, "arrivals.on_off.mean_on_s")
        _nested_get(cfg, "arrivals.on_off.mean_off_s")
        _nested_get(cfg, "arrivals.on_off.initial_state")
    elif process == "trace":
        _nested_get(cfg, "arrivals.timestamps_s")
    else:
        raise ValueError(f"unknown arrivals.process {process!r}")


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return float(ys[0])
    idx = min(len(ys) - 1, max(0, round((q / 100.0) * (len(ys) - 1))))
    return float(ys[idx])


def _summary(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None, "min": None, "max": None}
    return {
        "n": len(xs),
        "mean": float(sum(xs) / len(xs)),
        "p50": _pct(xs, 50),
        "p95": _pct(xs, 95),
        "p99": _pct(xs, 99),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


@dataclass
class Catalog:
    """Query × model service time and cost. Assignments are optional."""

    query_ids: list[str]
    models: list[str]
    cheap_model: str
    strong_model: str
    service_s: dict[tuple[str, str], float]
    cost: dict[tuple[str, str], float]
    assignments: dict[str, dict[str, str]] = field(default_factory=dict)
    service_time_note: str = (
        "SIMULATED service times. If sourced from frozen latency_s, that field "
        "already includes historical HTTP delay; simulated queues add more."
    )

    def mean_cost(self, model: str) -> float:
        vals = [self.cost[(model, q)] for q in self.query_ids]
        return float(sum(vals) / len(vals)) if vals else 0.0


def catalog_synthetic(cfg: Mapping[str, Any]) -> Catalog:
    cat = _nested_get(cfg, "catalog")
    cheap = str(cat["cheap_model"])
    strong = str(cat["strong_model"])
    models_cfg = _nested_get(cat, "models")
    models = list(models_cfg.keys())
    if cheap not in models_cfg or strong not in models_cfg:
        raise KeyError("catalog.cheap_model and strong_model must appear under catalog.models")
    if "query_ids" in cat and cat["query_ids"] is not None:
        qids = [str(q) for q in cat["query_ids"]]
    else:
        n = int(_nested_get(cfg, "n_requests"))
        qids = [f"q{i}" for i in range(n)]
    service: dict[tuple[str, str], float] = {}
    cost: dict[tuple[str, str], float] = {}
    for m, spec in models_cfg.items():
        if "service_s" not in spec or spec["service_s"] is None:
            raise KeyError(f"catalog.models.{m}.service_s must be set")
        if "cost" not in spec or spec["cost"] is None:
            raise KeyError(f"catalog.models.{m}.cost must be set")
        for q in qids:
            service[(str(m), q)] = float(spec["service_s"])
            cost[(str(m), q)] = float(spec["cost"])
    assigns: dict[str, dict[str, str]] = {}
    raw_a = cat.get("assignments") or {}
    for pname, mapping in raw_a.items():
        assigns[str(pname)] = {str(k): str(v) for k, v in mapping.items()}
    return Catalog(
        query_ids=qids,
        models=[str(m) for m in models],
        cheap_model=cheap,
        strong_model=strong,
        service_s=service,
        cost=cost,
        assignments=assigns,
        service_time_note="SIMULATED constant per-model service_s from YAML.",
    )


def catalog_from_item_matrix(
    cfg: Mapping[str, Any],
    matrix: Any,
    routing: Mapping[str, Any] | None,
    oracle_tiers: Mapping[str, int] | None,
    ecologic_tiers: Mapping[str, int] | None,
) -> Catalog:
    """Build a catalog from frozen Stage 1–2 measurements (read-only)."""
    cat = _nested_get(cfg, "catalog")
    cheap = str(cat["cheap_model"])
    strong = str(cat["strong_model"])
    tier_of = {1: "t1", 2: "t2", 3: "t3"}
    models = ["t1", "t2", "t3"]
    qids = list(matrix.item_ids)
    service: dict[tuple[str, str], float] = {}
    cost: dict[tuple[str, str], float] = {}
    for t, name in tier_of.items():
        for q in qids:
            service[(name, q)] = float(matrix.latency_s[(t, q)])
            cost[(name, q)] = float(matrix.usd[(t, q)])
    assigns: dict[str, dict[str, str]] = {}
    if ecologic_tiers is not None:
        assigns["ecologic"] = {str(q): tier_of[int(ecologic_tiers[q])] for q in qids}
    if oracle_tiers is not None:
        assigns["oracle"] = {str(q): tier_of[int(oracle_tiers[q])] for q in qids}
    if cheap not in models or strong not in models:
        raise KeyError("empirical cheap_model/strong_model must be t1, t2, or t3")
    return Catalog(
        query_ids=qids,
        models=models,
        cheap_model=cheap,
        strong_model=strong,
        service_s=service,
        cost=cost,
        assignments=assigns,
        service_time_note=(
            "SIMULATED. service_s is frozen latency_s (full historical HTTP round-trip). "
            "The DES still adds simulated queueing, cold starts, and router overhead."
        ),
    )


@dataclass
class Request:
    rid: int
    query_id: str
    model: str
    arrival_s: float
    service_s: float
    cost: float
    router_overhead_s: float
    router_cost_usd: float
    t_router_start: float | None = None
    t_router_done: float | None = None
    t_dispatch: float | None = None
    t_service_start: float | None = None
    t_complete: float | None = None
    queued: bool = False
    cold_start: bool = False
    instance_id: int | None = None
    queue_delay_s: float = 0.0
    cold_delay_s: float = 0.0


@dataclass(order=True)
class _E:
    time: float
    seq: int
    kind: str = field(compare=False)
    data: dict = field(compare=False, default_factory=dict)


def _policy_cfg(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    policies = _nested_get(cfg, "policies")
    if name not in policies:
        raise KeyError(f"policies.{name} must be specified in YAML")
    p = policies[name]
    for key in ("type", "router_overhead_s", "router_cost_usd"):
        if key not in p or p[key] is None:
            raise KeyError(f"policies.{name}.{key} must be specified (no implicit default)")
    return p


def mixture_f_strong(catalog: Catalog, cfg: Mapping[str, Any], policy_name: str) -> dict[str, Any]:
    p = _policy_cfg(cfg, policy_name)
    cheap_c = catalog.mean_cost(catalog.cheap_model)
    strong_c = catalog.mean_cost(catalog.strong_model)
    if "f_strong" in p and p["f_strong"] is not None:
        f = float(p["f_strong"])
        return {"f_strong": f, "clipped": False, "source": "yaml_f_strong", "cheap_mean": cheap_c, "strong_mean": strong_c}
    if "match_to" not in p or p["match_to"] is None:
        raise KeyError(
            f"policies.{policy_name} must set f_strong or match_to (no implicit mixture weight)"
        )
    target = str(p["match_to"])
    if target not in catalog.assignments:
        raise KeyError(f"cannot cost-match: assignments[{target!r}] unavailable")
    assign = catalog.assignments[target]
    costs = [catalog.cost[(assign[q], q)] for q in catalog.query_ids if q in assign]
    target_mean = float(sum(costs) / len(costs))
    denom = strong_c - cheap_c
    if abs(denom) < 1e-18:
        f = 0.0
        clipped = True
    else:
        f = (target_mean - cheap_c) / denom
        clipped = f < 0.0 or f > 1.0
        f = min(1.0, max(0.0, f))
    return {
        "f_strong": f,
        "clipped": clipped,
        "source": f"match_to:{target}",
        "target_mean_cost": target_mean,
        "cheap_mean": cheap_c,
        "strong_mean": strong_c,
    }


class DiscreteEventSimulator:
    def __init__(
        self,
        cfg: Mapping[str, Any],
        catalog: Catalog,
        policy_name: str,
        arrivals_s: np.ndarray,
        query_seq: list[str],
        rng: np.random.Generator,
    ):
        validate_config(cfg)
        if len(arrivals_s) != len(query_seq):
            raise ValueError("arrivals and query_seq length mismatch")
        self.cfg = cfg
        self.catalog = catalog
        self.policy_name = policy_name
        self.pcfg = _policy_cfg(cfg, policy_name)
        self.arrivals_s = np.asarray(arrivals_s, dtype=float)
        self.query_seq = query_seq
        self.rng = rng
        self.router_max = int(_nested_get(cfg, "router.max_concurrency"))
        self.sla = float(_nested_get(cfg, "sla.max_e2e_s"))
        self.idle_timeout = float(_nested_get(cfg, "serverless.idle_timeout_s"))
        params = params_from_mapping(dict(_nested_get(cfg, "serverless")))
        self.params = params
        self.pools = {m: ModelPool(m, params) for m in catalog.models}
        self.mix_info = None
        if str(self.pcfg["type"]) == "cost_matched_static":
            self.mix_info = mixture_f_strong(catalog, cfg, policy_name)
        self.events: list[_E] = []
        self.seq = 0
        self.router_in_flight = 0
        self.router_queue: list[int] = []
        self.requests: dict[int, Request] = {}
        self.completed: list[int] = []

    def _push(self, time: float, kind: str, **data: Any) -> None:
        heapq.heappush(self.events, _E(time=float(time), seq=self.seq, kind=kind, data=data))
        self.seq += 1

    def _choose_model(self, qid: str) -> str:
        kind = str(self.pcfg["type"])
        if kind == "always_cheap":
            return self.catalog.cheap_model
        if kind == "always_strong":
            return self.catalog.strong_model
        if kind == "cost_matched_static":
            assert self.mix_info is not None
            if self.rng.random() < self.mix_info["f_strong"]:
                return self.catalog.strong_model
            return self.catalog.cheap_model
        mapping = self.catalog.assignments.get(kind) or self.catalog.assignments.get(self.policy_name)
        if mapping is None:
            raise KeyError(f"policy {self.policy_name!r} type {kind!r} has no assignment map")
        if qid not in mapping:
            raise KeyError(f"no {kind} assignment for query {qid}")
        return mapping[qid]

    def run(self) -> dict[str, Any]:
        for i, (t, qid) in enumerate(zip(self.arrivals_s, self.query_seq)):
            self._push(float(t), "arrival", rid=i, query_id=qid)
        while self.events:
            ev = heapq.heappop(self.events)
            kind = ev.kind
            if kind == "arrival":
                self._on_arrival(ev.time, ev.data["rid"], ev.data["query_id"])
            elif kind == "router_done":
                self._on_router_done(ev.time, ev.data["rid"])
            elif kind == "cold_done":
                self._on_cold_done(ev.time, ev.data["rid"], ev.data["instance_id"], ev.data["model"])
            elif kind == "service_done":
                self._on_service_done(ev.time, ev.data["rid"], ev.data["instance_id"], ev.data["model"])
            elif kind == "idle_expire":
                self._on_idle_expire(ev.time, ev.data["model"], ev.data["instance_id"], ev.data["seq"])
            else:
                raise RuntimeError(f"unknown event {kind}")
        return self._metrics()

    def _on_arrival(self, now: float, rid: int, qid: str) -> None:
        model = self._choose_model(qid)
        overhead = float(self.pcfg["router_overhead_s"])
        rcost = float(self.pcfg["router_cost_usd"])
        req = Request(
            rid=rid,
            query_id=qid,
            model=model,
            arrival_s=now,
            service_s=float(self.catalog.service_s[(model, qid)]),
            cost=float(self.catalog.cost[(model, qid)]),
            router_overhead_s=overhead,
            router_cost_usd=rcost,
        )
        self.requests[rid] = req
        if overhead <= 0:
            req.t_router_start = now
            req.t_router_done = now
            self._dispatch(now, rid)
            return
        if self.router_in_flight < self.router_max:
            self._start_router(now, rid)
        else:
            self.router_queue.append(rid)

    def _start_router(self, now: float, rid: int) -> None:
        req = self.requests[rid]
        req.t_router_start = now
        self.router_in_flight += 1
        self._push(now + req.router_overhead_s, "router_done", rid=rid)

    def _on_router_done(self, now: float, rid: int) -> None:
        req = self.requests[rid]
        req.t_router_done = now
        self.router_in_flight -= 1
        if self.router_queue:
            nxt = self.router_queue.pop(0)
            self._start_router(now, nxt)
        self._dispatch(now, rid)

    def _dispatch(self, now: float, rid: int) -> None:
        req = self.requests[rid]
        pool = self.pools[req.model]
        admit = pool.try_admit(now, rid)
        if admit.queued:
            req.queued = True
            return
        req.instance_id = admit.instance_id
        req.t_dispatch = now
        req.cold_start = admit.cold
        req.queue_delay_s = 0.0
        if admit.cold:
            req.cold_delay_s = float(admit.t_ready) - now
            self._push(float(admit.t_ready), "cold_done", rid=rid, instance_id=admit.instance_id, model=req.model)
        else:
            self._begin_service(now, rid)

    def _begin_service(self, now: float, rid: int) -> None:
        req = self.requests[rid]
        req.t_service_start = now
        if req.t_router_done is not None:
            waited = now - req.t_router_done - req.cold_delay_s
            req.queue_delay_s = max(0.0, waited)
        self._push(now + req.service_s, "service_done", rid=rid, instance_id=req.instance_id, model=req.model)

    def _on_cold_done(self, now: float, rid: int, instance_id: int, model: str) -> None:
        pool = self.pools[model]
        pool.on_cold_ready(instance_id, now)
        self._begin_service(now, rid)
        self._admit_pulls(now, pool.fill_all(instance_id, now))

    def _on_service_done(self, now: float, rid: int, instance_id: int, model: str) -> None:
        req = self.requests[rid]
        req.t_complete = now
        self.completed.append(rid)
        pool = self.pools[model]
        pulls = pool.on_service_complete(instance_id, now)
        inst = pool._by_id(instance_id)
        if inst.in_flight == 0:
            self._push(
                now + self.idle_timeout,
                "idle_expire",
                model=model,
                instance_id=instance_id,
                seq=pool.idle_seq(instance_id),
            )
        self._admit_pulls(now, pulls)

    def _admit_pulls(self, now: float, pulls: list) -> None:
        for pulled in pulls:
            nxt = self.requests[pulled.request_id]
            nxt.queued = True
            nxt.instance_id = pulled.instance_id
            nxt.t_dispatch = now
            nxt.cold_start = pulled.cold
            nxt.cold_delay_s = 0.0
            self._begin_service(pulled.t_ready, pulled.request_id)

    def _on_idle_expire(self, now: float, model: str, instance_id: int, seq: int) -> None:
        self.pools[model].on_idle_expire(instance_id, seq, now)

    def _metrics(self) -> dict[str, Any]:
        reqs = [self.requests[i] for i in self.completed]
        e2e = [r.t_complete - r.arrival_s for r in reqs]  # type: ignore[operator]
        qdel = [r.queue_delay_s for r in reqs]
        cold_n = sum(1 for r in reqs if r.cold_start)
        inf_cost = [r.cost for r in reqs]
        tot_inf = float(sum(inf_cost))
        tot_router = float(sum(r.router_cost_usd for r in reqs))
        tot = tot_inf + tot_router
        sla_v = sum(1 for x in e2e if x > self.sla)
        t0 = float(self.arrivals_s[0]) if len(self.arrivals_s) else 0.0
        t_end = max((r.t_complete or 0.0) for r in reqs) if reqs else t0
        makespan = max(t_end - t0, 0.0)
        horizon = t_end  # from time 0
        n = len(reqs)
        rps = [c for p in self.pools.values() for c in p.requests_per_instance()]
        per_req = []
        for r in reqs:
            per_req.append(
                {
                    "rid": r.rid,
                    "query_id": r.query_id,
                    "model": r.model,
                    "arrival_s": r.arrival_s,
                    "end_to_end_s": (r.t_complete - r.arrival_s) if r.t_complete is not None else None,
                    "queue_delay_s": r.queue_delay_s,
                    "cold_start": r.cold_start,
                    "cold_delay_s": r.cold_delay_s,
                    "service_s": r.service_s,
                    "router_overhead_s": r.router_overhead_s,
                    "inference_cost": r.cost,
                    "router_cost_usd": r.router_cost_usd,
                    "instance_id": r.instance_id,
                    "sla_violated": (
                        (r.t_complete - r.arrival_s) > self.sla if r.t_complete is not None else None
                    ),
                }
            )
        return {
            "SIMULATED": True,
            "label": LABEL,
            "disclaimer": DISCLAIMER,
            "policy": self.policy_name,
            "policy_type": str(self.pcfg["type"]),
            "n_requests": int(len(self.arrivals_s)),
            "n_completed": n,
            "mixture": self.mix_info,
            "metrics": {
                "SIMULATED": True,
                "throughput_per_s": (n / horizon) if horizon > 0 else None,
                "throughput_per_s_makespan": (n / makespan) if makespan > 0 else None,
                "horizon_s": horizon,
                "makespan_s": makespan,
                "queue_delay_s": _summary(qdel),
                "end_to_end_latency_s": _summary(e2e),
                "p50_latency_s": _pct(e2e, 50),
                "p95_latency_s": _pct(e2e, 95),
                "p99_latency_s": _pct(e2e, 99),
                "cold_start_count": int(cold_n),
                "cold_start_rate": (cold_n / n) if n else None,
                "requests_per_instance": _summary([float(x) for x in rps]) if rps else _summary([]),
                "requests_per_instance_raw": rps,
                "cost_per_request": (tot / n) if n else None,
                "inference_cost_per_request": (tot_inf / n) if n else None,
                "total_realized_inference_cost": tot_inf,
                "total_realized_cost_including_router": tot,
                "sla_max_e2e_s": self.sla,
                "sla_violation_rate": (sla_v / n) if n else None,
                "sla_violation_count": sla_v,
                "mean_router_overhead_s": float(sum(r.router_overhead_s for r in reqs) / n) if n else None,
            },
            "pools": {m: p.snapshot() for m, p in self.pools.items()},
            "catalog_note": self.catalog.service_time_note,
            "per_request": per_req,
        }


def simulate(
    cfg: Mapping[str, Any],
    catalog: Catalog,
    policy_name: str,
    *,
    rng: np.random.Generator | None = None,
    arrivals_s: np.ndarray | None = None,
    query_seq: list[str] | None = None,
) -> dict[str, Any]:
    validate_config(cfg)
    seed = int(_nested_get(cfg, "seed"))
    rng = rng or np.random.default_rng(seed)
    n = int(_nested_get(cfg, "n_requests"))
    if arrivals_s is None:
        arrivals_s = arrivals_from_config(_nested_get(cfg, "arrivals"), n, rng)
    if query_seq is None:
        query_seq = [str(x) for x in rng.choice(catalog.query_ids, size=n, replace=True)]
    sim = DiscreteEventSimulator(cfg, catalog, policy_name, arrivals_s, query_seq, rng)
    out = sim.run()
    out["seed"] = seed
    out["arrivals_process"] = str(_nested_get(cfg, "arrivals.process"))
    return out

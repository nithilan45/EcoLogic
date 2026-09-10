"""Arrival processes for the SIMULATED workload generator.

All parameters come from the caller / YAML. This module does not insert
rates, duty cycles, or start times of its own.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

LABEL = "SIMULATED"


def _require(cfg: Mapping[str, Any], key: str) -> Any:
    if key not in cfg or cfg[key] is None:
        raise KeyError(f"arrivals.{key} must be specified in YAML (no implicit default)")
    return cfg[key]


def constant_arrivals(
    n: int,
    interarrival_s: float,
    start_s: float,
) -> np.ndarray:
    """Deterministic spacing. ``start_s`` and ``interarrival_s`` are required."""
    if n < 0:
        raise ValueError("n must be >= 0")
    if interarrival_s < 0:
        raise ValueError("interarrival_s must be >= 0")
    if n == 0:
        return np.zeros(0, dtype=float)
    return float(start_s) + float(interarrival_s) * np.arange(n, dtype=float)


def poisson_arrivals(
    n: int,
    rate_per_s: float,
    rng: np.random.Generator,
    start_s: float,
) -> np.ndarray:
    """Homogeneous Poisson process (exponential interarrivals)."""
    if n < 0:
        raise ValueError("n must be >= 0")
    if rate_per_s <= 0:
        raise ValueError("rate_per_s must be > 0")
    if n == 0:
        return np.zeros(0, dtype=float)
    ia = rng.exponential(1.0 / float(rate_per_s), size=n)
    return float(start_s) + np.cumsum(ia)


def on_off_arrivals(
    n: int,
    *,
    lambda_on_per_s: float,
    lambda_off_per_s: float,
    mean_on_s: float,
    mean_off_s: float,
    initial_state: str,
    rng: np.random.Generator,
    start_s: float,
) -> np.ndarray:
    """Two-state Markov-modulated Poisson (bursty ON/OFF).

    ``lambda_off_per_s`` is required and may be zero (no arrivals while OFF).
    State holding times are exponential with the given means.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if lambda_on_per_s < 0 or lambda_off_per_s < 0:
        raise ValueError("ON/OFF rates must be >= 0")
    if mean_on_s <= 0 or mean_off_s <= 0:
        raise ValueError("mean_on_s and mean_off_s must be > 0")
    state = str(initial_state).upper()
    if state not in {"ON", "OFF"}:
        raise ValueError("initial_state must be ON or OFF (quote it in YAML so it is not a boolean)")
    if n == 0:
        return np.zeros(0, dtype=float)

    t = float(start_s)
    hold = mean_on_s if state == "ON" else mean_off_s
    state_end = t + rng.exponential(hold)
    out = np.empty(n, dtype=float)
    i = 0
    guard = 0
    max_steps = max(10_000, n * 100)
    while i < n:
        guard += 1
        if guard > max_steps:
            raise RuntimeError("ON/OFF generator exceeded step cap; check rates")
        if t >= state_end:
            state = "OFF" if state == "ON" else "ON"
            hold = mean_on_s if state == "ON" else mean_off_s
            state_end = t + rng.exponential(hold)
            continue
        lam = lambda_on_per_s if state == "ON" else lambda_off_per_s
        if lam <= 0:
            t = state_end
            continue
        wait = rng.exponential(1.0 / lam)
        if t + wait > state_end:
            t = state_end
            continue
        t = t + wait
        out[i] = t
        i += 1
    return out


def trace_replay(timestamps_s: list[float] | np.ndarray, n: int) -> np.ndarray:
    """Replay recorded arrival epochs. Does not interpolate or jitter."""
    ts = np.asarray(timestamps_s, dtype=float).reshape(-1)
    if ts.size < n:
        raise ValueError(f"trace has {ts.size} timestamps; need n_requests={n}")
    if ts.size == 0:
        return ts
    if np.any(np.diff(ts) < -1e-15):
        raise ValueError("trace timestamps must be non-decreasing")
    return ts[:n].copy()


def arrivals_from_config(cfg: Mapping[str, Any], n: int, rng: np.random.Generator) -> np.ndarray:
    process = str(_require(cfg, "process")).lower()
    start_s = float(_require(cfg, "start_s"))
    if process == "constant":
        return constant_arrivals(n, float(_require(cfg, "interarrival_s")), start_s)
    if process == "poisson":
        return poisson_arrivals(n, float(_require(cfg, "rate_per_s")), rng, start_s)
    if process in {"on_off", "bursty", "bursty_on_off"}:
        oo = _require(cfg, "on_off")
        if not isinstance(oo, Mapping):
            raise TypeError("arrivals.on_off must be a mapping")
        return on_off_arrivals(
            n,
            lambda_on_per_s=float(_require(oo, "lambda_on_per_s")),
            lambda_off_per_s=float(_require(oo, "lambda_off_per_s")),
            mean_on_s=float(_require(oo, "mean_on_s")),
            mean_off_s=float(_require(oo, "mean_off_s")),
            initial_state=str(_require(oo, "initial_state")),
            rng=rng,
            start_s=start_s,
        )
    if process == "trace":
        return trace_replay(_require(cfg, "timestamps_s"), n)
    raise ValueError(f"unknown arrivals.process {process!r}")

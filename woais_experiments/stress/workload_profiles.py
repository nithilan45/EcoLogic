"""Named stress workload profiles. Deterministic given seed + multiplier.

Load multiplier ``m`` scales arrival *rate* (interarrivals shrink by ``1/m``).
Replay traces are time-compressed the same way. This module never sleeps and
never calls a provider API.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from woais_experiments.workloads.arrival_processes import (
    constant_arrivals,
    on_off_arrivals,
    poisson_arrivals,
    trace_replay,
)

PROFILE_NAMES = (
    "steady_low",
    "steady_medium",
    "steady_high",
    "poisson",
    "bursty_on_off",
    "sudden_spike",
    "ramp_up",
    "ramp_down",
    "periodic_bursts",
    "replay",
)

LOAD_MULTIPLIERS = (1, 2, 5, 10, 25, 50)

# Relative rates at 1x. Multiplier then scales these.
STEADY_LOW_RATE = 1.0
STEADY_MEDIUM_RATE = 4.0
STEADY_HIGH_RATE = 16.0


def _stable_int(text: str) -> int:
    """Process-independent digest. Never use Python's salted ``hash()``."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def profile_seed(base_seed: int, profile: str, multiplier: float) -> int:
    """Stable 32-bit seed so reruns of the same cell match."""
    h = _stable_int(f"{int(base_seed)}|{profile}|{float(multiplier):.6f}")
    return int(h % (2**31 - 1)) or 1


def scale_rate(rate_1x: float, multiplier: float) -> float:
    m = float(multiplier)
    if m <= 0:
        raise ValueError("load multiplier must be > 0")
    return float(rate_1x) * m


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def ramp_arrivals(
    n: int,
    *,
    rate_start: float,
    rate_end: float,
    start_s: float = 0.0,
) -> np.ndarray:
    """Deterministic arrivals under a linear rate path.

    Instantaneous rate λ(t) = λ0 + (λ1-λ0)*t/T with T chosen so the integrated
    count is n: mean λ is (λ0+λ1)/2, T = n / mean_λ.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return np.zeros(0, dtype=float)
    a, b = float(rate_start), float(rate_end)
    if a < 0 or b < 0:
        raise ValueError("ramp rates must be >= 0")
    if a == 0 and b == 0:
        raise ValueError("ramp rates cannot both be 0")
    mean = 0.5 * (a + b)
    duration = n / mean
    # Invert N(t) = a t + (b-a) t^2 / (2T) = k+1
    out = np.empty(n, dtype=float)
    slope = (b - a) / duration
    for k in range(n):
        target = float(k) + 1.0
        if abs(slope) < 1e-18:
            out[k] = target / a
        else:
            # 0.5*slope t^2 + a t - target = 0
            disc = a * a + 2.0 * slope * target
            t = (-a + np.sqrt(max(disc, 0.0))) / slope
            out[k] = t
    return float(start_s) + out


def spike_arrivals(
    n: int,
    *,
    base_rate: float,
    spike_rate: float,
    spike_frac: float = 0.2,
    start_s: float = 0.0,
) -> np.ndarray:
    """Quiet period then a sudden high-rate burst, then quiet again if leftover."""
    if n <= 0:
        return np.zeros(0, dtype=float)
    n_spike = max(1, int(round(n * float(spike_frac))))
    n_pre = max(0, (n - n_spike) // 2)
    n_post = n - n_spike - n_pre
    t = float(start_s)
    out: list[float] = []
    if n_pre:
        pre = constant_arrivals(n_pre, 1.0 / float(base_rate), t)
        out.extend(pre.tolist())
        t = float(pre[-1]) + 1.0 / float(base_rate)
    spike = constant_arrivals(n_spike, 1.0 / float(spike_rate), t)
    out.extend(spike.tolist())
    t = float(spike[-1]) + 1.0 / float(spike_rate)
    if n_post:
        post = constant_arrivals(n_post, 1.0 / float(base_rate), t)
        out.extend(post.tolist())
    return np.asarray(out[:n], dtype=float)


def periodic_burst_arrivals(
    n: int,
    *,
    lambda_on: float,
    period_s: float,
    duty: float,
    rng: np.random.Generator,
    start_s: float = 0.0,
) -> np.ndarray:
    """Deterministic ON windows of length duty*period; Poisson arrivals while ON."""
    if n <= 0:
        return np.zeros(0, dtype=float)
    if not 0.0 < duty <= 1.0:
        raise ValueError("duty must be in (0, 1]")
    if period_s <= 0 or lambda_on <= 0:
        raise ValueError("period_s and lambda_on must be > 0")
    on_s = float(period_s) * float(duty)
    t = float(start_s)
    out = np.empty(n, dtype=float)
    i = 0
    cycle = 0
    guard = 0
    while i < n:
        guard += 1
        if guard > max(10_000, n * 200):
            raise RuntimeError("periodic burst generator exceeded step cap")
        on_start = float(start_s) + cycle * float(period_s)
        on_end = on_start + on_s
        t = max(t, on_start)
        if t >= on_end:
            cycle += 1
            continue
        wait = rng.exponential(1.0 / float(lambda_on))
        if t + wait > on_end:
            cycle += 1
            t = on_end
            continue
        t = t + wait
        out[i] = t
        i += 1
    return out


def replay_arrivals(
    timestamps_s: Sequence[float],
    n: int,
    *,
    multiplier: float,
) -> np.ndarray:
    raw = trace_replay(list(timestamps_s), n)
    t0 = float(raw[0]) if raw.size else 0.0
    shifted = raw - t0
    return shifted / float(multiplier)


def _rates_1x(overrides: Mapping[str, float] | None) -> dict[str, float]:
    out = {"low": STEADY_LOW_RATE, "medium": STEADY_MEDIUM_RATE, "high": STEADY_HIGH_RATE}
    if overrides:
        for key in ("low", "medium", "high"):
            if key in overrides and overrides[key] is not None:
                out[key] = float(overrides[key])
    return out


def arrivals_for_profile(
    profile: str,
    n: int,
    *,
    multiplier: float = 1.0,
    seed: int = 20260909,
    start_s: float = 0.0,
    timestamps_s: Sequence[float] | None = None,
    rates_1x: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Return arrival epochs (seconds) and metadata. No I/O."""
    name = str(profile).lower()
    if name not in PROFILE_NAMES:
        raise ValueError(f"unknown profile {profile!r}; known: {PROFILE_NAMES}")
    m = float(multiplier)
    rates = _rates_1x(rates_1x)
    cell_seed = profile_seed(int(seed), name, m)
    rng = _rng(cell_seed)
    rate: float | None
    if name == "steady_low":
        rate = scale_rate(rates["low"], m)
        ts = constant_arrivals(n, 1.0 / rate, start_s)
        kind = "constant"
    elif name == "steady_medium":
        rate = scale_rate(rates["medium"], m)
        ts = constant_arrivals(n, 1.0 / rate, start_s)
        kind = "constant"
    elif name == "steady_high":
        rate = scale_rate(rates["high"], m)
        ts = constant_arrivals(n, 1.0 / rate, start_s)
        kind = "constant"
    elif name == "poisson":
        rate = scale_rate(rates["medium"], m)
        ts = poisson_arrivals(n, rate, rng, start_s)
        kind = "poisson"
    elif name == "bursty_on_off":
        rate = scale_rate(rates["high"], m)
        ts = on_off_arrivals(
            n,
            lambda_on_per_s=rate,
            lambda_off_per_s=0.0,
            mean_on_s=0.5,
            mean_off_s=0.5,
            initial_state="ON",
            rng=rng,
            start_s=start_s,
        )
        kind = "on_off"
    elif name == "sudden_spike":
        base = scale_rate(rates["low"], m)
        rate = scale_rate(rates["high"], m)
        ts = spike_arrivals(n, base_rate=base, spike_rate=rate, start_s=start_s)
        kind = "spike"
    elif name == "ramp_up":
        ts = ramp_arrivals(
            n,
            rate_start=scale_rate(rates["low"], m),
            rate_end=scale_rate(rates["high"], m),
            start_s=start_s,
        )
        kind = "ramp_up"
        rate = scale_rate(rates["medium"], m)
    elif name == "ramp_down":
        ts = ramp_arrivals(
            n,
            rate_start=scale_rate(rates["high"], m),
            rate_end=scale_rate(rates["low"], m),
            start_s=start_s,
        )
        kind = "ramp_down"
        rate = scale_rate(rates["medium"], m)
    elif name == "periodic_bursts":
        rate = scale_rate(rates["high"], m)
        ts = periodic_burst_arrivals(
            n, lambda_on=rate, period_s=1.0, duty=0.25, rng=rng, start_s=start_s
        )
        kind = "periodic_bursts"
    else:
        if timestamps_s is None:
            raise ValueError("replay profile requires timestamps_s")
        ts = replay_arrivals(timestamps_s, n, multiplier=m)
        ts = ts + float(start_s)
        kind = "replay"
        dur = float(ts[-1] - ts[0]) if ts.size > 1 else 0.0
        rate = (n / dur) if dur > 0 else None

    offered = None
    if ts.size > 1:
        dur = float(ts[-1] - ts[0])
        offered = ((n - 1) / dur) if dur > 0 else None
    return {
        "profile": name,
        "multiplier": m,
        "n": int(n),
        "seed": int(cell_seed),
        "base_seed": int(seed),
        "kind": kind,
        "arrivals_s": ts,
        "nominal_rate_per_s": rate,
        "offered_rate_per_s": offered,
        "horizon_s": float(ts[-1] - ts[0]) if ts.size else 0.0,
    }


def safe_multipliers(
    *,
    mode: str,
    paid: bool,
    max_multiplier: float | None = None,
    requested: Sequence[float] | None = None,
) -> list[float]:
    """Drop levels that would be unsafe for the backend. Never invent extra load.

    Simulated: all requested levels up to 50x.
    Measured stub / dry-run: cap 10x.
    Paid measured: cap 2x unless the operator passes an explicit ``max_multiplier``.
    """
    wanted = [float(x) for x in (requested or LOAD_MULTIPLIERS)]
    mode = str(mode).lower()
    if mode == "simulated":
        default_cap = 50.0
    elif paid:
        default_cap = 2.0
    else:
        default_cap = 10.0
    if max_multiplier is None:
        cap = default_cap
    elif paid and mode == "measured":
        cap = float(max_multiplier)
    else:
        cap = min(default_cap, float(max_multiplier))
    out = [m for m in wanted if m <= cap + 1e-12]
    if not out:
        raise ValueError(f"no safe load multipliers for mode={mode} paid={paid} cap={cap}")
    return out

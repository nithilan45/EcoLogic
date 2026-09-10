"""Serverless-style models parameterized by empirical wall-clock service times.

Honesty: TTFT, cold start, and idle timeout were never measured. `latency_s` is
the full HTTP round trip, including provider-side queueing. Cold-start penalties
and idle timeouts here are sensitivity knobs, not observations. Residual-after-
tokens is a *proxy* for unexplained delay (could be cold start, retries, or
load), not a labeled cold-start event.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from woais_experiments.frozen import ItemMatrix


def fit_linear_service(latency: np.ndarray, tokens: np.ndarray) -> dict:
    """latency ≈ alpha + beta * completion_tokens + residual."""
    x = np.asarray(tokens, dtype=float)
    y = np.asarray(latency, dtype=float)
    n = len(x)
    if n < 3 or np.allclose(x, x[0]):
        alpha = float(y.mean()) if n else 0.0
        resid = y - alpha
        return {
            "alpha": alpha, "beta": 0.0, "r2": 0.0,
            "residual_mean": float(resid.mean()) if n else 0.0,
            "residual_std": float(resid.std()) if n else 0.0,
            "residuals": resid,
        }
    A = np.vstack([np.ones(n), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    resid = y - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return {
        "alpha": float(coef[0]),
        "beta": float(coef[1]),
        "r2": r2,
        "residual_mean": float(resid.mean()),
        "residual_std": float(resid.std()),
        "residuals": resid,
    }


def flag_cold_like(residuals: np.ndarray, k: float = 2.0) -> np.ndarray:
    mu = residuals.mean()
    sd = residuals.std()
    if sd == 0:
        return np.zeros(len(residuals), dtype=bool)
    return residuals > (mu + k * sd)


def service_model_by_tier(matrix: ItemMatrix, k: float = 2.0) -> dict:
    out = {}
    for t in (1, 2, 3):
        lat = np.array([matrix.latency_s[(t, i)] for i in matrix.item_ids], dtype=float)
        tok = np.array([matrix.completion_tokens[(t, i)] for i in matrix.item_ids], dtype=float)
        fit = fit_linear_service(lat, tok)
        flags = flag_cold_like(fit["residuals"], k=k)
        rec = {key: val for key, val in fit.items() if key != "residuals"}
        rec["residual_outlier_frac"] = float(flags.mean())
        rec["cold_like_frac"] = rec["residual_outlier_frac"]
        rec["cold_like_n"] = int(flags.sum())
        rec["cold_like_note"] = (
            "residual_outlier_frac is a proxy from latency~tokens residuals, "
            "not a labeled cold-start measurement"
        )
        rec["mean_residual_when_flagged"] = (
            float(fit["residuals"][flags].mean()) if flags.any() else None
        )
        out[str(t)] = rec
    return out


def simulate_fcfs(
    service_times: np.ndarray,
    arrivals: np.ndarray,
    *,
    n_servers: int = 1,
    idle_timeout_s: float | None = None,
    cold_penalty_s: float = 0.0,
    first_request_cold: bool = True,
) -> dict:
    """Open-loop FCFS with optional scale-to-zero idle timeout.

    Each server remembers when it last finished. If the next job finds it idle
    longer than `idle_timeout_s`, `cold_penalty_s` is added before service.
    """
    s = np.asarray(service_times, dtype=float)
    a = np.asarray(arrivals, dtype=float)
    n = len(s)
    next_free = np.zeros(n_servers, dtype=float)
    last_finish = np.full(n_servers, np.nan)
    sojourn = np.zeros(n)
    wait = np.zeros(n)
    cold = np.zeros(n, dtype=bool)
    start = np.zeros(n)
    extra_s = np.zeros(n)

    for i in range(n):
        k = int(np.argmin(next_free))
        ready = max(a[i], next_free[k])
        extra = 0.0
        if idle_timeout_s is not None and cold_penalty_s > 0:
            if np.isnan(last_finish[k]):
                if first_request_cold:
                    extra = cold_penalty_s
                    cold[i] = True
            elif ready - last_finish[k] >= idle_timeout_s:
                extra = cold_penalty_s
                cold[i] = True
        extra_s[i] = extra
        begin = ready + extra
        finish = begin + s[i]
        start[i] = begin
        wait[i] = ready - a[i]
        sojourn[i] = finish - a[i]
        next_free[k] = finish
        last_finish[k] = finish

    horizon = float(max(a[-1], next_free.max())) if n else 0.0
    busy = float((s + extra_s).sum())
    util = busy / (n_servers * horizon) if horizon > 0 else 0.0
    return {
        "n": n,
        "n_servers": n_servers,
        "idle_timeout_s": idle_timeout_s,
        "cold_penalty_s": cold_penalty_s,
        "mean_sojourn_s": float(sojourn.mean()) if n else 0.0,
        "p50_sojourn_s": float(np.quantile(sojourn, 0.50)) if n else 0.0,
        "p95_sojourn_s": float(np.quantile(sojourn, 0.95)) if n else 0.0,
        "p99_sojourn_s": float(np.quantile(sojourn, 0.99)) if n else 0.0,
        "mean_wait_s": float(wait.mean()) if n else 0.0,
        "mean_queue_wait_s": float(wait.mean()) if n else 0.0,
        "mean_cold_s": float(extra_s.mean()) if n else 0.0,
        "cold_frac": float(cold.mean()) if n else 0.0,
        "utilization": util,
        "mean_service_s": float(s.mean()) if n else 0.0,
        "mean_wall_clock_s": float(s.mean()) if n else 0.0,
        "honesty": (
            "SIMULATED FCFS on historical HTTP round-trip times. Those times already "
            "include provider queueing, so adding simulated wait double-counts delay. "
            "Not a deployment measurement. mean_wait_s is queue wait only; cold is "
            "mean_cold_s; utilization includes cold busy time."
        ),
    }


def poisson_arrivals(n: int, rate_per_s: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if rate_per_s <= 0:
        raise ValueError("rate_per_s must be positive")
    ia = rng.exponential(1.0 / rate_per_s, size=n)
    return np.cumsum(ia)


def replay_policy(
    matrix: ItemMatrix,
    assign: dict[str, int],
    item_seq: Iterable[str],
    arrivals: np.ndarray,
    **sim_kwargs,
) -> dict:
    items = list(item_seq)
    service = np.array([matrix.latency_s[(assign[i], i)] for i in items], dtype=float)
    result = simulate_fcfs(service, arrivals, **sim_kwargs)
    result["mean_service_s"] = float(service.mean())
    result["mean_wall_clock_s"] = float(service.mean())
    result["service_time_is_http_rtt"] = True
    return result

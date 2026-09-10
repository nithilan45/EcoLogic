"""Synthetic arrival traces over frozen items. No new generations."""

from __future__ import annotations

import numpy as np

from woais_experiments.latency.serverless import poisson_arrivals


def open_loop_trace(
    item_ids: list[str],
    *,
    n_requests: int,
    arrival_rate_per_s: float,
    seed: int,
) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    seq = [str(x) for x in rng.choice(item_ids, size=n_requests, replace=True)]
    arrivals = poisson_arrivals(n_requests, arrival_rate_per_s, seed=seed + 1)
    return seq, arrivals

"""Wilson intervals and McNemar tests.

Formulas match `benchmark/analyze.py` (`wilson`, `mcnemar`) so published
p-values recompute without importing that module — `analyze` imports `api`,
which imports `httpx` at module load. A comparison test imports the legacy
functions when that stack is present.
"""

from __future__ import annotations

import math

Z = 1.959963984540054  # 95%, same constant as benchmark/analyze.py


def wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denom
    half = Z / denom * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (p, max(0.0, center - half), min(1.0, center + half))


def _binomtest_two_sided(k: int, n: int, p: float = 0.5) -> float:
    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n, p, alternative="two-sided").pvalue)
    except Exception:
        # Exact two-sided p-value: sum of point masses as likely as or less
        # likely than the observed count (matches scipy for p=0.5).
        def pmf(i: int) -> float:
            return math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

        observed = pmf(k)
        return float(min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= observed + 1e-15)))


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar on paired binary outcomes (same as analyze.mcnemar)."""
    b_only = sum(1 for x, y in zip(a, b) if x and not y)
    c_only = sum(1 for x, y in zip(a, b) if y and not x)
    n = b_only + c_only
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0, "note": "no discordant pairs"}
    p = _binomtest_two_sided(b_only, n, 0.5)
    return {"b": b_only, "c": c_only, "p_value": float(p), "note": ""}


def wilson_dict(k: int, n: int) -> dict:
    p, lo, hi = wilson(k, n)
    return {"p": p, "ci_lo": lo, "ci_hi": hi, "k": k, "n": n}


def matched_cost_fraction(cost: float, mean_weak: float, mean_strong: float) -> float:
    """Query-independent strong-call fraction whose expected cost equals `cost`."""
    denom = mean_strong - mean_weak
    if abs(denom) < 1e-18:
        return 0.0
    f = (cost - mean_weak) / denom
    return float(min(1.0, max(0.0, f)))


def mixture_accuracy(f_strong: float, acc_weak: float, acc_strong: float) -> float:
    return f_strong * acc_strong + (1.0 - f_strong) * acc_weak


def mixture_cost(f_strong: float, mean_weak: float, mean_strong: float) -> float:
    return f_strong * mean_strong + (1.0 - f_strong) * mean_weak

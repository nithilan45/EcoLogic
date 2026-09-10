"""Paired effect sizes. No resampling.

Differences are ``a - b`` at the query (pair) level. Missing values are
dropped, never filled. Cohen's ``d_z`` is the paired standardized mean;
Cliff's delta uses pair dominance unless ``paired=False``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

EPS = 1e-15


def as_1d(x: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr


def finite_pairs(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Keep pairs where both values are finite. Lengths must match."""
    aa, bb = as_1d(a), as_1d(b)
    if aa.shape != bb.shape:
        raise ValueError(f"paired series must have the same length; got {aa.size} and {bb.size}")
    ok = np.isfinite(aa) & np.isfinite(bb)
    meta = {
        "n": int(ok.sum()),
        "n_dropped": int((~ok).sum()),
        "n_input": int(aa.size),
    }
    return aa[ok], bb[ok], meta


def pair_by_query_id(
    a: Mapping[Any, float],
    b: Mapping[Any, float],
) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...], dict[str, int]]:
    """Inner-join two query maps. Unmatched queries are dropped, not imputed."""
    shared = [k for k in a.keys() if k in b]
    xs: list[float] = []
    ys: list[float] = []
    kept: list[Any] = []
    for qid in shared:
        x, y = float(a[qid]), float(b[qid])
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
            kept.append(qid)
    meta = {
        "n": len(kept),
        "n_dropped": len(shared) - len(kept),
        "n_input": len(shared),
        "n_unmatched_a": int(sum(1 for k in a if k not in b)),
        "n_unmatched_b": int(sum(1 for k in b if k not in a)),
    }
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), tuple(kept), meta


def differences(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> np.ndarray:
    aa, bb, _ = finite_pairs(a, b)
    return aa - bb


def mean_paired_difference(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float | None:
    d = differences(a, b)
    if d.size == 0:
        return None
    return float(d.mean())


def median_paired_difference(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float | None:
    d = differences(a, b)
    if d.size == 0:
        return None
    return float(np.median(d))


def hodges_lehmann_paired(d: Sequence[float] | np.ndarray, *, max_walsh: int = 5_000_000) -> float | None:
    """Median of Walsh averages ``(d_i + d_j)/2`` for ``i ≤ j``.

    This is the Hodges–Lehmann estimator associated with Wilcoxon signed-rank.
    Falls back to the sample median when the Walsh set would exceed ``max_walsh``.
    """
    arr = np.asarray(d, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return None
    n_walsh = n * (n + 1) // 2
    if n_walsh > int(max_walsh):
        return float(np.median(arr))
    i, j = np.triu_indices(n)
    return float(np.median(0.5 * (arr[i] + arr[j])))


def cohens_dz(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Paired Cohen's ``d_z = mean(d) / sd(d)`` with sample sd (ddof=1)."""
    d = differences(a, b)
    n = int(d.size)
    out: dict[str, Any] = {
        "name": "cohens_dz",
        "n": n,
        "estimate": None,
        "available": False,
        "reason": None,
        "definition": "mean(a-b) / sd(a-b) with sample sd (ddof=1)",
    }
    if n == 0:
        out["reason"] = "no finite pairs"
        return out
    mean = float(d.mean())
    if n == 1:
        out["reason"] = "Cohen's dz needs n>=2 to estimate sd"
        return out
    sd = float(d.std(ddof=1))
    if sd <= EPS:
        if abs(mean) <= EPS:
            out["estimate"] = 0.0
            out["available"] = True
            out["reason"] = "zero variance and zero mean; dz=0"
            return out
        out["reason"] = "Cohen's dz undefined: zero variance of differences with nonzero mean"
        return out
    out["estimate"] = mean / sd
    out["available"] = True
    return out


def cliffs_delta(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    paired: bool = True,
) -> dict[str, Any]:
    """Cliff's delta (dominance).

    Paired (default, query-level): ``(P(a>b) - P(a<b))`` over the n pairs,
    ties counted in the denominator. Unpaired: all ``n_a × n_b`` cross-pairs.
    Unpaired mode ignores pairing and is not appropriate for matched queries.
    """
    if paired:
        aa, bb, meta = finite_pairs(a, b)
        n = meta["n"]
        out: dict[str, Any] = {
            "name": "cliffs_delta",
            "paired": True,
            "n": n,
            "n_dropped": meta["n_dropped"],
            "estimate": None,
            "available": False,
            "n_positive": 0,
            "n_negative": 0,
            "n_ties": 0,
            "definition": "(#{a>b} - #{a<b}) / n  over paired queries; ties in the denominator",
        }
        if n == 0:
            out["reason"] = "no finite pairs"
            return out
        n_pos = int(np.sum(aa > bb))
        n_neg = int(np.sum(aa < bb))
        n_tie = int(n - n_pos - n_neg)
        out.update(
            {
                "estimate": (n_pos - n_neg) / n,
                "available": True,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "n_ties": n_tie,
            }
        )
        return out

    aa = as_1d(a)
    bb = as_1d(b)
    aa = aa[np.isfinite(aa)]
    bb = bb[np.isfinite(bb)]
    n_a, n_b = int(aa.size), int(bb.size)
    out = {
        "name": "cliffs_delta",
        "paired": False,
        "n": n_a * n_b,
        "n_a": n_a,
        "n_b": n_b,
        "estimate": None,
        "available": False,
        "definition": "(#{a_i>b_j} - #{a_i<b_j}) / (n_a n_b); unpaired, not for matched queries",
        "reason": None,
    }
    if n_a == 0 or n_b == 0:
        out["reason"] = "empty group"
        return out
    sign = np.sign(aa[:, None] - bb[None, :])
    out["estimate"] = float(sign.mean())
    out["available"] = True
    out["n_positive"] = int(np.sum(sign > 0))
    out["n_negative"] = int(np.sum(sign < 0))
    out["n_ties"] = int(np.sum(sign == 0))
    return out


def matched_pairs_rank_biserial(d: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Matched-pairs rank-biserial correlation from signed ranks of ``d = a-b``."""
    arr = np.asarray(d, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    nz = arr[np.abs(arr) > EPS]
    n = int(nz.size)
    out: dict[str, Any] = {
        "name": "matched_pairs_rank_biserial",
        "n_nonzero": n,
        "n": int(arr.size),
        "estimate": None,
        "available": False,
        "definition": "(T+ - T-) / (n(n+1)/2) on nonzero paired differences",
    }
    if n == 0:
        out["reason"] = "no nonzero paired differences"
        if arr.size and np.all(np.abs(arr) <= EPS):
            out["estimate"] = 0.0
            out["available"] = True
            out["reason"] = "all paired differences are zero; rank-biserial=0"
        return out
    from scipy.stats import rankdata

    ranks = np.asarray(rankdata(np.abs(nz), method="average"), dtype=float)
    t_plus = float(ranks[nz > 0].sum())
    t_minus = float(ranks[nz < 0].sum())
    denom = n * (n + 1) / 2.0
    out["estimate"] = (t_plus - t_minus) / denom
    out["available"] = True
    out["t_plus"] = t_plus
    out["t_minus"] = t_minus
    return out

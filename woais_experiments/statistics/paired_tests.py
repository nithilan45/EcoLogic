"""Paired hypothesis tests and Benjamini–Hochberg FDR.

Query-level pairing: under the null of exchangeable labels within a query,
signs of ``d_i = a_i - b_i`` are flipped. Wilcoxon signed-rank is reported
when it is defined (at least one nonzero difference). Multiplicity uses
Benjamini–Hochberg on a caller-specified family of raw p-values.

CI overlap is not treated as a test. See ``SIGNIFICANCE_NOTE``.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from woais_experiments.statistics.bootstrap import (
    DEFAULT_LEVEL,
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    SIGNIFICANCE_NOTE,
    estimate_payload,
    paired_bootstrap_effects,
)
from woais_experiments.statistics.effect_sizes import (
    as_1d,
    cliffs_delta,
    finite_pairs,
    hodges_lehmann_paired,
    matched_pairs_rank_biserial,
    pair_by_query_id,
)

DEFAULT_N_PERM = 10_000
EXACT_MAX_N = 16  # 2**16 sign patterns
ALPHA = 0.05
EPS = 1e-15

Stat1D = Callable[[np.ndarray], float]


def _mean_stat(d: np.ndarray) -> float:
    return float(d.mean())


def _median_stat(d: np.ndarray) -> float:
    return float(np.median(d))


def _dz_stat(d: np.ndarray) -> float:
    n = d.size
    if n < 2:
        return float("nan")
    sd = float(d.std(ddof=1))
    m = float(d.mean())
    if sd <= EPS:
        return 0.0 if abs(m) <= EPS else float("nan")
    return m / sd


def _cliff_stat(d: np.ndarray) -> float:
    return float(np.mean(d > 0.0) - np.mean(d < 0.0))


def _two_sided_count(obs: float, nulls: np.ndarray) -> int:
    return int(np.sum(np.abs(nulls) >= abs(obs) - 1e-15))


def _one_sided_count(obs: float, nulls: np.ndarray, *, greater: bool) -> int:
    if greater:
        return int(np.sum(nulls >= obs - 1e-15))
    return int(np.sum(nulls <= obs + 1e-15))


def _p_from_counts(count: int, n_draw: int, *, monte_carlo: bool) -> float:
    if n_draw <= 0:
        return float("nan")
    if monte_carlo:
        return float((1 + count) / (1 + n_draw))
    return float(count / n_draw)


def paired_permutation_test(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    statistic: Stat1D | str = "mean",
    *,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = DEFAULT_SEED,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Sign-flip permutation test of paired differences ``a - b``.

    Exact enumeration when ``n ≤ 16``; otherwise Monte Carlo with
    ``p = (1 + #{extreme}) / (1 + n_perm)``.
    """
    alt = alternative.lower()
    if alt not in {"two-sided", "greater", "less"}:
        raise ValueError(f"unknown alternative {alternative!r}")
    aa, bb, meta = finite_pairs(a, b)
    n = meta["n"]
    fn: Stat1D
    if isinstance(statistic, str):
        fn = {"mean": _mean_stat, "median": _median_stat, "cohens_dz": _dz_stat, "cliffs_delta": _cliff_stat}[statistic]
        stat_name = statistic
    else:
        fn = statistic
        stat_name = getattr(statistic, "__name__", "custom")

    out: dict[str, Any] = {
        "name": f"paired_permutation:{stat_name}",
        "n": n,
        "n_dropped": meta["n_dropped"],
        "alternative": alt,
        "pairing": "query-level",
        "available": False,
        "p_raw": None,
        "estimate": None,
        "method": None,
        "n_perm": int(n_perm),
        "seed": int(seed),
        "significance_note": SIGNIFICANCE_NOTE,
    }
    if n == 0:
        out["reason"] = "no finite pairs"
        return out
    d = aa - bb
    obs = fn(d)
    out["estimate"] = None if not np.isfinite(obs) else float(obs)
    if not np.isfinite(obs):
        out["reason"] = "observed statistic is non-finite"
        return out

    exact = n <= EXACT_MAX_N
    if exact:
        n_pat = 1 << n
        k = np.arange(n_pat, dtype=np.int64)
        bits = ((k[:, None] >> np.arange(n, dtype=np.int64)) & 1) * 2 - 1
        star = bits.astype(float) * d
        if stat_name == "mean":
            nulls = star.mean(axis=1)
        elif stat_name == "median":
            nulls = np.median(star, axis=1)
        elif stat_name == "cliffs_delta":
            nulls = np.mean(star > 0.0, axis=1) - np.mean(star < 0.0, axis=1)
        elif stat_name == "cohens_dz":
            m = star.mean(axis=1)
            sd = star.std(axis=1, ddof=1) if n > 1 else np.full(n_pat, np.nan)
            nulls = np.divide(m, sd, out=np.full(n_pat, np.nan), where=np.isfinite(sd) & (sd > EPS))
            nulls[(np.abs(m) <= EPS) & np.isfinite(sd) & (sd <= EPS)] = 0.0
        else:
            nulls = np.array([fn(star[i]) for i in range(n_pat)], dtype=float)
        finite = nulls[np.isfinite(nulls)]
        n_draws = int(nulls.size)
        n_finite = int(finite.size)
        n_nonfinite = n_draws - n_finite
        if n_finite == 0:
            out["reason"] = "all null statistics were non-finite"
            out["n_perm"] = n_draws
            out["n_nonfinite_nulls"] = n_nonfinite
            return out
        if alt == "two-sided":
            count = _two_sided_count(obs, finite)
        else:
            count = _one_sided_count(obs, finite, greater=(alt == "greater"))
        # Denominator is the number of *defined* nulls. Counting extremes only
        # on finite draws while dividing by all draws is anti-conservative.
        p = _p_from_counts(count, n_finite, monte_carlo=False)
        out.update(
            {
                "available": True,
                "p_raw": p,
                "method": "exact_sign_flip",
                "n_perm": n_draws,
                "n_finite_nulls": n_finite,
                "n_extreme": count,
                "n_nonfinite_nulls": n_nonfinite,
            }
        )
        return out

    if n_perm <= 0:
        out["reason"] = "n_perm<=0 and n too large for exact enumeration"
        return out

    rng = np.random.default_rng(seed)
    nulls_acc: list[np.ndarray] = []
    remaining = int(n_perm)
    chunk = max(1, min(n_perm, 2_000_000 // max(n, 1)))
    while remaining > 0:
        rows = min(remaining, chunk)
        signs = rng.integers(0, 2, size=(rows, n), endpoint=False) * 2 - 1
        star = signs.astype(float) * d
        if stat_name == "mean":
            nulls_acc.append(star.mean(axis=1))
        elif stat_name == "median":
            nulls_acc.append(np.median(star, axis=1))
        elif stat_name == "cliffs_delta":
            nulls_acc.append(np.mean(star > 0.0, axis=1) - np.mean(star < 0.0, axis=1))
        elif stat_name == "cohens_dz":
            m = star.mean(axis=1)
            sd = star.std(axis=1, ddof=1) if n > 1 else np.full(rows, np.nan)
            dz = np.divide(m, sd, out=np.full(rows, np.nan), where=np.isfinite(sd) & (sd > EPS))
            dz[(np.abs(m) <= EPS) & np.isfinite(sd) & (sd <= EPS)] = 0.0
            nulls_acc.append(dz)
        else:
            nulls_acc.append(np.array([fn(star[i]) for i in range(rows)], dtype=float))
        remaining -= rows
    all_nulls = np.concatenate(nulls_acc)
    n_draws = int(all_nulls.size)
    finite = all_nulls[np.isfinite(all_nulls)]
    n_finite = int(finite.size)
    n_nonfinite = n_draws - n_finite
    if n_finite == 0:
        out["reason"] = "all null statistics were non-finite"
        out["n_perm"] = n_draws
        out["n_nonfinite_nulls"] = n_nonfinite
        return out
    if alt == "two-sided":
        count = _two_sided_count(obs, finite)
    else:
        count = _one_sided_count(obs, finite, greater=(alt == "greater"))
    p = _p_from_counts(count, n_finite, monte_carlo=True)
    out.update(
        {
            "available": True,
            "p_raw": p,
            "method": "monte_carlo_sign_flip",
            "n_perm": n_draws,
            "n_finite_nulls": n_finite,
            "n_extreme": count,
            "n_nonfinite_nulls": n_nonfinite,
        }
    )
    return out


def wilcoxon_signed_rank(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    alternative: str = "two-sided",
    zero_method: str = "wilcox",
) -> dict[str, Any]:
    """Wilcoxon signed-rank on paired differences. Unavailable if all diffs are 0."""
    from scipy.stats import wilcoxon

    aa, bb, meta = finite_pairs(a, b)
    n = meta["n"]
    d = aa - bb if n else np.asarray([], dtype=float)
    n_zero = int(np.sum(np.abs(d) <= EPS)) if n else 0
    n_nonzero = n - n_zero
    rb = matched_pairs_rank_biserial(d) if n else {"estimate": None, "available": False}
    hl = hodges_lehmann_paired(d) if n else None
    payload = estimate_payload(
        name="wilcoxon_signed_rank",
        n=n,
        estimate=hl,
        effect_size={"matched_pairs_rank_biserial": rb.get("estimate"), "cliffs_delta": cliffs_delta(aa, bb).get("estimate") if n else None},
        available=False,
        alternative=alternative,
        zero_method=zero_method,
        n_zero=n_zero,
        n_nonzero=n_nonzero,
        hypothesis="P(d>0)=P(d<0) on ranks; median(d)=0 if differences are symmetric under H0",
    )
    payload["hodges_lehmann"] = hl
    payload["rank_biserial"] = rb
    if n == 0:
        payload["reason"] = "no finite pairs"
        return payload
    if n_nonzero == 0:
        payload["reason"] = "Wilcoxon signed-rank is not defined: all paired differences are zero"
        payload["p_raw"] = 1.0
        payload["estimate"] = 0.0
        payload["available"] = False
        return payload
    try:
        res = wilcoxon(
            d,
            alternative=alternative,
            zero_method=zero_method,
            method="auto",
        )
    except TypeError:
        res = wilcoxon(d, alternative=alternative, zero_method=zero_method)
    except ValueError as exc:
        payload["reason"] = f"Wilcoxon unavailable: {exc}"
        return payload
    payload["available"] = True
    payload["p_raw"] = float(res.pvalue)
    payload["statistic"] = float(res.statistic)
    payload["method"] = str(getattr(res, "method", "auto"))
    payload["reason"] = None
    return payload


def benjamini_hochberg(
    p_values: Sequence[float | None],
    *,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """BH step-up adjusted p-values. Non-finite entries are left as None and excluded from m."""
    raw = list(p_values)
    finite_idx = [i for i, p in enumerate(raw) if p is not None and np.isfinite(p)]
    m = len(finite_idx)
    adj: list[float | None] = [None] * len(raw)
    reject = [False] * len(raw)
    if m == 0:
        return {
            "n_tests": len(raw),
            "n_valid": 0,
            "alpha": float(alpha),
            "method": "benjamini_hochberg",
            "p_adjusted": adj,
            "reject": reject,
            "significance_note": SIGNIFICANCE_NOTE,
        }
    order = sorted(finite_idx, key=lambda i: (float(raw[i]), i))
    q = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        p = float(raw[order[rank - 1]])
        running = min(running, min(1.0, p * m / rank))
        q[rank - 1] = running
    for rank, i in enumerate(order):
        adj[i] = q[rank]
        reject[i] = q[rank] <= float(alpha)
    return {
        "n_tests": len(raw),
        "n_valid": m,
        "alpha": float(alpha),
        "method": "benjamini_hochberg",
        "p_adjusted": adj,
        "reject": reject,
        "significance_note": SIGNIFICANCE_NOTE,
        "rule": "reject H0_i iff p_adjusted[i] <= alpha; this is not a CI-overlap rule",
    }


def apply_bh(
    estimates: Sequence[Mapping[str, Any]],
    *,
    alpha: float = ALPHA,
    p_key: str = "p_raw",
) -> list[dict[str, Any]]:
    """Copy estimates and fill ``p_adjusted`` from BH on ``p_key``."""
    pvals = [e.get(p_key) for e in estimates]
    bh = benjamini_hochberg(pvals, alpha=alpha)
    out: list[dict[str, Any]] = []
    for est, p_adj in zip(estimates, bh["p_adjusted"]):
        row = dict(est)
        row["p_adjusted"] = p_adj
        row["bh_alpha"] = float(alpha)
        row["bh_n_valid"] = bh["n_valid"]
        out.append(row)
    return out


def _fill_p(estimate: dict[str, Any], perm: dict[str, Any]) -> dict[str, Any]:
    row = dict(estimate)
    row["p_raw"] = perm.get("p_raw")
    row["p_adjusted"] = None
    row["hypothesis_test"] = perm.get("name")
    row["permutation"] = {
        "method": perm.get("method"),
        "n_perm": perm.get("n_perm"),
        "n_extreme": perm.get("n_extreme"),
        "alternative": perm.get("alternative"),
        "available": perm.get("available"),
        "reason": perm.get("reason"),
    }
    if not row.get("available") and estimate.get("estimate") is not None:
        row["available"] = True
    return row


def paired_comparison(
    a: Sequence[float] | np.ndarray | Mapping[Any, float],
    b: Sequence[float] | np.ndarray | Mapping[Any, float],
    *,
    query_ids: Sequence[Any] | None = None,
    name_a: str = "a",
    name_b: str = "b",
    name: str | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Publication-grade paired comparison of two query-level series.

    Accepts aligned arrays or ``{query_id: value}`` maps (inner join).
    Each of mean, median, Cohen's dz, and Cliff's delta includes n, estimate,
    95% CI, raw p-value, and effect size. ``p_adjusted`` stays None until
    ``apply_bh`` / ``compare_many``.
    """
    dropped_maps = {}
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        aa, bb, kept, meta = pair_by_query_id(a, b)
        qids: tuple[Any, ...] | None = kept
        dropped_maps = {k: meta[k] for k in ("n_unmatched_a", "n_unmatched_b") if k in meta}
    else:
        if isinstance(a, Mapping) or isinstance(b, Mapping):
            raise TypeError("both series must be maps or both must be sequences")
        aa_all, bb_all = as_1d(a), as_1d(b)
        if aa_all.shape != bb_all.shape:
            raise ValueError("paired series must have the same length")
        ok = np.isfinite(aa_all) & np.isfinite(bb_all)
        aa, bb = aa_all[ok], bb_all[ok]
        meta = {"n": int(ok.sum()), "n_dropped": int((~ok).sum()), "n_input": int(aa_all.size)}
        if query_ids is not None:
            if len(query_ids) != meta["n_input"]:
                raise ValueError("query_ids length must match the input series")
            qids = tuple(qid for qid, keep in zip(query_ids, ok) if keep)
        else:
            qids = None

    effects = paired_bootstrap_effects(aa, bb, n_boot=n_boot, seed=seed, level=level)
    perm_mean = paired_permutation_test(aa, bb, "mean", n_perm=n_perm, seed=seed, alternative=alternative)
    perm_median = paired_permutation_test(aa, bb, "median", n_perm=n_perm, seed=seed, alternative=alternative)
    perm_dz = paired_permutation_test(aa, bb, "cohens_dz", n_perm=n_perm, seed=seed, alternative=alternative)
    perm_cliff = paired_permutation_test(aa, bb, "cliffs_delta", n_perm=n_perm, seed=seed, alternative=alternative)
    wil = wilcoxon_signed_rank(aa, bb, alternative=alternative)

    mean_est = _fill_p(effects["mean_paired_difference"], perm_mean)
    median_est = _fill_p(effects["median_paired_difference"], perm_median)
    # Wilcoxon is the rank test for the median/symmetry hypothesis when valid.
    median_est["wilcoxon_p_raw"] = wil.get("p_raw")
    median_est["wilcoxon_available"] = wil.get("available")
    if wil.get("available"):
        median_est["permutation_p_raw"] = median_est.get("p_raw")
        median_est["p_raw"] = wil.get("p_raw")
        median_est["hypothesis_test"] = "wilcoxon_signed_rank"

    dz_est = _fill_p(effects["cohens_dz"], perm_dz)
    cliff_est = _fill_p(effects["cliffs_delta"], perm_cliff)

    # Wilcoxon block also uses the median CI (location) plus its own p-value.
    wil_out = dict(wil)
    wil_out["ci_lo"] = median_est.get("ci_lo")
    wil_out["ci_hi"] = median_est.get("ci_hi")
    wil_out["level"] = float(level)
    wil_out["n_boot"] = int(n_boot)
    wil_out["p_adjusted"] = None
    wil_out["significance_note"] = SIGNIFICANCE_NOTE

    return {
        "schema_version": "1.0",
        "name": name or f"{name_a} - {name_b}",
        "name_a": name_a,
        "name_b": name_b,
        "difference": f"{name_a} - {name_b}",
        "n": int(meta["n"]),
        "n_dropped": int(meta["n_dropped"]),
        "n_input": int(meta["n_input"]),
        "n_queries_kept": None if qids is None else len(qids),
        **dropped_maps,
        "n_boot": int(n_boot),
        "n_perm": int(n_perm),
        "seed": int(seed),
        "level": float(level),
        "alternative": alternative,
        "pairing": "query-level",
        "primary_endpoint": "mean_paired_difference",
        "mean_paired_difference": mean_est,
        "median_paired_difference": median_est,
        "cohens_dz": dz_est,
        "cliffs_delta": cliff_est,
        "wilcoxon_signed_rank": wil_out,
        "significance_note": SIGNIFICANCE_NOTE,
        "p_adjusted": None,
        "p_adjusted_note": (
            "p_adjusted is filled by apply_bh / compare_many over a family of tests. "
            "The primary endpoint is mean_paired_difference; other endpoints are descriptive."
        ),
    }


def compare_many(
    comparisons: Iterable[Mapping[str, Any]],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
    alternative: str = "two-sided",
    alpha: float = ALPHA,
    family: str = "mean_paired_difference",
) -> dict[str, Any]:
    """Run several paired comparisons and BH-adjust ``family`` raw p-values.

    Each item is a mapping with ``a``, ``b``, and optional ``name``,
    ``name_a``, ``name_b``, ``query_ids``.
    """
    reports: list[dict[str, Any]] = []
    for i, spec in enumerate(comparisons):
        reports.append(
            paired_comparison(
                spec["a"],
                spec["b"],
                query_ids=spec.get("query_ids"),
                name_a=str(spec.get("name_a", "a")),
                name_b=str(spec.get("name_b", "b")),
                name=str(spec.get("name", f"comparison_{i}")),
                n_boot=n_boot,
                n_perm=n_perm,
                seed=seed,
                level=level,
                alternative=alternative,
            )
        )
    family_rows = [r[family] for r in reports]
    adjusted = apply_bh(family_rows, alpha=alpha)
    for report, adj in zip(reports, adjusted):
        report[family] = adj
        report["p_adjusted"] = adj.get("p_adjusted")
        report["bh_family"] = family
        report["bh_alpha"] = float(alpha)
    bh = benjamini_hochberg([r[family].get("p_raw") for r in reports], alpha=alpha)
    return {
        "schema_version": "1.0",
        "n_comparisons": len(reports),
        "family": family,
        "bh": bh,
        "significance_note": SIGNIFICANCE_NOTE,
        "comparisons": reports,
    }

"""Complementarity / predictability decomposition — the computational core.

Implements the three-level ladder from `theory.md`:

    S(b)   best expected utility at expected cost <= b using a QUERY-INDEPENDENT policy
    A+(b)  ...                                        using any ROUTER (function of x alone)
    A*(b)  ...                                        using an ORACLE (sees realised outcomes)

and the derived quantities

    kappa(b) = A*(b) - S(b)                complementarity  (how much is there to win)
    rho(b)   = (A+(b) - S(b)) / kappa(b)   predictability   (how much of it is a function of x)

Nothing here calls an API or fits a model; it is pure accounting over a matrix of
per-item utilities and per-item costs. Utilities are bounded in [0, 1] and may be
graded rather than binary (RouterBench ships partial credit on 6 of 86 evals).

Conventions
-----------
`u` and `c` are (n_items, n_models) float arrays. All frontiers are computed at a
budget expressed as an expected cost per item.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull


# ---------------------------------------------------------------------------
# S(b): the query-independent (static) frontier
# ---------------------------------------------------------------------------

def static_frontier_vertices(mean_cost: np.ndarray, mean_util: np.ndarray):
    """Vertices of the upper concave envelope of the per-model (cost, utility) points.

    A query-independent policy picks model m with fixed probability lambda_m, so its
    (cost, utility) is a convex combination of the per-model points. The best
    achievable utility at budget b is therefore the upper-left concave envelope of
    those points, which is what this returns (sorted by cost, strictly improving).
    """
    pts = np.column_stack([np.asarray(mean_cost, float), np.asarray(mean_util, float)])
    order = np.lexsort((-pts[:, 1], pts[:, 0]))
    pts = pts[order]

    # Keep only Pareto-efficient points (no cheaper point with >= utility).
    keep, best = [], -np.inf
    for i in range(len(pts)):
        if pts[i, 1] > best:
            keep.append(i)
            best = pts[i, 1]
    pts = pts[keep]
    if len(pts) <= 2:
        return pts

    # Upper concave envelope by monotone chain on the Pareto set.
    hull = [0]
    for i in range(1, len(pts)):
        while len(hull) >= 2:
            a, b_, c_ = pts[hull[-2]], pts[hull[-1]], pts[i]
            cross = (b_[0] - a[0]) * (c_[1] - a[1]) - (b_[1] - a[1]) * (c_[0] - a[0])
            if cross >= 0:      # b_ is on or below chord a->c_, so it is not a vertex
                hull.pop()
            else:
                break
        hull.append(i)
    return pts[hull]


def static_frontier_at(mean_cost: np.ndarray, mean_util: np.ndarray, budget: float) -> float:
    """S(budget): best query-independent expected utility at expected cost <= budget."""
    v = static_frontier_vertices(mean_cost, mean_util)
    costs, utils = v[:, 0], v[:, 1]
    if budget <= costs[0]:
        # Cheapest efficient model is already too expensive: only feasible if equal.
        return float(utils[0]) if np.isclose(budget, costs[0]) else float("nan")
    if budget >= costs[-1]:
        return float(utils[-1])
    j = int(np.searchsorted(costs, budget, side="right") - 1)
    if j >= len(costs) - 1:
        return float(utils[-1])
    w = (budget - costs[j]) / (costs[j + 1] - costs[j])
    return float(utils[j] + w * (utils[j + 1] - utils[j]))


# ---------------------------------------------------------------------------
# A*(b): the oracle frontier, by exact LP relaxation of the multiple-choice knapsack
# ---------------------------------------------------------------------------

def _assign_from_lambda(u: np.ndarray, c: np.ndarray, lam: float):
    """Per-item argmax of (utility - lam * cost). Ties resolved toward lower cost."""
    score = u - lam * c
    # np.argmax takes the first maximal column; order columns by cost so ties go cheap.
    return np.argmax(score, axis=1)


def oracle_frontier_at(u: np.ndarray, c: np.ndarray, budget: float, tol: float = 1e-12,
                       max_iter: int = 200):
    """A*(budget) by Lagrangian bisection on the single budget constraint.

    Solves  max_z  sum_i sum_m z_im u_im   s.t.  sum_i sum_m z_im c_im <= n*budget,
    sum_m z_im = 1, z >= 0. The LP relaxation of a multiple-choice knapsack has at
    most one fractional item, so bisecting the multiplier and then mixing the two
    bracketing assignments on a single item is exact.

    Returns (utility, realised_cost, lam).
    """
    u = np.asarray(u, float)
    c = np.asarray(c, float)
    n, k = u.shape
    cost_order = np.argsort(c.mean(axis=0))
    u, c = u[:, cost_order], c[:, cost_order]

    def eval_lam(lam):
        a = _assign_from_lambda(u, c, lam)
        rows = np.arange(n)
        return u[rows, a].mean(), c[rows, a].mean(), a

    # lam = 0 buys the best utility regardless of cost; lam -> inf buys the cheapest.
    hi_u, hi_c, _ = eval_lam(0.0)
    if hi_c <= budget + tol:
        return float(hi_u), float(hi_c), 0.0
    lam_lo, lam_hi = 0.0, 1.0
    for _ in range(200):
        _, cc, _ = eval_lam(lam_hi)
        if cc <= budget:
            break
        lam_hi *= 2.0
    else:
        raise RuntimeError("could not bracket the budget")

    for _ in range(max_iter):
        mid = 0.5 * (lam_lo + lam_hi)
        _, cc, _ = eval_lam(mid)
        if cc > budget:
            lam_lo = mid
        else:
            lam_hi = mid
        if lam_hi - lam_lo < 1e-14 * max(1.0, lam_hi):
            break

    u_lo, c_lo, _ = eval_lam(lam_lo)   # over budget, higher utility
    u_hi, c_hi, _ = eval_lam(lam_hi)   # under budget, lower utility
    if c_lo <= budget + tol:
        return float(u_lo), float(c_lo), lam_lo
    if abs(c_lo - c_hi) < tol:
        return float(u_hi), float(c_hi), lam_hi
    w = (budget - c_hi) / (c_lo - c_hi)          # mix to land exactly on the budget
    w = float(np.clip(w, 0.0, 1.0))
    return float(u_hi + w * (u_lo - u_hi)), float(budget), lam_hi


def oracle_integer_at(u: np.ndarray, c: np.ndarray, budget: float):
    """Best integral (one model per item) assignment under the budget, greedily repaired.

    Used only to report the discreteness gap; the LP value is the bound used in the
    decomposition.
    """
    lp_u, lp_c, lam = oracle_frontier_at(u, c, budget)
    u = np.asarray(u, float)
    c = np.asarray(c, float)
    n = u.shape[0]
    rows = np.arange(n)
    a = _assign_from_lambda(u, c, lam)
    cost = c[rows, a].mean()
    if cost <= budget:
        return float(u[rows, a].mean()), float(cost)
    # Downgrade the items that give up the least utility per dollar released.
    cheapest = np.argmin(c, axis=1)
    d_u = u[rows, a] - u[rows, cheapest]
    d_c = c[rows, a] - c[rows, cheapest]
    ratio = np.where(d_c > 0, d_u / np.maximum(d_c, 1e-300), np.inf)
    for i in np.argsort(ratio):
        if cost <= budget:
            break
        a[i] = cheapest[i]
        cost = c[rows, a].mean()
    return float(u[rows, a].mean()), float(cost)


# ---------------------------------------------------------------------------
# A router's realised value at matched cost
# ---------------------------------------------------------------------------

def router_value_at(u: np.ndarray, c: np.ndarray, scores: np.ndarray, budget: float):
    """Realised utility of a router that ranks by `scores` and spends up to `budget`.

    `scores[i, m]` is the router's PREDICTED utility for model m on item i. The
    router commits using its own predictions and per-model mean costs (it cannot
    know an item's realised token count before generating), but the cost it is
    charged is the REALISED per-item cost -- which is the Stage 8 correction applied
    to the routing literature's own accounting.
    """
    u = np.asarray(u, float)
    c = np.asarray(c, float)
    s = np.asarray(scores, float)
    n = u.shape[0]
    rows = np.arange(n)
    mean_c = c.mean(axis=0)

    def eval_lam(lam):
        a = np.argmax(s - lam * mean_c[None, :], axis=1)
        return u[rows, a].mean(), c[rows, a].mean(), a

    hi_u, hi_c, _ = eval_lam(0.0)
    if hi_c <= budget:
        return float(hi_u), float(hi_c)
    lam_lo, lam_hi = 0.0, 1.0
    for _ in range(200):
        if eval_lam(lam_hi)[1] <= budget:
            break
        lam_hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lam_lo + lam_hi)
        if eval_lam(mid)[1] > budget:
            lam_lo = mid
        else:
            lam_hi = mid
    u_lo, c_lo, _ = eval_lam(lam_lo)
    u_hi, c_hi, _ = eval_lam(lam_hi)
    if abs(c_lo - c_hi) < 1e-15:
        return float(u_hi), float(c_hi)
    w = float(np.clip((budget - c_hi) / (c_lo - c_hi), 0.0, 1.0))
    return float(u_hi + w * (u_lo - u_hi)), float(budget)


# ---------------------------------------------------------------------------
# Proposition 3: the router-free ceiling  gain <= sqrt(beta(1-beta)) * sd(delta)
# ---------------------------------------------------------------------------

def ceiling_from_sd(sd_delta: float, beta: float) -> float:
    """sqrt(beta*(1-beta)) * sd_delta -- see theory.md Proposition 3.

    Valid for ANY router at operating fraction beta, by Cauchy-Schwarz on
    Cov(z, delta) plus Var(z) <= beta(1-beta) for z in [0, 1] with mean beta.
    """
    return float(np.sqrt(max(beta * (1.0 - beta), 0.0)) * sd_delta)


def variance_components(reps: np.ndarray, k: int | None = None):
    """Split per-item outcome variance into between-item (signal) and within-item (noise).

    `reps[i, j]` is replicate j of a binary outcome for item i. With k i.i.d.
    replicates per item,

        Var(eta_hat) = Var(eta) + (1/k) E[eta(1-eta)]
        E[(k/(k-1)) eta_hat (1-eta_hat)] = E[eta(1-eta)]

    so the between-item variance of the true success probability is identified.
    Returns a dict; `var_between` is clipped at zero and `clipped` records whether
    the clip bound.
    """
    reps = np.asarray(reps, float)
    n, kk = reps.shape
    k = kk if k is None else k
    if k < 2:
        raise ValueError("need k >= 2 replicates to separate within from between")
    eta_hat = reps.mean(axis=1)
    var_hat = float(np.var(eta_hat, ddof=1))
    within = float(np.mean(k / (k - 1.0) * eta_hat * (1.0 - eta_hat)))
    raw = var_hat - within / k
    return {
        "n_items": int(n),
        "k": int(k),
        "mean": float(eta_hat.mean()),
        "var_eta_hat": var_hat,
        "within_var_E_eta_1_minus_eta": within,
        "var_between_raw": float(raw),
        "var_between": float(max(raw, 0.0)),
        "clipped": bool(raw < 0.0),
        "sd_between": float(np.sqrt(max(raw, 0.0))),
        "reliability": float(max(raw, 0.0) / var_hat) if var_hat > 0 else float("nan"),
    }


def sd_delta_from_replicates(reps_a: np.ndarray, reps_b: np.ndarray, k: int | None = None):
    """sd of delta(x) = eta_b(x) - eta_a(x), corrected for generation noise.

    Assumes replicates are i.i.d. given the item and independent across models (the
    two models are separate API calls, in our case to separate providers). Both
    assumptions are stated as limitations in theory.md; violating i.i.d. by positive
    within-item correlation would UNDERSTATE the noise term and therefore OVERSTATE
    the ceiling, i.e. err in routing's favour.
    """
    ra = np.asarray(reps_a, float)
    rb = np.asarray(reps_b, float)
    n, kk = ra.shape
    k = kk if k is None else k
    ea, eb = ra.mean(axis=1), rb.mean(axis=1)
    d_hat = eb - ea
    var_d_hat = float(np.var(d_hat, ddof=1))
    w_a = float(np.mean(k / (k - 1.0) * ea * (1.0 - ea)))
    w_b = float(np.mean(k / (k - 1.0) * eb * (1.0 - eb)))
    raw = var_d_hat - (w_a + w_b) / k
    return {
        "n_items": int(n),
        "k": int(k),
        "mean_delta": float(d_hat.mean()),
        "var_delta_hat_observed": var_d_hat,
        "within_a": w_a,
        "within_b": w_b,
        "var_delta_raw": float(raw),
        "var_delta": float(max(raw, 0.0)),
        "clipped": bool(raw < 0.0),
        "sd_delta": float(np.sqrt(max(raw, 0.0))),
        "sd_delta_observed": float(np.sqrt(var_d_hat)),
        "noise_share": float(min(1.0, ((w_a + w_b) / k) / var_d_hat)) if var_d_hat > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Proposition 5: the Bayes-optimal AUC implied by a given eta distribution
# ---------------------------------------------------------------------------

def bayes_auc_bound(sd_eta: float, mean_eta: float) -> float:
    """Upper bound on the Bayes AUC:  1/2 + sd_eta / (2*sqrt(2)*a*(1-a)).

    From AUC* = 1/2 + E|eta(X)-eta(X')| / (4 a (1-a)) and E|Z| <= sqrt(E Z^2).
    """
    a = float(mean_eta)
    if not (0.0 < a < 1.0):
        return float("nan")
    return float(min(1.0, 0.5 + sd_eta / (2.0 * np.sqrt(2.0) * a * (1.0 - a))))


def bayes_auc_beta_moment_match(mean_eta: float, sd_eta: float, n_grid: int = 4001):
    """Point estimate of AUC* assuming eta ~ Beta with the given mean and sd.

    A modelling assumption, reported separately from the distribution-free bound
    above: k = 3 replicates identify the first two moments of eta but not
    E|eta - eta'|, which the exact formula needs.
    """
    a, s = float(mean_eta), float(sd_eta)
    if not (0.0 < a < 1.0) or s <= 0:
        return {"feasible": False, "auc_star": 0.5, "alpha": None, "beta": None}
    v = s * s
    vmax = a * (1.0 - a)
    if v >= vmax:                       # outside the Beta family
        return {"feasible": False, "auc_star": float("nan"), "alpha": None, "beta": None}
    nu = vmax / v - 1.0
    alpha, bet = a * nu, (1.0 - a) * nu
    from scipy.stats import beta as beta_dist
    q = np.linspace(0.5 / n_grid, 1.0 - 0.5 / n_grid, n_grid)
    x = beta_dist.ppf(q, alpha, bet)
    # E|eta - eta'| via the L1 Gini form: 2 * E[(2F(x)-1) x]
    e_abs = float(2.0 * np.mean((2.0 * q - 1.0) * x))
    return {
        "feasible": True,
        "alpha": float(alpha),
        "beta": float(bet),
        "E_abs_diff": e_abs,
        "auc_star": float(min(1.0, 0.5 + e_abs / (4.0 * a * (1.0 - a)))),
    }


# ---------------------------------------------------------------------------
# Binary special case: the disagreement cells
# ---------------------------------------------------------------------------

def binary_cells(y_a: np.ndarray, y_b: np.ndarray):
    """p11/p10/p01/p00 and min(p01, p10), the exact gain of an unconstrained oracle
    over the better single model (theory.md Proposition 2c)."""
    ya = np.asarray(y_a, float) > 0.5
    yb = np.asarray(y_b, float) > 0.5
    p11 = float(np.mean(ya & yb))
    p10 = float(np.mean(ya & ~yb))
    p01 = float(np.mean(~ya & yb))
    p00 = float(np.mean(~ya & ~yb))
    return {
        "p11": p11, "p10": p10, "p01": p01, "p00": p00,
        "acc_a": p11 + p10, "acc_b": p11 + p01,
        "disagreement": p10 + p01,
        "kappa0_min_p01_p10": float(min(p01, p10)),
        "best_single": float(max(p11 + p10, p11 + p01)),
        "union": float(1.0 - p00),
    }


# ---------------------------------------------------------------------------
# Bootstrap over items
# ---------------------------------------------------------------------------

def bootstrap_ci(fn, n_items: int, n_boot: int = 2000, seed: int = 20260907, alpha: float = 0.05):
    """Percentile bootstrap over items. `fn(idx)` returns a scalar or 1-d array."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_items, n_items)
        out.append(fn(idx))
    arr = np.asarray(out, float)
    lo = np.nanpercentile(arr, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(arr, 100 * (1 - alpha / 2), axis=0)
    return lo, hi, arr


def holm_bonferroni(pvals, alpha: float = 0.05):
    """Holm-Bonferroni step-down. Returns (reject, adjusted_p)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj <= alpha, adj


def benjamini_hochberg(pvals, alpha: float = 0.05):
    """Benjamini-Hochberg FDR. Returns (reject, adjusted_p)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, m * p[i] / (rank + 1))
        adj[i] = min(1.0, running)
    return adj <= alpha, adj

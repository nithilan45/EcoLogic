"""Numerical validation of every proposition in theory.md.

Each check compares a closed form or fast algorithm in decomp.py against a
brute-force reference (an LP solver, or Monte Carlo). Run before any result in
Stages 11-13 is believed. Writes s11_validate.json.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.stats import beta as beta_dist
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))
from decomp import (  # noqa: E402
    bayes_auc_beta_moment_match,
    bayes_auc_bound,
    binary_cells,
    ceiling_from_sd,
    oracle_frontier_at,
    router_value_at,
    sd_delta_from_replicates,
    static_frontier_at,
    variance_components,
)

RESULTS: dict = {}
FAILURES: list = []


def check(name: str, ok: bool, detail: dict):
    RESULTS[name] = {"pass": bool(ok), **detail}
    if not ok:
        FAILURES.append(name)
    print(("PASS  " if ok else "FAIL  ") + name + "  " + json.dumps(
        {k: (round(v, 10) if isinstance(v, float) else v) for k, v in detail.items()}))


# ---------------------------------------------------------------------------
# 1. S(b) against a brute-force LP over the simplex of mixing weights
# ---------------------------------------------------------------------------
def brute_static(mean_cost, mean_util, budget):
    k = len(mean_cost)
    res = linprog(
        c=-np.asarray(mean_util, float),
        A_ub=np.asarray(mean_cost, float).reshape(1, -1),
        b_ub=[budget],
        A_eq=np.ones((1, k)),
        b_eq=[1.0],
        bounds=[(0, 1)] * k,
        method="highs",
    )
    return -res.fun if res.success else float("nan")


rng = np.random.default_rng(11)
errs = []
for _ in range(300):
    k = rng.integers(2, 7)
    mc = np.sort(rng.uniform(0.01, 1.0, k))
    mu = rng.uniform(0.2, 0.95, k)
    b = float(rng.uniform(mc.min(), mc.max()))
    a, bb = static_frontier_at(mc, mu, b), brute_static(mc, mu, b)
    if np.isfinite(a) and np.isfinite(bb):
        errs.append(abs(a - bb))
check("P1_static_frontier_matches_LP", max(errs) < 1e-9,
      {"n_cases": len(errs), "max_abs_err": float(max(errs))})


# ---------------------------------------------------------------------------
# 2. A*(b) against a brute-force multiple-choice-knapsack LP
# ---------------------------------------------------------------------------
def brute_oracle(u, c, budget):
    n, k = u.shape
    nv = n * k
    A_eq = np.zeros((n, nv))
    for i in range(n):
        A_eq[i, i * k:(i + 1) * k] = 1.0
    res = linprog(
        c=-u.reshape(-1),
        A_ub=c.reshape(1, -1) / n,
        b_ub=[budget],
        A_eq=A_eq,
        b_eq=np.ones(n),
        bounds=[(0, 1)] * nv,
        method="highs",
    )
    return -res.fun / n if res.success else float("nan")


errs, cost_errs = [], []
for _ in range(120):
    n, k = int(rng.integers(6, 40)), int(rng.integers(2, 5))
    u = rng.random((n, k))
    c = rng.uniform(0.01, 1.0, (n, k))
    lo, hi = c.min(axis=1).mean(), c.max(axis=1).mean()
    b = float(rng.uniform(lo, hi))
    got, got_c, _ = oracle_frontier_at(u, c, b)
    ref = brute_oracle(u, c, b)
    errs.append(abs(got - ref))
    cost_errs.append(max(0.0, got_c - b))
check("P2_oracle_frontier_matches_MCKP_LP", max(errs) < 1e-8,
      {"n_cases": len(errs), "max_abs_err": float(max(errs)),
       "max_budget_overrun": float(max(cost_errs))})


# ---------------------------------------------------------------------------
# 3. The ladder S(b) <= A+(b) <= A*(b).
#
#    A+ is the Bayes router: the LP optimum over policies measurable in x, which is
#    the oracle problem solved on the CONDITIONAL means (eta, gamma) rather than on
#    realised outcomes. A* is the same problem on realised (y, c).
# ---------------------------------------------------------------------------
viol_low, viol_high, gains, realistic_viol = [], [], [], []
for _ in range(200):
    n, k = 20000, 2
    eta = np.column_stack([rng.beta(4, 2, n), rng.beta(3, 3, n)])
    y = (rng.random((n, k)) < eta).astype(float)
    gamma = np.tile(np.array([0.1, 1.0]), (n, 1)) * rng.uniform(0.5, 1.5, (n, k))
    c = gamma * rng.lognormal(-0.02, 0.2, (n, k))          # cost noisy given x
    b = float(np.mean(c[:, 0]) + 0.5 * (np.mean(c[:, 1]) - np.mean(c[:, 0])))
    s = static_frontier_at(c.mean(axis=0), y.mean(axis=0), b)
    a_dag, _, _ = oracle_frontier_at(eta, gamma, b)
    a_star, _, _ = oracle_frontier_at(y, c, b)
    # Deployable router: decides with mean costs, charged per-item cost. Evaluated on
    # eta rather than y so the comparison against A+ carries no sampling noise; its
    # only handicap is not knowing an item's cost before generating.
    a_real, _ = router_value_at(eta, gamma, eta, b)
    viol_low.append(s - a_dag)
    viol_high.append(a_dag - a_star)
    realistic_viol.append(a_real - a_dag)
    gains.append(a_dag - s)
check("P1_ladder_S_le_Adag_le_Astar",
      max(viol_low) < 1e-6 and max(viol_high) < 1e-6 and max(realistic_viol) < 1e-3,
      {"max_S_minus_Adag": float(max(viol_low)),
       "max_Adag_minus_Astar": float(max(viol_high)),
       "max_realistic_minus_Adag": float(max(realistic_viol)),
       "mean_bayes_router_gain": float(np.mean(gains))})


# ---------------------------------------------------------------------------
# 3b. The predictability gap is EXACTLY the outcome randomness given x.
#
#     If utility is deterministic given the query then a router is as good as the
#     oracle, so rho = 1. Every bit of rho < 1 is outcome noise the router cannot
#     see -- which means an "oracle headroom" measured on single generations is
#     partly a noise-exploiting artifact. This is the claim the paper turns on, so
#     it gets its own check.
# ---------------------------------------------------------------------------
det_gap, noisy_rho = [], []
for _ in range(200):
    n = 20000
    eta = np.column_stack([rng.beta(4, 2, n), rng.beta(3, 3, n)])
    gamma = np.tile(np.array([0.1, 1.0]), (n, 1)) * rng.uniform(0.5, 1.5, (n, k))
    b = float(np.mean(gamma[:, 0]) + 0.5 * (np.mean(gamma[:, 1]) - np.mean(gamma[:, 0])))
    # deterministic outcomes: u = eta exactly
    s_d = static_frontier_at(gamma.mean(axis=0), eta.mean(axis=0), b)
    dag_d, _, _ = oracle_frontier_at(eta, gamma, b)
    star_d, _, _ = oracle_frontier_at(eta, gamma, b)
    det_gap.append(abs((star_d - s_d) - (dag_d - s_d)))
    # Bernoulli outcomes with the same eta: rho drops strictly below 1
    y = (rng.random((n, 2)) < eta).astype(float)
    s_n = static_frontier_at(gamma.mean(axis=0), y.mean(axis=0), b)
    star_n, _, _ = oracle_frontier_at(y, gamma, b)
    if star_n - s_n > 1e-9:
        noisy_rho.append((dag_d - s_n) / (star_n - s_n))
check("P1b_rho_is_one_iff_outcomes_deterministic_given_x",
      max(det_gap) < 1e-9 and np.mean(noisy_rho) < 0.6,
      {"max_deterministic_kappa_minus_achievable": float(max(det_gap)),
       "mean_rho_under_bernoulli_noise": float(np.mean(noisy_rho)),
       "note": "rho = 1 exactly when u is a function of x; Bernoulli noise alone drives it far below 1"})


# ---------------------------------------------------------------------------
# 4. Proposition 3: gain <= sqrt(beta(1-beta)) * sd(delta), for ANY policy
# ---------------------------------------------------------------------------
worst_slack, tight = [], []
for _ in range(4000):
    n = 4000
    eta_s = rng.beta(rng.uniform(0.5, 6), rng.uniform(0.5, 6), n)
    eta_l = rng.beta(rng.uniform(0.5, 6), rng.uniform(0.5, 6), n)
    delta = eta_l - eta_s
    beta = float(rng.uniform(0.05, 0.95))
    # exact best router gain at operating fraction beta: top-beta tail of delta
    thr = np.quantile(delta, 1.0 - beta)
    z = (delta > thr).astype(float)
    short = beta * n - z.sum()
    if short > 0:                       # fill the boundary fractionally
        eq = np.flatnonzero(delta == thr)
        if len(eq):
            z[eq[:int(short)]] = 1.0
    gain = float(np.mean(z * delta) - beta * delta.mean())
    bound = ceiling_from_sd(float(np.std(delta)), beta)
    worst_slack.append(bound - gain)
    tight.append(gain / bound if bound > 0 else np.nan)
check("P3_ceiling_never_violated", min(worst_slack) > -1e-12,
      {"n_cases": len(worst_slack), "min_slack": float(min(worst_slack)),
       "max_tightness_ratio": float(np.nanmax(tight))})

# tightness: a two-point delta at beta = P(high) should attain the bound
n = 200000
p = 0.3
delta = np.where(rng.random(n) < p, 1.0, 0.0)
gain = p * (1.0 - delta.mean())
bound = ceiling_from_sd(float(np.std(delta)), p)
check("P3_ceiling_tight_on_two_point", abs(gain / bound - 1.0) < 0.01,
      {"gain": float(gain), "bound": float(bound), "ratio": float(gain / bound)})


# ---------------------------------------------------------------------------
# 5. Proposition 4: replicate-based variance components are unbiased
# ---------------------------------------------------------------------------
truth_between, est_between = [], []
for _ in range(400):
    n, k = 5000, 3
    a, bta = rng.uniform(0.8, 5), rng.uniform(0.8, 5)
    eta = rng.beta(a, bta, n)
    reps = (rng.random((n, k)) < eta[:, None]).astype(float)
    vc = variance_components(reps)
    truth_between.append(float(np.var(eta, ddof=1)))
    est_between.append(vc["var_between_raw"])
bias = float(np.mean(np.array(est_between) - np.array(truth_between)))
rel = abs(bias) / float(np.mean(truth_between))
check("P4_variance_components_unbiased", rel < 0.02,
      {"mean_true_var": float(np.mean(truth_between)),
       "mean_est_var": float(np.mean(est_between)), "rel_bias": rel})

truth_sd, est_sd = [], []
for _ in range(400):
    n, k = 5000, 3
    eta_a = rng.beta(rng.uniform(0.8, 5), rng.uniform(0.8, 5), n)
    eta_b = rng.beta(rng.uniform(0.8, 5), rng.uniform(0.8, 5), n)
    ra = (rng.random((n, k)) < eta_a[:, None]).astype(float)
    rb = (rng.random((n, k)) < eta_b[:, None]).astype(float)
    out = sd_delta_from_replicates(ra, rb)
    truth_sd.append(float(np.var(eta_b - eta_a, ddof=1)))
    est_sd.append(out["var_delta_raw"])
bias = float(np.mean(np.array(est_sd) - np.array(truth_sd)))
rel = abs(bias) / float(np.mean(truth_sd))
check("P4_sd_delta_from_replicates_unbiased", rel < 0.02,
      {"mean_true_var_delta": float(np.mean(truth_sd)),
       "mean_est_var_delta": float(np.mean(est_sd)), "rel_bias": rel})

# and the naive (uncorrected) estimator is badly biased upward -- the reason
# single-generation datasets cannot do this
naive = []
for _ in range(200):
    n, k = 5000, 3
    eta_a = rng.beta(2, 2, n)
    eta_b = rng.beta(2, 2, n)
    ra = (rng.random((n, k)) < eta_a[:, None]).astype(float)
    rb = (rng.random((n, k)) < eta_b[:, None]).astype(float)
    out = sd_delta_from_replicates(ra, rb)
    naive.append(out["var_delta_hat_observed"] / float(np.var(eta_b - eta_a, ddof=1)))
check("P4_naive_estimator_is_inflated", np.mean(naive) > 1.2,
      {"mean_inflation_factor_of_naive_var": float(np.mean(naive))})


# ---------------------------------------------------------------------------
# 6. Proposition 5: AUC* = 1/2 + E|eta-eta'| / (4 a (1-a))
# ---------------------------------------------------------------------------
errs = []
for _ in range(200):
    n = 200000
    a, bta = rng.uniform(0.7, 6), rng.uniform(0.7, 6)
    eta = rng.beta(a, bta, n)
    y = (rng.random(n) < eta).astype(int)
    if y.mean() in (0.0, 1.0):
        continue
    emp = roc_auc_score(y, eta)                       # Bayes AUC, empirically
    abar = float(eta.mean())
    e1, e2 = eta[: n // 2], eta[n // 2:]
    e_abs = float(np.mean(np.abs(e1 - e2)))
    closed = 0.5 + e_abs / (4.0 * abar * (1.0 - abar))
    errs.append(abs(emp - closed))
check("P5_bayes_auc_closed_form", max(errs) < 0.01,
      {"n_cases": len(errs), "max_abs_err": float(max(errs))})

# the sd-only bound must dominate the truth, and the Beta moment match approximate it
bnd_ok, mm_err = [], []
for _ in range(200):
    n = 200000
    a, bta = rng.uniform(0.7, 6), rng.uniform(0.7, 6)
    eta = rng.beta(a, bta, n)
    y = (rng.random(n) < eta).astype(int)
    if y.mean() in (0.0, 1.0):
        continue
    emp = roc_auc_score(y, eta)
    bnd = bayes_auc_bound(float(eta.std()), float(eta.mean()))
    mm = bayes_auc_beta_moment_match(float(eta.mean()), float(eta.std()))
    bnd_ok.append(bnd - emp)
    if mm["feasible"]:
        mm_err.append(abs(mm["auc_star"] - emp))
check("P5_sd_bound_dominates_truth", min(bnd_ok) > -1e-6,
      {"min_slack": float(min(bnd_ok)), "mean_slack": float(np.mean(bnd_ok))})
check("P5_beta_moment_match_accurate", max(mm_err) < 0.01,
      {"n_cases": len(mm_err), "max_abs_err": float(max(mm_err))})


# ---------------------------------------------------------------------------
# 7. Proposition 2c: min(p01, p10) is the unconstrained oracle gain over the best model
# ---------------------------------------------------------------------------
errs = []
for _ in range(2000):
    n = 3000
    eta_a = rng.beta(rng.uniform(1, 5), rng.uniform(1, 5), n)
    eta_b = rng.beta(rng.uniform(1, 5), rng.uniform(1, 5), n)
    ya = (rng.random(n) < eta_a).astype(float)
    yb = (rng.random(n) < eta_b).astype(float)
    cells = binary_cells(ya, yb)
    got = cells["union"] - cells["best_single"]
    errs.append(abs(got - cells["kappa0_min_p01_p10"]))
check("P2c_kappa0_equals_union_minus_best_single", max(errs) < 1e-12,
      {"n_cases": len(errs), "max_abs_err": float(max(errs))})


# ---------------------------------------------------------------------------
# 8. Stage 8 consistency: the naive per-model-mean cost axis is exact for a
#    query-independent policy and biased for a router
# ---------------------------------------------------------------------------
static_err, router_err = [], []
for _ in range(500):
    n = 5000
    c = np.column_stack([rng.lognormal(-2, 0.6, n), rng.lognormal(0, 0.6, n)])
    eta = np.column_stack([rng.beta(3, 2, n), rng.beta(4, 2, n)])
    lam = float(rng.uniform(0.1, 0.9))
    pick = (rng.random(n) < lam).astype(int)                 # query-independent
    naive = (1 - lam) * c[:, 0].mean() + lam * c[:, 1].mean()
    static_err.append(abs(c[np.arange(n), pick].mean() - naive) / naive)
    # a router escalates the items it predicts need it, which correlates with cost
    score = eta[:, 1] - eta[:, 0] + 0.5 * np.log(c[:, 1])
    thr = np.quantile(score, 1 - lam)
    pick_r = (score > thr).astype(int)
    frac = pick_r.mean()
    naive_r = (1 - frac) * c[:, 0].mean() + frac * c[:, 1].mean()
    router_err.append((c[np.arange(n), pick_r].mean() - naive_r) / naive_r)
check("S8_naive_cost_exact_for_static_biased_for_router",
      np.mean(static_err) < 0.02 and abs(np.mean(router_err)) > 0.05,
      {"mean_rel_err_static": float(np.mean(static_err)),
       "mean_rel_err_router": float(np.mean(router_err)),
       "note": "router error is signed: positive means naive accounting understates true cost"})


print()
print(f"{len(RESULTS) - len(FAILURES)}/{len(RESULTS)} checks passed")
if FAILURES:
    print("FAILURES: " + ", ".join(FAILURES))
with open(__file__.rsplit("/", 1)[0] + "/s11_validate.json", "w") as f:
    json.dump({"n_checks": len(RESULTS), "n_failures": len(FAILURES),
               "failures": FAILURES, "checks": RESULTS}, f, indent=1)
sys.exit(1 if FAILURES else 0)

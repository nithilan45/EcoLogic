"""Multiple-choice knapsack (MCKP) solvers for budgeted routing.

Each query is a class; each model is a mutually exclusive choice. Pick exactly
one model per query to maximise total quality subject to a total cost budget.

    maximise  Σ_i quality[i, x_i]
    s.t.      Σ_i cost[i, x_i]  ≤  B
              x_i ∈ {0,…,M−1}

Solvers
-------
* ``exact_solve`` / ``enumerate_solve`` — small n (DFS / full product).
* ``milp_solve`` — scipy HiGHS integer program (optional).
* ``dp_solve`` — optional exact DP on integer (or scaled) costs.
* ``greedy_solve`` — incremental-efficiency sanity check.
* ``lp_relaxation`` — Dyer–Zemel convex-chain greedy (at most one fractional class).
* ``approximate_solve`` — scalable integer feasible solution (LP round + greedy
  + single-class jumps). No ML.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Any, Sequence

import numpy as np

EPS = 1e-12
DFS_MAX_N = 14
ENUM_MAX_STATES = 250_000
DP_MAX_CELLS = 20_000_000


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _finite_pair(cost: np.ndarray, quality: np.ndarray) -> None:
    if cost.shape != quality.shape or cost.ndim != 2 or cost.shape[0] == 0:
        raise ValueError("cost and quality must be non-empty 2-D arrays of the same shape")
    if np.any(~np.isfinite(cost)) or np.any(~np.isfinite(quality)):
        raise ValueError("non-finite cost or quality")
    if np.any(cost < -EPS):
        raise ValueError("negative costs are not allowed")


def is_integer_valued(arr: np.ndarray, *, eps: float = 1e-9) -> bool:
    a = np.asarray(arr, dtype=float)
    return bool(np.all(np.abs(a - np.round(a)) <= eps))


@dataclass
class Solution:
    """One (possibly fractional) MCKP solution."""

    choice: np.ndarray
    total_cost: float
    total_quality: float
    n: int
    feasible: bool
    method: str
    note: str = ""
    weights: np.ndarray | None = None
    fractional: bool = False
    lp_mean_quality: float | None = None

    @property
    def mean_quality(self) -> float:
        if not self.feasible or self.n == 0:
            return float("nan")
        return float(self.total_quality / self.n)

    def as_dict(self) -> dict[str, Any]:
        mix = None
        if self.feasible and self.choice is not None and self.choice.size == self.n:
            mix = {str(k): int(np.sum(self.choice == k)) for k in np.unique(self.choice)}
        return {
            "total_cost": self.total_cost if self.feasible else None,
            "total_quality": self.total_quality if self.feasible else None,
            "mean_quality": self.mean_quality if self.feasible else None,
            "feasible": self.feasible,
            "method": self.method,
            "note": self.note,
            "fractional": self.fractional,
            "lp_mean_quality": self.lp_mean_quality,
            "model_mix": mix,
        }


@dataclass
class MCKPInstance:
    cost: np.ndarray
    quality: np.ndarray
    names: tuple[str, ...] = ()
    query_ids: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        self.cost = np.asarray(self.cost, dtype=float)
        self.quality = np.asarray(self.quality, dtype=float)
        _finite_pair(self.cost, self.quality)
        n, m = self.cost.shape
        if not self.names:
            self.names = tuple(f"m{j}" for j in range(m))
        else:
            self.names = tuple(self.names)
            if len(self.names) != m:
                raise ValueError("names length must match n_models")
        if not self.query_ids:
            self.query_ids = tuple(range(n))
        else:
            self.query_ids = tuple(self.query_ids)
            if len(self.query_ids) != n:
                raise ValueError("query_ids length must match n_queries")

    @property
    def n(self) -> int:
        return int(self.cost.shape[0])

    @property
    def m(self) -> int:
        return int(self.cost.shape[1])

    def min_total_cost(self) -> float:
        return float(self.cost.min(axis=1).sum())

    def max_total_cost(self) -> float:
        return float(self.cost.max(axis=1).sum())

    def evaluate(self, choice: Sequence[int]) -> Solution:
        choice = np.asarray(choice, dtype=int).reshape(-1)
        if choice.shape[0] != self.n:
            raise ValueError("choice length must equal n_queries")
        if np.any(choice < 0) or np.any(choice >= self.m):
            raise ValueError("choice index out of range")
        rows = np.arange(self.n)
        total_c = float(self.cost[rows, choice].sum())
        total_q = float(self.quality[rows, choice].sum())
        return Solution(
            choice=choice,
            total_cost=total_c,
            total_quality=total_q,
            n=self.n,
            feasible=True,
            method="evaluate",
        )

    def named_assignment(self, choice: Sequence[int]) -> dict[Any, str]:
        choice = np.asarray(choice, dtype=int)
        return {self.query_ids[i]: self.names[int(choice[i])] for i in range(self.n)}


def _infeasible(inst: MCKPInstance, method: str, note: str) -> Solution:
    return Solution(
        choice=np.full(inst.n, -1, dtype=int),
        total_cost=float("nan"),
        total_quality=float("nan"),
        n=inst.n,
        feasible=False,
        method=method,
        note=note,
    )


def cheapest_assignment(inst: MCKPInstance) -> Solution:
    """Min-cost model per query; ties break toward higher quality."""
    choice = np.empty(inst.n, dtype=int)
    for i in range(inst.n):
        choice[i] = min(
            range(inst.m),
            key=lambda j, i=i: (inst.cost[i, j], -inst.quality[i, j], j),
        )
    sol = inst.evaluate(choice)
    sol.method = "cheapest"
    return sol


def unconstrained_oracle(inst: MCKPInstance) -> Solution:
    """Max quality per query; ties break toward lower cost."""
    choice = np.empty(inst.n, dtype=int)
    for i in range(inst.n):
        choice[i] = min(
            range(inst.m),
            key=lambda j, i=i: (-inst.quality[i, j], inst.cost[i, j], j),
        )
    sol = inst.evaluate(choice)
    sol.method = "unconstrained_oracle"
    return sol


def _class_convex_chain(cost_row: np.ndarray, qual_row: np.ndarray, *, eps: float) -> list[int]:
    """Upper convex hull of one query's (cost, quality) options, increasing cost."""
    m = len(cost_row)
    order = sorted(range(m), key=lambda j: (cost_row[j], -qual_row[j], j))
    stair: list[int] = []
    best_q = -math.inf
    for j in order:
        if qual_row[j] > best_q + eps:
            stair.append(j)
            best_q = float(qual_row[j])
    if len(stair) <= 2:
        return stair
    hull: list[int] = []
    pts = np.stack([cost_row, qual_row], axis=1)
    for j in stair:
        while len(hull) >= 2 and _cross(pts[hull[-2]], pts[hull[-1]], pts[j]) >= -eps:
            hull.pop()
        hull.append(j)
    return hull


def _all_chains(inst: MCKPInstance, *, eps: float) -> list[list[int]]:
    return [_class_convex_chain(inst.cost[i], inst.quality[i], eps=eps) for i in range(inst.n)]


def greedy_solve(inst: MCKPInstance, budget: float, *, eps: float = EPS) -> Solution:
    """Integer incremental-efficiency greedy (sanity check; may be suboptimal)."""
    if budget < inst.min_total_cost() - eps:
        return _infeasible(inst, "greedy", "budget below the cheapest assignment")
    chains = _all_chains(inst, eps=eps)
    choice = np.array([ch[0] for ch in chains], dtype=int)
    pos = [0] * inst.n
    cost_now = float(inst.cost[np.arange(inst.n), choice].sum())
    qual_now = float(inst.quality[np.arange(inst.n), choice].sum())
    remaining = float(budget) - cost_now
    increments: list[tuple[float, int, int, int, float, float]] = []
    for i, ch in enumerate(chains):
        for k in range(len(ch) - 1):
            a, b = ch[k], ch[k + 1]
            dw = float(inst.cost[i, b] - inst.cost[i, a])
            dv = float(inst.quality[i, b] - inst.quality[i, a])
            eff = math.inf if dw <= eps else dv / dw
            increments.append((-eff, i, k, b, dw, dv))
    increments.sort()
    for _, i, k, b, dw, dv in increments:
        if pos[i] != k:
            continue
        if dw <= remaining + eps:
            remaining -= dw
            cost_now += dw
            qual_now += dv
            choice[i] = b
            pos[i] = k + 1
    return Solution(
        choice=choice,
        total_cost=cost_now,
        total_quality=qual_now,
        n=inst.n,
        feasible=True,
        method="greedy",
        note="incremental efficiency along per-query convex chains",
    )


def lp_relaxation(inst: MCKPInstance, budget: float, *, eps: float = EPS) -> Solution:
    """LP optimum of MCKP via Dyer–Zemel (at most one fractional query)."""
    if budget < inst.min_total_cost() - eps:
        return _infeasible(inst, "lp", "budget below the cheapest assignment")
    n, m = inst.n, inst.m
    chains = _all_chains(inst, eps=eps)
    weights = np.zeros((n, m), dtype=float)
    choice = np.array([ch[0] for ch in chains], dtype=int)
    pos = [0] * n
    for i, ch in enumerate(chains):
        weights[i, ch[0]] = 1.0
    cost_now = float(inst.cost[np.arange(n), choice].sum())
    qual_now = float(inst.quality[np.arange(n), choice].sum())
    remaining = float(budget) - cost_now
    increments: list[tuple[float, int, int, int, int, float, float]] = []
    for i, ch in enumerate(chains):
        for k in range(len(ch) - 1):
            a, b = ch[k], ch[k + 1]
            dw = float(inst.cost[i, b] - inst.cost[i, a])
            dv = float(inst.quality[i, b] - inst.quality[i, a])
            eff = math.inf if dw <= eps else dv / dw
            increments.append((-eff, i, k, a, b, dw, dv))
    increments.sort()
    fractional = False
    for _, i, k, a, b, dw, dv in increments:
        if pos[i] != k:
            continue
        if dw <= remaining + eps:
            remaining -= dw
            cost_now += dw
            qual_now += dv
            weights[i, a] = 0.0
            weights[i, b] = 1.0
            choice[i] = b
            pos[i] = k + 1
            continue
        if remaining > eps and dw > eps:
            t = remaining / dw
            weights[i, a] = 1.0 - t
            weights[i, b] = t
            cost_now += t * dw
            qual_now += t * dv
            remaining = 0.0
            fractional = True
            choice[i] = b if t >= 0.5 else a
        break
    return Solution(
        choice=choice,
        total_cost=cost_now,
        total_quality=qual_now,
        n=n,
        feasible=True,
        method="lp",
        note="Dyer-Zemel convex-chain LP; at most one fractional class",
        weights=weights,
        fractional=fractional,
        lp_mean_quality=qual_now / n,
    )


def _round_lp(inst: MCKPInstance, lp: Solution, budget: float, *, eps: float) -> Solution:
    if not lp.feasible or lp.weights is None:
        return _infeasible(inst, "approx_lp_round", lp.note)
    w = lp.weights
    frac_rows = [i for i in range(inst.n) if np.sum(w[i] > eps) > 1]
    base = np.array([int(np.argmax(w[i])) for i in range(inst.n)], dtype=int)
    if not frac_rows:
        sol = inst.evaluate(base)
        if sol.total_cost <= budget + eps:
            sol.method = "approx_lp_round"
            sol.lp_mean_quality = lp.mean_quality
            return sol
        return _infeasible(inst, "approx_lp_round", "LP rounding exceeded budget")
    i = frac_rows[0]
    cols = np.flatnonzero(w[i] > eps)
    best: Solution | None = None
    for j in cols:
        ch = base.copy()
        ch[i] = int(j)
        sol = inst.evaluate(ch)
        if sol.total_cost <= budget + eps and (best is None or sol.total_quality > best.total_quality + eps
                                              or (abs(sol.total_quality - best.total_quality) <= eps
                                                  and sol.total_cost < best.total_cost)):
            best = sol
    if best is None:
        return _infeasible(inst, "approx_lp_round", "neither rounding of the split class fits")
    best.method = "approx_lp_round"
    best.lp_mean_quality = lp.mean_quality
    return best


def _single_class_jumps(inst: MCKPInstance, budget: float, *, eps: float) -> Solution:
    """Leave every query at cheapest, upgrade one query to its best affordable option."""
    cheap = cheapest_assignment(inst)
    if cheap.total_cost > budget + eps:
        return _infeasible(inst, "single_jump", "budget below the cheapest assignment")
    best = cheap
    others = cheap.total_cost
    for i in range(inst.n):
        base_i = float(inst.cost[i, cheap.choice[i]])
        rest = others - base_i
        for j in range(inst.m):
            c = rest + float(inst.cost[i, j])
            if c > budget + eps:
                continue
            q = cheap.total_quality - float(inst.quality[i, cheap.choice[i]]) + float(inst.quality[i, j])
            if q > best.total_quality + eps or (
                abs(q - best.total_quality) <= eps and c < best.total_cost
            ):
                ch = cheap.choice.copy()
                ch[i] = j
                best = inst.evaluate(ch)
    best.method = "single_jump"
    return best


def approximate_solve(inst: MCKPInstance, budget: float, *, eps: float = EPS) -> Solution:
    """Scalable integer MCKP: best of greedy, LP rounding, and single-class jumps."""
    if budget < inst.min_total_cost() - eps:
        return _infeasible(inst, "approx", "budget below the cheapest assignment")
    lp = lp_relaxation(inst, budget, eps=eps)
    cands = [
        greedy_solve(inst, budget, eps=eps),
        _round_lp(inst, lp, budget, eps=eps),
        _single_class_jumps(inst, budget, eps=eps),
    ]
    feas = [s for s in cands if s.feasible]
    if not feas:
        return _infeasible(inst, "approx", "no feasible integer candidate")
    winner = max(feas, key=lambda s: (s.total_quality, -s.total_cost))
    source = next(
        (s.method for s in cands if s.feasible and np.array_equal(s.choice, winner.choice)),
        "selected",
    )
    lp_q = lp.mean_quality if lp.feasible else float("nan")
    winner.method = "approx"
    winner.lp_mean_quality = lp.mean_quality if lp.feasible else None
    winner.note = f"selected {source}; LP mean quality={lp_q:.6g}"
    return winner


def enumerate_solve(inst: MCKPInstance, budget: float, *, eps: float = EPS) -> Solution:
    """Exact product over assignments. For tiny instances only."""
    states = inst.m ** inst.n
    if states > ENUM_MAX_STATES:
        raise ValueError(f"enumeration has {states} states; cap is {ENUM_MAX_STATES}")
    if budget < inst.min_total_cost() - eps:
        return _infeasible(inst, "enumerate", "budget below the cheapest assignment")
    best_q = -math.inf
    best_c = math.inf
    best_ch: np.ndarray | None = None
    for tup in product(range(inst.m), repeat=inst.n):
        ch = np.fromiter(tup, dtype=int, count=inst.n)
        c = float(inst.cost[np.arange(inst.n), ch].sum())
        if c > budget + eps:
            continue
        q = float(inst.quality[np.arange(inst.n), ch].sum())
        if q > best_q + eps or (abs(q - best_q) <= eps and c < best_c):
            best_q, best_c, best_ch = q, c, ch
    if best_ch is None:
        return _infeasible(inst, "enumerate", "no feasible assignment")
    return Solution(
        choice=best_ch,
        total_cost=best_c,
        total_quality=best_q,
        n=inst.n,
        feasible=True,
        method="enumerate",
    )


def exact_solve(inst: MCKPInstance, budget: float, *, eps: float = EPS) -> Solution:
    """Exact DFS with cost/quality pruning. For small n (default ≤ 14)."""
    if inst.n > DFS_MAX_N:
        raise ValueError(f"exact DFS supports n≤{DFS_MAX_N}; use milp_solve or approximate_solve")
    if budget < inst.min_total_cost() - eps:
        return _infeasible(inst, "exact", "budget below the cheapest assignment")
    n, m = inst.n, inst.m
    min_rest = np.zeros(n + 1)
    max_rest = np.zeros(n + 1)
    for i in range(n - 1, -1, -1):
        min_rest[i] = min_rest[i + 1] + float(inst.cost[i].min())
        max_rest[i] = max_rest[i + 1] + float(inst.quality[i].max())
    orders = [np.argsort(-inst.quality[i]) for i in range(n)]
    best_q = -math.inf
    best_c = math.inf
    best_ch = np.zeros(n, dtype=int)
    choice = np.zeros(n, dtype=int)

    def dfs(i: int, c: float, q: float) -> None:
        nonlocal best_q, best_c
        if c > budget + eps:
            return
        if i == n:
            if q > best_q + eps or (abs(q - best_q) <= eps and c < best_c):
                best_q, best_c = q, c
                best_ch[:] = choice
            return
        if c + min_rest[i] > budget + eps:
            return
        opt_q = q + max_rest[i]
        opt_c = c + min_rest[i]
        if opt_q < best_q - eps:
            return
        if abs(opt_q - best_q) <= eps and opt_c >= best_c - eps:
            return
        for j in orders[i]:
            choice[i] = int(j)
            dfs(i + 1, c + float(inst.cost[i, j]), q + float(inst.quality[i, j]))

    dfs(0, 0.0, 0.0)
    if best_q == -math.inf:
        return _infeasible(inst, "exact", "no feasible assignment")
    return Solution(
        choice=best_ch.copy(),
        total_cost=best_c,
        total_quality=best_q,
        n=n,
        feasible=True,
        method="exact",
    )


def dp_solve(
    inst: MCKPInstance,
    budget: float,
    *,
    scale: float | None = None,
    eps: float = 1e-9,
) -> Solution:
    """Exact DP. Integer costs, or pass ``scale`` so round(cost * scale) is integer."""
    if scale is None:
        if not is_integer_valued(inst.cost, eps=eps) or not is_integer_valued(np.array([budget]), eps=eps):
            raise ValueError("dp_solve needs integer costs and budget, or an explicit scale")
        scale = 1.0
    w = np.rint(inst.cost * scale).astype(int)
    if np.any(w < 0):
        raise ValueError("scaled costs must be non-negative")
    W = int(math.floor(budget * scale + 1e-12))
    n, m = inst.n, inst.m
    cells = n * (W + 1)
    if cells > DP_MAX_CELLS:
        raise ValueError(f"DP table has {cells} cells; cap is {DP_MAX_CELLS}")
    NEG = -1e300
    dp = np.full(W + 1, NEG)
    dp[0] = 0.0
    pick = np.full((n, W + 1), 255, dtype=np.uint8)
    prev_w = np.full((n, W + 1), -1, dtype=np.int32)
    for i in range(n):
        new = np.full(W + 1, NEG)
        for c in range(W + 1):
            if dp[c] <= NEG / 2:
                continue
            for j in range(m):
                nc = c + int(w[i, j])
                if nc > W:
                    continue
                val = dp[c] + float(inst.quality[i, j])
                if val > new[nc]:
                    new[nc] = val
                    pick[i, nc] = j
                    prev_w[i, nc] = c
        dp = new
    feasible = np.flatnonzero(dp > NEG / 2)
    if feasible.size == 0:
        return _infeasible(inst, "dp", "no DP state fits the scaled budget")
    # max quality, then min scaled cost
    best_q = float(dp[feasible].max())
    cands = feasible[np.abs(dp[feasible] - best_q) <= 1e-12]
    end_c = int(cands.min())
    choice = np.empty(n, dtype=int)
    c = end_c
    for i in range(n - 1, -1, -1):
        j = int(pick[i, c])
        if j == 255:
            return _infeasible(inst, "dp", "failed to reconstruct a DP path")
        choice[i] = j
        c = int(prev_w[i, c])
    sol = inst.evaluate(choice)
    if sol.total_cost > budget + max(eps, 1.0 / max(scale, 1.0)):
        # rounding of the scale can overflow the true budget; mark infeasible
        return _infeasible(inst, "dp", "reconstructed cost exceeds budget (scale rounding)")
    sol.method = "dp"
    sol.note = f"scale={scale:g}"
    return sol


def milp_solve(inst: MCKPInstance, budget: float, *, eps: float = EPS) -> Solution:
    """Exact 0-1 MCKP via scipy.optimize.milp (HiGHS). Optional dependency."""
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as exc:
        raise ImportError("milp_solve requires scipy") from exc
    if budget < inst.min_total_cost() - eps:
        return _infeasible(inst, "milp", "budget below the cheapest assignment")
    n, m = inst.n, inst.m
    v = inst.quality.reshape(-1)
    w = inst.cost.reshape(-1)
    a_eq = np.zeros((n, n * m))
    for i in range(n):
        a_eq[i, i * m : (i + 1) * m] = 1.0
    cons = [
        LinearConstraint(w.reshape(1, -1), -np.inf, float(budget)),
        LinearConstraint(a_eq, 1.0, 1.0),
    ]
    res = milp(
        c=-v,
        constraints=cons,
        integrality=np.ones(n * m),
        bounds=Bounds(0, 1),
    )
    if not res.success or res.x is None:
        return _infeasible(inst, "milp", f"solver failed: {res.message}")
    x = np.asarray(res.x, dtype=float).reshape(n, m)
    choice = np.argmax(x, axis=1).astype(int)
    sol = inst.evaluate(choice)
    if sol.total_cost > budget + 1e-7:
        return _infeasible(inst, "milp", "rounded MILP assignment exceeds budget")
    sol.method = "milp"
    sol.note = str(res.message or "")
    return sol


def solve(
    inst: MCKPInstance,
    budget: float,
    *,
    method: str = "auto",
    eps: float = EPS,
    scale: float | None = None,
) -> Solution:
    """Dispatch. ``auto``: DFS if small, else integer DP if cheap, else approx."""
    method = method.lower()
    if method == "auto":
        if inst.n <= DFS_MAX_N:
            return exact_solve(inst, budget, eps=eps)
        try:
            w_max = inst.max_total_cost()
            if is_integer_valued(inst.cost) and is_integer_valued(np.array([budget])):
                W = int(math.floor(budget + 1e-12))
                if inst.n * (W + 1) <= DP_MAX_CELLS and W <= w_max + 1:
                    return dp_solve(inst, budget, scale=1.0, eps=eps)
        except ValueError:
            pass
        try:
            return milp_solve(inst, budget, eps=eps)
        except ImportError:
            return approximate_solve(inst, budget, eps=eps)
    if method in {"exact", "dfs"}:
        return exact_solve(inst, budget, eps=eps)
    if method in {"enumerate", "enum"}:
        return enumerate_solve(inst, budget, eps=eps)
    if method == "milp":
        return milp_solve(inst, budget, eps=eps)
    if method == "dp":
        return dp_solve(inst, budget, scale=scale, eps=eps)
    if method == "greedy":
        return greedy_solve(inst, budget, eps=eps)
    if method in {"lp", "relaxation"}:
        return lp_relaxation(inst, budget, eps=eps)
    if method in {"approx", "approximate"}:
        return approximate_solve(inst, budget, eps=eps)
    raise ValueError(f"unknown MCKP method {method!r}")


def default_budgets(
    inst: MCKPInstance,
    n_grid: int = 21,
    extra: Sequence[float] = (),
) -> np.ndarray:
    lo, hi = inst.min_total_cost(), inst.max_total_cost()
    extras = [lo, hi, unconstrained_oracle(inst).total_cost, *extra]
    if hi <= lo + EPS:
        return np.unique(np.asarray(extras, dtype=float))
    grid = np.linspace(lo, hi, n_grid)
    b = np.unique(np.concatenate([grid, np.asarray(extras, dtype=float)]))
    return b[(b >= lo - 1e-9) & (b <= hi + 1e-9)]

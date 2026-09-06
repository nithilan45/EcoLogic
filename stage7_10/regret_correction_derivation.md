# Stage 8 — the exact correction term for confusion-matrix energy regret

## The problem

Stage 6 computed the same quantity two ways and got opposite signs. The confusion-matrix formula said the router **saved** energy relative to the oracle (-0.2496 J/item); measuring each item's actual energy said it **spent more** (+0.2840 J/item). One of them is answering a different question, and it is the formula.

## Setup and notation

Let `X` be an item drawn from the evaluation set (the empirical distribution is the probability measure). For each tier `j`, let `e_j(X)` be the **actual** energy of tier `j`'s response to item `X`, and let `e_j := E[e_j(X)]` be the tier mean. Let `t*(X)` be the oracle tier and `t^(X)` the router's tier. Write `C_ij := {t* = i, t^ = j}` for a confusion-matrix cell and `D_ij(X) := e_j(X) - e_i(X)`.

The two estimators are

```
R_true  = E[ e_{t^(X)}(X) - e_{t*(X)}(X) ]
R_naive = sum_{i != j} P(C_ij) * (e_j - e_i)
```

## Proposition

> **Proposition.** For any joint distribution of `(X, t*, t^)` with finite first moments,
>
> ```
> R_true = R_naive + sum_{i != j} Cov( 1{C_ij}, D_ij(X) )
> ```
>
> Consequently `R_naive = R_true` if and only if `sum_{i != j} Cov(1{C_ij}, D_ij(X)) = 0`. A sufficient condition is that each `D_ij(X)` be mean-independent of cell membership, `E[D_ij | C_ij] = E[D_ij]`; in particular the naive formula is exact whenever `e_j(X)` is constant within each tier, i.e. whenever energy does not vary across items.

**Proof.** Partition on the cells. Since the cells `C_ij` are exhaustive and mutually exclusive,

```
R_true = sum_{i,j} P(C_ij) * E[ e_j(X) - e_i(X) | C_ij ]
       = sum_{i != j} P(C_ij) * E[ D_ij(X) | C_ij ]
```

where the diagonal terms drop out because `D_ii(X) = 0` pointwise. Subtracting the naive estimator term by term over the same index set,

```
R_true - R_naive = sum_{i != j} P(C_ij) * ( E[D_ij | C_ij] - E[D_ij] )
```

For any event `A` and integrable `D`, the definition of covariance gives

```
Cov(1_A, D) = E[1_A * D] - E[1_A] * E[D]
            = P(A) * E[D | A] - P(A) * E[D]
            = P(A) * ( E[D | A] - E[D] )
```

Applying this with `A = C_ij` and `D = D_ij(X)` to each summand yields the claim. The 'if and only if' is immediate, and the sufficient condition follows because `E[D_ij | C_ij] = E[D_ij]` makes each summand zero. $\blacksquare$

This is direct algebra from the definition of covariance, not a deep result. It is worth writing down only because the naive formula is used to report energy savings and, as the numbers below show, it can carry the wrong sign.

### A note on notation

The correction is sometimes stated informally as `sum P(C_ij) * Cov(D_ij | C_ij)`. That expression does not type-check — a covariance needs two arguments, and a conditional variance of `D_ij` is not what appears here. The correct object is the covariance between the **cell indicator** and the energy difference, equivalently `P(C_ij)` times the **conditional mean shift** `E[D_ij | C_ij] - E[D_ij]`. Both forms above are exact and identical; the informal one captures the right intuition ("naive is exact only when energy is uncorrelated with routing within each cell") but is not a usable formula.

## Reconciliation on Stage 6's own data

CALIBRATION split, n = 360, variant R2, threshold tau = 0.2245. Tier means used by the naive formula: `e_1` = 2.142382 J, `e_2` = 1.064663 J, `e_3` = 14.329333 J.

| Quantity | Value (J/item) |
|---|---|
| `R_naive` (confusion-matrix formula) | -0.249623217593 |
| correction `sum Cov(1{C_ij}, D_ij)` | 0.533592662037 |
| **`R_naive` + correction** | **0.283969444444** |
| `R_true` (direct per-item measurement) | **0.283969444444** |
| absolute residual | 2.220e-16 |

**They reconcile exactly** — residual 2.220e-16, i.e. floating-point noise. The correction term is **+0.5336 J/item**, which is **1.9x the size of the true regret itself** and large enough to flip the sign: the naive formula reports -0.2496 where the truth is +0.2840.

The correction was derived first and evaluated once; it was not tuned to make the numbers agree.

## Where the error comes from, cell by cell

| Cell (t\* -> t^) | P(cell) | E[D] uncond. | E[D] given cell | Cov(1,D) | naive contribution |
|---|---|---|---|---|---|
| 1 -> 2 | 0.2944 | -1.0777 | +0.6742 | +0.515856 | -0.317329 |
| 1 -> 3 | 0.0139 | +12.1870 | +8.2560 | -0.054597 | +0.169263 |
| 2 -> 1 | 0.0083 | +1.0777 | +2.3755 | +0.010815 | +0.008981 |
| 2 -> 3 | 0.0111 | +13.2647 | +6.9555 | -0.070102 | +0.147385 |
| 3 -> 2 | 0.0194 | -13.2647 | -6.4956 | +0.131620 | -0.257924 |

The mechanism is visible in the two middle columns: within a cell, the energy difference `D_ij` is nothing like its unconditional average. The router sends *cheap* items to Tier 2 and the items it leaves on Tier 1 are the expensive, long-reasoning ones, so cell membership and energy are strongly dependent. Substituting tier means for per-item energies throws exactly that dependence away.

## Practical consequence

Any routing evaluation that reports energy or cost savings by multiplying a confusion matrix (or a routing distribution) by per-model average costs is making this substitution. It is safe only when per-item cost is approximately constant within each model. That condition fails hard for reasoning models, whose token counts vary by an order of magnitude across items — and it fails in the direction that flatters the router, because routers preferentially send short, easy items to cheap models. The fix requires no new modelling, only per-item cost data: report `R_true` directly, or report `R_naive` together with the correction term above.

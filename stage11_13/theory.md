# Why routers gain so little: a complementarity × predictability decomposition

**Status.** Every proposition here is proved below and checked numerically in
`s11_validate.py` against a brute-force reference (an LP solver or Monte Carlo).
`s11_validate.json` records **14/14 checks passing**. The propositions are
elementary — Cauchy–Schwarz, the law of total variance, a fractional knapsack —
and are presented as such. What is not elementary is that they identify *which*
of two multiplicative factors is the binding constraint on LLM routing, and that
one of the two is **not measurable at all** from the data the field publishes.

---

## 1. Setup

A query `x` is drawn from a workload `D`. There are `K` models. Running model
`m` on `x` yields a **utility** `u_m(x) ∈ [0,1]` (graded correctness; binary
correctness is the special case `u_m ∈ {0,1}`) and a **cost** `c_m(x) ≥ 0`
(dollars, tokens, or joules — the algebra does not care).

Crucially, `u_m(x)` and `c_m(x)` are **random even given `x`**. Decoding is
stochastic, and Stage 10(a) measured that it is stochastic even at temperature 0:
our Tier 1 model changes its *graded verdict* on 19.8% of items between identical
temperature-0 calls. Write

```
eta_m(x)   = E[ u_m(x) | x ]        the model's success probability on x
gamma_m(x) = E[ c_m(x) | x ]        its expected cost on x
```

A **policy** assigns queries to models. Let `z_m(x) ∈ [0,1]`, `Σ_m z_m(x) = 1`,
be the probability it picks model `m`. Its value and cost are

```
A(z) = E[ Σ_m z_m(x) u_m(x) ]        C(z) = E[ Σ_m z_m(x) c_m(x) ]
```

Three nested policy classes, distinguished only by **what they are allowed to
look at**:

| Class | May depend on | Reached value at budget `b` |
|---|---|---|
| `Π₀` **query-independent** | nothing | `S(b)` |
| `Π₁` **router** | `x` | `A†(b)` |
| `Π₂` **oracle** | `x` *and the realised* `u`, `c` | `A*(b)` |

where each value is the supremum of `A(z)` over that class subject to
`C(z) ≤ b`. `Π₀ ⊂ Π₁ ⊂ Π₂`: a constant is a function of `x`, and a function of
`x` is a function of `(x, u, c)`.

`Π₀` is the bar a router must clear. A member of `Π₀` is "always use model 3", or
"send 40% of traffic to the big model and 60% to the small one, at random" — the
matched-cost static mixture that Stages 1–9 argued is the correct baseline and
that the routing literature does not report. `Π₂` is the "oracle" whose headroom
routing papers quote.

---

## 2. The ladder and the decomposition

### Proposition 1 (ladder).
`S(b) ≤ A†(b) ≤ A*(b)`, and all three are concave and nondecreasing in `b`.

*Proof.* The inequalities are immediate from `Π₀ ⊂ Π₁ ⊂ Π₂`. For concavity: `A`
and `C` are linear in `z`, each class is convex and closed under mixing, so if
`z¹` is feasible at `b₁` and `z²` at `b₂` then `t z¹ + (1-t) z²` is feasible at
`t b₁ + (1-t) b₂` and attains at least `t A(z¹) + (1-t) A(z²)`. Monotonicity is
immediate since enlarging `b` enlarges the feasible set. ∎

### Definition (the two factors).
```
complementarity   kappa(b) = A*(b) - S(b)
predictability    rho(b)   = ( A†(b) - S(b) ) / kappa(b)     (kappa(b) > 0)
```
so that

> **achievable router gain at budget `b` = rho(b) × kappa(b).**

And for any *actual* router `ẑ` with `C(ẑ) ≤ b`,

```
realised gain  =  A(ẑ) - S(b)  =  rho(b) · kappa(b)  -  eps(ẑ),      eps(ẑ) ≥ 0
```

with `eps(ẑ)` the excess risk of `ẑ` against the Bayes router. **Three factors,
three different kinds of problem:**

| Factor | Question it answers | Whose fault a small value is |
|---|---|---|
| `kappa(b)` | do the models cover each other's failures at all? | the model *set* — fix it by choosing more complementary models |
| `rho(b)` | is which-model-will-succeed a function of the query? | the **task** — not fixable by any router |
| `eps(ẑ)` | did we fit a good enough router? | the **engineer** — fixable with better features, data, architecture |

The identity is true by construction. What makes it useful is that each factor is
separately measurable or boundable, and that they behave completely differently:
`kappa` is large and easy to measure, `eps` is what the field works on, and
`rho` — which turns out to be the binding constraint — is invisible in every
public routing dataset. §4 explains why.

### Proposition 1b (what `rho < 1` actually is).
`rho(b) = 1` for all `b` if `u_m(x)` and `c_m(x)` are deterministic given `x` for
every `m`. Consequently every unit of `rho < 1` is **outcome randomness given the
query**, not feature weakness or finite training data — those live in `eps`.

*Proof.* If `u, c` are `σ(x)`-measurable then `Π₁ = Π₂` as sets of policies with
the same value and cost functionals, so `A† = A*`. ∎

This is worth stating explicitly because it inverts how oracle headroom is
usually read. The "oracle" of a routing paper is computed from **one generation
per model per item**, so it is free to route each item to whichever model *happened*
to get that item right on that draw. When outcomes are noisy, much of what that
oracle collects is a coin flip it saw and the router cannot. Published oracle
headroom is therefore an **upper bound inflated by generation noise**, and
Proposition 3 quantifies by how much.

---

## 3. Two models: closed forms, and the ceiling

Take `K = 2`: a cheap model `s` and an expensive model `l`, and write
`beta` for the fraction of traffic sent to `l`, so the budget is
`b(beta) = (1-beta)·E[c_s] + beta·E[c_l]`. Let

```
delta(x) = eta_l(x) - eta_s(x)      the predictable advantage of the big model
D        = u_l - u_s                its realised advantage
```

Note `delta(x) = E[D | x]`. Let `CTE_beta(Z)` denote the mean of the upper
`beta`-tail of `Z` (the superquantile / conditional tail expectation).

### Proposition 2 (closed forms).
Assume per-model costs are constant (`c_m(x) ≡ c_m`, relaxed in §6) and
`E[u_l] ≥ E[u_s]`. Then for `beta ∈ [0,1]`:

**(a)** `S(b(beta)) = E[u_s] + beta·E[D]`.

**(b)** `A†(b(beta)) = E[u_s] + beta·CTE_beta(delta)`, attained by sending to `l`
the `beta` fraction of items with the largest `delta(x)`. Hence
```
A†(b) - S(b) = beta · ( CTE_beta(delta) - E[delta] )
```

**(c)** `A*(b(beta)) = E[u_s] + beta·CTE_beta(D)`, hence
`kappa(b) = beta·( CTE_beta(D) - E[D] )`, and therefore
```
rho(beta) = ( CTE_beta(delta) - E[delta] ) / ( CTE_beta(D) - E[D] )
```

**(d)** In the binary case, the gain of an *unconstrained* oracle over the better
single model is exactly
```
kappa_0 = min(p01, p10),      p01 = P(u_s=0, u_l=1),  p10 = P(u_s=1, u_l=0)
```

*Proof.* (a) A query-independent policy mixes the two points, and the value is
linear in `beta`, so the budget binds. (b) Maximise `E[u_s] + E[z·delta]` over
`z: X → [0,1]` with `E[z] ≤ beta`; this is a fractional knapsack whose solution
puts `z = 1` on the largest values of `delta`, giving `beta·CTE_beta(delta)`.
(c) Identical, with the oracle ranking by realised `D` instead. (d) Any policy has
value at most `E[max(u_s, u_l)] = 1 - p00`, attained by the cheapest-correct
oracle. Since `E[u_s] = p11+p10` and `E[u_l] = p11+p01`, we get
`(1-p00) - E[u_s] = p01` and `(1-p00) - E[u_l] = p10`, so the gain over the
better single model is `min(p01, p10)`. ∎

Part (c) is the interpretation that matters: **predictability is the ratio of the
tail dispersion of the conditional mean to the tail dispersion of the
realisation.** Since `delta = E[D|x]` and `CTE` is a coherent risk measure,
conditioning contracts it, which re-proves `rho ≤ 1`.

Part (d) is a useful screening statistic but is *not* `kappa`: it measures
headroom over the better single model, whereas `kappa(b)` measures headroom over
the matched-cost *mixture*, which is worse than the better single model whenever
`beta < 1`. A router can gain over a mixture even when `kappa_0 = 0`, simply by
spending a fixed budget where it does the most good. `kappa(b)` is the quantity
in the decomposition; `kappa_0` is reported alongside it because it is the
quantity a reader intuits.

### Proposition 3 (the router-free ceiling). ★
For **any** router at operating fraction `beta`,
```
A†(b(beta)) - S(b(beta))  ≤  sqrt( beta(1-beta) ) · sd(delta)  ≤  (1/2)·sd(delta)
```
and the bound is attained.

*Proof.* For any `z: X → [0,1]` with `E[z] = beta`,
```
E[z·delta] - beta·E[delta] = Cov(z, delta) ≤ sd(z)·sd(delta)
```
by Cauchy–Schwarz. A `[0,1]`-valued variable with mean `beta` has variance at
most `beta(1-beta)`, attained by the two-point `{0,1}` distribution. Substituting
gives the bound; the maximum over `beta` is at `beta = 1/2`. Tightness: take
`delta` two-valued and `z` its indicator, so `z` and `delta` are perfectly
correlated and both bounds bind. ∎

Two remarks.

- The bound needs **no assumption** on the distribution of `delta` — no
  continuity, no independence, no parametric family. It is a hard cap on every
  router that will ever be written for this workload and model pair.
- It is the **same covariance object** as the Stage 8 cost-accounting correction,
  `R_true = R_naive + Σ_{i≠j} Cov(1{t*=i, t̂=j}, e_j − e_i)`. There, a covariance
  between a routing indicator and a cost difference biases the cost axis; here, a
  covariance between a routing indicator and a *utility* difference caps the
  accuracy gain. Routing lives or dies on one covariance, measured on two
  different axes.

### Proposition 5 (the AUC a router could reach).
For a single model with `a = E[eta(X)]` and independent `X, X' ~ D`, the
Bayes-optimal AUC for predicting `u(x)` from `x` is exactly
```
AUC* = 1/2 + E| eta(X) - eta(X') | / ( 4 a (1-a) )
```
(with the usual half-credit convention for ties), and consequently
```
AUC* ≤ 1/2 + sd(eta) / ( 2·sqrt(2)·a·(1-a) )
```

*Proof.* Score by `eta`, which is Bayes-optimal for AUC. Write `u = eta(X)`,
`v = eta(X')` and `T_> = E[1{u>v} u (1-v)]`, `T_< = E[1{u<v} u (1-v)]`,
`T_= = E[1{u=v} u (1-v)]`. The AUC with half-credit is
`(T_> + T_=/2) / (a(1-a))`, and `T_> + T_< + T_= = E[u(1-v)] = a(1-a)`. By
exchangeability `T_< = E[1{u>v} v (1-u)]`, so
`T_> - T_< = E[1{u>v}(u-v)] = E|u-v|/2`. Solving,
`T_> + T_=/2 = a(1-a)/2 + E|u-v|/4`. The bound follows from
`E|u-v| ≤ sqrt(E(u-v)²) = sqrt(2)·sd(eta)`. ∎

This converts a quantity we can bound (`sd(eta)`, from replicates) into a
quantity we have already measured four independent times (held-out AUC ≈ 0.65–0.68
in Stages 5, 7, 7c across logistic regression, gradient boosting, random forest
and k-NN). If the two agree, our routers were already at the ceiling and the
"you tested a weak router" objection is answered without testing another router.

---

## 4. Estimating the two factors — and why only one of them is public

`kappa(b)` is easy: `S(b)` is a convex hull of `K` points and `A*(b)` is a
fractional multiple-choice knapsack, both computed directly from a matrix of
per-item utilities and costs. Any released routing dataset supports it.

`rho(b)` is hard, and the difficulty is not statistical but **structural**:
`rho` depends on `delta = E[D|x]`, and one generation per item does not identify
a conditional mean. All you get is

```
Var(D) = Var(delta) + E[ Var(D | x) ]   ≥  Var(delta)
```

so single-generation data yields only the loose bound `sd(delta) ≤ sd(D)`. Every
public routing dataset we are aware of — RouterBench, RouteLLM's released evals,
and the outcome matrices used by the routing literature generally — stores exactly
one generation per (model, prompt). **The quantity that determines whether routing
can work is not recoverable from the data the field publishes.**

### Proposition 4 (identification from replicates).
Suppose `k ≥ 2` conditionally i.i.d. replicate generations per item, binary
outcomes, and independence across models. Let `eta_hat_m` be an item's replicate
mean. Then

```
Var(eta_hat_m) = Var(eta_m) + (1/k)·E[ eta_m (1-eta_m) ]
E[ (k/(k-1))·eta_hat_m (1-eta_hat_m) ] = E[ eta_m (1-eta_m) ]
```

so `Var(eta_m)` is identified, and for `delta`,

```
Var_hat(delta) = Var(eta_hat_l - eta_hat_s) - (1/(k-1))·mean_i[ eta_hat_l(1-eta_hat_l) + eta_hat_s(1-eta_hat_s) ]
```

is unbiased for `Var(delta)`.

*Proof.* The first line is the law of total variance with
`Var(eta_hat | x) = eta(1-eta)/k`. For the second, `k·eta_hat ~ Bin(k, eta)` given
`x`, and `E[eta_hat(1-eta_hat) | x] = ((k-1)/k)·eta(1-eta)`. The `delta` line
combines both across the two models using conditional independence, with
`(1/k)·(k/(k-1)) = 1/(k-1)`. ∎

Validated in `s11_validate.py` (`P4_*`): relative bias below 0.6% at
`n = 5000, k = 3`, while the **naive** uncorrected estimator inflates
`Var(delta)` by a factor of **2.33** in the same simulation. That factor is the
size of the mistake a single-generation dataset forces.

Estimates are clipped at zero and clipping is reported when it binds.

**Assumptions, and which way they err.** (i) Replicates conditionally i.i.d.:
provider-side nondeterminism could be positively correlated within an item
(batching, caching), which would *understate* the within-item term and therefore
*overstate* `sd(delta)` and the ceiling — erring in routing's favour, which is the
conservative direction for our claim. (ii) Independence across models: our tiers
are separate API calls, two of them to different providers. (iii) Replicates were
drawn at the same temperature as the reported runs, so they measure the noise the
evaluation actually contains.

---

## 5. What the decomposition predicts, and how it can fail

Reading the three factors together:

- If `kappa` is small, the model set is redundant and routing is pointless. This
  is a **model-selection** finding.
- If `kappa` is large but `rho` is small, routing is **capped by the task**. No
  representation, no architecture, and no amount of training data helps, because
  the target is not a function of the input. The correct response is to stop
  building routers for that workload — or to reduce outcome noise first, e.g. by
  self-consistency, which changes `eta` and so changes the problem.
- If `kappa` and `rho` are both large but realised gains are small, the gap is
  `eps` and it **is** an engineering problem.

The pre-registered hypothesis (`prereg_stage11_13.md` §1) is the middle case:
`kappa` abundant, `rho` scarce. The three ways it can fail are written down
there, including the case where a fitted router captures most of `kappa` and our
Stage 5/7 negatives are revealed as a weak-router artifact.

---

## 6. Relaxations and limits of the theory

**Per-item costs.** Proposition 2 assumed constant per-model cost for legibility.
The implementation does not: `S(b)` uses per-model *mean* cost (exact for
query-independent policies, by the Stage 8 result), while `A*(b)` and every
router's realised cost use **per-item** costs, so all budget matching is on
realised spend. Proposition 3 extends by replacing `delta` with the
cost-normalised advantage; we report the constant-cost form because it is the one
whose right-hand side we can estimate, and note that per-item cost dispersion can
only add covariance for a router to exploit — so the reported ceiling is again the
generous direction.

**`K > 2`.** Propositions 1, 1b, 4 and 5 are stated for general `K`. Propositions
2 and 3 are two-model statements; for `K > 2` we compute `S`, `A*` and router
values directly (as the code does) and apply the pairwise ceiling to the relevant
pair at each operating point. A general-`K` ceiling would need a vector version of
the Cauchy–Schwarz step; we do not claim it.

**Utilities in `[0,1]`.** Everything above holds for graded utilities. Only
Proposition 2d and Proposition 4 are binary-specific — 4 because the within-item
variance formula uses the Bernoulli identity.

**What none of this shows.** The decomposition is descriptive: it says where a
routing gain must come from, not that any particular workload has low `rho`. `rho`
is a property of a (workload, model set) pair and must be measured each time. A
setting with high `rho` is perfectly consistent with this framework and would be
a positive result within it. Nothing here says routing cannot work; it says what
would have to be true for it to work, and gives the two measurements that decide
it.

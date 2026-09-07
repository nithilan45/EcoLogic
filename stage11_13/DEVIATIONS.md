# Deviations from `prereg_stage11_13.md`

Recorded in the style of the Stage 3 energy-budget deviation disclosed in
`stage7_10/evaluation_card.md` §6. Each entry states what was pre-registered,
what happened, why, and what it costs the paper.

---

## D1. **H3 is refuted. The pre-registered ceiling is too loose to bite, and we say so rather than reframing it.**

**Pre-registered (§1, §3).** H3: "the router-free upper bound on `A†(b) − S(b)` is
≤ 50% of `kappa`", i.e. at least half of measured complementarity is provably
unreachable by any router. Falsification condition, written in advance: "H3 fails
if the upper bound exceeds 50% of `kappa` (then the ceiling argument is vacuous
and we must retreat to 'we could not find a good router', which is a much weaker
paper)."

**What happened.** The bound of Proposition 3,
`gain ≤ sqrt(beta(1-beta))·sd(delta)`, evaluates at `beta = 0.5` to **8.4–15.6
accuracy points** across the three tier pairs, against a single-draw `kappa` of
**3.3–9.5 points**. The median ratio of bound to `kappa` is **2.82**, i.e. the
bound sits *above* the oracle it was supposed to constrain. It is a valid
inequality (`s11_validate.py` confirms it is never violated and is attained on a
two-point distribution) but it is **vacuous at these effect sizes**.

**Why.** The bound is Cauchy–Schwarz, tight only when `delta` is two-valued and
the routing indicator is perfectly correlated with it. Real `delta` distributions
are diffuse, so the bound loses roughly the factor by which `delta` fails to be
two-point. When `kappa` is a few accuracy points and `sd(delta)` is 0.10–0.31,
that slack swamps the quantity of interest.

**Consequence, and it is not cosmetic.** Our central Stage 5/7 claim was that
per-item tier success is *weakly predictable in principle*. **The replicate data
says the opposite.** Item-level difficulty is highly *reliable*: the share of
per-item outcome variance that is real between-item signal rather than
regeneration noise is

| Set | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| pool, temp 0.7, n = 5,000 | 0.895 | 0.873 | 0.942 |
| test, temp 0.0, n = 364 | 0.772 | 0.902 | 0.983 |

and the Proposition 5 point estimate of the Bayes-optimal AUC is **0.95–0.99**,
not 0.68. The pre-registered consistency check in §3.4 predicted the
replicate-implied `AUC*` would land within 0.05 of the measured 0.65–0.68
plateau. **It came out 0.99 against a measured 0.68 — a miss of 0.31, and the
check is recorded as INCONSISTENT.** That check was included precisely because it
could embarrass us (§3.4a), and it did.

**What this means for the claim.** Whether a given model answers a given item
correctly is *not* a coin flip. It is a stable property of the item that our
routers could not read off the prompt. So `rho ≈ 1` and the binding term in the
decomposition is `eps` — the **generalization gap from prompt text to item
difficulty** — not `rho`. The paper's claim changes from

> "per-item success is unpredictable, so routing is capped by the task"

to

> "per-item success is highly *reliable* yet only weakly *inferable from prompt
> text*; the gap is a generalization failure, it does not close with 4.2× more
> data, and it survives every router we tried."

That is a different claim, it is the one the data supports, and it is the one the
paper makes. The `kappa × rho − eps` decomposition itself is unaffected — what
changed is which factor we found to be binding, which is exactly the question the
decomposition was built to ask.

---

## D2. The split-replicate "semi-oracle" is not an upper bound on `A†`, and is no longer described as one.

**Pre-registered (§3.3).** "A policy allowed to see one realised replicate
`Y^(1)(x)` has a strictly larger information set than a router that sees `x`
alone. So the best such semi-oracle policy upper-bounds `A†`."

**The error.** That sentence is wrong. `Y^(1)` is a realised outcome, not a
function of `x`, so `sigma(Y^(1))` does **not** contain `sigma(x)`; the two
information sets are incomparable. Only their join `sigma(x, Y^(1))` contains
`sigma(x)`, and computing the optimum over that join requires modelling in `x`,
which reintroduces the router we were trying to avoid. The claim was caught while
implementing Stage 12, before any result depended on it.

**What is reported instead.** The split-replicate quantity is still computed and
still informative, but relabelled honestly: it is the **unbiased value of a
specific feasible policy that gets to peek at one graded outcome per model** —
information no deployed router has — evaluated on held-out replicates. It
captures only **34–55%** of the single-draw `kappa`. That is a statement about
how much of published oracle headroom survives when you cannot see the very draw
you are scored on. It is not a bound on `A†`, and the paper does not call it one.

---

## D3. Stage 12b (learning curves) added, not pre-registered.

**Addition.** Given D1, the binding term is `eps`, and the pre-registration
contains no instrument for it beyond the two data points Stage 7 supplies
(1,200 → 5,000 training items moved held-out AUC by 0.009). RouterBench's 36,494
items support a proper learning curve, so we add one: held-out AUC and realised
matched-cost gain against training-set size on a geometric grid.

**Status.** This is an **exploratory, non-pre-registered** analysis and is
labelled as such wherever it is reported. It is not used to judge H1 or H2, and
no verdict rests on it. It is reported because "would more data fix it?" is the
first question a reviewer asks about `eps`, and answering it with a curve is
better than declining to answer.

---

## D4. Router-fitting details.

**Pre-registered (§2.5).** Four router families, one `P(correct | x)` head per
model, hyperparameters chosen within training folds only.

**As run.** The two linear rungs use logistic regression on labels binarised at
0.5, exactly as pre-registered. The MLP and boosted-tree rungs are fitted as
**regressors on the graded score** rather than classifiers on the binarised
label, because 21.1% of RouterBench score cells are fractional and regression is
the natural estimator of `E[u | x]`. The pre-registration named the model classes
but not the loss; this is the choice that matches the estimand. The MLP is
fitted multi-output (one network, 11 heads) rather than as 11 independent
networks, for runtime; this can only *hurt* the MLP relative to independent fits,
so it is not a choice that flatters our conclusion.

**Fold structure.** 5-fold, grouped by item and stratified by benchmark family,
as pre-registered. Global folds (all families pooled) are used so every router
gets the maximum training data — 29,195 items — which is the setting least
favourable to a "you starved the router" objection.

---

## D5. Integer (discreteness) gap computed only at the headline operating point.

**Pre-registered (§2.4).** The integer MCKP solution "plus the integer solution
for the gap" for every pair and every `beta`.

**As run.** Computed at `beta = 0.5` only. The integer repair is an O(n) sweep
per instance and 55 pairs × 9 betas × 9 families made it the dominant cost of the
run for a quantity that Stages 5 and 7 already measured at ~0.00 pp. The LP value
is the one used in the decomposition, and it is computed everywhere.

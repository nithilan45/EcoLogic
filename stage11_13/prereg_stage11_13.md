# Pre-registration — Stages 11, 12, 13

**Committed before any of the analyses it governs were run.** Check `git log`:
this file's commit must precede every `stage11_13/s1*.py`, `.json` and `.md`
artifact and every `paper/` file. The pattern follows
`router_v2/PREREGISTRATION.md` and `stage7_10/prereg_stage7.md`.

Written by: the same evaluation effort as Stages 1–10. Nothing in
`raw_results/`, `results_report.md`, `router_v2/` or `stage7_10/` will be
modified; all new artifacts land in `stage11_13/`, `external_data/` and
`paper/`.

---

## 0. Why these stages exist

Stages 1–10 established a measurement claim: routing gains are reported against
the wrong baseline on a biased cost axis, and under a matched-cost
query-independent baseline the honest effect is small (EcoLogic: negative;
RouteLLM: +0.57 pp). Two objections remain unanswered and both are fatal to a
workshop submission.

1. **"You tested a weak router."** The predictability ceiling (AUC ≈ 0.68) rests
   on one representation (MiniLM) and four classical learners. A reviewer will
   say a fine-tuned LLM router would do better, and Stage 7c cannot refute that.
2. **"n = 1 system, $48 of data."** The headline is validated on our own stack
   plus one benchmark of one published router. That is a case study.

Stages 11–13 attack both, and reframe the contribution from "measure routers
better" to a **decomposition**: a router's achievable gain is the product of how
much the models complement each other and how much of that complementarity is
predictable from the query text alone.

---

## 1. The hypothesis, stated so it can fail

**H (decomposition).** For a set of models and a workload, let `S(b)` be the
best accuracy achievable at expected cost `b` by any *query-independent*
policy, `A*(b)` the best achievable by an oracle that sees the realised
per-model outcomes, and `A†(b)` the best achievable by any *router* — any
measurable function of the query `x` alone. Then

```
S(b)  <=  A†(b)  <=  A*(b)
```

and we define

```
complementarity   kappa(b) = A*(b) - S(b)        >= 0
predictability    rho(b)   = (A†(b) - S(b)) / kappa(b)   in [0, 1]
achievable gain   A†(b) - S(b) = rho(b) * kappa(b)
```

**H1 (complementarity is abundant).** Across model pairs and benchmarks,
`kappa` is large — median `kappa` over RouterBench model pairs is **>= 5.0
accuracy points** at the mid-budget operating point.

**H2 (predictability is scarce).** The *realised* gain of the best fitted
router, at matched cost, is a small fraction of `kappa` — median realised gain
over the same pairs is **<= 30% of kappa**.

**H3 (the ceiling is router-free, not router-specific).** On our own replicated
data, the router-free upper bound on `A†(b) - S(b)` derived in §3 is **<= 50% of
`kappa`** — i.e. at least half of the measured complementarity is provably
unreachable by *any* router, however strong.

**What would falsify each.** H1 fails if median `kappa` < 5.0 pp (then there is
nothing to route for, and the whole framing is wrong — the interesting claim
would become "models are redundant"). H2 fails if a fitted router captures
> 30% of `kappa` at median (then predictability is not the bottleneck and our
Stage 5/7 negatives were a weak-router artifact after all). H3 fails if the
upper bound exceeds 50% of `kappa` (then the ceiling argument is vacuous and we
must retreat to "we could not find a good router", which is a much weaker
paper). **We commit to reporting whichever of these outcomes occurs**, including
the case where H2 or H3 fails and the project's central claim is wrong.

**Explicit non-hypothesis.** We do not hypothesise that routing never works.
`rho` and `kappa` are workload- and model-set-dependent quantities; a setting
with high `rho` is entirely consistent with this framework and would be a
positive result within it.

---

## 2. Stage 11 — RouterBench external validation at scale

### 2.1 Data, fixed in advance

`withmartian/routerbench` from HuggingFace, files `routerbench_0shot.pkl`
(36,497 prompts × 11 models = 401,467 outcomes) and `routerbench_5shot.pkl`.
SHA of the dataset revision to be recorded in the manifest. **Primary analysis
uses the 0-shot file**; the 5-shot file is a pre-registered replication and its
results are reported whether or not they agree.

Per-item fields used: `prompt`, `eval_name`, per-model score, per-model
`total_cost`. Per-model `model_response` text is **not** used for the primary
analysis (we do not re-grade RouterBench).

### 2.2 Score handling, fixed in advance

Inspection *before* pre-registering (descriptive only, no hypothesis touched)
established that 6 of 86 `eval_name` values carry fractional scores:
`grade-school-math`, `mtbench`, `mtbench-math`, `mtbench-reference`,
`consensus_summary`, `chinese_idioms`. 21.1% of all score cells are fractional.

We checked whether the 0.25-step scores are means of 4 replicate generations,
which would have let us apply §3's replicate bound to RouterBench. **They are
not**: `gpt-4-1106-preview` on `grade-school-math` is modal at 0.75 and
`mistralai/mistral-7b-chat` modal at 0.25, with almost no mass at 0 or 1, which
is a graded partial-credit pattern and not a Binomial(4, p) pattern; and only
one `model_response` is stored per model per item. RouterBench therefore
**cannot** support any within-item variance decomposition. This is recorded here
so it cannot later look like a convenient omission.

Consequently:

- **The decomposition is defined for bounded utilities `u_m(x)` in [0,1]**, not
  only binary correctness, and the primary analysis runs on the scores as
  released. This is the honest choice: it requires no threshold.
- The **binary special case** (`min(p01, p10)`, McNemar-style disagreement
  cells) is reported only on the 79 strictly-binary `eval_name` values.
- Where a binary reading of a fractional eval is needed, the threshold is
  **`score >= 0.5`**, fixed here, with sensitivity at 0.25 and 0.75 reported.

### 2.3 Benchmark families, fixed in advance

`eval_name` values are grouped by prefix into these families, and no others:

| Family | Rule |
|---|---|
| `mmlu` | `eval_name` starts with `mmlu-` |
| `gsm8k` | `grade-school-math` |
| `hellaswag` | `hellaswag` |
| `winogrande` | `winogrande` |
| `arc` | `arc-challenge` |
| `mbpp` | `mbpp` |
| `mtbench` | starts with `mtbench` |
| `chinese` | starts with `chinese` or `Chinese` |
| `other` | everything else (`abstract2title`, `accounting_audit`, `bias_detection`, `consensus_summary`) |

`test-match` (n = 3) is **excluded** as a smoke-test artifact. Families with
n < 100 items are reported but excluded from the medians used to judge H1/H2.

### 2.4 Quantities computed

For every unordered model pair (55 of them) and every family, and for the full
11-model set:

- `S(b)`: upper concave envelope of the points `(mean cost_m, mean score_m)` —
  the matched-cost query-independent frontier. Computed exactly by convex hull.
- `A*(b)`: oracle frontier, by LP relaxation of the multiple-choice knapsack on
  realised per-item scores and costs, plus the integer solution for the gap.
- `A†(b)` **lower bounds**: the realised matched-cost accuracy of each fitted
  router in §2.5, evaluated out-of-fold.
- `kappa(b)`, and realised `rho_hat(b) = (A_router(b) - S(b)) / kappa(b)`.
- Binary-only: `p01`, `p10`, disagreement rate, and `min(p01, p10)` (the closed
  form for gain over the better single model, proved in Stage 11's theory doc).

**Operating point for the headline numbers**: the budget `b` equal to the cost
of sending 50% of traffic to the more expensive model of the pair, i.e.
`beta = 0.5`. Chosen because §3's bound is maximised there, so it is the most
favourable point for routing and therefore the conservative place to argue a
ceiling. Full `beta` sweeps on a grid of 0.1 to 0.9 in steps of 0.1 are
reported for every pair.

### 2.5 Router families, fixed in advance

Fitted on RouterBench prompts, per family, 5-fold cross-validation grouped by
item so no item appears in both fit and score:

1. TF-IDF word (1,2)-grams + char (3,5)-grams, logistic regression — the
   Stage 5 R1 representation.
2. `all-MiniLM-L6-v2` sentence embeddings + logistic regression — the Stage 5/7
   R2 representation.
3. `all-MiniLM-L6-v2` + 2-hidden-layer MLP (256, 64), early stopping on an
   inner split — **new, and the pre-registered answer to "you used a linear
   head"**.
4. Gradient boosting on MiniLM features.

One `P(correct | x)` head per model, then the matched-cost policy is: rank items
by the predicted score difference and send the top `beta` fraction to the
expensive model. This is the Bayes-optimal *form* given the predicted scores,
so the only thing being tested is the quality of the predictions.

Hyperparameters are selected **within training folds only**. No test-set
selection.

### 2.6 Statistics

- **Confidence intervals**: 2,000-resample nonparametric bootstrap over items
  (the sampling unit), percentile method, reported for `kappa`, realised gain
  and `rho_hat`. Bootstrap resamples items within family, preserving n.
- **Multiple comparisons**: any claim of the form "the router beats the matched
  static baseline for pair p" is tested across all 55 pairs and corrected by
  **Holm–Bonferroni** at family-wise alpha = 0.05. Benjamini–Hochberg
  (FDR 0.05) is reported alongside as the less conservative view. Both are
  reported; neither is chosen after seeing which is friendlier.
- **Paired tests** use the same items for router and baseline throughout.
- H1/H2/H3 are judged on the medians defined above, not on cherry-picked pairs.

---

## 3. Stage 12 — the router-free predictability ceiling

This is the stage that answers "you tested a weak router" without testing
another router.

### 3.1 The estimand

Write `eta_m(x) = E[Y_m(x) | x]`, the probability that model `m` answers `x`
correctly, averaging over generation randomness. A router sees only `x`, so the
best any router can do is act on `eta`. For two models `s` (cheap) and `l`
(expensive) at operating fraction `beta`, Stage 11's theory doc will prove

```
A†(b) - S(b) = beta * ( CTE_beta(delta) - E[delta] )   where delta(x) = eta_l(x) - eta_s(x)
             <= sqrt( beta * (1 - beta) ) * sd(delta)
```

`CTE_beta` is the mean of the top-`beta` tail. The inequality is the standard
mean–variance bound on a conditional tail expectation and is tight. So
**`sd(delta)` caps the gain of every router**, and at `beta = 0.5` the cap is
`sd(delta) / 2`.

### 3.2 Why our $48 of data is the only place this can be computed

`sd(delta)` is a property of `eta`, not of `Y`. Observed outcomes give
`Var(Y_l - Y_s) >= Var(delta)`, so single-generation datasets — RouterBench,
RouteLLM, and every other public routing dataset we are aware of — yield only a
loose bound. Separating the two requires **repeated generations of the same
prompt by the same model**, which Stage 7 has: k = 3 replicates for
5,000 pool items at temperature 0.7 (45,000 calls) and 364 test items at
temperature 0 (3,276 calls). By the law of total variance,

```
Var(eta_hat_m) = Var(eta_m) + (1/k) * E[eta_m (1 - eta_m)]
```

and `(k / (k-1)) * eta_hat (1 - eta_hat)` is unbiased for `eta (1 - eta)`, so
`Var(eta_m)` and `Var(delta)` are identified. Estimates are clipped at zero and
the clipping is reported when it binds.

### 3.3 A second, tighter, data-driven ceiling

A policy allowed to see one realised replicate `Y^(1)(x)` has a strictly larger
information set than a router that sees `x` alone. So the best such
"semi-oracle" policy upper-bounds `A†`. Fixed in advance: rank by
`Y^(1)_l - Y^(1)_s`, take the top `beta` fraction, and evaluate on the mean of
replicates 2 and 3. Held-out by construction across replicates.

Both bounds are reported. The *smaller* of the two is the headline ceiling; the
choice rule is fixed here, before either is computed.

### 3.4 Consistency check against the AUC plateau

Stage 11's theory doc will prove that the Bayes-optimal AUC for predicting
`Y_m` from `x` is

```
AUC*_m = 1/2 + E| eta_m(X) - eta_m(X') | / ( 4 * a_m * (1 - a_m) ),   a_m = E[eta_m]
```

for independent `X, X'`. Plugging the replicate-based estimate of the `eta`
distribution gives a predicted ceiling on AUC, which is compared to the
0.65–0.68 plateau that Stages 5, 7 and 7c measured with four different
learners. **Pre-registered prediction**: the replicate-based `AUC*` estimate
lands within **0.05** of the measured plateau. If instead `AUC*` comes out much
higher (say 0.85), the ceiling story is wrong, our routers really were weak,
and we will say so.

### 3.4a Direction of this check

Note this check can embarrass us and we are running it anyway: it is the one
place where an independent quantity (generation variance) predicts a number we
have already measured four times (held-out AUC), so it cannot be tuned.

---

## 4. Stage 13 — the router-strength ladder

Ordered by cost, each rung run only if the previous rung leaves the objection
open.

| Rung | Router | Status |
|---|---|---|
| R-a | MiniLM + MLP, MiniLM + GBM | zero API cost; part of Stage 11 |
| R-b | Prompted LLM-as-router, zero-shot and 8-shot, reading the token logprob of the "will succeed" token for a continuous score | metered |
| R-c | **Fine-tuned LLM router** on Together AI, LoRA on an 8B-class instruct model, trained on the Stage 7 TRAIN split labels | metered |

### 4.1 Cost gate, fixed in advance

Project spend to date is **$48.09** against a Stage 7 gate of $150. Stage 13's
gate is **$25 of additional spend**, and the total project gate is raised to
**$100**, still below the original $150 envelope.

Projection, to be checked against actuals and reported either way:

| Item | Volume | Unit | Projected |
|---|---|---|---|
| R-b prompted router, zero-shot | 5,364 items × 3 tiers ≈ 16,100 calls, ~400 prompt + 1 completion tokens | $0.18/M | ~$1.20 |
| R-b prompted router, 8-shot | same call count, ~1,400 prompt tokens | $0.18/M | ~$4.10 |
| R-c LoRA fine-tune, 3 epochs | 3,500 items × ~350 tok × 3 ≈ 3.7M tokens | ~$0.48/M | ~$1.80 |
| R-c fine-tuned inference | ~5,600 calls × ~400 tok | ~$0.36/M | ~$0.80 |
| Contingency (retries, a second fine-tune configuration) | | | ~$8.00 |
| **Total projected** | | | **~$16** |

If measured spend reaches $25 the stage stops where it is and the partial result
is reported as partial. If Together AI credits are exhausted — they blocked
Stage 7 twice — the rung is recorded as **BLOCKED** with the provider error, in
the style of Stage 7's original block report. **No result will be estimated,
extrapolated or simulated in place of a blocked run.**

### 4.2 What R-c has to beat, fixed in advance

The fine-tuned router is evaluated on the **same** Stage 7 CALIBRATION split
(n = 1,500) as every earlier router, with the frozen test set left alone unless
the calibration result warrants one final one-shot evaluation. Criteria:

- **C1**: mean held-out per-tier AUC exceeds the Stage 7c best (0.6816) by
  **>= 0.05**. This is the threshold at which "you tested a weak router"
  becomes a live objection again.
- **C2**: realised matched-cost gain over the static baseline exceeds the
  Stage 11 best classical router's by >= 1.0 pp.
- If C1 and C2 both fail, the ceiling claim survives its strongest available
  attack. If either succeeds, **the ceiling claim is withdrawn or narrowed in
  the paper**, and Stage 12's bound is re-examined for error.

### 4.3 One-shot discipline

The frozen Stage 7 test set has been scored once, in Stage 7. Stage 13 works on
TRAIN/CALIBRATION only. Any final test-set evaluation is a single shot with the
router's weights, threshold and featurizer already fixed, and is reported even
if it disagrees with the calibration result.

---

## 5. Paper

A workshop-length write-up (target: an ICLR 2027 workshop) lands in `paper/`.
Structure fixed here: abstract; the decomposition and its proofs; the
in-house study; RouterBench at scale; the ceiling; limitations. The limitations
section must carry forward every item from `stage7_10/s7_LIMITATIONS.md` and
`evaluation_card.md` §8 that still applies, and must state plainly what $48 of
in-house data cannot support.

**The paper does not get to choose its conclusion.** If H2 or H3 fails, the
paper's claim becomes "complementarity is abundant and predictability is
attainable, contrary to our Stage 5/7 negatives", and the earlier stages are
reinterpreted as a weak-router artifact.

---

## 6. Deviations

Any deviation from this document will be recorded in
`stage11_13/DEVIATIONS.md` with the reason, in the style of the Stage 3 energy
budget deviation disclosed in `evaluation_card.md` §6.

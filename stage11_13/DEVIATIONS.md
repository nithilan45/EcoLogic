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

---

## D6. 28 RouterBench 5-shot items dropped for missing score cells.

**Pre-registered (§2.2).** Score handling was fixed in advance for fractional
scores. Missing cells were not anticipated, because the 0-shot file has none.

**What happened.** `routerbench_5shot.pkl` has **154 missing score cells spread
over 28 `arc-challenge` items** out of 36,508 (0.077% of items, 0.038% of cells).
An item with a missing score for some model has no defined utility vector, so it
is **dropped rather than imputed**; the count is printed by the loader and stored
as `n_items_dropped_missing_cells` in the result JSON. Imputation was rejected
because every imputation rule (model mean, item mean, zero) changes `kappa` in a
direction we would have had to argue about. The 0-shot primary analysis is
unaffected: it drops nothing.

---

## D7. Stage 13b (end-to-end fine-tuned encoders) added, not pre-registered.

**Addition.** The pre-registered ladder (§4) has three rungs and every one of
them puts a *fitted head* on a *frozen* representation, or prompts a frozen LLM.
That leaves one version of "you tested a weak router" open: perhaps the MiniLM
sentence embedding simply does not contain the information, and a representation
*trained on the task* would. So we add a rung that unfreezes the encoder and
trains all of its weights on the routing labels, with a 3-logit head and
per-tier binary cross-entropy — including the **same** `all-MiniLM-L6-v2`
backbone whose frozen output gives 0.6816, so the comparison isolates exactly the
effect of unfreezing.

**Status.** **Exploratory and not pre-registered**, labelled as such wherever
reported. Fitted on the Stage 7 TRAIN split and scored once on the Stage 7
CALIBRATION split, which is the split every earlier rung reports; epoch selection
uses a 15% inner split carved out of TRAIN and never touches CALIBRATION. It is
judged against the same pre-registered C1 threshold (+0.05 AUC over 0.6816) as
the pre-registered rungs, so adding it cannot make the criterion easier to pass.
Cost $0; it runs on CPU.

---

## D8. Rung R-c (fine-tuned LLM router) is **BLOCKED**, and reported as blocked.

**Pre-registered (§4, §4.1).** R-c: "LoRA fine-tune on Together AI ... trained on
the Stage 7 TRAIN split labels", scored on CALIBRATION against criteria C1/C2,
with the instruction that if credits are exhausted we "say so explicitly and
document the cost gate rather than silently skipping. **No result will be
estimated, extrapolated or simulated in place of a blocked run.**"

**What happened.** The fine-tune itself **succeeded**. Job `ft-6567602b-4aa3`,
LoRA on `google/gemma-3-27b-it`, 3 epochs over the 10,500 prompt/completion
examples in `s13_ft_train.jsonl`, 1,491,033 tokens, **$6.71** booked to the
ledger from Together's own reported price. It then proved **unservable**, through
four independent routes, each probed rather than assumed and each recorded in
`s13_ftblocked.json` and `s13_endpoint_probe.json` with the provider's verbatim
error:

1. **Serverless LoRA** — `/v1/models` advertises 13 `*-Lora` inference targets
   including `google/gemma-3-27b-it-lora`, but **every one** returns
   `400 Unable to access non-serverless model` on this account, and calling the
   fine-tuned name directly returns `404 model_not_available`.
2. **Dedicated endpoints v1** — `POST /v1/endpoints` returns
   `403 endpoints_v1_create_access_disabled`. Together retired endpoint creation
   on v1 platform-wide; this is not an account limit.
3. **Dedicated endpoints v2** — v2 requires a *certified config*.
   `models.configs.list` returns **0 configs** for the merged fine-tune and 0 for
   the `gemma-3-27b-it` base, and gemma-3-27b is absent from the 43 v2-supported
   architectures (only gemma-4 variants are listed). A `validate_only` deployment
   create fails for want of a config.
4. **Re-fine-tuning on a base that v2 can serve** — `Qwen/Qwen3.5-9B` is both
   fine-tunable *and* has a certified v2 config (`cr_CeQCqcGQpVCeTctadrHjy`,
   BF16, 1× H100, $3.99/replica-hour), and it is the Tier-1 model of the system
   under study, so the router would cost no more to run than the cheapest tier it
   routes to. `POST /v1/fine-tunes` returns
   `402 insufficient_balance: "Required combined balance and credit limit:
   4.00 USD"`. Pay-as-you-go serverless calls still return 200, so the block is
   the upfront reserve a fine-tuning job requires, not the key.

**Consequence.** R-c is reported as **BLOCKED** with those four errors, at a
spend of **$9.71** against the $25 gate — the gate was never the binding
constraint, the provider's balance was. The C1/C2 verdict therefore rests on the
rungs that did run: the prompted 70B router (R-b, complete) and the end-to-end
fine-tuned encoders (Stage 13b, D7, complete). We note plainly what this costs
the paper: **a reviewer who believes a fine-tuned generative LLM router would
clear C1 has not been answered by a fine-tuned generative LLM router.** They have
been answered by a fine-tuned *encoder*, a prompted 70B model, a random forest
that memorises its training set, and a learning curve — which is weaker on that
specific axis, and the paper says so rather than papering over it.

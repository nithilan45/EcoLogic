# Evaluation card — EcoLogic tiered query routing

Written in the spirit of Model Cards (Mitchell et al., FAccT 2019) and
Datasheets for Datasets (Gebru et al., CACM 2021), adapted to document an
*evaluation* rather than a model or a dataset. The intent is that anyone
quoting a number from this project can find, in one place, what that number
does and does not license them to say.

Sections 1–7 document the audit of **this** router on **our** data. Section 7b
documents the generalisation of that audit — a decomposition of any router's
gain into complementarity, predictability and estimation error, measured on
RouterBench's 401,434 released outcomes — and is the part a reader interested in
routing generally should start from. Section 8 carries every limitation forward.

---

## 1. What is being evaluated

Not a model. A **routing policy**: the decision procedure that assigns an
incoming query to one of three model tiers.

Two routers are evaluated, against four reference policies:

| | Policy | What it establishes |
|---|---|---|
| Subject | **EcoLogic keyword classifier** (`classify_prompt_local_nlp`, imported unmodified from `backend/main.py`) | the deployed system |
| Subject | **Learned router** (per-tier `P(correct \| query)` + one cost threshold), two data regimes | whether a trained replacement helps |
| Reference | Always-Tier-1, Always-Tier-2 | static assignment — the bar a router must clear to justify existing |
| Reference | Always-frontier (`gpt-4o`) | quality ceiling |
| Reference | Random tier | sanity floor |
| Reference | Oracle (cheapest tier that answered correctly, post hoc) | efficiency ceiling |
| Reference | LP-relaxed and integer MCKP frontiers | the achievable (energy, accuracy) boundary |

Tiers: Tier 1 `Qwen/Qwen3.5-9B`, Tier 2 `openai/gpt-oss-20b` (both Together AI),
Tier 3 `gpt-4o` (OpenAI).

**The tier models are substitutes.** The paper specifies Gemma 3N E4B and
Apriel 1.6 15B; both were retired from Together AI's catalogue before this
evaluation ran. Tiers 1 and 2 are energy-adjacent replacements. This evaluation
therefore audits *EcoLogic's routing logic*, not EcoLogic's exact deployed
model stack.

### Primary claim under test

The paper's own limitations section states that "no formal quality evaluation
comparing tier outputs on matched query sets has been performed." This project
performs that evaluation, then goes further and asks whether a learned router
would do better.

---

## 2. Benchmarks, and why these

| Benchmark | Items | Role | Why chosen |
|---|---|---|---|
| HumanEval | 164 | code, original test set | graded by **executing** official test cases — no judge, no partial credit |
| MBPP | 974 total, used across pools and the new test set | code, training pools + Stage 7 test set | same objective execution grading; substitutes for HumanEval where HumanEval would contaminate |
| MMLU | 100 + 100 test, 2,695 pool | factual/knowledge | 4-way multiple choice, exact letter match |
| GSM8K | 100 + 100 test, 2,295 pool | reasoning | exact numeric final-answer match |

**Selection criterion: objective auto-gradability.** Every benchmark here can
be graded by executing code or matching a string, with no model in the grading
loop. An earlier iteration of this project used 24 hand-written questions and
an LLM judge; that report is retained at `quality_benchmark_report.md` and is
**superseded**, because judge-graded numbers could not be defended at the
precision the routing comparison needs.

### These benchmarks are explicitly NOT claimed to represent EcoLogic traffic

This is the most important limitation on the card and it is not recoverable by
any amount of extra rigour elsewhere. Real routing traffic is conversational,
multi-turn, frequently trivial ("thanks", "what about tomorrow"), and mostly not
objectively gradable. HumanEval/MBPP/MMLU/GSM8K are single-turn,
self-contained, difficulty-concentrated academic tasks. Consequences:

- Accuracy figures are accuracy **on these benchmarks**, not "EcoLogic answers
  N% of user queries correctly".
- The energy-savings figures depend on the *mix*. A workload with more trivial
  queries would make any router look better; one with more hard ones, worse.
- The oracle gap is a property of this item distribution.

No claim in any report in this project is conditioned on benchmark
representativeness, and none should be.

---

## 3. What the routing decision is sensitive to

A finding worth surfacing on its own, because it undermines the deployed
router's premise. The classifier was run twice on every item: once on the
**raw query** (what a user types) and once on the **wrapped prompt** actually
sent to the model (raw query plus benchmark-standard instructions such as
"Answer with only the letter").

| Routing input | Tier 1 / 2 / 3 | Accuracy | Energy |
|---|---|---|---|
| Raw query (used for all headline numbers) | 274 / 87 / 3 | 86.8% | 689 J |
| Wrapped prompt | 137 / 53 / 174 | 89.6% | 3,419 J |

**Agreement between the two: 53.0%.** Adding a fixed, content-free instruction
wrapper reroutes nearly half the corpus and multiplies energy use by 5. A router
whose output is this unstable under a formatting change is keying on surface
features rather than on task difficulty. Every headline number uses the raw
query, which is the favourable choice for the deployed system.

---

## 4. Measured vs modelled — read before quoting any joule figure

| Quantity | Status | Detail |
|---|---|---|
| Token counts | **measured** | provider `usage` field on every call; never estimated from word counts |
| Correctness | **measured** | code executed against official tests; MMLU/GSM8K exact match |
| Dollar cost | **measured** | provider-reported per-call cost, summed |
| Latency | **measured** | wall-clock per call |
| **Energy (joules)** | **MODELLED** | `total_tokens / 1000 × rate_tier`, rates 0.5 / 1.5 / 60.0 J per 1k tokens taken from EcoLogic's own source |
| Tier identity | **substituted** | Tiers 1–2 are replacements for retired models (§1) |

**No joule was measured anywhere in this project.** There is no power meter, no
GPU telemetry, no provider energy disclosure. The joule figures are a linear
transform of measured token counts under EcoLogic's own assumed rates, and they
inherit every error in those rates. Because the rates are per-token constants,
any conclusion of the form "policy A uses less energy than policy B" is, under
this model, a statement about token counts weighted by three constants — not an
independent physical measurement.

Two defences are applied and neither fully repairs this:

- **Sensitivity sweep**: all 27 combinations of ×0.2, ×1, ×5 per tier rate,
  reported for every energy claim, with any sign flips called out explicitly.
- **A second rate set** (1.1 / 2.0 / 60.0) reflecting the larger substitute
  models is reported alongside the paper's rates.

### And the accounting formula itself has a bias (Stages 8–9)

Independently of rate uncertainty, the *standard way* the field converts routing
decisions into savings is biased. Multiplying a confusion matrix or routing
distribution by per-model **average** cost is exact only when per-item cost is
constant within a model. Stage 8 derives the exact correction —
`R_true = R_naive + Σ_{i≠j} Cov(1{t*=i, t̂=j}, e_j(x) − e_i(x))` — and validates
it to 2×10⁻¹⁶ on our data, where the correction is **1.9× the size of the regret
being reported** and flips its sign. Stage 9 reproduces the sign flip on
RouteLLM's own released GSM8K data, where naive accounting overstates savings by
up to 6.3%. The bias always favours the router, and it grows with within-model
cost dispersion — so it is getting worse as the field routes among reasoning
models with variable-length thinking budgets.

---

## 5. Statistical treatment

- **95% Wilson intervals** on every accuracy figure.
- **McNemar exact tests** for all paired comparisons, computed on identical
  item sets and identical generations. All pre-registered verdicts rest on
  paired tests, never on interval overlap.
- **Temperature 0 is not deterministic** on either provider. This was measured,
  not assumed: on repeated temperature-0 calls, Tier 1 returned byte-identical
  text on only 9/21 items and token counts moved by tens of percent.
- **Generation-variance decomposition** (Stage 10a) was designed to quantify
  this properly — splitting accuracy variance into between-item and
  within-item (regeneration) components and reporting "sampling-only" versus
  "sampling+generation" interval widths. It is **complete**
  (`s7_generation_variance.md`), and three results follow. First, the Wilson
  intervals used throughout this project are **not too narrow**: they track the
  sampling+generation width closely, because a single-run evaluation's draw
  already contains the generation noise. Second, the share of interval width
  that is pure regeneration noise scales with output length — **4.9%** for terse
  `gpt-4o` but **47.0%** for the long-reasoning Tier 1, which changes its graded
  verdict on **19.8% of items (72/364)** between identical temperature-0 calls
  with output length moving ~1,936 tokens on average. A Wilson interval is
  routinely read as if reruns would land inside it and only a different item
  sample would move it; for that tier the reading is wrong. Third, Stage 7's
  1.10 pp verdict margin falls **inside** the router's own 2.70 pp
  sampling+generation half-width, so the decomposition and the pre-registered
  McNemar test agree that the margin is not separable from noise.

---

## 6. Pre-registration and one-shot discipline

| Document | Fixed in advance |
|---|---|
| `router_v2/PREREGISTRATION.md` | S1/S2 success criteria, one-shot rule, threshold rule with X = 10% filled in, oracle labelling convention, $40 cost gate |
| `stage7_10/prereg_stage7.md` | 5,000-item target, k=3 majority labels at temp 0.7, unchanged S1/S2, a third "partial support, inconclusive" outcome with its numeric trigger, the MBPP scaling ceiling, $150 cost gate, throughput contingency |

Both were committed **before** the data they govern existed, each in its own
commit ahead of the corresponding artifacts. Model and hyperparameter selection
happened by cross-validation within TRAIN only. Each frozen test set was
evaluated once, with the featurizer, weights and threshold already fixed.

One deviation, disclosed: in Stage 3 the pre-registered energy budget was
infeasible under the routing rule's original tier ordering, because Tier 1 —
despite a 3× lower per-token rate — emits ~6× more tokens and is therefore more
expensive per item than Tier 2. The routing rule was changed to cost-ordered
(order derived from TRAIN means only); the *pre-registered threshold-selection
rule and budget were not changed*, and the original ordering is retained as a
reported ablation. Routing-rule structure had not been pre-registered; the
threshold rule had.

---

## 7. Headline results

Original frozen test set, 364 items:

| Policy | Accuracy | Energy (J) |
|---|---|---|
| EcoLogic keyword router | 86.8% | 689.0 |
| Always Tier 1 | 87.6% | 677.1 |
| **Always Tier 2** | **92.3%** | **393.9** |
| Random tier | 90.7% | 2,556.3 |
| Always-frontier | 91.5% | 6,065.9 |
| Oracle | 96.4% | 385.5 |

The deployed router is **beaten on both axes simultaneously** by the static
policy "send everything to Tier 2" — lower accuracy (86.8% vs 92.3%) *and*
higher energy (689 J vs 394 J). It also fails to beat Always-Tier-1
(McNemar p = 0.25), so its keyword logic adds nothing over ignoring the query
entirely. Against the frontier it gives up 4.67 pp of accuracy
(McNemar p = 0.0095) to save 88.6% of energy, and it spends 78.7% more energy
than the oracle.

The learned router did not rescue this. Stage 5 (1,200 training items,
single-sample labels): neither S1 nor S2 met, with a 5.83 pp calibration gap to
the LP-relaxed frontier and a ~0 pp discreteness gap. Stage 7 retested whether
that gap is a data problem, with 4.2× the training data and k=3 majority-voted
labels, and the pre-registered verdict is **partial support, inconclusive**: on
a fresh 364-item frozen test set the scaled router reached 93.7% against
Always-Tier-2's 92.6%, moving from 0.27 pp behind (Stage 5) to 1.10 pp ahead —
but not significantly (McNemar p = 0.2188) and at 28.2% *more* energy, so
neither S1 nor S2 was met. Mechanistically the improvement is thin: held-out
per-tier head AUCs stayed flat (0.656 → 0.665 and 0.701 → 0.702) even though
cross-validation AUC rose (0.675 → 0.716), and the calibration gap closed only
0.83 pp to 5.00 pp with the discreteness gap still ~0. The remaining gap
therefore still looks structural — feature representation, or how predictable
per-item tier success is from a prompt at all. The design does not separate
those two explanations and this card does not claim it does.

---

## 7b. The explanation, and the external check (Stages 11–13)

Sections 1–7 audit *this* router. Stages 11–13 ask the more general question the
audit raises: when a routing gain is small, **which of three things is to
blame** — the model set, the task, or the router? For a workload and a model
set, let `S(b)` be the best expected utility at expected cost `b` achievable by
any **query-independent** policy, `A†(b)` by any **router** (any function of the
query), and `A*(b)` by an **oracle** that sees realised outcomes. Then
`S(b) ≤ A†(b) ≤ A*(b)`, and writing

```
complementarity  kappa(b) = A*(b) - S(b)          blames the model set
predictability   rho(b)   = (A†(b) - S(b))/kappa  blames the task
estimation error eps                              blames the engineer
```

any fitted router realises `rho·kappa − eps`. Five propositions with proofs are
in `../stage11_13/theory.md`; **14/14** are checked numerically against LP or
Monte-Carlo ground truth (`s11_validate.json`).

### Measured on RouterBench — 100× more data than this project bought

`withmartian/routerbench` (SHA-256 pinned in the result JSON): 36,494 prompts ×
11 models = 401,434 already-released outcomes with per-item graded score and
per-item dollar cost, across 8 benchmark families. All 55 model pairs × 8
families. **Cost of this analysis: $0.** Medians over 440 pair-family cells at
`beta = 0.5`, 95% bootstrap intervals:

| Quantity | 0-shot (primary) | 5-shot (pre-registered replication) |
|---|---|---|
| complementarity `kappa` | **12.11 pp** [11.15, 12.92] | 11.44 pp [10.66, 12.27] |
| best router's realised gain | **+0.585 pp** [0.442, 0.767] | +0.640 pp [0.519, 0.790] |
| best router's realised `rho` | **4.6%** [3.6, 5.9] | 5.5% [4.4, 6.8] |
| pairs with positive mean gain | 49/55 | 50/55 |
| surviving BH / Holm at 0.05 | 26 / 13 | 22 / 11 |

**Complementarity is abundant and routers capture about a twentieth of it.**
This is *not* a null result — routing works, and 13 pairs survive
Holm–Bonferroni. But reading gains against an always-frontier baseline instead
of a cost-matched query-independent one overstates the achievement by roughly
20×.

### We pre-registered an explanation and the data refuted it

The pre-registered hypothesis H3 was that **predictability** is the missing
factor — that per-item success is close to a coin flip, so most of `kappa` is
provably unreachable. Using `k = 3` replicate generations of 5,000 pool items
(the one thing no public routing dataset has), **it is not**: 77–98% of per-item
outcome variance is stable between-item signal rather than regeneration noise,
implying a Bayes-optimal AUC of 0.95–0.99 against the 0.65–0.72 that nine
router families actually reach. The pre-registered §3.4 consistency check,
written because it could embarrass us, came out **0.99 against 0.68 and is
recorded as INCONSISTENT**.

**So `rho ≈ 1` and the whole gap is `eps`.** The claim this project now makes is
not "per-item success is unpredictable" but "**per-item success is a highly
reliable property of the item that routers cannot read off the prompt text**."
Full accounting in `../stage11_13/DEVIATIONS.md` D1; the Cauchy–Schwarz ceiling
of Proposition 3 is also recorded as **vacuous at these effect sizes** (it
evaluates to 2.82× the measured `kappa`), which was the falsification rule
written down in advance.

### Nine router families, and the "weak router" objection

Same 1,500-item held-out split throughout. `ΔAUC` is a 2,000-resample paired
bootstrap over items against the frozen MiniLM + logistic reference refitted on
the same TRAIN split, Holm-corrected across the four new rungs.

| Router | Representation | mean AUC | ΔAUC [95%] | gain (pp) |
|---|---|---|---|---|
| k-NN, boosting, random forest, logistic | MiniLM (frozen) | 0.652–0.690 | — | up to +0.556 |
| Prompted LLM, zero-shot | Llama-3.3-70B | 0.6416 | −0.048 [−0.097, +0.004] | −0.444 |
| Prompted LLM, 4-shot | Llama-3.3-70B | **0.7105** | +0.021 [−0.019, +0.062] | **−1.311** |
| Fine-tuned end-to-end | MiniLM-L6 (**unfrozen**) | 0.6945 | +0.005 [−0.014, +0.025] | +0.533 |
| Fine-tuned end-to-end | MiniLM-L12 (**unfrozen**) | 0.6912 | +0.002 [−0.018, +0.023] | +0.533 |
| LoRA fine-tune, 3 epochs | Gemma-3-27B-it | **BLOCKED** | — | — |

The pre-registered criterion C1 — a stronger router must beat Stage 7c's 0.6816
by ≥ 0.05 for "you tested a weak router" to become a live objection again — is
**not met** (largest point estimate +0.029), C2 (matched-cost gain ≥ 1.0 pp over
the reference) is **not met**, and **no rung's ΔAUC is distinguishable from
zero** (smallest Holm-adjusted p = 0.26). A learning curve on RouterBench from
250 to 29,000 training items fits a power law with **asymptote 0.741 AUC**;
realised gain *does* improve with data (+1.47 pp at 8k → +2.10 pp at 29k), so
this is not "data doesn't help", but the extrapolated ceiling stays far below
what capturing 12 pp would need.

Three observations carry more than the verdict, and belong on this card because
they qualify how any router number here should be read:

- **Unfreezing the encoder is the honest answer to "you used frozen features",**
  and it buys +0.005 held out while its *inner-validation* AUC reaches 0.751.
  The capacity to fit routing labels exists and does not transfer.
- **The highest-AUC rung has the worst matched-cost gain.** The prompted 4-shot
  router ranks best by AUC and is *worse than ignoring the query* (−1.31 pp) at
  the same budget on the same items. AUC is per-model and invariant to monotone
  rescaling; a matched-cost policy ranks by the *difference* of two models'
  predicted utilities, so a router can order items correctly within each model
  and get every cross-model comparison wrong. The routing literature reports
  per-model AUC almost universally. **Here it disagrees in sign with the
  deployable quantity** — which is a caution against every AUC on this card,
  including ours.
- **The random forest is the diagnostic:** train AUC 0.9999 (perfect
  memorisation), held-out AUC *below* logistic regression. Ample capacity, no
  generalisation — the signature of a target that is not a smooth function of
  the input representation, not of an inadequate model class.

### What was blocked

The pre-registered top rung was a fine-tuned **generative** LLM router. It
trained without incident (`ft-6567602b-4aa3`, LoRA on `google/gemma-3-27b-it`,
1,491,033 tokens, **$6.71** booked from Together's own reported price) and then
proved **unservable** through four independently probed routes: no serverless
LoRA on the account; dedicated-endpoints v1 creation retired platform-wide
(`403 endpoints_v1_create_access_disabled`); no certified v2 serving config for
`gemma-3-27b-it`; and re-training on a v2-servable base refused with
`402 insufficient_balance`. Verbatim provider errors are in
`../stage11_13/s13_ftblocked.json` and `DEVIATIONS.md` D8. **Nothing was
estimated, extrapolated or simulated in its place**, per the pre-registration.
The cost to the argument is stated rather than hidden: *a reader who believes a
fine-tuned generative LLM router would clear +0.05 AUC has not been answered by
one.* They have been answered by a fine-tuned encoder, a prompted 70B model, a
memorising random forest, and a learning curve.

Additional spend for Stages 11–13: **$9.71** against a pre-registered $25 gate
enforced in code. The gate never bound; the provider's account balance did.
RouterBench (both releases), the decomposition, the learning curves, the
end-to-end encoders and all numerical validation cost **$0**.

---

## 8. Known limitations, carried forward from every stage

From `results_report.md`:
1. Energy is modelled, never measured (§4).
2. Benchmarks are not EcoLogic traffic (§2).
3. Single generation per item for the headline numbers; quantified in §5.
4. Tier 1/2 are substituted models (§1).
5. Tier 1 truncation at a 4,096-token cap was a confound; fixed by re-running
   everything at 16,384 and archiving the old run.

From `router_v2/LIMITATIONS.md`:
6. **Train/test distribution shift**: MBPP ≠ HumanEval. The router's largest
   deviation from static Tier 2 was on code, which is exactly where the shift is
   largest — so any explanation of *why* it failed is confounded.
7. **Label noise**: oracle labels came from single temperature-0 generations, so
   the router partly fit generation noise. 58/1,200 pool items (4.8%) had no
   correct answer from any tier; their labels are a stated convention, not
   ground truth.
8. **Small training set**: 840 TRAIN items, badly imbalanced (the informative
   class is failures, of which there are few). CALIBRATION head AUCs were 0.66
   and 0.70 — real signal, far too weak to beat a static policy already right
   ~92% of the time.
9. **The threshold rule was one reasonable choice** among several; it was fixed
   in advance, but a different X would give a different operating point.

From `stage7_10/prereg_stage7.md` and `s7_LIMITATIONS.md`:
10. **MBPP is exhausted.** Only 410 unused code items remained for the Stage 7
    pool against Stage 1's 400 — a 1.03× scale-up versus 5.7× for MMLU and
    GSM8K. Stage 7's null result is therefore confounded on the code subset by
    the inability to add code training data. Flagged before the run, not after.
11. **Stage 7 pool GSM8K items come from the train split** while test items come
    from the test split. This guarantees disjointness but introduces a (small)
    distribution difference.
12. **Stage 9's external check covers one benchmark, one router, one model
    pair**, with token counts reconstructed from released response text rather
    than published directly.

From `stage11_13/prereg_stage11_13.md` and `stage11_13/DEVIATIONS.md`:
13. **Null results are not proofs of absence.** Nine router families failing to
    clear +0.05 AUC bounds what *these* representations and *this* volume of
    supervision achieve. It does not prove no router can; the learning curve is
    an extrapolation, not a theorem.
14. **The top pre-registered rung was never served** (§7b, D8). The strongest
    remaining evidence against "you tested a weak router" is a fine-tuned
    *encoder*, not a fine-tuned generative LLM.
15. **Proposition 3's ceiling is vacuous at these effect sizes.** It is a
    correct inequality, tight on a two-point distribution, and 2.82× too loose
    on real data. It is retained because the falsification rule was
    pre-registered, not because it constrains anything here.
16. **RouterBench is not this workload either.** Its 8 families share the
    single-turn, auto-gradable character that limitation 2 already flags; it
    buys scale and model diversity, not traffic realism. Its outcomes are
    single generations, which is exactly why the in-house `k = 3` replicate data
    is not redundant with it (Proposition 4).
17. **Two RouterBench analyses are exploratory, not pre-registered**: the
    learning curves (D3) and the end-to-end encoder rungs (D7). Both are
    labelled as such wherever reported, and the encoders are still judged
    against the pre-registered +0.05 threshold, so adding them cannot make the
    criterion easier to pass.

---

## 9. Intended and unintended use of these results

**Appropriate**: as evidence that this keyword router underperforms static
assignment on objectively-gradable single-turn tasks; as a worked template for
auditing a routing system; as a demonstration that confusion-matrix cost
accounting is biased and by how much; as evidence that on RouterBench's 55 model
pairs, routing gains against a cost-matched query-independent baseline are real
but roughly 20× smaller than the available oracle headroom, and that the binding
constraint there is generalisation rather than outcome noise or model
redundancy.

**Not appropriate**: as a measurement of EcoLogic's real-world energy savings;
as a claim about the retired Gemma/Apriel tiers; as evidence that learned
routing cannot work in general (Stage 5 tests *one* family of zero-API-cost
routers on *this* workload, and Stage 7 shows only that scaling *this* router's
training data moves it from slightly behind to slightly ahead of a static
baseline, inconclusively); as evidence that **no** router can close the
RouterBench gap (§8 limitation 13 — nine families is a bound on these
representations, not a theorem); as a claim that a fine-tuned generative LLM
router would also fail (it was never served, §7b); as physical energy
measurement of any kind.

**Read AUC on this card with the §7b caution in hand.** The highest-AUC router
built anywhere in this project has the *worst* matched-cost gain. Per-model AUC
is the field's standard router metric and it is not a proxy for router value.

---

## 10. Artifacts

| Path | Contents |
|---|---|
| `raw_results/` | every prompt, response, token count, grade, routing decision for the original test set |
| `results_report.md` | Stages 1–4 write-up: four-policy comparison, oracle gap, sensitivity band, "what failed" |
| `router_v2/` | learned-router addendum: pre-registration, pools, ablation, threshold sweep, MCKP frontier, one-shot results, limitations |
| `stage7_10/` | Stage 7 retest, Stage 8 derivation, Stage 9 external check, Stage 10 documentation |
| `stage11_13/` | the decomposition (`theory.md`, proofs), its numerical validation, RouterBench at scale (both releases), replicate-based reliability, learning curves, the nine-rung router ladder, `DEVIATIONS.md` (including the refuted hypothesis and the blocked rung) |
| `external_data/` | RouterBench 0-shot and 5-shot releases as downloaded, SHA-256 recorded in the result JSONs |
| `paper/` | workshop paper draft (`main.tex`, `appendix.tex`) and a per-number provenance table in `paper/README.md` |
| `quality_benchmark_report.md` | **superseded** first-pass report, retained for provenance |

---

## 11. Completion status

| Stage | Status |
|---|---|
| 1–6 (audit, learned router, MCKP, regret) | complete |
| 7 (scaled retest of the calibration-gap hypothesis) | complete, all 48,276 calls generated. Verdict **partial support, inconclusive** — the pre-registered third outcome (`stage7_results.md`) |
| 8 (regret correction derivation) | complete, reconciles to 2×10⁻¹⁶ |
| 9 (external check on RouteLLM) | complete, within the scope limits in §4 |
| 10(a) (generation-variance decomposition) | complete, all three tiers and all policies |
| 10(b)(c)(d) (this card, framing, manifest) | complete |
| 11 (decomposition: formalise, prove, validate, measure on RouterBench) | complete; 14/14 propositions validated; H1 and H2 **supported** on both the 0-shot and the 5-shot release |
| 12 (router-free ceiling from k=3 replicates) | complete, **and it refuted our own pre-registered H3**; reported as such |
| 12b (learning curves on RouterBench) | complete, **exploratory, not pre-registered** (D3) |
| 13 (router-strength ladder) | **partly blocked.** Prompted 70B router complete; the fine-tuned **generative** LLM rung is **BLOCKED** through four probed routes (D8) |
| 13b (end-to-end fine-tuned encoders, in-house) | complete, **exploratory, not pre-registered** (D7) |
| 13c (the same, on RouterBench) | **running**; exploratory, and no number from it is quoted until it lands |
| paper | draft complete, `paper/main.tex` |

Total measured API spend across all stages: **$57.80** — Stages 1–4 $1.9497 +
`router_v2` $3.5143 + `stage7_10` $42.6240 + `stage11_13` $9.7070 =
**$57.7950**, against the original $150 envelope. Components and their sources
are in `reproducibility_manifest.md` §10 and §12.8.

What remains **not established** is stated in §8 and in `s7_LIMITATIONS.md`
rather than left implicit: chiefly that energy is modelled and never measured,
that the workload is four auto-gradable benchmarks rather than EcoLogic traffic,
that Stage 7's "partial support" label rests on a 1.10 pp margin smaller than
the evaluation's own 2.70 pp generation noise — so it should not be read as
evidence the learned router is better than static assignment — and that the
Stage 13 verdict rests on nine router families of which the strongest
pre-registered one was never served.

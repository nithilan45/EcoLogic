# Evaluation card — EcoLogic tiered query routing

Written in the spirit of Model Cards (Mitchell et al., FAccT 2019) and
Datasheets for Datasets (Gebru et al., CACM 2021), adapted to document an
*evaluation* rather than a model or a dataset. The intent is that anyone
quoting a number from this project can find, in one place, what that number
does and does not license them to say.

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
- **Generation-variance decomposition** (Stage 10a): the frozen Stage 7 test set
  was regenerated k=3 times per item per tier at temperature 0, and accuracy
  variance is split into between-item and within-item (regeneration) components,
  with "sampling-only" and "sampling+generation" interval widths reported
  separately in `s7_generation_variance.md`.
- **Temperature 0 is not deterministic** on either provider. This was measured,
  not assumed.

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
single-sample labels): neither S1 nor S2 met. Stage 7 retests the hypothesis
that the shortfall was a data problem, with ~4.2× the training data and
majority-voted labels; its verdict is in `stage7_results.md`.

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
    pool against Stage 1's 400 — a 1.02× scale-up versus 5.74× for MMLU and
    GSM8K. A null result on code in Stage 7 is confounded with the inability to
    add code training data. Flagged before the run, not after.
11. **Stage 7 pool GSM8K items come from the train split** while test items come
    from the test split. This guarantees disjointness but introduces a (small)
    distribution difference.
12. **Stage 9's external check covers one benchmark, one router, one model
    pair**, with token counts reconstructed from released response text rather
    than published directly.

---

## 9. Intended and unintended use of these results

**Appropriate**: as evidence that this keyword router underperforms static
assignment on objectively-gradable single-turn tasks; as a worked template for
auditing a routing system; as a demonstration that confusion-matrix cost
accounting is biased and by how much.

**Not appropriate**: as a measurement of EcoLogic's real-world energy savings;
as a claim about the retired Gemma/Apriel tiers; as evidence that learned
routing cannot work in general (Stages 5 and 7 test *one* family of
zero-API-cost routers on *this* workload); as physical energy measurement of
any kind.

---

## 10. Artifacts

| Path | Contents |
|---|---|
| `raw_results/` | every prompt, response, token count, grade, routing decision for the original test set |
| `results_report.md` | Stages 1–4 write-up: four-policy comparison, oracle gap, sensitivity band, "what failed" |
| `router_v2/` | learned-router addendum: pre-registration, pools, ablation, threshold sweep, MCKP frontier, one-shot results, limitations |
| `stage7_10/` | Stage 7 retest, Stage 8 derivation, Stage 9 external check, Stage 10 hardening and documentation |
| `quality_benchmark_report.md` | **superseded** first-pass report, retained for provenance |

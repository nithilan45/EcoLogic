# EcoLogic Routing Evaluation — Publication-Grade Benchmark

**Date:** 2026-09-05
**Item set:** 364 prompts (HumanEval 164, MMLU 100, GSM8K 100)
**Calls:** 1,092 graded generation calls (364 items x 3 tiers), plus 63 repeat
calls for a determinism check and 27 pilot calls for cost estimation.
**Failed calls in the graded run:** 0
**Total measured API cost (all runs, from provider usage fields):** **$1.9497**
**Grading:** 100% objective auto-grading. No LLM judge is used anywhere in this
report (Stage 5 of the addendum is therefore not applicable — see
[Judge hygiene](#judge-hygiene)).

All raw prompts, responses, token counts, routing decisions and grades are in
[`raw_results/`](raw_results/). Every number below is reproducible from
`raw_results/graded.jsonl` + `raw_results/routing.json` via
`python3 benchmark/analyze.py`.

---

## 1. Headline result

Two numbers were requested as "the paper". Here they are, plus a third that
turned out to matter more than either.

| Quantity | Value |
|---|---|
| **Quality gap** — accuracy EcoLogic gives up vs always-frontier | **-4.7 pp** (86.8% vs 91.5%), McNemar exact **p = 0.0095** |
| **Oracle gap** — energy EcoLogic leaves on the table vs oracle routing | **+303.4 J, i.e. 78.7% more energy than necessary** (689.0 J vs 385.5 J) |
| **The finding that undercuts both** | **A static "always Tier 2" policy beats EcoLogic on *both* axes at once: 92.3% accuracy (+5.5 pp) for 393.9 J (-42.8% energy).** |

EcoLogic's routing is Pareto-dominated by a policy with no classifier in it at
all. The router is also statistically indistinguishable from "always use
Tier 1": 86.8% vs 87.6%, McNemar p = 0.25.

The mechanism is precise and worth stating plainly. The classifier escalated
90 of 364 items away from Tier 1. Of those 90 escalations:

- **0** rescued an item that Tier 1 got wrong.
- **3** broke an item that Tier 1 got right (`HumanEval/127`, `gsm8k/1251`,
  `gsm8k/353`).

So on this item set every escalation the router made was either neutral or
harmful, while costing extra energy. The classifier's tier assignment carries
no usable signal about which prompts actually need a larger model.

---

## 2. Per-benchmark, per-tier accuracy (95% Wilson intervals)

| Benchmark | n | Tier 1 (Qwen3.5-9B) | Tier 2 (gpt-oss-20b) | Tier 3 (gpt-4o) |
|---|---|---|---|---|
| HumanEval (pass@1, executed) | 164 | 86.0% [79.8%, 90.5%] (141/164) | 96.3% [92.2%, 98.3%] (158/164) | 91.5% [86.2%, 94.8%] (150/164) |
| MMLU (multiple choice) | 100 | 83.0% [74.5%, 89.1%] (83/100) | 84.0% [75.6%, 89.9%] (84/100) | 88.0% [80.2%, 93.0%] (88/100) |
| GSM8K (final-answer match) | 100 | 95.0% [88.8%, 97.8%] (95/100) | 94.0% [87.5%, 97.2%] (94/100) | 95.0% [88.8%, 97.8%] (95/100) |
| **ALL** | **364** | **87.6% [83.9%, 90.6%]** (319/364) | **92.3% [89.1%, 94.6%]** (336/364) | **91.5% [88.2%, 93.9%]** (333/364) |

Two things stand out.

**The tier ordering is not monotonic in size or cost.** Tier 2 (gpt-oss-20b)
outscores Tier 3 (gpt-4o) on HumanEval by 4.9 pp and on the full set by
0.8 pp. The paper's premise that Tier 3 is the quality ceiling does not hold
against these benchmarks — the 20B open model is at least as good overall and
costs roughly 1/50th as much per token.

**The tiers are much closer than the routing premise assumes.** The spread
across all 364 items is 87.6% to 92.3%, under 5 pp. On GSM8K the three tiers
are within 1 pp of each other (95/94/95). Routing can only buy quality where
tiers actually differ, and on two of three benchmarks they barely do.

### MMLU by subject

| Subject | n | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|---|
| college_biology | 10 | 10/10 | 10/10 | 9/10 |
| college_chemistry | 10 | 8/10 | 8/10 | 8/10 |
| global_facts | 10 | 4/10 | 4/10 | 4/10 |
| high_school_computer_science | 10 | 10/10 | 10/10 | 10/10 |
| high_school_us_history | 10 | 9/10 | 9/10 | 9/10 |
| human_aging | 10 | 7/10 | 8/10 | 8/10 |
| machine_learning | 10 | 9/10 | 9/10 | 10/10 |
| philosophy | 10 | 8/10 | 8/10 | 10/10 |
| prehistory | 10 | 9/10 | 9/10 | 10/10 |
| world_religions | 10 | 9/10 | 9/10 | 10/10 |

All three tiers score 4/10 on `global_facts`, which is at chance-ish level for
4-way multiple choice and is a known-hard MMLU subject; it is not a tier
effect.

### McNemar exact tests, paired on identical items

| Benchmark | Pair | A>B | B>A | p-value | Significant at 0.05 |
|---|---|---|---|---|---|
| humaneval | T1 vs T2 | 3 | 20 | 0.00049 | yes |
| humaneval | T1 vs T3 | 9 | 18 | 0.122 | no |
| humaneval | T2 vs T3 | 13 | 5 | 0.096 | no |
| mmlu | T1 vs T2 | 2 | 3 | 1.0 | no |
| mmlu | T1 vs T3 | 1 | 6 | 0.125 | no |
| mmlu | T2 vs T3 | 2 | 6 | 0.289 | no |
| gsm8k | T1 vs T2 | 3 | 2 | 1.0 | no |
| gsm8k | T1 vs T3 | 2 | 2 | 1.0 | no |
| gsm8k | T2 vs T3 | 1 | 2 | 1.0 | no |
| **ALL** | T1 vs T2 | 8 | 25 | 0.0046 | yes |
| **ALL** | T1 vs T3 | 12 | 26 | 0.034 | yes |
| **ALL** | T2 vs T3 | 16 | 13 | 0.711 | no |

Only three of twelve comparisons reach significance, and all three are
Tier 1 losing. **Tier 2 vs Tier 3 is not significantly different on any
benchmark or overall** (p = 0.71 on the full set). There is no measurable
quality justification for escalating from Tier 2 to Tier 3 on this workload.

---

## 3. Four-policy comparison (router in the loop)

The real production classifier (`classify_prompt_local_nlp`, imported directly
from `backend/main.py` — not a reimplementation) was run over all 364 prompts,
and each item was scored using the response from the tier it selected.

| Policy | Accuracy [95% CI] | Total tokens | Energy (J) | Energy vs frontier | Tier mix (T1/T2/T3) |
|---|---|---|---|---|---|
| (a) EcoLogic routing | 86.8% [82.9%, 89.9%] | 1,161,679 | 689.0 | 0.114x | 274/87/3 |
| (b) Always-frontier (gpt-4o) | 91.5% [88.2%, 93.9%] | 101,098 | 6,065.9 | 1.000x | 0/0/364 |
| (c) Random tier | 90.7% [87.2%, 93.2%] | 579,165 | 2,556.3 | 0.421x | 125/107/132 |
| (d) Oracle routing | 96.4% [94.0%, 97.9%] | 295,402 | 385.5 | 0.064x | 160/194/10 |
| Always Tier 1 (added baseline) | 87.6% [83.9%, 90.6%] | 1,354,186 | 677.1 | 0.112x | 364/0/0 |
| Always Tier 2 (added baseline) | 92.3% [89.1%, 94.6%] | 262,573 | 393.9 | 0.065x | 0/364/0 |

**EcoLogic routing loses to the random-assignment sanity floor.** Random tier
assignment scores 90.7% against EcoLogic's 86.8%. Random does better simply
because it sends a third of traffic to each tier, and the two tiers EcoLogic
avoids are the more accurate ones. A router that underperforms random
assignment on quality is not selecting well; it is just selecting cheap.

I added the two static single-tier baselines because they are the policies a
router has to beat to justify its existence, and they were not in the original
comparison list. Always-Tier-2 dominates EcoLogic on both axes simultaneously.

Energy savings vs always-frontier, for the record: EcoLogic 88.6%. That
headline number is real under the paper's assumed rates, but Section 5 shows
it rests entirely on those rates, and Always-Tier-2 achieves 93.5% savings at
higher accuracy.

### Oracle gap detail

Oracle routing picks, per item, the lowest-energy tier that actually answered
correctly (post-hoc). Where no tier was correct, it is charged the
lowest-energy response and scored wrong.

- Oracle reaches **96.4%** accuracy for **385.5 J**.
- EcoLogic reaches **86.8%** for **689.0 J**.
- EcoLogic therefore spends **78.7% more energy than necessary** while
  scoring **9.6 pp lower** (McNemar p = 5.8e-11).
- On **35 of 364 items (9.6%)** EcoLogic's chosen tier was wrong while some
  other tier answered correctly. Those are recoverable losses a better router
  could capture.

Note the oracle is *more accurate than the frontier model* (96.4% vs 91.5%),
because per-item the tiers fail on different items. That is the real headroom
argument for routing: an ideal router would beat gpt-4o while using 6% of its
energy. EcoLogic's classifier captures none of that headroom.

---

## 4. Router behaviour and prompt sensitivity

| Routing input | Tier 1 | Tier 2 | Tier 3 | Accuracy | Energy |
|---|---|---|---|---|---|
| Bare user query (primary) | 274 | 87 | 3 | 86.8% | 689.0 J |
| Wrapped prompt as sent to model | 137 | 53 | 174 | 89.6% | 3,419.1 J |

The classifier is highly sensitive to prompt surface form. The two inputs
differ only by the benchmark instruction wrapper ("Answer this multiple-choice
question...", "Complete the following Python function..."), and they **agree on
only 53.0% of items**. With the wrapper, all 164 HumanEval items escalate to
Tier 3; without it, 134 of them route to Tier 1.

This is a routing-stability problem independent of accuracy: identical
underlying tasks get assigned different tiers, and therefore different energy
budgets, based on boilerplate phrasing. I report the bare-query routing as
primary because it is what a user actually types, which is what the deployed
classifier receives.

---

## 5. Energy accounting with uncertainty

Token counts are the real `usage` values returned by the providers, never
word-count estimates. Energy is `total_tokens / 1000 * rate_tier`, using the
rates in `backend/main.py`: **Tier 1 = 0.5, Tier 2 = 1.5, Tier 3 = 60 J per
1k tokens**.

| Policy | Energy, paper rates (0.5/1.5/60) | Energy, substitute-model rates (1.1/2.0/60) |
|---|---|---|
| (a) EcoLogic routing | 689.0 J | 1,379.5 J |
| (b) Always-frontier | 6,065.9 J | 6,065.9 J |
| (c) Random tier | 2,556.3 J | 2,875.5 J |
| (d) Oracle routing | 385.5 J | 549.5 J |

The second column re-rates Tiers 1 and 2 upward because the models actually
tested are larger than the retired ones the paper's rates describe (see
[Substitutions](#substituted-models)).

### The token-inflation effect

The single most important thing in this section: **EcoLogic routing consumes
11.5x more tokens than always-frontier** (1,161,679 vs 101,098). Tier 1 and
Tier 2 are reasoning models that emit long `reasoning_content` traces —
Tier 1 averages 3,880 completion tokens on HumanEval where gpt-4o averages 127.

So the 88.6% energy saving is not a saving on work done. It is entirely the
product of a 120x assumed per-token rate advantage (60 / 0.5) partially eaten
by an 11.5x token penalty. The net margin is only about **10.4x**, and that is
what the sensitivity analysis probes.

### Sensitivity: each tier's rate independently perturbed x0.2, x1, x5

27 rate combinations were evaluated.

- EcoLogic's energy savings vs always-frontier ranges from **-164.6% to
  +98.8%**.
- **The savings conclusion flips in 3 of 27 combinations.**

| Tier1 x | Tier2 x | Tier3 x | EcoLogic J | Frontier J | Savings % |
|---|---|---|---|---|---|
| 5 | 5 | 0.2 | 3,209 | 1,213 | **-164.6** |
| 5 | 1 | 0.2 | 2,852 | 1,213 | **-135.1** |
| 5 | 0.2 | 0.2 | 2,781 | 1,213 | **-129.2** |
| 1 | 5 | 0.2 | 1,007 | 1,213 | +17.0 |
| 1 | 1 | 0.2 | 650 | 1,213 | +46.4 |
| 0.2 | 5 | 5 | 802 | 30,329 | +97.4 |
| 0.2 | 1 | 5 | 444 | 30,329 | +98.5 |
| 0.2 | 0.2 | 5 | 373 | 30,329 | +98.8 |

**Answer to the question as posed: the savings conclusion does not survive the
full +/-5x band. It flips.** All three flips share one signature — Tier 1's
rate understated by 5x *and* Tier 3's rate overstated by 5x. That combination
is a 25x adverse swing against a 10.4x margin, so the sign reversal is
arithmetically forced rather than incidental.

This is a narrow but real failure mode, and it is not an exotic corner. The
flip condition is "the small reasoning model is less efficient per token than
assumed and gpt-4o is more efficient than assumed", which is exactly the
direction one would worry about given that the 0.5 J/1k figure is a modeled
estimate for a *different, retired* model and the 60 J/1k figure for gpt-4o is
itself an unsourced estimate. The savings claim is robust to rate error in 24
of 27 scenarios but is not unconditional.

Note also that Always-Tier-2 saves 93.5% under the paper's rates and holds up
better under perturbation, because it uses 4.4x fewer tokens than EcoLogic
routing (262,573 vs 1,161,679).

### Run-to-run energy variance

Because token counts drive energy and token counts are not reproducible at
temperature 0 (Section 6), the energy figures carry run-to-run noise. On the
21-item repeat sample, aggregate completion tokens moved **-20.6% (Tier 1)**,
**+37.5% (Tier 2)** and **-4.6% (Tier 3)** between two identical runs. Energy
numbers here should be read as having tens-of-percent run-to-run uncertainty
for the reasoning tiers, on top of the rate uncertainty above.

---

## 6. Determinism spot-check

All calls were made at `temperature = 0`. 21 items (7 per benchmark) were
re-run a second time, 63 calls total.

| Tier | Pairs | Byte-identical text | Identical token count | Identical grade |
|---|---|---|---|---|
| 1 (Qwen3.5-9B) | 21 | 9/21 | 2/21 | **21/21** |
| 2 (gpt-oss-20b) | 21 | 13/21 | 4/21 | **21/21** |
| 3 (gpt-4o) | 21 | 15/21 | 17/21 | **21/21** |

Provider-side nondeterminism at temperature 0 is substantial: fewer than half
of Tier 1 responses were byte-identical across two runs, and only 2 of 21 had
the same completion-token count. Per-item token counts moved by a mean of
19.6% (Tier 1), 29.9% (Tier 2) and 1.5% (Tier 3), with one Tier 2 item moving
384%.

**No grade flipped in the spot-check** (63/63 agreement), so the accuracy
figures look stable at this sample size. The energy figures do not — see
above. Full pair-level data in `raw_results/determinism.json`.

---

## 7. What failed

Everything in this section is a real observed problem, not a hypothetical.

**API errors in the final graded run: 0** of 1,092 calls. One earlier run hit
a single Together AI HTTP 429 rate-limit exhaustion on `HumanEval/61` (Tier 1)
after 5 exponential-backoff retries; it succeeded on a resume pass. No item is
missing a response from any tier, so all 364 items are fully paired for
McNemar.

**The first full run had to be discarded and re-run.** With a uniform
4,096-token cap, Tier 1 was truncated on 57/164 HumanEval, 21/100 MMLU and
13/100 GSM8K items, while gpt-4o never exceeded 458 completion tokens. The cap
was binding for one tier and irrelevant for another, so it was measuring the
cap, not the models: Tier 1 HumanEval scored 64.6% under that cap versus 86.0%
after raising it. The cap was raised to 16,384 for all tiers (non-binding for
Tier 2 and Tier 3) and everything was re-run from scratch. The discarded run
is preserved as `raw_results/responses_cap4096.jsonl.gz` and
`raw_results/graded_cap4096.jsonl.gz`. **Anyone benchmarking reasoning models
against non-reasoning models under a shared token cap should assume their
numbers are wrong until they check truncation rates per tier.**

**Residual truncation at 16,384 tokens:** Tier 1 still hit the cap on 18/164
HumanEval, 11/100 MMLU and 9/100 GSM8K items; Tier 2 on 1/164 HumanEval;
Tier 3 never. These are degenerate-repetition failures rather than
cap-too-tight cases (the model loops instead of concluding), so they are
counted as incorrect. Tier 1's accuracy is nonetheless still somewhat
depressed by a failure mode the other tiers do not exhibit, and its numbers
should be read with that in mind.

**Ungradable outputs:** 4 Tier 1 HumanEval responses (`no_code`) contained no
extractable code block at all — the model exhausted its budget mid-reasoning
without emitting an implementation. Scored incorrect, since a code request
that produces no code is a failure. No MMLU or GSM8K response was ungradable:
answer extraction succeeded on 100% of 600 responses.

**Benchmark items excluded: none.** All 164 HumanEval, 100 MMLU and 100 GSM8K
items were run and graded on all three tiers. No item was dropped for any
reason.

**Extraction leniencies applied equally to all tiers**, worth listing because
they are grading decisions rather than model behaviour:
- Where a tier returned empty `content` but non-empty `reasoning_content`
  (routine for Tier 1), the reasoning trace was used as the answer. Without
  this, Tier 1 would score near zero for reasons that have nothing to do with
  correctness.
- HumanEval candidates are extracted from the last ```python block containing
  the target function; the official prompt preamble (its imports) is prepended
  before executing the official test suite unmodified.
- GSM8K takes the last `Answer:` number, else the last `####` number, else the
  final number in the text. One Tier 2 item answered `47/3`, from which "47"
  was extracted; the gold answer was 15, so the grade is unaffected.

**Sandboxing caveat:** HumanEval candidates execute in a separate subprocess
with `python3 -I -S`, a 4 GB address-space limit, a 15 s CPU limit, a 256
process limit and a temp CWD. This is process isolation, not a container or
VM, and network access is not blocked. Adequate for benchmark code; not a
hostile-code sandbox.

---

## Substituted models

The paper's Tier 1 and Tier 2 models (`google/gemma-3n-E4B-it`,
`ServiceNow-AI/Apriel-1.6-15b-Thinker`) have been **retired from Together AI
serverless inference** and cannot be called. They were replaced with
energy-adjacent models, as agreed:

| Tier | Paper model | Model actually tested | Paper rate | Adjusted rate |
|---|---|---|---|---|
| 1 | google/gemma-3n-E4B-it | **Qwen/Qwen3.5-9B** | 0.5 J/1k | 1.1 J/1k |
| 2 | ServiceNow-AI/Apriel-1.6-15b-Thinker | **openai/gpt-oss-20b** | 1.5 J/1k | 2.0 J/1k |
| 3 | gpt-4o | **gpt-4o** (unchanged) | 60 J/1k | 60 J/1k |

This is the largest threat to validity in the whole report. The substitutes are
2-4x larger than the originals and, critically, both are *reasoning* models
that emit long chain-of-thought traces, which the retired models did not. The
11.5x token inflation in Section 5 is very likely an artifact of that
substitution rather than a property of EcoLogic's intended Tier 1/2. Results
here characterise EcoLogic's *routing policy* faithfully (the classifier is the
real one) but characterise the paper's *specific model lineup* only by analogy.

---

## Judge hygiene

Stage 5 of the addendum applies only "if any free-form grading remains". None
does. HumanEval is graded by executing the official test suites, MMLU by
multiple-choice letter match, GSM8K by numeric final-answer match. All 1,092
responses were graded mechanically, so no judge models, answer-order
randomisation, or Cohen's kappa are reported — there is nothing left for a
judge to score. This is a strict improvement over the previous report, whose
reasoning and code numbers depended on a single LLM judge.

---

## LIMITATIONS

**Energy is modeled, not measured.** Not one joule in this report was
measured. Every energy figure is `tokens / 1000 * an assumed constant`, using
rates hard-coded in `backend/main.py` with no cited derivation, no measurement
methodology, and no accounting for batching, hardware, utilisation, quantisation,
serving stack, PUE, or idle draw. The provider APIs expose token counts, not
power. Section 5 shows the headline savings claim flips sign under a +/-5x
perturbation of those constants, so the ranking of policies by energy should be
treated as a consequence of the assumptions rather than a finding. Nothing here
validates the rate constants themselves; a real result needs wall-power
measurement on known hardware.

**Benchmark is not real traffic.** HumanEval, MMLU and GSM8K are clean,
self-contained, single-turn academic tasks with verifiable answers. Real
EcoLogic traffic is presumably multi-turn, open-ended, unevenly distributed
across difficulty, and full of prompts with no ground truth. The 364-item mix
here (45% code, 27% multiple-choice, 27% grade-school math) is an artifact of
the addendum's spec, not a measured traffic distribution, and system-level
accuracy under any policy is a weighted average whose weights I chose. Section
4's prompt-sensitivity result is also a direct warning that routing behaviour
on benchmark-formatted prompts may not transfer to real ones. Additionally,
all three benchmarks are old, public, and near-certainly in the training data
of all three models, so absolute accuracies are optimistic; the paired
comparisons between tiers are the more trustworthy part.

**Single run.** One generation per item per tier, so per-item outcomes are one
sample, not an expectation. This is pass@1 with n=1, which is a noisy estimator
even where it is unbiased. The determinism check (Section 6) shows the
providers are not reproducible at temperature 0 even in text or token count, so
a re-run would produce different per-item grades and materially different energy
totals. The Wilson intervals quantify sampling error over *items* only; they do
not capture generation variance, and the true uncertainty on every accuracy
figure is therefore wider than the interval printed next to it. The oracle
policy is especially exposed, since selecting the cheapest tier that happened to
be right on a single sample overfits that sample and makes 96.4% an optimistic
ceiling.

**Other limits worth naming.** The three tiers are not the paper's models (see
[Substitutions](#substituted-models)). Tier 1 carries a degenerate-repetition
failure mode the others lack, which depresses its scores for reasons partly
unrelated to capability. The oracle's tie-breaking and its charge for
all-tiers-wrong items are conventions I chose, documented in
`benchmark/analyze.py`. MMLU is 100 items over 10 of 57 subjects, so
per-subject cells are n=10 and individually near-meaningless. And this
evaluation was built and run by the same process that is reporting on it, with
no independent replication.

---

## Reproducing

```bash
pip install httpx pyarrow scipy fastapi pydantic python-dotenv
export TOGETHER_API_KEY=...   # Tiers 1 and 2
export OPENAI_API_KEY=...     # Tier 3

python3 benchmark/build_benchmark.py     # seeded 364-item set (seed 20260905)
python3 benchmark/router.py              # real classifier over all items
python3 benchmark/run_benchmark.py       # 1,092 calls, temperature 0, resumable
python3 benchmark/grade.py               # executes HumanEval, extracts MMLU/GSM8K
python3 benchmark/run_benchmark.py --repeat 7
python3 benchmark/determinism.py
python3 benchmark/analyze.py             # policies, Wilson, McNemar, sensitivity
```

| File | Contents |
|---|---|
| `raw_results/benchmark_items.json` | All 364 items with the seeded selection manifest (MMLU subject/index, GSM8K indices) |
| `raw_results/graded.jsonl` | **Canonical raw data.** 1,092 rows: prompt, full response, reasoning trace, token counts, cost, latency, extracted answer, grade |
| `raw_results/responses.jsonl.gz` | Same run pre-grading |
| `raw_results/routing.json` | Real classifier's decision per item, for bare and wrapped prompts |
| `raw_results/analysis.json` | Every computed number, including all 27 sensitivity combinations |
| `raw_results/determinism.json` | Pair-level repeat-run comparison |
| `raw_results/tables.md` | Machine-generated tables |
| `raw_results/*_cap4096.*` | Discarded first run, kept as evidence for the truncation finding |
| `raw_results/responses_pilot.jsonl` | 27-call pilot used for the pre-run cost estimate |

Pre-run cost estimate was $0.74 projected from the pilot's real token counts
(worst case ~$15 had every call saturated the token cap), against a $25
threshold. Actual total: **$1.9497**.

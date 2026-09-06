# Addendum 4 (Stages 7–10) — completion summary

Nothing in `raw_results/`, `results_report.md` or `router_v2/` was modified.
Everything here is new, in `stage7_10/`.

## Status at a glance

| Stage | What it was for | Status |
|---|---|---|
| **7** | Scaled, cleaned retest of the calibration-gap hypothesis | **BLOCKED** — Together AI credit limit (HTTP 402) at 45.8% of generation. No verdict issued. |
| **8** | Derive the exact regret-decomposition correction term | **Complete.** Reconciles to residual 2.2×10⁻¹⁶. |
| **9** | External generalization check on published routers | **Complete**, within real scope limits. Correction reproduces on RouteLLM's own data. |
| **10(a)** | Generation-variance decomposition | **BLOCKED** with Stage 7 (needs its test set). |
| **10(b)** | Evaluation card | **Complete.** |
| **10(c)** | Contribution-type framing | **Complete.** |
| **10(d)** | Full reproducibility manifest, Stages 1–10 | **Complete.** |

Total spend this addendum: **$29.31** (all of it Stage 7 generation), against a
pre-registered gate of $150. The gate was never the binding constraint.

---

## Stage 7 — blocked, and exactly why

The Together AI account ran out of credits partway through generation. Both
Tier 1 (`Qwen/Qwen3.5-9B`) and Tier 2 (`openai/gpt-oss-20b`) are Together
hosted, so the failure removed two of three tiers. Confirmed persistent: after
halting all workers, a single 5-token call still returned HTTP 402
`credit_limit`. Tier 3 (`gpt-4o`, OpenAI) was unaffected and completed 100%.

```
HTTP 402: "Credit limit exceeded, please add credits."   type: credit_limit
```

| | Target | Succeeded |
|---|---|---|
| Pool, Tier 3 (OpenAI) | 15,000 | **15,000** (100%) |
| Pool, Tier 2 (Together) | 15,000 | 3,512 (23.4%) |
| Pool, Tier 1 (Together) | 15,000 | 3,374 (22.5%) |
| New test set, all tiers | 3,276 | 220 (6.7%) |
| **Total** | **48,276** | **22,106** (45.8%) |

Items usable under the pre-registered k=3 label rule: **1,039 of 5,000** pool
items, **all from TRAIN, none from CALIBRATION**, and **0 of 364** test items.

**No accuracy table, no S1/S2 determination, and no calibration-gap number is
reported for Stage 7.** Refitting on the salvage would have been dishonest for
three independent reasons: (1) 1,039 items is 0.87× Stage 1's 1,200, so the
"4–5× more data" premise the stage exists to test is unattainable — it would
test the label fix alone, at less data than before, which was not what was
pre-registered; (2) zero CALIBRATION items completed, so the pre-registered
threshold rule ("highest CALIBRATION accuracy at ≤10% of always-frontier
energy") has no data, and selecting on TRAIN is the exact leak the
pre-registration forbids; (3) zero test items completed, so the one-shot
evaluation cannot be run, and substituting the Stage 5 test set is explicitly
ruled out by this addendum.

**What did complete in Stage 7**: the pre-registration (`prereg_stage7.md`,
commit `97a08cb`, before any data existed); pool and test-set construction with
all five disjointness assertions printing **PASS**; the cost pilot ($47.47
projected); and the entire analysis pipeline, written and exercised end-to-end
on the partial data (grading ran clean on 21,886 responses).

**One measurement survives, clearly labelled as a partial diagnostic**, in
`stage7_results.md`: across 1,039 complete items, the k=3 samples disagreed 2–1
on **7.1%** of Tier 1 labels, **8.2%** of Tier 2, and **4.0%** of Tier 3. That
is the share of Stage 1's single-sample labels that were near coin flips, so it
bounds what the label-noise fix could ever have bought. It is suggestive, not a
test of the hypothesis.

**To resume**: add Together credits and re-run `s7_run.py`; resumption is keyed
on `(tier, item_id, sample_idx)`, so none of the $29.31 already spent is
repeated. Expected additional cost ≈ **$13**, wall time ≈ **1.5 h** at the
observed 341 Together calls/min. Pools, test set and seed are already fixed on
disk, so what resumes is the pre-registered experiment.

**The hypothesis remains open.** Whether the Stage 6 calibration gap is a
data-quantity problem or a structural one is not settled by this addendum, and
the reports say so rather than implying otherwise.

---

## Stage 8 — complete

Derived the exact correction to the confusion-matrix regret formula and stated
it as a proposition with proof:

```
R_true = R_naive + sum_{i != j} Cov( 1{t* = i, t^ = j},  e_j(x) - e_i(x) )
```

Equivalently, `P(cell)` times the conditional **mean shift**
`E[D_ij | cell] − E[D_ij]`. The naive formula is exact if and only if that sum
vanishes — in particular whenever per-item energy is constant within a tier.

**Reconciliation on Stage 6's own data**, the two numbers that disagreed:

| Quantity | J/item |
|---|---|
| `R_naive` (confusion-matrix formula) | −0.249623217593 |
| correction term | +0.533592662037 |
| **sum** | **+0.283969444444** |
| `R_true` (direct per-item measurement) | **+0.283969444444** |
| residual | 2.2×10⁻¹⁶ |

The correction is **1.9× the size of the true regret** and flips its sign. It
was derived first and evaluated once; it was not tuned to make the numbers
agree.

Two honesty notes recorded in the write-up: the requested form
`Σ P(cell)·Cov(e_j − e_i | cell)` is not well-formed (a covariance needs two
arguments), so the correct object is stated instead; and the proposition is
elementary algebra, presented as such rather than as a deep result.

→ `regret_correction_derivation.md`

---

## Stage 9 — complete, and it generalizes

**RouteLLM does release enough to check this, for one of its three
benchmarks.** `evals/gsm8k/gsm8k_responses.csv` ships per-item correctness
*and full response text* for both routed models, so per-item token cost is
recoverable with the models' real tokenizers. MMLU (57 files) and MT-Bench ship
correctness and judge scores only — no responses, so no cost reconstruction.
**No RouteLLM artifact publishes token counts or per-item cost as a numeric
field.**

Their cost axis is a **call fraction**: `routellm/evals/evaluate.py` contains no
occurrence of `cost` or `token`; every metric is computed against
`strong_percentage`. The README's "reduce costs by up to 85%" is a call-count
claim converted to dollars by assuming constant cost per call — exactly the
assumption Stage 8 characterises.

Using their released BERT router checkpoint, their decontamination list, their
threshold grid, and 1,307 GSM8K items:

- The correction reconciles **exactly** on external data (max residual over the
  whole sweep: **8.7×10⁻¹⁹**).
- **The sign flip reproduces.** At their 30%-strong operating point the naive
  formula says the router spends *more* than the oracle (+0.0000321 $/item)
  when it actually spends *less* (−0.0000923 $/item).
- **Naive accounting is biased toward the router at every operating point**,
  understating true cost by 0.4% to **6.3%** — i.e. overstating savings — and
  the bias grows as routing gets more aggressive.
- **The magnitude scales with within-model cost dispersion.** RouteLLM's GSM8K
  answers are short and similar in length (cost CV ≈ 0.36–0.43), so the effect
  is ~6%. Our Tier 1 is a reasoning model averaging ~5,300 completion tokens
  with an order-of-magnitude spread, and there the correction was 1.9× the
  regret. The problem is therefore getting **worse** as the field routes among
  reasoning models with variable-length thinking budgets.

**FrugalGPT** (Chen et al., 2023) was checked as the pre-registered fallback:
no author-released code or per-query cost data exists. As a cascade its exposure
is structurally greater; this was not verifiable and was not estimated. No
stand-in dataset was simulated.

→ `external_generalization.md`

---

## Stage 10

**(a) Generation-variance decomposition — BLOCKED.** Requires k=3
temperature-0 regenerations of the Stage 7 frozen test set; 0/364 items
completed. `s7_variance.py` is written and runs on resumption. Consequence: the
share of each reported Wilson interval attributable to pure regeneration noise
is currently unquantified, though the earlier finding that temperature 0 is not
deterministic (byte-identical text on only 9/21 Tier 1 repeats) still stands.

**(b) Evaluation card — complete.** `evaluation_card.md`: what is evaluated (a
routing policy, not a model), the four benchmarks and why (objective
auto-gradability, explicitly **not** claimed representative of EcoLogic
traffic), the prompt-wrapper sensitivity finding (the classifier agrees with
itself only **53.0%** of the time between raw query and wrapped prompt,
rerouting half the corpus and multiplying energy 5×), a measured-vs-modelled
table stating plainly that **no joule was measured anywhere**, pre-registration
and one-shot discipline including the one disclosed deviation, and twelve
limitations carried forward from every prior stage.

**(c) Contribution framing — complete.** `contribution_framing.md`: one
paragraph declaring the primary contribution as an evaluation-methodology
contribution carried by a negative result, plus notes on why it leads with the
audit rather than the derivation and where a reviewer will push hardest.

**(d) Reproducibility manifest — complete.** `reproducibility_manifest.md`:
every seed, package version, model string, endpoint, dataset revision,
item-set partition, energy rate, pre-registration commit and measured cost
across Stages 1–10, with reproducibility hazards flagged as **not pinned**
rather than glossed — chiefly that provider model weights have no immutable
revision and `gpt-4o` is a moving alias.

---

## Files

| File | Contents |
|---|---|
| `prereg_stage7.md` | Stage 7 pre-registration, committed before any data (`97a08cb`) |
| `stage7_results.md` | Stage 7 block report: what was collected, why no reduced analysis is honest, the surviving diagnostic, how to resume |
| `regret_correction_derivation.md` | Stage 8 proposition, proof, reconciliation |
| `external_generalization.md` | Stage 9 external check |
| `evaluation_card.md` | Stage 10(b) |
| `contribution_framing.md` | Stage 10(c) |
| `reproducibility_manifest.md` | Stage 10(d) |
| `s7_LIMITATIONS.md` | Limitations for Stages 7–10 |
| `build_pools.py`, `s7_run.py`, `s7_grade.py`, `s7_data.py`, `s7_fit.py`, `s7_calibrate.py`, `s7_final.py`, `s7_variance.py` | Stage 7 / 10(a) pipeline, written and exercised, awaiting credits |
| `regret_correction.py`, `external_check.py` | Stage 8 / 9 analysis |
| `s7_*_pool.json`, `s7_test_set.json` | the frozen selections (disjointness PASS) |
| `s7_pool_responses.jsonl.gz`, `s7_pool_graded.jsonl.gz`, `s7_pool_labels.json` | partial Stage 7 generations and grades |
| `regret_correction_validation.json`, `external_generalization.json` | machine-readable Stage 8 / 9 results |

The poster and earlier abstract work was not touched.

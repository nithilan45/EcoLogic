# Pre-registration — Stage 7 (scaled, cleaned retest of the calibration-gap hypothesis)

Committed **before** any Stage 7 data was generated, in its own commit ahead of
every other `stage7_10/` artifact. The git history of this file is the evidence.
Nothing below was edited after results were seen.

## Hypothesis under test

From Stage 6: the **5.83 pp calibration gap** between the learned router and the
LP-relaxed MCKP frontier is caused by *insufficient and noisy training data*,
not by architectural or problem-structure limits (the discreteness gap was
~0 pp, so the headroom is not blocked by having to pick one tier per item).

This stage tests that hypothesis. It does not assume the answer. The two obvious
alternatives — that the gap is a **feature-representation** limit, or that
per-item tier success is substantially **irreducibly unpredictable** from the
prompt alone — are not distinguished from each other by this design, and I will
not claim they are.

## Success criteria (unchanged from Stage 6, restated verbatim in force)

Evaluated on a **new** frozen test set, once:

- **S1** — the learned router beats Always-Tier-2 on accuracy, McNemar exact
  **p < 0.05**.
- **S2** — the learned router matches Always-Tier-2's accuracy (McNemar exact
  **p >= 0.05**) at **>= 15% lower energy**.

**If neither holds again, that is the finding.** It falsifies the
calibration-gap-is-a-data-problem hypothesis and indicates the gap is
structural — a limit of the feature representation or of prompt-level
predictability rather than of training-set size. That is a legitimate and
reportable outcome and will be written up as such, not as a failed experiment.

## Third outcome, pre-registered now

If accuracy improves only modestly — concretely, if the learned router closes
**1–2 pp** of the 4.1 pp accuracy gap to Always-Tier-2 **without** reaching
McNemar significance — that is recorded as **"partial support, inconclusive"**.
It will be reported under that label. It will not be rounded up to a clean
positive, and it will not be dismissed as a clean negative.

Operationally, the outcome label is assigned as:

| Condition | Label |
|---|---|
| S1 met | hypothesis **supported** |
| S2 met | hypothesis **supported** |
| accuracy gap to Always-Tier-2 narrows by >= 1.0 pp vs Stage 5's router, but neither S1 nor S2 | **partial support, inconclusive** |
| otherwise | hypothesis **not supported** |

Stage 5's learned router scored 92.03% against Always-Tier-2's 92.31% on the
*old* test set, a gap of 0.28 pp. Because Stage 7 uses a *different* test set,
the "narrows by >= 1.0 pp" comparison is made against Always-Tier-2 **measured
on the new test set**, comparing the new router's gap to Stage 5's 0.28 pp gap.

## Label-noise fix (fixed now, before running)

- **k = 3 independent samples per item per tier**, at **temperature 0.7** (not
  0). Temperature-0 repeats from the same model are the least informative
  possible resamples; 0.7 samples the model's actual output distribution.
- The per-tier grade for an item is the **majority grade over the 3 samples**
  (correct iff >= 2 of 3 samples grade correct), not a single-sample grade.
- The per-tier **energy** for an item is the **mean total tokens over the 3
  samples**, converted at the paper's J/1k rates.
- **Oracle label** = the lowest-energy tier whose majority grade is correct.
  Items where no tier's majority grade is correct are labelled with the
  lowest-energy tier and **flagged separately** as label noise, exactly as in
  Stage 1.

Grading is unchanged and fully objective: MBPP and code by executing official
test assertions in a resource-limited subprocess, MMLU by letter match, GSM8K by
numeric final-answer match. No LLM judge.

Generation config for the pool: temperature 0.7, **16,384-token cap** (unchanged;
the Stage 5 run established that reducing it measures the cap rather than the
model).

## Sample sizes, and an honest ceiling I cannot get around

Target training/calibration pool: **5,000 items** (4.2x Stage 1's 1,200), split
70/30 into TRAIN and CALIBRATION as before.

| Benchmark | Stage 7 pool | Stage 1 pool | Scale-up | Source |
|---|---|---|---|---|
| GSM8K | 2,295 | 400 | 5.7x | GSM8K **train** split (7,473 items; Stage 1 and both test sets use only the *test* split, so overlap is impossible by construction) |
| MMLU | 2,295 | 400 | 5.7x | MMLU test split, indices disjoint from all prior selections |
| MBPP | 410 | 400 | **1.03x** | all MBPP items not already used and not reserved for the new test set |
| **Total** | **5,000** | **1,200** | **4.2x** | |

**MBPP has only 974 problems in total and Stage 1 already consumed 400.** After
reserving 164 for the new frozen test set, exactly 410 remain. So the code
portion of the training pool **cannot be scaled** — it grows 1.03x while the
other two benchmarks grow 5.7x. This matters because Stage 5's router deviated
from static Tier 2 mainly on code. Stage 7 therefore tests "more data" unevenly,
and a null result on code will be **confounded** with the inability to add code
training data. I am stating this before running rather than discovering it
afterwards, and it will be repeated in the results and limitations.

New frozen test set: **364 items** (MBPP 164, MMLU 100, GSM8K 100), matching the
original set's shape. Generated at **temperature 0** to match the main
evaluation, with **k = 3** samples per item per tier so that Stage 10(a)'s
generation-variance decomposition uses the same responses; sample index 0 is the
primary generation for the headline numbers.

## Disjointness requirements

Asserted programmatically and printed as PASS/FAIL before any generation:

1. Zero item-ID overlap between the Stage 7 pool and the **original 364-item
   frozen test set**.
2. Zero item-ID overlap between the Stage 7 pool and **Stage 1's 1,200-item
   pool**. This is a fresh pool, not a superset: reusing Stage 1 items with new
   majority-voted labels would confound "more data" with "better labels".
3. Zero item-ID overlap between the **new** frozen test set and anything used
   anywhere in Stages 1–6 or in the Stage 7 pool.

The original Stage 5 test set is **not** re-evaluated in this addendum.

## Cost and throughput gates

Full generation is 5,000 items x 3 tiers x 3 samples = **45,000 calls**, plus
364 x 3 x 3 = **3,276** for the new test set: **48,276 calls**.

A pilot is run first and the projected total reported. **If the projection
exceeds $150, work stops and the user is asked before the full run.**

**Throughput contingency, fixed now.** Stage 1's 3,600 calls took 103 minutes,
so 48,276 calls is a many-hour job dominated by Tier 1's long reasoning traces.
The pool is **randomly shuffled before generation and generated in item order**,
so any prefix of completed items is itself a uniform random subsample of the
pool. If the run cannot complete, the analysis uses **the largest prefix of
items for which all 3 tiers x 3 samples succeeded**, and the actual N is reported
prominently in the results and in SUMMARY.md alongside the 5,000 target. Items
with any missing generation are excluded entirely rather than partially imputed.
A reduced N weakens the "more data" test proportionally, and that weakening will
be stated, not glossed.

## Analysis, fixed now

- Both variants (R1 TF-IDF + n-grams + hand features, R2 local MiniLM
  embeddings + hand features) are refit on the new majority-voted labels.
- Variant and hyperparameters are selected by cross-validation **within TRAIN
  only**, then confirmed once on CALIBRATION — identical procedure to Stage 2.
- The routing rule is the **cost-ordered** rule fixed at the end of Stage 3
  (cheapest expected energy first, ordering derived from TRAIN means only, Tier 3
  fallback). This is now fixed in advance rather than corrected mid-experiment.
- Threshold selection rule is **unchanged from the Stage 3 pre-registration**:
  highest CALIBRATION accuracy among thresholds whose CALIBRATION energy is
  <= **10%** of always-frontier energy on the same split; ties toward lower
  energy; if none qualifies, the lowest-energy threshold, reported as such.
- MCKP LP frontier, integer frontier, threshold sweep and the
  calibration/discreteness gap decomposition are recomputed on the new
  CALIBRATION split by the same code paths.
- The Stage 5 table is reproduced with the new router as a seventh row and
  Stage 5's original learned router as an eighth row, so the data-scaling effect
  is directly visible.

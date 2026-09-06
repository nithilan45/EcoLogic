# Addendum 4 (Stages 7–10) — completion summary

Nothing in `raw_results/`, `results_report.md` or `router_v2/` was modified.
Everything here is new, in `stage7_10/`.

## Status at a glance

| Stage | What it was for | Status |
|---|---|---|
| **7** | Scaled, cleaned retest of the calibration-gap hypothesis | **Hypothesis not supported** on the evidence in hand (gap 5.83 → 5.00 pp for 4.2× data). Pool 100% generated; pre-registered S1/S2 test-set verdict **pending 1,091 calls / $1.91** of OpenAI credit. |
| **8** | Derive the exact regret-decomposition correction term | **Complete.** Reconciles to residual 2.2×10⁻¹⁶. |
| **9** | External generalization check on published routers | **Complete**, within real scope limits. Correction reproduces on RouteLLM's own data. |
| **10(a)** | Generation-variance decomposition | **Pending** with the Stage 7 test set (same 1,091 calls). |
| **10(b)** | Evaluation card | **Complete.** |
| **10(c)** | Contribution-type framing | **Complete.** |
| **10(d)** | Full reproducibility manifest, Stages 1–10 | **Complete.** |

Total spend this addendum: **$41.11** (all of it Stage 7 generation), against a
pre-registered gate of $150. The gate was never the binding constraint; two
separate account credit limits were.

---

## Stage 7 — hypothesis not supported; formal verdict pending 1,091 calls

Together AI credits were topped up and the **pool generation completed in
full**: 45,000/45,000 calls, all 5,000 items with complete k=3 × 3-tier
generations, split 3,500 TRAIN / 1,500 CALIBRATION as pre-registered. The new
frozen test set has Tier 1 and Tier 2 complete (1,092 each) but **1,091 of
1,092 Tier 3 (`gpt-4o`) calls missing**, because the OpenAI account then hit its
own credit limit.

| | Target | Succeeded |
|---|---|---|
| Pool, all three tiers | 45,000 | **45,000** (100%) |
| Test set, Tiers 1 and 2 | 2,184 | **2,184** (100%) |
| Test set, Tier 3 (`gpt-4o`) | 1,092 | 1 (0.1%) |
| **Total** | **48,276** | **47,185** (97.7%) |

**The hypothesis-level answer is in, and it is negative.** Scaling the pool 4.2×
and denoising labels with majority-of-k=3 moved the calibration gap from Stage
6's **+5.83 pp** to **+5.00 pp** — 0.83 pp closed — while held-out per-tier head
AUCs stayed flat (Tier 1 0.656 → 0.665, Tier 2 0.701 → 0.702). The
cross-validation AUC did rise (0.675 → 0.716), but that is largely the labels
becoming easier to predict once denoised, not the heads generalising better. The
discreteness gap remained **+0.00 pp**, so the headroom is still not blocked by
one-tier-per-item. This points at the gap being **structural** — feature
representation or prompt-level predictability — which the pre-registration
anticipated and named as a legitimate finding.

**The pre-registered S1/S2 verdict is withheld**, because it is defined on the
frozen test set and the always-frontier and oracle rows need Tier 3. On the
CALIBRATION split neither criterion would hold — the router is +0.47 pp over
Always-Tier-2 (p = 0.167, not significant) at **67% more energy**, and it is
**significantly worse than Always-Tier-1** (−1.53 pp, p = 0.021) for only 4.8%
less energy, so Always-Tier-1 dominates it outright. That is indicative, not the
verdict, and the report says so.

**Label noise the fix actually removed**: the k=3 samples split 2–1 on **5.8%**
of Tier 1 labels, **8.0%** of Tier 2 and **4.0%** of Tier 3, across all 5,000
items. 192 items (3.8%) had no tier correct and are flagged as noise rather than
treated as ground truth.

**Four infrastructure fixes were needed to get the run to converge**, all
affecting only how calls are issued, never what is asked or graded: a hard
per-task timeout (a single stalled socket could pin a worker for ~25 min, and
with 80 workers the run silently flatlined at 0 calls/min), phase-granular HTTP
timeouts, 15-second keepalive expiry with a 25-minute cap per pass (throughput
decayed from ~200 calls/min toward zero as sockets went stale), and
proportional interleaving of the TRAIN and CALIBRATION queues — the previous
TRAIN-first order is why the earlier interruption left zero CALIBRATION items.

**To finish**: add **$1.91** of OpenAI credit (measured at $0.00175/call over
15,000 completed Tier 3 calls, not list-price estimated) and run `s7_run.py
--target test`, then grade, `s7_final.py` and `s7_variance.py`. Router weights,
variant, threshold τ = 0.2653, pools, test set and seed are all fixed on disk
and committed; the one-shot property is intact because the frozen test set has
not been scored.

→ `stage7_results.md`

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

**(a) Generation-variance decomposition — pending the same 1,091 calls.** The
k=3 temperature-0 regenerations of the Stage 7 frozen test set are complete for
Tiers 1 and 2 and missing for Tier 3, so the decomposition cannot be reported
for the policies that use the frontier tier. `s7_variance.py` is written and
runs as soon as Tier 3 lands. Consequence for now: the share of each reported
Wilson interval attributable to pure regeneration noise is unquantified, though
the earlier finding that temperature 0 is not deterministic (byte-identical text
on only 9/21 Tier 1 repeats) still stands.

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
| `stage7_results.md` | Stage 7 results: generation status, label-noise measurement, the two tests of the hypothesis, CALIBRATION policy table, what remains |
| `s7_model_comparison.md` | R1 vs R2 ablation on the scaled majority-voted pool, with Stage 2 columns for the data-scaling comparison |
| `regret_correction_derivation.md` | Stage 8 proposition, proof, reconciliation |
| `external_generalization.md` | Stage 9 external check |
| `evaluation_card.md` | Stage 10(b) |
| `contribution_framing.md` | Stage 10(c) |
| `reproducibility_manifest.md` | Stage 10(d) |
| `s7_LIMITATIONS.md` | Limitations for Stages 7–10 |
| `build_pools.py`, `s7_run.py`, `s7_grade.py`, `s7_data.py`, `s7_features.py`, `s7_fit.py`, `s7_calibrate.py`, `s7_calib_policies.py`, `s7_final.py`, `s7_variance.py` | Stage 7 / 10(a) pipeline; all run except `s7_final.py` and `s7_variance.py`, which need Tier 3 |
| `regret_correction.py`, `external_check.py` | Stage 8 / 9 analysis |
| `s7_*_pool.json`, `s7_test_set.json` | the frozen selections (disjointness PASS) |
| `s7_pool_labels.json`, `s7_pool_samples.csv.gz` | k=3 majority labels, and the per-sample grade/token/cost table all numbers derive from |
| `s7_router_model.pkl`, `s7_model_selection.json`, `s7_chosen_threshold.json` | the fixed router: weights, selected variant, pre-registered threshold |
| `s7_threshold_sweep.csv/.png`, `s7_mckp_frontier.csv/.png`, `s7_mckp_gaps.json`, `s7_calib_policies.json` | Stage 7 calibration sweep, MCKP frontier, gap decomposition, CALIBRATION policy table |
| `s7_run_accounting.json` | per-target call counts, failure kinds and measured spend |
| `regret_correction_validation.json`, `external_generalization.json` | machine-readable Stage 8 / 9 results |

The poster and earlier abstract work was not touched.

# Pre-registration — learned router (Addendum 3)

Committed **before** the training pool was built, before any model was fitted,
and before any threshold was selected. The git history of this file is the
evidence: it is committed in its own commit ahead of every other `router_v2/`
artifact. Nothing below was edited after results were seen.

## Success criteria (fixed)

Success is **either** of:

- **S1** — the learned router beats Always-Tier-2 on accuracy over the original
  frozen 364-item test set, McNemar exact **p < 0.05**.
- **S2** — the learned router matches Always-Tier-2's accuracy (no significant
  difference, McNemar exact **p >= 0.05**) at **>= 15% lower energy**.

If neither holds, the reported outcome is: *a properly trained router still does
not surpass static assignment on this workload.* That is a valid result and will
be written up as such. No iteration on the router after seeing the test-set
result.

## One-shot rule

The 364-item test set (HumanEval 164 + MMLU 100 + GSM8K 100, seed 20260905, as
listed in `raw_results/benchmark_items.json`) is **frozen**. It is evaluated
exactly once, in Stage 5, with the feature extractor, model weights and
threshold all already fixed. Per-item test-set outcomes are not inspected before
Stage 5. Model and hyperparameter selection happens by cross-validation *within
TRAIN*; the choice is confirmed once on CALIBRATION.

No new API calls are made against the frozen test items. Stage 5 reuses the
already-collected per-tier responses in `raw_results/graded.jsonl`, so the
learned router is scored on exactly the same generations as every baseline in
the previous report. Nothing in `raw_results/` is modified.

## Stage 3 threshold-selection rule (fixed, X filled in now)

> **Pick the threshold that achieves the highest CALIBRATION accuracy among
> those thresholds whose CALIBRATION energy is <= 10% of always-frontier energy
> on the same CALIBRATION split.**

**X = 10%.** Ties are broken toward lower energy; if no threshold satisfies the
constraint, the lowest-energy threshold available is taken and that fact is
reported.

Rationale for X = 10%, stated in advance: the deployed EcoLogic keyword
classifier operated at 11.4% of always-frontier energy on the test set. Holding
the learned router to <= 10% means it must be at least as energy-disciplined as
the system it replaces, so any accuracy improvement cannot be bought simply by
escalating more traffic to Tier 3. It also leaves real headroom above the 6.5%
that Always-Tier-2 uses, so the router is free to escalate selectively.

## Analysis decisions fixed in advance

- **Oracle label** = the lowest-energy tier that answered the item correctly.
  Where no tier answered correctly, the item is labelled with the lowest-energy
  tier (matching the convention in the previous report) and **flagged
  separately** as label noise rather than ground truth.
- **Router output** is a per-tier predicted P(correct), not a hard assignment,
  for Tier 1 and Tier 2. The threshold is a continuous knob swept in Stage 3.
- **Energy** = `total_tokens / 1000 * rate_tier` with the paper's rates
  (Tier 1 = 0.5, Tier 2 = 1.5, Tier 3 = 60 J per 1k tokens), real provider
  token counts, identical to the previous report.
- **Pool/test overlap** is asserted programmatically against the frozen test
  item IDs and the PASS/FAIL result printed. HumanEval items are not reused for
  training under any framing; MBPP substitutes for them in the training pool.
- **Generation config** is unchanged from the previous run: temperature 0,
  16,384-token cap for every tier.

## Cost gate

A pilot is run first and the projected pool cost reported. If the projection
exceeds **$40**, work stops and the user is asked before the full pool runs.

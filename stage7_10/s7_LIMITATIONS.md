# Limitations — Stages 7–10

Stage 7 is **blocked** (see `stage7_results.md`), so the first section is about
limitations that were already baked in *before* the block and would have
constrained the result even had it run. The later sections cover the stages that
did complete.

## Stage 7 (blocked)

### The blocking limitation

The Together AI account exhausted its credits 45.8% of the way through
generation. Tiers 1 and 2 are both Together-hosted, so this removed two of
three tiers. 1,039 of 5,000 pool items completed, all from TRAIN, none from
CALIBRATION, and none of the 364 test items. No verdict is issued and none can
be. Resumption needs ~$13 of Together credit.

### Limitations that would have applied anyway

**MBPP is exhausted, so the code portion could not be scaled.** This was
pre-registered before the run precisely because it is not fixable. MBPP has 974
problems in total; Stage 1 consumed 400 and 164 were reserved for the new test
set, leaving exactly 410 for the pool — a **1.02× scale-up against 5.74× for
MMLU and GSM8K**. Stage 5's router deviated from static Tier 2 mainly on code,
so a null result on code would have been confounded with the inability to add
code training data. Any future attempt at this experiment needs a larger code
corpus (e.g. APPS, CodeContests, LiveCodeBench), not more MBPP.

**Pool GSM8K items come from the train split, test items from the test
split.** This guarantees disjointness by construction, which is why it was
chosen, but it introduces a small distribution difference that the Stage 1
design did not have (Stage 1 used disjoint indices within the same split).

**The design cannot separate its two changes.** Stage 7 alters training-set
size *and* label quality simultaneously. Had it produced a positive result,
attributing that result to "more data" versus "cleaner labels" would have
required a third arm that was not budgeted. The hypothesis as stated
("insufficient/noisy training data") bundles both, so the test was valid for
the bundled claim and uninformative about which half mattered.

**A null result would not have distinguished the two structural
explanations.** "The gap is structural" could mean the feature representation
is inadequate (fixable with better features) or that per-tier success is
largely unpredictable from the prompt alone (not fixable by any router). The
pre-registration says this explicitly. Testing them apart needs a much stronger
feature extractor as a third arm.

**Ordering artifact in the generator.** Items are enqueued TRAIN-file-first,
so a truncated run yields TRAIN items only. The pre-registered throughput
contingency assumed a prefix of the *shuffled pool* would be a valid random
subsample — true of the pool as a whole, but it does not preserve the
TRAIN/CALIBRATION proportions, which is the second reason no reduced analysis
was possible. Interleaving the two files when enqueueing would have made the
contingency actually work, and is the one design fix worth making before
resuming.

**Temperature 0.7 labels, temperature 0 evaluation.** Pool labels deliberately
sample the output distribution while the test set is evaluated at temperature 0.
This is intentional — a majority over temperature-0 repeats would be a much
weaker de-noiser — but it means the router is trained on a slightly different
generation regime than the one it is scored in.

## Stage 8 (complete)

**The proposition is elementary.** It is two lines of algebra from the
definition of covariance, and it is stated as such. It is worth writing down
only because the biased formula is in active use, not because the derivation is
difficult.

**The user-supplied form of the correction does not type-check, and the exact
form differs.** The correction was requested as
`Σ P(t*=i,t̂=j) · Cov(e_j(x) − e_i(x) | t*=i, t̂=j)`, which is not
well-formed — a covariance needs two arguments. The correct object is
`Σ Cov(1{t*=i, t̂=j}, e_j(x) − e_i(x))`, equivalently `P(cell)` times the
conditional **mean shift**. The intended intuition is right and is preserved;
the formula as literally written is not the one that reconciles, and it was not
adjusted to force agreement.

**Validation is on one dataset with one router.** The reconciliation is exact
(residual 2×10⁻¹⁶) but exactness is guaranteed by the algebra — the
reconciliation checks the *implementation*, not the proposition. What is
dataset-specific, and therefore not general, is the *magnitude*: 1.9× the true
regret here. Stage 9 is what tests whether the magnitude matters elsewhere.

## Stage 9 (complete, with real scope limits)

**One benchmark, one router, one model pair.** GSM8K only (1,307 items after
RouteLLM's own decontamination), because it is the sole artifact in their
release carrying response text. MMLU (57 files) and MT-Bench ship correctness
and judge scores but no responses, so per-item cost is unrecoverable there. Of
RouteLLM's five routers, only `bert` runs without per-prompt OpenAI embedding
calls, so only `bert` was evaluated.

**Token counts are reconstructed, not published.** Tokenizing released response
text with the models' real tokenizers recovers completion tokens well but cannot
recover chat-template overhead, system-prompt tokens, or billing rounding.
Absolute dollar figures are approximate. The naive-versus-true *gap* is
insensitive to this because both estimators consume the same token counts.

**The prices are ours, not theirs.** Magnitudes in dollars depend on the
assumed price vector. The sign flip at the 30%-strong operating point does not:
it follows from the covariance between routing and response length and survives
any price vector in which the strong model is dearer.

**This is not a refutation of RouteLLM.** Their quality-versus-call-fraction
curves are untouched by any of this. Only the translation from call fraction to
cost is affected — which is, however, how the headline "up to 85% cost
reduction" is stated.

**FrugalGPT could not be checked at all.** No author-released code or
per-query cost data exists for Chen et al. (2023). As a cascade its exposure is
structurally greater (cost varies with both length and cascade depth), but that
was not verifiable and was not estimated.

## Stage 10(a) (blocked)

Depends on the Stage 7 frozen test set, of which 0/364 items have complete
generations. Not attempted. The script (`s7_variance.py`) is written and will
run on resumption.

Note for when it does run: with k=3 the between-item variance estimator is
unbiased but not non-negative, so it can come out negative when item difficulty
is indistinguishable from generation noise. Such values are reported raw in the
JSON and clipped to zero in the table, and a clipped value is informative
rather than an error.

## Stage 10(b)(c)(d) (complete)

**The evaluation card documents an evaluation, not a model or dataset**, so it
follows the *spirit* of Model Cards and Datasheets rather than either format
literally. Some standard sections (training data, intended users, ethical
considerations of a deployed artifact) have no clean analogue and are omitted
rather than padded.

**The reproducibility manifest cannot pin what matters most.** Seeds, package
versions, endpoints, item selections and prompts are all pinned. **Provider
model weights are not**, and `gpt-4o` is a moving alias. Nothing in this
project can assert that the weights served for a given slug are stable over
time, which caps achievable reproducibility regardless of local rigour.

**The contribution framing is a draft paragraph, not a claim of acceptance
worthiness.** It is scaffolding, and it concedes the weakest point (no joule
was measured) rather than defending it.

## Carried forward from earlier stages

Still binding on everything here:

1. **Energy is modelled, never measured** — a linear transform of real token
   counts under EcoLogic's own assumed J/1k rates. No power meter, no GPU
   telemetry, no provider disclosure. The ±5× sensitivity sweep bounds
   robustness to the rates; it does not turn the figures into measurements.
2. **Tiers 1 and 2 are substituted models.** The paper's Gemma 3N E4B and
   Apriel 1.6 15B slugs were retired before this evaluation. The routing logic
   is audited; the deployed model stack is not.
3. **Benchmarks are not EcoLogic traffic** — single-turn, self-contained,
   difficulty-concentrated academic tasks, chosen for objective
   auto-gradability and explicitly not claimed representative.
4. **Train/test distribution shift**: MBPP ≠ HumanEval, in the stage that ran
   and in the stage that was blocked.
5. **Prompt-wrapper sensitivity**: the deployed classifier agrees with itself
   only 53.0% of the time between the raw query and the wrapped prompt,
   multiplying energy 5× — so all reported routing behaviour is conditional on
   feeding it the raw query, which is the choice favourable to it.

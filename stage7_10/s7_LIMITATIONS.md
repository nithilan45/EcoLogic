# Limitations — Stages 7–10

Stage 7 is complete, with the pre-registered verdict **partial support,
inconclusive**. The limitations below constrain what that verdict does and does
not establish.

## Stage 7

### The verdict rests on a margin smaller than the evaluation's own noise

The router finished 1.10 pp above Always-Tier-2 on the frozen test set, on 6
discordant items out of 364 (5 to 1). Stage 10(a) puts the router's
sampling+generation half-width at **2.70 pp**, so that margin is *inside* the
noise of a single evaluation run. The "partial support" label is triggered by
the pre-registered ≥1.0 pp narrowing rule and is correctly assigned, but it
should not be read as evidence the router is better — two statistical routes
(McNemar, and the variance decomposition) independently say the comparison is
unresolved at this sample size.

### CALIBRATION and the frozen test set disagree about the router

On the 1,500-item CALIBRATION split the same fixed router is *significantly
worse* than Always-Tier-1 (−1.53 pp, p = 0.021) at 4.8% less energy, and only
+0.47 pp over Always-Tier-2 at 67% more energy. On the 364-item test set it is
1.10 pp above Always-Tier-2. These are not contradictory results — different
splits, and the larger one has ~4× the power to detect a deficit — but the data
does not tell us which reading is right, and the more pessimistic split is the
larger one. Note also that CALIBRATION is not held out in the same sense: the
threshold was selected on it (by the pre-registered rule, but selected on it
nonetheless), so if anything its numbers are mildly *optimistic* for the router.

### `gpt-4o` is not the quality ceiling the design assumes

Always-frontier reaches 89.8% on the test set, **below** Always-Tier-2's 92.6%,
driven by `gpt-4o` scoring lowest of the three tiers on MBPP (86.0% test, 85.3%
pool). Every "quality given up versus the frontier" framing in this project
inherits that, and the cause is not diagnosed here — prompt-wrapper interaction,
verbosity breaking answer extraction, and genuine weakness on this MBPP
formatting are not separated.

### Limitations that apply regardless

**MBPP is exhausted, so the code portion could not be scaled.** This was
pre-registered before the run precisely because it is not fixable. MBPP has 974
problems in total; Stage 1 consumed 400 and 164 were reserved for the new test
set, leaving exactly 410 for the pool — a **1.03× scale-up against 5.7× for
MMLU and GSM8K**. Stage 5's router deviated from static Tier 2 mainly on code,
so the null result is confounded on the code subset specifically by the
inability to add code training data, and the headline "4.2× more data" is uneven
across benchmarks by construction. Any future attempt at this experiment needs a
larger code corpus (e.g. APPS, CodeContests, LiveCodeBench), not more MBPP.

**Pool GSM8K items come from the train split, test items from the test
split.** This guarantees disjointness by construction, which is why it was
chosen, but it introduces a small distribution difference that the Stage 1
design did not have (Stage 1 used disjoint indices within the same split).

**The design cannot separate its two changes.** Stage 7 alters training-set
size *and* label quality simultaneously. The hypothesis as stated
("insufficient/noisy training data") bundles both, so the test is valid for the
bundled claim and uninformative about which half mattered. Because the result is
negative, this matters less than it would have for a positive result: neither
change helped, so there is no effect to attribute.

**The null result does not distinguish the two structural explanations.** "The
gap is structural" could mean the feature representation is inadequate (fixable
with better features) or that per-tier success is largely unpredictable from the
prompt alone (not fixable by any router). The pre-registration says this
explicitly, and the reported result does not separate them. Testing them apart
needs a much stronger feature extractor as a third arm.

**Cross-stage AUC comparisons are not exactly like-for-like.** The headline
"held-out AUC was flat" compares Stage 2's CALIBRATION split to Stage 7's, but
these are different item sets drawn from different pools, and their per-tier base
rates differ materially (Tier 1 0.856 vs 0.921). Base rate alone shifts AUC, so
the comparison supports "no visible improvement" and not a precise zero.

**Generation-order artifact, now fixed but not retroactive.** Items were
enqueued TRAIN-file-first, so the earlier interruption yielded TRAIN items only
and the pre-registered throughput contingency was unusable. The queues are now
interleaved proportionally, but the completed run therefore mixes the two
orderings; since resumption is keyed on `(tier, item_id, sample_idx)` and every
item ultimately completed, this affects only *when* each call was issued.

**Generations span a ~15-hour window across two credit interruptions.** Tier 3
pool generations predate the Together outage; most Tier 1 and 2 generations
postdate it. Provider weights behind `gpt-4o`, `Qwen/Qwen3.5-9B` and
`openai/gpt-oss-20b` carry no immutable revision, so between-tier comparisons on
the same item mix generations from different points in time. Probably small,
genuinely unquantified.

**345 calls were abandoned to a client-side timeout and reissued.** The
per-task ceiling that keeps stalled sockets from halting the run will also
abort a legitimate generation that needs more than 300 s. Tier 1's p90 latency
is ~76 s so this should be rare, but a systematically slow subpopulation (very
long reasoning traces) would be preferentially retried rather than preferentially
dropped — retries did eventually succeed for every pool call, so this is a
latency-sampling caveat, not missing data.

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

## Stage 10(a) (complete)

**k = 3 is a small number of replicates.** With three draws per item the
within-item estimate `p̂(1−p̂)` takes only the values 0 and 2/9, so per-item
generation variance is coarsely quantised; the *aggregate* over 364 items is
still a sound estimate, but no individual item's noise level is well measured.
The between-item estimator is unbiased but not non-negative and can come out
negative where item difficulty is indistinguishable from generation noise; such
values are reported raw in the JSON and clipped to zero in the table, and a
clipped value is informative rather than an error. None were clipped here.

**This measures one provider's nondeterminism on one day.** Temperature-0
variability comes from batching, kernel scheduling, MoE routing and
floating-point reduction order, all of which depend on the provider's serving
conditions at the time. The 19.8% Tier 1 flip rate is a measurement of this run,
not a stable property of the model.

**The decomposition assumes replicates are exchangeable within an item**, i.e.
that the three draws are independent conditional on the prompt. Calls issued
seconds apart against a shared batching queue may be correlated, which would
bias the within-item component **downward** and make the reported generation
noise, if anything, an underestimate.

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
4. **Train/test distribution shift**: MBPP ≠ HumanEval in Stage 1's pool, and
   Stage 7's fresh test set uses MBPP for code while the original frozen test
   set used HumanEval, so the two test sets are not interchangeable.
5. **Prompt-wrapper sensitivity**: the deployed classifier agrees with itself
   only 53.0% of the time between the raw query and the wrapped prompt,
   multiplying energy 5× — so all reported routing behaviour is conditional on
   feeding it the raw query, which is the choice favourable to it.

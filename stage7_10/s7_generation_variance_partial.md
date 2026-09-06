# Stage 10(a) — generation-variance decomposition (partial)

> **Partial.** Tier 3 generations for this test set are incomplete (OpenAI credit limit), so every policy that can escalate to them — and the oracle — is omitted here, as are all router-dependent policies. The latter is deliberate: leaving the frozen test set unscored for the router preserves the pre-registered one-shot evaluation. What is reported below is the part that does not depend on the missing tier, and it is the core Stage 10(a) quantity: how much a temperature-0 rerun moves accuracy.

The Stage 7 frozen test set (n = 364) was regenerated **k = 3 times per item per tier at temperature 0**. Replicates therefore differ only through provider-side nondeterminism, not sampling temperature — which is a distinct quantity from Stage 7's temperature-0.7 training samples.

## How much does a temperature-0 rerun actually change?

| Tier | Mean accuracy over k=3 | Items that flipped across replicates | Mean within-item token SD | Items with byte-identical token counts |
|---|---|---|---|---|
| Tier 1 | 83.2% | 72 (19.8%) | 1936.4 tokens | 6.3% |
| Tier 2 | 91.5% | 21 (5.8%) | 263.4 tokens | 7.1% |

A flipped item is one where the same tier, on the same prompt, at temperature 0, graded correct on some replicates and incorrect on others. Temperature 0 is not determinism.

The spread across tiers is the part worth noting. Tier 1 changes its graded verdict on **19.8%** of items between identical temperature-0 calls, with output length moving by 1936 tokens on average, and only 6.3% of items return the same token count twice. Nondeterminism of that size is not a rounding detail: it means a single-run accuracy figure for this tier is reproducible only to within a few points, and any two systems being compared through it need paired generations rather than independently-run numbers.

## Variance decomposition per policy

| Policy | Acc (k=3 mean) | between-item var | within-item var | within share | sampling-only ± | sampling+generation ± | Wilson ± (reported) |
|---|---|---|---|---|---|---|---|
| Always Tier 1 | 83.2% | 0.07444 | 0.06593 | 47.0% | 2.80 pp | 3.85 pp | 3.81 pp |
| Always Tier 2 | 91.5% | 0.05886 | 0.01923 | 24.6% | 2.49 pp | 2.87 pp | 2.71 pp |

## Reading this table

**The Wilson intervals in every table of this project are not too narrow.** That is the first thing to establish, because it is the natural worry. A single-generation evaluation draws one `Y_i` per item, and that draw already contains the generation noise, so `p(1-p)/n` is the right total variance for it. The Wilson column matches the sampling+generation column closely, which is the arithmetic confirming this.

**What the decomposition does show** is that a Wilson interval is routinely *read* as the wrong thing. Readers treat it as the uncertainty in the benchmark's verdict about these models — as if rerunning generation would land inside the interval and only a different item sample would move it. The 'within share' column says how much of the interval is instead pure regeneration noise that no amount of item sampling would reduce, and that rerunning the identical evaluation would reproduce differently.

**Consequence for tight comparisons.** Two policies whose accuracies differ by less than the sampling+generation half-width cannot be separated by a single run, however many items are used, unless the comparison is paired on identical generations — which is why the McNemar tests in this project are all paired on the same response set, and why the pre-registered verdict rests on a paired test rather than on overlapping intervals.

Negative raw between-item variance estimates (possible because the estimator is unbiased, not non-negative) are reported in the JSON and clipped to zero in the table; a clipped value means item difficulty is indistinguishable from pure generation noise at this sample size.

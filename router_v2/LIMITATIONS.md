# Limitations — learned router (Addendum 3)

Written to be read *before* the result is quoted anywhere. The headline outcome
was a pre-registered negative (neither S1 nor S2 met), and several of the items
below cut against the router; a couple cut in its favour. Both directions are
listed.

## The training pool is a different distribution from the frozen test set

This is the single largest threat to the result. All 164 HumanEval problems are
inside the frozen test set, so none could be used for training without
contaminating it. **MBPP substitutes for HumanEval in the training pool**, and
MBPP is not HumanEval: its problems are shorter, more formulaic, phrased as
one-line natural-language tasks with the assertions supplied, and drawn from a
different authoring process. The router therefore learned "what a code prompt
that Tier N will get right looks like" from MBPP and was asked to generalise to
HumanEval.

The same substitution applies less severely to the other two benchmarks. MMLU
training items come from the **47 subjects the frozen selection did not use**, so
the router never saw a training example from any of the 10 test subjects —
subject-level generalisation is being tested, not memorised. GSM8K training
items are from the same distribution as the test items (disjoint indices in the
same split), so GSM8K is the only benchmark where train and test are genuinely
exchangeable.

Concretely: the router's largest test-set deviation from static Tier 2 was on
code, and code is exactly where the train/test distribution shift is largest.
Any conclusion about *why* the router failed to beat static assignment is
confounded by this.

## Oracle training labels carry the same single-sample noise as before

Labels are derived from one generation per item per tier at temperature 0. The
previous report established that these providers are not reproducible at
temperature 0 (byte-identical text on only 9/21 Tier 1 repeats, token counts
moving tens of percent), so a re-run would relabel some items. An item where
Tier 2 happened to succeed and Tier 1 happened to fail gets a label that a
second run might reverse. The router is fitting partly to generation noise.

**58 of 1,200 pool items (4.8%) had no correct answer from any tier.** These are
flagged `no_tier_correct: true` in `pool_labels.json` and, per the
pre-registration, were labelled with the lowest-energy tier. That label is a
convention, not ground truth — there is no "right" tier for an item nobody got
right — so roughly 5% of the training signal is definitionally arbitrary.

## ~1,200 items is small for a learned classifier

840 TRAIN items split across three benchmarks, with two binary heads fitted on
top, is a small sample for text classification. The per-head base rates are
also badly imbalanced (Tier 1 correct 83.3%, Tier 2 correct 90.5% on TRAIN), so
each head has only ~80–140 negative examples to learn from — and the negatives
are the informative class, since the router's entire job is anticipating
failure. `class_weight="balanced"` compensates for the imbalance in fitting but
cannot manufacture information that is not there.

The discriminative signal that resulted is weak and uneven. On CALIBRATION the
heads reach AUC 0.66 (Tier 1) and 0.70 (Tier 2) for the selected variant. AUC in
the high 0.6s is real signal, not noise, but it is nowhere near strong enough to
beat a static policy that is already right ~92% of the time. A router only earns
its keep on items where tiers disagree, and there were few of those.

## The Stage 3 threshold rule was one reasonable choice among several

The pre-registered rule — highest calibration accuracy subject to energy <= 10%
of always-frontier — is defensible but arbitrary in two ways, and both mattered.

**The energy cap was binding and cost accuracy.** The unconstrained best
threshold on CALIBRATION was tau = 0.510 at 91.4% accuracy and 13.2% of frontier
energy. The pre-registered cap forced tau = 0.2245 at 90.6% and 9.0%. So the
rule gave up **0.83 pp of calibration accuracy** to honour a budget I picked in
advance. A rule with X = 15% would have selected a different, more accurate
threshold. I did not change it after seeing this, which is the point of
pre-registering, but the number reported is a consequence of that choice.

**X = 10% turned out to be nearly infeasible.** Always-Tier-1 alone costs 771 J
on CALIBRATION against a 516 J budget, so the constraint was only satisfiable at
all by routing most traffic to Tier 2. The budget I chose in advance, reasoning
from the keyword router's 11.4% operating point, was tighter than I realised
relative to Tier 1's token appetite.

## The routing rule's candidate ordering was corrected mid-experiment

My first implementation was a cascade that tested Tier 1 before Tier 2. That is
structurally wrong here: Tier 2 costs *less* energy than Tier 1 (383 J vs 771 J
on CALIBRATION) despite a 3x higher per-token rate, because Tier 1 emits ~6x
more tokens. A Tier-1-first cascade can therefore never reach a policy cheaper
than always-Tier-1, and **no threshold in its entire sweep satisfied the
pre-registered budget**.

I replaced it with a cost-ordered rule (cheapest expected energy first, ordering
derived from TRAIN-split means only) and re-ran Stage 3. Routing-rule structure
was not part of the pre-registration and the frozen test set was untouched at
that point, so this is a legitimate calibration-stage fix rather than a
post-hoc adjustment — but it *is* a change made after seeing calibration
results, and a stricter protocol would have fixed the rule in advance too. The
discarded index-ordered sweep is retained in `threshold_sweep.csv` under
`rule = index_ordered` and plotted as the grey ablation curve.

## The selection procedure picked the variant that is worse on CALIBRATION

Per the pre-registration, the variant was chosen by cross-validation **within
TRAIN**, where R2 (embeddings) beat R1 (TF-IDF), 0.6746 vs 0.6550 mean AUC. On
CALIBRATION the ordering reverses: R1 averages 0.7028 across the two heads
against R2's 0.6784, driven by a clearly better Tier 1 head (0.7148 vs 0.6558).

So the honest reading is that R1 and R2 are indistinguishable at this sample
size and the TRAIN-CV comparison was decided by noise. I carried R2 forward
because that is what the pre-registered procedure dictated, not because the
evidence favours it. Had R1 been selected, the test-set number would differ by
an unknown amount. This is a direct cost of selecting on 840 items.

## Single run, and the energy figures inherit every earlier caveat

Every limitation from the previous report still applies and is not repeated in
full here: energy is **modeled, not measured** (`tokens/1000 * an assumed
constant`, with the savings conclusion flipping sign in 3 of 27 perturbation
combinations); the benchmarks are old, public and near-certainly in training
data; and each item is a single generation, so the Wilson intervals capture
sampling error over items only and not generation variance.

Two additions specific to this stage. First, the pool run is also a single
sample, so both the labels and the per-tier energy statistics that define the
cost ordering carry run-to-run noise. Second, the test-set comparison against
Always-Tier-2 rests on **one discordant item** (McNemar b=0, c=1): the router
and static Tier 2 disagreed on essentially nothing. `p = 1` here means "these
two policies are indistinguishable on this test set", not "they are equivalent
in general" — a larger or harder test set could separate them in either
direction, and the 364-item set is simply not powerful enough to resolve a
difference this small.

## What the negative result does and does not license

It supports: *on this workload, with this training pool and this label quality,
a learned router did not surpass static Tier-2 assignment.*

It does not support: *learned routing does not work.* The oracle reaches 96.4%
against static Tier 2's 92.3%, so 4.1 pp of headroom demonstrably exists on the
test set, and the LP frontier analysis shows the discreteness gap is ~0 — the
headroom is capturable in principle by a better-calibrated router, not blocked
by the structure of the problem. What this experiment shows is that ~1,200
noisily-labelled items across a shifted distribution are not enough to capture
it. It also does not license any conclusion about the paper's actual Tier 1/2
models, which are retired and were substituted throughout.

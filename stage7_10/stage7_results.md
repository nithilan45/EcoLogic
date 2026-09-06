# Stage 7 — hypothesis not supported on the evidence now in; pre-registered test-set verdict still pending 1,091 calls

## Verdict, in one sentence

**The calibration-gap hypothesis is not supported by the evidence now in
hand**: scaling the training pool 4.2× (1,200 → 5,000 items) and replacing
single-sample labels with majority-of-k=3 labels moved the calibration gap from
Stage 6's **+5.83 pp** to **+5.00 pp**, closing 0.83 pp of it, while held-out
per-tier head AUCs were flat (Tier 1 0.656 → 0.665, Tier 2 0.701 → 0.702).

**That sentence is not yet the pre-registered S1/S2 verdict.** S1 and S2 are
defined on the *new frozen test set*, and 1,091 of its 1,092 Tier 3 (`gpt-4o`)
generations are missing because the OpenAI account hit its credit limit. The
formal verdict is therefore withheld, not guessed. Finishing it costs **$1.91**
of OpenAI credit (measured, not estimated — see [Remaining
work](#remaining-work-1091-calls-191)).

What separates the two statements: the hypothesis is about the **calibration
gap**, which is defined and measured on the CALIBRATION split, and that split is
100% generated. S1/S2 are about beating **Always-Tier-2 on a held-out test set**,
which needs the third tier for the always-frontier and oracle rows.

---

## Status of the generation run

Pool generation completed in full after the Together AI credits were topped up.
Resumption was keyed on `(tier, item_id, sample_idx)`, so no completed call was
repeated.

| Target | Calls needed | Succeeded | Share |
|---|---|---|---|
| **Pool** — Tier 1 (`Qwen/Qwen3.5-9B`) | 15,000 | **15,000** | 100% |
| **Pool** — Tier 2 (`openai/gpt-oss-20b`) | 15,000 | **15,000** | 100% |
| **Pool** — Tier 3 (`gpt-4o`) | 15,000 | **15,000** | 100% |
| **Test** — Tier 1 | 1,092 | **1,092** | 100% |
| **Test** — Tier 2 | 1,092 | **1,092** | 100% |
| **Test** — Tier 3 | 1,092 | 1 | **0.1%** |
| **Total** | 48,276 | 47,185 | 97.7% |

All 5,000 pool items have complete k=3 × 3-tier generations, split 3,500 TRAIN /
1,500 CALIBRATION exactly as pre-registered. Nothing was excluded for
incompleteness.

**Spend so far: $41.11** ($39.57 pool + $1.53 test), against the pre-registered
$150 gate and the pilot's $47.47 projection. Full per-tier accounting is in
`s7_run_accounting.json`.

### Failed attempts, and why they are not data loss

17,109 call attempts failed and were retried: 15,330 HTTP 402 (the credit
exhaustion, before and after top-up), 1,416 HTTP 429 (Together rate limiting),
18 HTTP 503, and 345 client-side task timeouts. None are billed and none appear
in the analysis; every one was retried to success except the 1,091 Tier 3
test-set calls. Three infrastructure fixes were needed to converge, all
committed and all affecting only *how* calls are issued, never what is asked or
graded:

- a hard per-task timeout, because `api.py`'s 5 retries at a 300 s request
  timeout let a single stalled socket pin a worker for ~25 minutes, and with 80
  workers the run silently flatlined at 0 calls/min;
- phase-granular HTTP timeouts (connect 30 s, read 300 s) so a stuck connect
  fails fast while a legitimate 16,384-token generation still has ~190 s to run;
- 15-second keepalive expiry plus a 25-minute cap per pass, because throughput
  started near 200 calls/min on a fresh connection pool and decayed toward zero
  as sockets went stale.

A fourth fix was to the enqueue order: TRAIN was previously generated before
CALIBRATION, which is why the earlier interruption left zero CALIBRATION items.
Splits are now interleaved proportionally, so any prefix is ~70/30. This makes
the pre-registered throughput contingency actually usable; it was not usable
before.

---

## The label-noise fix, and how much noise it removed

k=3 sampling at temperature 0.7 found genuine per-item disagreement, so the
majority-vote label is doing real work rather than averaging identical repeats:

| Tier | Items with a 2–1 sample split | Share of 5,000 |
|---|---|---|
| Tier 1 (`Qwen/Qwen3.5-9B`) | 290 | **5.8%** |
| Tier 2 (`openai/gpt-oss-20b`) | 402 | **8.0%** |
| Tier 3 (`gpt-4o`) | 201 | **4.0%** |

192 items (3.8%) had no tier whose majority grade was correct; per the
pre-registration these are labelled with the lowest-energy tier and flagged as
label noise rather than treated as ground truth. Oracle label mix: Tier 2 =
3,428, Tier 1 = 1,497, Tier 3 = 75.

So Stage 1's single-sample labels were wrong-by-coin-flip on roughly 6–8% of
per-tier judgements for the two cheap tiers, and Stage 7 removes that. This is
the "cleaned" half of "scaled and cleaned" actually landing.

---

## Did more, cleaner data help? Two measurements say mostly no

### 1. Held-out discrimination is flat

The per-tier heads are the mechanism the whole approach rests on: if they cannot
tell which queries a tier will get right, no threshold can route well.

| | Stage 2 (1,200 items, single-sample labels) | Stage 7 (5,000 items, k=3 majority labels) |
|---|---|---|
| TRAIN-CV mean AUC, R1 | 0.6550 | **0.7154** |
| TRAIN-CV mean AUC, R2 | 0.6746 | **0.7162** |
| CALIBRATION AUC, Tier 1 head (R2) | 0.6558 | **0.6650** |
| CALIBRATION AUC, Tier 2 head (R2) | 0.7010 | **0.7015** |

The TRAIN-CV numbers rose by ~0.04, which looks like the hypothesis being
confirmed — but the held-out CALIBRATION numbers moved by **+0.009 and
+0.0005**. The cross-validation gain is largely the labels becoming easier to
predict once denoised, not the heads generalising better. Both remain far below
the ~0.85 that a router would need to beat a static policy already correct ~92%
of the time.

Two caveats on that comparison, both real: the Stage 2 and Stage 7 CALIBRATION
splits are *different item sets* from different pools, and their base rates
differ (Tier 1 0.856 vs 0.921), which alone shifts AUC. So read the flatness as
"no visible improvement" rather than as a precise zero.

R2 (MiniLM embeddings) won TRAIN-CV again, 0.7162 vs 0.7154 — a 0.0008 margin,
i.e. the richer local representation is not meaningfully better than TF-IDF
here. Full grids in `s7_model_comparison.md`.

### 2. The calibration gap barely moved

Recomputed on the new 1,500-item CALIBRATION split by the same code path:

| Quantity | Stage 6 (n=360) | Stage 7 (n=1,500) |
|---|---|---|
| Router accuracy | — | 0.9060 |
| LP-relaxed frontier at the router's energy | — | 0.9560 |
| **Calibration gap** | **+5.83 pp** | **+5.00 pp** |
| **Discreteness gap** | +0.00 pp | **+0.00 pp** |

The gap closed by 0.83 pp for a 4.2× increase in data plus a label-noise fix.
The discreteness gap stayed at zero, so the headroom is still not blocked by
having to pick one tier per item — the LP relaxation and the integer MCKP agree
to within 0.04 pp at every budget on the frontier curve.

Per the pre-registration's framing, this points at the gap being **structural**:
a limit of the feature representation, or of how predictable per-item tier
success is from the prompt alone. This design deliberately does not distinguish
those two, and I am not claiming it does.

---

## CALIBRATION-split policy comparison

Reported because this split's data is complete. **This is not the frozen test
set and not the pre-registered verdict.** No policy, variant or threshold was
selected using this table: the variant was fixed by TRAIN-only CV and the
threshold τ = 0.2653 by the pre-registered ≤10%-of-frontier budget rule
(14 of 50 grid thresholds qualified; the rule applied as written, no fallback).

n = 1,500. Energy at the paper's J/1k-token rates.

| Policy | Accuracy | 95% Wilson CI | Energy (J) | % of frontier | Tier mix 1/2/3 |
|---|---|---|---|---|---|
| Always Tier 1 | 0.9213 | [0.9066, 0.9339] | 2,444.5 | 10.4% | 1500/0/0 |
| Always Tier 2 | 0.9013 | [0.8852, 0.9154] | 1,394.9 | 5.9% | 0/1500/0 |
| Always Tier 3 (frontier) | 0.9060 | [0.8902, 0.9197] | 23,613.6 | 100.0% | 0/0/1500 |
| Random | 0.9127 | [0.8973, 0.9259] | 9,252.5 | 39.2% | 486/500/514 |
| Oracle | 0.9560 | [0.9444, 0.9653] | 1,515.2 | 6.4% | 420/1063/17 |
| **Learned router (Stage 7)** | **0.9060** | [0.8902, 0.9197] | **2,326.0** | **9.9%** | 24/1423/53 |
| EcoLogic keyword router | 0.9113 | — | 2,757.2 | 11.7% | 1087/354/59 |

McNemar exact, learned router vs each row:

| Comparison | Δ accuracy | b / c | p | Δ energy |
|---|---|---|---|---|
| vs Always Tier 1 | **−1.53 pp** | 34 / 57 | **0.021** | −4.8% |
| vs Always Tier 2 | +0.47 pp | 13 / 6 | 0.167 | **+66.8%** |
| vs Always Tier 3 | +0.00 pp | 57 / 57 | 1.00 | −90.1% |
| vs Random | −0.67 pp | 35 / 45 | 0.314 | −74.9% |
| vs Oracle | −5.00 pp | 0 / 75 | 5.3e−23 | +53.5% |

Read plainly: on this split the router is **significantly worse than simply
always using Tier 1** (−1.53 pp, p = 0.021) while saving only 4.8% of energy,
so Always-Tier-1 dominates it outright. Against Always-Tier-2 it is +0.47 pp,
not significant, and spends **67% more energy** — so on this split neither S1
(needs p < 0.05) nor S2 (needs ≥15% *less* energy) would hold. Whether that
carries to the frozen test set is exactly what the missing 1,091 calls decide,
and I am not asserting it in advance.

One structural note worth flagging: on this pool Tier 1 is *more accurate* than
Tier 2 (0.9213 vs 0.9013) but also *more energy-hungry* (2,444 J vs 1,395 J),
because Tier 1 emits ~3,100 output tokens per call against Tier 2's ~427. The
cost ordering that the routing rule derives from TRAIN means is therefore
Tier 2 → Tier 1 → Tier 3, and the router sends 1,423 of 1,500 items to Tier 2.
It is, in effect, an expensive approximation of Always-Tier-2 plus 53 escalations.

---

## Remaining work: 1,091 calls, $1.91

Blocked by OpenAI credit exhaustion on `gpt-4o`:

```
FATAL: openai/gpt-4o HTTP 429 out of credit
```

The cost is measured from the 15,000 completed Tier 3 pool calls, not estimated
from list prices: $26.19 / 15,000 = **$0.00175 per call** at a measured 120
input / 145 output tokens, so 1,091 calls = **$1.91**. Together AI is funded and
unaffected; Tiers 1 and 2 of the test set are already complete.

Once credit is added:

```bash
python3 stage7_10/s7_run.py --target test    # resumes; issues only the 1,091 Tier 3 calls
python3 stage7_10/s7_grade.py --target test
python3 stage7_10/s7_final.py                # eight-policy one-shot table + S1/S2 verdict
python3 stage7_10/s7_variance.py             # Stage 10(a) variance decomposition
python3 stage7_10/s7_export_samples.py
```

Nothing needs re-selecting or re-seeding. The router weights
(`s7_router_model.pkl`), the variant, the threshold τ = 0.2653, the pools, the
test set and the seed (`20260906`) are all fixed on disk and committed, so the
run that finishes is the one that was pre-registered. The one-shot property is
intact: the frozen test set has not been scored.

### Resumption fidelity, for the record

Tier 3 pool generations were made before the Together interruption; Tier 1 and 2
generations for most items were made after it, across roughly a 15-hour window.
Provider weights behind `gpt-4o`, `Qwen/Qwen3.5-9B` and `openai/gpt-oss-20b` are
not version-pinned (`reproducibility_manifest.md` §3), so this run mixes
generations from different points in time. For between-tier comparisons on the
same item that is a real, probably small, confound, and it should be disclosed
if these results are published.

---

## Per-sample accuracy on the pool, for reference

All k=3 samples pooled at temperature 0.7, so **not** comparable to the
temperature-0 headline numbers elsewhere in this project.

| Tier | GSM8K | MBPP | MMLU |
|---|---|---|---|
| Tier 1 | 96.2% (6625/6885) | 91.7% (1128/1230) | 87.6% (6032/6885) |
| Tier 2 | 95.1% (6546/6885) | 95.4% (1173/1230) | 84.4% (5810/6885) |
| Tier 3 | 95.9% (6603/6885) | 85.3% (1049/1230) | 86.1% (5929/6885) |

Tier 3 (`gpt-4o`) scoring **lowest of the three on MBPP** (85.3%) is the clearest
anomaly in this table and it reproduced at full pool size, so it is not a
small-sample artifact. It is reported, not explained; the plausible causes
(prompt-wrapper interaction, verbosity that breaks the extraction, genuine
weakness on this MBPP formatting) are not separated here, and it directly
weakens the "Tier 3 is the quality ceiling" assumption that the always-frontier
baseline encodes.

## What the MBPP ceiling means for this stage

Restating the pre-registered caveat now that the numbers exist: MBPP has 974
problems total, Stage 1 consumed 400 and the new test set reserves 164, leaving
**410** for the pool. So GSM8K and MMLU scaled 5.7× while code scaled **1.03×**.
Stage 5's router deviated from static Tier 2 mainly on code. Any null result on
the code subset is therefore confounded with the inability to add code training
data, and the "4.2× more data" claim is uneven across benchmarks by
construction.

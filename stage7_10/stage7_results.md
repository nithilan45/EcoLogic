# Stage 7 — BLOCKED, no verdict issued

## Verdict

**No verdict. Stage 7 is blocked and the pre-registered hypothesis remains
untested**: the Together AI account ran out of credits partway through
generation (HTTP 402 `credit_limit`), leaving 1,039 of 5,000 pool items fully
generated — *fewer* than Stage 1's 1,200 — so the "4–5× more training data"
premise the stage exists to test is unattainable from the collected data, and
zero CALIBRATION-split items completed, which makes the pre-registered
threshold-selection step structurally impossible.

Per the addendum's instruction, this stage stops here. What follows is exactly
what was collected, what it cost, why the salvage does not constitute a reduced
version of the experiment, and what is needed to resume. **No accuracy table,
no S1/S2 determination and no calibration-gap number is reported for Stage 7,
because none can be computed honestly from this data.**

Stages 8, 9 and 10(b)(c)(d) do not depend on Stage 7 and are complete. Stage
10(a) depends on the Stage 7 test set and is blocked with it.

---

## What blocked it

```
HTTP 402: {"error": {"message": "Credit limit exceeded, please add credits.
           If you've already made a payment, please wait up to 5 minutes for
           balances to update and try again.", "type": "credit_limit"}}
```

Confirmed persistent, not transient: after halting all workers, a single
minimal call (`openai/gpt-oss-20b`, 5 max tokens) still returned HTTP 402.
Tiers 1 and 2 are both Together-hosted, so the failure removes two of the three
tiers. Tier 3 (`gpt-4o`, OpenAI) was unaffected and finished completely.

This is an account-funding limit, not a rate limit, a bug, or a
misconfiguration. It cannot be worked around from inside the run.

## What was collected before the limit hit

| | Target | Succeeded | Share |
|---|---|---|---|
| **Pool**, Tier 3 (`gpt-4o`, OpenAI) | 15,000 | **15,000** | 100% |
| **Pool**, Tier 2 (`openai/gpt-oss-20b`, Together) | 15,000 | 3,512 | 23.4% |
| **Pool**, Tier 1 (`Qwen/Qwen3.5-9B`, Together) | 15,000 | 3,374 | 22.5% |
| **Test set**, all tiers | 3,276 | 220 | 6.7% |
| **Total** | 48,276 | **22,106** | 45.8% |

13,383 calls returned HTTP 402 and 1,920 more on the test-set run. Failed calls
are recorded with their error and are not counted as data.

Items with **all 3 tiers × all 3 samples** present — the only items usable
under the pre-registered label rule:

| | Count |
|---|---|
| Pool items fully generated | **1,039** of 5,000 |
| ...from the TRAIN split | 1,039 |
| ...from the CALIBRATION split | **0** |
| Test-set items fully generated | **0** of 364 |

## Cost

| Provider | Successful calls | Spend |
|---|---|---|
| OpenAI (Tier 3) | 15,001 | $26.1903 |
| Together AI (Tiers 1–2) | 7,105 | $3.1156 |
| **Stage 7 total** | **22,106** | **$29.3059** |

The pilot projected **$47.47** for the full 48,276 calls, against a
pre-registered gate of $150. That projection was accurate — the run was not
stopped by cost overrun. Roughly **$13 of further spend** would finish it
(~23,100 remaining Together calls at the measured $0.000439/call ≈ $10.1, plus
1,091 OpenAI test-set calls ≈ $1.9), bringing Stage 7 to about **$42**, still
well under the gate.

## Why the salvaged data is not a reduced Stage 7

Three independent reasons, any one of which is sufficient:

1. **The premise is unattainable.** Stage 7 exists to test whether the Stage 6
   calibration gap is caused by insufficient training data. 1,039 complete
   items is **0.87× Stage 1's 1,200**, not 4.2×. Fitting a router on it would
   test *nothing about data scaling*; at best it would test the label-noise fix
   alone, at slightly less data than before — a different experiment that was
   not pre-registered and that cannot separate the two changes.
2. **There is no CALIBRATION split.** Every completed item is from TRAIN,
   because the generator processes the TRAIN file before the CALIBRATION file.
   The pre-registered threshold rule ("highest CALIBRATION accuracy among
   thresholds using ≤ 10% of always-frontier energy **on the same CALIBRATION
   split**") has no data to run on. Selecting a threshold on TRAIN instead
   would be exactly the leak the pre-registration forbids.
3. **There is no test set.** Zero of 364 new frozen-test items have complete
   generations, so the one-shot evaluation cannot be run at all. Substituting
   the Stage 5 test set is explicitly ruled out by this addendum ("do not touch
   the original Stage 5 test set again").

Refitting anyway and reporting a table would produce numbers that look like the
pre-registered experiment while answering a different question with a
train-set-selected threshold. That is precisely the failure the
pre-registration was written to prevent, so it was not done.

---

## One measurement that does survive, clearly labelled

The following is a **partial diagnostic from the incomplete pool**, not the
pre-registered experiment. It is reported because it is well-defined on the data
that exists, it required no threshold and no test set, and it bears directly on
whether the label-noise fix could plausibly have mattered.

**How much label noise did k=3 majority voting actually remove?** Over the 1,039
complete TRAIN items, generating 3 independent samples per tier at temperature
0.7 and asking how often the three samples disagreed:

| Tier | Items with a 2–1 sample split | Share |
|---|---|---|
| Tier 1 (`Qwen/Qwen3.5-9B`) | 74 / 1,039 | **7.1%** |
| Tier 2 (`openai/gpt-oss-20b`) | 85 / 1,039 | **8.2%** |
| Tier 3 (`gpt-4o`) | 42 / 1,039 | **4.0%** |

Items where no tier's majority grade was correct: 33 / 1,039 (3.2%), flagged as
label noise rather than treated as ground truth. Oracle label mix over the
complete items: Tier 2 = 692, Tier 1 = 328, Tier 3 = 19.

Read carefully, this bounds the upside the label fix was ever going to deliver.
A 2–1 split marks an item whose single-sample Stage 1 label was close to a coin
flip; roughly **7–8% of per-tier labels for the two cheap tiers were in that
category**, and majority voting fixes those in expectation. Stage 1's
CALIBRATION head AUCs were 0.66 and 0.70. Removing noise from ~8% of labels is
a real improvement in label quality, but it is not obviously enough to move
head AUCs to where a router would need them (roughly 0.85+) to beat a static
policy that is already right ~92% of the time. **This is a suggestive
observation, not a result** — it does not test the hypothesis, and the
pre-registered outcome remains "untested", not "not supported".

Per-sample accuracy on the complete items, for reference (all k samples pooled,
temperature 0.7, so these are *not* comparable to the temperature-0 headline
numbers elsewhere in this project):

| Tier | GSM8K | MBPP | MMLU |
|---|---|---|---|
| Tier 1 | 96.9% (1492/1539) | 94.5% (309/327) | 87.3% (1317/1508) |
| Tier 2 | 95.7% (1522/1590) | 97.1% (334/344) | 85.6% (1350/1578) |
| Tier 3 | 95.9% (6603/6885) | 85.3% (1049/1230) | 86.1% (5929/6885) |

The Tier 3 MBPP figure (85.3%, below both cheaper tiers) is unexpected and is
noted rather than explained; with the pool incomplete it is not worth
investigating here, since the code subset is small and the comparison is
confounded by which items happened to complete.

---

## What was completed successfully before the block

These parts of Stage 7 are done and are not affected by the credit limit:

- **Pool and test-set construction**, with all five disjointness assertions
  printing PASS (`build_pools.py`): a 5,000-item pool (MBPP 410, MMLU 2,295,
  GSM8K-train 2,295; 3,500 TRAIN / 1,500 CALIBRATION) and a fresh 364-item test
  set (MBPP 164, MMLU 100, GSM8K 100), with zero item-ID overlap against the
  original frozen test set, Stage 1's pool, and each other, and no HumanEval
  item reused anywhere.
- **Cost pilot**, projecting $47.47 against the $150 gate.
- **The full analysis pipeline**, written and exercised end-to-end on the
  partial data: `s7_grade.py` (k=3 majority labels), `s7_fit.py` (R1/R2 refit
  with TRAIN-only CV), `s7_calibrate.py` (threshold sweep + LP/integer MCKP
  frontier + calibration and discreteness gaps), `s7_final.py` (eight-policy
  one-shot table + verdict logic), `s7_variance.py` (Stage 10a). Grading runs
  clean on 21,886 responses.
- **The pre-registration** (`prereg_stage7.md`), committed at `97a08cb` before
  any data existed, including the MBPP scaling ceiling that would have
  confounded the code result even had the run finished.

## To resume

Add credits to the Together AI account, then:

```bash
python3 stage7_10/s7_run.py --target pool   # resumes; skips the 22k completed calls
python3 stage7_10/s7_run.py --target test
python3 stage7_10/s7_grade.py --target pool
python3 stage7_10/s7_grade.py --target test
python3 stage7_10/s7_fit.py
python3 stage7_10/s7_calibrate.py
python3 stage7_10/s7_final.py
python3 stage7_10/s7_variance.py           # Stage 10(a)
```

Resumption is keyed on `(tier, item_id, sample_idx)` against the existing
`s7_pool_responses.jsonl`, so no completed call is repeated and none of the
$29.31 already spent is wasted. Expected additional cost ≈ $13; expected wall
time ≈ 1.5 h at the observed 341 Together calls/min. Nothing needs re-selecting
or re-seeding: the pools, the test set and the seed (`20260906`) are already
fixed on disk, so the experiment that resumes is the one that was
pre-registered.

**One caveat on resumption fidelity, stated for the record.** The Tier 3 pool
generations were made before the block and the Tier 1/2 generations for most
items would be made after it. Provider model weights behind `gpt-4o`,
`Qwen/Qwen3.5-9B` and `openai/gpt-oss-20b` are not version-pinned (see
`reproducibility_manifest.md` §3), so a resumed run mixes generations from two
points in time. For a comparison *between tiers on the same item* this is a
real, if probably small, confound, and it should be disclosed if the resumed
results are published.

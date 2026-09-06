# Learned router (Addendum 3)

A new artifact alongside the previous audit, not a rewrite of it. Nothing in
`raw_results/` or `results_report.md` was modified; those stand as the "before"
baseline this compares against.

## Verdict

**Neither pre-registered success criterion was met: a properly trained router
still does not surpass static assignment on this workload.** On the frozen
364-item test set the learned router scored **92.0%** against Always-Tier-2's
**92.3%** (McNemar exact p = 1, one discordant item) while using **18.9% more
energy**. S1 required a significant accuracy win; S2 required parity at >= 15%
lower energy.

The router did, however, decisively beat the keyword classifier it replaces:
**86.8% -> 92.0% accuracy (+5.2 pp, McNemar p = 0.00055) at 32.0% less energy.**
It got there by learning, in effect, "always use Tier 2" — its test-set tier mix
is 19/340/5 — and then paying 18.9% extra energy for the 24 deviations, which
gained nothing.

## Stage results

| Stage | What happened | Artifact |
|---|---|---|
| Pre-registration | Success criteria and threshold rule (X = 10%) fixed and committed before any run | [`PREREGISTRATION.md`](PREREGISTRATION.md) |
| 1 — pool | 1,200 items (GSM8K 400, MMLU 400 from 47 unused subjects, MBPP 400) — overlap check **PASS**, 3,600 calls, 0 errors, $3.47 | [`train_pool.json`](train_pool.json), [`calibration_pool.json`](calibration_pool.json), `pool_graded.jsonl`, `pool_labels.json` |
| 2 — router | R2 (MiniLM embeddings) beat R1 (TF-IDF) on TRAIN CV, 0.6746 vs 0.6550 mean AUC | [`model_comparison.md`](model_comparison.md) |
| 3 — threshold | tau = 0.2245 by the pre-registered rule; 17/50 thresholds were within budget | [`threshold_sweep.csv`](threshold_sweep.csv), `threshold_sweep.png` |
| 4 — MCKP | calibration gap **+5.83 pp**, discreteness gap **~0 pp**, regret cross-check matched to 8e-17 | `mckp_frontier_v2.png`, [`regret_verification_v2.md`](regret_verification_v2.md), `mckp_gaps.json` |
| 5 — one shot | Verdict **NEITHER** | [`final_test_set_results.md`](final_test_set_results.md) |
| 6 — honesty | Distribution shift, label noise, sample size, threshold-rule arbitrariness, a mid-experiment rule fix | [`LIMITATIONS.md`](LIMITATIONS.md) |

## Two findings worth more than the verdict

**The gap is the router's, not the problem's.** At matched energy the LP-relaxed
MCKP frontier reaches 96.4% where the router reaches 90.6% — a **5.83 pp
calibration gap**. The **discreteness gap is ~0** (0.00 pp at both matched
points; 0.009 pp mean across the budget sweep), so being forced to pick one tier
per item costs essentially nothing. The headroom is real and capturable in
principle; the learned probabilities just are not sharp enough to capture it.

**The confusion-matrix regret formula flips sign against reality.** Using
per-tier mean energies, both computations agree exactly (−0.2496 J/item, matched
to 8e-17), suggesting the router *saves* energy versus the oracle. Using each
item's actual energy gives **+0.2840 J/item** — the router in fact spends more.
The identity is exact only when energy is a per-tier constant, and it is not:
tier choice correlates with item token count. The tidy formula and the truth
have opposite signs here.

## Reproducing

```bash
pip install httpx pyarrow scipy scikit-learn matplotlib sentence-transformers \
            fastapi pydantic python-dotenv
export TOGETHER_API_KEY=...   # Tiers 1 and 2
export OPENAI_API_KEY=...     # Tier 3

python3 router_v2/build_train_pool.py   # asserts disjointness from the frozen set
python3 router_v2/run_pool.py --pilot 4 # cost projection
python3 router_v2/run_pool.py           # 3,600 calls, temperature 0, resumable
python3 router_v2/grade_pool.py         # executes MBPP; writes oracle labels
python3 router_v2/train_router.py       # R1 vs R2, CV within TRAIN only
python3 router_v2/calibrate.py          # threshold sweep on CALIBRATION
python3 router_v2/mckp.py               # LP frontier, gaps, regret
python3 router_v2/final_test.py         # one shot at the frozen test set
```

Total measured API cost for this addendum: **$3.5143** ($3.4712 pool + $0.0431
pilot), 3,636 calls, 0 errors. Stage 5 adds no API calls: it reuses the
per-tier generations already in `raw_results/graded.jsonl`, so the learned
router is scored on identical responses to every baseline.

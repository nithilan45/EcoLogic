# Stage 7 — scaled, cleaned retest of the calibration-gap hypothesis

## Verdict

**Partial support, inconclusive.** Scaling the training pool 4.2x and replacing single-sample labels with k=3 majority votes narrowed the accuracy gap to Always-Tier-2 by 1.37 pp (Stage 5: 0.27 pp behind, Stage 7: -1.10 pp behind) without reaching significance (McNemar exact p = 0.2188); this is reported as partial support per the pre-registration, not as a positive result.

Pre-registered outcome label: **partial support, inconclusive**.

| Pre-registered criterion | Result |
|---|---|
| S1 — beats Always-Tier-2, McNemar p < 0.05 | not met |
| S2 — matches accuracy at >= 15% less energy | not met |
| Third outcome — gap to Always-Tier-2 narrows >= 1.0 pp | narrowed +1.37 pp (qualifies) |

## What changed relative to Stage 5

| | Stage 5 | Stage 7 |
|---|---|---|
| Training pool size | 1,200 items | **5000 items** (4.17x) |
| Labels | single sample, temp 0 | **majority of k=3, temp 0.7** |
| TRAIN items | 840 | 3500 |
| TRAIN-CV winner | R2 (0.6746 mean AUC) | R2 (0.7162) |
| Calibration gap to LP frontier | +5.83 pp | **+5.00 pp** |
| Discreteness gap | +0.00 pp | +0.00 pp |

The calibration-gap row is the direct test. If more and cleaner data were the binding constraint, the gap should have shrunk materially.

## Eight-policy comparison, new frozen test set (n = 364)

| Policy | Accuracy [95% Wilson CI] | Tokens | Energy (J) | vs frontier | Tier mix (T1/T2/T3) | McNemar vs Stage 7 router |
|---|---|---|---|---|---|---|
| EcoLogic keyword router | 85.2% [81.1%, 88.4%] (310/364) | 1,333,566 | 1,272.7 | 0.251x | 255/56/53 | p=1.474e-05 (41/10) |
| Always Tier 1 | 83.5% [79.4%, 87.0%] (304/364) | 1,932,931 | 966.5 | 0.191x | 364/0/0 | p=4.336e-07 (46/9) |
| Always Tier 2 | 92.6% [89.4%, 94.9%] (337/364) | 282,675 | 424.0 | 0.084x | 0/364/0 | p=0.2188 (5/1) |
| Random tier | 89.3% [85.7%, 92.1%] (325/364) | 714,922 | 2,012.2 | 0.397x | 126/124/114 | p=0.007 (24/8) |
| Oracle routing | 97.8% [95.7%, 98.9%] (356/364) | 379,207 | 475.1 | 0.094x | 103/254/7 | p=6.104e-05 (0/15) |
| Always-frontier (gpt-4o) | 89.8% [86.3%, 92.5%] (327/364) | 84,372 | 5,062.3 | 1.000x | 0/0/364 | p=0.02431 (24/10) |
| **Learned router — Stage 7 (scaled + k=3 labels)** | 93.7% [90.7%, 95.8%] (341/364) | 295,174 | 543.6 | 0.107x | 4/351/9 | — |
| Learned router — Stage 5 model, re-applied here | 92.9% [89.7%, 95.1%] (338/364) | 329,772 | 554.6 | 0.110x | 13/341/10 | p=0.4531 (5/2) |

McNemar columns are exact two-sided tests on the identical item set; `(b/c)` are discordant counts (Stage-7-router-only-correct / other-only-correct).

Row 8 is Stage 5's router — same weights, same threshold (tau = 0.2245), no re-tuning — scored on **this** test set, so rows 7 and 8 are a paired comparison isolating the effect of scaling the training data. For reference, on its own original test set the Stage 5 router scored 92.0% against Always-Tier-2's 92.3%; that number is not paired with anything in this table.

### Per-benchmark accuracy by policy

| Policy | MBPP (164) | MMLU (100) | GSM8K (100) |
|---|---|---|---|
| EcoLogic keyword router | 76.2% (125/164) | 89.0% (89/100) | 96.0% (96/100) |
| Always Tier 1 | 73.8% (121/164) | 88.0% (88/100) | 95.0% (95/100) |
| Always Tier 2 | 95.7% (157/164) | 85.0% (85/100) | 95.0% (95/100) |
| Random tier | 88.4% (145/164) | 84.0% (84/100) | 96.0% (96/100) |
| Oracle routing | 98.2% (161/164) | 96.0% (96/100) | 99.0% (99/100) |
| Always-frontier (gpt-4o) | 86.0% (141/164) | 88.0% (88/100) | 98.0% (98/100) |
| **Learned router — Stage 7 (scaled + k=3 labels)** | 95.7% (157/164) | 89.0% (89/100) | 95.0% (95/100) |
| Learned router — Stage 5 model, re-applied here | 94.5% (155/164) | 88.0% (88/100) | 95.0% (95/100) |

**The code column carries the pre-registered confound.** MBPP had only 410 unused items left for the training pool against 400 in Stage 1 — a 1.03x scale-up, versus 5.74x for MMLU and GSM8K. Any null result on code is therefore confounded with the inability to add code training data, exactly as flagged in `prereg_stage7.md` before the run.

## Pre-registered comparison in detail: Stage 7 router vs Always Tier 2

| Quantity | Value |
|---|---|
| Stage 7 router accuracy | 93.7% (341/364) |
| Always-Tier-2 accuracy | 92.6% (337/364) |
| Accuracy difference | +1.10 pp |
| McNemar exact p | 0.2188 (discordant 5/1) |
| Stage 7 router energy | 543.6 J |
| Always-Tier-2 energy | 424.0 J |
| Energy reduction vs Always-Tier-2 | -28.2% (>= 15% needed for S2) |
| **S1 met** | False |
| **S2 met** | False |

## Label noise the k=3 majority vote actually removed

| Tier | Items with a 2-1 sample split | Share |
|---|---|---|
| Tier 1 | 290 / 5,000 | 5.8% |
| Tier 2 | 402 / 5,000 | 8.0% |
| Tier 3 | 201 / 5,000 | 4.0% |

Items where no tier's majority grade was correct (label noise, flagged not treated as ground truth): **192** of 5,000. Oracle label mix: {'2': 3428, '1': 1497, '3': 75}.

This table quantifies what the label fix bought. A tier whose three samples disagree is an item whose single-sample Stage 1 label was a coin flip; the share of such items bounds how much label noise scaling could ever have removed.

## Energy-rate sensitivity for the Stage 7 router (27 combinations)

- Savings vs always-frontier range **-105.8% to 97.1%**; combinations where that conclusion flips: **3/27**.
- Savings vs Always-Tier-2 range **-855.1% to 4.9%**; combinations where the router uses *more* energy than Always-Tier-2: **24/27**.

| Tier1 x | Tier2 x | Tier3 x | Router J | Frontier J | vs frontier % | vs Always-T2 % |
|---|---|---|---|---|---|---|
| 5 | 5 | 0.2 | 2,084 | 1,012 | -105.8 | 1.7 |
| 1 | 5 | 0.2 | 2,028 | 1,012 | -100.3 | 4.3 |
| 0.2 | 5 | 0.2 | 2,017 | 1,012 | -99.2 | 4.9 |
| 5 | 1 | 0.2 | 494 | 1,012 | 51.2 | -16.5 |
| 1 | 1 | 0.2 | 438 | 1,012 | 56.7 | -3.3 |
| 0.2 | 0.2 | 5 | 743 | 25,312 | 97.1 | -776.0 |
| 1 | 0.2 | 5 | 754 | 25,312 | 97.0 | -789.2 |
| 5 | 0.2 | 5 | 810 | 25,312 | 96.8 | -855.1 |

(5 least-favourable then 3 most-favourable combinations.)

## Artifacts

- `prereg_stage7.md` — pre-registration, committed before any data
- `s7_model_comparison.md` — R1 vs R2 ablation on the scaled pool
- `s7_threshold_sweep.csv` / `.png` — CALIBRATION threshold sweep
- `s7_mckp_frontier.csv` / `.png` — LP and integer frontiers with both gaps
- `s7_final_results.json` — every number above, machine-readable
- `s7_run_report.md` — generation status, cost and failure accounting, the CALIBRATION-split analysis, and the infrastructure fixes the run required
- `s7_generation_variance.md` — Stage 10(a): how much a temperature-0 rerun moves each accuracy figure above
- `s7_LIMITATIONS.md` — what this stage does and does not establish

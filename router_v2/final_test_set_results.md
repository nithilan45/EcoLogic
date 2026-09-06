# Stage 5 — one-shot frozen test set results

## Pre-registration verdict

**Neither S1 nor S2 was met: a properly trained router still does not surpass static assignment on this workload.** The learned router scored 92.0% against Always-Tier-2's 92.3% (McNemar exact p = 1) using -18.9% energy relative to it (a >= 15% reduction was required for S2).

Evaluated once, as pre-registered: variant **R2**, threshold **tau = 0.2245**, both fixed before the test set was touched. No new API calls; the router selects among per-tier responses already collected for the previous report, so all six rows below are scored on identical generations.

## Six-policy comparison, frozen 364-item test set

| Policy | Accuracy [95% Wilson CI] | Tokens | Energy (J) | vs frontier | Tier mix (T1/T2/T3) | McNemar vs learned |
|---|---|---|---|---|---|---|
| EcoLogic keyword router (previous result) | 86.8% [82.9%, 89.9%] (316/364) | 1,161,679 | 689.0 | 0.114x | 274/87/3 | p=0.0005461 (24/5) |
| Always Tier 1 | 87.6% [83.9%, 90.6%] (319/364) | 1,354,186 | 677.1 | 0.112x | 364/0/0 | p=0.007 (24/8) |
| Always Tier 2 | 92.3% [89.1%, 94.6%] (336/364) | 262,573 | 393.9 | 0.065x | 0/364/0 | p=1 (0/1) |
| Random tier | 90.7% [87.2%, 93.2%] (330/364) | 579,165 | 2,556.3 | 0.421x | 125/107/132 | p=0.3593 (12/7) |
| Oracle routing | 96.4% [94.0%, 97.9%] (351/364) | 295,402 | 385.5 | 0.064x | 160/194/10 | p=3.052e-05 (0/16) |
| **Learned router (new)** | 92.0% [88.8%, 94.4%] (335/364) | 311,305 | 468.4 | 0.077x | 19/340/5 | — |
| Always-frontier (gpt-4o) *(reference)* | 91.5% [88.2%, 93.9%] (333/364) | 101,098 | 6,065.9 | 1.000x | 0/0/364 | p=0.8555 (16/14) |

McNemar columns are exact two-sided tests on the identical item set; `(b/c)` are the discordant counts (learned-only-correct / other-only-correct).

### Per-benchmark accuracy by policy

| Policy | HumanEval (164) | MMLU (100) | GSM8K (100) |
|---|---|---|---|
| EcoLogic keyword router (previous result) | 85.4% (140/164) | 83.0% (83/100) | 93.0% (93/100) |
| Always Tier 1 | 86.0% (141/164) | 83.0% (83/100) | 95.0% (95/100) |
| Always Tier 2 | 96.3% (158/164) | 84.0% (84/100) | 94.0% (94/100) |
| Random tier | 90.2% (148/164) | 86.0% (86/100) | 96.0% (96/100) |
| Oracle routing | 99.4% (163/164) | 90.0% (90/100) | 98.0% (98/100) |
| **Learned router (new)** | 95.7% (157/164) | 84.0% (84/100) | 94.0% (94/100) |

## Pre-registered comparison in detail: learned router vs Always Tier 2

| Quantity | Value |
|---|---|
| Learned router accuracy | 92.0% (335/364) |
| Always-Tier-2 accuracy | 92.3% (336/364) |
| Accuracy difference | -0.3 pp |
| McNemar exact p | 1 (discordant 0/1) |
| Learned router energy | 468.4 J |
| Always-Tier-2 energy | 393.9 J |
| Energy reduction vs Always-Tier-2 | -18.9% (>= 15% needed for S2) |
| **S1 met** | False |
| **S2 met** | False |

## Energy-rate sensitivity for the learned router (27 combinations)

- Savings vs always-frontier range **-67.0% to 98.6%**; combinations where that conclusion flips: **3/27**.
- Savings vs Always-Tier-2 range **-611.1% to 4.8%**; combinations where the learned router uses *more* energy than Always-Tier-2: **22/27**.

| Tier1 x | Tier2 x | Tier3 x | Learned J | Frontier J | vs frontier % | vs Always-T2 % |
|---|---|---|---|---|---|---|
| 5 | 5 | 0.2 | 2,026 | 1,213 | -67.0 | -2.9 |
| 1 | 5 | 0.2 | 1,901 | 1,213 | -56.7 | 3.5 |
| 0.2 | 5 | 0.2 | 1,875 | 1,213 | -54.6 | 4.8 |
| 5 | 1 | 0.2 | 541 | 1,213 | 55.4 | -37.4 |
| 1 | 1 | 0.2 | 416 | 1,213 | 65.7 | -5.6 |
| 0.2 | 0.2 | 5 | 410 | 30,329 | 98.6 | -420.0 |
| 1 | 0.2 | 5 | 435 | 30,329 | 98.6 | -451.9 |
| 5 | 0.2 | 5 | 560 | 30,329 | 98.2 | -611.1 |

(5 least-favourable then 3 most-favourable combinations.)

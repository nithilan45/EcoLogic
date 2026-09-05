### Per-tier accuracy by benchmark (95% Wilson CI)

| Benchmark | n | Tier 1 (Qwen3.5-9B) | Tier 2 (gpt-oss-20b) | Tier 3 (gpt-4o) |
|---|---|---|---|---|
| humaneval | 164 | 86.0% [79.8%, 90.5%] (141/164) | 96.3% [92.2%, 98.3%] (158/164) | 91.5% [86.2%, 94.8%] (150/164) |
| mmlu | 100 | 83.0% [74.5%, 89.1%] (83/100) | 84.0% [75.6%, 89.9%] (84/100) | 88.0% [80.2%, 93.0%] (88/100) |
| gsm8k | 100 | 95.0% [88.8%, 97.8%] (95/100) | 94.0% [87.5%, 97.2%] (94/100) | 95.0% [88.8%, 97.8%] (95/100) |
| **ALL (system-wide)** | 364 | 87.6% [83.9%, 90.6%] (319/364) | 92.3% [89.1%, 94.6%] (336/364) | 91.5% [88.2%, 93.9%] (333/364) |

### MMLU accuracy by subject

| Subject | n | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|---|
| college_biology | 10 | 10/10 | 10/10 | 9/10 |
| college_chemistry | 10 | 8/10 | 8/10 | 8/10 |
| global_facts | 10 | 4/10 | 4/10 | 4/10 |
| high_school_computer_science | 10 | 10/10 | 10/10 | 10/10 |
| high_school_us_history | 10 | 9/10 | 9/10 | 9/10 |
| human_aging | 10 | 7/10 | 8/10 | 8/10 |
| machine_learning | 10 | 9/10 | 9/10 | 10/10 |
| philosophy | 10 | 8/10 | 8/10 | 10/10 |
| prehistory | 10 | 9/10 | 9/10 | 10/10 |
| world_religions | 10 | 9/10 | 9/10 | 10/10 |

### McNemar exact test, paired tier-vs-tier on identical items

| Benchmark | Pair | A>B | B>A | p-value | Significant at 0.05 |
|---|---|---|---|---|---|
| humaneval | T1 vs T2 | 3 | 20 | 0.0004883 | yes |
| humaneval | T1 vs T3 | 9 | 18 | 0.1221 | no |
| humaneval | T2 vs T3 | 13 | 5 | 0.09625 | no |
| mmlu | T1 vs T2 | 2 | 3 | 1 | no |
| mmlu | T1 vs T3 | 1 | 6 | 0.125 | no |
| mmlu | T2 vs T3 | 2 | 6 | 0.2891 | no |
| gsm8k | T1 vs T2 | 3 | 2 | 1 | no |
| gsm8k | T1 vs T3 | 2 | 2 | 1 | no |
| gsm8k | T2 vs T3 | 1 | 2 | 1 | no |
| **ALL** | T1 vs T2 | 8 | 25 | 0.004551 | yes |
| **ALL** | T1 vs T3 | 12 | 26 | 0.03355 | yes |
| **ALL** | T2 vs T3 | 16 | 13 | 0.7111 | no |

### Four-policy comparison (identical 364-item set)

| Policy | Accuracy [95% CI] | Total tokens | Energy (J) | Energy vs frontier | Tier mix (T1/T2/T3) |
|---|---|---|---|---|---|
| (a) EcoLogic routing | 86.8% [82.9%, 89.9%] | 1,161,679 | 689.0 | 0.114x | 274/87/3 |
| (b) Always-frontier (gpt-4o) | 91.5% [88.2%, 93.9%] | 101,098 | 6,065.9 | 1.000x | 0/0/364 |
| (c) Random tier | 90.7% [87.2%, 93.2%] | 579,165 | 2,556.3 | 0.421x | 125/107/132 |
| (d) Oracle routing | 96.4% [94.0%, 97.9%] | 295,402 | 385.5 | 0.064x | 160/194/10 |
| Always Tier 1 (baseline) | 87.6% [83.9%, 90.6%] | 1,354,186 | 677.1 | 0.112x | 364/0/0 |
| Always Tier 2 (baseline) | 92.3% [89.1%, 94.6%] | 262,573 | 393.9 | 0.065x | 0/364/0 |

### Headline gaps

| Quantity | Value |
|---|---|
| EcoLogic accuracy | 86.8% |
| Always-frontier accuracy | 91.5% |
| **Quality gap** (frontier - EcoLogic) | **+4.7 pp** (McNemar p=0.009475) |
| EcoLogic energy | 689.0 J |
| Oracle energy | 385.5 J |
| **Oracle gap** (energy left on table) | **303.4 J = +78.7% over oracle** |
| EcoLogic energy savings vs frontier | 88.6% |
| Oracle accuracy (efficiency ceiling) | 96.4% |
| Always-Tier-1 accuracy / energy | 87.6% / 677.1 J |
| EcoLogic vs Always-Tier-1 | McNemar p=0.25 (discordant 0/3) |

### Energy under both rate sets

| Policy | Energy, paper rates (0.5/1.5/60 J per 1k) | Energy, substitute-model rates (1.1/2.0/60) |
|---|---|---|
| (a) EcoLogic routing | 689.0 J | 1,379.5 J |
| (b) Always-frontier (gpt-4o) | 6,065.9 J | 6,065.9 J |
| (c) Random tier | 2,556.3 J | 2,875.5 J |
| (d) Oracle routing | 385.5 J | 549.5 J |
| Always Tier 1 (baseline) | 677.1 J | 1,489.6 J |
| Always Tier 2 (baseline) | 393.9 J | 525.1 J |

### Energy-rate sensitivity: each tier's rate independently x0.2, x1, x5

- 27 rate combinations evaluated.
- EcoLogic energy savings vs always-frontier ranges **-164.6% to 98.8%**.
- Combinations where the savings conclusion flips (savings <= 0): **3**.

| Tier1 x | Tier2 x | Tier3 x | EcoLogic J | Frontier J | Savings % | Oracle gap % |
|---|---|---|---|---|---|---|
| 5 | 5 | 0.2 | 3,209 | 1,213 | -164.6 | 209.2 |
| 5 | 1 | 0.2 | 2,852 | 1,213 | -135.1 | 677.9 |
| 5 | 0.2 | 0.2 | 2,781 | 1,213 | -129.2 | 2338.6 |
| 1 | 5 | 0.2 | 1,007 | 1,213 | 17.0 | 102.8 |
| 1 | 1 | 0.2 | 650 | 1,213 | 46.4 | 127.0 |
| 0.2 | 0.2 | 5 | 373 | 30,329 | 98.8 | -28.5 |
| 0.2 | 1 | 5 | 444 | 30,329 | 98.5 | -26.2 |
| 0.2 | 5 | 5 | 802 | 30,329 | 97.4 | 0.2 |

(5 least-favourable combinations then 3 most-favourable.)

### Router behaviour

- Raw-query tier mix: {'1': 274, '2': 87, '3': 3}
- Wrapped-prompt tier mix: {'1': 137, '2': 53, '3': 174}
- Raw vs wrapped routing agreement: 53.0%
- Items EcoLogic got wrong where some other tier was right: 35/364 (9.6%)

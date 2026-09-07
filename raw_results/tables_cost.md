### Policy comparison on a MEASURED cost axis (n = 364 items)

Dollar cost from the providers' own `usage` fields at published prices. No modelled energy rates are used anywhere in this table.

| Policy | Accuracy [95% CI] | Cost (USD) | $/item | vs frontier | Tier mix (T1/T2/T3) | McNemar vs EcoLogic |
|---|---|---|---|---|---|---|
| EcoLogic routing (real classifier) | 86.8% [82.9%, 89.9%] (316/364) | $0.2867 | $0.000788 | 0.450x | 274/87/3 | — |
| Always Tier 1 | 87.6% [83.9%, 90.6%] (319/364) | $0.3342 | $0.000918 | 0.525x | 364/0/0 | p=0.25 (0/3) |
| Always Tier 2 | 92.3% [89.1%, 94.6%] (336/364) | $0.0418 | $0.000115 | 0.066x | 0/364/0 | p=0.0003249 (5/25) |
| Random tier | 90.7% [87.2%, 93.2%] (330/364) | $0.3628 | $0.000997 | 0.570x | 125/107/132 | p=0.006611 (5/19) |
| Always-frontier (gpt-4o) | 91.5% [88.2%, 93.9%] (333/364) | $0.6366 | $0.001749 | 1.000x | 0/0/364 | p=0.009475 (11/28) |
| Oracle (cheapest correct) | 96.4% [94.0%, 97.9%] (351/364) | $0.0470 | $0.000129 | 0.074x | 9/343/12 | p=5.821e-11 (0/35) |

**Always-Tier-2 dominates EcoLogic on the measured cost axis: True** (+5.5 pp accuracy at 6.87x the router's cost).

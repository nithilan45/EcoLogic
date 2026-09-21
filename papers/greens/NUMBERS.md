# NUMBERS.md — every reported digit → source

All energy figures are `total_tokens / 1000 * assumed J/1k`. **No joule was metered.**
Reconstruction: `python3 papers/greens/analysis.py` (reads frozen `raw_results/` only).

Rounded display values match `EVALUATION.md` / `results_report.md` unless noted.

## Energy model

| Quantity | Value | Source |
|---|---|---|
| Paper rates J/1k T1/T2/T3 | 0.5 / 1.5 / 60 | `backend/main.py` MODELS; `woais_experiments/configs/models.json` `paper_energy_per_1k`; `benchmark/api.py` |
| Substitute rates | 1.1 / 2.0 / 60 | `models.json` `subst_energy_per_1k`; `results_report.md` Substitutions |
| GPT-5 comparator in product | 500 J/1k | `backend/main.py` `GPT5_ENERGY_PER_1K` |
| Formula | tokens/1000 * rate | `benchmark/analyze.py` `energy()`; `woais_experiments/accounting/costs.py` `energy_j()` |

## Sample

| Quantity | Value | Source |
|---|---|---|
| n items | 364 | HumanEval 164 + MMLU 100 + GSM8K 100; `raw_results/benchmark_items.json`; `results_report.md` |
| Graded calls | 1,092 | 364×3; `raw_results/graded.jsonl` |
| Routing mix EcoLogic (raw) | 274 / 87 / 3 | `raw_results/routing.json` + analysis; `raw_results/tables.md` |
| Raw vs wrapped agreement | 53.0% | `results_report.md` §4; `raw_results/tables.md` |

## Table 1 — paper rates (rounded as in EVALUATION.md)

Exact values from `papers/greens/artifacts/policy_energy.csv` / `raw_results/analysis.json` `policies.*`.

| Policy | Acc | Energy J (exact → rounded) | Tokens | USD | Mix |
|---|---|---|---|---|---|
| EcoLogic | 316/364 = 86.813…% → 86.8%; Wilson [82.950%, 89.908%] → [82.9%, 89.9%] | 688.953 → **689** | 1,161,679 | 0.28674677 → $0.2867 | 274/87/3 |
| Always-T2 | 336/364 = 92.308…% → 92.3%; [89.096%, 94.647%] → [89.1%, 94.6%] | 393.8595 → **394** | 262,573 | 0.0417509 → $0.0418 | 0/364/0 |
| Always-T1 | 319/364 = 87.637…% → 87.6%; [83.859%, 90.627%] → [83.9%, 90.6%] | 677.093 → **677** | 1,354,186 | 0.33422946 → $0.3342 | 364/0/0 |
| Frontier | 333/364 = 91.484…% → 91.5%; [88.170%, 93.947%] → [88.2%, 93.9%] | 6065.88 → **6066** | 101,098 | 0.636595 → $0.6366 | 0/0/364 |
| Random | 330/364 = 90.659…% → 90.7%; [87.261%, 93.232%] | 2556.334 → **2556** | 579,165 | — | 125/107/132 |
| Oracle (energy) | 351/364 = 96.429…% → 96.4%; [93.967%, 97.910%] | 385.5215 → **386** | 295,402 | 0.0702458 energy-oracle USD (not the $0.0470 cheapest-USD oracle in `tables_cost.md`) | 160/194/10 |

USD column: `raw_results/tables_cost.md` / stored `usd` on the energy-paper assignments. **Do not** use `tables_cost.md` oracle $0.0470 as this paper's energy-oracle USD (different oracle objective).

## Derived headlines

| Quantity | Value | How |
|---|---|---|
| T2 accuracy advantage | 5.5 pp | 92.3 − 86.8 |
| T2 / Eco energy | 394/689 ≈ 0.5717 → **0.57×** | exact 393.8595/688.953 = 0.5717 |
| Eco / T2 energy | **1.749×** → ~1.75× | 688.953/393.8595 |
| Eco vs frontier J | 688.953/6065.88 = 0.11358 → **0.114×**; saving **88.642% → 88.6%** | `analysis.json` `ecologic_energy_savings_vs_frontier_pct` |
| Eco vs frontier USD | 0.28674677/0.636595 = 0.4504 → **0.450×**; saving **54.96% → 55.0%** | `tables_cost.md`; EVALUATION.md |
| Residual USD/J ratio | 0.4504/0.11358 ≈ **3.97 → ~4×** | EVALUATION.md “roughly 4x” |
| Token inflation | 1,161,679/101,098 = 11.491 → **11.5×** | `results_report.md` §5 |
| Rate ratio T3/T1 | 60/0.5 = **120×** | paper rates |
| McNemar Eco vs T2 | 5 / 25, p = 0.0003249 | `tables_cost.md`; `analysis.py` |
| McNemar Eco vs T1 | 0 / 3, p = 0.25 | `analysis.json` |
| McNemar Eco vs frontier | 11 / 28, p = 0.009475 | `analysis.json`; `results_report.md` quality gap |
| McNemar Eco vs oracle | 0 / 35, p = 5.82e-11 | `analysis.json` |
| McNemar T2 vs T3 overall | p = 0.711 | `results_report.md` |
| Escalations off T1 | 90; 0 rescues; 3 breaks | `results_report.md` §1 |
| Eco energy by chosen tier | T1 550.68 J (274 calls, 1,101,360 tok); T2 89.253 J (87); T3 49.02 J (3, 817 tok) | `artifacts/headline_dump.txt` |
| Wrapped-prompt Eco energy | 3419.1 J | `results_report.md` §4 (not a main table) |

## Per-benchmark modelled J (`artifacts/per_benchmark_energy.csv`)

| | HumanEval | MMLU | GSM8K |
|---|---|---|---|
| EcoLogic | 352.443 → 352 / 140/164=85.4% | 161.065 → 161 / 83% | 175.445 → 175 / 93% |
| Always-T2 | 220.294 → 220 / 158/164=96.3% | 76.325 → 76 / 84% | 97.241 → 97 / 94% |
| Always-T1 | 333.074 → 333 / 86.0% | 173.377 → 173 / 83% | 170.642 → 171 / 95% |
| Frontier | 2895.0 → 2895 / 91.5% | 875.52 → 876 / 88% | 2295.36 → 2295 / 95% |

Tier accuracies (results_report §2): HE 86.0/96.3/91.5; MMLU 83/84/88; GSM8K 95/94/95.

## Sensitivity 27 (`artifacts/sensitivity_27.csv`; `raw_results/analysis.json` `sensitivity`)

| Quantity | Value | Source |
|---|---|---|
| Band vs frontier | −164.551% … +98.770% → **−164.6% to +98.8%** | `analysis.json` savings_min/max; `results_report.md` §5 |
| Sign flips vs frontier | **3/27** | m=(5,0.2,0.2) −129.2%; (5,1,0.2) −135.1%; (5,5,0.2) −164.6% |
| Eco more energy than T2 | **19/27** | `analysis.py` unique to Paper B |
| Eco less energy than T2 | **8/27** | (m1,m2) ∈ {(0.2,1),(0.2,5),(1,5)} × all m3 |
| Saving vs T2 at paper rates | −74.92% (uses 1.75×) | combo (1,1,1) |
| Substitute-rate energies | Eco 1379.52 → 1380 J; T2 525.146 → 525 J; frontier 6065.88 | `analysis.json` `energy_both_rate_sets` |

## What we did **not** put in abstract / Table 1

- RouteLLM GSM8K 9-point sweep (one sentence in threats).
- Latency p50/p90, serverless DES.
- Learned-router AUC / pre-registration as a section.
- Energy-oracle USD $0.0702 as a headline (axis mixing).
- `tables_cost.md` USD-oracle $0.0470 as this paper's oracle energy companion.

## Failed / non-claims

- Physical wattmeter: none in repo.
- Full factorial was already in `raw_results/analysis.json`; we recomputed it offline from item tokens and added the Always-T2 ranking (19/27) which the original report did not headline.
- pdflatex may be absent on the author machine; `build.sh` documents the engine.

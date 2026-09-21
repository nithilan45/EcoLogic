# NUMBERS.md — every reported digit → source

Paper: `papers/icaart/main.tex`. Reconstruction: `python3 papers/icaart/analysis.py` → `papers/icaart/artifacts/`. Frozen trees were not modified.

Rounding: percentages to 1 decimal as in `raw_results/tables_cost.md` unless a 4-decimal identifier (AUC, quality 0.8889, p-values).

| Paper number | Meaning | Source path | Reconstruction |
|---|---|---|---|
| 364 | Stage 1–2 panel size | `raw_results/analysis_cost.json` `n_items`; `graded.jsonl` unique items | `artifacts/stage12_summary.json` `n` |
| 164 / 100 / 100 | HumanEval / MMLU / GSM8K counts | `EVALUATION.md`; `analysis.json` per-benchmark `n` | same |
| 86.8% (316/364) | EcoLogic accuracy | `raw_results/tables_cost.md`; `analysis_cost.json` `policies.ecologic` | `stage12_summary.json` `ecologic_acc_pct` |
| [82.9%, 89.9%] | EcoLogic 95% Wilson | `analysis_cost.json` `policies.ecologic.ci` | Wilson from 316/364 |
| $0.2867 | EcoLogic total USD | `tables_cost.md`; `analysis_cost.json` `cost_usd` 0.28674677 | reconstructed from `graded.jsonl`+`routing.json` |
| 274/87/3 | EcoLogic mix | `tables_cost.md`; `analysis_cost.json` `tier_mix` | reconstructed |
| 92.3% (336/364) | Always-T2 accuracy | `tables_cost.md` | `always_t2_acc_pct` |
| [89.1%, 94.6%] | Always-T2 Wilson | `analysis_cost.json` `always_t2.ci` | Wilson 336/364 |
| $0.0418 | Always-T2 USD | `tables_cost.md`; exact 0.0417509 | reconstructed |
| 0.0003249 | McNemar EcoLogic vs T2 | `tables_cost.md`; `analysis_cost.json` `mcnemar_vs_ecologic.always_t2.p_value` 0.0003249142318964005 | exact binomial b=5,c=25 |
| 5/25 | discordant pairs vs T2 | same `b`,`c` | reconstructed flags |
| +5.5 pp | T2−EcoLogic accuracy | `tables_cost.md`; 5.4945… in JSON | `accuracy_gap_t2_minus_eco_pp` |
| 6.87× / 6.868 | USD ratio EcoLogic/T2 | `tables_cost.md`; `cost_ratio_ecologic_over_t2` | `cost_ratio_eco_over_t2` |
| 87.6% (319/364), $0.3342 | Always-T1 | `tables_cost.md` | reconstructed |
| p=0.25 (0/3) | McNemar vs T1 | `analysis_cost.json` | reconstructed |
| 90.7% (330/364), $0.3628 | Random stored seed | `tables_cost.md` / `analysis_cost.json` (not re-rolled) | copied from stored |
| p=0.006611 (5/19) | McNemar vs random | `analysis_cost.json` | stored |
| 91.5% (333/364), $0.6366 | Always-frontier | `tables_cost.md` | reconstructed |
| p=0.009475 (11/28) | McNemar vs frontier | `analysis_cost.json` | reconstructed |
| 96.4% (351/364), $0.0470 | Oracle | `tables_cost.md` | reconstructed |
| p=5.821e-11 (0/35) | McNemar vs oracle | `analysis_cost.json` | reconstructed |
| 0.450×, 0.525×, 0.066×, 0.570×, 0.074× | vs frontier cost | `analysis_cost.json` `cost_vs_frontier` | reconstructed |
| 87.6 / 92.3 / 91.5% all-panel tiers | per-tier accuracy | `analysis.json` `tier_accuracy` `tier{1,2,3}/ALL` | checksum `analysis_cost` always_* |
| [83.9,90.6], [89.1,94.6], [88.2,93.9] | tier Wilson | `analysis.json` | Wilson |
| 95/94/95 GSM8K | per-tier GSM8K | `analysis.json` `tier*/gsm8k` | — |
| HumanEval 96.3% / 91.5% | T2 / T3 HE | `analysis.json` 158/164, 150/164 | — |
| p=0.711 (16/13) | T2 vs T3 McNemar ALL | `analysis.json` `mcnemar.ALL/t2_vs_t3` | — |
| 53.0% raw vs wrapped agreement | keyword stability | `woais_experiments/results/routing/stage12_policies.json` `agreement` 0.53021978; `results_report.md` | not re-derived in analysis.py |
| 164 HumanEval all to T3 when wrapped | `results_report.md` §4 | qualitative from routing wrapped | — |
| n=1307 | RouteLLM GSM8K after decontam | `stage7_10/s9_static_baselines.json` `n_items` | `routellm_summary.json` |
| 1319 | GSM8K rows before decontam | `stage7_10/external_generalization.json` `release_audit.gsm8k.n_rows` | — |
| +0.57 pp | mean cost-matched edge | `s9_static_baselines.json` `mean_router_edge_matched_cost_pp` 0.56627 | mean of 9 interior edges |
| +0.76 pp | mean call-fraction edge | `woais_experiments/results/external/audit/routellm_s9_committed.json` `mean_edge_matched_fraction_pp` 0.75567 | mean of 9 interior frac edges |
| 0.09–0.30 pp | advantage lost to cost matching | interior `advantage_lost_to_cost_matching_pp` min 0.0888 max 0.2968 | `lost_min/max_interior_pp` |
| 8 of 9 | points beating cost-matched mix | `n_interior_beating_matched_cost_mixture` | count of positive cost edges |
| 0 of 9 | points with MC p<0.05 | `n_interior_beating_matched_cost_mixture_p05` | all `mc_p_value` ≥ 0.0551 |
| p=0.039 | sign test | `sign_test_across_sweep.p_two_sided` 0.0390625 | binomial 8/9 two-sided |
| 20,000 | MC draws | `s9_static_baselines.json` `mc_draws` | — |
| 63.73% / 85.77% | Mixtral / GPT-4 acc | `s9` `accuracy.always_weak_pct` / `always_strong_pct` | — |
| Table B per-threshold acc/edges/p | sweep | `s9_static_baselines.json` `sweep` | `artifacts/routellm_gsm8k_sweep.csv` |
| prices 10/30 and 0.60/0.60 | $/1M tok | `s9` `prices_usd_per_1m`; `external_generalization.json` | input, not billed |
| R_naive = −0.2496, R_true = +0.2840 | EcoLogic calib identity | `stage7_10/regret_correction_validation.json` (−0.249623, +0.283969) | `accounting_summary.json` |
| correction +0.5336 | same | `correction` 0.53359266 | — |
| 1.88× / ~1.9× | correction / \|R_true\| | 0.53359/0.28397 ≈ 1.879 | `correction_over_abs_R_true` |
| residual 2.2×10^{-16} | S8 reconcile | `abs_residual` 2.220e-16 | — |
| n=360 calibration | S8 | `regret_correction_validation.json` `n_items` | — |
| $0.0199 / 6.95% | EcoLogic USD naive understatement | `stage12_ecologic_decomposition.json` naive 0.26681589 vs realized 0.28674677 | `eco_understatement_pct` |
| residual 1.1×10^{-19} | USD identity | `abs_residual` 1.084e-19 | — |
| up to 6.3% | RouteLLM naive understates cost | `external_generalization.json` `max_abs_cost_misestimate_pct` 6.3054 | rounded 6.3; table 6.31% at 10% strong |
| 8.7×10^{-19} | RouteLLM identity residual | `max_abs_residual` | — |
| 30% strong sign flip | RouteLLM | sweep row `strong_pct` 29.992, `sign_flip` true; R_naive +3.21e-5, R_true −9.23e-5 | `routellm_accounting_sweep.csv` |
| CV 0.36 / 0.43 | Mixtral / GPT-4 cost CV | `within_model_cost_cv` 0.356 / 0.428 | — |
| 0.8889 | held-out logistic = always_cheap = cost_matched_static quality | `woais_experiments/results/heldout/final_test_results.csv` 0.888888… | `heldout_summary.json` |
| $0.0001526 /item | same rows `realized_cost` | CSV | identical across those methods |
| n=243 test; 720/237/243 split | `frozen_config.json` `n` | — |
| quality advantage 0.0 | logistic vs CMS | CSV `quality_advantage_vs_cost_matched_static` | — |
| heuristic 0.860 | `ecologic_heuristic.quality` 0.860082 | CSV | — |
| oracle 0.942 | `oracle.quality` 0.942387 | CSV | — |
| 92.0% vs 92.3%, McNemar p=1 | Stage 5 one-shot | `router_v2/final_test_set_results.json` 0.92033 vs 0.92308; mcnemar p=1.0 b=0 c=1 | `stage5_prereg.json` |
| 0.6816 | logistic mean CALIB AUC | `stage7_10/s7_ceiling.json` 0.681642 | rounded 4 dp |
| 0.6547 / 0.6703 / 0.6521 | GB / RF / kNN calib AUC | same `models.*.mean_calibration_auc` | `ceiling_auc.csv` |
| 0.8140 / 0.9577 / 0.9999 / 0.7835 | train AUCs | same | RF train 0.99987 → 0.9999 |
| TRAIN 3500 / CALIB 1500 | `s7_ceiling.json` `n_train` `n_calibration` | — |
| −0.0114 AUC lift | `auc_lift_over_linear` | — |
| 19.8% (72/364) | T1 temp-0 flips | `s7_generation_variance.json` `always_t1.n_items_with_flip` 72; frac 0.19780 | `generation_variance_summary.json` |
| 1.4% (5/364) | T3 flips | `always_t3` 5 items | — |
| 47% | T1 within-share of CI | `within_share_of_total` 0.46971 → 47% | EVALUATION.md uses 47% |
| 4.9% | frontier within-share | `always_t3.within_share_of_total` 0.04884 | — |
| k=3, temperature 0 | `s7_generation_variance.json` | — |
| 1936 tokens | T1 mean within-item token SD | `s7_generation_variance.md` table (JSON policies do not store this scalar; markdown/docs) | cited from `s7_generation_variance.md` |
| 85% | RouteLLM README call-fraction slogan | `EVALUATION.md`; `external_generalization.md` | not recomputed |
| $5×10^{-5}$ | naive exact for random residual | `s9` `naive_formula_exact_for_random_residual_usd` 5.45e-5 | EVALUATION.md ~5e-5 |
| 15% energy (S2) | pre-reg criterion | `router_v2/PREREGISTRATION.md` | not a measured USD result |
| 60/20/20 | held-out split style | `frozen_config.json` 720/237/243 of 1200 | — |

## Analysis that did not run in this environment

- `pdflatex` / `bibtex` were not on `PATH` (no MacTeX/TeX Live). `build.sh` reconstructs tables; PDF compile is skipped until a TeX install is available.
- No paid API calls. RouteLLM query-level re-scoring was not re-run; Study B uses committed aggregates (`s9_static_baselines.json`), as `woais_experiments/results/external_router` has `available: false`.

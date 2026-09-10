# Adversarial audit of `woais_experiments/`

Reviewer stance: senior ML systems / research integrity. Scope is the live
`woais_experiments/` implementation, not frozen Stage 1–10 trees
(`raw_results/`, `router_v2/`, `stage7_10/`). Those hashed files were **not**
modified. SHA256 verification still passes.

Full suite: **259 tests, OK**
(`python3.13 -m unittest discover -s woais_experiments/tests -t . -v`).

The existing `test_offline_suite` regenerates **derived** files under
`woais_experiments/results/` (REPORT, JSON summaries). That is the offline
writer, not a frozen experimental tree.

---

## Verdict

No deployed keyword router was fit on evaluation labels. Classic leakage
(threshold search on test, scaler fit, RouteLLM re-training) is absent.

Several **real** accounting, statistics, systems, and integrity bugs were
present. They are fixed below. Headline Stage 1–2 USD conclusions are
unchanged (Always-T2 still dominates; keyword overhead in config is $0).

---

## Issues found and fixed

### 1. Missing USD / latency / tokens coerced to zero

| | |
|---|---|
| **Severity** | High |
| **File** | `woais_experiments/frozen.py` (`_normalize_call`, `matrix_from_calls`) |
| **Problem** | `usd or 0.0`, `latency_s or 0.0`, `total_tokens or 0`, and `bool(None)` treated missing measurements as free, instant, empty, or incorrect. That fabricates complete observations. |
| **Fix** | Keep `None` on ingest. Drop incomplete `(tier, item)` cells; record `n_dropped_incomplete`. Never zero-fill. |
| **Test** | `woais_experiments.tests.test_audit_fixes.TestNoZeroFillMissing.test_incomplete_calls_are_dropped_not_zeroed` |

### 2. Modelled energy stored as undifferentiated `cost`

| | |
|---|---|
| **Severity** | High |
| **File** | `woais_experiments/accounting/costs.py`, `woais_experiments/run_offline.py` |
| **Problem** | `tokens/1000 × paper_energy_per_1k` is not metered joules, but landed in the same `cost` field as USD with no provenance. |
| **Fix** | `evaluate_assignment` now emits `cost_kind`, `cost_unit`, `cost_provenance`. Energy axis is `modelled_energy` / `J` / `tokens/1000 * paper_energy_per_1k (not metered)`. Payload also has `energy_is_modelled`. |
| **Test** | `TestEnergyProvenance.test_energy_assignment_is_labeled_modelled` |

### 3. Dual USD / energy rate tables (`api.MODELS` vs `models.json`)

| | |
|---|---|
| **Severity** | Medium |
| **File** | `woais_experiments/accounting/costs.py` |
| **Problem** | Rates preferred live `benchmark.api` when httpx was importable, else `configs/models.json`. Silent drift. |
| **Fix** | Production path reads **only** `configs/models.json`. |
| **Test** | `TestEnergyProvenance.test_paper_energy_rates_do_not_import_api`; `test_accounting.TestAccounting.test_usage_and_cost_matches_published_rates` |

### 4. Headline policy USD omitted router overhead

| | |
|---|---|
| **Severity** | Medium |
| **File** | `woais_experiments/accounting/costs.py`, `woais_experiments/routing/policies.py` |
| **Problem** | `evaluate_assignment` summed inference USD only. Breakeven included overhead; stage12 tables did not. Config overhead is $0 today, so published USD did not move, but any non-zero overhead would have been understated. |
| **Fix** | EcoLogic policies add `overhead_per_query` from `router_overhead.ecologic_keyword`. Fields: `inference_cost`, `router_overhead_per_query`, `cost` = inference + overhead×n. Static policies get overhead 0. |
| **Test** | `TestRouterOverheadInHeadline.test_overhead_is_added_only_for_ecologic` |

### 5. `+∞` serialized as JSON `null`

| | |
|---|---|
| **Severity** | Medium |
| **File** | `woais_experiments/frozen.py` (`to_jsonable`, `write_result`); local `_jsonable` helpers |
| **Problem** | Non-finite floats became `None`. Oracle break-even `plus_infinity` looked missing. |
| **Fix** | JSON writes `"Infinity"` / `"-Infinity"` (`allow_nan=False`). NaN still `null`. |
| **Test** | `TestJsonInf.test_infinity_serializes_as_string_not_null` |

### 6. Bare `"latency"` assumed seconds

| | |
|---|---|
| **Severity** | Medium |
| **File** | `woais_experiments/external/schema.py` (`latency_to_ms`) |
| **Problem** | Auto mode treated unknown / `"latency"` as seconds (`×1000`). Millisecond dumps named `latency` were inflated 1000×. |
| **Fix** | Auto-convert only `*_ms` / `*_s` aliases. Bare `latency` raises unless `latency_unit` is `'ms'` or `'s'`. |
| **Test** | `test_external_adapter.TestRouteLLMAndDiscovery.test_bare_latency_column_requires_explicit_unit` |

### 7. Price-table fill labeled `realized_cost`

| | |
|---|---|
| **Severity** | Medium |
| **File** | `woais_experiments/external/schema.py`, `adapter.py`, `xroutebench.py` |
| **Problem** | Missing costs filled from $/1M tables were stored as `realized_cost` (metered language). |
| **Fix** | `CanonicalRow.cost_source` ∈ `{metered, price_table, missing}`. Notes say price-table fill is not metered. |
| **Test** | `test_external_adapter.TestRouteLLMAndDiscovery.test_price_table_fill_is_not_labeled_metered` |

### 8. CI overlap sold as `significant_05_quality`

| | |
|---|---|
| **Severity** | High |
| **File** | `woais_experiments/external/run_external_audit.py` (`bootstrap_router_vs_static`) |
| **Problem** | `significant_05_quality = (CI_lo > 0)`, or `point > 0` when `n_boot=0`. That is the CI-as-test pattern `SIGNIFICANCE_NOTE` forbids. |
| **Fix** | Flag is always `None`. CI kept as an interval. `ci_is_not_a_hypothesis_test: true`. Effect size = quality-advantage point. Bootstrap share `p_quality_advantage_le_zero` is descriptive. |
| **Test** | `test_external_adapter.TestAuditTwoToFifty.test_two_model_full_stack` |

### 9. Robustness: unadjusted p-values, no effect size, eval-set quality grid

| | |
|---|---|
| **Severity** | High (stats) / Medium (grid) |
| **File** | `woais_experiments/routing/robustness.py` |
| **Problem** | Dozens of paired sign-flip tests tagged `p<0.05` with no BH. Primary CSV had no effect size. `QUALITY_TARGETS` included **0.868** and **0.923**, the Stage 1–2 EcoLogic / Always-T2 accuracies. |
| **Fix** | Structural rows get Benjamini–Hochberg `p_adjusted`; significance strings use `p_adj` plus Cohen’s \(d_z\). Bootstrap rows are labeled sensitivity, not tests. Quality grid is a pre-registered 2.5 pp ladder on [0.80, 0.95]. |
| **Test** | `TestRobustnessStats.test_bh_rewrites_significance_with_effect_size`; `test_quality_grid_is_not_eval_peek`; `test_robustness.TestHelpers.test_quality_targets_are_preregistered_grid` |

### 10. Latency penalty imputed 0 for missing latency

| | |
|---|---|
| **Severity** | Low |
| **File** | `woais_experiments/routing/robustness.py` (`apply_latency_penalty`) |
| **Problem** | Non-finite latency added **$0**, i.e. “no latency cost.” |
| **Fix** | Require finite latency on every cell; drop incomplete rows before the penalty sweep. |
| **Test** | `TestRobustnessStats.test_latency_penalty_refuses_to_impute_zero` |

### 11. Break-even bootstrap CI for `raw_router_savings` used the wrong replicates

| | |
|---|---|
| **Severity** | High |
| **File** | `woais_experiments/accounting/breakeven.py` (`analyze_assignment`) |
| **Problem** | `_ci_payload(be_s, row["raw_router_savings"])` attached **break-even overhead** bootstrap draws to the raw-savings interval. Today the two point functionals coincide for USD, so numbers matched; the wiring was still wrong if they diverge. |
| **Fix** | Collect `boot_raw_*` from `sample[axis]["raw_router_savings"]` and pass those into `_ci_payload` for raw savings. |
| **Test** | `test_breakeven.TestNoFittingImports.test_raw_savings_bootstrap_uses_raw_replicates`; `TestPathsAndInfWiring.test_analyze_assignment_collects_raw_bootstrap_list` |

### 12. McNemar family unadjusted; no effect size

| | |
|---|---|
| **Severity** | Medium |
| **File** | `woais_experiments/routing/policies.py` (`evaluate_policies`) |
| **Problem** | One McNemar per policy vs EcoLogic, raw `p_value` only, same paired items. |
| **Fix** | BH over the family; `p_raw`, `p_adjusted`, `accuracy_delta` / `effect_size`. |
| **Test** | `TestMcnemarBh.test_policy_mcnemar_family_has_adjusted_p_and_effect_size` |

### 13. Sign-flip permutation dropped non-finite nulls from the denominator

| | |
|---|---|
| **Severity** | Low |
| **File** | `woais_experiments/statistics/paired_tests.py` |
| **Problem** | For \(d_z\), some sign patterns are non-finite; those draws were removed from `n_perm`, slightly anticonservative. |
| **Fix** | Denominator is the intended number of draws. Non-finite nulls are not counted as extreme (`n_nonfinite_nulls`). |
| **Test** | `TestPermutationDenominator.test_nonfinite_nulls_stay_in_the_denominator` |

### 14. Multiple endpoints in `paired_comparison` without a stated primary

| | |
|---|---|
| **Severity** | Medium |
| **File** | `woais_experiments/statistics/paired_tests.py` |
| **Problem** | Mean / median / \(d_z\) / Cliff’s δ each have a raw p. Easy to cherry-pick. `compare_many` BH-adjusts only `family` (default mean). |
| **Fix** | Record `primary_endpoint: mean_paired_difference`. Note that other endpoints are descriptive unless BH’d. |
| **Test** | Existing `test_paired_statistics` plus the new note; BH family tests already in `TestCompareMany` |

### 15. Wall-clock HTTP RTT called “service time”; queueing double-counts delay

| | |
|---|---|
| **Severity** | High |
| **File** | `woais_experiments/latency/serverless.py`, `woais_experiments/run_offline.py` REPORT writer |
| **Problem** | Frozen `latency_s` is full HTTP RTT (client + provider queue + generate). Feeding it to FCFS as exclusive service, then adding simulated wait, double-counts provider delay. REPORT said “mean service time.” |
| **Fix** | Outputs include `mean_wall_clock_s`, `honesty` (SIMULATED, double-count). REPORT says “wall-clock HTTP RTT,” not exclusive service, and labels sojourn SIMULATED. |
| **Test** | `test_serverless.TestServerlessModel.test_idle_timeout_adds_cold_penalty`; regenerated `results/REPORT.md` |

### 16. Cold start in wait but not utilization; wait mixed queue + cold

| | |
|---|---|
| **Severity** | Medium |
| **File** | `woais_experiments/latency/serverless.py` (`simulate_fcfs`) |
| **Problem** | `wait = begin - arrival` included cold. `busy = sum(service)` ignored cold, understating utilization. |
| **Fix** | `mean_wait_s` / `mean_queue_wait_s` = queue only (`ready - arrival`). `mean_cold_s` separate. Utilization uses `service + cold`. Residual outliers exposed as `residual_outlier_frac` (alias `cold_like_frac` kept, with a proxy note). |
| **Test** | `test_serverless.TestServerlessModel.test_idle_timeout_adds_cold_penalty` |

### 17. RouteLLM sweep sign-test reported as confirmatory

| | |
|---|---|
| **Severity** | High |
| **File** | `woais_experiments/run_offline.py` (`write_report`) |
| **Problem** | REPORT quoted `sign-test p = 0.0391` next to “0 significant at p<0.05.” Stage 9 JSON already calls that sign test descriptive (dependent interior points). |
| **Fix** | Prose: unadjusted interior McNemar count; sign count is **not** a Type-I-controlled test. |
| **Test** | Regenerated `results/REPORT.md` (offline suite); generator is `write_report` |

### 18. xRouteBench: `>20` tasks dropped **all** per-task jobs

| | |
|---|---|
| **Severity** | Medium |
| **File** | `woais_experiments/external/run_xroutebench.py` (`subset_jobs_from_meta`) |
| **Problem** | If a split had more than `MAX_TASK_SUBSETS` tasks, **no** per-task jobs were planned (silent omission, not a sample). |
| **Fix** | Keep a deterministic alphabetical cap of 20; do not return an empty plan. |
| **Test** | `test_xroutebench.TestAuditAndResume.test_too_many_tasks_are_capped_not_dropped` |

### 19. Complete-case model truncation ranked by coverage on the audited split

| | |
|---|---|
| **Severity** | Low |
| **File** | `woais_experiments/external/xroutebench.py` (`select_complete_rows`) |
| **Problem** | If `> MAX_MODELS`, models were ranked by coverage on the split being scored (including `test`). |
| **Fix** | Truncate by model **name** (frozen, not eval-ranked). |
| **Test** | `test_xroutebench.TestAuditAndResume.test_complete_case_truncation_is_alphabetical` |

### 20. Offline path: no environment metadata; absolute machine paths

| | |
|---|---|
| **Severity** | High / Medium |
| **File** | `woais_experiments/run_offline.py`, `woais_experiments/paths.py` (`repo_rel`) |
| **Problem** | `run_offline` wrote `summary.json` with host home-directory absolute paths and no git / Python / config hash / `PYTHONHASHSEED`. |
| **Fix** | Write `run_environment.json` (git, Python, platform, hash seed, config hash, seeds). Summary paths are repo-relative. |
| **Test** | `test_offline_suite.TestOfflineSuite.test_run_writes_results_and_preserves_hashes` (now emits `run_environment.json`); runner tests already stamp git + config hash |

### 21. Silent overwrite of `woais_experiments/results/`

| | |
|---|---|
| **Severity** | High (reproducibility of the offline entrypoint) |
| **File** | `woais_experiments/run_offline.py`, `woais_experiments/paths.py` |
| **Problem** | Default `write_result` policy is `replace`. `run_offline` clobbers canonical derived results with no confirmation. The CLI runner already uses `forbid` / `resume` / `force`. |
| **Fix** | Offline path logs a warning and points at `run_woais.py`. Policy left as `replace` so the existing publish test can regenerate derived JSON. Timestamped runs still refuse silent overwrite. |
| **Test** | `test_runner.TestAuditRun.test_refuses_overwrite_without_resume_or_force`; warning text in offline suite output |

---

## Reviewed and not treated as bugs

| Topic | Why it stands |
|---|---|
| Oracle / MCKP using `correct` labels | Explicit upper bounds (`oracle_usd`, budget oracle). |
| Keyword router | Frozen `routing.json` assignment; no τ search in this package. Cascade τ is tests-only. |
| In-sample static hull / cost-matched mix | Query-independent bound on the same panel as a frozen heuristic. Documented in `market_from_panel`. Not an OOS-trained static router. |
| Naive mix×μ vs realized per-query USD | Already the point of `aggregate_cost` / `cost_decomposition`. |
| Input vs output token prices | `per_query_cost.py` uses separate $/1M rates. |
| Workload DES | Already `SIMULATED`; rejects `simulated: false`. |
| No sklearn / StandardScaler routing fit | Confirmed; tests ban generation/HTTP clients. |
| Stage 1–2 vs calibration pools | Keyword table does not overlap `router_v2` / Stage 7 fit pools in this package’s loaders. |

---

## Residual (documented, not silent)

| Severity | Item |
|---|---|
| Low | `PYTHONHASHSEED` is **recorded** in `run_environment.json` but not forced before process start (must be set in the shell). |
| Low | Hugging Face xRouteBench cache is gitignored; a clean clone must re-download. Failures should be cache-miss errors, not invented rows. |
| Low | Optional RouteLLM CSV discovery can still look at env / `/tmp`; default offline path is committed Stage 9 JSON. |
| Low | Static hull remains in-sample. OOS static baselines would need a held-out split, which Stage 1–2 does not provide. |

---

## Integrity checks that passed

- No fabricated missing benchmark scores (`fillna` / invented splits). `token_num` is still refused as an I/O split.
- Energy is no longer machine-readable as undifferentiated measured cost.
- Simulated queueing is labeled SIMULATED in code and REPORT.
- Frozen hash manifest still matches (`test_hashes`, offline suite after-write check).
- No paid `chat()` / httpx generation path in the offline runner.

# Final technical audit — EcoLogic / WOAIS experimental stack

Reviewer stance: hostile ACM Middleware / WoAIS systems reviewer. Scope is
**code and experimental design only**. Frozen hashed trees (`raw_results/`,
`router_v2/`, `stage7_10/`) were not modified. This document does not write the
paper and does not optimize presentation.

Date of this pass: 2026-09-10.

---

## Verdict

**Reject as currently packaged for a double-blind systems artifact.**

The Stage 1–2 USD arithmetic on frozen `graded.jsonl` is internally consistent:
Always-Tier-2 dominates the keyword router on realized USD
(`n=364`; EcoLogic `$0.28674677` vs Always-T2 `$0.0417509`; accuracy
`316/364` vs `336/364`). Reconstruction of those USD headlines succeeds.
Frozen SHA256 verification succeeds (`67` files, `0` mismatches, `0` missing).
The keyword router does **not** read evaluation labels at decision time.

That is not enough.

The headline router has **no train/validation/test split**. The static hull is
an **in-sample** mean of the same 364 items. The oracle uses **hindsight
correctness**. Modelled energy uses **paper joule coefficients for retired
slugs** on substitute models. The discrete-event latency path **replays HTTP
RTT as exclusive service time and then adds queueing**. RouteLLM “external”
evidence is a **committed aggregate with a different population** (`n=1307`).
Git history still **deanonymizes the authors**. `validate-artifact` without a
clean clone is defined to fail, and git identity makes a full clone fail
anonymity even if the venv install succeeds.

This repository is **not publication-ready**.

Publication-ready would require every CRITICAL and HIGH **engineering** issue
to be closed. Git identity in `git log` is still CRITICAL and is not closable
without history rewrite, which this audit does not perform.

---

## What was verified in this pass

| Check | Result |
|---|---|
| Unit tests (`python3.13 -m unittest discover -s woais_experiments/tests -t . -q`) | **354 tests, OK** |
| Frozen hashes `EXISTING_RESULTS_SHA256.txt` | **ok**, 67 records, 0 mismatch, 0 missing |
| `reproduce_tables.reconstruct()` | **ok**, 0 failed (USD headlines + two framework aggregates + RouteLLM s9 keys) |
| External public smoke | **ok**, RouteLLM s9 `n_items=1307` |
| `run_woais.py audit --run-id final_audit_verify --no-symlink` | **ok** |
| `run_woais.py validate-artifact --skip-clone` | **exit 1** (required): `anonymity_scan` + `clean_clone_skipped` |
| `run_woais.py validate-artifact` (full clone, no skip) | **exit 1**. `clean_clone_ok=true`, `reconstruction_ok=true`, `secret_hits=0`. Sole remaining `critical`: `anonymity_scan` (`git:log` on the original repository). Elapsed ~123 s. |
| Secret scan | **ok**, 0 hits |
| Anonymity scan after path sanitization | **fail / critical**, **only `git:log`** remains in the tree scan |

Reconstruction **does not** re-derive energy, latency, robustness, stress,
ablations, or serverless tables. A green reconstruct means the USD Stage 1–2
headlines and a small RouteLLM summary still match, not that the artifact is
complete.

The full clone copies a source tree **without `.git`**, installs
`requirements-lock.txt`, and re-runs unit tests plus reconstruction. That
install test **passed**. It does not anonymize `git log` of the repository
the scanner runs against.

---

## Issue catalog

Severity: **CRITICAL** kills the claim or the artifact under double-blind
rules. **HIGH** would be a reject reason for a competent reviewer. **MEDIUM**
is a real defect that does not by itself kill the USD ranking. **LOW** is
hygiene.

Fixes from an earlier `AUDIT_REPORT.md` pass are treated as prior work. This
catalog is the residual plus issues closed in **this** pass.

### CRITICAL

#### C1. No train/val/test for the headline keyword router

| | |
|---|---|
| **Why it matters** | The only population for the paper’s routing claim is `n=364`. Thresholds, keyword lists, and the raw-vs-wrapped surface were not locked on a held-out split. In-sample evaluation cannot distinguish a router from a post-hoc description of the eval set. |
| **Where** | `woais_experiments/routing/policies.py` (`keyword_from_routing`, `build_stage12_policies`). Production decision list: `backend/main.py` `classify_prompt_local_nlp` via `deployment/router.py`. |
| **Fix** | Not fixable in code without a new experiment. Ablations (`routing/ablations.py`) *do* split 50/25/25 for **learned** variants; that protocol is not the Stage 1–2 keyword result. |
| **Test** | Cannot regression-test a missing split. `tests/test_ablations.py` only guards the ablation protocol. |
| **Status** | **Unresolved scientific weakness.** |

The router does not look up `matrix.correct` when assigning a tier. That is a
narrow statement. It is not a train/test discipline.

#### C2. Author identity in git history

| | |
|---|---|
| **Why it matters** | A double-blind artifact that greps `git log` finds configured identity strings (names, emails). Camera-ready anonymity is already broken for anyone who clones the git repo. |
| **Where** | `git log` (scanner: `reproducibility/anonymity_scan.py` `scan_git_metadata`). Configured needles: `reproducibility/anonymity_config.yaml`. |
| **Fix** | History rewrite is out of scope. Clean-clone now **omits `.git`** (`clean_clone_test.py` `IGNORE_DIR_NAMES`) so a *source copy* is not a git identity dump. The published git repository still leaks. |
| **Test** | `tests/test_reproducibility.py` `TestCleanCopy.test_git_metadata_dir_is_not_copied`; `tests/test_runner.py` `TestGitSnapshot` skips when git is absent. |
| **Status** | **Unresolved engineering.** `validate-artifact` remains `ok: false` while `anonymity_scan` is critical. |

#### C3. `--skip-clone` used to report a passing clone (closed this lineage; keep as a reject reason if anyone still cites skip-clone as validation)

| | |
|---|---|
| **Why it matters** | Declaring a publication artifact valid without installing the lockfile in a clean tree is false assurance. |
| **Where** | `reproducibility/validate.py` `run_validate_artifact` |
| **Fix** | `clean_clone.ok=False` when skipped; `critical` includes `clean_clone_skipped`. Process exits non-zero. |
| **Test** | `tests/test_final_audit.py` `TestSkipCloneIsNotAPass`; `tests/test_reproducibility.py` expects exit 1. |
| **Status** | **Fixed in code.** Skip-clone still cannot be sold as artifact validation. |

---

### HIGH

#### H1. In-sample static hull / quality-matched mix

| | |
|---|---|
| **Why it matters** | `market_from_panel` takes mean cost and mean quality on the **same** items being scored. The hull vertex Always-T2 is therefore an in-sample static oracle of expected quality/cost, not an OOS-trained comparator. A text-only router is being compared to a policy that sees eval-set averages. That is the opposite of “static baselines received less information.” If the paper sells this as a fair OOS bake-off, the comparison is invalid. |
| **Where** | `routing/static_baselines.py` `market_from_panel`; used from `routing/policies.py`, `routing/robustness.py`, `external/run_external_audit.py`. |
| **Fix** | Labeled in the docstring. Cannot make the Stage 1–2 hull OOS without a new split. |
| **Test** | `tests/test_static_baselines.py` |
| **Status** | **Unresolved scientific weakness.** Code is honest; the design is not an OOS test. |

#### H2. Oracle uses hindsight labels

| | |
|---|---|
| **Why it matters** | `oracle()` assigns, per item, the cheapest tier that is **correct on that item**. That is not a deployable router. Packing it next to Always-T2 without a role label would be a claim-killer. |
| **Where** | `routing/policies.py` `oracle`; `evaluate_assignment` sets `role=hindsight_oracle` when the name starts with `oracle`. |
| **Fix** | McNemar vs oracle is **excluded from BH** (`bh_family=upper_bound_excluded_from_fdr`). |
| **Test** | `tests/test_final_audit.py` `TestOracleExcludedFromMcnemarFdr` |
| **Status** | **Labeled in code.** Still HIGH if any table treats oracle as a peer policy. |

#### H3. Raw vs wrapped prompt surface

| | |
|---|---|
| **Why it matters** | Headline `ecologic` is `routing[i]["raw"]`. `ecologic_wrapped` exists and is different (Stage 1–2: wrapped accuracy `0.896` vs raw `0.868`, wrapped USD `$0.440` vs raw `$0.287`). Choosing one surface after seeing both is a researcher degree of freedom. It is not silent label leakage. |
| **Where** | `policies.build_stage12_policies` |
| **Fix** | Both policies are computed. Selection of `raw` as the production name is historical. |
| **Status** | **Unresolved scientific weakness** (researcher df). Not an accounting bug. |

#### H4. Energy is modelled; paper rates are for substitute models

| | |
|---|---|
| **Why it matters** | `energy_j = total_tokens/1000 * rate`. Rates `paper_energy_per_1k` are 0.5 / 1.5 / 60.0 for retired Gemma/Apriel-style coefficients, applied to Qwen / gpt-oss / gpt-4o tokens. Joules were never metered. Subst rates roughly double EcoLogic energy (`688.953 J` paper vs `1379.52 J` subst) while Always-T2 stays cheaper under both (`393.86 J` / `525.15 J`). A paper that says “measured energy” is false. |
| **Where** | `accounting/costs.py` `energy_j`, `paper_energy_rates`, `subst_energy_rates`; `configs/models.json`; `run_offline.py` `run_accounting`. |
| **Fix** | Provenance fields and `axis_energy_subst_rates` exist. Token basis is **total_tokens**, not output-only. |
| **Status** | **Unresolved scientific weakness.** Accounting no longer pretends joules were metered. |

#### H5. Serverless / DES uses HTTP RTT as exclusive service time

| | |
|---|---|
| **Why it matters** | Stored `latency_s` is provider HTTP round-trip, which already includes remote queueing. FCFS then adds simulated wait (`latency/serverless.py` `simulate_fcfs`; `workloads/simulator.py`). Sojourn is **not** additive delay on a dedicated GPU. Load conclusions from this DES are simulator artifacts until service time is isolated. |
| **Where** | `latency/serverless.py` `simulate_fcfs` honesty string; `workloads/serverless_model.py`. |
| **Fix** | Labeled `SIMULATED`. Residual after tokens is **not** a labeled cold start (`flag_cold_like`). |
| **Status** | **Unresolved scientific weakness** if sojourn-vs-load is a paper claim. Engineering labeling is in place. |

#### H6. RouteLLM external panel is a different population and is aggregate-first

| | |
|---|---|
| **Why it matters** | Committed Stage 9 table is `n=1307` GSM8K responses, not the EcoLogic `n=364` mix. There is no per-query EcoLogic-vs-RouteLLM pairing. Calling this “the same benchmark, routed” is invalid. |
| **Where** | `external/routellm.py`; `external/run_external_audit.py`; `stage7_10/s9_static_baselines.json`. |
| **Fix** | Adapter records committed summary when no local per-query CSV exists. Does not download. |
| **Status** | **Unresolved scientific weakness.** |

#### H7. Quality permutation p is not a hull test

| | |
|---|---|
| **Why it matters** | Robustness `p_raw` is paired quality vs Always-T2. The primary claim is USD hull domination. A non-significant quality test does not confirm domination; a significant quality gap does not refute it. Selling `p_raw` as the hull test is a wrong estimand. |
| **Where** | `routing/robustness.py` `score_hull` (`p_raw_estimand`, `primary_claim_tested_by_p_raw=False`). |
| **Fix** | Fields added. Survival is now hull-status flips, not “became beneficial.” Current summary: `survives_hull=False` (2 hull flips), `survives_structural=False` (16 flips), `survives_became_beneficial_only=True`. |
| **Test** | `tests/test_final_audit.py` `TestQualityPermIsNotHullTest`; `tests/test_robustness.py`. |
| **Status** | **Fixed in code.** Any prose that says the domination “survives robustness” using the old one-sided flag is still false. |

#### H8. Robustness bootstrap `n_boot=50`

| | |
|---|---|
| **Why it matters** | Fifty query-resamples cannot characterize tail instability. Flip rate `0` on 50 draws is underpowered, not proof of stability. |
| **Where** | `routing/robustness.py` `DEFAULT_N_BOOTSTRAP = 50` |
| **Fix** | Not changed (compute). Rows are labeled sensitivity, not tests. |
| **Status** | **Unresolved.** Underpowered sensitivity, honestly labeled if the paper does not overclaim. |

#### H9. Missing usage previously coerced to $0 (closed)

| | |
|---|---|
| **Why it matters** | `usd or 0` fabricates free inference. Failed chats with missing `usage` looked cheap. |
| **Where** | `benchmark/api.py` `usage_and_cost`; `quality_benchmark_harness.py`; `frozen.py` `matrix_from_calls`; this pass: `stress/analyze_stress.py` `cell_metrics`. |
| **Fix** | Incomplete cells dropped on ingest. Stress `cost_per_query` is now mean over **finite costs only**; missing-as-zero is a separate field. Pareto ranking **refuses** `None` cost instead of treating it as `$0`. Simulated DES rows no longer turn missing `inference_cost` into `0.0`. |
| **Test** | `tests/test_final_audit.py` `TestUsageMissingIsNotZeroUsd`, `TestStressCostDoesNotTreatMissingAsZero`; `tests/test_audit_fixes.py` `TestNoZeroFillMissing`. |
| **Status** | **Fixed in live code.** Frozen `graded.jsonl` was already complete for Stage 1–2; headline USD did not move. |

#### H10. Home-directory paths in derived JSON (closed this pass)

| | |
|---|---|
| **Why it matters** | Absolute workspace paths in `index.json` / `run.json` deanonymize a machine user. Anonymity scan was critical on derived results, not only git. |
| **Where** | `frozen.py` `to_jsonable`; `paths.py` `public_relpath`; writers that used `str(write_result(...))`. |
| **Fix** | JSON serialization converts in-repo paths to repo-relative and redacts home/temp absolutes. Writers store `public_relpath`. `run.json` / CLI summary use relative paths. |
| **Test** | `tests/test_final_audit.py` `TestArtifactPathsAreNotHomeAbsolute`; `tests/test_runner.py` asserts run.json has no `/Users/` or `/home/` prefixes. |
| **Status** | **Fixed for newly written artifacts.** Historical run directories under `results/runs/` may still contain old absolutes; that glob is excluded from the anonymity scan. Git identity remains. |

#### H11. Process-first “cold start” is not a platform cold start

| | |
|---|---|
| **Why it matters** | `ProcessLifecycle` labels request index 1 as `cold`. A warm container reused by the platform is still “cold” on first request in-process. Idle gaps do **not** relabel. Using this as Cloud Run / Lambda cold-start measurement is a misclassification. |
| **Where** | `deployment/lifecycle.py` |
| **Fix** | Basis strings `first_request_in_process` / `subsequent_request_same_process`. Residual-outlier flags in serverless are separately labeled as proxies. |
| **Status** | **Labeled.** HIGH if a paper reports these as cloud cold starts. |

#### H12. Retries: billed cost is final attempt only

| | |
|---|---|
| **Why it matters** | Nested retries can consume tokens/USD on failed attempts. Logs store `cost_scope=final_attempt_only`. Realized provider cost is not a sum of attempts. |
| **Where** | `deployment/infer.py` `call_with_retries`, `RequestRecord.cost_scope`. |
| **Fix** | Labeled. Not a silent change of the Stage 1–2 frozen USD (those calls were stored per completed cell). |
| **Status** | **Unresolved for live deployment accounting** unless the paper claims final-attempt cost only. |

#### H13. Success-conditioned latency vs excluding failures from the claim

| | |
|---|---|
| **Why it matters** | Primary p95 is successful requests. Failures remain in `p95_*_including_failures` and in failure-rate denominators. That is correct **if** both are reported. Reporting only success p95 while timeouts exist understates user-visible delay. |
| **Where** | `deployment/analyze_deployment.py` `summarize_group`; `stress/analyze_stress.py` `cell_metrics`. |
| **Fix** | Dual summaries. Stub/dry-run marked `cloud_measured=False`, `environment_kind=local_stub`. |
| **Test** | `tests/test_final_audit.py` `TestDeploymentLatencySuccessConditioned`, `TestStubIsNotCloud`, `TestStressSuccessLatency`. |
| **Status** | **Fixed in code.** HIGH if only the success p95 is quoted for a loaded system. |

#### H14. Reconstruction coverage is not the whole artifact

| | |
|---|---|
| **Why it matters** | `reproduce_tables.reconstruct` checks Stage 1–2 **USD** policy costs/accuracy, RouteLLM s9 keys, and two framework per-query sums. Energy, latency CDFs, robustness CSVs, stress analyses, ablations, and serverless JSON are **not** in the reconstruction set. “Tables reconstruct from raw” is true only for that subset. |
| **Where** | `reproducibility/reproduce_tables.py` `reconstruct` |
| **Fix** | Not expanded in this pass. |
| **Status** | **Unresolved engineering/science gap.** |

#### H15. `router_v2` energy budget informed by eval-set EcoLogic energy

| | |
|---|---|
| **Why it matters** | Cascade / `router_v2` calibration that sets an X% energy budget from test-set EcoLogic totals is peeking. τ-on-calibration-only is cleaner. This audit did not re-fit `router_v2`; the frozen tree is read-only. |
| **Where** | Frozen `router_v2/` (not modified). Live WOAIS stack evaluates the **keyword** router for Stage 1–2. |
| **Status** | **Unresolved** for any claim that uses `router_v2` as a confirmatory OOS result. |

#### H16. Multiple-comparison leftovers

| | |
|---|---|
| **Why it matters** | McNemar BH is over deployable policies vs EcoLogic (quality). Robustness BH is over structural rows. Those families are not the USD hull test. Dependent quality p-values in a mixed bag still overstate how many “significant” quality gaps exist. |
| **Where** | `routing/policies.py` `evaluate_policies`; `routing/robustness.py`. |
| **Status** | **Partially mitigated.** Residual: family definition is still a researcher df. |

#### H17. Undeclared PyYAML (closed)

| | |
|---|---|
| **Where** | `woais_experiments/requirements.txt` |
| **Fix** | `PyYAML` declared. |
| **Test** | `TestRequirementsDeclarePyYAML` |
| **Status** | **Fixed.** |

#### H18. Schema: `as_float(True)→1.0` and generic `id` alias (closed)

| | |
|---|---|
| **Why it matters** | Boolean quality would become 1.0. Column `id` is too generic for query identity (row index collision). |
| **Where** | `external/schema.py` |
| **Test** | `TestSchemaBooleansAndIds` |
| **Status** | **Fixed.** |

#### H19. Permutation denominator used all draws including non-finite (closed)

| | |
|---|---|
| **Where** | `statistics/paired_tests.py` |
| **Fix** | Denominator is finite nulls. |
| **Status** | **Fixed.** Anti-conservative p-values are no longer the default. |

#### H20. HTTP 429 classified as 5xx (closed)

| | |
|---|---|
| **Where** | `deployment/provider.py` |
| **Fix** | `error_type=http_429`. |
| **Test** | `TestRetryTaxonomyAndCostScope` |
| **Status** | **Fixed.** |

---

### MEDIUM

#### M1. `run_offline` overwrite policy is `replace`

Silent overwrite of `woais_experiments/results/**` during `test_offline_suite` /
`run_offline.run()`. The CLI runner uses forbid/resume. A reviewer who runs
the unittest suite **will mutate derived JSON**. Frozen hashes still pass
because hashed files are outside that write root.

#### M2. Energy token basis vs USD token basis

USD uses prompt and completion rates separately (`usd_from_tokens`). Energy
uses **total** tokens × a single J/1k rate. If the source paper’s J/1k was
decode-only, prompt tokens inflate every policy. Ranking vs Always-T2 can
still hold; absolute joules should not be compared to metered GPU energy.

#### M3. Percentile index is nearest-rank, not Hyndman–Fan type 7

`accounting/costs.py` `_percentile` and `latency/analyze_latency.py`
`percentile` share `round((q/100)*(n-1))`. Bootstrap CIs use NumPy linear
quantiles. Point p95 and CI endpoints are not the same functional.

#### M4. xRouteBench Hub fetch vs “no network” claims

`external/xroutebench.py` downloads the public dataset into
`woais_experiments/data/xroutebench/hf_cache/`. That is not a paid model API,
but it is a network dependency. `token_num` is **not** used as a cost
substitute (`token_num_not_used_for_cost`). Bare `latency` requires an
explicit unit.

#### M5. Deployment `lru_cache` on router config

`deployment/router.py` caches classifier/config loads (`maxsize=1`). This is
not response caching of LLM completions. Provider-side caches are unmeasured.
Do not attribute latency changes to routing without ruling out provider cache.

#### M6. Concurrent HTTP client

`deployment/benchmark.py` uses `ThreadPoolExecutor`. `ProcessLifecycle` is
locked. A shared `urllib` opener under threads is a classic source of
intermittent timing artifacts. Dry-run stubs will not show it.

#### M7. Simulated stress rows default to HTTP 200

`stress/stress_test.py` `simulated_rows` stamps `http_status=200` unless the
DES emits a failure. Timeouts inside the DES must be explicit or they enter
the success latency population.

#### M8. Occupancy uses `scheduled_arrival_s or 0.0`

Missing arrivals collapse to t=0 and inflate occupancy. Fine when timestamps
are always present (current generators).

#### M9. Ablations are a different estimator than production EcoLogic

`routing/ablations.py` fits TF-IDF / logistic / trees with train-only
preprocessing and val-only τ. Production EcoLogic is a keyword decision list.
Ablation “full” is gated to match production on the eval texts, but learned
variants are not the deployed system. Reporting ablation lifts as EcoLogic
component importance is a method mismatch.

#### M10. Clean-clone destination paths

Clone payload used to store host temp paths. Now `"<redacted-absolute>"`.

#### M11. Python `executable` in environment.json

Home venvs would leak. `environment_capture.python_info` redacts home
absolutes; `/Library/Frameworks/...` is left as-is (not an author home path).

#### M12. BH bag still mixes related quality tests

See H16. Remaining MEDIUM after the oracle exclusion.

---

### LOW

#### L1. `+∞` JSON encoding

Non-finite floats serialize as `"Infinity"` strings, not `null`. Oracle
break-even no longer looks “missing.”

#### L2. Router overhead is $0 in config

Headline USD adds `overhead_per_query` for EcoLogic names. Config is `$0`, so
published USD is unchanged. The wiring is tested.

#### L3. Dual rate tables historically (`api.MODELS` vs `models.json`)

Production path reads `configs/models.json` only.

#### L4. CLI used to print absolute `run_dir`

Now prints `public_relpath`. Logs are not the double-blind artifact; git is.

#### L5. Anonymity scanner excludes `results/reproducibility/**` and `results/runs/**`

Those directories can still contain host paths from older runs. They are not
in the scan, so they are a review residual if someone publishes the whole
`results/` tree.

---

## Thirty-point checklist (requested failure modes)

| # | Failure mode | Finding |
|---|---|---|
| 1 | Leakage | Keyword router does not read labels. **In-sample design** is the leak analogue (C1, H1). Ablation learned models: train-only fit, val-only τ, test scored once (`select_ablation_on_test=False`). |
| 2 | Train/val/test | **Absent** for Stage 1–2 keyword (C1). Present for ablations. |
| 3 | Router tuning on test | No automated τ search on Stage 1–2. Keyword lists are frozen in production source. Researcher df: raw vs wrapped (H3). |
| 4 | Cost-accounting mistakes | Frozen USD identity holds for complete cells. Missing-as-zero bugs were real and are closed in live code (H9). Energy is not USD (H4). |
| 5 | Pricing inconsistencies | Live path: `configs/models.json` only. Eval slugs ≠ product slugs (documented in that file). |
| 6 | Token-count inconsistencies | USD: in/out split. Energy: total tokens (M2). xRouteBench does not split `token_num` into cost. |
| 7 | Unmeasured as measured | Energy modelled. Serverless cold/idle knobs unmeasured. Residual outliers ≠ cold start. Stub ≠ cloud. |
| 8 | Inappropriate tests | CI-as-test was removed for external bootstrap. `p_raw` is not a hull test (H7). |
| 9 | Wrong bootstrap unit | Query-level paired resample (`statistics/bootstrap.py`). Correct unit for item-level routing. `n_boot=50` on robustness is too small (H8). |
| 10 | Multiple comparisons | BH on deployable McNemar and robustness structural rows. Oracle out of FDR. Residual family df (H16). |
| 11 | Cherry picking | Raw headline vs wrapped (H3). Robustness “survives” must not use `became_beneficial_only`. Reconstruction ignores most tables (H14). |
| 12 | Unsupported exclusions | Tail drops labeled `exclusion_basis=ecologic_realized_usd` / `ecologic_completion_tokens`. Incomplete calls dropped, not zero-filled. |
| 13 | Mismatched populations | Stage 1–2 `n=364` vs RouteLLM s9 `n=1307` (H6). Ablation test `n=91` is a subset protocol, not the headline n. |
| 14 | Static informational disadvantage | **Not found.** Hull is in-sample; static sees eval means. Unfairness runs **against** the router (H1). |
| 15 | Oracle impossible information | Yes. Labeled `hindsight_oracle` (H2). |
| 16 | Simulator-driven conclusions | Yes if DES sojourn is treated as deployment evidence (H5). |
| 17 | Cloud measurement artifacts | Dry-run/stub forced `cloud_measured=False`. Process-first cold ≠ platform cold (H11). |
| 18 | Retries silently changing cost | Final-attempt cost, labeled (H12). |
| 19 | Caching affecting latency | No local completion cache. Config `lru_cache` only (M5). Provider cache unmeasured. |
| 20 | Concurrency bugs | Thread pool + urllib (M6). Lifecycle counter is locked. |
| 21 | Race conditions | Same as 20. Not demonstrated in stubs. |
| 22 | Cold-start misclassification | Process index 1; residual outliers. Labeled (H11, H5). |
| 23 | Timeouts excluded from summaries | Primary p95 is success-conditioned; including-failures field exists (H13). |
| 24 | Failed requests excluded from denominators | Failure rate uses full issued set. Cost mean no longer treats missing USD as $0 (H9). |
| 25 | External benchmark parsing | Bool≠float; no generic `id`; bare latency needs unit; `token_num` unused for cost. |
| 26 | Missing-value handling | No zero-fill on ingest. External price-table fill labeled not metered. |
| 27 | Output-token accounting | USD uses completion tokens × output rate. Energy uses total tokens. |
| 28 | Reproducibility failures | Frozen hashes OK. Reconstruct OK on a **narrow** set. Skip-clone fails. Git anonymity fails. Full clone is the only honest install test. |
| 29 | Hidden dependencies | PyYAML now listed. Ablations need sklearn. xRouteBench needs `datasets` + Hub. |
| 30 | Accidental deanonymization | **Git log still CRITICAL.** Derived JSON home paths closed this pass. |

---

## Fixes applied in this pass (engineering)

1. `to_jsonable` / `public_relpath`: repo-relative artifact paths; home/temp redaction.
2. Run metadata (`run.json`, CLI summary, validate `written` map) no longer stores home absolutes.
3. Ablation / external / xRouteBench / deployment writers store `public_relpath`.
4. Clean clone **does not copy `.git`**.
5. Tests no longer require git identity in a source-only tree.
6. Stress `cost_per_query` denominator is finite costs; Pareto refuses missing cost.
7. Simulated inference cost is `None` when the DES omits it, not `$0`.
8. Environment `executable` redacts home venvs.
9. `AUDIT_REPORT.md` example path language no longer trips the home-path regex.

Regression tests: `woais_experiments/tests/test_final_audit.py` (extended),
`tests/test_reproducibility.py` (`TestCleanCopy`), `tests/test_runner.py`
(path + optional git).

---

## Residual scientific weaknesses (cannot be patched by JSON flags)

1. **No held-out evaluation of the keyword router** on the headline population.
2. **In-sample static hull** as the cost/quality comparator.
3. **Hindsight oracle** in the policy suite (even if labeled).
4. **Modelled energy** with substitute-model rates and total-token basis.
5. **DES double-counting** of delay if HTTP RTT is used as exclusive service.
6. **RouteLLM n=1307** is not the EcoLogic item set.
7. **Raw vs wrapped** is a researcher degree of freedom.
8. **`router_v2` budgets** that peek at eval energy, if cited as confirmation.
9. **Robustness** does **not** leave the domination claim invariant under hull-status (`survives_hull=False`).

A paper that states “Always-T2 is cheaper and more accurate than keyword
EcoLogic on this frozen 364-item panel under committed USD rates” is
supported by the arithmetic. A paper that states “we evaluated a held-out
router against fair OOS static baselines, measured energy, and validated
behavior under realistic serverless queueing, replicated on RouteLLM” is
**not** supported.

---

## Publication readiness

**No.**

Closed HIGH engineering in live code does not include:

- git author identity (CRITICAL, unresolved);
- incomplete reconstruction coverage (HIGH);
- underpowered robustness bootstrap (HIGH, scientific/compute);
- unlabeled use of oracle / in-sample hull / DES / energy (HIGH if claimed).

Do not describe this tree as a WOAIS-ready artifact while `git log` matches
the anonymity config and while `--skip-clone` is the validation that was
actually completed in a given environment.

---

## Verification commands

```text
python3.13 -m unittest discover -s woais_experiments/tests -t . -q
python3.13 -c "from woais_experiments.frozen import verify_frozen_hashes; print(verify_frozen_hashes())"
python3.13 -c "from woais_experiments.reproducibility.reproduce_tables import reconstruct, smoke_external_public; print(reconstruct()['ok'], smoke_external_public())"
python3.13 run_woais.py audit --run-id final_audit_verify --no-symlink
python3.13 run_woais.py validate-artifact --skip-clone --run-id final_audit_validate --no-symlink
python3.13 run_woais.py validate-artifact --run-id final_audit_fullclone --no-symlink
# this session: exit 1; clean_clone_ok=true; critical=["anonymity_scan"] only
```

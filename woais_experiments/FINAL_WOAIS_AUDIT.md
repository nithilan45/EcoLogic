# Final WoAIS Technical Audit

Reviewer stance: adversarial senior ML systems researcher for WoAIS at ACM/IFIP Middleware 2026. Scope is **code, experiments, and result validity only**. This document does not write or edit a paper. Frozen hashed trees (`raw_results/`, `router_v2/`, `stage7_10/`) were not modified.

Date: 2026-09-10.

---

## Executive verdict

**MAJOR WORK REQUIRED**

Stage 1–2 **measured USD** on the frozen 364-item panel is internally consistent and independently reconstructible: Always-Tier-2 dominates the keyword EcoLogic router on both cost and accuracy. That is a real negative result, not an accounting bug.

It is not a WoAIS-ready serverless systems artifact. The headline keyword router has **no held-out split**. The learned held-out protocol is mechanically clean and then **collapses to Always-Tier-2** (quality advantage 0, identical cost). There is **no paid Cloud Run/Lambda run** tagged honestly as `MEASURED_REAL_DEPLOYMENT`; committed v2 files are a local stub dry-run. Energy is modelled. The discrete-event “serverless” path is simulated and double-counts HTTP RTT. RouteLLM query-level assignments are **unavailable**. xRouteBench has **no learned-router decisions**. Git history deanonymizes authors, so `validate-artifact` remains `ok: false`.

Tests passing is not technical readiness.

---

## Test status

- tests run: **430** (`python3.13 -m unittest discover -s woais_experiments -p 'test*.py' -t . -q`)
- passed: **430**
- failed: **0**
- errors: **0**
- skipped: **0** (this checkout; clean-clone discovery of `woais_experiments/tests` only skips 1 git-dependent test)

Engineering fixes in this pass (dry-run suite tags, same-PID cold-start refuse, infeasible mix not treated as cost-matched) are covered by new regressions in `tests/test_deployment_v2.py` and `tests/test_heldout.py`.

---

## Historical artifact integrity

- files checked: **67** (`woais_experiments/EXISTING_RESULTS_SHA256.txt`)
- mismatches: **0**
- missing: **0**
- result: **ok**

Unexplained historical modification: **none**. Frozen trees were not rewritten to make tests pass.

---

## Critical issues

### C1. Keyword EcoLogic router evaluated in-sample on the headline panel

- **issue:** Production `classify_prompt_local_nlp` has no train/val/test. Keyword lists, thresholds, and raw-vs-wrapped surface were not locked on a held-out split. The paper-facing Stage 1–2 table scores that heuristic on the same `n=364` used to inspect it.
- **evidence:** `backend/main.py` `classify_prompt_local_nlp`; `benchmark/router.py` writes `raw_results/routing.json`; `woais_experiments/routing/policies.py` `build_stage12_policies` uses `routing[i]["raw"]`. No split protocol exists for this policy. The router does **not** read `matrix.correct` at decision time — that is a narrower statement than “no leakage.”
- **affected result:** Every Stage 1–2 EcoLogic accuracy/USD/latency headline on `n=364`.
- **fix/status:** **Unresolved scientific.** Not fixable without a new experiment. Held-out `heldout_v1` does not rehabilitate this policy (on that split the heuristic **loses** to Always-T2).

### C2. Author identity in Git history (double-blind artifact)

- **issue:** `git log` contains configured identity strings (names, emails). `validate-artifact` treats anonymity hits at `severity: critical`.
- **evidence:** `woais_experiments/reproducibility/anonymity_scan.py` `scan_git_metadata`; `anonymity_config.yaml` identities. Full `validate-artifact` this pass: `ok: false`, `critical: [anonymity_scan]`, `secret_hits: 0`, `reconstruction_ok: true`, `clean_clone_ok: true`. A mid-pass scan also hit a documentation example in `POST_FIX_VALIDATION.md` (`macos_home` regex); that string was removed. Current tree scan: **50 hits, files = `git:log` only.**
- **affected result:** Artifact evaluation / double-blind packaging, not USD arithmetic.
- **fix/status:** **Unresolved engineering.** History rewrite is out of scope. Clean-clone omits `.git` so a source zip is not a git identity dump. The git repository still leaks.

### C3. No genuine `MEASURED_REAL_DEPLOYMENT` cloud experiment

- **issue:** The v2 suite was designed to record paid serverless inference. Committed `results/deployment_real_v2/` is `generation_mode: dry_run`, `paid: false`, `serverless: false`, `environment_kind: local_stub`, `base_url: null`, **120 stub records**. There is no Cloud Run/Lambda timing or provider USD in this tree.
- **evidence:** `woais_experiments/results/deployment_real_v2/run.json`. Query set n=80 frozen; executed subset yields 120 local stub rows.
- **affected result:** Any serverless / cold-start / real-deployment claim citing v2.
- **fix/status:** **Unresolved scientific.** Requires a new paid run with `--allow-api --allow-cloud --max-cost-usd` against a non-loopback URL. Code now refuses to tag dry-run as `MEASURED_REAL_DEPLOYMENT` (this pass). **Committed JSON still carries the old suite tag** until a new run overwrites it; those files are not in the frozen hash manifest.

### C4. Held-out selected “learned router” is Always-Tier-2

- **issue:** After a frozen 60/20/20 protocol, validation selected logistic with `val_quality_advantage: 0.0`. On test, logistic equals `cost_matched_static` / `always_cheap` / `random_matched` to numerical identity (quality 0.8889, USD 0.0001526, strong fraction 0). Claiming that a learned router beats a cost-matched static policy on this experiment is false.
- **evidence:** `results/heldout/frozen_config.json`; `results/heldout/final_test_results.csv`. Independently recomputed: logistic quality and USD match cost-matched static exactly.
- **affected result:** Held-out learned-router vs static.
- **fix/status:** **Unresolved scientific (null result must remain visible).** Do not retune after seeing test. `TEST_EVALUATED.lock` is advisory (deletable; `--new-experiment` overwrites the same prefix).

---

## High issues

### H1. In-sample static hull on Stage 1–2

- `market_from_panel` uses unconditional means on the **same** 364 items. Always-T2 is an in-sample quality/cost vertex, not an OOS-trained comparator. Honest in `static_baselines.py` docstring. Invalid as a generalization bake-off.

### H2. Oracle / MCKP is hindsight; published sweep is approximate

- `policies.oracle` assigns the cheapest **correct** tier per item. Role is `hindsight_oracle`; McNemar vs oracle is excluded from BH. `results/routing/oracle_budget_sweep.json` uses `"method": "approx"` / greedy. Exact vs greedy is unit-tested on toys; the n=364 frontier is **not** certified exact. Must never be tabled as a deployable policy.

### H3. Researcher degrees of freedom: raw vs wrapped

- Wrapped EcoLogic: acc 0.896 / $0.440 vs raw 0.868 / $0.287. Production name is raw. Both are computed; choosing the published surface after seeing both is a df, not silent label leakage.

### H4. Energy is modelled, including substitute-model joule coefficients

- `run_offline` sets `energy_is_modelled: True`. Paper rates for retired slugs are applied to evaluation substitutes. Must not be cited as measured joules.

### H5. Serverless DES replays HTTP RTT then adds queueing

- Stored `latency_s` is full HTTP round trip. The simulator uses it as exclusive service time and then queues. Cold-start delay in YAML is a knob (`cold_start_delay_s: 3.0`), not a platform measurement. Outputs are labeled `SIMULATED` in code; easy to mis-cite as Cloud Run.

### H6. RouteLLM query-level sweep did not run

- `results/external_router/summary.json`: `available: false`. Scores not cached; `transformers` not installed. Threshold grid **is** predefined (`threshold_grid.json`, 21 linspace points, `frozen_before_quality_cost: true`). Stage 9 `s9_static_baselines.json` is a **committed aggregate** on a **different population** (`n=1307` GSM8K), not paired to EcoLogic’s 364. `historical_stage9_pointer` marks `not_this_protocol`.

### H7. xRouteBench has no learned-router assignments

- Combo JSON: `router_frontier: skipped`, `has_router_assignment: false`. Accounting uses unconstrained oracle vs static. Missing tokens/cost are dropped, not imputed. Oracle must not be described as external learned-router evidence.

### H8. Robustness bootstrap underpowered; p_raw is the wrong estimand for hull domination

- `results/routing/robustness.json`: `n_bootstrap: 50` (library default 10_000). `n_structural_flips: 16`, `survives_hull: false`, `survives_became_beneficial_only: true`. The last flag does **not** falsify domination. `p_raw` tests paired quality vs Always-T2, not USD hull domination (`primary_claim_tested_by_p_raw: false`).

### H9. `reconstruct()` is a thin headline checksum

- Verifies Stage 1–2 USD/accuracy, two framework aggregates, RouteLLM s9 keys. Does **not** reconstruct energy, latency CDFs, robustness, ablations, oracle sweep, stress, serverless DES, held-out, xRouteBench, or deployment. `result_provenance.json` covers that subset only.

### H10. Held-out data already used in `router_v2/`

- Same 1200-item pool drove earlier train/calibrate/feature work. New 60/20/20 split is ID-disjoint internally; **method contamination** from prior peeking at the pool is not controlled. Soft Jaccard 0.7–0.9 near-duplicates: 42 cross-split pairs (protocol only merges ≥0.9).

### H11. Retry cost is final-attempt only

- `cost_scope: final_attempt_only`. Failed attempts can be invisible to USD. Documented; full attempt accounting needs new logging.

### H12. Stress high multipliers can be dropped by safety caps

- Config lists 1×–50×. Measured stub cap drops 25×/50×; paid cap drops 5×+. Recorded as `multipliers_executed` vs requested — easy to over-claim “ran 50×” from YAML alone.

**Engineering HIGH closed this pass**

- Dry-run / local stub no longer **written** as `MEASURED_REAL_DEPLOYMENT` (`v2_measurement_type` → `DRY_RUN_LOCAL` / `MEASURED_PAID_LOCAL` / `MEASURED_REAL_DEPLOYMENT`).
- Same-OS-pid HTTP recycle no longer resets `ProcessLifecycle` (`keep_lifecycle_if_same_process`).
- `cost_match_feasible: false` no longer emits “savings vs cost-matched static.”

Committed `deployment_real_v2/*.json` still show the old suite tag (not a frozen-hash file; not regenerated).

---

## Medium issues

- Naive mix×mean **understates** EcoLogic realized USD by ~6.95% on Stage 1–2 (covariance identity reconciles). Mix×mean narratives bias EcoLogic’s cost **down**.
- Cost-matched static matches **expected cost on the hull**, not routing-rate mix. With a singleton hull (T2), “same quality” still prices at T2 even when EcoLogic quality is lower.
- Dual cost stacks: policy table uses stored `matrix.usd`; framework recomputes tokens×config. Match on Stage 1–2; would diverge if rates change post-hoc.
- Incomplete calls dropped (not zero-filled); Stage 1–2 has 0 incomplete. `analyze.py` skip is quieter than `matrix_from_calls`’s counter. Latency `selected_mean` can drop NaNs.
- Ablations: 50/25/25 resplit of the **same** n=364 (`n_test: 91`), not the held-out pool. Learned ablations train to **hindsight oracle** labels. `no_semantic` / `no_confidence` are degenerate copies of full (flagged).
- Deployment success-conditioned p95; failures in rate / `costs_all` but not primary latency.
- Package-check: 7 warnings (`.env.example`; duplicate-byte CSVs/JSONs). 0 critical.
- `TEST_EVALUATED.lock` does not hash fitted weights; `protocol_source_sha256` is now recorded on **new** locks only. Existing lock was not rewritten (would require re-eval).
- Framework “sign flip” tables are synthetic demos, not an exhaustive Stage 1–2 policy-pair inventory. External flip CSVs are headers-only because the sweep did not run.

---

## Low issues

- Overwrite / run-dir errors now serialize via `public_relpath` (repo-relative or `<redacted-absolute>`).
- Cold-like residual detector uses `np.isclose` so machine-precision linear fits are not outliers (`latency/serverless.py`). Still a **proxy**, not a labeled cold start.
- Path serialization uses pathlib containment, not a `/Users`/`/home` prefix list.
- Wilcoxon vs permutation p-values coexist; CI overlap is documented as not a test.
- ThreadPoolExecutor request logs are sorted by query id after concurrent HTTP (ordering is deterministic post-hoc).

---

## Held-out evaluation verdict

**Mechanical protocol: pass. Scientific claim that a learned router beats static on this split: fail (null).**

| Check | Result |
|---|---|
| Frozen `split_manifest.json` + SHA | Pass |
| Train 720 / val 237 / test 243; grouped; Stage12 disjoint | Pass |
| TF-IDF/scaler train-only | Pass |
| τ and HPs on val only; `test_used: false` | Pass |
| Exact / Jaccard≥0.9 cross-split dups | 0 |
| Soft paraphrases Jaccard 0.7–0.9 | 42 pairs (MEDIUM) |
| `TEST_EVALUATED.lock` | Blocks default CLI; bypassable |
| Selected method on test | **Identical to Always-T2 / cost-matched static** |
| Production EcoLogic heuristic on this split | Quality 0.860 vs static 0.889; ~5.7× USD |

Do not retune. Do not shop `final_test_results.csv` for a non-selected row (tree/threshold/heuristic are all written).

---

## Baseline fairness verdict

**Stage 1–2 USD comparison of EcoLogic vs Always-T2 is apples-to-apples on the same 364 complete items.** Failures are not silently dropped on this panel. Always-cheap / always-strong / random / hull interpolation are implemented; hull does not extrapolate (`feasible=False` outside `[c_lo,c_hi]`).

**Unfair if sold as OOS:** the hull sees eval-set means. Cost-matching is cost-on-hull, not routing-rate matching. On this price vector T2 Pareto-dominates T1 and T3, so “cost-matched static” **is** Always-T2.

Held-out cost-matched mix was frozen from **validation** (good) and is Always-T2.

Deployment v2 “cost-matched” mix was **infeasible** (`mix_fraction_strong: 0.0`, EcoLogic estimate below cheap). Analysis previously still emitted savings; **code now refuses that comparison.**

---

## Per-query accounting verdict

**Sound for Stage 1–2 measured USD.**

```
realized_i = in_i * p_in(model_i) + out_i * p_out(model_i) + overhead_i
```

Units: config USD/million → `/1e6` per token. Stored `usd` vs tokens×rates: 0 mismatches on graded.jsonl. Overhead config = $0. Naive − realized = Σ Cov(1{t=m}, c_m); EcoLogic residual ~1e-19, `reconciles: true`. Naive understates EcoLogic ~6.95%.

Independently reconstructed:

| Quantity | Value |
|---|---|
| EcoLogic USD | 0.28674677 |
| Always-T2 USD | 0.0417509 |
| EcoLogic correct | 316/364 |
| Always-T2 correct | 336/364 |
| Ratio | 6.868… |
| Always-T2 dominates | true |

---

## External learned-router verdict

**Independence of the routing rule is implemented; the experiment did not produce assignments.**

`route_by_score` uses score ≥ threshold only. Oracle is a separate `ORACLE_ASSIGNMENT` path. Grid frozen before quality/cost. Current artifacts correctly refuse to invent scores (`available: false`). Stage 9 aggregates are a different protocol/population. **No query-level routing-rate / realized-cost / cost-matched-static curve exists to recompute across all 21 thresholds.**

---

## xRouteBench verdict

**Candidate-model execution tables, not router decisions.** Parsing refuses `token_num` as I/O split; incomplete cells dropped. Oracle vs static is labeled skipped for the router frontier. Do not cite combo `oracle_quality_advantage` as EcoLogic or RouteLLM evidence.

---

## Oracle/MCKP verdict

Hindsight, label-using, correctly role-tagged. Published Stage 1–2 budget sweep is **approx**. Exact/MILP vs greedy tested on small synthetic instances (`test_oracle.py`, greedy can be suboptimal). `fraction_of_available_routing_value_captured` at router cost may be non-binding (budget ≥ unconstrained). Mid-budget points are not certified.

---

## Statistical validity verdict

Primary continuous path is **query-paired** bootstrap; BH on deployable McNemar; oracle excluded from FDR. No silent unpaired t-test in the main policy table. **Underpowered** robustness `n_boot=50`. Ablations use group-level CIs on held-out (`bootstrap_unit: group`, 170 groups) — better. Do not interpret CI overlap as a test. Do not treat `survives_became_beneficial_only` as a hull-survival result. Multiple thresholds on a future RouteLLM sweep would need FDR; that sweep is empty.

---

## Real deployment verdict

**No real serverless deployment was measured.** Committed v2 is local stub dry-run. Timing fields exist (`router_decision_ms`, `provider_request_ms`, `end_to_end_ms`, `queue_ms`); TTFT is often null on stubs. Energy is never labeled measured. Cold-start is process-init only in `lifecycle.py`; this pass **stops** fake same-PID resets. Clock: `time.perf_counter` for phase walls; lifecycle uses `time.monotonic`.

---

## Serverless simulation verdict

Labeled `SIMULATED`. Heap event ordering with monotonic seq. Idle timeout / cold delay are config knobs. Deterministic unit tests exist (`test_workload_simulator.py`, `test_serverless.py`). Not a substitute for Cloud Run. Residual “cold-like” flags are proxies.

---

## Stress-testing verdict

Measured vs simulated namespaces are separated (`assert_homogeneous`). High multipliers can be **capped** rather than run. Failures belong in failure-rate denominators; primary p95 is success-conditioned. Saturation “max sustainable throughput” is only defined for the tested stub/paid environment — not a platform SLA.

---

## Ablation verdict

Protocol: same internal split, val-only selection, `select_ablation_on_test: false`. Scope: **in-panel resplit**, not held-out. Oracle-supervised learned variants are ceilings. Degenerate feature ablations are flagged. All variants stored. Do not interpret feature flags causally.

---

## Falsification verdict

Robustness grid **does** search unfavorable prices, overhead, budgets, drop-tails, weaker statics, and **records flips** (`n_structural_flips: 16`, hull survival false). That is real negative evidence. Soft “became beneficial only” flag can **hide** non-survival if quoted alone. Held-out null and Always-T2 domination are the strongest falsifiers of “routing saves money.”

---

## Reproducibility verdict

- Lockfile install in a clean tree: **exercised** (`validate-artifact` clean-clone).
- Frozen hashes: **67/67**.
- Aggregate reconstruction of the **USD subset**: **ok**.
- Full `validate-artifact`: **`ok: false`** solely `anonymity_scan` (`git:log`) on this checkout.
- Secrets: **0**.
- Skip-clone is **not** a pass (`clean_clone_skipped` is critical).
- No undeclared paid-API env required for offline tests.

---

## Anonymity verdict

**Fail on a git clone; pass on a git-less source copy.** After removing a documentation path example that tripped `macos_home`, the working-tree scan is `git:log` only. Scanner critical because `scan_git_metadata: true` and history contains identities. Do not weaken the scanner. Do not claim a clean double-blind git artifact. On-disk `results/reproducibility/validate.json` is the full-clone run from this pass (`anonymity_files` may still list the now-fixed markdown until the validator is re-run).

---

## Result provenance verdict

**Safe for Stage 1–2 USD/accuracy headlines; unsafe as a blanket “all tables reconstruct from raw.”**

Traced this pass (10 values, all matched):

1. EcoLogic USD 0.28674677 ← `graded.jsonl` + `routing.json` + `evaluate_policies`
2. Always-T2 USD 0.0417509
3. EcoLogic correct 316
4. Always-T2 correct 336
5. n=364
6. cost ratio 6.868…
7. stored `accounting/stage12.json` EcoLogic USD
8. stored Always-T2 USD
9. held-out logistic quality = cost-matched static 0.8889
10. held-out logistic USD = cost-matched static 0.0001526

`reconstruct()` ok, n_failed=0. Energy/latency/robustness/deployment/xRoute/RouteLLM-query-level **not** in that chain.

---

## Remaining scientific limitations

- n=364, no OOS for the keyword router.
- T2 dominates T1/T3 on this price vector — routing has little room.
- Held-out learned methods collapse to the cheap static policy.
- Evaluation models ≠ production slugs.
- Latency is HTTP RTT, not isolated GPU service time.
- No metered energy, no TTFT, no platform cold-start IDs from Cloud Run/Lambda.
- RouteLLM/xRouteBench do not supply learned EcoLogic-comparable assignments.

---

## Results that are safe to claim

- On the frozen Stage 1–2 panel (`n=364`, complete three-tier calls), **Always-Tier-2 has higher accuracy (336/364 vs 316/364) and lower measured USD ($0.04175 vs $0.28675)** than the keyword EcoLogic policy that logs `routing.json` `raw` decisions. Independently reconstructed; hashes match.
- Naive mix×mean cost **is not** realized per-query cost; the covariance identity holds; EcoLogic’s naive figure **understates** realized USD by ~7% on this panel.
- Quality-matched / cost-matched static on this hull **is Always-T2** under committed prices.
- Keyword routing does not look up evaluation labels at decision time.
- Held-out `heldout_v1`: train-only preprocess, val-only selection, frozen split; **selected logistic does not beat cost-matched static** (identical test metrics).
- Energy in this package is **modelled**. Simulator output is **SIMULATED**. Dry-run stubs are **not** paid serverless measurements (new writes tagged `DRY_RUN_LOCAL`).
- Frozen SHA256: 67 files, 0 mismatches.

---

## Results that are NOT safe to claim

- That EcoLogic (keyword or learned) **beats** a cost-matched static policy on USD or quality in a held-out or deployable sense.
- That Stage 1–2 is an out-of-sample router evaluation.
- That the oracle is a deployable router or that the published budget sweep is exact MCKP.
- That energy, TTFT, or cold starts were **measured** on Cloud Run/Lambda.
- That `deployment_real_v2` is a real serverless benchmark (it is a local stub; 120 rows).
- That RouteLLM or xRouteBench demonstrate an **external learned router** on this artifact (assignments missing / oracle-only).
- That robustness `n_boot=50` “proves” stability, or that `survives_became_beneficial_only` means the hull claim survived.
- That `reconstruct()` / green tests imply the whole experimental stack was reproduced.
- That a git clone of this repo is double-blind anonymous.

---

## Required work before code freeze

1. **Do not freeze as a double-blind git artifact** until history is rewritten or the published bundle is git-less. Leave the anonymity scanner on.
2. **Either run a real paid Cloud Run/Lambda campaign** (all three safety flags, non-loopback URL, frozen query set) **or delete/relabel claims** that v2 is `MEASURED_REAL_DEPLOYMENT`. Optionally regenerate committed dry-run JSON with `DRY_RUN_LOCAL` so on-disk files match the new writer.
3. **Headline claims must match experiments:** keyword in-sample domination by T2; held-out learned router null vs T2. Do not add a routing-win sentence without a new experiment.
4. If a learned-router win is required for the workshop story: **new data / new pool**, not retuning `heldout_v1` after seeing test.
5. If RouteLLM is a contribution: install the public scorer, run the frozen 21-threshold sweep, store query-level assignments, recompute all thresholds (with FDR).
6. If oracle frontiers are cited: recompute n=364 with exact/MILP and report approximation error.
7. If robustness p-values are cited: rerun with `n_boot` on the order of 10⁴ and test the **USD hull** estimand.

Items 2–7 are **new experiments or honest claim cuts**, not more unit tests.

---

## Optional work

- Raise default robustness `n_boot` / `n_perm` for future runs (do not silently rewrite `robustness.json` to “improve” conclusions).
- Expand `reproduce_tables.reconstruct()` to latency/held-out/framework sign-flips.
- Namespace `--new-experiment` held-out outputs to a new prefix so the lock cannot overwrite `heldout_v1`.
- Report dual latency (success-only and including failures) in every deployment table.
- Log per-attempt USD on retries.
- Jaccard 0.7–0.9 near-dup analysis for the held-out split.

---

## Final freeze recommendation

**NO**

The USD negative result on the frozen 364-item panel is real and reconstructible; that is not permission to freeze a WoAIS serverless artifact. The keyword router is in-sample, the held-out learned router is a null, there is no paid cloud measurement, energy and DES latency are modelled/simulated, external routers did not emit assignments, and git history fails anonymity. Tests are green and hashes match; those are necessary and nowhere near sufficient. Cut the claim set to the Stage 1–2 domination result **or** run the missing experiments. Do not freeze until the published claims and the executed measurements are the same object.

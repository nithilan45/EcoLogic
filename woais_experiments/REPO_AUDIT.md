# Repository audit for WOAIS experiments

This document is the pre-implementation map of EcoLogic. It exists so the
`woais_experiments/` tree can reuse frozen measurements without touching them.
Nothing in `raw_results/`, `router_v2/`, `stage7_10/`, or the root quality-pilot
outputs was modified to produce this audit. SHA256 digests of those artifacts
are in `EXISTING_RESULTS_SHA256.txt`.

**Hard rules for this package**

- Do not regenerate, overwrite, or delete existing experimental outputs.
- Do not make provider API calls.
- Write new artifacts only under `woais_experiments/` (primarily `results/`).
- Prefer importing existing functions over copying them; if a legacy `main()`
  writes into a frozen directory, call the library functions instead of `main()`.

---

## 1. Repository structure

EcoLogic is a product (FastAPI + static frontend) plus a research audit of
whether LLM routers beat **query-independent static baselines** on **exact
per-item cost**. There is no installable Python package, no Jupyter notebook,
and no pre-existing unit-test suite. Scripts share code via `sys.path` inserts
into `benchmark/`, `router_v2/`, `stage7_10/`, and `backend/`.

| Path | Role |
|---|---|
| `backend/main.py` | Production keyword/NLP router and live inference |
| `benchmark/` | Stage 1–2 harness: item set, API calls, grading, routing log, analysis |
| `raw_results/` | Frozen Stage 1–2 generations, grades, routing, policy tables |
| `router_v2/` | Learned router (pre-reg → pool → train → calibrate → MCKP → one-shot) |
| `stage7_10/` | Scaled retest, regret correction, RouteLLM check, variance decomposition |
| `quality_benchmark_harness.py` | Earlier 24-question pilot (superseded) |
| `EVALUATION.md` | Research question, contributions, findings, reproduction map |

Eval models (pinned in `stage7_10/reproducibility_manifest.md`): Tier 1
`Qwen/Qwen3.5-9B`, Tier 2 `openai/gpt-oss-20b` (Together), Tier 3 `gpt-4o`
(OpenAI). Product `MODELS` in `backend/main.py` still name the retired Gemma /
Apriel slugs; evaluation uses substitutes.

---

## 2. Current routing logic

There is **one production router** and several **offline evaluation policies**.
There is **no bandit / online learner**, and the confidence-escalation step
described in `context.md` is **not implemented**.

### Production (served)

- **File:** `backend/main.py`
- **Function:** `classify_prompt_local_nlp(prompt) -> ClassificationResult`
- **Type:** keyword / linguistic heuristics (no ML)
- **Output:** `recommended_tier ∈ {1,2,3}` then `MODELS[f"tier{tier}"]`
- **Audit log:** `benchmark/router.py` runs the same function on the frozen item
  set and writes `raw_results/routing.json` with both `raw` (user query) and
  `wrapped` (templated prompt) decisions. All published EcoLogic numbers use
  `raw`. Raw vs wrapped agreement is only 53.0%.

### Learned cascade (eval only)

- **File:** `router_v2/calibrate.py` → `route(p1, p2, tau, order=(2,1))`
- **Rule:** first candidate tier (cost-ordered) whose `P(correct) ≥ τ`, else
  Tier 3. Observed order is Tier 2 < Tier 1 < Tier 3 because Tier 1 emits far
  more tokens.
- **Heads:** logistic regression on TF-IDF (R1) or MiniLM (R2) + hand features
  (`router_v2/train_router.py`, `features.py`).
- **Stage 7:** same cascade, larger pool, majority-vote labels
  (`stage7_10/s7_fit.py`, `s7_calibrate.py`, `s7_final.py`).

### Evaluation-only policies

Defined in `benchmark/analyze.py` `build_policies` and the USD twin
`benchmark/analyze_cost.py` `build_policies_cost`:

| Policy | Definition |
|---|---|
| `always_t1` / `always_t2` | Static single-tier |
| `frontier` | Always Tier 3 |
| `random` | Uniform random tier, seed `20260905` |
| `oracle` | Lowest-cost correct tier; if none correct, cheapest tier |
| `ecologic` | Keyword decisions from `routing.json` |

MCKP / LP frontiers in `router_v2/mckp.py` are **offline Pareto bounds**, not
runtime routers.

### RouteLLM (external, not served)

- No `routellm` dependency in the product.
- `stage7_10/external_check.py` clones/loads `lm-sys/RouteLLM` at
  `/tmp/routellm_chk`, reconstructs per-item $ with tiktoken + Mixtral
  tokenizer, runs HF `routellm/bert_gpt4_augmented`.
- `stage7_10/s9_static_baselines.py` compares that router to call-fraction-matched
  and **cost-matched** query-independent mixtures.

**Reuse:** import `classify_prompt_local_nlp` and `route` for read-only checks.
Do **not** re-run `calibrate.py`, `s7_calibrate.py`, `final_test.py`,
`s7_final.py`, `external_check.py`, or `s9_static_baselines.py` in place — they
write into frozen trees.

---

## 3. Where token counts, API cost, latency, identity, prompts, and responses live

Canonical per-call telemetry is produced by `benchmark/api.py`
(`usage_and_cost`, `chat`) and logged by `benchmark/run_benchmark.py`,
`router_v2/run_pool.py`, and `stage7_10/s7_run.py`.

| Quantity | Computed | Stored | Notes |
|---|---|---|---|
| Prompt / completion / total tokens | Provider `usage` | every graded/response row | Never word-count estimates |
| USD | `pt/1e6 * in + ct/1e6 * out` via `RATES_PER_MILLION` | `usd` on each row | Measured at call time |
| Energy (J) | `total_tokens/1000 * paper_energy_per_1k` | usually **computed** at analysis | Rates `{0.5, 1.5, 60}`; **no joule was metered** |
| Latency | `time.time()` around the full HTTP call | `latency_s` | Wall-clock only |
| Model identity | `MODELS[tier]` | `tier`, `model` | Eval slugs ≠ product slugs for T1/T2 |
| Prompts / responses | as sent / returned | full text in Stage 1–6 jsonl; Stage 7 full text **gitignored** | No hashing |
| Quality | `benchmark/grade.py` | `correct`, `gradable` | HumanEval exec, MMLU letter, GSM8K numeric |

**Stage 1–2 schema** (`raw_results/graded.jsonl`): `item_id`, `tier`, `model`,
`benchmark`, `prompt`, `content`, `answer`, `prompt_tokens`,
`completion_tokens`, `total_tokens`, `usd`, `latency_s`, `correct`,
`gradable`, `truncated`, `retries`, …

**Stage 7 compact schema** (`stage7_10/s7_{pool,test}_samples.csv.gz`):
`tier`, `model`, `item_id`, `sample_idx`, `benchmark`, `latency_s`,
`prompt_tokens`, `completion_tokens`, `total_tokens`, `usd`, `correct`,
`gradable`, `truncated`, `extracted` (capped at 120 chars). Full Stage 7
response text is not in git.

**Absent from the entire repo:** TTFT, cold-start flags, queue wait,
autoscaling / idle-timeout traces, prompt/response hashes, RL rewards.

**Reuse:** read frozen rows; recompute aggregates in this package. Do not call
`chat()`.

---

## 4. Static baselines and regret

### Static / oracle (identical item set)

- Energy axis: `benchmark/analyze.py` → `raw_results/analysis.json`, `tables.md`
- USD axis: `benchmark/analyze_cost.py` → `raw_results/analysis_cost.json`,
  `tables_cost.md`
- Headline (n = 364, measured $): Always-Tier-2 **dominates** EcoLogic
  (92.3% vs 86.8% accuracy at $0.0418 vs $0.2867, ratio **6.87×**).
- Stage 7 eight-policy table: `stage7_10/s7_final.py` → `s7_final_results.json`
- External two-model mixtures: `s9_static_baselines.py`
  - matched **call fraction** (usual practice)
  - matched **dollar cost** (correct bar)
  - naive per-model-mean cost is **exact** for query-independent assignment
    and **biased** for a real router

### Regret

Naive confusion-matrix formula vs per-item truth:

```
R_true  = E[ e_{t̂}(X) − e_{t*}(X) ]
R_naive = Σ_{i≠j} P(t*=i, t̂=j) * (ē_j − ē_i)
R_true  = R_naive + Σ_{i≠j} Cov( 1{C_ij}, e_j(X) − e_i(X) )
```

- Derivation + identity check: `stage7_10/regret_correction.py`
  (writes `regret_correction_derivation.md`, `regret_correction_validation.json`)
- Stage 6 calibration numbers: `R_naive ≈ −0.250`, correction `≈ +0.534`,
  `R_true ≈ +0.284` J/item, residual `2.2×10⁻¹⁶`
- External reproduction: `external_check.py` / `external_generalization.json`

**Reuse:** the algebra is a library function (reimplemented here as
`statistics/regret.py` so we never invoke `regret_correction.main()`, which
writes into `stage7_10/`). Import `analyze.wilson` and `analyze.mcnemar` for
the published CI / paired tests.

---

## 5. External RouteLLM experiments

| Artifact | Safe to reuse how |
|---|---|
| `stage7_10/s9_static_baselines.json` | **Read** — headline +0.57 pp mean vs cost-matched mix; 8/9 interior points; 0/9 significant; sign test p = 0.039 |
| `stage7_10/external_generalization.json` | **Read** — correction identity on RouteLLM GSM8K; naive understates cost up to 6.3% |
| `external_check.py`, `s9_static_baselines.py` | Logic reference only. Re-running requires `/tmp/routellm_chk` and **overwrites** the JSON above |
| RouteLLM source / GSM8K response CSV | **Not vendored.** Do not clone or download as part of the default offline suite |

FrugalGPT: no released per-query costs; documented as unverifiable.

**Reuse for WOAIS:** summarize and re-tabulate the committed JSON into
`woais_experiments/results/external/`. Do not invoke BERT or tokenizers in the
default path (those are API-adjacent / heavy and not needed to report the
already-computed sweep).

---

## 6. Workloads, latency, serverless

**Workloads** are frozen academic item sets, not arrival traces:

- Stage 1–2 test: HumanEval 164 + MMLU 100 + GSM8K 100, seed `20260905`
  (`raw_results/benchmark_items.json`)
- `router_v2` train/calibration pools (MBPP / MMLU / GSM8K)
- Stage 7: 5k pool + new 364 test, seed `20260906` (MBPP replaces HumanEval on
  the new test; HumanEval never in train)

“Mixed” means multi-benchmark composition, not open-loop traffic.

**Latency** is a stored wall-clock field (`latency_s`) with no decomposition.
On Stage 1–2, Tier 1 mean latency is ~40–45 s vs Tier 2 ~4–8 s vs Tier 3 ~0.5–2 s.
That is already enough to compare **routing policies on service time** without
new calls, and to parameterize a **queueing / scale-to-zero model**. It is
**not** a measurement of TTFT or cold start.

**Serverless:** Together/OpenAI HTTP chat APIs. Catalog notes that original
product slugs left Together serverless. No idle-timeout, replica-count, or
scale-to-zero experiment exists.

---

## 7. What can safely be reused (and what must not)

### Safe: read-only imports and frozen files

- `benchmark/api.py`: `MODELS`, `RATES_PER_MILLION`, `usage_and_cost` (do not
  call `chat`)
- `benchmark/analyze.py`: `wilson`, `mcnemar`, `load` (read-only if `main` is
  not invoked)
- `backend/main.py`: `classify_prompt_local_nlp` (local, no HTTP)
- `router_v2/calibrate.py`: `route` (pure NumPy) — prefer a local wrapper so
  importing `calibrate` does not pull training-stack side effects unless needed
- All committed JSON / JSONL / CSV / PNG / PKL listed in
  `EXISTING_RESULTS_SHA256.txt`

### Unsafe: scripts whose `main()` writes into frozen trees

Do not execute these in-process as `__main__` from this package:

`benchmark/analyze.py`, `analyze_cost.py`, `run_benchmark.py`, `grade.py`,
`router.py`, `router_v2/calibrate.py`, `mckp.py`, `final_test.py`,
`run_pool.py`, `stage7_10/s7_*.py` generation/calibration/final scripts,
`regret_correction.py`, `external_check.py`, `s9_static_baselines.py`.

### Gaps this package is meant to close *offline*

1. **Unified cost accounting** on USD and energy, including naive vs exact
   per-item cost for the keyword router on the frozen 364-item set (Stage 8
   ran on the learned router’s calibration split, not this table).
2. **Policy latency** (mean / tail) from stored `latency_s`.
3. **Serverless-style models** parameterized by empirical service times:
   residual-after-tokens as a cold-like proxy, open-loop FCFS, idle timeout
   sensitivity. Labelled as models, not measurements.
4. **Accuracy-optimal query-independent mixture** at matched cost for the
   three-tier pool.
5. **Automated tests** that the frozen hashes still match and that published
   Stage 1–2 USD numbers recompute.

---

## 8. Frozen-hash protocol

`EXISTING_RESULTS_SHA256.txt` records SHA256, byte size, and repo-relative path
for 67 result artifacts (Stage 1–2, router_v2 outputs, stage7_10 outputs, root
pilot). Source `.py` / preregistration markdown are not hashed as results.

`woais_experiments` verifies this manifest before and after every offline run.
A mismatch is a hard failure: it means a frozen file changed on disk.

Stage 7 full generations (`s7_*_responses.jsonl`, `s7_*_graded.jsonl`) are
gitignored and **not** in the manifest. Compact `s7_*_samples.csv.gz` files
are, and they are sufficient to recompute Stage 7 accuracy / tokens / USD /
latency.

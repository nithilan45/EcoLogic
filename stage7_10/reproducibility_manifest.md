# Reproducibility manifest — Stages 1–10

Every seed, model version string, package version, endpoint and dataset
revision used anywhere in this project. Compiled so that a year from now the
exact configuration of each stage is recoverable from one file.

Anything below marked **not pinned** is a genuine reproducibility hazard, not an
oversight being papered over.

## 1. Environment

| Item | Value |
|---|---|
| Python | 3.12.3 |
| Platform | Linux 6.12.94, x86-64, glibc 2.39 |
| Compute | CPU only (no GPU used anywhere; all local models run on CPU) |

| Package | Version |
|---|---|
| numpy | 2.4.4 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.5 (Stage 9 only) |
| httpx | 0.28.1 |
| matplotlib | 3.11.1 |
| sentence-transformers | 6.0.1 |
| transformers | 5.16.1 |
| tokenizers | 0.23.2 |
| torch | 2.14.0+cpu |
| tiktoken | 0.14.0 (Stage 9 only) |
| pyarrow | 25.0.1 |
| huggingface_hub | 1.30.0 |

## 2. Random seeds

| Seed | Used by | Governs |
|---|---|---|
| `20260905` | `benchmark/build_benchmark.py` | the **original** frozen 364-item test set: MMLU subject/index sampling, GSM8K index sampling |
| `20260905` | `benchmark/analyze.py` | random-tier baseline assignment, original test set |
| `7` | `benchmark/analyze.py` | nondeterminism spot-check item subsample |
| `771113` | `router_v2/build_train_pool.py` | Stage 1 pool selection (deliberately distinct from the test set's seed) |
| `771113` | `router_v2/train_router.py` | `StratifiedKFold(shuffle=True, random_state=…)` for TRAIN CV, Stages 2 **and 7** |
| `20260906` | `stage7_10/build_pools.py` | Stage 7 pool and new frozen test set selection, and the pool shuffle that makes any generated prefix a random subsample |
| `20260906` | `stage7_10/s7_final.py`, `s7_variance.py` | random-tier baseline on the new test set |

Deterministic by construction, no seed needed: oracle labelling, MCKP/LP solves
(HiGHS), threshold sweep grid (`np.linspace(0, 1, 50)`), MCKP budget grid
(`np.geomspace`), logistic-regression fits (`lbfgs`, deterministic given data).

**Not pinned:** provider-side generation. Temperature is 0 for every headline
number, but Stage 10(a) measures directly that temperature 0 is not
deterministic on either provider. No `seed` parameter was sent on any request.
Re-running the generation will not reproduce the responses byte-for-byte; it
should reproduce the conclusions within the sampling+generation intervals
reported in `s7_generation_variance.md`.

## 3. Models under test

| Tier | Model string | Provider | Role |
|---|---|---|---|
| 1 | `Qwen/Qwen3.5-9B` | Together AI | cheapest tier |
| 2 | `openai/gpt-oss-20b` | Together AI | middle tier |
| 3 | `gpt-4o` | OpenAI | frontier tier |

Tiers 1 and 2 are **energy-adjacent substitutes**. The paper specifies Gemma 3N
E4B and Apriel 1.6 15B; both slugs were retired from Together AI's catalogue
before this evaluation ran, which is documented in `results_report.md`. This is
a substitution, not a reproduction of the paper's exact system.

**Not pinned:** neither provider exposes an immutable revision for these
slugs. `gpt-4o` in particular is a moving alias. There is no way from the client
side to assert that the weights served in September 2026 are those served
earlier. This is the single largest reproducibility hazard in the project and it
cannot be fixed from here.

Retired models never called: `google/gemma-3n-e4b-it`, `servicenow/apriel-15b`.

## 4. Auxiliary models (all local, all CPU, zero marginal API cost)

| Model | Version pin | Used by |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | HF default revision, not pinned to a commit | router variant R2 features, Stages 2 and 7 |
| `routellm/bert_gpt4_augmented` | HF default revision, XLM-RoBERTa, `num_labels=3` | Stage 9, RouteLLM's own released router |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` tokenizer | vocab size 32,000 | Stage 9, weak-model token reconstruction |
| `tiktoken` `cl100k_base` | via `encoding_for_model("gpt-4-1106-preview")` | Stage 9, strong-model token reconstruction |

Judge models used in the **superseded** first-pass report only, and in no
current number: `MiniMaxAI/MiniMax-M3` (judge), `zai-org/GLM-5.3-Flash`
(abandoned, unparseable output). All current grading is objective.

## 5. API endpoints and generation configuration

| Item | Value |
|---|---|
| Together AI | `POST https://api.together.xyz/v1/chat/completions` |
| OpenAI | `POST https://api.openai.com/v1/chat/completions` |
| Together model catalogue (preflight) | `GET https://api.together.xyz/v1/models` |
| Temperature | **0.0** everywhere except the Stage 7 training pool, which is **0.7** by pre-registration |
| `max_tokens` | **16,384** for every tier and benchmark, all stages |
| Token counts | provider `usage` field only; never estimated from word counts |
| Retries | exponential backoff on 429/5xx; `credit_balance_exhausted` treated as fatal, not retried |
| Concurrency | Stages 1–6: 20 Together / 12 OpenAI workers. Stage 7: 80 Together / 24 OpenAI |
| Request timeouts | Stages 1–6: blanket 300 s per request. Stage 7: phase-granular (connect 30 s, read 300 s, write 60 s, pool 60 s) plus a 300 s hard ceiling per task and 15 s keepalive expiry — added mid-Stage-7 because stalled sockets were halting the run; affects only call issuance, never prompts or grading |
| User-Agent | `Mozilla/5.0 EcoLogicBenchmark/1.0` |

The 16,384 cap is load-bearing. At 4,096, Tier 1 was truncated on 57/164
HumanEval items while `gpt-4o` never exceeded 458 tokens, so the cap was
measuring itself rather than the models. The pre-4,096 run is retained at
`raw_results/responses_cap4096.jsonl`.

## 6. Datasets and revisions

| Dataset | Source | Split used | Notes |
|---|---|---|---|
| HumanEval | `openai_humaneval` parquet via HF | all 164 | original frozen test set only; **never** used for training |
| MMLU | `cais/mmlu` `all/test` parquet via HF | test, 14,042 items | 4-choice items only |
| GSM8K | `openai/gsm8k` `main/test` parquet via HF | test, 1,319 items | original test set + Stage 1 pool + new test set |
| GSM8K (train) | `raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl` | train, 7,473 items | Stage 7 pool only, so overlap with any test split is impossible by construction |
| MBPP | `raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl` | all 974 | Stage 1 pool 400; Stage 7 test 164; Stage 7 pool 410 (exhausts the dataset) |
| RouteLLM release | `github.com/lm-sys/RouteLLM` @ `0b64fda` | `evals/gsm8k/gsm8k_responses.csv` | Stage 9; decontaminated with their own `contaminated_prompts.jsonl` to 1,307 items |

**Not pinned:** HF dataset downloads are not pinned to dataset revisions. The
exact item IDs actually used are recorded in
`raw_results/benchmark_items.json`, `router_v2/{train,calibration}_pool.json`
and `stage7_10/s7_{train_pool,calibration_pool,test_set}.json`, so the item
selections are recoverable even if upstream changes.

## 7. Item-set partitions and disjointness

| Set | Size | Composition | Disjointness verified against |
|---|---|---|---|
| Original frozen test set | 364 | HumanEval 164, MMLU 100, GSM8K 100 | — (defined first) |
| Stage 1 pool | 1,200 | MBPP 400, MMLU 400, GSM8K 400 | original test set |
| Stage 1 TRAIN / CALIBRATION | 840 / 360 | 70/30 split | — |
| Stage 7 pool | 5,000 | MBPP 410, MMLU 2,295, GSM8K-train 2,295 | original test set, Stage 1 pool |
| Stage 7 TRAIN / CALIBRATION | 3,500 / 1,500 | 70/30 split | — |
| Stage 7 frozen test set | 364 | MBPP 164, MMLU 100, GSM8K 100 | original test set, Stage 1 pool, Stage 7 pool |

All Stage 7 checks printed **PASS** before any generation
(`stage7_10/build_pools.py`). HumanEval items appear in no training pool under
any framing.

## 8. Energy model

Modelled, not measured. Rates in joules per 1,000 tokens:

| Tier | Paper rate (used for all headline numbers) | Substitute-adjusted rate (sensitivity only) |
|---|---|---|
| 1 | 0.5 | 1.1 |
| 2 | 1.5 | 2.0 |
| 3 | 60.0 | 60.0 |

Energy = `total_tokens / 1000 × rate_tier`, with real provider token counts.
Sensitivity sweep: all 27 combinations of ×0.2, ×1, ×5 per tier.

## 9. Pre-registration commits

| Document | Commit | Covers |
|---|---|---|
| `router_v2/PREREGISTRATION.md` | committed ahead of every other `router_v2/` artifact | Stages 1–6: S1/S2, one-shot rule, threshold rule with X = 10%, oracle labelling, cost gate $40 |
| `stage7_10/prereg_stage7.md` | `97a08cb`, committed before any Stage 7 data existed | Stage 7: 5,000-item target, k=3 majority labels at temp 0.7, unchanged S1/S2, third "partial support" outcome, MBPP ceiling, $150 cost gate, throughput contingency |

## 10. Measured API cost

| Stage group | Calls | Cost |
|---|---|---|
| Superseded 24-question harness | 72 | $0.1216 |
| Stages 1–6 (original test set + Stage 1 pool) | ~4,700 | $3.5143 |
| Stage 7 (complete) | 48,276 of 48,276 planned | $42.6240 |
| Stage 8 | 0 (re-analysis of Stage 6 data) | $0.00 |
| Stage 9 | 0 (all local; RouteLLM's own released outputs) | $0.00 |
| **Total spent** | | **$46.26** |

Stage 7 split: the 45,000-call pool (Tier 1 $11.96, Tier 2 $1.42, Tier 3 $26.19)
and the 3,276-call test set ($3.05), all complete. The pilot projected $47.47
against the pre-registered $150 gate, so it was accurate to within 10% and cost
was never the binding constraint: the run was interrupted twice by **account
credit limits**, first Together AI (HTTP 402) mid-pool and then OpenAI (HTTP 429
`insufficient_quota`) on the test set's Tier 3 column, each resolved by adding
credit. Measured unit costs, useful for future projections: Together Tier 1
$0.000797/call, Tier 2 $0.000095/call, OpenAI Tier 3 $0.001746/call.

17,109 call attempts failed and were retried (15,330 HTTP 402, 1,416 HTTP 429,
18 HTTP 503, 345 client-side task timeouts). None are billed and all eventually
succeeded, so they are neither cost nor data loss. Full breakdown in
`stage7_10/s7_run_accounting.json`.

## 11. Commands, in order

```bash
# Stages 1-6 (previous reports)
python3 benchmark/build_benchmark.py
python3 benchmark/run_benchmark.py
python3 benchmark/grade.py
python3 benchmark/router.py
python3 benchmark/analyze.py
python3 router_v2/build_train_pool.py
python3 router_v2/run_pool.py
python3 router_v2/grade_pool.py
python3 router_v2/train_router.py
python3 router_v2/calibrate.py
python3 router_v2/mckp.py
python3 router_v2/final_test.py

# Stages 7-10
python3 stage7_10/build_pools.py            # prints the disjointness PASS
python3 stage7_10/s7_run.py --target pool --pilot 6
python3 stage7_10/s7_run.py --target pool
python3 stage7_10/s7_run.py --target test
python3 stage7_10/s7_grade.py --target pool
python3 stage7_10/s7_grade.py --target test
python3 stage7_10/s7_fit.py                 # R1/R2 refit, TRAIN-CV selection only
python3 stage7_10/s7_calibrate.py           # threshold sweep + LP/integer MCKP + gaps
python3 stage7_10/s7_calib_policies.py      # CALIBRATION policy table
python3 stage7_10/s7_final.py               # one-shot frozen test set + S1/S2 verdict
python3 stage7_10/s7_export_samples.py      # per-sample grade/token/cost table
python3 stage7_10/regret_correction.py      # Stage 8
python3 stage7_10/external_check.py         # Stage 9 (needs the RouteLLM clone)
python3 stage7_10/s7_variance.py            # Stage 10(a)
```

Two helper scripts wrap the above for unattended resumption after a credit
interruption, and are what actually produced the completed run:
`stage7_10/resume_when_funded.sh` (waits for Together AI, then loops the pool
and test generation until no calls are pending) and
`stage7_10/finish_when_funded.sh` (waits for OpenAI, then issues the outstanding
Tier 3 test calls and runs grading, the one-shot evaluation and Stage 10(a)).
Neither selects or tunes anything.

Credentials are read from `TOGETHER_API_KEY` and `OPENAI_API_KEY` in the
environment. No key is committed anywhere in this repository.

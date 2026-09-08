# Reproducibility manifest — Stages 1–13

Every seed, model version string, package version, endpoint and dataset
revision used anywhere in this project. Compiled so that a year from now the
exact configuration of each stage is recoverable from one file.

Anything below marked **not pinned** is a genuine reproducibility hazard, not an
oversight being papered over.

Sections 1–11 cover Stages 1–10. **Section 12 covers Stages 11–13** (the
decomposition, RouterBench, the router ladder) and is self-contained: it repeats
the seeds, models, datasets, costs and commands for that group rather than
amending the sections above, so nothing documenting the earlier stages changes
meaning.

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

| Stage group | Calls | Cost | Source |
|---|---|---|---|
| Stages 1–4 (original frozen test set, pilots, repeats, the archived 4,096-cap run, determinism checks) | ~1,900 | $1.9497 | `results_report.md` |
| Stages 5–6 / `router_v2` (Stage 1 training pool + calibration) | ~4,700 | $3.5143 | `router_v2/README.md` ($3.4712 pool + $0.0431) |
| Stage 7 (complete) | 48,276 of 48,276 planned | $42.6240 | `s7_run_accounting.json` |
| Stage 8 | 0 (re-analysis of Stage 6 data) | $0.00 | — |
| Stage 9 | 0 (all local; RouteLLM's own released outputs) | $0.00 | — |
| **Subtotal, Stages 1–10** | | **$48.0880** | |
| Stages 11–13 (§12.8) | 9,000 | $9.7070 | `s13_spend.json` |
| **Total spent** | | **$57.7950** | |

Quoted elsewhere as **$48.09** through Stage 10 and **$57.80** overall.

The superseded 24-question judge-graded harness cost a further **$0.1216**
(`quality_benchmark_report.md`). It is **excluded** from both totals because no
current number depends on it; a reader who wants every dollar ever spent on this
project should add it, for $57.9166. An earlier revision of this table both
omitted the Stages 1–4 row and included the superseded harness in its total,
which is why it read $46.26; the components above are each traceable to the file
named beside them and now sum to the totals the rest of the project quotes.

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

---

# 12. Stages 11–13 — the decomposition, RouterBench, and the router ladder

Same machine, same Python 3.12.3, CPU only. Additional or newly load-bearing
packages beyond §1:

| Package | Version | Used by |
|---|---|---|
| `together` | 2.32.0 | Stage 13 prompted and fine-tuned routers, endpoint probes |
| `torch` | 2.14.0+cpu | Stages 13b/13c end-to-end encoder fine-tuning |
| `transformers` | 5.16.1 | same (`AutoModel`, `AutoTokenizer`) |
| `pandas` | 3.0.5 | RouterBench pickle loading (no longer Stage 9 only) |
| `huggingface_hub` | 1.30.0 | RouterBench download |

`datasets` is **not** installed and is not needed: the RouterBench release ships
as pickled DataFrames, loaded with `pandas` directly.

## 12.1 Seeds

| Seed | Used by | Governs |
|---|---|---|
| `20260907` | `s11_routerbench.py` | 5-fold `StratifiedKFold(shuffle=True)` for out-of-fold router predictions, MLP and GBM initialisation |
| `20260907` | `s12_ceiling.py` | replicate resampling and bootstrap draws |
| `20260907` | `s12b_learning_curve.py` | the fixed 20% held-out split; training subsets drawn as `SEED + repeat` |
| `20260907` | `s13_llm_router.py` | few-shot exemplar selection, bootstrap resampling |
| `20260907` | `s13b_encoder_finetune.py`, `s13c_encoder_routerbench.py` | `torch.manual_seed`, the inner-validation split, batch order |
| `2000` resamples | all bootstrap intervals in Stages 11–13 | interval width |

Deterministic by construction: the static frontier (upper concave envelope), the
oracle frontier (Lagrangian multiple-choice-knapsack solve on per-item costs),
every closed form in `theory.md`, and the Holm–Bonferroni / Benjamini–Hochberg
corrections.

**Checked:** re-running `s11_validate.py` on a later date reproduced
`s11_validate.json` **byte-identically** (14/14 checks, every reported error
term unchanged), so the LP solves, the Monte-Carlo suites and the closed forms
are reproducible on this machine rather than merely seeded.

**Not pinned:** CPU-only PyTorch training is deterministic given the seed on
this machine but is not guaranteed bit-identical across BLAS builds or thread
counts. `OMP_NUM_THREADS=2` was set for the encoder runs. The reported held-out
AUCs should reproduce to the third decimal, not exactly.

## 12.2 External dataset — RouterBench

| File | SHA-256 | Items | Models |
|---|---|---|---|
| `external_data/routerbench_0shot.pkl` | `ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d` | 36,494 | 11 |
| `external_data/routerbench_5shot.pkl` | `fbd7d3d16fba2759a18fa0ad44409d3e0e92ba80d03d90827eed1a6d084c9ffe` | 36,480 analysed (28 dropped) | 11 |

Source: `withmartian/routerbench` on the HuggingFace Hub. `fetch_routerbench.py`
downloads both releases into `external_data/`, hashes them and **fails** if
either hash differs from the two above. **The Hub revision is not pinned** from
the dataset id alone, which is why the hash of each file as actually analysed is
recorded — in the `sha256` field of each result JSON and again in
`s11_routerbench_sha256.json`. A future download that hashes differently is a
different dataset and none of the numbers here describe it. Checked: a fresh
download from the Hub on 2026-09-07 reproduced
`ba4f77f1…d36d` for the 0-shot file, so the revision has not moved since the
analysis ran.

The pickles are ~270 MB together and are **gitignored, not committed**, so
fetching them is a required reproduction step rather than an optional one.

Item accounting, so the counts in the reports reconcile against the files:

| | 0-shot | 5-shot |
|---|---|---|
| rows in the pickle | 36,497 | 36,511 |
| less `eval_name == test-match` (a 3-item smoke test, "Once upon a ") | −3 | −3 |
| less items with any missing score or cost cell | −0 | −28 |
| **analysed** | **36,494** | **36,480** |

The `test-match` exclusion was **pre-registered** (`prereg_stage11_13.md` §2.2,
"`test-match` (n = 3) is excluded as a smoke-test artifact"), so it is not a
deviation. The missing cells were not anticipated: the 5-shot file has **154
missing score cells over 28 `arc-challenge` items** (0.077% of items), and those
items are **dropped, not imputed**, because every imputation rule would move
`kappa` in a direction requiring an argument. The count is printed by the loader
and stored as `n_items_dropped_missing_cells`. The 0-shot primary analysis drops
nothing on this ground. See `DEVIATIONS.md` D6.

The 11 models and 8 benchmark families are listed in
`s11_routerbench_{0shot,5shot}.json` under `models` and `family_sizes`. Families
are included at `n ≥ 100`, giving 8 families × 55 pairs = 440 cells.

**Not used:** the "Who Routes the Router" release was considered and not
analysed; nothing in this project depends on it, and it is not cited as if it
were.

## 12.3 Models

| Model string | Provider | Role |
|---|---|---|
| `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Together AI | prompted LLM router, zero-shot and 4-shot (Stage 13 rung R-b) |
| `google/gemma-3-27b-it` | Together AI | LoRA fine-tune base (rung R-c) — **trained, never served**, see §12.6 |
| `Qwen/Qwen3.5-9B` | Together AI | attempted second fine-tune base, refused for insufficient balance |
| `sentence-transformers/all-MiniLM-L6-v2` | local, CPU | frozen features for RouterBench routers; **and** the unfrozen backbone in 13b/13c |
| `sentence-transformers/all-MiniLM-L12-v2` | local, CPU | second unfrozen backbone in 13b |

`all-MiniLM-L6-v2` appears twice deliberately: Stage 13b fine-tunes end-to-end
**the same backbone** whose frozen output gives 0.6816 in Stage 7c, so the
comparison isolates the effect of unfreezing and nothing else.

Fine-tuned adapter produced and retained on the provider side:
`nithilankarthik-0f83/gemma-3-27b-it-s13router-f64c70b3`, from job
`ft-6567602b-4aa3`.

**Not pinned:** as in §3, no provider revision is exposed for the Together
slugs.

## 12.4 Router hyperparameters

RouterBench routers (`s11_routerbench.py`), all fitted out-of-fold within the
5-fold CV and never on their own scoring fold:

| Router | Configuration |
|---|---|
| TF-IDF + logistic | `TfidfVectorizer` word 1–2 grams, `LogisticRegression(lbfgs)` |
| MiniLM + logistic | 384-d frozen embedding, `LogisticRegression(lbfgs)` |
| MiniLM + MLP | hidden `(256, 64)`, `learning_rate_init=1e-3`, multi-output |
| MiniLM + boosted trees | `HistGradientBoostingClassifier`, `early_stopping=True` |

End-to-end encoders: `MAX_LEN=256`; Stage 13b `EPOCHS=6`, `BATCH=16`,
encoder LR `2e-5`, head LR `1e-3`, inner-validation fraction 0.15; Stage 13c
`EPOCHS=3`, `BATCH=32`, encoder LR `3e-5`, head LR `1e-3`, inner-validation
fraction 0.05. The head emits one logit per candidate model — 3 for Stage 13b's
tiers, 11 for Stage 13c's RouterBench models — each trained with binary
cross-entropy against that model's graded outcome. Dynamic padding with length
bucketing — chosen for CPU
throughput, and it changes no result because padding is masked out of the mean
pool either way. Epoch selection uses the inner split carved out of TRAIN and
**never touches the split every rung is scored on**.

Fine-tune job configuration: LoRA, 3 epochs, 10,500 prompt/completion examples
from `s13_ft_train.jsonl` (the Stage 7 TRAIN split, 3,500 items × 3 tiers),
1,491,033 tokens.

## 12.5 Splits

Stages 13/13b reuse **exactly** the Stage 7 partitions in §7 — fitted on the
3,500-item TRAIN split, scored once on the 1,500-item CALIBRATION split, which
is the split every earlier rung reports. Nothing new was partitioned in-house,
so no new disjointness check exists or is needed.

RouterBench (Stages 12b, 13c): a fixed **7,299-item held-out set** (20%) drawn
at `SEED`, leaving **29,195** training items. Stage 12b sweeps training subsets
from 250 to 29,000 of those; Stage 13c trains on 27,735 and holds 1,460 back as
inner validation. Stages 12b and 13c draw the held-out set with the same seed
and the same call, so their numbers are directly comparable, and 13c
**refits** the frozen logistic and MLP baselines on its own training split
rather than quoting the 5-fold CV numbers, so the end-to-end comparison is
like-for-like.

## 12.6 API configuration and the blocked rung

| Item | Value |
|---|---|
| Prompted router endpoint | `POST https://api.together.xyz/v1/chat/completions`, `logprobs` on, 1 completion token |
| Decision rule | `P(yes)` from the logprobs of the `yes`/`no` tokens, per tier |
| Fine-tuning endpoint | `POST https://api.together.xyz/v1/fine-tunes` |
| Endpoint probes | `/v1/models`, `/v1/hardware`, `/v1/endpoints`, and the v2 `models.configs.list` / deployments API via the SDK |
| Cost gate | **$25**, enforced in code; every call books the provider's own reported usage into `s13_spend.json` and the run aborts at the gate |

The fine-tuned generative rung is **BLOCKED**. Four routes were probed rather
than assumed, each with the provider's verbatim error recorded in
`s13_ftblocked.json` and `s13_endpoint_probe.json`:

1. serverless LoRA — `400 Unable to access non-serverless model` on all 13
   advertised `*-Lora` targets, `404 model_not_available` on the fine-tune;
2. dedicated endpoints v1 — `403 endpoints_v1_create_access_disabled`
   (retired platform-wide, not an account limit);
3. dedicated endpoints v2 — **0 certified configs** for both the merged
   fine-tune and the `gemma-3-27b-it` base; gemma-3-27b absent from the 43
   v2-supported architectures;
4. re-fine-tuning on a v2-servable base — `402 insufficient_balance`
   ("Required combined balance and credit limit: 4.00 USD").

Per the pre-registration, **no result was estimated, extrapolated or simulated
in its place**.

## 12.7 Pre-registration commit

| Document | Commit | Covers |
|---|---|---|
| `stage11_13/prereg_stage11_13.md` | `0b17dfa`, committed before any Stage 11–13 artifact existed | H1 (complementarity ≥ 5 pp), H2 (realised `rho` ≤ 30%), H3 (the ceiling hypothesis, since **refuted**), the §3.4 consistency check that came out INCONSISTENT, the C1/C2 router-ladder criteria, score handling, the $25 cost gate, and the rule that a blocked run is reported as blocked |

Deviations are in `stage11_13/DEVIATIONS.md`, D1–D8. D1 (H3 refuted), D2 (a
pre-registration error about what the peeking policy bounds), D3 and D7
(analyses added as exploratory), D6 (the 28 dropped items) and D8 (the blocked
rung) all change what the project claims and are written up in full rather than
summarised.

## 12.8 Measured API cost

| Item | Volume | Cost |
|---|---|---|
| Prompted router, zero-shot | 4,500 calls | $0.5837 |
| Prompted router, 4-shot | 4,500 calls | $2.4136 |
| LoRA fine-tune (Gemma-3-27B-it, 3 epochs) | 1,491,033 tokens | $6.7096 |
| Fine-tuned generative inference | **blocked** | $0.00 |
| End-to-end encoders (13b, 13c) | CPU | $0.00 |
| RouterBench both releases, theory, validation, learning curves | — | $0.00 |
| **Total, Stages 11–13** | 9,000 calls | **$9.7070** |

Against the pre-registered $25 gate and a projection of ~$16. The gate never
bound; the provider's account balance did. Project total **$57.80**.

## 12.9 Commands, in order

```bash
python3 stage11_13/fetch_routerbench.py         # download + hash-verify both releases
python3 stage11_13/s11_validate.py              # 14/14 propositions vs LP / Monte Carlo
python3 stage11_13/s11_routerbench.py 0shot     # primary analysis
python3 stage11_13/s11_routerbench.py 5shot     # pre-registered replication
python3 stage11_13/s12_ceiling.py               # replicate reliability, Bayes AUC, peeking policy
python3 stage11_13/s12b_learning_curve.py       # exploratory (D3)
python3 stage11_13/s13_llm_router.py probe      # catalogue + pricing preflight
python3 stage11_13/s13_llm_router.py prompted 0
python3 stage11_13/s13_llm_router.py prompted 4
python3 stage11_13/s13_llm_router.py build_ft   # writes s13_ft_train.jsonl
python3 stage11_13/s13_llm_router.py launch_ft
python3 stage11_13/s13_llm_router.py poll_ft
python3 stage11_13/s13_llm_router.py score_ft   # BLOCKED; records the provider errors
OMP_NUM_THREADS=2 python3 stage11_13/s13b_encoder_finetune.py          # exploratory (D7), hours on CPU
OMP_NUM_THREADS=2 python3 stage11_13/s13c_encoder_routerbench.py 0shot # exploratory, ~1 h on CPU
python3 stage11_13/s13_llm_router.py report     # ladder, paired bootstrap, C1/C2 verdict
cd paper && pdflatex main.tex && pdflatex main.tex
```

The `launch_ft2` / `poll_ft2` / `deploy_ft2` / `score_ft2` / `teardown_ft2`
subcommands are the fourth route in §12.6 — the attempt to re-train on a
v2-servable base. They are retained because `launch_ft2` is what returned
`402 insufficient_balance`, and a reader checking that claim should be able to
run the same call.

`s11_routerbench.py` caches MiniLM embeddings under `external_data/`; the cache
is derived and deliberately untracked, and is regenerated if absent or if the
item count changes.

Credentials: `TOGETHER_API_KEY` only (no OpenAI call is made in Stages 11–13).
No key is committed anywhere in this repository.

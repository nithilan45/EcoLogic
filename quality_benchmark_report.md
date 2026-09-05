# EcoLogic matched-query quality benchmark

**Run status: 72/72 inference calls succeeded.** Judge pass: 48/48 reasoning+code rows scored 0/1.

This is **not** the paper’s original stack. Gemma 3N E4B and Apriel 1.6 15B are gone from Together serverless. The operator approved energy-adjacent substitutes. **Tier 3 is not GPT-4o** (`OPENAI_API_KEY` was not provided).

Checked / generated: 2026-09-05T22:16:34Z.

## Models actually called

| Tier | Role in EcoLogic | Model used | Provider | EcoLogic-style J/1k* | Catalog check |
|------|------------------|------------|----------|----------------------|---------------|
| 1 | small / cheap | `Qwen/Qwen3.5-9B` | Together | ~1.1 (was 0.5 for 4B) | Present; smoke `pong` |
| 2 | mid | `openai/gpt-oss-20b` | Together | ~2.0 (was 1.5 for 15B) | Present; smoke `pong` |
| 3 | large | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Together | ~7.0 (paper used GPT-4o at 60) | Present; smoke `pong` |
| judge | not under test | `MiniMaxAI/MiniMax-M3` | Together | — | Present |

\*Same param-count scaling EcoLogic already uses. Not a lab joule meter.

Original IDs `google/gemma-3n-E4B-it` and `ServiceNow-AI/Apriel-1.6-15b-Thinker` were **absent** from Together `GET /v1/models` (275 ids). `google/gemma-4-E4B-it` is catalog-listed but **dedicated-only** (HTTP 400 `model_not_available` on serverless chat).

Smoke tests (prompt: `Reply with the single word: pong`) all returned `pong` before the 72-call batch.

## Factual accuracy (auto-graded substring match)

Logic unchanged: `reference.lower() in answer.lower()`. 8 questions × 3 tiers.

| Tier | Model | Correct / 8 | Accuracy |
|------|-------|-------------|----------|
| 1 | `Qwen/Qwen3.5-9B` | 8 / 8 | **100%** |
| 2 | `openai/gpt-oss-20b` | 7 / 8 | **87.5%** |
| 3 | `Llama-3.3-70B-Instruct-Turbo` | 8 / 8 | **100%** |

**Auto-grader miss, not a wrong painter:** Tier 2 f7 (*Who painted 'Starry Night'?*) said Vincent van Gogh, but used a Unicode narrow no-break space between `van` and `Gogh`, so `"van gogh"` did not match. Leave the scorer as-is; treat this 87.5% as a substring artifact. Semantically T2 also got Van Gogh.

## Judge-graded reasoning and code

Judge: `MiniMaxAI/MiniMax-M3` (not T1/T2/T3). Rubric: reasoning = addresses the trade-off/comparison without factual errors or dodge; code = would plausibly run and solve the task.

First judge attempt used `zai-org/GLM-5.3-Flash`, which only filled `reasoning_content` and never emitted a grade (39/48 unparseable). Those scores were **discarded** and the 48 rows were re-graded with MiniMax. Inference answers were not re-generated.

| Tier | Reasoning correct / 8 | Reasoning accuracy | Code correct / 8 | Code accuracy |
|------|----------------------:|-------------------:|-----------------:|--------------:|
| 1 | 3 / 8 | **37.5%** | 7 / 8 | **87.5%** |
| 2 | 8 / 8 | **100%** | 8 / 8 | **100%** |
| 3 | 7 / 8 | **87.5%** | 8 / 8 | **100%** |

## Anomalies (not smoothed over)

- **API errors:** none on the 72 eval calls. All HTTP 200. Retries: 0.
- **Empty/refused completions:** 0 empty answers. Qwen often puts chain-of-thought in `reasoning` / `reasoning_content`; 7 Tier 1 answers were thinking-process dumps used as the visible answer because `content` was empty.
- **Rate limits:** none observed.
- **Tier 1 reasoning collapse:** 5/8 reasoning fails (`r2, r3, r4, r5, r6`) because Qwen returned outlines / “here's a thinking process” instead of the analysis, or truncated mid-sentence (`r4`). Code fail `c2`: debounce snippet truncated at `function debounce(func,`.
- **Smaller tier beat larger — call these out:**
  1. **T2 reasoning 100% vs T3 87.5%.** The 70B tier lost `r5` (nuclear vs solar): judge flagged a factual error (solar 10–20 vs nuclear 10–40 gCO2/kWh; IPCC-style figures go the other way). T2’s `r5` scored 1. T1 scored 0 on `r5` (truncated thinking dump).
  2. **T1 and T3 both 100% factual vs T2 87.5%** on the auto-grader, driven only by f7’s Unicode space (see above). Not a real knowledge fail.
- **Tier 3 is Llama 70B, not GPT-4o.** Do not read these numbers as “GPT-4o vs small models.”
- **`gpt-oss-20b` is scheduled for Together serverless removal 2026-09-14.** This run was 2026-09-05.

## Cost

APIs did not return a billed dollar field. Cost below is **token usage × published per-1M rates**.

| Stage | Tokens | Estimated USD |
|-------|--------|---------------|
| Tier 1 (24 calls) | 18,240 | $0.00452 |
| Tier 2 (24 calls) | 17,752 | $0.00327 |
| Tier 3 (24 calls) | 10,221 | $0.01063 |
| Eval subtotal (72) | 46,213 | **$0.01842** |
| Judge (48 MiniMax calls) | — | **$0.02614** |
| **Total this run** | | **~$0.0446** |

Smoke tests were extra (3 cheap `pong` calls, on the order of $0.0001) and are not in the 72-call JSON. The failed GLM-Flash judge pass (~$0.011) is also extra vs the MiniMax re-grade in `judge.judge_estimated_usd`.

## What this does and does not show

On this 24-question matched set, the **mid Together model (`gpt-oss-20b`) matched or beat the 70B tier** on judge-graded reasoning and code, and the 9B Qwen tier was fine on short facts and most code but often failed to emit a finished reasoning answer.

That is a result for **these substitutes**, not a measurement of Gemma 3N vs Apriel vs GPT-4o. To compare GPT-4o, add `OPENAI_API_KEY` and set Tier 3 back to `gpt-4o`.

## Files

- `quality_benchmark_results.json` — all 72 prompt/response pairs plus `judge_score` / `judge_justification`
- `quality_benchmark_harness.py` — slugs, catalog preflight, smoke tests, MiniMax judge
- `quality_benchmark_report.md` — this file

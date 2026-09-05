# EcoLogic matched-query quality benchmark

> **Superseded by [`results_report.md`](results_report.md).** This report uses
> 24 hand-written questions and an LLM judge for the reasoning and code
> categories. The newer report replaces both with established benchmarks
> (HumanEval executed against official tests, MMLU, GSM8K), runs the real
> classifier in the loop against four routing policies, and reports Wilson
> intervals, McNemar tests and an energy sensitivity band. Prefer it. This file
> is kept for provenance.

**Run status: 72/72 inference calls succeeded.** Judge: 48/48 reasoning+code rows scored 0/1 (MiniMax M3).

This is **not** the paper’s original Together stack. Gemma 3N E4B and Apriel 1.6 15B are gone from Together serverless. Tiers 1–2 are energy-adjacent substitutes. **Tier 3 is GPT-4o** (OpenAI), after a billed retry.

Generated: 2026-09-05T22:33:01Z.

## Models actually called

| Tier | Role | Model | Provider | EcoLogic-style J/1k* | Smoke |
|------|------|-------|----------|----------------------|-------|
| 1 | small | `Qwen/Qwen3.5-9B` | Together | ~1.1 (paper 4B ≈ 0.5) | `pong` |
| 2 | mid | `openai/gpt-oss-20b` | Together | ~2.0 (paper 15B ≈ 1.5) | `pong` |
| 3 | large | `gpt-4o` (snapshot `gpt-4o-2024-08-06` on smoke) | OpenAI | 60 | `Pong` |
| judge | not under test | `MiniMaxAI/MiniMax-M3` | Together | — | — |

\*Param-count scaling, not a lab joule meter.

Original IDs `google/gemma-3n-E4B-it` and `ServiceNow-AI/Apriel-1.6-15b-Thinker` were absent from Together `GET /v1/models`. `google/gemma-4-E4B-it` is dedicated-only (HTTP 400 on serverless).

An earlier complete 72-call pass used `meta-llama/Llama-3.3-70B-Instruct-Turbo` for Tier 3 because OpenAI returned `credit_balance_exhausted`. After credits were added, **only Tier 3 was regenerated** with gpt-4o. Tiers 1–2 answers and MiniMax scores were not re-queried.

## Factual accuracy (auto-graded substring match)

Logic unchanged: `reference.lower() in answer.lower()`. 8 questions × 3 tiers.

| Tier | Model | Correct / 8 | Accuracy |
|------|-------|-------------|----------|
| 1 | `Qwen/Qwen3.5-9B` | 8 / 8 | **100%** |
| 2 | `openai/gpt-oss-20b` | 7 / 8 | **87.5%** |
| 3 | `gpt-4o` | 8 / 8 | **100%** |

**Auto-grader miss, not a wrong painter:** Tier 2 f7 named Vincent van Gogh but used a Unicode narrow no-break space, so `"van gogh"` did not match. Semantically T2 also got Van Gogh.

## Judge-graded reasoning and code

Same MiniMax M3 judge and rubric as Tiers 1–2. Reasoning: addresses the trade-off/comparison without factual errors or dodge. Code: would plausibly run and solve the task.

| Tier | Reasoning / 8 | Reasoning accuracy | Code / 8 | Code accuracy |
|------|--------------:|-------------------:|---------:|--------------:|
| 1 | 3 / 8 | **37.5%** | 7 / 8 | **87.5%** |
| 2 | 8 / 8 | **100%** | 8 / 8 | **100%** |
| 3 | 8 / 8 | **100%** | 7 / 8 | **87.5%** |

## Anomalies (not smoothed over)

- **API errors on the 72 eval calls:** none. All HTTP 200. Retries: 0. OpenAI first attempt (pre-credits) was HTTP 429 `credit_balance_exhausted`; that call was **not** counted in the 72.
- **Empty/refused:** 0. Seven Tier 1 answers were Qwen thinking-process dumps used as the visible answer because `content` was empty.
- **Rate limits on the successful 72:** none.
- **Tier 1 reasoning:** 5/8 fails (`r2`–`r6`) from outlines / truncated answers. Code fail `c2`: debounce cut off at `function debounce(func,`.
- **Smaller tier judged better than GPT-4o:**
  - **T2 code 100% vs T3 87.5%.** On `c5` (divide by zero), MiniMax scored T1 and T2 1, GPT-4o 0: GPT-4o’s fixes used undefined names (`some_value`, `denominator`) that would not run as written.
  - **T2 reasoning 100% ties GPT-4o 100%** on this 8-item set.
  - Factual auto-score T2 87.5% vs T1/T3 100% is the Unicode-space scorer miss on f7, not a knowledge fail.
- **`gpt-oss-20b` Together serverless removal date:** 2026-09-14 (this run: 2026-09-05).

## Cost

No provider returned a billed-dollar field. Figures are token counts × published per-1M rates.

| Stage | Tokens | Estimated USD |
|-------|--------|---------------|
| Tier 1 (24, Qwen 9B) | 18,240 | $0.00452 |
| Tier 2 (24, gpt-oss-20b) | 17,752 | $0.00327 |
| Tier 3 (24, gpt-4o) | 8,290 | $0.07971 |
| Eval subtotal (72) | 44,282 | **$0.08750** |
| Judge (MiniMax; 48 original + 16 T3 regrade) | — | **$0.03262** |
| **Total in results JSON** | | **~$0.120** |

Extra, not in the 72-row usage: smoke `pong`/`Pong` calls; failed OpenAI quota smoke; discarded GLM-5.3-Flash judge pass; Llama 70B Tier 3 that was later replaced.

## What this shows

On this 24-question matched set, **gpt-oss-20b matched GPT-4o on reasoning (8/8) and beat it on code (8/8 vs 7/8)**. Qwen 9B was perfect on short facts and most code, but often failed to emit a finished reasoning answer.

This is still not Gemma 3N vs Apriel vs GPT-4o. It is Qwen 9B vs gpt-oss-20b vs GPT-4o.

## Files

- `quality_benchmark_results.json` — 72 prompt/response pairs; Tier 3 model field is `gpt-4o`
- `quality_benchmark_harness.py` — `--rerun-tier 3` used for the GPT-4o replacement
- `quality_benchmark_report.md` — this file

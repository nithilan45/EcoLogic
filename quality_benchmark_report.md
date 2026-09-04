# EcoLogic matched-query quality benchmark

**Run status: DID NOT RUN. 0 of 72 inference calls succeeded.**

This is not a quality comparison. No per-tier accuracy numbers exist because the three models were never called on the 24-question set. The tables below are explicit non-results, not estimates.

Checked: 2026-09-04 (UTC). Sources: Together AI `GET https://api.together.xyz/v1/models` (unauthenticated), Together [serverless catalog](https://docs.together.ai/docs/serverless-models), Together [deprecations](https://docs.together.ai/docs/deprecations), OpenAI [GPT-4o model page](https://developers.openai.com/api/docs/models/gpt-4o), EcoLogic production slugs in `backend/main.py`.

## Why the batch was aborted

Two independent blockers. Either one is enough to stop.

### 1. Both Together AI EcoLogic models are gone from serverless

The harness draft used **placeholder slugs that are not Together IDs**:

| Tier | Placeholder in original harness | Real Together ID (production + docs) | Together serverless status on 2026-09-04 |
|------|----------------------------------|--------------------------------------|------------------------------------------|
| 1 | `google/gemma-3n-e4b-it` | `google/gemma-3n-E4B-it` | **Removed 2026-08-25.** Dedicated on-demand: No |
| 2 | `servicenow/apriel-15b` | `ServiceNow-AI/Apriel-1.6-15b-Thinker` | **Removed 2026-04-03.** Dedicated on-demand: No |
| 3 | `gpt-4o` | `gpt-4o` | Still listed as the OpenAI API alias |

`quality_benchmark_harness.py` was updated to the real IDs (`google/gemma-3n-E4B-it`, `ServiceNow-AI/Apriel-1.6-15b-Thinker`, `gpt-4o`). It was **not** pointed at a substitute model. Together's current serverless chat catalog (docs, 2026-09-04) does not list any Gemma 3N or Apriel model. The closest Gemma name still mentioned in Together dedicated-endpoint examples is `google/gemma-4-E4B-it`, which is a **different model** and was not used.

Together's deprecation page also says neither EcoLogic Together model is supported as an on-demand dedicated endpoint after removal.

### 2. No provider credentials in this environment

| Variable | Present? | Live check |
|----------|----------|------------|
| `TOGETHER_API_KEY` | **No** | `GET /v1/models` → HTTP 401 body `Missing API key` |
| `OPENAI_API_KEY` | **No** | `GET /v1/models` → HTTP 401 |
| `ANTHROPIC_API_KEY` (judge) | **No** | Not set; judge pass was never started |

Together's `/v1/models` endpoint is **not public**. An unauthenticated request does not return the catalog; it returns 401. The live catalog confirmation therefore comes from Together's published serverless-models and deprecations pages, not from an authenticated `GET /v1/models` dump.

Smoke tests (one cheap call per tier) were **not** issued. The harness aborts before billed calls when keys are missing or when Together no longer lists the two EcoLogic slugs.

## Factual accuracy (auto-graded substring match)

Not computed. Would have been 8 questions × 3 tiers.

| Tier | Model | Correct / 8 | Accuracy |
|------|-------|-------------|----------|
| 1 | `google/gemma-3n-E4B-it` | — / 8 | **n/a — 0 calls** |
| 2 | `ServiceNow-AI/Apriel-1.6-15b-Thinker` | — / 8 | **n/a — 0 calls** |
| 3 | `gpt-4o` | — / 8 | **n/a — 0 calls** |

Factual auto-scoring logic in the harness is unchanged: `reference.lower() in answer.lower()`. It was never applied to a model output.

## Judge-graded accuracy (reasoning and code)

Not computed. Would have been 8 reasoning + 8 code questions × 3 tiers, scored 0/1 by a model that is **not** one of the three tiers (not GPT-4o). No judge model was called.

| Tier | Reasoning correct / 8 | Reasoning accuracy | Code correct / 8 | Code accuracy |
|------|----------------------:|-------------------:|-----------------:|--------------:|
| 1 | — | **n/a — 0 calls** | — | **n/a — 0 calls** |
| 2 | — | **n/a — 0 calls** | — | **n/a — 0 calls** |
| 3 | — | **n/a — 0 calls** | — | **n/a — 0 calls** |

No `judge_score` / `judge_justification` fields were filled, because there were no responses to grade.

## Anomalies

These are run-blocking failures, not evaluation findings:

- **API auth errors:** Together `GET /v1/models` HTTP 401 `Missing API key`. OpenAI `GET /v1/models` HTTP 401. Together `POST /v1/chat/completions` without a key: HTTP 401 `missing_api_key`.
- **Model slugs:** original harness placeholders were wrong; the real IDs match `backend/main.py` and Together's deprecation history, and both real Together IDs are **removed**.
- **Empty / refused responses:** none. No completions were requested.
- **Rate-limit retries:** none. No billed traffic.
- **Smaller tier judged better than a larger tier:** **not observed**, because nothing was judged. Do not read this as "tiers are equal."

## Cost

**$0.00 billed for this run.** Zero chat-completion requests were sent. There is no usage object to sum.

Published rates (for if this is re-run later, not incurred today):

- Gemma 3N E4B on Together, last published while live: $0.06 input / $0.12 output per 1M tokens (Together changelog; EcoLogic README still quotes the older $0.02 / $0.04).
- Apriel 1.6 15B Thinker: advertised as $0 while it was a Together serverless model.
- GPT-4o (OpenAI docs, 2026-09-04): $2.50 input / $10.00 output per 1M tokens. Alias `gpt-4o` is still documented, with snapshots `gpt-4o-2024-11-20`, `gpt-4o-2024-08-06`, and deprecated `gpt-4o-2024-05-13`. ChatGPT consumer retirement of GPT-4o does not by itself retire the API alias.

## What would be required to actually produce the 72-call table

1. A Together (or other) endpoint that still serves **exactly** `google/gemma-3n-E4B-it` and `ServiceNow-AI/Apriel-1.6-15b-Thinker`. Substituting Gemma 4, a Qwen, or any other "similar-size" model would not be an EcoLogic-tier evaluation.
2. Working `TOGETHER_API_KEY` and `OPENAI_API_KEY`.
3. Re-run `python3 quality_benchmark_harness.py` (smoke tests first, then 72 calls).
4. A judge model that is not GPT-4o (and not Gemma 3N / Apriel) to fill `judge_score` and `judge_justification` on the 48 reasoning/code rows.

Until those exist, any numeric "Tier 1 vs Tier 2 vs Tier 3 quality" table for this design would be fabricated.

## Files in this repo

- `quality_benchmark_harness.py` — ready to run; real slugs; aborts if keys or catalog membership fail.
- `quality_benchmark_results.json` — machine output of this attempt: `successful_calls: 0`, `responses: []`.
- `quality_benchmark_report.md` — this file.

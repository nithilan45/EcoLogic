"""
EcoLogic matched-query quality benchmark.

This script is the runnable harness for a 24-question x 3-tier matched
evaluation (72 inference calls). It will refuse to run if:

- TOGETHER_API_KEY or OPENAI_API_KEY is missing
- Either Together model slug is absent from GET /v1/models
- A smoke-test call to any of the three tiers fails

It will NOT silently substitute a different model. Together AI removed both
EcoLogic Together slugs from serverless inference (see quality_benchmark_report.md).

Usage:
    export TOGETHER_API_KEY=...
    export OPENAI_API_KEY=...
    python3 quality_benchmark_harness.py
"""

import os
import json
import sys
import time
import asyncio
from datetime import datetime, timezone

import httpx

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
TOGETHER_MODELS_URL = "https://api.together.xyz/v1/models"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODELS_URL = "https://api.openai.com/v1/models"

# Verified against Together deprecation history + EcoLogic production
# (backend/main.py). Placeholder slugs in the original draft were wrong:
#   "google/gemma-3n-e4b-it"  -> real id was "google/gemma-3n-E4B-it"
#   "servicenow/apriel-15b"   -> real id was "ServiceNow-AI/Apriel-1.6-15b-Thinker"
# Together then REMOVED both from serverless:
#   google/gemma-3n-E4B-it              removed 2026-08-25 (dedicated: No)
#   ServiceNow-AI/Apriel-1.6-15b-Thinker removed 2026-04-03 (dedicated: No)
# gpt-4o remains the current OpenAI API alias (snapshots: gpt-4o-2024-11-20,
# gpt-4o-2024-08-06, gpt-4o-2024-05-13).
MODELS = {
    1: {"provider": "together", "model": "google/gemma-3n-E4B-it"},
    2: {"provider": "together", "model": "ServiceNow-AI/Apriel-1.6-15b-Thinker"},
    3: {"provider": "openai", "model": "gpt-4o"},
}

# Fallback published rates used only if the API does not return a dollar cost.
# Together changelog (historical, while Gemma 3N was live): $0.06 / $0.12 per 1M.
# Apriel 1.6 was advertised as free. OpenAI GPT-4o (docs, 2026): $2.50 / $10 per 1M.
FALLBACK_RATES_PER_MILLION = {
    "google/gemma-3n-E4B-it": {"input": 0.06, "output": 0.12},
    "ServiceNow-AI/Apriel-1.6-15b-Thinker": {"input": 0.00, "output": 0.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

QUESTIONS = [
    # -- factual (auto-gradable via substring match) --
    {"id": "f1", "category": "factual", "prompt": "What is the capital of Australia?", "reference": "Canberra"},
    {"id": "f2", "category": "factual", "prompt": "Who wrote 'Pride and Prejudice'?", "reference": "Jane Austen"},
    {"id": "f3", "category": "factual", "prompt": "What is the atomic number of carbon?", "reference": "6"},
    {"id": "f4", "category": "factual", "prompt": "In what year did the Titanic sink?", "reference": "1912"},
    {"id": "f5", "category": "factual", "prompt": "What is the largest planet in the solar system?", "reference": "Jupiter"},
    {"id": "f6", "category": "factual", "prompt": "What is the chemical formula for table salt?", "reference": "NaCl"},
    {"id": "f7", "category": "factual", "prompt": "Who painted 'Starry Night'?", "reference": "Van Gogh"},
    {"id": "f8", "category": "factual", "prompt": "What is the smallest prime number?", "reference": "2"},

    # -- reasoning/analysis (rubric-graded, e.g. by human or judge model) --
    {"id": "r1", "category": "reasoning", "prompt": "Compare the trade-offs of monolithic vs microservice architectures.", "reference": None},
    {"id": "r2", "category": "reasoning", "prompt": "Analyze the main causes of urban housing shortages.", "reference": None},
    {"id": "r3", "category": "reasoning", "prompt": "What are the pros and cons of universal basic income?", "reference": None},
    {"id": "r4", "category": "reasoning", "prompt": "Contrast supervised and unsupervised learning, with an example of each.", "reference": None},
    {"id": "r5", "category": "reasoning", "prompt": "Evaluate the environmental trade-offs of nuclear power vs solar power.", "reference": None},
    {"id": "r6", "category": "reasoning", "prompt": "Step by step, explain why inflation erodes purchasing power.", "reference": None},
    {"id": "r7", "category": "reasoning", "prompt": "Assess the risks of over-reliance on a single cloud provider.", "reference": None},
    {"id": "r8", "category": "reasoning", "prompt": "Compare federated learning and centralized training for privacy.", "reference": None},

    # -- code (rubric-graded: does it run, does it solve the stated task) --
    {"id": "c1", "category": "code", "prompt": "Write a Python function that returns the nth Fibonacci number.", "reference": None},
    {"id": "c2", "category": "code", "prompt": "Write a JavaScript function that debounces another function.", "reference": None},
    {"id": "c3", "category": "code", "prompt": "Implement binary search in Java.", "reference": None},
    {"id": "c4", "category": "code", "prompt": "Write a SQL query to find the second-highest salary in an employees table.", "reference": None},
    {"id": "c5", "category": "code", "prompt": "Debug this Python: `def f(x): return x / 0` -- explain the error and fix it.", "reference": None},
    {"id": "c6", "category": "code", "prompt": "Write a Rust function that reverses a string.", "reference": None},
    {"id": "c7", "category": "code", "prompt": "Implement a simple LRU cache in Python.", "reference": None},
    {"id": "c8", "category": "code", "prompt": "Write a regex to validate an email address, in Python.", "reference": None},
]


def _message_content(payload: dict) -> str:
    msg = payload["choices"][0]["message"]
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def _usage_and_cost(model: str, payload: dict) -> dict:
    usage = payload.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    rates = FALLBACK_RATES_PER_MILLION.get(model, {"input": None, "output": None})
    estimated_usd = None
    if rates["input"] is not None and rates["output"] is not None:
        estimated_usd = (
            prompt_tokens / 1_000_000 * rates["input"]
            + completion_tokens / 1_000_000 * rates["output"]
        )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_usd": estimated_usd,
        "usage_raw": usage,
    }


async def list_together_model_ids(client: httpx.AsyncClient) -> list[str]:
    resp = await client.get(
        TOGETHER_MODELS_URL,
        headers={"Authorization": f"Bearer {TOGETHER_API_KEY}"},
        timeout=60,
    )
    if resp.status_code == 401:
        raise SystemExit(
            "Together /v1/models returned 401. TOGETHER_API_KEY is missing or invalid."
        )
    resp.raise_for_status()
    data = resp.json()
    items = data if isinstance(data, list) else (data.get("data") or [])
    ids = []
    for m in items:
        if isinstance(m, dict):
            mid = m.get("id") or m.get("name")
            if mid:
                ids.append(mid)
        elif isinstance(m, str):
            ids.append(m)
    return ids


async def openai_model_exists(client: httpx.AsyncClient, model: str) -> bool:
    resp = await client.get(
        OPENAI_MODELS_URL,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        timeout=60,
    )
    if resp.status_code == 401:
        raise SystemExit(
            "OpenAI /v1/models returned 401. OPENAI_API_KEY is missing or invalid."
        )
    resp.raise_for_status()
    data = resp.json()
    ids = {m.get("id") for m in data.get("data", []) if isinstance(m, dict)}
    return model in ids


async def call_together(client: httpx.AsyncClient, model: str, prompt: str) -> dict:
    last_err = None
    for attempt in range(1, 4):
        try:
            resp = await client.post(
                TOGETHER_URL,
                headers={"Authorization": f"Bearer {TOGETHER_API_KEY}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                },
                timeout=60,
            )
            if resp.status_code in (429, 500, 502, 503):
                last_err = f"HTTP {resp.status_code}: {resp.text[:500]}"
                await asyncio.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            payload = resp.json()
            return {
                "answer": _message_content(payload),
                "http_status": resp.status_code,
                "retries": attempt - 1,
                **_usage_and_cost(model, payload),
                "error": None,
            }
        except httpx.HTTPStatusError as e:
            last_err = f"HTTP {e.response.status_code}: {e.response.text[:500]}"
            break
        except Exception as e:
            last_err = str(e)
            await asyncio.sleep(2 ** attempt)
    return {
        "answer": "",
        "http_status": None,
        "retries": 3,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_usd": None,
        "usage_raw": {},
        "error": last_err,
    }


async def call_openai(client: httpx.AsyncClient, model: str, prompt: str) -> dict:
    last_err = None
    for attempt in range(1, 4):
        try:
            resp = await client.post(
                OPENAI_URL,
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                },
                timeout=60,
            )
            if resp.status_code in (429, 500, 502, 503):
                last_err = f"HTTP {resp.status_code}: {resp.text[:500]}"
                await asyncio.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            payload = resp.json()
            return {
                "answer": _message_content(payload),
                "http_status": resp.status_code,
                "retries": attempt - 1,
                **_usage_and_cost(model, payload),
                "error": None,
            }
        except httpx.HTTPStatusError as e:
            last_err = f"HTTP {e.response.status_code}: {e.response.text[:500]}"
            break
        except Exception as e:
            last_err = str(e)
            await asyncio.sleep(2 ** attempt)
    return {
        "answer": "",
        "http_status": None,
        "retries": 3,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_usd": None,
        "usage_raw": {},
        "error": last_err,
    }


async def smoke_test(client: httpx.AsyncClient) -> None:
    print("=== Smoke tests (one cheap call per tier) ===")
    prompt = "Reply with the single word: pong"
    for tier, cfg in MODELS.items():
        print(f"\n--- Tier {tier} ({cfg['provider']}: {cfg['model']}) ---")
        if cfg["provider"] == "together":
            result = await call_together(client, cfg["model"], prompt)
        else:
            result = await call_openai(client, cfg["model"], prompt)
        print("RAW RESPONSE:")
        print(json.dumps(result, indent=2)[:4000])
        if result.get("error") or not (result.get("answer") or "").strip():
            raise SystemExit(
                f"Smoke test FAILED for Tier {tier} ({cfg['model']}): "
                f"{result.get('error') or 'empty response'}. "
                "Aborting before the 72-call batch."
            )
    print("\nAll three smoke tests succeeded.\n")


def write_aborted_results(blockers: list[str], extra: dict | None = None) -> None:
    payload = {
        "run_status": "aborted_before_any_inference",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "expected_calls": 72,
        "successful_calls": 0,
        "blockers": blockers,
        "models": MODELS,
        "responses": [],
    }
    if extra:
        payload.update(extra)
    with open("quality_benchmark_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("Wrote quality_benchmark_results.json (0 inference calls).")


async def run_all():
    blockers = []
    if not TOGETHER_API_KEY:
        blockers.append("TOGETHER_API_KEY is not set in this environment.")
    if not OPENAI_API_KEY:
        blockers.append("OPENAI_API_KEY is not set in this environment.")
    if blockers:
        write_aborted_results(blockers)
        raise SystemExit(
            "Set TOGETHER_API_KEY and OPENAI_API_KEY before running. "
            "This harness makes real, billed API calls. " + " ".join(blockers)
        )

    async with httpx.AsyncClient() as client:
        print("Querying Together GET /v1/models ...")
        try:
            together_ids = await list_together_model_ids(client)
        except SystemExit:
            raise
        except Exception as e:
            write_aborted_results([f"Together /v1/models request failed: {e}"])
            raise SystemExit(f"Together /v1/models request failed: {e}")

        missing = []
        for tier in (1, 2):
            slug = MODELS[tier]["model"]
            if slug not in together_ids:
                missing.append(slug)

        print(f"Together catalog returned {len(together_ids)} model ids.")
        for needle in ("gemma-3n", "gemma-3n-E4B", "Apriel", "apriel"):
            hits = [i for i in together_ids if needle.lower() in i.lower()]
            print(f"  catalog matches for {needle!r}: {hits or '(none)'}")

        if missing:
            msg = (
                "Together AI catalog does not currently list the EcoLogic Together "
                f"models: {missing}. Refusing to substitute a different model. "
                "See Together deprecations: google/gemma-3n-E4B-it removed "
                "2026-08-25; ServiceNow-AI/Apriel-1.6-15b-Thinker removed 2026-04-03."
            )
            write_aborted_results(
                [msg],
                extra={
                    "together_catalog_size": len(together_ids),
                    "together_catalog_matches": {
                        "gemma-3n": [i for i in together_ids if "gemma-3n" in i.lower()],
                        "apriel": [i for i in together_ids if "apriel" in i.lower()],
                    },
                },
            )
            raise SystemExit(msg)

        print("Querying OpenAI GET /v1/models ...")
        try:
            gpt4o_ok = await openai_model_exists(client, MODELS[3]["model"])
        except SystemExit:
            raise
        except Exception as e:
            write_aborted_results([f"OpenAI /v1/models request failed: {e}"])
            raise SystemExit(f"OpenAI /v1/models request failed: {e}")
        if not gpt4o_ok:
            msg = "OpenAI catalog does not currently list gpt-4o."
            write_aborted_results([msg])
            raise SystemExit(msg)

        await smoke_test(client)

        results = []
        for tier, cfg in MODELS.items():
            for q in QUESTIONS:
                if cfg["provider"] == "together":
                    call = await call_together(client, cfg["model"], q["prompt"])
                else:
                    call = await call_openai(client, cfg["model"], q["prompt"])
                answer = call.get("answer") or ""
                results.append({
                    "tier": tier,
                    "model": cfg["model"],
                    "question_id": q["id"],
                    "category": q["category"],
                    "prompt": q["prompt"],
                    "reference": q["reference"],
                    "answer": answer,
                    "auto_correct": (
                        q["reference"].lower() in answer.lower()
                        if q["reference"] and answer else None
                    ),
                    "judge_score": None,
                    "judge_justification": None,
                    "http_status": call.get("http_status"),
                    "retries": call.get("retries"),
                    "error": call.get("error"),
                    "prompt_tokens": call.get("prompt_tokens"),
                    "completion_tokens": call.get("completion_tokens"),
                    "total_tokens": call.get("total_tokens"),
                    "estimated_usd": call.get("estimated_usd"),
                })
                time.sleep(0.15)

    successes = sum(1 for r in results if r.get("error") is None and (r.get("answer") or "").strip())
    payload = {
        "run_status": "complete" if successes == 72 else "incomplete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "expected_calls": 72,
        "successful_calls": successes,
        "models": MODELS,
        "responses": results,
    }
    with open("quality_benchmark_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(results)} responses to quality_benchmark_results.json "
          f"({successes}/72 successful).")
    print("Factual (auto-graded) accuracy by tier:")
    for tier in (1, 2, 3):
        factual = [r for r in results if r["tier"] == tier and r["category"] == "factual"]
        graded = [r for r in factual if r["auto_correct"] is not None]
        if not graded:
            print(f"  Tier {tier}: n/a (no auto-gradable answers)")
            continue
        acc = sum(1 for r in graded if r["auto_correct"]) / len(graded)
        print(f"  Tier {tier}: {acc:.1%} ({sum(1 for r in graded if r['auto_correct'])}/{len(graded)})")
    print("\nReasoning/code responses need a rubric pass (human or judge-model) "
          "before scoring -- see 'auto_correct: null' entries in the JSON.")
    if successes < 72:
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(run_all())

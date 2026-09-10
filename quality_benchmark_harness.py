"""
EcoLogic matched-query quality benchmark.

Original Together slugs (Gemma 3N E4B, Apriel 1.6 15B) are gone from serverless.
Tier 1/2 use energy-adjacent Together replacements. Tier 3 is gpt-4o.

Usage:
    export TOGETHER_API_KEY=...
    export OPENAI_API_KEY=...
    python3 quality_benchmark_harness.py
    python3 quality_benchmark_harness.py --rerun-tier 3
"""

import os
import json
import re
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

UA = "Mozilla/5.0 EcoLogicBenchmark/1.0"
COMMON_HEADERS = {"User-Agent": UA, "Content-Type": "application/json"}

MAX_TOKENS = 1024
SMOKE_MAX_TOKENS = 256
# Not one of the three tiers under test. GLM-5.3-Flash only wrote
# reasoning_content and never emitted a grade; MiniMax M3 returns SCORE/WHY.
JUDGE_MODEL = "MiniMaxAI/MiniMax-M3"

# Energy-adjacent replacements for retired Together IDs; Tier 3 is gpt-4o.
MODELS = {
    1: {"provider": "together", "model": "Qwen/Qwen3.5-9B", "energy_per_1k_tokens": 1.1},
    2: {"provider": "together", "model": "openai/gpt-oss-20b", "energy_per_1k_tokens": 2.0},
    3: {"provider": "openai", "model": "gpt-4o", "energy_per_1k_tokens": 60},
}

FALLBACK_RATES_PER_MILLION = {
    "Qwen/Qwen3.5-9B": {"input": 0.17, "output": 0.25},
    "openai/gpt-oss-20b": {"input": 0.05, "output": 0.20},
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"input": 1.04, "output": 1.04},
    "zai-org/GLM-5.3-Flash": {"input": 0.15, "output": 0.50},
    "MiniMaxAI/MiniMax-M3": {"input": 0.30, "output": 1.20},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

QUESTIONS = [
    {"id": "f1", "category": "factual", "prompt": "What is the capital of Australia?", "reference": "Canberra"},
    {"id": "f2", "category": "factual", "prompt": "Who wrote 'Pride and Prejudice'?", "reference": "Jane Austen"},
    {"id": "f3", "category": "factual", "prompt": "What is the atomic number of carbon?", "reference": "6"},
    {"id": "f4", "category": "factual", "prompt": "In what year did the Titanic sink?", "reference": "1912"},
    {"id": "f5", "category": "factual", "prompt": "What is the largest planet in the solar system?", "reference": "Jupiter"},
    {"id": "f6", "category": "factual", "prompt": "What is the chemical formula for table salt?", "reference": "NaCl"},
    {"id": "f7", "category": "factual", "prompt": "Who painted 'Starry Night'?", "reference": "Van Gogh"},
    {"id": "f8", "category": "factual", "prompt": "What is the smallest prime number?", "reference": "2"},
    {"id": "r1", "category": "reasoning", "prompt": "Compare the trade-offs of monolithic vs microservice architectures.", "reference": None},
    {"id": "r2", "category": "reasoning", "prompt": "Analyze the main causes of urban housing shortages.", "reference": None},
    {"id": "r3", "category": "reasoning", "prompt": "What are the pros and cons of universal basic income?", "reference": None},
    {"id": "r4", "category": "reasoning", "prompt": "Contrast supervised and unsupervised learning, with an example of each.", "reference": None},
    {"id": "r5", "category": "reasoning", "prompt": "Evaluate the environmental trade-offs of nuclear power vs solar power.", "reference": None},
    {"id": "r6", "category": "reasoning", "prompt": "Step by step, explain why inflation erodes purchasing power.", "reference": None},
    {"id": "r7", "category": "reasoning", "prompt": "Assess the risks of over-reliance on a single cloud provider.", "reference": None},
    {"id": "r8", "category": "reasoning", "prompt": "Compare federated learning and centralized training for privacy.", "reference": None},
    {"id": "c1", "category": "code", "prompt": "Write a Python function that returns the nth Fibonacci number.", "reference": None},
    {"id": "c2", "category": "code", "prompt": "Write a JavaScript function that debounces another function.", "reference": None},
    {"id": "c3", "category": "code", "prompt": "Implement binary search in Java.", "reference": None},
    {"id": "c4", "category": "code", "prompt": "Write a SQL query to find the second-highest salary in an employees table.", "reference": None},
    {"id": "c5", "category": "code", "prompt": "Debug this Python: `def f(x): return x / 0` -- explain the error and fix it.", "reference": None},
    {"id": "c6", "category": "code", "prompt": "Write a Rust function that reverses a string.", "reference": None},
    {"id": "c7", "category": "code", "prompt": "Implement a simple LRU cache in Python.", "reference": None},
    {"id": "c8", "category": "code", "prompt": "Write a regex to validate an email address, in Python.", "reference": None},
]


def _stringify_content(content) -> str:
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


def _message_fields(payload: dict) -> tuple[str, str]:
    msg = payload["choices"][0]["message"]
    content = _stringify_content(msg.get("content")).strip()
    reasoning = _stringify_content(
        msg.get("reasoning") or msg.get("reasoning_content")
    ).strip()
    return content, reasoning


def _usage_and_cost(model: str, payload: dict) -> dict:
    usage = payload.get("usage")
    empty = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "estimated_usd": None,
        "usage_raw": usage if isinstance(usage, dict) else {},
        "usage_complete": False,
    }
    if not isinstance(usage, dict):
        return empty
    if usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None:
        return empty
    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    total_tokens = (
        int(usage["total_tokens"])
        if usage.get("total_tokens") is not None
        else prompt_tokens + completion_tokens
    )
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
        "usage_complete": True,
    }


def _auth_headers(provider: str) -> dict:
    headers = dict(COMMON_HEADERS)
    if provider == "together":
        headers["Authorization"] = f"Bearer {TOGETHER_API_KEY}"
    else:
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
    return headers


async def list_together_model_ids(client: httpx.AsyncClient) -> list[str]:
    resp = await client.get(
        TOGETHER_MODELS_URL,
        headers=_auth_headers("together"),
        timeout=60,
    )
    if resp.status_code == 401:
        raise SystemExit("Together /v1/models returned 401. TOGETHER_API_KEY is missing or invalid.")
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
        headers=_auth_headers("openai"),
        timeout=60,
    )
    if resp.status_code == 401:
        raise SystemExit("OpenAI /v1/models returned 401. OPENAI_API_KEY is missing or invalid.")
    resp.raise_for_status()
    data = resp.json()
    ids = {m.get("id") for m in data.get("data", []) if isinstance(m, dict)}
    return model in ids


async def _post_chat(client: httpx.AsyncClient, provider: str, url: str, model: str, prompt: str, max_tokens: int) -> dict:
    last_err = None
    retries = 0
    for attempt in range(1, 5):
        try:
            resp = await client.post(
                url,
                headers=_auth_headers(provider),
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                },
                timeout=120,
            )
            if resp.status_code == 429 and (
                "insufficient_quota" in resp.text
                or "credit_balance_exhausted" in resp.text
            ):
                raise SystemExit(
                    f"{provider} {model} returned HTTP 429 insufficient_quota / "
                    "credit_balance_exhausted. Add billing credits and retry. "
                    f"Body: {resp.text[:400]}"
                )
            if resp.status_code in (429, 500, 502, 503):
                last_err = f"HTTP {resp.status_code}: {resp.text[:500]}"
                retries += 1
                await asyncio.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            payload = resp.json()
            content, reasoning = _message_fields(payload)
            answer = content or reasoning
            return {
                "answer": answer,
                "content": content,
                "reasoning": reasoning,
                "http_status": resp.status_code,
                "retries": retries,
                **_usage_and_cost(model, payload),
                "error": None,
            }
        except httpx.HTTPStatusError as e:
            last_err = f"HTTP {e.response.status_code}: {e.response.text[:500]}"
            if e.response.status_code in (429, 500, 502, 503) and attempt < 4:
                retries += 1
                await asyncio.sleep(2 ** attempt)
                continue
            break
        except Exception as e:
            last_err = str(e)
            retries += 1
            await asyncio.sleep(2 ** attempt)
    return {
        "answer": "",
        "content": "",
        "reasoning": "",
        "http_status": None,
        "retries": retries,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_usd": None,
        "usage_raw": {},
        "error": last_err,
    }


async def call_model(client: httpx.AsyncClient, cfg: dict, prompt: str, max_tokens: int = MAX_TOKENS) -> dict:
    if cfg["provider"] == "together":
        return await _post_chat(client, "together", TOGETHER_URL, cfg["model"], prompt, max_tokens)
    return await _post_chat(client, "openai", OPENAI_URL, cfg["model"], prompt, max_tokens)


async def smoke_test(client: httpx.AsyncClient) -> None:
    print("=== Smoke tests (one cheap call per tier) ===")
    prompt = "Reply with the single word: pong"
    for tier, cfg in MODELS.items():
        print(f"\n--- Tier {tier} ({cfg['provider']}: {cfg['model']}) ---")
        result = await call_model(client, cfg, prompt, max_tokens=SMOKE_MAX_TOKENS)
        printable = {k: v for k, v in result.items() if k != "usage_raw"}
        printable["answer"] = (printable.get("answer") or "")[:500]
        printable["content"] = (printable.get("content") or "")[:500]
        printable["reasoning"] = (printable.get("reasoning") or "")[:500]
        print("RAW RESPONSE:")
        print(json.dumps(printable, indent=2)[:4000])
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
        "judge_model": JUDGE_MODEL,
        "responses": [],
    }
    if extra:
        payload.update(extra)
    with open("quality_benchmark_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("Wrote quality_benchmark_results.json (0 inference calls).")


def parse_judge_output(text: str) -> tuple[int | None, str]:
    text = (text or "").strip()
    m = re.search(r"SCORE:\s*([01])\b", text, re.IGNORECASE)
    if m:
        why = ""
        wm = re.search(r"WHY:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if wm:
            why = wm.group(1).strip().splitlines()[0].strip()
        return int(m.group(1)), why or text[:300]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    blob = match.group(0) if match else text
    try:
        data = json.loads(blob)
        score = data.get("score")
        if score in (0, 1, "0", "1"):
            return int(score), str(data.get("justification") or "").strip() or text[:300]
    except Exception:
        pass
    m = re.search(r'"score"\s*:\s*([01])', text)
    if m:
        return int(m.group(1)), text[:300]
    return None, f"unparseable judge output: {text[:300]}"


def judge_prompt(row: dict) -> str:
    if row["category"] == "reasoning":
        rubric = (
            "Score 1 if the answer correctly addresses the actual trade-offs or "
            "comparison asked for, without factual errors or dodging the question. "
            "Score 0 otherwise."
        )
    else:
        rubric = (
            "Score 1 if the code would plausibly run and correctly solve the stated "
            "task (check logic, not just that code-shaped text was produced). "
            "Score 0 otherwise."
        )
    return (
        "Grade the answer. Think briefly, then output exactly two lines:\n"
        "SCORE: 0 or SCORE: 1\n"
        "WHY: one sentence\n"
        f"Rubric: {rubric}\n\n"
        f"Question: {row['prompt']}\n\n"
        f"Answer:\n{row['answer']}"
    )


async def judge_rows(client: httpx.AsyncClient, results: list[dict], tiers: set[int] | None = None) -> dict:
    judge_cfg = {"provider": "together", "model": JUDGE_MODEL}
    judge_calls = 0
    judge_errors = 0
    judge_retries = 0
    judge_cost = 0.0
    for row in results:
        if tiers is not None and row["tier"] not in tiers:
            continue
        if row["category"] not in ("reasoning", "code"):
            continue
        if row.get("error") or not (row.get("answer") or "").strip():
            row["judge_score"] = 0
            row["judge_justification"] = "No usable model answer to grade."
            row["judge_model"] = JUDGE_MODEL
            continue
        call = await call_model(client, judge_cfg, judge_prompt(row), max_tokens=700)
        judge_calls += 1
        judge_retries += int(call.get("retries") or 0)
        if call.get("estimated_usd"):
            judge_cost += call["estimated_usd"]
        if call.get("error"):
            judge_errors += 1
            row["judge_score"] = None
            row["judge_justification"] = f"judge error: {call['error']}"
            row["judge_model"] = JUDGE_MODEL
            continue
        score, why = parse_judge_output(call.get("answer") or "")
        row["judge_score"] = score
        row["judge_justification"] = why
        row["judge_model"] = JUDGE_MODEL
        await asyncio.sleep(0.15)
    return {
        "judge_model": JUDGE_MODEL,
        "judge_calls": judge_calls,
        "judge_errors": judge_errors,
        "judge_retries": judge_retries,
        "judge_estimated_usd": judge_cost,
    }


def save_results(payload: dict) -> None:
    with open("quality_benchmark_results.json", "w") as f:
        json.dump(payload, f, indent=2)


async def run_all():
    needs_together = any(c["provider"] == "together" for c in MODELS.values()) or True
    needs_openai = any(c["provider"] == "openai" for c in MODELS.values())
    blockers = []
    if needs_together and not TOGETHER_API_KEY:
        blockers.append("TOGETHER_API_KEY is not set in this environment.")
    if needs_openai and not OPENAI_API_KEY:
        blockers.append("OPENAI_API_KEY is not set in this environment.")
    if blockers:
        write_aborted_results(blockers)
        raise SystemExit(" ".join(blockers))

    notes = []

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
        for tier, cfg in MODELS.items():
            if cfg["provider"] == "together" and cfg["model"] not in together_ids:
                missing.append(cfg["model"])
        if JUDGE_MODEL not in together_ids:
            missing.append(JUDGE_MODEL)

        print(f"Together catalog returned {len(together_ids)} model ids.")
        for slug in [MODELS[t]["model"] for t in (1, 2, 3)] + [JUDGE_MODEL]:
            print(f"  {slug}: {'YES' if slug in together_ids else 'NO'}")

        if missing:
            msg = f"Together catalog missing required slugs: {missing}. Refusing to substitute."
            write_aborted_results([msg], extra={"together_catalog_size": len(together_ids)})
            raise SystemExit(msg)

        if needs_openai:
            print("Querying OpenAI GET /v1/models ...")
            gpt4o_ok = await openai_model_exists(client, MODELS[3]["model"])
            if not gpt4o_ok:
                msg = f"OpenAI catalog does not currently list {MODELS[3]['model']}."
                write_aborted_results([msg])
                raise SystemExit(msg)

        await smoke_test(client)

        results = []
        for tier, cfg in MODELS.items():
            for q in QUESTIONS:
                print(f"Calling T{tier} {q['id']} ...", flush=True)
                call = await call_model(client, cfg, q["prompt"])
                answer = call.get("answer") or ""
                results.append({
                    "tier": tier,
                    "model": cfg["model"],
                    "question_id": q["id"],
                    "category": q["category"],
                    "prompt": q["prompt"],
                    "reference": q["reference"],
                    "answer": answer,
                    "reasoning": call.get("reasoning") or "",
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
                await asyncio.sleep(0.15)

        successes = sum(
            1 for r in results if r.get("error") is None and (r.get("answer") or "").strip()
        )
        payload = {
            "run_status": "inference_done_pending_judge" if successes == 72 else "incomplete",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "expected_calls": 72,
            "successful_calls": successes,
            "models": MODELS,
            "notes": notes,
            "responses": results,
        }
        save_results(payload)
        print(f"Wrote {len(results)} responses ({successes}/72 successful). Starting judge...")

        judge_meta = await judge_rows(client, results)
        payload["run_status"] = "complete" if successes == 72 else "incomplete"
        payload["judge"] = judge_meta
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        payload["responses"] = results
        save_results(payload)

def print_summary(results: list[dict], successes: int) -> None:
    print(f"Wrote quality_benchmark_results.json ({successes}/72 successful).")
    print("Factual (auto-graded) accuracy by tier:")
    for tier in (1, 2, 3):
        factual = [r for r in results if r["tier"] == tier and r["category"] == "factual"]
        graded = [r for r in factual if r["auto_correct"] is not None]
        if not graded:
            print(f"  Tier {tier}: n/a")
            continue
        n = sum(1 for r in graded if r["auto_correct"])
        print(f"  Tier {tier}: {n/len(graded):.1%} ({n}/{len(graded)})")
    print("Judge-graded reasoning/code:")
    for tier in (1, 2, 3):
        for cat in ("reasoning", "code"):
            rows = [r for r in results if r["tier"] == tier and r["category"] == cat]
            scored = [r for r in rows if r.get("judge_score") in (0, 1)]
            if not scored:
                print(f"  Tier {tier} {cat}: n/a")
                continue
            n = sum(r["judge_score"] for r in scored)
            print(f"  Tier {tier} {cat}: {n/len(scored):.1%} ({n}/{len(scored)})")


async def judge_only() -> None:
    if not TOGETHER_API_KEY:
        raise SystemExit("TOGETHER_API_KEY is not set.")
    with open("quality_benchmark_results.json") as f:
        payload = json.load(f)
    results = payload.get("responses") or []
    if len(results) != 72:
        raise SystemExit(f"Expected 72 responses to re-judge, found {len(results)}.")
    async with httpx.AsyncClient() as client:
        together_ids = await list_together_model_ids(client)
        if JUDGE_MODEL not in together_ids:
            raise SystemExit(f"Judge model {JUDGE_MODEL} not in Together catalog.")
        print(f"Re-judging 48 reasoning/code rows with {JUDGE_MODEL} ...")
        judge_meta = await judge_rows(client, results)
    successes = sum(
        1 for r in results if r.get("error") is None and (r.get("answer") or "").strip()
    )
    payload["run_status"] = "complete" if successes == 72 else "incomplete"
    payload["judge"] = judge_meta
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["responses"] = results
    save_results(payload)
    print_summary(results, successes)
    if successes < 72:
        sys.exit(2)


async def rerun_tier(tier: int) -> None:
    cfg = MODELS[tier]
    if cfg["provider"] == "together" and not TOGETHER_API_KEY:
        raise SystemExit("TOGETHER_API_KEY is not set.")
    if cfg["provider"] == "openai" and not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY is not set.")
    if not TOGETHER_API_KEY:
        raise SystemExit("TOGETHER_API_KEY is required for the MiniMax judge.")
    with open("quality_benchmark_results.json") as f:
        payload = json.load(f)
    results = payload.get("responses") or []
    if len(results) != 72:
        raise SystemExit(f"Expected 72 existing responses, found {len(results)}.")

    async with httpx.AsyncClient() as client:
        if cfg["provider"] == "openai":
            ok = await openai_model_exists(client, cfg["model"])
            if not ok:
                raise SystemExit(f"OpenAI catalog does not list {cfg['model']}.")
        print(f"=== Smoke test Tier {tier} ({cfg['model']}) ===")
        smoke = await call_model(
            client, cfg, "Reply with the single word: pong", max_tokens=SMOKE_MAX_TOKENS
        )
        print(json.dumps({k: (v[:400] if isinstance(v, str) else v) for k, v in smoke.items() if k != "usage_raw"}, indent=2)[:2000])
        if smoke.get("error") or not (smoke.get("answer") or "").strip():
            raise SystemExit(
                f"Smoke test FAILED for Tier {tier}: {smoke.get('error') or 'empty response'}"
            )
        print("Smoke test succeeded. Replacing existing Tier", tier, "rows...")

        new_rows = []
        for q in QUESTIONS:
            print(f"Calling T{tier} {q['id']} ...", flush=True)
            call = await call_model(client, cfg, q["prompt"])
            answer = call.get("answer") or ""
            new_rows.append({
                "tier": tier,
                "model": cfg["model"],
                "question_id": q["id"],
                "category": q["category"],
                "prompt": q["prompt"],
                "reference": q["reference"],
                "answer": answer,
                "reasoning": call.get("reasoning") or "",
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
            await asyncio.sleep(0.15)

        kept = [r for r in results if r["tier"] != tier]
        results = kept + new_rows
        results.sort(key=lambda r: (r["tier"], r["question_id"]))
        payload["models"] = MODELS
        payload["responses"] = results
        payload["notes"] = list(payload.get("notes") or []) + [
            f"Tier {tier} rerun with {cfg['model']} at {datetime.now(timezone.utc).isoformat()}"
        ]
        save_results(payload)

        print(f"Judging Tier {tier} reasoning/code with {JUDGE_MODEL} ...")
        extra = await judge_rows(client, results, tiers={tier})
        old = payload.get("judge") or {}
        payload["judge"] = {
            "judge_model": JUDGE_MODEL,
            "judge_calls": int(old.get("judge_calls") or 0) + extra["judge_calls"],
            "judge_errors": int(old.get("judge_errors") or 0) + extra["judge_errors"],
            "judge_retries": int(old.get("judge_retries") or 0) + extra["judge_retries"],
            "judge_estimated_usd": float(old.get("judge_estimated_usd") or 0)
            + float(extra["judge_estimated_usd"] or 0),
            "last_rerun_tier": extra,
        }

    successes = sum(
        1 for r in results if r.get("error") is None and (r.get("answer") or "").strip()
    )
    payload["run_status"] = "complete" if successes == 72 else "incomplete"
    payload["successful_calls"] = successes
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["responses"] = results
    save_results(payload)
    print_summary(results, successes)
    if successes < 72:
        sys.exit(2)


if __name__ == "__main__":
    if "--judge-only" in sys.argv:
        asyncio.run(judge_only())
    elif "--rerun-tier" in sys.argv:
        idx = sys.argv.index("--rerun-tier")
        asyncio.run(rerun_tier(int(sys.argv[idx + 1])))
    else:
        asyncio.run(run_all())

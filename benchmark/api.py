"""Shared provider client for the EcoLogic publication-grade benchmark.

All generation calls run at temperature 0. Token counts come from the
provider `usage` field, never from word-count estimates.
"""

import asyncio
import os
import httpx

TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

UA = "Mozilla/5.0 EcoLogicBenchmark/1.0"
COMMON_HEADERS = {"User-Agent": UA, "Content-Type": "application/json"}

# Tier models. Tier 1/2 are energy-adjacent Together replacements for the
# retired Gemma 3N E4B / Apriel 1.6 15B slugs; Tier 3 is the paper's gpt-4o.
# paper_energy_per_1k = the rate in backend/main.py (the paper's own numbers).
# subst_energy_per_1k = rate adjusted for the larger substitute models.
MODELS = {
    1: {
        "provider": "together",
        "model": "Qwen/Qwen3.5-9B",
        "paper_energy_per_1k": 0.5,
        "subst_energy_per_1k": 1.1,
    },
    2: {
        "provider": "together",
        "model": "openai/gpt-oss-20b",
        "paper_energy_per_1k": 1.5,
        "subst_energy_per_1k": 2.0,
    },
    3: {
        "provider": "openai",
        "model": "gpt-4o",
        "paper_energy_per_1k": 60.0,
        "subst_energy_per_1k": 60.0,
    },
}

GPT5_ENERGY_PER_1K = 500.0

RATES_PER_MILLION = {
    "Qwen/Qwen3.5-9B": {"input": 0.17, "output": 0.25},
    "openai/gpt-oss-20b": {"input": 0.05, "output": 0.20},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

# Uniform across tiers and benchmarks on purpose, and set high enough that the
# cap is not binding for any tier. Tier 1/2 are reasoning models that spend
# most of their budget in reasoning_content: at 4096 Tier 1 was truncated on
# 57/164 HumanEval items while gpt-4o never exceeded 458 tokens, which measures
# the cap rather than the model. See raw_results/responses_cap4096.jsonl.
MAX_TOKENS = {"humaneval": 16384, "gsm8k": 16384, "mmlu": 16384}


def auth_headers(provider: str) -> dict:
    headers = dict(COMMON_HEADERS)
    key = os.environ.get("TOGETHER_API_KEY" if provider == "together" else "OPENAI_API_KEY")
    headers["Authorization"] = f"Bearer {key}"
    return headers


def _stringify(content) -> str:
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


def message_fields(payload: dict) -> tuple[str, str]:
    msg = payload["choices"][0]["message"]
    content = _stringify(msg.get("content")).strip()
    reasoning = _stringify(msg.get("reasoning") or msg.get("reasoning_content")).strip()
    return content, reasoning


def usage_and_cost(model: str, payload: dict) -> dict:
    usage = payload.get("usage")
    finish = (payload["choices"][0].get("finish_reason") if payload.get("choices") else None)
    empty = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "usd": None,
        "finish_reason": finish,
        "usage_complete": False,
    }
    if not isinstance(usage, dict):
        return empty
    if usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None:
        return empty
    pt = int(usage["prompt_tokens"])
    ct = int(usage["completion_tokens"])
    tt = int(usage["total_tokens"]) if usage.get("total_tokens") is not None else pt + ct
    rates = RATES_PER_MILLION.get(model)
    usd = None
    if rates:
        usd = pt / 1_000_000 * rates["input"] + ct / 1_000_000 * rates["output"]
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
        "usd": usd,
        "finish_reason": finish,
        "usage_complete": True,
    }


# Phase-granular rather than a blanket 300s: a long generation legitimately
# needs a long *read* (a 16,384-token completion takes ~190s), but a connect or
# pool acquisition that takes minutes means a stale/stuck connection, and under
# high concurrency waiting 300s on those pins every worker and halts the run.
REQUEST_TIMEOUT = httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=60.0)


async def chat(
    client: httpx.AsyncClient,
    tier: int,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    seed: int | None = None,
) -> dict:
    cfg = MODELS[tier]
    provider = cfg["provider"]
    url = TOGETHER_URL if provider == "together" else OPENAI_URL
    body = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        body["seed"] = seed

    last_err = None
    retries = 0
    for attempt in range(1, 6):
        try:
            resp = await client.post(url, headers=auth_headers(provider), json=body,
                                     timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429 and (
                "insufficient_quota" in resp.text or "credit_balance_exhausted" in resp.text
            ):
                raise SystemExit(
                    f"{provider}/{cfg['model']} HTTP 429 out of credit. Body: {resp.text[:300]}"
                )
            if resp.status_code in (408, 429, 500, 502, 503, 504, 520, 524):
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                retries += 1
                await asyncio.sleep(min(2 ** attempt, 30))
                continue
            resp.raise_for_status()
            payload = resp.json()
            content, reasoning = message_fields(payload)
            return {
                "model": cfg["model"],
                "content": content,
                "reasoning": reasoning,
                "answer": content or reasoning,
                "http_status": resp.status_code,
                "retries": retries,
                "error": None,
                **usage_and_cost(cfg["model"], payload),
            }
        except SystemExit:
            raise
        except httpx.HTTPStatusError as e:
            last_err = f"HTTP {e.response.status_code}: {e.response.text[:300]}"
            break
        except Exception as e:  # network/timeout/decode
            last_err = f"{type(e).__name__}: {e}"
            retries += 1
            await asyncio.sleep(min(2 ** attempt, 30))
    return {
        "model": cfg["model"],
        "content": "",
        "reasoning": "",
        "answer": "",
        "http_status": None,
        "retries": retries,
        "error": last_err,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "usd": None,
        "finish_reason": None,
        "usage_complete": False,
    }

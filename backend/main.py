import os
import re
import json
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio

load_dotenv()

app = FastAPI(title="EcoLogic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model configuration - optimized for energy efficiency (ultra-low energy models)
MODELS = {
    "tier1": {
        "name": "google/gemma-3n-E4B-it",
        "provider": "together",
        "energy_per_1k_tokens": 0.5,  # Joules (4B effective params, FP8, extremely efficient)
    },
    "tier2": {
        "name": "ServiceNow-AI/Apriel-1.6-15b-Thinker",
        "provider": "together",
        "energy_per_1k_tokens": 1.5,  # Joules (15B params, FREE, frontier performance)
    },
    "tier3": {
        "name": "gpt-4o",
        "provider": "openai",
        "energy_per_1k_tokens": 60,
    },
}

CHATGPT_ENERGY_PER_1K = 60  # Baseline for comparison (GPT-4o)


class QueryRequest(BaseModel):
    prompt: str


class QueryResponse(BaseModel):
    response: str
    tier: int
    model: str
    tokens: int
    energy_used: float
    energy_saved: float
    escalated: bool


class ClassificationResult(BaseModel):
    difficulty: str
    risk: str
    recommended_tier: int
    reason: str


async def classify_prompt_nlp(prompt: str) -> ClassificationResult:
    """Advanced NLP-based classification using ultra-low-energy Gemma 3N model."""
    classification_prompt = f"""Analyze this user query and classify it for AI model routing.

Query: "{prompt}"

Classify into ONE tier:
- Tier 1: Simple factual questions, definitions, basic explanations (80% of queries)
- Tier 2: Comparisons, multi-step reasoning, analysis requiring deeper thought
- Tier 3: Code generation/debugging, medical/legal advice, technical implementation

Respond in JSON format:
{{"tier": 1, "reason": "brief explanation"}}

Be conservative - default to Tier 1 unless clearly complex."""

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "google/gemma-3n-E4B-it",
                "messages": [
                    {"role": "user", "content": classification_prompt}
                ],
                "max_tokens": 100,
                "temperature": 0.3,
                "response_format": {"type": "json_object"}
            },
            timeout=10.0,
        )
        
        if response.status_code != 200:
            # Fallback to Tier 1 on error
            return ClassificationResult(
                difficulty="easy",
                risk="low",
                recommended_tier=1,
                reason="Classification fallback"
            )
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        try:
            result = json.loads(content)
            tier = result.get("tier", 1)
            reason = result.get("reason", "NLP classification")
            
            # Map tier to difficulty
            difficulty_map = {1: "easy", 2: "medium", 3: "hard"}
            risk_map = {1: "low", 2: "low", 3: "high"}
            
            return ClassificationResult(
                difficulty=difficulty_map.get(tier, "easy"),
                risk=risk_map.get(tier, "low"),
                recommended_tier=min(max(tier, 1), 3),
                reason=reason
            )
        except json.JSONDecodeError:
            # Fallback to Tier 1
            return ClassificationResult(
                difficulty="easy",
                risk="low",
                recommended_tier=1,
                reason="Parse error - defaulting to Tier 1"
            )


async def query_together(model: str, prompt: str) -> tuple[str, int]:
    """Query Together AI API."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Be concise and direct. Keep responses under 200 words unless more detail is specifically requested."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 512,
                "temperature": 0.7,
            },
            timeout=30.0,
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Together API error: {response.text}")
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", int(len(content.split()) * 1.3))
        
        return content, int(tokens)


async def query_openai(model: str, prompt: str) -> tuple[str, int]:
    """Query OpenAI API."""
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-key-here":
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Be concise and direct. Keep responses under 200 words unless more detail is specifically requested."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 512,
                "temperature": 0.7,
            },
            timeout=30.0,
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {response.text}")
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", int(len(content.split()) * 1.3))
        
        return content, int(tokens)


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """Main endpoint: classify prompt, route to appropriate tier, return response with energy stats."""
    
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    # Step 1: Classify the prompt using NLP (ultra-low energy Gemma 3N)
    classification = await classify_prompt_nlp(prompt)
    tier = classification.recommended_tier
    tier_key = f"tier{tier}"
    
    # Step 2: Route to appropriate model
    model_config = MODELS[tier_key]
    model_name = model_config["name"]
    provider = model_config["provider"]
    
    if provider == "together":
        response_text, tokens = await query_together(model_name, prompt)
    else:
        response_text, tokens = await query_openai(model_name, prompt)
    
    # Step 3: Calculate energy
    energy_per_1k = model_config["energy_per_1k_tokens"]
    energy_used = (tokens / 1000) * energy_per_1k
    energy_if_chatgpt = (tokens / 1000) * CHATGPT_ENERGY_PER_1K
    energy_saved = energy_if_chatgpt - energy_used
    
    return QueryResponse(
        response=response_text,
        tier=tier,
        model=model_name,
        tokens=tokens,
        energy_used=round(energy_used, 2),
        energy_saved=round(max(0, energy_saved), 2),
        escalated=tier > 1,
    )


async def stream_together(model: str, prompt: str, tier: int, energy_per_1k: float):
    """Stream response from Together AI API."""
    full_content = ""
    
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "https://api.together.xyz/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Be concise and direct. Keep responses under 200 words unless more detail is specifically requested."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 512,
                "temperature": 0.7,
                "stream": True,
            },
            timeout=60.0,
        ) as response:
            # Send initial metadata
            yield f"data: {json.dumps({'type': 'meta', 'tier': tier, 'model': model})}\n\n"
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_content += content
                            yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                    except json.JSONDecodeError:
                        continue
    
    # Calculate and send final energy stats
    tokens = int(len(full_content.split()) * 1.3)
    energy_used = (tokens / 1000) * energy_per_1k
    energy_if_chatgpt = (tokens / 1000) * CHATGPT_ENERGY_PER_1K
    energy_saved = max(0, energy_if_chatgpt - energy_used)
    
    yield f"data: {json.dumps({'type': 'done', 'tokens': tokens, 'energy_used': round(energy_used, 2), 'energy_saved': round(energy_saved, 2)})}\n\n"


async def stream_openai(model: str, prompt: str, tier: int, energy_per_1k: float):
    """Stream response from OpenAI API."""
    full_content = ""
    
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Be concise and direct. Keep responses under 200 words unless more detail is specifically requested."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 512,
                "temperature": 0.7,
                "stream": True,
            },
            timeout=60.0,
        ) as response:
            # Send initial metadata
            yield f"data: {json.dumps({'type': 'meta', 'tier': tier, 'model': model})}\n\n"
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_content += content
                            yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                    except json.JSONDecodeError:
                        continue
    
    # Calculate and send final energy stats
    tokens = int(len(full_content.split()) * 1.3)
    energy_used = (tokens / 1000) * energy_per_1k
    energy_if_chatgpt = (tokens / 1000) * CHATGPT_ENERGY_PER_1K
    energy_saved = max(0, energy_if_chatgpt - energy_used)
    
    yield f"data: {json.dumps({'type': 'done', 'tokens': tokens, 'energy_used': round(energy_used, 2), 'energy_saved': round(energy_saved, 2)})}\n\n"


@app.post("/query/stream")
async def handle_query_stream(request: QueryRequest):
    """Streaming endpoint: returns Server-Sent Events with response chunks."""
    
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    # Classify the prompt using NLP
    classification = await classify_prompt_nlp(prompt)
    tier = classification.recommended_tier
    tier_key = f"tier{tier}"
    
    model_config = MODELS[tier_key]
    model_name = model_config["name"]
    provider = model_config["provider"]
    energy_per_1k = model_config["energy_per_1k_tokens"]
    
    if provider == "together":
        generator = stream_together(model_name, prompt, tier, energy_per_1k)
    else:
        generator = stream_openai(model_name, prompt, tier, energy_per_1k)
    
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "EcoLogic API is running"}

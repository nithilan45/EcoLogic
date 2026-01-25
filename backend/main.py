import os
import re
import json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

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

# Model configuration - optimized for energy efficiency
MODELS = {
    "tier1": {
        "name": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "provider": "together",
        "energy_per_1k_tokens": 1,  # Joules (small, efficient 3B model)
    },
    "tier2": {
        "name": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "provider": "together",
        "energy_per_1k_tokens": 4,
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


# Keywords for fast classification (no API call needed)
TIER3_KEYWORDS = [
    r'\bcode\b', r'\bfunction\b', r'\bdebug\b', r'\bpython\b', r'\bjavascript\b',
    r'\bjava\b', r'\bc\+\+\b', r'\btypescript\b', r'\brust\b', r'\bgo\b',
    r'\bsql\b', r'\bhtml\b', r'\bcss\b', r'\bapi\b', r'\bscript\b',
    r'\bprogram\b', r'\balgorithm\b', r'\bmedical\b', r'\blegal\b',
    r'\bdiagnos', r'\bsymptom', r'\blawsuit\b', r'\bcontract\b',
    r'\bgpt-4\b', r'\bbest quality\b', r'\bwrite.*code\b', r'\bfix.*bug\b',
]

TIER2_KEYWORDS = [
    r'\bcompare\b.*\b(and|vs|versus|to)\b', r'\bcontrast\b',
    r'\bstep.by.step\b', r'\bmulti.?step\b', r'\bchain.*logic\b',
    r'\banalyze.*and.*then\b', r'\bfirst.*then.*finally\b',
]


def classify_prompt(prompt: str) -> ClassificationResult:
    """Fast keyword-based classification - no API call needed."""
    lower = prompt.lower()
    
    # Check for Tier 3 keywords
    for pattern in TIER3_KEYWORDS:
        if re.search(pattern, lower):
            return ClassificationResult(
                difficulty="hard",
                risk="high",
                recommended_tier=3,
                reason="Code or specialized query"
            )
    
    # Check for Tier 2 keywords
    for pattern in TIER2_KEYWORDS:
        if re.search(pattern, lower):
            return ClassificationResult(
                difficulty="medium",
                risk="low",
                recommended_tier=2,
                reason="Multi-step or comparison query"
            )
    
    # Default to Tier 1 for everything else
    return ClassificationResult(
        difficulty="easy",
        risk="low",
        recommended_tier=1,
        reason="General query"
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
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.7,
            },
            timeout=60.0,
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
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.7,
            },
            timeout=60.0,
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
    
    # Step 1: Classify the prompt (instant - keyword-based)
    classification = classify_prompt(prompt)
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


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "EcoLogic API is running"}

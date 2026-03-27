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
from typing import Set, List, Tuple

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

# GPT-5 baseline: ~8.6x more energy than GPT-4 per query (URI AI Lab, 2025)
GPT5_ENERGY_PER_1K = 500


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


# Programming language keywords for code detection
PROGRAMMING_LANGS = {
    'python', 'javascript', 'java', 'c++', 'cpp', 'typescript', 'rust', 'go', 
    'golang', 'ruby', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl',
    'haskell', 'elixir', 'dart', 'lua', 'sql', 'html', 'css', 'react', 'vue',
    'angular', 'node', 'django', 'flask', 'spring', 'bash', 'shell', 'powershell'
}

# Code action verbs (imperative)
CODE_ACTIONS = {'write', 'code', 'implement', 'create', 'build', 'develop', 'program', 
                'script', 'generate', 'make', 'design'}
CODE_DEBUG = {'debug', 'fix', 'error', 'bug', 'issue', 'solve', 'troubleshoot', 'repair'}
CODE_NOUNS = {'function', 'algorithm', 'api', 'class', 'method', 'module', 'library', 
              'framework', 'package', 'application', 'app', 'script', 'program'}

# High-risk domain keywords
HIGH_RISK_DOMAINS = {
    'medical', 'legal', 'health', 'disease', 'diagnosis', 'symptom', 'medicine',
    'drug', 'lawsuit', 'contract', 'sue', 'attorney', 'doctor', 'patient', 'treatment'
}
HIGH_RISK_ACTIONS = {'diagnose', 'treat', 'prescribe', 'advise', 'recommend', 'cure'}

# Comparison indicators
COMPARISON_WORDS = {'compare', 'contrast', 'versus', 'vs', 'difference', 'differ', 
                    'distinguish', 'similarities'}
ANALYSIS_WORDS = {'analyze', 'evaluate', 'assess', 'examine', 'investigate', 'review',
                  'critique', 'discuss'}

# Question starters (usually simple/Tier 1)
SIMPLE_STARTERS = {'what is', 'who is', 'where is', 'when is', 'when was', 'who was',
                   'what are', 'define', 'explain', 'describe', 'tell me about'}


def simple_tokenize(text: str) -> List[str]:
    """Simple tokenization - split on spaces and punctuation."""
    return re.findall(r'\b\w+\b', text.lower())


def classify_prompt_local_nlp(prompt: str) -> ClassificationResult:
    """
    Advanced local NLP classification - no API calls, instant (<5ms), zero external energy.
    Uses linguistic pattern analysis, intent detection, and contextual rules.
    """
    
    prompt_lower = prompt.lower()
    tokens = simple_tokenize(prompt)
    words_set = set(tokens)
    
    # Analyze sentence structure
    first_word = tokens[0] if tokens else ""
    first_three = ' '.join(tokens[:3])
    
    # === TIER 1 PRIORITY CHECK (Definitional/Informational) ===
    # Check this FIRST to avoid false positives with tech terms
    
    if any(prompt_lower.startswith(starter) for starter in SIMPLE_STARTERS):
        return ClassificationResult(
            difficulty="easy",
            risk="low",
            recommended_tier=1,
            reason="Factual question"
        )
    
    # === TIER 3 DETECTION (Code, Medical, Legal) ===
    
    # 1. Code Implementation Detection
    has_code_action = bool(CODE_ACTIONS & words_set)
    has_code_debug = bool(CODE_DEBUG & words_set)
    has_prog_lang = bool(PROGRAMMING_LANGS & words_set)
    has_code_noun = bool(CODE_NOUNS & words_set)
    
    # Strong code signals: action + language
    if has_code_action and has_prog_lang:
        return ClassificationResult(
            difficulty="hard",
            risk="high",
            recommended_tier=3,
            reason="Code implementation request"
        )
    
    # Debug requests
    if has_code_debug and (has_prog_lang or has_code_noun):
        return ClassificationResult(
            difficulty="hard",
            risk="high",
            recommended_tier=3,
            reason="Code debugging request"
        )
    
    # Technical code queries (function, algorithm, etc. + language)
    if has_code_noun and has_prog_lang:
        return ClassificationResult(
            difficulty="hard",
            risk="high",
            recommended_tier=3,
            reason="Technical implementation query"
        )
    
    # 2. High-Risk Domain Detection (Medical/Legal)
    has_risk_domain = bool(HIGH_RISK_DOMAINS & words_set)
    has_risk_action = bool(HIGH_RISK_ACTIONS & words_set)
    
    if has_risk_domain and has_risk_action:
        return ClassificationResult(
            difficulty="hard",
            risk="high",
            recommended_tier=3,
            reason="High-risk domain advice"
        )
    
    # === TIER 2 DETECTION (Comparisons, Analysis) ===
    
    # 1. Comparison Queries
    has_comparison = bool(COMPARISON_WORDS & words_set)
    
    # Check if it's a genuine comparison (not "what is the difference")
    if has_comparison:
        # Definitional questions about differences → Tier 1
        if any(starter in first_three for starter in ['what is', 'what are']):
            pass  # Will fall to Tier 1 below
        else:
            # Genuine comparison request → Tier 2
            return ClassificationResult(
                difficulty="medium",
                risk="low",
                recommended_tier=2,
                reason="Comparative analysis"
            )
    
    # 2. Analytical Queries
    has_analysis = bool(ANALYSIS_WORDS & words_set)
    
    if has_analysis:
        return ClassificationResult(
            difficulty="medium",
            risk="low",
            recommended_tier=2,
            reason="Analytical reasoning required"
        )
    
    # 3. Multi-step indicators
    multi_step_phrases = ['step by step', 'pros and cons', 'advantages and disadvantages',
                          'first then', 'both', 'each']
    if any(phrase in prompt_lower for phrase in multi_step_phrases):
        return ClassificationResult(
            difficulty="medium",
            risk="low",
            recommended_tier=2,
            reason="Multi-step reasoning"
        )
    
    # 4. Complexity heuristics
    # Long, complex questions likely need more reasoning
    word_count = len(tokens)
    question_marks = prompt.count('?')
    
    if word_count > 25 and question_marks >= 2:
        # Multiple complex questions
        return ClassificationResult(
            difficulty="medium",
            risk="low",
            recommended_tier=2,
            reason="Multi-part complex query"
        )
    
    # === TIER 1 DEFAULT (Simple factual queries) ===
    
    # Definitional questions (what/who/where/when/why)
    if any(prompt_lower.startswith(starter) for starter in SIMPLE_STARTERS):
        return ClassificationResult(
            difficulty="easy",
            risk="low",
            recommended_tier=1,
            reason="Factual question"
        )
    
    # General knowledge (short, simple)
    if word_count <= 10 and question_marks <= 1:
        return ClassificationResult(
            difficulty="easy",
            risk="low",
            recommended_tier=1,
            reason="Simple query"
        )
    
    # Default to Tier 1 (conservative approach)
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
    
    # Step 1: Classify the prompt using local NLP (instant, zero API calls)
    classification = classify_prompt_local_nlp(prompt)
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
    
    # Step 3: Calculate energy (compared to GPT-5 baseline)
    energy_per_1k = model_config["energy_per_1k_tokens"]
    energy_used = (tokens / 1000) * energy_per_1k
    energy_if_gpt5 = (tokens / 1000) * GPT5_ENERGY_PER_1K
    energy_saved = energy_if_gpt5 - energy_used
    
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
    
    yield f"data: {json.dumps({'type': 'meta', 'tier': tier, 'model': model})}\n\n"
    
    try:
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
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_content += content
                                yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                        except (json.JSONDecodeError, IndexError, KeyError):
                            continue
    except Exception:
        pass
    
    tokens = max(1, int(len(full_content.split()) * 1.3))
    energy_used = (tokens / 1000) * energy_per_1k
    energy_if_gpt5 = (tokens / 1000) * GPT5_ENERGY_PER_1K
    energy_saved = max(0, energy_if_gpt5 - energy_used)
    
    yield f"data: {json.dumps({'type': 'done', 'tokens': tokens, 'energy_used': round(energy_used, 2), 'energy_saved': round(energy_saved, 2)})}\n\n"


async def stream_openai(model: str, prompt: str, tier: int, energy_per_1k: float):
    """Stream response from OpenAI API."""
    full_content = ""
    
    yield f"data: {json.dumps({'type': 'meta', 'tier': tier, 'model': model})}\n\n"
    
    try:
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
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_content += content
                                yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                        except (json.JSONDecodeError, IndexError, KeyError):
                            continue
    except Exception:
        pass
    
    tokens = max(1, int(len(full_content.split()) * 1.3))
    energy_used = (tokens / 1000) * energy_per_1k
    energy_if_gpt5 = (tokens / 1000) * GPT5_ENERGY_PER_1K
    energy_saved = max(0, energy_if_gpt5 - energy_used)
    
    yield f"data: {json.dumps({'type': 'done', 'tokens': tokens, 'energy_used': round(energy_used, 2), 'energy_saved': round(energy_saved, 2)})}\n\n"


@app.post("/query/stream")
async def handle_query_stream(request: QueryRequest):
    """Streaming endpoint: returns Server-Sent Events with response chunks."""
    
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    # Classify the prompt using local NLP
    classification = classify_prompt_local_nlp(prompt)
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

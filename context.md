# EcoLogic — context.md

## 0) One-liner
EcoLogic is a minimalist, sustainable alternative to ChatGPT that reduces the environmental cost of AI answers by governing which model is used (low, medium, or high energy) based on the user’s question.

---

## 1) Product goal
Build a ChatGPT-style Q&A system where:
- Most queries are answered by **very low-energy open-source local models**
- Medium complexity escalates to a **stronger open-source model**
- Only genuinely hard / high-stakes queries escalate to **ChatGPT**
- Every response shows **energy saved vs using ChatGPT for this question**

Success = EcoLogic answers well while proving that high-energy AI is rarely necessary.

---

## 2) Core principles
1) **Minimalism-first UI**
   - One input box, one output area, no clutter
   - Optional “Details” toggle: model tier, energy used, energy saved
2) **Governance over performance**
   - EcoLogic decides whether high-energy AI is justified
3) **Escalate only when necessary**
4) **Transparency**
   - Users can see what energy was saved by not defaulting to ChatGPT

---

## 3) Fixed model tiers (NO ambiguity)

### Tier 1 — VERY LOW ENERGY (default)
**Model:** `Llama-3.2-3B-Instruct-Turbo` (Meta, open-source)  
Runs via Together AI

Why:
- Extremely small (3 billion parameters)
- Very fast
- Excellent reasoning for its size
- Ideal for classification, summaries, simple Q&A

Energy baseline:
- **~1 J per 1K tokens**

---

### Tier 2 — MEDIUM ENERGY (open-source)
**Model:** `Llama-3.1-8B-Instruct-Turbo` (Meta, open-source)  
Runs via Together AI

Why:
- Strong reasoning (8 billion parameters)
- Can handle structured and multi-step responses
- Still vastly cheaper than ChatGPT

Energy baseline:
- **~4 J per 1K tokens**

---

### Tier 3 — HIGH ENERGY (paid)
**Model:** GPT-4o (OpenAI)

Why:
- Best general reasoning
- Reserved for:
  - high-risk
  - high-complexity
  - or explicitly requested best-quality outputs

Energy baseline:
- **~60 J per 1K tokens**  
(≈10× Tier 2, ≈60–100× Tier 1)

---

## 4) Governance system (routing logic)

EcoLogic enforces **environmental restraint** through routing.

### Step A — Prompt classification (using Tier 1)
Return JSON:
```json
{
  "difficulty": "easy|medium|hard",
  "risk": "low|high",
  "recommended_tier": 1,
  "reason": "..."
}
```

### Step B — Routing policy
easy + low risk → Tier 1 (`phi-3-mini`)

medium → Tier 2 (`llama3.1:8b`)

hard OR high risk → Tier 3 (ChatGPT)

### Step C — Confidence check
If answer confidence < threshold → escalate one tier

---

## 5) Energy math (explicit + visible)

### Per-request energy estimate
Let:

T = tokens generated

E₁ = Tier 1 energy per 1K tokens ≈ 1 J

E₂ = Tier 2 energy per 1K tokens ≈ 6 J

E₃ = Tier 3 energy per 1K tokens ≈ 60 J

Energy used:

Tier 1: (T / 1000) × 1 J

Tier 2: (T / 1000) × 6 J

Tier 3: (T / 1000) × 60 J

### Energy saved vs ChatGPT
If EcoLogic used Tier X:

Energy saved =

```
((T / 1000) × 60) − ((T / 1000) × EX)
```

Where EX is:

1 J for Tier 1

6 J for Tier 2

60 J for Tier 3

### Example
If a 900-token answer is produced using Tier 1:

ChatGPT cost:
0.9 × 60 = 54 J

EcoLogic Tier 1 cost:
0.9 × 1 = 0.9 J

Energy saved:
53.1 J (~98.3% reduction)

This number is shown to the user in the UI.

---

## 6) UI requirements (minimalist)
Default view:

Prompt box (centered)

Answer area

Send button

Details toggle:

Model used (Tier 1 / 2 / 3)

Estimated energy used

Energy saved vs ChatGPT

“Escalated?” yes/no

No charts. No clutter. Just numbers.

---

## 7) Why these model choices matter
EcoLogic does NOT pretend all models are equal.

It explicitly proves:

Most questions do not require frontier models

Open-source models can handle the majority of use cases

Environmental savings are structural, not cosmetic

---

## 8) Tech stack
Local inference
Ollama

phi-3-mini

llama3.1:8b

High-energy tier
ChatGPT API only

Backend
Python + FastAPI

Frontend
Next.js or simple React

Single centered chat view

---

## 9) MVP completion criteria
EcoLogic is complete when:

Tier 1 answers most queries correctly

Tier 3 usage is rare and justified

Every answer displays energy saved vs ChatGPT

Logs prove average energy per query << ChatGPT baseline

---

### Why this is now powerful
You now have:
- **Fixed models**
- **Open-source priority**
- **Real math**
- **Visible environmental savings**
- A system that can be **measured, benchmarked, and defended**

This turns EcoLogic from:
> “green AI idea”  
into  
> **an enforceable, provable AI governance system**

Which is exactly what makes this elite.

# EcoLogic 🌱

> **A minimalist, sustainable alternative to ChatGPT that reduces the environmental cost of AI by routing queries to the smallest model capable of answering them.**

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://ecologic.bio)
[![API](https://img.shields.io/badge/API-Railway-purple)](https://ecologic-production.up.railway.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**🌐 Live Site:** [ecologic.bio](https://ecologic.bio)  
**🔌 API Endpoint:** [ecologic-production.up.railway.app](https://ecologic-production.up.railway.app)

EcoLogic is a ChatGPT-style Q&A system with a critical difference: it governs which model gets used based on query complexity. Most queries are answered by ultra-low-energy open-source models, only escalating to larger models when truly necessary. Every response displays the energy saved versus using ChatGPT by default.

**Mission:** Prove that high-energy AI is rarely necessary.

---

## Table of Contents

1. [How It Works: Model-Choosing Logic](#how-it-works-model-choosing-logic)
2. [Project Architecture](#project-architecture)
3. [Frontend Structure](#frontend-structure)
4. [Backend Structure](#backend-structure)
5. [File Structure](#file-structure)
6. [Energy Calculations](#energy-calculations)
7. [API Documentation](#api-documentation)
8. [Setup & Installation](#setup--installation)
9. [Deployment](#deployment)
10. [Design Philosophy](#design-philosophy)
11. [Tech Stack](#tech-stack)
12. [Performance & Benchmarks](#performance--benchmarks)
13. [Security & Privacy](#security--privacy)
14. [Roadmap](#roadmap)
15. [Contributing](#contributing)
16. [FAQ](#faq)

---

## How It Works: Model-Choosing Logic

EcoLogic uses a **3-tier model selection system** powered by fast, keyword-based classification to route queries to the most energy-efficient model capable of handling them.

### The Three Tiers

| Tier | Model | Provider | Energy Cost | Use Case |
|------|-------|----------|-------------|----------|
| **Tier 1** | Llama 3 8B Instruct Lite | Together AI (Meta) | 1 J/1k tokens | General questions, simple queries |
| **Tier 2** | Llama 3.3 70B Instruct Turbo | Together AI (Meta) | 4 J/1k tokens | Multi-step reasoning, comparisons |
| **Tier 3** | GPT-4o | OpenAI | 60 J/1k tokens | Code generation, debugging, specialized domains |

*Energy baseline: GPT-4o uses 60 Joules per 1,000 tokens*

### Classification Algorithm

The system performs **instant, zero-API-call classification** using regex keyword matching:

#### ⚡ Tier 3 Keywords (High Complexity)
Triggers: Code-related terms or specialized domains
- **Programming**: `code`, `function`, `debug`, `python`, `javascript`, `java`, `c++`, `typescript`, `rust`, `go`, `sql`, `html`, `css`, `api`, `script`, `program`, `algorithm`
- **Code Actions**: `write.*code`, `fix.*bug`
- **Specialized Domains**: `medical`, `legal`, `diagnos*`, `symptom*`, `lawsuit`, `contract`
- **Explicit Requests**: `gpt-4`, `best quality`

**Example**: *"Write a Python function to parse JSON"* → **Tier 3** (contains "python" and "function")

**Note:** Together AI's model offerings changed - the original Llama 3.2 3B and 3.1 8B "Turbo" models are now on-demand/dedicated only. The project now uses the "Lite" and serverless "Turbo" variants which maintain the same energy efficiency goals.

#### 🔶 Tier 2 Keywords (Medium Complexity)
Triggers: Multi-step reasoning or comparison queries
- **Comparisons**: `compare.*and`, `compare.*vs`, `compare.*versus`, `compare.*to`, `contrast`
- **Multi-step**: `step by step`, `multi-step`, `chain.*logic`, `analyze.*and.*then`, `first.*then.*finally`

**Example**: *"Compare React and Vue frameworks"* → **Tier 2** (contains "compare.*and")

#### ✅ Tier 1 (Default)
Everything else routes to Tier 1 for maximum energy efficiency
- General knowledge questions
- Simple factual queries
- Definitions and explanations

**Example**: *"What is the capital of France?"* → **Tier 1** (no matching keywords)

### Request Flow

```
User Query
    ↓
[Keyword Classification] ← Instant, no API call
    ↓
Tier Selected (1, 2, or 3)
    ↓
[Model Invocation] ← Single API call to selected tier
    ↓
[Energy Calculation]
    ├─ Energy used by selected model
    └─ Energy saved vs. GPT-4o baseline
    ↓
Response + Energy Stats
```

### Energy Calculation

For each response, the system calculates:

1. **Tokens Used**: Total tokens in the response (from API or estimated as `words × 1.3`)
2. **Energy Used**: `(tokens / 1000) × tier_energy_rate`
3. **Energy Saved**: `(tokens / 1000) × 60` - Energy Used
   - Compared against GPT-4o baseline (60 J/1k tokens)

**Example Calculation**:
- Query: "What is Python?" → Tier 1 (3B model)
- Response: 150 words ≈ 195 tokens
- Energy Used: `(195 / 1000) × 1 = 0.195 J`
- Energy if GPT-4o: `(195 / 1000) × 60 = 11.7 J`
- **Energy Saved: 11.5 J (98.3% reduction!)**

### API Endpoints

#### POST `/query`
Standard query with full response
```json
{
  "prompt": "What is machine learning?"
}
```

**Response**:
```json
{
  "response": "Machine learning is...",
  "tier": 1,
  "model": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
  "tokens": 245,
  "energy_used": 0.25,
  "energy_saved": 14.45,
  "escalated": false
}
```

#### POST `/query/stream`
Streaming response with Server-Sent Events (SSE)
```json
{
  "prompt": "Explain quantum computing"
}
```

**Event Stream**:
```
data: {"type": "meta", "tier": 1, "model": "meta-llama/Llama-3.2-3B-Instruct-Turbo"}

data: {"type": "content", "content": "Quantum"}
data: {"type": "content", "content": " computing"}
...
data: {"type": "done", "tokens": 312, "energy_used": 0.31, "energy_saved": 18.41}
```

## Setup & Installation

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.8+ | Backend runtime |
| pip | Latest | Package management |
| Together AI API Key | - | Tier 1 & 2 models ([Get one here](https://api.together.xyz/)) |
| OpenAI API Key | - | Tier 3 model (Optional, [Get one here](https://platform.openai.com/api-keys)) |

### Local Development Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/EcoLogic.git
cd EcoLogic
```

#### 2. Install Python Dependencies
```bash
cd backend
pip install -r requirements.txt
```

Or use a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 3. Configure Environment Variables
Copy the example environment file:
```bash
cp .env.example .env
```

Then edit `.env` and add your API keys:
```env
TOGETHER_API_KEY=your-together-ai-api-key-here
OPENAI_API_KEY=your-openai-api-key-here  # Optional
```

**Getting API Keys:**
- **Together AI**: Sign up at [api.together.xyz](https://api.together.xyz/), navigate to API Keys
- **OpenAI**: Sign up at [platform.openai.com](https://platform.openai.com/), create an API key

**Note:** OpenAI key is optional. If not provided, Tier 3 queries will fail gracefully with an error message.

#### 4. Run the Backend
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

**Flags explained:**
- `--reload`: Auto-restart on code changes (development only)
- `--host 0.0.0.0`: Accept connections from any IP
- `--port 8080`: Port number (default 8000)

Backend will be available at: `http://localhost:8080`

#### 5. Serve the Frontend

**Option A: Simple HTTP Server (Python)**
```bash
# From project root
python -m http.server 3000
```
Open `http://localhost:3000`

**Option B: Simple HTTP Server (Node.js)**
```bash
npx serve .
```

**Option C: Open Files Directly**
```bash
open index.html  # macOS
start index.html # Windows
xdg-open index.html # Linux
```

**Note:** For full functionality (API calls), you need to update `API_URL` in `app.js`:
```javascript
// Line 10 in app.js
const API_URL = "http://localhost:8080";  // Change from production URL
```

#### 6. Test the Setup

**Health Check:**
```bash
curl http://localhost:8080/health
```

Expected response:
```json
{"status":"ok","message":"EcoLogic API is running"}
```

**Test Query:**
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is photosynthesis?"}'
```

Expected response:
```json
{
  "response": "Photosynthesis is...",
  "tier": 1,
  "model": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
  "tokens": 150,
  "energy_used": 0.15,
  "energy_saved": 8.85,
  "escalated": false
}
```

### Troubleshooting

**Issue: `ModuleNotFoundError`**
```bash
# Solution: Reinstall dependencies
pip install -r backend/requirements.txt
```

**Issue: `TOGETHER_API_KEY not found`**
```bash
# Solution: Check .env file exists and has the key
cat .env
```

**Issue: CORS errors in browser**
```bash
# Solution: Ensure backend CORS middleware is configured (already done)
# Or use the same origin for frontend and backend
```

**Issue: Port already in use**
```bash
# Find process using port 8080
lsof -i :8080  # macOS/Linux
netstat -ano | findstr :8080  # Windows

# Kill the process or use a different port
uvicorn main:app --port 8081
```

---

## Deployment

EcoLogic is designed for separate frontend and backend deployment.

### Backend Deployment (Railway)

**Current Production:** [https://ecologic-production.up.railway.app](https://ecologic-production.up.railway.app)

#### Railway Configuration (`railway.json`)
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "cd backend && python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

#### Deploy to Railway

1. **Install Railway CLI:**
```bash
npm install -g @railway/cli
```

2. **Login:**
```bash
railway login
```

3. **Initialize Project:**
```bash
railway init
```

4. **Set Environment Variables:**
```bash
railway variables set TOGETHER_API_KEY=your-key-here
railway variables set OPENAI_API_KEY=your-key-here
```

5. **Deploy:**
```bash
railway up
```

6. **Get URL:**
```bash
railway domain
```

**Alternative Platforms:**
- **Render**: Create new Web Service, connect GitHub repo
- **Fly.io**: `fly launch` and `fly deploy`
- **Heroku**: `heroku create && git push heroku main`
- **AWS Lambda**: Use Mangum adapter for FastAPI

### Frontend Deployment (Vercel)

**Vercel is ideal for static sites with zero config.**

#### Vercel Configuration (`vercel.json`)
```json
{
  "buildCommand": "",
  "outputDirectory": ".",
  "rewrites": [
    { "source": "/search", "destination": "/search.html" }
  ]
}
```

**Features:**
- No build command (vanilla HTML/JS)
- Serves from root directory
- URL rewrite for clean `/search` URL

#### Deploy to Vercel

1. **Install Vercel CLI:**
```bash
npm install -g vercel
```

2. **Deploy:**
```bash
vercel
```

3. **Production Deploy:**
```bash
vercel --prod
```

4. **Update API URL:**
After deploying backend, update `app.js` line 10:
```javascript
const API_URL = "https://your-backend.railway.app";
```
Then redeploy frontend.

**Alternative Platforms:**
- **Netlify**: Drag & drop folder or connect GitHub
- **GitHub Pages**: Enable in repo settings
- **Cloudflare Pages**: Connect GitHub repo
- **AWS S3 + CloudFront**: Static website hosting

### Environment Variables in Production

**Backend Environment Variables (Railway/Render/etc):**
```bash
TOGETHER_API_KEY=your_key
OPENAI_API_KEY=your_key
PORT=8080  # Usually auto-set by platform
```

**Frontend Configuration:**
Update `API_URL` in `app.js` to point to deployed backend:
```javascript
const API_URL = "https://ecologic-production.up.railway.app";
```

### Monitoring & Logs

**Railway:**
```bash
railway logs
```

**Vercel:**
```bash
vercel logs
```

**Health Monitoring:**
Set up uptime monitoring with:
- [UptimeRobot](https://uptimerobot.com/)
- [Pingdom](https://www.pingdom.com/)
- [Better Uptime](https://betteruptime.com/)

Ping endpoint: `https://your-backend/health`

---

## Design Philosophy

EcoLogic is built on four core principles:

### 1. **Minimalism Over Features**

> "Perfection is achieved not when there is nothing more to add, but when there is nothing left to take away." — Antoine de Saint-Exupéry

**In Practice:**
- Single input box, single output area
- No sign-up, no account system, no friction
- No charts, graphs, or visual noise
- Energy stats are shown, not hidden in settings
- 3 model tiers (not 5, not 10)
- Vanilla JavaScript (no React/Vue overhead)
- 835 lines of CSS for entire site
- Zero build process

**Why It Matters:**
- Fewer features = less code = less energy to serve
- Simpler UX = faster decisions = less cognitive load
- Minimal JS bundle = faster page loads = less energy

### 2. **Governance Over Performance**

> "The best code is no code. The second best is the smallest model that works."

**In Practice:**
- EcoLogic *decides* which model to use (not the user)
- Default is always Tier 1 (3B model)
- Escalation requires specific keywords (code, medical, legal)
- No "max quality" toggle (prevents users from always choosing GPT-4)
- Transparent reasoning (shows tier + reason)

**Why It Matters:**
- Users default to "best quality" without governance
- ChatGPT always uses GPT-4, even for "What is 2+2?"
- Structural energy savings require policy enforcement
- Transparency builds trust in smaller models

### 3. **Transparency Over Opacity**

> "If you can measure it, you can improve it."

**In Practice:**
- Energy used displayed on every response
- Energy saved (vs GPT-4o) shown in green
- Model tier visible (1/2/3)
- Token count included in API response
- Escalation flag (`escalated: true/false`)
- No hidden AI calls (1 classification = 0 API calls)

**Why It Matters:**
- Users see the impact of their queries
- Developers can audit energy consumption
- Accountability drives better behavior
- Proves that smaller models work

### 4. **Open Source Over Proprietary**

> "Open source models are environmentally defensible."

**In Practice:**
- Tier 1: Meta Llama 3.2 3B (open weights)
- Tier 2: Meta Llama 3.1 8B (open weights)
- Tier 3: GPT-4o (closed, only for high-risk queries)
- Backend code is open source (MIT License)
- Energy calculations are auditable
- No vendor lock-in (swap models easily)

**Why It Matters:**
- Open models democratize AI
- Community can audit and improve
- Not dependent on OpenAI/Anthropic pricing
- Smaller models iterate faster

---

## Tech Stack

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.8+ | Backend language (async-first) |
| **FastAPI** | 0.109.0 | Modern async web framework |
| **Uvicorn** | 0.27.0 | ASGI server (production-ready) |
| **Pydantic** | 2.5.3 | Data validation & serialization |
| **httpx** | 0.26.0 | Async HTTP client (for API calls) |
| **python-dotenv** | 1.0.0 | Environment variable management |
| **OpenAI SDK** | 1.12.0 | OpenAI API client (Tier 3 only) |

**Why FastAPI?**
- Native async/await support (better performance)
- Automatic OpenAPI/Swagger docs (`/docs` endpoint)
- Type hints + Pydantic = runtime validation
- Built-in SSE support via `StreamingResponse`
- Modern Python (3.8+) with excellent DX

**Why Not Flask/Django?**
- Flask: Synchronous by default (slower for I/O)
- Django: Too heavy for a simple API
- FastAPI: Async-first, lightweight, perfect fit

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| **Vanilla JavaScript** | ES6+ | Client-side logic (no framework) |
| **HTML5** | - | Semantic markup |
| **CSS3** | - | Styling with custom properties |
| **marked.js** | 12.0+ (CDN) | Markdown rendering |
| **Google Fonts** | - | Typography (Source Serif 4, Cormorant Garamond, Inter) |

**Why Vanilla JS?**
- No build process (instant local dev)
- Minimal bundle size (~10KB total JS)
- Native browser APIs are powerful enough
- Easier to audit and modify
- Faster page loads = less energy

**Why Not React/Vue/Svelte?**
- Overkill for 2 simple pages
- Adds 40-100KB+ to bundle
- Requires build process (Webpack/Vite)
- More complexity = harder to maintain
- Goes against minimalism principle

### APIs & External Services

| Service | Purpose | Cost |
|---------|---------|------|
| **Together AI** | Llama 3.2 3B (Tier 1) | ~$0.06 / 1M tokens |
| **Together AI** | Llama 3.1 8B (Tier 2) | ~$0.20 / 1M tokens |
| **OpenAI** | GPT-4o (Tier 3) | ~$2.50 / 1M input tokens |

**Cost Comparison:**
- 1,000 queries (avg 200 tokens each) on Tier 1: **~$0.012**
- Same queries on GPT-4o: **~$0.50**
- **Savings: 97.6%**

### Infrastructure

| Component | Platform | Purpose |
|-----------|----------|---------|
| **Backend Hosting** | Railway | API server (Python + Uvicorn) |
| **Frontend Hosting** | Vercel | Static files (HTML/CSS/JS) |
| **DNS & CDN** | Cloudflare (optional) | Caching + DDoS protection |

**Why Railway?**
- Zero-config Python deployment
- Automatic HTTPS
- Environment variable management
- Built-in monitoring and logs
- Free tier available

**Why Vercel?**
- Instant static deployments
- Global CDN (fast worldwide)
- Automatic HTTPS
- Preview deployments for PRs
- Free tier (generous)

### Development Tools

| Tool | Purpose |
|------|---------|
| **Git** | Version control |
| **GitHub** | Code hosting + CI/CD |
| **VS Code** | Code editor |
| **Postman/curl** | API testing |
| **Chrome DevTools** | Frontend debugging |

### Monitoring & Analytics

**Current:** None (intentional minimalism)

**Potential Future:**
- [Sentry](https://sentry.io/) for error tracking
- [PostHog](https://posthog.com/) for privacy-friendly analytics
- [Plausible](https://plausible.io/) for lightweight analytics
- Custom logging (energy savings per query)

---

## Key Features

✅ **Zero-Classification Energy Cost**: Uses regex patterns instead of AI for classification  
✅ **Transparent Energy Tracking**: Shows exact energy used vs. saved on every query  
✅ **Automatic Routing**: No user intervention required  
✅ **Streaming Support**: Real-time responses with Server-Sent Events  
✅ **Fallback Support**: Gracefully handles missing API keys  
✅ **No Build Process**: Vanilla JS for instant local development  
✅ **Keyboard Shortcuts**: Enter to send, Escape to clear (planned)  
✅ **Markdown Rendering**: Full support for formatted responses  
✅ **Scroll Animations**: Smooth reveal animations on landing page  
✅ **Mobile Responsive**: Works on all screen sizes  
✅ **Accessibility**: Reduced motion support, semantic HTML  
✅ **Open Source**: MIT License, audit-friendly code  

## Example Queries

| Query | Selected Tier | Reason |
|-------|---------------|--------|
| *"What is photosynthesis?"* | Tier 1 | General knowledge |
| *"Compare Python and Ruby"* | Tier 2 | Comparison query |
| *"Write a sorting algorithm in JavaScript"* | Tier 3 | Code generation |
| *"Explain the water cycle"* | Tier 1 | Simple explanation |
| *"Debug this SQL query: SELECT * FROM..."* | Tier 3 | Contains "debug" and "sql" |
| *"First explain vectors, then show dot product"* | Tier 2 | Multi-step reasoning |

---

## Project Architecture

EcoLogic follows a clean client-server architecture with a static frontend and a FastAPI backend.

```
┌────────────────────────────────────────────────────────┐
│                    CLIENT LAYER                        │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Landing Page (index.html)                             │
│  ├─ Hero sections with scroll animations              │
│  ├─ Educational content about AI energy use           │
│  └─ CTA button → Search page                          │
│                                                        │
│  Chat Interface (search.html)                         │
│  ├─ Minimalist Q&A interface                          │
│  ├─ Real-time streaming responses                     │
│  └─ Energy statistics display                         │
│                                                        │
└─────────────────┬──────────────────────────────────────┘
                  │ HTTP/SSE
                  ↓
┌────────────────────────────────────────────────────────┐
│                   SERVER LAYER                         │
├────────────────────────────────────────────────────────┤
│                                                        │
│  FastAPI Backend (backend/main.py)                    │
│  ├─ Keyword Classifier                                │
│  │  └─ Instant regex-based tier selection             │
│  ├─ Query Router                                      │
│  │  ├─ Tier 1: Together AI (Llama 3.2 3B)            │
│  │  ├─ Tier 2: Together AI (Llama 3.1 8B)            │
│  │  └─ Tier 3: OpenAI (GPT-4o)                        │
│  └─ Energy Calculator                                 │
│     └─ Compute savings vs GPT-4o baseline             │
│                                                        │
└─────────────────┬──────────────────────────────────────┘
                  │ API Calls
                  ↓
┌────────────────────────────────────────────────────────┐
│              EXTERNAL SERVICES                         │
├────────────────────────────────────────────────────────┤
│  ┌──────────────────┐      ┌──────────────────┐      │
│  │  Together AI API │      │   OpenAI API     │      │
│  │  (Llama models)  │      │   (GPT-4o)       │      │
│  └──────────────────┘      └──────────────────┘      │
└────────────────────────────────────────────────────────┘
```

---

## Frontend Structure

### Pages

#### 1. **Landing Page** (`index.html`)
**Purpose:** Educational marketing page that explains the environmental cost of AI

**Components:**
- **Hero Section**: Dramatic full-screen headline with scroll reveal
- **Impact Slides**: Full-screen sections showcasing AI energy statistics
  - "One ChatGPT query = 12 minutes of LED power"
  - "10,000 homes could be powered by daily ChatGPT queries"
- **Intro Section**: EcoLogic branding with call-to-action
- **How It Works**: 3-step process explanation
  1. Classify (difficulty and risk assessment)
  2. Route (to appropriate tier)
  3. Reveal (energy stats)
- **Model Tiers**: Visual cards displaying the 3 tiers
- **CTA Banner**: Secondary call-to-action
- **Sources**: Links to Epoch AI and IEA research

**Animations:**
- Scroll-triggered reveals using Intersection Observer
- Staggered content animations with CSS transforms
- Smooth fade-in effects with cubic-bezier easing
- Mobile-optimized animation parameters

**Script:** `landing.js` (156 lines)
- Implements IntersectionObserver for scroll animations
- Mobile-responsive rootMargin adjustments
- Unobserves elements after animation completes
- Hero auto-triggers on page load

#### 2. **Chat Interface** (`search.html`)
**Purpose:** Minimalist Q&A interface for asking questions

**Components:**
- **Header**: Simple logo linking back to landing page
- **Chat Welcome**: Centered welcome message (hidden after first query)
- **Chat Response Area**: 
  - Markdown-rendered response text
  - Typing cursor animation during streaming
  - Energy statistics display:
    - Tier used (1/2/3)
    - Energy used (Joules)
    - Energy saved vs ChatGPT (Joules)
- **Input Footer**:
  - Rounded text input with send button
  - SVG arrow icon
  - Helper text explaining default tier

**Features:**
- Real-time streaming responses (Server-Sent Events)
- Markdown support via marked.js
- Loading states with disabled inputs
- Keyboard shortcuts (Enter to send)
- Error handling with user-friendly messages

**Script:** `app.js` (103 lines)
- Connects to `/query/stream` endpoint
- Parses SSE stream with TextDecoder
- Updates UI progressively as chunks arrive
- Formats energy values with 1 decimal place
- Renders markdown with `marked.parse()`

### Styling System (`styles.css`)

**Design Tokens:**
```css
--bg: #f3f8f5        /* Soft sage background */
--fg: #0b0b0b        /* Near-black text */
--muted: #4f5b56     /* Muted gray for secondary text */
--accent: #2a6b57    /* Forest green accent */
--border: #d7e1db    /* Subtle borders */
--panel: #ffffff     /* White panels */
--soft: #eef4f0      /* Soft input backgrounds */
```

**Typography:**
- **Primary Font**: Source Serif 4 (body, paragraphs)
- **Display Font**: Cormorant Garamond (headlines, italic)
- **Monospace Font**: Inter (energy statistics, numbers)

**Key Design Patterns:**
1. **Minimalism**: No unnecessary UI chrome or decoration
2. **Whitespace**: Generous padding and spacing
3. **Subtle Borders**: 1px solid borders for separation
4. **Soft Shadows**: Minimal use of box-shadow for depth
5. **Smooth Transitions**: 150-180ms ease transitions
6. **Responsive Typography**: clamp() for fluid scaling
7. **Scroll Animations**: CSS transforms with cubic-bezier
8. **Accessibility**: Reduced motion support via `@media (prefers-reduced-motion)`

**Animation System:**
- `.reveal` class: Translate-Y with opacity fade-in
- `.reveal-slide` class: Scale + translate for hero sections
- `.typing-cursor` class: Blinking cursor animation
- Staggered delays using `--delay` CSS custom property
- Mobile-optimized transforms (reduced distances)

---

## Backend Structure

### Core Files

#### `backend/main.py` (363 lines)
**FastAPI application implementing the entire backend logic**

### Key Components

#### 1. **Configuration** (Lines 24-46)

```python
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODELS = {
    "tier1": {
        "name": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        "provider": "together",
        "energy_per_1k_tokens": 1,
    },
    "tier2": {
        "name": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "provider": "together",
        "energy_per_1k_tokens": 4,
    },
    "tier3": {
        "name": "gpt-4o",
        "provider": "openai",
        "energy_per_1k_tokens": 60,
    },
}
```

**Why These Models:**
- **Tier 1 (8B Lite)**: Extremely efficient, INT4 quantized, handles 80%+ of queries, serverless
- **Tier 2 (70B Turbo)**: Better reasoning, FP8 quantized, still open-source and efficient, serverless
- **Tier 3 (GPT-4o)**: Reserved for code and specialized domains

#### 2. **Classification System** (Lines 70-117)

**Keyword Patterns:**
```python
TIER3_KEYWORDS = [
    r'\bcode\b', r'\bfunction\b', r'\bdebug\b', 
    r'\bpython\b', r'\bjavascript\b', r'\bjava\b',
    r'\bc\+\+\b', r'\btypescript\b', r'\brust\b',
    r'\bgo\b', r'\bsql\b', r'\bhtml\b', r'\bcss\b',
    r'\bapi\b', r'\bscript\b', r'\bprogram\b',
    r'\balgorithm\b', r'\bmedical\b', r'\blegal\b',
    r'\bdiagnos', r'\bsymptom', r'\blawsuit\b',
    r'\bcontract\b', r'\bgpt-4\b', r'\bbest quality\b',
    r'\bwrite.*code\b', r'\bfix.*bug\b',
]

TIER2_KEYWORDS = [
    r'\bcompare\b.*\b(and|vs|versus|to)\b',
    r'\bcontrast\b', r'\bstep.by.step\b',
    r'\bmulti.?step\b', r'\bchain.*logic\b',
    r'\banalyze.*and.*then\b',
    r'\bfirst.*then.*finally\b',
]
```

**`classify_prompt(prompt: str)` Function:**
- Converts prompt to lowercase
- Iterates through Tier 3 patterns first (priority)
- Falls back to Tier 2 patterns
- Defaults to Tier 1 for everything else
- Returns `ClassificationResult` with difficulty, risk, tier, reason
- **No API calls** = instant classification

**Why Keyword-Based?**
- **Speed**: 0ms classification (no network latency)
- **Cost**: $0 per classification
- **Energy**: No additional model inference
- **Transparency**: Rules are auditable and adjustable
- **Scalability**: Handles unlimited queries without rate limits

#### 3. **API Clients** (Lines 120-182)

**`query_together(model, prompt)`:**
- Async HTTP client using `httpx`
- POST to `https://api.together.xyz/v1/chat/completions`
- System message: "Be concise and direct. Keep responses under 200 words unless more detail is specifically requested."
- Max tokens: 512
- Temperature: 0.7
- Returns tuple: (response_text, token_count)
- Error handling with HTTPException

**`query_openai(model, prompt)`:**
- Similar structure to Together AI client
- POST to `https://api.openai.com/v1/chat/completions`
- Checks for valid API key
- Same parameters (max_tokens, temperature)
- Returns tuple: (response_text, token_count)

#### 4. **Endpoints**

**POST `/query`** (Lines 185-222)
Non-streaming endpoint that returns full response

**Request:**
```json
{
  "prompt": "What is machine learning?"
}
```

**Response:**
```json
{
  "response": "Machine learning is...",
  "tier": 1,
  "model": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
  "tokens": 245,
  "energy_used": 0.25,
  "energy_saved": 14.45,
  "escalated": false
}
```

**Flow:**
1. Validate prompt (non-empty)
2. Classify prompt → get tier
3. Route to appropriate model API
4. Calculate energy used and saved
5. Return structured response

**POST `/query/stream`** (Lines 327-357)
Streaming endpoint using Server-Sent Events (SSE)

**Request:**
```json
{
  "prompt": "Explain quantum computing"
}
```

**Event Stream:**
```
data: {"type": "meta", "tier": 1, "model": "meta-llama/Llama-3.2-3B-Instruct-Turbo"}

data: {"type": "content", "content": "Quantum"}
data: {"type": "content", "content": " computing"}
data: {"type": "content", "content": " is"}
...
data: {"type": "done", "tokens": 312, "energy_used": 0.31, "energy_saved": 18.41}
```

**Flow:**
1. Classify prompt → get tier
2. Select streaming function (Together or OpenAI)
3. Stream response chunks in real-time
4. Send metadata first (tier, model)
5. Stream content chunks progressively
6. Calculate and send final energy stats

**GET `/health`** (Lines 360-362)
Health check endpoint for monitoring

**Response:**
```json
{
  "status": "ok",
  "message": "EcoLogic API is running"
}
```

#### 5. **Streaming Implementation** (Lines 225-325)

**`stream_together(model, prompt, tier, energy_per_1k)`:**
- Opens persistent HTTP connection with `client.stream()`
- Enables `stream: True` in API request
- Iterates through response lines asynchronously
- Parses SSE format: lines starting with `"data: "`
- Accumulates full content for token counting
- Yields JSON events to client
- Calculates final energy stats after stream completes

**`stream_openai(model, prompt, tier, energy_per_1k)`:**
- Identical structure to `stream_together`
- Different API endpoint and headers
- Same SSE parsing and event yielding logic

**Why Streaming?**
- **Better UX**: Users see responses immediately
- **Perceived Speed**: Feels faster than waiting for full response
- **Engagement**: Visual feedback during generation
- **Modern Standard**: Matches ChatGPT UX expectations

### Middleware & Configuration

**CORS Middleware:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
- Allows requests from any origin (frontend can be hosted separately)
- Required for cross-origin API calls

**Environment Variables:**
- `TOGETHER_API_KEY`: Required for Tier 1 & 2
- `OPENAI_API_KEY`: Optional, only needed if Tier 3 queries occur
- Loaded via `python-dotenv` from `.env` file

---

## File Structure

```
EcoLogic/
├── backend/
│   ├── main.py                  # FastAPI backend (363 lines)
│   ├── requirements.txt         # Python dependencies
│   └── __pycache__/            # Compiled Python files (gitignored)
│
├── index.html                   # Landing page (145 lines)
├── search.html                  # Chat interface (65 lines)
├── styles.css                   # Global styles (835 lines)
├── landing.js                   # Landing page animations (47 lines)
├── app.js                       # Chat page logic (103 lines)
│
├── requirements.txt             # Python dependencies (duplicate)
├── vercel.json                  # Vercel frontend deployment config
├── railway.json                 # Railway backend deployment config
├── .env                         # Environment variables (gitignored)
├── .gitignore                   # Git ignore rules
├── context.md                   # Original project specification
└── README.md                    # This file
```

### Dependencies

**Backend (`backend/requirements.txt`):**
```
fastapi==0.109.0         # Modern async web framework
uvicorn==0.27.0          # ASGI server for FastAPI
python-dotenv==1.0.0     # Environment variable management
httpx==0.26.0            # Async HTTP client
openai==1.12.0           # OpenAI Python SDK
pydantic==2.5.3          # Data validation and serialization
```

**Frontend:**
- **No build step required** (vanilla JS)
- **marked.js** (via CDN): Markdown rendering
- Uses native browser APIs:
  - Fetch API for HTTP requests
  - ReadableStream for SSE parsing
  - Intersection Observer for scroll animations

**Why No Framework?**
- **Simplicity**: No build process, no dependencies
- **Performance**: Minimal JS bundle size
- **Sustainability**: Fewer bytes = less energy to transfer
- **Maintainability**: Easy to understand and modify

---

## Contributing

Contributions welcome! Here are areas where you can help:

### Improving Classification Logic
1. Edit keyword patterns in `backend/main.py` (lines 71-84)
2. Test with diverse query types
3. Submit PR with example queries and expected tiers

### Adding Features
- [ ] User accounts and query history
- [ ] Cumulative energy savings dashboard
- [ ] A/B testing: EcoLogic vs always-GPT-4
- [ ] Export energy reports
- [ ] Custom tier selection (override)
- [ ] Model performance metrics

### UI Enhancements
- [ ] Dark mode toggle
- [ ] Accessibility improvements
- [ ] Multi-language support
- [ ] Mobile app (React Native)

### Documentation
- [ ] API reference documentation
- [ ] Deployment guides for other platforms
- [ ] Video walkthrough
- [ ] Case studies with real usage data

## License

MIT License - see LICENSE file for details

---

**Built with restraint. Designed for impact.**
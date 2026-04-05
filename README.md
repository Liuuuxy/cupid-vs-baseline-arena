# CUPID vs Baseline Arena

A web-based user study platform comparing **CUPID** (Contextual UCB with Pairwise preferences, Indicators, and Dueling bandits) against an **LMArena-style baseline** for interactive model selection. Supports both **text** (LLM) and **image** (text-to-image) generation modes.

## Architecture

```
frontend/          React + Vite (TypeScript)
backend/           FastAPI + BoTorch (Python)
```

- **Frontend** serves the study UI and proxies API calls to the backend during development.
- **Backend** runs the CUPID algorithm (GP posterior, bUCB scoring, auxiliary LLM routing) and the baseline (Bradley-Terry), generates model responses via OpenRouter / Runware, and logs per-round timing data.

## Prerequisites

- Python 3.11+
- Node.js 18+
- An [OpenRouter](https://openrouter.ai/) API key (for LLM text generation)

## Setup

### 1. Clone

```bash
git clone https://github.com/<your-org>/cupid-vs-baseline-arena.git
cd cupid-vs-baseline-arena
```

### 2. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in `backend/`:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

Optional environment variables:

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | *(none -- saves to local JSON)* |
| `RUNWARE_API_KEY` | Runware API key for image mode | *(none)* |
| `MODEL_POOL_PATH` | Path to LLM model pool CSV | `./model-pool-llm.csv` |

### 3. Frontend

```bash
cd frontend
npm install
```

## Running locally

Open two terminals:

**Terminal 1 -- Backend** (from `backend/`):

```bash
source venv/bin/activate
uvicorn app:app --reload --port 8000
```

**Terminal 2 -- Frontend** (from `frontend/`):

```bash
npm run dev
```

The frontend starts at **http://localhost:3000** and proxies API requests to the backend at port 8000.

## Per-round timing instrumentation

Each `/interact` call measures five components (wall-clock seconds):

| Component | Key | Description |
|---|---|---|
| T1 | `gp_update` | GP posterior update (Laplace approx, Hessian, model fit) |
| T2 | `ucb_scoring` | bUCB scoring + arm selection (belief marginalization, top-2) |
| T3 | `aux_llm` | Auxiliary LLM call (Grok 4.1 Fast -- constraint routing) |
| T4 | `llm_duel` | LLM response generation (shared across methods) |
| T5 | `belief_update` | Bayesian belief + budget penalty update |

Derived fields: `total` = T1+T2+T3+T4+T5, `cupid_overhead` = T1+T2+T3+T5.

Timing data is:
- Embedded in each history entry (`timing` field)
- Saved per-session to `backend/session_data/timing_log_{session_id}.json`
- Retrievable via `GET /session/{session_id}/timing`

## Results

Study results are saved to `backend/session_data/results_{session_id}.json`. If a `DATABASE_URL` is configured, results are also persisted to PostgreSQL.

## Project structure

```
backend/
  app.py                  FastAPI app (CUPID + baseline logic, timing, API calls)
  model-pool-llm.csv      LLM model pool (25 models)
  image_model_pool.csv    Image model pool
  requirements.txt        Python dependencies
  session_data/           Output directory for results + timing logs
frontend/
  src/App.tsx             Main study UI
  vite.config.ts          Dev server config (proxy to backend)
```

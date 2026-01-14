# CUPID vs Baseline Arena

A dual-arena web application for comparing two LLM model selection algorithms:
- **CUPID (System A)**: Contextual Bandit with Gaussian Process and directional feedback learning
- **Baseline (System B)**: Bradley-Terry arena-style selection

## Architecture

```
Frontend (React + Vite + Tailwind)
         │
         ▼
    FastAPI Backend
         │
         ├── CUPID Driver (GP-based contextual bandit)
         │   └── Grok LLM for feedback analysis
         │
         └── Baseline Driver (Bradley-Terry)

         │
         ▼
    OpenRouter API
    (15 LLM models)
```

## Features

### Dual-Arena Comparison
- **Row 1 (CUPID)**: Uses contextual Gaussian Process with UCB selection
  - Analyzes user feedback text to extract directional preferences
  - Uses Grok LLM to understand what the user wants (cheaper, faster, better at code, etc.)
  - Updates direction bonus for model selection based on feedback

- **Row 2 (Baseline)**: Standard Bradley-Terry rating system
  - Arena-style pair selection based on uncertainty and ratings
  - No feedback learning - purely outcome-based

### Feedback Loop (The Brain)
When users provide text feedback on CUPID selections:
1. Feedback is sent to Grok LLM for analysis
2. Direction is extracted (e.g., "I want a cheaper model")
3. Router determines which models align with this direction
4. CUPID's GP model is updated with direction bonus
5. Next round selection incorporates learned preferences

### Cost Tracking
- Tracks input/output token costs for every LLM call
- Displays per-round and cumulative costs
- Shows cost breakdown on summary page

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- OpenRouter API key

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your OpenRouter API key
export OPENROUTER_API_KEY="sk-or-your-key-here"

# Run the server
uvicorn main:app --reload --port 8000
```

### Frontend Setup

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

The app will be available at `http://localhost:5173`

## API Endpoints

### POST /interact
Main interaction endpoint for running experiment rounds.

**Request:**
```json
{
  "prompt": "Explain recursion with a simple example",
  "session_id": "optional-session-id",
  "prev_round_info": {
    "cupid_winner": "left",
    "cupid_feedback": "More concise explanation",
    "baseline_winner": "right"
  }
}
```

**Response:**
```json
{
  "round": 1,
  "total_cost": 0.00234,
  "round_cost": 0.00234,
  "direction_extracted": "I want more concise responses",
  "cupid_pair": {
    "left": { "model_id": "1", "model_name": "GPT-4o", "text": "...", "cost": 0.0005 },
    "right": { "model_id": "3", "model_name": "Claude 3.5 Sonnet", "text": "...", "cost": 0.0006 }
  },
  "baseline_pair": {
    "left": { "model_id": "2", "model_name": "GPT-4o Mini", "text": "...", "cost": 0.0001 },
    "right": { "model_id": "4", "model_name": "Claude 3 Haiku", "text": "...", "cost": 0.0002 }
  }
}
```

### GET /summary
Get experiment summary with rankings and total cost.

### POST /reset
Reset session state to start fresh.

### GET /health
Health check endpoint.

## Model Pool

The system includes 15 LLM models with varying characteristics:

| Model | Input Price | Output Price | Speed | Intelligence |
|-------|-------------|--------------|-------|--------------|
| GPT-4o | $0.0025 | $0.01 | 4.5 | 5 |
| GPT-4o Mini | $0.00015 | $0.0006 | 5 | 4 |
| Claude 3.5 Sonnet | $0.003 | $0.015 | 4 | 5 |
| Claude 3 Haiku | $0.00025 | $0.00125 | 5 | 3.5 |
| Gemini 2.0 Flash | Free | Free | 4.5 | 4 |
| ... | ... | ... | ... | ... |

## Algorithm Details

### CUPID (Contextual UCB with Preference-Informed Direction)

```python
# Selection formula
UCB = mu + beta * sqrt(var) + B * direction_bonus

where:
- mu: estimated quality from pairwise comparisons
- var: uncertainty (reduces as more comparisons observed)
- beta: exploration-exploitation trade-off (default: 1.5)
- B: bonus scale based on UCB spread
- direction_bonus: 0/1 vector from router indicating direction alignment
```

### Baseline (Bradley-Terry)

```python
# BT rating update via MLE gradient ascent
theta[i] += lr * (wins[i,j] - N[i,j] * sigmoid(theta[i] - theta[j]))

# Pair selection uses softmax over ratings with uncertainty bonus
p_anchor = softmax(alpha_score * theta + alpha_unc / sqrt(comparisons))
```

## Project Structure

```
cupid-vs-baseline-arena/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── bandit_wrapper.py    # CUPID & Baseline algorithm implementations
│   ├── model_pool.csv       # LLM model metadata
│   └── requirements.txt     # Python dependencies
├── App.tsx                  # Main React component
├── types.ts                 # TypeScript type definitions
├── index.tsx                # React entry point
├── index.html               # HTML template with Tailwind
├── index.css                # Additional styles
├── package.json             # Node dependencies
└── README.md                # This file
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| OPENROUTER_API_KEY | Your OpenRouter API key | Yes |

## License

Research project - MIT License

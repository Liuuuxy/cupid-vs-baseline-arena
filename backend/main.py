"""
FastAPI Backend for CUPID vs Baseline Arena
============================================

This application provides a REST API for the dual-arena bandit experiment:
- System A (CUPID): Contextual bandit with GP and feedback-driven direction
- System B (Baseline): Bradley-Terry arena-style selection

Endpoints:
- POST /interact: Main interaction endpoint for running rounds
- GET /summary/{session_id}: Get experiment summary
- GET /health: Health check
"""

import os
import uuid
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from bandit_wrapper import bandit_manager


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class PrevRoundInfo(BaseModel):
    cupid_winner: Optional[str] = None  # 'left' or 'right'
    cupid_feedback: Optional[str] = None
    baseline_winner: Optional[str] = None  # 'left' or 'right'


class InteractRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None  # If not provided, uses default session
    prev_round_info: Optional[PrevRoundInfo] = None


class ModelResponse(BaseModel):
    model_id: str
    model_name: str
    text: str
    cost: float


class PairResponse(BaseModel):
    left: ModelResponse
    right: ModelResponse


class CupidPairResponse(PairResponse):
    context_info: Optional[dict] = None


class ArenaResponse(BaseModel):
    round: int
    total_cost: float
    round_cost: float
    direction_extracted: Optional[str] = None
    cupid_pair: dict
    baseline_pair: dict
    rankings: Optional[dict] = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize bandit manager on startup."""
    # Ensure model pool is loaded
    print("CUPID vs Baseline Arena - Backend Starting")
    print(f"OpenRouter API Key configured: {'Yes' if os.environ.get('OPENROUTER_API_KEY') else 'No (set OPENROUTER_API_KEY env var)'}")
    yield
    print("Backend shutting down")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CUPID vs Baseline Arena",
    description="Dual-arena bandit experiment comparing CUPID (contextual GP) with Bradley-Terry baseline",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "cupid-vs-baseline-arena",
        "api_key_set": bool(os.environ.get("OPENROUTER_API_KEY")),
    }


@app.post("/interact", response_model=ArenaResponse)
async def interact(request: InteractRequest):
    """
    Main interaction endpoint.

    This endpoint:
    1. Accepts the user's prompt and previous round votes/feedback
    2. Updates the bandit states based on previous votes
    3. Analyzes feedback text to extract direction (CUPID only)
    4. Selects new model pairs for both systems
    5. Calls all 4 models concurrently via OpenRouter
    6. Returns responses with cost tracking

    The feedback loop (brain):
    - User's text feedback is passed to Grok to extract a directional vector
    - This vector updates the CUPID GP's direction bonus before next selection
    """
    # Use default session if not provided
    session_id = request.session_id or "default"

    # Extract previous round info
    prev_cupid_vote = None
    prev_cupid_feedback = None
    prev_baseline_vote = None

    if request.prev_round_info:
        prev_cupid_vote = request.prev_round_info.cupid_winner
        prev_cupid_feedback = request.prev_round_info.cupid_feedback
        prev_baseline_vote = request.prev_round_info.baseline_winner

    try:
        result = await bandit_manager.process_round(
            session_id=session_id,
            prompt=request.prompt,
            prev_cupid_vote=prev_cupid_vote,
            prev_cupid_feedback=prev_cupid_feedback,
            prev_baseline_vote=prev_baseline_vote,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary/{session_id}")
async def get_summary(session_id: str):
    """
    Get experiment summary for a session.

    Returns:
    - Total rounds completed
    - Total cost incurred
    - Current rankings for both systems
    - Best model for each system
    """
    summary = bandit_manager.get_summary(session_id)
    if "error" in summary:
        raise HTTPException(status_code=404, detail=summary["error"])
    return summary


@app.get("/summary")
async def get_default_summary():
    """Get summary for the default session."""
    return await get_summary("default")


@app.post("/reset/{session_id}")
async def reset_session(session_id: str):
    """Reset a session to start fresh."""
    if session_id in bandit_manager._sessions:
        del bandit_manager._sessions[session_id]
    return {"status": "reset", "session_id": session_id}


@app.post("/reset")
async def reset_default_session():
    """Reset the default session."""
    return await reset_session("default")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

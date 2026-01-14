"""
Bandit Wrapper Module
=====================
This module provides a clean wrapper around the CUPID (Contextual Bandit with GP)
and Baseline (Bradley-Terry) algorithms for the web application.

Key Features:
- Persistent state management via singleton pattern (mocking Redis)
- CUPID algorithm with Gaussian Process and directional feedback
- Bradley-Terry baseline with arena-style pair selection
- Feedback analysis using Grok LLM for direction extraction
"""

import os
import math
import json
import re
import asyncio
import httpx
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Path to model pool CSV
MODEL_POOL_PATH = Path(__file__).parent / "model_pool.csv"


def _load_model_pool() -> pd.DataFrame:
    """Load model pool CSV."""
    if MODEL_POOL_PATH.exists():
        df = pd.read_csv(MODEL_POOL_PATH)
    else:
        raise FileNotFoundError(f"Model pool not found at {MODEL_POOL_PATH}")

    drop_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


# Load model pool at module level
df = _load_model_pool()
num_models = df.shape[0]


def model_name_from_id(model_id: int) -> str:
    """Get OpenRouter model identifier from our CSV 'id'."""
    row = df.loc[df["id"] == int(model_id)]
    if row.empty:
        raise ValueError(f"No model with id={model_id}")
    return str(row["model-id"].iloc[0])


def model_display_name(model_id: int) -> str:
    """Get human-readable model name from our CSV 'id'."""
    row = df.loc[df["id"] == int(model_id)]
    if row.empty:
        return f"Model {model_id}"
    return str(row["model"].iloc[0])


def get_model_prices(model_id: int) -> Tuple[float, float]:
    """Return (input_price, output_price) for the given model_id."""
    row = df.loc[df["id"] == int(model_id)]
    if row.empty:
        return 0.0, 0.0
    r = row.iloc[0]
    cin = float(r.get("input-price", 0) or 0)
    cout = float(r.get("output-price", 0) or 0)
    return cin, cout


# ---------------------------------------------------------------------------
# OpenRouter API Calls
# ---------------------------------------------------------------------------
async def call_openrouter(
    model_id: int,
    prompt: str,
    system_prompt: Optional[str] = None,
    timeout: float = 60.0
) -> Dict[str, Any]:
    """
    Call OpenRouter API asynchronously.
    Returns dict with 'text', 'cost', 'model_id', 'model_name'.
    """
    model_name = model_name_from_id(model_id)
    display_name = model_display_name(model_id)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_name,
        "messages": messages,
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(
                OPENROUTER_URL,
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

            # Extract text and cost
            text = ""
            if data.get("choices"):
                text = data["choices"][0].get("message", {}).get("content", "")

            usage = data.get("usage", {}) or {}
            cost = float(usage.get("cost", 0) or 0)

            # If cost not provided, estimate from token counts
            if cost == 0:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                in_price, out_price = get_model_prices(model_id)
                cost = (prompt_tokens * in_price / 1000) + (completion_tokens * out_price / 1000)

            return {
                "text": text,
                "cost": cost,
                "model_id": model_id,
                "model_name": display_name,
            }
        except Exception as e:
            return {
                "text": f"[Error calling {display_name}: {str(e)}]",
                "cost": 0.0,
                "model_id": model_id,
                "model_name": display_name,
            }


async def analyze_feedback_with_grok(
    feedback_text: str,
    last_model_csv: Optional[str] = None,
    constraints_text: Optional[str] = None,
) -> str:
    """
    Use Grok to analyze user feedback and extract a directional preference.
    Returns a direction string like "I want a cheaper model" or "I need faster responses".
    """
    if not feedback_text.strip():
        return ""

    model_section = last_model_csv if last_model_csv else "(no model selected yet)"

    system_content = (
        "You are analyzing user feedback about AI model selection.\n\n"
        "You will be given:\n"
        "  - The user's feedback text about why they made a choice\n"
        "  - Information about the last chosen model (if any)\n"
        "  - Any existing constraints\n\n"
        "Your job:\n"
        "1) Extract the user's preference direction from their feedback.\n"
        "2) Produce a short natural-language DIRECTION (one sentence) that captures what they want.\n"
        "   Examples: 'I want a cheaper model.', 'I need faster responses.',\n"
        "   'I want better code quality.', 'I prefer more concise answers.'.\n"
        "3) Focus on ONE main preference. Do not combine multiple items.\n"
        "4) Output ONLY the DIRECTION sentence, without quotes or explanations.\n"
        "5) If the feedback is unclear or too vague, output 'NONE'.\n"
    )

    user_content = (
        f"User feedback: {feedback_text}\n\n"
        f"Last chosen model info:\n{model_section}\n\n"
        f"Existing constraints: {constraints_text or '(none)'}\n\n"
        "What direction/preference does this feedback indicate?"
    )

    payload = {
        "model": "x-ai/grok-2-1212",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(OPENROUTER_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            direction = data["choices"][0]["message"]["content"].strip()
            if direction.upper() == "NONE" or not direction:
                return ""
            return direction
        except Exception:
            return ""


async def get_router_support_mask(
    prompt: str,
    direction_text: str,
    last_model_csv: Optional[str] = None,
) -> List[int]:
    """
    Use Grok to determine which models align with the user's direction.
    Returns a list of 0/1 for each model in the pool.
    """
    df_text = df.drop(columns=["model-id"], errors="ignore").to_csv(index=False)
    last_section = last_model_csv if last_model_csv else "(none yet)"

    full_prompt = prompt
    if direction_text:
        full_prompt = prompt.rstrip() + f"\n\nDIRECTION: {direction_text}"

    system_content = (
        "You are choosing which language models align with the user's preferences.\n\n"
        f"There are {num_models} models in the CSV below.\n\n"
        "Your task:\n"
        "1) For each model row, decide if it matches the user's DIRECTION.\n"
        f"2) Output exactly {num_models} integers (space-separated), one per row.\n"
        "   Use '1' if the model is acceptable, '0' if it is not.\n"
        "3) Output no other words.\n\n"
        f"Last chosen model:\n{last_section}\n\n"
        f"All models CSV:\n{df_text}\n"
    )

    payload = {
        "model": "x-ai/grok-2-1212",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": full_prompt},
        ],
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(OPENROUTER_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            raw = data["choices"][0]["message"]["content"].strip()
            bits = [int(x) for x in re.findall(r"\b[01]\b", raw)]

            # Pad or truncate to num_models
            if len(bits) < num_models:
                bits += [0] * (num_models - len(bits))
            return bits[:num_models]
        except Exception:
            return [1] * num_models  # Default: all models acceptable


# ---------------------------------------------------------------------------
# Bradley-Terry Model (Baseline)
# ---------------------------------------------------------------------------
def bt_grad(theta: np.ndarray, wins: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Gradient of log-likelihood for Bradley-Terry model."""
    K = theta.shape[0]
    grad = np.zeros_like(theta)
    for i in range(K):
        for j in range(K):
            Nij = N[i, j]
            if i == j or Nij == 0:
                continue
            diff = theta[i] - theta[j]
            p = 1.0 / (1.0 + np.exp(-np.clip(diff, -30, 30)))
            w_ij = wins[i, j]
            g = w_ij - Nij * p
            grad[i] += g
            grad[j] -= g
    grad -= grad.mean()  # Center for identifiability
    return grad


def fit_bt(
    theta_init: np.ndarray,
    wins: np.ndarray,
    N: np.ndarray,
    steps: int = 50,
    lr: float = 0.01
) -> np.ndarray:
    """Simple gradient ascent MLE for Bradley-Terry ratings."""
    theta = theta_init.copy()
    if wins.sum() == 0:
        theta -= theta.mean()
        return theta
    for _ in range(steps):
        g = bt_grad(theta, wins, N)
        theta += lr * g
        theta -= theta.mean()
    return theta


def choose_pair_bt(
    theta_hat: np.ndarray,
    wins: np.ndarray,
    N: np.ndarray,
    alpha_score: float = 1.0,
    alpha_unc: float = 1.0,
    tau: float = 1.0,
    eps_random: float = 0.05,
) -> Tuple[int, int]:
    """
    LMArena-style pair selection for Bradley-Terry baseline.
    """
    K = theta_hat.shape[0]
    counts_per_arm = N.sum(axis=1)
    s = 1.0 / np.sqrt(np.maximum(counts_per_arm, 1.0))

    centered_theta = theta_hat - theta_hat.mean()
    base = np.exp(alpha_score * centered_theta) * (1.0 + alpha_unc * s)
    base = np.maximum(base, 1e-8)
    p_anchor = base / base.sum()

    # Choose anchor
    if np.random.rand() < eps_random:
        i = int(np.random.randint(K))
    else:
        i = int(np.random.choice(K, p=p_anchor))

    # Choose opponent
    w_opp = base.copy()
    w_opp[i] = 0.0
    closeness = np.exp(-((theta_hat[i] - theta_hat) ** 2) / (tau ** 2))
    w_opp *= closeness
    w_opp[i] = 0.0

    if w_opp.sum() <= 0.0:
        choices = [j for j in range(K) if j != i]
        j = int(np.random.choice(choices))
    else:
        p_opp = w_opp / w_opp.sum()
        j = int(np.random.choice(K, p=p_opp))

    return i, j


# ---------------------------------------------------------------------------
# CUPID GP-based Contextual Bandit (Simplified for Web)
# ---------------------------------------------------------------------------
@dataclass
class CupidState:
    """State for the CUPID contextual bandit algorithm."""

    num_arms: int = num_models
    beta: float = 1.5
    rho: float = 1.3
    kappa: float = 1.6

    # GP surrogate: simplified as mean and variance estimates per arm
    mu: np.ndarray = field(default_factory=lambda: np.zeros(num_models))
    var: np.ndarray = field(default_factory=lambda: np.ones(num_models))

    # Win matrix for pairwise comparisons
    wins: np.ndarray = field(default_factory=lambda: np.zeros((num_models, num_models)))
    comparisons: np.ndarray = field(default_factory=lambda: np.zeros((num_models, num_models)))

    # Direction bonus from feedback
    direction_bonus: np.ndarray = field(default_factory=lambda: np.zeros(num_models))

    # Cooldown tracking
    recent_arms: deque = field(default_factory=lambda: deque(maxlen=5))

    # Current direction text
    current_direction: str = ""

    def select_pair(self) -> Tuple[int, int]:
        """Select a pair of arms using UCB with direction bonus."""
        # Calculate UCB scores
        ucb = self.mu + self.beta * np.sqrt(self.var)

        # Add direction bonus (scaled by spread)
        spread = np.std(ucb) if ucb.std() > 0 else 1.0
        bonus_scale = self.rho * spread
        bonus_scale = min(bonus_scale, self.kappa * spread)

        if self.direction_bonus.sum() > 0:
            ucb = ucb + bonus_scale * self.direction_bonus

        # Apply cooldown penalty
        cooldown = self._cooldown_vector()
        ucb = ucb - 0.1 * cooldown

        # Get top 2
        order = np.argsort(ucb)[::-1]
        i, j = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])

        return i, j

    def update(self, winner_idx: int, loser_idx: int):
        """Update state after observing a comparison result."""
        # Update win matrix
        self.wins[winner_idx, loser_idx] += 1
        self.comparisons[winner_idx, loser_idx] += 1
        self.comparisons[loser_idx, winner_idx] += 1

        # Update mean estimates (simplified Bayesian update)
        self.mu[winner_idx] += 0.1 * (1.0 - self.mu[winner_idx])
        self.mu[loser_idx] += 0.1 * (0.0 - self.mu[loser_idx])

        # Reduce variance for compared arms
        self.var[winner_idx] *= 0.95
        self.var[loser_idx] *= 0.95

        # Track recent arms for cooldown
        self.recent_arms.append(winner_idx)
        self.recent_arms.append(loser_idx)

    def update_direction_bonus(self, mask: List[int]):
        """Update direction bonus from router mask."""
        self.direction_bonus = np.array(mask, dtype=float)

    def _cooldown_vector(self) -> np.ndarray:
        """Get cooldown vector based on recent selections."""
        counts = Counter(self.recent_arms)
        vec = np.zeros(self.num_arms)
        for idx, cnt in counts.items():
            if 0 <= idx < self.num_arms:
                vec[idx] = float(cnt)
        return vec

    def get_ranking(self) -> List[int]:
        """Get current model ranking by estimated quality."""
        return list(np.argsort(self.mu)[::-1])


@dataclass
class BaselineState:
    """State for the Bradley-Terry baseline algorithm."""

    num_arms: int = num_models

    # BT ratings
    theta: np.ndarray = field(default_factory=lambda: np.zeros(num_models))

    # Win matrix
    wins: np.ndarray = field(default_factory=lambda: np.zeros((num_models, num_models)))
    N: np.ndarray = field(default_factory=lambda: np.zeros((num_models, num_models)))

    def select_pair(self) -> Tuple[int, int]:
        """Select a pair using arena-style sampling."""
        return choose_pair_bt(self.theta, self.wins, self.N)

    def update(self, winner_idx: int, loser_idx: int):
        """Update after observing a comparison result."""
        self.wins[winner_idx, loser_idx] += 1
        self.N[winner_idx, loser_idx] += 1
        self.N[loser_idx, winner_idx] += 1

        # Refit BT model
        self.theta = fit_bt(self.theta, self.wins, self.N)

    def get_ranking(self) -> List[int]:
        """Get current model ranking by BT rating."""
        return list(np.argsort(self.theta)[::-1])


# ---------------------------------------------------------------------------
# Bandit Manager - Singleton for State Persistence
# ---------------------------------------------------------------------------
@dataclass
class SessionState:
    """Complete state for a user session."""
    session_id: str
    round_number: int = 0
    total_cost: float = 0.0
    cupid: CupidState = field(default_factory=CupidState)
    baseline: BaselineState = field(default_factory=BaselineState)

    # Last selected models for context
    last_cupid_winner_id: Optional[int] = None
    last_baseline_winner_id: Optional[int] = None

    # Constraints (can be set by user)
    constraints: List[str] = field(default_factory=list)


class BanditManager:
    """
    Singleton manager for bandit state persistence.
    In production, replace with Redis or database storage.
    """

    _instance = None
    _sessions: Dict[str, SessionState] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_or_create_session(self, session_id: str) -> SessionState:
        """Get existing session or create new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get existing session if it exists."""
        return self._sessions.get(session_id)

    async def process_round(
        self,
        session_id: str,
        prompt: str,
        prev_cupid_vote: Optional[str] = None,  # 'left' or 'right'
        prev_cupid_feedback: Optional[str] = None,
        prev_baseline_vote: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a round of the dual-arena experiment.

        1. Update state based on previous votes (if any)
        2. Analyze feedback and update direction (CUPID only)
        3. Select new pairs for both systems
        4. Call OpenRouter for all 4 model responses
        5. Return results with cost tracking
        """
        session = self.get_or_create_session(session_id)
        round_cost = 0.0

        # ---------------------
        # Step 1: Update state from previous round votes
        # ---------------------
        if prev_cupid_vote and session.round_number > 0:
            # Determine winner/loser indices for CUPID
            cupid_last = getattr(session, '_last_cupid_pair', None)
            if cupid_last:
                left_idx, right_idx = cupid_last
                winner_idx = left_idx if prev_cupid_vote == 'left' else right_idx
                loser_idx = right_idx if prev_cupid_vote == 'left' else left_idx
                session.cupid.update(winner_idx, loser_idx)
                session.last_cupid_winner_id = winner_idx + 1  # 1-based

        if prev_baseline_vote and session.round_number > 0:
            baseline_last = getattr(session, '_last_baseline_pair', None)
            if baseline_last:
                left_idx, right_idx = baseline_last
                winner_idx = left_idx if prev_baseline_vote == 'left' else right_idx
                loser_idx = right_idx if prev_baseline_vote == 'left' else left_idx
                session.baseline.update(winner_idx, loser_idx)
                session.last_baseline_winner_id = winner_idx + 1

        # ---------------------
        # Step 2: Analyze feedback for CUPID direction
        # ---------------------
        direction_text = ""
        if prev_cupid_feedback and prev_cupid_feedback.strip():
            # Get last model info for context
            last_model_csv = ""
            if session.last_cupid_winner_id:
                row = df.loc[df["id"] == session.last_cupid_winner_id]
                if not row.empty:
                    last_model_csv = row.to_csv(index=False)

            # Use Grok to analyze feedback
            direction_text = await analyze_feedback_with_grok(
                feedback_text=prev_cupid_feedback,
                last_model_csv=last_model_csv,
                constraints_text=" & ".join(session.constraints) if session.constraints else None,
            )
            session.cupid.current_direction = direction_text

            # Get router mask based on direction
            if direction_text:
                mask = await get_router_support_mask(
                    prompt=prompt,
                    direction_text=direction_text,
                    last_model_csv=last_model_csv,
                )
                session.cupid.update_direction_bonus(mask)

        # ---------------------
        # Step 3: Select pairs
        # ---------------------
        cupid_i, cupid_j = session.cupid.select_pair()
        baseline_i, baseline_j = session.baseline.select_pair()

        # Store for next round's update
        session._last_cupid_pair = (cupid_i, cupid_j)
        session._last_baseline_pair = (baseline_i, baseline_j)

        # Convert indices to model IDs (1-based)
        cupid_left_id = cupid_i + 1
        cupid_right_id = cupid_j + 1
        baseline_left_id = baseline_i + 1
        baseline_right_id = baseline_j + 1

        # ---------------------
        # Step 4: Call all 4 models in parallel
        # ---------------------
        system_prompt = (
            "You may receive a prompt that contains a line starting with 'DIRECTION:'. "
            "That line describes internal routing preferences and is NOT part of the user's question. "
            "When answering, completely ignore any DIRECTION lines. Answer only the user's actual question."
        )

        # Build prompts (CUPID may include direction)
        cupid_prompt = prompt
        if direction_text:
            cupid_prompt = f"{prompt}\n\nDIRECTION: {direction_text}"

        # Fire all 4 API calls concurrently
        results = await asyncio.gather(
            call_openrouter(cupid_left_id, cupid_prompt, system_prompt),
            call_openrouter(cupid_right_id, cupid_prompt, system_prompt),
            call_openrouter(baseline_left_id, prompt, system_prompt),
            call_openrouter(baseline_right_id, prompt, system_prompt),
            return_exceptions=True,
        )

        # Process results
        cupid_left = results[0] if not isinstance(results[0], Exception) else {
            "text": f"Error: {results[0]}", "cost": 0, "model_id": cupid_left_id, "model_name": model_display_name(cupid_left_id)
        }
        cupid_right = results[1] if not isinstance(results[1], Exception) else {
            "text": f"Error: {results[1]}", "cost": 0, "model_id": cupid_right_id, "model_name": model_display_name(cupid_right_id)
        }
        baseline_left = results[2] if not isinstance(results[2], Exception) else {
            "text": f"Error: {results[2]}", "cost": 0, "model_id": baseline_left_id, "model_name": model_display_name(baseline_left_id)
        }
        baseline_right = results[3] if not isinstance(results[3], Exception) else {
            "text": f"Error: {results[3]}", "cost": 0, "model_id": baseline_right_id, "model_name": model_display_name(baseline_right_id)
        }

        # Sum costs
        round_cost = (
            cupid_left.get("cost", 0) +
            cupid_right.get("cost", 0) +
            baseline_left.get("cost", 0) +
            baseline_right.get("cost", 0)
        )

        # ---------------------
        # Step 5: Update session and return
        # ---------------------
        session.round_number += 1
        session.total_cost += round_cost

        return {
            "round": session.round_number,
            "total_cost": session.total_cost,
            "round_cost": round_cost,
            "direction_extracted": direction_text,
            "cupid_pair": {
                "left": {
                    "model_id": str(cupid_left_id),
                    "model_name": cupid_left.get("model_name", ""),
                    "text": cupid_left.get("text", ""),
                    "cost": cupid_left.get("cost", 0),
                },
                "right": {
                    "model_id": str(cupid_right_id),
                    "model_name": cupid_right.get("model_name", ""),
                    "text": cupid_right.get("text", ""),
                    "cost": cupid_right.get("cost", 0),
                },
                "context_info": {
                    "direction": direction_text,
                    "bonus_active": session.cupid.direction_bonus.sum() > 0,
                },
            },
            "baseline_pair": {
                "left": {
                    "model_id": str(baseline_left_id),
                    "model_name": baseline_left.get("model_name", ""),
                    "text": baseline_left.get("text", ""),
                    "cost": baseline_left.get("cost", 0),
                },
                "right": {
                    "model_id": str(baseline_right_id),
                    "model_name": baseline_right.get("model_name", ""),
                    "text": baseline_right.get("text", ""),
                    "cost": baseline_right.get("cost", 0),
                },
            },
            "rankings": {
                "cupid": session.cupid.get_ranking()[:5],  # Top 5
                "baseline": session.baseline.get_ranking()[:5],
            },
        }

    def get_summary(self, session_id: str) -> Dict[str, Any]:
        """Get experiment summary for a session."""
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}

        cupid_ranking = session.cupid.get_ranking()
        baseline_ranking = session.baseline.get_ranking()

        def ranking_to_models(ranking: List[int]) -> List[Dict]:
            return [
                {
                    "rank": i + 1,
                    "model_id": idx + 1,
                    "model_name": model_display_name(idx + 1),
                }
                for i, idx in enumerate(ranking[:10])
            ]

        return {
            "session_id": session_id,
            "total_rounds": session.round_number,
            "total_cost": session.total_cost,
            "cupid_ranking": ranking_to_models(cupid_ranking),
            "baseline_ranking": ranking_to_models(baseline_ranking),
            "cupid_best_model": model_display_name(cupid_ranking[0] + 1) if cupid_ranking else None,
            "baseline_best_model": model_display_name(baseline_ranking[0] + 1) if baseline_ranking else None,
        }


# Global singleton instance
bandit_manager = BanditManager()

"""
FastAPI Backend for LLM Model Selection User Study
Compares CUPID algorithm vs LMArena-style Baseline
"""

import os
import re
import uuid
import json
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter, deque
from datetime import datetime
import asyncio

import numpy as np
import pandas as pd
import torch
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Database imports
try:
    from databases import Database

    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    print(
        "Warning: databases package not available. Results will not be saved to database."
    )

# Conditional imports for CUPID GP model
try:
    from botorch.fit import fit_gpytorch_model
    from botorch.models.pairwise_gp import (
        PairwiseGP,
        PairwiseLaplaceMarginalLogLikelihood,
    )

    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False
    print("Warning: botorch not available. CUPID algorithm will be limited.")


# ================== Configuration ==================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}" if OPENROUTER_API_KEY else "",
    "Content-Type": "application/json",
}

# Model pool path - adjust as needed
MODEL_POOL_PATH = os.environ.get("MODEL_POOL_PATH", "./model-pool.csv")
MODEL_POOL_LOCAL = os.environ.get("MODEL_POOL_LOCAL", "./model-pool.csv")

# Database configuration
# Set DATABASE_URL in your environment (Render provides this automatically for PostgreSQL)
DATABASE_URL = os.environ.get("DATABASE_URL", "")
# Render uses 'postgres://' but asyncpg needs 'postgresql://'
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Initialize database connection
database = Database(DATABASE_URL) if DATABASE_URL and DATABASE_AVAILABLE else None


# ================== Load Model Pool ==================
def _load_model_pool() -> pd.DataFrame:
    """Load model pool CSV."""
    if os.path.exists(MODEL_POOL_PATH):
        df = pd.read_csv(MODEL_POOL_PATH)
    elif os.path.exists(MODEL_POOL_LOCAL):
        df = pd.read_csv(MODEL_POOL_LOCAL)
    else:
        # Create a minimal mock model pool for testing
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "model-id": [
                    "openai/gpt-4o-mini",
                    "anthropic/claude-3-haiku",
                    "google/gemini-flash-1.5",
                    "meta-llama/llama-3.1-8b-instruct",
                    "mistralai/mistral-7b-instruct",
                ],
                "model": [
                    "GPT-4o Mini",
                    "Claude 3 Haiku",
                    "Gemini Flash",
                    "Llama 3.1 8B",
                    "Mistral 7B",
                ],
                "intelligence": [85, 82, 80, 75, 72],
                "speed": [90, 88, 92, 85, 87],
                "input-price": [0.15, 0.25, 0.075, 0.05, 0.05],
                "output-price": [0.60, 1.25, 0.30, 0.05, 0.05],
            }
        )

    drop_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


MODEL_POOL = _load_model_pool()

if "id" not in MODEL_POOL.columns:
    raise RuntimeError("model-pool.csv must contain an 'id' column.")
if "model-id" not in MODEL_POOL.columns:
    raise RuntimeError("model-pool.csv must contain a 'model-id' column.")

num_models = MODEL_POOL.shape[0]
MODEL_IDS: List[int] = list(range(1, num_models + 1))


def _model_name_from_id(model_id: int) -> str:
    """Lookup OpenRouter model name from our CSV 'id'."""
    row = MODEL_POOL.loc[MODEL_POOL["id"] == int(model_id)]
    if row.empty:
        raise ValueError(f"No row in model pool with id={model_id}")
    return str(row["model-id"].iloc[0])


def _model_display_name_from_id(model_id: int) -> str:
    """Get display name for a model."""
    row = MODEL_POOL.loc[MODEL_POOL["id"] == int(model_id)]
    if row.empty:
        return f"Model {model_id}"
    if "model" in row.columns:
        return str(row["model"].iloc[0])
    return str(row["model-id"].iloc[0])


def _id_to_index(model_id: int) -> int:
    """Map 1-based model ID -> 0-based index."""
    idx = int(model_id) - 1
    if idx < 0 or idx >= num_models:
        raise ValueError(f"Model ID {model_id} is out of range [1, {num_models}]")
    return idx


def _get_model_pricing(model_id: int) -> Tuple[float, float]:
    """Get input and output price per 1M tokens for a model."""
    try:
        row = MODEL_POOL[MODEL_POOL["id"] == model_id].iloc[0]
        input_price = float(row.get("input-price", 0) or 0)
        output_price = float(row.get("output-price", 0) or 0)
        return input_price, output_price
    except (IndexError, KeyError):
        return 0.0, 0.0


# ================== OpenRouter API Calls ==================
def call_openrouter(prompt: str, model_id: int) -> Dict[str, Any]:
    """Generate response from a specific model via OpenRouter."""
    model_name = _model_name_from_id(model_id)

    # Mock response for testing without API key
    if not OPENROUTER_API_KEY:
        # Estimate mock cost based on model pricing
        input_price, output_price = _get_model_pricing(model_id)
        mock_prompt_tokens = int(len(prompt.split()) * 1.3) + 30
        mock_completion_tokens = 150
        mock_cost = (mock_prompt_tokens * input_price / 1_000_000) + (
            mock_completion_tokens * output_price / 1_000_000
        )
        return {
            "text": f"[Mock response from {_model_display_name_from_id(model_id)}] This is a simulated response for testing. Your query was about: {prompt[:100]}...",
            "cost": mock_cost,
            "prompt_tokens": mock_prompt_tokens,
            "completion_tokens": mock_completion_tokens,
        }

    system_content = "You are a helpful AI assistant. Answer the user's question concisely and accurately."

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
        "usage": {
            "include": True  # Enable usage accounting to get cost data
        },
    }

    try:
        response = requests.post(
            url=OPENROUTER_URL,
            headers=OPENROUTER_HEADERS,
            data=json.dumps(payload),
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        text = ""
        if isinstance(data.get("choices"), list) and data["choices"]:
            msg = data["choices"][0].get("message", {})
            text = msg.get("content", "")

        # Get usage data from response (enabled by usage.include: true)
        usage = data.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        cost = float(usage.get("cost", 0) or 0)

        return {
            "text": text,
            "cost": cost,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "raw": data,
        }
    except Exception as e:
        return {
            "text": f"[Error calling model: {str(e)}]",
            "cost": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": str(e),
        }


# ================== CUPID Algorithm Components ==================
if BOTORCH_AVAILABLE:
    torch.set_default_dtype(torch.double)

    def encode_arm(model_id: int) -> torch.Tensor:
        e = torch.zeros(num_models, dtype=torch.double)
        e[_id_to_index(model_id)] = 1.0
        return e

    def encode_context(ctx_id: int) -> torch.Tensor:
        e = torch.zeros(num_models, dtype=torch.double)
        e[_id_to_index(ctx_id)] = 1.0
        return e

    def build_point(arm_id: int, ctx_id: int) -> torch.Tensor:
        return torch.cat([encode_arm(int(arm_id)), encode_context(int(ctx_id))], dim=0)

    def block_features(arms: List[int], ctx_id: int) -> torch.Tensor:
        return torch.stack([build_point(a, ctx_id) for a in arms], dim=0)

    @torch.no_grad()
    def _safe_init_model(train_X: torch.Tensor) -> PairwiseGP:
        try:
            return PairwiseGP(train_X, None)
        except Exception:
            dummy = torch.tensor([[0, 1]], dtype=torch.long, device=train_X.device)
            return PairwiseGP(train_X, dummy)

    def fit_model(
        train_X: torch.Tensor, comps_wl, min_fit_pairs: int = 5
    ) -> PairwiseGP:
        if comps_wl is None or (
            isinstance(comps_wl, torch.Tensor) and comps_wl.numel() == 0
        ):
            return _safe_init_model(train_X)
        model = PairwiseGP(train_X, comps_wl)
        if comps_wl.shape[0] >= min_fit_pairs:
            mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)
        return model

    @torch.no_grad()
    def posterior_stats(model: PairwiseGP, Xtest: torch.Tensor):
        post = model.posterior(Xtest)
        mvn = getattr(post, "mvn", None)
        mean = post.mean.squeeze(-1) if mvn is None else mvn.mean
        cov = post.covariance_matrix if mvn is None else mvn.covariance_matrix
        return mean, cov

    def pairwin_prob_from_mu_cov(mu, cov, i: int, j: int, eps: float = 1e-9) -> float:
        var = (cov[i, i] + cov[j, j] - 2.0 * cov[i, j]).clamp_min(eps)
        z = (mu[i] - mu[j]) / var.sqrt()
        normal = torch.distributions.Normal(0.0, 1.0)
        return float(normal.cdf(z).clamp(1e-6, 1 - 1e-6))


class DiscreteBelief:
    """Belief distribution over contexts for CUPID."""

    def __init__(self, n: int):
        self.logp = (
            torch.zeros(n, dtype=torch.double) if BOTORCH_AVAILABLE else np.zeros(n)
        )

    def probs(self):
        if BOTORCH_AVAILABLE:
            return torch.softmax(self.logp, dim=0)
        else:
            exp_logp = np.exp(self.logp - np.max(self.logp))
            return exp_logp / exp_logp.sum()

    def entropy(self):
        p = self.probs()
        if BOTORCH_AVAILABLE:
            p = p.clamp_min(1e-12)
            return float(-(p * p.log()).sum())
        else:
            p = np.clip(p, 1e-12, None)
            return float(-(p * np.log(p)).sum())

    def sample(self):
        p = self.probs()
        if BOTORCH_AVAILABLE:
            return int(torch.multinomial(p, 1).item())
        else:
            return int(np.random.choice(len(p), p=p))

    def map(self):
        p = self.probs()
        if BOTORCH_AVAILABLE:
            return int(torch.argmax(p).item())
        else:
            return int(np.argmax(p))

    def bayes_update(self, log_lik):
        if BOTORCH_AVAILABLE:
            self.logp = self.logp + log_lik
        else:
            self.logp = self.logp + np.array(log_lik)


def _calc_spread(vec, method: str = "iqr") -> float:
    """Calculate spread of scores."""
    if BOTORCH_AVAILABLE:
        if vec.numel() < 2:
            return 0.0
        if method == "stdev":
            return float(max(torch.std(vec, unbiased=False).item(), 0.0))
        q = torch.quantile(vec, torch.tensor([0.25, 0.75], dtype=vec.dtype))
        return float((q[1] - q[0]).clamp_min(0.0).item())
    else:
        if len(vec) < 2:
            return 0.0
        if method == "stdev":
            return float(max(np.std(vec), 0.0))
        q25, q75 = np.percentile(vec, [25, 75])
        return float(max(q75 - q25, 0.0))


def _base_ucb(
    mu,
    cov,
    beta: float,
    cooldown_vec=None,
    cooldown_w: float = 0.0,
    cost_vec=None,
    penalty_p: float = 0.0,
):
    """Compute base UCB scores."""
    if BOTORCH_AVAILABLE:
        var = cov.diag().clamp_min(0.0)
        base = mu + beta * var.sqrt()
        if cooldown_vec is not None and cooldown_w > 0.0:
            base = base - cooldown_w * cooldown_vec.to(base)
        if cost_vec is not None and penalty_p > 0.0:
            base = base - penalty_p * cost_vec.to(base)
        return base
    else:
        var = np.maximum(np.diag(cov), 0.0)
        base = mu + beta * np.sqrt(var)
        if cooldown_vec is not None and cooldown_w > 0.0:
            base = base - cooldown_w * np.array(cooldown_vec)
        if cost_vec is not None and penalty_p > 0.0:
            base = base - penalty_p * np.array(cost_vec)
        return base


def _calc_bonus_B(base_scores, rho: float, kappa: float, method: str) -> float:
    """Calculate router bonus magnitude."""
    S = _calc_spread(base_scores, method)
    if S <= 0.0 or rho <= 0.0:
        return 0.01
    B = rho * S
    Bmax = kappa * S if kappa > 0 else B
    return float(max(0.01, min(B, Bmax)))


# ================== Baseline (Bradley-Terry) Components ==================
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
            p = 1.0 / (1.0 + np.exp(-diff))
            w_ij = wins[i, j]
            g = w_ij - Nij * p
            grad[i] += g
            grad[j] -= g
    grad -= grad.mean()
    return grad


def fit_bt(
    theta_init: np.ndarray,
    wins: np.ndarray,
    N: np.ndarray,
    steps: int = 50,
    lr: float = 0.01,
) -> np.ndarray:
    """Fit Bradley-Terry model via gradient ascent."""
    theta = theta_init.copy()
    if wins.sum() == 0:
        theta -= theta.mean()
        return theta
    for _ in range(steps):
        g = bt_grad(theta, wins, N)
        theta += lr * g
        theta -= theta.mean()
    return theta


def choose_pair_baseline(
    theta_hat: np.ndarray,
    wins: np.ndarray,
    N: np.ndarray,
    alpha_score: float = 1.0,
    alpha_unc: float = 1.0,
    tau: float = 1.0,
    eps_random: float = 0.05,
) -> Tuple[int, int]:
    """LMArena-style pair selection for baseline."""
    K = theta_hat.shape[0]
    counts_per_arm = N.sum(axis=1)
    s = 1.0 / np.sqrt(np.maximum(counts_per_arm, 1.0))

    centered_theta = theta_hat - theta_hat.mean()
    base = np.exp(alpha_score * centered_theta) * (1.0 + alpha_unc * s)
    base = np.maximum(base, 1e-8)
    p_anchor = base / base.sum()

    if np.random.rand() < eps_random:
        i = int(np.random.randint(K))
    else:
        i = int(np.random.choice(K, p=p_anchor))

    w_opp = base.copy()
    w_opp[i] = 0.0
    closeness = np.exp(-((theta_hat[i] - theta_hat) ** 2) / (tau**2))
    w_opp *= closeness
    w_opp[i] = 0.0

    if w_opp.sum() <= 0.0:
        choices = [j for j in range(K) if j != i]
        j = int(np.random.choice(choices))
    else:
        p_opp = w_opp / w_opp.sum()
        j = int(np.random.choice(K, p=p_opp))

    return i, j


# ================== Session State Management ==================
class CUPIDState:
    """State for CUPID algorithm."""

    def __init__(self, arms: List[int], contexts: List[int]):
        self.arms = arms
        self.contexts = contexts
        self.K = len(arms)
        self.M = len(contexts)

        # GP components
        if BOTORCH_AVAILABLE:
            prior_anchor_ctx = contexts[0]
            self.train_X = block_features(arms, prior_anchor_ctx)
            self.index_of: Dict[Tuple[int, int], int] = {
                (a_idx, 0): a_idx for a_idx in range(self.K)
            }
            self.comps_wl = None
            self.model = fit_model(self.train_X, self.comps_wl)

        self.belief = DiscreteBelief(self.M)
        self.recent_arms: deque = deque(maxlen=5)

        # Current pair being shown
        self.current_left_idx: Optional[int] = None
        self.current_right_idx: Optional[int] = None
        self.current_ctx_idx: Optional[int] = None

        # Parameters
        self.beta = 1.5
        self.rho = 1.30
        self.kappa = 1.60
        self.spread_method = "iqr"
        self.min_fit_pairs = 5

        # Cost tracking
        self.total_cost = 0.0
        self.round_count = 0

        # History for tracking
        self.history: List[Dict] = []

    @property
    def current_left_id(self) -> Optional[int]:
        """Get the actual model ID for current left arm."""
        if self.current_left_idx is not None and 0 <= self.current_left_idx < len(
            self.arms
        ):
            return self.arms[self.current_left_idx]
        return None

    @property
    def current_right_id(self) -> Optional[int]:
        """Get the actual model ID for current right arm."""
        if self.current_right_idx is not None and 0 <= self.current_right_idx < len(
            self.arms
        ):
            return self.arms[self.current_right_idx]
        return None

    def select_pair(self, direction_text: str = "") -> Tuple[int, int]:
        """Select next pair using expected UCB."""
        if not BOTORCH_AVAILABLE:
            # Fallback to random selection
            import random

            indices = list(range(self.K))
            i = random.choice(indices)
            j = random.choice([x for x in indices if x != i])
            self.current_left_idx = i
            self.current_right_idx = j
            self.current_ctx_idx = 0
            return self.arms[i], self.arms[j]

        # Build cost vector
        arm_cost_list = []
        for mid in self.arms:
            row = MODEL_POOL.loc[MODEL_POOL["id"] == int(mid)]
            if row.empty:
                arm_cost_list.append(0.0)
                continue
            r = row.iloc[0]
            cin = float(r.get("input-price", 0) or 0)
            cout = float(r.get("output-price", 0) or 0)
            arm_cost_list.append(cin + cout)
        cost_vec = torch.tensor(arm_cost_list, dtype=torch.double)

        # Cooldown vector
        cooldown_vec = torch.zeros(self.K, dtype=torch.double)
        c = Counter(self.recent_arms)
        for idx, cnt in c.items():
            if 0 <= idx < self.K:
                cooldown_vec[idx] = float(cnt)

        # Expected UCB selection
        p = self.belief.probs()
        expected_base = torch.zeros(self.K, dtype=torch.double)

        for cidx, ctx in enumerate(self.contexts):
            mu, cov = posterior_stats(self.model, block_features(self.arms, ctx))
            base_ctx = _base_ucb(mu, cov, self.beta, cooldown_vec, 0.0, cost_vec, 0.0)
            expected_base += p[cidx] * base_ctx

        B = _calc_bonus_B(
            expected_base, rho=self.rho, kappa=self.kappa, method=self.spread_method
        )
        ucb = expected_base  # Can add router bonus here if needed

        k = min(2, ucb.numel())
        top = torch.topk(ucb, k=k).indices.tolist()

        i = top[0]
        j = top[1] if len(top) > 1 else top[0]

        self.current_left_idx = i
        self.current_right_idx = j
        self.current_ctx_idx = self.belief.map()

        return self.arms[i], self.arms[j]

    def update_with_vote(self, winner_is_left: bool):
        """Update GP model with user's vote."""
        if self.current_left_idx is None or self.current_right_idx is None:
            return

        i = self.current_left_idx
        j = self.current_right_idx
        winner_arm = i if winner_is_left else j
        loser_arm = j if winner_is_left else i

        if not BOTORCH_AVAILABLE:
            self.recent_arms.append(winner_arm)
            self.recent_arms.append(loser_arm)
            self.round_count += 1
            return

        self.history.append(
            {
                "left": self.current_left_id,
                "right": self.current_right_id,
                "winner_id": self.current_left_id
                if winner_is_left
                else self.current_right_id,
            }
        )

        # Belief update
        log_liks = []
        for ctx in self.contexts:
            mu_ctx, cov_ctx = posterior_stats(
                self.model, block_features(self.arms, ctx)
            )
            p_i_wins = pairwin_prob_from_mu_cov(mu_ctx, cov_ctx, i, j)
            like = p_i_wins if winner_arm == i else (1.0 - p_i_wins)
            log_liks.append(
                float(
                    torch.log(torch.tensor(like, dtype=torch.double).clamp_min(1e-12))
                )
            )
        self.belief.bayes_update(torch.tensor(log_liks, dtype=torch.double))

        # Add training pair
        ctx_for_training = self.belief.map()
        self.train_X, self.index_of, _ = self._ensure_context_block_rows(
            ctx_for_training
        )

        w_row = self.index_of[(winner_arm, ctx_for_training)]
        l_row = self.index_of[(loser_arm, ctx_for_training)]
        pair = torch.tensor([[w_row, l_row]], dtype=torch.long)

        if self.comps_wl is None or (
            isinstance(self.comps_wl, torch.Tensor) and self.comps_wl.numel() == 0
        ):
            self.comps_wl = pair
        else:
            self.comps_wl = torch.cat([self.comps_wl, pair], dim=0)

        # Refit GP
        self.recent_arms.append(winner_arm)
        self.recent_arms.append(loser_arm)
        self.model = fit_model(
            self.train_X, self.comps_wl, min_fit_pairs=self.min_fit_pairs
        )
        self.round_count += 1

    def _ensure_context_block_rows(self, ctx_idx: int):
        """Ensure all arms have rows for the given context."""
        block_row_ids = []
        for a_idx in range(self.K):
            key = (a_idx, ctx_idx)
            if key not in self.index_of:
                phi = build_point(self.arms[a_idx], self.contexts[ctx_idx]).view(1, -1)
                row_id = self.train_X.shape[0]
                self.train_X = torch.cat([self.train_X, phi], dim=0)
                self.index_of[key] = row_id
                block_row_ids.append(row_id)
            else:
                block_row_ids.append(self.index_of[key])
        return self.train_X, self.index_of, block_row_ids


class BaselineState:
    """State for Bradley-Terry baseline algorithm."""

    def __init__(self, arms: List[int]):
        self.arms = arms
        self.K = len(arms)

        # BT model components
        self.wins = np.zeros((self.K, self.K), dtype=np.int64)
        self.N_mat = np.zeros((self.K, self.K), dtype=np.int64)
        self.theta_hat = np.zeros(self.K, dtype=float)

        # Current pair being shown
        self.current_left_idx: Optional[int] = None
        self.current_right_idx: Optional[int] = None

        # Cost tracking
        self.total_cost = 0.0
        self.round_count = 0

        # History for tracking
        self.history: List[Dict] = []

    @property
    def current_left_id(self) -> Optional[int]:
        """Get the actual model ID for current left arm."""
        if self.current_left_idx is not None and 0 <= self.current_left_idx < len(
            self.arms
        ):
            return self.arms[self.current_left_idx]
        return None

    @property
    def current_right_id(self) -> Optional[int]:
        """Get the actual model ID for current right arm."""
        if self.current_right_idx is not None and 0 <= self.current_right_idx < len(
            self.arms
        ):
            return self.arms[self.current_right_idx]
        return None

    def select_pair(self) -> Tuple[int, int]:
        """Select next pair using LMArena-style sampling."""
        i, j = choose_pair_baseline(self.theta_hat, self.wins, self.N_mat)
        self.current_left_idx = i
        self.current_right_idx = j
        return self.arms[i], self.arms[j]

    def update_with_vote(self, winner_is_left: bool):
        """Update BT model with user's vote."""
        if self.current_left_idx is None or self.current_right_idx is None:
            return

        i = self.current_left_idx
        j = self.current_right_idx
        winner_idx = i if winner_is_left else j
        loser_idx = j if winner_is_left else i

        self.wins[winner_idx, loser_idx] += 1
        self.N_mat[winner_idx, loser_idx] += 1
        self.N_mat[loser_idx, winner_idx] += 1

        # Refit BT model periodically
        self.round_count += 1
        if self.round_count % 5 == 0 or self.round_count == 1:
            self.theta_hat = fit_bt(
                self.theta_hat, self.wins, self.N_mat, steps=50, lr=0.01
            )


class ModelStats(BaseModel):
    id: int
    intelligence: Optional[int] = None
    speed: Optional[int] = None
    reasoning: Optional[int] = None
    input_price: Optional[float] = None
    output_price: Optional[float] = None
    context_window: Optional[int] = None
    max_output: Optional[int] = None
    text_input: Optional[bool] = None
    image_input: Optional[bool] = None
    voice_input: Optional[bool] = None
    function_calling: Optional[bool] = None
    structured_output: Optional[bool] = None
    knowledge_cutoff: Optional[str] = None


class SessionState:
    """Combined state for a user study session."""

    def __init__(self):
        self.lock = asyncio.Lock()
        arms = MODEL_IDS.copy()
        contexts = MODEL_IDS.copy()

        self.cupid = CUPIDState(arms, contexts)
        self.baseline = BaselineState(arms)
        self.round_count = 0

        # Routing cost for language feedback bias calculation (CUPID only)
        self.routing_cost: float = 0.0

        # Track final converged models
        self.final_cupid_model_id: Optional[int] = None
        self.final_baseline_model_id: Optional[int] = None

        # User study metadata
        self.budget_cost: Optional[float] = None
        self.budget_rounds: Optional[int] = None
        self.persona_id: Optional[str] = None
        self.demographics: Optional[Dict[str, Any]] = None

        # History tracking
        self.history: List[Dict[str, Any]] = []
        self.evaluation: Optional[Dict[str, Any]] = None


def get_model_stats(model_id: int) -> Optional[ModelStats]:
    """Get model statistics from the CSV (without revealing the name)."""
    
    def safe_int(val):
        """Safely convert value to int, handling comma-formatted numbers."""
        if pd.isna(val) or val is None:
            return None
        if isinstance(val, (int, float)):
            return int(val)
        # Handle string with commas like "400,000"
        try:
            return int(str(val).replace(",", ""))
        except (ValueError, TypeError):
            return None
    
    def safe_float(val):
        """Safely convert value to float."""
        if pd.isna(val) or val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(str(val).replace(",", ""))
        except (ValueError, TypeError):
            return None
    
    def safe_bool(val):
        """Safely convert value to bool."""
        if pd.isna(val) or val is None:
            return None
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            return val.lower() in ('true', '1', 'yes')
        return None
    
    try:
        row = MODEL_POOL.loc[MODEL_POOL["id"] == int(model_id)]
        if row.empty:
            return None
        row = row.iloc[0]

        return ModelStats(
            id=int(model_id),
            intelligence=safe_int(row.get("intelligence")),
            speed=safe_int(row.get("speed")),
            reasoning=safe_int(row.get("reasoning")),
            input_price=safe_float(row.get("input-price")),
            output_price=safe_float(row.get("output-price")),
            context_window=safe_int(row.get("window-context")),
            max_output=safe_int(row.get("max-output")),
            text_input=safe_bool(row.get("text-input")),
            image_input=safe_bool(row.get("image-input")),
            voice_input=safe_bool(row.get("voice-input")),
            function_calling=safe_bool(row.get("function-calling")),
            structured_output=safe_bool(row.get("structured-output")),
            knowledge_cutoff=str(row.get("knowledge-cutoff", ""))
            if pd.notna(row.get("knowledge-cutoff"))
            else None,
        )
    except Exception as e:
        print(f"[get_model_stats] Error for model {model_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


# Global session storage (in production, use Redis or database)
sessions: Dict[str, SessionState] = {}

# Storage for completed sessions (for persistence)
completed_sessions: Dict[str, Dict[str, Any]] = {}


def get_or_create_session(session_id: Optional[str] = None) -> Tuple[str, SessionState]:
    """Get existing session or create new one."""
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]

    new_id = str(uuid.uuid4())
    sessions[new_id] = SessionState()
    return new_id, sessions[new_id]


# ================== FastAPI App ==================
app = FastAPI(
    title="LLM Model Selection User Study API",
    description="Backend for comparing CUPID vs Baseline algorithms",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================== Database Events ==================
@app.on_event("startup")
async def startup():
    """Initialize database connection and create tables."""
    print("=" * 60)
    print("[startup] LLM Model Selection User Study API starting...")
    print(f"[startup] DATABASE_URL configured: {bool(DATABASE_URL)}")
    if DATABASE_URL:
        # Show masked URL for debugging (hide password)
        try:
            from urllib.parse import urlparse
            parsed = urlparse(DATABASE_URL)
            masked_url = f"{parsed.scheme}://{parsed.username}:***@{parsed.hostname}:{parsed.port}{parsed.path}"
            print(f"[startup] DATABASE_URL (masked): {masked_url}")
        except:
            print(f"[startup] DATABASE_URL (first 30 chars): {DATABASE_URL[:30]}...")
    print(f"[startup] DATABASE_AVAILABLE: {DATABASE_AVAILABLE}")
    print(f"[startup] database object: {database is not None}")
    print("=" * 60)
    
    if database:
        try:
            await database.connect()
            print("[startup] ✅ Database connected successfully")

            # Create results table if it doesn't exist
            await database.execute("""
                CREATE TABLE IF NOT EXISTS study_results (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) UNIQUE NOT NULL,
                    persona_group VARCHAR(50),
                    results_json JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes separately
            await database.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id ON study_results(session_id)
            """)
            await database.execute("""
                CREATE INDEX IF NOT EXISTS idx_persona_group ON study_results(persona_group)
            """)
            await database.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON study_results(created_at)
            """)

            # Test the connection by counting rows
            result = await database.fetch_one("SELECT COUNT(*) as count FROM study_results")
            print(f"[startup] ✅ Database table ready - {result['count']} existing records")
        except Exception as e:
            print(f"[startup] ⚠️ Database connection failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("[startup] ℹ️ Database not configured - results will not be saved automatically")
        if not DATABASE_URL:
            print("[startup]    → DATABASE_URL environment variable is not set")
        if not DATABASE_AVAILABLE:
            print("[startup]    → databases package is not installed")


@app.on_event("shutdown")
async def shutdown():
    """Close database connection."""
    if database:
        await database.disconnect()
        print("Database disconnected")


# ================== Request/Response Models ==================
class InteractRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Session ID for continuity")
    prompt: str = Field(..., description="User's prompt to send to models")
    previous_vote: Optional[str] = Field(
        None,
        description="Previous vote: 'cupid_left', 'cupid_right', 'baseline_left', 'baseline_right' or null for first round",
    )
    feedback_text: Optional[str] = Field(
        "", description="User's feedback/direction for model selection"
    )
    # Support sending both votes at once
    cupid_vote: Optional[str] = Field(None, description="CUPID vote: 'left' or 'right'")
    baseline_vote: Optional[str] = Field(
        None, description="Baseline vote: 'left' or 'right'"
    )
    # Budget and persona (sent on first round)
    budget_cost: Optional[float] = Field(None, description="Maximum cost budget")
    budget_rounds: Optional[int] = Field(None, description="Maximum number of rounds")
    persona_id: Optional[str] = Field(None, description="Assigned persona ID")
    persona_group: Optional[str] = Field(
        None, description="Persona group: 'traditional', 'expert', or 'preference'"
    )
    expert_subject: Optional[str] = Field(
        None, description="Expert subject for expert group"
    )
    constraints: Optional[List[Dict[str, Any]]] = Field(
        None, description="Hard constraints for traditional group"
    )
    demographics: Optional[Dict[str, Any]] = Field(
        None, description="User demographics"
    )


class ModelResponse(BaseModel):
    model_id: int
    model_name: str
    text: str
    cost: float


class InteractResponse(BaseModel):
    session_id: str
    round: int
    total_cost: float  # Keep for backward compatibility
    cupid_cost: float  # Separate cost for System A
    baseline_cost: float  # Separate cost for System B
    routing_cost: float  # Cost for language feedback routing model
    cLeft: ModelResponse
    cRight: ModelResponse
    bLeft: ModelResponse
    bRight: ModelResponse
    cLeftStats: Optional[ModelStats] = None
    cRightStats: Optional[ModelStats] = None
    bLeftStats: Optional[ModelStats] = None
    bRightStats: Optional[ModelStats] = None


class ChatRequest(BaseModel):
    session_id: str
    message: str
    system: str = Field(..., description="'cupid' or 'baseline'")


class ChatResponse(BaseModel):
    response: str
    cost: float


class SessionInfoResponse(BaseModel):
    session_id: str
    round_count: int
    cupid_rounds: int
    baseline_rounds: int
    num_models: int
    total_cost: float


class SaveSessionRequest(BaseModel):
    demographics: Optional[Dict[str, Any]] = None
    persona: Optional[Dict[str, Any]] = None
    persona_group: Optional[str] = None  # 'traditional', 'expert', or 'preference'
    expert_subject: Optional[str] = None  # For expert group
    constraints: Optional[List[Dict[str, Any]]] = None  # For traditional group
    budget: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, Any]]] = None
    evaluation: Optional[Dict[str, Any]] = None
    final_cost: Optional[float] = None
    final_cost_a: Optional[float] = None  # System A total cost
    final_cost_b: Optional[float] = None  # System B total cost
    terminated_early: Optional[bool] = None  # True if user clicked "I'm Satisfied"
    open_test_rounds_a: Optional[int] = None  # Open testing rounds for System A
    open_test_rounds_b: Optional[int] = None  # Open testing rounds for System B


# ================== API Endpoints ==================
@app.get("/")
async def root():
    return {
        "message": "LLM Model Selection User Study API",
        "endpoints": {
            "/interact": "POST - Main interaction endpoint",
            "/session/{session_id}": "GET - Get session info",
            "/models": "GET - List available models",
            "/health": "GET - Health check",
        },
    }


@app.get("/health")
async def health_check():
    db_status = "not_configured"
    db_record_count = None
    db_error = None
    
    if database:
        try:
            # Test database connection
            await database.execute("SELECT 1")
            db_status = "connected"
            # Get record count
            result = await database.fetch_one("SELECT COUNT(*) as count FROM study_results")
            db_record_count = result['count'] if result else 0
        except Exception as e:
            db_status = "error"
            db_error = str(e)

    return {
        "status": "healthy",
        "botorch_available": BOTORCH_AVAILABLE,
        "num_models": num_models,
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "database_status": db_status,
        "database_record_count": db_record_count,
        "database_error": db_error,
        "active_sessions": len(sessions),
    }


@app.get("/debug/database")
async def debug_database():
    """Debug endpoint to check database status and contents."""
    if not database:
        return {
            "error": "Database not configured",
            "DATABASE_URL_set": bool(DATABASE_URL),
            "DATABASE_AVAILABLE": DATABASE_AVAILABLE,
        }
    
    try:
        # Check connection
        await database.execute("SELECT 1")
        
        # Get all records
        records = await database.fetch_all(
            "SELECT id, session_id, persona_group, created_at FROM study_results ORDER BY created_at DESC LIMIT 20"
        )
        
        # Get total count
        count_result = await database.fetch_one("SELECT COUNT(*) as count FROM study_results")
        
        return {
            "status": "connected",
            "total_records": count_result['count'] if count_result else 0,
            "recent_records": [
                {
                    "id": r["id"],
                    "session_id": r["session_id"],
                    "persona_group": r["persona_group"],
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                }
                for r in records
            ],
            "database_url_masked": DATABASE_URL[:20] + "..." if DATABASE_URL else None,
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


@app.get("/models")
async def list_models():
    """List all available models in the pool."""
    models = []
    for _, row in MODEL_POOL.iterrows():
        models.append(
            {
                "id": int(row["id"]),
                "model_id": row["model-id"],
                "name": row.get("model", row["model-id"]),
                "input_price": row.get("input-price", 0),
                "output_price": row.get("output-price", 0),
            }
        )
    return {"models": models, "count": len(models)}


@app.get("/model-pool-stats")
async def get_model_pool_stats():
    """Get model pool statistics for constraint sampling.
    Returns only the attributes shown in model cards for constraint generation."""

    def safe_int(val):
        """Safely convert value to int, handling comma-formatted numbers."""
        if pd.isna(val) or val is None:
            return None
        if isinstance(val, (int, float)):
            return int(val)
        # Handle string with commas like "400,000"
        try:
            return int(str(val).replace(",", ""))
        except (ValueError, TypeError):
            return None

    def safe_float(val):
        """Safely convert value to float."""
        if pd.isna(val) or val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(str(val).replace(",", ""))
        except (ValueError, TypeError):
            return None

    stats = []
    for _, row in MODEL_POOL.iterrows():
        stats.append(
            {
                "id": int(row["id"]),
                "intelligence": safe_int(row.get("intelligence")),
                "speed": safe_int(row.get("speed")),
                "reasoning": safe_int(row.get("reasoning")),
                "input_price": safe_float(row.get("input-price")),
                "output_price": safe_float(row.get("output-price")),
                "context_window": safe_int(row.get("window-context")),
                "max_output": safe_int(row.get("max-output")),
            }
        )
    return {"models": stats, "count": len(stats)}


# ================== Save Results Endpoint ==================
class SaveResultsRequest(BaseModel):
    """Request model for saving study results."""

    session_id: str
    timestamp: str
    demographics: Optional[Dict[str, Any]] = None
    persona_group: Optional[str] = None
    expert_subject: Optional[str] = None
    constraints: Optional[List[Dict[str, Any]]] = None
    budget: Optional[Dict[str, Any]] = None
    initial_preference: Optional[str] = None
    final_state: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, Any]]] = None
    open_testing: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None


@app.post("/save-results")
async def save_results(request: SaveResultsRequest):
    """
    Save study results to database.
    This endpoint is called when the user completes the study.
    """
    print(f"[save-results] Received save request for session: {request.session_id}")
    print(f"[save-results] Persona group: {request.persona_group}")
    print(f"[save-results] Initial preference: {request.initial_preference}")
    print(f"[save-results] History length: {len(request.history) if request.history else 0}")
    print(f"[save-results] Database configured: {database is not None}")
    
    if not database:
        print("[save-results] ❌ Database not configured")
        return {
            "success": False,
            "message": "Database not configured. Please download results manually.",
            "saved": False,
        }

    try:
        # Convert request to dict for JSON storage
        results_dict = request.model_dump()
        print(f"[save-results] Converted to dict, keys: {list(results_dict.keys())}")

        # Check count before insert
        before_count = await database.fetch_one("SELECT COUNT(*) as count FROM study_results")
        print(f"[save-results] Records before insert: {before_count['count']}")

        # Use upsert (INSERT ... ON CONFLICT) to avoid race conditions
        print(f"[save-results] Executing database insert...")
        await database.execute(
            query="""
            INSERT INTO study_results (session_id, persona_group, results_json)
            VALUES (:session_id, :persona_group, :results_json)
            ON CONFLICT (session_id) DO UPDATE SET
                results_json = EXCLUDED.results_json,
                persona_group = EXCLUDED.persona_group,
                created_at = CURRENT_TIMESTAMP
            """,
            values={
                "session_id": request.session_id,
                "persona_group": request.persona_group,
                "results_json": json.dumps(results_dict),
            },
        )
        
        # Verify the insert worked
        after_count = await database.fetch_one("SELECT COUNT(*) as count FROM study_results")
        print(f"[save-results] Records after insert: {after_count['count']}")
        
        # Try to fetch the specific record we just inserted
        verification = await database.fetch_one(
            "SELECT session_id, persona_group, created_at FROM study_results WHERE session_id = :session_id",
            values={"session_id": request.session_id}
        )
        
        if verification:
            print(f"[save-results] ✅ Verified record exists: session_id={verification['session_id']}, created_at={verification['created_at']}")
        else:
            print(f"[save-results] ⚠️ Could not verify record - SELECT returned None")
        
        print(f"[save-results] ✅ Successfully saved session {request.session_id}")
        return {
            "success": True,
            "message": "Results saved successfully",
            "saved": True,
            "session_id": request.session_id,
            "verified": verification is not None,
        }

    except Exception as e:
        print(f"[save-results] ❌ Error saving results: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Failed to save results: {str(e)}",
            "saved": False,
        }


@app.get("/results")
async def get_all_results():
    """
    Get all saved study results (for research/admin use).
    Returns summary statistics and list of sessions.
    """
    if not database:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        # Get count by persona group
        counts = await database.fetch_all(
            query="""
            SELECT persona_group, COUNT(*) as count 
            FROM study_results 
            GROUP BY persona_group
            """
        )

        # Get recent sessions
        recent = await database.fetch_all(
            query="""
            SELECT session_id, persona_group, created_at 
            FROM study_results 
            ORDER BY created_at DESC 
            LIMIT 50
            """
        )

        total = sum(row["count"] for row in counts)

        return {
            "total_sessions": total,
            "by_group": {row["persona_group"]: row["count"] for row in counts},
            "recent_sessions": [
                {
                    "session_id": row["session_id"],
                    "persona_group": row["persona_group"],
                    "created_at": row["created_at"].isoformat()
                    if row["created_at"]
                    else None,
                }
                for row in recent
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/results/{session_id}")
async def get_session_results(session_id: str):
    """Get results for a specific session."""
    if not database:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        result = await database.fetch_one(
            query="SELECT * FROM study_results WHERE session_id = :session_id",
            values={"session_id": session_id},
        )

        if not result:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "session_id": result["session_id"],
            "persona_group": result["persona_group"],
            "created_at": result["created_at"].isoformat()
            if result["created_at"]
            else None,
            "results": json.loads(result["results_json"]),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/results/export/all")
async def export_all_results():
    """Export all results as JSON array (for download/analysis)."""
    if not database:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        results = await database.fetch_all(
            query="SELECT results_json FROM study_results ORDER BY created_at"
        )

        return {
            "count": len(results),
            "results": [json.loads(row["results_json"]) for row in results],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/session/{session_id}")
async def get_session_info(session_id: str) -> SessionInfoResponse:
    """Get information about a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    state = sessions[session_id]
    total_cost = state.cupid.total_cost + state.baseline.total_cost

    return SessionInfoResponse(
        session_id=session_id,
        round_count=state.round_count,
        cupid_rounds=state.cupid.round_count,
        baseline_rounds=state.baseline.round_count,
        num_models=num_models,
        total_cost=total_cost,
    )


@app.post("/interact", response_model=InteractResponse)
async def interact(request: InteractRequest):
    """
    Main interaction endpoint.

    Takes user prompt, previous vote(s), and feedback text.
    Returns 4 model responses: cLeft, cRight (CUPID) and bLeft, bRight (Baseline).
    """
    # Get or create session
    session_id, state = get_or_create_session(request.session_id)
    
    is_new_session = request.session_id is None
    if is_new_session:
        print(f"[interact] 🆕 New session created: {session_id}")
        print(f"[interact]    Persona group: {request.persona_group}")
        print(f"[interact]    Initial preference/feedback: {request.feedback_text}")
        print(f"[interact]    Budget: {request.budget_rounds} rounds, ${request.budget_cost}")
    else:
        print(f"[interact] 🔄 Round {state.round_count + 1} for session: {session_id}")
        if request.feedback_text:
            print(f"[interact]    Feedback: {request.feedback_text}")

    async with state.lock:
        # On first round, save budget and persona
        if request.budget_cost is not None:
            state.budget_cost = request.budget_cost
        if request.budget_rounds is not None:
            state.budget_rounds = request.budget_rounds
        if request.persona_id is not None:
            state.persona_id = request.persona_id
        if request.demographics is not None:
            state.demographics = request.demographics

        # Process votes - support both old format (previous_vote) and new format (cupid_vote/baseline_vote)

        # Handle CUPID vote
        cupid_winner_is_left = None
        if request.cupid_vote:
            cupid_winner_is_left = request.cupid_vote.lower() == "left"
        elif request.previous_vote and "cupid" in request.previous_vote.lower():
            cupid_winner_is_left = "left" in request.previous_vote.lower()

        if cupid_winner_is_left is not None:
            # Track the winning model as the final converged model
            if (
                state.cupid.current_left_id is not None
                and state.cupid.current_right_id is not None
            ):
                state.final_cupid_model_id = (
                    state.cupid.current_left_id
                    if cupid_winner_is_left
                    else state.cupid.current_right_id
                )
            state.cupid.update_with_vote(cupid_winner_is_left)

        # Handle Baseline vote
        baseline_winner_is_left = None
        if request.baseline_vote:
            baseline_winner_is_left = request.baseline_vote.lower() == "left"
        elif request.previous_vote and "baseline" in request.previous_vote.lower():
            baseline_winner_is_left = "left" in request.previous_vote.lower()

        if baseline_winner_is_left is not None:
            # Track the winning model as the final converged model
            if (
                state.baseline.current_left_id is not None
                and state.baseline.current_right_id is not None
            ):
                state.final_baseline_model_id = (
                    state.baseline.current_left_id
                    if baseline_winner_is_left
                    else state.baseline.current_right_id
                )
            state.baseline.update_with_vote(baseline_winner_is_left)

        # Select new pairs
        cupid_left_id, cupid_right_id = state.cupid.select_pair(
            request.feedback_text or ""
        )
        baseline_left_id, baseline_right_id = state.baseline.select_pair()

        # Call OpenRouter for all 4 models
        prompt = request.prompt

        # CUPID responses
        cupid_left_result = call_openrouter(prompt, cupid_left_id)
        cupid_right_result = call_openrouter(prompt, cupid_right_id)

        # Baseline responses
        baseline_left_result = call_openrouter(prompt, baseline_left_id)
        baseline_right_result = call_openrouter(prompt, baseline_right_id)

        # Update costs
        round_cost = (
            cupid_left_result["cost"]
            + cupid_right_result["cost"]
            + baseline_left_result["cost"]
            + baseline_right_result["cost"]
        )
        state.cupid.total_cost += cupid_left_result["cost"] + cupid_right_result["cost"]
        state.baseline.total_cost += (
            baseline_left_result["cost"] + baseline_right_result["cost"]
        )

        # Increment round
        state.round_count += 1

        # Calculate total cost across both systems
        total_cost = state.cupid.total_cost + state.baseline.total_cost

        # Track history for this round
        state.history.append(
            {
                "round": state.round_count,
                "prompt": prompt,
                "cupid_left_id": cupid_left_id,
                "cupid_right_id": cupid_right_id,
                "baseline_left_id": baseline_left_id,
                "baseline_right_id": baseline_right_id,
                "feedback": request.feedback_text,
                "cupid_vote": request.cupid_vote,
                "baseline_vote": request.baseline_vote,
                "round_cost": round_cost,
                "total_cost": total_cost,
            }
        )

        # Get model stats (without names)
        c_left_stats = get_model_stats(cupid_left_id)
        c_right_stats = get_model_stats(cupid_right_id)
        b_left_stats = get_model_stats(baseline_left_id)
        b_right_stats = get_model_stats(baseline_right_id)

        return InteractResponse(
            session_id=session_id,
            round=state.round_count,
            total_cost=total_cost,
            cupid_cost=state.cupid.total_cost,
            baseline_cost=state.baseline.total_cost,
            routing_cost=getattr(state, "routing_cost", 0.0),
            cLeft=ModelResponse(
                model_id=cupid_left_id,
                model_name=_model_display_name_from_id(cupid_left_id),
                text=cupid_left_result["text"],
                cost=cupid_left_result["cost"],
            ),
            cRight=ModelResponse(
                model_id=cupid_right_id,
                model_name=_model_display_name_from_id(cupid_right_id),
                text=cupid_right_result["text"],
                cost=cupid_right_result["cost"],
            ),
            bLeft=ModelResponse(
                model_id=baseline_left_id,
                model_name=_model_display_name_from_id(baseline_left_id),
                text=baseline_left_result["text"],
                cost=baseline_left_result["cost"],
            ),
            bRight=ModelResponse(
                model_id=baseline_right_id,
                model_name=_model_display_name_from_id(baseline_right_id),
                text=baseline_right_result["text"],
                cost=baseline_right_result["cost"],
            ),
            cLeftStats=c_left_stats,
            cRightStats=c_right_stats,
            bLeftStats=b_left_stats,
            bRightStats=b_right_stats,
        )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    del sessions[session_id]
    return {"message": "Session deleted", "session_id": session_id}


@app.post("/session/{session_id}/save")
async def save_session(session_id: str, request: SaveSessionRequest):
    """Save session data for a completed user study."""
    import json
    import os
    from datetime import datetime

    # Get session state if it exists
    state = sessions.get(session_id)

    # Compile all session data
    session_data = {
        "session_id": session_id,
        "saved_at": datetime.now().isoformat(),
        "demographics": request.demographics,
        "persona": request.persona,
        "persona_group": request.persona_group,
        "expert_subject": request.expert_subject,
        "constraints": request.constraints,
        "budget": request.budget,
        "history": request.history or (state.history if state else []),
        "evaluation": request.evaluation,
        "final_cost": request.final_cost,
        "final_cost_a": request.final_cost_a,
        "final_cost_b": request.final_cost_b,
        "terminated_early": request.terminated_early,
        "open_test_rounds_a": request.open_test_rounds_a,
        "open_test_rounds_b": request.open_test_rounds_b,
        "backend_history": state.history if state else [],
        "budget_settings": {
            "cost": state.budget_cost if state else None,
            "rounds": state.budget_rounds if state else None,
        }
        if state
        else None,
    }

    # Store in memory
    completed_sessions[session_id] = session_data

    # Save to file for persistence
    output_dir = "./session_data"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/session_{session_id}.json"
    try:
        with open(filename, "w") as f:
            json.dump(session_data, f, indent=2, default=str)
    except Exception as e:
        print(f"Warning: Could not save session to file: {e}")

    return {
        "message": "Session saved successfully",
        "session_id": session_id,
        "filename": filename,
    }


@app.get("/session/{session_id}/data")
async def get_session_data(session_id: str):
    """Get saved session data."""
    if session_id in completed_sessions:
        return completed_sessions[session_id]

    # Try loading from file
    import json

    filename = f"./session_data/session_{session_id}.json"
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session data not found")


@app.get("/sessions/list")
async def list_sessions():
    """List all saved sessions."""
    import os
    import glob

    sessions_list = []

    # From memory
    for sid, data in completed_sessions.items():
        sessions_list.append(
            {"session_id": sid, "saved_at": data.get("saved_at"), "source": "memory"}
        )

    # From files
    try:
        files = glob.glob("./session_data/session_*.json")
        for f in files:
            sid = f.split("session_")[-1].replace(".json", "")
            if sid not in completed_sessions:
                sessions_list.append(
                    {"session_id": sid, "source": "file", "filename": f}
                )
    except Exception:
        pass

    return {"sessions": sessions_list, "count": len(sessions_list)}


@app.post("/chat", response_model=ChatResponse)
async def chat_with_model(request: ChatRequest):
    """
    Chat endpoint for open testing phase.
    Allows users to chat with the final converged model from either system.
    """
    session_id = request.session_id

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    state = sessions[session_id]

    # Determine which model to use based on system selection
    if request.system == "cupid":
        # Use the most recently selected CUPID model (winner of last round)
        if state.final_cupid_model_id:
            model_id = state.final_cupid_model_id
        elif state.cupid.history:
            # Get most recent winner
            last_entry = state.cupid.history[-1]
            model_id = last_entry.get(
                "winner_id", state.cupid.arms[0] if state.cupid.arms else MODEL_IDS[0]
            )
        else:
            model_id = state.cupid.arms[0] if state.cupid.arms else MODEL_IDS[0]
    else:  # baseline
        if state.final_baseline_model_id:
            model_id = state.final_baseline_model_id
        elif state.baseline.history:
            last_entry = state.baseline.history[-1]
            model_id = last_entry.get(
                "winner_id",
                state.baseline.arms[0] if state.baseline.arms else MODEL_IDS[0],
            )
        else:
            model_id = state.baseline.arms[0] if state.baseline.arms else MODEL_IDS[0]

    # Call the model
    result = call_openrouter(request.message, model_id)

    # Track cost
    if request.system == "cupid":
        state.cupid.total_cost += result["cost"]
    else:
        state.baseline.total_cost += result["cost"]

    return ChatResponse(response=result["text"], cost=result["cost"])


# ================== Main ==================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

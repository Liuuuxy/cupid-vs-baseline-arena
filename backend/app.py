"""
FastAPI Backend for Model Selection User Study
- Text Mode: Full persona groups (traditional/expert/preference) with constraints (original)
- Image Mode: Preference-only (simplified)
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
from enum import Enum

import numpy as np
import pandas as pd
import torch
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Database imports
try:
    from databases import Database

    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    print("Warning: databases package not available.")

# Runware SDK import
try:
    from runware import Runware, IImageInference

    RUNWARE_SDK_AVAILABLE = True
except ImportError:
    RUNWARE_SDK_AVAILABLE = False
    print("Warning: runware SDK not available. Will use REST API fallback.")

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


# ================== Mode Enum ==================
class ArenaMode(str, Enum):
    TEXT = "text"
    IMAGE = "image"


# ================== Configuration ==================
# OpenRouter for LLM text generation
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}" if OPENROUTER_API_KEY else "",
    "Content-Type": "application/json",
}

# Runware for text-to-image generation
RUNWARE_API_KEY = os.environ.get("RUNWARE_API_KEY", "")

# Model pool paths
MODEL_POOL_PATH = os.environ.get("MODEL_POOL_PATH", "./model-pool.csv")
MODEL_POOL_LOCAL = os.environ.get("MODEL_POOL_LOCAL", "./model-pool.csv")
MODEL_POOL_IMAGE_PATH = os.environ.get(
    "MODEL_POOL_IMAGE_PATH", "./image_model_pool.csv"
)

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

database = Database(DATABASE_URL) if DATABASE_URL and DATABASE_AVAILABLE else None

# Global Runware client instance
runware_client: Optional[Runware] = None


# ================== Load Model Pools ==================
def _load_model_pool() -> pd.DataFrame:
    """Load LLM model pool CSV (original)."""
    if os.path.exists(MODEL_POOL_PATH):
        df = pd.read_csv(MODEL_POOL_PATH)
    elif os.path.exists(MODEL_POOL_LOCAL):
        df = pd.read_csv(MODEL_POOL_LOCAL)
    else:
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


def _load_image_model_pool() -> pd.DataFrame:
    """Load text-to-image model pool CSV."""
    if os.path.exists(MODEL_POOL_IMAGE_PATH):
        df = pd.read_csv(MODEL_POOL_IMAGE_PATH)
    else:
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "model-id": [
                    "openai:1@1",
                    "openai:2@3",
                    "openai:2@2",
                    "openai:1@2",
                    "openai:4@1",
                    "google:2@3",
                    "google:2@1",
                    "google:2@2",
                    "google:1@2",
                    "google:1@1",
                    "google:4@1",
                    "google:4@2",
                ],
                "model": [
                    "GPT Image 1",
                    "DallE 3",
                    "DallE 2",
                    "GPT Image 1 Mini",
                    "GPT Image 1.5",
                    "Imagen 4.0 Fast",
                    "Imagen 4.0 Preview",
                    "Imagen 4.0 Ultra",
                    "Imagen 3.0 Fast",
                    "Imagen 3.0",
                    "Gemini Flash Image 2.5 (Nano Banana)",
                    "Nano Banana Pro",
                ],
            }
        )
    df.columns = df.columns.str.strip()
    drop_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


# Load both model pools
MODEL_POOL = _load_model_pool()
MODEL_POOL_IMAGE = _load_image_model_pool()

# Validate model pools
for pool_name, pool in [("LLM", MODEL_POOL), ("Image", MODEL_POOL_IMAGE)]:
    if "id" not in pool.columns:
        raise RuntimeError(f"{pool_name} model pool must contain an 'id' column.")
    if "model-id" not in pool.columns:
        raise RuntimeError(f"{pool_name} model pool must contain a 'model-id' column.")

MODEL_IDS: List[int] = list(range(1, MODEL_POOL.shape[0] + 1))
MODEL_IDS_IMAGE: List[int] = list(range(1, MODEL_POOL_IMAGE.shape[0] + 1))


def get_model_pool(mode: ArenaMode) -> pd.DataFrame:
    return MODEL_POOL if mode == ArenaMode.TEXT else MODEL_POOL_IMAGE


def get_model_ids(mode: ArenaMode) -> List[int]:
    return MODEL_IDS if mode == ArenaMode.TEXT else MODEL_IDS_IMAGE


def _model_name_from_id(model_id: int, mode: ArenaMode = ArenaMode.TEXT) -> str:
    pool = get_model_pool(mode)
    row = pool.loc[pool["id"] == int(model_id)]
    if row.empty:
        raise ValueError(f"No row in model pool with id={model_id}")
    return str(row["model-id"].iloc[0])


def _model_display_name_from_id(model_id: int, mode: ArenaMode = ArenaMode.TEXT) -> str:
    pool = get_model_pool(mode)
    row = pool.loc[pool["id"] == int(model_id)]
    if row.empty:
        return f"Model {model_id}"
    if "model" in row.columns:
        return str(row["model"].iloc[0])
    return str(row["model-id"].iloc[0])


def _id_to_index(model_id: int, mode: ArenaMode = ArenaMode.TEXT) -> int:
    num_models = len(get_model_ids(mode))
    idx = int(model_id) - 1
    if idx < 0 or idx >= num_models:
        raise ValueError(f"Model ID {model_id} is out of range [1, {num_models}]")
    return idx


# ================== Database Functions for Image Results ==================
async def init_image_database():
    """Initialize the image results table in the existing database."""
    if not database:
        return

    create_table_query = """
    CREATE TABLE IF NOT EXISTS image_results (
        id SERIAL PRIMARY KEY,
        task_uuid VARCHAR(255) UNIQUE,
        session_id VARCHAR(255),
        model_id INTEGER,
        model_name VARCHAR(255),
        display_name VARCHAR(255),
        prompt TEXT,
        image_url TEXT,
        width INTEGER,
        height INTEGER,
        seed BIGINT,
        cost DECIMAL(10, 6),
        algorithm VARCHAR(50),
        round_number INTEGER,
        position VARCHAR(10),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata JSONB
    );
    """

    try:
        await database.execute(create_table_query)
        # Create indexes separately to avoid errors if they already exist
        try:
            await database.execute(
                "CREATE INDEX IF NOT EXISTS idx_image_results_session ON image_results(session_id);"
            )
        except Exception:
            pass
        try:
            await database.execute(
                "CREATE INDEX IF NOT EXISTS idx_image_results_model ON image_results(model_id);"
            )
        except Exception:
            pass
        try:
            await database.execute(
                "CREATE INDEX IF NOT EXISTS idx_image_results_created ON image_results(created_at);"
            )
        except Exception:
            pass
        print("Image results table initialized successfully.")
    except Exception as e:
        print(f"Warning: Could not create image_results table: {e}")


async def init_study_results_table():
    """Initialize the study_results table for storing complete study data."""
    if not database:
        return

    create_table_query = """
    CREATE TABLE IF NOT EXISTS study_results (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(255) UNIQUE,
        timestamp TIMESTAMP,
        mode VARCHAR(20),
        demographics JSONB,
        persona_group VARCHAR(50),
        expert_subject VARCHAR(100),
        constraints JSONB,
        budget JSONB,
        initial_preference VARCHAR(50),
        final_state JSONB,
        history JSONB,
        open_testing JSONB,
        evaluation JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    try:
        await database.execute(create_table_query)
        try:
            await database.execute(
                "CREATE INDEX IF NOT EXISTS idx_study_results_session ON study_results(session_id);"
            )
        except Exception:
            pass
        try:
            await database.execute(
                "CREATE INDEX IF NOT EXISTS idx_study_results_persona ON study_results(persona_group);"
            )
        except Exception:
            pass
        try:
            await database.execute(
                "CREATE INDEX IF NOT EXISTS idx_study_results_created ON study_results(created_at);"
            )
        except Exception:
            pass
        print("Study results table initialized successfully.")
    except Exception as e:
        print(f"Warning: Could not create study_results table: {e}")


async def save_image_result(
    task_uuid: str,
    session_id: Optional[str],
    model_id: int,
    model_name: str,
    display_name: str,
    prompt: str,
    image_url: str,
    width: int,
    height: int,
    seed: int,
    cost: float,
    algorithm: Optional[str] = None,
    round_number: Optional[int] = None,
    position: Optional[str] = None,
    metadata: Optional[Dict] = None,
):
    """Save image generation result to the database."""
    if not database:
        return

    insert_query = """
    INSERT INTO image_results 
    (task_uuid, session_id, model_id, model_name, display_name, prompt, image_url, 
     width, height, seed, cost, algorithm, round_number, position, metadata)
    VALUES 
    (:task_uuid, :session_id, :model_id, :model_name, :display_name, :prompt, :image_url,
     :width, :height, :seed, :cost, :algorithm, :round_number, :position, :metadata)
    ON CONFLICT (task_uuid) DO UPDATE SET
        image_url = EXCLUDED.image_url,
        cost = EXCLUDED.cost,
        metadata = EXCLUDED.metadata
    """

    try:
        await database.execute(
            insert_query,
            {
                "task_uuid": task_uuid,
                "session_id": session_id,
                "model_id": model_id,
                "model_name": model_name,
                "display_name": display_name,
                "prompt": prompt,
                "image_url": image_url,
                "width": width,
                "height": height,
                "seed": seed,
                "cost": cost,
                "algorithm": algorithm,
                "round_number": round_number,
                "position": position,
                "metadata": json.dumps(metadata) if metadata else None,
            },
        )
    except Exception as e:
        print(f"Warning: Could not save image result to database: {e}")


# ================== API Call Functions ==================
def call_openrouter(prompt: str, model_id: int) -> Dict[str, Any]:
    """Generate LLM text response via OpenRouter (original)."""
    model_name = _model_name_from_id(model_id, ArenaMode.TEXT)

    try:
        row = MODEL_POOL[MODEL_POOL["id"] == model_id].iloc[0]
        input_price = float(row.get("input-price", 0) or 0)
        output_price = float(row.get("output-price", 0) or 0)
    except (IndexError, KeyError):
        input_price, output_price = 0.0, 0.0

    if not OPENROUTER_API_KEY:
        mock_prompt_tokens = int(len(prompt.split()) * 1.3) + 30
        mock_completion_tokens = 150
        mock_cost = (mock_prompt_tokens * input_price / 1_000_000) + (
            mock_completion_tokens * output_price / 1_000_000
        )
        return {
            "text": f"[Mock response from {_model_display_name_from_id(model_id, ArenaMode.TEXT)}] This is a simulated response. Your query: {prompt[:100]}...",
            "cost": mock_cost,
            "prompt_tokens": mock_prompt_tokens,
            "completion_tokens": mock_completion_tokens,
        }

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Answer concisely and accurately.",
            },
            {"role": "user", "content": prompt},
        ],
        "usage": {"include": True},
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=OPENROUTER_HEADERS,
            data=json.dumps(payload),
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        text = ""
        if isinstance(data.get("choices"), list) and data["choices"]:
            text = data["choices"][0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        return {
            "text": text,
            "cost": float(usage.get("cost", 0) or 0),
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        }
    except Exception as e:
        return {
            "text": f"[Error: {str(e)}]",
            "cost": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": str(e),
        }


async def call_runware(
    prompt: str,
    model_id: int,
    width: int = 1024,
    height: int = 1024,
    session_id: Optional[str] = None,
    algorithm: Optional[str] = None,
    round_number: Optional[int] = None,
    position: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate image via Runware SDK (WebSocket-based)."""
    global runware_client

    model_name = _model_name_from_id(model_id, ArenaMode.IMAGE)
    display_name = _model_display_name_from_id(model_id, ArenaMode.IMAGE)
    task_uuid = str(uuid.uuid4())
    price_per_image = 0.002

    # Mock response if no API key
    if not RUNWARE_API_KEY:
        result = {
            "imageUrl": f"https://via.placeholder.com/{width}x{height}.png?text={display_name.replace(' ', '+')}",
            "cost": price_per_image,
            "seed": 12345678,
            "taskUUID": task_uuid,
        }
        # Save mock result to database
        await save_image_result(
            task_uuid=task_uuid,
            session_id=session_id,
            model_id=model_id,
            model_name=model_name,
            display_name=display_name,
            prompt=prompt,
            image_url=result["imageUrl"],
            width=width,
            height=height,
            seed=result["seed"],
            cost=result["cost"],
            algorithm=algorithm,
            round_number=round_number,
            position=position,
            metadata={"mock": True},
        )
        return result

    # Use Runware SDK if available
    if RUNWARE_SDK_AVAILABLE and runware_client:
        try:
            request = IImageInference(
                positivePrompt=prompt,
                model=model_name,
                width=width,
                height=height,
                numberResults=1,
                outputFormat="JPG",
                includeCost=True,  # Enable cost reporting in response
            )

            images = await runware_client.imageInference(requestImage=request)

            if images and len(images) > 0:
                image = images[0]
                # Access cost from the response - it's included when includeCost=True
                image_cost = price_per_image  # Default fallback
                if hasattr(image, "cost") and image.cost is not None:
                    image_cost = float(image.cost)

                result = {
                    "imageUrl": image.imageURL,
                    "cost": image_cost,
                    "seed": int(getattr(image, "seed", 0) or 0),
                    "taskUUID": getattr(image, "taskUUID", task_uuid) or task_uuid,
                }

                # Save result to image database
                await save_image_result(
                    task_uuid=result["taskUUID"],
                    session_id=session_id,
                    model_id=model_id,
                    model_name=model_name,
                    display_name=display_name,
                    prompt=prompt,
                    image_url=result["imageUrl"],
                    width=width,
                    height=height,
                    seed=result["seed"],
                    cost=result["cost"],
                    algorithm=algorithm,
                    round_number=round_number,
                    position=position,
                    metadata={"sdk": "runware-python", "model": model_name},
                )
                # In the SDK response handling section, add:
                print(
                    f"Runware response - cost: {getattr(image, 'cost', 'NOT_FOUND')}, attrs: {dir(image)}"
                )
                return result
            else:
                return {
                    "imageUrl": "",
                    "cost": 0.0,
                    "seed": 0,
                    "taskUUID": task_uuid,
                    "error": "No image returned",
                }

        except Exception as e:
            print(f"Runware SDK error: {e}")
            # Fall through to REST API fallback

    # # REST API fallback
    # RUNWARE_URL = "https://api.runware.ai/v1"
    # RUNWARE_HEADERS = {
    #     "Authorization": f"Bearer {RUNWARE_API_KEY}",
    #     "Content-Type": "application/json",
    # }

    # payload = [{
    #     "taskType": "imageInference",
    #     "taskUUID": task_uuid,
    #     "model": model_name,
    #     "positivePrompt": prompt,
    #     "width": width,
    #     "height": height,
    #     "numberResults": 1,
    #     "outputType": "URL",
    #     "outputFormat": "JPG",
    #     "includeCost": True,
    # }]

    # try:
    #     response = requests.post(
    #         f"{RUNWARE_URL}/tasks",
    #         headers=RUNWARE_HEADERS,
    #         json=payload,
    #         timeout=120
    #     )
    #     response.raise_for_status()
    #     data = response.json()

    #     if "data" in data and len(data["data"]) > 0:
    #         api_result = data["data"][0]
    #         result = {
    #             "imageUrl": api_result.get("imageURL", ""),
    #             "cost": float(api_result.get("cost", price_per_image) or price_per_image),
    #             "seed": api_result.get("seed", 0),
    #             "taskUUID": task_uuid,
    #         }

    #         # Save result to image database
    #         await save_image_result(
    #             task_uuid=task_uuid,
    #             session_id=session_id,
    #             model_id=model_id,
    #             model_name=model_name,
    #             display_name=display_name,
    #             prompt=prompt,
    #             image_url=result["imageUrl"],
    #             width=width,
    #             height=height,
    #             seed=result["seed"],
    #             cost=result["cost"],
    #             algorithm=algorithm,
    #             round_number=round_number,
    #             position=position,
    #             metadata={"sdk": "rest-api", "model": model_name}
    #         )

    #         return result
    #     else:
    #         errors = data.get("errors", [])
    #         error_msg = errors[0].get("message", "Unknown error") if errors else "No image returned"
    #         return {"imageUrl": "", "cost": 0.0, "seed": 0, "taskUUID": task_uuid, "error": error_msg}
    # except Exception as e:
    #     print(f"Runware API error: {e}")
    #     return {"imageUrl": "", "cost": 0.0, "seed": 0, "taskUUID": task_uuid, "error": str(e)}


# ================== CUPID Algorithm Components (Original) ==================
if BOTORCH_AVAILABLE:
    torch.set_default_dtype(torch.double)

    def encode_arm(model_id: int, num_models: int) -> torch.Tensor:
        e = torch.zeros(num_models, dtype=torch.double)
        e[model_id - 1] = 1.0
        return e

    def encode_context(ctx_id: int, num_models: int) -> torch.Tensor:
        e = torch.zeros(num_models, dtype=torch.double)
        e[ctx_id - 1] = 1.0
        return e

    def build_point(arm_id: int, ctx_id: int, num_models: int) -> torch.Tensor:
        return torch.cat(
            [encode_arm(arm_id, num_models), encode_context(ctx_id, num_models)], dim=0
        )

    def block_features(arms: List[int], ctx_id: int, num_models: int) -> torch.Tensor:
        return torch.stack([build_point(a, ctx_id, num_models) for a in arms], dim=0)

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


class DiscreteBelief:
    def __init__(self, n: int):
        self.logp = (
            torch.zeros(n, dtype=torch.double) if BOTORCH_AVAILABLE else np.zeros(n)
        )

    def probs(self):
        if BOTORCH_AVAILABLE:
            return torch.softmax(self.logp, dim=0)
        exp_logp = np.exp(self.logp - np.max(self.logp))
        return exp_logp / exp_logp.sum()

    def map(self):
        p = self.probs()
        return int(torch.argmax(p).item()) if BOTORCH_AVAILABLE else int(np.argmax(p))

    def bayes_update(self, log_lik):
        if BOTORCH_AVAILABLE:
            self.logp = self.logp + log_lik
        else:
            self.logp = self.logp + np.array(log_lik)


def _calc_spread(vec, method: str = "iqr") -> float:
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


# ================== Baseline (Bradley-Terry) Components ==================
def bt_grad(theta: np.ndarray, wins: np.ndarray, N: np.ndarray) -> np.ndarray:
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
    def __init__(
        self, arms: List[int], contexts: List[int], mode: ArenaMode = ArenaMode.TEXT
    ):
        self.arms = arms
        self.contexts = contexts
        self.K = len(arms)
        self.M = len(contexts)
        self.mode = mode
        self.num_models = len(arms)

        if BOTORCH_AVAILABLE:
            prior_anchor_ctx = contexts[0]
            self.train_X = block_features(arms, prior_anchor_ctx, self.num_models)
            self.index_of: Dict[Tuple[int, int], int] = {
                (a_idx, 0): a_idx for a_idx in range(self.K)
            }
            self.comps_wl = None
            self.model = fit_model(self.train_X, self.comps_wl)

        self.belief = DiscreteBelief(self.M)
        self.recent_arms: deque = deque(maxlen=5)
        self.current_left_idx: Optional[int] = None
        self.current_right_idx: Optional[int] = None
        self.current_ctx_idx: Optional[int] = None
        self.beta = 1.5
        self.rho = 1.30
        self.kappa = 1.60
        self.spread_method = "iqr"
        self.min_fit_pairs = 5
        self.total_cost = 0.0
        self.round_count = 0
        self.history: List[Dict] = []

    @property
    def current_left_id(self) -> Optional[int]:
        if self.current_left_idx is not None and 0 <= self.current_left_idx < len(
            self.arms
        ):
            return self.arms[self.current_left_idx]
        return None

    @property
    def current_right_id(self) -> Optional[int]:
        if self.current_right_idx is not None and 0 <= self.current_right_idx < len(
            self.arms
        ):
            return self.arms[self.current_right_idx]
        return None

    def select_pair(self, direction_text: str = "") -> Tuple[int, int]:
        if not BOTORCH_AVAILABLE:
            import random

            indices = list(range(self.K))
            i = random.choice(indices)
            j = random.choice([x for x in indices if x != i])
            self.current_left_idx, self.current_right_idx, self.current_ctx_idx = (
                i,
                j,
                0,
            )
            return self.arms[i], self.arms[j]

        cost_vec = torch.zeros(self.K, dtype=torch.double)
        cooldown_vec = torch.zeros(self.K, dtype=torch.double)
        c = Counter(self.recent_arms)
        for idx, cnt in c.items():
            if 0 <= idx < self.K:
                cooldown_vec[idx] = float(cnt)

        p = self.belief.probs()
        expected_base = torch.zeros(self.K, dtype=torch.double)

        for cidx, ctx in enumerate(self.contexts):
            mu, cov = posterior_stats(
                self.model, block_features(self.arms, ctx, self.num_models)
            )
            base_ctx = _base_ucb(mu, cov, self.beta, cooldown_vec, 0.0, cost_vec, 0.0)
            expected_base += p[cidx] * base_ctx

        ucb = expected_base
        k = min(2, ucb.numel())
        top = torch.topk(ucb, k=k).indices.tolist()

        i, j = top[0], top[1] if len(top) > 1 else top[0]
        self.current_left_idx, self.current_right_idx = i, j
        self.current_ctx_idx = self.belief.map()

        return self.arms[i], self.arms[j]

    def update_with_vote(self, winner_is_left: bool):
        if self.current_left_idx is None or self.current_right_idx is None:
            return

        i, j = self.current_left_idx, self.current_right_idx
        winner_arm = i if winner_is_left else j
        loser_arm = j if winner_is_left else i

        if not BOTORCH_AVAILABLE:
            self.recent_arms.extend([winner_arm, loser_arm])
            self.round_count += 1
            return

        for cidx, ctx in enumerate(self.contexts):
            fi = build_point(self.arms[i], ctx, self.num_models)
            fj = build_point(self.arms[j], ctx, self.num_models)

            for key, feat in [((i, cidx), fi), ((j, cidx), fj)]:
                if key not in self.index_of:
                    self.index_of[key] = self.train_X.shape[0]
                    self.train_X = torch.cat([self.train_X, feat.unsqueeze(0)], dim=0)

        map_ctx = self.belief.map()
        idx_w, idx_l = (
            self.index_of[(winner_arm, map_ctx)],
            self.index_of[(loser_arm, map_ctx)],
        )

        new_comp = torch.tensor([[idx_w, idx_l]], dtype=torch.long)
        self.comps_wl = (
            new_comp
            if self.comps_wl is None
            else torch.cat([self.comps_wl, new_comp], dim=0)
        )
        self.model = fit_model(self.train_X, self.comps_wl, self.min_fit_pairs)

        self.recent_arms.extend([winner_arm, loser_arm])
        self.round_count += 1


class BaselineState:
    def __init__(self, arms: List[int], mode: ArenaMode = ArenaMode.TEXT):
        self.arms = arms
        self.K = len(arms)
        self.mode = mode
        self.wins = np.zeros((self.K, self.K), dtype=np.float64)
        self.N = np.zeros((self.K, self.K), dtype=np.float64)
        self.theta = np.zeros(self.K, dtype=np.float64)
        self.current_i: Optional[int] = None
        self.current_j: Optional[int] = None
        self.total_cost = 0.0
        self.round_count = 0
        self.history: List[Dict] = []

    @property
    def current_left_id(self) -> Optional[int]:
        return (
            self.arms[self.current_i]
            if self.current_i is not None and 0 <= self.current_i < len(self.arms)
            else None
        )

    @property
    def current_right_id(self) -> Optional[int]:
        return (
            self.arms[self.current_j]
            if self.current_j is not None and 0 <= self.current_j < len(self.arms)
            else None
        )

    def select_pair(self) -> Tuple[int, int]:
        i, j = choose_pair_baseline(self.theta, self.wins, self.N)
        self.current_i, self.current_j = i, j
        return self.arms[i], self.arms[j]

    def update_with_vote(self, winner_is_left: bool):
        if self.current_i is None or self.current_j is None:
            return
        i, j = self.current_i, self.current_j
        winner_idx = i if winner_is_left else j
        loser_idx = j if winner_is_left else i
        self.wins[winner_idx, loser_idx] += 1
        self.N[i, j] += 1
        self.N[j, i] += 1
        self.theta = fit_bt(self.theta, self.wins, self.N)
        self.round_count += 1


class SessionState:
    def __init__(self, mode: ArenaMode = ArenaMode.TEXT):
        self.mode = mode
        model_ids = get_model_ids(mode)
        self.cupid = CUPIDState(model_ids, model_ids, mode)
        self.baseline = BaselineState(model_ids, mode)
        self.round_count = 0
        self.history: List[Dict] = []
        self.budget_cost: Optional[float] = None
        self.budget_rounds: Optional[int] = None
        self.routing_cost = 0.0
        self.final_cupid_model_id: Optional[int] = None
        self.final_baseline_model_id: Optional[int] = None


sessions: Dict[str, SessionState] = {}
completed_sessions: Dict[str, Dict] = {}


# ================== Pydantic Models ==================
class ModelResponse(BaseModel):
    model_id: int
    model_name: str
    text: str  # For text mode
    content: Optional[str] = None  # For image mode (imageUrl)
    cost: float
    content_type: str = "text"


class InteractRequest(BaseModel):
    session_id: Optional[str] = None
    prompt: str
    mode: ArenaMode = ArenaMode.TEXT
    previous_vote: Optional[str] = None
    cupid_vote: Optional[str] = None
    baseline_vote: Optional[str] = None
    feedback_text: Optional[str] = None
    budget_cost: Optional[float] = None
    budget_rounds: Optional[int] = None
    persona_group: Optional[str] = None
    expert_subject: Optional[str] = None
    constraints: Optional[List[Dict]] = None
    demographics: Optional[Dict] = None
    width: Optional[int] = 1024
    height: Optional[int] = 1024


class InteractResponse(BaseModel):
    session_id: str
    round: int
    mode: ArenaMode = ArenaMode.TEXT
    total_cost: float
    cupid_cost: float
    baseline_cost: float
    routing_cost: float
    cLeft: ModelResponse
    cRight: ModelResponse
    bLeft: ModelResponse
    bRight: ModelResponse
    cLeftStats: Optional[Dict] = None
    cRightStats: Optional[Dict] = None
    bLeftStats: Optional[Dict] = None
    bRightStats: Optional[Dict] = None


class ChatRequest(BaseModel):
    session_id: str
    message: str
    system: str
    mode: Optional[ArenaMode] = None
    width: Optional[int] = 1024
    height: Optional[int] = 1024


class ChatResponse(BaseModel):
    response: str
    content: Optional[str] = None
    cost: float
    content_type: str = "text"


class SaveSessionRequest(BaseModel):
    demographics: Optional[Dict] = None
    persona: Optional[str] = None
    persona_group: Optional[str] = None
    expert_subject: Optional[str] = None
    constraints: Optional[List[Dict]] = None
    budget: Optional[Dict] = None
    history: Optional[List[Dict]] = None
    evaluation: Optional[Dict] = None
    final_cost: Optional[float] = None
    final_cost_a: Optional[float] = None
    final_cost_b: Optional[float] = None
    terminated_early: Optional[bool] = None
    open_test_rounds_a: Optional[int] = None
    open_test_rounds_b: Optional[int] = None
    side_by_side_rounds: Optional[int] = None


class SaveResultsRequest(BaseModel):
    """Full study results object from frontend."""

    session_id: Optional[str] = None
    timestamp: Optional[str] = None
    demographics: Optional[Dict] = None
    persona_group: Optional[str] = None
    expert_subject: Optional[str] = None
    constraints: Optional[List[Dict]] = None
    budget: Optional[Dict] = None
    initial_preference: Optional[str] = None
    final_state: Optional[Dict] = None
    history: Optional[List[Dict]] = None
    open_testing: Optional[Dict] = None
    evaluation: Optional[Dict] = None


# ================== FastAPI App ==================
app = FastAPI(title="Model Selection Arena - Text & Image")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global runware_client

    # Connect to database
    if database:
        await database.connect()
        # Initialize tables
        await init_image_database()
        await init_study_results_table()

    # Initialize Runware SDK client
    if RUNWARE_SDK_AVAILABLE and RUNWARE_API_KEY:
        try:
            runware_client = Runware(api_key=RUNWARE_API_KEY)
            await runware_client.connect()
            print("Runware SDK connected successfully.")
        except Exception as e:
            print(f"Warning: Could not connect Runware SDK: {e}")
            runware_client = None


@app.on_event("shutdown")
async def shutdown():
    global runware_client

    if database:
        await database.disconnect()

    # Disconnect Runware client
    if runware_client:
        try:
            await runware_client.disconnect()
        except Exception as e:
            print(f"Warning: Error disconnecting Runware client: {e}")


def get_model_stats(model_id: int, mode: ArenaMode = ArenaMode.TEXT) -> Optional[Dict]:
    pool = get_model_pool(mode)
    try:
        row = pool[pool["id"] == model_id].iloc[0]
        if mode == ArenaMode.TEXT:
            return {
                "id": model_id,
                "intelligence": int(row.get("intelligence"))
                if pd.notna(row.get("intelligence"))
                else None,
                "speed": int(row.get("speed")) if pd.notna(row.get("speed")) else None,
                "reasoning": int(row.get("reasoning"))
                if pd.notna(row.get("reasoning"))
                else None,
                "input_price": float(row.get("input-price"))
                if pd.notna(row.get("input-price"))
                else None,
                "output_price": float(row.get("output-price"))
                if pd.notna(row.get("output-price"))
                else None,
                "context_window": int(row.get("context_window"))
                if pd.notna(row.get("context_window"))
                else None,
                "max_output": int(row.get("max_output"))
                if pd.notna(row.get("max_output"))
                else None,
            }
        else:
            return {"id": model_id, "model": str(row.get("model", ""))}
    except (IndexError, KeyError):
        return None


@app.get("/")
async def root():
    return {
        "message": "Model Selection Arena API - Text & Image",
        "modes": ["text", "image"],
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "llm_models": len(MODEL_IDS),
        "image_models": len(MODEL_IDS_IMAGE),
        "runware_sdk": RUNWARE_SDK_AVAILABLE and runware_client is not None,
        "database": database is not None,
    }


@app.get("/model-pool-stats")
async def get_model_pool_stats(mode: ArenaMode = Query(ArenaMode.TEXT)):
    pool = get_model_pool(mode)
    models = []
    for _, row in pool.iterrows():
        if mode == ArenaMode.TEXT:
            models.append(
                {
                    "id": int(row["id"]),
                    "intelligence": int(row.get("intelligence"))
                    if pd.notna(row.get("intelligence"))
                    else None,
                    "speed": int(row.get("speed"))
                    if pd.notna(row.get("speed"))
                    else None,
                    "reasoning": int(row.get("reasoning"))
                    if pd.notna(row.get("reasoning"))
                    else None,
                    "input_price": float(row.get("input-price"))
                    if pd.notna(row.get("input-price"))
                    else None,
                    "output_price": float(row.get("output-price"))
                    if pd.notna(row.get("output-price"))
                    else None,
                    "context_window": int(row.get("context_window"))
                    if pd.notna(row.get("context_window"))
                    else None,
                    "max_output": int(row.get("max_output"))
                    if pd.notna(row.get("max_output"))
                    else None,
                }
            )
        else:
            models.append({"id": int(row["id"]), "model": str(row.get("model", ""))})
    return {"models": models, "count": len(models), "mode": mode}


@app.get("/image-results")
async def get_image_results(
    session_id: Optional[str] = None,
    model_id: Optional[int] = None,
    algorithm: Optional[str] = None,
    limit: int = Query(100, le=1000),
    offset: int = 0,
):
    """Retrieve image generation results from the database."""
    if not database:
        raise HTTPException(status_code=503, detail="Database not available")

    query = "SELECT * FROM image_results WHERE 1=1"
    params = {}

    if session_id:
        query += " AND session_id = :session_id"
        params["session_id"] = session_id
    if model_id:
        query += " AND model_id = :model_id"
        params["model_id"] = model_id
    if algorithm:
        query += " AND algorithm = :algorithm"
        params["algorithm"] = algorithm

    query += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
    params["limit"] = limit
    params["offset"] = offset

    try:
        results = await database.fetch_all(query, params)
        return {"results": [dict(r) for r in results], "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interact", response_model=InteractResponse)
async def interact(request: InteractRequest):
    mode = request.mode
    prompt = request.prompt
    width = request.width or 1024
    height = request.height or 1024

    if request.session_id and request.session_id in sessions:
        session_id = request.session_id
        state = sessions[session_id]

        if state.mode != mode:
            raise HTTPException(status_code=400, detail=f"Session mode mismatch")

        if request.cupid_vote:
            winner_is_left = request.cupid_vote == "left"
            state.cupid.update_with_vote(winner_is_left)
            state.final_cupid_model_id = (
                state.cupid.current_left_id
                if winner_is_left
                else state.cupid.current_right_id
            )

        if request.baseline_vote:
            winner_is_left = request.baseline_vote == "left"
            state.baseline.update_with_vote(winner_is_left)
            state.final_baseline_model_id = (
                state.baseline.current_left_id
                if winner_is_left
                else state.baseline.current_right_id
            )
    else:
        session_id = f"sess_{mode.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        state = SessionState(mode)

        if request.budget_cost is not None:
            state.budget_cost = request.budget_cost
        if request.budget_rounds is not None:
            state.budget_rounds = request.budget_rounds

        sessions[session_id] = state

    cupid_left_id, cupid_right_id = state.cupid.select_pair(request.feedback_text or "")
    baseline_left_id, baseline_right_id = state.baseline.select_pair()

    # Generate responses based on mode
    if mode == ArenaMode.TEXT:
        cupid_left_result = call_openrouter(prompt, cupid_left_id)
        cupid_right_result = call_openrouter(prompt, cupid_right_id)
        baseline_left_result = call_openrouter(prompt, baseline_left_id)
        baseline_right_result = call_openrouter(prompt, baseline_right_id)

        c_left = ModelResponse(
            model_id=cupid_left_id,
            model_name=_model_display_name_from_id(cupid_left_id, mode),
            text=cupid_left_result["text"],
            cost=cupid_left_result["cost"],
            content_type="text",
        )
        c_right = ModelResponse(
            model_id=cupid_right_id,
            model_name=_model_display_name_from_id(cupid_right_id, mode),
            text=cupid_right_result["text"],
            cost=cupid_right_result["cost"],
            content_type="text",
        )
        b_left = ModelResponse(
            model_id=baseline_left_id,
            model_name=_model_display_name_from_id(baseline_left_id, mode),
            text=baseline_left_result["text"],
            cost=baseline_left_result["cost"],
            content_type="text",
        )
        b_right = ModelResponse(
            model_id=baseline_right_id,
            model_name=_model_display_name_from_id(baseline_right_id, mode),
            text=baseline_right_result["text"],
            cost=baseline_right_result["cost"],
            content_type="text",
        )
    else:
        # Image mode - use async call_runware with session tracking
        cupid_left_result = await call_runware(
            prompt,
            cupid_left_id,
            width,
            height,
            session_id=session_id,
            algorithm="cupid",
            round_number=state.round_count + 1,
            position="left",
        )
        cupid_right_result = await call_runware(
            prompt,
            cupid_right_id,
            width,
            height,
            session_id=session_id,
            algorithm="cupid",
            round_number=state.round_count + 1,
            position="right",
        )
        baseline_left_result = await call_runware(
            prompt,
            baseline_left_id,
            width,
            height,
            session_id=session_id,
            algorithm="baseline",
            round_number=state.round_count + 1,
            position="left",
        )
        baseline_right_result = await call_runware(
            prompt,
            baseline_right_id,
            width,
            height,
            session_id=session_id,
            algorithm="baseline",
            round_number=state.round_count + 1,
            position="right",
        )

        c_left = ModelResponse(
            model_id=cupid_left_id,
            model_name=_model_display_name_from_id(cupid_left_id, mode),
            text=cupid_left_result["imageUrl"],
            content=cupid_left_result["imageUrl"],
            cost=cupid_left_result["cost"],
            content_type="image",
        )
        c_right = ModelResponse(
            model_id=cupid_right_id,
            model_name=_model_display_name_from_id(cupid_right_id, mode),
            text=cupid_right_result["imageUrl"],
            content=cupid_right_result["imageUrl"],
            cost=cupid_right_result["cost"],
            content_type="image",
        )
        b_left = ModelResponse(
            model_id=baseline_left_id,
            model_name=_model_display_name_from_id(baseline_left_id, mode),
            text=baseline_left_result["imageUrl"],
            content=baseline_left_result["imageUrl"],
            cost=baseline_left_result["cost"],
            content_type="image",
        )
        b_right = ModelResponse(
            model_id=baseline_right_id,
            model_name=_model_display_name_from_id(baseline_right_id, mode),
            text=baseline_right_result["imageUrl"],
            content=baseline_right_result["imageUrl"],
            cost=baseline_right_result["cost"],
            content_type="image",
        )

    # Update costs
    state.cupid.total_cost += c_left.cost + c_right.cost
    state.baseline.total_cost += b_left.cost + b_right.cost
    state.round_count += 1

    total_cost = state.cupid.total_cost + state.baseline.total_cost

    # Build history entry
    history_entry = {
        "round": state.round_count,
        "prompt": prompt,
        "cupid_left_id": cupid_left_id,
        "cupid_right_id": cupid_right_id,
        "baseline_left_id": baseline_left_id,
        "baseline_right_id": baseline_right_id,
        "cupid_vote": request.cupid_vote,
        "baseline_vote": request.baseline_vote,
        "total_cost": total_cost,
    }

    # Add image URLs for image mode
    if mode == ArenaMode.IMAGE:
        history_entry["cupid_left_image_url"] = cupid_left_result.get("imageUrl", "")
        history_entry["cupid_right_image_url"] = cupid_right_result.get("imageUrl", "")
        history_entry["baseline_left_image_url"] = baseline_left_result.get(
            "imageUrl", ""
        )
        history_entry["baseline_right_image_url"] = baseline_right_result.get(
            "imageUrl", ""
        )
        history_entry["cupid_left_cost"] = cupid_left_result.get("cost", 0)
        history_entry["cupid_right_cost"] = cupid_right_result.get("cost", 0)
        history_entry["baseline_left_cost"] = baseline_left_result.get("cost", 0)
        history_entry["baseline_right_cost"] = baseline_right_result.get("cost", 0)

    state.history.append(history_entry)

    return InteractResponse(
        session_id=session_id,
        round=state.round_count,
        mode=mode,
        total_cost=total_cost,
        cupid_cost=state.cupid.total_cost,
        baseline_cost=state.baseline.total_cost,
        routing_cost=state.routing_cost,
        cLeft=c_left,
        cRight=c_right,
        bLeft=b_left,
        bRight=b_right,
        cLeftStats=get_model_stats(cupid_left_id, mode),
        cRightStats=get_model_stats(cupid_right_id, mode),
        bLeftStats=get_model_stats(baseline_left_id, mode),
        bRightStats=get_model_stats(baseline_right_id, mode),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_with_model(request: ChatRequest):
    session_id = request.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    state = sessions[session_id]
    mode = request.mode or state.mode
    width = request.width or 1024
    height = request.height or 1024

    if request.system == "cupid":
        model_id = state.final_cupid_model_id or (
            state.cupid.arms[0] if state.cupid.arms else get_model_ids(mode)[0]
        )
    else:
        model_id = state.final_baseline_model_id or (
            state.baseline.arms[0] if state.baseline.arms else get_model_ids(mode)[0]
        )

    if mode == ArenaMode.TEXT:
        result = call_openrouter(request.message, model_id)
        response_text = result.get("text", "")
        cost = result.get("cost", 0)
        content_type = "text"
    else:
        result = await call_runware(
            request.message,
            model_id,
            width,
            height,
            session_id=session_id,
            algorithm=request.system,
        )
        response_text = result.get("imageUrl", "")
        cost = result.get("cost", 0)
        content_type = "image"

    if request.system == "cupid":
        state.cupid.total_cost += cost
    else:
        state.baseline.total_cost += cost

    return ChatResponse(
        response=response_text,
        content=response_text,
        cost=cost,
        content_type=content_type,
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return {"message": "Session deleted", "session_id": session_id}


@app.post("/session/{session_id}/save")
async def save_session(session_id: str, request: SaveSessionRequest):
    state = sessions.get(session_id)
    session_data = {
        "session_id": session_id,
        "saved_at": datetime.now().isoformat(),
        "mode": state.mode.value if state else None,
        "demographics": request.demographics,
        "persona_group": request.persona_group,
        "expert_subject": request.expert_subject,
        "constraints": request.constraints,
        "budget": request.budget,
        "history": request.history or (state.history if state else []),
        "evaluation": request.evaluation,
        "final_cost_a": request.final_cost_a,
        "final_cost_b": request.final_cost_b,
        "side_by_side_rounds": request.side_by_side_rounds,
    }
    completed_sessions[session_id] = session_data

    output_dir = "./session_data"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/session_{session_id}.json"
    try:
        with open(filename, "w") as f:
            json.dump(session_data, f, indent=2, default=str)
    except Exception as e:
        print(f"Warning: Could not save session to file: {e}")

    return {"message": "Session saved successfully", "session_id": session_id}


@app.get("/session/{session_id}/data")
async def get_session_data(session_id: str):
    if session_id in completed_sessions:
        return completed_sessions[session_id]
    filename = f"./session_data/session_{session_id}.json"
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session data not found")


@app.post("/save-results")
async def save_results(request: SaveResultsRequest):
    """Save complete study results to database."""
    if not database:
        return {"success": False, "saved": False, "message": "Database not available"}

    # Determine mode from session_id
    mode = "text"
    if request.session_id and "image" in request.session_id:
        mode = "image"

    # Parse timestamp
    timestamp = None
    if request.timestamp:
        try:
            timestamp = datetime.fromisoformat(request.timestamp.replace("Z", "+00:00"))
        except Exception:
            timestamp = datetime.now()
    else:
        timestamp = datetime.now()

    insert_query = """
    INSERT INTO study_results 
    (session_id, timestamp, mode, demographics, persona_group, expert_subject, 
     constraints, budget, initial_preference, final_state, history, open_testing, evaluation)
    VALUES 
    (:session_id, :timestamp, :mode, :demographics, :persona_group, :expert_subject,
     :constraints, :budget, :initial_preference, :final_state, :history, :open_testing, :evaluation)
    ON CONFLICT (session_id) DO UPDATE SET
        timestamp = EXCLUDED.timestamp,
        demographics = EXCLUDED.demographics,
        persona_group = EXCLUDED.persona_group,
        expert_subject = EXCLUDED.expert_subject,
        constraints = EXCLUDED.constraints,
        budget = EXCLUDED.budget,
        initial_preference = EXCLUDED.initial_preference,
        final_state = EXCLUDED.final_state,
        history = EXCLUDED.history,
        open_testing = EXCLUDED.open_testing,
        evaluation = EXCLUDED.evaluation
    """

    try:
        await database.execute(
            insert_query,
            {
                "session_id": request.session_id,
                "timestamp": timestamp,
                "mode": mode,
                "demographics": json.dumps(request.demographics)
                if request.demographics
                else None,
                "persona_group": request.persona_group,
                "expert_subject": request.expert_subject,
                "constraints": json.dumps(request.constraints)
                if request.constraints
                else None,
                "budget": json.dumps(request.budget) if request.budget else None,
                "initial_preference": request.initial_preference,
                "final_state": json.dumps(request.final_state)
                if request.final_state
                else None,
                "history": json.dumps(request.history) if request.history else None,
                "open_testing": json.dumps(request.open_testing)
                if request.open_testing
                else None,
                "evaluation": json.dumps(request.evaluation)
                if request.evaluation
                else None,
            },
        )

        # Also save to file as backup
        output_dir = "./session_data"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/results_{request.session_id}.json"
        try:
            with open(filename, "w") as f:
                json.dump(request.dict(), f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save results to file: {e}")

        return {
            "success": True,
            "saved": True,
            "message": "Results saved successfully",
            "session_id": request.session_id,
        }
    except Exception as e:
        print(f"Error saving results to database: {e}")
        return {"success": False, "saved": False, "message": str(e)}


@app.get("/study-results")
async def get_study_results(
    session_id: Optional[str] = None,
    persona_group: Optional[str] = None,
    mode: Optional[str] = None,
    limit: int = Query(100, le=1000),
    offset: int = 0,
):
    """Retrieve study results from the database."""
    if not database:
        raise HTTPException(status_code=503, detail="Database not available")

    query = "SELECT * FROM study_results WHERE 1=1"
    params = {}

    if session_id:
        query += " AND session_id = :session_id"
        params["session_id"] = session_id
    if persona_group:
        query += " AND persona_group = :persona_group"
        params["persona_group"] = persona_group
    if mode:
        query += " AND mode = :mode"
        params["mode"] = mode

    query += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
    params["limit"] = limit
    params["offset"] = offset

    try:
        results = await database.fetch_all(query, params)
        return {"results": [dict(r) for r in results], "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

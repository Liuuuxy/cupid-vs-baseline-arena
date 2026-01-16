// Types for the User Study Frontend
// These match the FastAPI backend response/request models

export interface ModelResponse {
  model_id: number;
  model_name: string;
  text: string;
  cost: number;
}

export interface ModelStats {
  id: number;
  intelligence?: number;
  speed?: number;
  reasoning?: number;
  input_price?: number;
  output_price?: number;
  context_window?: number;
  max_output?: number;
  text_input?: boolean;
  image_input?: boolean;
  voice_input?: boolean;
  function_calling?: boolean;
  structured_output?: boolean;
  knowledge_cutoff?: string;
}

export interface InteractRequest {
  session_id?: string | null;
  prompt: string;
  previous_vote?: string | null;
  feedback_text?: string;
  cupid_vote?: string | null;
  baseline_vote?: string | null;
  budget_cost?: number;
  budget_rounds?: number;
  persona_id?: string;
  demographics?: Record<string, any>;
}

export interface InteractResponse {
  session_id: string;
  round: number;
  total_cost: number;
  cLeft: ModelResponse;
  cRight: ModelResponse;
  bLeft: ModelResponse;
  bRight: ModelResponse;
  cLeftStats?: ModelStats;
  cRightStats?: ModelStats;
  bLeftStats?: ModelStats;
  bRightStats?: ModelStats;
}

// Transformed state for the UI
export interface ArenaState {
  session_id: string;
  round: number;
  total_cost: number;
  cupid_pair: {
    left: ModelResponse;
    right: ModelResponse;
    left_stats?: ModelStats;
    right_stats?: ModelStats;
  };
  baseline_pair: {
    left: ModelResponse;
    right: ModelResponse;
    left_stats?: ModelStats;
    right_stats?: ModelStats;
  };
}

export interface SessionInfo {
  session_id: string;
  round_count: number;
  cupid_rounds: number;
  baseline_rounds: number;
  num_models: number;
  total_cost: number;
}

export interface HealthResponse {
  status: string;
  botorch_available: boolean;
  num_models: number;
  openrouter_configured: boolean;
}

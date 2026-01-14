export interface ModelResponse {
  model_id: string;
  model_name: string;
  text: string;
  cost: number;
}

export interface ArenaState {
  round: number;
  total_cost: number;
  cupid_pair: {
    left: ModelResponse;
    right: ModelResponse;
    context_id: number;
  };
  baseline_pair: {
    left: ModelResponse;
    right: ModelResponse;
  };
}

export interface InteractRequest {
  prompt: string;
  prev_round_info?: {
    cupid_winner: 'left' | 'right' | null;
    cupid_feedback: string;
    baseline_winner: 'left' | 'right' | null;
  };
}

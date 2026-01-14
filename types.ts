export interface ModelResponse {
  model_id: string;
  model_name: string;
  text: string;
  cost: number;
}

export interface CupidContextInfo {
  direction: string;
  bonus_active: boolean;
}

export interface CupidPair {
  left: ModelResponse;
  right: ModelResponse;
  context_info?: CupidContextInfo;
}

export interface BaselinePair {
  left: ModelResponse;
  right: ModelResponse;
}

export interface Rankings {
  cupid: number[];
  baseline: number[];
}

export interface ArenaState {
  round: number;
  total_cost: number;
  round_cost: number;
  direction_extracted?: string;
  cupid_pair: CupidPair;
  baseline_pair: BaselinePair;
  rankings?: Rankings;
}

export interface PrevRoundInfo {
  cupid_winner: 'left' | 'right' | null;
  cupid_feedback: string;
  baseline_winner: 'left' | 'right' | null;
}

export interface InteractRequest {
  prompt: string;
  session_id?: string;
  prev_round_info?: PrevRoundInfo;
}

export interface RankingEntry {
  rank: number;
  model_id: number;
  model_name: string;
}

export interface ExperimentSummary {
  session_id: string;
  total_rounds: number;
  total_cost: number;
  cupid_ranking: RankingEntry[];
  baseline_ranking: RankingEntry[];
  cupid_best_model: string | null;
  baseline_best_model: string | null;
}

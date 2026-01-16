import React, { useState, useCallback } from 'react';
import {
  ArrowRight, MessageSquare, Activity, Cpu, FlaskConical, User, CheckCircle,
  GraduationCap, Briefcase, FileText, Star, DollarSign, Zap, Brain,
  Image, Clock, Settings, X, Info, RefreshCw, AlertCircle, Download
} from 'lucide-react';

// --- API CONFIGURATION ---
const API_URL = 'https://cupid-vs-baseline-arena.onrender.com';

// --- TYPES ---
type Phase = 'consent' | 'calibration' | 'interaction' | 'evaluation';

interface ModelResponse {
  model_id: number;
  model_name: string;
  text: string;
  cost: number;
}

interface ModelStats {
  id: number;
  intelligence: number;
  speed: number;
  reasoning: number;
  input_price: number;
  output_price: number;
  context_window: number;
  max_output: number;
  text_input: boolean;
  image_input: boolean;
  voice_input: boolean;
  function_calling: boolean;
  structured_output: boolean;
  knowledge_cutoff: string;
}

interface ArenaState {
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

interface Persona {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  priorities: string[];
}

interface Demographics {
  age: number | '';
  education: string;
  major: string;
  familiarity: number;
}

interface BudgetConstraints {
  maxCost: number;
  maxRounds: number;
}

interface RoundHistory {
  round: number;
  prompt: string;
  cupid_left_id: number;
  cupid_right_id: number;
  cupid_vote: 'left' | 'right';
  baseline_left_id: number;
  baseline_right_id: number;
  baseline_vote: 'left' | 'right';
  feedback: string;
  timestamp: string;
}

// --- CONSTANTS ---
const DEFAULT_MAX_ROUNDS = 10;
const DEFAULT_MAX_COST = 1.0;

const PERSONAS: Persona[] = [
  {
    id: 'budget',
    title: 'The Budget Optimizer',
    description: 'You prioritize cost-efficiency above all. You want the best quality-to-price ratio and prefer models with lower input/output costs.',
    icon: <DollarSign className="text-green-500" size={32} />,
    priorities: ['input-price', 'output-price', 'cached-input']
  },
  {
    id: 'performance',
    title: 'The Performance Maximizer',
    description: 'You demand the highest intelligence and reasoning capabilities. Cost is secondary to getting the most accurate and thoughtful responses.',
    icon: <Brain className="text-purple-500" size={32} />,
    priorities: ['intelligence', 'reasoning']
  },
  {
    id: 'speed',
    title: 'The Speed Enthusiast',
    description: 'You value fast response times above all. You need quick iterations and cannot wait for slow models, even if they are smarter.',
    icon: <Zap className="text-yellow-500" size={32} />,
    priorities: ['speed']
  },
  {
    id: 'developer',
    title: 'The Developer',
    description: 'You need models that support function calling, structured outputs, and have good API features for building applications.',
    icon: <Settings className="text-blue-500" size={32} />,
    priorities: ['function-calling', 'structured-output', 'streaming']
  },
  {
    id: 'researcher',
    title: 'The Researcher',
    description: 'You work with long documents and need large context windows. You also value recent knowledge cutoffs for up-to-date information.',
    icon: <FileText className="text-orange-500" size={32} />,
    priorities: ['window-context', 'max-output', 'knowledge-cutoff']
  },
  {
    id: 'creative',
    title: 'The Multimodal Creator',
    description: 'You work with various media types and need models that can handle images, voice, and produce creative outputs.',
    icon: <Image className="text-pink-500" size={32} />,
    priorities: ['image-input', 'voice-input', 'image-output', 'audio-output']
  }
];

const EDUCATION_LEVELS = [
  "High School", "Undergraduate", "Master's Degree", "PhD / Doctorate", "Other"
];

const MAJORS = [
  "Computer Science", "Engineering", "Business", "Arts & Humanities",
  "Social Sciences", "Natural Sciences", "Mathematics", "Other"
];

// --- MAIN COMPONENT ---
const App: React.FC = () => {
  // --- STATE ---
  const [phase, setPhase] = useState<Phase>('consent');
  const [sessionId, setSessionId] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  // Calibration State
  const [demographics, setDemographics] = useState<Demographics>({
    age: '',
    education: EDUCATION_LEVELS[1],
    major: MAJORS[0],
    familiarity: 3
  });
  const [assignedPersona, setAssignedPersona] = useState<Persona | null>(null);
  const [budgetConstraints, setBudgetConstraints] = useState<BudgetConstraints>({
    maxCost: DEFAULT_MAX_COST,
    maxRounds: DEFAULT_MAX_ROUNDS
  });

  // Interaction State
  const [loading, setLoading] = useState<boolean>(false);
  const [init, setInit] = useState<boolean>(true);
  const [prompt, setPrompt] = useState<string>('');
  const [nextPrompt, setNextPrompt] = useState<string>('');
  const [arenaState, setArenaState] = useState<ArenaState | null>(null);
  const [cupidVote, setCupidVote] = useState<'left' | 'right' | null>(null);
  const [baselineVote, setBaselineVote] = useState<'left' | 'right' | null>(null);
  const [feedback, setFeedback] = useState<string>('');
  const [roundHistory, setRoundHistory] = useState<RoundHistory[]>([]);

  // Model Info Modal
  const [showModelInfo, setShowModelInfo] = useState<{ system: 'cupid' | 'baseline', side: 'left' | 'right' } | null>(null);

  // Evaluation State
  const [evalRatingA, setEvalRatingA] = useState<number>(0);
  const [evalRatingB, setEvalRatingB] = useState<number>(0);
  const [evalComment, setEvalComment] = useState<string>('');
  const [finished, setFinished] = useState<boolean>(false);

  // --- API HELPERS ---
  const fetchNextRound = useCallback(async (isFirst: boolean = false, currentPrompt?: string) => {
    setLoading(true);
    setError(null);

    const promptToUse = currentPrompt || prompt;

    try {
      const payload: any = {
        session_id: isFirst ? null : sessionId,
        prompt: promptToUse,
        previous_vote: null,
        feedback_text: feedback || '',
      };

      // Send votes if not first round
      if (!isFirst && cupidVote) {
        payload.cupid_vote = cupidVote;
      }
      if (!isFirst && baselineVote) {
        payload.baseline_vote = baselineVote;
      }

      // Send budget constraints on first round
      if (isFirst) {
        payload.budget_cost = budgetConstraints.maxCost;
        payload.budget_rounds = budgetConstraints.maxRounds;
        payload.persona_id = assignedPersona?.id;
        payload.demographics = demographics;
      }

      console.log('Sending request:', payload);

      const response = await fetch(`${API_URL}/interact`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || `API Error: ${response.status}`);
      }

      const data = await response.json();
      console.log('Received:', data);

      // Save session ID
      if (isFirst) {
        setSessionId(data.session_id);
      }

      // Record history before updating state (for non-first rounds)
      if (!isFirst && arenaState && cupidVote && baselineVote) {
        const historyEntry: RoundHistory = {
          round: arenaState.round,
          prompt: prompt,
          cupid_left_id: arenaState.cupid_pair.left.model_id,
          cupid_right_id: arenaState.cupid_pair.right.model_id,
          cupid_vote: cupidVote,
          baseline_left_id: arenaState.baseline_pair.left.model_id,
          baseline_right_id: arenaState.baseline_pair.right.model_id,
          baseline_vote: baselineVote,
          feedback: feedback,
          timestamp: new Date().toISOString()
        };
        setRoundHistory(prev => [...prev, historyEntry]);
      }

      // Transform response to arena state
      const newState: ArenaState = {
        session_id: data.session_id,
        round: data.round,
        total_cost: data.total_cost,
        cupid_pair: {
          left: data.cLeft,
          right: data.cRight,
          left_stats: data.cLeftStats,
          right_stats: data.cRightStats,
        },
        baseline_pair: {
          left: data.bLeft,
          right: data.bRight,
          left_stats: data.bLeftStats,
          right_stats: data.bRightStats,
        }
      };

      setArenaState(newState);
      setPrompt(promptToUse);
      setNextPrompt('');
      setCupidVote(null);
      setBaselineVote(null);
      setFeedback('');

    } catch (err) {
      console.error("Failed to fetch round:", err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(`Error: ${errorMessage}. Make sure the backend is running at ${API_URL}`);
    } finally {
      setLoading(false);
    }
  }, [prompt, sessionId, cupidVote, baselineVote, feedback, budgetConstraints, assignedPersona, demographics, arenaState]);

  // Save session data to backend
  const saveSessionData = useCallback(async () => {
    try {
      await fetch(`${API_URL}/session/${sessionId}/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          demographics,
          persona: assignedPersona,
          budget: budgetConstraints,
          history: roundHistory,
          evaluation: {
            rating_a: evalRatingA,
            rating_b: evalRatingB,
            comment: evalComment
          },
          final_cost: arenaState?.total_cost
        })
      });
    } catch (e) {
      console.error('Failed to save session:', e);
    }
  }, [sessionId, demographics, assignedPersona, budgetConstraints, roundHistory, evalRatingA, evalRatingB, evalComment, arenaState]);

  // --- HANDLERS ---
  const handleConsent = () => {
    const newSessionId = `sess_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setSessionId(newSessionId);

    // Randomly assign persona
    const randomPersona = PERSONAS[Math.floor(Math.random() * PERSONAS.length)];
    setAssignedPersona(randomPersona);

    setPhase('calibration');
  };

  const handleCalibrationSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!demographics.age) {
      setError("Please enter your age.");
      return;
    }
    if (budgetConstraints.maxRounds < 3) {
      setError("Minimum 3 rounds required.");
      return;
    }
    if (budgetConstraints.maxCost < 0.01) {
      setError("Minimum budget is $0.01.");
      return;
    }
    setError(null);
    setPhase('interaction');
  };

  const startSession = async () => {
    if (!prompt.trim()) {
      setError("Please enter a prompt to start.");
      return;
    }
    setError(null);
    await fetchNextRound(true, prompt);
    setInit(false);
  };

  const handleSubmitRound = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!cupidVote || !baselineVote) {
      setError("Please vote on both systems before proceeding.");
      return;
    }

    // Check if it's the last round
    const isLastRound = arenaState && (
      arenaState.round >= budgetConstraints.maxRounds ||
      arenaState.total_cost >= budgetConstraints.maxCost
    );

    if (isLastRound) {
      // Record final round before transitioning
      if (arenaState && cupidVote && baselineVote) {
        const historyEntry: RoundHistory = {
          round: arenaState.round,
          prompt: prompt,
          cupid_left_id: arenaState.cupid_pair.left.model_id,
          cupid_right_id: arenaState.cupid_pair.right.model_id,
          cupid_vote: cupidVote,
          baseline_left_id: arenaState.baseline_pair.left.model_id,
          baseline_right_id: arenaState.baseline_pair.right.model_id,
          baseline_vote: baselineVote,
          feedback: feedback,
          timestamp: new Date().toISOString()
        };
        setRoundHistory(prev => [...prev, historyEntry]);
      }
      setPhase('evaluation');
      return;
    }

    // Not last round - require next prompt
    if (!nextPrompt.trim()) {
      setError("Please enter a prompt for the next round.");
      return;
    }

    setError(null);
    await fetchNextRound(false, nextPrompt);
  };

  const handleFinalSubmit = async () => {
    await saveSessionData();
    setFinished(true);
  };

  const downloadResults = () => {
    const results = {
      session_id: sessionId,
      demographics,
      persona: assignedPersona,
      budget: budgetConstraints,
      history: roundHistory,
      evaluation: {
        rating_a: evalRatingA,
        rating_b: evalRatingB,
        comment: evalComment
      },
      final_state: arenaState
    };

    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `study_results_${sessionId}.json`;
    a.click();
  };

  // --- RENDER HELPERS ---

  const getModelStats = (system: 'cupid' | 'baseline', side: 'left' | 'right'): ModelStats | null => {
    if (!arenaState) return null;
    const pair = system === 'cupid' ? arenaState.cupid_pair : arenaState.baseline_pair;
    return side === 'left' ? pair.left_stats || null : pair.right_stats || null;
  };

  const renderModelInfoModal = () => {
    if (!showModelInfo) return null;

    const stats = getModelStats(showModelInfo.system, showModelInfo.side);
    const label = showModelInfo.side === 'left' ? 'A' : 'B';
    const systemLabel = showModelInfo.system === 'cupid' ? 'System A' : 'System B';

    return (
      <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4" onClick={() => setShowModelInfo(null)}>
        <div className="bg-white rounded-2xl max-w-md w-full p-6 relative" onClick={e => e.stopPropagation()}>
          <button
            onClick={() => setShowModelInfo(null)}
            className="absolute top-4 right-4 text-gray-400 hover:text-gray-600"
          >
            <X size={20} />
          </button>

          <h3 className="text-xl font-bold mb-4">Model {label} Stats</h3>
          <p className="text-sm text-gray-500 mb-4">{systemLabel} • {showModelInfo.side === 'left' ? 'Left' : 'Right'} Option</p>

          {stats ? (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-50 p-3 rounded-lg">
                  <div className="text-xs text-gray-500">Intelligence</div>
                  <div className="text-lg font-bold">{stats.intelligence}/5</div>
                </div>
                <div className="bg-gray-50 p-3 rounded-lg">
                  <div className="text-xs text-gray-500">Speed</div>
                  <div className="text-lg font-bold">{stats.speed}/5</div>
                </div>
                <div className="bg-gray-50 p-3 rounded-lg">
                  <div className="text-xs text-gray-500">Input Price</div>
                  <div className="text-lg font-bold">${stats.input_price}</div>
                </div>
                <div className="bg-gray-50 p-3 rounded-lg">
                  <div className="text-xs text-gray-500">Output Price</div>
                  <div className="text-lg font-bold">${stats.output_price}</div>
                </div>
              </div>

              <div className="bg-gray-50 p-3 rounded-lg">
                <div className="text-xs text-gray-500 mb-2">Capabilities</div>
                <div className="flex flex-wrap gap-2">
                  {stats.reasoning && <span className="bg-purple-100 text-purple-700 text-xs px-2 py-1 rounded">Reasoning</span>}
                  {stats.image_input && <span className="bg-blue-100 text-blue-700 text-xs px-2 py-1 rounded">Image Input</span>}
                  {stats.voice_input && <span className="bg-green-100 text-green-700 text-xs px-2 py-1 rounded">Voice Input</span>}
                  {stats.function_calling && <span className="bg-orange-100 text-orange-700 text-xs px-2 py-1 rounded">Functions</span>}
                  {stats.structured_output && <span className="bg-pink-100 text-pink-700 text-xs px-2 py-1 rounded">Structured</span>}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <span className="text-gray-500">Context Window:</span>
                  <span className="ml-2 font-mono">{stats.context_window?.toLocaleString() || 'N/A'}</span>
                </div>
                <div>
                  <span className="text-gray-500">Max Output:</span>
                  <span className="ml-2 font-mono">{stats.max_output?.toLocaleString() || 'N/A'}</span>
                </div>
              </div>

              <p className="text-xs text-gray-400 mt-4 italic">
                Note: Model name is hidden until evaluation phase.
              </p>
            </div>
          ) : (
            <div className="text-gray-500 text-center py-8">
              Model statistics not available
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderModelCard = (
    systemName: string,
    side: 'left' | 'right',
    data: ModelResponse | undefined,
    voteState: 'left' | 'right' | null,
    setVote: (v: 'left' | 'right') => void,
    colorClass: string,
    system: 'cupid' | 'baseline'
  ) => {
    if (!data) {
      return (
        <div className="animate-pulse h-64 bg-gray-100 rounded-lg flex items-center justify-center">
          <span className="text-gray-400">Loading...</span>
        </div>
      );
    }

    const isSelected = voteState === side;
    const label = side === 'left' ? 'A' : 'B';

    const borderColor = isSelected
      ? (colorClass === 'violet' ? 'border-violet-600' : 'border-emerald-600')
      : 'border-gray-200 hover:border-gray-300';

    const bgColor = isSelected
      ? (colorClass === 'violet' ? 'bg-violet-50' : 'bg-emerald-50')
      : 'bg-white';

    const buttonBg = isSelected
      ? (colorClass === 'violet' ? 'bg-violet-600' : 'bg-emerald-600')
      : 'bg-gray-100';

    return (
      <div
        className={`
          relative p-4 rounded-xl border-2 transition-all duration-200 flex flex-col md:h-full min-h-[300px]
          ${borderColor} ${bgColor}
          ${isSelected ? 'shadow-lg scale-[1.01]' : ''}
        `}
      >
        {/* Header with Model Label and Info Button */}
        <div className="flex justify-between items-center mb-2">
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowModelInfo({ system, side });
            }}
            className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700 font-mono"
          >
            Model {label} <Info size={12} />
          </button>
          <span className="text-xs text-gray-400">${data.cost.toFixed(5)}</span>
        </div>

        {/* Response Content */}
        <div
          onClick={() => setVote(side)}
          className="flex-grow cursor-pointer overflow-y-auto h-48 md:h-auto md:max-h-80 text-gray-700 whitespace-pre-wrap font-sans text-sm leading-relaxed"
        >
          {data.text || <span className="text-gray-400 italic">No response</span>}
        </div>

        {/* Selection Button */}
        <div
          onClick={() => setVote(side)}
          className={`mt-4 text-center font-bold py-2 rounded cursor-pointer transition ${buttonBg} ${isSelected ? 'text-white' : 'text-gray-400'}`}
        >
          {isSelected ? '✓ SELECTED' : 'Select This Output'}
        </div>
      </div>
    );
  };

  const renderEvalCard = (
    systemLabel: string,
    algorithmName: string,
    bestModelName: string,
    totalCost: number,
    rating: number,
    setRating: (r: number) => void,
    colorClass: string,
    icon: React.ReactNode,
    winCount: number
  ) => {
    return (
      <div className={`border-2 ${colorClass === 'violet' ? 'border-violet-200 bg-violet-50' : 'border-emerald-200 bg-emerald-50'} rounded-xl p-6 relative overflow-hidden`}>
        <div className={`absolute top-0 right-0 ${colorClass === 'violet' ? 'bg-violet-200 text-violet-800' : 'bg-emerald-200 text-emerald-800'} text-xs font-bold px-3 py-1 rounded-bl-lg`}>
          {systemLabel}
        </div>

        <div className="flex items-center gap-2 mb-4">
          <div className={`p-2 bg-white rounded-lg shadow-sm ${colorClass === 'violet' ? 'text-violet-600' : 'text-emerald-600'}`}>
            {icon}
          </div>
          <h3 className={`text-xl font-bold ${colorClass === 'violet' ? 'text-violet-900' : 'text-emerald-900'}`}>{algorithmName}</h3>
        </div>

        <div className="space-y-3 mb-6">
          <div className="bg-white p-3 rounded-lg border border-gray-100 shadow-sm flex justify-between items-center">
            <span className="text-sm text-gray-500">Final Best Model</span>
            <span className="font-mono font-bold text-gray-800">{bestModelName}</span>
          </div>
          <div className="bg-white p-3 rounded-lg border border-gray-100 shadow-sm flex justify-between items-center">
            <span className="text-sm text-gray-500">Total Cost</span>
            <span className="font-mono font-bold text-gray-800 flex items-center">
              <DollarSign size={14} />
              {totalCost.toFixed(4)}
            </span>
          </div>
          <div className="bg-white p-3 rounded-lg border border-gray-100 shadow-sm flex justify-between items-center">
            <span className="text-sm text-gray-500">Your Selections</span>
            <span className="font-mono font-bold text-gray-800">{winCount} rounds</span>
          </div>
        </div>

        <div>
          <label className={`block text-sm font-bold ${colorClass === 'violet' ? 'text-violet-900' : 'text-emerald-900'} mb-2 text-center`}>
            Rate this System (1-5)
          </label>
          <div className="flex justify-center gap-2">
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                key={star}
                onClick={() => setRating(star)}
                className={`p-1 transition-transform hover:scale-110 focus:outline-none ${star <= rating ? 'text-yellow-500' : 'text-gray-300'}`}
              >
                <Star size={32} fill={star <= rating ? "currentColor" : "none"} />
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  };

  // --- PHASE RENDERS ---

  // CONSENT PHASE
  if (phase === 'consent') {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="max-w-3xl w-full bg-white shadow-xl rounded-2xl overflow-hidden flex flex-col">
          <div className="bg-gradient-to-r from-violet-600 to-emerald-600 p-6 md:p-8 text-white text-center">
            <h1 className="text-2xl md:text-3xl font-bold">CUPID vs Baseline</h1>
            <p className="opacity-90">Research Study on LLM Selection Algorithms</p>
          </div>

          <div className="p-6 md:p-8 overflow-y-auto max-h-[60vh] prose prose-sm max-w-none text-gray-700">
            <h2 className="text-center font-bold text-xl mb-4 text-black">Consent Form</h2>
            <p>
              I am a graduate student under the direction of Professor Ransalu Senanayake in the Ira A.
              Fulton School of Engineering at Arizona State University. I am conducting a research study to
              understand how users select and evaluate Large Language Models (LLMs) to better align AI
              systems with human preferences.
            </p>
            <p>
              I am inviting your participation, which will involve completing an interactive web-based session.
              The study is divided into three phases:
            </p>
            <ol>
              <li><strong>Calibration:</strong> You will answer basic demographic questions, set your budget constraints, and be assigned a "user persona".</li>
              <li><strong>Interaction:</strong> You will use a chat interface to enter prompts and vote on responses from different AI models (names hidden).</li>
              <li><strong>Evaluation:</strong> You will see the actual model names and rate the final recommendations.</li>
            </ol>
            <p>
              You can choose not to answer any specific question or skip a turn; this will not affect your participation.
              This session will take approximately <strong>15-30 minutes</strong> depending on your budget settings.
            </p>
            <p>
              Your participation is voluntary. You must be 18 years or older, proficient in English,
              and have basic familiarity with AI chatbots. All information obtained is strictly confidential.
              Anonymous data may be used for future research.
            </p>
            <p className="text-xs text-gray-500 mt-4 border-t pt-4">
              Contacts: xinyua11@asu.edu, snguye88@asu.edu, ransalu@asu.edu<br />
              ASU Office of Research Integrity and Assurance: (480) 965-6788
            </p>
          </div>

          <div className="p-4 md:p-6 bg-gray-50 border-t flex flex-col items-center gap-4">
            <p className="text-xs md:text-sm text-gray-600 text-center max-w-xl">
              By clicking the button below, you acknowledge that you have read the information above,
              are at least 18 years of age, and agree to participate in this study.
            </p>
            <button
              onClick={handleConsent}
              className="bg-black text-white px-8 py-3 rounded-full font-bold hover:bg-gray-800 transition-transform transform hover:scale-105 flex items-center"
            >
              <CheckCircle size={20} className="mr-2" /> I AGREE
            </button>
          </div>
        </div>
      </div>
    );
  }

  // CALIBRATION PHASE
  if (phase === 'calibration') {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
        <div className="max-w-5xl w-full grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Left: Demographics Form */}
          <div className="bg-white p-6 md:p-8 rounded-2xl shadow-lg">
            <h2 className="text-xl font-bold mb-6 flex items-center">
              <User className="mr-2" /> Demographics
            </h2>

            {error && (
              <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm flex items-center gap-2">
                <AlertCircle size={16} />
                {error}
              </div>
            )}

            <form onSubmit={handleCalibrationSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Age</label>
                <input
                  type="number"
                  required
                  min="18"
                  className="w-full border rounded p-2"
                  value={demographics.age}
                  onChange={e => setDemographics({ ...demographics, age: parseInt(e.target.value) || '' })}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Education Level</label>
                <select
                  className="w-full border rounded p-2"
                  value={demographics.education}
                  onChange={e => setDemographics({ ...demographics, education: e.target.value })}
                >
                  {EDUCATION_LEVELS.map(l => <option key={l} value={l}>{l}</option>)}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Major / Field</label>
                <select
                  className="w-full border rounded p-2"
                  value={demographics.major}
                  onChange={e => setDemographics({ ...demographics, major: e.target.value })}
                >
                  {MAJORS.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  AI Familiarity (1 = Novice, 5 = Expert)
                </label>
                <div className="flex items-center gap-4">
                  <input
                    type="range"
                    min="1" max="5"
                    className="flex-grow h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                    value={demographics.familiarity}
                    onChange={e => setDemographics({ ...demographics, familiarity: parseInt(e.target.value) })}
                  />
                  <span className="font-bold text-lg w-8 text-center">{demographics.familiarity}</span>
                </div>
              </div>

              <hr className="my-4" />

              {/* Budget Constraints */}
              <h3 className="text-lg font-bold flex items-center gap-2">
                <DollarSign size={18} /> Budget Constraints
              </h3>

              <input
                type="number"
                min="0.000001"
                max="0.999999"
                step="0.01"
                className="w-full border rounded p-2"
                value={budgetConstraints.maxCost}
                onChange={(e) => {
                  // 1. Allow typing freely so you can type "0.5" without it deleting "0"
                  setBudgetConstraints({ ...budgetConstraints, maxCost: parseFloat(e.target.value) });
                }}
                onBlur={(e) => {
                  // 2. When user leaves the field, force the limit strictly > 0 and < 1
                  let val = parseFloat(e.target.value);

                  if (isNaN(val) || val <= 0) val = 0.01; // Minimum default
                  if (val >= 1) val = 0.99;               // Maximum limit

                  setBudgetConstraints({ ...budgetConstraints, maxCost: val });
                }}
              />

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Maximum Rounds
                </label>
                <input
                  type="number"
                  min="3"
                  max="50"
                  className="w-full border rounded p-2"
                  value={budgetConstraints.maxRounds}
                  onChange={e => setBudgetConstraints({ ...budgetConstraints, maxRounds: parseInt(e.target.value) || 3 })}
                />
                <p className="text-xs text-gray-500 mt-1">Number of comparison rounds (3-50)</p>
              </div>

              <div className="pt-4">
                <button
                  type="submit"
                  className="w-full bg-blue-600 text-white py-3 rounded-lg font-bold hover:bg-blue-700 transition"
                >
                  Continue to Experiment
                </button>
              </div>
            </form>
          </div>

          {/* Center: Persona Card */}
          <div className="flex flex-col justify-center">
            <div className="bg-gradient-to-br from-indigo-900 to-purple-800 text-white p-8 rounded-2xl shadow-xl transform lg:rotate-1 hover:rotate-0 transition duration-500 border-4 border-yellow-400/30">
              <div className="uppercase tracking-widest text-xs font-bold text-yellow-300 mb-2">Assigned Persona</div>
              <div className="bg-white/10 w-16 h-16 rounded-full flex items-center justify-center mb-4 backdrop-blur-sm">
                {assignedPersona?.icon}
              </div>
              <h3 className="text-2xl font-bold mb-4">{assignedPersona?.title}</h3>
              <p className="text-indigo-100 leading-relaxed">
                {assignedPersona?.description}
              </p>
              <div className="mt-6 text-xs text-indigo-300">
                <span className="font-bold">Key Priorities:</span>
                <div className="flex flex-wrap gap-1 mt-2">
                  {assignedPersona?.priorities.map(p => (
                    <span key={p} className="bg-indigo-700/50 px-2 py-0.5 rounded text-indigo-100">{p}</span>
                  ))}
                </div>
              </div>
              <div className="mt-6 text-sm text-indigo-300 italic border-t border-indigo-700 pt-4">
                Please adopt this mindset when evaluating the AI responses.
              </div>
            </div>
          </div>

          {/* Right: Instructions */}
          <div className="bg-white p-6 md:p-8 rounded-2xl shadow-lg">
            <h2 className="text-xl font-bold mb-4">How It Works</h2>
            <div className="space-y-4 text-sm text-gray-600">
              <div className="flex gap-3">
                <div className="bg-violet-100 text-violet-700 w-8 h-8 rounded-full flex items-center justify-center font-bold flex-shrink-0">1</div>
                <div>
                  <p className="font-bold text-gray-800">Enter Prompts</p>
                  <p>You'll provide questions/tasks for AI models to answer.</p>
                </div>
              </div>
              <div className="flex gap-3">
                <div className="bg-violet-100 text-violet-700 w-8 h-8 rounded-full flex items-center justify-center font-bold flex-shrink-0">2</div>
                <div>
                  <p className="font-bold text-gray-800">Compare Outputs</p>
                  <p>Two systems will each show you 2 responses. Pick the better one from each pair.</p>
                </div>
              </div>
              <div className="flex gap-3">
                <div className="bg-violet-100 text-violet-700 w-8 h-8 rounded-full flex items-center justify-center font-bold flex-shrink-0">3</div>
                <div>
                  <p className="font-bold text-gray-800">Model Names Hidden</p>
                  <p>You won't see which model produced each response until the end.</p>
                </div>
              </div>
              <div className="flex gap-3">
                <div className="bg-emerald-100 text-emerald-700 w-8 h-8 rounded-full flex items-center justify-center font-bold flex-shrink-0">4</div>
                <div>
                  <p className="font-bold text-gray-800">Final Evaluation</p>
                  <p>After all rounds, you'll see which models were selected and rate the systems.</p>
                </div>
              </div>
            </div>

            <div className="mt-6 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
              <p className="text-sm text-yellow-800">
                <strong>Tip:</strong> Consider your persona's priorities when making selections. Think about what matters most to your assigned role.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // INTERACTION PHASE
  if (phase === 'interaction') {
    // Initial prompt entry
    if (init) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50 font-sans p-4">
          <div className="max-w-md w-full p-8 bg-white shadow-xl rounded-2xl text-center relative overflow-hidden">
            <div className="mb-4 flex justify-center text-violet-500">
              {assignedPersona?.icon}
            </div>

            <h1 className="text-2xl font-bold mb-2">Ready to Start?</h1>
            <p className="text-gray-600 mb-2">
              Acting as <strong>{assignedPersona?.title}</strong>
            </p>
            <p className="text-sm text-gray-500 mb-6">
              Budget: ${budgetConstraints.maxCost} • Max {budgetConstraints.maxRounds} rounds
            </p>

            {error && (
              <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm flex items-center gap-2">
                <AlertCircle size={16} />
                {error}
              </div>
            )}

            <div className="mb-4 text-left">
              <label className="block text-sm font-medium text-gray-700 mb-1">Enter your first prompt:</label>
              <textarea
                className="w-full border rounded-lg p-3 focus:ring-2 focus:ring-violet-500 outline-none"
                rows={3}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="e.g., Write a python function to calculate fibonacci numbers..."
              />
            </div>

            <button
              onClick={startSession}
              disabled={!prompt.trim() || loading}
              className="w-full bg-black text-white py-3 rounded-lg font-bold hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <RefreshCw size={16} className="animate-spin" />
                  Starting...
                </>
              ) : (
                'Launch Arena'
              )}
            </button>
          </div>
        </div>
      );
    }

    // Main interaction view
    const isLastRound = arenaState && (
      arenaState.round >= budgetConstraints.maxRounds ||
      arenaState.total_cost >= budgetConstraints.maxCost
    );

    return (
      <div className="min-h-screen bg-gray-50 flex flex-col">
        {renderModelInfoModal()}

        {/* Header */}
        <header className="bg-white border-b sticky top-0 z-10">
          <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="bg-gradient-to-r from-violet-500 to-emerald-500 text-white px-2 py-1 rounded text-xs font-bold">STUDY</div>
              <h1 className="font-bold text-lg hidden sm:block">CUPID Arena</h1>
              <span className="bg-gray-100 text-gray-600 text-xs px-2 py-0.5 rounded border font-mono truncate max-w-[150px]">
                {assignedPersona?.title}
              </span>
            </div>

            <div className="flex items-center space-x-4 text-sm font-mono">
              <div className="flex items-center">
                <span className="text-gray-400 mr-2">ROUND</span>
                <span className="font-bold">{arenaState?.round || 0} / {budgetConstraints.maxRounds}</span>
              </div>
              <div className="flex items-center text-green-700 bg-green-50 px-3 py-1 rounded-full">
                <DollarSign size={14} className="mr-1" />
                <span className="font-bold">{arenaState?.total_cost.toFixed(4) || "0.0000"}</span>
                <span className="text-green-500 ml-1">/ {budgetConstraints.maxCost}</span>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-grow max-w-7xl mx-auto px-4 py-4 md:py-8 w-full flex flex-col gap-6 pb-40 md:pb-8">

          {/* Loading Overlay */}
          {loading && (
            <div className="fixed inset-0 bg-white/80 backdrop-blur-sm z-50 flex items-center justify-center">
              <div className="flex flex-col items-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-black mb-4"></div>
                <p className="font-mono text-sm">Generating 4 LLM Responses...</p>
              </div>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 flex items-center gap-2">
              <AlertCircle size={20} />
              {error}
            </div>
          )}

          {/* Current Prompt */}
          <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
            <span className="text-xs font-bold text-gray-400 uppercase tracking-wide">Current Prompt</span>
            <p className="text-base md:text-lg text-gray-800 font-medium mt-1">{prompt}</p>
          </div>

          {/* SYSTEM A: CUPID */}
          <section>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-violet-600 font-bold flex items-center gap-2">
                <Cpu size={18} /> SYSTEM A
              </h2>
              <span className="text-xs text-gray-400 px-2 py-0.5">Which response is better?</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:min-h-[28rem]">
              {renderModelCard('A1', 'left', arenaState?.cupid_pair.left, cupidVote, setCupidVote, 'violet', 'cupid')}
              {renderModelCard('A2', 'right', arenaState?.cupid_pair.right, cupidVote, setCupidVote, 'violet', 'cupid')}
            </div>

            {/* Feedback for CUPID */}
            <div className="mt-4 bg-violet-50 p-4 rounded-lg border border-violet-100">
              <label className="flex items-center text-sm font-bold text-violet-900 mb-2">
                <MessageSquare size={16} className="mr-2" />
                Direction Feedback (Optional):
              </label>
              <input
                type="text"
                className="w-full border border-violet-200 rounded p-2 text-sm focus:ring-2 focus:ring-violet-500 outline-none"
                placeholder="e.g., 'I want faster responses', 'Prefer cheaper models'..."
                value={feedback}
                onChange={(e) => setFeedback(e.target.value)}
              />
            </div>
          </section>

          <hr className="border-gray-200" />

          {/* SYSTEM B: BASELINE */}
          <section>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-emerald-600 font-bold flex items-center gap-2">
                <FlaskConical size={18} /> SYSTEM B
              </h2>
              <span className="text-xs text-gray-400 px-2 py-0.5">Which response is better?</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:min-h-[28rem]">
              {renderModelCard('B1', 'left', arenaState?.baseline_pair.left, baselineVote, setBaselineVote, 'emerald', 'baseline')}
              {renderModelCard('B2', 'right', arenaState?.baseline_pair.right, baselineVote, setBaselineVote, 'emerald', 'baseline')}
            </div>
          </section>

          {/* Sticky Footer */}
          <div className="fixed bottom-0 left-0 w-full md:sticky md:bottom-4 z-40 bg-white p-4 shadow-[0_-5px_20px_rgba(0,0,0,0.1)] md:shadow-2xl border-t md:border border-gray-200 md:rounded-xl">
            <div className="max-w-7xl mx-auto flex flex-col md:flex-row gap-4 items-center justify-between">
              <div className="text-sm text-gray-500 flex items-center gap-2">
                <span className={`w-3 h-3 rounded-full ${cupidVote ? 'bg-violet-500' : 'bg-gray-300'}`}></span>
                <span>A: {cupidVote || '—'}</span>
                <span className="mx-2">|</span>
                <span className={`w-3 h-3 rounded-full ${baselineVote ? 'bg-emerald-500' : 'bg-gray-300'}`}></span>
                <span>B: {baselineVote || '—'}</span>
                {isLastRound && <span className="ml-4 text-orange-600 font-bold">Final Round!</span>}
              </div>

              <div className="flex gap-4 w-full md:w-auto">
                {!isLastRound && (
                  <input
                    type="text"
                    placeholder="Enter next prompt (required)"
                    className={`flex-grow border rounded px-3 py-2 text-sm w-full md:w-64 ${!nextPrompt.trim() && cupidVote && baselineVote ? 'border-red-300 bg-red-50' : ''}`}
                    value={nextPrompt}
                    onChange={(e) => setNextPrompt(e.target.value)}
                  />
                )}
                <button
                  onClick={handleSubmitRound}
                  disabled={loading}
                  className="bg-black text-white px-6 py-2 rounded-lg font-bold flex items-center justify-center hover:bg-gray-800 disabled:opacity-50 whitespace-nowrap"
                >
                  {isLastRound ? 'Finish & Evaluate' : 'Submit'}
                  <ArrowRight size={16} className="ml-2" />
                </button>
              </div>
            </div>
          </div>
        </main>
      </div>
    );
  }

  // EVALUATION PHASE
  if (phase === 'evaluation') {
    if (finished) {
      return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
          <div className="max-w-xl w-full bg-white shadow-xl rounded-2xl p-12 text-center">
            <CheckCircle className="mx-auto text-green-500 mb-6" size={80} />
            <h1 className="text-3xl font-bold mb-2">Thank You!</h1>
            <p className="text-gray-600 mb-8">
              Your responses have been recorded. Your participation helps us improve AI model selection algorithms.
            </p>

            <button
              onClick={downloadResults}
              className="mb-4 w-full bg-blue-600 text-white py-3 rounded-lg font-bold hover:bg-blue-700 transition flex items-center justify-center gap-2"
            >
              <Download size={18} /> Download Your Results
            </button>

            <p className="text-sm text-gray-400">
              Session ID: {sessionId} <br />
              Rounds completed: {roundHistory.length}<br />
              You may close this window.
            </p>
          </div>
        </div>
      );
    }

    // Calculate stats
    const cupidWins = roundHistory.filter(r => r.cupid_vote).length;
    const baselineWins = roundHistory.filter(r => r.baseline_vote).length;

    // Get final model names (now revealed!)
    const cupidFinalModel = arenaState?.cupid_pair.left.model_name || 'Unknown';
    const baselineFinalModel = arenaState?.baseline_pair.left.model_name || 'Unknown';
    const cupidCost = (arenaState?.total_cost || 0) / 2;
    const baselineCost = (arenaState?.total_cost || 0) / 2;

    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="max-w-4xl w-full bg-white shadow-xl rounded-2xl overflow-hidden flex flex-col">
          <div className="bg-black p-6 text-white text-center">
            <h1 className="text-2xl font-bold">Evaluation Phase</h1>
            <p className="opacity-90">Model names revealed! Rate both systems.</p>
          </div>

          <div className="p-4 md:p-8 bg-gray-50 flex-grow overflow-y-auto max-h-[80vh]">
            <div className="text-center mb-8">
              <h2 className="text-xl font-bold text-gray-800">Final Results</h2>
              <p className="text-gray-600">
                You completed {roundHistory.length} rounds • Total cost: ${arenaState?.total_cost.toFixed(4)}
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-10">
              {renderEvalCard(
                "SYSTEM A",
                "CUPID (Your Algorithm)",
                cupidFinalModel,
                cupidCost,
                evalRatingA,
                setEvalRatingA,
                "violet",
                <Cpu size={24} />,
                cupidWins
              )}

              {renderEvalCard(
                "SYSTEM B",
                "Baseline (Bradley-Terry)",
                baselineFinalModel,
                baselineCost,
                evalRatingB,
                setEvalRatingB,
                "emerald",
                <FlaskConical size={24} />,
                baselineWins
              )}
            </div>

            <div className="max-w-2xl mx-auto space-y-6 pb-8">
              <div>
                <label className="block text-sm font-bold text-gray-700 mb-2">
                  Final Comments (Optional)
                </label>
                <textarea
                  className="w-full border rounded-lg p-3 h-24 focus:ring-2 focus:ring-black outline-none bg-white"
                  placeholder="Any feedback on the models or the experiment process?"
                  value={evalComment}
                  onChange={(e) => setEvalComment(e.target.value)}
                />
              </div>

              <button
                onClick={handleFinalSubmit}
                disabled={evalRatingA === 0 || evalRatingB === 0}
                className="w-full bg-black text-white py-4 rounded-lg font-bold hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center justify-center"
              >
                Submit Evaluation & Finish <ArrowRight className="ml-2" />
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return null;
};

export default App;

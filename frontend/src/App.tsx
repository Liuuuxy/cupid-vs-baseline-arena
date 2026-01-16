import React, { useState, useCallback } from 'react';
import {
  ArrowRight, MessageSquare, User, CheckCircle,
  Star, DollarSign, Zap, Brain, X, Info, RefreshCw,
  AlertCircle, Download, Send, MessageCircle, Target, Sparkles
} from 'lucide-react';

// --- API CONFIGURATION ---
const API_URL = 'https://cupid-vs-baseline-arena.onrender.com';

// --- TYPES ---
type Phase = 'consent' | 'calibration' | 'interaction' | 'openTesting' | 'evaluation';

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
  cupid_cost: number;
  baseline_cost: number;
  routing_cost: number;
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
  final_model_a?: ModelResponse;
  final_model_b?: ModelResponse;
}

interface Persona {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  sampleQuestions: string[];
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

interface OpenTestMessage {
  role: 'user' | 'assistant';
  content: string;
  system: 'A' | 'B';
}

// --- CONSTANTS ---
const DEFAULT_MAX_ROUNDS = 10;
const DEFAULT_MAX_COST = 1.0;

// Simple, non-technical personas designed for complex questions (TEXT ONLY)
const PERSONAS: Persona[] = [
  {
    id: 'everyday',
    title: 'Everyday Helper',
    description: 'You want helpful, clear answers for daily questions like planning trips, writing emails, or understanding complex topics in simple terms.',
    icon: <MessageCircle className="text-blue-500" size={32} />,
    sampleQuestions: [
      "Help me plan a 3-day trip to New York on a budget",
      "Write a polite email to my landlord about a maintenance issue",
      "Explain how credit scores work in simple terms"
    ]
  },
  {
    id: 'analytical',
    title: 'Deep Thinker',
    description: 'You need thorough, well-reasoned answers for complex questions that require careful analysis and multiple perspectives.',
    icon: <Brain className="text-purple-500" size={32} />,
    sampleQuestions: [
      "What are the pros and cons of working from home vs office?",
      "Compare different investment strategies for retirement",
      "Analyze the ethical implications of AI in healthcare"
    ]
  },
  {
    id: 'creative',
    title: 'Creative Mind',
    description: 'You value imaginative, unique responses for creative projects like writing stories, brainstorming ideas, or solving problems in new ways.',
    icon: <Sparkles className="text-pink-500" size={32} />,
    sampleQuestions: [
      "Write a short story about a time traveler who can only go back 5 minutes",
      "Give me 10 unique business ideas for a food truck",
      "Help me come up with creative names for my photography business"
    ]
  },
  {
    id: 'practical',
    title: 'Problem Solver',
    description: 'You need quick, accurate solutions for practical problems like debugging code, fixing things, or following step-by-step instructions.',
    icon: <Target className="text-orange-500" size={32} />,
    sampleQuestions: [
      "Why is my Python code throwing a KeyError and how do I fix it?",
      "What's the step-by-step process to change a car tire?",
      "How do I set up a home WiFi network for best coverage?"
    ]
  },
  {
    id: 'efficient',
    title: 'Time Saver',
    description: 'You prioritize fast, concise answers without unnecessary detail. You want to get things done quickly and efficiently.',
    icon: <Zap className="text-yellow-500" size={32} />,
    sampleQuestions: [
      "Summarize this article in 3 bullet points",
      "Give me a quick recipe for dinner with chicken and rice",
      "What are the main keyboard shortcuts for Excel?"
    ]
  }
];

const EDUCATION_LEVELS = [
  "High School", "Some College", "Bachelor's Degree", "Master's Degree", "PhD / Doctorate", "Other"
];

const MAJORS = [
  "Computer Science / IT", "Engineering", "Business / Finance", "Arts & Design",
  "Social Sciences", "Natural Sciences", "Healthcare", "Education", "Other"
];

// --- MAIN COMPONENT ---
const App: React.FC = () => {
  const [phase, setPhase] = useState<Phase>('consent');
  const [sessionId, setSessionId] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  const [demographics, setDemographics] = useState<Demographics>({
    age: '',
    education: EDUCATION_LEVELS[2],
    major: MAJORS[0],
    familiarity: 3
  });
  const [assignedPersona, setAssignedPersona] = useState<Persona | null>(null);
  const [budgetConstraints, setBudgetConstraints] = useState<BudgetConstraints>({
    maxCost: DEFAULT_MAX_COST,
    maxRounds: DEFAULT_MAX_ROUNDS
  });

  const [loading, setLoading] = useState<boolean>(false);
  const [init, setInit] = useState<boolean>(true);
  const [prompt, setPrompt] = useState<string>('');
  const [nextPrompt, setNextPrompt] = useState<string>('');
  const [arenaState, setArenaState] = useState<ArenaState | null>(null);
  const [cupidVote, setCupidVote] = useState<'left' | 'right' | null>(null);
  const [baselineVote, setBaselineVote] = useState<'left' | 'right' | null>(null);
  const [feedback, setFeedback] = useState<string>('');
  const [baselineFakeFeedback, setBaselineFakeFeedback] = useState<string>(''); // Fake - NOT stored
  const [roundHistory, setRoundHistory] = useState<RoundHistory[]>([]);

  const [showModelInfo, setShowModelInfo] = useState<{ system: 'cupid' | 'baseline', side: 'left' | 'right' } | null>(null);

  const [openTestMessages, setOpenTestMessages] = useState<OpenTestMessage[]>([]);
  const [openTestInput, setOpenTestInput] = useState<string>('');
  const [openTestSystem, setOpenTestSystem] = useState<'A' | 'B'>('A');
  const [openTestLoading, setOpenTestLoading] = useState<boolean>(false);

  const [evalRatingA, setEvalRatingA] = useState<number>(0);
  const [evalRatingB, setEvalRatingB] = useState<number>(0);
  const [evalComment, setEvalComment] = useState<string>('');
  const [finished, setFinished] = useState<boolean>(false);

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

      if (!isFirst && cupidVote) payload.cupid_vote = cupidVote;
      if (!isFirst && baselineVote) payload.baseline_vote = baselineVote;

      if (isFirst) {
        payload.budget_cost = budgetConstraints.maxCost;
        payload.budget_rounds = budgetConstraints.maxRounds;
        payload.persona_id = assignedPersona?.id;
        payload.demographics = demographics;
      }

      const response = await fetch(`${API_URL}/interact`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || `API Error: ${response.status}`);
      }

      const data = await response.json();
      if (isFirst) setSessionId(data.session_id);

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

      // SEPARATE costs: cupid_cost, baseline_cost, routing_cost
      const newState: ArenaState = {
        session_id: data.session_id,
        round: data.round,
        cupid_cost: data.cupid_cost || 0,
        baseline_cost: data.baseline_cost || 0,
        routing_cost: data.routing_cost || 0,
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
        },
        final_model_a: data.final_model_a,
        final_model_b: data.final_model_b,
      };

      setArenaState(newState);
      setPrompt(promptToUse);
      setNextPrompt('');
      setCupidVote(null);
      setBaselineVote(null);
      setFeedback('');
      setBaselineFakeFeedback('');

    } catch (err) {
      console.error("Failed to fetch round:", err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(`Error: ${errorMessage}. Please check your connection and try again.`);
    } finally {
      setLoading(false);
    }
  }, [prompt, sessionId, cupidVote, baselineVote, feedback, budgetConstraints, assignedPersona, demographics, arenaState]);

  const sendOpenTestMessage = async () => {
    if (!openTestInput.trim() || openTestLoading) return;

    const userMsg: OpenTestMessage = { role: 'user', content: openTestInput, system: openTestSystem };
    setOpenTestMessages(prev => [...prev, userMsg]);
    setOpenTestInput('');
    setOpenTestLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          message: openTestInput,
          system: openTestSystem === 'A' ? 'cupid' : 'baseline'
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const assistantMsg: OpenTestMessage = {
          role: 'assistant',
          content: data.response || 'No response received',
          system: openTestSystem
        };
        setOpenTestMessages(prev => [...prev, assistantMsg]);
      } else {
        const assistantMsg: OpenTestMessage = {
          role: 'assistant',
          content: '[Demo mode: Free chat endpoint not connected. In the full version, you would chat with the final model here.]',
          system: openTestSystem
        };
        setOpenTestMessages(prev => [...prev, assistantMsg]);
      }
    } catch (e) {
      console.error('Chat error:', e);
      const assistantMsg: OpenTestMessage = {
        role: 'assistant',
        content: '[Connection error. Please try again.]',
        system: openTestSystem
      };
      setOpenTestMessages(prev => [...prev, assistantMsg]);
    } finally {
      setOpenTestLoading(false);
    }
  };

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
          evaluation: { rating_a: evalRatingA, rating_b: evalRatingB, comment: evalComment },
          final_cost_a: (arenaState?.cupid_cost || 0) + (arenaState?.routing_cost || 0),
          final_cost_b: arenaState?.baseline_cost || 0,
        })
      });
    } catch (e) {
      console.error('Failed to save session:', e);
    }
  }, [sessionId, demographics, assignedPersona, budgetConstraints, roundHistory, evalRatingA, evalRatingB, evalComment, arenaState]);

  const handleConsent = () => {
    const newSessionId = `sess_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setSessionId(newSessionId);
    const randomPersona = PERSONAS[Math.floor(Math.random() * PERSONAS.length)];
    setAssignedPersona(randomPersona);
    setPhase('calibration');
  };

  const handleCalibrationSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!demographics.age) { setError("Please enter your age (you may skip other fields)."); return; }
    if (budgetConstraints.maxRounds < 3) { setError("Minimum 3 rounds required."); return; }
    setError(null);
    setPhase('interaction');
  };

  const startSession = async () => {
    if (!prompt.trim()) { setError("Please enter a prompt to start."); return; }
    setError(null);
    await fetchNextRound(true, prompt);
    setInit(false);
  };

  const handleSubmitRound = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!cupidVote || !baselineVote) { setError("Please select your preferred response from both systems."); return; }

    // System A cost includes routing cost
    const systemACost = (arenaState?.cupid_cost || 0) + (arenaState?.routing_cost || 0);
    const systemBCost = arenaState?.baseline_cost || 0;
    const totalCost = systemACost + systemBCost;
    const isLastRound = arenaState && (arenaState.round >= budgetConstraints.maxRounds || totalCost >= budgetConstraints.maxCost);

    if (isLastRound) {
      if (arenaState && cupidVote && baselineVote) {
        const historyEntry: RoundHistory = {
          round: arenaState.round, prompt: prompt,
          cupid_left_id: arenaState.cupid_pair.left.model_id, cupid_right_id: arenaState.cupid_pair.right.model_id, cupid_vote: cupidVote,
          baseline_left_id: arenaState.baseline_pair.left.model_id, baseline_right_id: arenaState.baseline_pair.right.model_id, baseline_vote: baselineVote,
          feedback: feedback, timestamp: new Date().toISOString()
        };
        setRoundHistory(prev => [...prev, historyEntry]);
      }
      setPhase('openTesting');
      return;
    }

    if (!nextPrompt.trim()) { setError("Please enter your next question to continue."); return; }
    setError(null);
    await fetchNextRound(false, nextPrompt);
  };

  const handleFinalSubmit = async () => { await saveSessionData(); setFinished(true); };

  const downloadResults = () => {
    const results = {
      session_id: sessionId, demographics, persona: assignedPersona, budget: budgetConstraints, history: roundHistory,
      evaluation: { rating_a: evalRatingA, rating_b: evalRatingB, comment: evalComment },
      final_costs: { system_a: (arenaState?.cupid_cost || 0) + (arenaState?.routing_cost || 0), system_b: arenaState?.baseline_cost || 0 }
    };
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `llm_matching_results_${sessionId}.json`; a.click();
  };

  const getModelStats = (system: 'cupid' | 'baseline', side: 'left' | 'right'): ModelStats | null => {
    if (!arenaState) return null;
    const pair = system === 'cupid' ? arenaState.cupid_pair : arenaState.baseline_pair;
    return side === 'left' ? pair.left_stats || null : pair.right_stats || null;
  };

  // Modal showing ALL model information from Excel (without name)
  const renderModelInfoModal = () => {
    if (!showModelInfo) return null;
    const stats = getModelStats(showModelInfo.system, showModelInfo.side);
    const label = showModelInfo.side === 'left' ? '1' : '2';
    const systemLabel = showModelInfo.system === 'cupid' ? 'System A' : 'System B';

    return (
      <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4" onClick={() => setShowModelInfo(null)}>
        <div className="bg-white rounded-2xl max-w-lg w-full p-6 relative max-h-[90vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
          <button onClick={() => setShowModelInfo(null)} className="absolute top-4 right-4 text-gray-400 hover:text-gray-600"><X size={20} /></button>
          <h3 className="text-xl font-bold mb-2">{systemLabel} - Option {label}</h3>
          <p className="text-sm text-gray-500 mb-4">Model specifications (name hidden)</p>
          {stats ? (
            <div className="space-y-4">
              {/* Performance Metrics */}
              <div className="bg-gradient-to-r from-purple-50 to-blue-50 p-4 rounded-lg">
                <h4 className="font-bold text-gray-700 mb-3">Performance Scores</h4>
                <div className="grid grid-cols-3 gap-3">
                  <div className="text-center"><div className="text-2xl font-bold text-purple-600">{stats.intelligence || '—'}</div><div className="text-xs text-gray-500">Intelligence</div></div>
                  <div className="text-center"><div className="text-2xl font-bold text-blue-600">{stats.speed || '—'}</div><div className="text-xs text-gray-500">Speed</div></div>
                  <div className="text-center"><div className="text-2xl font-bold text-indigo-600">{stats.reasoning ? 'Yes' : 'No'}</div><div className="text-xs text-gray-500">Reasoning</div></div>
                </div>
              </div>
              {/* Pricing */}
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-bold text-gray-700 mb-3">Pricing (per 1M tokens)</h4>
                <div className="grid grid-cols-2 gap-3">
                  <div><div className="text-lg font-bold text-green-600">${stats.input_price || '—'}</div><div className="text-xs text-gray-500">Input</div></div>
                  <div><div className="text-lg font-bold text-green-700">${stats.output_price || '—'}</div><div className="text-xs text-gray-500">Output</div></div>
                </div>
              </div>
              {/* Capacity */}
              <div className="bg-orange-50 p-4 rounded-lg">
                <h4 className="font-bold text-gray-700 mb-3">Capacity</h4>
                <div className="grid grid-cols-2 gap-3">
                  <div><div className="text-lg font-bold text-orange-600">{stats.context_window?.toLocaleString() || '—'}</div><div className="text-xs text-gray-500">Context Window</div></div>
                  <div><div className="text-lg font-bold text-orange-700">{stats.max_output?.toLocaleString() || '—'}</div><div className="text-xs text-gray-500">Max Output</div></div>
                </div>
              </div>
              {/* Capabilities - TEXT ONLY focus */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-bold text-gray-700 mb-3">Capabilities</h4>
                <div className="flex flex-wrap gap-2">
                  {stats.text_input && <span className="bg-blue-100 text-blue-700 text-xs px-2 py-1 rounded">Text Input ✓</span>}
                  {stats.function_calling && <span className="bg-orange-100 text-orange-700 text-xs px-2 py-1 rounded">Function Calling</span>}
                  {stats.structured_output && <span className="bg-pink-100 text-pink-700 text-xs px-2 py-1 rounded">Structured Output</span>}
                </div>
                <p className="text-xs text-gray-400 mt-2">Note: This study focuses on text-only interactions.</p>
              </div>
              {/* Knowledge Cutoff */}
              {stats.knowledge_cutoff && <div className="text-sm text-gray-600 bg-gray-50 p-3 rounded-lg"><span className="font-medium">Knowledge Cutoff:</span> {stats.knowledge_cutoff}</div>}
              {/* Feedback guidance */}
              <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                <p className="text-sm text-yellow-800 font-medium mb-1">💡 Use these specs for your feedback!</p>
                <p className="text-xs text-yellow-700">Examples: "I prefer the faster model", "Better reasoning ability", "More affordable pricing", "Larger context window"</p>
              </div>
            </div>
          ) : (<div className="text-gray-500 text-center py-8">Model statistics not available</div>)}
        </div>
      </div>
    );
  };

  const renderModelCard = (side: 'left' | 'right', data: ModelResponse | undefined, voteState: 'left' | 'right' | null, setVote: (v: 'left' | 'right') => void, colorClass: string, system: 'cupid' | 'baseline') => {
    if (!data) return <div className="animate-pulse h-64 bg-gray-100 rounded-lg flex items-center justify-center"><span className="text-gray-400">Loading...</span></div>;
    const isSelected = voteState === side;
    const label = side === 'left' ? '1' : '2';
    const borderColor = isSelected ? (colorClass === 'violet' ? 'border-violet-600' : 'border-emerald-600') : 'border-gray-200 hover:border-gray-300';
    const bgColor = isSelected ? (colorClass === 'violet' ? 'bg-violet-50' : 'bg-emerald-50') : 'bg-white';
    const buttonBg = isSelected ? (colorClass === 'violet' ? 'bg-violet-600' : 'bg-emerald-600') : 'bg-gray-100';

    return (
      <div className={`relative p-4 rounded-xl border-2 transition-all duration-200 flex flex-col md:h-full min-h-[300px] ${borderColor} ${bgColor} ${isSelected ? 'shadow-lg scale-[1.01]' : ''}`}>
        <div className="flex justify-between items-center mb-2">
          <button onClick={(e) => { e.stopPropagation(); setShowModelInfo({ system, side }); }} className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 font-medium bg-blue-50 px-2 py-1 rounded"><Info size={14} /> View Model Specs</button>
          <span className="text-xs text-gray-400">${data.cost.toFixed(5)}</span>
        </div>
        <div onClick={() => setVote(side)} className="flex-grow cursor-pointer overflow-y-auto h-48 md:h-auto md:max-h-80 text-gray-700 whitespace-pre-wrap font-sans text-sm leading-relaxed">{data.text || <span className="text-gray-400 italic">No response</span>}</div>
        <div onClick={() => setVote(side)} className={`mt-4 text-center font-bold py-3 rounded-lg cursor-pointer transition ${buttonBg} ${isSelected ? 'text-white' : 'text-gray-400 hover:text-gray-600'}`}>{isSelected ? '✓ SELECTED' : `Select Option ${label}`}</div>
      </div>
    );
  };

  // Evaluation card - NO model names shown, just System A/B
  const renderEvalCard = (systemLabel: string, totalCost: number, rating: number, setRating: (r: number) => void, colorClass: string, winCount: number) => (
    <div className={`border-2 ${colorClass === 'violet' ? 'border-violet-200 bg-violet-50' : 'border-emerald-200 bg-emerald-50'} rounded-xl p-6 relative overflow-hidden`}>
      <div className={`absolute top-0 right-0 ${colorClass === 'violet' ? 'bg-violet-200 text-violet-800' : 'bg-emerald-200 text-emerald-800'} text-xs font-bold px-3 py-1 rounded-bl-lg`}>{systemLabel}</div>
      <div className="space-y-3 mb-6 mt-4">
        <div className="bg-white p-3 rounded-lg border border-gray-100 shadow-sm flex justify-between items-center"><span className="text-sm text-gray-500">Total Cost</span><span className="font-mono font-bold text-gray-800 flex items-center"><DollarSign size={14} />{totalCost.toFixed(4)}</span></div>
        <div className="bg-white p-3 rounded-lg border border-gray-100 shadow-sm flex justify-between items-center"><span className="text-sm text-gray-500">Times You Preferred</span><span className="font-mono font-bold text-gray-800">{winCount} rounds</span></div>
      </div>
      <div>
        <label className={`block text-sm font-bold ${colorClass === 'violet' ? 'text-violet-900' : 'text-emerald-900'} mb-2 text-center`}>Overall satisfaction (1-5)</label>
        <div className="flex justify-center gap-2">{[1, 2, 3, 4, 5].map((star) => (<button key={star} onClick={() => setRating(star)} className={`p-1 transition-transform hover:scale-110 focus:outline-none ${star <= rating ? 'text-yellow-500' : 'text-gray-300'}`}><Star size={32} fill={star <= rating ? "currentColor" : "none"} /></button>))}</div>
      </div>
    </div>
  );

  // ==================== PHASE RENDERS ====================

  // CONSENT PHASE - Updated for LLM Matching
  if (phase === 'consent') {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="max-w-3xl w-full bg-white shadow-xl rounded-2xl overflow-hidden flex flex-col">
          <div className="bg-gradient-to-r from-blue-600 to-indigo-600 p-6 md:p-8 text-white text-center">
            <h1 className="text-2xl md:text-3xl font-bold">LLM Matching Study</h1>
            <p className="opacity-90">Discovering Your Preferred AI Response Style</p>
          </div>
          <div className="p-6 md:p-8 overflow-y-auto max-h-[60vh] prose prose-sm max-w-none text-gray-700">
            <h2 className="text-center font-bold text-xl mb-4 text-black">Participant Information</h2>
            <p>We are researchers at Arizona State University studying <strong>how to better match people with AI responses that fit their preferences</strong>. In this study, you will read AI-generated responses and tell us which ones you prefer.</p>
            <h3 className="text-lg font-semibold mt-4">What You'll Do</h3>
            <ol>
              <li><strong>Setup:</strong> Answer a few optional questions about yourself and see your assigned "testing mindset."</li>
              <li><strong>Compare Responses:</strong> Enter questions and see responses from two matching systems. Pick the response you prefer from each pair. <strong>Model identities are never revealed</strong> — you only see "Option 1" and "Option 2."</li>
              <li><strong>Free Testing:</strong> After the comparison rounds, chat freely with both systems' final selections.</li>
              <li><strong>Rate the Systems:</strong> Provide a final rating based on your experience with the responses — <strong>not based on any model names</strong> (which remain hidden).</li>
            </ol>
            <h3 className="text-lg font-semibold mt-4">Important Notes</h3>
            <ul>
              <li><strong>Model names hidden:</strong> You will never see which AI model produced any response.</li>
              <li><strong>Text only:</strong> This study involves text responses only (no images, audio, or video).</li>
              <li><strong>Time:</strong> Approximately 15-30 minutes</li>
            </ul>
            <h3 className="text-lg font-semibold mt-4">Privacy & Confidentiality</h3>
            <p><strong>All responses you provide and your demographic information are strictly confidential.</strong> Data will be used only for research purposes and stored securely. No personally identifying information will be published or shared. Anonymous, aggregated data may be used in academic publications.</p>
            <p className="text-xs text-gray-500 mt-4 border-t pt-4">Questions? Contact: xinyua11@asu.edu, snguye88@asu.edu, ransalu@asu.edu<br />ASU Office of Research Integrity and Assurance: (480) 965-6788</p>
          </div>
          <div className="p-4 md:p-6 bg-gray-50 border-t flex flex-col items-center gap-4">
            <p className="text-xs md:text-sm text-gray-600 text-center max-w-xl">By clicking below, you confirm you are at least 18 years old and agree to participate. You may withdraw at any time.</p>
            <button onClick={handleConsent} className="bg-blue-600 text-white px-8 py-3 rounded-full font-bold hover:bg-blue-700 transition-transform transform hover:scale-105 flex items-center"><CheckCircle size={20} className="mr-2" /> I Agree to Participate</button>
          </div>
        </div>
      </div>
    );
  }

  // CALIBRATION PHASE - Simple personas, text-only focus
  if (phase === 'calibration') {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
        <div className="max-w-5xl w-full grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: Demographics Form */}
          <div className="bg-white p-6 md:p-8 rounded-2xl shadow-lg">
            <h2 className="text-xl font-bold mb-6 flex items-center"><User className="mr-2" /> About You</h2>
            {error && <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm flex items-center gap-2"><AlertCircle size={16} />{error}</div>}
            <form onSubmit={handleCalibrationSubmit} className="space-y-4">
              <div><label className="block text-sm font-medium text-gray-700 mb-1">Age *</label><input type="number" min="18" className="w-full border rounded p-2" value={demographics.age} onChange={e => setDemographics({ ...demographics, age: parseInt(e.target.value) || '' })} placeholder="Required" /></div>
              <div><label className="block text-sm font-medium text-gray-700 mb-1">Education (optional - skip if you prefer)</label><select className="w-full border rounded p-2" value={demographics.education} onChange={e => setDemographics({ ...demographics, education: e.target.value })}><option value="">Prefer not to say</option>{EDUCATION_LEVELS.map(l => <option key={l} value={l}>{l}</option>)}</select></div>
              <div><label className="block text-sm font-medium text-gray-700 mb-1">Field (optional)</label><select className="w-full border rounded p-2" value={demographics.major} onChange={e => setDemographics({ ...demographics, major: e.target.value })}><option value="">Prefer not to say</option>{MAJORS.map(m => <option key={m} value={m}>{m}</option>)}</select></div>
              <div><label className="block text-sm font-medium text-gray-700 mb-1">How often do you use AI chatbots? (optional)</label><div className="flex items-center gap-2 text-sm text-gray-500"><span>Rarely</span><input type="range" min="1" max="5" className="flex-grow" value={demographics.familiarity} onChange={e => setDemographics({ ...demographics, familiarity: parseInt(e.target.value) })} /><span>Daily</span></div></div>
              <hr className="my-4" />
              <h3 className="text-lg font-bold flex items-center gap-2"><DollarSign size={18} /> Study Settings</h3>
              <div><label className="block text-sm font-medium text-gray-700 mb-1">Max Budget ($)</label><input type="number" step="any" min="0.10" className="w-full border rounded p-2" value={budgetConstraints.maxCost} onChange={e => setBudgetConstraints({ ...budgetConstraints, maxCost: parseFloat(e.target.value) || 0.10 })} /><p className="text-xs text-gray-500 mt-1">Total API cost limit for your session</p></div>
              <div><label className="block text-sm font-medium text-gray-700 mb-1">Number of Rounds (3-20)</label><input type="number" min="3" max="20" className="w-full border rounded p-2" value={budgetConstraints.maxRounds} onChange={e => setBudgetConstraints({ ...budgetConstraints, maxRounds: parseInt(e.target.value) || 3 })} /><p className="text-xs text-gray-500 mt-1">How many comparison rounds</p></div>
              <button type="submit" className="w-full bg-blue-600 text-white py-3 rounded-lg font-bold hover:bg-blue-700 transition mt-4">Continue</button>
            </form>
          </div>

          {/* Center: Persona Card - Simple, non-technical */}
          <div className="flex flex-col justify-center">
            <div className="bg-gradient-to-br from-blue-900 to-indigo-800 text-white p-8 rounded-2xl shadow-xl">
              <div className="uppercase tracking-widest text-xs font-bold text-blue-300 mb-2">Your Testing Mindset</div>
              <div className="bg-white/10 w-16 h-16 rounded-full flex items-center justify-center mb-4">{assignedPersona?.icon}</div>
              <h3 className="text-2xl font-bold mb-4">{assignedPersona?.title}</h3>
              <p className="text-blue-100 leading-relaxed mb-4">{assignedPersona?.description}</p>
              <div className="mt-4 pt-4 border-t border-blue-700">
                <div className="text-xs text-blue-300 font-bold mb-2">Try questions like these:</div>
                <ul className="text-sm text-blue-100 space-y-2">{assignedPersona?.sampleQuestions.map((q, i) => (<li key={i} className="flex items-start gap-2"><span className="text-blue-400">•</span><span className="italic">"{q}"</span></li>))}</ul>
              </div>
            </div>
          </div>

          {/* Right: Instructions */}
          <div className="bg-white p-6 md:p-8 rounded-2xl shadow-lg">
            <h2 className="text-xl font-bold mb-4">How It Works</h2>
            <div className="space-y-4 text-sm text-gray-600">
              <div className="flex gap-3"><div className="bg-blue-100 text-blue-700 w-8 h-8 rounded-full flex items-center justify-center font-bold flex-shrink-0">1</div><div><p className="font-bold text-gray-800">Ask Questions</p><p>Type questions that fit your assigned mindset.</p></div></div>
              <div className="flex gap-3"><div className="bg-blue-100 text-blue-700 w-8 h-8 rounded-full flex items-center justify-center font-bold flex-shrink-0">2</div><div><p className="font-bold text-gray-800">Compare Responses</p><p>Each system shows 2 options. Pick the one you prefer.</p></div></div>
              <div className="flex gap-3"><div className="bg-blue-100 text-blue-700 w-8 h-8 rounded-full flex items-center justify-center font-bold flex-shrink-0">3</div><div><p className="font-bold text-gray-800">Give Feedback</p><p>Click "View Model Specs" to see details, then tell us what you liked.</p></div></div>
              <div className="flex gap-3"><div className="bg-green-100 text-green-700 w-8 h-8 rounded-full flex items-center justify-center font-bold flex-shrink-0">4</div><div><p className="font-bold text-gray-800">Free Testing & Rating</p><p>Chat freely with final selections, then rate your experience.</p></div></div>
            </div>
            <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200"><p className="text-sm text-blue-800"><strong>Remember:</strong> Model names are never shown. Focus on the quality and style of responses.</p></div>
            <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200"><p className="text-sm text-gray-600"><strong>Text Only:</strong> This study uses text responses only — no images, audio, or video.</p></div>
          </div>
        </div>
      </div>
    );
  }

  // INTERACTION PHASE
  if (phase === 'interaction') {
    if (init) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
          <div className="max-w-lg w-full p-8 bg-white shadow-xl rounded-2xl text-center">
            <div className="mb-4 flex justify-center">{assignedPersona?.icon}</div>
            <h1 className="text-2xl font-bold mb-2">Ready to Begin</h1>
            <p className="text-gray-600 mb-2">You're the <strong>{assignedPersona?.title}</strong></p>
            <p className="text-sm text-gray-500 mb-6">Budget: ${budgetConstraints.maxCost} • {budgetConstraints.maxRounds} rounds</p>
            {error && <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm flex items-center gap-2"><AlertCircle size={16} />{error}</div>}
            <div className="mb-4 text-left">
              <label className="block text-sm font-medium text-gray-700 mb-1">Enter your first question:</label>
              <textarea className="w-full border rounded-lg p-3 focus:ring-2 focus:ring-blue-500 outline-none resize-none" rows={5} value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder={assignedPersona?.sampleQuestions[0] || "Type your question here..."} />
            </div>
            <button onClick={startSession} disabled={!prompt.trim() || loading} className="w-full bg-blue-600 text-white py-3 rounded-lg font-bold hover:bg-blue-700 disabled:opacity-50 transition flex items-center justify-center gap-2">{loading ? (<><RefreshCw size={16} className="animate-spin" />Starting...</>) : 'Start Comparing'}</button>
          </div>
        </div>
      );
    }

    // SEPARATE costs: System A = cupid + routing, System B = baseline only
    const systemACost = (arenaState?.cupid_cost || 0) + (arenaState?.routing_cost || 0);
    const systemBCost = arenaState?.baseline_cost || 0;
    const totalCost = systemACost + systemBCost;
    const isLastRound = arenaState && (arenaState.round >= budgetConstraints.maxRounds || totalCost >= budgetConstraints.maxCost);

    return (
      <div className="min-h-screen bg-gray-50 flex flex-col">
        {renderModelInfoModal()}
        <header className="bg-white border-b sticky top-0 z-10">
          <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
            <div className="flex items-center space-x-2"><div className="bg-blue-600 text-white px-2 py-1 rounded text-xs font-bold">LLM MATCHING</div><span className="bg-gray-100 text-gray-600 text-xs px-2 py-0.5 rounded border truncate max-w-[120px]">{assignedPersona?.title}</span></div>
            <div className="flex items-center space-x-3 text-sm font-mono">
              <div className="flex items-center"><span className="text-gray-400 mr-1">Round</span><span className="font-bold">{arenaState?.round || 0}/{budgetConstraints.maxRounds}</span></div>
              {/* SEPARATE costs per system */}
              <div className="hidden sm:flex items-center gap-2">
                <span className="text-violet-600 bg-violet-50 px-2 py-0.5 rounded text-xs">A: ${systemACost.toFixed(4)}</span>
                <span className="text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded text-xs">B: ${systemBCost.toFixed(4)}</span>
              </div>
            </div>
          </div>
        </header>
        <main className="flex-grow max-w-7xl mx-auto px-4 py-4 w-full flex flex-col gap-6 pb-56 md:pb-8">
          {loading && <div className="fixed inset-0 bg-white/80 backdrop-blur-sm z-50 flex items-center justify-center"><div className="flex flex-col items-center"><div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div><p className="font-mono text-sm">Getting responses...</p></div></div>}
          {error && <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 flex items-center gap-2"><AlertCircle size={20} />{error}</div>}
          <div className="bg-white p-4 rounded-lg shadow-sm border"><span className="text-xs font-bold text-gray-400 uppercase">Your Question</span><p className="text-gray-800 font-medium mt-1">{prompt}</p></div>

          {/* SYSTEM A - includes routing cost */}
          <section>
            <div className="flex items-center justify-between mb-3"><h2 className="text-violet-600 font-bold text-lg">System A</h2><span className="text-xs text-violet-500 bg-violet-50 px-2 py-1 rounded">Total Cost: ${systemACost.toFixed(4)}</span></div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">{renderModelCard('left', arenaState?.cupid_pair.left, cupidVote, setCupidVote, 'violet', 'cupid')}{renderModelCard('right', arenaState?.cupid_pair.right, cupidVote, setCupidVote, 'violet', 'cupid')}</div>
            {/* Real feedback for System A - STORED */}
            <div className="mt-4 bg-violet-50 p-4 rounded-lg border border-violet-100">
              <label className="flex items-center text-sm font-bold text-violet-900 mb-2"><MessageSquare size={16} className="mr-2" />What influenced your choice? (based on Model Specs)</label>
              <input type="text" className="w-full border border-violet-200 rounded p-2 text-sm focus:ring-2 focus:ring-violet-500 outline-none" placeholder="e.g., 'Higher intelligence score', 'Faster speed', 'Better reasoning', 'Lower price'..." value={feedback} onChange={(e) => setFeedback(e.target.value)} />
              <p className="text-xs text-violet-500 mt-2">💡 Click <strong>"View Model Specs"</strong> on each option to see detailed attributes like intelligence, speed, pricing, and capabilities.</p>
            </div>
          </section>
          <hr className="border-gray-200" />
          {/* SYSTEM B - separate cost */}
          <section>
            <div className="flex items-center justify-between mb-3"><h2 className="text-emerald-600 font-bold text-lg">System B</h2><span className="text-xs text-emerald-500 bg-emerald-50 px-2 py-1 rounded">Total Cost: ${systemBCost.toFixed(4)}</span></div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">{renderModelCard('left', arenaState?.baseline_pair.left, baselineVote, setBaselineVote, 'emerald', 'baseline')}{renderModelCard('right', arenaState?.baseline_pair.right, baselineVote, setBaselineVote, 'emerald', 'baseline')}</div>
            {/* FAKE feedback for System B - NOT stored, just for UI consistency */}
            <div className="mt-4 bg-emerald-50 p-4 rounded-lg border border-emerald-100">
              <label className="flex items-center text-sm font-bold text-emerald-900 mb-2"><MessageSquare size={16} className="mr-2" />What influenced your choice? (based on Model Specs, optional)</label>
              <input type="text" className="w-full border border-emerald-200 rounded p-2 text-sm focus:ring-2 focus:ring-emerald-500 outline-none" placeholder="e.g., 'Higher intelligence score', 'Faster speed', 'Better reasoning', 'Lower price'..." value={baselineFakeFeedback} onChange={(e) => setBaselineFakeFeedback(e.target.value)} />
              <p className="text-xs text-violet-500 mt-2">💡 Click <strong>"View Model Specs"</strong> on each option to see detailed attributes like intelligence, speed, pricing, and capabilities.</p>
            </div>
          </section>
          {/* Footer with LARGER next prompt input */}
          <div className="fixed bottom-0 left-0 w-full md:sticky md:bottom-4 z-40 bg-white p-4 shadow-lg border-t md:border md:rounded-xl">
            <div className="max-w-7xl mx-auto flex flex-col gap-4">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2"><span className={`w-3 h-3 rounded-full ${cupidVote ? 'bg-violet-500' : 'bg-gray-300'}`}></span><span>A: {cupidVote ? `Option ${cupidVote === 'left' ? '1' : '2'}` : '—'}</span><span className="mx-2">|</span><span className={`w-3 h-3 rounded-full ${baselineVote ? 'bg-emerald-500' : 'bg-gray-300'}`}></span><span>B: {baselineVote ? `Option ${baselineVote === 'left' ? '1' : '2'}` : '—'}</span></div>
                {isLastRound && <span className="text-orange-600 font-bold">Final Round!</span>}
              </div>
              {/* LARGER multi-line prompt input */}
              {!isLastRound && (
                <textarea
                  placeholder="Enter your next question here (required to continue)...

You can write longer, more detailed questions across multiple lines."
                  className={`w-full border rounded-lg px-3 py-3 text-sm resize-none ${!nextPrompt.trim() && cupidVote && baselineVote ? 'border-red-300 bg-red-50' : ''}`}
                  rows={4}
                  value={nextPrompt}
                  onChange={(e) => setNextPrompt(e.target.value)}
                />
              )}
              <button onClick={handleSubmitRound} disabled={loading} className="w-full md:w-auto md:self-end bg-blue-600 text-white px-8 py-3 rounded-lg font-bold hover:bg-blue-700 disabled:opacity-50 transition">{isLastRound ? 'Continue to Free Testing →' : 'Submit & Next →'}</button>
            </div>
          </div>
        </main>
      </div>
    );
  }

  // OPEN TESTING PHASE - Chat with final converged models
  if (phase === 'openTesting') {
    return (
      <div className="min-h-screen bg-gray-50 flex flex-col">
        <header className="bg-white border-b p-4"><div className="max-w-4xl mx-auto flex items-center justify-between"><div><h1 className="text-xl font-bold">Free Testing Phase</h1><p className="text-sm text-gray-500">Chat with both final models as much as you'd like</p></div><button onClick={() => setPhase('evaluation')} className="bg-blue-600 text-white px-6 py-2 rounded-lg font-bold hover:bg-blue-700">I'm Done Testing → Rate Systems</button></div></header>
        <main className="flex-grow max-w-4xl mx-auto w-full p-4 flex flex-col">
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4"><p className="text-sm text-yellow-800"><strong>Take your time!</strong> Test both systems with any questions you like. When you're satisfied, click "I'm Done" above to provide your final ratings.</p></div>
          <div className="flex gap-2 mb-4">
            <button onClick={() => setOpenTestSystem('A')} className={`flex-1 py-3 rounded-lg font-bold transition ${openTestSystem === 'A' ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}>Chat with System A</button>
            <button onClick={() => setOpenTestSystem('B')} className={`flex-1 py-3 rounded-lg font-bold transition ${openTestSystem === 'B' ? 'bg-emerald-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}>Chat with System B</button>
          </div>
          <div className="flex-grow bg-white rounded-xl border overflow-hidden flex flex-col min-h-[400px]">
            <div className="flex-grow overflow-y-auto p-4 space-y-4">
              {openTestMessages.filter(m => m.system === openTestSystem).length === 0 && <div className="text-center text-gray-400 py-12"><p className="text-lg mb-2">Start chatting with System {openTestSystem}</p><p className="text-sm">Ask any questions to test its responses</p></div>}
              {openTestMessages.filter(m => m.system === openTestSystem).map((msg, i) => (<div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}><div className={`max-w-[80%] p-3 rounded-lg whitespace-pre-wrap ${msg.role === 'user' ? (openTestSystem === 'A' ? 'bg-violet-600 text-white' : 'bg-emerald-600 text-white') : 'bg-gray-100 text-gray-800'}`}>{msg.content}</div></div>))}
              {openTestLoading && <div className="flex justify-start"><div className="bg-gray-100 p-3 rounded-lg"><RefreshCw size={16} className="animate-spin" /></div></div>}
            </div>
            <div className="border-t p-4 flex gap-2"><input type="text" className="flex-grow border rounded-lg px-4 py-2" placeholder={`Ask System ${openTestSystem} anything...`} value={openTestInput} onChange={(e) => setOpenTestInput(e.target.value)} onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && sendOpenTestMessage()} /><button onClick={sendOpenTestMessage} disabled={openTestLoading || !openTestInput.trim()} className={`px-4 py-2 rounded-lg font-bold ${openTestSystem === 'A' ? 'bg-violet-600' : 'bg-emerald-600'} text-white disabled:opacity-50`}><Send size={18} /></button></div>
          </div>
        </main>
      </div>
    );
  }

  // EVALUATION PHASE - NO model names, just System A/B
  if (phase === 'evaluation') {
    if (finished) return (<div className="min-h-screen bg-gray-50 flex items-center justify-center p-4"><div className="max-w-xl w-full bg-white shadow-xl rounded-2xl p-12 text-center"><CheckCircle className="mx-auto text-green-500 mb-6" size={80} /><h1 className="text-3xl font-bold mb-2">Thank You!</h1><p className="text-gray-600 mb-8">Your feedback helps us improve AI matching systems.</p><button onClick={downloadResults} className="mb-4 w-full bg-blue-600 text-white py-3 rounded-lg font-bold hover:bg-blue-700 flex items-center justify-center gap-2"><Download size={18} /> Download Your Data</button><p className="text-sm text-gray-400">Session: {sessionId}<br />You may close this window.</p></div></div>);

    const cupidWins = roundHistory.filter(r => r.cupid_vote).length;
    const baselineWins = roundHistory.filter(r => r.baseline_vote).length;
    const systemACost = (arenaState?.cupid_cost || 0) + (arenaState?.routing_cost || 0);
    const systemBCost = arenaState?.baseline_cost || 0;

    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="max-w-4xl w-full bg-white shadow-xl rounded-2xl overflow-hidden">
          <div className="bg-blue-600 p-6 text-white text-center"><h1 className="text-2xl font-bold">Final Evaluation</h1><p className="opacity-90">Rate both systems based on your experience with the responses</p></div>
          <div className="p-4 md:p-8 bg-gray-50">
            <div className="text-center mb-8"><p className="text-gray-600">You completed {roundHistory.length} comparison rounds</p><p className="text-xs text-gray-400 mt-2">(Model identities remain hidden — rate based on response quality only)</p></div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-10">{renderEvalCard("System A", systemACost, evalRatingA, setEvalRatingA, "violet", cupidWins)}{renderEvalCard("System B", systemBCost, evalRatingB, setEvalRatingB, "emerald", baselineWins)}</div>
            <div className="max-w-2xl mx-auto space-y-6">
              <div><label className="block text-sm font-bold text-gray-700 mb-2">Any final thoughts? (optional)</label><textarea className="w-full border rounded-lg p-3 h-24 bg-white" placeholder="What worked well? What could be improved? Any other feedback?" value={evalComment} onChange={(e) => setEvalComment(e.target.value)} /></div>
              <button onClick={handleFinalSubmit} disabled={evalRatingA === 0 || evalRatingB === 0} className="w-full bg-blue-600 text-white py-4 rounded-lg font-bold hover:bg-blue-700 disabled:opacity-50 transition flex items-center justify-center">Submit & Finish <ArrowRight className="ml-2" size={18} /></button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return null;
};

export default App;

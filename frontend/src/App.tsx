import React, { useState, useCallback } from 'react';
import {
  ArrowRight, MessageSquare, User, CheckCircle,
  Star, DollarSign, Zap, Brain, X, Info, RefreshCw, 
  AlertCircle, Download, Send, MessageCircle, Target, Sparkles,
  BookOpen, Heart, ThumbsUp, Settings, HelpCircle
} from 'lucide-react';

// --- API CONFIGURATION ---
const API_URL = 'https://cupid-vs-baseline-arena.onrender.com';

// --- TYPES ---
type Phase = 'consent' | 'calibration' | 'interaction' | 'openTesting' | 'evaluation';
type PersonaGroup = 'traditional' | 'expert' | 'preference';

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

// --- RATING SCALE LABELS (5-point) ---
const RATING_LABELS = [
  { value: 1, label: "Very bad / Far from good", shortLabel: "Very Bad" },
  { value: 2, label: "Bad model", shortLabel: "Bad" },
  { value: 3, label: "Average model, could be better", shortLabel: "Average" },
  { value: 4, label: "Model does well for what it should be", shortLabel: "Good" },
  { value: 5, label: "Aligns very well / Goes beyond expectations", shortLabel: "Excellent" }
];

// --- BUDGET PROBABILITY DISTRIBUTION ---
// System assigns budget/rounds from this distribution
const BUDGET_DISTRIBUTION = [
  { maxRounds: 5, maxCost: 0.5, probability: 0.2 },
  { maxRounds: 8, maxCost: 0.8, probability: 0.3 },
  { maxRounds: 10, maxCost: 1.0, probability: 0.3 },
  { maxRounds: 15, maxCost: 1.5, probability: 0.15 },
  { maxRounds: 20, maxCost: 2.0, probability: 0.05 }
];

// Sample budget from distribution
function sampleBudget(): BudgetConstraints {
  const rand = Math.random();
  let cumulative = 0;
  for (const bucket of BUDGET_DISTRIBUTION) {
    cumulative += bucket.probability;
    if (rand <= cumulative) {
      return { maxRounds: bucket.maxRounds, maxCost: bucket.maxCost };
    }
  }
  return { maxRounds: 10, maxCost: 1.0 }; // Default fallback
}

// --- PERSONA GROUPS (3 modes) ---

// Traditional: System assigns hard constraints/persona
const TRADITIONAL_PERSONAS: Persona[] = [
  {
    id: 'traditional_analytical',
    title: 'Analytical Thinker',
    description: 'The system has assigned you a persona that values thorough analysis, logical reasoning, and well-structured responses. Focus on questions requiring deep thinking.',
    icon: <Brain className="text-purple-500" size={32} />,
    sampleQuestions: [
      "Compare the economic implications of universal basic income vs. negative income tax",
      "Analyze the trade-offs between privacy and security in digital systems",
      "What are the long-term societal impacts of remote work becoming permanent?"
    ]
  },
  {
    id: 'traditional_creative',
    title: 'Creative Explorer',
    description: 'The system has assigned you a persona that values creativity, imagination, and unique perspectives. Focus on questions that require innovative thinking.',
    icon: <Sparkles className="text-pink-500" size={32} />,
    sampleQuestions: [
      "Design a city of the future that solves current urban problems",
      "Write a compelling opening for a mystery novel set in space",
      "Propose an unconventional solution to reduce food waste globally"
    ]
  },
  {
    id: 'traditional_efficient',
    title: 'Efficiency Seeker',
    description: 'The system has assigned you a persona that values speed, conciseness, and practical utility. Focus on getting accurate answers quickly.',
    icon: <Zap className="text-yellow-500" size={32} />,
    sampleQuestions: [
      "Give me the 3 most important factors when choosing a laptop",
      "What's the fastest way to learn basic SQL?",
      "Summarize the key points of effective time management"
    ]
  }
];

// Subject Expert: User selects answers based on domain accuracy
const EXPERT_PERSONAS: Persona[] = [
  {
    id: 'expert_science',
    title: 'Science & Technology Expert',
    description: 'You have expertise in science/technology. Evaluate responses based on FACTUAL ACCURACY and technical correctness. Choose models that demonstrate deep, accurate domain knowledge.',
    icon: <BookOpen className="text-blue-500" size={32} />,
    sampleQuestions: [
      "Explain the mechanism of CRISPR-Cas9 gene editing",
      "How do transformers work in modern language models?",
      "Describe the physics behind quantum entanglement"
    ]
  },
  {
    id: 'expert_business',
    title: 'Business & Finance Expert',
    description: 'You have expertise in business/finance. Evaluate responses based on FACTUAL ACCURACY and professional correctness. Choose models that demonstrate sound business knowledge.',
    icon: <DollarSign className="text-green-500" size={32} />,
    sampleQuestions: [
      "Explain the differences between DCF and comparable company analysis",
      "What are the key metrics for evaluating a SaaS business?",
      "Describe the implications of Basel III regulations on banking"
    ]
  },
  {
    id: 'expert_medical',
    title: 'Healthcare & Medical Expert',
    description: 'You have expertise in healthcare/medicine. Evaluate responses based on MEDICAL ACCURACY and clinical correctness. Choose models that demonstrate proper medical knowledge.',
    icon: <Heart className="text-red-500" size={32} />,
    sampleQuestions: [
      "Explain the pathophysiology of Type 2 diabetes",
      "What are the current first-line treatments for hypertension?",
      "Describe the mechanism of action of SSRIs"
    ]
  }
];

// Personal Preference: Open-ended, user decides what matters
const PREFERENCE_PERSONAS: Persona[] = [
  {
    id: 'preference_open',
    title: 'Your Personal Preference',
    description: 'No constraints! Ask any questions you want and choose responses based entirely on YOUR OWN preferences. What matters to you? Helpfulness? Creativity? Accuracy? Tone? You decide!',
    icon: <ThumbsUp className="text-indigo-500" size={32} />,
    sampleQuestions: [
      "Ask anything that matters to you personally",
      "Test the models on topics you care about",
      "Choose based on whatever criteria feels right to you"
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

  // Persona group selection
  const [personaGroup, setPersonaGroup] = useState<PersonaGroup | null>(null);
  const [selectedExpertPersona, setSelectedExpertPersona] = useState<Persona | null>(null);

  const [demographics, setDemographics] = useState<Demographics>({
    age: '',
    education: EDUCATION_LEVELS[2],
    major: MAJORS[0],
    familiarity: 3
  });
  const [assignedPersona, setAssignedPersona] = useState<Persona | null>(null);
  const [budgetConstraints, setBudgetConstraints] = useState<BudgetConstraints>({
    maxCost: 1.0,
    maxRounds: 10
  });

  const [loading, setLoading] = useState<boolean>(false);
  const [init, setInit] = useState<boolean>(true);
  const [prompt, setPrompt] = useState<string>('');
  const [nextPrompt, setNextPrompt] = useState<string>('');
  const [arenaState, setArenaState] = useState<ArenaState | null>(null);
  const [cupidVote, setCupidVote] = useState<'left' | 'right' | null>(null);
  const [baselineVote, setBaselineVote] = useState<'left' | 'right' | null>(null);
  const [feedback, setFeedback] = useState<string>('');
  const [baselineFakeFeedback, setBaselineFakeFeedback] = useState<string>('');
  const [roundHistory, setRoundHistory] = useState<RoundHistory[]>([]);

  // Show routing info modal
  const [showRoutingInfo, setShowRoutingInfo] = useState<boolean>(false);
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
        payload.persona_group = personaGroup;
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
  }, [prompt, sessionId, cupidVote, baselineVote, feedback, budgetConstraints, assignedPersona, demographics, arenaState, personaGroup]);

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
          content: '[Demo mode: Free chat endpoint not connected.]', 
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
          persona_group: personaGroup,
          budget: budgetConstraints,
          history: roundHistory,
          evaluation: { rating_a: evalRatingA, rating_b: evalRatingB, comment: evalComment },
          final_cost_a: (arenaState?.cupid_cost || 0) + (arenaState?.routing_cost || 0),
          final_cost_b: arenaState?.baseline_cost || 0,
          terminated_early: roundHistory.length < budgetConstraints.maxRounds
        })
      });
    } catch (e) {
      console.error('Failed to save session:', e);
    }
  }, [sessionId, demographics, assignedPersona, budgetConstraints, roundHistory, evalRatingA, evalRatingB, evalComment, arenaState, personaGroup]);

  const handleConsent = () => {
    const newSessionId = `sess_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setSessionId(newSessionId);
    // System assigns budget from probability distribution
    const assignedBudget = sampleBudget();
    setBudgetConstraints(assignedBudget);
    setPhase('calibration');
  };

  const handlePersonaGroupSelect = (group: PersonaGroup) => {
    setPersonaGroup(group);
    
    if (group === 'traditional') {
      // Randomly assign a traditional persona
      const randomPersona = TRADITIONAL_PERSONAS[Math.floor(Math.random() * TRADITIONAL_PERSONAS.length)];
      setAssignedPersona(randomPersona);
    } else if (group === 'preference') {
      // Use the open preference persona
      setAssignedPersona(PREFERENCE_PERSONAS[0]);
    }
    // For 'expert', user will select from list
  };

  const handleExpertPersonaSelect = (persona: Persona) => {
    setSelectedExpertPersona(persona);
    setAssignedPersona(persona);
  };

  const handleCalibrationSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!demographics.age) { setError("Please enter your age."); return; }
    if (!personaGroup) { setError("Please select a testing mode."); return; }
    if (personaGroup === 'expert' && !selectedExpertPersona) { setError("Please select your area of expertise."); return; }
    setError(null);
    setPhase('interaction');
  };

  const startSession = async () => {
    if (!prompt.trim()) { setError("Please enter a prompt to start."); return; }
    setError(null);
    await fetchNextRound(true, prompt);
    setInit(false);
  };

  // Handle "I am satisfied" - early termination
  const handleSatisfied = () => {
    // Record current round if votes exist
    if (arenaState && cupidVote && baselineVote) {
      const historyEntry: RoundHistory = {
        round: arenaState.round, prompt: prompt,
        cupid_left_id: arenaState.cupid_pair.left.model_id, cupid_right_id: arenaState.cupid_pair.right.model_id, cupid_vote: cupidVote,
        baseline_left_id: arenaState.baseline_pair.left.model_id, baseline_right_id: arenaState.baseline_pair.right.model_id, baseline_vote: baselineVote,
        feedback: feedback, timestamp: new Date().toISOString()
      };
      setRoundHistory(prev => [...prev, historyEntry]);
    }
    // Skip remaining rounds, go to open testing
    setPhase('openTesting');
  };

  const handleSubmitRound = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!cupidVote || !baselineVote) { setError("Please select your preferred response from both systems."); return; }

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
      session_id: sessionId, demographics, persona: assignedPersona, persona_group: personaGroup, 
      budget: budgetConstraints, history: roundHistory,
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

  // Routing Info Modal
  const renderRoutingInfoModal = () => {
    if (!showRoutingInfo) return null;
    return (
      <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4" onClick={() => setShowRoutingInfo(false)}>
        <div className="bg-white rounded-2xl max-w-lg w-full p-6 relative" onClick={e => e.stopPropagation()}>
          <button onClick={() => setShowRoutingInfo(false)} className="absolute top-4 right-4 text-gray-400 hover:text-gray-600"><X size={20} /></button>
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2"><Settings className="text-violet-600" /> How Routing Works</h3>
          
          <div className="space-y-4 text-sm text-gray-700">
            <div className="bg-violet-50 p-4 rounded-lg border border-violet-200">
              <h4 className="font-bold text-violet-900 mb-2">System A uses "Smart Routing"</h4>
              <p>System A uses your feedback to intelligently route to models that better match your preferences. This routing has an additional cost.</p>
            </div>
            
            <div className="bg-emerald-50 p-4 rounded-lg border border-emerald-200">
              <h4 className="font-bold text-emerald-900 mb-2">System B uses "Standard Selection"</h4>
              <p>System B uses a standard algorithm without language-based routing. No additional routing cost.</p>
            </div>
            
            <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
              <h4 className="font-bold text-yellow-900 mb-2">💡 Strategy Tips</h4>
              <ul className="list-disc list-inside space-y-1 text-yellow-800">
                <li><strong>Use routing early:</strong> Provide detailed feedback in the first few rounds to help System A learn your preferences quickly.</li>
                <li><strong>Use when far from ideal:</strong> If responses are way off from what you want, detailed feedback helps the routing model correct course.</li>
                <li><strong>Skip when satisfied:</strong> Once models are close to your preference, you can provide less feedback to save on routing costs.</li>
              </ul>
            </div>
          </div>
          
          <button onClick={() => setShowRoutingInfo(false)} className="w-full mt-6 bg-blue-600 text-white py-2 rounded-lg font-bold hover:bg-blue-700">Got it!</button>
        </div>
      </div>
    );
  };

  // Model Info Modal
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
              <div className="bg-gradient-to-r from-purple-50 to-blue-50 p-4 rounded-lg">
                <h4 className="font-bold text-gray-700 mb-3">Performance</h4>
                <div className="grid grid-cols-3 gap-3">
                  <div className="text-center"><div className="text-2xl font-bold text-purple-600">{stats.intelligence || '—'}</div><div className="text-xs text-gray-500">Intelligence</div></div>
                  <div className="text-center"><div className="text-2xl font-bold text-blue-600">{stats.speed || '—'}</div><div className="text-xs text-gray-500">Speed</div></div>
                  <div className="text-center"><div className="text-2xl font-bold text-indigo-600">{stats.reasoning ? 'Yes' : 'No'}</div><div className="text-xs text-gray-500">Reasoning</div></div>
                </div>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-bold text-gray-700 mb-3">Pricing (per 1M tokens)</h4>
                <div className="grid grid-cols-2 gap-3">
                  <div><div className="text-lg font-bold text-green-600">${stats.input_price || '—'}</div><div className="text-xs text-gray-500">Input</div></div>
                  <div><div className="text-lg font-bold text-green-700">${stats.output_price || '—'}</div><div className="text-xs text-gray-500">Output</div></div>
                </div>
              </div>
              <div className="bg-orange-50 p-4 rounded-lg">
                <h4 className="font-bold text-gray-700 mb-3">Capacity</h4>
                <div className="grid grid-cols-2 gap-3">
                  <div><div className="text-lg font-bold text-orange-600">{stats.context_window?.toLocaleString() || '—'}</div><div className="text-xs text-gray-500">Context Window</div></div>
                  <div><div className="text-lg font-bold text-orange-700">{stats.max_output?.toLocaleString() || '—'}</div><div className="text-xs text-gray-500">Max Output</div></div>
                </div>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-bold text-gray-700 mb-3">Capabilities</h4>
                <div className="flex flex-wrap gap-2">
                  {stats.text_input && <span className="bg-blue-100 text-blue-700 text-xs px-2 py-1 rounded">Text ✓</span>}
                  {stats.function_calling && <span className="bg-orange-100 text-orange-700 text-xs px-2 py-1 rounded">Functions</span>}
                  {stats.structured_output && <span className="bg-pink-100 text-pink-700 text-xs px-2 py-1 rounded">Structured</span>}
                </div>
              </div>
              {stats.knowledge_cutoff && <div className="text-sm text-gray-600 bg-gray-50 p-3 rounded-lg"><span className="font-medium">Knowledge Cutoff:</span> {stats.knowledge_cutoff}</div>}
            </div>
          ) : (<div className="text-gray-500 text-center py-8">Stats not available</div>)}
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
          <button onClick={(e) => { e.stopPropagation(); setShowModelInfo({ system, side }); }} className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 font-medium bg-blue-50 px-2 py-1 rounded"><Info size={14} /> View Specs</button>
          <span className="text-xs text-gray-400">${data.cost.toFixed(5)}</span>
        </div>
        <div onClick={() => setVote(side)} className="flex-grow cursor-pointer overflow-y-auto h-48 md:h-auto md:max-h-80 text-gray-700 whitespace-pre-wrap font-sans text-sm leading-relaxed">{data.text || <span className="text-gray-400 italic">No response</span>}</div>
        <div onClick={() => setVote(side)} className={`mt-4 text-center font-bold py-3 rounded-lg cursor-pointer transition ${buttonBg} ${isSelected ? 'text-white' : 'text-gray-400 hover:text-gray-600'}`}>{isSelected ? '✓ SELECTED' : `Select Option ${label}`}</div>
      </div>
    );
  };

  // New rating card with 5-point scale and labels
  const renderEvalCard = (systemLabel: string, totalCost: number, rating: number, setRating: (r: number) => void, colorClass: string, winCount: number) => (
    <div className={`border-2 ${colorClass === 'violet' ? 'border-violet-200 bg-violet-50' : 'border-emerald-200 bg-emerald-50'} rounded-xl p-6 relative overflow-hidden`}>
      <div className={`absolute top-0 right-0 ${colorClass === 'violet' ? 'bg-violet-200 text-violet-800' : 'bg-emerald-200 text-emerald-800'} text-xs font-bold px-3 py-1 rounded-bl-lg`}>{systemLabel}</div>
      <div className="space-y-3 mb-6 mt-4">
        <div className="bg-white p-3 rounded-lg border border-gray-100 shadow-sm flex justify-between items-center"><span className="text-sm text-gray-500">Total Cost</span><span className="font-mono font-bold text-gray-800 flex items-center"><DollarSign size={14} />{totalCost.toFixed(4)}</span></div>
        <div className="bg-white p-3 rounded-lg border border-gray-100 shadow-sm flex justify-between items-center"><span className="text-sm text-gray-500">Times Preferred</span><span className="font-mono font-bold text-gray-800">{winCount} rounds</span></div>
      </div>
      <div>
        <label className={`block text-sm font-bold ${colorClass === 'violet' ? 'text-violet-900' : 'text-emerald-900'} mb-3 text-center`}>Rate this system</label>
        <div className="space-y-2">
          {RATING_LABELS.map((item) => (
            <button
              key={item.value}
              onClick={() => setRating(item.value)}
              className={`w-full p-3 rounded-lg text-left transition-all flex items-center gap-3 ${
                rating === item.value 
                  ? (colorClass === 'violet' ? 'bg-violet-600 text-white' : 'bg-emerald-600 text-white')
                  : 'bg-white border border-gray-200 hover:border-gray-300 text-gray-700'
              }`}
            >
              <span className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${
                rating === item.value ? 'bg-white/20' : 'bg-gray-100'
              }`}>{item.value}</span>
              <span className="text-sm">{item.label}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );

  // ==================== PHASE RENDERS ====================

  // CONSENT PHASE
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
            <p>We are researchers at Arizona State University studying <strong>how to better match people with AI responses that fit their preferences</strong>.</p>
            
            <h3 className="text-lg font-semibold mt-4">What You'll Do</h3>
            <ol>
              <li><strong>Choose Your Mode:</strong> Select how you want to evaluate models — by assigned criteria, domain expertise, or personal preference.</li>
              <li><strong>Compare Responses:</strong> Enter questions and pick responses you prefer. The system assigns your session length automatically.</li>
              <li><strong>End When Satisfied:</strong> You can end early anytime after the first round if you're satisfied with the results.</li>
              <li><strong>Final Testing:</strong> Chat freely with both final models, then rate them.</li>
            </ol>

            <h3 className="text-lg font-semibold mt-4">How the Systems Work</h3>
            <p><strong>System A</strong> uses "smart routing" that learns from your feedback to find better matches. This routing has an additional cost, so use it strategically — give detailed feedback early or when results are far from your preference.</p>
            <p><strong>System B</strong> uses standard selection without language-based routing.</p>

            <h3 className="text-lg font-semibold mt-4">Important Notes</h3>
            <ul>
              <li><strong>Skip anything:</strong> You may skip any question without penalty.</li>
              <li><strong>Model names hidden:</strong> You will never see which AI model produced any response.</li>
              <li><strong>Text only:</strong> This study involves text responses only.</li>
            </ul>

            <h3 className="text-lg font-semibold mt-4">Privacy & Confidentiality</h3>
            <p><strong>All responses and demographic information are strictly confidential.</strong> Data is used only for research and stored securely.</p>
            <p className="text-xs text-gray-500 mt-4 border-t pt-4">Questions? Contact: xinyua11@asu.edu, snguye88@asu.edu, ransalu@asu.edu<br />ASU IRB: (480) 965-6788</p>
          </div>
          <div className="p-4 md:p-6 bg-gray-50 border-t flex flex-col items-center gap-4">
            <p className="text-xs md:text-sm text-gray-600 text-center max-w-xl">By clicking below, you confirm you are at least 18 years old and agree to participate.</p>
            <button onClick={handleConsent} className="bg-blue-600 text-white px-8 py-3 rounded-full font-bold hover:bg-blue-700 transition-transform transform hover:scale-105 flex items-center"><CheckCircle size={20} className="mr-2" /> I Agree to Participate</button>
          </div>
        </div>
      </div>
    );
  }

  // CALIBRATION PHASE - Three persona groups
  if (phase === 'calibration') {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
        <div className="max-w-5xl w-full">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-gray-800">Setup Your Session</h1>
            <p className="text-gray-600 mt-2">Session automatically configured: <span className="font-mono bg-gray-200 px-2 py-1 rounded">{budgetConstraints.maxRounds} rounds • ${budgetConstraints.maxCost} budget</span></p>
          </div>

          {error && <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 flex items-center gap-2"><AlertCircle size={20} />{error}</div>}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left: Demographics & Mode Selection */}
            <div className="space-y-6">
              {/* Demographics */}
              <div className="bg-white p-6 rounded-2xl shadow-lg">
                <h2 className="text-xl font-bold mb-4 flex items-center"><User className="mr-2" /> About You</h2>
                <div className="space-y-4">
                  <div><label className="block text-sm font-medium text-gray-700 mb-1">Age *</label><input type="number" min="18" className="w-full border rounded p-2" value={demographics.age} onChange={e => setDemographics({ ...demographics, age: parseInt(e.target.value) || '' })} placeholder="Required" /></div>
                  <div><label className="block text-sm font-medium text-gray-700 mb-1">Education (optional)</label><select className="w-full border rounded p-2" value={demographics.education} onChange={e => setDemographics({ ...demographics, education: e.target.value })}><option value="">Prefer not to say</option>{EDUCATION_LEVELS.map(l => <option key={l} value={l}>{l}</option>)}</select></div>
                  <div><label className="block text-sm font-medium text-gray-700 mb-1">AI chatbot experience</label><div className="flex items-center gap-2 text-sm text-gray-500"><span>Rarely</span><input type="range" min="1" max="5" className="flex-grow" value={demographics.familiarity} onChange={e => setDemographics({ ...demographics, familiarity: parseInt(e.target.value) })} /><span>Daily</span></div></div>
                </div>
              </div>

              {/* Mode Selection */}
              <div className="bg-white p-6 rounded-2xl shadow-lg">
                <h2 className="text-xl font-bold mb-4">Choose Your Testing Mode *</h2>
                <div className="space-y-3">
                  <button
                    onClick={() => handlePersonaGroupSelect('traditional')}
                    className={`w-full p-4 rounded-xl border-2 text-left transition-all ${personaGroup === 'traditional' ? 'border-purple-500 bg-purple-50' : 'border-gray-200 hover:border-gray-300'}`}
                  >
                    <div className="flex items-center gap-3">
                      <Settings className={`${personaGroup === 'traditional' ? 'text-purple-600' : 'text-gray-400'}`} size={24} />
                      <div>
                        <div className="font-bold text-gray-800">Traditional Mode</div>
                        <div className="text-sm text-gray-500">System assigns you a persona with specific evaluation criteria</div>
                      </div>
                    </div>
                  </button>

                  <button
                    onClick={() => handlePersonaGroupSelect('expert')}
                    className={`w-full p-4 rounded-xl border-2 text-left transition-all ${personaGroup === 'expert' ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'}`}
                  >
                    <div className="flex items-center gap-3">
                      <BookOpen className={`${personaGroup === 'expert' ? 'text-blue-600' : 'text-gray-400'}`} size={24} />
                      <div>
                        <div className="font-bold text-gray-800">Subject Expert Mode</div>
                        <div className="text-sm text-gray-500">Evaluate based on domain accuracy — choose your field of expertise</div>
                      </div>
                    </div>
                  </button>

                  <button
                    onClick={() => handlePersonaGroupSelect('preference')}
                    className={`w-full p-4 rounded-xl border-2 text-left transition-all ${personaGroup === 'preference' ? 'border-indigo-500 bg-indigo-50' : 'border-gray-200 hover:border-gray-300'}`}
                  >
                    <div className="flex items-center gap-3">
                      <ThumbsUp className={`${personaGroup === 'preference' ? 'text-indigo-600' : 'text-gray-400'}`} size={24} />
                      <div>
                        <div className="font-bold text-gray-800">Personal Preference Mode</div>
                        <div className="text-sm text-gray-500">Open-ended — evaluate based on whatever matters to you</div>
                      </div>
                    </div>
                  </button>
                </div>

                {/* Expert sub-selection */}
                {personaGroup === 'expert' && (
                  <div className="mt-4 pt-4 border-t">
                    <label className="block text-sm font-bold text-gray-700 mb-3">Select Your Expertise Area:</label>
                    <div className="space-y-2">
                      {EXPERT_PERSONAS.map(p => (
                        <button
                          key={p.id}
                          onClick={() => handleExpertPersonaSelect(p)}
                          className={`w-full p-3 rounded-lg border text-left flex items-center gap-3 transition-all ${selectedExpertPersona?.id === p.id ? 'border-blue-500 bg-blue-100' : 'border-gray-200 hover:border-gray-300'}`}
                        >
                          {p.icon}
                          <span className="font-medium">{p.title}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Right: Persona Details & Instructions */}
            <div className="space-y-6">
              {/* Assigned Persona Display */}
              {assignedPersona && (
                <div className="bg-gradient-to-br from-blue-900 to-indigo-800 text-white p-6 rounded-2xl shadow-xl">
                  <div className="uppercase tracking-widest text-xs font-bold text-blue-300 mb-2">
                    {personaGroup === 'traditional' ? 'System-Assigned Persona' : personaGroup === 'expert' ? 'Your Expertise' : 'Your Mode'}
                  </div>
                  <div className="bg-white/10 w-14 h-14 rounded-full flex items-center justify-center mb-4">{assignedPersona.icon}</div>
                  <h3 className="text-xl font-bold mb-3">{assignedPersona.title}</h3>
                  <p className="text-blue-100 text-sm leading-relaxed mb-4">{assignedPersona.description}</p>
                  <div className="pt-4 border-t border-blue-700">
                    <div className="text-xs text-blue-300 font-bold mb-2">Example questions:</div>
                    <ul className="text-sm text-blue-100 space-y-1">
                      {assignedPersona.sampleQuestions.slice(0, 2).map((q, i) => (
                        <li key={i} className="flex items-start gap-2"><span className="text-blue-400">•</span><span className="italic text-xs">"{q}"</span></li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}

              {/* Instructions */}
              <div className="bg-white p-6 rounded-2xl shadow-lg">
                <h2 className="text-xl font-bold mb-4">How It Works</h2>
                <div className="space-y-3 text-sm text-gray-600">
                  <div className="flex gap-3"><div className="bg-blue-100 text-blue-700 w-7 h-7 rounded-full flex items-center justify-center font-bold flex-shrink-0 text-sm">1</div><div><p className="font-bold text-gray-800">Ask & Compare</p><p>Type questions, compare responses from two systems.</p></div></div>
                  <div className="flex gap-3"><div className="bg-blue-100 text-blue-700 w-7 h-7 rounded-full flex items-center justify-center font-bold flex-shrink-0 text-sm">2</div><div><p className="font-bold text-gray-800">Give Feedback</p><p>System A learns from your feedback (costs more). Use strategically!</p></div></div>
                  <div className="flex gap-3"><div className="bg-green-100 text-green-700 w-7 h-7 rounded-full flex items-center justify-center font-bold flex-shrink-0 text-sm">3</div><div><p className="font-bold text-gray-800">End When Ready</p><p>Click "I'm Satisfied" anytime after round 1 to proceed to final testing.</p></div></div>
                </div>

                <div className="mt-4 p-3 bg-yellow-50 rounded-lg border border-yellow-200">
                  <p className="text-xs text-yellow-800"><strong>💡 Routing Tip:</strong> Give detailed feedback early to help System A learn quickly, or when results are far off. Save on routing costs when you're already getting good matches.</p>
                </div>
              </div>

              <form onSubmit={handleCalibrationSubmit}>
                <button type="submit" className="w-full bg-blue-600 text-white py-4 rounded-xl font-bold hover:bg-blue-700 transition text-lg">Continue to Experiment →</button>
              </form>
            </div>
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
            <p className="text-gray-600 mb-2"><strong>{assignedPersona?.title}</strong></p>
            <p className="text-sm text-gray-500 mb-4">Up to {budgetConstraints.maxRounds} rounds • ${budgetConstraints.maxCost} budget</p>
            
            <button onClick={() => setShowRoutingInfo(true)} className="mb-4 text-sm text-blue-600 hover:text-blue-800 flex items-center justify-center gap-1 mx-auto">
              <HelpCircle size={16} /> How does routing work?
            </button>

            {error && <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm flex items-center gap-2"><AlertCircle size={16} />{error}</div>}
            <div className="mb-4 text-left">
              <label className="block text-sm font-medium text-gray-700 mb-1">Enter your first question:</label>
              <textarea className="w-full border rounded-lg p-3 focus:ring-2 focus:ring-blue-500 outline-none resize-none" rows={5} value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder={assignedPersona?.sampleQuestions[0] || "Type your question here..."} />
            </div>
            <button onClick={startSession} disabled={!prompt.trim() || loading} className="w-full bg-blue-600 text-white py-3 rounded-lg font-bold hover:bg-blue-700 disabled:opacity-50 transition flex items-center justify-center gap-2">{loading ? (<><RefreshCw size={16} className="animate-spin" />Starting...</>) : 'Start Comparing'}</button>
          </div>
          {renderRoutingInfoModal()}
        </div>
      );
    }

    const systemACost = (arenaState?.cupid_cost || 0) + (arenaState?.routing_cost || 0);
    const systemBCost = arenaState?.baseline_cost || 0;
    const totalCost = systemACost + systemBCost;
    const isLastRound = arenaState && (arenaState.round >= budgetConstraints.maxRounds || totalCost >= budgetConstraints.maxCost);
    const canEndEarly = arenaState && arenaState.round >= 1;

    return (
      <div className="min-h-screen bg-gray-50 flex flex-col">
        {renderModelInfoModal()}
        {renderRoutingInfoModal()}
        
        <header className="bg-white border-b sticky top-0 z-10">
          <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="bg-blue-600 text-white px-2 py-1 rounded text-xs font-bold">LLM MATCHING</div>
              <span className="bg-gray-100 text-gray-600 text-xs px-2 py-0.5 rounded border truncate max-w-[100px]">{assignedPersona?.title}</span>
            </div>
            <div className="flex items-center space-x-3 text-sm font-mono">
              <div className="flex items-center"><span className="text-gray-400 mr-1">Round</span><span className="font-bold">{arenaState?.round || 0}/{budgetConstraints.maxRounds}</span></div>
              <div className="hidden sm:flex items-center gap-2">
                <span className="text-violet-600 bg-violet-50 px-2 py-0.5 rounded text-xs">A: ${systemACost.toFixed(4)}</span>
                <span className="text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded text-xs">B: ${systemBCost.toFixed(4)}</span>
              </div>
              <button onClick={() => setShowRoutingInfo(true)} className="text-gray-400 hover:text-gray-600"><HelpCircle size={18} /></button>
            </div>
          </div>
        </header>

        <main className="flex-grow max-w-7xl mx-auto px-4 py-4 w-full flex flex-col gap-6 pb-56 md:pb-8">
          {loading && <div className="fixed inset-0 bg-white/80 backdrop-blur-sm z-50 flex items-center justify-center"><div className="flex flex-col items-center"><div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div><p className="font-mono text-sm">Getting responses...</p></div></div>}
          {error && <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 flex items-center gap-2"><AlertCircle size={20} />{error}</div>}
          
          <div className="bg-white p-4 rounded-lg shadow-sm border"><span className="text-xs font-bold text-gray-400 uppercase">Your Question</span><p className="text-gray-800 font-medium mt-1">{prompt}</p></div>
          
          {/* System A */}
          <section>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <h2 className="text-violet-600 font-bold text-lg">System A</h2>
                <span className="text-xs text-violet-500 bg-violet-50 px-2 py-1 rounded">Smart Routing</span>
              </div>
              <span className="text-xs text-violet-500 bg-violet-50 px-2 py-1 rounded">Cost: ${systemACost.toFixed(4)}</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">{renderModelCard('left', arenaState?.cupid_pair.left, cupidVote, setCupidVote, 'violet', 'cupid')}{renderModelCard('right', arenaState?.cupid_pair.right, cupidVote, setCupidVote, 'violet', 'cupid')}</div>
            <div className="mt-4 bg-violet-50 p-4 rounded-lg border border-violet-100">
              <label className="flex items-center text-sm font-bold text-violet-900 mb-2"><MessageSquare size={16} className="mr-2" />Feedback for Routing (helps System A learn)</label>
              <input type="text" className="w-full border border-violet-200 rounded p-2 text-sm focus:ring-2 focus:ring-violet-500 outline-none" placeholder="e.g., 'Need more detail', 'Too technical', 'Prefer concise answers'..." value={feedback} onChange={(e) => setFeedback(e.target.value)} />
              <p className="text-xs text-violet-500 mt-2">💡 Detailed feedback costs more but improves matching. Use strategically!</p>
            </div>
          </section>

          <hr className="border-gray-200" />

          {/* System B */}
          <section>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <h2 className="text-emerald-600 font-bold text-lg">System B</h2>
                <span className="text-xs text-emerald-500 bg-emerald-50 px-2 py-1 rounded">Standard</span>
              </div>
              <span className="text-xs text-emerald-500 bg-emerald-50 px-2 py-1 rounded">Cost: ${systemBCost.toFixed(4)}</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">{renderModelCard('left', arenaState?.baseline_pair.left, baselineVote, setBaselineVote, 'emerald', 'baseline')}{renderModelCard('right', arenaState?.baseline_pair.right, baselineVote, setBaselineVote, 'emerald', 'baseline')}</div>
            <div className="mt-4 bg-emerald-50 p-4 rounded-lg border border-emerald-100">
              <label className="flex items-center text-sm font-bold text-emerald-900 mb-2"><MessageSquare size={16} className="mr-2" />Notes (optional, not used for routing)</label>
              <input type="text" className="w-full border border-emerald-200 rounded p-2 text-sm focus:ring-2 focus:ring-emerald-500 outline-none" placeholder="Any thoughts..." value={baselineFakeFeedback} onChange={(e) => setBaselineFakeFeedback(e.target.value)} />
            </div>
          </section>

          {/* Footer */}
          <div className="fixed bottom-0 left-0 w-full md:sticky md:bottom-4 z-40 bg-white p-4 shadow-lg border-t md:border md:rounded-xl">
            <div className="max-w-7xl mx-auto flex flex-col gap-4">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <span className={`w-3 h-3 rounded-full ${cupidVote ? 'bg-violet-500' : 'bg-gray-300'}`}></span>
                  <span>A: {cupidVote ? `Opt ${cupidVote === 'left' ? '1' : '2'}` : '—'}</span>
                  <span className="mx-2">|</span>
                  <span className={`w-3 h-3 rounded-full ${baselineVote ? 'bg-emerald-500' : 'bg-gray-300'}`}></span>
                  <span>B: {baselineVote ? `Opt ${baselineVote === 'left' ? '1' : '2'}` : '—'}</span>
                </div>
                <div className="flex items-center gap-2">
                  {isLastRound && <span className="text-orange-600 font-bold">Final Round!</span>}
                  {canEndEarly && !isLastRound && (
                    <button 
                      onClick={handleSatisfied}
                      className="bg-green-100 text-green-700 px-4 py-2 rounded-lg font-medium hover:bg-green-200 transition text-sm"
                    >
                      ✓ I'm Satisfied — End Early
                    </button>
                  )}
                </div>
              </div>
              
              {!isLastRound && (
                <textarea
                  placeholder="Enter your next question (required to continue)..."
                  className={`w-full border rounded-lg px-3 py-3 text-sm resize-none ${!nextPrompt.trim() && cupidVote && baselineVote ? 'border-red-300 bg-red-50' : ''}`}
                  rows={4}
                  value={nextPrompt}
                  onChange={(e) => setNextPrompt(e.target.value)}
                />
              )}
              
              <button onClick={handleSubmitRound} disabled={loading} className="w-full md:w-auto md:self-end bg-blue-600 text-white px-8 py-3 rounded-lg font-bold hover:bg-blue-700 disabled:opacity-50 transition">
                {isLastRound ? 'Continue to Free Testing →' : 'Submit & Next →'}
              </button>
            </div>
          </div>
        </main>
      </div>
    );
  }

  // OPEN TESTING PHASE
  if (phase === 'openTesting') {
    return (
      <div className="min-h-screen bg-gray-50 flex flex-col">
        <header className="bg-white border-b p-4">
          <div className="max-w-4xl mx-auto flex items-center justify-between">
            <div><h1 className="text-xl font-bold">Free Testing Phase</h1><p className="text-sm text-gray-500">Chat with both final models as much as you'd like</p></div>
            <button onClick={() => setPhase('evaluation')} className="bg-blue-600 text-white px-6 py-2 rounded-lg font-bold hover:bg-blue-700">I'm Done → Rate Systems</button>
          </div>
        </header>
        <main className="flex-grow max-w-4xl mx-auto w-full p-4 flex flex-col">
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4"><p className="text-sm text-yellow-800"><strong>Take your time!</strong> Test both systems freely. Click "I'm Done" when ready to provide your final ratings.</p></div>
          <div className="flex gap-2 mb-4">
            <button onClick={() => setOpenTestSystem('A')} className={`flex-1 py-3 rounded-lg font-bold transition ${openTestSystem === 'A' ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}>System A</button>
            <button onClick={() => setOpenTestSystem('B')} className={`flex-1 py-3 rounded-lg font-bold transition ${openTestSystem === 'B' ? 'bg-emerald-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}>System B</button>
          </div>
          <div className="flex-grow bg-white rounded-xl border overflow-hidden flex flex-col min-h-[400px]">
            <div className="flex-grow overflow-y-auto p-4 space-y-4">
              {openTestMessages.filter(m => m.system === openTestSystem).length === 0 && <div className="text-center text-gray-400 py-12"><p className="text-lg mb-2">Chat with System {openTestSystem}</p><p className="text-sm">Ask any questions to test</p></div>}
              {openTestMessages.filter(m => m.system === openTestSystem).map((msg, i) => (<div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}><div className={`max-w-[80%] p-3 rounded-lg whitespace-pre-wrap ${msg.role === 'user' ? (openTestSystem === 'A' ? 'bg-violet-600 text-white' : 'bg-emerald-600 text-white') : 'bg-gray-100 text-gray-800'}`}>{msg.content}</div></div>))}
              {openTestLoading && <div className="flex justify-start"><div className="bg-gray-100 p-3 rounded-lg"><RefreshCw size={16} className="animate-spin" /></div></div>}
            </div>
            <div className="border-t p-4 flex gap-2"><input type="text" className="flex-grow border rounded-lg px-4 py-2" placeholder={`Ask System ${openTestSystem}...`} value={openTestInput} onChange={(e) => setOpenTestInput(e.target.value)} onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && sendOpenTestMessage()} /><button onClick={sendOpenTestMessage} disabled={openTestLoading || !openTestInput.trim()} className={`px-4 py-2 rounded-lg font-bold ${openTestSystem === 'A' ? 'bg-violet-600' : 'bg-emerald-600'} text-white disabled:opacity-50`}><Send size={18} /></button></div>
          </div>
        </main>
      </div>
    );
  }

  // EVALUATION PHASE - New 5-point scale
  if (phase === 'evaluation') {
    if (finished) return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="max-w-xl w-full bg-white shadow-xl rounded-2xl p-12 text-center">
          <CheckCircle className="mx-auto text-green-500 mb-6" size={80} />
          <h1 className="text-3xl font-bold mb-2">Thank You!</h1>
          <p className="text-gray-600 mb-8">Your feedback helps us improve AI matching systems.</p>
          <button onClick={downloadResults} className="mb-4 w-full bg-blue-600 text-white py-3 rounded-lg font-bold hover:bg-blue-700 flex items-center justify-center gap-2"><Download size={18} /> Download Your Data</button>
          <p className="text-sm text-gray-400">Session: {sessionId}</p>
        </div>
      </div>
    );

    const cupidWins = roundHistory.filter(r => r.cupid_vote).length;
    const baselineWins = roundHistory.filter(r => r.baseline_vote).length;
    const systemACost = (arenaState?.cupid_cost || 0) + (arenaState?.routing_cost || 0);
    const systemBCost = arenaState?.baseline_cost || 0;

    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="max-w-4xl w-full bg-white shadow-xl rounded-2xl overflow-hidden">
          <div className="bg-blue-600 p-6 text-white text-center">
            <h1 className="text-2xl font-bold">Final Evaluation</h1>
            <p className="opacity-90">Rate both systems based on your experience</p>
          </div>
          <div className="p-4 md:p-8 bg-gray-50">
            <div className="text-center mb-8">
              <p className="text-gray-600">You completed {roundHistory.length} comparison round{roundHistory.length !== 1 ? 's' : ''}</p>
              <p className="text-xs text-gray-400 mt-2">(Model identities remain hidden — rate based on response quality only)</p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-10">
              {renderEvalCard("System A", systemACost, evalRatingA, setEvalRatingA, "violet", cupidWins)}
              {renderEvalCard("System B", systemBCost, evalRatingB, setEvalRatingB, "emerald", baselineWins)}
            </div>
            <div className="max-w-2xl mx-auto space-y-6">
              <div>
                <label className="block text-sm font-bold text-gray-700 mb-2">Any final thoughts? (optional)</label>
                <textarea className="w-full border rounded-lg p-3 h-24 bg-white" placeholder="What worked well? What could be improved?" value={evalComment} onChange={(e) => setEvalComment(e.target.value)} />
              </div>
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

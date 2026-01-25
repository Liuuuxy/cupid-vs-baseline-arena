import React, { useState, useCallback, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  ArrowRight, ArrowLeft, Image, User, CheckCircle,
  Star, DollarSign, Zap, X, Info, RefreshCw,
  AlertCircle, Send, MessageCircle, Target, Sparkles,
  BookOpen, ThumbsUp, Settings, Layers, MessageSquare, Type, Wand2, Camera
} from 'lucide-react';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import 'katex/dist/katex.min.css';

// --- API CONFIGURATION ---
const API_URL = 'https://cupid-vs-baseline-arena.onrender.com';  // Update this
const SURVEY_URL = "https://your-survey-url.com";

// --- MODE TYPE ---
type ArenaMode = 'text' | 'image';

// --- MARKDOWN COMPONENT FOR TEXT MODE ---
const Markdown: React.FC<{ content: string; className?: string }> = ({ content, className = '' }) => {
  const preprocessContent = (text: string) => {
    if (!text) return '';
    return text
      .replace(/\\\[/g, '$$')
      .replace(/\\\]/g, '$$')
      .replace(/\\\(/g, '$')
      .replace(/\\\)/g, '$');
  };

  return (
    <ReactMarkdown
      className={`prose prose-sm max-w-none ${className}`}
      remarkPlugins={[remarkMath, remarkGfm]}
      rehypePlugins={[rehypeKatex]}
      components={{
        h1: ({ children }) => <h1 className="text-xl font-bold mt-4 mb-2 text-gray-900">{children}</h1>,
        h2: ({ children }) => <h2 className="text-lg font-bold mt-3 mb-2 text-gray-900">{children}</h2>,
        p: ({ children }) => <p className="mb-2 text-gray-700 leading-relaxed">{children}</p>,
        ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-1 text-gray-700">{children}</ul>,
        ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-1 text-gray-700">{children}</ol>,
        code: ({ className, children, ...props }) => {
          const isInline = !className;
          if (isInline) {
            return <code className="bg-gray-100 text-pink-600 px-1 py-0.5 rounded text-xs font-mono" {...props}>{children}</code>;
          }
          return (
            <code className="block bg-gray-900 text-green-400 p-3 rounded-lg text-xs font-mono overflow-x-auto my-2" {...props}>
              {children}
            </code>
          );
        },
        pre: ({ children }) => <pre className="bg-gray-900 rounded-lg overflow-x-auto my-2">{children}</pre>,
        blockquote: ({ children }) => (
          <blockquote className="border-l-4 border-blue-400 pl-4 py-1 my-2 bg-blue-50 text-gray-700 italic">
            {children}
          </blockquote>
        ),
        a: ({ href, children }) => (
          <a href={href} className="text-blue-600 hover:text-blue-800 underline" target="_blank" rel="noopener noreferrer">
            {children}
          </a>
        ),
        strong: ({ children }) => <strong className="font-bold text-gray-900">{children}</strong>,
      }}
    >
      {preprocessContent(content)}
    </ReactMarkdown>
  );
};

// --- IMAGE DISPLAY COMPONENT ---
const ImageDisplay: React.FC<{
  imageUrl: string;
  alt: string;
  className?: string;
  loading?: boolean;
}> = ({ imageUrl, alt, className = '', loading = false }) => {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);

  useEffect(() => {
    setImageLoaded(false);
    setImageError(false);
  }, [imageUrl]);

  if (loading) {
    return (
      <div className={`flex items-center justify-center bg-gray-100 rounded-lg ${className}`} style={{ minHeight: '300px' }}>
        <div className="text-center">
          <RefreshCw className="animate-spin mx-auto mb-2 text-blue-500" size={32} />
          <p className="text-sm text-gray-500">Generating image...</p>
        </div>
      </div>
    );
  }

  if (imageError || !imageUrl) {
    return (
      <div className={`flex items-center justify-center bg-gray-100 rounded-lg ${className}`} style={{ minHeight: '300px' }}>
        <div className="text-center text-gray-400">
          <AlertCircle className="mx-auto mb-2" size={32} />
          <p className="text-sm">Failed to load image</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`relative ${className}`}>
      {!imageLoaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100 rounded-lg">
          <RefreshCw className="animate-spin text-blue-500" size={24} />
        </div>
      )}
      <img
        src={imageUrl}
        alt={alt}
        className={`w-full h-auto rounded-lg shadow-md transition-opacity duration-300 ${imageLoaded ? 'opacity-100' : 'opacity-0'}`}
        onLoad={() => setImageLoaded(true)}
        onError={() => setImageError(true)}
      />
    </div>
  );
};

// --- CONTENT DISPLAY (handles both text and image) ---
const ContentDisplay: React.FC<{
  content: string;
  contentType: 'text' | 'image';
  loading?: boolean;
  className?: string;
}> = ({ content, contentType, loading = false, className = '' }) => {
  if (loading) {
    return (
      <div className={`flex items-center justify-center bg-gray-50 rounded-lg p-8 ${className}`}>
        <RefreshCw className="animate-spin text-blue-500 mr-2" size={20} />
        <span className="text-gray-500">
          {contentType === 'image' ? 'Generating image...' : 'Generating response...'}
        </span>
      </div>
    );
  }

  if (contentType === 'image') {
    return <ImageDisplay imageUrl={content} alt="Generated image" className={className} />;
  }

  return (
    <div className={`bg-gray-50 rounded-lg p-4 ${className}`}>
      <Markdown content={content} />
    </div>
  );
};

// --- TYPES ---
type Phase = 'modeSelect' | 'consent' | 'calibration' | 'interaction' | 'openTesting' | 'evaluation';
type PersonaGroup = 'traditional' | 'expert' | 'preference';

interface ModelResponse {
  model_id: number;
  model_name: string;
  content: string;
  cost: number;
  content_type: 'text' | 'image';
}

interface ArenaState {
  session_id: string;
  round: number;
  mode: ArenaMode;
  cupid_cost: number;
  baseline_cost: number;
  routing_cost: number;
  cupid_pair: { left: ModelResponse; right: ModelResponse; };
  baseline_pair: { left: ModelResponse; right: ModelResponse; };
}

interface Constraint {
  attribute: string;
  operator: '>=' | '<=' | '==' | '>';
  value: number | boolean | string;
  displayName: string;
  unit?: string;
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
  cupid_vote: 'left' | 'right';
  baseline_vote: 'left' | 'right';
  feedback: string;
  timestamp: string;
}

// --- CONFIG BASED ON MODE ---
const getModeConfig = (mode: ArenaMode) => {
  if (mode === 'text') {
    return {
      title: 'LLM Model Arena',
      subtitle: 'Compare text generation models',
      icon: <MessageSquare className="text-blue-500" size={28} />,
      promptLabel: 'Enter your question or prompt',
      promptPlaceholder: 'E.g., Explain quantum computing in simple terms...',
      preferenceLabel: 'What kind of LLM are you looking for?',
      preferencePlaceholder: 'E.g., I need fast, accurate responses for coding questions...',
      expertSubjects: [
        { id: 'science', label: 'Science & Research', icon: <BookOpen className="text-blue-500" size={24} /> },
        { id: 'coding', label: 'Programming & Tech', icon: <Settings className="text-green-500" size={24} /> },
        { id: 'writing', label: 'Writing & Content', icon: <Type className="text-purple-500" size={24} /> },
        { id: 'business', label: 'Business & Finance', icon: <DollarSign className="text-yellow-500" size={24} /> },
        { id: 'education', label: 'Education & Learning', icon: <BookOpen className="text-orange-500" size={24} /> },
      ],
      constraintColumns: [
        { key: 'intelligence', displayName: 'Intelligence Rating', unit: '', isCost: false },
        { key: 'speed', displayName: 'Speed Rating', unit: '', isCost: false },
        { key: 'input_price', displayName: 'Input Price', unit: '$/1M tokens', isCost: true },
        { key: 'output_price', displayName: 'Output Price', unit: '$/1M tokens', isCost: true },
      ],
      budgetRange: { minCost: 0.5, maxCost: 1.5 },
      ratingLabels: [
        { value: 1, label: "Very poor responses", shortLabel: "Very Poor" },
        { value: 2, label: "Below average quality", shortLabel: "Poor" },
        { value: 3, label: "Average, acceptable", shortLabel: "Average" },
        { value: 4, label: "Good quality responses", shortLabel: "Good" },
        { value: 5, label: "Excellent, exceeds expectations", shortLabel: "Excellent" }
      ],
    };
  } else {
    return {
      title: 'Text-to-Image Arena',
      subtitle: 'Compare image generation models',
      icon: <Wand2 className="text-purple-500" size={28} />,
      promptLabel: 'Enter your image prompt',
      promptPlaceholder: 'E.g., A serene mountain landscape at sunset with a small cabin...',
      preferenceLabel: 'What kind of images are you looking for?',
      preferencePlaceholder: 'E.g., I need photorealistic product images...',
      expertSubjects: [], // Not used for image mode
      constraintColumns: [], // Not used for image mode
      budgetRange: { minCost: 0.5, maxCost: 1 },
      ratingLabels: [
        { value: 1, label: "Very poor quality images", shortLabel: "Very Poor" },
        { value: 2, label: "Below average quality", shortLabel: "Poor" },
        { value: 3, label: "Average, acceptable", shortLabel: "Average" },
        { value: 4, label: "Good quality images", shortLabel: "Good" },
        { value: 5, label: "Excellent quality", shortLabel: "Excellent" }
      ],
    };
  }
};

// --- CONSTANTS ---
const EDUCATION_LEVELS = [
  "High School", "Some College", "Bachelor's Degree", "Master's Degree", "PhD / Doctorate", "Other"
];

const MAJORS_TEXT = [
  "Computer Science / IT", "Engineering", "Business / Finance", "Arts & Design",
  "Social Sciences", "Natural Sciences", "Healthcare", "Education", "Other"
];

const MAJORS_IMAGE = [
  "Art & Design", "Computer Science / IT", "Engineering", "Business / Finance",
  "Photography", "Marketing", "Architecture", "Other"
];

const BUDGET_RATING_LABELS = [
  { value: 1, label: 'Heavily exceeded the budget', color: 'text-red-600' },
  { value: 2, label: 'Somewhat exceeded the budget', color: 'text-orange-600' },
  { value: 3, label: 'Just right', color: 'text-green-600' },
  { value: 4, label: 'Saved some money', color: 'text-blue-600' },
  { value: 5, label: 'Very efficient cost-wise', color: 'text-purple-600' },
];

const OPEN_TESTING_MAX_ROUNDS = 10;

// --- MAIN APP COMPONENT ---
const App: React.FC = () => {
  // Core state
  const [phase, setPhase] = useState<Phase>('modeSelect');
  const [mode, setMode] = useState<ArenaMode>('text');
  const [sessionId, setSessionId] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  // User profile (text mode only for constraints)
  const [personaGroup, setPersonaGroup] = useState<PersonaGroup | null>(null);
  const [selectedExpertSubject, setSelectedExpertSubject] = useState<string | null>(null);
  const [assignedConstraints, setAssignedConstraints] = useState<Constraint[]>([]);
  const [demographics, setDemographics] = useState<Demographics>({
    age: '',
    education: EDUCATION_LEVELS[2],
    major: MAJORS_TEXT[0],
    familiarity: 3
  });
  const [budgetConstraints, setBudgetConstraints] = useState<BudgetConstraints>({
    maxCost: 1.0,
    maxRounds: 10
  });

  // Arena state
  const [init, setInit] = useState<boolean>(true);
  const [prompt, setPrompt] = useState<string>('');
  const [nextPrompt, setNextPrompt] = useState<string>('');
  const [arenaState, setArenaState] = useState<ArenaState | null>(null);
  const [cupidVote, setCupidVote] = useState<'left' | 'right' | null>(null);
  const [baselineVote, setBaselineVote] = useState<'left' | 'right' | null>(null);
  const [feedbackA, setFeedbackA] = useState<string>('');
  const [initialPreference, setInitialPreference] = useState<string>('');
  const [roundHistory, setRoundHistory] = useState<RoundHistory[]>([]);

  // Testing phase state
  interface SideBySideRound {
    prompt: string;
    contentA: string;
    contentB: string;
    costA: number;
    costB: number;
  }
  const [sideBySideRounds, setSideBySideRounds] = useState<SideBySideRound[]>([]);
  const [openTestInput, setOpenTestInput] = useState<string>('');
  const [openTestLoading, setOpenTestLoading] = useState<boolean>(false);

  // Evaluation state
  const [evalRatingA, setEvalRatingA] = useState<number>(0);
  const [evalRatingB, setEvalRatingB] = useState<number>(0);
  const [evalBudgetRatingA, setEvalBudgetRatingA] = useState<number>(0);
  const [evalBudgetRatingB, setEvalBudgetRatingB] = useState<number>(0);
  const [evalComment, setEvalComment] = useState<string>('');
  const [finished, setFinished] = useState<boolean>(false);
  const [hasTestedModels, setHasTestedModels] = useState<boolean>(false);

  // Model pool data (text mode only)
  const [modelPoolData, setModelPoolData] = useState<any[]>([]);

  // Get mode config
  const config = getModeConfig(mode);

  // Fetch model pool on mode change (text mode only)
  useEffect(() => {
    const fetchModelPool = async () => {
      if (mode === 'text') {
        try {
          const response = await fetch(`${API_URL}/model-pool-stats?mode=${mode}`);
          if (response.ok) {
            const data = await response.json();
            setModelPoolData(data.models || []);
          }
        } catch (err) {
          console.error('Failed to fetch model pool stats:', err);
        }
      }
    };
    fetchModelPool();
  }, [mode]);

  // Scroll to top on phase change
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [phase, arenaState?.round, init]);

  // Sample budget based on mode
  const sampleBudget = useCallback(() => {
    const { minCost, maxCost } = config.budgetRange;
    const minRounds = 5;
    const maxRounds = 15;

    const randomCost = Math.round((minCost + Math.random() * (maxCost - minCost)) * 1000) / 1000;
    const randomRounds = Math.floor(minRounds + Math.random() * (maxRounds - minRounds + 1));

    return { maxRounds: randomRounds, maxCost: randomCost };
  }, [config.budgetRange]);

  // Sample constraints (text mode only)
  const sampleConstraints = useCallback((nObjectives: number = 3) => {
    const constraints: Constraint[] = [];
    if (modelPoolData.length === 0 || mode === 'image') return constraints;

    const shuffled = [...config.constraintColumns].sort(() => Math.random() - 0.5);
    const selectedCols = shuffled.slice(0, Math.min(nObjectives, shuffled.length));

    for (const col of selectedCols) {
      const values = modelPoolData.map(m => m[col.key]).filter(v => v !== undefined && v !== null);
      if (values.length === 0) continue;

      const samples: number[] = [];
      for (let i = 0; i < 5; i++) {
        samples.push(values[Math.floor(Math.random() * values.length)]);
      }

      let chosenValue: number;
      let operator: '>=' | '<=';

      if (col.isCost) {
        chosenValue = Math.min(...samples);
        operator = '<=';
      } else {
        chosenValue = Math.max(...samples);
        operator = '>=';
      }

      constraints.push({
        attribute: col.key,
        operator,
        value: chosenValue,
        displayName: col.displayName,
        unit: col.unit
      });
    }

    return constraints;
  }, [modelPoolData, config.constraintColumns, mode]);

  // Format constraint for display
  const formatConstraint = (c: Constraint): string => {
    const opMap: Record<string, string> = { '>=': '≥', '<=': '≤', '==': '=', '>': '>' };
    const valueStr = typeof c.value === 'number' ? c.value.toLocaleString() : String(c.value);
    return `${c.displayName} ${opMap[c.operator]} ${valueStr}${c.unit ? ` ${c.unit}` : ''}`;
  };

  // Fetch next round
  const fetchNextRound = useCallback(async (isFirst: boolean = false, currentPrompt?: string) => {
    setLoading(true);
    setError(null);
    const promptToUse = currentPrompt || prompt;

    try {
      const payload: any = {
        session_id: isFirst ? null : sessionId,
        prompt: promptToUse,
        mode: mode,
        feedback_text: isFirst ? (initialPreference || '') : (feedbackA || ''),
      };

      if (!isFirst && cupidVote) payload.cupid_vote = cupidVote;
      if (!isFirst && baselineVote) payload.baseline_vote = baselineVote;

      if (isFirst) {
        payload.budget_cost = budgetConstraints.maxCost;
        payload.budget_rounds = budgetConstraints.maxRounds;
        payload.persona_group = personaGroup || 'preference'; // Default to preference for image mode
        payload.expert_subject = selectedExpertSubject;
        payload.constraints = assignedConstraints;
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
        setRoundHistory(prev => [...prev, {
          round: arenaState.round,
          prompt: prompt,
          cupid_vote: cupidVote,
          baseline_vote: baselineVote,
          feedback: feedbackA,
          timestamp: new Date().toISOString(),
        }]);
      }

      const newState: ArenaState = {
        session_id: data.session_id,
        round: data.round,
        mode: data.mode,
        cupid_cost: data.cupid_cost || 0,
        baseline_cost: data.baseline_cost || 0,
        routing_cost: data.routing_cost || 0,
        cupid_pair: { left: data.cLeft, right: data.cRight },
        baseline_pair: { left: data.bLeft, right: data.bRight },
      };

      setArenaState(newState);
      setPrompt(promptToUse);
      setNextPrompt('');
      setCupidVote(null);
      setBaselineVote(null);
      setFeedbackA('');

    } catch (err) {
      console.error("Failed to fetch round:", err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(`Error: ${errorMessage}. Please check your connection and try again.`);
    } finally {
      setLoading(false);
    }
  }, [prompt, sessionId, cupidVote, baselineVote, feedbackA, initialPreference, budgetConstraints, personaGroup, selectedExpertSubject, assignedConstraints, demographics, arenaState, mode]);

  // Send side-by-side message
  const sendSideBySideMessage = async () => {
    if (!openTestInput.trim() || openTestLoading) return;

    if (sideBySideRounds.length >= OPEN_TESTING_MAX_ROUNDS) {
      setError(`Maximum of ${OPEN_TESTING_MAX_ROUNDS} comparison rounds reached.`);
      return;
    }

    const currentPrompt = openTestInput;
    setOpenTestInput('');
    setOpenTestLoading(true);
    setError(null);

    try {
      const [responseA, responseB] = await Promise.all([
        fetch(`${API_URL}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId, message: currentPrompt, system: 'cupid', mode }),
        }),
        fetch(`${API_URL}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId, message: currentPrompt, system: 'baseline', mode }),
        })
      ]);

      let contentA = '', contentB = '';
      let costA = 0, costB = 0;

      if (responseA.ok) {
        const dataA = await responseA.json();
        contentA = dataA.content || '';
        costA = dataA.cost || 0;
      }
      if (responseB.ok) {
        const dataB = await responseB.json();
        contentB = dataB.content || '';
        costB = dataB.cost || 0;
      }

      setSideBySideRounds(prev => [...prev, { prompt: currentPrompt, contentA, contentB, costA, costB }]);
      setHasTestedModels(true);

    } catch (e) {
      console.error('Chat error:', e);
      setError('Connection error. Please try again.');
    } finally {
      setOpenTestLoading(false);
    }
  };

  // Save session
  const saveSessionData = useCallback(async () => {
    try {
      await fetch(`${API_URL}/session/${sessionId}/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          demographics,
          persona_group: personaGroup || 'preference',
          expert_subject: selectedExpertSubject,
          constraints: assignedConstraints,
          budget: budgetConstraints,
          history: roundHistory,
          evaluation: {
            quality_rating_a: evalRatingA,
            quality_rating_b: evalRatingB,
            budget_rating_a: evalBudgetRatingA,
            budget_rating_b: evalBudgetRatingB,
            comment: evalComment
          },
          final_cost_a: (arenaState?.cupid_cost || 0) + (arenaState?.routing_cost || 0),
          final_cost_b: arenaState?.baseline_cost || 0,
          side_by_side_rounds: sideBySideRounds.length,
        })
      });
    } catch (e) {
      console.error('Failed to save session:', e);
    }
  }, [sessionId, demographics, personaGroup, selectedExpertSubject, assignedConstraints, budgetConstraints, roundHistory, evalRatingA, evalRatingB, evalBudgetRatingA, evalBudgetRatingB, evalComment, arenaState, sideBySideRounds]);

  // Handlers
  const handleModeSelect = (selectedMode: ArenaMode) => {
    setMode(selectedMode);
    setDemographics(prev => ({
      ...prev,
      major: selectedMode === 'text' ? MAJORS_TEXT[0] : MAJORS_IMAGE[0]
    }));
    setPhase('consent');
  };

  const handleConsent = () => {
    const newSessionId = `sess_${mode}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setSessionId(newSessionId);
    const assignedBudget = sampleBudget();
    setBudgetConstraints(assignedBudget);

    // Image mode skips calibration (no persona groups/constraints)
    if (mode === 'image') {
      setPersonaGroup('preference');
      setPhase('interaction');
    } else {
      setPhase('calibration');
    }
  };

  const handlePersonaGroupSelect = (group: PersonaGroup) => {
    setPersonaGroup(group);
    if (group === 'traditional') {
      const numConstraints = 2 + Math.floor(Math.random() * 3);
      setAssignedConstraints(sampleConstraints(numConstraints));
    } else {
      setAssignedConstraints([]);
    }
    if (group !== 'expert') {
      setSelectedExpertSubject(null);
    }
  };

  const handleCalibrationSubmit = () => {
    if (!demographics.age) { setError("Please enter your age."); return; }
    if (!personaGroup) { setError("Please select a testing mode."); return; }
    if (personaGroup === 'expert' && !selectedExpertSubject) { setError("Please select your area of expertise."); return; }
    setError(null);
    setPhase('interaction');
  };

  const startSession = async () => {
    if (!prompt.trim()) { setError("Please enter a prompt to start."); return; }
    if (!initialPreference.trim()) { setError("Please describe what you're looking for."); return; }
    setError(null);
    await fetchNextRound(true, prompt);
    setInit(false);
  };

  const handleSatisfied = () => {
    if (arenaState && cupidVote && baselineVote) {
      setRoundHistory(prev => [...prev, {
        round: arenaState.round,
        prompt: prompt,
        cupid_vote: cupidVote,
        baseline_vote: baselineVote,
        feedback: feedbackA,
        timestamp: new Date().toISOString(),
      }]);
    }
    setPhase('openTesting');
  };

  const handleSubmitRound = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!cupidVote || !baselineVote) {
      setError(`Please select your preferred ${mode === 'image' ? 'image' : 'response'} from both systems.`);
      return;
    }

    const systemACost = (arenaState?.cupid_cost || 0) + (arenaState?.routing_cost || 0);
    const systemBCost = arenaState?.baseline_cost || 0;
    const totalCost = systemACost + systemBCost;
    const isLastRound = arenaState && (arenaState.round >= budgetConstraints.maxRounds || totalCost >= budgetConstraints.maxCost);

    if (isLastRound) {
      if (arenaState && cupidVote && baselineVote) {
        setRoundHistory(prev => [...prev, {
          round: arenaState.round,
          prompt: prompt,
          cupid_vote: cupidVote,
          baseline_vote: baselineVote,
          feedback: feedbackA,
          timestamp: new Date().toISOString(),
        }]);
      }
      setPhase('openTesting');
    } else {
      await fetchNextRound(false, nextPrompt || prompt);
    }
  };

  const handleFinalSubmit = async () => {
    await saveSessionData();
    setFinished(true);
  };

  // Render content pair for comparison
  const renderContentPair = (
    leftContent: ModelResponse,
    rightContent: ModelResponse,
    vote: 'left' | 'right' | null,
    setVote: (v: 'left' | 'right') => void,
    systemLabel: string
  ) => (
    <div className="mb-6">
      <div className="flex items-center gap-2 mb-3">
        <span className="font-bold text-gray-800">{systemLabel}</span>
        <span className="text-xs text-gray-500">Click to select preferred {mode === 'image' ? 'image' : 'response'}</span>
      </div>
      <div className="grid grid-cols-2 gap-4">
        {/* Left option */}
        <div
          onClick={() => setVote('left')}
          className={`cursor-pointer rounded-xl border-2 p-3 transition-all ${vote === 'left'
            ? 'border-green-500 bg-green-50 ring-2 ring-green-300'
            : 'border-gray-200 hover:border-blue-300 hover:shadow-md'
            }`}
        >
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-gray-600">Option A</span>
            {vote === 'left' && (
              <span className="bg-green-500 text-white text-xs px-2 py-1 rounded-full flex items-center gap-1">
                <CheckCircle size={12} /> Selected
              </span>
            )}
          </div>
          <ContentDisplay
            content={leftContent.content}
            contentType={leftContent.content_type}
            className="min-h-[150px]"
          />
        </div>

        {/* Right option */}
        <div
          onClick={() => setVote('right')}
          className={`cursor-pointer rounded-xl border-2 p-3 transition-all ${vote === 'right'
            ? 'border-green-500 bg-green-50 ring-2 ring-green-300'
            : 'border-gray-200 hover:border-blue-300 hover:shadow-md'
            }`}
        >
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-gray-600">Option B</span>
            {vote === 'right' && (
              <span className="bg-green-500 text-white text-xs px-2 py-1 rounded-full flex items-center gap-1">
                <CheckCircle size={12} /> Selected
              </span>
            )}
          </div>
          <ContentDisplay
            content={rightContent.content}
            contentType={rightContent.content_type}
            className="min-h-[150px]"
          />
        </div>
      </div>
    </div>
  );

  // Render rating card
  const renderEvalCard = (
    systemLabel: string,
    totalCost: number,
    rating: number,
    setRating: (v: number) => void,
    budgetRating: number,
    setBudgetRating: (v: number) => void
  ) => (
    <div className="bg-white border rounded-2xl p-6 shadow-sm">
      <h3 className="text-lg font-bold text-gray-800 mb-4">{systemLabel}</h3>

      <div className="mb-4 p-3 bg-gray-50 rounded-lg">
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Total Cost:</span>
          <span className="font-bold text-gray-800">${totalCost.toFixed(4)}</span>
        </div>
      </div>

      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          {mode === 'image' ? 'Image' : 'Response'} Quality Rating
        </label>
        <div className="flex gap-2">
          {config.ratingLabels.map((r) => (
            <button
              key={r.value}
              onClick={() => setRating(r.value)}
              className={`flex-1 py-2 px-1 text-xs rounded-lg border transition-all ${rating === r.value
                ? 'bg-blue-500 text-white border-blue-500'
                : 'bg-white text-gray-600 border-gray-300 hover:border-blue-300'
                }`}
            >
              {r.shortLabel}
            </button>
          ))}
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Budget Compliance</label>
        <div className="flex gap-2">
          {BUDGET_RATING_LABELS.map((r) => (
            <button
              key={r.value}
              onClick={() => setBudgetRating(r.value)}
              className={`flex-1 py-2 px-1 text-xs rounded-lg border transition-all ${budgetRating === r.value
                ? 'bg-purple-500 text-white border-purple-500'
                : 'bg-white text-gray-600 border-gray-300 hover:border-purple-300'
                }`}
              title={r.label}
            >
              {r.value}
            </button>
          ))}
        </div>
      </div>
    </div>
  );

  const systemACost = (arenaState?.cupid_cost || 0) + (arenaState?.routing_cost || 0);
  const systemBCost = arenaState?.baseline_cost || 0;

  // ================== RENDER PHASES ==================

  // MODE SELECT PHASE
  if (phase === 'modeSelect') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-xl max-w-2xl w-full overflow-hidden">
          <div className="bg-gradient-to-r from-purple-600 to-blue-600 p-6 text-white text-center">
            <h1 className="text-2xl font-bold flex items-center justify-center gap-3">
              <Sparkles size={28} />
              Model Selection Arena
            </h1>
            <p className="mt-2 opacity-90">Choose which type of AI models to compare</p>
          </div>
          <div className="p-8">
            <div className="grid grid-cols-2 gap-6">
              {/* Text/LLM Mode */}
              <button
                onClick={() => handleModeSelect('text')}
                className="p-6 rounded-2xl border-2 border-gray-200 hover:border-blue-500 hover:bg-blue-50 transition-all text-center group"
              >
                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-blue-200 transition-colors">
                  <MessageSquare className="text-blue-600" size={32} />
                </div>
                <h2 className="text-xl font-bold text-gray-800 mb-2">LLM Arena</h2>
                <p className="text-sm text-gray-600">
                  Compare text generation models like GPT, Claude, Gemini, and more
                </p>
                <div className="mt-4 flex flex-wrap justify-center gap-2">
                  <span className="bg-blue-100 text-blue-700 text-xs px-2 py-1 rounded">Chat</span>
                  <span className="bg-blue-100 text-blue-700 text-xs px-2 py-1 rounded">Code</span>
                  <span className="bg-blue-100 text-blue-700 text-xs px-2 py-1 rounded">Writing</span>
                </div>
              </button>

              {/* Image Mode */}
              <button
                onClick={() => handleModeSelect('image')}
                className="p-6 rounded-2xl border-2 border-gray-200 hover:border-purple-500 hover:bg-purple-50 transition-all text-center group"
              >
                <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-purple-200 transition-colors">
                  <Wand2 className="text-purple-600" size={32} />
                </div>
                <h2 className="text-xl font-bold text-gray-800 mb-2">Image Arena</h2>
                <p className="text-sm text-gray-600">
                  Compare text-to-image models like DALL·E, Imagen, and more
                </p>
                <div className="mt-4 flex flex-wrap justify-center gap-2">
                  <span className="bg-purple-100 text-purple-700 text-xs px-2 py-1 rounded">Art</span>
                  <span className="bg-purple-100 text-purple-700 text-xs px-2 py-1 rounded">Photos</span>
                  <span className="bg-purple-100 text-purple-700 text-xs px-2 py-1 rounded">Design</span>
                </div>
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // CONSENT PHASE
  if (phase === 'consent') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-xl max-w-2xl w-full overflow-hidden">
          <div className="bg-gradient-to-r from-purple-600 to-blue-600 p-6 text-white">
            <h1 className="text-2xl font-bold flex items-center gap-3">
              {config.icon}
              {config.title}
            </h1>
            <p className="mt-2 opacity-90">{config.subtitle}</p>
          </div>
          <div className="p-6 space-y-4">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h2 className="font-bold text-blue-800 mb-2">📋 Study Overview</h2>
              <ul className="text-sm text-blue-700 space-y-1">
                <li>• You'll compare {mode === 'image' ? 'images' : 'responses'} from different AI models</li>
                <li>• Enter prompts and select outputs you prefer</li>
                <li>• The system learns your preferences</li>
                <li>• Takes approximately 10-15 minutes</li>
              </ul>
            </div>
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <h2 className="font-bold text-gray-800 mb-2">🔒 Privacy</h2>
              <p className="text-sm text-gray-600">
                Your responses are anonymous. Data is used only for research purposes.
              </p>
            </div>
            <div className="flex gap-4">
              <button
                onClick={() => setPhase('modeSelect')}
                className="flex-1 bg-gray-200 text-gray-700 py-4 rounded-xl font-bold hover:bg-gray-300 transition-all flex items-center justify-center"
              >
                <ArrowLeft className="mr-2" size={18} /> Back
              </button>
              <button
                onClick={handleConsent}
                className="flex-1 bg-gradient-to-r from-purple-600 to-blue-600 text-white py-4 rounded-xl font-bold hover:from-purple-700 hover:to-blue-700 transition-all flex items-center justify-center gap-2"
              >
                I Agree — Start Study <ArrowRight size={18} />
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // CALIBRATION PHASE (TEXT MODE ONLY)
  if (phase === 'calibration') {
    const majors = MAJORS_TEXT;

    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50 p-4">
        <div className="max-w-3xl mx-auto">
          <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
            <div className="bg-gradient-to-r from-purple-600 to-blue-600 p-6 text-white">
              <h1 className="text-2xl font-bold">Setup Your Profile</h1>
              <p className="mt-2 opacity-90">Help us understand your background</p>
            </div>

            <div className="p-6 space-y-6">
              {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-center gap-2">
                  <AlertCircle size={18} />
                  {error}
                </div>
              )}

              {/* Demographics */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Age</label>
                  <input
                    type="number"
                    min="18"
                    max="100"
                    value={demographics.age}
                    onChange={(e) => setDemographics({ ...demographics, age: parseInt(e.target.value) || '' })}
                    className="w-full border rounded-lg px-3 py-2"
                    placeholder="Enter your age"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Education</label>
                  <select
                    value={demographics.education}
                    onChange={(e) => setDemographics({ ...demographics, education: e.target.value })}
                    className="w-full border rounded-lg px-3 py-2"
                  >
                    {EDUCATION_LEVELS.map(level => (
                      <option key={level} value={level}>{level}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Field / Background</label>
                <select
                  value={demographics.major}
                  onChange={(e) => setDemographics({ ...demographics, major: e.target.value })}
                  className="w-full border rounded-lg px-3 py-2"
                >
                  {majors.map(major => (
                    <option key={major} value={major}>{major}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Familiarity with LLMs (1-5)
                </label>
                <div className="flex gap-2">
                  {[1, 2, 3, 4, 5].map(val => (
                    <button
                      key={val}
                      onClick={() => setDemographics({ ...demographics, familiarity: val })}
                      className={`flex-1 py-2 rounded-lg border transition-all ${demographics.familiarity === val
                        ? 'bg-blue-500 text-white border-blue-500'
                        : 'bg-white text-gray-600 border-gray-300 hover:border-blue-300'
                        }`}
                    >
                      {val}
                    </button>
                  ))}
                </div>
              </div>

              {/* Testing Mode Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Select Testing Mode</label>
                <div className="grid grid-cols-3 gap-3">
                  <button
                    onClick={() => handlePersonaGroupSelect('traditional')}
                    className={`p-4 rounded-xl border-2 transition-all text-center ${personaGroup === 'traditional'
                      ? 'border-purple-500 bg-purple-50'
                      : 'border-gray-200 hover:border-purple-300'
                      }`}
                  >
                    <Target className="mx-auto mb-2 text-purple-500" size={24} />
                    <div className="font-medium text-sm">Requirements</div>
                  </button>
                  <button
                    onClick={() => handlePersonaGroupSelect('expert')}
                    className={`p-4 rounded-xl border-2 transition-all text-center ${personaGroup === 'expert'
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-blue-300'
                      }`}
                  >
                    <BookOpen className="mx-auto mb-2 text-blue-500" size={24} />
                    <div className="font-medium text-sm">Expert</div>
                  </button>
                  <button
                    onClick={() => handlePersonaGroupSelect('preference')}
                    className={`p-4 rounded-xl border-2 transition-all text-center ${personaGroup === 'preference'
                      ? 'border-indigo-500 bg-indigo-50'
                      : 'border-gray-200 hover:border-indigo-300'
                      }`}
                  >
                    <ThumbsUp className="mx-auto mb-2 text-indigo-500" size={24} />
                    <div className="font-medium text-sm">Preference</div>
                  </button>
                </div>
              </div>

              {/* Expert Subject Selection */}
              {personaGroup === 'expert' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Select Your Expertise Area</label>
                  <div className="grid grid-cols-2 gap-3">
                    {config.expertSubjects.map(subject => (
                      <button
                        key={subject.id}
                        onClick={() => setSelectedExpertSubject(subject.id)}
                        className={`p-3 rounded-xl border-2 transition-all flex items-center gap-3 ${selectedExpertSubject === subject.id
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 hover:border-blue-300'
                          }`}
                      >
                        {subject.icon}
                        <span className="font-medium text-sm">{subject.label}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Constraints display */}
              {personaGroup === 'traditional' && assignedConstraints.length > 0 && (
                <div className="bg-purple-50 border border-purple-200 rounded-xl p-4">
                  <h3 className="font-bold text-purple-800 mb-2 flex items-center gap-2">
                    <Target size={18} /> Your Assigned Requirements
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {assignedConstraints.map((c, i) => (
                      <span key={i} className="bg-purple-100 text-purple-800 text-sm px-3 py-1 rounded-full">
                        {formatConstraint(c)}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Budget info */}
              <div className="bg-gray-50 border border-gray-200 rounded-xl p-4">
                <h3 className="font-bold text-gray-800 mb-2 flex items-center gap-2">
                  <DollarSign size={18} /> Your Budget
                </h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Max Cost:</span>
                    <span className="font-bold ml-2">${budgetConstraints.maxCost.toFixed(3)}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Max Rounds:</span>
                    <span className="font-bold ml-2">{budgetConstraints.maxRounds}</span>
                  </div>
                </div>
              </div>

              <button
                onClick={handleCalibrationSubmit}
                className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-4 rounded-xl font-bold hover:from-purple-700 hover:to-blue-700 transition-all flex items-center justify-center gap-2"
              >
                Continue <ArrowRight size={18} />
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // INTERACTION PHASE
  if (phase === 'interaction') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50 p-4">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="bg-white rounded-2xl shadow-lg mb-4 p-4">
            <div className="flex justify-between items-center">
              <div>
                <h1 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                  {config.icon}
                  {config.title}
                </h1>
                <p className="text-sm text-gray-500">
                  Round {arenaState?.round || 0} / {budgetConstraints.maxRounds}
                </p>
              </div>
              <div className="flex gap-4 text-sm">
                <div className="bg-green-50 px-3 py-1 rounded-full">
                  <span className="text-green-700">Budget: ${budgetConstraints.maxCost.toFixed(3)}</span>
                </div>
                <div className="bg-blue-50 px-3 py-1 rounded-full">
                  <span className="text-blue-700">
                    Spent: ${((arenaState?.cupid_cost || 0) + (arenaState?.baseline_cost || 0)).toFixed(4)}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-4 flex items-center gap-2">
              <AlertCircle size={18} />
              {error}
            </div>
          )}

          {/* Initial prompt input */}
          {init ? (
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h2 className="text-lg font-bold text-gray-800 mb-4">Start by entering your first prompt</h2>

              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {config.preferenceLabel}
                </label>
                <textarea
                  value={initialPreference}
                  onChange={(e) => setInitialPreference(e.target.value)}
                  className="w-full border rounded-lg px-4 py-3 h-20"
                  placeholder={config.preferencePlaceholder}
                />
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {config.promptLabel}
                </label>
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  className="w-full border rounded-lg px-4 py-3 h-24"
                  placeholder={config.promptPlaceholder}
                />
              </div>

              <button
                onClick={startSession}
                disabled={loading}
                className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-4 rounded-xl font-bold hover:from-purple-700 hover:to-blue-700 transition-all flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <RefreshCw className="animate-spin" size={18} />
                    {mode === 'image' ? 'Generating Images...' : 'Generating Responses...'}
                  </>
                ) : (
                  <>
                    Generate <Sparkles size={18} />
                  </>
                )}
              </button>
            </div>
          ) : arenaState && (
            <>
              {/* Current prompt display */}
              <div className="bg-white rounded-xl shadow-sm p-4 mb-4">
                <div className="flex items-start gap-3">
                  <MessageCircle className="text-purple-500 mt-1" size={20} />
                  <div>
                    <p className="text-sm text-gray-500">Your prompt:</p>
                    <p className="text-gray-800 font-medium">{prompt}</p>
                  </div>
                </div>
              </div>

              {loading ? (
                <div className="bg-white rounded-2xl shadow-lg p-12 text-center">
                  <RefreshCw className="animate-spin mx-auto mb-4 text-purple-500" size={48} />
                  <p className="text-gray-600">
                    {mode === 'image' ? 'Generating images...' : 'Generating responses...'}
                  </p>
                </div>
              ) : (
                <form onSubmit={handleSubmitRound}>
                  {/* System A */}
                  <div className="bg-white rounded-2xl shadow-lg p-6 mb-4">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center text-white font-bold">A</div>
                      <span className="font-bold text-gray-800">System A</span>
                    </div>
                    {renderContentPair(
                      arenaState.cupid_pair.left,
                      arenaState.cupid_pair.right,
                      cupidVote,
                      setCupidVote,
                      ''
                    )}
                  </div>

                  {/* System B */}
                  <div className="bg-white rounded-2xl shadow-lg p-6 mb-4">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold">B</div>
                      <span className="font-bold text-gray-800">System B</span>
                    </div>
                    {renderContentPair(
                      arenaState.baseline_pair.left,
                      arenaState.baseline_pair.right,
                      baselineVote,
                      setBaselineVote,
                      ''
                    )}
                  </div>

                  {/* Controls */}
                  <div className="bg-white rounded-2xl shadow-lg p-6">
                    <div className="mb-4">
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Feedback for next round (optional)
                      </label>
                      <input
                        type="text"
                        value={feedbackA}
                        onChange={(e) => setFeedbackA(e.target.value)}
                        className="w-full border rounded-lg px-4 py-2"
                        placeholder={mode === 'image' ? 'E.g., more realistic, brighter colors...' : 'E.g., more detailed, shorter response...'}
                      />
                    </div>

                    <div className="mb-4">
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Next prompt (or keep same)
                      </label>
                      <textarea
                        value={nextPrompt}
                        onChange={(e) => setNextPrompt(e.target.value)}
                        className="w-full border rounded-lg px-4 py-2 h-20"
                        placeholder={prompt}
                      />
                    </div>

                    <div className="flex gap-4">
                      {arenaState.round >= 1 && (
                        <button
                          type="button"
                          onClick={handleSatisfied}
                          disabled={!cupidVote || !baselineVote}
                          className="flex-1 bg-green-500 text-white py-3 rounded-xl font-bold hover:bg-green-600 disabled:opacity-50 transition-all flex items-center justify-center gap-2"
                        >
                          <CheckCircle size={18} /> I'm Satisfied — Test Models
                        </button>
                      )}
                      <button
                        type="submit"
                        disabled={!cupidVote || !baselineVote || loading}
                        className="flex-1 bg-gradient-to-r from-purple-600 to-blue-600 text-white py-3 rounded-xl font-bold hover:from-purple-700 hover:to-blue-700 disabled:opacity-50 transition-all flex items-center justify-center gap-2"
                      >
                        {loading ? <RefreshCw className="animate-spin" size={18} /> : <>Continue <ArrowRight size={18} /></>}
                      </button>
                    </div>
                  </div>
                </form>
              )}
            </>
          )}
        </div>
      </div>
    );
  }

  // OPEN TESTING PHASE
  if (phase === 'openTesting') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50 p-4">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="bg-white rounded-2xl shadow-lg mb-4 p-4">
            <div className="flex justify-between items-center">
              <div>
                <h1 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                  <Layers className="text-purple-500" size={24} />
                  Side-by-Side Testing
                </h1>
                <p className="text-sm text-gray-500">
                  Test {sideBySideRounds.length} / {OPEN_TESTING_MAX_ROUNDS} comparisons
                </p>
              </div>
              <button
                onClick={() => setPhase('evaluation')}
                className="bg-green-500 text-white px-4 py-2 rounded-lg font-medium hover:bg-green-600 transition-all flex items-center gap-2"
              >
                Proceed to Rating <ArrowRight size={16} />
              </button>
            </div>
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-4 flex items-center gap-2">
              <AlertCircle size={18} />
              {error}
            </div>
          )}

          {/* Prompt input */}
          <div className="bg-white rounded-2xl shadow-lg p-6 mb-4">
            <div className="flex gap-3">
              <textarea
                value={openTestInput}
                onChange={(e) => setOpenTestInput(e.target.value)}
                className="flex-1 border rounded-lg px-4 py-3"
                placeholder={config.promptPlaceholder}
                rows={2}
                disabled={openTestLoading}
              />
              <button
                onClick={sendSideBySideMessage}
                disabled={openTestLoading || !openTestInput.trim()}
                className="bg-gradient-to-r from-purple-600 to-blue-600 text-white px-6 rounded-lg font-medium hover:from-purple-700 hover:to-blue-700 disabled:opacity-50 transition-all flex items-center gap-2"
              >
                {openTestLoading ? <RefreshCw className="animate-spin" size={18} /> : <><Send size={18} /> Generate</>}
              </button>
            </div>
          </div>

          {/* Side-by-side results */}
          {sideBySideRounds.map((round, idx) => (
            <div key={idx} className="bg-white rounded-2xl shadow-lg p-6 mb-4">
              <div className="mb-4">
                <div className="flex items-start gap-3">
                  <MessageCircle className="text-purple-500 mt-1" size={20} />
                  <p className="text-gray-800 font-medium">{round.prompt}</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="border rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <div className="w-6 h-6 bg-purple-500 rounded-full flex items-center justify-center text-white text-xs font-bold">A</div>
                    <span className="font-medium text-gray-700">System A</span>
                    <span className="text-xs text-gray-400 ml-auto">${round.costA.toFixed(4)}</span>
                  </div>
                  <ContentDisplay
                    content={round.contentA}
                    contentType={mode === 'image' ? 'image' : 'text'}
                  />
                </div>
                <div className="border rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-white text-xs font-bold">B</div>
                    <span className="font-medium text-gray-700">System B</span>
                    <span className="text-xs text-gray-400 ml-auto">${round.costB.toFixed(4)}</span>
                  </div>
                  <ContentDisplay
                    content={round.contentB}
                    contentType={mode === 'image' ? 'image' : 'text'}
                  />
                </div>
              </div>
            </div>
          ))}

          {openTestLoading && (
            <div className="bg-white rounded-2xl shadow-lg p-8 text-center">
              <RefreshCw className="animate-spin mx-auto mb-4 text-purple-500" size={32} />
              <p className="text-gray-600">
                {mode === 'image' ? 'Generating images...' : 'Generating responses...'}
              </p>
            </div>
          )}

          {sideBySideRounds.length === 0 && !openTestLoading && (
            <div className="bg-white rounded-2xl shadow-lg p-12 text-center">
              {mode === 'image' ? (
                <Image className="mx-auto mb-4 text-gray-300" size={48} />
              ) : (
                <MessageSquare className="mx-auto mb-4 text-gray-300" size={48} />
              )}
              <p className="text-gray-500">Enter a prompt above to test both matched models</p>
            </div>
          )}
        </div>
      </div>
    );
  }

  // EVALUATION PHASE
  if (phase === 'evaluation') {
    if (finished) {
      return (
        <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-xl max-w-md w-full p-8 text-center">
            <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <CheckCircle className="text-green-500" size={32} />
            </div>
            <h1 className="text-2xl font-bold text-gray-800 mb-2">Thank You!</h1>
            <p className="text-gray-600 mb-6">
              Your responses have been saved. Thank you for participating!
            </p>
            {SURVEY_URL && (
              <a
                href={SURVEY_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="block w-full bg-blue-500 text-white py-3 rounded-lg font-medium hover:bg-blue-600 transition-all mb-4"
              >
                Complete Follow-up Survey
              </a>
            )}
            <button
              onClick={() => window.location.reload()}
              className="w-full bg-gray-200 text-gray-700 py-3 rounded-lg font-medium hover:bg-gray-300 transition-all"
            >
              Start New Session
            </button>
          </div>
        </div>
      );
    }

    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50 p-4">
        <div className="max-w-4xl mx-auto">
          <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
            <div className="bg-gradient-to-r from-purple-600 to-blue-600 p-6 text-white">
              <h1 className="text-2xl font-bold">Final Evaluation</h1>
              <p className="mt-2 opacity-90">Rate each system based on quality and budget adherence</p>
            </div>

            <div className="p-6">
              <div className="text-center mb-6">
                <p className="text-gray-600">
                  You completed {roundHistory.length} comparison rounds and {sideBySideRounds.length} test generations
                </p>
              </div>

              <div className="grid grid-cols-2 gap-6 mb-8">
                {renderEvalCard("System A", systemACost, evalRatingA, setEvalRatingA, evalBudgetRatingA, setEvalBudgetRatingA)}
                {renderEvalCard("System B", systemBCost, evalRatingB, setEvalRatingB, evalBudgetRatingB, setEvalBudgetRatingB)}
              </div>

              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Any final thoughts? (optional)
                </label>
                <textarea
                  value={evalComment}
                  onChange={(e) => setEvalComment(e.target.value)}
                  className="w-full border rounded-lg px-4 py-3 h-24"
                  placeholder="What worked well? What could be improved?"
                />
              </div>

              <div className="flex gap-4">
                <button
                  onClick={() => setPhase('openTesting')}
                  className="flex-1 bg-gray-200 text-gray-700 py-4 rounded-lg font-bold hover:bg-gray-300 transition flex items-center justify-center"
                >
                  <ArrowLeft className="mr-2" size={18} /> Back to Testing
                </button>
                <button
                  onClick={handleFinalSubmit}
                  disabled={evalRatingA === 0 || evalRatingB === 0 || evalBudgetRatingA === 0 || evalBudgetRatingB === 0 || !hasTestedModels}
                  className="flex-1 bg-gradient-to-r from-purple-600 to-blue-600 text-white py-4 rounded-lg font-bold hover:from-purple-700 hover:to-blue-700 disabled:opacity-50 transition flex items-center justify-center"
                >
                  Submit & Finish <ArrowRight className="ml-2" size={18} />
                </button>
              </div>

              {!hasTestedModels && (
                <p className="text-center text-amber-600 text-sm mt-4">
                  ⚠️ Please test both models at least once before rating.
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return null;
};

export default App;

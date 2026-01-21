import React, { useState, useCallback, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  ArrowRight, ArrowLeft, MessageSquare, User, CheckCircle,
  Star, DollarSign, Zap, Brain, X, Info, RefreshCw,
  AlertCircle, Download, Send, MessageCircle, Target, Sparkles,
  BookOpen, Heart, ThumbsUp, Settings, HelpCircle
} from 'lucide-react';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import 'katex/dist/katex.min.css';
// --- API CONFIGURATION ---
const API_URL = 'https://cupid-vs-baseline-arena.onrender.com';

// --- MARKDOWN COMPONENT WITH STYLING ---
const Markdown: React.FC<{ content: string; className?: string }> = ({ content, className = '' }) => {
  // FIX: Pre-process the content to convert OpenAI's LaTeX style (\[ ... \]) 
  // to the format remark-math expects ($$ ... $$)
  const preprocessContent = (text: string) => {
    if (!text) return '';
    return text
      // Replace block math \[ ... \] with $$ ... $$
      .replace(/\\\[/g, '$$')
      .replace(/\\\]/g, '$$')
      // Replace inline math \( ... \) with $ ... $
      .replace(/\\\(/g, '$')
      .replace(/\\\)/g, '$');
  };

  return (
    <ReactMarkdown
      className={`prose prose-sm max-w-none ${className}`}
      remarkPlugins={[remarkMath, remarkGfm]}
      rehypePlugins={[rehypeKatex]}
      components={{
        // ... (Keep all your existing custom components: h1, h2, code, table, etc.) ...
        // Headers
        h1: ({ children }) => <h1 className="text-xl font-bold mt-4 mb-2 text-gray-900">{children}</h1>,
        h2: ({ children }) => <h2 className="text-lg font-bold mt-3 mb-2 text-gray-900">{children}</h2>,
        h3: ({ children }) => <h3 className="text-base font-bold mt-2 mb-1 text-gray-900">{children}</h3>,
        h4: ({ children }) => <h4 className="text-sm font-bold mt-2 mb-1 text-gray-800">{children}</h4>,
        // Paragraphs
        p: ({ children }) => <p className="mb-2 text-gray-700 leading-relaxed">{children}</p>,
        // Lists
        ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-1 text-gray-700">{children}</ul>,
        ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-1 text-gray-700">{children}</ol>,
        li: ({ children }) => <li className="text-gray-700">{children}</li>,
        // Code
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
        // Blockquote
        blockquote: ({ children }) => (
          <blockquote className="border-l-4 border-blue-400 pl-4 py-1 my-2 bg-blue-50 text-gray-700 italic">
            {children}
          </blockquote>
        ),
        // Links
        a: ({ href, children }) => (
          <a href={href} className="text-blue-600 hover:text-blue-800 underline" target="_blank" rel="noopener noreferrer">
            {children}
          </a>
        ),
        // Strong/Bold
        strong: ({ children }) => <strong className="font-bold text-gray-900">{children}</strong>,
        // Emphasis/Italic
        em: ({ children }) => <em className="italic">{children}</em>,
        // Horizontal Rule
        hr: () => <hr className="my-4 border-gray-300" />,
        // Tables
        table: ({ children }) => (
          <div className="overflow-x-auto my-2">
            <table className="min-w-full border border-gray-300 text-sm">{children}</table>
          </div>
        ),
        thead: ({ children }) => <thead className="bg-gray-100">{children}</thead>,
        tbody: ({ children }) => <tbody>{children}</tbody>,
        tr: ({ children }) => <tr className="border-b border-gray-200">{children}</tr>,
        th: ({ children }) => <th className="px-3 py-2 text-left font-bold text-gray-700 border-r border-gray-200">{children}</th>,
        td: ({ children }) => <td className="px-3 py-2 text-gray-700 border-r border-gray-200">{children}</td>,
      }}
    >
      {preprocessContent(content)}
    </ReactMarkdown>
  );
};

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
  intelligence?: number | null;
  speed?: number | null;
  reasoning?: number | null;
  input_price?: number | null;
  output_price?: number | null;
  context_window?: number | null;
  max_output?: number | null;
  text_input?: boolean | null;
  image_input?: boolean | null;
  voice_input?: boolean | null;
  function_calling?: boolean | null;
  structured_output?: boolean | null;
  knowledge_cutoff?: string | null;
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
  cupid_left_id: number;
  cupid_right_id: number;
  cupid_vote: 'left' | 'right';
  baseline_left_id: number;
  baseline_right_id: number;
  baseline_vote: 'left' | 'right';
  feedback: string;
  timestamp: string;
  // Cost details for each round
  cupid_left_cost?: number;
  cupid_right_cost?: number;
  baseline_left_cost?: number;
  baseline_right_cost?: number;
  routing_cost?: number;
  cupid_total_cost?: number;  // Running total for cupid system
  baseline_total_cost?: number;  // Running total for baseline system
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

// --- BUDGET RANDOM RANGE ---
// Cost: 0.5 to 1.5, Rounds: 5 to 15 (randomly selected)
function sampleBudget(): BudgetConstraints {
  const minCost = 0.5;
  const maxCost = 1.5;
  const minRounds = 5;
  const maxRounds = 15;

  // Random cost between 0.5 and 1.5 (rounded to 2 decimal places)
  const randomCost = Math.round((minCost + Math.random() * (maxCost - minCost)) * 100) / 100;
  // Random rounds between 5 and 15 (integer)
  const randomRounds = Math.floor(minRounds + Math.random() * (maxRounds - minRounds + 1));

  return { maxRounds: randomRounds, maxCost: randomCost };
}

// --- CONSTRAINT SAMPLING (Based on Constraint_Picker.ipynb) ---
// Objective columns that can be used for constraints (matches model card attributes)
const OBJECTIVE_COLUMNS = [
  { key: 'intelligence', displayName: 'Intelligence Rating', unit: '', isCost: false },
  { key: 'speed', displayName: 'Speed Rating', unit: '', isCost: false },
  { key: 'reasoning', displayName: 'Reasoning Capability', unit: '', isCost: false, isBoolean: true },
  { key: 'input_price', displayName: 'Input Price', unit: '$/1M tokens', isCost: true },
  { key: 'output_price', displayName: 'Output Price', unit: '$/1M tokens', isCost: true },
  { key: 'context_window', displayName: 'Context Window', unit: 'tokens', isCost: false },
  { key: 'max_output', displayName: 'Max Output', unit: 'tokens', isCost: false },
];

// Model pool data type
interface ModelPoolStats {
  id: number;
  intelligence: number | null;
  speed: number | null;
  reasoning: number | null;
  input_price: number | null;
  output_price: number | null;
  context_window: number | null;
  max_output: number | null;
}

// Sample constraints using min-max rule from Constraint_Picker.ipynb
function sampleConstraintsDynamic(modelPoolData: ModelPoolStats[], nObjectives: number = 3, kSamples: number = 5): Constraint[] {
  const constraints: Constraint[] = [];

  if (modelPoolData.length === 0) return constraints;

  // Shuffle and pick n objective columns
  const shuffled = [...OBJECTIVE_COLUMNS].sort(() => Math.random() - 0.5);
  const selectedCols = shuffled.slice(0, Math.min(nObjectives, shuffled.length));

  for (const col of selectedCols) {
    // Get all values for this column from model pool
    const values = modelPoolData.map(m => (m as any)[col.key]).filter(v => v !== undefined && v !== null);

    if (values.length === 0) continue;

    // Sample k values
    const samples: number[] = [];
    for (let i = 0; i < kSamples; i++) {
      samples.push(values[Math.floor(Math.random() * values.length)]);
    }

    // For cost columns: use MIN (user wants low cost, so we set upper bound)
    // For benefit columns: use MAX (user wants high value, so we set lower bound)
    let chosenValue: number | boolean;
    let operator: '>=' | '<=' | '==';

    if (col.isBoolean) {
      // For boolean, randomly decide if required or not
      chosenValue = Math.random() > 0.5;
      operator = '==';
    } else if (col.isCost) {
      // Cost: user wants <= this value
      chosenValue = Math.min(...samples);
      operator = '<=';
    } else {
      // Benefit: user wants >= this value
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
}

function formatConstraint(c: Constraint): string {
  const opMap: Record<string, string> = { '>=': '≥', '<=': '≤', '==': '=', '>': '>' };

  // For boolean constraints, show cleaner format
  if (typeof c.value === 'boolean') {
    if (c.value === true) {
      return `${c.displayName}: Required`;
    } else {
      return `${c.displayName}: Not Required`;
    }
  }

  const valueStr = typeof c.value === 'number' ? c.value.toLocaleString() : String(c.value);
  return `${c.displayName} ${opMap[c.operator]} ${valueStr}${c.unit ? ` ${c.unit}` : ''}`;
}

// --- BUDGET COMPLIANCE RATING LABELS ---
const BUDGET_RATING_LABELS = [
  { value: 1, label: 'Heavily exceeded the budget', color: 'text-red-600' },
  { value: 2, label: 'Somewhat exceeded the budget', color: 'text-orange-600' },
  { value: 3, label: 'Just right', color: 'text-green-600' },
  { value: 4, label: 'Saved some money', color: 'text-blue-600' },
  { value: 5, label: 'Very efficient cost-wise', color: 'text-purple-600' },
];

// --- TUTORIAL STEPS ---
const INTERACTION_TUTORIAL_STEPS = [
  {
    title: 'Welcome to the Interaction Stage',
    description: 'In this stage, you will ask questions and interact with the system. Based on your inputs, the system will match you with a suitable model.',
    icon: '🎯'
  },
  {
    title: 'Selecting Your Preferred Output',
    description: 'For each query, you\'ll see two responses from different models. Click on the one you prefer. Your choices help the system learn your preferences.',
    icon: '👆'
  },
  {
    title: 'Optional Language Feedback',
    description: 'You can provide feedback like "give me a cheaper model" or "I need smarter responses" to guide the system toward your ideal model.',
    icon: '💬'
  },
  {
    title: 'Review Model Specification', // UPDATED TERM
    description: 'Check the model specifications (intelligence, speed, price, etc.) to see if they match your requirements. This helps you make informed decisions.',
    icon: '📋'
  },
  {
    title: 'Confirm When Satisfied',
    description: 'When you\'re happy with the model selection, click "I\'m Satisfied" to proceed to the testing phase. You can end early after the first round.',
    icon: '✅'
  }
];

const TESTING_TUTORIAL_STEPS = [
  {
    title: 'Welcome to the Testing Stage',
    description: 'Now you\'ll test both matched models side-by-side. The same prompt goes to both systems so you can compare their outputs directly.',
    icon: '🔬'
  },
  {
    title: 'Focus on Output Quality',
    description: 'Pay attention to accuracy, helpfulness, clarity, and relevance of responses. This will help you decide which system found a better model for you.',
    icon: '⭐'
  },
  {
    title: 'Compare Side-by-Side',
    description: 'Each prompt shows System A and System B responses next to each other. You have up to 10 rounds to test thoroughly.',
    icon: '↔️'
  },
  {
    title: 'Proceed to Rating',
    description: 'After testing, you\'ll rate both systems on model quality and budget compliance. Make sure to test enough to form a clear opinion!',
    icon: '📊'
  }
];

// --- EXPERT SUBJECTS ---
const EXPERT_SUBJECTS = [
  { id: 'science', label: 'Science & Technology', icon: <BookOpen className="text-blue-500" size={24} /> },
  { id: 'business', label: 'Business & Finance', icon: <DollarSign className="text-green-500" size={24} /> },
  { id: 'medical', label: 'Healthcare & Medicine', icon: <Heart className="text-red-500" size={24} /> },
  { id: 'engineering', label: 'Engineering', icon: <Settings className="text-orange-500" size={24} /> },
  { id: 'law', label: 'Law & Legal', icon: <BookOpen className="text-purple-500" size={24} /> },
];

const EDUCATION_LEVELS = [
  "High School", "Some College", "Bachelor's Degree", "Master's Degree", "PhD / Doctorate", "Other"
];

const MAJORS = [
  "Computer Science / IT", "Engineering", "Business / Finance", "Arts & Design",
  "Social Sciences", "Natural Sciences", "Healthcare", "Education", "Other"
];

// --- OPEN TESTING CONFIG ---
const OPEN_TESTING_MAX_ROUNDS = 10;

// --- MAIN COMPONENT ---
const App: React.FC = () => {
  const [phase, setPhase] = useState<Phase>('consent');
  const [sessionId, setSessionId] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  // Model pool data fetched from backend for constraint sampling
  const [modelPoolData, setModelPoolData] = useState<ModelPoolStats[]>([]);
  const [modelPoolLoading, setModelPoolLoading] = useState<boolean>(true);

  const [personaGroup, setPersonaGroup] = useState<PersonaGroup | null>(null);
  const [selectedExpertSubject, setSelectedExpertSubject] = useState<string | null>(null);
  const [assignedConstraints, setAssignedConstraints] = useState<Constraint[]>([]);

  const [demographics, setDemographics] = useState<Demographics>({
    age: '',
    education: EDUCATION_LEVELS[2],
    major: MAJORS[0],
    familiarity: 3
  });
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
  const [feedbackA, setFeedbackA] = useState<string>('');
  const [feedbackB, setFeedbackB] = useState<string>('');
  const [roundHistory, setRoundHistory] = useState<RoundHistory[]>([]);

  const [showModelInfo, setShowModelInfo] = useState<{ system: 'cupid' | 'baseline', side: 'left' | 'right' } | null>(null);

  const [openTestMessages, setOpenTestMessages] = useState<OpenTestMessage[]>([]);
  const [openTestInput, setOpenTestInput] = useState<string>('');
  const [openTestSystem, setOpenTestSystem] = useState<'A' | 'B'>('A');
  const [openTestLoading, setOpenTestLoading] = useState<boolean>(false);
  const [openTestRoundsA, setOpenTestRoundsA] = useState<number>(0);
  const [openTestRoundsB, setOpenTestRoundsB] = useState<number>(0);

  // Side-by-side comparison state for open testing
  interface SideBySideRound {
    prompt: string;
    responseA: string;
    responseB: string;
    costA: number;
    costB: number;
  }
  const [sideBySideRounds, setSideBySideRounds] = useState<SideBySideRound[]>([]);

  const [evalRatingA, setEvalRatingA] = useState<number>(0);
  const [evalRatingB, setEvalRatingB] = useState<number>(0);
  const [evalBudgetRatingA, setEvalBudgetRatingA] = useState<number>(0);
  const [evalBudgetRatingB, setEvalBudgetRatingB] = useState<number>(0);
  const [evalComment, setEvalComment] = useState<string>('');
  const [finished, setFinished] = useState<boolean>(false);

  // Tutorial state
  const [showInteractionTutorial, setShowInteractionTutorial] = useState<boolean>(false);
  const [showTestingTutorial, setShowTestingTutorial] = useState<boolean>(false);
  const [tutorialStep, setTutorialStep] = useState<number>(0);
  const [hasSeenInteractionTutorial, setHasSeenInteractionTutorial] = useState<boolean>(false);
  const [hasSeenTestingTutorial, setHasSeenTestingTutorial] = useState<boolean>(false);

  // Gate for rating - must have tested at least once
  const [hasTestedModels, setHasTestedModels] = useState<boolean>(false);

  // Download reminder state
  const [showDownloadReminder, setShowDownloadReminder] = useState<boolean>(false);
  const [hasDownloaded, setHasDownloaded] = useState<boolean>(false);

  // Fetch model pool data from backend on mount
  useEffect(() => {
    const fetchModelPool = async () => {
      try {
        const response = await fetch(`${API_URL}/model-pool-stats`);
        if (response.ok) {
          const data = await response.json();
          setModelPoolData(data.models || []);
        }
      } catch (err) {
        console.error('Failed to fetch model pool stats:', err);
      } finally {
        setModelPoolLoading(false);
      }
    };
    fetchModelPool();
  }, []);

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [phase, arenaState?.round, init]);

  const fetchNextRound = useCallback(async (isFirst: boolean = false, currentPrompt?: string) => {
    setLoading(true);
    setError(null);
    const promptToUse = currentPrompt || prompt;

    try {
      const payload: any = {
        session_id: isFirst ? null : sessionId,
        prompt: promptToUse,
        previous_vote: null,
        feedback_text: feedbackA || '',
      };

      if (!isFirst && cupidVote) payload.cupid_vote = cupidVote;
      if (!isFirst && baselineVote) payload.baseline_vote = baselineVote;

      if (isFirst) {
        payload.budget_cost = budgetConstraints.maxCost;
        payload.budget_rounds = budgetConstraints.maxRounds;
        payload.persona_group = personaGroup;
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
        const historyEntry: RoundHistory = {
          round: arenaState.round,
          prompt: prompt,
          cupid_left_id: arenaState.cupid_pair.left.model_id,
          cupid_right_id: arenaState.cupid_pair.right.model_id,
          cupid_vote: cupidVote,
          baseline_left_id: arenaState.baseline_pair.left.model_id,
          baseline_right_id: arenaState.baseline_pair.right.model_id,
          baseline_vote: baselineVote,
          feedback: feedbackA,
          timestamp: new Date().toISOString(),
          // Cost details
          cupid_left_cost: arenaState.cupid_pair.left.cost,
          cupid_right_cost: arenaState.cupid_pair.right.cost,
          baseline_left_cost: arenaState.baseline_pair.left.cost,
          baseline_right_cost: arenaState.baseline_pair.right.cost,
          routing_cost: arenaState.routing_cost,
          cupid_total_cost: arenaState.cupid_cost,
          baseline_total_cost: arenaState.baseline_cost,
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
      setFeedbackA('');
      setFeedbackB('');

    } catch (err) {
      console.error("Failed to fetch round:", err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(`Error: ${errorMessage}. Please check your connection and try again.`);
    } finally {
      setLoading(false);
    }
  }, [prompt, sessionId, cupidVote, baselineVote, feedbackA, budgetConstraints, personaGroup, selectedExpertSubject, assignedConstraints, demographics, arenaState]);

  const sendOpenTestMessage = async () => {
    if (!openTestInput.trim() || openTestLoading) return;

    if (openTestSystem === 'A' && openTestRoundsA >= OPEN_TESTING_MAX_ROUNDS) {
      setError(`System A has reached the maximum of ${OPEN_TESTING_MAX_ROUNDS} rounds.`);
      return;
    }
    if (openTestSystem === 'B' && openTestRoundsB >= OPEN_TESTING_MAX_ROUNDS) {
      setError(`System B has reached the maximum of ${OPEN_TESTING_MAX_ROUNDS} rounds.`);
      return;
    }

    const userMsg: OpenTestMessage = { role: 'user', content: openTestInput, system: openTestSystem };
    setOpenTestMessages(prev => [...prev, userMsg]);
    setOpenTestInput('');
    setOpenTestLoading(true);
    setError(null);

    if (openTestSystem === 'A') setOpenTestRoundsA(prev => prev + 1);
    else setOpenTestRoundsB(prev => prev + 1);

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
          content: '[Demo mode: Chat endpoint not connected.]',
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

  // Send to BOTH systems for side-by-side comparison
  const sendSideBySideMessage = async () => {
    if (!openTestInput.trim() || openTestLoading) return;

    const totalRounds = sideBySideRounds.length;
    if (totalRounds >= OPEN_TESTING_MAX_ROUNDS) {
      setError(`Maximum of ${OPEN_TESTING_MAX_ROUNDS} comparison rounds reached.`);
      return;
    }

    const currentPrompt = openTestInput;
    setOpenTestInput('');
    setOpenTestLoading(true);
    setError(null);

    try {
      // Send to both systems in parallel
      const [responseA, responseB] = await Promise.all([
        fetch(`${API_URL}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionId,
            message: currentPrompt,
            system: 'cupid'
          }),
        }),
        fetch(`${API_URL}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionId,
            message: currentPrompt,
            system: 'baseline'
          }),
        })
      ]);

      let textA = '[Connection error]';
      let textB = '[Connection error]';
      let costA = 0;
      let costB = 0;

      if (responseA.ok) {
        const dataA = await responseA.json();
        textA = dataA.response || 'No response received';
        costA = dataA.cost || 0;
      }
      if (responseB.ok) {
        const dataB = await responseB.json();
        textB = dataB.response || 'No response received';
        costB = dataB.cost || 0;
      }

      setSideBySideRounds(prev => [...prev, {
        prompt: currentPrompt,
        responseA: textA,
        responseB: textB,
        costA,
        costB
      }]);

      setOpenTestRoundsA(prev => prev + 1);
      setOpenTestRoundsB(prev => prev + 1);
      setHasTestedModels(true); // Mark that user has tested the models

    } catch (e) {
      console.error('Chat error:', e);
      setError('Connection error. Please try again.');
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
          persona_group: personaGroup,
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
          terminated_early: roundHistory.length < budgetConstraints.maxRounds,
          open_test_rounds_a: openTestRoundsA,
          open_test_rounds_b: openTestRoundsB,
          side_by_side_rounds: sideBySideRounds.length,
        })
      });
    } catch (e) {
      console.error('Failed to save session:', e);
    }
  }, [sessionId, demographics, personaGroup, selectedExpertSubject, assignedConstraints, budgetConstraints, roundHistory, evalRatingA, evalRatingB, evalBudgetRatingA, evalBudgetRatingB, evalComment, arenaState, openTestRoundsA, openTestRoundsB, sideBySideRounds]);

  const handleConsent = () => {
    const newSessionId = `sess_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setSessionId(newSessionId);
    const assignedBudget = sampleBudget();
    setBudgetConstraints(assignedBudget);
    setPhase('calibration');
  };

  const handlePersonaGroupSelect = (group: PersonaGroup) => {
    setPersonaGroup(group);

    if (group === 'traditional') {
      // Sample 2-4 constraints using backend model pool data
      const numConstraints = 2 + Math.floor(Math.random() * 3); // 2, 3, or 4
      const constraints = sampleConstraintsDynamic(modelPoolData, numConstraints, 5);
      setAssignedConstraints(constraints);
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
    setInit(true);
    setPhase('interaction');
  };

  const handleFinalSubmit = async () => {
    await saveSessionData();
    setFinished(true);
  };

  const handleDownloadAndFinish = () => {
    // Generate JSON file
    const cupidFinalModelId = arenaState?.final_model_a?.model_id;
    const baselineFinalModelId = arenaState?.final_model_b?.model_id;

    // Helper to get stats safely
    const getStats = (id?: number) => modelPoolData.find(m => m.id === id);

    const cupidFinalStats = getStats(cupidFinalModelId);
    const baselineFinalStats = getStats(baselineFinalModelId);

    // Calculate costs
    const interactionPhaseCupidCost = (arenaState?.cupid_cost || 0);
    const interactionPhaseBaselineCost = roundHistory.length > 0 ? (roundHistory[roundHistory.length - 1].baseline_total_cost || 0) : 0;
    const interactionPhaseRoutingCost = roundHistory.reduce((sum, r) => sum + (r.routing_cost || 0), 0);

    // Open testing costs
    const openTestCostA = sideBySideRounds.reduce((sum, r) => sum + r.costA, 0);
    const openTestCostB = sideBySideRounds.reduce((sum, r) => sum + r.costB, 0);

    const results = {
      session_id: sessionId,
      timestamp: new Date().toISOString(),
      demographics,
      persona_group: personaGroup,
      expert_subject: selectedExpertSubject,
      constraints: assignedConstraints,
      budget: budgetConstraints,

      // Comprehensive final state
      final_state: {
        system_a: {
          label: 'System A (CUPID)',
          algorithm: 'CUPID (Pairwise GP with language feedback routing)',
          final_model_id: cupidFinalModelId,
          final_model_name: arenaState?.final_model_a?.model_name || null,
          final_model_stats: cupidFinalStats || null,
          interaction_phase_cost: interactionPhaseCupidCost,
          routing_cost_total: interactionPhaseRoutingCost,
          interaction_phase_cost_with_routing: interactionPhaseCupidCost + interactionPhaseRoutingCost,
          open_test_rounds: sideBySideRounds.length,
          open_test_cost: openTestCostA,
          total_cost: interactionPhaseCupidCost + interactionPhaseRoutingCost + openTestCostA,
          total_rounds: roundHistory.length,
        },
        system_b: {
          label: 'System B (Baseline)',
          algorithm: 'Bradley-Terry Baseline',
          final_model_id: baselineFinalModelId,
          final_model_name: arenaState?.final_model_b?.model_name || null,
          final_model_stats: baselineFinalStats || null,
          interaction_phase_cost: interactionPhaseBaselineCost,
          open_test_rounds: sideBySideRounds.length,
          open_test_cost: openTestCostB,
          total_cost: interactionPhaseBaselineCost + openTestCostB,
          total_rounds: roundHistory.length,
        }
      },

      history: roundHistory,
      open_testing_chat: sideBySideRounds,
      evaluation: {
        quality_rating_a: evalRatingA,
        quality_rating_b: evalRatingB,
        budget_rating_a: evalBudgetRatingA,
        budget_rating_b: evalBudgetRatingB,
        comment: evalComment
      }
    };

    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `study_results_${sessionId}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setHasDownloaded(true);
  };

  const renderModelInfoModal = () => {
    if (!showModelInfo || !arenaState) return null;

    // UPDATED: Modal visuals to be cleaner
    const pair = showModelInfo.system === 'cupid' ? arenaState.cupid_pair : arenaState.baseline_pair;
    const stats = showModelInfo.side === 'left' ? pair.left_stats : pair.right_stats;
    const modelName = showModelInfo.side === 'left' ? pair.left.model_name : pair.right.model_name;

    if (!stats) return null;

    return (
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={() => setShowModelInfo(null)}>
        <div className="bg-white rounded-xl shadow-2xl max-w-md w-full overflow-hidden animate-in fade-in zoom-in duration-200" onClick={e => e.stopPropagation()}>
          <div className="bg-gradient-to-r from-gray-800 to-gray-900 p-4 text-white flex justify-between items-center">
            <div>
              <h3 className="font-bold text-lg">Model Specification</h3> {/* UPDATED TERM */}
              <p className="text-xs text-gray-400 font-mono">{modelName}</p>
            </div>
            <button onClick={() => setShowModelInfo(null)} className="hover:bg-white/20 p-1 rounded transition">
              <X size={20} />
            </button>
          </div>

          <div className="p-6 space-y-4">
            <div className="bg-purple-50 p-4 rounded-lg">
              <h4 className="font-bold text-gray-700 mb-3">Performance Ratings</h4>
              <div className="grid grid-cols-3 gap-3">
                <div className="text-center"><div className="text-2xl font-bold text-purple-600">{stats.intelligence ?? '—'}</div><div className="text-xs text-gray-500">Intelligence</div></div>
                <div className="text-center"><div className="text-2xl font-bold text-blue-600">{stats.speed ?? '—'}</div><div className="text-xs text-gray-500">Speed</div></div>
                <div className="text-center"><div className="text-2xl font-bold text-indigo-600">{stats.reasoning ? 'Yes' : 'No'}</div><div className="text-xs text-gray-500">Reasoning</div></div>
              </div>
            </div>

            {/* UPDATED PRICING SECTION */}
            <div className="bg-green-50 p-4 rounded-lg">
              <h4 className="font-bold text-gray-700 mb-3">Pricing</h4>
              <div className="grid grid-cols-1 gap-2">
                <div className="flex justify-between items-center border-b border-green-100 pb-2">
                  <div className="text-sm text-gray-600">Input Price</div>
                  <div className="text-right">
                    <span className="text-lg font-bold text-green-600">${stats.input_price ?? '—'}</span>
                    <span className="text-xs text-gray-500 ml-1">/ 1M tokens</span>
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <div className="text-sm text-gray-600">Output Price</div>
                  <div className="text-right">
                    <span className="text-lg font-bold text-green-700">${stats.output_price ?? '—'}</span>
                    <span className="text-xs text-gray-500 ml-1">/ 1M tokens</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-orange-50 p-4 rounded-lg">
              <h4 className="font-bold text-gray-700 mb-3">Capacity</h4>
              <div className="grid grid-cols-2 gap-3">
                <div><div className="text-lg font-bold text-orange-600">{stats.context_window?.toLocaleString() ?? '—'}</div><div className="text-xs text-gray-500">Context Window</div></div>
                <div><div className="text-lg font-bold text-orange-700">{stats.max_output?.toLocaleString() ?? '—'}</div><div className="text-xs text-gray-500">Max Output</div></div>
              </div>
            </div>

            <button onClick={() => setShowModelInfo(null)} className="w-full py-3 bg-gray-100 font-bold text-gray-700 rounded-lg hover:bg-gray-200 transition">
              Close
            </button>
          </div>
        </div>
      </div>
    );
  };

  const renderModelCard = (side: 'left' | 'right', data: ModelResponse, vote: 'left' | 'right' | null, setVote: (v: 'left' | 'right') => void, color: string, system: 'cupid' | 'baseline') => {
    const isSelected = vote === side;
    const label = side === 'left' ? '1' : '2';
    // Access stats safely
    const pair = system === 'cupid' ? arenaState?.cupid_pair : arenaState?.baseline_pair;
    const stats = side === 'left' ? pair?.left_stats : pair?.right_stats;

    return (
      <div
        className={`relative border-2 rounded-xl p-4 transition-all duration-200 flex flex-col h-full bg-white ${isSelected ? `border-${color}-500 shadow-lg bg-${color}-50` : 'border-gray-200 hover:border-gray-300'}`}
      >
        <div className="flex justify-between items-start mb-3">
          <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-white ${isSelected ? `bg-${color}-600` : 'bg-gray-400'}`}>
            {isSelected ? <CheckCircle size={18} /> : label}
          </div>
          <button
            onClick={(e) => { e.stopPropagation(); setShowModelInfo({ system, side }); }}
            className="text-gray-400 hover:text-gray-600 p-1 rounded hover:bg-gray-100 flex items-center gap-1 text-xs"
            title="View Model Specification" // UPDATED TERM
          >
            <Info size={16} /> Specification
          </button>
        </div>

        <div className={`flex-grow rounded-lg p-3 mb-3 text-sm leading-relaxed overflow-hidden ${isSelected ? 'bg-white' : 'bg-gray-50'} ${isSelected && color === 'blue' ? 'text-blue-900' : ''} ${isSelected ? 'ring-2 ring-offset-1 ring-' + color + '-200 scale-[1.01]' : ''}`}>
          <div className="flex justify-between items-center mb-2">
            <span className="text-xs font-bold text-gray-500">Output {label}</span>
            {/* Cost with explanation tooltip */}
            <div className="group relative">
              <span className="text-xs text-gray-400 cursor-help flex items-center gap-1">
                <DollarSign size={12} /> {(data.cost ?? 0).toFixed(5)} <Info size={10} className="text-gray-300" />
              </span>
              <div className="absolute right-0 top-full mt-1 w-48 bg-gray-800 text-white text-xs rounded-lg p-2 hidden group-hover:block z-10 shadow-lg">
                Cost for this response based on input/output tokens and model pricing (per 1M tokens).
              </div>
            </div>
          </div>

          {/* Markdown rendered content */}
          <div onClick={() => setVote(side)} className="flex-grow cursor-pointer overflow-y-auto h-48 md:h-auto md:max-h-64 mb-3">
            {data.text ? (
              <Markdown content={data.text} />
            ) : (
              <span className="text-gray-400 italic">No response</span>
            )}
          </div>

          {/* Model Info - Expanded by default for Traditional Group */}
          {personaGroup === 'traditional' && (
            <div className="border-t pt-3 mt-2">
              <div className="text-xs font-bold text-gray-600 mb-2 flex items-center gap-1">
                <Info size={12} /> Model Specification {/* UPDATED TERM */}
              </div>
              {stats ? (
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="bg-purple-50 p-2 rounded">
                    <span className="text-gray-500">Intelligence:</span>
                    <span className="font-bold text-purple-700 ml-1">{stats.intelligence ?? '—'}</span>
                  </div>
                  <div className="bg-blue-50 p-2 rounded">
                    <span className="text-gray-500">Speed:</span>
                    <span className="font-bold text-blue-700 ml-1">{stats.speed ?? '—'}</span>
                  </div>
                  <div className="bg-green-50 p-2 rounded col-span-2">
                    <span className="text-gray-500">Price:</span>
                    <span className="font-bold text-green-700 ml-1">${stats.input_price} / ${stats.output_price}</span>
                  </div>
                </div>
              ) : (
                <div className="text-xs text-gray-400 italic">Model specifications not available</div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  // --- RENDERING ---

  // 1. CONSENT & LANDING
  if (phase === 'consent') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-indigo-900 to-blue-900 flex items-center justify-center p-4">
        <div className="max-w-2xl bg-white rounded-3xl shadow-2xl overflow-hidden">
          <div className="p-8 md:p-12">
            <div className="flex justify-center mb-6">
              <div className="bg-blue-100 p-4 rounded-full">
                <Brain className="text-blue-600 w-12 h-12" />
              </div>
            </div>
            <h1 className="text-3xl font-bold text-center text-gray-900 mb-4">AI Model Selection Study</h1>
            <p className="text-gray-600 text-center mb-8 text-lg">
              Help us understand how people choose AI models. You will interact with two AI systems and evaluate their recommendations.
            </p>

            <div className="space-y-4 mb-8">
              <div className="flex items-start gap-4">
                <div className="bg-purple-100 p-2 rounded-lg text-purple-600 font-bold">1</div>
                <div>
                  <h3 className="font-bold text-gray-900">Interact</h3>
                  <p className="text-sm text-gray-600">Ask questions and provide feedback to find the best model.</p>
                </div>
              </div>
              <div className="flex items-start gap-4">
                <div className="bg-pink-100 p-2 rounded-lg text-pink-600 font-bold">2</div>
                <div>
                  <h3 className="font-bold text-gray-900">Test</h3>
                  <p className="text-sm text-gray-600">Compare the recommended models side-by-side.</p>
                </div>
              </div>
              <div className="flex items-start gap-4">
                <div className="bg-green-100 p-2 rounded-lg text-green-600 font-bold">3</div>
                <div>
                  <h3 className="font-bold text-gray-900">Rate</h3>
                  <p className="text-sm text-gray-600">Evaluate the quality and cost-efficiency of the systems.</p>
                </div>
              </div>
            </div>

            <button
              onClick={handleConsent}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-4 rounded-xl text-lg transition-all transform hover:scale-[1.02] shadow-lg flex items-center justify-center gap-2"
            >
              Start Study <ArrowRight />
            </button>
            <p className="text-xs text-gray-400 text-center mt-4">
              By clicking Start, you consent to participate in this anonymous research study.
            </p>
          </div>
        </div>
      </div>
    );
  }

  // 2. CALIBRATION (Demographics & Persona)
  if (phase === 'calibration') {
    return (
      <div className="min-h-screen bg-gray-50 py-12 px-4">
        <div className="max-w-xl mx-auto space-y-6">
          <div className="bg-white p-6 rounded-2xl shadow-lg">
            <h2 className="text-xl font-bold mb-4">About You</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Age</label>
                <input type="number" className="w-full border rounded-lg p-3" value={demographics.age} onChange={e => setDemographics({ ...demographics, age: parseInt(e.target.value) || '' })} placeholder="Enter your age" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Education Level</label>
                <select className="w-full border rounded-lg p-3" value={demographics.education} onChange={e => setDemographics({ ...demographics, education: e.target.value })}>
                  {EDUCATION_LEVELS.map(l => <option key={l} value={l}>{l}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Field of Study / Profession</label>
                <select className="w-full border rounded-lg p-3" value={demographics.major} onChange={e => setDemographics({ ...demographics, major: e.target.value })}>
                  {MAJORS.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Familiarity with AI (1-5)</label>
                <div className="flex items-center gap-2 text-sm text-gray-500"><span>Rarely</span><input type="range" min="1" max="5" className="flex-grow" value={demographics.familiarity} onChange={e => setDemographics({ ...demographics, familiarity: parseInt(e.target.value) })} /><span>Daily</span></div></div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-2xl shadow-lg">
            <h2 className="text-xl font-bold mb-4">Choose Your Group *</h2>
            <div className="space-y-3">
              <button
                onClick={() => handlePersonaGroupSelect('traditional')}
                className={`w-full p-4 rounded-xl border-2 text-left transition-all ${personaGroup === 'traditional' ? 'border-purple-500 bg-purple-50' : 'border-gray-200 hover:border-gray-300'}`}
              >
                <div className="flex items-center gap-3">
                  <Settings className={`${personaGroup === 'traditional' ? 'text-purple-600' : 'text-gray-400'}`} size={24} />
                  <div>
                    <div className="font-bold text-gray-800">Traditional User</div>
                    <div className="text-sm text-gray-500">I have specific constraints (e.g., budget, speed).</div>
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
                    <div className="font-bold text-gray-800">Expert User</div>
                    <div className="text-sm text-gray-500">I want the best model for my specific field.</div>
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
                    <div className="font-bold text-gray-800">Preference User</div>
                    <div className="text-sm text-gray-500">I want a model that matches my personal taste.</div>
                  </div>
                </div>
              </button>
            </div>

            {personaGroup === 'expert' && (
              <div className="mt-4 animate-in fade-in slide-in-from-top-2">
                <label className="block text-sm font-bold text-gray-700 mb-2">Select Your Expertise:</label>
                <div className="grid grid-cols-1 gap-2">
                  {EXPERT_SUBJECTS.map(sub => (
                    <button
                      key={sub.id}
                      onClick={() => setSelectedExpertSubject(sub.id)}
                      className={`flex items-center gap-3 p-3 rounded-lg border text-left transition ${selectedExpertSubject === sub.id ? 'bg-blue-100 border-blue-400 ring-1 ring-blue-400' : 'bg-gray-50 border-gray-200 hover:bg-gray-100'}`}
                    >
                      {sub.icon}
                      <span className="text-sm font-medium">{sub.label}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Show assigned constraints if traditional */}
            {personaGroup === 'traditional' && assignedConstraints.length > 0 && (
              <div className="mt-4 p-4 bg-purple-50 rounded-lg border border-purple-100 animate-in fade-in">
                <h3 className="text-sm font-bold text-purple-800 mb-2">Assigned Constraints (Randomly Generated):</h3>
                <ul className="text-sm space-y-1">
                  {assignedConstraints.map((c, i) => (
                    <li key={i} className="flex items-center gap-2">
                      <CheckCircle size={14} className="text-purple-600" />
                      <span>{formatConstraint(c)}</span>
                    </li>
                  ))}
                </ul>
                <p className="text-xs text-purple-600 mt-2 italic">You will need to find a model that meets these requirements.</p>
              </div>
            )}
          </div>

          {error && (
            <div className="bg-red-50 text-red-700 p-4 rounded-lg flex items-center gap-2">
              <AlertCircle size={20} />
              {error}
            </div>
          )}

          <button
            onClick={handleCalibrationSubmit}
            disabled={modelPoolLoading}
            className="w-full bg-black text-white py-4 rounded-xl font-bold text-lg hover:bg-gray-800 transition disabled:opacity-50"
          >
            {modelPoolLoading ? 'Loading...' : 'Start Interaction'}
          </button>
        </div>
      </div>
    );
  }

  // 3. INTERACTION PHASE
  if (phase === 'interaction') {
    // Show tutorial overlay if requested
    if (showInteractionTutorial) {
      const step = INTERACTION_TUTORIAL_STEPS[tutorialStep];
      return (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl max-w-md w-full p-6 shadow-2xl animate-in zoom-in-95 duration-200">
            <div className="flex justify-center mb-4 text-4xl">{step.icon}</div>
            <h2 className="text-xl font-bold text-center mb-2">{step.title}</h2>
            <p className="text-gray-600 text-center mb-6">{step.description}</p>
            <div className="flex gap-3">
              <button
                onClick={() => {
                  if (tutorialStep < INTERACTION_TUTORIAL_STEPS.length - 1) {
                    setTutorialStep(prev => prev + 1);
                  } else {
                    setShowInteractionTutorial(false);
                    setHasSeenInteractionTutorial(true);
                  }
                }}
                className="flex-grow bg-blue-600 text-white py-3 rounded-xl font-bold hover:bg-blue-700 transition"
              >
                {tutorialStep < INTERACTION_TUTORIAL_STEPS.length - 1 ? 'Next' : 'Got it!'}
              </button>
            </div>
            <div className="flex justify-center gap-1 mt-4">
              {INTERACTION_TUTORIAL_STEPS.map((_, i) => (
                <div key={i} className={`w-2 h-2 rounded-full ${i === tutorialStep ? 'bg-blue-600' : 'bg-gray-300'}`} />
              ))}
            </div>
          </div>
        </div>
      );
    }

    if (init && !arenaState) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
          <div className="max-w-lg w-full p-8 bg-white shadow-xl rounded-2xl text-center">
            {/* Stage Introduction */}
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl p-4 mb-6 text-left">
              <div className="flex items-center gap-2 mb-2">
                <Target className="text-blue-600" size={20} />
                <span className="font-bold text-blue-800">Stage 1: Interaction</span>
              </div>
              <p className="text-sm text-blue-700">
                You will ask questions and interact with the system. Based on your inputs and preferences, the system will match you with a suitable model.
              </p>
            </div>

            <Brain className="mx-auto text-blue-600 mb-4" size={48} />
            <h1 className="text-2xl font-bold mb-2">Ready to Begin</h1>

            {/* REMOVED BUDGET DISPLAY AS REQUESTED */}
            {/* <p className="text-sm text-gray-500 mb-4">Budget: ${budgetConstraints.maxCost} • Up to {budgetConstraints.maxRounds} rounds</p> */}

            {/* Tutorial button */}
            <button
              onClick={() => { setShowInteractionTutorial(true); setTutorialStep(0); }}
              className="mb-4 text-sm text-blue-600 hover:text-blue-800 underline flex items-center justify-center gap-1 mx-auto"
            >
              <HelpCircle size={14} /> View Tutorial Again
            </button>

            {personaGroup === 'traditional' && assignedConstraints.length > 0 && (
              <div className="mb-4 p-4 bg-purple-50 rounded-lg text-left border border-purple-200">
                <p className="text-sm font-bold text-purple-800 mb-2">Your Model Requirements:</p>
                <ul className="text-xs text-purple-700 space-y-1">
                  {assignedConstraints.map((c, i) => (
                    <li key={i}>• {formatConstraint(c)}</li>
                  ))}
                </ul>
              </div>
            )}

            <div className="space-y-4 text-left">
              <div>
                <label className="block text-sm font-bold text-gray-700 mb-1">Enter your first question/prompt:</label>
                <textarea
                  className="w-full border rounded-lg p-3 h-24 focus:ring-2 focus:ring-blue-500 outline-none"
                  placeholder="E.g., Write a python script to sort a list..."
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                />
              </div>
              <button
                onClick={() => fetchNextRound(true)}
                disabled={loading || !prompt.trim()}
                className="w-full bg-blue-600 text-white py-3 rounded-lg font-bold hover:bg-blue-700 transition disabled:opacity-50"
              >
                {loading ? 'Starting...' : 'Start Interaction'}
              </button>
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="min-h-screen bg-gray-50 flex flex-col md:flex-row relative">
        {renderModelInfoModal()}

        {/* Sidebar */}
        <div className="w-full md:w-64 bg-white border-r border-gray-200 p-4 flex flex-col gap-4 overflow-y-auto h-48 md:h-screen sticky top-0 md:relative">
          <div className="mb-2">
            <h1 className="font-bold text-xl text-gray-900 flex items-center gap-2"><Sparkles className="text-yellow-500" /> Arena</h1>
            <p className="text-xs text-gray-500">Round {arenaState?.round} / {budgetConstraints.maxRounds}</p>
          </div>

          <div className="p-3 bg-gray-100 rounded-lg">
            <div className="text-xs font-bold text-gray-500 uppercase mb-2">Your Task</div>
            <div className="text-sm font-medium">
              {personaGroup === 'traditional' ? 'Find a model satisfying constraints' :
                personaGroup === 'expert' ? `Find the best model for ${EXPERT_SUBJECTS.find(s => s.id === selectedExpertSubject)?.label}` :
                  'Find your preferred model'}
            </div>
          </div>

          {personaGroup === 'traditional' && (
            <div className="p-3 bg-purple-50 border border-purple-100 rounded-lg">
              <div className="text-xs font-bold text-purple-800 uppercase mb-2">Constraints</div>
              <ul className="text-xs space-y-2">
                {assignedConstraints.map((c, i) => (
                  <li key={i} className="flex items-start gap-1">
                    <span className="mt-0.5">•</span>
                    <span>{formatConstraint(c)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="mt-auto">
            <button onClick={() => { setPhase('openTesting'); setHasTestedModels(false); }} className="w-full py-2 border border-gray-300 rounded-lg text-sm text-gray-600 hover:bg-gray-100">
              I'm Satisfied (Skip to Test)
            </button>
          </div>
        </div>

        {/* Main Chat Area */}
        <div className="flex-grow max-w-7xl mx-auto px-4 py-4 w-full flex flex-col gap-6 pb-56 md:pb-8">
          {loading && <div className="fixed inset-0 bg-white/80 backdrop-blur-sm z-50 flex items-center justify-center"><div className="flex flex-col items-center"><div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div><p className="font-mono text-sm">Getting responses...</p></div></div>}

          {error && <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 flex items-center gap-2"><AlertCircle size={20} />{error}</div>}

          {/* No Chat History Reminder */}
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 flex items-center gap-2">
            <AlertCircle size={16} className="text-amber-600 flex-shrink-0" />
            <p className="text-sm text-amber-800"><strong>Note:</strong> There is no chat history. Each round is a one-time session — the models do not remember previous queries.</p>
          </div>

          {/* Session Reminder Panel */}
          {/* BUDGET REMOVED */}
          {/* <div className="bg-gradient-to-r from-slate-50 to-blue-50 border border-slate-200 rounded-xl p-4"> ... </div> */}

          {/* NEW: Traditional Group Instructions */}
          {personaGroup === 'traditional' && (
            <div className="bg-purple-50 border border-purple-200 p-4 rounded-lg shadow-sm">
              <div className="flex items-start gap-3">
                <div className="bg-purple-100 p-2 rounded-full text-purple-600">
                  <Info size={20} />
                </div>
                <div>
                  <h3 className="font-bold text-purple-900 text-lg">Instruction</h3>
                  <p className="text-purple-800">
                    Compare the <strong className="text-purple-950">Model Specification</strong> (click the Info button on each card) with your <strong className="text-purple-950">Assigned Constraints</strong> on the left.
                  </p>
                  <p className="text-sm text-purple-700 mt-1">
                    Select the model that best satisfies your constraints, even if the text quality is similar.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Chat Bubble (User) */}
          <div className="self-end max-w-2xl bg-gray-100 rounded-2xl rounded-tr-none p-4 shadow-sm border border-gray-200"><p className="text-xs text-gray-500 mb-1 font-bold">You</p><p className="text-gray-800 font-medium mt-1">{prompt}</p></div>

          {/* System A */}
          <section>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-blue-600 font-bold text-lg">System A</h2>
              <span className="text-xs text-blue-500 bg-blue-50 px-2 py-1 rounded">Cost: ${((arenaState?.cupid_pair.left.cost || 0) + (arenaState?.cupid_pair.right.cost || 0)).toFixed(5)}</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">{renderModelCard('left', arenaState?.cupid_pair.left!, cupidVote, setCupidVote, 'blue', 'cupid')}{renderModelCard('right', arenaState?.cupid_pair.right!, cupidVote, setCupidVote, 'blue', 'cupid')}</div>

            <div className="mt-4 bg-blue-50 p-4 rounded-lg border border-blue-100">
              <label className="flex items-center text-sm font-bold text-blue-900 mb-2 gap-2">
                <MessageSquare size={16} />
                Feedback for System A
              </label>
              <textarea
                className="w-full border border-blue-200 rounded-lg p-3 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                rows={2}
                placeholder={personaGroup === 'traditional'
                  ? "Explain your choice (e.g., 'Model A satisfies the speed constraint but is too expensive')."
                  : "What worked well? What could be improved?"}
                value={feedbackA}
                onChange={(e) => setFeedbackA(e.target.value)}
              />
              <div className="flex justify-between items-center mt-2">
                <span className="text-xs text-blue-400">Feedback guides the next model selection.</span>
                <div className="flex gap-2">
                  {['Cheaper', 'Smarter', 'Faster'].map(tag => (
                    <button key={tag} onClick={() => setFeedbackA(prev => (prev ? prev + ' ' : '') + tag)} className="text-xs bg-white border border-blue-200 px-2 py-1 rounded-full text-blue-600 hover:bg-blue-100 transition">
                      + {tag}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </section>

          {/* System B */}
          <section className="border-t pt-6">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-gray-600 font-bold text-lg">System B</h2>
              <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">Cost: ${((arenaState?.baseline_pair.left.cost || 0) + (arenaState?.baseline_pair.right.cost || 0)).toFixed(5)}</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">{renderModelCard('left', arenaState?.baseline_pair.left!, baselineVote, setBaselineVote, 'gray', 'baseline')}{renderModelCard('right', arenaState?.baseline_pair.right!, baselineVote, setBaselineVote, 'gray', 'baseline')}</div>
          </section>

          {/* Actions */}
          <div className="sticky bottom-4 z-40 bg-white/90 backdrop-blur border border-gray-200 shadow-2xl p-4 rounded-2xl flex flex-col gap-3">
            <div className="flex gap-3">
              <input type="text" className="flex-grow border rounded-lg px-4 py-3 shadow-sm focus:ring-2 focus:ring-black outline-none" placeholder="Enter prompt for next round..." value={nextPrompt} onChange={(e) => setNextPrompt(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') fetchNextRound(false, nextPrompt); }} />
              <button
                onClick={() => fetchNextRound(false, nextPrompt)}
                disabled={(!cupidVote || !baselineVote) && !loading}
                className="bg-black text-white px-6 py-3 rounded-lg font-bold hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center gap-2 whitespace-nowrap"
              >
                Next Round <ArrowRight size={18} />
              </button>
            </div>
            <div className="flex justify-between items-center px-1">
              <p className="text-xs text-gray-500 flex items-center gap-1">
                {!cupidVote || !baselineVote ? <><AlertCircle size={12} className="text-orange-500" /> Please select a preferred model for BOTH systems.</> : <><CheckCircle size={12} className="text-green-500" /> Ready to proceed.</>}
              </p>
              <button onClick={() => { setPhase('openTesting'); setHasTestedModels(false); }} className="text-xs text-gray-400 hover:text-gray-800 underline">I'm satisfied with the current models</button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // 4. OPEN TESTING PHASE
  if (phase === 'openTesting') {
    // Tutorial
    if (showTestingTutorial) {
      const step = TESTING_TUTORIAL_STEPS[tutorialStep];
      return (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl max-w-md w-full p-6 shadow-2xl animate-in zoom-in-95 duration-200">
            <div className="flex justify-center mb-4 text-4xl">{step.icon}</div>
            <h2 className="text-xl font-bold text-center mb-2">{step.title}</h2>
            <p className="text-gray-600 text-center mb-6">{step.description}</p>
            <div className="flex gap-3">
              <button
                onClick={() => {
                  if (tutorialStep < TESTING_TUTORIAL_STEPS.length - 1) {
                    setTutorialStep(prev => prev + 1);
                  } else {
                    setShowTestingTutorial(false);
                    setHasSeenTestingTutorial(true);
                  }
                }}
                className="flex-grow bg-blue-600 text-white py-3 rounded-xl font-bold hover:bg-blue-700 transition"
              >
                {tutorialStep < TESTING_TUTORIAL_STEPS.length - 1 ? 'Next' : 'Got it!'}
              </button>
            </div>
            <div className="flex justify-center gap-1 mt-4">
              {TESTING_TUTORIAL_STEPS.map((_, i) => (
                <div key={i} className={`w-2 h-2 rounded-full ${i === tutorialStep ? 'bg-blue-600' : 'bg-gray-300'}`} />
              ))}
            </div>
          </div>
        </div>
      );
    }

    // Initial Tutorial Trigger
    if (!hasSeenTestingTutorial) {
      setTimeout(() => { setShowTestingTutorial(true); setTutorialStep(0); }, 500);
    }

    return (
      <div className="min-h-screen bg-gray-50 flex flex-col h-screen">
        <header className="bg-white border-b px-6 py-4 flex justify-between items-center shadow-sm z-10">
          <div>
            <h1 className="font-bold text-xl flex items-center gap-2"><Target className="text-blue-600" /> Testing Phase</h1>
            <p className="text-xs text-gray-500">Compare both systems side-by-side</p>
          </div>
          <button onClick={() => setPhase('evaluation')} className="bg-green-600 text-white px-4 py-2 rounded-lg font-bold hover:bg-green-700 transition text-sm flex items-center gap-2">
            Finish & Rate <ArrowRight size={16} />
          </button>
        </header>

        <div className="flex-grow flex flex-col md:flex-row overflow-hidden">
          {/* Main Chat Area */}
          <div className="flex-grow flex flex-col p-4 md:p-6 overflow-hidden max-w-5xl mx-auto w-full">
            {error && <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm">{error}</div>}

            {/* Progress indicator */}
            {!hasTestedModels && (
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-center mb-4">
                <p className="text-sm text-amber-800">
                  ⚠️ <strong>Required:</strong> Test the models at least once before proceeding to rating.
                </p>
              </div>
            )}

            {/* Round counter */}
            <div className="text-center text-sm text-gray-500 mb-2">
              Rounds: {sideBySideRounds.length} / {OPEN_TESTING_MAX_ROUNDS}
              {hasTestedModels && <span className="ml-2 text-green-600">✓ Ready to rate</span>}
            </div>

            {/* Side-by-side comparison rounds */}
            <div className="flex-grow overflow-y-auto space-y-6 pb-4">
              {sideBySideRounds.length === 0 && !openTestLoading && (
                <div className="text-center text-gray-400 py-12 bg-white rounded-xl border">
                  <p className="text-lg mb-2">Enter a prompt to compare both models</p>
                  <p className="text-sm">Your prompt will be sent to both System A and System B simultaneously</p>
                </div>
              )}

              {sideBySideRounds.map((round, i) => (
                <div key={i} className="bg-white rounded-xl border overflow-hidden shadow-sm">
                  {/* Prompt */}
                  <div className="bg-gray-100 p-4 border-b">
                    <span className="text-xs font-bold text-gray-500 uppercase">Your Prompt (Round {i + 1})</span>
                    <p className="text-gray-800 mt-1">{round.prompt}</p>
                  </div>
                  {/* Side-by-side responses */}
                  <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x">
                    <div className="p-4 bg-blue-50/30">
                      <span className="text-xs font-bold text-blue-600 uppercase mb-2 block">System A Response</span>
                      <Markdown content={round.responseA} />
                    </div>
                    <div className="p-4 bg-gray-50/30">
                      <span className="text-xs font-bold text-gray-600 uppercase mb-2 block">System B Response</span>
                      <Markdown content={round.responseB} />
                    </div>
                  </div>
                </div>
              ))}

              {openTestLoading && (
                <div className="bg-white rounded-xl border p-8 flex justify-center items-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                </div>
              )}
            </div>

            {/* Input */}
            <div className="mt-4 flex gap-2">
              <input
                className="flex-grow border rounded-lg px-4 py-3 shadow-sm focus:ring-2 focus:ring-blue-500 outline-none"
                placeholder="Type a prompt to test both models..."
                value={openTestInput}
                onChange={e => setOpenTestInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') sendSideBySideMessage(); }}
                disabled={openTestLoading || sideBySideRounds.length >= OPEN_TESTING_MAX_ROUNDS}
              />
              <button
                onClick={sendSideBySideMessage}
                disabled={openTestLoading || !openTestInput.trim() || sideBySideRounds.length >= OPEN_TESTING_MAX_ROUNDS}
                className="bg-blue-600 text-white px-6 py-3 rounded-lg font-bold hover:bg-blue-700 disabled:opacity-50 transition"
              >
                <Send size={20} />
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // 5. EVALUATION & FINISH
  if (phase === 'evaluation') {
    if (finished) {
      return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
          <div className="max-w-xl w-full bg-white rounded-3xl shadow-2xl overflow-hidden">
            <div className="bg-gradient-to-r from-orange-500 to-red-500 p-6 text-white text-center">
              <AlertCircle className="mx-auto mb-3" size={48} />
              <h1 className="text-2xl font-bold">Important: Save Your Results!</h1>
              <p className="opacity-90 mt-2">Please download your results before closing this page</p>
            </div>

            <div className="p-8">
              {/* Step 1: Download */}
              <div className={`rounded-xl p-6 mb-6 border-2 ${hasDownloaded ? 'bg-green-50 border-green-300' : 'bg-yellow-50 border-yellow-400'}`}>
                <div className="flex items-center gap-3 mb-4">
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-bold ${hasDownloaded ? 'bg-green-500' : 'bg-yellow-500'}`}>
                    {hasDownloaded ? <CheckCircle size={24} /> : '1'}
                  </div>
                  <div>
                    <span className="font-bold text-gray-800 text-lg">Download Your Results</span>
                    {hasDownloaded && <span className="ml-2 text-green-600 text-sm font-medium">✓ Downloaded</span>}
                  </div>
                </div>
                <p className="text-sm text-gray-600 mb-4">
                  Your study data will be saved as a JSON file. You will need to upload this file in the next step.
                </p>
                <button
                  onClick={handleDownloadAndFinish}
                  className={`w-full py-4 rounded-xl font-bold flex items-center justify-center gap-3 transition text-lg ${hasDownloaded ? 'bg-green-100 text-green-700 hover:bg-green-200' : 'bg-gradient-to-r from-orange-500 to-red-500 text-white hover:from-orange-600 hover:to-red-600 shadow-lg'}`}
                >
                  <Download size={24} /> {hasDownloaded ? 'Download Again' : 'Download Results'}
                </button>
              </div>

              {/* Step 2: Upload Reminder */}
              <div className={`transition-opacity duration-500 ${hasDownloaded ? 'opacity-100' : 'opacity-50 pointer-events-none'}`}>
                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 mb-6 text-left border-2 border-blue-400 shadow-lg">
                  <div className="flex items-center gap-2 mb-3">
                    <div className="bg-blue-600 text-white w-8 h-8 rounded-full flex items-center justify-center font-bold">→</div>
                    <span className="font-bold text-gray-800 text-lg">Final Step: Upload Your Results</span>
                  </div>
                  <p className="text-sm text-gray-600 mb-4">
                    Please upload the JSON file you downloaded to our survey. Your submission is <strong>completely anonymous</strong>.
                  </p>
                  <a
                    href="https://asuengineering.co1.qualtrics.com/jfe/form/SV_6YiJbesl1iMmrT8"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-4 rounded-xl font-bold hover:from-blue-700 hover:to-indigo-700 flex items-center justify-center gap-2 transition shadow-lg text-lg"
                  >
                    <ArrowRight size={20} /> Go to Survey & Upload File
                  </a>
                </div>
              </div>

              <p className="text-sm text-gray-400 text-center">Session: {sessionId}</p>
            </div>
          </div>
        </div>
      );
    }

    const cupidWins = roundHistory.filter(r => r.cupid_vote).length;
    const baselineWins = roundHistory.filter(r => r.baseline_vote).length;
    const systemACost = (arenaState?.cupid_cost || 0) + (arenaState?.routing_cost || 0);
    const systemBCost = arenaState?.baseline_cost || 0;

    return (
      <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-4 py-12">
        <div className="max-w-4xl w-full bg-white rounded-3xl shadow-2xl p-8 md:p-12">
          <h1 className="text-3xl font-bold text-center mb-2">Final Evaluation</h1>
          <p className="text-gray-600 text-center mb-8">Please rate the systems based on your experience.</p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
            {/* System A Card */}
            <div className="bg-blue-50 border border-blue-100 rounded-2xl p-6">
              <h2 className="text-xl font-bold text-blue-800 mb-4 flex items-center gap-2"><Sparkles /> System A</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-bold text-blue-900 mb-2">Quality Rating</label>
                  <div className="flex gap-1">
                    {[1, 2, 3, 4, 5].map(v => (
                      <button key={v} onClick={() => setEvalRatingA(v)} className={`flex-1 py-2 rounded-lg font-bold border ${evalRatingA === v ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-500 border-gray-200'}`}>
                        {v}
                      </button>
                    ))}
                  </div>
                  <div className="flex justify-between text-xs text-gray-500 mt-1 px-1"><span>Very Bad</span><span>Excellent</span></div>
                </div>

                <div>
                  <label className="block text-sm font-bold text-blue-900 mb-2">Budget Compliance</label>
                  <div className="flex gap-1">
                    {[1, 2, 3, 4, 5].map(v => (
                      <button key={v} onClick={() => setEvalBudgetRatingA(v)} className={`flex-1 py-2 rounded-lg font-bold border ${evalBudgetRatingA === v ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-500 border-gray-200'}`}>
                        {v}
                      </button>
                    ))}
                  </div>
                  <div className="flex justify-between text-xs text-gray-500 mt-1 px-1"><span>Poor</span><span>Excellent</span></div>
                </div>
              </div>
            </div>

            {/* System B Card */}
            <div className="bg-gray-50 border border-gray-100 rounded-2xl p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2"><Target /> System B</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-bold text-gray-900 mb-2">Quality Rating</label>
                  <div className="flex gap-1">
                    {[1, 2, 3, 4, 5].map(v => (
                      <button key={v} onClick={() => setEvalRatingB(v)} className={`flex-1 py-2 rounded-lg font-bold border ${evalRatingB === v ? 'bg-gray-800 text-white border-gray-800' : 'bg-white text-gray-500 border-gray-200'}`}>
                        {v}
                      </button>
                    ))}
                  </div>
                  <div className="flex justify-between text-xs text-gray-500 mt-1 px-1"><span>Very Bad</span><span>Excellent</span></div>
                </div>

                <div>
                  <label className="block text-sm font-bold text-gray-900 mb-2">Budget Compliance</label>
                  <div className="flex gap-1">
                    {[1, 2, 3, 4, 5].map(v => (
                      <button key={v} onClick={() => setEvalBudgetRatingB(v)} className={`flex-1 py-2 rounded-lg font-bold border ${evalBudgetRatingB === v ? 'bg-gray-800 text-white border-gray-800' : 'bg-white text-gray-500 border-gray-200'}`}>
                        {v}
                      </button>
                    ))}
                  </div>
                  <div className="flex justify-between text-xs text-gray-500 mt-1 px-1"><span>Poor</span><span>Excellent</span></div>
                </div>
              </div>
            </div>
          </div>

          <div className="mb-8">
            <label className="block text-sm font-bold text-gray-700 mb-2">Final Comments (Optional)</label>
            <textarea
              className="w-full border rounded-lg p-3 h-24 bg-white"
              placeholder="What worked well? What could be improved?"
              value={evalComment}
              onChange={(e) => setEvalComment(e.target.value)}
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
              disabled={evalRatingA === 0 || evalRatingB === 0 || evalBudgetRatingA === 0 || evalBudgetRatingB === 0}
              className="flex-1 bg-blue-600 text-white py-4 rounded-lg font-bold hover:bg-blue-700 disabled:opacity-50 transition flex items-center justify-center"
            >
              Submit & Finish <ArrowRight className="ml-2" size={18} />
            </button>
          </div>
        </div>
      </div>
    );
  }

  return null;
};

export default App;
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
// Objective columns that can be used for constraints
const OBJECTIVE_COLUMNS = [
  { key: 'intelligence', displayName: 'Intelligence Rating', unit: '', isCost: false },
  { key: 'speed', displayName: 'Speed Rating', unit: '', isCost: false },
  { key: 'reasoning', displayName: 'Reasoning Capability', unit: '', isCost: false, isBoolean: true },
  { key: 'input_price', displayName: 'Input Price', unit: '$/1M tokens', isCost: true },
  { key: 'output_price', displayName: 'Output Price', unit: '$/1M tokens', isCost: true },
  { key: 'context_window', displayName: 'Context Window', unit: 'tokens', isCost: false },
  { key: 'max_output', displayName: 'Max Output', unit: 'tokens', isCost: false },
];

// Model pool data for constraint sampling (simplified version)
// In production, this would come from the backend
const MODEL_POOL_DATA = [
  { id: 1, intelligence: 3, speed: 3, reasoning: 0, input_price: 5, output_price: 15, context_window: 128000, max_output: 16384 },
  { id: 2, intelligence: 4, speed: 3, reasoning: 0, input_price: 2, output_price: 8, context_window: 1047576, max_output: 32768 },
  { id: 3, intelligence: 4, speed: 3, reasoning: 1, input_price: 1.1, output_price: 4.4, context_window: 200000, max_output: 100000 },
  { id: 4, intelligence: 1, speed: 2, reasoning: 0, input_price: 3, output_price: 4, context_window: 16385, max_output: 4096 },
  { id: 5, intelligence: 1, speed: 2, reasoning: 0, input_price: 1.5, output_price: 2, context_window: 4096, max_output: 4096 },
  { id: 6, intelligence: 1, speed: 2, reasoning: 0, input_price: 0.5, output_price: 1.5, context_window: 16385, max_output: 4096 },
  { id: 7, intelligence: 2, speed: 3, reasoning: 0, input_price: 10, output_price: 30, context_window: 128000, max_output: 4096 },
  { id: 8, intelligence: 2, speed: 3, reasoning: 0, input_price: 10, output_price: 30, context_window: 128000, max_output: 4096 },
  { id: 9, intelligence: 3, speed: 4, reasoning: 0, input_price: 0.4, output_price: 1.6, context_window: 1047576, max_output: 32768 },
  { id: 10, intelligence: 2, speed: 5, reasoning: 0, input_price: 0.1, output_price: 0.4, context_window: 1047576, max_output: 32768 },
  { id: 11, intelligence: 2, speed: 3, reasoning: 0, input_price: 30, output_price: 60, context_window: 8192, max_output: 8192 },
  { id: 12, intelligence: 3, speed: 3, reasoning: 0, input_price: 2.5, output_price: 10, context_window: 128000, max_output: 16384 },
  { id: 13, intelligence: 2, speed: 4, reasoning: 0, input_price: 0.15, output_price: 0.6, context_window: 128000, max_output: 16384 },
  { id: 14, intelligence: 2, speed: 4, reasoning: 0, input_price: 0.15, output_price: 0.6, context_window: 128000, max_output: 16384 },
  { id: 15, intelligence: 3, speed: 3, reasoning: 0, input_price: 2.5, output_price: 10, context_window: 128000, max_output: 16384 },
  { id: 16, intelligence: 2, speed: 4, reasoning: 0, input_price: 0.15, output_price: 0.6, context_window: 128000, max_output: 16384 },
  { id: 17, intelligence: 3, speed: 4, reasoning: 1, input_price: 0.25, output_price: 2, context_window: 400000, max_output: 128000 },
  { id: 18, intelligence: 2, speed: 5, reasoning: 1, input_price: 0.05, output_price: 0.4, context_window: 400000, max_output: 128000 },
  { id: 19, intelligence: 3, speed: 3, reasoning: 0, input_price: 2.5, output_price: 10, context_window: 128000, max_output: 16384 },
  { id: 20, intelligence: 4, speed: 4, reasoning: 1, input_price: 1.25, output_price: 10, context_window: 400000, max_output: 128000 },
  { id: 21, intelligence: 4, speed: 3, reasoning: 1, input_price: 1.25, output_price: 10, context_window: 400000, max_output: 128000 },
  { id: 22, intelligence: 4, speed: 3, reasoning: 1, input_price: 1.1, output_price: 4.4, context_window: 200000, max_output: 100000 },
  { id: 23, intelligence: 5, speed: 1, reasoning: 1, input_price: 2, output_price: 8, context_window: 200000, max_output: 100000 },
  { id: 24, intelligence: 4, speed: 4, reasoning: 1, input_price: 1.75, output_price: 14, context_window: 400000, max_output: 128000 },
  { id: 25, intelligence: 4, speed: 4, reasoning: 1, input_price: 1.75, output_price: 14, context_window: 400000, max_output: 128000 },
];

// Sample constraints using min-max rule from Constraint_Picker.ipynb
function sampleConstraintsDynamic(nObjectives: number = 3, kSamples: number = 5): Constraint[] {
  const constraints: Constraint[] = [];
  
  // Shuffle and pick n objective columns
  const shuffled = [...OBJECTIVE_COLUMNS].sort(() => Math.random() - 0.5);
  const selectedCols = shuffled.slice(0, Math.min(nObjectives, shuffled.length));
  
  for (const col of selectedCols) {
    // Get all values for this column from model pool
    const values = MODEL_POOL_DATA.map(m => (m as any)[col.key]).filter(v => v !== undefined && v !== null);
    
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

function sampleConstraints(): Constraint[] {
  // Sample 2-4 constraints randomly
  const numConstraints = 2 + Math.floor(Math.random() * 3); // 2, 3, or 4
  return sampleConstraintsDynamic(numConstraints, 5);
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
    title: 'Review Model Information',
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
      const constraints = sampleConstraints();
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
    // Always go to interaction phase, show tutorial if first time
    setPhase('interaction');
    if (!hasSeenInteractionTutorial) {
      setShowInteractionTutorial(true);
      setTutorialStep(0);
    }
  };

  const handleSkipTutorial = () => {
    setShowInteractionTutorial(false);
    setShowTestingTutorial(false);
    setHasSeenInteractionTutorial(true);
    setHasSeenTestingTutorial(true);
  };

  const handleNextTutorialStep = () => {
    if (showInteractionTutorial) {
      if (tutorialStep < INTERACTION_TUTORIAL_STEPS.length - 1) {
        setTutorialStep(tutorialStep + 1);
      } else {
        setShowInteractionTutorial(false);
        setHasSeenInteractionTutorial(true);
        setTutorialStep(0);
        // Phase is already 'interaction', no need to change
      }
    } else if (showTestingTutorial) {
      if (tutorialStep < TESTING_TUTORIAL_STEPS.length - 1) {
        setTutorialStep(tutorialStep + 1);
      } else {
        setShowTestingTutorial(false);
        setHasSeenTestingTutorial(true);
        setTutorialStep(0);
      }
    }
  };

  const handlePrevTutorialStep = () => {
    if (tutorialStep > 0) {
      setTutorialStep(tutorialStep - 1);
    }
  };

  const startSession = async () => {
    if (!prompt.trim()) { setError("Please enter a query to start."); return; }
    setError(null);
    await fetchNextRound(true, prompt);
    setInit(false);
  };

  const handleSatisfied = () => {
    if (arenaState && cupidVote && baselineVote) {
      const historyEntry: RoundHistory = {
        round: arenaState.round, prompt: prompt,
        cupid_left_id: arenaState.cupid_pair.left.model_id, cupid_right_id: arenaState.cupid_pair.right.model_id, cupid_vote: cupidVote,
        baseline_left_id: arenaState.baseline_pair.left.model_id, baseline_right_id: arenaState.baseline_pair.right.model_id, baseline_vote: baselineVote,
        feedback: feedbackA, timestamp: new Date().toISOString()
      };
      setRoundHistory(prev => [...prev, historyEntry]);
    }
    // Show testing tutorial for first time
    if (!hasSeenTestingTutorial) {
      setShowTestingTutorial(true);
      setTutorialStep(0);
    }
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
          feedback: feedbackA, timestamp: new Date().toISOString()
        };
        setRoundHistory(prev => [...prev, historyEntry]);
      }
      // Show testing tutorial for first time
      if (!hasSeenTestingTutorial) {
        setShowTestingTutorial(true);
        setTutorialStep(0);
      }
      setPhase('openTesting');
      return;
    }

    if (!nextPrompt.trim()) { setError("Please enter your next query to continue."); return; }
    setError(null);
    await fetchNextRound(false, nextPrompt);
  };

  const handleFinalSubmit = async () => { await saveSessionData(); setFinished(true); };

  const downloadResults = () => {
    const systemACost = (arenaState?.cupid_cost || 0) + (arenaState?.routing_cost || 0);
    const systemBCost = arenaState?.baseline_cost || 0;
    
    const results = {
      session_id: sessionId, 
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
      final_costs: { 
        system_a: systemACost, 
        system_b: systemBCost 
      },
      open_test_rounds: { 
        system_a: openTestRoundsA, 
        system_b: openTestRoundsB,
        side_by_side_rounds: sideBySideRounds.length
      },
      side_by_side_test_data: sideBySideRounds,
      // Add cupid and baseline labels for clarity
      cupid: {
        label: 'System A',
        algorithm: 'CUPID (Pairwise GP with routing)',
        total_cost: arenaState?.cupid_cost || 0,
        routing_cost: arenaState?.routing_cost || 0,
        quality_rating: evalRatingA,
        budget_rating: evalBudgetRatingA
      },
      baseline: {
        label: 'System B',
        algorithm: 'Bradley-Terry Baseline',
        total_cost: arenaState?.baseline_cost || 0,
        quality_rating: evalRatingB,
        budget_rating: evalBudgetRatingB
      }
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
          <p className="text-sm text-gray-500 mb-4">Model specifications from OpenAI (name hidden)</p>
          {stats ? (
            <div className="space-y-4">
              <div className="bg-gradient-to-r from-purple-50 to-blue-50 p-4 rounded-lg">
                <h4 className="font-bold text-gray-700 mb-3">Performance Ratings</h4>
                <div className="grid grid-cols-3 gap-3">
                  <div className="text-center"><div className="text-2xl font-bold text-purple-600">{stats.intelligence || '—'}</div><div className="text-xs text-gray-500">Intelligence</div></div>
                  <div className="text-center"><div className="text-2xl font-bold text-blue-600">{stats.speed || '—'}</div><div className="text-xs text-gray-500">Speed</div></div>
                  <div className="text-center"><div className="text-2xl font-bold text-indigo-600">{stats.reasoning ? 'Yes' : 'No'}</div><div className="text-xs text-gray-500">Reasoning</div></div>
                </div>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-bold text-gray-700 mb-3">Pricing (per 1M tokens)</h4>
                <div className="grid grid-cols-2 gap-3">
                  <div><div className="text-lg font-bold text-green-600">${stats.input_price || '—'}</div><div className="text-xs text-gray-500">Input Cost</div></div>
                  <div><div className="text-lg font-bold text-green-700">${stats.output_price || '—'}</div><div className="text-xs text-gray-500">Output Cost</div></div>
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
                  {stats.function_calling && <span className="bg-orange-100 text-orange-700 text-xs px-2 py-1 rounded">Function Calling</span>}
                  {stats.structured_output && <span className="bg-pink-100 text-pink-700 text-xs px-2 py-1 rounded">Structured Output</span>}
                </div>
              </div>
              {stats.knowledge_cutoff && <div className="text-sm text-gray-600 bg-gray-50 p-3 rounded-lg"><span className="font-medium">Knowledge Cutoff:</span> {stats.knowledge_cutoff}</div>}
            </div>
          ) : (<div className="text-gray-500 text-center py-8">Stats not available</div>)}
        </div>
      </div>
    );
  };

  // Model Card with Markdown rendering
  const renderModelCard = (side: 'left' | 'right', data: ModelResponse | undefined, voteState: 'left' | 'right' | null, setVote: (v: 'left' | 'right') => void, colorClass: string, system: 'cupid' | 'baseline') => {
    if (!data) return <div className="animate-pulse h-64 bg-gray-100 rounded-lg flex items-center justify-center"><span className="text-gray-400">Loading...</span></div>;
    const isSelected = voteState === side;
    const label = side === 'left' ? '1' : '2';
    const borderColor = isSelected ? 'border-blue-600' : 'border-gray-200 hover:border-gray-300';
    const bgColor = isSelected ? 'bg-blue-50' : 'bg-white';
    const buttonBg = isSelected ? 'bg-blue-600' : 'bg-gray-100';
    const stats = getModelStats(system, side);

    return (
      <div className={`relative p-4 rounded-xl border-2 transition-all duration-200 flex flex-col ${borderColor} ${bgColor} ${isSelected ? 'shadow-lg scale-[1.01]' : ''}`}>
        <div className="flex justify-between items-center mb-2">
          <span className="text-xs font-bold text-gray-500">Output {label}</span>
          {/* Cost with explanation tooltip */}
          <div className="group relative">
            <span className="text-xs text-gray-400 cursor-help flex items-center gap-1">
              <DollarSign size={12} />
              {data.cost.toFixed(5)}
              <Info size={10} className="text-gray-300" />
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
        {personaGroup === 'traditional' && stats && (
          <div className="border-t pt-3 mt-2">
            <div className="text-xs font-bold text-gray-600 mb-2 flex items-center gap-1">
              <Info size={12} /> Model Specifications
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-purple-50 p-2 rounded">
                <span className="text-gray-500">Intelligence:</span>
                <span className="font-bold text-purple-700 ml-1">{stats.intelligence || '—'}</span>
              </div>
              <div className="bg-blue-50 p-2 rounded">
                <span className="text-gray-500">Speed:</span>
                <span className="font-bold text-blue-700 ml-1">{stats.speed || '—'}</span>
              </div>
              <div className="bg-indigo-50 p-2 rounded">
                <span className="text-gray-500">Reasoning:</span>
                <span className="font-bold text-indigo-700 ml-1">{stats.reasoning ? 'Yes' : 'No'}</span>
              </div>
              <div className="bg-green-50 p-2 rounded">
                <span className="text-gray-500">Input:</span>
                <span className="font-bold text-green-700 ml-1">${stats.input_price}/1M</span>
              </div>
              <div className="bg-green-50 p-2 rounded">
                <span className="text-gray-500">Output:</span>
                <span className="font-bold text-green-700 ml-1">${stats.output_price}/1M</span>
              </div>
              <div className="bg-orange-50 p-2 rounded">
                <span className="text-gray-500">Context:</span>
                <span className="font-bold text-orange-700 ml-1">{stats.context_window?.toLocaleString() || '—'}</span>
              </div>
              <div className="bg-orange-50 p-2 rounded">
                <span className="text-gray-500">Max Output:</span>
                <span className="font-bold text-orange-700 ml-1">{stats.max_output?.toLocaleString() || '—'}</span>
              </div>
              <div className="bg-gray-50 p-2 rounded">
                <span className="text-gray-500">Func Call:</span>
                <span className="font-bold text-gray-700 ml-1">{stats.function_calling ? 'Yes' : 'No'}</span>
              </div>
            </div>
          </div>
        )}

        <div onClick={() => setVote(side)} className={`mt-3 text-center font-bold py-3 rounded-lg cursor-pointer transition ${buttonBg} ${isSelected ? 'text-white' : 'text-gray-400 hover:text-gray-600'}`}>{isSelected ? '✓ PREFERRED' : `Select Output ${label}`}</div>
      </div>
    );
  };

  const renderEvalCard = (systemLabel: string, totalCost: number, rating: number, setRating: (r: number) => void, budgetRating: number, setBudgetRating: (r: number) => void, winCount: number) => (
    <div className="border-2 border-blue-200 bg-blue-50 rounded-xl p-6 relative overflow-hidden">
      <div className="absolute top-0 right-0 bg-blue-200 text-blue-800 text-xs font-bold px-3 py-1 rounded-bl-lg">{systemLabel}</div>
      <div className="space-y-3 mb-6 mt-4">
        <div className="bg-white p-3 rounded-lg border border-gray-100 shadow-sm flex justify-between items-center"><span className="text-sm text-gray-500">Total Cost</span><span className="font-mono font-bold text-gray-800 flex items-center"><DollarSign size={14} />{totalCost.toFixed(4)}</span></div>
        <div className="bg-white p-3 rounded-lg border border-gray-100 shadow-sm flex justify-between items-center"><span className="text-sm text-gray-500">Times Preferred</span><span className="font-mono font-bold text-gray-800">{winCount} rounds</span></div>
      </div>
      
      {/* Quality Rating */}
      <div className="mb-6">
        <label className="block text-sm font-bold text-blue-900 mb-3 text-center">⭐ Rate Model Quality</label>
        <div className="space-y-2">
          {RATING_LABELS.map((item) => (
            <button
              key={item.value}
              onClick={() => setRating(item.value)}
              className={`w-full p-3 rounded-lg text-left transition-all flex items-center gap-3 ${rating === item.value
                ? 'bg-blue-600 text-white'
                : 'bg-white border border-gray-200 hover:border-gray-300 text-gray-700'
                }`}
            >
              <span className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${rating === item.value ? 'bg-white/20' : 'bg-gray-100'
                }`}>{item.value}</span>
              <span className="text-sm">{item.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Budget Compliance Rating */}
      <div>
        <label className="block text-sm font-bold text-green-800 mb-3 text-center">💰 Rate Budget Compliance</label>
        <div className="space-y-2">
          {BUDGET_RATING_LABELS.map((item) => (
            <button
              key={item.value}
              onClick={() => setBudgetRating(item.value)}
              className={`w-full p-2 rounded-lg text-left transition-all flex items-center gap-3 ${budgetRating === item.value
                ? 'bg-green-600 text-white'
                : 'bg-white border border-gray-200 hover:border-gray-300 text-gray-700'
                }`}
            >
              <span className={`w-7 h-7 rounded-full flex items-center justify-center font-bold text-xs ${budgetRating === item.value ? 'bg-white/20' : 'bg-gray-100'
                }`}>{item.value}</span>
              <span className="text-xs">{item.label}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );

  // Tutorial Modal Component
  const renderTutorialModal = () => {
    if (!showInteractionTutorial && !showTestingTutorial) return null;
    
    const steps = showInteractionTutorial ? INTERACTION_TUTORIAL_STEPS : TESTING_TUTORIAL_STEPS;
    const currentStep = steps[tutorialStep];
    const totalSteps = steps.length;
    const title = showInteractionTutorial ? 'Interaction Tutorial' : 'Testing Tutorial';
    
    return (
      <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl max-w-lg w-full p-8 relative">
          {/* Progress indicator */}
          <div className="flex justify-center gap-2 mb-6">
            {steps.map((_, idx) => (
              <div 
                key={idx} 
                className={`w-3 h-3 rounded-full transition-all ${idx === tutorialStep ? 'bg-blue-600 scale-110' : idx < tutorialStep ? 'bg-blue-300' : 'bg-gray-200'}`}
              />
            ))}
          </div>
          
          {/* Content */}
          <div className="text-center mb-8">
            <div className="text-5xl mb-4">{currentStep.icon}</div>
            <h3 className="text-xl font-bold text-gray-800 mb-3">{currentStep.title}</h3>
            <p className="text-gray-600 leading-relaxed">{currentStep.description}</p>
          </div>
          
          {/* Navigation */}
          <div className="flex gap-3">
            {tutorialStep > 0 ? (
              <button 
                onClick={handlePrevTutorialStep}
                className="flex-1 py-3 px-4 rounded-lg border border-gray-200 text-gray-600 hover:bg-gray-50 font-medium flex items-center justify-center gap-2"
              >
                <ArrowLeft size={18} /> Back
              </button>
            ) : (
              <button 
                onClick={handleSkipTutorial}
                className="flex-1 py-3 px-4 rounded-lg border border-gray-200 text-gray-500 hover:bg-gray-50 font-medium"
              >
                Skip Tutorial
              </button>
            )}
            <button 
              onClick={handleNextTutorialStep}
              className="flex-1 py-3 px-4 rounded-lg bg-blue-600 text-white hover:bg-blue-700 font-medium flex items-center justify-center gap-2"
            >
              {tutorialStep === totalSteps - 1 ? (
                <>Let's Start! <CheckCircle size={18} /></>
              ) : (
                <>Next <ArrowRight size={18} /></>
              )}
            </button>
          </div>
          
          {/* Step counter */}
          <p className="text-center text-xs text-gray-400 mt-4">
            Step {tutorialStep + 1} of {totalSteps}
          </p>
        </div>
      </div>
    );
  };

  // ==================== PHASE RENDERS ====================

  // CONSENT PHASE
  if (phase === 'consent') {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="max-w-3xl w-full bg-white shadow-xl rounded-2xl overflow-hidden flex flex-col">
          <div className="bg-gradient-to-r from-blue-600 to-indigo-600 p-6 md:p-8 text-white text-center">
            <h1 className="text-2xl md:text-3xl font-bold">LLM Matchmaking Study</h1>
            <p className="opacity-90">Find Your Dream Model</p>
            <p className="text-xs mt-2 opacity-75">IRB ID: STUDY00023557</p>
          </div>
          <div className="p-6 md:p-8 overflow-y-auto max-h-[60vh] prose prose-sm max-w-none text-gray-700">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
              <p className="font-semibold text-blue-800 mb-1">🎯 Goal of This Study</p>
              <p className="text-blue-700 text-sm">Help us evaluate and compare two AI model matching systems. Your ratings will determine which system is better at finding the right model for your needs.</p>
            </div>
            
            <p>Thank you for participating in this study. In this experiment, you will help us compare two systems by interacting with both and providing your preferences. Your feedback will help us improve LLM matchmaking systems and how they are presented to users.</p>

            <h3 className="text-lg font-semibold mt-4">Settings</h3>
            <p>There are <strong>25 LLM models from OpenAI</strong> ready for you to use. As a user, it might be difficult choosing a suitable model for your task. Our system will help you find your dream model.</p>

            <h3 className="text-lg font-semibold mt-4">What You Will Do</h3>
            <p>You will interact with <strong>two systems concurrently</strong>. You will be shown two outputs for the same task or query in each system. The cost of the two systems will be shown separately and you will see how much you spent on each system.</p>
            <p>Your task is to <strong>compare the two outputs</strong> in each system and indicate which one you prefer.</p>
            <p>In both systems, you can provide <strong>language feedback</strong>. This gives you an option to dictate the system to your personal preference. For example, you could ask for a cheaper model: <em>"Please give me a cheaper model"</em>, or a model that is more capable: <em>"Please give me a smarter model."</em></p>
            <p>You will repeat this process for multiple rounds with different queries. <strong>If you are satisfied with your model, you can opt to end the drafting process.</strong></p>
            <p>After the drafting process, you are allowed to <strong>play with your chosen models</strong> from the two systems (up to 10 rounds each). You then will assign a rating for each model.</p>

            <h3 className="text-lg font-semibold mt-4">Instructions for Comparison</h3>
            <p>Focus on the quality and the cost of the outputs. You will not see the model's name, but will be provided some information about the LLMs such as its intelligence ratings, input cost, output cost, etc. from OpenAI.</p>
            <p><strong>There are no right or wrong answers</strong>—choose the output that you think is better overall.</p>

            <h3 className="text-lg font-semibold mt-4">Important Notes</h3>
            <ul>
              <li>Please <strong>do not try to guess which system/LLM produced which output</strong>. Focus on your genuine preference.</li>
              <li>Take your time to read and understand each output before making a choice.</li>
              <li>Your responses are anonymous and will be used only for research purposes.</li>
            </ul>

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

  // CALIBRATION PHASE
  if (phase === 'calibration') {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
        <div className="max-w-5xl w-full">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-gray-800">Setup Your Session</h1>
            <p className="text-gray-600 mt-2">
              Session configuration: <span className="font-mono bg-gray-200 px-2 py-1 rounded">{budgetConstraints.maxRounds} rounds • ${budgetConstraints.maxCost} budget</span>
            </p>
          </div>

          {error && <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 flex items-center gap-2"><AlertCircle size={20} />{error}</div>}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-6">
              <div className="bg-white p-6 rounded-2xl shadow-lg">
                <h2 className="text-xl font-bold mb-4 flex items-center"><User className="mr-2" /> About You</h2>
                <div className="space-y-4">
                  <div><label className="block text-sm font-medium text-gray-700 mb-1">Age *</label><input type="number" min="18" className="w-full border rounded p-2" value={demographics.age} onChange={e => setDemographics({ ...demographics, age: parseInt(e.target.value) || '' })} placeholder="Required" /></div>
                  <div><label className="block text-sm font-medium text-gray-700 mb-1">Education (optional)</label><select className="w-full border rounded p-2" value={demographics.education} onChange={e => setDemographics({ ...demographics, education: e.target.value })}><option value="">Prefer not to say</option>{EDUCATION_LEVELS.map(l => <option key={l} value={l}>{l}</option>)}</select></div>
                  <div><label className="block text-sm font-medium text-gray-700 mb-1">AI chatbot experience</label><div className="flex items-center gap-2 text-sm text-gray-500"><span>Rarely</span><input type="range" min="1" max="5" className="flex-grow" value={demographics.familiarity} onChange={e => setDemographics({ ...demographics, familiarity: parseInt(e.target.value) })} /><span>Daily</span></div></div>
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
                        <div className="font-bold text-gray-800">Traditional Group</div>
                        <div className="text-sm text-gray-500">You will be assigned a persona with specific model requirements</div>
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
                        <div className="font-bold text-gray-800">Subject Expert Group</div>
                        <div className="text-sm text-gray-500">Play as an expert/student in your field seeking the best model</div>
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
                        <div className="font-bold text-gray-800">Personal Preference Group</div>
                        <div className="text-sm text-gray-500">Ask anything you want and choose based on your own criteria</div>
                      </div>
                    </div>
                  </button>
                </div>

                {personaGroup === 'expert' && (
                  <div className="mt-4 pt-4 border-t">
                    <label className="block text-sm font-bold text-gray-700 mb-3">Select Your Field of Expertise:</label>
                    <div className="space-y-2">
                      {EXPERT_SUBJECTS.map(s => (
                        <button
                          key={s.id}
                          onClick={() => setSelectedExpertSubject(s.id)}
                          className={`w-full p-3 rounded-lg border text-left flex items-center gap-3 transition-all ${selectedExpertSubject === s.id ? 'border-blue-500 bg-blue-100' : 'border-gray-200 hover:border-gray-300'}`}
                        >
                          {s.icon}
                          <span className="font-medium">{s.label}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="space-y-6">
              {personaGroup === 'traditional' && assignedConstraints.length > 0 && (
                <div className="bg-gradient-to-br from-purple-900 to-indigo-800 text-white p-6 rounded-2xl shadow-xl">
                  <div className="uppercase tracking-widest text-xs font-bold text-purple-300 mb-2">Your Assigned Requirements</div>
                  <p className="text-purple-100 text-sm mb-4">As a user seeking assistance from LLM, you need a model that meets these specifications:</p>
                  <div className="space-y-2">
                    {assignedConstraints.map((c, i) => (
                      <div key={i} className="bg-white/10 p-3 rounded-lg flex items-center gap-2">
                        <CheckCircle size={18} className="text-green-400" />
                        <span className="font-mono text-sm">{formatConstraint(c)}</span>
                      </div>
                    ))}
                  </div>
                  <p className="text-purple-200 text-xs mt-4">Focus on finding a model that satisfies these constraints while considering the output quality and cost.</p>
                </div>
              )}

              {personaGroup === 'expert' && selectedExpertSubject && (
                <div className="bg-gradient-to-br from-blue-900 to-indigo-800 text-white p-6 rounded-2xl shadow-xl">
                  <div className="uppercase tracking-widest text-xs font-bold text-blue-300 mb-2">Subject Expert Instructions</div>
                  <p className="text-blue-100 text-sm mb-4">
                    You will play the role of a <strong>{EXPERT_SUBJECTS.find(s => s.id === selectedExpertSubject)?.label}</strong> expert/student seeking assistance from LLM, but you are yet to know which model is good for your field.
                  </p>
                  <p className="text-blue-200 text-sm mb-3">Focus on the <strong>quality</strong> and the <strong>cost</strong> of the outputs. Choose the output that best demonstrates domain knowledge and accuracy.</p>
                  <div className="bg-yellow-500/20 border border-yellow-400/30 rounded-lg p-3 mt-3">
                    <p className="text-yellow-200 text-sm font-medium">💡 <strong>Tip:</strong> Ask hard, challenging, or technical questions in your field to properly test each model's capabilities. This will help you distinguish which model is truly better for your domain.</p>
                  </div>
                </div>
              )}

              {personaGroup === 'preference' && (
                <div className="bg-gradient-to-br from-indigo-900 to-purple-800 text-white p-6 rounded-2xl shadow-xl">
                  <div className="uppercase tracking-widest text-xs font-bold text-indigo-300 mb-2">Personal Preference Instructions</div>
                  <p className="text-indigo-100 text-sm mb-4">
                    You can <strong>ask anything you want</strong>. Focus on the quality and the cost of the outputs.
                  </p>
                  <p className="text-indigo-200 text-sm">Choose based on whatever criteria matters to you — there are no right or wrong answers!</p>
                </div>
              )}

              <div className="bg-white p-6 rounded-2xl shadow-lg">
                <h2 className="text-xl font-bold mb-4">How It Works</h2>
                <div className="space-y-3 text-sm text-gray-600">
                  <div className="flex gap-3"><div className="bg-blue-100 text-blue-700 w-7 h-7 rounded-full flex items-center justify-center font-bold flex-shrink-0 text-sm">1</div><div><p className="font-bold text-gray-800">Enter Queries</p><p>Ask questions and compare two outputs from each system.</p></div></div>
                  <div className="flex gap-3"><div className="bg-blue-100 text-blue-700 w-7 h-7 rounded-full flex items-center justify-center font-bold flex-shrink-0 text-sm">2</div><div><p className="font-bold text-gray-800">Select & Feedback</p><p>Choose your preferred output. Provide feedback to guide the system (e.g., "cheaper model", "smarter model").</p></div></div>
                  <div className="flex gap-3"><div className="bg-green-100 text-green-700 w-7 h-7 rounded-full flex items-center justify-center font-bold flex-shrink-0 text-sm">3</div><div><p className="font-bold text-gray-800">End When Satisfied</p><p>Click "I'm Satisfied" anytime to proceed to play with your chosen models.</p></div></div>
                  <div className="flex gap-3"><div className="bg-green-100 text-green-700 w-7 h-7 rounded-full flex items-center justify-center font-bold flex-shrink-0 text-sm">4</div><div><p className="font-bold text-gray-800">Rate the Systems</p><p>Rate each system based on model quality and budget adherence.</p></div></div>
                </div>
              </div>

              <button 
                type="button" 
                onClick={handleCalibrationSubmit}
                className="w-full bg-blue-600 text-white py-4 rounded-xl font-bold hover:bg-blue-700 transition text-lg"
              >
                Start Experiment →
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // INTERACTION PHASE
  if (phase === 'interaction') {
    // Show tutorial modal
    if (showInteractionTutorial) {
      return (
        <div className="min-h-screen bg-gray-50">
          {renderTutorialModal()}
        </div>
      );
    }

    if (init) {
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
            <p className="text-sm text-gray-500 mb-4">Budget: ${budgetConstraints.maxCost} • Up to {budgetConstraints.maxRounds} rounds</p>

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

            {error && <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm flex items-center gap-2"><AlertCircle size={16} />{error}</div>}
            <div className="mb-4 text-left">
              <label className="block text-sm font-medium text-gray-700 mb-1">Enter your first query:</label>
              <textarea className="w-full border rounded-lg p-3 focus:ring-2 focus:ring-blue-500 outline-none resize-none" rows={5} value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="Type your question or task here..." />
            </div>
            <button onClick={startSession} disabled={!prompt.trim() || loading} className="w-full bg-blue-600 text-white py-3 rounded-lg font-bold hover:bg-blue-700 disabled:opacity-50 transition flex items-center justify-center gap-2">{loading ? (<><RefreshCw size={16} className="animate-spin" />Starting...</>) : 'Start Comparing'}</button>
          </div>
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

        <header className="bg-white border-b sticky top-0 z-10">
          <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="bg-blue-600 text-white px-2 py-1 rounded text-xs font-bold">LLM MATCHMAKING</div>
            </div>
            <div className="flex items-center space-x-3 text-sm font-mono">
              <div className="flex items-center"><span className="text-gray-400 mr-1">Round</span><span className="font-bold">{arenaState?.round || 0}/{budgetConstraints.maxRounds}</span></div>
              <div className="hidden sm:flex items-center gap-2">
                <span className="text-blue-600 bg-blue-50 px-2 py-0.5 rounded text-xs">A: ${systemACost.toFixed(4)}</span>
                <span className="text-blue-600 bg-blue-50 px-2 py-0.5 rounded text-xs">B: ${systemBCost.toFixed(4)}</span>
              </div>
            </div>
          </div>
        </header>

        <main className="flex-grow max-w-7xl mx-auto px-4 py-4 w-full flex flex-col gap-6 pb-56 md:pb-8">
          {loading && <div className="fixed inset-0 bg-white/80 backdrop-blur-sm z-50 flex items-center justify-center"><div className="flex flex-col items-center"><div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div><p className="font-mono text-sm">Getting responses...</p></div></div>}
          {error && <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 flex items-center gap-2"><AlertCircle size={20} />{error}</div>}

          {/* No Chat History Reminder */}
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 flex items-center gap-2">
            <AlertCircle size={16} className="text-amber-600 flex-shrink-0" />
            <p className="text-sm text-amber-800"><strong>Note:</strong> There is no chat history. Each round is a one-time session — the models do not remember previous queries.</p>
          </div>

          {/* Session Reminder Panel */}
          <div className="bg-gradient-to-r from-slate-50 to-blue-50 border border-slate-200 rounded-xl p-4">
            <div className="flex flex-wrap items-start gap-4">
              {/* Budget Info */}
              <div className="flex items-center gap-3 bg-white px-3 py-2 rounded-lg border border-slate-200 shadow-sm">
                <DollarSign size={18} className="text-green-600" />
                <div className="text-sm">
                  <span className="text-gray-500">Budget:</span>
                  <span className="font-bold text-gray-800 ml-1">${budgetConstraints.maxCost}</span>
                  <span className="text-gray-400 mx-1">•</span>
                  <span className="text-gray-500">Rounds:</span>
                  <span className="font-bold text-gray-800 ml-1">{budgetConstraints.maxRounds}</span>
                </div>
              </div>

              {/* Traditional Group - Constraints */}
              {personaGroup === 'traditional' && assignedConstraints.length > 0 && (
                <div className="flex-1 min-w-[280px]">
                  <div className="flex items-center gap-2 mb-2">
                    <Target size={16} className="text-purple-600" />
                    <span className="text-xs font-bold text-purple-700 uppercase">Your Model Requirements</span>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {assignedConstraints.map((c, i) => (
                      <span key={i} className="bg-purple-100 text-purple-800 text-xs px-2 py-1 rounded-full font-medium border border-purple-200">
                        {formatConstraint(c)}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Expert Group - Subject */}
              {personaGroup === 'expert' && selectedExpertSubject && (
                <div className="flex items-center gap-2 bg-blue-100 px-3 py-2 rounded-lg border border-blue-200">
                  <BookOpen size={16} className="text-blue-600" />
                  <div className="text-sm">
                    <span className="text-blue-700 font-medium">Expert Mode:</span>
                    <span className="font-bold text-blue-900 ml-1">{EXPERT_SUBJECTS.find(s => s.id === selectedExpertSubject)?.label}</span>
                  </div>
                </div>
              )}

              {/* Preference Group */}
              {personaGroup === 'preference' && (
                <div className="flex items-center gap-2 bg-indigo-100 px-3 py-2 rounded-lg border border-indigo-200">
                  <ThumbsUp size={16} className="text-indigo-600" />
                  <span className="text-sm font-medium text-indigo-800">Personal Preference Mode — Choose based on your own criteria</span>
                </div>
              )}
            </div>
          </div>

          {/* Traditional Group: Review Model Info Instruction */}
          {personaGroup === 'traditional' && (
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
              <p className="text-sm text-purple-800">
                <strong>📋 Important:</strong> Review the <strong>Model Information</strong> shown below each response to check if it meets your requirements. 
                Use the feedback fields to guide both systems toward models that satisfy your constraints.
              </p>
            </div>
          )}

          <div className="bg-white p-4 rounded-lg shadow-sm border"><span className="text-xs font-bold text-gray-400 uppercase">Your Query</span><p className="text-gray-800 font-medium mt-1">{prompt}</p></div>

          {/* System A */}
          <section>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-blue-600 font-bold text-lg">System A</h2>
              <span className="text-xs text-blue-500 bg-blue-50 px-2 py-1 rounded">Cost: ${systemACost.toFixed(4)}</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">{renderModelCard('left', arenaState?.cupid_pair.left, cupidVote, setCupidVote, 'blue', 'cupid')}{renderModelCard('right', arenaState?.cupid_pair.right, cupidVote, setCupidVote, 'blue', 'cupid')}</div>
            <div className="mt-4 bg-blue-50 p-4 rounded-lg border border-blue-100">
              <label className="flex items-center text-sm font-bold text-blue-900 mb-2"><MessageSquare size={16} className="mr-2" />Language Feedback (optional)</label>
              <input type="text" className="w-full border border-blue-200 rounded p-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none" placeholder='e.g., "Please give me a cheaper model" or "Please give me a smarter model"' value={feedbackA} onChange={(e) => setFeedbackA(e.target.value)} />
            </div>
          </section>

          <hr className="border-gray-200" />

          {/* System B */}
          <section>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-blue-600 font-bold text-lg">System B</h2>
              <span className="text-xs text-blue-500 bg-blue-50 px-2 py-1 rounded">Cost: ${systemBCost.toFixed(4)}</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">{renderModelCard('left', arenaState?.baseline_pair.left, baselineVote, setBaselineVote, 'blue', 'baseline')}{renderModelCard('right', arenaState?.baseline_pair.right, baselineVote, setBaselineVote, 'blue', 'baseline')}</div>
            <div className="mt-4 bg-blue-50 p-4 rounded-lg border border-blue-100">
              <label className="flex items-center text-sm font-bold text-blue-900 mb-2"><MessageSquare size={16} className="mr-2" />Language Feedback (optional)</label>
              <input type="text" className="w-full border border-blue-200 rounded p-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none" placeholder='e.g., "Please give me a cheaper model" or "Please give me a smarter model"' value={feedbackB} onChange={(e) => setFeedbackB(e.target.value)} />
            </div>
          </section>

          {/* Footer */}
          <div className="fixed bottom-0 left-0 w-full md:sticky md:bottom-4 z-40 bg-white p-4 shadow-lg border-t md:border md:rounded-xl">
            <div className="max-w-7xl mx-auto flex flex-col gap-4">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <span className={`w-3 h-3 rounded-full ${cupidVote ? 'bg-blue-500' : 'bg-gray-300'}`}></span>
                  <span>A: {cupidVote ? `Output ${cupidVote === 'left' ? '1' : '2'}` : '—'}</span>
                  <span className="mx-2">|</span>
                  <span className={`w-3 h-3 rounded-full ${baselineVote ? 'bg-blue-500' : 'bg-gray-300'}`}></span>
                  <span>B: {baselineVote ? `Output ${baselineVote === 'left' ? '1' : '2'}` : '—'}</span>
                </div>
                <div className="flex items-center gap-2">
                  {isLastRound && <span className="text-orange-600 font-bold">Final Round!</span>}
                  {canEndEarly && !isLastRound && (
                    <button
                      onClick={handleSatisfied}
                      className="bg-green-100 text-green-700 px-4 py-2 rounded-lg font-medium hover:bg-green-200 transition text-sm"
                    >
                      ✓ I'm Satisfied — End Drafting
                    </button>
                  )}
                </div>
              </div>

              {!isLastRound && (
                <textarea
                  placeholder="Enter your next query (required to continue)..."
                  className={`w-full border rounded-lg px-3 py-3 text-sm resize-none ${!nextPrompt.trim() && cupidVote && baselineVote ? 'border-red-300 bg-red-50' : ''}`}
                  rows={4}
                  value={nextPrompt}
                  onChange={(e) => setNextPrompt(e.target.value)}
                />
              )}

              <button onClick={handleSubmitRound} disabled={loading} className="w-full md:w-auto md:self-end bg-blue-600 text-white px-8 py-3 rounded-lg font-bold hover:bg-blue-700 disabled:opacity-50 transition">
                {isLastRound ? 'Continue to Play with Models →' : 'Submit & Next →'}
              </button>
            </div>
          </div>
        </main>
      </div>
    );
  }

  // OPEN TESTING PHASE - Side-by-side comparison
  if (phase === 'openTesting') {
    // Show tutorial modal
    if (showTestingTutorial) {
      return (
        <div className="min-h-screen bg-gray-50">
          {renderTutorialModal()}
        </div>
      );
    }

    const canChat = sideBySideRounds.length < OPEN_TESTING_MAX_ROUNDS;

    return (
      <div className="min-h-screen bg-gray-50 flex flex-col">
        <header className="bg-white border-b p-4">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <div>
              <h1 className="text-xl font-bold">Stage 2: Test & Compare</h1>
              <p className="text-sm text-gray-500">Test both matched models side-by-side ({sideBySideRounds.length}/{OPEN_TESTING_MAX_ROUNDS} rounds)</p>
            </div>
            <div className="flex items-center gap-3">
              <button 
                onClick={() => { setShowTestingTutorial(true); setTutorialStep(0); }}
                className="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1"
              >
                <HelpCircle size={14} /> Tutorial
              </button>
              <button 
                onClick={() => setPhase('evaluation')} 
                disabled={!hasTestedModels}
                className="bg-blue-600 text-white px-6 py-2 rounded-lg font-bold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                I'm Done → Rate Systems
              </button>
            </div>
          </div>
        </header>
        
        <main className="flex-grow max-w-7xl mx-auto w-full p-4 flex flex-col gap-4">
          {/* Stage Introduction */}
          <div className="bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2">
              <Sparkles className="text-green-600" size={20} />
              <span className="font-bold text-green-800">Testing Stage</span>
            </div>
            <p className="text-sm text-green-700">
              Now test both matched models by sending the same prompt to both. Compare their outputs side-by-side to determine which system found a better model for you. 
              <strong> You must test at least once before rating.</strong>
            </p>
          </div>

          {/* Instructions */}
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <p className="text-sm text-yellow-800">
              <strong>Focus on the quality of output.</strong> Send the same prompt to both models and compare their responses side by side. 
              This will help you determine which model better suits your needs.
            </p>
          </div>

          {/* Group-specific reminder */}
          {personaGroup === 'expert' && selectedExpertSubject && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
              <p className="text-sm text-blue-800">
                <strong>Reminder:</strong> You are evaluating as a <strong>{EXPERT_SUBJECTS.find(s => s.id === selectedExpertSubject)?.label}</strong> expert. 
                Ask technical questions in your field to test model capabilities.
              </p>
            </div>
          )}

          {personaGroup === 'preference' && (
            <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3">
              <p className="text-sm text-indigo-800">
                <strong>Reminder:</strong> You are in Personal Preference mode. Choose based on whatever criteria matters most to you.
              </p>
            </div>
          )}

          {personaGroup === 'traditional' && assignedConstraints.length > 0 && (
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
              <p className="text-sm text-purple-800">
                <strong>Reminder - Your Requirements:</strong> {assignedConstraints.map(c => formatConstraint(c)).join(' • ')}
              </p>
            </div>
          )}

          {error && <div className="p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm">{error}</div>}

          {/* Progress indicator */}
          {!hasTestedModels && (
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-center">
              <p className="text-sm text-amber-800">
                ⚠️ <strong>Required:</strong> Test the models at least once before proceeding to rating.
              </p>
            </div>
          )}

          {/* Round counter */}
          <div className="text-center text-sm text-gray-500">
            Rounds: {sideBySideRounds.length} / {OPEN_TESTING_MAX_ROUNDS}
            {hasTestedModels && <span className="ml-2 text-green-600">✓ Ready to rate</span>}
          </div>

          {/* Side-by-side comparison rounds */}
          <div className="flex-grow overflow-y-auto space-y-6">
            {sideBySideRounds.length === 0 && !openTestLoading && (
              <div className="text-center text-gray-400 py-12 bg-white rounded-xl border">
                <p className="text-lg mb-2">Enter a prompt to compare both models</p>
                <p className="text-sm">Your prompt will be sent to both System A and System B simultaneously</p>
              </div>
            )}

            {sideBySideRounds.map((round, i) => (
              <div key={i} className="bg-white rounded-xl border overflow-hidden">
                {/* Prompt */}
                <div className="bg-gray-100 p-4 border-b">
                  <span className="text-xs font-bold text-gray-500 uppercase">Your Prompt (Round {i + 1})</span>
                  <p className="text-gray-800 mt-1">{round.prompt}</p>
                </div>
                
                {/* Side-by-side responses */}
                <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x">
                  {/* System A Response */}
                  <div className="p-4">
                    <div className="flex justify-between items-center mb-3">
                      <span className="font-bold text-blue-600">System A</span>
                      <span className="text-xs text-gray-400">${round.costA.toFixed(5)}</span>
                    </div>
                    <div className="prose prose-sm max-w-none">
                      <Markdown content={round.responseA} />
                    </div>
                  </div>
                  
                  {/* System B Response */}
                  <div className="p-4">
                    <div className="flex justify-between items-center mb-3">
                      <span className="font-bold text-blue-600">System B</span>
                      <span className="text-xs text-gray-400">${round.costB.toFixed(5)}</span>
                    </div>
                    <div className="prose prose-sm max-w-none">
                      <Markdown content={round.responseB} />
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {openTestLoading && (
              <div className="bg-white rounded-xl border p-8 text-center">
                <RefreshCw size={24} className="animate-spin mx-auto text-blue-600 mb-2" />
                <p className="text-gray-500 text-sm">Getting responses from both models...</p>
              </div>
            )}
          </div>

          {/* Input area */}
          <div className="bg-white rounded-xl border p-4 sticky bottom-4">
            <div className="flex gap-2">
              <input
                type="text"
                className="flex-grow border rounded-lg px-4 py-3"
                placeholder={canChat ? "Enter your prompt (will be sent to both models)..." : `Maximum ${OPEN_TESTING_MAX_ROUNDS} rounds reached`}
                value={openTestInput}
                onChange={(e) => setOpenTestInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && canChat && sendSideBySideMessage()}
                disabled={!canChat || openTestLoading}
              />
              <button
                onClick={sendSideBySideMessage}
                disabled={openTestLoading || !openTestInput.trim() || !canChat}
                className="px-6 py-3 rounded-lg font-bold bg-blue-600 text-white disabled:opacity-50 flex items-center gap-2"
              >
                <Send size={18} /> Compare
              </button>
            </div>
          </div>
        </main>
      </div>
    );
  }

  // EVALUATION PHASE
  if (phase === 'evaluation') {
    if (finished) return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="max-w-xl w-full bg-white shadow-xl rounded-2xl p-12 text-center">
          <CheckCircle className="mx-auto text-green-500 mb-6" size={80} />
          <h1 className="text-3xl font-bold mb-2">Thank You!</h1>
          <p className="text-gray-600 mb-6">Your feedback helps us improve LLM matchmaking systems.</p>
          
          {/* Step 1: Download */}
          <div className="bg-gray-50 rounded-xl p-6 mb-6 text-left">
            <div className="flex items-center gap-2 mb-3">
              <div className="bg-blue-600 text-white w-6 h-6 rounded-full flex items-center justify-center text-sm font-bold">1</div>
              <span className="font-bold text-gray-800">Download your results</span>
            </div>
            <button onClick={downloadResults} className="w-full bg-blue-600 text-white py-3 rounded-lg font-bold hover:bg-blue-700 flex items-center justify-center gap-2">
              <Download size={18} /> Download JSON File
            </button>
          </div>

          {/* Step 2: Upload to Survey */}
          <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl p-6 mb-6 text-left border border-green-200">
            <div className="flex items-center gap-2 mb-3">
              <div className="bg-green-600 text-white w-6 h-6 rounded-full flex items-center justify-center text-sm font-bold">2</div>
              <span className="font-bold text-gray-800">Submit your results (anonymous)</span>
            </div>
            <p className="text-sm text-gray-600 mb-4">
              Please upload the JSON file you just downloaded to our survey. Your submission is <strong>completely anonymous</strong>.
            </p>
            <a 
              href="https://asuengineering.co1.qualtrics.com/jfe/form/SV_6YiJbesl1iMmrT8" 
              target="_blank" 
              rel="noopener noreferrer"
              className="w-full bg-green-600 text-white py-3 rounded-lg font-bold hover:bg-green-700 flex items-center justify-center gap-2 transition"
            >
              <ArrowRight size={18} /> Go to Survey & Upload
            </a>
          </div>

          <p className="text-sm text-gray-400">Session: {sessionId}</p>
        </div>
      </div>
    );

    const cupidWins = roundHistory.filter(r => r.cupid_vote).length;
    const baselineWins = roundHistory.filter(r => r.baseline_vote).length;
    const systemACost = (arenaState?.cupid_cost || 0) + (arenaState?.routing_cost || 0);
    const systemBCost = arenaState?.baseline_cost || 0;

    // Get final model stats for traditional group
    const finalCupidStats = arenaState?.cupid_pair?.left_stats || arenaState?.cupid_pair?.right_stats || null;
    const finalBaselineStats = arenaState?.baseline_pair?.left_stats || arenaState?.baseline_pair?.right_stats || null;

    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="max-w-5xl w-full bg-white shadow-xl rounded-2xl overflow-hidden">
          <div className="bg-blue-600 p-6 text-white text-center relative">
            {/* Go Back Button */}
            <button 
              onClick={() => setPhase('openTesting')} 
              className="absolute left-4 top-1/2 -translate-y-1/2 bg-white/20 hover:bg-white/30 text-white px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2 transition"
            >
              <ArrowLeft size={16} /> Back to Testing
            </button>
            <h1 className="text-2xl font-bold">Final Evaluation</h1>
            <p className="opacity-90">Rate each system based on model quality and budget adherence</p>
          </div>
          
          <div className="p-4 md:p-8 bg-gray-50">
            {/* Session Info */}
            <div className="text-center mb-6">
              <p className="text-gray-600">You completed {roundHistory.length} comparison round{roundHistory.length !== 1 ? 's' : ''}</p>
              <p className="text-xs text-gray-400 mt-2">(Model identities remain hidden — rate based on your experience)</p>
            </div>

            {/* Group-specific reminder panel */}
            <div className="mb-8">
              {personaGroup === 'traditional' && assignedConstraints.length > 0 && (
                <div className="bg-purple-50 border border-purple-200 rounded-xl p-4 mb-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Target size={18} className="text-purple-600" />
                    <span className="font-bold text-purple-800">Your Model Requirements</span>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {assignedConstraints.map((c, i) => (
                      <span key={i} className="bg-purple-100 text-purple-800 text-sm px-3 py-1 rounded-full font-medium border border-purple-200">
                        {formatConstraint(c)}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {personaGroup === 'expert' && selectedExpertSubject && (
                <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 mb-4">
                  <div className="flex items-center gap-2">
                    <BookOpen size={18} className="text-blue-600" />
                    <span className="font-bold text-blue-800">Expert Mode:</span>
                    <span className="text-blue-700">{EXPERT_SUBJECTS.find(s => s.id === selectedExpertSubject)?.label}</span>
                  </div>
                  <p className="text-sm text-blue-600 mt-2">Rate based on domain knowledge accuracy and helpfulness for your field.</p>
                </div>
              )}

              {personaGroup === 'preference' && (
                <div className="bg-indigo-50 border border-indigo-200 rounded-xl p-4 mb-4">
                  <div className="flex items-center gap-2">
                    <ThumbsUp size={18} className="text-indigo-600" />
                    <span className="font-bold text-indigo-800">Personal Preference Mode</span>
                  </div>
                  <p className="text-sm text-indigo-600 mt-2">Rate based on your personal criteria — quality, style, helpfulness, or whatever matters to you.</p>
                </div>
              )}

              {/* Budget reminder */}
              <div className="bg-gray-100 rounded-lg p-3 flex items-center justify-center gap-4 text-sm">
                <span className="text-gray-500">Your Budget:</span>
                <span className="font-bold text-gray-800">${budgetConstraints.maxCost}</span>
                <span className="text-gray-400">•</span>
                <span className="text-gray-500">Max Rounds:</span>
                <span className="font-bold text-gray-800">{budgetConstraints.maxRounds}</span>
              </div>
            </div>

            {/* Rating cards with optional model info for traditional group */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-10">
              <div className="space-y-4">
                {renderEvalCard("System A", systemACost, evalRatingA, setEvalRatingA, evalBudgetRatingA, setEvalBudgetRatingA, cupidWins)}
                {/* Model info for traditional group */}
                {personaGroup === 'traditional' && finalCupidStats && (
                  <div className="bg-white border rounded-xl p-4">
                    <h4 className="text-sm font-bold text-gray-600 mb-3 flex items-center gap-2">
                      <Info size={14} /> System A - Final Model Specs
                    </h4>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="bg-purple-50 p-2 rounded"><span className="text-gray-500">Intelligence:</span><span className="font-bold text-purple-700 ml-1">{finalCupidStats.intelligence || '—'}</span></div>
                      <div className="bg-blue-50 p-2 rounded"><span className="text-gray-500">Speed:</span><span className="font-bold text-blue-700 ml-1">{finalCupidStats.speed || '—'}</span></div>
                      <div className="bg-indigo-50 p-2 rounded"><span className="text-gray-500">Reasoning:</span><span className="font-bold text-indigo-700 ml-1">{finalCupidStats.reasoning ? 'Yes' : 'No'}</span></div>
                      <div className="bg-green-50 p-2 rounded"><span className="text-gray-500">Input:</span><span className="font-bold text-green-700 ml-1">${finalCupidStats.input_price}/1M</span></div>
                      <div className="bg-green-50 p-2 rounded"><span className="text-gray-500">Output:</span><span className="font-bold text-green-700 ml-1">${finalCupidStats.output_price}/1M</span></div>
                      <div className="bg-orange-50 p-2 rounded"><span className="text-gray-500">Context:</span><span className="font-bold text-orange-700 ml-1">{finalCupidStats.context_window?.toLocaleString() || '—'}</span></div>
                    </div>
                  </div>
                )}
              </div>
              
              <div className="space-y-4">
                {renderEvalCard("System B", systemBCost, evalRatingB, setEvalRatingB, evalBudgetRatingB, setEvalBudgetRatingB, baselineWins)}
                {/* Model info for traditional group */}
                {personaGroup === 'traditional' && finalBaselineStats && (
                  <div className="bg-white border rounded-xl p-4">
                    <h4 className="text-sm font-bold text-gray-600 mb-3 flex items-center gap-2">
                      <Info size={14} /> System B - Final Model Specs
                    </h4>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="bg-purple-50 p-2 rounded"><span className="text-gray-500">Intelligence:</span><span className="font-bold text-purple-700 ml-1">{finalBaselineStats.intelligence || '—'}</span></div>
                      <div className="bg-blue-50 p-2 rounded"><span className="text-gray-500">Speed:</span><span className="font-bold text-blue-700 ml-1">{finalBaselineStats.speed || '—'}</span></div>
                      <div className="bg-indigo-50 p-2 rounded"><span className="text-gray-500">Reasoning:</span><span className="font-bold text-indigo-700 ml-1">{finalBaselineStats.reasoning ? 'Yes' : 'No'}</span></div>
                      <div className="bg-green-50 p-2 rounded"><span className="text-gray-500">Input:</span><span className="font-bold text-green-700 ml-1">${finalBaselineStats.input_price}/1M</span></div>
                      <div className="bg-green-50 p-2 rounded"><span className="text-gray-500">Output:</span><span className="font-bold text-green-700 ml-1">${finalBaselineStats.output_price}/1M</span></div>
                      <div className="bg-orange-50 p-2 rounded"><span className="text-gray-500">Context:</span><span className="font-bold text-orange-700 ml-1">{finalBaselineStats.context_window?.toLocaleString() || '—'}</span></div>
                    </div>
                  </div>
                )}
              </div>
            </div>
            
            <div className="max-w-2xl mx-auto space-y-6">
              <div>
                <label className="block text-sm font-bold text-gray-700 mb-2">Any final thoughts? (optional)</label>
                <textarea className="w-full border rounded-lg p-3 h-24 bg-white" placeholder="What worked well? What could be improved?" value={evalComment} onChange={(e) => setEvalComment(e.target.value)} />
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
                  className="flex-1 bg-blue-600 text-white py-4 rounded-lg font-bold hover:bg-blue-700 disabled:opacity-50 transition flex items-center justify-center"
                >
                  Submit & Finish <ArrowRight className="ml-2" size={18} />
                </button>
              </div>
              {!hasTestedModels && (
                <p className="text-center text-amber-600 text-sm mt-2">
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

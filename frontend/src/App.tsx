import React, { useState, useCallback, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  ArrowRight, ArrowLeft, MessageSquare, User, CheckCircle,
  Star, DollarSign, Zap, Brain, X, Info, RefreshCw,
  AlertCircle, Download, Send, MessageCircle, Target, Sparkles,
  BookOpen, Heart, ThumbsUp, Settings, HelpCircle, Database,
  Wand2, Image as ImageIcon
} from 'lucide-react';

// --- MODE TYPE ---
type ArenaMode = 'text' | 'image';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import 'katex/dist/katex.min.css';
// --- API CONFIGURATION ---
const API_URL = 'https://cupid-vs-baseline-arena.onrender.com';
// const API_URL = 'http://localhost:8000';
const SURVEY_URL = "https://asuengineering.co1.qualtrics.com/jfe/form/SV_6YiJbesl1iMmrT8";

// --- Image World ACCESS GATE ---
// NOTE: This is only a *UI-level* gate. Because this is client-side code, a determined person can still
// discover the code in the built JS bundle. For stronger protection, enforce an access token on the backend.
//
// Change this string to whatever access code you want to give only to Image-study participants.
// (Example: put this code in the Image Qualtrics instructions.)
const IMAGE_ARENA_ACCESS_CODE = 'image';
const IMAGE_ARENA_UNLOCK_STORAGE_KEY = 'image_arena_unlocked_v1';

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

// --- IMAGE DISPLAY COMPONENT ---
const ImageDisplay: React.FC<{
  imageUrl: string;
  alt?: string;
  className?: string;
  loading?: boolean;
}> = ({ imageUrl, alt = "Generated image", className = '', loading = false }) => {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);

  useEffect(() => {
    setImageLoaded(false);
    setImageError(false);
  }, [imageUrl]);

  if (loading) {
    return (
      <div className={`flex items-center justify-center bg-gray-100 rounded-lg w-full h-full ${className}`} style={{ minHeight: 'min(200px, 100%)' }}>
        <div className="text-center">
          <RefreshCw className="animate-spin mx-auto mb-2 text-blue-500" size={32} />
          <p className="text-sm text-gray-500">Generating image...</p>
        </div>
      </div>
    );
  }

  if (imageError || !imageUrl) {
    return (
      <div className={`flex items-center justify-center bg-gray-100 rounded-lg w-full h-full ${className}`} style={{ minHeight: 'min(200px, 100%)' }}>
        <div className="text-center text-gray-400">
          <AlertCircle className="mx-auto mb-2" size={32} />
          <p className="text-sm">Failed to load image</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`relative w-full h-full max-h-[60vh] flex items-center justify-center bg-gray-50 rounded-lg shadow-md overflow-hidden ${className}`}>
      {!imageLoaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100 rounded-lg">
          <RefreshCw className="animate-spin text-blue-500" size={24} />
        </div>
      )}
      <img
        src={imageUrl}
        alt={alt}
        className={`block max-w-full max-h-full object-contain transition-opacity duration-300 ${imageLoaded ? 'opacity-100' : 'opacity-0'}`}
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
    return <ImageDisplay imageUrl={content} className={className} />;
  }

  return <Markdown content={content} className={className} />;
};

// --- TYPES ---
type Phase = 'modeSelect' | 'consent' | 'calibration' | 'interaction' | 'openTesting' | 'evaluation';
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
  const minCost = 0.7;
  const maxCost = 1.5;
  const minRounds = 2;
  const maxRounds = 2;

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
const INTERACTION_TUTORIAL_STEPS_TEXT = [
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
    title: 'Review Model Specification',
    description: 'Check the model specifications (intelligence, speed, price, etc.) to see if they match your requirements. This helps you make informed decisions.',
    icon: '📋'
  },
  {
    title: 'Confirm When Satisfied',
    description: 'When you\'re happy with the model selection, click "I\'m Satisfied" to proceed to the testing phase. You can end early after the first round.',
    icon: '✅'
  }
];

const INTERACTION_TUTORIAL_STEPS_IMAGE = [
  {
    title: 'Welcome to Image Generation',
    description: 'Enter image prompts and the system will match you with the best text-to-image generation model based on your preferences.',
    icon: '🎨'
  },
  {
    title: 'Selecting Your Preferred Image',
    description: 'For each prompt, you\'ll see two generated images from different models. Click on the one you prefer. Your choices help the system learn your style.',
    icon: '👆'
  },
  {
    title: 'Optional Feedback',
    description: 'You can provide feedback like "more realistic" or "more artistic style" to guide the system toward your ideal image generation model.',
    icon: '💬'
  },
  {
    title: 'Confirm When Satisfied',
    description: 'When you\'re happy with the image quality, click "I\'m Satisfied" to proceed to the testing phase.',
    icon: '✅'
  }
];

const TESTING_TUTORIAL_STEPS_TEXT = [
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

const TESTING_TUTORIAL_STEPS_IMAGE = [
  {
    title: 'Welcome to Image Testing',
    description: 'Now you\'ll test both matched image models side-by-side. The same prompt goes to both systems so you can compare their generated images directly.',
    icon: '🔬'
  },
  {
    title: 'Focus on Image Quality',
    description: 'Pay attention to visual quality, accuracy to your prompt, style consistency, and detail. This will help you decide which system found a better model for you.',
    icon: '⭐'
  },
  {
    title: 'Compare Side-by-Side',
    description: 'Each prompt shows System A and System B images next to each other. You have up to 10 rounds to test thoroughly.',
    icon: '↔️'
  },
  {
    title: 'Proceed to Rating',
    description: 'After testing, you\'ll rate both systems on image quality and budget compliance. Make sure to test enough to form a clear opinion!',
    icon: '📊'
  }
];

// Legacy aliases for compatibility
const INTERACTION_TUTORIAL_STEPS = INTERACTION_TUTORIAL_STEPS_TEXT;
const TESTING_TUTORIAL_STEPS = TESTING_TUTORIAL_STEPS_TEXT;

// --- EXPERT SUBJECTS ---
const EXPERT_SUBJECTS = [
  { id: 'science', label: 'Art & Science', icon: <BookOpen className="text-blue-500" size={24} /> },
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
  const [phase, setPhase] = useState<Phase>('modeSelect');
  const [mode, setMode] = useState<ArenaMode>('text');
  const [sessionId, setSessionId] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  // ---- Image World access gate (soft UI-level lock) ----
  const [imageGateOpen, setImageGateOpen] = useState<boolean>(false);
  const [imageGateCode, setImageGateCode] = useState<string>('');
  const [imageGateError, setImageGateError] = useState<string | null>(null);
  const [imageArenaUnlocked, setImageArenaUnlocked] = useState<boolean>(() => {
    try {
      return sessionStorage.getItem(IMAGE_ARENA_UNLOCK_STORAGE_KEY) === '1';
    } catch {
      return false;
    }
  });

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
  const [initialPreference, setInitialPreference] = useState<string>('');
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
  const [hasOpenedSurvey, setHasOpenedSurvey] = useState(false);
  const [confirmedUploaded, setConfirmedUploaded] = useState(false);


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

  // Optional: allow direct links that pre-select an arena.
  // Examples:
  //   - Text/LLM participants:  https://.../?mode=text
  //   - Image participants:     https://.../?mode=image&code=YOUR_CODE
  useEffect(() => {
    try {
      const params = new URLSearchParams(window.location.search);
      const modeParam = (params.get('mode') || params.get('arena') || '').toLowerCase();
      const codeParam = params.get('code') || params.get('image_code') || params.get('pass');

      if (modeParam === 'text') {
        setMode('text');
        setPhase('consent');
        return;
      }

      if (modeParam === 'image') {
        if (codeParam && codeParam === IMAGE_ARENA_ACCESS_CODE) {
          setImageArenaUnlocked(true);
          try { sessionStorage.setItem(IMAGE_ARENA_UNLOCK_STORAGE_KEY, '1'); } catch { }
          setMode('image');
          setPhase('consent');
        } else {
          // Keep them on modeSelect, but prompt for the code.
          setImageGateOpen(true);
          setImageGateError('Image World requires an access code.');
        }
      }
    } catch {
      // Ignore URL parsing errors and fall back to normal flow.
    }
    // Intentionally run once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (showDownloadReminder) {
      setHasOpenedSurvey(false);
      setConfirmedUploaded(false);
    }
  }, [showDownloadReminder]);


  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [phase, arenaState?.round, init]);

  // ---- Image World gating helpers ----
  const openImageGate = () => {
    setImageGateOpen(true);
    setImageGateCode('');
    setImageGateError(null);
  };

  const closeImageGate = () => {
    setImageGateOpen(false);
    setImageGateCode('');
    setImageGateError(null);
  };

  const unlockImageArena = () => {
    setImageArenaUnlocked(true);
    try {
      sessionStorage.setItem(IMAGE_ARENA_UNLOCK_STORAGE_KEY, '1');
    } catch {
      // ignore
    }
  };

  const attemptUnlockImageArena = () => {
    const code = imageGateCode.trim();
    if (!code) {
      setImageGateError('Please enter the access code.');
      return;
    }
    if (code !== IMAGE_ARENA_ACCESS_CODE) {
      setImageGateError('Incorrect access code. Please use the code provided in your study instructions.');
      return;
    }

    unlockImageArena();
    closeImageGate();
    setMode('image');
    setPhase('consent');
  };

  const handleSelectTextMode = () => {
    setMode('text');
    setPhase('consent');
  };

  const handleSelectImageMode = () => {
    if (imageArenaUnlocked) {
      setMode('image');
      setPhase('consent');
      return;
    }
    openImageGate();
  };

  const renderImageGateModal = () => {
    if (!imageGateOpen) return null;

    return (
      <div
        className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4"
        onClick={closeImageGate}
      >
        <div
          className="bg-white rounded-2xl max-w-md w-full p-6 relative shadow-2xl"
          onClick={(e) => e.stopPropagation()}
        >
          <button
            onClick={closeImageGate}
            className="absolute top-4 right-4 text-gray-400 hover:text-gray-600"
            aria-label="Close"
          >
            <X size={20} />
          </button>

          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-full bg-purple-100 flex items-center justify-center">
              <Wand2 className="text-purple-600" size={20} />
            </div>
            <div>
              <h3 className="text-lg font-bold text-gray-900">Image World Access</h3>
              <p className="text-xs text-gray-500">Access code required</p>
            </div>
          </div>

          <p className="text-sm text-gray-600 mt-3">
            If you are participating in the <strong>Image Generation</strong> study, enter the access code from your instructions.
          </p>

          <div className="mt-4">
            <label className="block text-xs font-semibold text-gray-700 mb-1">Access code</label>
            <input
              type="password"
              value={imageGateCode}
              onChange={(e) => setImageGateCode(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && attemptUnlockImageArena()}
              className="w-full border rounded-lg px-3 py-2 focus:ring-2 focus:ring-purple-500 outline-none"
              placeholder="Enter code"
              autoFocus
            />
            {imageGateError && (
              <div className="mt-2 text-sm text-red-600 flex items-start gap-2">
                <AlertCircle size={16} className="mt-0.5" />
                <span>{imageGateError}</span>
              </div>
            )}
          </div>

          <div className="flex gap-3 mt-6">
            <button
              onClick={closeImageGate}
              className="flex-1 py-2.5 rounded-lg border border-gray-200 text-gray-700 hover:bg-gray-50 font-semibold"
            >
              Cancel
            </button>
            <button
              onClick={attemptUnlockImageArena}
              className="flex-1 py-2.5 rounded-lg bg-purple-600 text-white hover:bg-purple-700 font-semibold disabled:opacity-50"
              disabled={!imageGateCode.trim()}
            >
              Unlock
            </button>
          </div>
        </div>
      </div>
    );
  };

  const fetchNextRound = useCallback(async (isFirst: boolean = false, currentPrompt?: string) => {
    setLoading(true);
    setError(null);
    const promptToUse = currentPrompt || prompt;

    try {
      const payload: any = {
        session_id: isFirst ? null : sessionId,
        prompt: promptToUse,
        mode: mode,
        previous_vote: null,
        feedback_text: isFirst ? (initialPreference || '') : (feedbackA || ''),
      };

      if (!isFirst && cupidVote) payload.cupid_vote = cupidVote;
      if (!isFirst && baselineVote) payload.baseline_vote = baselineVote;

      if (isFirst) {
        payload.budget_cost = budgetConstraints.maxCost;
        payload.budget_rounds = budgetConstraints.maxRounds;
        payload.persona_group = mode === 'image' ? 'preference' : personaGroup;
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
  }, [prompt, sessionId, cupidVote, baselineVote, feedbackA, initialPreference, budgetConstraints, personaGroup, selectedExpertSubject, assignedConstraints, demographics, arenaState, mode]);

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
            system: 'cupid',
            mode: mode
          }),
        }),
        fetch(`${API_URL}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionId,
            message: currentPrompt,
            system: 'baseline',
            mode: mode
          }),
        })
      ]);

      let textA = '[Connection error]';
      let textB = '[Connection error]';
      let costA = 0;
      let costB = 0;

      if (responseA.ok) {
        const dataA = await responseA.json();
        textA = dataA.response || dataA.content || 'No response received';
        costA = dataA.cost || 0;
      }
      if (responseB.ok) {
        const dataB = await responseB.json();
        textB = dataB.response || dataB.content || 'No response received';
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
    // Safety check: don't allow Image mode unless they've unlocked it.
    // (Prevents accidental entry if someone navigates in a weird order.)
    if (mode === 'image' && !imageArenaUnlocked) {
      setError('Image World requires an access code.');
      setPhase('modeSelect');
      openImageGate();
      return;
    }

    const newSessionId = `sess_${mode}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setSessionId(newSessionId);
    const assignedBudget = sampleBudget();
    setBudgetConstraints(assignedBudget);

    // Image mode skips calibration (preference-only, no constraints)
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
    if (!prompt.trim()) { setError(mode === 'image' ? "Please enter an image prompt to start." : "Please enter a query to start."); return; }
    if (!initialPreference.trim()) { setError(mode === 'image' ? "Please describe what kind of images you want to generate." : "Please describe what kind of LLM you're looking for."); return; }
    setError(null);
    await fetchNextRound(true, prompt);
    setInit(false);
  };

  const handleSatisfied = async () => {
    if (arenaState && cupidVote && baselineVote) {
      const historyEntry: RoundHistory = {
        round: arenaState.round, prompt: prompt,
        cupid_left_id: arenaState.cupid_pair.left.model_id, cupid_right_id: arenaState.cupid_pair.right.model_id, cupid_vote: cupidVote,
        baseline_left_id: arenaState.baseline_pair.left.model_id, baseline_right_id: arenaState.baseline_pair.right.model_id, baseline_vote: baselineVote,
        feedback: feedbackA, timestamp: new Date().toISOString(),
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
      await submitFinalVotes();
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
          feedback: feedbackA, timestamp: new Date().toISOString(),
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
        await submitFinalVotes();
      }
      // Show testing tutorial for first time
      if (!hasSeenTestingTutorial) {
        setShowTestingTutorial(true);
        setTutorialStep(0);
      }
      setPhase('openTesting');
      return;
    }

    if (!nextPrompt.trim()) { setError(mode === 'image' ? "Please enter your next image prompt to continue." : "Please enter your next query to continue."); return; }
    setError(null);
    await fetchNextRound(false, nextPrompt);
  };

  // Build results object (shared between save and download)
  const buildResultsObject = () => {
    const lastRound = roundHistory[roundHistory.length - 1];
    const cupidFinalModelId = lastRound ?
      (lastRound.cupid_vote === 'left' ? lastRound.cupid_left_id : lastRound.cupid_right_id) : null;
    const baselineFinalModelId = lastRound ?
      (lastRound.baseline_vote === 'left' ? lastRound.baseline_left_id : lastRound.baseline_right_id) : null;

    const cupidFinalStats = lastRound?.cupid_vote === 'left'
      ? arenaState?.cupid_pair?.left_stats
      : arenaState?.cupid_pair?.right_stats;
    const baselineFinalStats = lastRound?.baseline_vote === 'left'
      ? arenaState?.baseline_pair?.left_stats
      : arenaState?.baseline_pair?.right_stats;

    const interactionPhaseCupidCost = arenaState?.cupid_cost || 0;
    const interactionPhaseBaselineCost = arenaState?.baseline_cost || 0;
    const interactionPhaseRoutingCost = arenaState?.routing_cost || 0;

    const openTestCostA = sideBySideRounds.reduce((sum, r) => sum + r.costA, 0);
    const openTestCostB = sideBySideRounds.reduce((sum, r) => sum + r.costB, 0);

    return {
      session_id: sessionId,
      timestamp: new Date().toISOString(),
      demographics,
      persona_group: personaGroup,
      expert_subject: selectedExpertSubject,
      constraints: assignedConstraints,
      budget: budgetConstraints,
      initial_preference: initialPreference,

      final_state: {
        system_a: {
          label: 'System A (CUPID)',
          algorithm: 'CUPID (Pairwise GP with language feedback routing)',
          final_model_id: cupidFinalModelId,
          final_model_name: arenaState?.final_model_a?.model_name || null,
          final_model_stats: cupidFinalStats || null,
          interaction_phase_cost: interactionPhaseCupidCost,
          routing_cost_total: interactionPhaseRoutingCost,
          total_cost: interactionPhaseCupidCost + interactionPhaseRoutingCost,
          open_test_rounds: sideBySideRounds.length,
          open_test_cost: openTestCostA,
          total_rounds: roundHistory.length,
        },
        system_b: {
          label: 'System B (Baseline)',
          algorithm: 'Bradley-Terry Baseline',
          final_model_id: baselineFinalModelId,
          final_model_name: arenaState?.final_model_b?.model_name || null,
          final_model_stats: baselineFinalStats || null,
          interaction_phase_cost: interactionPhaseBaselineCost,
          total_cost: interactionPhaseBaselineCost,
          open_test_rounds: sideBySideRounds.length,
          open_test_cost: openTestCostB,
          total_rounds: roundHistory.length,
        },
        terminated_early: roundHistory.length < budgetConstraints.maxRounds,
      },

      history: roundHistory,

      open_testing: {
        rounds: sideBySideRounds.length,
        data: sideBySideRounds,
      },

      evaluation: {
        quality_rating_a: evalRatingA,
        quality_rating_b: evalRatingB,
        budget_rating_a: evalBudgetRatingA,
        budget_rating_b: evalBudgetRatingB,
        comment: evalComment
      },
    };
  };

  // Save results to database
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [saveMessage, setSaveMessage] = useState('');

  const saveResultsToDatabase = async () => {
    setSaveStatus('saving');
    console.log('[saveResultsToDatabase] Starting save...');
    try {
      const results = buildResultsObject();
      console.log('[saveResultsToDatabase] Built results object:', {
        session_id: results.session_id,
        persona_group: results.persona_group,
        initial_preference: results.initial_preference,
        history_length: results.history?.length,
        has_evaluation: !!results.evaluation
      });

      const response = await fetch(`${API_URL}/save-results`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(results),
      });

      console.log('[saveResultsToDatabase] Response status:', response.status);
      const data = await response.json();
      console.log('[saveResultsToDatabase] Response data:', data);

      if (data.success && data.saved) {
        setSaveStatus('saved');
        setSaveMessage('Results saved to database successfully!');
        console.log('[saveResultsToDatabase] ✅ Save successful');
        return true;
      } else {
        setSaveStatus('error');
        setSaveMessage(data.message || 'Failed to save results');
        console.error('[saveResultsToDatabase] ❌ Save failed:', data.message);
        return false;
      }
    } catch (err) {
      console.error('[saveResultsToDatabase] ❌ Error:', err);
      setSaveStatus('error');
      setSaveMessage('Failed to connect to server. Please download results manually.');
      return false;
    }
  };


  const verifyImageResultsSaved = async () => {
    setSaveStatus('saving');
    setSaveMessage('Checking image results in database...');
    console.log('[verifyImageResultsSaved] Starting check...', { sessionId });

    if (!sessionId) {
      setSaveStatus('error');
      setSaveMessage('Missing session ID. Please download results manually.');
      return false;
    }

    try {
      const url = `${API_URL}/image-results?session_id=${encodeURIComponent(sessionId)}&limit=1`;
      const response = await fetch(url);
      console.log('[verifyImageResultsSaved] Response status:', response.status);

      let data: any = null;
      try {
        data = await response.json();
      } catch (_) {
        data = null;
      }

      if (!response.ok) {
        const msg =
          (data && (data.detail || data.message)) ||
          `Failed to verify image results (HTTP ${response.status}). Please download results manually.`;
        setSaveStatus('error');
        setSaveMessage(msg);
        return false;
      }

      const count = typeof data?.count === 'number' ? data.count : 0;

      if (count > 0) {
        setSaveStatus('saved');
        setSaveMessage('Image results saved to database successfully!');
        console.log('[verifyImageResultsSaved] ✅ Image results found for this session');
        return true;
      }

      setSaveStatus('error');
      setSaveMessage('No image results found in the database for this session. Please download the JSON as a backup.');
      console.warn('[verifyImageResultsSaved] ⚠️ No image results returned');
      return false;

    } catch (err) {
      console.error('[verifyImageResultsSaved] ❌ Error:', err);
      setSaveStatus('error');
      setSaveMessage('Failed to connect to server. Please download results manually.');
      return false;
    }
  };

  const submitFinalVotes = useCallback(async () => {
    if (!sessionId || !cupidVote || !baselineVote) return;

    try {
      const res = await fetch(`${API_URL}/vote`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          mode,
          cupid_vote: cupidVote,
          baseline_vote: baselineVote,
        }),
      });

      if (!res.ok) {
        // Don’t block the user, but log so you can see it in devtools
        const err = await res.json().catch(() => null);
        console.warn('Final vote sync failed:', res.status, err);
      }
    } catch (e) {
      console.warn('Final vote sync error:', e);
    }
  }, [sessionId, mode, cupidVote, baselineVote]);


  const handleFinalSubmit = async () => {
    await saveSessionData();

    // Save final results once. Backend routes text vs image into separate tables.
    await saveResultsToDatabase();

    // Show final steps / (optional) download modal
    setShowDownloadReminder(true);
  };

  // const handleDownloadAndFinish = () => {
  //   // downloadResults();
  //   setHasDownloaded(true);
  // };

  const handleFinishStudy = () => {
    setShowDownloadReminder(false);
    setFinished(true);
  };

  const downloadResults = () => {
    const results = buildResultsObject();

    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${mode === 'image' ? 'image_gen' : 'llm'}_matching_results_${sessionId}.json`;
    a.click();
    setHasDownloaded(true);
  };

  const getModelStats = (system: 'cupid' | 'baseline', side: 'left' | 'right'): ModelStats | null => {
    if (!arenaState) return null;
    const pair = system === 'cupid' ? arenaState.cupid_pair : arenaState.baseline_pair;
    return side === 'left' ? pair.left_stats || null : pair.right_stats || null;
  };

  // Model Specification Modal
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
                  <div className="text-center"><div className="text-2xl font-bold text-purple-600">{stats.intelligence ?? '—'}</div><div className="text-xs text-gray-500">Intelligence</div></div>
                  <div className="text-center"><div className="text-2xl font-bold text-blue-600">{stats.speed ?? '—'}</div><div className="text-xs text-gray-500">Speed</div></div>
                  <div className="text-center"><div className="text-2xl font-bold text-indigo-600">{stats.reasoning ? 'Yes' : 'No'}</div><div className="text-xs text-gray-500">Reasoning</div></div>
                </div>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-bold text-gray-700 mb-3">Pricing</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-white p-3 rounded-lg border border-green-200">
                    <div className="text-xs text-gray-500 mb-1">Input Price</div>
                    <div className="text-lg font-bold text-green-600">${stats.input_price ?? '—'} <span className="text-xs font-normal text-gray-500">/ 1M tokens</span></div>
                  </div>
                  <div className="bg-white p-3 rounded-lg border border-green-200">
                    <div className="text-xs text-gray-500 mb-1">Output Price</div>
                    <div className="text-lg font-bold text-green-700">${stats.output_price ?? '—'} <span className="text-xs font-normal text-gray-500">/ 1M tokens</span></div>
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
    const isSelected = voteState === side;
    const label = side === 'left' ? '1' : '2';
    const borderColor = isSelected ? 'border-blue-600' : 'border-gray-200 hover:border-gray-300';
    const bgColor = isSelected ? 'bg-blue-50' : 'bg-white';
    const buttonBg = isSelected ? 'bg-blue-600' : 'bg-gray-100';
    const stats = getModelStats(system, side);

    // Show loading state if data not available
    if (!data) {
      return (
        <div className={`relative p-4 rounded-xl border-2 transition-all duration-200 flex flex-col ${borderColor} ${bgColor}`}>
          <div className="animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
            <div className="h-32 bg-gray-100 rounded mb-3"></div>
            {personaGroup === 'traditional' && (
              <div className="border-t pt-3 mt-2">
                <div className="text-xs font-bold text-purple-700 mb-2 flex items-center gap-1">
                  <Target size={12} /> Model Specifications
                </div>
                <div className="text-xs text-purple-400 italic p-2 bg-purple-50 rounded animate-pulse">
                  Loading model specifications...
                </div>
              </div>
            )}
            <div className="h-10 bg-gray-200 rounded mt-3"></div>
          </div>
        </div>
      );
    }

    return (
      <div className={`relative p-4 rounded-xl border-2 transition-all duration-200 flex flex-col ${borderColor} ${bgColor} ${isSelected ? 'shadow-lg scale-[1.01]' : ''}`}>
        <div className="flex justify-between items-center mb-2">
          <span className="text-xs font-bold text-gray-500">Output {label}</span>
          {/* Hide cost for traditional group to reduce info overload - focus on specs */}
          {personaGroup !== 'traditional' && (
            <div className="group relative">
              <span className="text-xs text-gray-400 cursor-help flex items-center gap-1">
                <DollarSign size={12} />
                {(data.cost ?? 0).toFixed(5)}
              </span>
            </div>
          )}
        </div>

        {/* Content rendered based on mode */}
        <div
          onClick={() => setVote(side)}
          className={`flex-grow cursor-pointer mb-3 ${mode === 'image'
            ? 'h-48 md:h-64 flex items-center justify-center overflow-hidden'
            : 'overflow-y-auto h-48 md:h-auto md:max-h-64'
            }`}
        >
          {data.text ? (
            mode === 'image' ? (
              <ImageDisplay imageUrl={data.text} alt={`Generated image ${label}`} />
            ) : (
              <Markdown content={data.text} />
            )
          ) : (
            <span className="text-gray-400 italic">{mode === 'image' ? 'No image generated' : 'No response'}</span>
          )}
        </div>

        {/* Model Specification - Simplified for Traditional Group */}
        {personaGroup === 'traditional' && (
          <div className="border-t pt-3 mt-2">
            <div className="text-xs font-bold text-purple-700 mb-2 flex items-center gap-1">
              <Target size={12} /> Model Specifications
            </div>
            {stats ? (
              <div className="grid grid-cols-4 gap-1.5 text-xs">
                <div className="bg-purple-50 p-1.5 rounded text-center">
                  <div className="text-gray-500 text-[10px]">Intelligence</div>
                  <div className="font-bold text-purple-700">{stats.intelligence ?? '—'}</div>
                </div>
                <div className="bg-blue-50 p-1.5 rounded text-center">
                  <div className="text-gray-500 text-[10px]">Speed</div>
                  <div className="font-bold text-blue-700">{stats.speed ?? '—'}</div>
                </div>
                <div className="bg-indigo-50 p-1.5 rounded text-center">
                  <div className="text-gray-500 text-[10px]">Reasoning</div>
                  <div className="font-bold text-indigo-700">{stats.reasoning ? 'Yes' : 'No'}</div>
                </div>
                <div className="bg-green-50 p-1.5 rounded text-center">
                  <div className="text-gray-500 text-[10px]">Input $/1M</div>
                  <div className="font-bold text-green-700">${stats.input_price ?? '—'}</div>
                </div>
                <div className="bg-green-50 p-1.5 rounded text-center">
                  <div className="text-gray-500 text-[10px]">Output $/1M</div>
                  <div className="font-bold text-green-700">${stats.output_price ?? '—'}</div>
                </div>
                <div className="bg-orange-50 p-1.5 rounded text-center">
                  <div className="text-gray-500 text-[10px]">Context</div>
                  <div className="font-bold text-orange-700 text-[10px]">{stats.context_window?.toLocaleString() ?? '—'}</div>
                </div>
                <div className="bg-orange-50 p-1.5 rounded text-center">
                  <div className="text-gray-500 text-[10px]">Max Output</div>
                  <div className="font-bold text-orange-700 text-[10px]">{stats.max_output?.toLocaleString() ?? '—'}</div>
                </div>
                <div className="bg-gray-50 p-1.5 rounded text-center">
                  <div className="text-gray-500 text-[10px]">Func Call</div>
                  <div className="font-bold text-gray-700">{stats.function_calling ? 'Yes' : 'No'}</div>
                </div>
              </div>
            ) : (
              <div className="text-xs text-purple-400 italic p-2 bg-purple-50 rounded animate-pulse">
                Loading specifications...
              </div>
            )}
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
        <div className="bg-white p-3 rounded-lg border border-gray-100 shadow-sm flex justify-between items-center"><span className="text-sm text-gray-500">Total Cost</span><span className="font-mono font-bold text-gray-800 flex items-center"><DollarSign size={14} />{totalCost.toFixed(5)}</span></div>
        <div className="bg-white p-3 rounded-lg border border-gray-100 shadow-sm flex justify-between items-center"><span className="text-sm text-gray-500">Rounds Completed</span><span className="font-mono font-bold text-gray-800">{winCount}</span></div>
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

  // MODE SELECTION PHASE
  if (phase === 'modeSelect') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-xl max-w-2xl w-full overflow-hidden">
          <div className="bg-gradient-to-r from-purple-600 to-blue-600 p-6 text-white text-center">
            <h1 className="text-2xl font-bold flex items-center justify-center gap-3">
              <Sparkles size={28} />
              LLM Selection
            </h1>
            <p className="mt-2 opacity-90">Choose which type of AI models to compare</p>
          </div>
          <div className="p-8">
            <div className="grid grid-cols-2 gap-6">
              {/* Text/LLM Mode */}
              <button
                onClick={handleSelectTextMode}
                className="p-6 rounded-2xl border-2 border-gray-200 hover:border-blue-500 hover:bg-blue-50 transition-all text-center group"
              >
                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-blue-200 transition-colors">
                  <MessageSquare className="text-blue-600" size={32} />
                </div>
                <h2 className="text-xl font-bold text-gray-800 mb-2">LLM World</h2>
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
                onClick={handleSelectImageMode}
                className="p-6 rounded-2xl border-2 border-gray-200 hover:border-purple-500 hover:bg-purple-50 transition-all text-center group"
              >
                <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-purple-200 transition-colors">
                  <Wand2 className="text-purple-600" size={32} />
                </div>
                <h2 className="text-xl font-bold text-gray-800 mb-2">Image World</h2>
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

            {/* Image World access modal (only shows when Image World is gated) */}
            {renderImageGateModal()}
          </div>
        </div>
      </div>
    );
  }

  // CONSENT PHASE
  if (phase === 'consent') {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="max-w-3xl w-full bg-white shadow-xl rounded-2xl overflow-hidden flex flex-col">
          <div
            className={`p-6 md:p-8 text-white text-center ${mode === 'image'
              ? 'bg-gradient-to-r from-purple-600 to-indigo-600'
              : 'bg-gradient-to-r from-blue-600 to-indigo-600'
              }`}
          >
            <h1 className="text-2xl md:text-3xl font-bold">
              {mode === 'image' ? 'Image Generation Study' : 'LLM Selection Study'}
            </h1>
            <p className="opacity-90">
              {mode === 'image' ? 'Find Your Ideal Image Model' : 'Find Your Dream Model'}
            </p>
            <p className="text-xs mt-2 opacity-75">IRB ID: STUDY00023557</p>
          </div>

          {/* ✅ Replace the consent body with this conditional block */}
          <div className="p-6 md:p-8 overflow-y-auto max-h-[60vh] prose prose-sm max-w-none text-gray-700">
            {mode === 'image' ? (
              <>
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                  <p className="font-semibold text-blue-800 mb-1">🎯 Goal of This Study</p>
                  <p className="text-blue-700 text-sm">
                    Help us compare two image model selection systems. Your job is to choose which generated images you prefer (considering
                    quality + cost) so we can see which system learns your preferences better.
                  </p>
                </div>

                <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 mb-4">
                  <p className="font-semibold mb-1">TL;DR</p>
                  <p className="text-sm m-0">
                    Describe an image you want → each system shows a <strong>duel (2 images)</strong> → pick a winner in{' '}
                    <strong>System A</strong> and in <strong>System B</strong> → optional “cheaper / more realistic / more creative”
                    feedback → repeat → then play & rate.
                  </p>
                </div>

                <h3 className="text-lg font-semibold mt-2">What you’ll see</h3>
                <ul className="ml-5 list-disc">
                  <li>
                    Two panels: <strong>System A</strong> and <strong>System B</strong>.
                  </li>
                  <li>
                    In <strong>each</strong> system, you will see <strong>two candidate images</strong> for the same prompt.
                  </li>
                  <li>
                    Costs are tracked <strong>separately</strong> for System A vs System B (you’ll see how much you spent in each).
                  </li>
                  <li>
                    Model names are hidden, but you may see metadata (e.g., ratings, estimated costs). The systems choose among{' '}
                    <strong>12 text-to-image models</strong> from <strong>Google</strong> and <strong>OpenAI</strong>.
                  </li>
                </ul>

                <h3 className="text-lg font-semibold mt-4">What you do each round (the exact loop)</h3>
                <ol className="ml-5 list-decimal">
                  <li>
                    <strong>Enter an image prompt</strong> (anything realistic: photorealistic, illustration, logo idea, product mock,
                    scene description, etc.).
                  </li>
                  <li>
                    <strong>Read the duels</strong>:
                    <ul className="ml-5 list-disc mt-1">
                      <li>System A shows Image A1 vs Image A2</li>
                      <li>System B shows Image B1 vs Image B2</li>
                    </ul>
                  </li>
                  <li>
                    <strong>Pick winners (two choices every round)</strong>:
                    <ul className="ml-5 list-disc mt-1">
                      <li>Choose the winner in <strong>System A</strong></li>
                      <li>Choose the winner in <strong>System B</strong></li>
                    </ul>
                  </li>
                  <li>
                    <strong>(Optional) Give language feedback</strong> to guide what you want next (examples:{' '}
                    <em>“cheaper model”</em>, <em>“more realistic”</em>, <em>“more creative”</em>, <em>“better text rendering”</em>). This
                    is optional and should reflect your preference, not “correctness.”
                  </li>
                  <li>
                    <strong>Repeat</strong> until you run out of rounds or you feel satisfied and choose to stop early.
                  </li>
                </ol>

                <h3 className="text-lg font-semibold mt-4">How to compare (what “better” means)</h3>
                <p className="text-sm">Choose based on what you personally value. Most participants weigh:</p>
                <ul className="ml-5 list-disc">
                  <li>
                    <strong>Image quality</strong> (visual appeal, sharpness, coherence)
                  </li>
                  <li>
                    <strong>Prompt fit</strong> (does it match what you asked for?)
                  </li>
                  <li>
                    <strong>Style fit</strong> (realistic vs stylized, composition, vibe)
                  </li>
                  <li>
                    <strong>Cost</strong> (is the extra cost worth it?)
                  </li>
                </ul>
                <p className="text-sm">It’s okay if your preferences change as you see more outputs — just choose what you prefer in the moment.</p>

                <h3 className="text-lg font-semibold mt-4">After drafting: play & rate</h3>
                <p className="text-sm">
                  After the drafting/matching phase, you can freely “play” with the final chosen model from each system (up to{' '}
                  <strong>10 rounds per system</strong>), then rate each system on:
                </p>
                <ul className="ml-5 list-disc">
                  <li>Overall image quality</li>
                  <li>Budget compliance</li>
                </ul>

                <h3 className="text-lg font-semibold mt-4">Important notes</h3>
                <ul className="ml-5 list-disc text-sm">
                  <li>
                    Please <strong>do not try to guess</strong> which system/model produced which image. Focus on your preference.
                  </li>
                  <li>Take your time to view both duels before choosing winners.</li>
                  <li>Your responses are anonymous and used only for research purposes.</li>
                </ul>

                <div className="bg-amber-50 border border-amber-300 rounded-lg p-4 mt-4">
                  <p className="font-semibold text-amber-800 mb-1">⚠️ Privacy Notice</p>
                  <p className="text-amber-700 text-sm">
                    Please <strong>do not enter any personal or sensitive information</strong> in your prompts. Avoid names, addresses,
                    phone numbers, or identifying details. Prompts and generated images will be collected for research.
                  </p>
                </div>

                <p className="text-xs text-gray-500 mt-4 border-t pt-4">
                  Questions? Contact: xinyua11@asu.edu, snguye88@asu.edu, ransalu@asu.edu
                  <br />
                  ASU IRB: (480) 965-6788
                </p>
              </>
            ) : (
              <>
                {/* ✅ LLM version kept EXACTLY the same as your previous content */}
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                  <p className="font-semibold text-blue-800 mb-1">🎯 Goal of This Study</p>
                  <p className="text-blue-700 text-sm">
                    Help us compare two LLM selection systems. Your job is to choose which answers you prefer (considering quality + cost)
                    so we can see which system learns your preferences better.
                  </p>
                </div>

                <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 mb-4">
                  <p className="font-semibold mb-1">TL;DR</p>
                  <p className="text-sm m-0">
                    Ask a question → each system shows a <strong>duel (2 answers)</strong> → pick a winner in <strong>System A</strong> and in{" "}
                    <strong>System B</strong> → optional “cheaper / smarter / longer context” feedback → repeat → then play & rate.
                  </p>
                </div>

                <h3 className="text-lg font-semibold mt-2">What you’ll see</h3>
                <ul className="ml-5 list-disc">
                  <li>
                    Two panels: <strong>System A</strong> and <strong>System B</strong>.
                  </li>
                  <li>
                    In <strong>each</strong> system, you will see <strong>two candidate outputs</strong> for the same query.
                  </li>
                  <li>
                    Costs are tracked <strong>separately</strong> for System A vs System B (you’ll see how much you spent in each).
                  </li>
                  <li>
                    Model names are hidden, but you may see metadata (e.g., ratings, input/output costs). The systems choose among{" "}
                    <strong>25 OpenAI models</strong>.
                  </li>
                </ul>

                {/* Constraint-based (Traditional group) note */}
                <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-4 mt-4">
                  <p className="font-semibold text-emerald-900 mb-1">✅ If you were assigned constraints (Traditional group)</p>
                  <p className="text-emerald-800 text-sm m-0">
                    You may see a small list of constraints (e.g., “input price ≤ $2/1M”, “reasoning required”, “intelligence ≥ 4”).
                    In this mode, treat constraints as <strong>requirements</strong>.
                  </p>
                  <ul className="ml-5 list-disc text-sm text-emerald-800 mt-2">
                    <li>
                      When picking winners in each duel, <strong>prefer answers that come from models satisfying the constraints</strong>.
                    </li>
                    <li>
                      If <strong>both</strong> options satisfy constraints, choose based on <strong>quality + cost</strong> as usual.
                    </li>
                    <li>
                      If <strong>neither</strong> satisfies constraints, pick the option that seems <strong>closest</strong> (and you can use feedback like “I want reasoning model”).
                    </li>
                  </ul>
                </div>

                <h3 className="text-lg font-semibold mt-4">What you do each round (the exact loop)</h3>
                <ol className="ml-5 list-decimal">
                  <li>
                    <strong>Enter a query</strong> (anything realistic: explain, write, debug, brainstorm, etc.).
                  </li>
                  <li>
                    <strong>Read the duels</strong>:
                    <ul className="ml-5 list-disc mt-1">
                      <li>System A shows Answer A1 vs Answer A2</li>
                      <li>System B shows Answer B1 vs Answer B2</li>
                    </ul>
                  </li>
                  <li>
                    <strong>Pick winners (two choices every round)</strong>:
                    <ul className="ml-5 list-disc mt-1">
                      <li>Choose the winner in <strong>System A</strong></li>
                      <li>Choose the winner in <strong>System B</strong></li>
                    </ul>
                  </li>
                  <li>
                    <strong>(Optional) Give language feedback</strong> to guide what you want next (examples:{" "}
                    <em>“cheaper model”</em>, <em>“stronger reasoning”</em>). This is optional
                    and should reflect your preference, not “correctness.”
                  </li>
                  <li>
                    <strong>Repeat</strong> until you run out of rounds or you feel satisfied and choose to stop early.
                  </li>
                </ol>

                <h3 className="text-lg font-semibold mt-4">How to compare (what “better” means)</h3>
                <p className="text-sm">
                  Choose based on what you personally value. Most participants weigh:
                </p>
                <ul className="ml-5 list-disc">
                  <li><strong>Quality</strong> (helpfulness, correctness, clarity)</li>
                  <li><strong>Cost</strong> (is the extra cost worth it?)</li>
                  <li><strong>Style fit</strong> (detail level, tone, structure)</li>
                </ul>
                <p className="text-sm">
                  It’s okay if your preferences change as you see more outputs — just choose what you prefer in the moment.
                </p>

                <h3 className="text-lg font-semibold mt-4">After drafting: play & rate</h3>
                <p className="text-sm">
                  After the drafting/matching phase, you can freely “play” with the final chosen model from each system (up to{" "}
                  <strong>10 rounds per system</strong>), then rate each system on:
                </p>
                <ul className="ml-5 list-disc">
                  <li>Overall output quality</li>
                  <li>Budget compliance</li>
                </ul>

                <h3 className="text-lg font-semibold mt-4">Important notes</h3>
                <ul className="ml-5 list-disc text-sm">
                  <li>
                    Please <strong>do not try to guess</strong> which system/model produced which output. Focus on your preference.
                  </li>
                  <li>Take your time to read both duels before choosing winners.</li>
                  <li>Your responses are anonymous and used only for research purposes.</li>
                </ul>

                <div className="bg-amber-50 border border-amber-300 rounded-lg p-4 mt-4">
                  <p className="font-semibold text-amber-800 mb-1">⚠️ Privacy Notice</p>
                  <p className="text-amber-700 text-sm">
                    Please <strong>do not enter any personal or sensitive information</strong> in your prompts. Avoid names, addresses,
                    phone numbers, or identifying details. Prompts and model responses will be collected for research.
                  </p>
                </div>

                <p className="text-xs text-gray-500 mt-4 border-t pt-4">
                  Questions? Contact: xinyua11@asu.edu, snguye88@asu.edu, ransalu@asu.edu
                  <br />
                  ASU IRB: (480) 965-6788
                </p>
              </>
            )}
          </div>

          <div className="p-4 md:p-6 bg-gray-50 border-t flex flex-col items-center gap-4">
            <p className="text-xs md:text-sm text-gray-600 text-center max-w-xl">
              By clicking below, you confirm you are at least 18 years old and agree to participate.
            </p>
            <button
              onClick={handleConsent}
              className="bg-blue-600 text-white px-8 py-3 rounded-full font-bold hover:bg-blue-700 transition-transform transform hover:scale-105 flex items-center"
            >
              <CheckCircle size={20} className="mr-2" /> I Agree to Participate
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
            <p className="text-sm text-gray-500 mb-4">You have up to {budgetConstraints.maxRounds} rounds to find your ideal model</p>

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
              <label className="block text-sm font-medium text-gray-700 mb-1">
                {mode === 'image' ? 'Enter your first image prompt:' : 'Enter your first query:'}
              </label>
              <textarea className="w-full border rounded-lg p-3 focus:ring-2 focus:ring-blue-500 outline-none resize-none" rows={5} value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder={mode === 'image' ? "Describe the image you want to generate..." : "Type your question or task here..."} />
              <p className="text-xs text-amber-600 mt-2 flex items-center gap-1">
                <AlertCircle size={12} />
                <span><strong>Privacy:</strong> Do not enter personal information. Prompts and responses are collected for research.</span>
              </p>
            </div>

            {/* Initial Preference Field */}
            <div className="mb-4 bg-gradient-to-r from-indigo-50 to-purple-50 p-4 rounded-lg border border-indigo-200 text-left">
              <div className="flex items-center gap-2 mb-2">
                <Sparkles size={18} className="text-indigo-600" />
                <label className="text-sm font-bold text-indigo-900">
                  {mode === 'image' ? 'What kind of images do you want to generate?' : 'What kind of LLM would you like?'} <span className="text-red-500">*</span>
                </label>
              </div>
              <p className="text-xs text-indigo-700 mb-3">
                {mode === 'image'
                  ? 'Tell the system your preferences to help find the best image generation model for you. For example: "realistic photos", "artistic style", "anime illustrations", "high detail", etc.'
                  : 'Tell the system your preferences to help find the best model for you. For example: "a cheap model", "a friendly assistant", "technology-focused", "fast responses", etc.'
                }
              </p>
              <input
                type="text"
                className="w-full border border-indigo-200 rounded-lg p-3 text-sm focus:ring-2 focus:ring-indigo-500 outline-none bg-white"
                placeholder={mode === 'image'
                  ? 'e.g., "I want realistic photos", "I prefer artistic illustrations", "Something with vibrant colors"'
                  : 'e.g., "I want a cheap model", "I prefer detailed explanations", "Something fast and concise"'
                }
                value={initialPreference}
                onChange={(e) => setInitialPreference(e.target.value)}
              />
            </div>

            <button onClick={startSession} disabled={!prompt.trim() || !initialPreference.trim() || loading} className="w-full bg-blue-600 text-white py-3 rounded-lg font-bold hover:bg-blue-700 disabled:opacity-50 transition flex items-center justify-center gap-2">{loading ? (<><RefreshCw size={16} className="animate-spin" />Starting...</>) : 'Start Comparing'}</button>
          </div>
        </div>
      );
    }

    const systemACost = (arenaState?.cupid_cost || 0) + (arenaState?.routing_cost || 0);
    const systemBCost = arenaState?.baseline_cost || 0;
    const totalCost = systemACost + systemBCost;
    const isLastRound = arenaState && (arenaState.round >= budgetConstraints.maxRounds || totalCost >= budgetConstraints.maxCost);
    const halfwayPoint = Math.ceil(budgetConstraints.maxRounds / 2);
    const canEndEarly = arenaState && arenaState.round >= halfwayPoint;

    return (
      <div className="min-h-screen bg-gray-50 flex flex-col">
        {renderModelInfoModal()}

        {/* Simplified Header */}
        <header className="bg-white border-b sticky top-0 z-20">
          <div className="max-w-7xl mx-auto px-4 h-14 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className={`${mode === 'image' ? 'bg-purple-600' : 'bg-blue-600'} text-white px-2 py-1 rounded text-xs font-bold`}>
                {mode === 'image' ? 'IMAGE GENERATION' : 'LLM MATCHMAKING'}
              </div>
            </div>
            <div className="flex items-center space-x-3 text-sm font-mono">
              <div className="flex items-center"><span className="text-gray-400 mr-1">Round</span><span className="font-bold">{arenaState?.round || 0}/{budgetConstraints.maxRounds}</span></div>
            </div>
          </div>
        </header>

        {/* Sticky Constraints Panel for Traditional Group */}
        {personaGroup === 'traditional' && assignedConstraints.length > 0 && (
          <div className="sticky top-14 z-10 bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-lg">
            <div className="max-w-7xl mx-auto px-4 py-3">
              <div className="flex flex-wrap items-center gap-3">
                <div className="flex items-center gap-2">
                  <Target size={18} className="text-purple-200" />
                  <span className="font-bold text-sm">Your Requirements:</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {assignedConstraints.map((c, i) => (
                    <span key={i} className="bg-white/20 text-white text-xs px-3 py-1 rounded-full font-medium backdrop-blur-sm border border-white/30">
                      {formatConstraint(c)}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        <main className="flex-grow max-w-7xl mx-auto px-4 py-4 w-full flex flex-col gap-5 pb-56 md:pb-8">
          {loading && <div className="fixed inset-0 bg-white/80 backdrop-blur-sm z-50 flex items-center justify-center"><div className="flex flex-col items-center"><div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div><p className="font-mono text-sm">Getting responses...</p></div></div>}
          {error && <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 flex items-center gap-2"><AlertCircle size={20} />{error}</div>}

          {/* Task Instructions based on persona group */}
          {personaGroup === 'traditional' && (
            <div className="bg-purple-50 border-2 border-purple-300 rounded-xl p-4">
              <div className="flex items-start gap-3">
                <div className="bg-purple-600 text-white p-2 rounded-lg flex-shrink-0">
                  <Target size={20} />
                </div>
                <div>
                  <h3 className="font-bold text-purple-900 mb-1">Your Task: Find a Model that Meets Your Requirements</h3>
                  <p className="text-sm text-purple-800">
                    Compare the <strong>Model Specifications</strong> below each response against your requirements shown above.
                    Select the output from the model that better matches your constraints, then use the feedback field to guide the system
                    (e.g., "I need a cheaper model" or "Give me higher intelligence").
                  </p>
                </div>
              </div>
            </div>
          )}

          {personaGroup === 'expert' && selectedExpertSubject && (
            <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 flex items-center gap-3">
              <BookOpen size={20} className="text-blue-600 flex-shrink-0" />
              <div>
                <span className="font-bold text-blue-900">Expert Mode: {EXPERT_SUBJECTS.find(s => s.id === selectedExpertSubject)?.label}</span>
                <p className="text-sm text-blue-700">Select the model that performs best for your domain expertise. Use feedback to guide the system.</p>
              </div>
            </div>
          )}

          {personaGroup === 'preference' && (
            <div className="bg-indigo-50 border border-indigo-200 rounded-xl p-4 flex items-center gap-3">
              <ThumbsUp size={20} className="text-indigo-600 flex-shrink-0" />
              <div>
                <span className="font-bold text-indigo-900">Personal Preference Mode</span>
                <p className="text-sm text-indigo-700">Choose based on your own criteria — quality, style, helpfulness, or anything you value.</p>
              </div>
            </div>
          )}

          {/* No Chat History Note - simplified */}
          <div className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-2 flex items-center gap-2">
            <AlertCircle size={14} className="text-amber-600 flex-shrink-0" />
            <p className="text-xs text-amber-700"><strong>Note:</strong> Models don't remember previous queries. Each round is independent. <strong>Do not enter personal information</strong> — prompts and responses are collected for research.</p>
          </div>

          <div className="bg-white p-4 rounded-lg shadow-sm border"><span className="text-xs font-bold text-gray-400 uppercase">Your Query</span><p className="text-gray-800 font-medium mt-1">{prompt}</p></div>

          {/* System A */}
          <section>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-blue-600 font-bold text-lg">System A</h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">{renderModelCard('left', arenaState?.cupid_pair.left, cupidVote, setCupidVote, 'blue', 'cupid')}{renderModelCard('right', arenaState?.cupid_pair.right, cupidVote, setCupidVote, 'blue', 'cupid')}</div>
            <div className="mt-4 bg-blue-50 p-4 rounded-lg border border-blue-100">
              <label className="flex items-center text-sm font-bold text-blue-900 mb-2"><MessageSquare size={16} className="mr-2" />Language Feedback (optional)</label>
              <input
                type="text"
                className="w-full border border-blue-200 rounded p-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                placeholder={personaGroup === 'traditional'
                  ? 'e.g., "I need a cheaper model", "Give me higher intelligence", "I want faster speed"'
                  : 'e.g., "Please give me a smarter model" or "I prefer more concise responses"'
                }
                value={feedbackA}
                onChange={(e) => setFeedbackA(e.target.value)}
              />
            </div>
          </section>

          <hr className="border-gray-200" />

          {/* System B */}
          <section>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-blue-600 font-bold text-lg">System B</h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">{renderModelCard('left', arenaState?.baseline_pair.left, baselineVote, setBaselineVote, 'blue', 'baseline')}{renderModelCard('right', arenaState?.baseline_pair.right, baselineVote, setBaselineVote, 'blue', 'baseline')}</div>
            <div className="mt-4 bg-blue-50 p-4 rounded-lg border border-blue-100">
              <label className="flex items-center text-sm font-bold text-blue-900 mb-2"><MessageSquare size={16} className="mr-2" />Language Feedback (optional)</label>
              <input
                type="text"
                className="w-full border border-blue-200 rounded p-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                placeholder={personaGroup === 'traditional'
                  ? 'e.g., "I need a cheaper model", "Give me higher intelligence", "I want faster speed"'
                  : 'e.g., "Please give me a smarter model" or "I prefer more concise responses"'
                }
                value={feedbackB}
                onChange={(e) => setFeedbackB(e.target.value)}
              />
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
                      disabled={!cupidVote || !baselineVote}
                      className={`px-4 py-2 rounded-lg font-medium transition text-sm ${cupidVote && baselineVote
                        ? 'bg-green-100 text-green-700 hover:bg-green-200'
                        : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                        }`}
                      title={!cupidVote || !baselineVote ? 'Please select your preferred response from both systems first' : ''}
                    >
                      ✓ I'm Satisfied — End Drafting
                    </button>
                  )}
                </div>
              </div>

              {!isLastRound && (
                <textarea
                  placeholder={mode === 'image' ? "Enter your next image prompt (required to continue)..." : "Enter your next query (required to continue)..."}
                  className={`w-full border rounded-lg px-3 py-3 text-sm resize-none ${!nextPrompt.trim() && cupidVote && baselineVote ? 'border-red-300 bg-red-50' : ''}`}
                  rows={4}
                  value={nextPrompt}
                  onChange={(e) => setNextPrompt(e.target.value)}
                />
              )}

              <button onClick={handleSubmitRound} disabled={loading} className="w-full md:w-auto md:self-end bg-blue-600 text-white px-8 py-3 rounded-lg font-bold hover:bg-blue-700 disabled:opacity-50 transition">
                {isLastRound ? (mode === 'image' ? 'Continue to Test Image Models →' : 'Continue to Play with Models →') : 'Submit & Next →'}
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
              <span className="text-yellow-700 ml-1">• <strong>Do not enter personal information</strong> — prompts and responses are collected for research.</span>
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
                      <ContentDisplay content={round.responseA} contentType={mode === 'image' ? 'image' : 'text'} />
                    </div>
                  </div>

                  {/* System B Response */}
                  <div className="p-4">
                    <div className="flex justify-between items-center mb-3">
                      <span className="font-bold text-blue-600">System B</span>
                      <span className="text-xs text-gray-400">${round.costB.toFixed(5)}</span>
                    </div>
                    <div className="prose prose-sm max-w-none">
                      <ContentDisplay content={round.responseB} contentType={mode === 'image' ? 'image' : 'text'} />
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {openTestLoading && (
              <div className="bg-white rounded-xl border p-8 text-center">
                <RefreshCw size={24} className="animate-spin mx-auto text-blue-600 mb-2" />
                <p className="text-gray-500 text-sm">
                  {mode === 'image' ? 'Generating images from both models...' : 'Getting responses from both models...'}
                </p>
              </div>
            )}
          </div>

          {/* Input area */}
          <div className="bg-white rounded-xl border p-4 sticky bottom-4">
            <div className="flex gap-2">
              <input
                type="text"
                className="flex-grow border rounded-lg px-4 py-3"
                placeholder={canChat ? (mode === 'image' ? "Enter your image prompt (will be sent to both models)..." : "Enter your prompt (will be sent to both models)...") : `Maximum ${OPEN_TESTING_MAX_ROUNDS} rounds reached`}
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
    // Download Reminder Modal
    if (showDownloadReminder) {
      return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
          <div className="max-w-lg w-full bg-white shadow-2xl rounded-2xl overflow-hidden">
            {/* Header - changes based on save status */}
            <div className={`p-6 text-white text-center ${saveStatus === 'saved' ? 'bg-gradient-to-r from-green-500 to-emerald-600' : 'bg-gradient-to-r from-orange-500 to-red-500'}`}>
              {saveStatus === 'saved' ? (
                <Database className="mx-auto mb-3" size={48} />
              ) : saveStatus === 'saving' ? (
                <RefreshCw className="mx-auto mb-3 animate-spin" size={48} />
              ) : (
                <AlertCircle className="mx-auto mb-3" size={48} />
              )}
              <h1 className="text-2xl font-bold">
                {saveStatus === 'saved' ? 'Results Saved to Database!' : saveStatus === 'saving' ? 'Saving Results...' : 'Final Steps (Required)'}
              </h1>
              <p className="opacity-90 mt-2">
                {saveStatus === 'saved' ? (
                  <>Your data has been automatically saved. Download a backup copy if you'd like.</>
                ) : (
                  <><strong>Step 1: Download</strong> your JSON → <strong>Step 2: Upload</strong> it to the survey</>
                )}
              </p>
              {saveStatus !== 'saved' && (
                <p className="text-sm mt-2 opacity-90">
                  <strong>Important:</strong> Downloading does <u>not</u> submit your participation — you must upload the file.
                </p>
              )}
            </div>

            <div className="p-8">
              {/* Database Save Status Indicator */}
              {(saveStatus === 'saving' || saveStatus === 'saved' || saveStatus === 'error') && (
                <div className={`rounded-xl p-4 mb-5 border-2 ${saveStatus === 'saved' ? 'bg-green-50 border-green-300' :
                  saveStatus === 'error' ? 'bg-red-50 border-red-300' :
                    'bg-blue-50 border-blue-300'
                  }`}>
                  <div className="flex items-center gap-3">
                    {saveStatus === 'saved' && <CheckCircle className="text-green-600" size={24} />}
                    {saveStatus === 'error' && <AlertCircle className="text-red-600" size={24} />}
                    {saveStatus === 'saving' && <RefreshCw className="text-blue-600 animate-spin" size={24} />}
                    <div>
                      <span className={`font-bold ${saveStatus === 'saved' ? 'text-green-800' :
                        saveStatus === 'error' ? 'text-red-800' :
                          'text-blue-800'
                        }`}>
                        {saveStatus === 'saved' ? '✓ Auto-Saved to Database' :
                          saveStatus === 'error' ? 'Database Save Failed' :
                            'Saving to Database...'}
                      </span>
                      {saveMessage && <p className="text-sm text-gray-600 mt-1">{saveMessage}</p>}
                    </div>
                  </div>
                </div>
              )}


              {/* Strong warning (always visible until confirmed) - only if NOT saved to database */}
              {!confirmedUploaded && saveStatus !== 'saved' && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
                  <p className="text-red-700 text-sm flex items-center gap-2">
                    <AlertCircle size={16} />
                    <strong>Do not close this page yet.</strong> Downloading does not submit — you must upload the JSON and submit the survey.
                  </p>
                </div>
              )}

              {/* Finish button - enabled immediately if saved to database */}
              <button
                onClick={handleFinishStudy}
                disabled={saveStatus !== 'saved' && (!hasDownloaded || !confirmedUploaded)}
                className={`w-full py-4 rounded-xl font-bold flex items-center justify-center gap-3 transition text-lg ${saveStatus === 'saved' || (hasDownloaded && confirmedUploaded)
                  ? "bg-green-600 text-white hover:bg-green-700"
                  : "bg-gray-200 text-gray-400 cursor-not-allowed"
                  }`}
              >
                {saveStatus === 'saved' ? 'Continue' : 'I Uploaded & Submitted — Finish'} <CheckCircle size={22} />
              </button>

              {saveStatus !== 'saved' && (!hasDownloaded || !confirmedUploaded) && (
                <p className="text-center text-gray-400 text-sm mt-3">
                  Finish is enabled after you upload the JSON and submit the survey.
                </p>
              )}

            </div>
          </div>
        </div>
      );
    }


    if (finished) return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="max-w-xl w-full bg-white shadow-xl rounded-2xl p-12 text-center">
          <CheckCircle className="mx-auto text-green-500 mb-6" size={80} />
          <h1 className="text-3xl font-bold mb-2">Thank You!</h1>
          <p className="text-gray-600 mb-6">
            {saveStatus === 'saved'
              ? 'Your results have been saved. You may close this tab.'
              : 'Your submission is complete. You may close this tab.'
            }
          </p>

          {/* Show different content based on save status */}
          {saveStatus === 'saved' ? (
            <div className="bg-green-50 border-2 border-green-200 rounded-xl p-6 text-left">
              <div className="flex items-center gap-3 mb-2">
                <Database className="text-green-600" size={24} />
                <span className="font-bold text-green-800">Results Saved Successfully</span>
              </div>
              <p className="text-sm text-gray-600">
                Your study data has been automatically saved to our database. No further action required!
              </p>
              {hasDownloaded && (
                <p className="text-xs text-gray-500 mt-2">A backup copy was also downloaded to your device.</p>
              )}
            </div>
          ) : (
            <div className="bg-gray-50 border rounded-xl p-4 text-left">
              <p className="text-sm text-gray-600">
                If you need the survey link again:
              </p>
              <a
                href={SURVEY_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-2 inline-flex items-center justify-center w-full bg-blue-600 text-white py-3 rounded-lg font-bold hover:bg-blue-700 transition"
              >
                <ArrowRight size={18} className="mr-2" /> Open Survey
              </a>
            </div>
          )}

          <p className="text-sm text-gray-400 mt-6">Session: {sessionId}</p>
        </div>
      </div>
    );



    const cupidWins = roundHistory.filter(r => r.cupid_vote).length;
    const baselineWins = roundHistory.filter(r => r.baseline_vote).length;
    const systemACost = (arenaState?.cupid_cost || 0) + (arenaState?.routing_cost || 0);
    const systemBCost = arenaState?.baseline_cost || 0;

    // Get final model stats for traditional group - use the voted model's stats
    const lastRound = roundHistory[roundHistory.length - 1];
    const finalCupidStats = lastRound?.cupid_vote === 'left'
      ? arenaState?.cupid_pair?.left_stats
      : arenaState?.cupid_pair?.right_stats;
    const finalBaselineStats = lastRound?.baseline_vote === 'left'
      ? arenaState?.baseline_pair?.left_stats
      : arenaState?.baseline_pair?.right_stats;

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
            </div>

            {/* Rating cards with optional model info for traditional group */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-10">
              <div className="space-y-4">
                {renderEvalCard("System A", systemACost, evalRatingA, setEvalRatingA, evalBudgetRatingA, setEvalBudgetRatingA, cupidWins)}
                {/* Model info for traditional group */}
                {personaGroup === 'traditional' && (
                  <div className="bg-white border rounded-xl p-4">
                    <h4 className="text-sm font-bold text-gray-600 mb-3 flex items-center gap-2">
                      <Info size={14} /> System A - Final Model Specs
                    </h4>
                    {finalCupidStats ? (
                      <div className="grid grid-cols-4 gap-2 text-xs">
                        <div className="bg-purple-50 p-2 rounded text-center"><div className="text-gray-500 text-[10px]">Intelligence</div><div className="font-bold text-purple-700">{finalCupidStats.intelligence ?? '—'}</div></div>
                        <div className="bg-blue-50 p-2 rounded text-center"><div className="text-gray-500 text-[10px]">Speed</div><div className="font-bold text-blue-700">{finalCupidStats.speed ?? '—'}</div></div>
                        <div className="bg-indigo-50 p-2 rounded text-center"><div className="text-gray-500 text-[10px]">Reasoning</div><div className="font-bold text-indigo-700">{finalCupidStats.reasoning ? 'Yes' : 'No'}</div></div>
                        <div className="bg-green-50 p-2 rounded text-center"><div className="text-gray-500 text-[10px]">Input $/1M</div><div className="font-bold text-green-700">${finalCupidStats.input_price ?? '—'}</div></div>
                        <div className="bg-green-50 p-2 rounded text-center"><div className="text-gray-500 text-[10px]">Output $/1M</div><div className="font-bold text-green-700">${finalCupidStats.output_price ?? '—'}</div></div>
                        <div className="bg-orange-50 p-2 rounded text-center"><div className="text-gray-500 text-[10px]">Context</div><div className="font-bold text-orange-700 text-[10px]">{finalCupidStats.context_window?.toLocaleString() ?? '—'}</div></div>
                        <div className="bg-orange-50 p-2 rounded text-center"><div className="text-gray-500 text-[10px]">Max Output</div><div className="font-bold text-orange-700 text-[10px]">{finalCupidStats.max_output?.toLocaleString() ?? '—'}</div></div>
                        <div className="bg-gray-50 p-2 rounded text-center"><div className="text-gray-500 text-[10px]">Func Call</div><div className="font-bold text-gray-700">{finalCupidStats.function_calling ? 'Yes' : 'No'}</div></div>
                      </div>
                    ) : (
                      <div className="text-xs text-gray-400 italic">Model specifications not available</div>
                    )}
                  </div>
                )}
              </div>

              <div className="space-y-4">
                {renderEvalCard("System B", systemBCost, evalRatingB, setEvalRatingB, evalBudgetRatingB, setEvalBudgetRatingB, baselineWins)}
                {/* Model info for traditional group */}
                {personaGroup === 'traditional' && (
                  <div className="bg-white border rounded-xl p-4">
                    <h4 className="text-sm font-bold text-gray-600 mb-3 flex items-center gap-2">
                      <Info size={14} /> System B - Final Model Specs
                    </h4>
                    {finalBaselineStats ? (
                      <div className="grid grid-cols-4 gap-2 text-xs">
                        <div className="bg-purple-50 p-2 rounded text-center"><div className="text-gray-500 text-[10px]">Intelligence</div><div className="font-bold text-purple-700">{finalBaselineStats.intelligence ?? '—'}</div></div>
                        <div className="bg-blue-50 p-2 rounded text-center"><div className="text-gray-500 text-[10px]">Speed</div><div className="font-bold text-blue-700">{finalBaselineStats.speed ?? '—'}</div></div>
                        <div className="bg-indigo-50 p-2 rounded text-center"><div className="text-gray-500 text-[10px]">Reasoning</div><div className="font-bold text-indigo-700">{finalBaselineStats.reasoning ? 'Yes' : 'No'}</div></div>
                        <div className="bg-green-50 p-2 rounded text-center"><div className="text-gray-500 text-[10px]">Input $/1M</div><div className="font-bold text-green-700">${finalBaselineStats.input_price ?? '—'}</div></div>
                        <div className="bg-green-50 p-2 rounded text-center"><div className="text-gray-500 text-[10px]">Output $/1M</div><div className="font-bold text-green-700">${finalBaselineStats.output_price ?? '—'}</div></div>
                        <div className="bg-orange-50 p-2 rounded text-center"><div className="text-gray-500 text-[10px]">Context</div><div className="font-bold text-orange-700 text-[10px]">{finalBaselineStats.context_window?.toLocaleString() ?? '—'}</div></div>
                        <div className="bg-orange-50 p-2 rounded text-center"><div className="text-gray-500 text-[10px]">Max Output</div><div className="font-bold text-orange-700 text-[10px]">{finalBaselineStats.max_output?.toLocaleString() ?? '—'}</div></div>
                        <div className="bg-gray-50 p-2 rounded text-center"><div className="text-gray-500 text-[10px]">Func Call</div><div className="font-bold text-gray-700">{finalBaselineStats.function_calling ? 'Yes' : 'No'}</div></div>
                      </div>
                    ) : (
                      <div className="text-xs text-gray-400 italic">Model specifications not available</div>
                    )}
                  </div>
                )}
              </div>
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
    );
  }

  return null;
};

export default App;

import React, { useState, useCallback } from 'react';
import { ArenaState, InteractRequest } from './types';
import { ArrowRight, MessageSquare, DollarSign, Activity, Cpu, Trophy, RefreshCw, Sparkles, AlertCircle, CheckCircle } from 'lucide-react';

// API Configuration
const API_URL = 'http://localhost:8000';

interface LoadingState {
  cupidLeft: boolean;
  cupidRight: boolean;
  baselineLeft: boolean;
  baselineRight: boolean;
}

const App: React.FC = () => {
  // Application State
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingStates, setLoadingStates] = useState<LoadingState>({
    cupidLeft: false,
    cupidRight: false,
    baselineLeft: false,
    baselineRight: false,
  });
  const [init, setInit] = useState<boolean>(true);
  const [prompt, setPrompt] = useState<string>('');
  const [nextPrompt, setNextPrompt] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  // Arena Data
  const [arenaState, setArenaState] = useState<ArenaState | null>(null);
  const [showSummary, setShowSummary] = useState<boolean>(false);
  const [summary, setSummary] = useState<any>(null);

  // User Inputs for current round
  const [cupidVote, setCupidVote] = useState<'left' | 'right' | null>(null);
  const [baselineVote, setBaselineVote] = useState<'left' | 'right' | null>(null);
  const [feedback, setFeedback] = useState<string>('');

  // Direction extracted from feedback
  const [extractedDirection, setExtractedDirection] = useState<string>('');

  const fetchNextRound = useCallback(async (isFirst: boolean = false) => {
    setLoading(true);
    setError(null);

    // Set all quadrants to loading
    setLoadingStates({
      cupidLeft: true,
      cupidRight: true,
      baselineLeft: true,
      baselineRight: true,
    });

    const currentPrompt = isFirst ? prompt : (nextPrompt || prompt);

    const payload: InteractRequest = {
      prompt: currentPrompt || "Tell me a fun fact about space.",
      prev_round_info: isFirst ? undefined : {
        cupid_winner: cupidVote,
        cupid_feedback: feedback,
        baseline_winner: baselineVote
      }
    };

    try {
      const res = await fetch(`${API_URL}/interact`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `API Error: ${res.statusText}`);
      }

      const data = await res.json();
      setArenaState(data);
      setExtractedDirection(data.direction_extracted || '');

      // Update prompt for display
      if (!isFirst && nextPrompt) {
        setPrompt(nextPrompt);
      }

      // Reset inputs for next round
      setCupidVote(null);
      setBaselineVote(null);
      setFeedback('');
      setNextPrompt('');
    } catch (err: any) {
      console.error("Failed to fetch round", err);
      setError(err.message || "Error connecting to backend. Ensure FastAPI is running on port 8000.");
    } finally {
      setLoading(false);
      setLoadingStates({
        cupidLeft: false,
        cupidRight: false,
        baselineLeft: false,
        baselineRight: false,
      });
    }
  }, [prompt, nextPrompt, cupidVote, baselineVote, feedback]);

  const startSession = async () => {
    setInit(false);
    await fetchNextRound(true);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!cupidVote || !baselineVote) {
      setError("Please vote on both arenas before proceeding.");
      return;
    }
    fetchNextRound(false);
  };

  const fetchSummary = async () => {
    try {
      const res = await fetch(`${API_URL}/summary`);
      if (res.ok) {
        const data = await res.json();
        setSummary(data);
        setShowSummary(true);
      }
    } catch (err) {
      console.error("Failed to fetch summary", err);
    }
  };

  const resetExperiment = async () => {
    try {
      await fetch(`${API_URL}/reset`, { method: 'POST' });
      setArenaState(null);
      setShowSummary(false);
      setSummary(null);
      setInit(true);
      setPrompt('');
      setNextPrompt('');
      setCupidVote(null);
      setBaselineVote(null);
      setFeedback('');
      setExtractedDirection('');
      setError(null);
    } catch (err) {
      console.error("Failed to reset", err);
    }
  };

  // --- Render Helpers ---

  const renderModelCard = (
    systemName: string,
    side: 'left' | 'right',
    data: any,
    voteState: 'left' | 'right' | null,
    setVote: (v: 'left' | 'right') => void,
    isLoading: boolean,
    colorScheme: 'cupid' | 'baseline'
  ) => {
    if (isLoading || !data) {
      return (
        <div className={`relative p-4 rounded-xl border-2 border-gray-200 bg-white flex flex-col h-full`}>
          <div className="flex justify-between items-center mb-2">
            <div className="h-4 w-24 bg-gray-200 rounded animate-pulse"></div>
            <div className="h-4 w-16 bg-gray-200 rounded animate-pulse"></div>
          </div>
          <div className="flex-grow space-y-2">
            <div className="h-4 bg-gray-100 rounded animate-pulse"></div>
            <div className="h-4 bg-gray-100 rounded animate-pulse w-5/6"></div>
            <div className="h-4 bg-gray-100 rounded animate-pulse w-4/6"></div>
            <div className="h-4 bg-gray-100 rounded animate-pulse w-3/6"></div>
          </div>
          <div className="mt-auto pt-4">
            <div className="h-10 bg-gray-100 rounded animate-pulse"></div>
          </div>
        </div>
      );
    }

    const isSelected = voteState === side;
    const borderColor = colorScheme === 'cupid'
      ? (isSelected ? 'border-violet-500' : 'border-gray-200 hover:border-violet-300')
      : (isSelected ? 'border-emerald-500' : 'border-gray-200 hover:border-emerald-300');
    const bgColor = colorScheme === 'cupid'
      ? (isSelected ? 'bg-violet-50' : 'bg-white')
      : (isSelected ? 'bg-emerald-50' : 'bg-white');
    const buttonBg = colorScheme === 'cupid'
      ? (isSelected ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-500 hover:bg-violet-100')
      : (isSelected ? 'bg-emerald-500 text-white' : 'bg-gray-100 text-gray-500 hover:bg-emerald-100');

    return (
      <div
        onClick={() => setVote(side)}
        className={`
          relative p-4 rounded-xl border-2 cursor-pointer transition-all duration-200 flex flex-col h-full
          ${borderColor} ${bgColor}
          ${isSelected ? 'shadow-lg scale-[1.01]' : ''}
        `}
      >
        {isSelected && (
          <div className={`absolute -top-2 -right-2 ${colorScheme === 'cupid' ? 'bg-violet-500' : 'bg-emerald-500'} rounded-full p-1`}>
            <CheckCircle size={16} className="text-white" />
          </div>
        )}

        <div className="flex justify-between items-center mb-2 text-xs text-gray-500 font-mono">
          <span className="font-semibold text-gray-700">{data.model_name}</span>
          <span className="flex items-center text-green-600">
            <DollarSign size={10} />
            {data.cost?.toFixed(6) || '0.000000'}
          </span>
        </div>

        <div className="flex-grow overflow-y-auto max-h-60 mb-4 text-gray-700 text-sm whitespace-pre-wrap leading-relaxed">
          {data.text}
        </div>

        <div className={`mt-auto text-center font-bold py-2 rounded-lg transition-all ${buttonBg}`}>
          {isSelected ? (
            <span className="flex items-center justify-center gap-2">
              <CheckCircle size={16} /> SELECTED
            </span>
          ) : (
            'Click to Select'
          )}
        </div>
      </div>
    );
  };

  // Summary Modal
  if (showSummary && summary) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="max-w-2xl w-full bg-white rounded-2xl shadow-xl p-8">
          <div className="flex items-center justify-center mb-6">
            <Trophy size={48} className="text-yellow-500" />
          </div>
          <h1 className="text-2xl font-bold text-center mb-2">Experiment Summary</h1>
          <p className="text-gray-500 text-center mb-6">Session completed with {summary.total_rounds} rounds</p>

          <div className="grid grid-cols-2 gap-6 mb-8">
            {/* CUPID Results */}
            <div className="bg-violet-50 rounded-xl p-4">
              <h3 className="font-bold text-violet-700 flex items-center gap-2 mb-3">
                <Cpu size={18} /> CUPID Winner
              </h3>
              <p className="text-xl font-bold text-violet-900">{summary.cupid_best_model}</p>
              <div className="mt-3 text-sm text-violet-600">
                <p className="font-semibold mb-1">Top 5 Ranking:</p>
                <ol className="list-decimal list-inside space-y-1">
                  {summary.cupid_ranking?.slice(0, 5).map((m: any) => (
                    <li key={m.rank}>{m.model_name}</li>
                  ))}
                </ol>
              </div>
            </div>

            {/* Baseline Results */}
            <div className="bg-emerald-50 rounded-xl p-4">
              <h3 className="font-bold text-emerald-700 flex items-center gap-2 mb-3">
                <Activity size={18} /> Baseline Winner
              </h3>
              <p className="text-xl font-bold text-emerald-900">{summary.baseline_best_model}</p>
              <div className="mt-3 text-sm text-emerald-600">
                <p className="font-semibold mb-1">Top 5 Ranking:</p>
                <ol className="list-decimal list-inside space-y-1">
                  {summary.baseline_ranking?.slice(0, 5).map((m: any) => (
                    <li key={m.rank}>{m.model_name}</li>
                  ))}
                </ol>
              </div>
            </div>
          </div>

          <div className="bg-gray-100 rounded-xl p-4 mb-6">
            <div className="flex justify-between items-center">
              <span className="text-gray-600 font-medium">Total Cost Incurred</span>
              <span className="text-2xl font-bold text-green-600 flex items-center">
                <DollarSign size={24} />
                {summary.total_cost?.toFixed(6) || '0.000000'}
              </span>
            </div>
          </div>

          <button
            onClick={resetExperiment}
            className="w-full bg-black text-white py-3 rounded-lg font-bold hover:bg-gray-800 flex items-center justify-center gap-2"
          >
            <RefreshCw size={18} /> Start New Experiment
          </button>
        </div>
      </div>
    );
  }

  // Initial Screen
  if (init) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-violet-50 to-emerald-50">
        <div className="max-w-lg w-full p-8 bg-white shadow-xl rounded-2xl">
          <div className="mb-6 flex justify-center">
            <div className="relative">
              <Cpu size={48} className="text-violet-500" />
              <Activity size={24} className="text-emerald-500 absolute -bottom-1 -right-1" />
            </div>
          </div>
          <h1 className="text-3xl font-bold text-center mb-2">CUPID vs Baseline</h1>
          <p className="text-gray-500 text-center mb-2">Dual-Arena Bandit Experiment</p>
          <p className="text-sm text-gray-400 text-center mb-6">
            Compare contextual bandits with GP (CUPID) against Bradley-Terry baseline
          </p>

          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Enter your first prompt:
            </label>
            <textarea
              className="w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-violet-500 focus:border-violet-500 outline-none resize-none"
              rows={4}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="e.g., Explain the concept of recursion with a simple example..."
            />
          </div>

          <div className="bg-gray-50 rounded-lg p-4 mb-6 text-sm text-gray-600">
            <p className="font-semibold mb-2">How it works:</p>
            <ul className="space-y-1 text-gray-500">
              <li>- Row 1 (CUPID): Uses contextual GP with your feedback</li>
              <li>- Row 2 (Baseline): Standard Bradley-Terry selection</li>
              <li>- Your feedback text helps CUPID learn your preferences</li>
            </ul>
          </div>

          <button
            onClick={startSession}
            disabled={!prompt.trim()}
            className="w-full bg-gradient-to-r from-violet-500 to-emerald-500 text-white py-3 rounded-lg font-bold hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            Start Experiment
          </button>

          <p className="text-xs text-gray-400 text-center mt-4">
            Requires OPENROUTER_API_KEY environment variable on backend
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b sticky top-0 z-10 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="bg-gradient-to-r from-violet-500 to-emerald-500 text-white px-2 py-1 rounded text-xs font-bold">
              RESEARCH
            </div>
            <h1 className="font-bold text-lg">Bandit Arena</h1>
          </div>

          <div className="flex items-center space-x-6 text-sm font-mono">
            <div className="flex items-center">
              <span className="text-gray-400 mr-2">ROUND</span>
              <span className="font-bold text-lg">{arenaState?.round || 0}</span>
            </div>
            <div className="flex items-center text-green-700 bg-green-50 px-3 py-1.5 rounded-full">
              <DollarSign size={14} className="mr-1" />
              <span className="font-bold">{arenaState?.total_cost?.toFixed(5) || "0.00000"}</span>
            </div>
            <button
              onClick={fetchSummary}
              className="text-gray-500 hover:text-gray-700 flex items-center gap-1 px-3 py-1.5 border rounded-lg hover:bg-gray-50"
            >
              <Trophy size={14} /> Summary
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow max-w-7xl mx-auto px-4 py-6 w-full flex flex-col gap-6">

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
            <AlertCircle className="text-red-500 flex-shrink-0" size={20} />
            <p className="text-red-700 text-sm">{error}</p>
            <button
              onClick={() => setError(null)}
              className="ml-auto text-red-400 hover:text-red-600"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Loading Overlay */}
        {loading && (
          <div className="fixed inset-0 bg-white/80 backdrop-blur-sm z-50 flex items-center justify-center">
            <div className="flex flex-col items-center bg-white p-8 rounded-2xl shadow-xl">
              <div className="relative">
                <div className="animate-spin rounded-full h-16 w-16 border-4 border-violet-200 border-t-violet-500"></div>
                <Sparkles className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-violet-500" size={24} />
              </div>
              <p className="font-semibold mt-4">Processing 4 LLM Streams...</p>
              <p className="text-sm text-gray-500 mt-1">Updating Gaussian Process & Bradley-Terry Models</p>
            </div>
          </div>
        )}

        {/* Current Prompt Display */}
        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200">
          <span className="text-xs font-bold text-gray-400 uppercase tracking-wide">Current Prompt</span>
          <p className="text-lg text-gray-800 font-medium mt-1">{prompt}</p>

          {/* Show extracted direction if any */}
          {extractedDirection && (
            <div className="mt-3 bg-violet-50 border border-violet-200 rounded-lg p-3">
              <span className="text-xs font-bold text-violet-500 uppercase tracking-wide flex items-center gap-1">
                <Sparkles size={12} /> Direction Extracted from Feedback
              </span>
              <p className="text-sm text-violet-700 mt-1">{extractedDirection}</p>
            </div>
          )}
        </div>

        {/* --- ROW 1: CUPID SYSTEM --- */}
        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-violet-600 font-bold flex items-center gap-2">
              <Cpu size={18} /> SYSTEM A (CUPID)
            </h2>
            <span className="text-xs bg-violet-100 text-violet-700 px-2 py-0.5 rounded-full">
              Contextual GP + Direction Feedback
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4" style={{ minHeight: '24rem' }}>
            {renderModelCard('A1', 'left', arenaState?.cupid_pair?.left, cupidVote, setCupidVote, loadingStates.cupidLeft, 'cupid')}
            {renderModelCard('A2', 'right', arenaState?.cupid_pair?.right, cupidVote, setCupidVote, loadingStates.cupidRight, 'cupid')}
          </div>

          {/* Feedback Input for CUPID */}
          <div className="mt-4 bg-violet-50 p-4 rounded-xl border border-violet-100">
            <label className="flex items-center text-sm font-bold text-violet-900 mb-2">
              <MessageSquare size={16} className="mr-2" />
              Why did you choose this? (Feeds into CUPID's direction learning)
            </label>
            <input
              type="text"
              className="w-full border border-violet-200 rounded-lg p-3 text-sm focus:ring-2 focus:ring-violet-500 focus:border-violet-500 outline-none"
              placeholder="e.g., 'More concise', 'Better code formatting', 'More detailed explanation'..."
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
            />
            <p className="text-xs text-violet-400 mt-2 flex items-center gap-1">
              <Sparkles size={10} /> This feedback is analyzed by Grok to extract directional preferences for the GP model
            </p>
          </div>
        </section>

        <hr className="border-gray-200" />

        {/* --- ROW 2: BASELINE SYSTEM --- */}
        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-emerald-600 font-bold flex items-center gap-2">
              <Activity size={18} /> SYSTEM B (Baseline)
            </h2>
            <span className="text-xs bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded-full">
              Bradley-Terry Arena Selection
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4" style={{ minHeight: '24rem' }}>
            {renderModelCard('B1', 'left', arenaState?.baseline_pair?.left, baselineVote, setBaselineVote, loadingStates.baselineLeft, 'baseline')}
            {renderModelCard('B2', 'right', arenaState?.baseline_pair?.right, baselineVote, setBaselineVote, loadingStates.baselineRight, 'baseline')}
          </div>
        </section>

        {/* Sticky Footer for Action */}
        <div className="sticky bottom-4 bg-white p-4 rounded-xl shadow-2xl border border-gray-200">
          <div className="flex flex-col md:flex-row gap-4 items-center">
            <div className="flex-grow w-full md:w-auto">
              <label className="text-xs text-gray-500 font-medium mb-1 block">Next prompt (leave empty to reuse current)</label>
              <input
                type="text"
                placeholder="Enter next prompt or leave empty..."
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-violet-500 outline-none"
                value={nextPrompt}
                onChange={(e) => setNextPrompt(e.target.value)}
              />
            </div>

            <button
              onClick={handleSubmit}
              disabled={!cupidVote || !baselineVote || loading}
              className="w-full md:w-auto bg-gradient-to-r from-violet-500 to-emerald-500 text-white px-8 py-3 rounded-lg font-bold flex items-center justify-center hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              Submit & Next Round <ArrowRight size={18} className="ml-2" />
            </button>
          </div>

          {(!cupidVote || !baselineVote) && (
            <p className="text-xs text-amber-600 mt-2 text-center">
              Please select one option from each row before proceeding
            </p>
          )}
        </div>
      </main>
    </div>
  );
};

export default App;

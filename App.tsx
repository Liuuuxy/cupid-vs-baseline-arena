import React, { useState, useEffect } from 'react';
import { ArenaState, InteractRequest } from './types';
import { ArrowRight, MessageSquare, DollarSign, Activity, Cpu } from 'lucide-react';

// API Configuration
const API_URL = 'http://localhost:8000';

const App: React.FC = () => {
  // Application State
  const [loading, setLoading] = useState<boolean>(false);
  const [init, setInit] = useState<boolean>(true);
  const [prompt, setPrompt] = useState<string>('');
  
  // Arena Data
  const [arenaState, setArenaState] = useState<ArenaState | null>(null);

  // User Inputs for current round
  const [cupidVote, setCupidVote] = useState<'left' | 'right' | null>(null);
  const [baselineVote, setBaselineVote] = useState<'left' | 'right' | null>(null);
  const [feedback, setFeedback] = useState<string>('');

  // Initial Fetch
  const startSession = async () => {
    setLoading(true);
    try {
      // For the first call, we send just the prompt (or empty if purely generic start)
      // In this app flow, user provides prompt first.
      await fetchNextRound(true);
    } catch (e) {
      console.error(e);
    } finally {
      setInit(false);
      setLoading(false);
    }
  };

  const fetchNextRound = async (isFirst: boolean = false) => {
    setLoading(true);
    
    const payload: InteractRequest = {
      prompt: prompt || "Tell me a fun fact about space.", // Default prompt if empty
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
        throw new Error(`API Error: ${res.statusText}`);
      }
      const data = await res.json();
      setArenaState(data);
      
      // Reset inputs for next round
      setCupidVote(null);
      setBaselineVote(null);
      setFeedback('');
    } catch (err) {
      console.error("Failed to fetch round", err);
      alert("Error connecting to backend. Ensure FastAPI is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!cupidVote || !baselineVote) {
      alert("Please vote on both arenas before proceeding.");
      return;
    }
    fetchNextRound(false);
  };

  // --- Render Helpers ---

  const renderModelCard = (
    systemName: string, 
    side: 'left' | 'right', 
    data: any, 
    voteState: 'left' | 'right' | null, 
    setVote: (v: 'left' | 'right') => void,
    colorClass: string
  ) => {
    if (!data) return <div className="animate-pulse h-64 bg-gray-100 rounded-lg"></div>;
    
    const isSelected = voteState === side;

    return (
      <div 
        onClick={() => setVote(side)}
        className={`
          relative p-4 rounded-xl border-2 cursor-pointer transition-all duration-200 flex flex-col h-full
          ${isSelected ? `border-${colorClass}-500 bg-${colorClass}-50 shadow-lg scale-[1.01]` : 'border-gray-200 hover:border-gray-300 bg-white'}
        `}
      >
        <div className="flex justify-between items-center mb-2 text-xs text-gray-500 font-mono">
          <span>{data.model_name}</span>
          <span className="flex items-center"><DollarSign size={10} />{data.cost.toFixed(5)}</span>
        </div>
        
        <div className="flex-grow prose prose-sm overflow-y-auto max-h-60 mb-4 text-gray-700 whitespace-pre-wrap">
           {data.text}
        </div>

        <div className={`mt-auto text-center font-bold py-2 rounded ${isSelected ? `bg-${colorClass}-500 text-white` : 'bg-gray-100 text-gray-400'}`}>
          {isSelected ? 'SELECTED' : 'Select This Output'}
        </div>
      </div>
    );
  };

  if (init) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 font-sans">
        <div className="max-w-md w-full p-8 bg-white shadow-xl rounded-2xl text-center">
          <div className="mb-6 flex justify-center">
            <Activity size={48} className="text-cupid-500" />
          </div>
          <h1 className="text-2xl font-bold mb-2">CUPID vs Baseline</h1>
          <p className="text-gray-600 mb-6">Dual-Arena Bandit Experiment</p>
          
          <div className="mb-4 text-left">
            <label className="block text-sm font-medium text-gray-700 mb-1">Enter your first prompt:</label>
            <textarea 
              className="w-full border rounded-lg p-3 focus:ring-2 focus:ring-cupid-500 outline-none"
              rows={3}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="e.g., Write a python script to merge sort..."
            />
          </div>

          <button 
            onClick={startSession}
            disabled={!prompt}
            className="w-full bg-black text-white py-3 rounded-lg font-bold hover:bg-gray-800 disabled:opacity-50 transition"
          >
            Start Experiment
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="bg-gradient-to-r from-cupid-500 to-baseline-500 text-white px-2 py-1 rounded text-xs font-bold">RESEARCH</div>
            <h1 className="font-bold text-lg">Bandit Arena</h1>
          </div>
          
          <div className="flex items-center space-x-6 text-sm font-mono">
             <div className="flex items-center">
               <span className="text-gray-400 mr-2">ROUND</span>
               <span className="font-bold">{arenaState?.round || 0}</span>
             </div>
             <div className="flex items-center text-green-700 bg-green-50 px-3 py-1 rounded-full">
               <DollarSign size={14} className="mr-1" />
               <span className="font-bold">{arenaState?.total_cost.toFixed(4) || "0.0000"}</span>
             </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow max-w-7xl mx-auto px-4 py-8 w-full flex flex-col gap-8">
        
        {/* Loading Overlay */}
        {loading && (
          <div className="fixed inset-0 bg-white/80 backdrop-blur-sm z-50 flex items-center justify-center">
            <div className="flex flex-col items-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-black mb-4"></div>
              <p className="font-mono text-sm">Processing 4 LLM Streams & Updating GPs...</p>
            </div>
          </div>
        )}

        {/* Current Prompt Display */}
        <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
           <span className="text-xs font-bold text-gray-400 uppercase tracking-wide">Current Prompt</span>
           <p className="text-lg text-gray-800 font-medium mt-1">{prompt}</p>
        </div>

        {/* --- ROW 1: CUPID SYSTEM --- */}
        <section>
          <div className="flex items-center justify-between mb-3">
             <h2 className="text-cupid-600 font-bold flex items-center gap-2">
               <Cpu size={18} /> SYSTEM A (CUPID)
             </h2>
             <span className="text-xs bg-cupid-100 text-cupid-700 px-2 py-0.5 rounded">Contextual Bandit</span>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 h-96">
            {renderModelCard('A1', 'left', arenaState?.cupid_pair.left, cupidVote, setCupidVote, 'cupid')}
            {renderModelCard('A2', 'right', arenaState?.cupid_pair.right, cupidVote, setCupidVote, 'cupid')}
          </div>

          {/* Feedback Input for CUPID */}
          <div className="mt-4 bg-indigo-50 p-4 rounded-lg border border-indigo-100">
            <label className="flex items-center text-sm font-bold text-indigo-900 mb-2">
              <MessageSquare size={16} className="mr-2" />
              Reasoning (Optional):
            </label>
            <input 
              type="text" 
              className="w-full border border-indigo-200 rounded p-2 text-sm focus:ring-2 focus:ring-cupid-500 outline-none"
              placeholder="Why did you choose this? (e.g., 'More concise', 'Better code style')..."
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
            />
            <p className="text-xs text-indigo-400 mt-1">
              * This feedback is fed into the Grok LLM to update the Bandit's latent direction vector.
            </p>
          </div>
        </section>

        <hr className="border-gray-200" />

        {/* --- ROW 2: BASELINE SYSTEM --- */}
        <section>
          <div className="flex items-center justify-between mb-3">
             <h2 className="text-baseline-600 font-bold flex items-center gap-2">
               <Activity size={18} /> SYSTEM B (Baseline)
             </h2>
             <span className="text-xs bg-baseline-100 text-baseline-700 px-2 py-0.5 rounded">Bradley-Terry</span>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 h-96">
            {renderModelCard('B1', 'left', arenaState?.baseline_pair.left, baselineVote, setBaselineVote, 'baseline')}
            {renderModelCard('B2', 'right', arenaState?.baseline_pair.right, baselineVote, setBaselineVote, 'baseline')}
          </div>
        </section>

        {/* Sticky Footer for Action */}
        <div className="sticky bottom-4 bg-white p-4 rounded-xl shadow-2xl border border-gray-200 flex flex-col md:flex-row gap-4 items-center justify-between">
           <div className="text-sm text-gray-500">
             Next prompt will be requested after submission.
           </div>
           
           <div className="flex gap-4 w-full md:w-auto">
             <input 
                type="text" 
                placeholder="Next prompt (optional, or reuse current)" 
                className="flex-grow border rounded px-3 py-2 text-sm"
                onChange={(e) => setPrompt(e.target.value)} // Ideally we'd have a separate state for next prompt, but reuse for simplicity
             />
             <button 
               onClick={handleSubmit}
               className="bg-black text-white px-6 py-2 rounded-lg font-bold flex items-center hover:bg-gray-800"
             >
               Submit & Next Round <ArrowRight size={16} className="ml-2" />
             </button>
           </div>
        </div>
      </main>
    </div>
  );
};

export default App;
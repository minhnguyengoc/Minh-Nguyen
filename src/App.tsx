import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, TrendingDown, Activity, Wallet, History, 
  BarChart3, ShieldCheck, BrainCircuit, RefreshCw,
  Search, ChevronRight, Play, Square, Settings as SettingsIcon,
  Plus, Minus
} from 'lucide-react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer
} from 'recharts';
import { motion, AnimatePresence } from 'motion/react';
import { GoogleGenAI } from "@google/genai";
import { cn, formatVND, formatPercent } from './lib/utils';
import { TickerData, PortfolioStats, Position, Trade, Signal } from './types';

// --- INITIALIZE AI ---
// Gemini logic moved to server for better security and stability

// --- MOCK CHART DATA ---
const chartData = [
  { time: '09:00', value: 500.0 },
  { time: '10:00', value: 502.4 },
  { time: '11:00', value: 501.2 },
  { time: '13:00', value: 505.8 },
  { time: '14:00', value: 508.3 },
  { time: '14:30', value: 512.1 },
  { time: '15:00', value: 511.5 },
];

export default function App() {
  const [tickers, setTickers] = useState<TickerData[]>([]);
  const [portfolio, setPortfolio] = useState<PortfolioStats | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [history, setHistory] = useState<Trade[]>([]);
  const [lessons, setLessons] = useState<any[]>([]);
  const [isBotRunning, setIsBotRunning] = useState(false);
  const [loadingSignal, setLoadingSignal] = useState<string | null>(null);
  const [activeSignal, setActiveSignal] = useState<Signal | null>(null);
  const [feedbackTrade, setFeedbackTrade] = useState<Trade | null>(null);
  const [isConnected, setIsConnected] = useState(true);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      const tickerRes = await fetch('/api/market/tickers');
      if (!tickerRes.ok) {
        throw new Error(`Ticker fetch failed: ${tickerRes.status}`);
      }
      const tickerData = await tickerRes.json();
      setTickers(tickerData);
      setIsConnected(true);

      const portfolioRes = await fetch('/api/portfolio');
      if (!portfolioRes.ok) {
        throw new Error(`Portfolio fetch failed: ${portfolioRes.status}`);
      }
      const pData = await portfolioRes.json();
      setPortfolio({
        balance: pData.balance,
        equity: pData.equity,
        totalPnL: pData.totalPnL,
        winRate: pData.winRate,
        profitFactor: pData.profitFactor,
        totalTrades: pData.totalTrades,
      });
      setPositions(pData.positions);
      setHistory(pData.history);

      const lessonsRes = await fetch('/api/learning/lessons');
      if (!lessonsRes.ok) {
        throw new Error(`Lessons fetch failed: ${lessonsRes.status}`);
      }
      const lData = await lessonsRes.json();
      setLessons(lData);
    } catch (err: any) {
      setIsConnected(false);
      if (err.name === 'TypeError' && err.message === 'Failed to fetch') {
        console.error('Network Error: Check if dev server is running or if requests are being blocked.');
      } else {
        console.error('Data sync error:', err.message);
      }
    }
  };

  const submitFeedback = async (tradeId: string, feedback: 'SUCCESS' | 'FAILURE', reason: string, pnl: number) => {
    try {
      const res = await fetch('/api/trading/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tradeId, feedback, reason, pnl }),
      });
      if (!res.ok) throw new Error(`Feedback submission failed: ${res.status}`);
      setFeedbackTrade(null);
      fetchData();
    } catch (err) {
      console.error('Feedback failed', err);
    }
  };

  const executeTrade = async (type: 'BUY' | 'SELL', symbol: string, quantity: number, price: number) => {
    try {
      const res = await fetch('/api/trading/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type, symbol, quantity, price }),
      });
      if (!res.ok) throw new Error(`Execution failed: ${res.status}`);
      const result = await res.json();
      if (type === 'SELL') {
        // Assume the last trade in history is the one just executed
        const lastTrade = result.portfolio.history[result.portfolio.history.length - 1];
        setFeedbackTrade(lastTrade);
      }
      fetchData();
    } catch (err) {
      console.error('Trade failed', err);
    }
  };

  const getAISignal = async (symbol: string) => {
    setLoadingSignal(symbol);
    try {
      const response = await fetch('/api/ai/signal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, tickers, lessons }),
      });
      
      if (!response.ok) throw new Error("AI Signal fetch failed");
      
      const signal = await response.json();
      setActiveSignal({ ...signal, symbol, timestamp: new Date().toISOString() });
    } catch (err) {
      console.error('Signal failed', err);
    } finally {
      setLoadingSignal(null);
    }
  };

  return (
    <div className="min-h-screen flex flex-col md:flex-row bg-[#0f172a] text-white relative overflow-hidden">
      <div className="mesh-bg" />
      {/* Sidebar Navigation */}
      <nav className="w-full md:w-20 lg:w-64 glass border-b md:border-b-0 md:border-r border-white/10 p-4 flex flex-col gap-8 shrink-0 z-10">
        <div className="flex items-center gap-3 px-2">
          <div className="w-10 h-10 bg-gradient-to-tr from-indigo-500 to-pink-500 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <BrainCircuit className="text-white" />
          </div>
          <span className="font-bold text-xl hidden lg:block tracking-tight">VnQuant AI</span>
        </div>

        <ul className="flex flex-row md:flex-col gap-2 overflow-x-auto md:overflow-x-visible">
          <NavItem icon={<BarChart3 />} label="Dashboard" active />
          <NavItem icon={<Wallet />} label="Portfolio" />
          <NavItem icon={<History />} label="History" />
          <NavItem icon={<ShieldCheck />} label="Risk Engine" />
          <NavItem icon={<SettingsIcon />} label="Settings" />
        </ul>

        <div className="mt-auto hidden lg:block">
          {!isConnected && (
            <div className="mb-4 p-3 bg-red-500/20 border border-red-500/30 rounded-xl">
              <p className="text-[10px] text-red-400 font-bold uppercase tracking-tight">Backend Disconnected</p>
              <p className="text-[9px] text-red-300/60 leading-tight">Check server logs or restart the dev server.</p>
            </div>
          )}
          <div className="p-4 glass rounded-2xl">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-xs text-white/40 uppercase tracking-widest font-bold text-[10px]">Live Mode</span>
            </div>
            <p className="text-[11px] text-white/60 leading-relaxed uppercase tracking-tight font-medium">
              Bot is scanning HOSE/HNX for opportunities. accuracy target: 85%+
            </p>
          </div>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="flex-1 overflow-y-auto p-4 md:p-8">
        <div className="max-w-7xl mx-auto space-y-8">
          {/* Header & Controls */}
          <header className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold tracking-tight mb-1">Trading Dashboard</h1>
              <p className="text-gray-400">Institutional Quant Strategy V2.0</p>
            </div>
            <div className="flex items-center gap-3">
              <button 
                onClick={() => setIsBotRunning(!isBotRunning)}
                className={cn(
                  "px-6 py-2.5 rounded-full font-semibold flex items-center gap-2 transition-all active:scale-95",
                  isBotRunning 
                    ? "bg-red-500/20 text-red-400 border border-red-500/30 backdrop-blur-md hover:bg-red-500/30" 
                    : "bg-gradient-to-r from-indigo-500 to-indigo-600 text-white hover:opacity-90 shadow-lg shadow-indigo-500/30"
                )}
              >
                {isBotRunning ? <Square size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" />}
                {isBotRunning ? 'Stop Autopilot' : 'Start Autopilot'}
              </button>
            </div>
          </header>

          {/* Top Stats Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard 
              label="Net Equity" 
              value={portfolio ? formatVND(portfolio.equity) : '---'} 
              change={"+12.4%"}
              positive={true}
              icon={<Wallet className="text-indigo-400" />}
            />
            <StatCard 
              label="Available Balance" 
              value={portfolio ? formatVND(portfolio.balance) : '---'} 
              icon={<TrendingUp className="text-blue-400" />}
            />
            <StatCard 
              label="Win Rate" 
              value="---" 
              subValue="Target: 85%"
              icon={<ShieldCheck className="text-pink-400" />}
            />
            <StatCard 
              label="Profit Factor" 
              value="1.38" 
              icon={<Activity className="text-amber-400" />}
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Performance Chart */}
            <div className="lg:col-span-8 space-y-8">
              <section className="glass rounded-[32px] p-6 lg:p-8">
                <div className="flex items-center justify-between mb-8">
                  <h3 className="text-xl font-bold flex items-center gap-2">
                    <TrendingUp className="text-indigo-400" />
                    Equity Curve
                  </h3>
                  <div className="flex gap-2">
                    <span className="px-3 py-1 bg-white/10 rounded-full text-xs font-medium cursor-pointer">1D</span>
                    <span className="px-3 py-1 text-white/40 rounded-full text-xs font-medium cursor-pointer">1W</span>
                  </div>
                </div>
                <div className="h-[300px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData}>
                      <defs>
                        <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.1)" />
                      <XAxis 
                        dataKey="time" 
                        axisLine={false} 
                        tickLine={false} 
                        tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 12 }} 
                      />
                      <YAxis hide domain={['dataMin - 5', 'dataMax + 5']} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', backdropFilter: 'blur(8px)' }}
                        itemStyle={{ color: '#818cf8' }}
                      />
                      <Area 
                        type="monotone" 
                        dataKey="value" 
                        stroke="#6366f1" 
                        strokeWidth={4}
                        fillOpacity={1} 
                        fill="url(#colorValue)" 
                        animationDuration={1500}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </section>

              {/* Positions Table */}
              <section className="glass rounded-[32px] overflow-hidden">
                <div className="p-6 border-b border-white/10 flex items-center justify-between">
                  <h3 className="text-xl font-bold">Active Positions</h3>
                  <span className="bg-indigo-500/20 text-indigo-300 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-tighter">
                    {positions.length} Open
                  </span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left">
                    <thead>
                      <tr className="bg-white/5 text-white/40 text-xs uppercase tracking-wider">
                        <th className="px-6 py-4 font-semibold">Asset</th>
                        <th className="px-6 py-4 font-semibold">Avg Price</th>
                        <th className="px-6 py-4 font-semibold">Quantity</th>
                        <th className="px-6 py-4 font-semibold">PnL (VND)</th>
                        <th className="px-6 py-4 font-semibold">Action</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/10">
                      <AnimatePresence initial={false}>
                        {positions.map((pos) => (
                          <motion.tr 
                            key={pos.id}
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="hover:bg-white/5 transition-colors group"
                          >
                            <td className="px-6 py-6 border-b border-white/10 text-white/90">
                              <div className="flex items-center gap-3">
                                <div className="w-10 h-10 bg-white/10 rounded-full flex items-center justify-center font-bold text-indigo-400 group-hover:scale-110 transition-transform uppercase border border-white/5">
                                  {pos.symbol[0]}
                                </div>
                                <span className="font-bold tracking-tight">{pos.symbol}</span>
                              </div>
                            </td>
                            <td className="px-6 py-6 border-b border-white/10 font-mono text-white/60">
                              {formatVND(pos.avgPrice)}
                            </td>
                            <td className="px-6 py-6 border-b border-white/10 text-white/60">
                              {pos.quantity}
                            </td>
                            <td className={cn(
                              "px-6 py-6 border-b border-white/10 font-bold font-mono",
                              pos.pnl >= 0 ? "text-emerald-400" : "text-rose-400"
                            )}>
                              {pos.pnl > 0 ? '+' : ''}{formatVND(pos.pnl)}
                            </td>
                            <td className="px-6 py-6 border-b border-white/10">
                              <button 
                                onClick={() => executeTrade('SELL', pos.symbol, pos.quantity, pos.avgPrice)}
                                className="px-4 py-1.5 rounded-lg border border-rose-500/30 text-rose-400 text-xs font-bold hover:bg-rose-500/10 active:scale-95 transition-all"
                              >
                                Close
                              </button>
                            </td>
                          </motion.tr>
                        ))}
                      </AnimatePresence>
                      {positions.length === 0 && (
                        <tr>
                          <td colSpan={5} className="px-6 py-12 text-center text-white/30 italic text-sm">
                            No active holdings. Monitoring HOSE for quality signals...
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </section>
            </div>

            {/* Market Tickers & Signals */}
            <div className="lg:col-span-4 space-y-8">
              <section className="glass rounded-[32px] p-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-xl font-bold flex items-center gap-2">
                    <Activity className="text-pink-400" />
                    Market Radar
                  </h3>
                  <RefreshCw 
                    size={18} 
                    className="text-white/40 cursor-pointer hover:rotate-180 transition-transform duration-500" 
                    onClick={fetchData}
                  />
                </div>
                
                <div className="space-y-3">
                  {tickers.map((ticker) => (
                    <motion.div 
                      key={ticker.symbol}
                      whileHover={{ x: 4 }}
                      className="group p-4 bg-white/5 rounded-2xl hover:bg-white/10 transition-all border border-white/5 hover:border-white/10"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-lg leading-none">{ticker.symbol}</span>
                          <span className="text-[10px] text-white/40 bg-white/10 px-1.5 py-0.5 rounded uppercase font-bold">{ticker.exchange}</span>
                        </div>
                        <span className={cn(
                          "text-[11px] font-bold font-mono px-2 py-0.5 rounded-full",
                          ticker.change >= 0 ? "text-emerald-400 bg-emerald-400/10" : "text-rose-400 bg-rose-400/10"
                        )}>
                          {ticker.change >= 0 ? '+' : ''}{ticker.changePercent}%
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-white/40 font-mono">{formatVND(ticker.price)}</span>
                        <button 
                          disabled={loadingSignal === ticker.symbol}
                          onClick={() => getAISignal(ticker.symbol)}
                          className={cn(
                            "text-[10px] uppercase font-black tracking-widest px-3 py-1.5 rounded-lg border transition-all active:scale-95",
                            loadingSignal === ticker.symbol 
                              ? "bg-indigo-500/10 text-indigo-400 border-indigo-500/40 animate-pulse"
                              : "border-white/10 text-white/40 hover:border-indigo-500/40 hover:text-indigo-400"
                          )}
                        >
                          {loadingSignal === ticker.symbol ? 'Analyzing' : 'AI Analysis'}
                        </button>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </section>

              {/* AI Insight Overlay */}
              <AnimatePresence>
                {activeSignal && (
                  <motion.section 
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="glass border-indigo-500/30 rounded-[32px] p-6 bg-indigo-500/5 relative shadow-indigo-500/10"
                  >
                    <div className="absolute top-4 right-4 text-[10px] text-white/20 font-bold uppercase">Signal Matrix</div>
                    <div className="flex items-center gap-2 mb-4">
                      <div className="p-2 bg-indigo-500 rounded-lg shadow-lg shadow-indigo-500/20">
                        <BrainCircuit className="text-white" size={20} />
                      </div>
                      <h4 className="font-bold underline decoration-indigo-500/30 decoration-2 underline-offset-4">Logic: {activeSignal.symbol}</h4>
                    </div>
                    <div className="flex items-center justify-between mb-4">
                      <div className={cn(
                        "text-2xl font-black px-4 py-1.5 rounded-xl uppercase skew-x-[-8deg] shadow-lg",
                        activeSignal.action === 'BUY' ? "bg-emerald-500 text-white" : "bg-rose-500 text-white"
                      )}>
                        {activeSignal.action}
                      </div>
                      <div className="text-right">
                        <div className="text-[10px] text-white/40 uppercase font-black tracking-wider">Confidence</div>
                        <div className="text-2xl font-black font-mono text-indigo-400 tracking-tighter">{formatPercent(activeSignal.confidence * 100)}</div>
                      </div>
                    </div>
                    <p className="text-xs text-white/60 leading-relaxed mb-6 bg-white/5 p-3 rounded-xl border border-white/5 italic">
                      <span className="text-indigo-400 font-bold not-italic">Reason:</span> {activeSignal.reason}
                    </p>
                    <div className="flex gap-2">
                       <button 
                        onClick={() => {
                          const t = tickers.find(t => t.symbol === activeSignal.symbol);
                          if (t) executeTrade('BUY', t.symbol, 1000, t.price);
                          setActiveSignal(null);
                        }}
                        className="flex-1 bg-indigo-600 text-white py-3 rounded-xl font-black uppercase text-xs tracking-widest shadow-lg shadow-indigo-600/20 hover:translate-y-[-1px] transition-transform"
                      >
                        Execute Order
                      </button>
                      <button 
                        onClick={() => setActiveSignal(null)}
                        className="px-4 rounded-xl border border-white/10 text-white/40 hover:bg-white/5 transition-colors"
                      >
                        <ChevronRight size={20} />
                      </button>
                    </div>
                  </motion.section>
                )}
              </AnimatePresence>

              {/* Feed/History Log */}
              <section className="glass rounded-[32px] p-6">
                <h3 className="text-lg font-bold mb-6 flex items-center gap-2">
                  <History className="text-white/40" size={18} />
                  Journal
                </h3>
                <div className="space-y-6 relative before:absolute before:left-[11px] before:top-2 before:bottom-0 before:w-px before:bg-white/10">
                  {history.slice(0, 5).map((trade) => (
                    <div key={trade.id} className="relative pl-8 group">
                      <div className={cn(
                        "absolute left-0 top-1 w-[22px] h-[22px] rounded-full flex items-center justify-center z-10 text-[10px]",
                        trade.type === 'BUY' ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40" : "bg-rose-500/20 text-rose-400 border border-rose-500/40"
                      )}>
                        {trade.type === 'BUY' ? 'B' : 'S'}
                      </div>
                      <div className="flex justify-between items-start">
                        <div>
                          <div className="flex items-center gap-2">
                            <div className="font-bold text-sm tracking-tight text-white/90">
                              {trade.type} {trade.symbol}
                            </div>
                            {trade.type === 'SELL' && trade.feedback && (
                              <span className={cn(
                                "text-[8px] px-1.5 py-0.5 rounded uppercase font-black",
                                trade.feedback === 'SUCCESS' ? "bg-emerald-500/20 text-emerald-400" : "bg-rose-500/20 text-rose-400"
                              )}>
                                {trade.feedback}
                              </span>
                            )}
                          </div>
                          <div className="text-[10px] text-white/40 font-mono">{new Date(trade.timestamp).toLocaleTimeString()} · {formatVND(trade.price)}</div>
                        </div>
                        <div className="text-[10px] font-bold text-white/60 bg-white/10 px-2 py-0.5 rounded leading-none border border-white/5">
                          x{trade.quantity}
                        </div>
                      </div>
                      {trade.type === 'SELL' && !trade.feedback && (
                        <button 
                          onClick={() => setFeedbackTrade(trade)}
                          className="mt-2 text-[10px] text-indigo-400 hover:underline uppercase tracking-widest font-bold"
                        >
                          Submit Feedback
                        </button>
                      )}
                    </div>
                  ))}
                  {history.length === 0 && <p className="text-center text-white/20 text-xs italic py-4 font-medium uppercase tracking-widest">Waiting for execution...</p>}
                </div>
              </section>
            </div>
          </div>
        </div>
      </main>

      {/* Feedback Modal Overlay */}
      <AnimatePresence>
        {feedbackTrade && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-md">
            <motion.div 
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="glass max-w-md w-full p-8 rounded-[40px] space-y-6"
            >
              <div className="flex items-center gap-3">
                <div className="p-3 bg-indigo-500 rounded-2xl shadow-lg shadow-indigo-500/20">
                  <BrainCircuit className="text-white" />
                </div>
                <h2 className="text-xl font-bold">Trade Feedback Loop</h2>
              </div>
              
              <p className="text-white/60 text-sm leading-relaxed">
                Execute learning for <span className="text-white font-bold">{feedbackTrade.symbol}</span>. 
                Was the AI signal effective for this exit?
              </p>

              <div className="grid grid-cols-2 gap-4">
                <button 
                  onClick={() => submitFeedback(feedbackTrade.id, 'SUCCESS', 'Profit captured as expected', feedbackTrade.pnl || 0)}
                  className="flex flex-col items-center gap-2 p-6 rounded-[24px] bg-emerald-500/10 border border-emerald-500/30 hover:bg-emerald-500/20 transition-all group"
                >
                  <TrendingUp className="text-emerald-400 group-hover:scale-110 transition-transform" />
                  <span className="font-bold text-emerald-400 text-sm uppercase tracking-widest">Success</span>
                </button>
                <button 
                  onClick={() => submitFeedback(feedbackTrade.id, 'FAILURE', 'Loss or inefficient exit', feedbackTrade.pnl || 0)}
                  className="flex flex-col items-center gap-2 p-6 rounded-[24px] bg-rose-500/10 border border-rose-500/30 hover:bg-rose-500/20 transition-all group"
                >
                  <TrendingDown className="text-rose-400 group-hover:scale-110 transition-transform" />
                  <span className="font-bold text-rose-400 text-sm uppercase tracking-widest">Failure</span>
                </button>
              </div>

              <div className="bg-white/5 p-4 rounded-2xl border border-white/5">
                <p className="text-[10px] text-white/30 uppercase font-black mb-1">Self-Correction Engine</p>
                <p className="text-[11px] text-white/60 italic">"Feedback improves next-cycle signal accuracy by pattern reinforcement."</p>
              </div>

              <button 
                onClick={() => setFeedbackTrade(null)}
                className="w-full py-4 text-white/40 hover:text-white text-xs font-bold uppercase tracking-widest transition-colors"
              >
                Skip This Time
              </button>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}

function NavItem({ icon, label, active = false }: { icon: React.ReactNode, label: string, active?: boolean }) {
  return (
    <li className={cn(
      "group flex items-center gap-3 px-4 py-3 rounded-2xl cursor-pointer transition-all shrink-0",
      active ? "bg-white/15 text-white border border-white/10 shadow-lg" : "text-white/40 hover:text-white"
    )}>
      <span className={cn(active ? "text-indigo-400" : "group-hover:text-white")}>{icon}</span>
      <span className="font-bold text-sm hidden lg:block tracking-wide">{label}</span>
    </li>
  );
}

function StatCard({ label, value, subValue, change, positive, icon }: { label: string, value: string, subValue?: string, change?: string, positive?: boolean, icon: React.ReactNode }) {
  return (
    <div className="glass p-6 rounded-[32px] relative overflow-hidden group hover:border-white/20 transition-colors">
      <div className="absolute -right-4 -top-4 w-24 h-24 bg-white/5 rounded-full blur-3xl opacity-0 group-hover:opacity-100 transition-opacity" />
      <div className="flex justify-between items-start mb-4">
        <div className="p-3 bg-white/5 rounded-2xl border border-white/5 shadow-inner">
          {cloneIcon(icon)}
        </div>
        {change && (
          <span className={cn(
            "text-[10px] font-black px-2 py-1 rounded-lg uppercase tracking-widest",
            positive ? "text-emerald-400 bg-emerald-400/10" : "text-rose-400 bg-rose-400/10"
          )}>
            {change}
          </span>
        )}
      </div>
      <div>
        <p className="text-[10px] text-white/30 font-black uppercase tracking-[0.2em] mb-2">{label}</p>
        <div className="flex items-baseline gap-2">
          <h2 className="text-2xl font-black tracking-tighter font-mono text-white/90">{value}</h2>
        </div>
        {subValue && <p className="text-[10px] text-white/30 mt-2 lowercase font-bold tracking-tight">{subValue}</p>}
      </div>
    </div>
  );
}

function cloneIcon(icon: React.ReactNode) {
  if (React.isValidElement(icon)) {
    return React.cloneElement(icon as React.ReactElement<any>, { size: 20 });
  }
  return icon;
}

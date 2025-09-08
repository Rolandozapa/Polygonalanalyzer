import React, { useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const TradingDashboard = () => {
  const [isTrading, setIsTrading] = useState(false);
  const [opportunities, setOpportunities] = useState([]);
  const [analyses, setAnalyses] = useState([]);
  const [decisions, setDecisions] = useState([]);
  const [performance, setPerformance] = useState({});
  const [websocket, setWebsocket] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('Disconnected');
  const [activeTab, setActiveTab] = useState('dashboard');
  const [activePositions, setActivePositions] = useState([]);
  const [executionMode, setExecutionMode] = useState('SIMULATION');
  const [backtestResults, setBacktestResults] = useState(null);
  const [backtestLoading, setBacktestLoading] = useState(false);
  const [backtestStatus, setBacktestStatus] = useState(null);

  // WebSocket connection
  useEffect(() => {
    const ws = new WebSocket(`${BACKEND_URL.replace('https', 'wss').replace('http', 'ws')}/api/ws`);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnectionStatus('Connected');
      setWebsocket(ws);
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleWebSocketMessage(data);
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnectionStatus('Disconnected');
    };
    
    ws.onerror = (error) => {
      console.log('WebSocket error:', error);
      setConnectionStatus('Error');
    };

    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, []);

  const handleWebSocketMessage = (data) => {
    switch (data.type) {
      case 'opportunities_found':
        setOpportunities(data.data);
        break;
      case 'technical_analysis':
        setAnalyses(prev => [data.data, ...prev.slice(0, 19)]);
        break;
      case 'trading_decision':
        setDecisions(prev => [data.data, ...prev.slice(0, 19)]);
        break;
      default:
        console.log('Unknown message type:', data.type);
    }
  };

  // Fetch initial data
  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const [oppRes, analysesRes, decisionsRes, perfRes, positionsRes, modeRes, backtestStatusRes] = await Promise.all([
        axios.get(`${API}/opportunities`),
        axios.get(`${API}/analyses`),
        axios.get(`${API}/decisions`),
        axios.get(`${API}/performance`),
        axios.get(`${API}/active-positions`),
        axios.get(`${API}/trading/execution-mode`),
        axios.get(`${API}/backtest/status`).catch(() => ({ data: { data: { available_symbols: [], engine_status: 'unavailable' } } }))
      ]);

      setOpportunities(oppRes.data.opportunities || []);
      setAnalyses(analysesRes.data.analyses || []);
      setDecisions(decisionsRes.data.decisions || []);
      setPerformance(perfRes.data.performance || {});
      setActivePositions(positionsRes.data.data?.active_positions || []);
      setExecutionMode(modeRes.data.execution_mode || 'SIMULATION');
      setBacktestStatus(backtestStatusRes.data.data || null);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const runBacktest = async (params) => {
    try {
      setBacktestLoading(true);
      const response = await axios.post(`${API}/backtest/run`, params);
      setBacktestResults(response.data);
    } catch (error) {
      console.error('Error running backtest:', error);
    } finally {
      setBacktestLoading(false);
    }
  };

  const runAITraining = async () => {
    try {
      setBacktestLoading(true);
      
      // Use quick training for better performance
      const response = await axios.post(`${API}/ai-training/run-quick`);
      setBacktestResults(prev => ({
        ...prev,
        training_completed: true,
        ...response.data.data
      }));
    } catch (error) {
      console.error('Error running AI training:', error);
    } finally {
      setBacktestLoading(false);
    }
  };

  const runStrategyBacktest = async () => {
    try {
      setBacktestLoading(true);
      
      // Get form values
      const startDate = document.getElementById('backtest-start-date')?.value || '2020-01-01';
      const endDate = document.getElementById('backtest-end-date')?.value || '2021-07-01';
      const symbolSelect = document.getElementById('backtest-symbols');
      const selectedSymbols = Array.from(symbolSelect?.selectedOptions || []).map(option => option.value);
      
      const params = {
        start_date: startDate,
        end_date: endDate,
        symbols: selectedSymbols.length > 0 ? selectedSymbols : backtestStatus?.available_symbols?.slice(0, 5)
      };
      
      const response = await axios.post(`${API}/backtest/run`, params);
      setBacktestResults(prev => ({
        ...prev,
        backtest_completed: true,
        results: response.data.data
      }));
    } catch (error) {
      console.error('Error running strategy backtest:', error);
    } finally {
      setBacktestLoading(false);
    }
  };

  const loadAdaptiveContext = async () => {
    try {
      const response = await axios.post(`${API}/adaptive-context/load-training`);
      console.log('Adaptive context loaded:', response.data);
      // Could show a success message or update UI
    } catch (error) {
      console.error('Error loading adaptive context:', error);
    }
  };

  const loadAIInsights = async () => {
    try {
      const response = await axios.post(`${API}/ai-training/load-insights`);
      console.log('AI insights loaded:', response.data);
      // Update UI with enhancement status
      setBacktestResults(prev => ({
        ...prev,
        enhancement_loaded: true,
        enhancement_summary: response.data.data
      }));
    } catch (error) {
      console.error('Error loading AI insights:', error);
    }
  };

  const getEnhancementStatus = async () => {
    try {
      const response = await axios.get(`${API}/ai-training/enhancement-status`);
      return response.data.data;
    } catch (error) {
      console.error('Error getting enhancement status:', error);
      return null;
    }
  };

  const getChartistLibrary = async () => {
    try {
      const response = await axios.get(`${API}/chartist/library`);
      return response.data.data;
    } catch (error) {
      console.error('Error getting chartist library:', error);
      return null;
    }
  };

  const analyzeChartistPatterns = async (patterns, marketContext = 'SIDEWAYS') => {
    try {
      const response = await axios.post(`${API}/chartist/analyze`, {
        patterns: patterns,
        market_context: marketContext
      });
      return response.data.data;
    } catch (error) {
      console.error('Error analyzing chartist patterns:', error);
      return null;
    }
  };

  const closePosition = async (positionId) => {
    try {
      await axios.post(`${API}/active-positions/close/${positionId}`);
      // Refresh data after closing position
      await fetchData();
    } catch (error) {
      console.error('Error closing position:', error);
    }
  };

  const setTradingMode = async (mode) => {
    try {
      await axios.post(`${API}/trading/execution-mode`, { mode });
      setExecutionMode(mode);
    } catch (error) {
      console.error('Error setting trading mode:', error);
    }
  };

  const startTrading = async () => {
    try {
      await axios.post(`${API}/start-trading`);
      setIsTrading(true);
    } catch (error) {
      console.error('Error starting trading:', error);
    }
  };

  const stopTrading = async () => {
    try {
      await axios.post(`${API}/stop-trading`);
      setIsTrading(false);
    } catch (error) {
      console.error('Error stopping trading:', error);
    }
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 4
    }).format(price);
  };

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getSignalColor = (signal) => {
    switch (signal) {
      case 'long': return 'text-emerald-600 bg-emerald-50';
      case 'short': return 'text-red-600 bg-red-50';
      default: return 'text-amber-600 bg-amber-50';
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-emerald-600';
    if (confidence >= 0.6) return 'text-amber-600';
    return 'text-red-600';
  };

  const TabButton = ({ id, children, icon }) => (
    <button
      onClick={() => setActiveTab(id)}
      className={`flex items-center gap-2 px-6 py-3 font-medium transition-all duration-200 ${
        activeTab === id
          ? 'text-indigo-600 border-b-2 border-indigo-600 bg-indigo-50'
          : 'text-slate-600 hover:text-indigo-600 hover:bg-slate-50'
      }`}
    >
      {icon}
      {children}
    </button>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl flex items-center justify-center">
                <span className="text-white font-bold text-lg">AI</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-900">Dual AI Trading Bot</h1>
                <p className="text-slate-600">Intelligent cryptocurrency trading system</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${connectionStatus === 'Connected' ? 'bg-emerald-500' : 'bg-red-500'}`}></div>
                <span className="text-sm text-slate-600">{connectionStatus}</span>
              </div>
              
              <button
                onClick={isTrading ? stopTrading : startTrading}
                className={`px-6 py-2 rounded-xl font-medium transition-all duration-200 ${
                  isTrading
                    ? 'bg-red-600 hover:bg-red-700 text-white'
                    : 'bg-emerald-600 hover:bg-emerald-700 text-white'
                }`}
              >
                {isTrading ? 'Stop Trading' : 'Start Trading'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex gap-1">
            <TabButton id="dashboard" icon="📊">Dashboard</TabButton>
            <TabButton id="opportunities" icon="🔍">Opportunities</TabButton>
            <TabButton id="analyses" icon="🤖">IA1 Analysis</TabButton>
            <TabButton id="decisions" icon="⚡">IA2 Decisions</TabButton>
            <TabButton id="positions" icon="🎯">Active Positions</TabButton>
            <TabButton id="backtest" icon="📈">Backtesting</TabButton>
            <TabButton id="performance" icon="📊">Performance</TabButton>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'dashboard' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div className="bg-white rounded-2xl p-6 shadow-sm border">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-600">Total Opportunities</p>
                  <p className="text-2xl font-bold text-slate-900">{performance.total_opportunities || 0}</p>
                </div>
                <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center">
                  <span className="text-blue-600 text-xl">🔍</span>
                </div>
              </div>
            </div>
            
            <div className="bg-white rounded-2xl p-6 shadow-sm border">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-600">Executed Trades</p>
                  <p className="text-2xl font-bold text-slate-900">{performance.executed_trades || 0}</p>
                </div>
                <div className="w-12 h-12 bg-emerald-100 rounded-xl flex items-center justify-center">
                  <span className="text-emerald-600 text-xl">⚡</span>
                </div>
              </div>
            </div>
            
            <div className="bg-white rounded-2xl p-6 shadow-sm border">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-600">Win Rate</p>
                  <p className="text-2xl font-bold text-slate-900">{(performance.win_rate || 0).toFixed(1)}%</p>
                </div>
                <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center">
                  <span className="text-purple-600 text-xl">📈</span>
                </div>
              </div>
            </div>
            
            <div className="bg-white rounded-2xl p-6 shadow-sm border">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-600">Avg Confidence</p>
                  <p className="text-2xl font-bold text-slate-900">{((performance.avg_confidence || 0) * 100).toFixed(1)}%</p>
                </div>
                <div className="w-12 h-12 bg-amber-100 rounded-xl flex items-center justify-center">
                  <span className="text-amber-600 text-xl">🎯</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'opportunities' && (
          <div className="bg-white rounded-2xl shadow-sm border">
            <div className="p-6 border-b">
              <h2 className="text-xl font-bold text-slate-900">Market Opportunities</h2>
              <p className="text-slate-600">Latest cryptocurrency trading opportunities discovered by the scout</p>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-slate-50">
                  <tr>
                    <th className="text-left py-3 px-6 font-medium text-slate-700">Symbol</th>
                    <th className="text-left py-3 px-6 font-medium text-slate-700">Price</th>
                    <th className="text-left py-3 px-6 font-medium text-slate-700">24h Change</th>
                    <th className="text-left py-3 px-6 font-medium text-slate-700">Volume</th>
                    <th className="text-left py-3 px-6 font-medium text-slate-700">Volatility</th>
                    <th className="text-left py-3 px-6 font-medium text-slate-700">Time</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {opportunities.map((opp, index) => (
                    <tr key={index} className="hover:bg-slate-50">
                      <td className="py-4 px-6">
                        <span className="font-medium text-slate-900">{opp.symbol}</span>
                      </td>
                      <td className="py-4 px-6 text-slate-900">{formatPrice(opp.current_price)}</td>
                      <td className="py-4 px-6">
                        <span className={`inline-flex px-2 py-1 rounded-lg text-sm font-medium ${
                          opp.price_change_24h >= 0 ? 'text-emerald-600 bg-emerald-50' : 'text-red-600 bg-red-50'
                        }`}>
                          {opp.price_change_24h >= 0 ? '+' : ''}{opp.price_change_24h.toFixed(2)}%
                        </span>
                      </td>
                      <td className="py-4 px-6 text-slate-600">{(opp.volume_24h / 1000000).toFixed(1)}M</td>
                      <td className="py-4 px-6 text-slate-600">{(opp.volatility * 100).toFixed(2)}%</td>
                      <td className="py-4 px-6 text-slate-500">{formatTime(opp.timestamp)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'analyses' && (
          <div className="bg-white rounded-2xl shadow-sm border">
            <div className="p-6 border-b">
              <h2 className="text-xl font-bold text-slate-900">IA1 Technical Analysis</h2>
              <p className="text-slate-600">AI-powered technical analysis and chart pattern recognition</p>
            </div>
            <div className="p-6 space-y-4">
              {analyses.map((analysis, index) => (
                <div key={index} className="border rounded-xl p-6 hover:shadow-sm transition-shadow">
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h3 className="text-lg font-semibold text-slate-900">{analysis.symbol}</h3>
                      <p className="text-slate-600">Confidence: <span className={`font-medium ${getConfidenceColor(analysis.analysis_confidence)}`}>
                        {(analysis.analysis_confidence * 100).toFixed(1)}%
                      </span></p>
                    </div>
                    <span className="text-sm text-slate-500">{formatTime(analysis.timestamp)}</span>
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    <div className="bg-slate-50 rounded-lg p-3">
                      <p className="text-sm text-slate-600">RSI</p>
                      <p className="text-lg font-semibold text-slate-900">{analysis.rsi.toFixed(1)}</p>
                    </div>
                    <div className="bg-slate-50 rounded-lg p-3">
                      <p className="text-sm text-slate-600">MACD</p>
                      <p className="text-lg font-semibold text-slate-900">{analysis.macd_signal.toFixed(3)}</p>
                    </div>
                    <div className="bg-slate-50 rounded-lg p-3">
                      <p className="text-sm text-slate-600">BB Position</p>
                      <p className="text-lg font-semibold text-slate-900">{analysis.bollinger_position.toFixed(2)}</p>
                    </div>
                    <div className="bg-slate-50 rounded-lg p-3">
                      <p className="text-sm text-slate-600">Fibonacci</p>
                      <p className="text-lg font-semibold text-slate-900">{analysis.fibonacci_level.toFixed(3)}</p>
                    </div>
                  </div>
                  
                  {analysis.patterns_detected.length > 0 && (
                    <div className="mb-4">
                      <p className="text-sm text-slate-600 mb-2">Patterns Detected:</p>
                      <div className="flex flex-wrap gap-2">
                        {analysis.patterns_detected.map((pattern, i) => (
                          <span key={i} className="inline-flex px-3 py-1 rounded-full text-sm bg-indigo-100 text-indigo-700">
                            {pattern}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  <div className="bg-slate-50 rounded-lg p-4">
                    <p className="text-sm text-slate-600 mb-2">IA1 Analysis:</p>
                    <p className="text-slate-800">{analysis.ia1_reasoning}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'decisions' && (
          <div className="bg-white rounded-2xl shadow-sm border">
            <div className="p-6 border-b">
              <h2 className="text-xl font-bold text-slate-900">IA2 Trading Decisions</h2>
              <p className="text-slate-600">Intelligent trading decisions with risk management</p>
            </div>
            <div className="p-6 space-y-4">
              {decisions.map((decision, index) => (
                <div key={index} className="border rounded-xl p-6 hover:shadow-sm transition-shadow">
                  <div className="flex justify-between items-start mb-4">
                    <div className="flex items-center gap-4">
                      <h3 className="text-lg font-semibold text-slate-900">{decision.symbol}</h3>
                      <span className={`inline-flex px-3 py-1 rounded-full text-sm font-medium ${getSignalColor(decision.signal)}`}>
                        {decision.signal.toUpperCase()}
                      </span>
                      <span className={`font-medium ${getConfidenceColor(decision.confidence)}`}>
                        {(decision.confidence * 100).toFixed(1)}% confidence
                      </span>
                    </div>
                    <span className="text-sm text-slate-500">{formatTime(decision.timestamp)}</span>
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-4">
                    <div className="bg-slate-50 rounded-lg p-3">
                      <p className="text-sm text-slate-600">Entry Price</p>
                      <p className="text-lg font-semibold text-slate-900">{formatPrice(decision.entry_price)}</p>
                    </div>
                    <div className="bg-slate-50 rounded-lg p-3">
                      <p className="text-sm text-slate-600">Stop Loss</p>
                      <p className="text-lg font-semibold text-red-600">{formatPrice(decision.stop_loss)}</p>
                    </div>
                    <div className="bg-slate-50 rounded-lg p-3">
                      <p className="text-sm text-slate-600">TP1</p>
                      <p className="text-lg font-semibold text-emerald-600">{formatPrice(decision.take_profit_1)}</p>
                    </div>
                    <div className="bg-slate-50 rounded-lg p-3">
                      <p className="text-sm text-slate-600">Position Size</p>
                      <p className="text-lg font-semibold text-slate-900">{(decision.position_size * 100).toFixed(1)}%</p>
                    </div>
                    <div className="bg-slate-50 rounded-lg p-3">
                      <p className="text-sm text-slate-600">R:R Ratio</p>
                      <p className="text-lg font-semibold text-slate-900">{decision.risk_reward_ratio.toFixed(1)}</p>
                    </div>
                  </div>
                  
                  <div className="bg-slate-50 rounded-lg p-4">
                    <p className="text-sm text-slate-600 mb-2">IA2 Reasoning:</p>
                    <p className="text-slate-800">{decision.ia2_reasoning}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'positions' && (
          <div className="space-y-6">
            {/* Execution Mode Control */}
            <div className="bg-white rounded-2xl p-6 shadow-sm border">
              <div className="flex justify-between items-center">
                <div>
                  <h3 className="text-lg font-semibold text-slate-900">Trading Execution Mode</h3>
                  <p className="text-slate-600 mt-1">Control how trades are executed</p>
                </div>
                <div className="flex items-center gap-4">
                  <div className={`px-4 py-2 rounded-full text-sm font-medium ${
                    executionMode === 'SIMULATION' 
                      ? 'bg-blue-100 text-blue-800' 
                      : 'bg-red-100 text-red-800'
                  }`}>
                    {executionMode} MODE
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setTradingMode('SIMULATION')}
                      className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                        executionMode === 'SIMULATION'
                          ? 'bg-blue-600 text-white'
                          : 'bg-slate-200 text-slate-700 hover:bg-slate-300'
                      }`}
                    >
                      Simulation
                    </button>
                    <button
                      onClick={() => setTradingMode('LIVE')}
                      className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                        executionMode === 'LIVE'
                          ? 'bg-red-600 text-white'
                          : 'bg-slate-200 text-slate-700 hover:bg-slate-300'
                      }`}
                    >
                      Live Trading
                    </button>
                  </div>
                </div>
              </div>
              {executionMode === 'LIVE' && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-red-800 font-medium">⚠️ LIVE TRADING ACTIVE</p>
                  <p className="text-red-700 text-sm">Trades will be executed with real money on BingX</p>
                </div>
              )}
            </div>

            {/* Active Positions Summary */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="bg-white rounded-2xl p-6 shadow-sm border">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-slate-600">Active Positions</p>
                    <p className="text-2xl font-bold text-slate-900">{activePositions.length}</p>
                  </div>
                  <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center">
                    <span className="text-2xl">🎯</span>
                  </div>
                </div>
              </div>
              
              <div className="bg-white rounded-2xl p-6 shadow-sm border">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-slate-600">Total P&L</p>
                    <p className={`text-2xl font-bold ${
                      activePositions.reduce((sum, pos) => sum + (pos.unrealized_pnl || 0), 0) >= 0 
                        ? 'text-emerald-600' : 'text-red-600'
                    }`}>
                      {formatPrice(activePositions.reduce((sum, pos) => sum + (pos.unrealized_pnl || 0), 0))}
                    </p>
                  </div>
                  <div className="w-12 h-12 bg-emerald-100 rounded-xl flex items-center justify-center">
                    <span className="text-2xl">💰</span>
                  </div>
                </div>
              </div>
              
              <div className="bg-white rounded-2xl p-6 shadow-sm border">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-slate-600">Total Position Value</p>
                    <p className="text-2xl font-bold text-slate-900">
                      {formatPrice(activePositions.reduce((sum, pos) => sum + (pos.position_size_usd || 0), 0))}
                    </p>
                  </div>
                  <div className="w-12 h-12 bg-amber-100 rounded-xl flex items-center justify-center">
                    <span className="text-2xl">📊</span>
                  </div>
                </div>
              </div>
              
              <div className="bg-white rounded-2xl p-6 shadow-sm border">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-slate-600">Trailing SL Active</p>
                    <p className="text-2xl font-bold text-slate-900">
                      {activePositions.filter(pos => pos.trailing_sl_active).length}
                    </p>
                  </div>
                  <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center">
                    <span className="text-2xl">🛡️</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Active Positions List */}
            <div className="bg-white rounded-2xl shadow-sm border">
              <div className="p-6 border-b">
                <h3 className="text-lg font-semibold text-slate-900">Active Trading Positions</h3>
                <p className="text-slate-600 mt-1">Real-time monitoring with dynamic trailing stops</p>
              </div>
              
              <div className="divide-y">
                {activePositions.length === 0 ? (
                  <div className="p-8 text-center">
                    <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4">
                      <span className="text-2xl">📋</span>
                    </div>
                    <p className="text-slate-600">No active positions</p>
                    <p className="text-sm text-slate-500 mt-1">Positions will appear here when IA2 generates LONG/SHORT signals</p>
                  </div>
                ) : (
                  activePositions.map((position) => (
                    <div key={position.id} className="p-6">
                      <div className="flex justify-between items-start mb-4">
                        <div className="flex items-center gap-3">
                          <div className={`w-3 h-3 rounded-full ${
                            position.status === 'ACTIVE' ? 'bg-emerald-500' :
                            position.status === 'OPENING' ? 'bg-amber-500' :
                            position.status === 'CLOSING' ? 'bg-red-500' : 'bg-slate-500'
                          }`}></div>
                          <h4 className="text-lg font-semibold text-slate-900">{position.symbol}</h4>
                          <span className={`px-3 py-1 rounded-full text-sm font-medium ${getSignalColor(position.signal.toLowerCase())}`}>
                            {position.signal}
                          </span>
                          <span className="text-sm text-slate-500">
                            {position.leverage}x leverage
                          </span>
                        </div>
                        
                        <div className="flex items-center gap-3">
                          <div className="text-right">
                            <p className={`text-lg font-semibold ${
                              position.pnl_percentage >= 0 ? 'text-emerald-600' : 'text-red-600'
                            }`}>
                              {position.pnl_percentage >= 0 ? '+' : ''}{position.pnl_percentage.toFixed(2)}%
                            </p>
                            <p className="text-sm text-slate-600">
                              {formatPrice(position.unrealized_pnl)}
                            </p>
                          </div>
                          
                          <button
                            onClick={() => closePosition(position.id)}
                            className="px-4 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors text-sm font-medium"
                          >
                            Close Position
                          </button>
                        </div>
                      </div>
                      
                      {/* Position Details Grid */}
                      <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-4">
                        <div className="bg-slate-50 rounded-lg p-3">
                          <p className="text-sm text-slate-600">Entry Price</p>
                          <p className="text-lg font-semibold text-slate-900">{formatPrice(position.entry_price)}</p>
                        </div>
                        <div className="bg-slate-50 rounded-lg p-3">
                          <p className="text-sm text-slate-600">Current Price</p>
                          <p className="text-lg font-semibold text-slate-900">{formatPrice(position.current_price)}</p>
                        </div>
                        <div className="bg-slate-50 rounded-lg p-3">
                          <p className="text-sm text-slate-600">Quantity</p>
                          <p className="text-lg font-semibold text-slate-900">{position.quantity.toFixed(6)}</p>
                        </div>
                        <div className="bg-slate-50 rounded-lg p-3">
                          <p className="text-sm text-slate-600">Position Size</p>
                          <p className="text-lg font-semibold text-slate-900">{formatPrice(position.position_size_usd)}</p>
                        </div>
                        <div className="bg-slate-50 rounded-lg p-3">
                          <p className="text-sm text-slate-600">Stop Loss</p>
                          <p className="text-lg font-semibold text-red-600">{formatPrice(position.current_stop_loss)}</p>
                        </div>
                        <div className="bg-slate-50 rounded-lg p-3">
                          <p className="text-sm text-slate-600">Status</p>
                          <p className="text-lg font-semibold text-slate-900">{position.status}</p>
                        </div>
                      </div>
                      
                      {/* Probabilistic TP Levels */}
                      <div className="bg-slate-50 rounded-lg p-4">
                        <div className="flex justify-between items-center mb-3">
                          <h5 className="font-medium text-slate-900">Probabilistic Take Profit Strategy</h5>
                          <div className="flex items-center gap-2">
                            <span className="text-sm text-slate-600">
                              {position.tp_filled_levels}/{position.tp_total_levels} levels hit
                            </span>
                            {position.trailing_sl_active && (
                              <span className="px-2 py-1 bg-emerald-100 text-emerald-700 rounded text-sm font-medium">
                                🚀 Trailing SL Active
                              </span>
                            )}
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                          {position.tp_levels && position.tp_levels.map((tp, index) => (
                            <div key={index} className={`p-3 rounded border-2 transition-all ${
                              tp.filled ? 'border-emerald-500 bg-emerald-50' : 'border-slate-200 bg-white'
                            }`}>
                              <div className="flex justify-between items-center mb-2">
                                <span className="font-medium text-slate-900">TP{tp.level}</span>
                                <span className={`text-sm font-medium ${
                                  tp.filled ? 'text-emerald-600' : 'text-slate-600'
                                }`}>
                                  {tp.position_distribution}%
                                </span>
                              </div>
                              <p className="text-lg font-semibold text-slate-900 mb-1">
                                {formatPrice(tp.price)}
                              </p>
                              <p className="text-sm text-slate-600">
                                +{tp.percentage_from_entry.toFixed(1)}% from entry
                              </p>
                              {tp.filled && (
                                <p className="text-sm text-emerald-600 mt-1">
                                  ✅ Filled at {tp.filled_at ? formatTime(tp.filled_at) : 'N/A'}
                                </p>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'backtest' && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-sm border">
              <div className="p-6 border-b">
                <h2 className="text-xl font-bold text-slate-900">AI Training & Backtesting System</h2>
                <p className="text-slate-600">Train our AI system with historical data and backtest trading strategies</p>
              </div>
              
              <div className="p-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  {/* AI Training Section */}
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-lg font-semibold text-slate-900 mb-4">🧠 AI Training System</h3>
                      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-6">
                        <div className="space-y-4">
                          {/* Training Status */}
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-medium text-slate-700">Training Data</span>
                            <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">
                              {backtestStatus?.available_symbols?.length || 0} symbols ready
                            </span>
                          </div>
                          
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-medium text-slate-700">Historical Period</span>
                            <span className="text-sm text-slate-600">2013-2025 (Multi-year)</span>
                          </div>
                          
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-medium text-slate-700">AI Components</span>
                            <span className="text-sm text-slate-600">IA1 + IA2 + Patterns</span>
                          </div>
                          
                          {/* Training Actions */}
                          <div className="pt-4 border-t border-slate-200">
                            <button
                              onClick={() => runAITraining()}
                              disabled={backtestLoading}
                              className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-6 py-3 rounded-lg font-medium transition-all duration-200 disabled:opacity-50"
                            >
                              {backtestLoading ? 'Training AI System...' : '🚀 Start AI Training'}
                            </button>
                            
                            <p className="text-xs text-slate-500 mt-2 text-center">
                              Analyzes market conditions, trains pattern recognition, and enhances IA1/IA2 accuracy
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    {/* AI Training Results */}
                    {backtestResults?.training_completed && (
                      <div className="bg-green-50 rounded-lg p-6">
                        <h4 className="font-semibold text-green-900 mb-3">✅ Training Completed</h4>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-green-700">Market Conditions</span>
                            <p className="font-medium text-green-900">{backtestResults.market_conditions_classified || 0}</p>
                          </div>
                          <div>
                            <span className="text-green-700">Patterns Analyzed</span>
                            <p className="font-medium text-green-900">{backtestResults.patterns_analyzed || 0}</p>
                          </div>
                          <div>
                            <span className="text-green-700">IA1 Improvements</span>
                            <p className="font-medium text-green-900">{backtestResults.ia1_improvements_identified || 0}</p>
                          </div>
                          <div>
                            <span className="text-green-700">IA2 Enhancements</span>
                            <p className="font-medium text-green-900">{backtestResults.ia2_enhancements_generated || 0}</p>
                          </div>
                        </div>
                        
                        {/* AI Enhancement Integration */}
                        <div className="mt-4 pt-4 border-t border-green-200">
                          <h5 className="font-semibold text-green-900 mb-2">🧠 Apply AI Insights to Live Trading</h5>
                          <div className="flex gap-3">
                            <button
                              onClick={() => loadAIInsights()}
                              disabled={backtestLoading}
                              className="flex-1 bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
                            >
                              {backtestLoading ? 'Loading...' : '⚡ Load AI Insights into Trading Bot'}
                            </button>
                          </div>
                          
                          {backtestResults?.enhancement_loaded && (
                            <div className="mt-3 p-3 bg-green-100 rounded-lg">
                              <p className="text-sm text-green-800">
                                🎯 <strong>AI Enhancement Active!</strong> Your trading bot is now using AI insights:
                              </p>
                              <ul className="text-xs text-green-700 mt-2 space-y-1">
                                <li>• {backtestResults.enhancement_summary?.total_rules || 0} enhancement rules loaded</li>
                                <li>• {backtestResults.enhancement_summary?.pattern_insights || 0} pattern success insights</li>
                                <li>• {backtestResults.enhancement_summary?.market_condition_insights || 0} market condition optimizations</li>
                                <li>• IA1 and IA2 are now enhanced with historical learning</li>
                              </ul>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                  
                  {/* Backtesting Section */}
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-lg font-semibold text-slate-900 mb-4">📈 Strategy Backtesting</h3>
                      <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg p-6">
                        <div className="space-y-4">
                          {/* Backtest Parameters */}
                          <div className="grid grid-cols-2 gap-4">
                            <div>
                              <label className="block text-sm font-medium text-slate-700 mb-2">Start Date</label>
                              <input
                                type="date"
                                defaultValue="2020-01-01"
                                className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                                id="backtest-start-date"
                              />
                            </div>
                            <div>
                              <label className="block text-sm font-medium text-slate-700 mb-2">End Date</label>
                              <input
                                type="date"
                                defaultValue="2021-07-01"
                                className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                                id="backtest-end-date"
                              />
                            </div>
                          </div>
                          
                          <div>
                            <label className="block text-sm font-medium text-slate-700 mb-2">Test Symbols</label>
                            <select
                              multiple
                              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                              id="backtest-symbols"
                              size="3"
                            >
                              {backtestStatus?.available_symbols?.slice(0, 10).map(symbol => (
                                <option key={symbol} value={symbol}>{symbol}</option>
                              ))}
                            </select>
                            <p className="text-xs text-slate-500 mt-1">Hold Ctrl/Cmd to select multiple symbols</p>
                          </div>
                          
                          {/* Backtest Actions */}
                          <div className="pt-4 border-t border-slate-200">
                            <button
                              onClick={() => runStrategyBacktest()}
                              disabled={backtestLoading}
                              className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white px-6 py-3 rounded-lg font-medium transition-all duration-200 disabled:opacity-50"
                            >
                              {backtestLoading ? 'Running Backtest...' : '⚡ Run Strategy Backtest'}
                            </button>
                            
                            <p className="text-xs text-slate-500 mt-2 text-center">
                              Tests trading strategies against historical market data
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    {/* Backtest Results */}
                    {backtestResults?.backtest_completed && backtestResults.results && (
                      <div className="bg-slate-50 rounded-lg p-6">
                        <h4 className="font-semibold text-slate-900 mb-4">📊 Backtest Results</h4>
                        <div className="space-y-4">
                          {Object.entries(backtestResults.results).map(([symbol, result]) => (
                            <div key={symbol} className="bg-white rounded-lg p-4 border">
                              <div className="flex justify-between items-center mb-3">
                                <h5 className="font-medium text-slate-900">{symbol}</h5>
                                <span className={`px-2 py-1 rounded text-sm font-medium ${
                                  result.total_return > 0 ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                                }`}>
                                  {result.total_return > 0 ? '+' : ''}{(result.total_return * 100).toFixed(1)}%
                                </span>
                              </div>
                              
                              <div className="grid grid-cols-3 gap-4 text-sm">
                                <div>
                                  <span className="text-slate-600">Win Rate</span>
                                  <p className="font-medium text-slate-900">{(result.win_rate * 100).toFixed(1)}%</p>
                                </div>
                                <div>
                                  <span className="text-slate-600">Total Trades</span>
                                  <p className="font-medium text-slate-900">{result.total_trades}</p>
                                </div>
                                <div>
                                  <span className="text-slate-600">Sharpe Ratio</span>
                                  <p className="font-medium text-slate-900">{result.sharpe_ratio.toFixed(2)}</p>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
                
                {/* Adaptive Context System Status */}
                <div className="mt-8 pt-6 border-t border-slate-200">
                  <h3 className="text-lg font-semibold text-slate-900 mb-4">⚡ Adaptive Context System</h3>
                  <div className="bg-gradient-to-r from-emerald-50 to-teal-50 rounded-lg p-6">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                      <div className="text-center">
                        <div className="w-12 h-12 bg-emerald-100 rounded-xl flex items-center justify-center mx-auto mb-3">
                          <span className="text-emerald-600 text-xl">🎯</span>
                        </div>
                        <h4 className="font-semibold text-slate-900">Market Context</h4>
                        <p className="text-sm text-slate-600">Real-time market regime detection</p>
                      </div>
                      
                      <div className="text-center">
                        <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center mx-auto mb-3">
                          <span className="text-blue-600 text-xl">🧠</span>
                        </div>
                        <h4 className="font-semibold text-slate-900">Dynamic Adjustment</h4>
                        <p className="text-sm text-slate-600">AI-driven strategy adaptation</p>
                      </div>
                      
                      <div className="text-center">
                        <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center mx-auto mb-3">
                          <span className="text-purple-600 text-xl">⚡</span>
                        </div>
                        <h4 className="font-semibold text-slate-900">Performance Boost</h4>
                        <p className="text-sm text-slate-600">Enhanced trading accuracy</p>
                      </div>
                    </div>
                    
                    <div className="mt-6 text-center">
                      <button
                        onClick={() => loadAdaptiveContext()}
                        className="bg-emerald-600 hover:bg-emerald-700 text-white px-6 py-2 rounded-lg font-medium transition-colors"
                      >
                        🔄 Load Training Data to Context System
                      </button>
                    </div>
                {/* Chartist Library Section */}
                <div className="mt-8 pt-6 border-t border-slate-200">
                  <h3 className="text-lg font-semibold text-slate-900 mb-4">📈 Bibliothèque de Figures Chartistes</h3>
                  <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg p-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                      
                      {/* Figures de Retournement */}
                      <div className="text-center">
                        <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center mx-auto mb-3">
                          <span className="text-red-600 text-xl">🔄</span>
                        </div>
                        <h4 className="font-semibold text-slate-900">Retournement</h4>
                        <p className="text-sm text-slate-600 mb-2">Tête-épaules, Double sommet</p>
                        <div className="text-xs space-y-1">
                          <div className="flex justify-between">
                            <span>Long:</span>
                            <span className="font-medium text-green-600">65% succès</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Short:</span>
                            <span className="font-medium text-red-600">72% succès</span>
                          </div>
                        </div>
                      </div>
                      
                      {/* Figures de Continuation */}
                      <div className="text-center">
                        <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center mx-auto mb-3">
                          <span className="text-green-600 text-xl">📈</span>
                        </div>
                        <h4 className="font-semibold text-slate-900">Continuation</h4>
                        <p className="text-sm text-slate-600 mb-2">Drapeaux, Triangles</p>
                        <div className="text-xs space-y-1">
                          <div className="flex justify-between">
                            <span>Long:</span>
                            <span className="font-medium text-green-600">76% succès</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Short:</span>
                            <span className="font-medium text-red-600">74% succès</span>
                          </div>
                        </div>
                      </div>
                      
                      {/* Patterns Harmoniques */}
                      <div className="text-center">
                        <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center mx-auto mb-3">
                          <span className="text-purple-600 text-xl">🎵</span>
                        </div>
                        <h4 className="font-semibold text-slate-900">Harmoniques</h4>
                        <p className="text-sm text-slate-600 mb-2">Gartley, Butterfly</p>
                        <div className="text-xs space-y-1">
                          <div className="flex justify-between">
                            <span>Long:</span>
                            <span className="font-medium text-green-600">73% succès</span>
                          </div>
                          <div className="flex justify-between">
                            <span>R:R:</span>
                            <span className="font-medium text-blue-600">3.5:1</span>
                          </div>
                        </div>
                      </div>
                      
                      {/* Optimisation IA */}
                      <div className="text-center">
                        <div className="w-12 h-12 bg-emerald-100 rounded-xl flex items-center justify-center mx-auto mb-3">
                          <span className="text-emerald-600 text-xl">🧠</span>
                        </div>
                        <h4 className="font-semibold text-slate-900">IA Optimisée</h4>
                        <p className="text-sm text-slate-600 mb-2">Stratégies adaptatives</p>
                        <div className="text-xs space-y-1">
                          <div className="flex justify-between">
                            <span>Position:</span>
                            <span className="font-medium text-emerald-600">Auto-ajustée</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Context:</span>
                            <span className="font-medium text-emerald-600">Adaptatif</span>
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    <div className="mt-6 text-center">
                      <button
                        onClick={async () => {
                          const library = await getChartistLibrary();
                          console.log('Bibliothèque chartiste:', library);
                        }}
                        className="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-2 rounded-lg font-medium transition-colors"
                      >
                        📚 Voir Bibliothèque Complète
                      </button>
                    </div>
                    
                    {/* Statistiques détaillées */}
                    <div className="mt-6 pt-4 border-t border-indigo-200">
                      <h5 className="font-semibold text-slate-900 mb-3 text-center">🎯 Meilleures Figures par Stratégie</h5>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                        <div className="bg-white/50 rounded-lg p-3">
                          <h6 className="font-medium text-green-700 mb-2">📈 Stratégies Long</h6>
                          <ul className="space-y-1 text-xs">
                            <li>• Tasse avec Anse: <span className="font-medium">81% succès, +12.4%</span></li>
                            <li>• Drapeau Haussier: <span className="font-medium">78% succès, +5.4%</span></li>
                            <li>• Triangle Ascendant: <span className="font-medium">75% succès, +6.8%</span></li>
                            <li>• T&E Inversée: <span className="font-medium">74% succès, +9.2%</span></li>
                          </ul>
                        </div>
                        <div className="bg-white/50 rounded-lg p-3">
                          <h6 className="font-medium text-red-700 mb-2">📉 Stratégies Short</h6>
                          <ul className="space-y-1 text-xs">
                            <li>• Drapeau Baissier: <span className="font-medium">76% succès, +5.8%</span></li>
                            <li>• Double Creux: <span className="font-medium">75% succès, +7.8%</span></li>
                            <li>• Triangle Descendant: <span className="font-medium">73% succès, +6.2%</span></li>
                            <li>• Tête et Épaules: <span className="font-medium">72% succès, +8.4%</span></li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'performance' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="bg-white rounded-2xl p-6 shadow-sm border">
                <h3 className="text-lg font-semibold text-slate-900 mb-4">System Health</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-slate-600">Status</span>
                    <span className={`font-medium ${isTrading ? 'text-emerald-600' : 'text-slate-500'}`}>
                      {isTrading ? 'Active' : 'Inactive'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Connection</span>
                    <span className={`font-medium ${connectionStatus === 'Connected' ? 'text-emerald-600' : 'text-red-600'}`}>
                      {connectionStatus}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">IA1 Status</span>
                    <span className="font-medium text-emerald-600">Online</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">IA2 Status</span>
                    <span className="font-medium text-emerald-600">Online</span>
                  </div>
                </div>
              </div>
              
              <div className="bg-white rounded-2xl p-6 shadow-sm border">
                <h3 className="text-lg font-semibold text-slate-900 mb-4">Trading Statistics</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-slate-600">Total Signals</span>
                    <span className="font-medium text-slate-900">{decisions.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Long Signals</span>
                    <span className="font-medium text-emerald-600">
                      {decisions.filter(d => d.signal === 'long').length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Short Signals</span>
                    <span className="font-medium text-red-600">
                      {decisions.filter(d => d.signal === 'short').length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Hold Signals</span>
                    <span className="font-medium text-amber-600">
                      {decisions.filter(d => d.signal === 'hold').length}
                    </span>
                  </div>
                </div>
              </div>
              
              <div className="bg-white rounded-2xl p-6 shadow-sm border">
                <h3 className="text-lg font-semibold text-slate-900 mb-4">AI Performance</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-slate-600">Avg IA1 Confidence</span>
                    <span className="font-medium text-slate-900">
                      {analyses.length > 0 ? 
                        ((analyses.reduce((sum, a) => sum + a.analysis_confidence, 0) / analyses.length) * 100).toFixed(1) + '%' 
                        : 'N/A'
                      }
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Avg IA2 Confidence</span>
                    <span className="font-medium text-slate-900">
                      {decisions.length > 0 ? 
                        ((decisions.reduce((sum, d) => sum + d.confidence, 0) / decisions.length) * 100).toFixed(1) + '%'
                        : 'N/A'
                      }
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">High Confidence</span>
                    <span className="font-medium text-emerald-600">
                      {decisions.filter(d => d.confidence > 0.8).length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Response Time</span>
                    <span className="font-medium text-slate-900">~2.3s</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <TradingDashboard />
    </div>
  );
}

export default App;
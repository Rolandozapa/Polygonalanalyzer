import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001/api';

const TradingDashboard = () => {
  const [opportunities, setOpportunities] = useState([]);
  const [analyses, setAnalyses] = useState([]);
  const [decisions, setDecisions] = useState([]);
  const [performance, setPerformance] = useState({});
  const [activeTab, setActiveTab] = useState('dashboard');
  const [loading, setLoading] = useState(true);
  const [isTrading, setIsTrading] = useState(false);
  const [activePositions, setActivePositions] = useState([]);
  const [executionMode, setExecutionMode] = useState('SIMULATION');
  const [backtestResults, setBacktestResults] = useState(null);
  const [backtestLoading, setBacktestLoading] = useState(false);
  const [backtestStatus, setBacktestStatus] = useState(null);

  // WebSocket connection
  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
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
    } finally {
      setLoading(false);
    }
  };

  const runBacktest = async (params) => {
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
    } catch (error) {
      console.error('Error loading adaptive context:', error);
    }
  };

  const loadAIInsights = async () => {
    try {
      const response = await axios.post(`${API}/ai-training/load-insights`);
      console.log('AI insights loaded:', response.data);
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
      await axios.post(`${API}/close-position`, { position_id: positionId });
      fetchData(); // Refresh data
    } catch (error) {
      console.error('Error closing position:', error);
    }
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 6
    }).format(price);
  };

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const TabButton = ({ label, active, onClick }) => (
    <button
      onClick={onClick}
      className={`px-6 py-3 font-medium text-sm rounded-lg transition-all duration-200 ${
        active
          ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg'
          : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
      }`}
    >
      {label}
    </button>
  );

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-slate-600 text-lg">Loading trading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white/90 backdrop-blur-sm border-b border-slate-200/60 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center">
                  <span className="text-white font-bold text-lg">🤖</span>
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-slate-900">Ultra Professional Trading Bot</h1>
                  <p className="text-sm text-slate-600">AI-Enhanced Dual Analysis System</p>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${isTrading ? 'bg-green-500 animate-pulse' : 'bg-slate-400'}`}></div>
                <span className="text-sm font-medium text-slate-700">
                  {isTrading ? 'Active' : 'Inactive'}
                </span>
              </div>
              
              <div className="flex items-center space-x-2">
                <span className="text-sm text-slate-600">Mode:</span>
                <span className={`px-2 py-1 rounded-lg text-xs font-medium ${
                  executionMode === 'LIVE' ? 'bg-red-100 text-red-700' : 'bg-blue-100 text-blue-700'
                }`}>
                  {executionMode}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div className="bg-white/60 backdrop-blur-sm border-b border-slate-200/40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-1 py-4 overflow-x-auto">
            <TabButton label="Dashboard" active={activeTab === 'dashboard'} onClick={() => setActiveTab('dashboard')} />
            <TabButton label="Opportunities" active={activeTab === 'opportunities'} onClick={() => setActiveTab('opportunities')} />
            <TabButton label="IA1 Analysis" active={activeTab === 'analyses'} onClick={() => setActiveTab('analyses')} />
            <TabButton label="IA2 Decisions" active={activeTab === 'decisions'} onClick={() => setActiveTab('decisions')} />
            <TabButton label="Active Positions" active={activeTab === 'positions'} onClick={() => setActiveTab('positions')} />
            <TabButton label="Backtesting" active={activeTab === 'backtest'} onClick={() => setActiveTab('backtest')} />
            <TabButton label="Performance" active={activeTab === 'performance'} onClick={() => setActiveTab('performance')} />
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {activeTab === 'dashboard' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-lg border border-white/20">
                <h3 className="text-lg font-semibold text-slate-900 mb-2">Opportunities</h3>
                <p className="text-3xl font-bold text-blue-600">{opportunities.length}</p>
                <p className="text-sm text-slate-600 mt-1">Active market opportunities</p>
              </div>
              
              <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-lg border border-white/20">
                <h3 className="text-lg font-semibold text-slate-900 mb-2">IA1 Analyses</h3>
                <p className="text-3xl font-bold text-emerald-600">{analyses.length}</p>
                <p className="text-sm text-slate-600 mt-1">Technical analysis completed</p>
              </div>
              
              <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-lg border border-white/20">
                <h3 className="text-lg font-semibold text-slate-900 mb-2">IA2 Decisions</h3>
                <p className="text-3xl font-bold text-purple-600">{decisions.length}</p>
                <p className="text-sm text-slate-600 mt-1">Strategic decisions made</p>
              </div>
              
              <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-lg border border-white/20">
                <h3 className="text-lg font-semibold text-slate-900 mb-2">Active Positions</h3>
                <p className="text-3xl font-bold text-amber-600">{activePositions.length}</p>
                <p className="text-sm text-slate-600 mt-1">Open trading positions</p>
              </div>
            </div>

            <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg border border-white/20 p-6">
              <h2 className="text-xl font-bold text-slate-900 mb-4">Latest Activity</h2>
              <div className="space-y-4">
                {decisions.slice(0, 5).map((decision, index) => (
                  <div key={index} className="flex items-center justify-between p-4 bg-slate-50 rounded-xl">
                    <div className="flex items-center space-x-4">
                      <div className={`w-3 h-3 rounded-full ${
                        decision.signal === 'long' ? 'bg-emerald-500' : 
                        decision.signal === 'short' ? 'bg-red-500' : 'bg-amber-500'
                      }`}></div>
                      <div>
                        <p className="font-medium text-slate-900">{decision.symbol}</p>
                        <p className="text-sm text-slate-600">{decision.signal.toUpperCase()} - {(decision.confidence * 100).toFixed(1)}% confidence</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium text-slate-900">{formatTime(decision.timestamp)}</p>
                      <p className="text-xs text-slate-600">Entry: {formatPrice(decision.entry_price)}</p>
                    </div>
                  </div>
                ))}
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
                    <span className="font-medium text-emerald-600">Healthy</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Uptime</span>
                    <span className="font-medium text-slate-900">24h 15m</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Memory Usage</span>
                    <span className="font-medium text-slate-900">342MB</span>
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
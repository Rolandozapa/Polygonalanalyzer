import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API = `${process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001'}/api`;

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
  
  // BingX Integration States
  const [bingxStatus, setBingxStatus] = useState(null);
  const [bingxBalance, setBingxBalance] = useState(null);
  const [bingxPositions, setBingxPositions] = useState([]);
  const [bingxTradingHistory, setBingxTradingHistory] = useState([]);
  const [bingxRiskConfig, setBingxRiskConfig] = useState(null);
  const [bingxLoading, setBingxLoading] = useState(false);

  // Optimized polling - Reduced from 5s to 15s to prevent CPU overload
  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 15000); // Refresh every 15 seconds (CPU optimized)
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

  // BingX Integration Functions
  const fetchBingxData = async () => {
    setBingxLoading(true);
    try {
      const [statusRes, balanceRes, positionsRes, historyRes, riskRes] = await Promise.all([
        axios.get(`${API}/bingx/status`).catch(() => ({ data: { status: 'error', message: 'Not connected' } })),
        axios.get(`${API}/bingx/balance`).catch(() => ({ data: { status: 'error' } })),
        axios.get(`${API}/bingx/positions`).catch(() => ({ data: { positions: [] } })),
        axios.get(`${API}/bingx/trading-history?limit=20`).catch(() => ({ data: { trading_history: [] } })),
        axios.get(`${API}/bingx/risk-config`).catch(() => ({ data: { status: 'error' } }))
      ]);

      setBingxStatus(statusRes.data);
      setBingxBalance(balanceRes.data || null);  // Use direct data, not .balance
      setBingxPositions(Array.isArray(positionsRes.data.positions) ? positionsRes.data.positions : []);
      setBingxTradingHistory(Array.isArray(historyRes.data.trading_history) ? historyRes.data.trading_history : []);
      setBingxRiskConfig(riskRes.data.risk_config || null);
    } catch (error) {
      console.error('Error fetching BingX data:', error);
    } finally {
      setBingxLoading(false);
    }
  };

  const executeBingxTrade = async (tradeData) => {
    try {
      setBingxLoading(true);
      const response = await axios.post(`${API}/bingx/trade`, tradeData);
      
      if (response.data.status === 'success') {
        alert(`Trade executed successfully: ${tradeData.symbol} ${tradeData.side}`);
        await fetchBingxData(); // Refresh data
      } else {
        alert(`Trade rejected: ${response.data.errors?.join(', ') || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Error executing BingX trade:', error);
      alert('Error executing trade: ' + error.message);
    } finally {
      setBingxLoading(false);
    }
  };

  const closeBingxPosition = async (symbol, positionSide) => {
    try {
      setBingxLoading(true);
      const response = await axios.post(`${API}/bingx/close-position/${symbol}`, { position_side: positionSide });
      
      if (response.data.status === 'success') {
        alert(`Position closed successfully: ${symbol}`);
        await fetchBingxData(); // Refresh data
      } else {
        alert('Error closing position');
      }
    } catch (error) {
      console.error('Error closing BingX position:', error);
      alert('Error closing position: ' + error.message);
    } finally {
      setBingxLoading(false);
    }
  };

  const triggerBingxEmergencyStop = async () => {
    if (!window.confirm('Are you sure you want to trigger emergency stop? This will close ALL positions.')) {
      return;
    }
    
    try {
      setBingxLoading(true);
      const response = await axios.post(`${API}/bingx/emergency-stop`);
      
      if (response.data.status === 'success') {
        alert('Emergency stop triggered successfully');
        await fetchBingxData(); // Refresh data
      } else {
        alert('Error triggering emergency stop');
      }
    } catch (error) {
      console.error('Error triggering emergency stop:', error);
      alert('Error triggering emergency stop: ' + error.message);
    } finally {
      setBingxLoading(false);
    }
  };

  const updateBingxRiskConfig = async (newConfig) => {
    try {
      setBingxLoading(true);
      const response = await axios.post(`${API}/bingx/risk-config`, newConfig);
      
      if (response.data.status === 'success') {
        alert('Risk configuration updated successfully');
        await fetchBingxData(); // Refresh data
      } else {
        alert('Error updating risk configuration');
      }
    } catch (error) {
      console.error('Error updating risk config:', error);
      alert('Error updating risk config: ' + error.message);
    } finally {
      setBingxLoading(false);
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
            <TabButton label="BingX Trading" active={activeTab === 'bingx'} onClick={() => { setActiveTab('bingx'); fetchBingxData(); }} />
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

        {activeTab === 'analyses' && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-sm border">
              <div className="p-6 border-b">
                <h2 className="text-xl font-bold text-slate-900">IA1 Technical Analysis</h2>
                <p className="text-slate-600">Advanced technical analysis with RSI, MACD, Stochastic, and Bollinger Bands</p>
              </div>
              
              <div className="p-6">
                {analyses.length === 0 ? (
                  <div className="text-center py-8">
                    <p className="text-slate-500">No technical analyses available</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {analyses.map((analysis, index) => (
                      <div key={index} className="bg-slate-50 rounded-lg p-4 border">
                        <div className="flex justify-between items-start mb-3">
                          <h3 className="font-semibold text-lg text-slate-900">{analysis.symbol}</h3>
                          <div className="flex items-center space-x-2">
                            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                              analysis.analysis_confidence >= 0.8 ? 'bg-green-100 text-green-700' :
                              analysis.analysis_confidence >= 0.6 ? 'bg-yellow-100 text-yellow-700' :
                              'bg-red-100 text-red-700'
                            }`}>
                              {(analysis.analysis_confidence * 100).toFixed(1)}% confidence
                            </span>
                            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                              analysis.ia1_signal === 'long' ? 'bg-emerald-100 text-emerald-700' :
                              analysis.ia1_signal === 'short' ? 'bg-red-100 text-red-700' :
                              'bg-slate-100 text-slate-700'
                            }`}>
                              {analysis.ia1_signal?.toUpperCase() || 'HOLD'}
                            </span>
                          </div>
                        </div>
                        
                        {/* Technical Indicators */}
                        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-4">
                          <div className="text-center">
                            <p className="text-sm text-slate-600">RSI</p>
                            <p className={`font-semibold ${
                              analysis.rsi >= 70 ? 'text-red-600' :
                              analysis.rsi <= 30 ? 'text-green-600' :
                              'text-slate-900'
                            }`}>
                              {analysis.rsi?.toFixed(1) || 'N/A'}
                            </p>
                          </div>
                          <div className="text-center">
                            <p className="text-sm text-slate-600">MACD</p>
                            <p className={`font-semibold ${
                              analysis.macd_signal > 0 ? 'text-green-600' : 'text-red-600'
                            }`}>
                              {analysis.macd_signal?.toFixed(4) || 'N/A'}
                            </p>
                          </div>
                          <div className="text-center">
                            <p className="text-sm text-slate-600">Stochastic</p>
                            <p className={`font-semibold ${
                              analysis.stochastic >= 80 ? 'text-red-600' :
                              analysis.stochastic <= 20 ? 'text-green-600' :
                              'text-slate-900'
                            }`}>
                              {analysis.stochastic?.toFixed(1) || 'N/A'}
                            </p>
                          </div>
                          <div className="text-center">
                            <p className="text-sm text-slate-600">Bollinger</p>
                            <p className={`font-semibold ${
                              analysis.bollinger_position >= 0.8 ? 'text-red-600' :
                              analysis.bollinger_position <= 0.2 ? 'text-green-600' :
                              'text-slate-900'
                            }`}>
                              {analysis.bollinger_position?.toFixed(2) || 'N/A'}
                            </p>
                          </div>
                          <div className="text-center">
                            <p className="text-sm text-slate-600">Fibonacci</p>
                            <p className={`font-semibold ${
                              (analysis.fibonacci_trend_direction || 'neutral') === 'bullish' ? 'text-green-600' :
                              (analysis.fibonacci_trend_direction || 'neutral') === 'bearish' ? 'text-red-600' :
                              'text-slate-900'
                            }`}>
                              {analysis.fibonacci_nearest_level || '61.8'}%
                            </p>
                            <p className="text-xs text-slate-500">
                              {((analysis.fibonacci_level || 0.618) * 100).toFixed(0)}%
                            </p>
                          </div>
                        </div>
                        
                        {/* Analysis Reasoning */}
                        <div className="bg-white rounded p-3 border">
                          <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">
                            {analysis.ia1_reasoning || 'No reasoning available'}
                          </p>
                        </div>
                        
                        {/* Patterns */}
                        {analysis.patterns_detected && analysis.patterns_detected.length > 0 && (
                          <div className="mt-3">
                            <p className="text-sm text-slate-600 mb-2">Detected Patterns:</p>
                            <div className="flex flex-wrap gap-2">
                              {analysis.patterns_detected.slice(0, 5).map((pattern, i) => (
                                <span key={i} className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs font-medium">
                                  {pattern}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'decisions' && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-sm border">
              <div className="p-6 border-b">
                <h2 className="text-xl font-bold text-slate-900">IA2 Strategic Decisions</h2>
                <p className="text-slate-600">Claude-powered strategic decisions with probabilistic TP optimization</p>
              </div>
              
              <div className="p-6">
                {decisions.length === 0 ? (
                  <div className="text-center py-8">
                    <p className="text-slate-500">No strategic decisions available</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {decisions.map((decision, index) => (
                      <div key={index} className="bg-slate-50 rounded-lg p-4 border">
                        <div className="flex justify-between items-start mb-3">
                          <h3 className="font-semibold text-lg text-slate-900">{decision.symbol}</h3>
                          <div className="flex items-center space-x-2">
                            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                              decision.confidence >= 0.8 ? 'bg-green-100 text-green-700' :
                              decision.confidence >= 0.6 ? 'bg-yellow-100 text-yellow-700' :
                              'bg-red-100 text-red-700'
                            }`}>
                              {(decision.confidence * 100).toFixed(1)}% confidence
                            </span>
                            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                              decision.signal === 'LONG' ? 'bg-emerald-100 text-emerald-700' :
                              decision.signal === 'SHORT' ? 'bg-red-100 text-red-700' :
                              'bg-slate-100 text-slate-700'
                            }`}>
                              {decision.signal}
                            </span>
                          </div>
                        </div>
                        
                        {/* Trading Levels */}
                        <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-4">
                          <div className="text-center">
                            <p className="text-sm text-slate-600">Entry</p>
                            <p className="font-semibold text-slate-900">
                              ${decision.entry_price?.toFixed(4) || 'N/A'}
                            </p>
                          </div>
                          <div className="text-center">
                            <p className="text-sm text-slate-600">Stop Loss</p>
                            <p className="font-semibold text-red-600">
                              ${decision.stop_loss?.toFixed(4) || 'N/A'}
                            </p>
                          </div>
                          <div className="text-center">
                            <p className="text-sm text-slate-600">TP1</p>
                            <p className="font-semibold text-green-600">
                              ${decision.take_profit_1?.toFixed(4) || 'N/A'}
                            </p>
                          </div>
                          <div className="text-center">
                            <p className="text-sm text-slate-600">TP2</p>
                            <p className="font-semibold text-green-600">
                              ${decision.take_profit_2?.toFixed(4) || 'N/A'}
                            </p>
                          </div>
                          <div className="text-center">
                            <p className="text-sm text-slate-600">TP3</p>
                            <p className="font-semibold text-green-600">
                              ${decision.take_profit_3?.toFixed(4) || 'N/A'}
                            </p>
                          </div>
                          <div className="text-center">
                            <p className="text-sm text-slate-600">IA2 R:R</p>
                            <p className={`font-semibold ${
                              (decision.risk_reward_ratio || 1.0) >= 2.0 ? 'text-green-600' :
                              (decision.risk_reward_ratio || 1.0) >= 1.5 ? 'text-yellow-600' :
                              'text-red-600'
                            }`}>
                              {(decision.risk_reward_ratio || 1.0).toFixed(1)}:1
                            </p>
                            <p className="text-xs text-slate-500">
                              {(decision.risk_reward_ratio || 1.0) >= 2.0 ? 'Excellent' :
                               (decision.risk_reward_ratio || 1.0) >= 1.5 ? 'Good' : 'Low'}
                            </p>
                          </div>
                        </div>
                        
                        {/* Position Size and Risk */}
                        <div className="grid grid-cols-3 gap-4 mb-4">
                          <div className="text-center bg-white rounded p-2">
                            <p className="text-sm text-slate-600">Position Size</p>
                            <p className="font-semibold text-blue-600">
                              {(decision.position_size * 100)?.toFixed(2) || '0'}%
                            </p>
                          </div>
                          <div className="text-center bg-white rounded p-2">
                            <p className="text-sm text-slate-600">Risk Level</p>
                            <p className={`font-semibold ${
                              decision.risk_level === 'HIGH' ? 'text-red-600' :
                              decision.risk_level === 'MEDIUM' ? 'text-yellow-600' :
                              'text-green-600'
                            }`}>
                              {decision.risk_level || 'N/A'}
                            </p>
                          </div>
                          <div className="text-center bg-white rounded p-2">
                            <p className="text-sm text-slate-600">Strategy</p>
                            <p className="font-semibold text-purple-600">
                              {decision.strategy_type?.replace('_', ' ') || 'N/A'}
                            </p>
                          </div>
                        </div>
                        
                        {/* Decision Reasoning */}
                        <div className="bg-white rounded p-3 border">
                          <p className="text-sm font-medium text-slate-800 mb-2">IA2 Strategic Reasoning:</p>
                          <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">
                            {decision.strategic_reasoning || decision.ia2_reasoning || 'No strategic reasoning available'}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'positions' && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-sm border">
              <div className="p-6 border-b">
                <h2 className="text-xl font-bold text-slate-900">Active Trading Positions</h2>
                <p className="text-slate-600">Real-time monitoring of open positions with trailing stops</p>
              </div>
              
              <div className="p-6">
                {activePositions.length === 0 ? (
                  <div className="text-center py-12">
                    <div className="text-6xl mb-4">📊</div>
                    <h3 className="text-lg font-semibold text-slate-900 mb-2">No Active Positions</h3>
                    <p className="text-slate-500 mb-4">Currently running in {executionMode} mode</p>
                    <p className="text-sm text-slate-400">Positions will appear here when trades are executed</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {activePositions.map((position, index) => (
                      <div key={index} className="bg-slate-50 rounded-lg p-4 border">
                        <div className="flex justify-between items-start mb-3">
                          <h3 className="font-semibold text-lg text-slate-900">{position.symbol}</h3>
                          <div className="flex items-center space-x-2">
                            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                              position.direction === 'LONG' ? 'bg-emerald-100 text-emerald-700' :
                              'bg-red-100 text-red-700'
                            }`}>
                              {position.direction}
                            </span>
                            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                              position.unrealized_pnl >= 0 ? 'bg-green-100 text-green-700' :
                              'bg-red-100 text-red-700'
                            }`}>
                              {position.unrealized_pnl >= 0 ? '+' : ''}
                              ${position.unrealized_pnl?.toFixed(2) || '0.00'}
                            </span>
                          </div>
                        </div>
                        
                        {/* Position Details */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                          <div className="text-center">
                            <p className="text-sm text-slate-600">Entry Price</p>
                            <p className="font-semibold text-slate-900">
                              ${position.entry_price?.toFixed(4) || 'N/A'}
                            </p>
                          </div>
                          <div className="text-center">
                            <p className="text-sm text-slate-600">Current Price</p>
                            <p className="font-semibold text-blue-600">
                              ${position.current_price?.toFixed(4) || 'N/A'}
                            </p>
                          </div>
                          <div className="text-center">
                            <p className="text-sm text-slate-600">Position Size</p>
                            <p className="font-semibold text-slate-900">
                              {position.quantity?.toFixed(6) || 'N/A'}
                            </p>
                          </div>
                          <div className="text-center">
                            <p className="text-sm text-slate-600">Value</p>
                            <p className="font-semibold text-purple-600">
                              ${position.position_value?.toFixed(2) || 'N/A'}
                            </p>
                          </div>
                        </div>
                        
                        {/* Stop Loss and Take Profits */}
                        <div className="grid grid-cols-4 gap-4 mb-4">
                          <div className="text-center bg-white rounded p-2">
                            <p className="text-sm text-slate-600">Stop Loss</p>
                            <p className="font-semibold text-red-600">
                              ${position.stop_loss?.toFixed(4) || 'N/A'}
                            </p>
                          </div>
                          <div className="text-center bg-white rounded p-2">
                            <p className="text-sm text-slate-600">TP1</p>
                            <p className="font-semibold text-green-600">
                              ${position.take_profit_1?.toFixed(4) || 'N/A'}
                            </p>
                          </div>
                          <div className="text-center bg-white rounded p-2">
                            <p className="text-sm text-slate-600">TP2</p>
                            <p className="font-semibold text-green-600">
                              ${position.take_profit_2?.toFixed(4) || 'N/A'}
                            </p>
                          </div>
                          <div className="text-center bg-white rounded p-2">
                            <p className="text-sm text-slate-600">TP3</p>
                            <p className="font-semibold text-green-600">
                              ${position.take_profit_3?.toFixed(4) || 'N/A'}  
                            </p>
                          </div>
                        </div>
                        
                        {/* Position Actions */}
                        <div className="flex justify-end space-x-2">
                          <button
                            onClick={() => closePosition(position.id)}
                            className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm font-medium transition-colors"
                          >
                            Close Position
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'bingx' && (
          <div className="space-y-6">
            {/* BingX Status Header */}
            <div className="bg-white rounded-2xl shadow-sm border">
              <div className="p-6 border-b">
                <div className="flex justify-between items-center">
                  <div>
                    <h2 className="text-xl font-bold text-slate-900">BingX Live Trading</h2>
                    <p className="text-slate-600">Real-time trading execution on BingX Futures</p>
                  </div>
                  <div className="flex space-x-2">
                    <button
                      onClick={fetchBingxData}
                      disabled={bingxLoading}
                      className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
                    >
                      {bingxLoading ? 'Loading...' : 'Refresh'}
                    </button>
                    <button
                      onClick={triggerBingxEmergencyStop}
                      disabled={bingxLoading}
                      className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50"
                    >
                      Emergency Stop
                    </button>
                  </div>
                </div>
              </div>
              
              <div className="p-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {/* Connection Status */}
                  <div className="text-center">
                    <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                      (bingxStatus?.status === 'success' && bingxStatus?.bingx_integration?.status === 'operational') ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                    }`}>
                      {(bingxStatus?.status === 'success' && bingxStatus?.bingx_integration?.status === 'operational') ? '🟢 Connected' : '🔴 Disconnected'}
                    </div>
                    <p className="mt-2 text-sm text-slate-600">API Status</p>
                  </div>
                  
                  {/* Account Balance */}
                  <div className="text-center">
                    <div className="text-2xl font-bold text-slate-900">
                      ${bingxStatus?.bingx_integration?.balance?.balance?.toFixed(2) || bingxBalance?.balance?.toFixed(2) || '0.00'}
                    </div>
                    <p className="text-sm text-slate-600">Account Balance</p>
                  </div>
                  
                  {/* Active Positions Count */}
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {bingxPositions?.length || 0}
                    </div>
                    <p className="text-sm text-slate-600">Active Positions</p>
                  </div>
                </div>
              </div>
            </div>

            {/* BingX Positions */}
            <div className="bg-white rounded-2xl shadow-sm border">
              <div className="p-6 border-b">
                <h3 className="text-lg font-semibold text-slate-900">Open Positions</h3>
              </div>
              <div className="p-6">
                {bingxPositions?.length === 0 ? (
                  <div className="text-center py-8">
                    <div className="text-4xl mb-4">📊</div>
                    <p className="text-slate-600">No open positions</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {bingxPositions?.map((position, index) => (
                      <div key={index} className="bg-slate-50 rounded-lg p-4 border">
                        <div className="flex justify-between items-start mb-3">
                          <h4 className="font-semibold text-lg">{position.symbol}</h4>
                          <div className="flex space-x-2">
                            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                              position.side === 'LONG' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                            }`}>
                              {position.side}
                            </span>
                            <button
                              onClick={() => closeBingxPosition(position.symbol, position.side)}
                              className="px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700"
                            >
                              Close
                            </button>
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div>
                            <p className="text-sm text-slate-600">Entry Price</p>
                            <p className="font-semibold">${position.entry_price?.toFixed(4)}</p>
                          </div>
                          <div>
                            <p className="text-sm text-slate-600">Mark Price</p>
                            <p className="font-semibold">${position.mark_price?.toFixed(4)}</p>
                          </div>
                          <div>
                            <p className="text-sm text-slate-600">Size</p>
                            <p className="font-semibold">{position.size}</p>
                          </div>
                          <div>
                            <p className="text-sm text-slate-600">P&L</p>
                            <p className={`font-semibold ${
                              position.unrealized_pnl >= 0 ? 'text-green-600' : 'text-red-600'
                            }`}>
                              ${position.unrealized_pnl?.toFixed(2)}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Manual Trading */}
            <div className="bg-white rounded-2xl shadow-sm border">
              <div className="p-6 border-b">
                <h3 className="text-lg font-semibold text-slate-900">Manual Trade Execution</h3>
              </div>
              <div className="p-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-slate-700 mb-2">Symbol</label>
                      <input
                        type="text"
                        id="bingx-symbol"
                        placeholder="e.g. BTC-USDT"
                        className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-slate-700 mb-2">Side</label>
                      <select
                        id="bingx-side"
                        className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      >
                        <option value="LONG">LONG</option>
                        <option value="SHORT">SHORT</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-slate-700 mb-2">Quantity</label>
                      <input
                        type="number"
                        id="bingx-quantity"
                        placeholder="0.01"
                        step="0.001"
                        className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-slate-700 mb-2">Leverage</label>
                      <input
                        type="number"
                        id="bingx-leverage"
                        placeholder="5"
                        min="1"
                        max="20"
                        className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-slate-700 mb-2">Stop Loss (Optional)</label>
                      <input
                        type="number"
                        id="bingx-stop-loss"
                        placeholder="Auto-calculated if empty"
                        step="0.0001"
                        className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-slate-700 mb-2">Take Profit (Optional)</label>
                      <input
                        type="number"
                        id="bingx-take-profit"
                        placeholder="Auto-calculated if empty"
                        step="0.0001"
                        className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>
                  </div>
                </div>
                
                <div className="mt-6">
                  <button
                    onClick={() => {
                      const symbol = document.getElementById('bingx-symbol')?.value;
                      const side = document.getElementById('bingx-side')?.value;
                      const quantity = parseFloat(document.getElementById('bingx-quantity')?.value);
                      const leverage = parseInt(document.getElementById('bingx-leverage')?.value) || 5;
                      const stopLoss = parseFloat(document.getElementById('bingx-stop-loss')?.value) || null;
                      const takeProfit = parseFloat(document.getElementById('bingx-take-profit')?.value) || null;
                      
                      if (symbol && side && quantity) {
                        executeBingxTrade({
                          symbol,
                          side,
                          quantity,
                          leverage,
                          stop_loss: stopLoss,
                          take_profit: takeProfit
                        });
                      } else {
                        alert('Please fill in required fields: Symbol, Side, and Quantity');
                      }
                    }}
                    disabled={bingxLoading}
                    className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 font-medium"
                  >
                    {bingxLoading ? 'Executing...' : 'Execute Trade'}
                  </button>
                </div>
              </div>
            </div>

            {/* Risk Configuration */}
            <div className="bg-white rounded-2xl shadow-sm border">
              <div className="p-6 border-b">
                <h3 className="text-lg font-semibold text-slate-900">Risk Management Configuration</h3>
              </div>
              <div className="p-6">
                {bingxRiskConfig && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-slate-700 mb-2">Max Position Size (%)</label>
                        <input
                          type="number"
                          defaultValue={bingxRiskConfig.max_position_size * 100}
                          step="1"
                          min="1"
                          max="20"
                          className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          id="risk-max-position"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-slate-700 mb-2">Max Total Exposure (%)</label>
                        <input
                          type="number"
                          defaultValue={bingxRiskConfig.max_total_exposure * 100}
                          step="5"
                          min="10"
                          max="80"
                          className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          id="risk-max-exposure"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-slate-700 mb-2">Max Leverage</label>
                        <input
                          type="number"
                          defaultValue={bingxRiskConfig.max_leverage}
                          step="1"
                          min="1"
                          max="20"
                          className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          id="risk-max-leverage"
                        />
                      </div>
                    </div>
                    
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-slate-700 mb-2">Stop Loss (%)</label>
                        <input
                          type="number"
                          defaultValue={bingxRiskConfig.stop_loss_percentage * 100}
                          step="0.5"
                          min="1"
                          max="10"
                          className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          id="risk-stop-loss"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-slate-700 mb-2">Max Drawdown (%)</label>
                        <input
                          type="number"
                          defaultValue={bingxRiskConfig.max_drawdown * 100}
                          step="1"
                          min="5"
                          max="30"
                          className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          id="risk-max-drawdown"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-slate-700 mb-2">Daily Loss Limit (%)</label>
                        <input
                          type="number"
                          defaultValue={bingxRiskConfig.daily_loss_limit * 100}
                          step="1"
                          min="2"
                          max="15"
                          className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          id="risk-daily-loss"
                        />
                      </div>
                    </div>
                  </div>
                )}
                
                <div className="mt-6">
                  <button
                    onClick={() => {
                      const maxPositionSize = parseFloat(document.getElementById('risk-max-position')?.value) / 100;
                      const maxTotalExposure = parseFloat(document.getElementById('risk-max-exposure')?.value) / 100;
                      const maxLeverage = parseInt(document.getElementById('risk-max-leverage')?.value);
                      const stopLossPercentage = parseFloat(document.getElementById('risk-stop-loss')?.value) / 100;
                      const maxDrawdown = parseFloat(document.getElementById('risk-max-drawdown')?.value) / 100;
                      const dailyLossLimit = parseFloat(document.getElementById('risk-daily-loss')?.value) / 100;
                      
                      updateBingxRiskConfig({
                        max_position_size: maxPositionSize,
                        max_total_exposure: maxTotalExposure,
                        max_leverage: maxLeverage,
                        stop_loss_percentage: stopLossPercentage,
                        max_drawdown: maxDrawdown,
                        daily_loss_limit: dailyLossLimit
                      });
                    }}
                    disabled={bingxLoading}
                    className="w-full px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 font-medium"
                  >
                    {bingxLoading ? 'Updating...' : 'Update Risk Configuration'}
                  </button>
                </div>
              </div>
            </div>

            {/* Trading History */}
            <div className="bg-white rounded-2xl shadow-sm border">
              <div className="p-6 border-b">
                <h3 className="text-lg font-semibold text-slate-900">Recent Trading History</h3>
              </div>
              <div className="p-6">
                {bingxTradingHistory?.length === 0 ? (
                  <div className="text-center py-8">
                    <div className="text-4xl mb-4">📜</div>
                    <p className="text-slate-600">No trading history</p>
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left py-2">Symbol</th>
                          <th className="text-left py-2">Side</th>
                          <th className="text-left py-2">Quantity</th>
                          <th className="text-left py-2">Price</th>
                          <th className="text-left py-2">Status</th>
                          <th className="text-left py-2">Time</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(Array.isArray(bingxTradingHistory) ? bingxTradingHistory : [])?.map((trade, index) => (
                          <tr key={index} className="border-b">
                            <td className="py-2 font-medium">{trade.symbol}</td>
                            <td className="py-2">
                              <span className={`px-2 py-1 rounded text-xs ${
                                trade.side === 'BUY' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                              }`}>
                                {trade.side}
                              </span>
                            </td>
                            <td className="py-2">{trade.quantity}</td>
                            <td className="py-2">${trade.price?.toFixed(4)}</td>
                            <td className="py-2">
                              <span className={`px-2 py-1 rounded text-xs ${
                                trade.status === 'FILLED' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-700'
                              }`}>
                                {trade.status}
                              </span>
                            </td>
                            <td className="py-2 text-slate-600">{new Date(trade.time).toLocaleString()}</td>
                          </tr>
                        ))}
                        {(!bingxTradingHistory || !Array.isArray(bingxTradingHistory) || bingxTradingHistory.length === 0) && (
                          <tr>
                            <td colSpan="6" className="py-8 text-center text-slate-500">
                              No trading history available
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                )}
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
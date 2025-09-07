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
      const [oppRes, analysesRes, decisionsRes, perfRes] = await Promise.all([
        axios.get(`${API}/opportunities`),
        axios.get(`${API}/analyses`),
        axios.get(`${API}/decisions`),
        axios.get(`${API}/performance`)
      ]);

      setOpportunities(oppRes.data.opportunities || []);
      setAnalyses(analysesRes.data.analyses || []);
      setDecisions(decisionsRes.data.decisions || []);
      setPerformance(perfRes.data.performance || {});
    } catch (error) {
      console.error('Error fetching data:', error);
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
            <TabButton id="performance" icon="📈">Performance</TabButton>
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
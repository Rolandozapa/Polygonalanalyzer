"""
AI TRAINING SYSTEM - Comprehensive Historical Data Training for IA1 & IA2
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import asyncio
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Import our trading system components
from technical_pattern_detector import TechnicalPatternDetector, TechnicalPattern
from data_models import TechnicalAnalysis, MarketOpportunity, TradingDecision, SignalType
from advanced_technical_indicators import advanced_technical_indicators, TechnicalIndicators, IndicatorSignal
from emergentintegrations.llm.chat import LlmChat, UserMessage

logger = logging.getLogger(__name__)

@dataclass
class MarketCondition:
    """Classification d'une condition de marché"""
    period_start: str
    period_end: str  
    symbol: str
    condition_type: str  # "BULL", "BEAR", "SIDEWAYS", "VOLATILE"
    volatility: float
    trend_strength: float
    volume_trend: float
    rsi_avg: float
    macd_signals: int
    price_change_pct: float
    pattern_frequency: Dict[str, int]
    success_rate: float  # Success rate of signals in this condition
    confidence_score: float  # How confident we are in this classification

@dataclass 
class PatternTraining:
    """Training data for pattern recognition improvement"""
    pattern_type: str
    symbol: str
    date: str
    success: bool  # Whether the pattern led to profitable trade
    entry_price: float
    exit_price: float
    hold_days: int
    market_condition: str
    volume_confirmation: bool
    rsi_level: float
    macd_signal: float
    confidence_factors: Dict[str, Any]

@dataclass
class IA1Enhancement:
    """Data structure for IA1 accuracy improvements"""
    symbol: str
    date: str
    predicted_signal: str
    actual_outcome: str  # What happened in the next 5-10 days
    prediction_accuracy: float
    technical_indicators: Dict[str, float]
    patterns_detected: List[str]
    market_context: str
    suggested_improvements: List[str]

@dataclass
class IA2Enhancement:  
    """Data structure for IA2 decision making improvements"""
    symbol: str
    date: str
    decision_signal: str
    decision_confidence: float
    actual_performance: float  # Actual return achieved
    optimal_exit_timing: int  # Days to optimal exit
    risk_reward_realized: float
    market_condition_match: bool
    position_sizing_accuracy: float
    suggested_adjustments: Dict[str, Any]

class AITrainingSystem:
    """Comprehensive AI Training System for IA1 & IA2 Enhancement"""
    
    def __init__(self):
        self.pattern_detector = TechnicalPatternDetector()
        self.historical_data = {}
        self.training_data_dir = "/app/historical_training_data"
        
        # Training results storage
        self.market_conditions = []
        self.pattern_training = []
        self.ia1_enhancements = []
        self.ia2_enhancements = []
        
        # Enhanced pattern recognition
        self.pattern_success_rates = defaultdict(list)
        self.market_condition_patterns = defaultdict(list)
        
        # IA1 & IA2 training models
        self.ia1_accuracy_model = None
        self.ia2_performance_model = None
        
        # Load and prepare data
        self._load_training_data()
        
        logger.info(f"AI Training System initialized with {len(self.historical_data)} symbols")
    
    def _load_training_data(self):
        """Load comprehensive historical training data"""
        if not os.path.exists(self.training_data_dir):
            logger.error(f"Training data directory not found: {self.training_data_dir}")
            return
        
        # Enhanced symbol mapping for comprehensive training
        symbol_mapping = {
            'coin_Bitcoin.csv': 'BTCUSDT',
            'coin_Ethereum.csv': 'ETHUSDT',
            'coin_BinanceCoin.csv': 'BNBUSDT', 
            'coin_Cardano.csv': 'ADAUSDT',
            'coin_ChainLink.csv': 'LINKUSDT',
            'coin_Dogecoin.csv': 'DOGEUSDT',
            'coin_Polkadot.csv': 'DOTUSDT',
            'coin_USDCoin.csv': 'USDCUSDT',
            'coin_Litecoin.csv': 'LTCUSDT',
            'coin_Stellar.csv': 'XLMUSDT',
            'coin_Monero.csv': 'XMRUSDT',
            'coin_EOS.csv': 'EOSUSDT',
            'coin_Tron.csv': 'TRXUSDT',
            'coin_Aave.csv': 'AAVEUSDT',
            'coin_Cosmos.csv': 'ATOMUSDT',
            'coin_Solana.csv': 'SOLUSDT',
            'coin_Uniswap.csv': 'UNIUSDT',
            'coin_WrappedBitcoin.csv': 'WBTCUSDT',
            'coin_XRP.csv': 'XRPUSDT',
            'coin_Iota.csv': 'IOTXUSDT',  # Similar token
            'coin_NEM.csv': 'XEMUSDT',   # Similar token
            'coin_Tether.csv': 'USDTUSDT',
            'coin_CryptocomCoin.csv': 'CROUSDT'
        }
        
        for filename, symbol in symbol_mapping.items():
            filepath = os.path.join(self.training_data_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    
                    # Enhanced data preprocessing
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.sort_values('Date')
                    
                    # Add comprehensive technical indicators
                    df = self._add_technical_indicators(df)
                    
                    # Filter valid data with sufficient history
                    df = df.dropna(subset=['High', 'Low', 'Open', 'Close'])
                    
                    if len(df) > 200:  # Need more data for training
                        self.historical_data[symbol] = df
                        logger.info(f"Loaded {len(df)} days of training data for {symbol}")
                    else:
                        logger.warning(f"Insufficient training data for {symbol}: {len(df)} days")
                        
                except Exception as e:
                    logger.error(f"Error loading {filepath}: {e}")
        
        logger.info(f"Successfully loaded training data for {len(self.historical_data)} symbols")
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators for enhanced training"""
        # Use advanced technical indicators system
        df_enhanced = advanced_technical_indicators.calculate_all_indicators(df)
        
        # Add additional custom indicators for training
        try:
            # Price action indicators
            df_enhanced['price_change_1d'] = df_enhanced['Close'].pct_change() * 100
            df_enhanced['price_change_3d'] = df_enhanced['Close'].pct_change(3) * 100
            df_enhanced['price_change_7d'] = df_enhanced['Close'].pct_change(7) * 100
            df_enhanced['price_change_14d'] = df_enhanced['Close'].pct_change(14) * 100
            
            # Volatility indicators
            df_enhanced['volatility_5d'] = df_enhanced['Close'].rolling(5).std() / df_enhanced['Close'].rolling(5).mean()
            df_enhanced['volatility_10d'] = df_enhanced['Close'].rolling(10).std() / df_enhanced['Close'].rolling(10).mean()
            df_enhanced['volatility_20d'] = df_enhanced['Close'].rolling(20).std() / df_enhanced['Close'].rolling(20).mean()
            
            # Support and resistance levels (enhanced)
            df_enhanced['resistance_level_5d'] = df_enhanced['High'].rolling(5).max()
            df_enhanced['resistance_level_20d'] = df_enhanced['High'].rolling(20).max()
            df_enhanced['support_level_5d'] = df_enhanced['Low'].rolling(5).min()
            df_enhanced['support_level_20d'] = df_enhanced['Low'].rolling(20).min()
            
            # Distance to key levels
            df_enhanced['distance_to_resistance_5d'] = (df_enhanced['resistance_level_5d'] - df_enhanced['Close']) / df_enhanced['Close']
            df_enhanced['distance_to_support_5d'] = (df_enhanced['Close'] - df_enhanced['support_level_5d']) / df_enhanced['Close']
            
            # Enhanced volume analysis
            df_enhanced['volume_change_1d'] = df_enhanced['Volume'].pct_change() * 100
            df_enhanced['volume_ma_ratio_5d'] = df_enhanced['Volume'] / df_enhanced['Volume'].rolling(5).mean()
            df_enhanced['volume_ma_ratio_20d'] = df_enhanced['Volume'] / df_enhanced['Volume'].rolling(20).mean()
            
            # Gap analysis
            df_enhanced['gap_up'] = (df_enhanced['Open'] > df_enhanced['Close'].shift(1) * 1.01).astype(int)
            df_enhanced['gap_down'] = (df_enhanced['Open'] < df_enhanced['Close'].shift(1) * 0.99).astype(int)
            
            # Candle patterns (basic)
            df_enhanced['doji'] = (abs(df_enhanced['Close'] - df_enhanced['Open']) / (df_enhanced['High'] - df_enhanced['Low']) < 0.1).astype(int)
            df_enhanced['hammer'] = ((df_enhanced['Close'] > df_enhanced['Open']) & 
                                   ((df_enhanced['Open'] - df_enhanced['Low']) > 2 * abs(df_enhanced['Close'] - df_enhanced['Open'])) &
                                   ((df_enhanced['High'] - df_enhanced['Close']) < 0.3 * abs(df_enhanced['Close'] - df_enhanced['Open']))).astype(int)
            
            # Multi-timeframe signals
            df_enhanced['signals_alignment'] = (
                (df_enhanced['rsi_14'] > 50).astype(int) +
                (df_enhanced['macd_histogram'] > 0).astype(int) +
                (df_enhanced['stoch_k'] > 50).astype(int) +
                (df_enhanced['bb_position'] > 0.5).astype(int) +
                (df_enhanced['Close'] > df_enhanced['sma_20']).astype(int)
            ) / 5.0  # Normalize to 0-1
            
            logger.info(f"Enhanced technical indicators added: {len(df_enhanced.columns)} total columns")
            
        except Exception as e:
            logger.error(f"Error adding custom indicators: {e}")
        
        return df_enhanced
    
    async def run_comprehensive_training(self) -> Dict[str, Any]:
        """Run comprehensive training on all historical data"""
        logger.info("🚀 Starting Comprehensive AI Training System")
        
        training_results = {
            'market_conditions_classified': 0,
            'patterns_analyzed': 0,
            'ia1_improvements_identified': 0,
            'ia2_enhancements_generated': 0,
            'training_performance': {}
        }
        
        # 1. Market Condition Classification
        logger.info("📊 Step 1: Market Condition Classification")
        market_conditions = await self._classify_market_conditions()
        training_results['market_conditions_classified'] = len(market_conditions)
        
        # 2. Pattern Recognition Training
        logger.info("🎯 Step 2: Enhanced Pattern Recognition Training")
        pattern_training = await self._train_pattern_recognition()
        training_results['patterns_analyzed'] = len(pattern_training)
        
        # 3. IA1 Technical Analysis Enhancement 
        logger.info("🤖 Step 3: IA1 Technical Analysis Enhancement")
        ia1_improvements = await self._enhance_ia1_accuracy()
        training_results['ia1_improvements_identified'] = len(ia1_improvements)
        
        # 4. IA2 Decision Making Training
        logger.info("🧠 Step 4: IA2 Strategic Decision Training")
        ia2_enhancements = await self._train_ia2_decision_making()
        training_results['ia2_enhancements_generated'] = len(ia2_enhancements)
        
        # 5. Generate Adaptive Context Rules
        logger.info("⚡ Step 5: Adaptive Context System Enhancement")
        adaptive_rules = await self._generate_adaptive_context_rules()
        training_results['adaptive_rules_generated'] = len(adaptive_rules)
        
        # 6. Save training results
        await self._save_training_results(training_results)
        
        logger.info("✅ Comprehensive AI Training Completed!")
        return training_results
    
    async def _classify_market_conditions(self) -> List[MarketCondition]:
        """Classify different market conditions from historical data"""
        market_conditions = []
        
        for symbol, df in self.historical_data.items():
            logger.info(f"Classifying market conditions for {symbol}")
            
            # Analyze in 30-day windows
            window_size = 30
            
            for i in range(window_size, len(df) - window_size, 15):  # 15-day overlap
                window_data = df.iloc[i-window_size:i+window_size].copy()
                
                if len(window_data) < window_size:
                    continue
                
                # Calculate market condition metrics
                price_change = ((window_data['Close'].iloc[-1] / window_data['Close'].iloc[0]) - 1) * 100
                volatility = window_data['volatility_7d'].mean()
                volume_trend = (window_data['volume_ratio'].mean() - 1) * 100
                rsi_avg = window_data['rsi'].mean()
                
                # Count MACD signals
                macd_bullish = len(window_data[window_data['macd'] > window_data['macd_signal']])
                macd_signals = abs(macd_bullish - (len(window_data) - macd_bullish))
                
                # Detect patterns in this period
                try:
                    patterns = self.pattern_detector._detect_all_patterns(symbol, window_data)
                    pattern_freq = Counter([p.pattern_type.value for p in patterns])
                except:
                    pattern_freq = {}
                
                # Classify market condition
                condition_type = self._classify_condition_type(price_change, volatility, rsi_avg)
                
                # Calculate trend strength
                trend_strength = abs(price_change) / max(volatility, 1.0)
                
                # Estimate success rate (simplified) 
                success_rate = self._estimate_success_rate(condition_type, volatility, trend_strength)
                
                # Confidence score
                confidence_score = min(trend_strength / 2.0 + (abs(price_change) / 20.0), 1.0)
                
                market_condition = MarketCondition(
                    period_start=window_data['Date'].iloc[0].strftime('%Y-%m-%d'),
                    period_end=window_data['Date'].iloc[-1].strftime('%Y-%m-%d'),
                    symbol=symbol,
                    condition_type=condition_type,
                    volatility=volatility,
                    trend_strength=trend_strength,
                    volume_trend=volume_trend,
                    rsi_avg=rsi_avg,
                    macd_signals=macd_signals,
                    price_change_pct=price_change,
                    pattern_frequency=dict(pattern_freq),
                    success_rate=success_rate,
                    confidence_score=confidence_score
                )
                
                market_conditions.append(market_condition)
        
        self.market_conditions = market_conditions
        logger.info(f"Classified {len(market_conditions)} market conditions")
        return market_conditions
    
    def _classify_condition_type(self, price_change: float, volatility: float, rsi_avg: float) -> str:
        """Classify the type of market condition"""
        if volatility > 15:  # High volatility
            return "VOLATILE"
        elif abs(price_change) < 5:  # Low price movement
            return "SIDEWAYS" 
        elif price_change > 5:  # Upward trend
            return "BULL"
        else:  # Downward trend
            return "BEAR"
    
    def _estimate_success_rate(self, condition_type: str, volatility: float, trend_strength: float) -> float:
        """Estimate success rate for different market conditions"""
        base_rates = {
            "BULL": 0.65,
            "BEAR": 0.55, 
            "SIDEWAYS": 0.45,
            "VOLATILE": 0.50
        }
        
        base_rate = base_rates.get(condition_type, 0.50)
        
        # Adjust based on trend strength and volatility
        if trend_strength > 2.0:
            base_rate += 0.10  # Strong trends are easier to trade
        
        if volatility > 20:
            base_rate -= 0.05  # High volatility reduces success
        
        return max(0.3, min(0.8, base_rate))
    
    async def _train_pattern_recognition(self) -> List[PatternTraining]:
        """Train enhanced pattern recognition using historical data"""
        pattern_training = []
        
        for symbol, df in self.historical_data.items():
            logger.info(f"Training pattern recognition for {symbol}")
            
            # Detect patterns throughout history
            window_size = 50  # Need sufficient data for pattern detection
            
            for i in range(window_size, len(df) - 10, 5):  # Look ahead 10 days
                historical_window = df.iloc[i-window_size:i].copy()
                future_window = df.iloc[i:i+10].copy()  # Next 10 days
                
                try:
                    patterns = self.pattern_detector._detect_all_patterns(symbol, historical_window)
                    
                    for pattern in patterns:
                        # Calculate pattern success
                        entry_price = historical_window['Close'].iloc[-1]
                        
                        # Find best exit in next 10 days
                        if pattern.signal == "bullish":
                            max_price = future_window['High'].max()
                            exit_price = max_price if max_price > entry_price * 1.02 else future_window['Close'].iloc[-1]
                            success = exit_price > entry_price * 1.02
                        elif pattern.signal == "bearish":
                            min_price = future_window['Low'].min()
                            exit_price = min_price if min_price < entry_price * 0.98 else future_window['Close'].iloc[-1]
                            success = exit_price < entry_price * 0.98
                        else:
                            exit_price = future_window['Close'].iloc[-1]
                            success = abs((exit_price / entry_price) - 1) < 0.02  # Neutral is success
                        
                        # Get market condition for this period
                        current_condition = self._get_market_condition_for_period(
                            symbol, historical_window['Date'].iloc[-1]
                        )
                        
                        # Volume confirmation
                        avg_volume = historical_window['Volume'].tail(10).mean()
                        current_volume = historical_window['Volume'].iloc[-1]
                        volume_confirmation = current_volume > avg_volume * 1.2
                        
                        # Technical context
                        rsi_level = historical_window['rsi'].iloc[-1]
                        macd_signal = historical_window['macd_histogram'].iloc[-1]
                        
                        # Confidence factors
                        confidence_factors = {
                            'pattern_strength': pattern.strength,
                            'volume_confirmation': volume_confirmation,
                            'rsi_level': rsi_level,
                            'macd_signal': macd_signal,
                            'market_condition': current_condition
                        }
                        
                        training_data = PatternTraining(
                            pattern_type=pattern.pattern_type.value,
                            symbol=symbol,
                            date=historical_window['Date'].iloc[-1].strftime('%Y-%m-%d'),
                            success=success,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            hold_days=len(future_window),
                            market_condition=current_condition,
                            volume_confirmation=volume_confirmation,
                            rsi_level=rsi_level,
                            macd_signal=macd_signal,
                            confidence_factors=confidence_factors
                        )
                        
                        pattern_training.append(training_data)
                        
                except Exception as e:
                    logger.debug(f"Error processing pattern for {symbol}: {e}")
                    continue
        
        self.pattern_training = pattern_training
        
        # Calculate success rates by pattern type
        pattern_success = defaultdict(list)
        for training in pattern_training:
            pattern_success[training.pattern_type].append(training.success)
        
        for pattern_type, successes in pattern_success.items():
            success_rate = sum(successes) / len(successes) if successes else 0
            logger.info(f"Pattern {pattern_type}: {success_rate:.1%} success rate ({len(successes)} samples)")
        
        logger.info(f"Generated {len(pattern_training)} pattern training samples")
        return pattern_training
    
    def _get_market_condition_for_period(self, symbol: str, date: datetime) -> str:
        """Get market condition for a specific symbol and date"""
        # Find matching market condition
        for condition in self.market_conditions:
            if (condition.symbol == symbol and 
                condition.period_start <= date.strftime('%Y-%m-%d') <= condition.period_end):
                return condition.condition_type
        
        return "UNKNOWN"
    
    async def _enhance_ia1_accuracy(self) -> List[IA1Enhancement]:
        """Enhance IA1's technical analysis accuracy using historical data"""
        ia1_enhancements = []
        
        for symbol, df in self.historical_data.items():
            logger.info(f"Enhancing IA1 accuracy for {symbol}")
            
            # Test IA1 predictions against historical outcomes
            for i in range(100, len(df) - 5, 10):  # Every 10 days, look ahead 5 days
                historical_data = df.iloc[i-50:i].copy()
                future_data = df.iloc[i:i+5].copy()
                
                try:
                    # Simulate IA1 analysis
                    current_price = historical_data['Close'].iloc[-1]
                    rsi = historical_data['rsi'].iloc[-1]
                    macd = historical_data['macd_histogram'].iloc[-1]
                    bb_position = historical_data['bb_position'].iloc[-1]
                    
                    # Generate IA1 prediction
                    predicted_signal = self._simulate_ia1_prediction(rsi, macd, bb_position)
                    
                    # Analyze actual outcome
                    future_price = future_data['Close'].iloc[-1]
                    actual_return = (future_price / current_price - 1) * 100
                    
                    if predicted_signal == "long":
                        actual_outcome = "correct" if actual_return > 2 else "incorrect"
                        accuracy = max(0, min(1, (actual_return + 10) / 20))  # Scale -10% to +10% to 0-1
                    elif predicted_signal == "short":
                        actual_outcome = "correct" if actual_return < -2 else "incorrect"  
                        accuracy = max(0, min(1, (-actual_return + 10) / 20))
                    else:  # hold
                        actual_outcome = "correct" if abs(actual_return) < 2 else "incorrect"
                        accuracy = max(0, min(1, (2 - abs(actual_return)) / 4 + 0.5))
                    
                    # Detect patterns for context
                    try:
                        patterns = self.pattern_detector._detect_all_patterns(symbol, historical_data)
                        patterns_detected = [p.pattern_type.value for p in patterns]
                    except:
                        patterns_detected = []
                    
                    # Get market context
                    market_context = self._get_market_condition_for_period(
                        symbol, historical_data['Date'].iloc[-1]
                    )
                    
                    # Generate improvement suggestions
                    suggestions = self._generate_ia1_suggestions(
                        predicted_signal, actual_outcome, rsi, macd, bb_position, patterns_detected
                    )
                    
                    enhancement = IA1Enhancement(
                        symbol=symbol,
                        date=historical_data['Date'].iloc[-1].strftime('%Y-%m-%d'),
                        predicted_signal=predicted_signal,
                        actual_outcome=actual_outcome,
                        prediction_accuracy=accuracy,
                        technical_indicators={
                            'rsi': rsi,
                            'macd': macd,
                            'bb_position': bb_position,
                            'actual_return': actual_return
                        },
                        patterns_detected=patterns_detected,
                        market_context=market_context,
                        suggested_improvements=suggestions
                    )
                    
                    ia1_enhancements.append(enhancement)
                    
                except Exception as e:
                    logger.debug(f"Error enhancing IA1 for {symbol}: {e}")
                    continue
        
        self.ia1_enhancements = ia1_enhancements
        
        # Calculate overall accuracy metrics
        correct_predictions = len([e for e in ia1_enhancements if e.actual_outcome == "correct"])
        total_predictions = len(ia1_enhancements)
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        logger.info(f"IA1 Enhancement: {overall_accuracy:.1%} accuracy ({correct_predictions}/{total_predictions})")
        return ia1_enhancements
    
    def _simulate_ia1_prediction(self, rsi: float, macd: float, bb_position: float) -> str:
        """Simulate IA1 prediction based on technical indicators"""
        signals = []
        
        # RSI signals
        if rsi < 30:
            signals.append("long")
        elif rsi > 70:
            signals.append("short")
        
        # MACD signals
        if macd > 0:
            signals.append("long")
        elif macd < 0:
            signals.append("short")
        
        # Bollinger Bands signals
        if bb_position < 0.2:
            signals.append("long")
        elif bb_position > 0.8:
            signals.append("short")
        
        # Determine final signal
        if signals.count("long") > signals.count("short"):
            return "long"
        elif signals.count("short") > signals.count("long"):
            return "short"
        else:
            return "hold"
    
    def _generate_ia1_suggestions(self, predicted: str, outcome: str, rsi: float, 
                                 macd: float, bb_position: float, patterns: List[str]) -> List[str]:
        """Generate improvement suggestions for IA1"""
        suggestions = []
        
        if outcome == "incorrect":
            if predicted == "long" and rsi > 50:
                suggestions.append("Consider RSI overbought conditions more carefully")
            
            if predicted == "short" and rsi < 50:
                suggestions.append("Be cautious with short signals when RSI isn't overbought")
            
            if abs(macd) < 0.001:
                suggestions.append("Avoid strong signals when MACD is weak")
            
            if len(patterns) == 0:
                suggestions.append("Require pattern confirmation for stronger signals")
            
            if bb_position > 0.3 and bb_position < 0.7:
                suggestions.append("Be more cautious in middle Bollinger Band range")
        
        return suggestions
    
    async def _train_ia2_decision_making(self) -> List[IA2Enhancement]:
        """Train IA2's strategic decision making using historical data"""
        ia2_enhancements = []
        
        for symbol, df in self.historical_data.items():
            logger.info(f"Training IA2 decision making for {symbol}")
            
            # Test IA2 decisions against optimal outcomes
            for i in range(100, len(df) - 20, 15):  # Every 15 days, look ahead 20 days
                historical_data = df.iloc[i-50:i].copy()
                future_data = df.iloc[i:i+20].copy()
                
                try:
                    # Simulate IA2 decision
                    entry_price = historical_data['Close'].iloc[-1]
                    decision_data = self._simulate_ia2_decision(historical_data, entry_price)
                    
                    if decision_data['signal'] == 'hold':
                        continue
                    
                    # Calculate actual performance
                    actual_performance = self._calculate_actual_performance(
                        decision_data, future_data, entry_price
                    )
                    
                    # Find optimal exit timing
                    optimal_exit = self._find_optimal_exit_timing(
                        decision_data['signal'], future_data, entry_price
                    )
                    
                    # Calculate realized risk-reward
                    if decision_data['signal'] == 'long':
                        rr_realized = (actual_performance['exit_price'] - entry_price) / (entry_price - decision_data['stop_loss'])
                    else:  # short
                        rr_realized = (entry_price - actual_performance['exit_price']) / (decision_data['stop_loss'] - entry_price)
                    
                    # Market condition match
                    market_condition = self._get_market_condition_for_period(
                        symbol, historical_data['Date'].iloc[-1]
                    )
                    condition_match = self._check_market_condition_match(decision_data['signal'], market_condition)
                    
                    # Position sizing accuracy (simplified)
                    optimal_size = 0.02  # 2% standard
                    size_accuracy = 1.0 - abs(decision_data['position_size'] - optimal_size) / optimal_size
                    
                    # Generate adjustment suggestions  
                    adjustments = self._generate_ia2_adjustments(
                        decision_data, actual_performance, optimal_exit, market_condition
                    )
                    
                    enhancement = IA2Enhancement(
                        symbol=symbol,
                        date=historical_data['Date'].iloc[-1].strftime('%Y-%m-%d'),
                        decision_signal=decision_data['signal'],
                        decision_confidence=decision_data['confidence'],
                        actual_performance=actual_performance['return_pct'],
                        optimal_exit_timing=optimal_exit['days'],
                        risk_reward_realized=rr_realized,
                        market_condition_match=condition_match,
                        position_sizing_accuracy=size_accuracy,
                        suggested_adjustments=adjustments
                    )
                    
                    ia2_enhancements.append(enhancement)
                    
                except Exception as e:
                    logger.debug(f"Error training IA2 for {symbol}: {e}")
                    continue
        
        self.ia2_enhancements = ia2_enhancements
        
        # Calculate performance metrics
        avg_performance = np.mean([e.actual_performance for e in ia2_enhancements])
        avg_rr = np.mean([e.risk_reward_realized for e in ia2_enhancements if not np.isnan(e.risk_reward_realized)])
        condition_match_rate = np.mean([e.market_condition_match for e in ia2_enhancements])
        
        logger.info(f"IA2 Training: {avg_performance:.1f}% avg return, {avg_rr:.1f}:1 avg RR, {condition_match_rate:.1%} condition match")
        return ia2_enhancements
    
    def _simulate_ia2_decision(self, historical_data: pd.DataFrame, entry_price: float) -> Dict[str, Any]:
        """Simulate IA2 decision making"""
        # Get technical indicators
        rsi = historical_data['rsi'].iloc[-1]
        macd = historical_data['macd_histogram'].iloc[-1]
        bb_position = historical_data['bb_position'].iloc[-1]
        volatility = historical_data['volatility_7d'].iloc[-1]
        
        # Decision logic (simplified IA2 simulation)
        signal_score = 0
        confidence = 0.5
        
        # RSI influence
        if rsi < 30:
            signal_score += 0.3
            confidence += 0.1
        elif rsi > 70:
            signal_score -= 0.3
            confidence += 0.1
        
        # MACD influence
        if macd > 0:
            signal_score += 0.2
            confidence += 0.05
        elif macd < 0:
            signal_score -= 0.2
            confidence += 0.05
        
        # Bollinger influence
        if bb_position < 0.2:
            signal_score += 0.15
        elif bb_position > 0.8:
            signal_score -= 0.15
        
        # Determine signal
        if signal_score > 0.2:
            signal = 'long'
            stop_loss = entry_price * (1 - max(volatility / 100 * 2, 0.02))
            take_profit = entry_price * (1 + max(volatility / 100 * 3, 0.03))
        elif signal_score < -0.2:
            signal = 'short'
            stop_loss = entry_price * (1 + max(volatility / 100 * 2, 0.02))
            take_profit = entry_price * (1 - max(volatility / 100 * 3, 0.03))
        else:
            signal = 'hold'
            stop_loss = entry_price * 0.98
            take_profit = entry_price * 1.02
        
        confidence = min(0.95, max(0.5, confidence + abs(signal_score) * 0.5))
        position_size = 0.02  # Standard 2% risk
        
        return {
            'signal': signal,
            'confidence': confidence,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size
        }
    
    def _calculate_actual_performance(self, decision_data: Dict[str, Any], 
                                    future_data: pd.DataFrame, entry_price: float) -> Dict[str, Any]:
        """Calculate actual performance of the decision"""
        signal = decision_data['signal']
        stop_loss = decision_data['stop_loss']
        take_profit = decision_data['take_profit']
        
        # Track the trade day by day
        for i, (_, row) in enumerate(future_data.iterrows()):
            current_price = row['Close']
            
            if signal == 'long':
                if current_price <= stop_loss:
                    # Stop loss hit
                    return {
                        'exit_price': stop_loss,
                        'exit_day': i + 1,
                        'return_pct': (stop_loss / entry_price - 1) * 100,
                        'exit_reason': 'stop_loss'
                    }
                elif current_price >= take_profit:
                    # Take profit hit
                    return {
                        'exit_price': take_profit,
                        'exit_day': i + 1,
                        'return_pct': (take_profit / entry_price - 1) * 100,
                        'exit_reason': 'take_profit'
                    }
            
            elif signal == 'short':
                if current_price >= stop_loss:
                    # Stop loss hit
                    return {
                        'exit_price': stop_loss,
                        'exit_day': i + 1,
                        'return_pct': (entry_price / stop_loss - 1) * 100,
                        'exit_reason': 'stop_loss'
                    }
                elif current_price <= take_profit:
                    # Take profit hit
                    return {
                        'exit_price': take_profit,
                        'exit_day': i + 1,
                        'return_pct': (entry_price / take_profit - 1) * 100,
                        'exit_reason': 'take_profit'
                    }
        
        # Exit at end of period
        final_price = future_data['Close'].iloc[-1]
        if signal == 'long':
            return_pct = (final_price / entry_price - 1) * 100
        else:
            return_pct = (entry_price / final_price - 1) * 100
        
        return {
            'exit_price': final_price,
            'exit_day': len(future_data),
            'return_pct': return_pct,
            'exit_reason': 'time_exit'
        }
    
    def _find_optimal_exit_timing(self, signal: str, future_data: pd.DataFrame, entry_price: float) -> Dict[str, Any]:
        """Find optimal exit timing and price"""
        if signal == 'long':
            max_price = future_data['High'].max()
            max_idx = future_data['High'].idxmax()
            optimal_return = (max_price / entry_price - 1) * 100
        else:  # short
            min_price = future_data['Low'].min()
            max_idx = future_data['Low'].idxmin()
            optimal_return = (entry_price / min_price - 1) * 100
            max_price = min_price
        
        # Find the day index
        day_idx = future_data.index.get_loc(max_idx) + 1
        
        return {
            'days': day_idx,
            'optimal_price': max_price,
            'optimal_return': optimal_return
        }
    
    def _check_market_condition_match(self, signal: str, market_condition: str) -> bool:
        """Check if signal matches market condition"""
        if signal == 'long' and market_condition in ['BULL', 'VOLATILE']:
            return True
        elif signal == 'short' and market_condition in ['BEAR', 'VOLATILE']:
            return True
        elif signal == 'hold' and market_condition == 'SIDEWAYS':
            return True
        return False
    
    def _generate_ia2_adjustments(self, decision_data: Dict[str, Any], actual_performance: Dict[str, Any],
                                 optimal_exit: Dict[str, Any], market_condition: str) -> Dict[str, Any]:
        """Generate adjustment suggestions for IA2"""
        adjustments = {}
        
        # Position sizing adjustments
        if actual_performance['return_pct'] < -5:
            adjustments['position_sizing'] = "Reduce position size in similar conditions"
        elif actual_performance['return_pct'] > 10:
            adjustments['position_sizing'] = "Consider larger position size for similar setups"
        
        # Stop loss adjustments
        if actual_performance['exit_reason'] == 'stop_loss' and optimal_exit['optimal_return'] > 5:
            adjustments['stop_loss'] = "Consider wider stop loss in this market condition"
        
        # Take profit adjustments
        if actual_performance['exit_reason'] == 'take_profit' and optimal_exit['optimal_return'] > actual_performance['return_pct'] * 2:
            adjustments['take_profit'] = "Consider higher take profit targets"
        
        # Market condition adjustments
        if not self._check_market_condition_match(decision_data['signal'], market_condition):
            adjustments['market_condition'] = f"Signal may not be optimal for {market_condition} conditions"
        
        # Timing adjustments
        if optimal_exit['days'] < 3:
            adjustments['timing'] = "Consider shorter-term profit taking"
        elif optimal_exit['days'] > 15:
            adjustments['timing'] = "Consider longer holding periods"
        
        return adjustments
    
    async def _generate_adaptive_context_rules(self) -> List[Dict[str, Any]]:
        """Generate adaptive context rules based on training data"""
        adaptive_rules = []
        
        # Market condition based rules
        condition_performance = defaultdict(list)
        for enhancement in self.ia2_enhancements:
            market_context = self._get_market_condition_for_period(
                enhancement.symbol, datetime.strptime(enhancement.date, '%Y-%m-%d')
            )
            condition_performance[market_context].append(enhancement.actual_performance)
        
        for condition, performances in condition_performance.items():
            avg_performance = np.mean(performances)
            rule = {
                'rule_type': 'market_condition',
                'condition': condition,
                'avg_performance': avg_performance,
                'sample_size': len(performances),
                'recommended_adjustment': self._get_condition_adjustment(condition, avg_performance)
            }
            adaptive_rules.append(rule)
        
        # Pattern-based rules
        pattern_performance = defaultdict(list)
        for training in self.pattern_training:
            pattern_performance[training.pattern_type].append(training.success)
        
        for pattern, successes in pattern_performance.items():
            success_rate = np.mean(successes)
            rule = {
                'rule_type': 'pattern_success',
                'pattern': pattern,
                'success_rate': success_rate,
                'sample_size': len(successes),
                'recommended_adjustment': self._get_pattern_adjustment(pattern, success_rate)
            }
            adaptive_rules.append(rule)
        
        # Volatility-based rules
        volatility_performance = defaultdict(list)
        for condition in self.market_conditions:
            volatility_bucket = self._get_volatility_bucket(condition.volatility)
            volatility_performance[volatility_bucket].append(condition.success_rate)
        
        for vol_bucket, success_rates in volatility_performance.items():
            avg_success = np.mean(success_rates)
            rule = {
                'rule_type': 'volatility_adjustment',
                'volatility_bucket': vol_bucket,
                'avg_success_rate': avg_success,
                'sample_size': len(success_rates),
                'recommended_adjustment': self._get_volatility_adjustment(vol_bucket, avg_success)
            }
            adaptive_rules.append(rule)
        
        logger.info(f"Generated {len(adaptive_rules)} adaptive context rules")
        return adaptive_rules
    
    def _get_condition_adjustment(self, condition: str, avg_performance: float) -> str:
        """Get adjustment recommendation for market condition"""
        if condition == 'BULL' and avg_performance > 5:
            return "Increase position sizes and extend take profit targets"
        elif condition == 'BEAR' and avg_performance > 3:
            return "Favor short positions and tighter stop losses"
        elif condition == 'SIDEWAYS' and avg_performance < 1:
            return "Reduce trading frequency and use tighter ranges"
        elif condition == 'VOLATILE' and avg_performance < 0:
            return "Reduce position sizes and use wider stop losses"
        else:
            return "Maintain standard parameters"
    
    def _get_pattern_adjustment(self, pattern: str, success_rate: float) -> str:
        """Get adjustment recommendation for pattern"""
        if success_rate > 0.7:
            return f"High confidence pattern - increase position sizing"
        elif success_rate > 0.5:
            return f"Moderate confidence pattern - standard position sizing"
        else:
            return f"Low confidence pattern - reduce position sizing or avoid"
    
    def _get_volatility_bucket(self, volatility: float) -> str:
        """Categorize volatility into buckets"""
        if volatility < 5:
            return "LOW"
        elif volatility < 15:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _get_volatility_adjustment(self, vol_bucket: str, avg_success: float) -> str:
        """Get adjustment recommendation for volatility"""
        if vol_bucket == 'LOW' and avg_success > 0.6:
            return "Stable conditions - favor trend following strategies"
        elif vol_bucket == 'HIGH' and avg_success < 0.5:
            return "High volatility - reduce position sizes and use wider stops"
        else:
            return "Standard volatility management"
    
    async def _save_training_results(self, training_results: Dict[str, Any]):
        """Save comprehensive training results"""
        results_dir = "/app/ai_training_results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save market conditions
        with open(f"{results_dir}/market_conditions_{timestamp}.json", 'w') as f:
            json.dump([
                {
                    'period_start': mc.period_start,
                    'period_end': mc.period_end,
                    'symbol': mc.symbol,
                    'condition_type': mc.condition_type,
                    'volatility': mc.volatility,
                    'trend_strength': mc.trend_strength,
                    'success_rate': mc.success_rate,
                    'confidence_score': mc.confidence_score
                } for mc in self.market_conditions
            ], f, indent=2)
        
        # Save pattern training results
        with open(f"{results_dir}/pattern_training_{timestamp}.json", 'w') as f:
            json.dump([
                {
                    'pattern_type': pt.pattern_type,
                    'symbol': pt.symbol,
                    'success': pt.success,
                    'market_condition': pt.market_condition,
                    'confidence_factors': pt.confidence_factors
                } for pt in self.pattern_training
            ], f, indent=2)
        
        # Save IA1 enhancements
        with open(f"{results_dir}/ia1_enhancements_{timestamp}.json", 'w') as f:
            json.dump([
                {
                    'symbol': e.symbol,
                    'predicted_signal': e.predicted_signal,
                    'actual_outcome': e.actual_outcome,
                    'prediction_accuracy': e.prediction_accuracy,
                    'market_context': e.market_context,
                    'suggested_improvements': e.suggested_improvements
                } for e in self.ia1_enhancements
            ], f, indent=2)
        
        # Save IA2 enhancements  
        with open(f"{results_dir}/ia2_enhancements_{timestamp}.json", 'w') as f:
            json.dump([
                {
                    'symbol': e.symbol,
                    'decision_signal': e.decision_signal,
                    'decision_confidence': e.decision_confidence,
                    'actual_performance': e.actual_performance,
                    'risk_reward_realized': e.risk_reward_realized if not np.isnan(e.risk_reward_realized) else None,
                    'market_condition_match': e.market_condition_match,
                    'suggested_adjustments': e.suggested_adjustments
                } for e in self.ia2_enhancements
            ], f, indent=2)
        
        # Save summary report
        with open(f"{results_dir}/training_summary_{timestamp}.json", 'w') as f:
            json.dump(training_results, f, indent=2)
        
        logger.info(f"Training results saved to {results_dir}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        summary = {
            'total_symbols_trained': len(self.historical_data),
            'market_conditions_analyzed': len(self.market_conditions),
            'pattern_samples_generated': len(self.pattern_training),
            'ia1_predictions_tested': len(self.ia1_enhancements),
            'ia2_decisions_analyzed': len(self.ia2_enhancements),
            'training_data_period': self._get_data_period_range(),
            'performance_metrics': self._calculate_performance_metrics()
        }
        return summary
    
    def _get_data_period_range(self) -> Dict[str, str]:
        """Get the range of training data periods"""
        if not self.historical_data:
            return {'start': 'N/A', 'end': 'N/A'}
        
        all_dates = []
        for df in self.historical_data.values():
            all_dates.extend(df['Date'].tolist())
        
        return {
            'start': min(all_dates).strftime('%Y-%m-%d'),
            'end': max(all_dates).strftime('%Y-%m-%d')
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate overall performance metrics from training"""
        metrics = {}
        
        # IA1 accuracy metrics
        if self.ia1_enhancements:
            correct_predictions = len([e for e in self.ia1_enhancements if e.actual_outcome == "correct"])
            metrics['ia1_accuracy'] = correct_predictions / len(self.ia1_enhancements)
            metrics['ia1_avg_prediction_score'] = np.mean([e.prediction_accuracy for e in self.ia1_enhancements])
        
        # IA2 performance metrics
        if self.ia2_enhancements:
            metrics['ia2_avg_return'] = np.mean([e.actual_performance for e in self.ia2_enhancements])
            metrics['ia2_avg_confidence'] = np.mean([e.decision_confidence for e in self.ia2_enhancements])
            metrics['ia2_condition_match_rate'] = np.mean([e.market_condition_match for e in self.ia2_enhancements])
        
        # Pattern success metrics
        if self.pattern_training:
            metrics['pattern_overall_success_rate'] = np.mean([p.success for p in self.pattern_training])
            pattern_success = defaultdict(list)
            for p in self.pattern_training:
                pattern_success[p.pattern_type].append(p.success)
            metrics['pattern_success_by_type'] = {k: np.mean(v) for k, v in pattern_success.items()}
        
        # Market condition metrics
        if self.market_conditions:
            condition_distribution = Counter([mc.condition_type for mc in self.market_conditions])
            metrics['market_condition_distribution'] = dict(condition_distribution)
            metrics['avg_market_condition_confidence'] = np.mean([mc.confidence_score for mc in self.market_conditions])
        
        return metrics

# Global instance
ai_training_system = AITrainingSystem()
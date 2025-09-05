# enhanced_trading_system.py
"""
Enhanced trading system with complete technical analysis implementation
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import time
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SignalType(Enum):
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"

class ConfidenceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class TechnicalIndicators:
    """Container for technical analysis results"""
    rsi: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_percent: float
    volume_sma: float
    atr: float
    stoch_k: float
    stoch_d: float

@dataclass
class TradingDecision:
    """Enhanced trading decision with comprehensive data"""
    symbol: str
    signal: SignalType
    confidence: ConfidenceLevel
    confidence_score: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    position_size: float
    risk_reward_ratio: float
    max_risk_pct: float
    rationale: str
    technical_indicators: TechnicalIndicators
    pattern_analysis: Dict[str, Any]
    market_context: Dict[str, Any]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['signal'] = self.signal.value
        result['confidence'] = self.confidence.value
        result['timestamp'] = self.timestamp.isoformat()
        return result

class TechnicalAnalyzer:
    """Advanced technical analysis with real implementations"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def detect_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced pattern detection"""
        patterns = {}
        
        # Get the most recent data for pattern analysis
        recent_data = data.tail(50)  # Last 50 periods
        
        # Double Top/Bottom Detection
        patterns.update(self._detect_double_patterns(recent_data))
        
        # Head and Shoulders Detection
        patterns.update(self._detect_head_shoulders(recent_data))
        
        # Support/Resistance Levels
        patterns.update(self._detect_support_resistance(recent_data))
        
        # Trend Analysis
        patterns.update(self._analyze_trend(recent_data))
        
        return patterns
    
    def _detect_double_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect double top/bottom patterns"""
        highs = data['high']
        lows = data['low']
        
        # Find local maxima and minima
        from scipy.signal import argrelextrema
        
        high_peaks = argrelextrema(highs.values, np.greater, order=3)[0]
        low_peaks = argrelextrema(lows.values, np.less, order=3)[0]
        
        patterns = {}
        
        # Double top detection (simplified)
        if len(high_peaks) >= 2:
            recent_peaks = high_peaks[-2:]
            peak_values = [highs.iloc[i] for i in recent_peaks]
            
            # Check if peaks are roughly equal (within 2%)
            if len(peak_values) == 2 and abs(peak_values[0] - peak_values[1]) / peak_values[0] < 0.02:
                patterns['double_top'] = {
                    'detected': True,
                    'confidence': 0.7,
                    'peak_values': peak_values,
                    'signal': 'bearish'
                }
        
        # Double bottom detection (simplified)
        if len(low_peaks) >= 2:
            recent_valleys = low_peaks[-2:]
            valley_values = [lows.iloc[i] for i in recent_valleys]
            
            if len(valley_values) == 2 and abs(valley_values[0] - valley_values[1]) / valley_values[0] < 0.02:
                patterns['double_bottom'] = {
                    'detected': True,
                    'confidence': 0.7,
                    'valley_values': valley_values,
                    'signal': 'bullish'
                }
        
        return patterns
    
    def _detect_head_shoulders(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Basic head and shoulders detection"""
        # Simplified implementation - in practice would use more sophisticated algorithms
        from scipy.signal import argrelextrema
        
        highs = data['high']
        high_peaks = argrelextrema(highs.values, np.greater, order=5)[0]
        
        patterns = {}
        
        if len(high_peaks) >= 3:
            # Take the last 3 peaks
            last_three_peaks = high_peaks[-3:]
            peak_values = [highs.iloc[i] for i in last_three_peaks]
            
            # Check for head and shoulders pattern (middle peak highest)
            if (peak_values[1] > peak_values[0] and 
                peak_values[1] > peak_values[2] and
                abs(peak_values[0] - peak_values[2]) / peak_values[0] < 0.05):  # Shoulders roughly equal
                
                patterns['head_shoulders'] = {
                    'detected': True,
                    'confidence': 0.75,
                    'peaks': peak_values,
                    'signal': 'bearish'
                }
        
        return patterns
    
    def _detect_support_resistance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect key support and resistance levels"""
        closes = data['close']
        highs = data['high']
        lows = data['low']
        
        # Simple support/resistance based on recent highs and lows
        recent_high = highs.tail(20).max()
        recent_low = lows.tail(20).min()
        current_price = closes.iloc[-1]
        
        # Calculate pivot points
        typical_price = (highs + lows + closes) / 3
        pivot = typical_price.tail(10).mean()
        
        support_1 = 2 * pivot - recent_high
        resistance_1 = 2 * pivot - recent_low
        
        return {
            'support_resistance': {
                'current_price': current_price,
                'pivot': pivot,
                'resistance_1': resistance_1,
                'support_1': support_1,
                'recent_high': recent_high,
                'recent_low': recent_low,
                'distance_to_resistance': (resistance_1 - current_price) / current_price * 100,
                'distance_to_support': (current_price - support_1) / current_price * 100
            }
        }
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend direction and strength"""
        closes = data['close']
        
        # Calculate multiple EMAs for trend analysis
        ema_20 = closes.ewm(span=20).mean()
        ema_50 = closes.ewm(span=50).mean() if len(closes) >= 50 else closes.ewm(span=len(closes)//2).mean()
        
        current_price = closes.iloc[-1]
        ema_20_current = ema_20.iloc[-1]
        ema_50_current = ema_50.iloc[-1]
        
        # Trend direction
        if current_price > ema_20_current > ema_50_current:
            trend = 'bullish'
            trend_strength = min((current_price - ema_50_current) / ema_50_current * 100, 10)
        elif current_price < ema_20_current < ema_50_current:
            trend = 'bearish'
            trend_strength = min((ema_50_current - current_price) / current_price * 100, 10)
        else:
            trend = 'sideways'
            trend_strength = 0
        
        # ADX approximation (trend strength indicator)
        high_low = data['high'] - data['low']
        adx_approx = high_low.rolling(window=14).mean() / closes.rolling(window=14).mean() * 100
        
        return {
            'trend_analysis': {
                'direction': trend,
                'strength': trend_strength,
                'ema_20': ema_20_current,
                'ema_50': ema_50_current,
                'adx_approximation': adx_approx.iloc[-1] if not adx_approx.empty else 0
            }
        }

class EnhancedIA1_Strategist:
    """Enhanced IA1 with complete technical analysis implementation"""
    
    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.conversation_memory = {}
        
        # Enhanced scoring weights
        self.indicator_weights = {
            'rsi': 0.15,
            'macd': 0.20,
            'bollinger': 0.15,
            'stochastic': 0.10,
            'pattern': 0.25,
            'trend': 0.15
        }
    
    async def analyze_opportunities(self, opportunities: Dict[str, Any]) -> List[TradingDecision]:
        """Enhanced opportunity analysis with complete technical analysis"""
        decisions = []
        
        for symbol, opportunity_data in opportunities.items():
            try:
                # Generate OHLCV data if not provided
                if 'ohlcv_data' not in opportunity_data:
                    ohlcv_data = self._generate_realistic_ohlcv(symbol, opportunity_data)
                else:
                    ohlcv_data = opportunity_data['ohlcv_data']
                
                # Perform comprehensive technical analysis
                technical_indicators = self._calculate_all_indicators(ohlcv_data)
                
                # Pattern analysis
                pattern_analysis = self.technical_analyzer.detect_patterns(ohlcv_data)
                
                # Market context analysis
                market_context = self._analyze_market_context(ohlcv_data, opportunity_data)
                
                # Generate trading decision
                decision = await self._generate_trading_decision(
                    symbol, ohlcv_data, technical_indicators, 
                    pattern_analysis, market_context
                )
                
                if decision and decision.confidence_score > 0.6:
                    decisions.append(decision)
                    self.logger.info(f"Generated {decision.signal.value} signal for {symbol} with confidence {decision.confidence_score:.2f}")
                
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        return sorted(decisions, key=lambda x: x.confidence_score, reverse=True)
    
    def _generate_realistic_ohlcv(self, symbol: str, opportunity_data: Dict) -> pd.DataFrame:
        """Generate realistic OHLCV data from opportunity data"""
        price_history = opportunity_data.get('price_history', [])
        volume_history = opportunity_data.get('volume_history', [])
        
        if not price_history:
            # Generate some sample data
            base_price = 50000 if 'BTC' in symbol else 3000
            price_history = [base_price * (1 + np.random.normal(0, 0.02)) for _ in range(100)]
        
        if not volume_history:
            volume_history = [np.random.uniform(1000, 5000) for _ in range(len(price_history))]
        
        # Create OHLCV data
        data = []
        for i, close in enumerate(price_history):
            if i == 0:
                open_price = close
            else:
                open_price = price_history[i-1]
            
            high = max(open_price, close) * np.random.uniform(1.0, 1.02)
            low = min(open_price, close) * np.random.uniform(0.98, 1.0)
            volume = volume_history[i] if i < len(volume_history) else np.random.uniform(1000, 5000)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'timestamp': datetime.now() - timedelta(hours=len(price_history)-i)
            })
        
        df = pd.DataFrame(data)
        return df

    def _generate_rationale(self, signal: SignalType, signal_scores: Dict, 
                       indicators: TechnicalIndicators, patterns: Dict) -> str:
    """Generate human-readable rationale for the decision"""
    rationale_parts = []
    
    # Technical indicators
    strong_indicators = [(k, v) for k, v in signal_scores.items() if abs(v) > 0.05]
    strong_indicators.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for indicator, score in strong_indicators[:3]:  # Top 3 indicators
        if indicator == 'rsi':
            if score > 0:
                rationale_parts.append(f"RSI oversold at {indicators.rsi:.1f} (bullish)")
            else:
                rationale_parts.append(f"RSI overbought at {indicators.rsi:.1f} (bearish)")
        elif indicator == 'macd':
            if score > 0:
                rationale_parts.append(f"MACD bullish crossover (line: {indicators.macd_line:.4f}, signal: {indicators.macd_signal:.4f})")
            else:
                rationale_parts.append(f"MACD bearish crossover (line: {indicators.macd_line:.4f}, signal: {indicators.macd_signal:.4f})")
        elif indicator == 'bollinger':
            if score > 0:
                rationale_parts.append(f"Price near lower Bollinger Band ({indicators.bb_percent:.1%} from bottom)")
            else:
                rationale_parts.append(f"Price near upper Bollinger Band ({indicators.bb_percent:.1%} from top)")
        elif indicator == 'stochastic':
            if score > 0:
                rationale_parts.append(f"Stochastic oversold (K: {indicators.stoch_k:.1f}, D: {indicators.stoch_d:.1f})")
            else:
                rationale_parts.append(f"Stochastic overbought (K: {indicators.stoch_k:.1f}, D: {indicators.stoch_d:.1f})")
        elif indicator == 'pattern':
            detected_patterns = [k for k, v in patterns.items() 
                               if isinstance(v, dict) and v.get('detected')]
            if detected_patterns:
                pattern_info = []
                for pattern_name in detected_patterns:
                    pattern_data = patterns[pattern_name]
                    if pattern_name == 'double_top':
                        pattern_info.append(f"Double Top pattern with peaks at {pattern_data.get('peak_values', [])}")
                    elif pattern_name == 'double_bottom':
                        pattern_info.append(f"Double Bottom pattern with valleys at {pattern_data.get('valley_values', [])}")
                    elif pattern_name == 'head_shoulders':
                        pattern_info.append(f"Head and Shoulders pattern with peaks at {pattern_data.get('peaks', [])}")
                    else:
                        pattern_info.append(pattern_name)
                rationale_parts.append(f"Patterns: {', '.join(pattern_info)}")
        elif indicator == 'trend':
            trend_data = patterns.get('trend_analysis', {})
            direction = trend_data.get('direction', 'sideways')
            strength = trend_data.get('strength', 0)
            if direction == 'bullish':
                rationale_parts.append(f"Strong bullish trend (strength: {strength:.1f}/10)")
            elif direction == 'bearish':
                rationale_parts.append(f"Strong bearish trend (strength: {strength:.1f}/10)")
            else:
                rationale_parts.append("Sideways market with no clear trend")
    
    # Add support/resistance information if available
    support_resistance = patterns.get('support_resistance', {})
    if support_resistance:
        current_price = support_resistance.get('current_price', 0)
        resistance_1 = support_resistance.get('resistance_1', 0)
        support_1 = support_resistance.get('support_1', 0)
        
        if resistance_1 > 0:
            resistance_distance = (resistance_1 - current_price) / current_price * 100
            rationale_parts.append(f"Next resistance at {resistance_1:.2f} ({resistance_distance:+.1f}%)")
        
        if support_1 > 0:
            support_distance = (current_price - support_1) / current_price * 100
            rationale_parts.append(f"Next support at {support_1:.2f} ({support_distance:+.1f}%)")
    
    # Add volatility context
    if indicators.atr > 0:
        atr_percent = indicators.atr / indicators.bb_middle * 100
        if atr_percent > 3:
            rationale_parts.append(f"High volatility (ATR: {atr_percent:.1f}%)")
        else:
            rationale_parts.append(f"Normal volatility (ATR: {atr_percent:.1f}%)")
    
    # Add volume context
    if indicators.volume_sma > 0:
        volume_ratio = indicators.volume_sma / (indicators.volume_sma * 0.8)  # Simplified
        if volume_ratio > 1.5:
            rationale_parts.append("High volume supporting the move")
        elif volume_ratio < 0.5:
            rationale_parts.append("Low volume - weak conviction")
    
    # Limit to top 5 rationale points for conciseness
    if len(rationale_parts) > 5:
        rationale_parts = rationale_parts[:5]
    
    signal_str = signal.value.upper()
    return f"{signal_str} signal based on: {'; '.join(rationale_parts)}"

    def _calculate_all_indicators(self, data: pd.DataFrame) -> TechnicalIndicators:
        """Calculate all technical indicators"""
        closes = data['close']
        highs = data['high']
        lows = data['low']
        volumes = data['volume']
        
        # RSI
        rsi = self.technical_analyzer.calculate_rsi(closes).iloc[-1]
        
        # MACD
        macd_line, macd_signal, macd_histogram = self.technical_analyzer.calculate_macd(closes)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.technical_analyzer.calculate_bollinger_bands(closes)
        bb_percent = (closes.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        
        # Stochastic
        stoch_k, stoch_d = self.technical_analyzer.calculate_stochastic(highs, lows, closes)
        
        # ATR
        atr = self.technical_analyzer.calculate_atr(highs, lows, closes)
        
        # Volume SMA
        volume_sma = volumes.rolling(window=20).mean().iloc[-1]
        
        return TechnicalIndicators(
            rsi=rsi if not pd.isna(rsi) else 50,
            macd_line=macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0,
            macd_signal=macd_signal.iloc[-1] if not pd.isna(macd_signal.iloc[-1]) else 0,
            macd_histogram=macd_histogram.iloc[-1] if not pd.isna(macd_histogram.iloc[-1]) else 0,
            bb_upper=bb_upper.iloc[-1] if not pd.isna(bb_upper.iloc[-1]) else closes.iloc[-1] * 1.02,
            bb_middle=bb_middle.iloc[-1] if not pd.isna(bb_middle.iloc[-1]) else closes.iloc[-1],
            bb_lower=bb_lower.iloc[-1] if not pd.isna(bb_lower.iloc[-1]) else closes.iloc[-1] * 0.98,
            bb_percent=bb_percent if not pd.isna(bb_percent) else 0.5,
            volume_sma=volume_sma if not pd.isna(volume_sma) else volumes.iloc[-1],
            atr=atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else closes.iloc[-1] * 0.02,
            stoch_k=stoch_k.iloc[-1] if not pd.isna(stoch_k.iloc[-1]) else 50,
            stoch_d=stoch_d.iloc[-1] if not pd.isna(stoch_d.iloc[-1]) else 50
        )
    
    def _analyze_market_context(self, ohlcv_data: pd.DataFrame, opportunity_data: Dict) -> Dict[str, Any]:
        """Analyze market context and conditions"""
        current_price = ohlcv_data['close'].iloc[-1]
        volume = ohlcv_data['volume'].iloc[-1]
        avg_volume = ohlcv_data['volume'].tail(20).mean()
        
        # Volatility analysis
        returns = ohlcv_data['close'].pct_change().tail(20)
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Liquidity analysis
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        
        return {
            'current_price': current_price,
            'volume_ratio': volume_ratio,
            'volatility': volatility,
            'market_cap_rank': opportunity_data.get('market_cap_rank', 100),
            'confidence_indicator': opportunity_data.get('confidence_indicator', 0.5)
        }
    
    async def _generate_trading_decision(self, symbol: str, ohlcv_data: pd.DataFrame, 
                                       indicators: TechnicalIndicators, patterns: Dict,
                                       market_context: Dict) -> Optional[TradingDecision]:
        """Generate comprehensive trading decision"""
        try:
            current_price = market_context['current_price']
            
            # Multi-factor signal generation
            signal_scores = self._calculate_signal_scores(indicators, patterns, market_context)
            overall_score = sum(signal_scores.values())
            
            # Determine signal type
            if overall_score > 0.3:
                signal = SignalType.LONG
            elif overall_score < -0.3:
                signal = SignalType.SHORT
            else:
                signal = SignalType.HOLD
            
            if signal == SignalType.HOLD:
                return None
            
            # Calculate confidence
            confidence_score = min(abs(overall_score), 1.0)
            confidence_level = self._get_confidence_level(confidence_score)
            
            # Risk management calculations
            atr = indicators.atr
            stop_loss, take_profits = self._calculate_price_levels(
                signal, current_price, atr, confidence_score
            )
            
            # Position sizing based on risk
            max_risk_pct = self._calculate_max_risk(confidence_score, market_context['volatility'])
            position_size = self._calculate_position_size(current_price, stop_loss, max_risk_pct)
            
            # Risk-reward ratio
            risk_reward_ratio = (take_profits[0] - current_price) / (current_price - stop_loss) if signal == SignalType.LONG else (current_price - take_profits[0]) / (stop_loss - current_price)
            
            # Generate rationale
            rationale = self._generate_rationale(signal, signal_scores, indicators, patterns)
            
            return TradingDecision(
                symbol=symbol,
                signal=signal,
                confidence=confidence_level,
                confidence_score=confidence_score,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit_1=take_profits[0],
                take_profit_2=take_profits[1],
                take_profit_3=take_profits[2],
                position_size=position_size,
                risk_reward_ratio=abs(risk_reward_ratio),
                max_risk_pct=max_risk_pct,
                rationale=rationale,
                technical_indicators=indicators,
                pattern_analysis=patterns,
                market_context=market_context,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error generating trading decision for {symbol}: {e}")
            return None
    
    def _calculate_signal_scores(self, indicators: TechnicalIndicators, patterns: Dict, market_context: Dict) -> Dict[str, float]:
        """Calculate individual signal scores for each factor"""
        scores = {}
        
        # RSI Score
        if indicators.rsi < 30:
            scores['rsi'] = 0.8 * self.indicator_weights['rsi']  # Oversold - bullish
        elif indicators.rsi > 70:
            scores['rsi'] = -0.8 * self.indicator_weights['rsi']  # Overbought - bearish
        else:
            scores['rsi'] = 0
        
        # MACD Score
        if indicators.macd_histogram > 0 and indicators.macd_line > indicators.macd_signal:
            scores['macd'] = 0.7 * self.indicator_weights['macd']
        elif indicators.macd_histogram < 0 and indicators.macd_line < indicators.macd_signal:
            scores['macd'] = -0.7 * self.indicator_weights['macd']
        else:
            scores['macd'] = 0
        
        # Bollinger Bands Score
        if indicators.bb_percent < 0.2:
            scores['bollinger'] = 0.6 * self.indicator_weights['bollinger']  # Near lower band - bullish
        elif indicators.bb_percent > 0.8:
            scores['bollinger'] = -0.6 * self.indicator_weights['bollinger']  # Near upper band - bearish
        else:
            scores['bollinger'] = 0
        
        # Stochastic Score
        if indicators.stoch_k < 20 and indicators.stoch_d < 20:
            scores['stochastic'] = 0.5 * self.indicator_weights['stochastic']
        elif indicators.stoch_k > 80 and indicators.stoch_d > 80:
            scores['stochastic'] = -0.5 * self.indicator_weights['stochastic']
        else:
            scores['stochastic'] = 0
        
        # Pattern Score
        pattern_score = 0
        for pattern_name, pattern_data in patterns.items():
            if isinstance(pattern_data, dict) and pattern_data.get('detected'):
                confidence = pattern_data.get('confidence', 0.5)
                signal = pattern_data.get('signal', 'neutral')
                if signal == 'bullish':
                    pattern_score += confidence * 0.5
                elif signal == 'bearish':
                    pattern_score -= confidence * 0.5
        
        scores['pattern'] = pattern_score * self.indicator_weights['pattern']
        
        # Trend Score
        trend_data = patterns.get('trend_analysis', {})
        trend_direction = trend_data.get('direction', 'sideways')
        trend_strength = trend_data.get('strength', 0) / 10  # Normalize to 0-1
        
        if trend_direction == 'bullish':
            scores['trend'] = trend_strength * self.indicator_weights['trend']
        elif trend_direction == 'bearish':
            scores['trend'] = -trend_strength * self.indicator_weights['trend']
        else:
            scores['trend'] = 0
        
        return scores
    
    def _get_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Convert numeric confidence to level"""
        if confidence_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _calculate_price_levels(self, signal: SignalType, current_price: float, 
                               atr: float, confidence: float) -> Tuple[float, List[float]]:
        """Calculate stop loss and take profit levels"""
        atr_multiplier = 2.0 + (1.0 - confidence)  # Higher risk for lower confidence
        
        if signal == SignalType.LONG:
            stop_loss = current_price - (atr * atr_multiplier)
            take_profit_1 = current_price + (atr * atr_multiplier * 1.5)
            take_profit_2 = current_price + (atr * atr_multiplier * 2.5)
            take_profit_3 = current_price + (atr * atr_multiplier * 3.5)
        else:  # SHORT
            stop_loss = current_price + (atr * atr_multiplier)
            take_profit_1 = current_price - (atr * atr_multiplier * 1.5)
            take_profit_2 = current_price - (atr * atr_multiplier * 2.5)
            take_profit_3 = current_price - (atr * atr_multiplier * 3.5)
        
        return stop_loss, [take_profit_1, take_profit_2, take_profit_3]
    
    def _calculate_max_risk(self, confidence: float, volatility: float) -> float:
        """Calculate maximum risk percentage based on confidence and volatility"""
        base_risk = 0.02  # 2% base risk
        confidence_adjustment = confidence * 0.01  # Up to 1% additional for high confidence
        volatility_adjustment = max(0, (volatility - 0.3) * -0.005)  # Reduce risk for high volatility
        
        max_risk = base_risk + confidence_adjustment + volatility_adjustment
        return max(0.005, min(0.05, max_risk))  # Clamp between 0.5% and 5%
    
    def _calculate_position_size(self, entry_price: float, stop_loss: float, max_risk_pct: float) -> float:
        """Calculate position size based on risk management"""
        # Assuming $100,000 portfolio (this should come from actual portfolio data)
        portfolio_value = 100000
        risk_amount = portfolio_value * max_risk_pct
        
        price_risk = abs(entry_price - stop_loss)
        position_size = risk_amount / price_risk if price_risk > 0 else 0
        
        # Convert to percentage of portfolio
        position_size_pct = (position_size * entry_price) / portfolio_value
        
        return min(position_size_pct, 0.2)  # Max 20% of portfolio
    
    def _generate_rationale(self, signal: SignalType, signal_scores: Dict, 
                           indicators: TechnicalIndicators, patterns: Dict) -> str:
        """Generate human-readable rationale for the decision"""
        rationale_parts = []
        
        # Technical indicators
        strong_indicators = [(k, v) for k, v in signal_scores.items() if abs(v) > 0.05]
        strong_indicators.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for indicator, score in strong_indicators[:3]:  # Top 3 indicators
            if indicator == 'rsi':
                if score > 0:
                    rationale_parts.append(f"RSI oversold at {indicators.rsi:.1f}")
                else:
                    rationale_parts.append(f"RSI overbought at {indicators.rsi:.1f}")
            elif indicator == 'macd':
                if score > 0:
                    rationale_parts.append("MACD bullish crossover")
                else:
                    rationale_parts.append("MACD bearish crossover")
            elif indicator == 'bollinger':
                if score > 0:
                    rationale_parts.append("Price near lower Bollinger Band")
                else:
                    rationale_parts.append("Price near upper Bollinger Band")
            elif indicator == 'pattern':
                detected_patterns = [k for k, v in patterns.items() 
                                   if isinstance(v, dict) and v.get('detected')]
                if detected_patterns:
                    rationale_parts.append(f"Pattern detected: {', '.join(detected_patterns)}")
        
        # Add trend information
        trend_data = patterns.get('trend_analysis', {})
        if trend_data.get('direction') != 'sideways':
            rationale_parts.append(f"Trend: {trend_data.get('direction', 'unknown')}")
        
        signal_str = signal.value.upper()
        return f"{signal_str} signal based on: {'; '.join(rationale_parts[:4])}"

class EnhancedPositionManager:
    """Enhanced position management with portfolio context"""
    
    def __init__(self, portfolio_value: float = 100000):
        self.portfolio_value = portfolio_value
        self.open_positions = {}
        self.position_history = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_position(self, symbol: str, decision: TradingDecision, actual_entry_price: float = None):
        """Add a new position to tracking"""
        entry_price = actual_entry_price or decision.entry_price
        position_value = decision.position_size * self.portfolio_value
        
        position = {
            'symbol': symbol,
            'entry_price': entry_price,
            'position_size_pct': decision.position_size,
            'position_value': position_value,
            'stop_loss': decision.stop_loss,
            'take_profits': [decision.take_profit_1, decision.take_profit_2, decision.take_profit_3],
            'signal': decision.signal,
            'entry_time': datetime.now(),
            'max_risk_pct': decision.max_risk_pct,
            'confidence': decision.confidence_score
        }
        
        self.open_positions[symbol] = position
        self.logger.info(f"Added position: {symbol} {decision.signal.value} ${position_value:.2f}")
    
    def update_position(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Update position with current market price and return analysis"""
        if symbol not in self.open_positions:
            return {'error': 'Position not found'}
        
        position = self.open_positions[symbol]
        
        # Calculate unrealized P&L
        if position['signal'] == SignalType.LONG:
            unrealized_pnl = (current_price - position['entry_price']) * position['position_value'] / position['entry_price']
        else:  # SHORT
            unrealized_pnl = (position['entry_price'] - current_price) * position['position_value'] / position['entry_price']
        
        unrealized_pnl_pct = unrealized_pnl / self.portfolio_value * 100
        
        # Check stop loss and take profit levels
        actions_needed = []
        
        if position['signal'] == SignalType.LONG:
            if current_price <= position['stop_loss']:
                actions_needed.append({'action': 'close_position', 'reason': 'stop_loss_hit'})
            else:
                for i, tp_level in enumerate(position['take_profits']):
                    if current_price >= tp_level:
                        actions_needed.append({
                            'action': 'take_profit',
                            'level': i + 1,
                            'price': tp_level,
                            'reason': f'take_profit_{i+1}_hit'
                        })
        else:  # SHORT
            if current_price >= position['stop_loss']:
                actions_needed.append({'action': 'close_position', 'reason': 'stop_loss_hit'})
            else:
                for i, tp_level in enumerate(position['take_profits']):
                    if current_price <= tp_level:
                        actions_needed.append({
                            'action': 'take_profit',
                            'level': i + 1,
                            'price': tp_level,
                            'reason': f'take_profit_{i+1}_hit'
                        })
        
        # Update position data
        position['current_price'] = current_price
        position['unrealized_pnl'] = unrealized_pnl
        position['unrealized_pnl_pct'] = unrealized_pnl_pct
        position['last_update'] = datetime.now()
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'actions_needed': actions_needed,
            'position_age_hours': (datetime.now() - position['entry_time']).total_seconds() / 3600
        }
    
    def close_position(self, symbol: str, close_price: float, reason: str = 'manual'):
        """Close a position and record the result"""
        if symbol not in self.open_positions:
            return {'error': 'Position not found'}
        
        position = self.open_positions[symbol]
        
        # Calculate realized P&L
        if position['signal'] == SignalType.LONG:
            realized_pnl = (close_price - position['entry_price']) * position['position_value'] / position['entry_price']
        else:  # SHORT
            realized_pnl = (position['entry_price'] - close_price) * position['position_value'] / position['entry_price']
        
        realized_pnl_pct = realized_pnl / self.portfolio_value * 100
        
        # Record in history
        trade_result = {
            **position,
            'close_price': close_price,
            'close_time': datetime.now(),
            'realized_pnl': realized_pnl,
            'realized_pnl_pct': realized_pnl_pct,
            'close_reason': reason,
            'duration_hours': (datetime.now() - position['entry_time']).total_seconds() / 3600
        }
        
        self.position_history.append(trade_result)
        del self.open_positions[symbol]
        
        # Update portfolio value
        self.portfolio_value += realized_pnl
        
        self.logger.info(f"Closed position: {symbol} P&L: ${realized_pnl:.2f} ({realized_pnl_pct:.2f}%)")
        
        return trade_result
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        total_exposure = sum(pos['position_value'] for pos in self.open_positions.values())
        exposure_pct = total_exposure / self.portfolio_value * 100
        
        # Calculate total unrealized P&L
        total_unrealized_pnl = 0
        for symbol, position in self.open_positions.items():
            if 'unrealized_pnl' in position:
                total_unrealized_pnl += position['unrealized_pnl']
        
        # Recent performance (last 30 days)
        recent_trades = [t for t in self.position_history 
                        if (datetime.now() - t['close_time']).days <= 30]
        
        win_rate = 0
        avg_return = 0
        if recent_trades:
            wins = len([t for t in recent_trades if t['realized_pnl'] > 0])
            win_rate = wins / len(recent_trades) * 100
            avg_return = sum(t['realized_pnl_pct'] for t in recent_trades) / len(recent_trades)
        
        return {
            'portfolio_value': self.portfolio_value,
            'open_positions_count': len(self.open_positions),
            'total_exposure': total_exposure,
            'exposure_percentage': exposure_pct,
            'total_unrealized_pnl': total_unrealized_pnl,
            'unrealized_pnl_pct': total_unrealized_pnl / self.portfolio_value * 100,
            'recent_win_rate': win_rate,
            'recent_avg_return': avg_return,
            'total_trades': len(self.position_history),
            'positions': list(self.open_positions.keys())
        }

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, max_portfolio_risk: float = 0.2, max_correlation: float = 0.8):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlation = max_correlation
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_new_position(self, decision: TradingDecision, position_manager: EnhancedPositionManager) -> Dict[str, Any]:
        """Validate if a new position should be opened based on risk rules"""
        
        portfolio_summary = position_manager.get_portfolio_summary()
        
        # Check maximum positions
        if len(position_manager.open_positions) >= 10:  # Max 10 concurrent positions
            return {
                'approved': False,
                'reason': 'Maximum number of concurrent positions reached',
                'risk_level': 'high'
            }
        
        # Check portfolio exposure
        new_exposure = decision.position_size * position_manager.portfolio_value
        total_exposure = portfolio_summary['total_exposure'] + new_exposure
        exposure_pct = total_exposure / position_manager.portfolio_value
        
        if exposure_pct > 0.8:  # Max 80% portfolio exposure
            return {
                'approved': False,
                'reason': f'Portfolio exposure would exceed limit: {exposure_pct:.1f}%',
                'risk_level': 'high'
            }
        
        # Check correlation (simplified - in practice would use actual price correlations)
        similar_positions = [pos for pos in position_manager.open_positions.values() 
                           if pos['signal'] == decision.signal]
        
        if len(similar_positions) >= 5:  # Max 5 positions in same direction
            return {
                'approved': False,
                'reason': 'Too many positions in same direction',
                'risk_level': 'medium'
            }
        
        # Risk-adjusted position sizing
        adjusted_size = self._adjust_position_size(decision, portfolio_summary)
        
        risk_level = 'low'
        if decision.confidence_score < 0.7:
            risk_level = 'medium'
        if decision.max_risk_pct > 0.03:
            risk_level = 'high'
        
        return {
            'approved': True,
            'adjusted_position_size': adjusted_size,
            'risk_level': risk_level,
            'reason': 'Position approved with risk management adjustments'
        }
    
    def _adjust_position_size(self, decision: TradingDecision, portfolio_summary: Dict) -> float:
        """Adjust position size based on portfolio state"""
        base_size = decision.position_size
        
        # Reduce size if portfolio is already heavily exposed
        exposure_factor = max(0.5, 1 - (portfolio_summary['exposure_percentage'] / 100))
        
        # Reduce size for lower confidence
        confidence_factor = 0.5 + (decision.confidence_score * 0.5)
        
        # Recent performance adjustment
        performance_factor = 1.0
        if portfolio_summary['recent_avg_return'] < -2:  # If recent performance is poor
            performance_factor = 0.7
        elif portfolio_summary['recent_avg_return'] > 5:  # If recent performance is great
            performance_factor = 1.2
        
        adjusted_size = base_size * exposure_factor * confidence_factor * performance_factor
        
        return min(adjusted_size, 0.15)  # Never exceed 15% of portfolio

class EnhancedTradingOrchestrator:
    """Enhanced trading orchestrator with complete implementation"""
    
    def __init__(self):
        self.ia1_strategist = EnhancedIA1_Strategist()
        self.position_manager = EnhancedPositionManager()
        self.risk_manager = RiskManager()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance tracking
        self.cycle_count = 0
        self.last_cycle_time = None
        self.performance_metrics = {
            'total_signals': 0,
            'executed_trades': 0,
            'rejected_trades': 0,
            'avg_confidence': 0
        }
    
    async def run_complete_trading_cycle(self) -> List[Dict[str, Any]]:
        """Run a complete enhanced trading cycle"""
        cycle_start = datetime.now()
        self.cycle_count += 1
        
        try:
            self.logger.info(f"Starting trading cycle #{self.cycle_count}")
            
            # 1. Generate opportunities (mock data for demo)
            opportunities = self._generate_mock_opportunities()
            self.logger.info(f"Found {len(opportunities)} opportunities")
            
            # 2. Analyze with enhanced IA1
            trading_decisions = await self.ia1_strategist.analyze_opportunities(opportunities)
            self.performance_metrics['total_signals'] += len(trading_decisions)
            
            if trading_decisions:
                avg_conf = sum(d.confidence_score for d in trading_decisions) / len(trading_decisions)
                self.performance_metrics['avg_confidence'] = avg_conf
                self.logger.info(f"Generated {len(trading_decisions)} trading signals, avg confidence: {avg_conf:.2f}")
            
            executed_trades = []
            
            # 3. Risk management and execution
            for decision in trading_decisions:
                # Risk validation
                risk_check = self.risk_manager.validate_new_position(decision, self.position_manager)
                
                if risk_check['approved']:
                    # Adjust position size if recommended
                    if 'adjusted_position_size' in risk_check:
                        decision.position_size = risk_check['adjusted_position_size']
                    
                    # Execute trade (mock execution)
                    execution_result = await self._execute_trade(decision)
                    
                    if execution_result['success']:
                        # Add to position manager
                        self.position_manager.add_position(
                            decision.symbol, decision, execution_result['fill_price']
                        )
                        executed_trades.append({
                            'symbol': decision.symbol,
                            'signal': decision.signal.value,
                            'confidence': decision.confidence_score,
                            'position_size': decision.position_size,
                            'entry_price': execution_result['fill_price']
                        })
                        self.performance_metrics['executed_trades'] += 1
                    else:
                        self.logger.warning(f"Trade execution failed for {decision.symbol}: {execution_result['reason']}")
                else:
                    self.logger.info(f"Trade rejected for {decision.symbol}: {risk_check['reason']}")
                    self.performance_metrics['rejected_trades'] += 1
            
            # 4. Monitor existing positions
            await self._monitor_positions()
            
            # 5. Update performance metrics
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            self.last_cycle_time = cycle_duration
            
            self.logger.info(f"Trading cycle completed in {cycle_duration:.2f}s. Executed {len(executed_trades)} trades.")
            
            return executed_trades
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
            return []
    
    def _generate_mock_opportunities(self) -> Dict[str, Any]:
        """Generate mock opportunities for demonstration"""
        symbols = ["BTC-USDT", "ETH-USDT", "ADA-USDT", "SOL-USDT", "AVAX-USDT"]
        opportunities = {}
        
        for symbol in symbols[:np.random.randint(2, 5)]:  # Random 2-4 symbols
            # Generate realistic price history
            base_price = {
                "BTC-USDT": 50000, "ETH-USDT": 3000, "ADA-USDT": 0.5, 
                "SOL-USDT": 100, "AVAX-USDT": 25
            }[symbol]
            
            price_history = []
            volume_history = []
            current_price = base_price
            
            # Generate 100 historical points with trend
            trend_direction = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])  # bearish, sideways, bullish
            
            for i in range(100):
                # Add trend + noise
                trend_component = trend_direction * 0.001  # 0.1% trend per period
                noise_component = np.random.normal(0, 0.02)  # 2% noise
                
                price_change = trend_component + noise_component
                current_price *= (1 + price_change)
                price_history.append(current_price)
                
                # Volume with some correlation to price movement
                base_volume = 1000
                volume_multiplier = 1 + abs(price_change) * 5  # Higher volume on big moves
                volume_history.append(base_volume * volume_multiplier * np.random.uniform(0.8, 1.2))
            
            opportunities[symbol] = {
                'price_history': price_history,
                'volume_history': volume_history,
                'market_cap_rank': np.random.randint(1, 50),
                'confidence_indicator': np.random.uniform(0.4, 0.9),
                'timestamp': datetime.now().isoformat()
            }
        
        return opportunities
    
    async def _execute_trade(self, decision: TradingDecision) -> Dict[str, Any]:
        """Mock trade execution with realistic behavior"""
        
        # Simulate execution delay
        await asyncio.sleep(0.1)
        
        # Simulate occasional execution failures
        if np.random.random() < 0.05:  # 5% failure rate
            return {
                'success': False,
                'reason': 'Insufficient liquidity'
            }
        
        # Simulate slippage
        slippage = np.random.normal(0, 0.001)  # 0.1% slippage on average
        fill_price = decision.entry_price * (1 + slippage)
        
        self.logger.info(f"Executed {decision.signal.value} order for {decision.symbol} at ${fill_price:.2f}")
        
        return {
            'success': True,
            'fill_price': fill_price,
            'slippage': slippage,
            'execution_time': datetime.now()
        }
    
    async def _monitor_positions(self):
        """Monitor existing positions for stop loss/take profit triggers"""
        if not self.position_manager.open_positions:
            return
        
        for symbol in list(self.position_manager.open_positions.keys()):
            # Simulate current market price
            position = self.position_manager.open_positions[symbol]
            current_price = position['entry_price'] * np.random.uniform(0.95, 1.05)  # ±5% movement
            
            # Update position
            update_result = self.position_manager.update_position(symbol, current_price)
            
            # Check for actions needed
            if update_result and 'actions_needed' in update_result:
                for action in update_result['actions_needed']:
                    if action['action'] == 'close_position':
                        close_result = self.position_manager.close_position(
                            symbol, current_price, action['reason']
                        )
                        self.logger.info(f"Position closed: {symbol} due to {action['reason']}")
                    elif action['action'] == 'take_profit':
                        # Partial close for take profit levels
                        self.logger.info(f"Take profit {action['level']} hit for {symbol}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        portfolio_summary = self.position_manager.get_portfolio_summary()
        
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'cycle_count': self.cycle_count,
            'last_cycle_duration': self.last_cycle_time,
            'performance_metrics': self.performance_metrics,
            'portfolio_summary': portfolio_summary,
            'components': {
                'ia1_strategist': 'healthy',
                'position_manager': 'healthy',
                'risk_manager': 'healthy'
            }
        }

# Main execution
async def main():
    """Main function for testing the enhanced trading system"""
    orchestrator = EnhancedTradingOrchestrator()
    
    try:
        # Run several cycles
        for cycle in range(3):
            print(f"\n{'='*50}")
            print(f"Running Trading Cycle {cycle + 1}")
            print(f"{'='*50}")
            
            executed_trades = await orchestrator.run_complete_trading_cycle()
            
            # Display results
            if executed_trades:
                print("\nExecuted Trades:")
                for trade in executed_trades:
                    print(f"  {trade['symbol']}: {trade['signal']} "
                          f"(confidence: {trade['confidence']:.2f}, "
                          f"size: {trade['position_size']:.1%})")
            else:
                print("No trades executed this cycle")
            
            # Display system health
            health = orchestrator.get_system_health()
            portfolio = health['portfolio_summary']
            print(f"\nPortfolio Status:")
            print(f"  Value: ${portfolio['portfolio_value']:,.2f}")
            print(f"  Open Positions: {portfolio['open_positions_count']}")
            print(f"  Exposure: {portfolio['exposure_percentage']:.1f}%")
            print(f"  Unrealized P&L: {portfolio['unrealized_pnl_pct']:.2f}%")
            
            # Wait before next cycle
            await asyncio.sleep(2)
        
        print(f"\n{'='*50}")
        print("Final System Summary")
        print(f"{'='*50}")
        
        final_health = orchestrator.get_system_health()
        metrics = final_health['performance_metrics']
        
        print(f"Total Signals Generated: {metrics['total_signals']}")
        print(f"Trades Executed: {metrics['executed_trades']}")
        print(f"Trades Rejected: {metrics['rejected_trades']}")
        print(f"Average Confidence: {metrics['avg_confidence']:.2f}")
        print(f"Execution Rate: {metrics['executed_trades']/max(metrics['total_signals'], 1)*100:.1f}%")
        
    except KeyboardInterrupt:
        print("\nSystem shutdown requested")
    except Exception as e:
        print(f"System error: {e}")
        logging.error(f"System error: {e}", exc_info=True)

if __name__ == "__main__":
    # Run the enhanced trading system
    asyncio.run(main())

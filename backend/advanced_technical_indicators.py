"""
ADVANCED TECHNICAL INDICATORS SYSTEM - REFORGED EDITION
Complete technical analysis system for IA1 with:
- ATR, ADX, MACD, Bollinger Bands, VWAP, RSI, Volume, Market Regime
- Enhanced 10-regime detection
- Multi-timeframe analysis
- Trading implications
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES (Critical for system compatibility)
# =============================================================================

@dataclass
class TechnicalIndicators:
    """Complete technical indicators structure for IA1"""
    # RSI
    rsi_14: float = 0.0
    rsi_9: float = 0.0
    rsi_21: float = 0.0
    rsi_divergence: bool = False
    rsi_overbought: bool = False
    rsi_oversold: bool = False
    
    # MACD
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    macd_bullish_crossover: bool = False
    macd_bearish_crossover: bool = False
    macd_above_zero: bool = False
    
    # Stochastic
    stoch_k: float = 0.0
    stoch_d: float = 0.0
    stoch_overbought: bool = False
    stoch_oversold: bool = False
    stoch_bullish_crossover: bool = False
    stoch_bearish_crossover: bool = False
    
    # Bollinger Bands
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0
    bb_position: float = 0.0
    bb_squeeze: bool = False
    bb_expansion: bool = False
    
    # ATR (Volatility)
    atr: float = 0.0
    atr_percentage: float = 0.0
    volatility_ratio: float = 1.0
    
    # ADX (Trend Strength)
    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0
    adx_trend: str = "WEAK"  # WEAK, MODERATE, STRONG
    
    # VWAP
    vwap: float = 0.0
    vwap_distance: float = 0.0
    above_vwap: bool = False
    
    # Volume
    volume_ratio: float = 1.0
    volume_trend: str = "NEUTRAL"
    volume_surge: bool = False
    
    # Market Regime
    market_regime: str = "RANGING"
    regime_confidence: float = 0.5
    trading_implications: List[str] = None
    
    # Multi-timeframe
    trend_alignment: str = "MIXED"
    timeframe_score: float = 0.5


@dataclass
class IndicatorSignal:
    """Signal with direction and strength"""
    direction: str  # "BULLISH", "BEARISH", "NEUTRAL"
    strength: float  # 0-1
    confidence: float  # 0-1
    reasoning: str


# =============================================================================
# MARKET REGIME DETECTION (Enhanced from v3)
# =============================================================================

class MarketRegimeDetailed(Enum):
    """Detailed market regimes for AI analysis"""
    TRENDING_UP_STRONG = "TRENDING_UP_STRONG"
    TRENDING_UP_MODERATE = "TRENDING_UP_MODERATE"
    TRENDING_DOWN_STRONG = "TRENDING_DOWN_STRONG"
    TRENDING_DOWN_MODERATE = "TRENDING_DOWN_MODERATE"
    RANGING_TIGHT = "RANGING_TIGHT"
    RANGING_WIDE = "RANGING_WIDE"
    CONSOLIDATION = "CONSOLIDATION"
    BREAKOUT_BULLISH = "BREAKOUT_BULLISH"
    BREAKOUT_BEARISH = "BREAKOUT_BEARISH"
    VOLATILE = "VOLATILE"


class AdvancedRegimeDetector:
    """
    Advanced market regime detector with 10 detailed classifications
    Enhanced with dynamic thresholds, persistence tracking, and transition detection
    """
    
    def __init__(self, lookback_period: int = 50, persistence_length: int = 5):
        self.lookback_period = lookback_period
        self.persistence_length = persistence_length
        self.regime_history = []  # Track last N regimes for persistence
    
    def detect_detailed_regime(self, df: pd.DataFrame) -> Dict:
        """Main regime detection with full analysis, persistence, and transitions"""
        if len(df) < self.lookback_period:
            return self._get_default_regime()
        
        try:
            # 1. Calculate dynamic thresholds based on asset characteristics
            thresholds = self._get_dynamic_thresholds(df)
            
            # 2. Calculate regime indicators
            indicators = self._calculate_regime_indicators(df, thresholds)
            
            # 3. Classify regime
            regime, base_confidence, scores = self._classify_regime(indicators, thresholds)
            
            # 4. Check for regime transitions
            previous_regime = self.regime_history[-1] if self.regime_history else regime
            transition_detected = self._detect_regime_transition(regime, previous_regime, indicators)
            if transition_detected:
                regime = transition_detected
                logger.info(f"🔄 Regime transition detected: {previous_regime.value} → {regime.value}")
            
            # 5. Assess persistence and adjust confidence
            persistence_score = self._assess_regime_persistence(regime)
            adjusted_confidence = base_confidence * (0.7 + 0.3 * persistence_score)
            
            # 6. Update history
            self.regime_history.append(regime)
            if len(self.regime_history) > self.persistence_length:
                self.regime_history.pop(0)
            
            # 7. Calculate signal consistency
            signal_consistency = self._calculate_signal_consistency(indicators)
            final_confidence = adjusted_confidence * signal_consistency
            
            return {
                'regime': regime.value,
                'confidence': round(final_confidence, 3),
                'base_confidence': round(base_confidence, 3),
                'persistence_score': round(persistence_score, 3),
                'signal_consistency': round(signal_consistency, 3),
                'scores': scores,
                'indicators': indicators,
                'thresholds': thresholds,
                'transition_detected': transition_detected is not None,
                'interpretation': self._interpret_regime(regime, final_confidence),
                'trading_implications': self._get_trading_implications(regime)
            }
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return self._get_default_regime()
    
    def _get_dynamic_thresholds(self, df: pd.DataFrame) -> Dict:
        """Calculate dynamic thresholds based on asset volatility"""
        close = df['close']
        volatility = close.pct_change().std()
        
        return {
            'bb_squeeze': 0.015 + volatility * 10,  # Dynamic BB squeeze threshold
            'bb_expansion': 0.045 + volatility * 15,  # Dynamic expansion threshold
            'adx_strong': max(20, min(30, 20 + volatility * 50)),  # Adaptive ADX
            'adx_weak': max(15, min(25, 15 + volatility * 30)),
            'volume_surge': 1.8 + volatility * 5,  # Adaptive volume surge
            'slope_strong': 0.001 + volatility * 2,  # Adaptive slope thresholds
            'range_tight': max(3, min(7, 5 - volatility * 100)),  # Adaptive range
            'range_wide': max(8, min(15, 10 + volatility * 100))
        }
    
    def _calculate_regime_indicators(self, df: pd.DataFrame, thresholds: Dict) -> Dict:
        """Calculate all indicators for regime classification"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        indicators = {}
        
        # 1. TREND INDICATORS
        indicators['adx'] = self._calculate_adx(df, 14)
        indicators['adx_strength'] = 'STRONG' if indicators['adx'] > 25 else 'WEAK' if indicators['adx'] < 20 else 'MODERATE'
        
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        indicators['sma_20_slope'] = self._calculate_slope(sma_20.tail(10))
        indicators['sma_50_slope'] = self._calculate_slope(sma_50.tail(20))
        indicators['above_sma_20'] = (close.iloc[-1] > sma_20.iloc[-1])
        indicators['above_sma_50'] = (close.iloc[-1] > sma_50.iloc[-1])
        
        # 2. RANGE/CONSOLIDATION INDICATORS
        atr = self._calculate_atr(df, 14)
        indicators['atr_pct'] = (atr.iloc[-1] / close.iloc[-1]) * 100
        
        atr_50 = self._calculate_atr(df, 50)
        indicators['volatility_ratio'] = atr.iloc[-1] / atr_50.iloc[-1] if atr_50.iloc[-1] > 0 else 1.0
        
        bb_width = self._calculate_bb_width(df, 20)
        indicators['bb_width'] = bb_width.iloc[-1]
        indicators['bb_squeeze'] = bb_width.iloc[-1] < thresholds['bb_squeeze']
        indicators['bb_expansion'] = bb_width.iloc[-1] > thresholds['bb_expansion']
        
        high_20 = high.rolling(20).max()
        low_20 = low.rolling(20).min()
        indicators['range_pct'] = ((high_20.iloc[-1] - low_20.iloc[-1]) / close.iloc[-1]) * 100
        
        # 3. MOMENTUM INDICATORS
        rsi = self._calculate_rsi(close, 14)
        indicators['rsi'] = rsi.iloc[-1]
        indicators['rsi_trend'] = self._calculate_slope(rsi.tail(10))
        
        macd, macd_signal = self._calculate_macd(close)
        indicators['macd'] = macd.iloc[-1]
        indicators['macd_signal'] = macd_signal.iloc[-1]
        indicators['macd_histogram'] = macd.iloc[-1] - macd_signal.iloc[-1]
        
        # 4. VOLUME INDICATORS
        if 'volume' in df.columns:
            volume_sma = df['volume'].rolling(20).mean()
            indicators['volume_ratio'] = df['volume'].iloc[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1.0
        else:
            indicators['volume_ratio'] = 1.0
        
        return indicators
    
    def _classify_regime(self, ind: Dict, thresholds: Dict) -> Tuple[MarketRegimeDetailed, float, Dict]:
        """Classify market regime based on indicators"""
        scores = {regime: 0 for regime in MarketRegimeDetailed}
        
        # TRENDING UP STRONG (use dynamic thresholds)
        if (ind['adx'] > thresholds['adx_strong'] and 
            ind['sma_20_slope'] > thresholds['slope_strong'] and 
            ind['above_sma_20'] and ind['above_sma_50']):
            scores[MarketRegimeDetailed.TRENDING_UP_STRONG] += 3
            if ind['rsi'] > 50 and ind['macd_histogram'] > 0:
                scores[MarketRegimeDetailed.TRENDING_UP_STRONG] += 2
        
        # TRENDING UP MODERATE
        elif (ind['above_sma_20'] and 
              0 < ind['sma_20_slope'] <= thresholds['slope_strong']):
            scores[MarketRegimeDetailed.TRENDING_UP_MODERATE] += 3
            if ind['adx'] > thresholds['adx_weak']:
                scores[MarketRegimeDetailed.TRENDING_UP_MODERATE] += 1
        
        # TRENDING DOWN STRONG (use dynamic thresholds)
        if (ind['adx'] > thresholds['adx_strong'] and 
            ind['sma_20_slope'] < -thresholds['slope_strong'] and 
            not ind['above_sma_20'] and not ind['above_sma_50']):
            scores[MarketRegimeDetailed.TRENDING_DOWN_STRONG] += 3
            if ind['rsi'] < 50 and ind['macd_histogram'] < 0:
                scores[MarketRegimeDetailed.TRENDING_DOWN_STRONG] += 2
        
        # TRENDING DOWN MODERATE
        elif (not ind['above_sma_20'] and 
              -thresholds['slope_strong'] <= ind['sma_20_slope'] < 0):
            scores[MarketRegimeDetailed.TRENDING_DOWN_MODERATE] += 3
            if ind['adx'] > thresholds['adx_weak']:
                scores[MarketRegimeDetailed.TRENDING_DOWN_MODERATE] += 1
        
        # CONSOLIDATION (BB Squeeze)
        if ind['bb_squeeze']:
            scores[MarketRegimeDetailed.CONSOLIDATION] += 4
            if abs(ind['sma_20_slope']) < 0.001:
                scores[MarketRegimeDetailed.CONSOLIDATION] += 2
        
        # RANGING TIGHT
        if (ind['range_pct'] < 5 and ind['adx'] < 20 and not ind['bb_squeeze']):
            scores[MarketRegimeDetailed.RANGING_TIGHT] += 3
            if abs(ind['sma_20_slope']) < 0.001:
                scores[MarketRegimeDetailed.RANGING_TIGHT] += 2
        
        # RANGING WIDE
        if (5 <= ind['range_pct'] < 10 and ind['adx'] < 25):
            scores[MarketRegimeDetailed.RANGING_WIDE] += 3
        
        # BREAKOUT BULLISH
        if (ind['volume_ratio'] > 1.5 and ind['sma_20_slope'] > 0.003 and 
            ind['above_sma_20'] and ind['volatility_ratio'] > 1.2):
            scores[MarketRegimeDetailed.BREAKOUT_BULLISH] += 4
        
        # BREAKOUT BEARISH
        if (ind['volume_ratio'] > 1.5 and ind['sma_20_slope'] < -0.003 and 
            not ind['above_sma_20'] and ind['volatility_ratio'] > 1.2):
            scores[MarketRegimeDetailed.BREAKOUT_BEARISH] += 4
        
        # VOLATILE
        if (ind['volatility_ratio'] > 1.5 and ind['range_pct'] > 10):
            scores[MarketRegimeDetailed.VOLATILE] += 3
            if ind['adx'] < 20:
                scores[MarketRegimeDetailed.VOLATILE] += 2
        
        # Determine winning regime
        max_score = max(scores.values())
        if max_score == 0:
            return MarketRegimeDetailed.RANGING_TIGHT, 0.5, scores
        
        regime = max(scores, key=scores.get)
        confidence = scores[regime] / max(sum(scores.values()), 1)
        
        return regime, confidence, scores
    
    def _interpret_regime(self, regime: MarketRegimeDetailed, confidence: float) -> str:
        """Human-readable regime interpretation"""
        interpretations = {
            MarketRegimeDetailed.TRENDING_UP_STRONG: f"🚀 Strong uptrend (confidence: {confidence:.1%})",
            MarketRegimeDetailed.TRENDING_UP_MODERATE: f"📈 Moderate uptrend (confidence: {confidence:.1%})",
            MarketRegimeDetailed.TRENDING_DOWN_STRONG: f"📉 Strong downtrend (confidence: {confidence:.1%})",
            MarketRegimeDetailed.TRENDING_DOWN_MODERATE: f"📊 Moderate downtrend (confidence: {confidence:.1%})",
            MarketRegimeDetailed.RANGING_TIGHT: f"🔄 Tight range (confidence: {confidence:.1%})",
            MarketRegimeDetailed.RANGING_WIDE: f"↔️ Wide range (confidence: {confidence:.1%})",
            MarketRegimeDetailed.CONSOLIDATION: f"⏳ Consolidation (confidence: {confidence:.1%})",
            MarketRegimeDetailed.BREAKOUT_BULLISH: f"🎯 Bullish breakout (confidence: {confidence:.1%})",
            MarketRegimeDetailed.BREAKOUT_BEARISH: f"⚠️ Bearish breakout (confidence: {confidence:.1%})",
            MarketRegimeDetailed.VOLATILE: f"⚡ Volatile market (confidence: {confidence:.1%})"
        }
        return interpretations.get(regime, f"Unknown regime (confidence: {confidence:.1%})")
    
    def _get_trading_implications(self, regime: MarketRegimeDetailed) -> List[str]:
        """Trading implications for each regime"""
        implications = {
            MarketRegimeDetailed.TRENDING_UP_STRONG: [
                "✅ Long positions favored",
                "✅ Buy on pullbacks",
                "✅ Wide stop-losses",
                "✅ High targets"
            ],
            MarketRegimeDetailed.TRENDING_UP_MODERATE: [
                "✅ Moderate long positions",
                "⚠️ Tight risk management",
                "✅ Conservative stop-losses",
                "✅ Moderate targets"
            ],
            MarketRegimeDetailed.TRENDING_DOWN_STRONG: [
                "✅ Short positions favored",
                "✅ Sell on rallies",
                "✅ Wide stop-losses",
                "✅ High targets"
            ],
            MarketRegimeDetailed.TRENDING_DOWN_MODERATE: [
                "✅ Moderate short positions",
                "⚠️ Tight risk management",
                "✅ Conservative stop-losses",
                "✅ Moderate targets"
            ],
            MarketRegimeDetailed.RANGING_TIGHT: [
                "🔄 Range trading",
                "✅ Buy support, sell resistance",
                "⚠️ Avoid breakout trades",
                "✅ Reduced position sizing"
            ],
            MarketRegimeDetailed.RANGING_WIDE: [
                "🔄 Range trading moderate",
                "✅ Breakout trades possible",
                "⚠️ Wide stop-losses needed",
                "✅ Monitor for breakouts"
            ],
            MarketRegimeDetailed.CONSOLIDATION: [
                "⏳ Accumulation phase",
                "🎯 Prepare for breakout",
                "✅ Reduced position sizing",
                "⚠️ Avoid directional trades"
            ],
            MarketRegimeDetailed.BREAKOUT_BULLISH: [
                "🎯 Long entry on confirmation",
                "✅ Stop-loss below breakout",
                "✅ Target based on previous range",
                "⚠️ Verify volume"
            ],
            MarketRegimeDetailed.BREAKOUT_BEARISH: [
                "🎯 Short entry on confirmation",
                "✅ Stop-loss above breakout",
                "✅ Target based on previous range",
                "⚠️ Verify volume"
            ],
            MarketRegimeDetailed.VOLATILE: [
                "⚡ High risk",
                "✅ Reduced position sizing",
                "⚠️ Tight stop-losses mandatory",
                "🎯 Short-term trading only"
            ]
        }
        return implications.get(regime, ["Standard strategy recommended"])
    
    def _assess_regime_persistence(self, current_regime: MarketRegimeDetailed) -> float:
        """
        Assess regime persistence over recent history
        Returns score 0-1 (1 = high persistence, 0 = unstable)
        """
        if len(self.regime_history) < 2:
            return 0.5  # Neutral if no history
        
        # Count how many recent regimes match current
        recent = self.regime_history[-min(len(self.regime_history), self.persistence_length):]
        same_count = sum(1 for r in recent if r == current_regime)
        persistence = same_count / len(recent)
        
        return persistence
    
    def _detect_regime_transition(self, current_regime: MarketRegimeDetailed, 
                                  previous_regime: MarketRegimeDetailed,
                                  indicators: Dict) -> Optional[MarketRegimeDetailed]:
        """
        Detect transitions between regimes (especially breakouts from consolidation)
        Returns new regime if transition detected, None otherwise
        """
        # Breakout from consolidation
        if previous_regime in [MarketRegimeDetailed.CONSOLIDATION, 
                              MarketRegimeDetailed.RANGING_TIGHT]:
            
            # Check for volume surge + volatility expansion
            if (indicators['volume_ratio'] > 1.5 and 
                indicators.get('bb_expansion', False) and
                abs(indicators['macd_histogram']) > 0.5):
                
                # Determine breakout direction
                if (indicators['macd_histogram'] > 0 and 
                    indicators['sma_20_slope'] > 0 and
                    indicators['above_sma_20']):
                    return MarketRegimeDetailed.BREAKOUT_BULLISH
                
                elif (indicators['macd_histogram'] < 0 and 
                      indicators['sma_20_slope'] < 0 and
                      not indicators['above_sma_20']):
                    return MarketRegimeDetailed.BREAKOUT_BEARISH
        
        # Reversal from strong trend to consolidation
        if previous_regime in [MarketRegimeDetailed.TRENDING_UP_STRONG,
                              MarketRegimeDetailed.TRENDING_DOWN_STRONG]:
            if (indicators['adx'] < 20 and 
                indicators.get('bb_squeeze', False) and
                abs(indicators['sma_20_slope']) < 0.001):
                return MarketRegimeDetailed.CONSOLIDATION
        
        return None
    
    def _calculate_signal_consistency(self, indicators: Dict) -> float:
        """
        Calculate consistency between different indicators
        Returns score 0-1 (1 = all signals aligned, 0 = conflicting)
        """
        consistency_score = 0.0
        total_checks = 0
        
        # 1. Trend consistency (ADX + SMA alignment)
        if indicators['adx'] > 20:
            if indicators['sma_20_slope'] > 0 and indicators['above_sma_20']:
                consistency_score += 1.0
            elif indicators['sma_20_slope'] < 0 and not indicators['above_sma_20']:
                consistency_score += 1.0
            else:
                consistency_score += 0.3  # Partial alignment
        else:
            consistency_score += 0.5  # Neutral for ranging
        total_checks += 1
        
        # 2. Momentum consistency (RSI + MACD alignment)
        rsi_bullish = indicators['rsi'] > 50
        macd_bullish = indicators['macd_histogram'] > 0
        if rsi_bullish == macd_bullish:
            consistency_score += 1.0
        else:
            consistency_score += 0.2  # Slight misalignment penalty
        total_checks += 1
        
        # 3. Volume consistency (Volume confirms price movement)
        if indicators['volume_ratio'] > 1.2:
            if indicators['sma_20_slope'] * indicators['macd_histogram'] > 0:
                consistency_score += 1.0  # Volume confirms direction
            else:
                consistency_score += 0.4  # Volume but conflicting signals
        else:
            consistency_score += 0.6  # Normal volume, neutral
        total_checks += 1
        
        # 4. Volatility consistency
        if 0.8 < indicators['volatility_ratio'] < 1.5:
            consistency_score += 1.0  # Normal volatility
        elif indicators['volatility_ratio'] > 1.5 and indicators.get('bb_expansion', False):
            consistency_score += 0.8  # High but expanding (expected)
        else:
            consistency_score += 0.5  # Unusual volatility
        total_checks += 1
        
        return consistency_score / total_checks
    
    # Helper calculation methods
    def _calculate_slope(self, series: pd.Series) -> float:
        """Calculate slope of a series"""
        if len(series) < 2:
            return 0.0
        x = np.arange(len(series))
        y = series.values
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    
    def _calculate_adx(self, df: pd.DataFrame, period: int) -> float:
        """
        Correct ADX calculation using Wilder's method
        Returns proper ADX value (0-100)
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        
        # Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        dm_plus[dm_plus < 0] = 0
        dm_minus[dm_minus < 0] = 0
        dm_plus[(dm_plus < dm_minus)] = 0
        dm_minus[(dm_minus < dm_plus)] = 0
        
        # Smoothed directional movement
        dm_plus_smooth = dm_plus.ewm(span=period, adjust=False).mean()
        dm_minus_smooth = dm_minus.ewm(span=period, adjust=False).mean()
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # DX (Directional Index)
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 0.0001)
        
        # ADX (smoothed DX)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()
    
    def _calculate_bb_width(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Bollinger Bands width as percentage"""
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return (upper - lower) / sma
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 0.001)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        return macd, macd_signal
    
    def _get_default_regime(self) -> Dict:
        """Default regime when insufficient data"""
        return {
            'regime': MarketRegimeDetailed.VOLATILE.value,
            'confidence': 0.1,
            'scores': {},
            'indicators': {},
            'interpretation': "Insufficient data for analysis",
            'trading_implications': ["Wait for more data before trading"]
        }


# =============================================================================
# MAIN TECHNICAL INDICATORS CLASS
# =============================================================================

class AdvancedTechnicalIndicators:
    """
    Complete technical analysis system for IA1
    Calculates: ATR, ADX, MACD, Bollinger Bands, VWAP, RSI, Volume, Market Regime
    """
    
    def __init__(self):
        self.regime_detector = AdvancedRegimeDetector(lookback_period=50)
        logger.info("✅ Advanced Technical Indicators System initialized")
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """
        Calculate ALL technical indicators for IA1
        Returns complete TechnicalIndicators dataclass
        """
        try:
            if len(df) < 50:
                return TechnicalIndicators()
            
            close = df['close']
            high = df['high']
            low = df['low']
            
            # 1. RSI (Multiple periods)
            rsi_14 = self._calculate_rsi(close, 14).iloc[-1]
            rsi_9 = self._calculate_rsi(close, 9).iloc[-1]
            rsi_21 = self._calculate_rsi(close, 21).iloc[-1]
            
            # 2. MACD
            macd, macd_signal = self._calculate_macd(close)
            macd_line = macd.iloc[-1]
            macd_sig = macd_signal.iloc[-1]
            macd_hist = macd_line - macd_sig
            
            # Check for crossovers
            macd_bullish_cross = (macd.iloc[-2] < macd_signal.iloc[-2] and 
                                 macd.iloc[-1] > macd_signal.iloc[-1])
            macd_bearish_cross = (macd.iloc[-2] > macd_signal.iloc[-2] and 
                                 macd.iloc[-1] < macd_signal.iloc[-1])
            
            # 3. Stochastic
            stoch_k, stoch_d = self._calculate_stochastic(df, 14, 3, 3)
            
            # 4. Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close, 20, 2)
            bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
            bb_position = ((close.iloc[-1] - bb_lower) / (bb_upper - bb_lower) 
                          if bb_upper != bb_lower else 0.5)
            
            # BB Squeeze detection
            bb_width_series = self._calculate_bb_width_series(df, 20)
            bb_avg_width = bb_width_series.rolling(20).mean().iloc[-1]
            bb_squeeze = bb_width < bb_avg_width * 0.8
            bb_expansion = bb_width > bb_avg_width * 1.2
            
            # 5. ATR (Volatility)
            atr_series = self._calculate_atr(df, 14)
            atr = atr_series.iloc[-1]
            atr_pct = (atr / close.iloc[-1]) * 100
            
            atr_50 = self._calculate_atr(df, 50).iloc[-1]
            volatility_ratio = atr / atr_50 if atr_50 > 0 else 1.0
            
            # 6. ADX (Trend Strength)
            adx, plus_di, minus_di = self._calculate_adx_full(df, 14)
            adx_trend = "STRONG" if adx > 25 else "WEAK" if adx < 20 else "MODERATE"
            
            # 7. VWAP
            vwap = self._calculate_vwap(df)
            vwap_distance = ((close.iloc[-1] - vwap) / vwap) * 100 if vwap > 0 else 0
            above_vwap = close.iloc[-1] > vwap
            
            # 8. Volume
            if 'volume' in df.columns:
                volume_sma = df['volume'].rolling(20).mean().iloc[-1]
                volume_ratio = df['volume'].iloc[-1] / volume_sma if volume_sma > 0 else 1.0
                volume_trend = "INCREASING" if volume_ratio > 1.2 else "DECREASING" if volume_ratio < 0.8 else "NEUTRAL"
                volume_surge = volume_ratio > 2.0
            else:
                volume_ratio = 1.0
                volume_trend = "NEUTRAL"
                volume_surge = False
            
            # 9. Market Regime (10 detailed classifications)
            regime_data = self.regime_detector.detect_detailed_regime(df)
            
            # 10. Multi-timeframe trend alignment
            trend_alignment, tf_score = self._calculate_trend_alignment(df)
            
            return TechnicalIndicators(
                # RSI
                rsi_14=rsi_14,
                rsi_9=rsi_9,
                rsi_21=rsi_21,
                rsi_divergence=False,  # Can be enhanced later
                rsi_overbought=(rsi_14 > 70),
                rsi_oversold=(rsi_14 < 30),
                
                # MACD
                macd_line=macd_line,
                macd_signal=macd_sig,
                macd_histogram=macd_hist,
                macd_bullish_crossover=macd_bullish_cross,
                macd_bearish_crossover=macd_bearish_cross,
                macd_above_zero=(macd_line > 0),
                
                # Stochastic
                stoch_k=stoch_k,
                stoch_d=stoch_d,
                stoch_overbought=(stoch_k > 80),
                stoch_oversold=(stoch_k < 20),
                stoch_bullish_crossover=False,  # Can be enhanced
                stoch_bearish_crossover=False,
                
                # Bollinger Bands
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                bb_width=bb_width,
                bb_position=bb_position,
                bb_squeeze=bb_squeeze,
                bb_expansion=bb_expansion,
                
                # ATR
                atr=atr,
                atr_percentage=atr_pct,
                volatility_ratio=volatility_ratio,
                
                # ADX
                adx=adx,
                plus_di=plus_di,
                minus_di=minus_di,
                adx_trend=adx_trend,
                
                # VWAP
                vwap=vwap,
                vwap_distance=vwap_distance,
                above_vwap=above_vwap,
                
                # Volume
                volume_ratio=volume_ratio,
                volume_trend=volume_trend,
                volume_surge=volume_surge,
                
                # Market Regime
                market_regime=regime_data['regime'],
                regime_confidence=regime_data['confidence'],
                trading_implications=regime_data['trading_implications'],
                
                # Multi-timeframe
                trend_alignment=trend_alignment,
                timeframe_score=tf_score
            )
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return TechnicalIndicators()
    
    # =========================================================================
    # INDIVIDUAL INDICATOR CALCULATIONS
    # =========================================================================
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 0.001)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_macd(self, prices: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period=14, k_smooth=3, d_smooth=3) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        k_raw = 100 * ((df['close'] - low_min) / (high_max - low_min))
        k = k_raw.rolling(window=k_smooth).mean()
        d = k.rolling(window=d_smooth).mean()
        
        return float(k.iloc[-1]), float(d.iloc[-1])
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period=20, std_dev=2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return float(upper.iloc[-1]), float(middle.iloc[-1]), float(lower.iloc[-1])
    
    def _calculate_bb_width_series(self, df: pd.DataFrame, period=20) -> pd.Series:
        """Calculate Bollinger Band width as series"""
        close = df['close']
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return (upper - lower) / sma
    
    def _calculate_atr(self, df: pd.DataFrame, period=14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()
    
    def _calculate_adx_full(self, df: pd.DataFrame, period=14) -> Tuple[float, float, float]:
        """Calculate ADX with +DI and -DI"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        
        # Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        dm_plus[dm_plus < 0] = 0
        dm_minus[dm_minus < 0] = 0
        dm_plus[(dm_plus < dm_minus)] = 0
        dm_minus[(dm_minus < dm_plus)] = 0
        
        # Directional Indicators
        di_plus = 100 * dm_plus.ewm(span=period, adjust=False).mean() / atr
        di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr
        
        # DX and ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return (float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0,
                float(di_plus.iloc[-1]) if not pd.isna(di_plus.iloc[-1]) else 0.0,
                float(di_minus.iloc[-1]) if not pd.isna(di_minus.iloc[-1]) else 0.0)
    
    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price"""
        if 'volume' not in df.columns:
            return float(df['close'].iloc[-1])
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).sum() / df['volume'].sum()
        return float(vwap)
    
    def _calculate_trend_alignment(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Calculate multi-timeframe trend alignment"""
        close = df['close']
        
        # Multiple EMAs
        ema_9 = close.ewm(span=9).mean().iloc[-1]
        ema_21 = close.ewm(span=21).mean().iloc[-1]
        ema_50 = close.ewm(span=50).mean().iloc[-1]
        
        current_price = close.iloc[-1]
        
        # Score based on alignment
        score = 0
        if current_price > ema_9: score += 1
        if current_price > ema_21: score += 1
        if current_price > ema_50: score += 1
        if ema_9 > ema_21: score += 1
        if ema_21 > ema_50: score += 1
        
        tf_score = score / 5.0
        
        if score >= 4:
            alignment = "STRONG_BULLISH"
        elif score == 3:
            alignment = "BULLISH"
        elif score == 2:
            alignment = "MIXED"
        elif score == 1:
            alignment = "BEARISH"
        else:
            alignment = "STRONG_BEARISH"
        
        return alignment, tf_score
    
    # =========================================================================
    # ENHANCED S/R INTEGRATION (from our previous work)
    # =========================================================================
    
    def calculate_enhanced_sr_levels(self, df: pd.DataFrame) -> Dict:
        """Calculate enhanced S/R levels (integration with enhanced_support_resistance.py)"""
        try:
            from enhanced_support_resistance import enhanced_sr_calculator
            return enhanced_sr_calculator.calculate_enhanced_support_resistance(df)
        except Exception as e:
            logger.error(f"Error calculating enhanced S/R: {e}")
            current_price = df['close'].iloc[-1]
            return {
                'supports': [],
                'resistances': [],
                'current_price': current_price,
                'primary_support': current_price * 0.95,
                'primary_resistance': current_price * 1.05,
                'support_strength': 0.5,
                'resistance_strength': 0.5
            }
    
    def calculate_context_aware_rr(self, df: pd.DataFrame, direction: str, 
                                   entry_price: Optional[float] = None) -> Dict:
        """Calculate context-aware RR (integration with enhanced_support_resistance.py)"""
        try:
            from enhanced_support_resistance import enhanced_sr_calculator
            sr_levels = self.calculate_enhanced_sr_levels(df)
            rr_analysis = enhanced_sr_calculator.calculate_enhanced_rr(
                df, direction, entry_price, sr_levels
            )
            
            return {
                'entry_price': rr_analysis.entry_price,
                'stop_loss': rr_analysis.stop_loss,
                'take_profit': rr_analysis.take_profit,
                'risk': rr_analysis.risk,
                'reward': rr_analysis.reward,
                'rr_ratio': rr_analysis.rr_ratio,
                'confidence_score': rr_analysis.confidence_score,
                'trade_valid': rr_analysis.trade_valid,
                'recommendation': rr_analysis.recommendation,
                'reasoning': rr_analysis.reasoning,
                'market_context': rr_analysis.market_context
            }
        except Exception as e:
            logger.error(f"Error calculating context-aware RR: {e}")
            return {'error': str(e)}


# =============================================================================
# GLOBAL INSTANCES (for backward compatibility)
# =============================================================================

advanced_indicators = AdvancedTechnicalIndicators()
advanced_technical_indicators = advanced_indicators  # Alias for compatibility

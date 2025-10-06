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
    
    # VWAP Position (replaces MFI for volume-price analysis)
    vwap_position: str = "NEUTRAL"
    
    # Market Regime
    market_regime: str = "RANGING"
    regime_confidence: float = 0.5
    trading_implications: List[str] = None
    
    # Multi-timeframe
    trend_alignment: str = "MIXED"
    timeframe_score: float = 0.5
    
    # EMAs for trend analysis
    ema_9: float = 0.0
    ema_21: float = 0.0
    ema_50: float = 0.0
    ema_200: float = 0.0
    sma_50: float = 0.0
    
    # Trend hierarchy analysis
    trend_hierarchy: str = "neutral"
    trend_momentum: str = "neutral"
    price_vs_emas: str = "mixed"
    ema_cross_signal: str = "none"
    trend_strength_score: float = 0.5
    
    # Trade Type Recommendation (NEW)
    trade_type: str = "SWING"  # SCALP, INTRADAY, SWING, POSITION
    trade_duration_estimate: str = "1-3 days"
    optimal_timeframe: str = "4H"  # Optimal chart timeframe for this setup
    minimum_rr_threshold: float = 2.0  # Minimum RR required for this trade type


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
    MERGED with v4: bar counter, stability score, ML multipliers
    """
    
    def __init__(self, lookback_period: int = 50, history_size: int = 200):
        self.lookback_period = lookback_period
        self.regime_history = []  # Track last N regimes for persistence
        self.current_regime = None  # v4: Current regime tracking
        self.regime_start_bar = 0  # v4: Bar counter for persistence
        self.bar_count = 0  # v4: Total bar count
    
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
            
            # 5. Calculate regime persistence (v4: bar counter method)
            persistence = self._calculate_regime_persistence_v4(regime)
            
            # 6. Calculate stability score (v4: NEW)
            stability_score = self._calculate_stability_score()
            
            # 7. Adjust confidence with persistence (v4 formula)
            combined_confidence = 0.7 * base_confidence + 0.3 * adjusted_confidence
            
            # 8. Apply stability adjustment (v4)
            final_confidence = combined_confidence * (0.9 + 0.1 * stability_score)
            final_confidence = min(1.0, final_confidence)
            
            # 9. Update history
            self._update_regime_history(regime)
            
            return {
                'regime': regime.value,
                'confidence': round(final_confidence, 3),
                'base_confidence': round(base_confidence, 3),
                'technical_consistency': round(adjusted_confidence, 3),
                'combined_confidence': round(combined_confidence, 3),
                'regime_persistence': persistence,  # v4: bar count
                'stability_score': round(stability_score, 3),  # v4: NEW
                'regime_transition_alert': self._detect_regime_transition_v4(),  # v4: enhanced
                'scores': scores,
                'indicators': indicators,
                'thresholds': thresholds,
                'transition_detected': transition_detected is not None,
                'interpretation': self._interpret_regime(regime, final_confidence),
                'trading_implications': self._get_trading_implications_v4(regime, persistence, final_confidence),  # v4: enhanced
                'ml_confidence_multiplier': self._get_ml_confidence_multiplier(final_confidence),  # v4: NEW
                'regime_multiplier': self._get_regime_multiplier(regime),  # v4: NEW
                'fresh_regime': persistence < 15  # v4: NEW flag
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
        
        # RANGING TIGHT (use dynamic thresholds)
        if (ind['range_pct'] < thresholds['range_tight'] and 
            ind['adx'] < thresholds['adx_weak'] and 
            not ind['bb_squeeze']):
            scores[MarketRegimeDetailed.RANGING_TIGHT] += 3
            if abs(ind['sma_20_slope']) < thresholds['slope_strong'] * 0.5:
                scores[MarketRegimeDetailed.RANGING_TIGHT] += 2
        
        # RANGING WIDE (use dynamic thresholds)
        if (thresholds['range_tight'] <= ind['range_pct'] < thresholds['range_wide'] and 
            ind['adx'] < thresholds['adx_strong']):
            scores[MarketRegimeDetailed.RANGING_WIDE] += 3
        
        # BREAKOUT BULLISH (use dynamic thresholds)
        if (ind['volume_ratio'] > thresholds['volume_surge'] and 
            ind['sma_20_slope'] > thresholds['slope_strong'] * 1.5 and 
            ind['above_sma_20'] and 
            ind['volatility_ratio'] > 1.2):
            scores[MarketRegimeDetailed.BREAKOUT_BULLISH] += 4
            if ind.get('bb_expansion', False):
                scores[MarketRegimeDetailed.BREAKOUT_BULLISH] += 1
        
        # BREAKOUT BEARISH (use dynamic thresholds)
        if (ind['volume_ratio'] > thresholds['volume_surge'] and 
            ind['sma_20_slope'] < -thresholds['slope_strong'] * 1.5 and 
            not ind['above_sma_20'] and 
            ind['volatility_ratio'] > 1.2):
            scores[MarketRegimeDetailed.BREAKOUT_BEARISH] += 4
            if ind.get('bb_expansion', False):
                scores[MarketRegimeDetailed.BREAKOUT_BEARISH] += 1
        
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
    
    def _get_ml_confidence_multiplier(self, confidence: float) -> float:
        """
        v4: ML confidence multiplier for position sizing
        High confidence = larger positions
        """
        if confidence >= 0.85:
            return 1.3
        elif confidence >= 0.75:
            return 1.15
        elif confidence >= 0.65:
            return 1.0
        elif confidence >= 0.55:
            return 0.85
        else:
            return 0.7
    
    def _get_regime_multiplier(self, regime: MarketRegimeDetailed) -> float:
        """
        v4: Regime-based multiplier for position sizing
        Strong trends = larger positions, ranging/volatile = smaller
        """
        multipliers = {
            MarketRegimeDetailed.TRENDING_UP_STRONG: 1.2,
            MarketRegimeDetailed.TRENDING_UP_MODERATE: 1.0,
            MarketRegimeDetailed.TRENDING_DOWN_STRONG: 1.2,
            MarketRegimeDetailed.TRENDING_DOWN_MODERATE: 1.0,
            MarketRegimeDetailed.BREAKOUT_BULLISH: 1.5,
            MarketRegimeDetailed.BREAKOUT_BEARISH: 1.5,
            MarketRegimeDetailed.CONSOLIDATION: 0.5,
            MarketRegimeDetailed.RANGING_TIGHT: 0.6,
            MarketRegimeDetailed.RANGING_WIDE: 0.8,
            MarketRegimeDetailed.VOLATILE: 0.3
        }
        return multipliers.get(regime, 1.0)
    
    def _get_trading_implications_v4(self, regime: MarketRegimeDetailed, 
                                    persistence: int, confidence: float) -> List[str]:
        """
        v4: Enhanced trading implications with persistence and confidence
        """
        base_implications = self._get_trading_implications(regime)
        
        # Add persistence-based implications
        if persistence < 15:
            base_implications.append("🆕 Fresh regime - Early entry opportunity")
            base_implications.append("⚡ Recommended: SCALP/INTRADAY for momentum capture")
        elif 15 <= persistence < 40:
            base_implications.append("📊 Developing regime - Strong setup")
            base_implications.append("📈 Recommended: INTRADAY/SWING trading")
        elif persistence >= 40 and confidence > 0.7:
            base_implications.append("⚠️ Mature regime - Consider tightening stops")
            base_implications.append("📊 Recommended: SWING/POSITION with trailing stops")
        elif persistence >= 40 and confidence <= 0.7:
            base_implications.append("⚠️ Mature weakening regime - Reduce exposure")
            base_implications.append("🔄 Recommended: EXIT positions or SCALP only")
        
        # Add confidence-based implications
        if confidence > 0.85:
            base_implications.append("💪 High confidence - Larger position sizing recommended")
        elif confidence < 0.55:
            base_implications.append("⚠️ Low confidence - Reduced position sizing advised")
        
        return base_implications
    
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
    
    def _calculate_regime_persistence_v4(self, current_regime: MarketRegimeDetailed) -> int:
        """
        v4: Calculate regime persistence using bar counter (more precise)
        Returns: number of bars since regime started
        """
        if current_regime != self.current_regime:
            # Changement de régime
            self.current_regime = current_regime
            self.regime_start_bar = self.bar_count
            return 0
        
        # Même régime, calculer la persistence
        persistence = self.bar_count - self.regime_start_bar
        return persistence
    
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
    
    def _calculate_stability_score(self) -> float:
        """
        v4: Calculate stability score based on regime changes frequency
        Returns: 0-1 (1 = stable, 0 = very unstable)
        """
        if len(self.regime_history) < 10:
            return 0.5  # Neutral if insufficient history
        
        # Count regime changes in last 20 bars
        recent_history = self.regime_history[-20:] if len(self.regime_history) >= 20 else self.regime_history
        changes = sum(1 for i in range(1, len(recent_history)) 
                     if recent_history[i] != recent_history[i-1])
        
        # Score inversely proportional to changes
        # 0 changes = 1.0 (very stable)
        # 10+ changes = 0.0 (very unstable)
        stability = max(0.0, 1.0 - (changes / 10.0))
        return stability
    
    def _update_regime_history(self, regime: MarketRegimeDetailed):
        """Update regime history and bar count"""
        self.regime_history.append(regime)
        self.bar_count += 1
        
        # Keep last 200 regimes
        if len(self.regime_history) > 200:
            self.regime_history.pop(0)
    
    def _detect_regime_transition_v4(self) -> str:
        """
        v4: Enhanced transition detection
        Returns: STABLE, EARLY_WARNING, IMMINENT_CHANGE, INSUFFICIENT_DATA
        """
        if len(self.regime_history) < 5:
            return "INSUFFICIENT_DATA"
        
        recent = self.regime_history[-5:]
        
        # All 5 identical = stable
        if len(set(recent)) == 1:
            return "STABLE"
        
        # Last 3 all different = imminent change
        if len(set(recent[-3:])) == 3:
            return "IMMINENT_CHANGE"
        
        # Last 2 different from rest = early warning
        if recent[-1] != recent[-2] or recent[-2] != recent[-3]:
            return "EARLY_WARNING"
        
        return "STABLE"
    
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
        Calculate technical consistency with v4 weighted approach
        Weights: Trend 35%, Momentum 35%, Volume 15%, Volatility 15%
        """
        consistency_score = 0.0
        max_score = 0.0
        
        # 1. Trend Coherence (35%)
        trend_score = 0.0
        trend_max = 3.5
        
        # SMA alignment
        if indicators['sma_20_slope'] > 0 and indicators['above_sma_20']:
            trend_score += 1.0
        elif indicators['sma_20_slope'] < 0 and not indicators['above_sma_20']:
            trend_score += 1.0
        
        # Distance aux moyennes cohérente
        if abs(indicators.get('distance_sma_20', 0)) < 10:
            trend_score += 0.5
        
        # ADX confirme la tendance
        if indicators['adx'] > 25:
            trend_score += 1.0
        elif indicators['adx'] > 20:
            trend_score += 0.5
        
        # Alignement SMA 20/50
        if (indicators['sma_20_slope'] > 0 and indicators['sma_50_slope'] > 0) or \
           (indicators['sma_20_slope'] < 0 and indicators['sma_50_slope'] < 0):
            trend_score += 1.0
        
        consistency_score += trend_score
        max_score += trend_max
        
        # 2. Momentum Coherence (35%)
        momentum_score = 0.0
        momentum_max = 3.5
        
        # RSI et MACD alignés
        if indicators.get('rsi_trend', 0) > 0 and indicators['macd_histogram'] > 0:
            momentum_score += 1.5
        elif indicators.get('rsi_trend', 0) < 0 and indicators['macd_histogram'] < 0:
            momentum_score += 1.5
        
        # RSI dans zone normale
        rsi_zone = indicators.get('rsi_zone', 'NORMAL')
        if rsi_zone in ['NEUTRAL', 'NORMAL']:
            momentum_score += 1.0
        elif rsi_zone in ['OVERBOUGHT', 'OVERSOLD']:
            momentum_score += 0.3
        
        # Tendance MACD cohérente
        if abs(indicators.get('macd_trend', 0)) > 0.01:
            momentum_score += 1.0
        
        consistency_score += momentum_score
        max_score += momentum_max
        
        # 3. Volume Coherence (15%)
        volume_score = 0.0
        volume_max = 1.5
        
        # Volume confirme la direction
        if indicators['volume_ratio'] > 1.0 and indicators.get('volume_trend', 0) > 0:
            volume_score += 1.0
        elif indicators['volume_ratio'] < 1.0 and indicators.get('volume_trend', 0) < 0:
            volume_score += 0.5
        
        # Volume pas anormalement élevé
        if 0.8 <= indicators['volume_ratio'] <= 2.0:
            volume_score += 0.5
        
        consistency_score += volume_score
        max_score += volume_max
        
        # 4. Volatility Coherence (15%)
        volatility_score = 0.0
        volatility_max = 1.5
        
        # Volatilité stable
        if 0.8 <= indicators['volatility_ratio'] <= 1.2:
            volatility_score += 1.0
        elif 0.6 <= indicators['volatility_ratio'] <= 1.5:
            volatility_score += 0.5
        
        # Volatilité appropriée
        if indicators['atr_pct'] < 5.0:
            volatility_score += 0.5
        
        consistency_score += volatility_score
        max_score += volatility_max
        
        # Score final normalisé
        final_consistency = consistency_score / max_score if max_score > 0 else 0.5
        return min(1.0, final_consistency)
    
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
            
            # 11. EMAs for trend analysis
            ema_9 = close.ewm(span=9).mean().iloc[-1] if len(close) >= 9 else close.iloc[-1]
            ema_21 = close.ewm(span=21).mean().iloc[-1] if len(close) >= 21 else close.iloc[-1]
            ema_50 = close.ewm(span=50).mean().iloc[-1] if len(close) >= 50 else close.iloc[-1]
            ema_200 = close.ewm(span=200).mean().iloc[-1] if len(close) >= 200 else close.iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.iloc[-1]
            
            # 12. Trend hierarchy analysis
            current_price = close.iloc[-1]
            trend_hierarchy, trend_momentum, price_vs_emas, ema_cross_signal, trend_strength_score = self._calculate_trend_hierarchy(
                current_price, ema_9, ema_21, ema_50, ema_200, sma_50
            )
            
            # 13. Trade Type Recommendation (NEW)
            trade_type, duration, timeframe, min_rr = self._determine_trade_type(
                regime_data, adx, volatility_ratio, atr_pct, indicators
            )
            
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
                timeframe_score=tf_score,
                
                # EMAs for trend analysis
                ema_9=ema_9,
                ema_21=ema_21,
                ema_50=ema_50,
                ema_200=ema_200,
                sma_50=sma_50,
                
                # Trend hierarchy analysis
                trend_hierarchy=trend_hierarchy,
                trend_momentum=trend_momentum,
                price_vs_emas=price_vs_emas,
                ema_cross_signal=ema_cross_signal,
                trend_strength_score=trend_strength_score,
                
                # Trade Type Recommendation
                trade_type=trade_type,
                trade_duration_estimate=duration,
                optimal_timeframe=timeframe,
                minimum_rr_threshold=min_rr
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
    
    def _determine_trade_type(self, regime_data: Dict, adx: float, volatility_ratio: float,
                              atr_pct: float, indicators: Dict) -> Tuple[str, str, str, float]:
        """
        Determine optimal trade type based on market conditions
        Returns: (trade_type, duration_estimate, optimal_timeframe, minimum_rr)
        
        Trade Types:
        - SCALP: Very short-term (minutes to hours), high volatility, fresh regimes
        - INTRADAY: Same-day trades (hours), moderate conditions
        - SWING: Multi-day trades (1-7 days), trending markets
        - POSITION: Long-term (weeks+), strong established trends
        """
        
        regime = regime_data.get('regime', 'RANGING_TIGHT')
        persistence = regime_data.get('regime_persistence', 0)
        fresh_regime = regime_data.get('fresh_regime', False)
        confidence = regime_data.get('confidence', 0.5)
        
        # Scoring system for trade type
        scalp_score = 0
        intraday_score = 0
        swing_score = 0
        position_score = 0
        
        # 1. VOLATILITY ANALYSIS
        if volatility_ratio > 1.5 and atr_pct > 3.0:
            # High volatility = scalping opportunities
            scalp_score += 3
            intraday_score += 2
        elif 1.2 < volatility_ratio <= 1.5:
            # Moderate volatility = intraday/swing
            intraday_score += 2
            swing_score += 2
        elif volatility_ratio < 1.0 and atr_pct < 2.0:
            # Low volatility = swing/position
            swing_score += 2
            position_score += 2
        
        # 2. ADX (TREND STRENGTH)
        if adx > 30:
            # Very strong trend = swing/position
            swing_score += 3
            position_score += 3
        elif 25 <= adx <= 30:
            # Strong trend = swing
            swing_score += 4
            position_score += 1
        elif 20 <= adx < 25:
            # Moderate trend = intraday/swing
            intraday_score += 2
            swing_score += 2
        else:  # adx < 20
            # Weak trend = scalp/intraday
            scalp_score += 2
            intraday_score += 2
        
        # 3. REGIME TYPE
        if 'BREAKOUT' in regime:
            # Breakouts = scalp/intraday for quick profits
            scalp_score += 3
            intraday_score += 4
        elif 'TRENDING' in regime and 'STRONG' in regime:
            # Strong trends = swing/position
            swing_score += 4
            position_score += 2
        elif 'TRENDING' in regime and 'MODERATE' in regime:
            # Moderate trends = intraday/swing
            intraday_score += 3
            swing_score += 3
        elif 'RANGING' in regime:
            # Ranging = scalp/intraday
            scalp_score += 3
            intraday_score += 3
        elif 'CONSOLIDATION' in regime:
            # Consolidation = wait or scalp
            scalp_score += 2
        elif 'VOLATILE' in regime:
            # Volatile = scalp only
            scalp_score += 5
        
        # 4. REGIME PERSISTENCE
        if fresh_regime or persistence < 15:
            # Fresh regime = capture early momentum (scalp/intraday)
            scalp_score += 2
            intraday_score += 3
        elif 15 <= persistence < 40:
            # Developing regime = intraday/swing
            intraday_score += 2
            swing_score += 3
        elif persistence >= 40 and confidence > 0.7:
            # Mature strong regime = swing/position
            swing_score += 2
            position_score += 3
        elif persistence >= 40 and confidence <= 0.7:
            # Mature weakening = exit or scalp
            scalp_score += 1
        
        # 5. BOLLINGER BAND SQUEEZE
        bb_squeeze = indicators.get('bb_squeeze', False)
        if bb_squeeze:
            # Squeeze = prepare for breakout (scalp/intraday)
            scalp_score += 2
            intraday_score += 2
        
        # 6. VOLUME SURGE
        volume_surge = indicators.get('volume_surge', False)
        if volume_surge:
            # Volume spike = short-term opportunity
            scalp_score += 2
            intraday_score += 2
        
        # Determine winner
        scores = {
            'SCALP': scalp_score,
            'INTRADAY': intraday_score,
            'SWING': swing_score,
            'POSITION': position_score
        }
        
        trade_type = max(scores, key=scores.get)
        max_score = scores[trade_type]
        
        # Default to SWING if scores too low or tied
        if max_score < 3:
            trade_type = 'SWING'
        
        # Determine duration, optimal timeframe, and minimum RR
        if trade_type == 'SCALP':
            duration = "5min - 2 hours"
            timeframe = "1M/5M"
            min_rr = 1.0  # Quick profits, lower RR acceptable
        elif trade_type == 'INTRADAY':
            duration = "2 hours - 24 hours"
            timeframe = "15M/1H"
            min_rr = 1.5  # Moderate RR for intraday
        elif trade_type == 'SWING':
            duration = "1-7 days"
            timeframe = "4H/1D"
            min_rr = 2.0  # Standard swing trading RR
        else:  # POSITION
            duration = "1-4 weeks"
            timeframe = "1D/1W"
            min_rr = 2.5  # Higher RR for position trades
        
        return trade_type, duration, timeframe, min_rr
    
    def _calculate_trend_hierarchy(self, current_price: float, ema_9: float, ema_21: float, 
                                 ema_50: float, ema_200: float, sma_50: float) -> tuple:
        """Calculate trend hierarchy and related metrics"""
        
        # Price position relative to EMAs
        above_ema9 = current_price > ema_9
        above_ema21 = current_price > ema_21
        above_ema50 = current_price > ema_50
        above_ema200 = current_price > ema_200
        above_sma50 = current_price > sma_50
        
        # EMA alignment
        ema9_above_21 = ema_9 > ema_21
        ema21_above_50 = ema_21 > ema_50
        ema50_above_200 = ema_50 > ema_200
        
        # Calculate trend strength score
        bullish_signals = sum([above_ema9, above_ema21, above_ema50, above_ema200, above_sma50, 
                              ema9_above_21, ema21_above_50, ema50_above_200])
        trend_strength_score = bullish_signals / 8.0  # 8 total signals
        
        # Determine trend hierarchy
        if bullish_signals >= 7:
            trend_hierarchy = "strong_bull"
        elif bullish_signals >= 5:
            trend_hierarchy = "weak_bull"
        elif bullish_signals <= 1:
            trend_hierarchy = "strong_bear"
        elif bullish_signals <= 3:
            trend_hierarchy = "weak_bear"
        else:
            trend_hierarchy = "neutral"
        
        # Determine price vs EMAs
        if above_ema9 and above_ema21 and above_ema50:
            price_vs_emas = "above_all"
        elif above_ema9 and above_ema21:
            price_vs_emas = "above_fast"
        elif not above_ema9 and not above_ema21 and not above_ema50:
            price_vs_emas = "below_all"
        elif not above_ema9 and not above_ema21:
            price_vs_emas = "below_fast"
        else:
            price_vs_emas = "mixed"
        
        # Determine EMA cross signal
        if ema9_above_21 and ema21_above_50:
            ema_cross_signal = "golden_cross"
        elif not ema9_above_21 and not ema21_above_50:
            ema_cross_signal = "death_cross"
        else:
            ema_cross_signal = "none"
        
        # Determine trend momentum
        if trend_strength_score > 0.7:
            trend_momentum = "strong_bullish"
        elif trend_strength_score > 0.6:
            trend_momentum = "bullish"
        elif trend_strength_score < 0.3:
            trend_momentum = "strong_bearish"
        elif trend_strength_score < 0.4:
            trend_momentum = "bearish"
        else:
            trend_momentum = "neutral"
        
        return trend_hierarchy, trend_momentum, price_vs_emas, ema_cross_signal, trend_strength_score
    
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

"""
ADAPTIVE CONTEXT SYSTEM - Dynamic Strategy Adjustment Based on Real-Time Market Conditions
Enhanced with AI Training Data for Better Context Understanding
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import json
from enum import Enum

logger = logging.getLogger(__name__)

class MarketRegime(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    VOLATILE = "VOLATILE"
    TRANSITION = "TRANSITION"

class ContextAdjustmentType(str, Enum):
    POSITION_SIZING = "position_sizing"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    SIGNAL_THRESHOLD = "signal_threshold"
    PATTERN_WEIGHT = "pattern_weight"
    RISK_REWARD = "risk_reward"

@dataclass
class MarketContext:
    """Current market context with enhanced AI-driven analysis"""
    current_regime: MarketRegime
    regime_confidence: float
    volatility_level: float  # 0-1 scale
    trend_strength: float   # 0-1 scale
    volume_trend: float     # Relative volume change
    pattern_environment: str  # Which patterns are more reliable now
    rsi_environment: str    # Overbought, oversold, neutral zones
    macd_environment: str   # Bullish, bearish, choppy
    market_stress_level: float  # 0-1 scale of market stress
    liquidity_condition: str   # High, medium, low liquidity
    correlation_breakdown: bool  # Are usual correlations breaking down?
    news_sentiment: str    # Positive, neutral, negative (if available)
    timestamp: datetime
    context_duration: int  # How long this context has been active (hours)

@dataclass 
class ContextualAdjustment:
    """Specific adjustment recommendation based on context"""
    adjustment_type: ContextAdjustmentType
    original_value: float
    adjusted_value: float
    adjustment_factor: float
    reasoning: str
    confidence: float
    expected_improvement: float
    applicable_symbols: List[str]
    context_conditions: Dict[str, Any]

@dataclass
class AdaptiveRule:
    """Dynamic rule that adjusts based on market context"""
    rule_id: str
    rule_name: str
    trigger_conditions: Dict[str, Any]
    adjustments: List[ContextualAdjustment]
    success_rate: float
    sample_size: int
    last_triggered: Optional[datetime]
    effectiveness_score: float
    is_active: bool

class AdaptiveContextSystem:
    """Enhanced Adaptive Context System with AI Training Integration"""
    
    def __init__(self):
        self.current_context = None
        self.context_history = deque(maxlen=168)  # Keep 1 week of hourly contexts
        self.adaptive_rules = []
        self.context_transitions = deque(maxlen=50)  # Track context changes
        
        # Performance tracking
        self.adjustment_performance = defaultdict(list)
        self.regime_detection_accuracy = deque(maxlen=100)
        
        # Market data for context analysis
        self.market_data_buffer = defaultdict(lambda: deque(maxlen=100))
        
        # Integration with AI training data
        self.trained_market_conditions = []
        self.pattern_success_rates = {}
        self.ia1_accuracy_patterns = {}
        self.ia2_performance_patterns = {}
        
        logger.info("Adaptive Context System initialized")
    
    def load_ai_training_data(self, ai_training_system):
        """Load and integrate AI training data for enhanced context understanding"""
        try:
            # Load market conditions from training
            self.trained_market_conditions = ai_training_system.market_conditions
            
            # Extract pattern success rates by context
            pattern_context_success = defaultdict(lambda: defaultdict(list))
            for pattern_data in ai_training_system.pattern_training:
                pattern_context_success[pattern_data.pattern_type][pattern_data.market_condition].append(pattern_data.success)
            
            # Calculate success rates
            for pattern_type, contexts in pattern_context_success.items():
                self.pattern_success_rates[pattern_type] = {}
                for context, successes in contexts.items():
                    self.pattern_success_rates[pattern_type][context] = {
                        'success_rate': np.mean(successes),
                        'sample_size': len(successes),
                        'confidence': min(len(successes) / 10, 1.0)  # Confidence based on sample size
                    }
            
            # Extract IA1 accuracy patterns
            ia1_context_accuracy = defaultdict(list)
            for ia1_data in ai_training_system.ia1_enhancements:
                ia1_context_accuracy[ia1_data.market_context].append(ia1_data.prediction_accuracy)
            
            for context, accuracies in ia1_context_accuracy.items():
                self.ia1_accuracy_patterns[context] = {
                    'avg_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'sample_size': len(accuracies)
                }
            
            # Extract IA2 performance patterns
            ia2_context_performance = defaultdict(list)
            for ia2_data in ai_training_system.ia2_enhancements:
                context = self._infer_context_from_enhancement(ia2_data)
                ia2_context_performance[context].append(ia2_data.actual_performance)
            
            for context, performances in ia2_context_performance.items():
                self.ia2_performance_patterns[context] = {
                    'avg_performance': np.mean(performances),
                    'volatility': np.std(performances),
                    'sample_size': len(performances)
                }
            
            # Generate adaptive rules from training data
            self._generate_adaptive_rules_from_training()
            
            logger.info(f"Loaded AI training data: {len(self.trained_market_conditions)} market conditions, "
                       f"{len(self.pattern_success_rates)} pattern types, "
                       f"{len(self.adaptive_rules)} adaptive rules generated")
            
        except Exception as e:
            logger.error(f"Error loading AI training data: {e}")
    
    def _infer_context_from_enhancement(self, ia2_data) -> str:
        """Infer market context from IA2 enhancement data"""
        # This is a simplified inference - in practice, you'd use more sophisticated logic
        if ia2_data.actual_performance > 5:
            return "BULL"
        elif ia2_data.actual_performance < -3:
            return "BEAR"
        elif abs(ia2_data.actual_performance) < 2:
            return "SIDEWAYS"  
        else:
            return "VOLATILE"
    
    def _generate_adaptive_rules_from_training(self):
        """Generate adaptive rules based on AI training insights"""
        rules = []
        
        # Rule 1: Pattern-based position sizing
        for pattern_type, contexts in self.pattern_success_rates.items():
            for context, stats in contexts.items():
                if stats['sample_size'] >= 5:  # Minimum sample requirement
                    adjustment_factor = 0.5 + (stats['success_rate'] * 1.0)  # 0.5 to 1.5 multiplier
                    
                    rule = AdaptiveRule(
                        rule_id=f"pattern_sizing_{pattern_type}_{context}",
                        rule_name=f"Pattern-based sizing for {pattern_type} in {context}",
                        trigger_conditions={
                            'pattern_detected': pattern_type,
                            'market_regime': context,
                            'min_confidence': 0.6
                        },
                        adjustments=[
                            ContextualAdjustment(
                                adjustment_type=ContextAdjustmentType.POSITION_SIZING,
                                original_value=0.02,  # Standard 2%
                                adjusted_value=0.02 * adjustment_factor,
                                adjustment_factor=adjustment_factor,
                                reasoning=f"{pattern_type} has {stats['success_rate']:.1%} success in {context} conditions",
                                confidence=stats['confidence'],
                                expected_improvement=stats['success_rate'] - 0.5,
                                applicable_symbols=[],  # Apply to all
                                context_conditions={'market_regime': context}
                            )
                        ],
                        success_rate=stats['success_rate'],
                        sample_size=stats['sample_size'],
                        last_triggered=None,
                        effectiveness_score=stats['success_rate'] * stats['confidence'],
                        is_active=True
                    )
                    rules.append(rule)
        
        # Rule 2: IA1 accuracy-based signal threshold adjustments
        for context, accuracy_stats in self.ia1_accuracy_patterns.items():
            if accuracy_stats['sample_size'] >= 10:
                # If IA1 is less accurate in this context, raise thresholds
                threshold_adjustment = 2.0 - accuracy_stats['avg_accuracy']  # Higher threshold for lower accuracy
                
                rule = AdaptiveRule(
                    rule_id=f"ia1_threshold_{context}",
                    rule_name=f"IA1 threshold adjustment for {context}",
                    trigger_conditions={
                        'market_regime': context,
                        'ia1_confidence': ('>', 0.5)
                    },
                    adjustments=[
                        ContextualAdjustment(
                            adjustment_type=ContextAdjustmentType.SIGNAL_THRESHOLD,
                            original_value=0.75,  # Standard threshold
                            adjusted_value=0.75 * threshold_adjustment,
                            adjustment_factor=threshold_adjustment,
                            reasoning=f"IA1 accuracy is {accuracy_stats['avg_accuracy']:.1%} in {context}, adjusting thresholds",
                            confidence=min(1.0, accuracy_stats['sample_size'] / 20),
                            expected_improvement=abs(accuracy_stats['avg_accuracy'] - 0.75),
                            applicable_symbols=[],
                            context_conditions={'market_regime': context}
                        )
                    ],
                    success_rate=accuracy_stats['avg_accuracy'],
                    sample_size=accuracy_stats['sample_size'],
                    last_triggered=None,
                    effectiveness_score=abs(accuracy_stats['avg_accuracy'] - 0.5) * 2,
                    is_active=True
                )
                rules.append(rule)
        
        # Rule 3: IA2 performance-based risk-reward adjustments
        for context, perf_stats in self.ia2_performance_patterns.items():
            if perf_stats['sample_size'] >= 10:
                # Adjust risk-reward based on historical performance
                if perf_stats['avg_performance'] > 3:
                    rr_adjustment = 1.2  # Increase targets in high-performance contexts
                elif perf_stats['avg_performance'] < 0:
                    rr_adjustment = 0.8  # Reduce targets in poor-performance contexts
                else:
                    rr_adjustment = 1.0
                
                if rr_adjustment != 1.0:
                    rule = AdaptiveRule(
                        rule_id=f"ia2_rr_{context}",
                        rule_name=f"IA2 risk-reward adjustment for {context}",
                        trigger_conditions={
                            'market_regime': context,
                            'ia2_confidence': ('>', 0.6)
                        },
                        adjustments=[
                            ContextualAdjustment(
                                adjustment_type=ContextAdjustmentType.RISK_REWARD,
                                original_value=2.0,  # Standard 2:1 RR
                                adjusted_value=2.0 * rr_adjustment,
                                adjustment_factor=rr_adjustment,
                                reasoning=f"IA2 avg performance is {perf_stats['avg_performance']:.1f}% in {context}",
                                confidence=min(1.0, perf_stats['sample_size'] / 15),
                                expected_improvement=abs(perf_stats['avg_performance'] / 10),
                                applicable_symbols=[],
                                context_conditions={'market_regime': context}
                            )
                        ],
                        success_rate=0.6 + (perf_stats['avg_performance'] / 20),  # Approximate success rate
                        sample_size=perf_stats['sample_size'],
                        last_triggered=None,
                        effectiveness_score=abs(perf_stats['avg_performance'] / 5),
                        is_active=True
                    )
                    rules.append(rule)
        
        self.adaptive_rules = rules
        logger.info(f"Generated {len(rules)} adaptive rules from AI training data")
    
    async def analyze_current_context(self, market_data: Dict[str, Any]) -> MarketContext:
        """Analyze current market context with enhanced AI-driven insights"""
        try:
            # Extract market metrics
            symbols_data = market_data.get('symbols', {})
            if not symbols_data:
                return self._get_default_context()
            
            # Calculate market-wide metrics
            price_changes = []
            volumes = []
            rsi_values = []
            macd_values = []
            volatilities = []
            
            for symbol, data in symbols_data.items():
                if isinstance(data, dict):
                    price_changes.append(data.get('price_change_24h', 0))
                    volumes.append(data.get('volume_ratio', 1.0))
                    rsi_values.append(data.get('rsi', 50))
                    macd_values.append(data.get('macd_signal', 0))
                    volatilities.append(data.get('volatility', 5))
            
            if not price_changes:
                return self._get_default_context()
            
            # Calculate aggregate metrics
            avg_price_change = np.mean(price_changes)
            avg_volatility = np.mean(volatilities)
            avg_volume_ratio = np.mean(volumes)
            avg_rsi = np.mean(rsi_values)
            avg_macd = np.mean(macd_values)
            
            # Determine market regime using AI-enhanced logic
            regime = self._determine_regime_ai_enhanced(
                avg_price_change, avg_volatility, avg_rsi, avg_macd
            )
            
            # Calculate regime confidence using trained patterns
            regime_confidence = self._calculate_regime_confidence(regime, {
                'price_change': avg_price_change,
                'volatility': avg_volatility,
                'rsi': avg_rsi,
                'macd': avg_macd
            })
            
            # Analyze pattern environment
            pattern_environment = self._analyze_pattern_environment(regime, avg_volatility)
            
            # Determine market stress level
            stress_level = self._calculate_market_stress(avg_volatility, price_changes)
            
            # Create context
            context = MarketContext(
                current_regime=regime,
                regime_confidence=regime_confidence,
                volatility_level=min(avg_volatility / 20, 1.0),  # Normalize to 0-1
                trend_strength=abs(avg_price_change) / max(avg_volatility, 1.0),
                volume_trend=avg_volume_ratio - 1.0,
                pattern_environment=pattern_environment,
                rsi_environment=self._categorize_rsi_environment(avg_rsi),
                macd_environment=self._categorize_macd_environment(avg_macd),
                market_stress_level=stress_level,
                liquidity_condition=self._assess_liquidity_condition(avg_volume_ratio),
                correlation_breakdown=self._detect_correlation_breakdown(symbols_data),
                news_sentiment="neutral",  # Could be enhanced with news analysis
                timestamp=datetime.now(),
                context_duration=self._calculate_context_duration(regime)
            )
            
            # Update context history
            self.current_context = context
            self.context_history.append(context)
            
            # Check for context transitions
            if len(self.context_history) > 1:
                prev_context = self.context_history[-2]
                if prev_context.current_regime != context.current_regime:
                    self.context_transitions.append({
                        'from_regime': prev_context.current_regime.value,
                        'to_regime': context.current_regime.value,
                        'timestamp': context.timestamp,
                        'confidence': context.regime_confidence
                    })
            
            logger.info(f"Market context: {regime.value} (confidence: {regime_confidence:.1%}), "
                       f"volatility: {context.volatility_level:.1%}, stress: {stress_level:.1%}")
            
            return context
            
        except Exception as e:
            logger.error(f"Error analyzing market context: {e}")
            return self._get_default_context()
    
    def _determine_regime_ai_enhanced(self, price_change: float, volatility: float, 
                                    rsi: float, macd: float) -> MarketRegime:
        """Determine market regime using AI-enhanced logic"""
        # Base regime determination
        if volatility > 15:
            base_regime = MarketRegime.VOLATILE
        elif abs(price_change) < 2:
            base_regime = MarketRegime.SIDEWAYS
        elif price_change > 2:
            base_regime = MarketRegime.BULL
        else:
            base_regime = MarketRegime.BEAR
        
        # Enhance with trained patterns
        if self.trained_market_conditions:
            # Find similar historical conditions
            similar_conditions = []
            for condition in self.trained_market_conditions:
                similarity = self._calculate_condition_similarity({
                    'price_change': price_change,
                    'volatility': volatility,
                    'rsi': rsi,
                    'macd': macd
                }, condition)
                
                if similarity > 0.7:  # High similarity threshold
                    similar_conditions.append((condition, similarity))
            
            # If we have similar historical conditions, use their classification
            if similar_conditions:
                # Weight by similarity and confidence
                regime_weights = defaultdict(float)
                total_weight = 0
                
                for condition, similarity in similar_conditions:
                    weight = similarity * condition.confidence_score
                    regime_weights[condition.condition_type] += weight
                    total_weight += weight
                
                if total_weight > 0:
                    # Normalize weights
                    for regime in regime_weights:
                        regime_weights[regime] /= total_weight
                    
                    # Choose regime with highest weight
                    best_regime = max(regime_weights.items(), key=lambda x: x[1])
                    if best_regime[1] > 0.4:  # Confidence threshold
                        try:
                            return MarketRegime(best_regime[0])
                        except:
                            pass  # Fall back to base regime
        
        return base_regime
    
    def _calculate_condition_similarity(self, current: Dict[str, float], 
                                      historical_condition) -> float:
        """Calculate similarity between current and historical market conditions"""
        try:
            # Normalize differences
            price_diff = abs(current['price_change'] - historical_condition.price_change_pct) / 20
            vol_diff = abs(current['volatility'] - historical_condition.volatility) / 15
            rsi_diff = abs(current['rsi'] - historical_condition.rsi_avg) / 50
            
            # Calculate similarity (0-1, where 1 is identical)
            avg_diff = (price_diff + vol_diff + rsi_diff) / 3
            similarity = max(0, 1 - avg_diff)
            
            return similarity
            
        except Exception as e:
            logger.debug(f"Error calculating condition similarity: {e}")
            return 0.0
    
    def _calculate_regime_confidence(self, regime: MarketRegime, metrics: Dict[str, float]) -> float:
        """Calculate confidence in regime determination using AI insights"""
        base_confidence = 0.7
        
        # Boost confidence if we have training data for this regime
        if regime.value in self.ia1_accuracy_patterns:
            pattern_data = self.ia1_accuracy_patterns[regime.value]
            if pattern_data['sample_size'] > 10:
                accuracy_boost = (pattern_data['avg_accuracy'] - 0.5) * 0.4
                base_confidence += accuracy_boost
        
        # Adjust based on clear regime signals
        if regime == MarketRegime.VOLATILE and metrics['volatility'] > 20:
            base_confidence += 0.15
        elif regime == MarketRegime.BULL and metrics['price_change'] > 5:
            base_confidence += 0.15
        elif regime == MarketRegime.BEAR and metrics['price_change'] < -5:
            base_confidence += 0.15
        elif regime == MarketRegime.SIDEWAYS and abs(metrics['price_change']) < 1:
            base_confidence += 0.10
        
        return min(0.95, max(0.5, base_confidence))
    
    def _analyze_pattern_environment(self, regime: MarketRegime, volatility: float) -> str:
        """Analyze which patterns are most reliable in current environment"""
        if regime == MarketRegime.BULL:
            return "bullish_breakouts_favorable"
        elif regime == MarketRegime.BEAR:
            return "bearish_patterns_reliable"
        elif regime == MarketRegime.VOLATILE:
            return "reversal_patterns_preferred"
        else:
            return "range_patterns_suitable"
    
    def _categorize_rsi_environment(self, avg_rsi: float) -> str:
        """Categorize RSI environment"""
        if avg_rsi > 70:
            return "overbought_zone"
        elif avg_rsi < 30:
            return "oversold_zone"
        elif avg_rsi > 55:
            return "bullish_zone"
        elif avg_rsi < 45:
            return "bearish_zone"
        else:
            return "neutral_zone"
    
    def _categorize_macd_environment(self, avg_macd: float) -> str:
        """Categorize MACD environment"""
        if avg_macd > 0.01:
            return "strong_bullish"
        elif avg_macd > 0:
            return "weak_bullish"
        elif avg_macd < -0.01:
            return "strong_bearish"
        elif avg_macd < 0:
            return "weak_bearish"
        else:
            return "neutral"
    
    def _calculate_market_stress(self, volatility: float, price_changes: List[float]) -> float:
        """Calculate market stress level"""
        vol_stress = min(volatility / 25, 1.0)  # Normalize volatility
        
        # Calculate dispersion of price changes
        if len(price_changes) > 1:
            dispersion = np.std(price_changes) / max(np.mean(np.abs(price_changes)), 1.0)
            dispersion_stress = min(dispersion / 3.0, 1.0)
        else:
            dispersion_stress = 0.0
        
        return (vol_stress + dispersion_stress) / 2
    
    def _assess_liquidity_condition(self, avg_volume_ratio: float) -> str:
        """Assess market liquidity condition"""
        if avg_volume_ratio > 1.5:
            return "high"
        elif avg_volume_ratio > 0.8:
            return "medium"
        else:
            return "low"
    
    def _detect_correlation_breakdown(self, symbols_data: Dict[str, Any]) -> bool:
        """Detect if usual market correlations are breaking down"""
        # Simplified correlation breakdown detection
        # In practice, this would analyze correlation matrices
        price_changes = []
        for data in symbols_data.values():
            if isinstance(data, dict):
                price_changes.append(data.get('price_change_24h', 0))
        
        if len(price_changes) > 5:
            # If price changes have very high variance, correlations might be breaking
            variance = np.var(price_changes)
            return variance > 100  # Threshold for correlation breakdown
        
        return False
    
    def _calculate_context_duration(self, current_regime: MarketRegime) -> int:
        """Calculate how long current context has been active"""
        if not self.context_history:
            return 0
        
        duration = 0
        for context in reversed(self.context_history):
            if context.current_regime == current_regime:
                duration += 1
            else:
                break
        
        return duration
    
    def _get_default_context(self) -> MarketContext:
        """Get default context when analysis fails"""
        return MarketContext(
            current_regime=MarketRegime.SIDEWAYS,
            regime_confidence=0.5,
            volatility_level=0.3,
            trend_strength=0.2,
            volume_trend=0.0,
            pattern_environment="neutral",
            rsi_environment="neutral_zone",
            macd_environment="neutral",
            market_stress_level=0.3,
            liquidity_condition="medium",
            correlation_breakdown=False,
            news_sentiment="neutral",
            timestamp=datetime.now(),
            context_duration=0
        )
    
    def get_contextual_adjustments(self, base_decision: Dict[str, Any]) -> List[ContextualAdjustment]:
        """Get contextual adjustments for a trading decision"""
        if not self.current_context:
            return []
        
        applicable_adjustments = []
        
        for rule in self.adaptive_rules:
            if not rule.is_active:
                continue
            
            # Check if rule conditions are met
            if self._check_rule_conditions(rule, base_decision):
                for adjustment in rule.adjustments:
                    # Verify context conditions
                    if self._check_context_conditions(adjustment.context_conditions):
                        applicable_adjustments.append(adjustment)
                        
                        # Update rule tracking
                        rule.last_triggered = datetime.now()
        
        # Sort by expected improvement
        applicable_adjustments.sort(key=lambda x: x.expected_improvement, reverse=True)
        
        return applicable_adjustments[:5]  # Return top 5 adjustments
    
    def _check_rule_conditions(self, rule: AdaptiveRule, decision: Dict[str, Any]) -> bool:
        """Check if rule conditions are met for current decision"""
        conditions = rule.trigger_conditions
        
        # Check market regime
        if 'market_regime' in conditions:
            if self.current_context.current_regime.value != conditions['market_regime']:
                return False
        
        # Check pattern detection
        if 'pattern_detected' in conditions:
            detected_patterns = decision.get('patterns_detected', [])
            if conditions['pattern_detected'] not in detected_patterns:
                return False
        
        # Check confidence thresholds
        for conf_key in ['ia1_confidence', 'ia2_confidence', 'min_confidence']:
            if conf_key in conditions:
                condition = conditions[conf_key]
                if isinstance(condition, tuple) and len(condition) == 2:
                    operator, threshold = condition
                    decision_conf = decision.get(conf_key.replace('_confidence', '_confidence'), 0.5)
                    
                    if operator == '>' and decision_conf <= threshold:
                        return False
                    elif operator == '<' and decision_conf >= threshold:
                        return False
                elif isinstance(condition, (int, float)):
                    decision_conf = decision.get(conf_key.replace('_confidence', '_confidence'), 0.5)
                    if decision_conf < condition:
                        return False
        
        return True
    
    def _check_context_conditions(self, context_conditions: Dict[str, Any]) -> bool:
        """Check if context conditions are met"""
        if not self.current_context:
            return False
        
        for key, value in context_conditions.items():
            if key == 'market_regime':
                if self.current_context.current_regime.value != value:
                    return False
            elif key == 'min_volatility':
                if self.current_context.volatility_level < value:
                    return False
            elif key == 'max_stress':
                if self.current_context.market_stress_level > value:
                    return False
        
        return True
    
    def apply_adjustments_to_decision(self, base_decision: Dict[str, Any], 
                                    adjustments: List[ContextualAdjustment]) -> Dict[str, Any]:
        """Apply contextual adjustments to a trading decision"""
        adjusted_decision = base_decision.copy()
        
        applied_adjustments = []
        
        for adjustment in adjustments:
            try:
                if adjustment.adjustment_type == ContextAdjustmentType.POSITION_SIZING:
                    adjusted_decision['position_size'] = adjustment.adjusted_value
                    applied_adjustments.append(f"Position size: {adjustment.original_value:.1%} → {adjustment.adjusted_value:.1%}")
                
                elif adjustment.adjustment_type == ContextAdjustmentType.STOP_LOSS:
                    # Adjust stop loss distance
                    entry_price = adjusted_decision.get('entry_price', 1.0)
                    original_sl = adjusted_decision.get('stop_loss', entry_price * 0.98)
                    sl_distance = abs(entry_price - original_sl) * adjustment.adjustment_factor
                    
                    if adjusted_decision.get('signal') == 'long':
                        adjusted_decision['stop_loss'] = entry_price - sl_distance
                    else:
                        adjusted_decision['stop_loss'] = entry_price + sl_distance
                    
                    applied_adjustments.append(f"Stop loss adjusted by {adjustment.adjustment_factor:.1f}x")
                
                elif adjustment.adjustment_type == ContextAdjustmentType.TAKE_PROFIT:
                    # Adjust take profit distances
                    entry_price = adjusted_decision.get('entry_price', 1.0)
                    for tp_key in ['take_profit_1', 'take_profit_2', 'take_profit_3']:
                        if tp_key in adjusted_decision:
                            original_tp = adjusted_decision[tp_key]
                            tp_distance = abs(original_tp - entry_price) * adjustment.adjustment_factor
                            
                            if adjusted_decision.get('signal') == 'long':
                                adjusted_decision[tp_key] = entry_price + tp_distance
                            else:
                                adjusted_decision[tp_key] = entry_price - tp_distance
                    
                    applied_adjustments.append(f"Take profit adjusted by {adjustment.adjustment_factor:.1f}x")
                
                elif adjustment.adjustment_type == ContextAdjustmentType.SIGNAL_THRESHOLD:
                    # This would be applied at the signal generation level
                    adjusted_decision['threshold_adjustment'] = adjustment.adjustment_factor
                    applied_adjustments.append(f"Signal threshold adjusted by {adjustment.adjustment_factor:.1f}x")
                
                elif adjustment.adjustment_type == ContextAdjustmentType.RISK_REWARD:
                    # Adjust risk-reward ratio
                    adjusted_decision['target_risk_reward'] = adjustment.adjusted_value
                    applied_adjustments.append(f"Target R:R: {adjustment.original_value:.1f}:1 → {adjustment.adjusted_value:.1f}:1")
                
            except Exception as e:
                logger.error(f"Error applying adjustment {adjustment.adjustment_type}: {e}")
                continue
        
        if applied_adjustments:
            adjusted_decision['adaptive_adjustments'] = applied_adjustments
            adjusted_decision['context_regime'] = self.current_context.current_regime.value
            adjusted_decision['context_confidence'] = self.current_context.regime_confidence
        
        return adjusted_decision
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the adaptive context system"""
        status = {
            'current_context': None,
            'active_rules': len([r for r in self.adaptive_rules if r.is_active]),
            'total_rules': len(self.adaptive_rules),
            'context_history_length': len(self.context_history),
            'recent_transitions': list(self.context_transitions)[-5:],
            'training_data_loaded': len(self.trained_market_conditions) > 0,
            'pattern_success_rates_available': len(self.pattern_success_rates) > 0,
            'ia1_accuracy_patterns_available': len(self.ia1_accuracy_patterns) > 0,
            'ia2_performance_patterns_available': len(self.ia2_performance_patterns) > 0
        }
        
        if self.current_context:
            status['current_context'] = {
                'regime': self.current_context.current_regime.value,
                'confidence': self.current_context.regime_confidence,
                'volatility_level': self.current_context.volatility_level,
                'trend_strength': self.current_context.trend_strength,
                'stress_level': self.current_context.market_stress_level,
                'duration_hours': self.current_context.context_duration,
                'timestamp': self.current_context.timestamp.isoformat()
            }
        
        return status

# Global instance
adaptive_context_system = AdaptiveContextSystem()
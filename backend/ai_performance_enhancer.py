"""
AI PERFORMANCE ENHANCER - Apply Training Insights to Improve Real-Time Trading
Integrates AI training results into IA1 and IA2 decision-making processes
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import json
import os
from ai_training_system import ai_training_system
from chartist_learning_system import chartist_learning_system, TradingDirection

logger = logging.getLogger(__name__)

@dataclass
class EnhancementRule:
    """Rule for enhancing trading decisions based on training insights"""
    rule_id: str
    rule_type: str  # "pattern_success", "market_condition", "ia1_accuracy", "ia2_performance"
    trigger_conditions: Dict[str, Any]
    enhancement_action: Dict[str, Any]
    success_rate: float
    confidence: float
    sample_size: int
    last_applied: Optional[datetime] = None
    application_count: int = 0
    effectiveness_score: float = 0.0

@dataclass
class TradingEnhancement:
    """Enhancement applied to a trading decision"""
    enhancement_type: str
    original_value: Any
    enhanced_value: Any
    enhancement_factor: float
    reasoning: str
    confidence: float
    rule_id: str

class AIPerformanceEnhancer:
    """Enhance trading bot performance using AI training insights"""
    
    def __init__(self):
        self.enhancement_rules = []
        self.applied_enhancements = deque(maxlen=1000)  # Track recent enhancements
        
        # Training data insights
        self.pattern_success_rates = {}
        self.market_condition_performance = {}
        self.ia1_accuracy_by_context = {}
        self.ia2_optimal_parameters = {}
        
        # Performance tracking
        self.enhancement_effectiveness = defaultdict(list)
        self.rule_performance = {}
        
        logger.info("AI Performance Enhancer initialized")
    
    def load_training_insights(self, ai_training_system):
        """Load insights from AI training system"""
        try:
            logger.info("Loading AI training insights for performance enhancement...")
            
            # 1. Extract pattern success rates by market condition
            self._extract_pattern_insights(ai_training_system.pattern_training)
            
            # 2. Extract market condition performance patterns
            self._extract_market_condition_insights(ai_training_system.market_conditions)
            
            # 3. Extract IA1 accuracy patterns
            self._extract_ia1_insights(ai_training_system.ia1_enhancements)
            
            # 4. Extract IA2 optimization insights
            self._extract_ia2_insights(ai_training_system.ia2_enhancements)
            
            # 5. Generate enhancement rules
            self._generate_enhancement_rules()
            
            # 6. NOUVEAU: Intégrer les insights des figures chartistes
            self._integrate_chartist_insights()
            
            logger.info(f"Loaded training insights: {len(self.enhancement_rules)} enhancement rules generated")
            
        except Exception as e:
            logger.error(f"Error loading training insights: {e}")
    
    def _integrate_chartist_insights(self):
        """Intègre les insights du système d'apprentissage des figures chartistes"""
        try:
            # Générer les stratégies chartistes optimisées
            chartist_strategies = chartist_learning_system.generate_chartist_strategies()
            
            # Convertir les stratégies chartistes en règles d'amélioration
            for strategy_name, strategy in chartist_strategies.items():
                
                # Règle de position sizing basée sur les figures chartistes
                position_rule = EnhancementRule(
                    rule_id=f"chartist_{strategy.pattern_type.value}_{strategy.direction.value}",
                    rule_type="chartist_pattern",
                    trigger_conditions={
                        'patterns_detected': [strategy.pattern_type.value],
                        'trading_direction': strategy.direction.value,
                        'market_context': strategy.market_context_filter
                    },
                    enhancement_action={
                        'type': 'chartist_position_sizing',
                        'factor': strategy.position_sizing_factor,
                        'max_risk': strategy.max_risk_per_trade,
                        'expected_rr': strategy.take_profit_targets[-1] if strategy.take_profit_targets else 2.0,
                        'reasoning': f"Figure chartiste {strategy.pattern_type.value} : {strategy.success_probability:.1%} de succès en {strategy.direction.value}"
                    },
                    success_rate=strategy.success_probability,
                    confidence=min(strategy.success_probability * 1.2, 0.95),
                    sample_size=50,  # Basé sur l'expertise chartiste
                    effectiveness_score=strategy.success_probability * strategy.position_sizing_factor
                )
                
                self.enhancement_rules.append(position_rule)
                
                # Règle de confirmation pour IA1
                if strategy.success_probability > 0.6:
                    ia1_rule = EnhancementRule(
                        rule_id=f"chartist_ia1_{strategy.pattern_type.value}",
                        rule_type="chartist_ia1_boost",
                        trigger_conditions={
                            'patterns_detected': [strategy.pattern_type.value],
                            'market_context': strategy.market_context_filter
                        },
                        enhancement_action={
                            'type': 'confidence_adjustment',
                            'factor': 1.0 + (strategy.success_probability - 0.5) * 0.4,  # Boost jusqu'à 20%
                            'reasoning': f"Figure chartiste {strategy.pattern_type.value} validée par l'analyse historique"
                        },
                        success_rate=strategy.success_probability,
                        confidence=0.8,
                        sample_size=30,
                        effectiveness_score=strategy.success_probability * 0.8
                    )
                    
                    self.enhancement_rules.append(ia1_rule)
            
            logger.info(f"Intégré {len(chartist_strategies)} stratégies chartistes dans {len([r for r in self.enhancement_rules if 'chartist' in r.rule_id])} règles d'amélioration")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'intégration des insights chartistes: {e}")

    def enhance_ia1_analysis_with_chartist(self, analysis: Dict[str, Any], market_context: str) -> Dict[str, Any]:
        """Améliore l'analyse IA1 avec les recommandations des figures chartistes"""
        enhanced_analysis = analysis.copy()
        applied_enhancements = []
        
        try:
            patterns_detected = analysis.get('patterns_detected', [])
            
            if patterns_detected:
                # Obtenir les recommandations chartistes
                from technical_pattern_detector import PatternType, TechnicalPattern
                
                # Créer des objets TechnicalPattern pour l'analyse
                mock_patterns = []
                for pattern_name in patterns_detected:
                    try:
                        pattern_type = PatternType(pattern_name)
                        mock_pattern = TechnicalPattern(
                            symbol=analysis.get('symbol', ''),
                            pattern_type=pattern_type,
                            confidence=0.8,
                            strength=0.7,
                            entry_price=0.0,
                            target_price=0.0,
                            stop_loss=0.0,
                            volume_confirmation=True
                        )
                        mock_patterns.append(mock_pattern)
                    except:
                        continue
                
                # Obtenir les recommandations chartistes
                recommendations = chartist_learning_system.get_pattern_recommendations(
                    mock_patterns, market_context
                )
                
                # Appliquer les améliorations basées sur les recommandations
                for rec in recommendations:
                    if rec['success_probability'] > 0.6:
                        # Boost de confiance pour les patterns fiables
                        confidence_boost = (rec['success_probability'] - 0.5) * 0.3
                        current_confidence = enhanced_analysis.get('analysis_confidence', 0.7)
                        enhanced_analysis['analysis_confidence'] = min(0.95, current_confidence + confidence_boost)
                        
                        # Ajouter les informations chartistes
                        enhanced_analysis['chartist_recommendation'] = {
                            'pattern_name': rec['pattern_name'],
                            'recommended_direction': rec['recommended_direction'],
                            'success_probability': rec['success_probability'],
                            'risk_reward_ratio': rec['risk_reward_ratio']
                        }
                        
                        enhancement = TradingEnhancement(
                            enhancement_type='chartist_confidence_boost',
                            original_value=current_confidence,
                            enhanced_value=enhanced_analysis['analysis_confidence'],
                            enhancement_factor=1 + confidence_boost,
                            reasoning=f"Figure chartiste {rec['pattern_name']} : {rec['success_probability']:.1%} de succès",
                            confidence=0.8,
                            rule_id=f"chartist_{rec['pattern_type']}"
                        )
                        
                        applied_enhancements.append(enhancement)
                        
                        logger.info(f"🎯 Amélioration chartiste IA1: {rec['pattern_name']} (+{confidence_boost:.1%} confiance)")
            
            if applied_enhancements:
                enhanced_analysis['chartist_enhancements'] = [
                    {
                        'type': e.enhancement_type,
                        'reasoning': e.reasoning,
                        'confidence': e.confidence
                    } for e in applied_enhancements
                ]
            
        except Exception as e:
            logger.error(f"Erreur dans l'amélioration chartiste IA1: {e}")
        
        return enhanced_analysis

    def enhance_ia2_decision_with_chartist(self, decision: Dict[str, Any], analysis: Dict[str, Any], 
                                         market_context: str) -> Dict[str, Any]:
        """Améliore la décision IA2 avec les insights des figures chartistes"""
        enhanced_decision = decision.copy()
        applied_enhancements = []
        
        try:
            patterns_detected = analysis.get('patterns_detected', [])
            signal_type = decision.get('signal', 'hold')
            
            if patterns_detected and signal_type != 'hold':
                # Convertir la direction de trading
                trading_direction = TradingDirection.LONG if signal_type == 'long' else TradingDirection.SHORT
                
                # Créer des patterns mock pour l'analyse
                from technical_pattern_detector import PatternType, TechnicalPattern
                mock_patterns = []
                
                for pattern_name in patterns_detected:
                    try:
                        pattern_type = PatternType(pattern_name)
                        mock_pattern = TechnicalPattern(
                            symbol=decision.get('symbol', ''),
                            pattern_type=pattern_type,
                            confidence=0.8,
                            strength=0.7,
                            entry_price=decision.get('entry_price', 0.0),
                            target_price=0.0,
                            stop_loss=0.0,
                            volume_confirmation=True
                        )
                        mock_patterns.append(mock_pattern)
                    except:
                        continue
                
                # Obtenir les recommandations chartistes
                recommendations = chartist_learning_system.get_pattern_recommendations(
                    mock_patterns, market_context
                )
                
                # Appliquer les améliorations
                for rec in recommendations:
                    if rec['recommended_direction'] == trading_direction.value:
                        # Ajuster la taille de position
                        current_size = enhanced_decision.get('position_size', 0.02)
                        chartist_factor = rec['position_sizing_factor']
                        enhanced_size = current_size * chartist_factor
                        enhanced_size = max(0.005, min(0.05, enhanced_size))  # Limites sécuritaires
                        
                        enhanced_decision['position_size'] = enhanced_size
                        
                        # Ajuster le risk-reward
                        optimal_rr = rec['risk_reward_ratio']
                        enhanced_decision['risk_reward_ratio'] = optimal_rr
                        
                        # Recalculer les take profits
                        entry_price = enhanced_decision.get('entry_price', 0)
                        stop_loss = enhanced_decision.get('stop_loss', 0)
                        
                        if entry_price > 0 and stop_loss > 0:
                            if signal_type == 'long':
                                risk_distance = entry_price - stop_loss
                                tp_targets = rec['take_profit_targets']
                                enhanced_decision['take_profit_1'] = entry_price + (risk_distance * tp_targets[0])
                                if len(tp_targets) > 1:
                                    enhanced_decision['take_profit_2'] = entry_price + (risk_distance * tp_targets[1])
                                if len(tp_targets) > 2:
                                    enhanced_decision['take_profit_3'] = entry_price + (risk_distance * tp_targets[2])
                                    
                            else:  # short
                                risk_distance = stop_loss - entry_price
                                tp_targets = rec['take_profit_targets']
                                enhanced_decision['take_profit_1'] = entry_price - (risk_distance * tp_targets[0])
                                if len(tp_targets) > 1:
                                    enhanced_decision['take_profit_2'] = entry_price - (risk_distance * tp_targets[1])
                                if len(tp_targets) > 2:
                                    enhanced_decision['take_profit_3'] = entry_price - (risk_distance * tp_targets[2])
                        
                        # Ajouter les informations chartistes
                        enhanced_decision['chartist_optimization'] = {
                            'pattern_name': rec['pattern_name'],
                            'success_probability': rec['success_probability'],
                            'position_sizing_factor': chartist_factor,
                            'risk_reward_optimized': optimal_rr,
                            'estimated_duration': rec['estimated_duration_days']
                        }
                        
                        enhancement = TradingEnhancement(
                            enhancement_type='chartist_position_optimization',
                            original_value=current_size,
                            enhanced_value=enhanced_size,
                            enhancement_factor=chartist_factor,
                            reasoning=f"Optimisation chartiste {rec['pattern_name']}: {rec['success_probability']:.1%} succès, RR {optimal_rr:.1f}:1",
                            confidence=rec['success_probability'],
                            rule_id=f"chartist_{rec['pattern_type']}"
                        )
                        
                        applied_enhancements.append(enhancement)
                        
                        logger.info(f"🎯 Optimisation chartiste IA2: {rec['pattern_name']} - Position {current_size:.1%} → {enhanced_size:.1%}")
                        
                        break  # Utiliser seulement la meilleure recommandation
            
            if applied_enhancements:
                enhanced_decision['chartist_enhancements'] = [
                    {
                        'type': e.enhancement_type,
                        'reasoning': e.reasoning,
                        'confidence': e.confidence,
                        'factor': e.enhancement_factor
                    } for e in applied_enhancements
                ]
            
        except Exception as e:
            logger.error(f"Erreur dans l'optimisation chartiste IA2: {e}")
        
        return enhanced_decision
    
    def _extract_pattern_insights(self, pattern_training: List):
        """Extract pattern success rate insights"""
        pattern_context_success = defaultdict(lambda: defaultdict(list))
        
        for pattern_data in pattern_training:
            pattern_type = pattern_data.pattern_type
            market_condition = pattern_data.market_condition
            success = pattern_data.success
            
            pattern_context_success[pattern_type][market_condition].append(success)
        
        # Calculate success rates
        for pattern_type, contexts in pattern_context_success.items():
            self.pattern_success_rates[pattern_type] = {}
            for context, successes in contexts.items():
                if len(successes) >= 5:  # Minimum sample size
                    success_rate = np.mean(successes)
                    confidence = min(len(successes) / 20, 1.0)  # Confidence based on sample size
                    
                    self.pattern_success_rates[pattern_type][context] = {
                        'success_rate': success_rate,
                        'sample_size': len(successes),
                        'confidence': confidence
                    }
        
        logger.info(f"Extracted insights for {len(self.pattern_success_rates)} pattern types")
    
    def _extract_market_condition_insights(self, market_conditions: List):
        """Extract market condition performance insights"""
        condition_performance = defaultdict(list)
        
        for condition in market_conditions:
            condition_type = condition.condition_type
            success_rate = condition.success_rate
            volatility = condition.volatility
            trend_strength = condition.trend_strength
            
            condition_performance[condition_type].append({
                'success_rate': success_rate,
                'volatility': volatility,
                'trend_strength': trend_strength,
                'confidence': condition.confidence_score
            })
        
        # Calculate aggregated insights
        for condition_type, performances in condition_performance.items():
            if len(performances) >= 3:
                avg_success = np.mean([p['success_rate'] for p in performances])
                avg_volatility = np.mean([p['volatility'] for p in performances])
                avg_trend_strength = np.mean([p['trend_strength'] for p in performances])
                
                self.market_condition_performance[condition_type] = {
                    'avg_success_rate': avg_success,
                    'avg_volatility': avg_volatility,
                    'avg_trend_strength': avg_trend_strength,
                    'sample_size': len(performances),
                    'optimal_position_size': self._calculate_optimal_position_size(avg_success, avg_volatility)
                }
        
        logger.info(f"Extracted insights for {len(self.market_condition_performance)} market conditions")
    
    def _extract_ia1_insights(self, ia1_enhancements: List):
        """Extract IA1 accuracy improvement insights"""
        context_accuracy = defaultdict(list)
        
        for enhancement in ia1_enhancements:
            context = enhancement.market_context
            accuracy = enhancement.prediction_accuracy
            
            context_accuracy[context].append(accuracy)
        
        # Calculate accuracy patterns
        for context, accuracies in context_accuracy.items():
            if len(accuracies) >= 5:
                avg_accuracy = np.mean(accuracies)
                std_accuracy = np.std(accuracies)
                
                self.ia1_accuracy_by_context[context] = {
                    'avg_accuracy': avg_accuracy,
                    'std_accuracy': std_accuracy,
                    'sample_size': len(accuracies),
                    'confidence_adjustment': self._calculate_confidence_adjustment(avg_accuracy)
                }
        
        logger.info(f"Extracted IA1 insights for {len(self.ia1_accuracy_by_context)} market contexts")
    
    def _extract_ia2_insights(self, ia2_enhancements: List):
        """Extract IA2 performance optimization insights"""
        performance_patterns = defaultdict(list)
        
        for enhancement in ia2_enhancements:
            signal = enhancement.decision_signal
            performance = enhancement.actual_performance
            confidence = enhancement.decision_confidence
            rr_realized = enhancement.risk_reward_realized
            
            if not np.isnan(rr_realized):
                performance_patterns[signal].append({
                    'performance': performance,
                    'confidence': confidence,
                    'rr_realized': rr_realized,
                    'optimal_exit_timing': enhancement.optimal_exit_timing
                })
        
        # Calculate optimal parameters
        for signal, performances in performance_patterns.items():
            if len(performances) >= 5:
                avg_performance = np.mean([p['performance'] for p in performances])
                avg_rr = np.mean([p['rr_realized'] for p in performances])
                avg_timing = np.mean([p['optimal_exit_timing'] for p in performances])
                
                self.ia2_optimal_parameters[signal] = {
                    'avg_performance': avg_performance,
                    'optimal_rr': avg_rr,
                    'optimal_timing_days': avg_timing,
                    'sample_size': len(performances),
                    'confidence_threshold': self._calculate_optimal_confidence_threshold(performances)
                }
        
        logger.info(f"Extracted IA2 insights for {len(self.ia2_optimal_parameters)} signal types")
    
    def _calculate_optimal_position_size(self, success_rate: float, volatility: float) -> float:
        """Calculate optimal position size based on success rate and volatility"""
        base_size = 0.02  # 2% base
        
        # Kelly Criterion inspired adjustment
        success_adjustment = (success_rate - 0.5) * 2  # -1 to +1
        volatility_adjustment = max(0.5, 1 - (volatility / 20))  # Reduce size for high volatility
        
        optimal_size = base_size * (1 + success_adjustment * 0.5) * volatility_adjustment
        return max(0.005, min(0.05, optimal_size))  # Clamp between 0.5% and 5%
    
    def _calculate_confidence_adjustment(self, accuracy: float) -> float:
        """Calculate confidence adjustment factor based on historical accuracy"""
        if accuracy > 0.8:
            return 1.2  # Boost confidence for high accuracy contexts
        elif accuracy > 0.6:
            return 1.0  # Neutral for moderate accuracy
        else:
            return 0.8  # Reduce confidence for low accuracy contexts
    
    def _calculate_optimal_confidence_threshold(self, performances: List[Dict]) -> float:
        """Calculate optimal confidence threshold for IA2 decisions"""
        # Sort by performance and find the threshold that maximizes return
        sorted_perfs = sorted(performances, key=lambda x: x['confidence'])
        
        best_threshold = 0.6
        best_avg_return = -float('inf')
        
        # Test different confidence thresholds
        for threshold in np.arange(0.5, 0.95, 0.05):
            filtered_perfs = [p for p in sorted_perfs if p['confidence'] >= threshold]
            if len(filtered_perfs) >= 3:
                avg_return = np.mean([p['performance'] for p in filtered_perfs])
                if avg_return > best_avg_return:
                    best_avg_return = avg_return
                    best_threshold = threshold
        
        return best_threshold
    
    def _generate_enhancement_rules(self):
        """Generate enhancement rules from training insights"""
        rules = []
        
        # 1. Pattern-based enhancement rules
        for pattern_type, contexts in self.pattern_success_rates.items():
            for context, stats in contexts.items():
                if stats['sample_size'] >= 5 and stats['confidence'] > 0.3:
                    
                    # Position sizing rule based on pattern success rate
                    position_factor = 0.5 + (stats['success_rate'] - 0.5)  # 0.5 to 1.5 multiplier
                    
                    rule = EnhancementRule(
                        rule_id=f"pattern_sizing_{pattern_type}_{context}",
                        rule_type="pattern_success",
                        trigger_conditions={
                            'patterns_detected': [pattern_type],
                            'market_condition': context
                        },
                        enhancement_action={
                            'type': 'position_sizing',
                            'factor': position_factor,
                            'reasoning': f"{pattern_type} has {stats['success_rate']:.1%} success in {context} conditions"
                        },
                        success_rate=stats['success_rate'],
                        confidence=stats['confidence'],
                        sample_size=stats['sample_size'],
                        effectiveness_score=stats['success_rate'] * stats['confidence']
                    )
                    rules.append(rule)
        
        # 2. Market condition enhancement rules
        for condition, stats in self.market_condition_performance.items():
            if stats['sample_size'] >= 3:
                
                rule = EnhancementRule(
                    rule_id=f"market_condition_{condition}",
                    rule_type="market_condition",
                    trigger_conditions={
                        'market_condition': condition
                    },
                    enhancement_action={
                        'type': 'position_sizing',
                        'optimal_size': stats['optimal_position_size'],
                        'reasoning': f"Optimal position size for {condition} market conditions"
                    },
                    success_rate=stats['avg_success_rate'],
                    confidence=min(stats['sample_size'] / 10, 1.0),
                    sample_size=stats['sample_size'],
                    effectiveness_score=stats['avg_success_rate']
                )
                rules.append(rule)
        
        # 3. IA1 accuracy enhancement rules
        for context, stats in self.ia1_accuracy_by_context.items():
            if stats['sample_size'] >= 5:
                
                rule = EnhancementRule(
                    rule_id=f"ia1_accuracy_{context}",
                    rule_type="ia1_accuracy",
                    trigger_conditions={
                        'market_context': context
                    },
                    enhancement_action={
                        'type': 'confidence_adjustment',
                        'factor': stats['confidence_adjustment'],
                        'reasoning': f"IA1 accuracy is {stats['avg_accuracy']:.1%} in {context} context"
                    },
                    success_rate=stats['avg_accuracy'],
                    confidence=min(stats['sample_size'] / 20, 1.0),
                    sample_size=stats['sample_size'],
                    effectiveness_score=abs(stats['avg_accuracy'] - 0.5) * 2
                )
                rules.append(rule)
        
        # 4. IA2 optimization rules
        for signal, stats in self.ia2_optimal_parameters.items():
            if stats['sample_size'] >= 5:
                
                # Risk-reward optimization rule
                if stats['optimal_rr'] > 1.5:
                    rule = EnhancementRule(
                        rule_id=f"ia2_rr_{signal}",
                        rule_type="ia2_performance",
                        trigger_conditions={
                            'signal_type': signal
                        },
                        enhancement_action={
                            'type': 'risk_reward_adjustment',
                            'optimal_rr': stats['optimal_rr'],
                            'confidence_threshold': stats['confidence_threshold'],
                            'reasoning': f"Optimal R:R for {signal} signals is {stats['optimal_rr']:.1f}:1"
                        },
                        success_rate=min(stats['avg_performance'] / 5, 1.0),  # Normalize performance to success rate
                        confidence=min(stats['sample_size'] / 15, 1.0),
                        sample_size=stats['sample_size'],
                        effectiveness_score=stats['avg_performance'] / 5
                    )
                    rules.append(rule)
        
        # Sort rules by effectiveness
        rules.sort(key=lambda x: x.effectiveness_score, reverse=True)
        
        self.enhancement_rules = rules
        logger.info(f"Generated {len(rules)} enhancement rules")
    
    def enhance_ia1_analysis(self, analysis: Dict[str, Any], market_context: str) -> Dict[str, Any]:
        """Enhance IA1 analysis using training insights"""
        enhanced_analysis = analysis.copy()
        applied_enhancements = []
        
        try:
            # Find applicable rules
            applicable_rules = self._find_applicable_rules(
                rule_type="ia1_accuracy",
                context={'market_context': market_context}
            )
            
            for rule in applicable_rules[:3]:  # Apply top 3 rules
                enhancement = self._apply_enhancement_rule(rule, enhanced_analysis)
                if enhancement:
                    applied_enhancements.append(enhancement)
                    
                    # Apply confidence adjustment
                    if enhancement.enhancement_type == 'confidence_adjustment':
                        original_confidence = enhanced_analysis.get('analysis_confidence', 0.7)
                        enhanced_confidence = original_confidence * enhancement.enhancement_factor
                        enhanced_analysis['analysis_confidence'] = min(0.95, max(0.5, enhanced_confidence))
            
            # Apply pattern-based enhancements
            patterns_detected = analysis.get('patterns_detected', [])
            for pattern in patterns_detected:
                pattern_rules = self._find_applicable_rules(
                    rule_type="pattern_success",
                    context={'patterns_detected': [pattern], 'market_condition': market_context}
                )
                
                for rule in pattern_rules[:1]:  # Apply top rule per pattern
                    enhancement = self._apply_enhancement_rule(rule, enhanced_analysis)
                    if enhancement:
                        applied_enhancements.append(enhancement)
            
            if applied_enhancements:
                enhanced_analysis['ai_enhancements'] = [
                    {
                        'type': e.enhancement_type,
                        'reasoning': e.reasoning,
                        'confidence': e.confidence,
                        'rule_id': e.rule_id
                    } for e in applied_enhancements
                ]
                
                logger.info(f"Applied {len(applied_enhancements)} enhancements to IA1 analysis for {analysis.get('symbol', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error enhancing IA1 analysis: {e}")
        
        return enhanced_analysis
    
    def enhance_ia2_decision(self, decision: Dict[str, Any], analysis: Dict[str, Any], market_context: str) -> Dict[str, Any]:
        """Enhance IA2 decision using training insights"""
        enhanced_decision = decision.copy()
        applied_enhancements = []
        
        try:
            signal_type = decision.get('signal', 'hold')
            
            # Find applicable rules
            applicable_rules = self._find_applicable_rules(
                rule_type="ia2_performance",
                context={'signal_type': signal_type}
            )
            
            # Apply IA2 optimization rules
            for rule in applicable_rules[:2]:  # Apply top 2 rules
                enhancement = self._apply_enhancement_rule(rule, enhanced_decision)
                if enhancement:
                    applied_enhancements.append(enhancement)
                    
                    # Apply risk-reward adjustments
                    if enhancement.enhancement_type == 'risk_reward_adjustment':
                        optimal_rr = rule.enhancement_action.get('optimal_rr', 2.0)
                        
                        # Adjust take profit levels based on optimal R:R
                        entry_price = enhanced_decision.get('entry_price', 0)
                        stop_loss = enhanced_decision.get('stop_loss', 0)
                        
                        if entry_price > 0 and stop_loss > 0:
                            if signal_type == 'long':
                                risk_distance = entry_price - stop_loss
                                optimal_tp = entry_price + (risk_distance * optimal_rr)
                                enhanced_decision['take_profit_1'] = optimal_tp
                            elif signal_type == 'short':
                                risk_distance = stop_loss - entry_price
                                optimal_tp = entry_price - (risk_distance * optimal_rr)
                                enhanced_decision['take_profit_1'] = optimal_tp
            
            # Apply market condition rules
            condition_rules = self._find_applicable_rules(
                rule_type="market_condition",
                context={'market_condition': market_context}
            )
            
            for rule in condition_rules[:1]:  # Apply top condition rule
                enhancement = self._apply_enhancement_rule(rule, enhanced_decision)
                if enhancement:
                    applied_enhancements.append(enhancement)
                    
                    # Apply position sizing optimization
                    if enhancement.enhancement_type == 'position_sizing':
                        optimal_size = rule.enhancement_action.get('optimal_size', 0.02)
                        enhanced_decision['position_size'] = optimal_size
            
            # Apply pattern-based position sizing
            patterns_detected = analysis.get('patterns_detected', [])
            for pattern in patterns_detected:
                pattern_rules = self._find_applicable_rules(
                    rule_type="pattern_success",
                    context={'patterns_detected': [pattern], 'market_condition': market_context}
                )
                
                for rule in pattern_rules[:1]:  # Apply top pattern rule
                    enhancement = self._apply_enhancement_rule(rule, enhanced_decision)
                    if enhancement:
                        applied_enhancements.append(enhancement)
                        
                        # Apply pattern-based position sizing
                        if enhancement.enhancement_type == 'position_sizing':
                            current_size = enhanced_decision.get('position_size', 0.02)
                            factor = enhancement.enhancement_factor
                            enhanced_size = current_size * factor
                            enhanced_decision['position_size'] = max(0.005, min(0.05, enhanced_size))
            
            if applied_enhancements:
                enhanced_decision['ai_enhancements'] = [
                    {
                        'type': e.enhancement_type,
                        'original_value': e.original_value,
                        'enhanced_value': e.enhanced_value,
                        'reasoning': e.reasoning,
                        'confidence': e.confidence,
                        'rule_id': e.rule_id
                    } for e in applied_enhancements
                ]
                
                logger.info(f"Applied {len(applied_enhancements)} enhancements to IA2 decision for {decision.get('symbol', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error enhancing IA2 decision: {e}")
        
        return enhanced_decision
    
    def _find_applicable_rules(self, rule_type: str, context: Dict[str, Any]) -> List[EnhancementRule]:
        """Find enhancement rules applicable to the given context"""
        applicable_rules = []
        
        for rule in self.enhancement_rules:
            if rule.rule_type != rule_type:
                continue
            
            # Check if rule conditions match context
            if self._check_rule_conditions(rule.trigger_conditions, context):
                applicable_rules.append(rule)
        
        # Sort by effectiveness score
        applicable_rules.sort(key=lambda x: x.effectiveness_score, reverse=True)
        
        return applicable_rules
    
    def _check_rule_conditions(self, conditions: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if rule conditions match the given context"""
        for key, value in conditions.items():
            if key not in context:
                return False
            
            context_value = context[key]
            
            if isinstance(value, list):
                # Check if any of the context values match
                if isinstance(context_value, list):
                    if not any(item in value for item in context_value):
                        return False
                else:
                    if context_value not in value:
                        return False
            else:
                if context_value != value:
                    return False
        
        return True
    
    def _apply_enhancement_rule(self, rule: EnhancementRule, target: Dict[str, Any]) -> Optional[TradingEnhancement]:
        """Apply an enhancement rule to a target analysis or decision"""
        try:
            action = rule.enhancement_action
            enhancement_type = action.get('type')
            
            if enhancement_type == 'confidence_adjustment':
                factor = action.get('factor', 1.0)
                original_value = target.get('confidence', 0.7)
                enhanced_value = original_value * factor
                
                enhancement = TradingEnhancement(
                    enhancement_type=enhancement_type,
                    original_value=original_value,
                    enhanced_value=enhanced_value,
                    enhancement_factor=factor,
                    reasoning=action.get('reasoning', ''),
                    confidence=rule.confidence,
                    rule_id=rule.rule_id
                )
                
                # Update rule tracking
                rule.last_applied = datetime.now()
                rule.application_count += 1
                
                return enhancement
            
            elif enhancement_type == 'position_sizing':
                if 'factor' in action:
                    factor = action.get('factor', 1.0)
                    original_value = target.get('position_size', 0.02)
                    enhanced_value = original_value * factor
                elif 'optimal_size' in action:
                    original_value = target.get('position_size', 0.02)
                    enhanced_value = action.get('optimal_size', 0.02)
                    factor = enhanced_value / original_value if original_value > 0 else 1.0
                else:
                    return None
                
                enhancement = TradingEnhancement(
                    enhancement_type=enhancement_type,
                    original_value=original_value,
                    enhanced_value=enhanced_value,
                    enhancement_factor=factor,
                    reasoning=action.get('reasoning', ''),
                    confidence=rule.confidence,
                    rule_id=rule.rule_id
                )
                
                # Update rule tracking
                rule.last_applied = datetime.now()
                rule.application_count += 1
                
                return enhancement
            
            elif enhancement_type == 'risk_reward_adjustment':
                optimal_rr = action.get('optimal_rr', 2.0)
                original_value = target.get('risk_reward_ratio', 2.0)
                
                enhancement = TradingEnhancement(
                    enhancement_type=enhancement_type,
                    original_value=original_value,
                    enhanced_value=optimal_rr,
                    enhancement_factor=optimal_rr / original_value if original_value > 0 else 1.0,
                    reasoning=action.get('reasoning', ''),
                    confidence=rule.confidence,
                    rule_id=rule.rule_id
                )
                
                # Update rule tracking
                rule.last_applied = datetime.now()
                rule.application_count += 1
                
                return enhancement
            
        except Exception as e:
            logger.error(f"Error applying enhancement rule {rule.rule_id}: {e}")
        
        return None
    
    def get_enhancement_summary(self) -> Dict[str, Any]:
        """Get summary of enhancement system performance"""
        active_rules = len([r for r in self.enhancement_rules if r.application_count > 0])
        total_applications = sum(r.application_count for r in self.enhancement_rules)
        
        # Rule type distribution
        rule_types = defaultdict(int)
        for rule in self.enhancement_rules:
            rule_types[rule.rule_type] += 1
        
        # Most effective rules
        top_rules = sorted(self.enhancement_rules, key=lambda x: x.effectiveness_score, reverse=True)[:5]
        
        return {
            'total_rules': len(self.enhancement_rules),
            'active_rules': active_rules,
            'total_applications': total_applications,
            'rule_type_distribution': dict(rule_types),
            'pattern_insights': len(self.pattern_success_rates),
            'market_condition_insights': len(self.market_condition_performance),
            'ia1_context_insights': len(self.ia1_accuracy_by_context),
            'ia2_optimization_insights': len(self.ia2_optimal_parameters),
            'top_rules': [
                {
                    'rule_id': rule.rule_id,
                    'rule_type': rule.rule_type,
                    'effectiveness_score': rule.effectiveness_score,
                    'application_count': rule.application_count,
                    'success_rate': rule.success_rate
                } for rule in top_rules
            ]
        }

# Global instance
ai_performance_enhancer = AIPerformanceEnhancer()
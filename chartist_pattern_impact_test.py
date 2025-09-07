#!/usr/bin/env python3
"""
Chartist Pattern Impact Analysis Test Suite
Focus: Analyser l'Impact des Patterns Chartistes sur la Qualité d'Interprétation d'IA1

This test suite specifically addresses the review request to analyze how chartist patterns
impact IA1's interpretation quality, including:
1. Explicit pattern usage in analyses
2. Pattern influence on recommendations
3. Technical analysis quality improvements
4. Conceptual integration
5. Concrete examples
6. Prompt engineering effectiveness
"""

import asyncio
import json
import logging
import os
import sys
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add backend to path
sys.path.append('/app/backend')

import requests
from motor.motor_asyncio import AsyncIOMotorClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChartistPatternImpactAnalyzer:
    """Analyzer for Chartist Pattern Impact on IA1 Interpretation Quality"""
    
    def __init__(self):
        # Get backend URL from frontend env
        try:
            with open('/app/frontend/.env', 'r') as f:
                for line in f:
                    if line.startswith('REACT_APP_BACKEND_URL='):
                        backend_url = line.split('=')[1].strip()
                        break
                else:
                    backend_url = "http://localhost:8001"
        except Exception:
            backend_url = "http://localhost:8001"
        
        self.api_url = f"{backend_url}/api"
        logger.info(f"Testing backend at: {self.api_url}")
        
        # MongoDB connection for direct data access
        self.mongo_client = None
        self.db = None
        
        # Test results
        self.test_results = []
        
        # Pattern analysis results
        self.pattern_analysis = {
            'explicit_usage': [],
            'recommendation_influence': [],
            'quality_improvements': [],
            'conceptual_integration': [],
            'concrete_examples': [],
            'prompt_effectiveness': []
        }
        
    async def setup_database(self):
        """Setup database connection"""
        try:
            # Get MongoDB URL from backend env
            mongo_url = "mongodb://localhost:27017"  # Default
            try:
                with open('/app/backend/.env', 'r') as f:
                    for line in f:
                        if line.startswith('MONGO_URL='):
                            mongo_url = line.split('=')[1].strip().strip('"')
                            break
            except Exception:
                pass
            
            self.mongo_client = AsyncIOMotorClient(mongo_url)
            self.db = self.mongo_client['myapp']
            logger.info("✅ Database connection established")
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            
    async def cleanup_database(self):
        """Cleanup database connection"""
        if self.mongo_client:
            self.mongo_client.close()
            
    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if details:
            logger.info(f"   Details: {details}")
        
        self.test_results.append({
            'test': test_name,
            'success': success,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
        
    def get_analyses_from_api(self):
        """Helper method to get analyses from API"""
        try:
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            if response.status_code != 200:
                return None, f"API error: {response.status_code}"
                
            data = response.json()
            
            # Handle API response format
            if isinstance(data, dict) and 'analyses' in data:
                analyses = data['analyses']
            else:
                analyses = data
                
            return analyses, None
        except Exception as e:
            return None, f"Exception: {str(e)}"
    
    def get_decisions_from_api(self):
        """Helper method to get IA2 decisions from API"""
        try:
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            if response.status_code != 200:
                return None, f"API error: {response.status_code}"
                
            data = response.json()
            
            # Handle API response format
            if isinstance(data, dict) and 'decisions' in data:
                decisions = data['decisions']
            else:
                decisions = data
                
            return decisions, None
        except Exception as e:
            return None, f"Exception: {str(e)}"
            
    async def test_explicit_pattern_usage(self):
        """Test 1: Utilisation Explicite des Patterns - Verify IA1 explicitly mentions detected patterns"""
        logger.info("\n🔍 TEST 1: Utilisation Explicite des Patterns")
        
        try:
            analyses, error = self.get_analyses_from_api()
            
            if error:
                self.log_test_result("Explicit Pattern Usage", False, error)
                return
                
            if not analyses:
                self.log_test_result("Explicit Pattern Usage", False, "No IA1 analyses found")
                return
                
            explicit_mentions = 0
            pattern_explanations = 0
            impact_descriptions = 0
            
            for analysis in analyses:
                symbol = analysis.get('symbol', 'UNKNOWN')
                patterns_detected = analysis.get('patterns_detected', [])
                ia1_reasoning = analysis.get('ia1_reasoning', '')
                
                if patterns_detected:
                    # Check for explicit pattern mentions
                    explicit_pattern_phrases = [
                        "Le pattern", "La formation", "pattern indique", "formation suggère",
                        "MASTER PATTERN", "PATTERN", "bearish_channel", "diamond_bottom",
                        "Golden Cross Formation", "Trend Continuation", "Volume Spike"
                    ]
                    
                    has_explicit_mention = any(phrase in ia1_reasoning for phrase in explicit_pattern_phrases)
                    if has_explicit_mention:
                        explicit_mentions += 1
                        logger.info(f"   ✅ {symbol}: Explicit pattern mention found")
                        
                        # Look for pattern explanations
                        explanation_phrases = [
                            "suggests", "indicates", "pattern", "formation", "breakout", 
                            "support", "resistance", "channel", "triangle", "diamond"
                        ]
                        
                        if any(phrase in ia1_reasoning.lower() for phrase in explanation_phrases):
                            pattern_explanations += 1
                            
                        # Look for impact on recommendations
                        impact_phrases = [
                            "PRIMARY BASIS", "strategic decision", "pattern is", "formation is",
                            "suggests", "indicates", "recommends", "advise"
                        ]
                        
                        if any(phrase in ia1_reasoning for phrase in impact_phrases):
                            impact_descriptions += 1
                            
                        # Store example for detailed analysis
                        self.pattern_analysis['explicit_usage'].append({
                            'symbol': symbol,
                            'patterns': patterns_detected,
                            'reasoning_snippet': ia1_reasoning[:300] + "..." if len(ia1_reasoning) > 300 else ia1_reasoning,
                            'has_explicit_mention': has_explicit_mention,
                            'has_explanation': pattern_explanations > 0,
                            'has_impact_description': impact_descriptions > 0
                        })
                    else:
                        logger.info(f"   ⚠️ {symbol}: No explicit pattern mention despite {len(patterns_detected)} patterns detected")
                        
            success = explicit_mentions > 0 and pattern_explanations > 0
            details = f"Explicit mentions: {explicit_mentions}/{len([a for a in analyses if a.get('patterns_detected')])}, Pattern explanations: {pattern_explanations}, Impact descriptions: {impact_descriptions}"
            
            self.log_test_result("Explicit Pattern Usage", success, details)
            
        except Exception as e:
            self.log_test_result("Explicit Pattern Usage", False, f"Exception: {str(e)}")
            
    async def test_recommendation_influence(self):
        """Test 2: Influence sur les Recommandations - Check if patterns influence LONG/SHORT/HOLD signals"""
        logger.info("\n🔍 TEST 2: Influence sur les Recommandations")
        
        try:
            analyses, error = self.get_analyses_from_api()
            
            if error:
                self.log_test_result("Recommendation Influence", False, error)
                return
                
            # Analyze pattern influence on recommendations
            pattern_influenced_signals = 0
            confidence_adjustments = 0
            signal_distribution = {'long': 0, 'short': 0, 'hold': 0}
            
            for analysis in analyses:
                symbol = analysis.get('symbol', 'UNKNOWN')
                patterns_detected = analysis.get('patterns_detected', [])
                ia1_signal = analysis.get('ia1_signal', 'hold')
                confidence = analysis.get('analysis_confidence', 0)
                ia1_reasoning = analysis.get('ia1_reasoning', '')
                
                signal_distribution[ia1_signal] += 1
                
                if patterns_detected:
                    # Check if pattern influences signal decision
                    pattern_signal_phrases = [
                        "LONG", "SHORT", "HOLD", "bullish", "bearish", "neutral",
                        "buy", "sell", "wait", "breakout", "breakdown", "continuation"
                    ]
                    
                    has_signal_influence = any(phrase in ia1_reasoning for phrase in pattern_signal_phrases)
                    if has_signal_influence:
                        pattern_influenced_signals += 1
                        
                    # Check for confidence adjustments based on patterns
                    confidence_phrases = [
                        "confidence", "strength", "probability", "certainty", "conviction"
                    ]
                    
                    if any(phrase in ia1_reasoning.lower() for phrase in confidence_phrases):
                        confidence_adjustments += 1
                        
                    # Store example for analysis
                    self.pattern_analysis['recommendation_influence'].append({
                        'symbol': symbol,
                        'patterns': patterns_detected,
                        'signal': ia1_signal,
                        'confidence': confidence,
                        'has_signal_influence': has_signal_influence,
                        'reasoning_snippet': ia1_reasoning[:200] + "..." if len(ia1_reasoning) > 200 else ia1_reasoning
                    })
                    
                    logger.info(f"   📊 {symbol}: {ia1_signal.upper()} signal, {len(patterns_detected)} patterns, confidence: {confidence:.2f}")
                    
            # Check if patterns create signal diversity (not all HOLD)
            signal_diversity = len([s for s in signal_distribution.values() if s > 0])
            trading_signals = signal_distribution['long'] + signal_distribution['short']
            
            success = pattern_influenced_signals > 0 and signal_diversity >= 2 and trading_signals > 0
            details = f"Pattern-influenced signals: {pattern_influenced_signals}, Signal distribution: {signal_distribution}, Confidence adjustments: {confidence_adjustments}"
            
            self.log_test_result("Recommendation Influence", success, details)
            
        except Exception as e:
            self.log_test_result("Recommendation Influence", False, f"Exception: {str(e)}")
            
    async def test_technical_analysis_quality(self):
        """Test 3: Qualité de l'Analyse Technique - Evaluate if analyses are more detailed with patterns"""
        logger.info("\n🔍 TEST 3: Qualité de l'Analyse Technique")
        
        try:
            analyses, error = self.get_analyses_from_api()
            
            if error:
                self.log_test_result("Technical Analysis Quality", False, error)
                return
                
            # Compare analyses with and without patterns
            with_patterns = []
            without_patterns = []
            
            for analysis in analyses:
                symbol = analysis.get('symbol', 'UNKNOWN')
                patterns_detected = analysis.get('patterns_detected', [])
                ia1_reasoning = analysis.get('ia1_reasoning', '')
                confidence = analysis.get('analysis_confidence', 0)
                
                analysis_metrics = {
                    'symbol': symbol,
                    'reasoning_length': len(ia1_reasoning),
                    'confidence': confidence,
                    'has_entry_exit': any(phrase in ia1_reasoning.lower() for phrase in ['entry', 'exit', 'target', 'stop', 'breakout']),
                    'has_technical_details': any(phrase in ia1_reasoning.lower() for phrase in ['rsi', 'macd', 'bollinger', 'support', 'resistance']),
                    'has_justification': any(phrase in ia1_reasoning.lower() for phrase in ['because', 'since', 'due to', 'suggests', 'indicates'])
                }
                
                if patterns_detected:
                    analysis_metrics['patterns_count'] = len(patterns_detected)
                    with_patterns.append(analysis_metrics)
                else:
                    without_patterns.append(analysis_metrics)
                    
            # Calculate quality metrics
            if with_patterns:
                avg_length_with_patterns = sum(a['reasoning_length'] for a in with_patterns) / len(with_patterns)
                avg_confidence_with_patterns = sum(a['confidence'] for a in with_patterns) / len(with_patterns)
                entry_exit_with_patterns = sum(1 for a in with_patterns if a['has_entry_exit']) / len(with_patterns)
                technical_details_with_patterns = sum(1 for a in with_patterns if a['has_technical_details']) / len(with_patterns)
                
                logger.info(f"   📊 Analyses WITH patterns ({len(with_patterns)}):")
                logger.info(f"      Average reasoning length: {avg_length_with_patterns:.0f} chars")
                logger.info(f"      Average confidence: {avg_confidence_with_patterns:.2f}")
                logger.info(f"      Entry/exit points: {entry_exit_with_patterns:.1%}")
                logger.info(f"      Technical details: {technical_details_with_patterns:.1%}")
                
                # Store quality analysis
                self.pattern_analysis['quality_improvements'] = {
                    'with_patterns_count': len(with_patterns),
                    'avg_length_with_patterns': avg_length_with_patterns,
                    'avg_confidence_with_patterns': avg_confidence_with_patterns,
                    'entry_exit_rate': entry_exit_with_patterns,
                    'technical_details_rate': technical_details_with_patterns
                }
                
            if without_patterns:
                avg_length_without_patterns = sum(a['reasoning_length'] for a in without_patterns) / len(without_patterns)
                avg_confidence_without_patterns = sum(a['confidence'] for a in without_patterns) / len(without_patterns)
                
                logger.info(f"   📊 Analyses WITHOUT patterns ({len(without_patterns)}):")
                logger.info(f"      Average reasoning length: {avg_length_without_patterns:.0f} chars")
                logger.info(f"      Average confidence: {avg_confidence_without_patterns:.2f}")
                
            # Quality assessment
            quality_indicators = 0
            if with_patterns:
                if avg_confidence_with_patterns > 0.7:
                    quality_indicators += 1
                if entry_exit_with_patterns > 0.5:
                    quality_indicators += 1
                if technical_details_with_patterns > 0.8:
                    quality_indicators += 1
                if avg_length_with_patterns > 500:
                    quality_indicators += 1
                    
            success = quality_indicators >= 3 and len(with_patterns) > 0
            details = f"Quality indicators met: {quality_indicators}/4, Analyses with patterns: {len(with_patterns)}, Average confidence: {avg_confidence_with_patterns:.2f}"
            
            self.log_test_result("Technical Analysis Quality", success, details)
            
        except Exception as e:
            self.log_test_result("Technical Analysis Quality", False, f"Exception: {str(e)}")
            
    async def test_conceptual_integration(self):
        """Test 4: Intégration Conceptuelle - Test pattern understanding and integration"""
        logger.info("\n🔍 TEST 4: Intégration Conceptuelle")
        
        try:
            analyses, error = self.get_analyses_from_api()
            
            if error:
                self.log_test_result("Conceptual Integration", False, error)
                return
                
            # Look for conceptual understanding of different pattern types
            harmonic_patterns = 0
            consolidation_patterns = 0
            volatility_patterns = 0
            trend_patterns = 0
            
            conceptual_understanding = 0
            
            for analysis in analyses:
                symbol = analysis.get('symbol', 'UNKNOWN')
                patterns_detected = analysis.get('patterns_detected', [])
                ia1_reasoning = analysis.get('ia1_reasoning', '')
                
                if patterns_detected:
                    # Check for different pattern categories
                    for pattern in patterns_detected:
                        pattern_lower = pattern.lower()
                        
                        # Harmonic patterns
                        if any(term in pattern_lower for term in ['gartley', 'butterfly', 'bat', 'crab', 'harmonic']):
                            harmonic_patterns += 1
                            
                        # Consolidation patterns
                        if any(term in pattern_lower for term in ['consolidation', 'rectangle', 'pennant', 'flag', 'triangle']):
                            consolidation_patterns += 1
                            
                        # Volatility patterns
                        if any(term in pattern_lower for term in ['volatility', 'spike', 'breakout', 'breakdown']):
                            volatility_patterns += 1
                            
                        # Trend patterns
                        if any(term in pattern_lower for term in ['trend', 'continuation', 'reversal', 'channel']):
                            trend_patterns += 1
                            
                    # Check for conceptual understanding in reasoning
                    conceptual_phrases = [
                        'implications', 'suggests', 'indicates', 'pattern', 'formation',
                        'breakout', 'support', 'resistance', 'trend', 'reversal',
                        'continuation', 'consolidation', 'volatility'
                    ]
                    
                    if any(phrase in ia1_reasoning.lower() for phrase in conceptual_phrases):
                        conceptual_understanding += 1
                        
                    # Store conceptual analysis
                    self.pattern_analysis['conceptual_integration'].append({
                        'symbol': symbol,
                        'patterns': patterns_detected,
                        'has_conceptual_understanding': any(phrase in ia1_reasoning.lower() for phrase in conceptual_phrases),
                        'reasoning_snippet': ia1_reasoning[:250] + "..." if len(ia1_reasoning) > 250 else ia1_reasoning
                    })
                    
                    logger.info(f"   📊 {symbol}: {len(patterns_detected)} patterns, conceptual understanding: {'✅' if conceptual_understanding > 0 else '❌'}")
                    
            # Pattern diversity assessment
            pattern_categories = sum([
                1 if harmonic_patterns > 0 else 0,
                1 if consolidation_patterns > 0 else 0,
                1 if volatility_patterns > 0 else 0,
                1 if trend_patterns > 0 else 0
            ])
            
            success = conceptual_understanding > 0 and pattern_categories >= 2
            details = f"Conceptual understanding: {conceptual_understanding} analyses, Pattern categories: {pattern_categories}/4 (Harmonic: {harmonic_patterns}, Consolidation: {consolidation_patterns}, Volatility: {volatility_patterns}, Trend: {trend_patterns})"
            
            self.log_test_result("Conceptual Integration", success, details)
            
        except Exception as e:
            self.log_test_result("Conceptual Integration", False, f"Exception: {str(e)}")
            
    async def test_concrete_examples(self):
        """Test 5: Exemples Concrets - Find specific examples of pattern usage"""
        logger.info("\n🔍 TEST 5: Exemples Concrets")
        
        try:
            analyses, error = self.get_analyses_from_api()
            
            if error:
                self.log_test_result("Concrete Examples", False, error)
                return
                
            # Look for specific pattern usage examples
            triple_top_examples = []
            gartley_examples = []
            pattern_target_examples = []
            
            for analysis in analyses:
                symbol = analysis.get('symbol', 'UNKNOWN')
                patterns_detected = analysis.get('patterns_detected', [])
                ia1_reasoning = analysis.get('ia1_reasoning', '')
                ia1_signal = analysis.get('ia1_signal', 'hold')
                
                if patterns_detected:
                    # Look for specific pattern mentions with implications
                    reasoning_lower = ia1_reasoning.lower()
                    
                    # Triple Top pattern examples
                    if any(term in reasoning_lower for term in ['triple', 'top', 'reversal', 'bearish']):
                        triple_top_examples.append({
                            'symbol': symbol,
                            'patterns': patterns_detected,
                            'signal': ia1_signal,
                            'reasoning': ia1_reasoning[:200] + "..."
                        })
                        
                    # Gartley/Harmonic pattern examples
                    if any(term in reasoning_lower for term in ['gartley', 'harmonic', 'bullish', 'continuation']):
                        gartley_examples.append({
                            'symbol': symbol,
                            'patterns': patterns_detected,
                            'signal': ia1_signal,
                            'reasoning': ia1_reasoning[:200] + "..."
                        })
                        
                    # Pattern with target adjustment examples
                    if any(term in reasoning_lower for term in ['target', 'entry', 'breakout', 'resistance', 'support']):
                        pattern_target_examples.append({
                            'symbol': symbol,
                            'patterns': patterns_detected,
                            'signal': ia1_signal,
                            'has_targets': True,
                            'reasoning': ia1_reasoning[:200] + "..."
                        })
                        
            # Log concrete examples
            if triple_top_examples:
                logger.info(f"   🎯 Triple Top/Reversal Examples: {len(triple_top_examples)}")
                for example in triple_top_examples[:2]:
                    logger.info(f"      {example['symbol']}: {example['signal'].upper()} - {example['patterns']}")
                    
            if gartley_examples:
                logger.info(f"   🎯 Gartley/Harmonic Examples: {len(gartley_examples)}")
                for example in gartley_examples[:2]:
                    logger.info(f"      {example['symbol']}: {example['signal'].upper()} - {example['patterns']}")
                    
            if pattern_target_examples:
                logger.info(f"   🎯 Pattern Target Examples: {len(pattern_target_examples)}")
                for example in pattern_target_examples[:3]:
                    logger.info(f"      {example['symbol']}: {example['signal'].upper()} - Target adjustment based on patterns")
                    
            # Store examples
            self.pattern_analysis['concrete_examples'] = {
                'triple_top_examples': triple_top_examples,
                'gartley_examples': gartley_examples,
                'pattern_target_examples': pattern_target_examples
            }
            
            total_examples = len(triple_top_examples) + len(gartley_examples) + len(pattern_target_examples)
            success = total_examples > 0 and len(pattern_target_examples) > 0
            details = f"Total concrete examples: {total_examples} (Triple Top: {len(triple_top_examples)}, Gartley: {len(gartley_examples)}, Target adjustment: {len(pattern_target_examples)})"
            
            self.log_test_result("Concrete Examples", success, details)
            
        except Exception as e:
            self.log_test_result("Concrete Examples", False, f"Exception: {str(e)}")
            
    async def test_prompt_engineering_effectiveness(self):
        """Test 6: Prompt Engineering Effectiveness - Verify prompt produces sophisticated responses"""
        logger.info("\n🔍 TEST 6: Prompt Engineering Effectiveness")
        
        try:
            analyses, error = self.get_analyses_from_api()
            
            if error:
                self.log_test_result("Prompt Engineering Effectiveness", False, error)
                return
                
            # Analyze prompt effectiveness
            sophisticated_responses = 0
            pattern_integration_in_json = 0
            analysis_field_quality = 0
            reasoning_field_quality = 0
            
            for analysis in analyses:
                symbol = analysis.get('symbol', 'UNKNOWN')
                patterns_detected = analysis.get('patterns_detected', [])
                ia1_reasoning = analysis.get('ia1_reasoning', '')
                confidence = analysis.get('analysis_confidence', 0)
                
                if patterns_detected:
                    # Check for sophisticated response indicators
                    sophistication_indicators = [
                        len(ia1_reasoning) > 500,  # Detailed reasoning
                        confidence > 0.7,  # High confidence
                        'analysis' in ia1_reasoning.lower(),  # Structured analysis
                        'reasoning' in ia1_reasoning.lower(),  # Clear reasoning
                        any(pattern.lower() in ia1_reasoning.lower() for pattern in patterns_detected)  # Pattern integration
                    ]
                    
                    if sum(sophistication_indicators) >= 3:
                        sophisticated_responses += 1
                        
                    # Check for pattern integration in structured format
                    if patterns_detected and any(pattern.lower() in ia1_reasoning.lower() for pattern in patterns_detected):
                        pattern_integration_in_json += 1
                        
                    # Check analysis field quality
                    if 'analysis' in ia1_reasoning.lower() and len(ia1_reasoning) > 300:
                        analysis_field_quality += 1
                        
                    # Check reasoning field quality
                    if 'reasoning' in ia1_reasoning.lower() and any(word in ia1_reasoning.lower() for word in ['suggests', 'indicates', 'because', 'since']):
                        reasoning_field_quality += 1
                        
                    logger.info(f"   📊 {symbol}: Sophistication score: {sum(sophistication_indicators)}/5, Length: {len(ia1_reasoning)} chars")
                    
            # Store prompt effectiveness analysis
            self.pattern_analysis['prompt_effectiveness'] = {
                'sophisticated_responses': sophisticated_responses,
                'pattern_integration_rate': pattern_integration_in_json,
                'analysis_field_quality': analysis_field_quality,
                'reasoning_field_quality': reasoning_field_quality,
                'total_analyses_with_patterns': len([a for a in analyses if a.get('patterns_detected')])
            }
            
            total_with_patterns = len([a for a in analyses if a.get('patterns_detected')])
            success = (sophisticated_responses > 0 and 
                      pattern_integration_in_json > 0 and 
                      total_with_patterns > 0)
            
            details = f"Sophisticated responses: {sophisticated_responses}/{total_with_patterns}, Pattern integration: {pattern_integration_in_json}, Analysis quality: {analysis_field_quality}, Reasoning quality: {reasoning_field_quality}"
            
            self.log_test_result("Prompt Engineering Effectiveness", success, details)
            
        except Exception as e:
            self.log_test_result("Prompt Engineering Effectiveness", False, f"Exception: {str(e)}")
            
    async def generate_comprehensive_report(self):
        """Generate comprehensive report on chartist pattern impact"""
        logger.info("\n📊 GENERATING COMPREHENSIVE CHARTIST PATTERN IMPACT REPORT")
        logger.info("=" * 80)
        
        # Analyze overall pattern impact
        analyses, _ = self.get_analyses_from_api()
        
        if analyses:
            total_analyses = len(analyses)
            analyses_with_patterns = len([a for a in analyses if a.get('patterns_detected')])
            
            logger.info(f"📈 OVERALL PATTERN INTEGRATION STATISTICS:")
            logger.info(f"   Total IA1 analyses: {total_analyses}")
            logger.info(f"   Analyses with patterns: {analyses_with_patterns} ({analyses_with_patterns/total_analyses:.1%})")
            
            # Pattern usage statistics
            all_patterns = []
            for analysis in analyses:
                patterns = analysis.get('patterns_detected', [])
                all_patterns.extend(patterns)
                
            unique_patterns = list(set(all_patterns))
            logger.info(f"   Unique pattern types detected: {len(unique_patterns)}")
            logger.info(f"   Most common patterns: {unique_patterns[:5]}")
            
            # Signal distribution with patterns
            signal_dist = {'long': 0, 'short': 0, 'hold': 0}
            for analysis in analyses:
                if analysis.get('patterns_detected'):
                    signal = analysis.get('ia1_signal', 'hold')
                    signal_dist[signal] += 1
                    
            logger.info(f"   Signal distribution (with patterns): {signal_dist}")
            
            # Quality metrics
            if self.pattern_analysis['quality_improvements']:
                quality = self.pattern_analysis['quality_improvements']
                logger.info(f"   Average confidence with patterns: {quality.get('avg_confidence_with_patterns', 0):.2f}")
                logger.info(f"   Entry/exit point rate: {quality.get('entry_exit_rate', 0):.1%}")
                
        # Specific findings
        logger.info(f"\n🔍 SPECIFIC FINDINGS:")
        
        # Explicit usage findings
        explicit_usage = self.pattern_analysis.get('explicit_usage', [])
        if explicit_usage:
            explicit_count = len([e for e in explicit_usage if e['has_explicit_mention']])
            logger.info(f"   ✅ Explicit pattern mentions: {explicit_count}/{len(explicit_usage)} analyses")
            
            # Show best example
            best_example = max(explicit_usage, key=lambda x: len(x['reasoning_snippet']))
            logger.info(f"   🏆 Best example: {best_example['symbol']} - {best_example['patterns']}")
            logger.info(f"      Reasoning: {best_example['reasoning_snippet'][:150]}...")
            
        # Recommendation influence findings
        rec_influence = self.pattern_analysis.get('recommendation_influence', [])
        if rec_influence:
            influenced_count = len([r for r in rec_influence if r['has_signal_influence']])
            logger.info(f"   ✅ Pattern-influenced recommendations: {influenced_count}/{len(rec_influence)} analyses")
            
        # Concrete examples findings
        concrete = self.pattern_analysis.get('concrete_examples', {})
        if concrete:
            total_concrete = (len(concrete.get('triple_top_examples', [])) + 
                            len(concrete.get('gartley_examples', [])) + 
                            len(concrete.get('pattern_target_examples', [])))
            logger.info(f"   ✅ Concrete pattern usage examples: {total_concrete}")
            
        logger.info(f"\n🎯 CHARTIST PATTERN IMPACT ASSESSMENT:")
        
        # Calculate overall impact score
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        impact_score = passed_tests / total_tests if total_tests > 0 else 0
        
        if impact_score >= 0.8:
            logger.info(f"   🟢 HIGH IMPACT: Chartist patterns significantly improve IA1 interpretation quality")
        elif impact_score >= 0.6:
            logger.info(f"   🟡 MODERATE IMPACT: Chartist patterns provide some improvement to IA1 analysis")
        else:
            logger.info(f"   🔴 LOW IMPACT: Chartist patterns have limited effect on IA1 interpretation quality")
            
        logger.info(f"   Impact Score: {impact_score:.1%} ({passed_tests}/{total_tests} tests passed)")
        
        return impact_score
            
    async def run_comprehensive_analysis(self):
        """Run comprehensive chartist pattern impact analysis"""
        logger.info("🚀 Starting Chartist Pattern Impact Analysis")
        logger.info("Focus: Analyser l'Impact des Patterns Chartistes sur la Qualité d'Interprétation d'IA1")
        logger.info("=" * 80)
        
        await self.setup_database()
        
        # Run all analysis tests
        await self.test_explicit_pattern_usage()
        await self.test_recommendation_influence()
        await self.test_technical_analysis_quality()
        await self.test_conceptual_integration()
        await self.test_concrete_examples()
        await self.test_prompt_engineering_effectiveness()
        
        # Generate comprehensive report
        impact_score = await self.generate_comprehensive_report()
        
        await self.cleanup_database()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 CHARTIST PATTERN IMPACT ANALYSIS SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed (Impact Score: {impact_score:.1%})")
        
        if impact_score >= 0.8:
            logger.info("🎉 HIGH IMPACT - Chartist patterns significantly enhance IA1 interpretation quality!")
        elif impact_score >= 0.6:
            logger.info("⚠️ MODERATE IMPACT - Chartist patterns provide some improvements to IA1 analysis")
        else:
            logger.info("❌ LOW IMPACT - Chartist pattern integration needs significant improvement")
            
        return passed_tests, total_tests, impact_score

async def main():
    """Main analysis execution"""
    analyzer = ChartistPatternImpactAnalyzer()
    passed, total, impact_score = await analyzer.run_comprehensive_analysis()
    
    # Exit with appropriate code based on impact score
    if impact_score >= 0.6:
        sys.exit(0)  # Acceptable impact
    else:
        sys.exit(1)  # Low impact needs attention

if __name__ == "__main__":
    asyncio.run(main())
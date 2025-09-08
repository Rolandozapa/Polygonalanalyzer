#!/usr/bin/env python3
"""
Backend Testing Suite for Chartist Learning System Integration
Focus: Testing the new chartist learning system with pattern-based strategies and market context adaptation
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import subprocess

# Add backend to path
sys.path.append('/app/backend')

import requests
from motor.motor_asyncio import AsyncIOMotorClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChartistLearningSystemTestSuite:
    """Test suite for Chartist Learning System Integration"""
    
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
        logger.info(f"Testing Chartist Learning System at: {self.api_url}")
        
        # MongoDB connection for direct data access
        self.mongo_client = None
        self.db = None
        
        # Test results
        self.test_results = []
        
        # Chartist system specific test data
        self.chartist_library = None
        self.chartist_analysis = None
        self.ai_training_insights = None
        
        # Test patterns for chartist analysis
        self.test_patterns = [
            "head_and_shoulders",
            "bullish_flag", 
            "ascending_triangle",
            "cup_with_handle",
            "double_bottom",
            "falling_wedge"
        ]
        
        # Market contexts to test
        self.market_contexts = ["BULL", "BEAR", "SIDEWAYS", "VOLATILE"]
        
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
        """Helper method to get analyses from API with proper format handling"""
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
        """Helper method to get decisions from API"""
        try:
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            if response.status_code != 200:
                return None, f"API error: {response.status_code}"
                
            data = response.json()
            # Extract decisions from response if needed
            if isinstance(data, dict) and 'decisions' in data:
                decisions = data['decisions']
            else:
                decisions = data
            return decisions, None
        except Exception as e:
            return None, f"Exception: {str(e)}"
    
    async def test_chartist_library_endpoint(self):
        """Test 1: /api/chartist/library - Should return complete chartist figures library with statistics"""
        logger.info("\n🔍 TEST 1: Chartist Library Endpoint")
        
        try:
            response = requests.get(f"{self.api_url}/chartist/library", timeout=15)
            
            if response.status_code != 200:
                self.log_test_result("Chartist Library Endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            
            # Validate response structure
            if not data.get('success', False):
                self.log_test_result("Chartist Library Endpoint", False, f"API returned success=False: {data.get('message', 'No message')}")
                return
            
            # Validate library data
            library_data = data.get('data', {})
            expected_fields = ['total_patterns', 'pattern_categories', 'patterns', 'statistics']
            missing_fields = [field for field in expected_fields if field not in library_data]
            
            if missing_fields:
                self.log_test_result("Chartist Library Endpoint", False, f"Missing library fields: {missing_fields}")
                return
            
            # Store for later tests
            self.chartist_library = library_data
            
            # Validate pattern data
            patterns = library_data.get('patterns', {})
            total_patterns = library_data.get('total_patterns', 0)
            statistics = library_data.get('statistics', {})
            
            logger.info(f"   📊 Total patterns: {total_patterns}")
            logger.info(f"   📊 Pattern categories: {library_data.get('pattern_categories', [])}")
            logger.info(f"   📊 Statistics: {statistics}")
            
            # Check for key chartist patterns
            key_patterns_found = 0
            for pattern in self.test_patterns:
                if pattern in patterns:
                    pattern_data = patterns[pattern]
                    success_rate = pattern_data.get('success_rate', 0)
                    avg_return = pattern_data.get('average_return', 0)
                    sample_size = pattern_data.get('sample_size', 0)
                    
                    logger.info(f"   🎯 {pattern}: Success rate {success_rate}%, Avg return {avg_return}%, Samples: {sample_size}")
                    key_patterns_found += 1
                else:
                    logger.info(f"   ⚠️ Missing pattern: {pattern}")
            
            # Validate that library has meaningful data
            has_meaningful_data = (total_patterns >= 10 and 
                                 key_patterns_found >= 4 and
                                 len(statistics) > 0)
            
            success = has_meaningful_data
            details = f"Total patterns: {total_patterns}, Key patterns found: {key_patterns_found}/{len(self.test_patterns)}, Has statistics: {len(statistics) > 0}"
            
            self.log_test_result("Chartist Library Endpoint", success, details)
            
        except Exception as e:
            self.log_test_result("Chartist Library Endpoint", False, f"Exception: {str(e)}")
    
    async def test_chartist_analyze_endpoint(self):
        """Test 2: /api/chartist/analyze - Should provide recommendations based on patterns and market context"""
        logger.info("\n🔍 TEST 2: Chartist Analyze Endpoint")
        
        try:
            # Test with different pattern combinations and market contexts
            test_cases = [
                {
                    "patterns": ["head_and_shoulders", "bullish_flag"],
                    "market_context": "BULL",
                    "symbol": "BTCUSDT"
                },
                {
                    "patterns": ["ascending_triangle", "cup_with_handle"],
                    "market_context": "BEAR", 
                    "symbol": "ETHUSDT"
                },
                {
                    "patterns": ["double_bottom", "falling_wedge"],
                    "market_context": "SIDEWAYS",
                    "symbol": "SOLUSDT"
                },
                {
                    "patterns": ["bullish_flag", "ascending_triangle"],
                    "market_context": "VOLATILE",
                    "symbol": "ADAUSDT"
                }
            ]
            
            successful_analyses = 0
            total_analyses = len(test_cases)
            analysis_results = []
            
            for i, test_case in enumerate(test_cases):
                logger.info(f"   🧪 Test case {i+1}: {test_case['patterns']} in {test_case['market_context']} market")
                
                try:
                    response = requests.post(
                        f"{self.api_url}/chartist/analyze",
                        json=test_case,
                        timeout=15
                    )
                    
                    if response.status_code != 200:
                        logger.info(f"      ❌ HTTP {response.status_code}: {response.text}")
                        continue
                    
                    data = response.json()
                    
                    if not data.get('success', False):
                        logger.info(f"      ❌ API error: {data.get('message', 'No message')}")
                        continue
                    
                    # Validate analysis data
                    analysis_data = data.get('data', {})
                    expected_fields = ['recommendations', 'position_sizing', 'risk_reward_optimization', 'market_adaptation']
                    
                    if all(field in analysis_data for field in expected_fields):
                        successful_analyses += 1
                        analysis_results.append(analysis_data)
                        
                        # Log key metrics
                        recommendations = analysis_data.get('recommendations', {})
                        position_sizing = analysis_data.get('position_sizing', {})
                        risk_reward = analysis_data.get('risk_reward_optimization', {})
                        
                        logger.info(f"      ✅ Success: {recommendations.get('signal', 'N/A')} signal")
                        logger.info(f"         Position size: {position_sizing.get('recommended_size', 'N/A')}%")
                        logger.info(f"         Risk/Reward: {risk_reward.get('ratio', 'N/A')}")
                    else:
                        missing = [f for f in expected_fields if f not in analysis_data]
                        logger.info(f"      ❌ Missing fields: {missing}")
                        
                except Exception as e:
                    logger.info(f"      ❌ Exception: {str(e)}")
            
            # Store for later tests
            if analysis_results:
                self.chartist_analysis = analysis_results[0]
            
            # Validate overall success
            success_rate = (successful_analyses / total_analyses) * 100
            success = success_rate >= 75  # At least 75% success rate
            
            details = f"Successful analyses: {successful_analyses}/{total_analyses} ({success_rate:.1f}%)"
            
            self.log_test_result("Chartist Analyze Endpoint", success, details)
            
        except Exception as e:
            self.log_test_result("Chartist Analyze Endpoint", False, f"Exception: {str(e)}")
    
    async def test_ai_training_load_insights_chartist_integration(self):
        """Test 3: /api/ai-training/load-insights - Should now integrate chartist strategies"""
        logger.info("\n🔍 TEST 3: AI Training Load Insights with Chartist Integration")
        
        try:
            response = requests.post(f"{self.api_url}/ai-training/load-insights", timeout=15)
            
            if response.status_code != 200:
                self.log_test_result("AI Training Load Insights Chartist Integration", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            
            # Validate response structure
            if not data.get('success', False):
                self.log_test_result("AI Training Load Insights Chartist Integration", False, f"API returned success=False: {data.get('message', 'No message')}")
                return
            
            # Validate enhancement summary with chartist integration
            enhancement_summary = data.get('data', {})
            
            logger.info(f"   📊 Enhancement summary: {enhancement_summary}")
            
            # Check for chartist-specific enhancements
            total_rules = enhancement_summary.get('total_rules', 0)
            pattern_insights = enhancement_summary.get('pattern_insights', 0)
            chartist_insights = enhancement_summary.get('chartist_insights', 0)
            market_condition_insights = enhancement_summary.get('market_condition_insights', 0)
            
            logger.info(f"   📊 Total enhancement rules: {total_rules}")
            logger.info(f"   📊 Pattern insights: {pattern_insights}")
            logger.info(f"   📊 Chartist insights: {chartist_insights}")
            logger.info(f"   📊 Market condition insights: {market_condition_insights}")
            
            # Store for later tests
            self.ai_training_insights = enhancement_summary
            
            # Validate that chartist insights were loaded
            chartist_integration = (total_rules > 0 and 
                                  pattern_insights > 0 and 
                                  chartist_insights > 0 and
                                  market_condition_insights > 0)
            
            success = chartist_integration
            details = f"Total rules: {total_rules}, Pattern insights: {pattern_insights}, Chartist insights: {chartist_insights}, Market insights: {market_condition_insights}"
            
            self.log_test_result("AI Training Load Insights Chartist Integration", success, details)
            
        except Exception as e:
            self.log_test_result("AI Training Load Insights Chartist Integration", False, f"Exception: {str(e)}")
    
    async def test_chartist_improvements_in_ia1_ia2(self):
        """Test 4: Verify chartist improvements are applied to IA1 and IA2 decisions"""
        logger.info("\n🔍 TEST 4: Chartist Improvements in IA1 and IA2 Decisions")
        
        try:
            # Get recent trading decisions
            decisions, error = self.get_decisions_from_api()
            
            if error:
                self.log_test_result("Chartist Improvements in IA1 IA2", False, f"Failed to get decisions: {error}")
                return
            
            if not decisions:
                self.log_test_result("Chartist Improvements in IA1 IA2", False, "No trading decisions found")
                return
            
            # Analyze decisions for chartist enhancements
            chartist_analysis = {
                'total_decisions': len(decisions),
                'ia1_chartist_enhanced': 0,
                'ia2_chartist_enhanced': 0,
                'confidence_boosts': 0,
                'position_size_adjustments': 0,
                'pattern_based_decisions': 0,
                'success_rate_adjustments': 0
            }
            
            chartist_examples = []
            
            for decision in decisions:
                symbol = decision.get('symbol', 'UNKNOWN')
                ia1_reasoning = decision.get('ia1_analysis', '')
                ia2_reasoning = decision.get('ia2_reasoning', '')
                confidence = decision.get('confidence', 0)
                position_size = decision.get('position_size', 0)
                
                # Check for chartist keywords in IA1 analysis
                chartist_ia1_keywords = ['chartist', 'pattern', 'cup with handle', 'head and shoulders', 'triangle', 'flag', 'wedge', 'double bottom', 'double top']
                if any(keyword in ia1_reasoning.lower() for keyword in chartist_ia1_keywords):
                    chartist_analysis['ia1_chartist_enhanced'] += 1
                
                # Check for chartist keywords in IA2 reasoning
                chartist_ia2_keywords = ['success rate', 'pattern-based', 'chartist', 'position sizing based on', 'risk-reward optimization', 'pattern confidence']
                if any(keyword in ia2_reasoning.lower() for keyword in chartist_ia2_keywords):
                    chartist_analysis['ia2_chartist_enhanced'] += 1
                
                # Check for confidence boosts (high confidence decisions)
                if confidence > 0.8:
                    chartist_analysis['confidence_boosts'] += 1
                
                # Check for position size adjustments (non-standard sizes)
                if position_size > 0 and position_size != 0.02:  # Not default 2%
                    chartist_analysis['position_size_adjustments'] += 1
                
                # Check for pattern-based decisions
                pattern_keywords = ['pattern', 'formation', 'breakout', 'support', 'resistance']
                if any(keyword in (ia1_reasoning + ia2_reasoning).lower() for keyword in pattern_keywords):
                    chartist_analysis['pattern_based_decisions'] += 1
                
                # Check for success rate mentions
                if 'success rate' in (ia1_reasoning + ia2_reasoning).lower() or '% success' in (ia1_reasoning + ia2_reasoning).lower():
                    chartist_analysis['success_rate_adjustments'] += 1
                
                # Collect examples
                if (chartist_analysis['ia1_chartist_enhanced'] > len(chartist_examples) or 
                    chartist_analysis['ia2_chartist_enhanced'] > len(chartist_examples)):
                    chartist_examples.append({
                        'symbol': symbol,
                        'confidence': confidence,
                        'position_size': position_size,
                        'has_ia1_chartist': any(keyword in ia1_reasoning.lower() for keyword in chartist_ia1_keywords),
                        'has_ia2_chartist': any(keyword in ia2_reasoning.lower() for keyword in chartist_ia2_keywords)
                    })
            
            logger.info(f"   📊 Chartist Analysis:")
            logger.info(f"      Total decisions: {chartist_analysis['total_decisions']}")
            logger.info(f"      IA1 chartist enhanced: {chartist_analysis['ia1_chartist_enhanced']}")
            logger.info(f"      IA2 chartist enhanced: {chartist_analysis['ia2_chartist_enhanced']}")
            logger.info(f"      Confidence boosts: {chartist_analysis['confidence_boosts']}")
            logger.info(f"      Position size adjustments: {chartist_analysis['position_size_adjustments']}")
            logger.info(f"      Pattern-based decisions: {chartist_analysis['pattern_based_decisions']}")
            logger.info(f"      Success rate adjustments: {chartist_analysis['success_rate_adjustments']}")
            
            # Show examples
            if chartist_examples:
                logger.info(f"   📝 Chartist Enhanced Examples:")
                for example in chartist_examples[:3]:
                    logger.info(f"      {example['symbol']}: Confidence {example['confidence']:.2f}, Position {example['position_size']:.3f}%, IA1 chartist: {example['has_ia1_chartist']}, IA2 chartist: {example['has_ia2_chartist']}")
            
            # Validate that chartist improvements are being applied
            chartist_integration_rate = 0
            if chartist_analysis['total_decisions'] > 0:
                chartist_integration_rate = ((chartist_analysis['ia1_chartist_enhanced'] + chartist_analysis['ia2_chartist_enhanced']) / (chartist_analysis['total_decisions'] * 2)) * 100
            
            # Check if chartist system is working
            chartist_working = (chartist_analysis['ia1_chartist_enhanced'] > 0 or 
                              chartist_analysis['ia2_chartist_enhanced'] > 0 or
                              chartist_analysis['pattern_based_decisions'] > 0 or
                              chartist_analysis['success_rate_adjustments'] > 0)
            
            success = chartist_working and chartist_integration_rate > 10  # At least 10% integration rate
            details = f"IA1 enhanced: {chartist_analysis['ia1_chartist_enhanced']}, IA2 enhanced: {chartist_analysis['ia2_chartist_enhanced']}, Integration rate: {chartist_integration_rate:.1f}%"
            
            self.log_test_result("Chartist Improvements in IA1 IA2", success, details)
            
        except Exception as e:
            self.log_test_result("Chartist Improvements in IA1 IA2", False, f"Exception: {str(e)}")
    
    async def test_market_context_adaptation(self):
        """Test 5: Test multiple market contexts (BULL, BEAR, SIDEWAYS, VOLATILE)"""
        logger.info("\n🔍 TEST 5: Market Context Adaptation")
        
        try:
            context_results = {}
            
            for market_context in self.market_contexts:
                logger.info(f"   🧪 Testing {market_context} market context")
                
                try:
                    # Test chartist analysis in different market contexts
                    test_payload = {
                        "patterns": ["bullish_flag", "ascending_triangle"],
                        "market_context": market_context,
                        "symbol": "BTCUSDT"
                    }
                    
                    response = requests.post(
                        f"{self.api_url}/chartist/analyze",
                        json=test_payload,
                        timeout=15
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('success', False):
                            analysis_data = data.get('data', {})
                            
                            # Extract key metrics for this market context
                            recommendations = analysis_data.get('recommendations', {})
                            position_sizing = analysis_data.get('position_sizing', {})
                            risk_reward = analysis_data.get('risk_reward_optimization', {})
                            market_adaptation = analysis_data.get('market_adaptation', {})
                            
                            context_results[market_context] = {
                                'success': True,
                                'signal': recommendations.get('signal', 'HOLD'),
                                'confidence': recommendations.get('confidence', 0),
                                'position_size': position_sizing.get('recommended_size', 0),
                                'risk_reward_ratio': risk_reward.get('ratio', 0),
                                'adaptation_strategy': market_adaptation.get('strategy', 'none')
                            }
                            
                            logger.info(f"      ✅ {market_context}: {context_results[market_context]['signal']} signal, {context_results[market_context]['confidence']:.2f} confidence")
                        else:
                            context_results[market_context] = {'success': False, 'error': 'API returned success=False'}
                            logger.info(f"      ❌ {market_context}: API error")
                    else:
                        context_results[market_context] = {'success': False, 'error': f'HTTP {response.status_code}'}
                        logger.info(f"      ❌ {market_context}: HTTP {response.status_code}")
                        
                except Exception as e:
                    context_results[market_context] = {'success': False, 'error': str(e)}
                    logger.info(f"      ❌ {market_context}: Exception {str(e)}")
            
            # Analyze results across market contexts
            successful_contexts = sum(1 for result in context_results.values() if result['success'])
            total_contexts = len(self.market_contexts)
            
            # Check for context-specific adaptations
            adaptations_detected = 0
            unique_signals = set()
            unique_position_sizes = set()
            
            for context, result in context_results.items():
                if result['success']:
                    unique_signals.add(result['signal'])
                    unique_position_sizes.add(result['position_size'])
                    
                    # Check if adaptation strategy is context-specific
                    if result.get('adaptation_strategy', 'none') != 'none':
                        adaptations_detected += 1
            
            logger.info(f"   📊 Market Context Results:")
            logger.info(f"      Successful contexts: {successful_contexts}/{total_contexts}")
            logger.info(f"      Unique signals: {list(unique_signals)}")
            logger.info(f"      Unique position sizes: {list(unique_position_sizes)}")
            logger.info(f"      Adaptations detected: {adaptations_detected}")
            
            # Validate market context adaptation
            context_adaptation_working = (successful_contexts >= 3 and  # At least 3 contexts work
                                        len(unique_signals) > 1 and    # Different signals for different contexts
                                        adaptations_detected > 0)       # Some adaptations detected
            
            success = context_adaptation_working
            details = f"Successful contexts: {successful_contexts}/{total_contexts}, Unique signals: {len(unique_signals)}, Adaptations: {adaptations_detected}"
            
            self.log_test_result("Market Context Adaptation", success, details)
            
        except Exception as e:
            self.log_test_result("Market Context Adaptation", False, f"Exception: {str(e)}")
    
    async def test_position_sizing_optimization(self):
        """Test 6: Verify position sizing optimization based on pattern success rates"""
        logger.info("\n🔍 TEST 6: Position Sizing Optimization Based on Pattern Success Rates")
        
        try:
            # Test different patterns with known success rates
            test_patterns = [
                {"pattern": "cup_with_handle", "expected_success_rate": 81},  # High success rate
                {"pattern": "head_and_shoulders", "expected_success_rate": 65},  # Medium success rate
                {"pattern": "double_top", "expected_success_rate": 45}  # Lower success rate
            ]
            
            position_sizing_results = []
            
            for pattern_test in test_patterns:
                pattern = pattern_test["pattern"]
                expected_rate = pattern_test["expected_success_rate"]
                
                logger.info(f"   🧪 Testing position sizing for {pattern} (expected {expected_rate}% success)")
                
                try:
                    test_payload = {
                        "patterns": [pattern],
                        "market_context": "BULL",
                        "symbol": "BTCUSDT"
                    }
                    
                    response = requests.post(
                        f"{self.api_url}/chartist/analyze",
                        json=test_payload,
                        timeout=15
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('success', False):
                            analysis_data = data.get('data', {})
                            position_sizing = analysis_data.get('position_sizing', {})
                            
                            recommended_size = position_sizing.get('recommended_size', 0)
                            success_rate_factor = position_sizing.get('success_rate_factor', 0)
                            pattern_confidence = position_sizing.get('pattern_confidence', 0)
                            
                            position_sizing_results.append({
                                'pattern': pattern,
                                'expected_success_rate': expected_rate,
                                'recommended_size': recommended_size,
                                'success_rate_factor': success_rate_factor,
                                'pattern_confidence': pattern_confidence
                            })
                            
                            logger.info(f"      ✅ {pattern}: {recommended_size}% position size (factor: {success_rate_factor:.2f})")
                        else:
                            logger.info(f"      ❌ {pattern}: API error")
                    else:
                        logger.info(f"      ❌ {pattern}: HTTP {response.status_code}")
                        
                except Exception as e:
                    logger.info(f"      ❌ {pattern}: Exception {str(e)}")
            
            # Analyze position sizing optimization
            if len(position_sizing_results) >= 2:
                # Sort by expected success rate
                sorted_results = sorted(position_sizing_results, key=lambda x: x['expected_success_rate'], reverse=True)
                
                # Check if higher success rate patterns get larger position sizes
                position_size_correlation = True
                for i in range(len(sorted_results) - 1):
                    current = sorted_results[i]
                    next_pattern = sorted_results[i + 1]
                    
                    if current['recommended_size'] <= next_pattern['recommended_size']:
                        position_size_correlation = False
                        logger.info(f"      ⚠️ Position size correlation issue: {current['pattern']} ({current['expected_success_rate']}%) has {current['recommended_size']}% vs {next_pattern['pattern']} ({next_pattern['expected_success_rate']}%) has {next_pattern['recommended_size']}%")
                
                # Check for meaningful position size differences
                max_size = max(result['recommended_size'] for result in position_sizing_results)
                min_size = min(result['recommended_size'] for result in position_sizing_results)
                size_range = max_size - min_size
                
                meaningful_differences = size_range > 0.5  # At least 0.5% difference
                
                logger.info(f"   📊 Position Sizing Analysis:")
                logger.info(f"      Position size correlation: {position_size_correlation}")
                logger.info(f"      Size range: {size_range:.2f}% ({min_size:.2f}% - {max_size:.2f}%)")
                logger.info(f"      Meaningful differences: {meaningful_differences}")
                
                # Validate optimization
                optimization_working = (position_size_correlation and meaningful_differences)
                
                success = optimization_working
                details = f"Correlation: {position_size_correlation}, Size range: {size_range:.2f}%, Patterns tested: {len(position_sizing_results)}"
                
            else:
                success = False
                details = f"Insufficient data: only {len(position_sizing_results)} patterns tested successfully"
            
            self.log_test_result("Position Sizing Optimization", success, details)
            
        except Exception as e:
            self.log_test_result("Position Sizing Optimization", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_tests(self):
        """Run all Chartist Learning System tests"""
        logger.info("🚀 Starting Chartist Learning System Test Suite")
        logger.info("=" * 80)
        
        await self.setup_database()
        
        # Run all tests in sequence
        await self.test_chartist_library_endpoint()
        await self.test_chartist_analyze_endpoint()
        await self.test_ai_training_load_insights_chartist_integration()
        await self.test_chartist_improvements_in_ia1_ia2()
        await self.test_market_context_adaptation()
        await self.test_position_sizing_optimization()
        
        await self.cleanup_database()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 CHARTIST LEARNING SYSTEM TEST SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # Specific analysis for French review request
        logger.info("\n" + "=" * 80)
        logger.info("📋 ANALYSE POUR LA DEMANDE DE RÉVISION (FRANÇAIS)")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 TOUS LES TESTS RÉUSSIS - Le système d'apprentissage chartiste fonctionne correctement!")
            logger.info("✅ Bibliothèque des figures chartistes accessible avec statistiques")
            logger.info("✅ Analyse chartiste avec recommandations basées sur les patterns")
            logger.info("✅ Intégration des stratégies chartistes dans l'entraînement IA")
            logger.info("✅ Améliorations chartistes appliquées aux décisions IA1 et IA2")
            logger.info("✅ Adaptation selon le contexte de marché (BULL, BEAR, SIDEWAYS, VOLATILE)")
            logger.info("✅ Optimisation des tailles de position selon les taux de succès des patterns")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ FONCTIONNEMENT PARTIEL - Quelques problèmes détectés dans le système chartiste")
            logger.info("🔍 Vérifiez les tests échoués pour des problèmes spécifiques")
        else:
            logger.info("❌ PROBLÈMES CRITIQUES - Le système d'apprentissage chartiste nécessite une attention")
            logger.info("🚨 Problèmes majeurs avec l'intégration chartiste")
            
        # Key features verification
        logger.info("\n📝 VÉRIFICATION DES FONCTIONNALITÉS CLÉS:")
        
        # Check specific features mentioned in French review request
        features_verified = []
        features_failed = []
        
        # Check chartist library
        library_working = any("Chartist Library Endpoint" in result['test'] and result['success'] for result in self.test_results)
        if library_working:
            features_verified.append("✅ Bibliothèque complète des figures chartistes avec statistiques")
        else:
            features_failed.append("❌ Bibliothèque des figures chartistes non accessible")
        
        # Check chartist analysis
        analysis_working = any("Chartist Analyze Endpoint" in result['test'] and result['success'] for result in self.test_results)
        if analysis_working:
            features_verified.append("✅ Analyse chartiste avec recommandations patterns")
        else:
            features_failed.append("❌ Analyse chartiste non fonctionnelle")
        
        # Check AI integration
        ai_integration = any("AI Training Load Insights Chartist Integration" in result['test'] and result['success'] for result in self.test_results)
        if ai_integration:
            features_verified.append("✅ Intégration des stratégies chartistes dans l'IA")
        else:
            features_failed.append("❌ Intégration IA des stratégies chartistes échouée")
        
        # Check IA1/IA2 improvements
        ia_improvements = any("Chartist Improvements in IA1 IA2" in result['test'] and result['success'] for result in self.test_results)
        if ia_improvements:
            features_verified.append("✅ Améliorations chartistes appliquées à IA1 et IA2")
        else:
            features_failed.append("❌ Améliorations chartistes IA1/IA2 non appliquées")
        
        # Check market context adaptation
        market_adaptation = any("Market Context Adaptation" in result['test'] and result['success'] for result in self.test_results)
        if market_adaptation:
            features_verified.append("✅ Adaptation selon contexte marché (BULL/BEAR/SIDEWAYS/VOLATILE)")
        else:
            features_failed.append("❌ Adaptation contexte marché non fonctionnelle")
        
        # Check position sizing optimization
        position_optimization = any("Position Sizing Optimization" in result['test'] and result['success'] for result in self.test_results)
        if position_optimization:
            features_verified.append("✅ Optimisation tailles position selon taux succès patterns")
        else:
            features_failed.append("❌ Optimisation tailles position non fonctionnelle")
        
        for feature in features_verified:
            logger.info(f"   {feature}")
        
        for failure in features_failed:
            logger.info(f"   {failure}")
            
        # Expected outcomes verification
        logger.info("\n🎯 VÉRIFICATION DES RÉSULTATS ATTENDUS:")
        
        expected_outcomes = [
            "Stratégies optimisées long/short basées sur figures chartistes",
            "Ajustement automatique tailles position selon taux succès (ex: Tasse avec Anse = 81%)",
            "Optimisation ratios risque/récompense selon figures détectées", 
            "Adaptation stratégies selon contexte marché",
            "Intégration recommandations chartistes dans IA1 (boost confiance) et IA2 (optimisation position/R:R)"
        ]
        
        for outcome in expected_outcomes:
            logger.info(f"   📋 {outcome}")
            
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = ChartistLearningSystemTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
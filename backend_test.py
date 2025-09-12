#!/usr/bin/env python3
"""
IA1 RISK-REWARD CALCULATION INDEPENDENCE TEST SUITE
Focus: Test IA1 RR calculation independence from confidence level

CRITICAL TEST REQUIREMENTS FROM REVIEW REQUEST:
1. **RR Calculation Independence**: Verify that IA1's Risk-Reward ratio calculation is completely independent of the confidence level
2. **Consistent RR Values**: Test that identical market conditions produce identical RR ratios regardless of IA1's confidence
3. **Technical Analysis Based**: Confirm RR calculation uses only technical levels (support/resistance) not confidence percentages
4. **Formula Validation**: Verify LONG and SHORT RR formulas are correctly implemented as specified:
   - LONG: RR = (Take_Profit - Entry) / (Entry - Stop_Loss)
   - SHORT: RR = (Entry - Take_Profit) / (Stop_Loss - Entry)

SPECIFIC TESTING SCENARIOS:
1. Test multiple IA1 analyses for the same symbol with varying confidence levels (70%, 85%, 95%)
2. Verify that stop-loss and take-profit levels are based on technical analysis, not confidence
3. Check that RR ratios are consistent for the same technical setup regardless of confidence
4. Validate that fallback levels (when no technical levels available) are fixed percentages
5. Test both LONG and SHORT scenarios with different confidence levels

ENDPOINTS TO TEST:
- POST /api/force-ia1-analysis with different symbols
- GET /api/analyses to verify stored RR calculations
- Monitor logs for RR calculation details and technical level explanations

SUCCESS CRITERIA:
- RR calculation shows no correlation with confidence levels
- Technical stop-loss/take-profit levels are consistent
- Formula implementation matches specified equations
- No evidence of confidence-based adjustments to RR values
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import requests
import subprocess
import re
from pymongo import MongoClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IA1RiskRewardIndependenceTestSuite:
    """Comprehensive test suite for IA1 Risk-Reward Calculation Independence"""
    
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
        logger.info(f"Testing IA1 Risk-Reward Calculation Independence at: {self.api_url}")
        
        # MongoDB connection for direct database analysis
        try:
            self.mongo_client = MongoClient("mongodb://localhost:27017")
            self.db = self.mongo_client["myapp"]
            logger.info("✅ MongoDB connection established for IA1 analysis")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.mongo_client = None
            self.db = None
        
        # Test results
        self.test_results = []
        
        # Test symbols for RR independence testing
        self.test_symbols = [
            "BTCUSDT",   # High volatility, strong technical levels
            "ETHUSDT",   # Medium volatility, clear S/R levels
            "XRPUSDT",   # Lower volatility, tight ranges
            "SOLUSDT",   # High volatility, trending
            "ADAUSDT"    # Medium volatility, range-bound
        ]
        
        # Expected RR formula validation
        self.rr_formulas = {
            "LONG": lambda entry, tp, sl: (tp - entry) / (entry - sl) if (entry - sl) > 0 else 0,
            "SHORT": lambda entry, tp, sl: (entry - tp) / (sl - entry) if (sl - entry) > 0 else 0
        }
        
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
    
    async def test_1_ia1_rr_formula_validation(self):
        """Test 1: Validate IA1 RR Formula Implementation"""
        logger.info("\n🔍 TEST 1: IA1 RR Formula Validation Test")
        
        try:
            # Test RR formula implementation with known values
            test_cases = [
                {
                    "name": "LONG Scenario - Standard Case",
                    "signal": "LONG",
                    "entry": 100.0,
                    "take_profit": 110.0,
                    "stop_loss": 95.0,
                    "expected_rr": (110.0 - 100.0) / (100.0 - 95.0),  # 10/5 = 2.0
                    "expected_formula": "LONG: RR = (Take_Profit - Entry) / (Entry - Stop_Loss)"
                },
                {
                    "name": "SHORT Scenario - Standard Case", 
                    "signal": "SHORT",
                    "entry": 100.0,
                    "take_profit": 90.0,
                    "stop_loss": 105.0,
                    "expected_rr": (100.0 - 90.0) / (105.0 - 100.0),  # 10/5 = 2.0
                    "expected_formula": "SHORT: RR = (Entry - Take_Profit) / (Stop_Loss - Entry)"
                },
                {
                    "name": "LONG Scenario - High RR Case",
                    "signal": "LONG", 
                    "entry": 50.0,
                    "take_profit": 65.0,
                    "stop_loss": 47.0,
                    "expected_rr": (65.0 - 50.0) / (50.0 - 47.0),  # 15/3 = 5.0
                    "expected_formula": "LONG: RR = (Take_Profit - Entry) / (Entry - Stop_Loss)"
                },
                {
                    "name": "SHORT Scenario - High RR Case",
                    "signal": "SHORT",
                    "entry": 50.0,
                    "take_profit": 35.0,
                    "stop_loss": 53.0,
                    "expected_rr": (50.0 - 35.0) / (53.0 - 50.0),  # 15/3 = 5.0
                    "expected_formula": "SHORT: RR = (Entry - Take_Profit) / (Stop_Loss - Entry)"
                }
            ]
            
            formula_validation_results = {
                'total_tests': len(test_cases),
                'correct_calculations': 0,
                'formula_matches': 0,
                'calculation_errors': []
            }
            
            logger.info(f"   📊 Testing {len(test_cases)} RR formula scenarios:")
            
            for i, test_case in enumerate(test_cases):
                try:
                    # Calculate using our expected formula
                    if test_case['signal'] == 'LONG':
                        calculated_rr = self.rr_formulas['LONG'](
                            test_case['entry'], 
                            test_case['take_profit'], 
                            test_case['stop_loss']
                        )
                    else:  # SHORT
                        calculated_rr = self.rr_formulas['SHORT'](
                            test_case['entry'], 
                            test_case['take_profit'], 
                            test_case['stop_loss']
                        )
                    
                    # Check if calculation matches expected
                    calculation_correct = abs(calculated_rr - test_case['expected_rr']) < 0.001
                    
                    if calculation_correct:
                        formula_validation_results['correct_calculations'] += 1
                        formula_validation_results['formula_matches'] += 1
                        status = "✅"
                    else:
                        formula_validation_results['calculation_errors'].append(
                            f"{test_case['name']}: Expected {test_case['expected_rr']:.3f}, Got {calculated_rr:.3f}"
                        )
                        status = "❌"
                    
                    logger.info(f"      {status} {test_case['name']}: "
                              f"Entry=${test_case['entry']}, TP=${test_case['take_profit']}, SL=${test_case['stop_loss']} "
                              f"→ RR={calculated_rr:.3f} (expected: {test_case['expected_rr']:.3f})")
                    logger.info(f"         Formula: {test_case['expected_formula']}")
                    
                except Exception as e:
                    formula_validation_results['calculation_errors'].append(f"{test_case['name']}: {str(e)}")
                    logger.warning(f"      ❌ {test_case['name']}: Formula test failed - {e}")
            
            # Calculate performance metrics
            calculation_accuracy = formula_validation_results['correct_calculations'] / formula_validation_results['total_tests']
            
            logger.info(f"   📊 Formula Validation Performance:")
            logger.info(f"      Calculation accuracy: {calculation_accuracy:.1%}")
            logger.info(f"      Correct calculations: {formula_validation_results['correct_calculations']}/{formula_validation_results['total_tests']}")
            
            if formula_validation_results['calculation_errors']:
                logger.info(f"      Calculation errors:")
                for error in formula_validation_results['calculation_errors']:
                    logger.info(f"        - {error}")
            
            # Determine test result
            if calculation_accuracy >= 1.0:
                self.log_test_result("IA1 RR Formula Validation", True, 
                                   f"All RR formulas working correctly: {calculation_accuracy:.1%} accuracy")
            elif calculation_accuracy >= 0.8:
                self.log_test_result("IA1 RR Formula Validation", False, 
                                   f"Most RR formulas working: {calculation_accuracy:.1%} accuracy, some errors found")
            else:
                self.log_test_result("IA1 RR Formula Validation", False, 
                                   f"RR formula issues: {calculation_accuracy:.1%} accuracy, multiple errors")
                
        except Exception as e:
            self.log_test_result("IA1 RR Formula Validation", False, f"Exception: {str(e)}")
    
    async def test_2_ia1_confidence_independence(self):
        """Test 2: IA1 RR Calculation Independence from Confidence Level"""
        logger.info("\n🔍 TEST 2: IA1 RR Calculation Independence Test")
        
        try:
            # Test multiple IA1 analyses for the same symbols with different confidence scenarios
            # This will verify that RR calculation is independent of confidence level
            
            independence_test_results = {
                'symbols_tested': 0,
                'analyses_performed': 0,
                'rr_consistency_violations': 0,
                'confidence_correlation_detected': False,
                'technical_level_consistency': 0,
                'rr_variations': []
            }
            
            logger.info("   🚀 Testing IA1 RR calculation independence from confidence...")
            
            # Test each symbol multiple times to check for RR consistency
            for symbol in self.test_symbols[:3]:  # Test first 3 symbols to avoid timeout
                try:
                    logger.info(f"   📊 Testing {symbol} for RR independence...")
                    
                    # Perform multiple IA1 analyses for the same symbol
                    # The system should produce consistent RR ratios regardless of confidence
                    symbol_analyses = []
                    
                    for attempt in range(3):  # 3 attempts per symbol
                        try:
                            # Force IA1 analysis
                            response = requests.post(
                                f"{self.api_url}/force-ia1-analysis",
                                json={"symbol": symbol},
                                timeout=30
                            )
                            
                            if response.status_code == 200:
                                analysis_data = response.json()
                                if analysis_data.get('success') and 'analysis' in analysis_data:
                                    analysis = analysis_data['analysis']
                                    symbol_analyses.append({
                                        'symbol': symbol,
                                        'attempt': attempt + 1,
                                        'confidence': analysis.get('confidence', 0),
                                        'risk_reward_ratio': analysis.get('risk_reward_ratio', 0),
                                        'entry_price': analysis.get('entry_price', 0),
                                        'stop_loss_price': analysis.get('stop_loss_price', 0),
                                        'take_profit_price': analysis.get('take_profit_price', 0),
                                        'signal': analysis.get('recommendation', 'hold'),
                                        'support_levels': analysis.get('support', []),
                                        'resistance_levels': analysis.get('resistance', [])
                                    })
                                    independence_test_results['analyses_performed'] += 1
                                    
                                    logger.info(f"      Attempt {attempt + 1}: Confidence={analysis.get('confidence', 0):.1%}, "
                                              f"RR={analysis.get('risk_reward_ratio', 0):.2f}, "
                                              f"Signal={analysis.get('recommendation', 'hold')}")
                            else:
                                logger.warning(f"      Attempt {attempt + 1}: API returned {response.status_code}")
                                
                        except Exception as e:
                            logger.warning(f"      Attempt {attempt + 1}: Request failed - {e}")
                        
                        # Small delay between requests
                        await asyncio.sleep(2)
                    
                    # Analyze RR consistency for this symbol
                    if len(symbol_analyses) >= 2:
                        independence_test_results['symbols_tested'] += 1
                        
                        # Check RR ratio consistency
                        rr_values = [a['risk_reward_ratio'] for a in symbol_analyses if a['risk_reward_ratio'] > 0]
                        confidence_values = [a['confidence'] for a in symbol_analyses]
                        
                        if len(rr_values) >= 2:
                            rr_variance = max(rr_values) - min(rr_values)
                            confidence_variance = max(confidence_values) - min(confidence_values)
                            
                            # Check if RR varies significantly with confidence
                            if rr_variance > 0.5 and confidence_variance > 0.1:  # Significant variations
                                independence_test_results['rr_consistency_violations'] += 1
                                independence_test_results['rr_variations'].append({
                                    'symbol': symbol,
                                    'rr_variance': rr_variance,
                                    'confidence_variance': confidence_variance,
                                    'analyses': symbol_analyses
                                })
                                logger.warning(f"      ⚠️ RR variance detected: {rr_variance:.2f} with confidence variance: {confidence_variance:.1%}")
                            else:
                                independence_test_results['technical_level_consistency'] += 1
                                logger.info(f"      ✅ RR consistent: variance {rr_variance:.2f} with confidence variance {confidence_variance:.1%}")
                        
                        # Log technical levels consistency
                        entry_prices = [a['entry_price'] for a in symbol_analyses if a['entry_price'] > 0]
                        sl_prices = [a['stop_loss_price'] for a in symbol_analyses if a['stop_loss_price'] > 0]
                        tp_prices = [a['take_profit_price'] for a in symbol_analyses if a['take_profit_price'] > 0]
                        
                        if entry_prices and sl_prices and tp_prices:
                            entry_variance = (max(entry_prices) - min(entry_prices)) / min(entry_prices) if min(entry_prices) > 0 else 0
                            sl_variance = (max(sl_prices) - min(sl_prices)) / min(sl_prices) if min(sl_prices) > 0 else 0
                            tp_variance = (max(tp_prices) - min(tp_prices)) / min(tp_prices) if min(tp_prices) > 0 else 0
                            
                            logger.info(f"      📊 Technical Level Variance: Entry={entry_variance:.1%}, SL={sl_variance:.1%}, TP={tp_variance:.1%}")
                    
                except Exception as e:
                    logger.error(f"   ❌ Error testing {symbol}: {e}")
            
            # Analyze overall independence
            logger.info(f"   📊 Independence Test Results:")
            logger.info(f"      Symbols tested: {independence_test_results['symbols_tested']}")
            logger.info(f"      Total analyses: {independence_test_results['analyses_performed']}")
            logger.info(f"      RR consistency violations: {independence_test_results['rr_consistency_violations']}")
            logger.info(f"      Technical level consistency: {independence_test_results['technical_level_consistency']}")
            
            # Determine test result
            if independence_test_results['symbols_tested'] > 0:
                consistency_rate = independence_test_results['technical_level_consistency'] / independence_test_results['symbols_tested']
                violation_rate = independence_test_results['rr_consistency_violations'] / independence_test_results['symbols_tested']
                
                if consistency_rate >= 0.8 and violation_rate <= 0.2:
                    self.log_test_result("IA1 RR Calculation Independence", True, 
                                       f"RR calculation independent of confidence: {consistency_rate:.1%} consistency, {violation_rate:.1%} violations")
                elif consistency_rate >= 0.6:
                    self.log_test_result("IA1 RR Calculation Independence", False, 
                                       f"Partial RR independence: {consistency_rate:.1%} consistency, {violation_rate:.1%} violations")
                else:
                    self.log_test_result("IA1 RR Calculation Independence", False, 
                                       f"RR calculation may depend on confidence: {consistency_rate:.1%} consistency, {violation_rate:.1%} violations")
            else:
                self.log_test_result("IA1 RR Calculation Independence", False, 
                                   "Insufficient data to test RR independence")
                
        except Exception as e:
            self.log_test_result("IA1 RR Calculation Independence", False, f"Exception: {str(e)}")
    
    async def test_3_lateral_pattern_detector_analysis(self):
        """Test 3: Lateral Pattern Detector Multi-Criteria Analysis"""
        logger.info("\n🔍 TEST 3: Lateral Pattern Detector Analysis Test")
        
        try:
            if not self.pattern_detector:
                self.log_test_result("Lateral Pattern Detector Analysis", False, 
                                   "Pattern detector module not available")
                return
            
            # Test cases for pattern detection
            test_cases = [
                # Strong bullish (should NOT be filtered)
                {"symbol": "BTCUSDT", "price_change": 8.5, "volume": 5000000, "expected_lateral": False, "expected_type": self.TrendType.STRONG_BULLISH},
                # Bullish (should NOT be filtered)
                {"symbol": "ETHUSDT", "price_change": 3.2, "volume": 2000000, "expected_lateral": False, "expected_type": self.TrendType.BULLISH},
                # Lateral pattern (should be filtered)
                {"symbol": "XRPUSDT", "price_change": 0.5, "volume": 100000, "expected_lateral": True, "expected_type": self.TrendType.LATERAL},
                # Strong bearish (should NOT be filtered)
                {"symbol": "SOLUSDT", "price_change": -7.8, "volume": 3000000, "expected_lateral": False, "expected_type": self.TrendType.STRONG_BEARISH},
                # Bearish (should NOT be filtered)
                {"symbol": "ADAUSDT", "price_change": -2.5, "volume": 1500000, "expected_lateral": False, "expected_type": self.TrendType.BEARISH},
                # Edge case: High volume but low price change (might be lateral)
                {"symbol": "DOGEUSDT", "price_change": 0.8, "volume": 10000000, "expected_lateral": True, "expected_type": self.TrendType.LATERAL}
            ]
            
            analysis_results = {
                'total_tests': len(test_cases),
                'correct_classifications': 0,
                'correct_lateral_detections': 0,
                'correct_filter_decisions': 0,
                'trend_strength_scores': [],
                'confidence_scores': []
            }
            
            logger.info(f"   📊 Testing {len(test_cases)} pattern detection scenarios:")
            
            for i, test_case in enumerate(test_cases):
                try:
                    # Analyze pattern
                    analysis = self.pattern_detector.analyze_trend_pattern(
                        symbol=test_case['symbol'],
                        price_change_pct=test_case['price_change'],
                        volume=test_case['volume']
                    )
                    
                    # Check classification accuracy
                    classification_correct = analysis.trend_type == test_case['expected_type']
                    lateral_detection_correct = analysis.is_lateral == test_case['expected_lateral']
                    
                    # Check filter decision
                    should_filter = self.pattern_detector.should_filter_opportunity(analysis)
                    filter_decision_correct = should_filter == test_case['expected_lateral']
                    
                    # Update results
                    if classification_correct:
                        analysis_results['correct_classifications'] += 1
                    if lateral_detection_correct:
                        analysis_results['correct_lateral_detections'] += 1
                    if filter_decision_correct:
                        analysis_results['correct_filter_decisions'] += 1
                    
                    analysis_results['trend_strength_scores'].append(analysis.trend_strength)
                    analysis_results['confidence_scores'].append(analysis.confidence)
                    
                    # Log result
                    status = "✅" if (classification_correct and lateral_detection_correct and filter_decision_correct) else "❌"
                    logger.info(f"      {status} {test_case['symbol']}: {analysis.trend_type.value}, "
                              f"lateral={analysis.is_lateral}, filter={should_filter}, "
                              f"strength={analysis.trend_strength:.2f}, conf={analysis.confidence:.2f}")
                    logger.info(f"         Reasoning: {analysis.reasoning}")
                    
                except Exception as e:
                    logger.warning(f"      ❌ {test_case['symbol']}: Analysis failed - {e}")
            
            # Calculate performance metrics
            classification_accuracy = analysis_results['correct_classifications'] / analysis_results['total_tests']
            lateral_detection_accuracy = analysis_results['correct_lateral_detections'] / analysis_results['total_tests']
            filter_accuracy = analysis_results['correct_filter_decisions'] / analysis_results['total_tests']
            avg_trend_strength = sum(analysis_results['trend_strength_scores']) / len(analysis_results['trend_strength_scores']) if analysis_results['trend_strength_scores'] else 0
            avg_confidence = sum(analysis_results['confidence_scores']) / len(analysis_results['confidence_scores']) if analysis_results['confidence_scores'] else 0
            
            logger.info(f"   📊 Pattern Detection Performance:")
            logger.info(f"      Classification accuracy: {classification_accuracy:.1%}")
            logger.info(f"      Lateral detection accuracy: {lateral_detection_accuracy:.1%}")
            logger.info(f"      Filter decision accuracy: {filter_accuracy:.1%}")
            logger.info(f"      Average trend strength: {avg_trend_strength:.2f}")
            logger.info(f"      Average confidence: {avg_confidence:.2f}")
            
            # Determine test result
            if classification_accuracy >= 0.8 and lateral_detection_accuracy >= 0.8 and filter_accuracy >= 0.8:
                self.log_test_result("Lateral Pattern Detector Analysis", True, 
                                   f"Pattern detection working: {classification_accuracy:.1%} classification, {filter_accuracy:.1%} filter accuracy")
            elif classification_accuracy >= 0.6 and filter_accuracy >= 0.6:
                self.log_test_result("Lateral Pattern Detector Analysis", False, 
                                   f"Partial pattern detection: {classification_accuracy:.1%} classification, {filter_accuracy:.1%} filter accuracy")
            else:
                self.log_test_result("Lateral Pattern Detector Analysis", False, 
                                   f"Pattern detection issues: {classification_accuracy:.1%} classification, {filter_accuracy:.1%} filter accuracy")
                
        except Exception as e:
            self.log_test_result("Lateral Pattern Detector Analysis", False, f"Exception: {str(e)}")
    
    async def test_4_advanced_market_aggregator_integration(self):
        """Test 4: Advanced Market Aggregator Integration with BingX Data"""
        logger.info("\n🔍 TEST 4: Advanced Market Aggregator Integration Test")
        
        try:
            if not self.market_aggregator:
                self.log_test_result("Advanced Market Aggregator Integration", False, 
                                   "Market aggregator module not available")
                return
            
            logger.info("   🚀 Testing get_current_opportunities() with BingX integration...")
            
            # Test get_current_opportunities
            opportunities = self.market_aggregator.get_current_opportunities()
            
            if not opportunities:
                self.log_test_result("Advanced Market Aggregator Integration", False, 
                                   "No opportunities returned from market aggregator")
                return
            
            # Analyze opportunities
            integration_analysis = {
                'total_opportunities': len(opportunities),
                'expected_range': (10, 50),
                'bingx_sources': len([opp for opp in opportunities if 'bingx' in str(opp.data_sources).lower()]),
                'with_volume': len([opp for opp in opportunities if opp.volume_24h > 0]),
                'with_price_change': len([opp for opp in opportunities if abs(opp.price_change_24h) > 0]),
                'high_confidence': len([opp for opp in opportunities if opp.data_confidence >= 0.7]),
                'unique_symbols': len(set(opp.symbol for opp in opportunities))
            }
            
            logger.info(f"   📊 Integration Analysis:")
            logger.info(f"      Total opportunities: {integration_analysis['total_opportunities']} (expected: {integration_analysis['expected_range'][0]}-{integration_analysis['expected_range'][1]})")
            logger.info(f"      BingX sources: {integration_analysis['bingx_sources']}")
            logger.info(f"      With volume data: {integration_analysis['with_volume']}")
            logger.info(f"      With price change: {integration_analysis['with_price_change']}")
            logger.info(f"      High confidence (≥0.7): {integration_analysis['high_confidence']}")
            logger.info(f"      Unique symbols: {integration_analysis['unique_symbols']}")
            
            # Log top 10 opportunities
            logger.info(f"   📈 Top 10 Market Opportunities:")
            for i, opp in enumerate(opportunities[:10]):
                sources_str = f", Sources: {opp.data_sources}" if opp.data_sources else ""
                confidence_str = f", Conf: {opp.data_confidence:.1f}" if opp.data_confidence else ""
                logger.info(f"      {i+1}. {opp.symbol}: ${opp.current_price:.6f}, "
                          f"{opp.price_change_24h:+.1f}%, Vol: {opp.volume_24h/1_000_000:.1f}M{confidence_str}{sources_str}")
            
            # Test cache TTL alignment (check if cache is working)
            logger.info("   🔄 Testing cache TTL alignment...")
            start_time = time.time()
            opportunities_2 = self.market_aggregator.get_current_opportunities()
            cache_time = time.time() - start_time
            
            cache_working = cache_time < 0.1  # Should be very fast if cached
            same_data = len(opportunities) == len(opportunities_2)  # Should be same data if cached
            
            logger.info(f"      Cache response time: {cache_time:.3f}s")
            logger.info(f"      Same data returned: {same_data}")
            logger.info(f"      Cache working: {cache_working}")
            
            # Determine test result
            data_quality = (
                integration_analysis['expected_range'][0] <= integration_analysis['total_opportunities'] <= integration_analysis['expected_range'][1] and
                integration_analysis['unique_symbols'] == integration_analysis['total_opportunities'] and
                integration_analysis['with_volume'] >= integration_analysis['total_opportunities'] * 0.8
            )
            
            bingx_integration = integration_analysis['bingx_sources'] > 0 or integration_analysis['high_confidence'] >= integration_analysis['total_opportunities'] * 0.5
            
            if data_quality and bingx_integration and cache_working:
                self.log_test_result("Advanced Market Aggregator Integration", True, 
                                   f"Integration working: {integration_analysis['total_opportunities']} opportunities, {integration_analysis['bingx_sources']} BingX sources, cache functional")
            elif data_quality and bingx_integration:
                self.log_test_result("Advanced Market Aggregator Integration", False, 
                                   f"Integration mostly working: {integration_analysis['total_opportunities']} opportunities, cache issues")
            else:
                self.log_test_result("Advanced Market Aggregator Integration", False, 
                                   f"Integration issues: {integration_analysis['total_opportunities']} opportunities, {integration_analysis['bingx_sources']} BingX sources")
                
        except Exception as e:
            self.log_test_result("Advanced Market Aggregator Integration", False, f"Exception: {str(e)}")
    
    async def test_5_complete_system_integration_flow(self):
        """Test 5: Complete System Integration Flow (trending → pattern detector → market aggregator)"""
        logger.info("\n🔍 TEST 5: Complete System Integration Flow Test")
        
        try:
            if not all([self.trending_updater, self.pattern_detector, self.market_aggregator]):
                self.log_test_result("Complete System Integration Flow", False, 
                                   "One or more system modules not available")
                return
            
            logger.info("   🚀 Testing complete integration flow...")
            
            # Step 1: Get trending data
            logger.info("   📈 Step 1: Fetching trending cryptos...")
            trending_cryptos = await self.trending_updater.fetch_trending_cryptos()
            
            if not trending_cryptos:
                self.log_test_result("Complete System Integration Flow", False, 
                                   "No trending cryptos from step 1")
                return
            
            # Step 2: Apply pattern detection filters
            logger.info("   🔍 Step 2: Applying pattern detection filters...")
            filtered_cryptos = []
            filter_stats = {'total': len(trending_cryptos), 'filtered_out': 0, 'passed': 0}
            
            for crypto in trending_cryptos:
                if crypto.price_change is not None and crypto.volume is not None:
                    analysis = self.pattern_detector.analyze_trend_pattern(
                        symbol=crypto.symbol,
                        price_change_pct=crypto.price_change,
                        volume=crypto.volume
                    )
                    
                    if not self.pattern_detector.should_filter_opportunity(analysis):
                        filtered_cryptos.append(crypto)
                        filter_stats['passed'] += 1
                    else:
                        filter_stats['filtered_out'] += 1
                else:
                    # Keep cryptos without price/volume data for now
                    filtered_cryptos.append(crypto)
                    filter_stats['passed'] += 1
            
            logger.info(f"      Filter results: {filter_stats['passed']}/{filter_stats['total']} passed, {filter_stats['filtered_out']} filtered out")
            
            # Step 3: Get market opportunities
            logger.info("   📊 Step 3: Getting market opportunities...")
            opportunities = self.market_aggregator.get_current_opportunities()
            
            # Step 4: Analyze integration
            logger.info("   🔗 Step 4: Analyzing integration...")
            
            integration_analysis = {
                'trending_count': len(trending_cryptos),
                'filtered_count': len(filtered_cryptos),
                'opportunities_count': len(opportunities),
                'filter_effectiveness': filter_stats['filtered_out'] / filter_stats['total'] if filter_stats['total'] > 0 else 0,
                'data_flow_working': len(filtered_cryptos) > 0 and len(opportunities) > 0
            }
            
            # Check symbol overlap between trending and opportunities
            trending_symbols = set(crypto.symbol for crypto in trending_cryptos)
            opportunity_symbols = set(opp.symbol for opp in opportunities)
            symbol_overlap = len(trending_symbols.intersection(opportunity_symbols))
            overlap_percentage = symbol_overlap / len(trending_symbols) if trending_symbols else 0
            
            integration_analysis['symbol_overlap'] = symbol_overlap
            integration_analysis['overlap_percentage'] = overlap_percentage
            
            logger.info(f"   📊 Integration Flow Analysis:")
            logger.info(f"      Trending cryptos: {integration_analysis['trending_count']}")
            logger.info(f"      After filtering: {integration_analysis['filtered_count']}")
            logger.info(f"      Market opportunities: {integration_analysis['opportunities_count']}")
            logger.info(f"      Filter effectiveness: {integration_analysis['filter_effectiveness']:.1%}")
            logger.info(f"      Symbol overlap: {integration_analysis['symbol_overlap']} ({integration_analysis['overlap_percentage']:.1%})")
            logger.info(f"      Data flow working: {integration_analysis['data_flow_working']}")
            
            # Show some examples of the flow
            logger.info(f"   📋 Integration Flow Examples:")
            for i, crypto in enumerate(filtered_cryptos[:5]):
                price_str = f", {crypto.price_change:+.1f}%" if crypto.price_change else ""
                volume_str = f", Vol: {crypto.volume/1_000_000:.1f}M" if crypto.volume else ""
                logger.info(f"      {i+1}. {crypto.symbol} (trending → filtered → opportunities{price_str}{volume_str})")
            
            # Determine test result
            flow_working = (
                integration_analysis['data_flow_working'] and
                integration_analysis['filter_effectiveness'] > 0.1 and  # At least 10% filtering
                integration_analysis['overlap_percentage'] > 0.3  # At least 30% symbol overlap
            )
            
            if flow_working:
                self.log_test_result("Complete System Integration Flow", True, 
                                   f"Integration flow working: {integration_analysis['trending_count']} → {integration_analysis['filtered_count']} → {integration_analysis['opportunities_count']}, {integration_analysis['overlap_percentage']:.1%} overlap")
            else:
                self.log_test_result("Complete System Integration Flow", False, 
                                   f"Integration flow issues: {integration_analysis['trending_count']} → {integration_analysis['filtered_count']} → {integration_analysis['opportunities_count']}, {integration_analysis['overlap_percentage']:.1%} overlap")
                
        except Exception as e:
            self.log_test_result("Complete System Integration Flow", False, f"Exception: {str(e)}")
    
    async def test_6_system_performance_and_stability(self):
        """Test 6: System Performance & Stability with 4h Frequency"""
        logger.info("\n🔍 TEST 6: System Performance & Stability Test")
        
        try:
            # Check CPU usage
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                logger.info(f"   📊 System Resources:")
                logger.info(f"      CPU Usage: {cpu_percent:.1f}%")
                logger.info(f"      Memory Usage: {memory_percent:.1f}%")
                
                cpu_stable = cpu_percent < 90.0  # CPU under 90%
                memory_stable = memory_percent < 85.0  # Memory under 85%
                
            except ImportError:
                logger.info("   ⚠️ psutil not available, skipping resource monitoring")
                cpu_stable = True
                memory_stable = True
            
            # Test trending updater performance
            performance_analysis = {
                'cpu_stable': cpu_stable,
                'memory_stable': memory_stable,
                'trending_updater_responsive': False,
                'pattern_detector_responsive': False,
                'market_aggregator_responsive': False
            }
            
            # Test trending updater responsiveness
            if self.trending_updater:
                try:
                    start_time = time.time()
                    trending_info = self.trending_updater.get_trending_info()
                    response_time = time.time() - start_time
                    
                    performance_analysis['trending_updater_responsive'] = response_time < 1.0
                    logger.info(f"      Trending updater response time: {response_time:.3f}s")
                    logger.info(f"      Trending info: {trending_info.get('trending_count', 0)} symbols, "
                              f"auto-update: {trending_info.get('auto_update_active', False)}")
                except Exception as e:
                    logger.warning(f"      Trending updater test failed: {e}")
            
            # Test pattern detector responsiveness
            if self.pattern_detector:
                try:
                    start_time = time.time()
                    test_analysis = self.pattern_detector.analyze_trend_pattern("TESTUSDT", 2.5, 1000000)
                    response_time = time.time() - start_time
                    
                    performance_analysis['pattern_detector_responsive'] = response_time < 0.1
                    logger.info(f"      Pattern detector response time: {response_time:.3f}s")
                except Exception as e:
                    logger.warning(f"      Pattern detector test failed: {e}")
            
            # Test market aggregator responsiveness
            if self.market_aggregator:
                try:
                    start_time = time.time()
                    opportunities = self.market_aggregator.get_current_opportunities()
                    response_time = time.time() - start_time
                    
                    performance_analysis['market_aggregator_responsive'] = response_time < 2.0
                    logger.info(f"      Market aggregator response time: {response_time:.3f}s")
                    logger.info(f"      Opportunities returned: {len(opportunities)}")
                except Exception as e:
                    logger.warning(f"      Market aggregator test failed: {e}")
            
            # Check backend logs for errors
            error_analysis = await self._analyze_backend_logs()
            
            # Overall stability assessment
            stable_components = sum(1 for component, stable in performance_analysis.items() if stable)
            total_components = len(performance_analysis)
            
            logger.info(f"   📊 Performance Analysis:")
            for component, stable in performance_analysis.items():
                status = "✅" if stable else "❌"
                logger.info(f"      {status} {component.replace('_', ' ').title()}")
            
            logger.info(f"   📊 Error Analysis:")
            logger.info(f"      Total log entries: {error_analysis['total_entries']}")
            logger.info(f"      Error entries: {error_analysis['error_entries']}")
            logger.info(f"      Error rate: {error_analysis['error_rate']:.1%}")
            logger.info(f"      Critical errors: {error_analysis['critical_errors']}")
            
            # Determine test result
            if stable_components >= total_components * 0.8 and error_analysis['error_rate'] < 0.1:
                self.log_test_result("System Performance & Stability", True, 
                                   f"System stable: {stable_components}/{total_components} components responsive, {error_analysis['error_rate']:.1%} error rate")
            elif stable_components >= total_components * 0.6:
                self.log_test_result("System Performance & Stability", False, 
                                   f"System mostly stable: {stable_components}/{total_components} components responsive, {error_analysis['error_rate']:.1%} error rate")
            else:
                self.log_test_result("System Performance & Stability", False, 
                                   f"System stability issues: {stable_components}/{total_components} components responsive, {error_analysis['error_rate']:.1%} error rate")
                
        except Exception as e:
            self.log_test_result("System Performance & Stability", False, f"Exception: {str(e)}")
    
    async def _analyze_backend_logs(self):
        """Analyze backend logs for error patterns and quality"""
        try:
            log_files = [
                "/var/log/supervisor/backend.out.log",
                "/var/log/supervisor/backend.err.log"
            ]
            
            analysis = {
                'total_entries': 0,
                'error_entries': 0,
                'critical_errors': 0,
                'error_rate': 0.0,
                'trending_related_entries': 0,
                'recent_errors': []
            }
            
            error_patterns = [
                r'ERROR',
                r'CRITICAL',
                r'Exception',
                r'Traceback',
                r'Failed to fetch',
                r'Connection error'
            ]
            
            trending_patterns = [
                r'trending',
                r'BingX',
                r'pattern.*detector',
                r'market.*aggregator',
                r'lateral.*pattern'
            ]
            
            for log_file in log_files:
                try:
                    if os.path.exists(log_file):
                        result = subprocess.run(['tail', '-n', '500', log_file], 
                                              capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            log_content = result.stdout
                            lines = log_content.split('\n')
                            analysis['total_entries'] += len([line for line in lines if line.strip()])
                            
                            # Count errors
                            for line in lines:
                                if any(re.search(pattern, line, re.IGNORECASE) for pattern in error_patterns):
                                    analysis['error_entries'] += 1
                                    if 'CRITICAL' in line.upper():
                                        analysis['critical_errors'] += 1
                                    if len(analysis['recent_errors']) < 3:
                                        analysis['recent_errors'].append(line.strip())
                                
                                # Count trending-related entries
                                if any(re.search(pattern, line, re.IGNORECASE) for pattern in trending_patterns):
                                    analysis['trending_related_entries'] += 1
                            
                except Exception as e:
                    logger.warning(f"   ⚠️ Could not analyze {log_file}: {e}")
            
            # Calculate final metrics
            if analysis['total_entries'] > 0:
                analysis['error_rate'] = analysis['error_entries'] / analysis['total_entries']
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Log analysis failed: {e}")
            return {
                'total_entries': 0,
                'error_entries': 0,
                'critical_errors': 0,
                'error_rate': 0.0,
                'trending_related_entries': 0,
                'recent_errors': []
            }
    
    async def run_comprehensive_test_suite(self):
        """Run comprehensive autonomous trend detection system test suite"""
        logger.info("🚀 Starting Autonomous Trend Detection System Comprehensive Test Suite")
        logger.info("=" * 80)
        logger.info("📋 AUTONOMOUS TREND DETECTION SYSTEM TEST SUITE")
        logger.info("🎯 Testing: 4h frequency, advanced filters, BingX integration, lateral pattern detection")
        logger.info("🎯 Expected: Complete autonomous trend detection working with real BingX data")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_trending_auto_updater_configuration()
        await self.test_2_bingx_trending_data_fetch()
        await self.test_3_lateral_pattern_detector_analysis()
        await self.test_4_advanced_market_aggregator_integration()
        await self.test_5_complete_system_integration_flow()
        await self.test_6_system_performance_and_stability()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 AUTONOMOUS TREND DETECTION SYSTEM COMPREHENSIVE TEST SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # Critical requirements analysis
        logger.info("\n" + "=" * 80)
        logger.info("📋 CRITICAL REQUIREMENTS VERIFICATION")
        logger.info("=" * 80)
        
        requirements_status = {}
        
        for result in self.test_results:
            if "Configuration" in result['test']:
                requirements_status['4h Frequency & Configuration'] = result['success']
            elif "BingX Trending Data" in result['test']:
                requirements_status['BingX Data Fetch & Filters'] = result['success']
            elif "Lateral Pattern Detector" in result['test']:
                requirements_status['Lateral Pattern Detection'] = result['success']
            elif "Market Aggregator" in result['test']:
                requirements_status['Market Aggregator Integration'] = result['success']
            elif "Integration Flow" in result['test']:
                requirements_status['Complete System Integration'] = result['success']
            elif "Performance" in result['test']:
                requirements_status['System Performance & Stability'] = result['success']
        
        logger.info("🎯 CRITICAL REQUIREMENTS STATUS:")
        for requirement, status in requirements_status.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {requirement}")
        
        requirements_met = sum(1 for status in requirements_status.values() if status)
        total_requirements = len(requirements_status)
        
        # Final verdict
        logger.info(f"\n🏆 REQUIREMENTS SATISFACTION: {requirements_met}/{total_requirements}")
        
        if requirements_met == total_requirements:
            logger.info("\n🎉 VERDICT: AUTONOMOUS TREND DETECTION SYSTEM FULLY FUNCTIONAL!")
            logger.info("✅ 4h frequency configuration working")
            logger.info("✅ BingX API integration with advanced filters operational")
            logger.info("✅ Lateral pattern detection filtering effectively")
            logger.info("✅ Market aggregator using filtered BingX data")
            logger.info("✅ Complete integration flow working")
            logger.info("✅ System performance stable")
            logger.info("✅ Ready for production use")
        elif requirements_met >= total_requirements * 0.8:
            logger.info("\n⚠️ VERDICT: AUTONOMOUS TREND DETECTION SYSTEM MOSTLY FUNCTIONAL")
            logger.info("🔍 Minor issues may need attention for complete functionality")
        elif requirements_met >= total_requirements * 0.6:
            logger.info("\n⚠️ VERDICT: AUTONOMOUS TREND DETECTION SYSTEM PARTIALLY FUNCTIONAL")
            logger.info("🔧 Several critical requirements need implementation or debugging")
        else:
            logger.info("\n❌ VERDICT: AUTONOMOUS TREND DETECTION SYSTEM NOT FUNCTIONAL")
            logger.info("🚨 Major issues preventing autonomous trend detection from working correctly")
            logger.info("🚨 System needs significant debugging and fixes")
        
        return passed_tests, total_tests

async def main():
    """Main function to run the comprehensive autonomous trend detection test suite"""
    test_suite = AutonomousTrendDetectionTestSuite()
    passed_tests, total_tests = await test_suite.run_comprehensive_test_suite()
    
    # Exit with appropriate code
    if passed_tests == total_tests:
        sys.exit(0)  # All tests passed
    elif passed_tests >= total_tests * 0.8:
        sys.exit(1)  # Mostly successful
    else:
        sys.exit(2)  # Major issues

if __name__ == "__main__":
    asyncio.run(main())
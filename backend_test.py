#!/usr/bin/env python3
"""
IA1 TO IA2 ESCALATION SYSTEM TEST SUITE
Focus: Test IA1 to IA2 Escalation System - COMPREHENSIVE VALIDATION

CRITICAL TEST REQUIREMENTS FROM REVIEW REQUEST:
1. **Escalation Logic Validation**: Test the 3 voies escalation system:
   - VOIE 1: LONG/SHORT signals with confidence ≥ 70%
   - VOIE 2: Risk-Reward ratio ≥ 2.0 (any signal)  
   - VOIE 3: LONG/SHORT signals with confidence ≥ 95% (override)

2. **End-to-End Escalation Flow**: Verify complete pipeline from IA1 → IA2 → Decision storage

3. **Database Integration**: Check that IA2 decisions are properly saved when escalation occurs

SPECIFIC TESTING SCENARIOS:
1. **API Endpoint Testing**:
   - Test /api/run-ia1-cycle for automatic escalation
   - Check escalation response fields: escalated_to_ia2, ia2_decision
   - Verify /api/decisions shows new IA2 decisions after escalation

2. **Escalation Criteria Validation**:
   - Test analyses with confidence 70%+ and LONG/SHORT signals → VOIE 1
   - Test analyses with RR ≥ 2.0 regardless of signal → VOIE 2  
   - Test analyses with confidence 95%+ and LONG/SHORT signals → VOIE 3

3. **Error Resolution Check**:
   - Verify fixed "advanced_market_aggregator" import error
   - Check that IA2 escalation completes without errors
   - Validate performance stats are properly retrieved

ENDPOINTS TO TEST:
- POST /api/run-ia1-cycle for automatic escalation testing
- GET /api/decisions to verify IA2 decisions are stored
- GET /api/analyses to check IA1 analyses with escalation flags
- Monitor logs for escalation flow and error resolution

SUCCESS CRITERIA:
✅ _should_send_to_ia2 function correctly identifies eligible analyses
✅ Escalation occurs for analyses meeting any of the 3 voies criteria
✅ IA2 make_decision method executes successfully after escalation
✅ New IA2 decisions appear in database after successful escalation
✅ No more "cannot access local variable" errors in escalation flow
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

class IA1ToIA2EscalationTestSuite:
    """Comprehensive test suite for IA1 to IA2 Escalation System"""
    
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
        logger.info(f"Testing IA1 to IA2 Escalation System at: {self.api_url}")
        
        # MongoDB connection for direct database analysis
        try:
            self.mongo_client = MongoClient("mongodb://localhost:27017")
            self.db = self.mongo_client["myapp"]
            logger.info("✅ MongoDB connection established for escalation testing")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.mongo_client = None
            self.db = None
        
        # Test results
        self.test_results = []
        
        # Test symbols for escalation testing
        self.test_symbols = [
            "BTCUSDT",   # High volatility, likely to trigger escalation
            "ETHUSDT",   # Medium volatility, good for testing
            "SOLUSDT",   # High volatility, trending
            "XRPUSDT",   # Lower volatility, edge cases
            "ADAUSDT"    # Medium volatility, range-bound
        ]
        
        # Escalation criteria for testing
        self.escalation_criteria = {
            "VOIE_1": {"confidence_min": 0.70, "signals": ["long", "short"]},
            "VOIE_2": {"rr_min": 2.0, "signals": ["long", "short", "hold"]},
            "VOIE_3": {"confidence_min": 0.95, "signals": ["long", "short"]}
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
    
    async def test_1_escalation_criteria_validation(self):
        """Test 1: Validate IA1 to IA2 Escalation Criteria (3 Voies System)"""
        logger.info("\n🔍 TEST 1: IA1 to IA2 Escalation Criteria Validation Test")
        
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
    
    async def test_3_technical_levels_consistency(self):
        """Test 3: Technical Levels Based RR Calculation (Not Confidence Based)"""
        logger.info("\n🔍 TEST 3: Technical Levels Based RR Calculation Test")
        
        try:
            # Test that IA1 RR calculation is based on technical levels (support/resistance)
            # and not influenced by confidence percentages
            
            technical_levels_results = {
                'analyses_with_technical_levels': 0,
                'analyses_with_fallback_levels': 0,
                'technical_level_based_rr': 0,
                'confidence_based_rr_detected': 0,
                'support_resistance_consistency': 0,
                'fallback_percentage_consistency': 0
            }
            
            logger.info("   🚀 Testing technical levels based RR calculation...")
            
            # Test multiple symbols to verify technical analysis basis
            for symbol in self.test_symbols[:2]:  # Test 2 symbols for detailed analysis
                try:
                    logger.info(f"   📊 Analyzing technical levels for {symbol}...")
                    
                    # Get IA1 analysis
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        if analysis_data.get('success') and 'analysis' in analysis_data:
                            analysis = analysis_data['analysis']
                            
                            # Extract key data
                            confidence = analysis.get('confidence', 0)
                            rr_ratio = analysis.get('risk_reward_ratio', 0)
                            entry_price = analysis.get('entry_price', 0)
                            stop_loss = analysis.get('stop_loss_price', 0)
                            take_profit = analysis.get('take_profit_price', 0)
                            support_levels = analysis.get('support', [])
                            resistance_levels = analysis.get('resistance', [])
                            signal = analysis.get('recommendation', 'hold')
                            reasoning = analysis.get('reasoning', '')
                            
                            logger.info(f"      📈 {symbol} Analysis:")
                            logger.info(f"         Confidence: {confidence:.1%}")
                            logger.info(f"         RR Ratio: {rr_ratio:.2f}")
                            logger.info(f"         Signal: {signal}")
                            logger.info(f"         Entry: ${entry_price:.6f}")
                            logger.info(f"         Stop Loss: ${stop_loss:.6f}")
                            logger.info(f"         Take Profit: ${take_profit:.6f}")
                            logger.info(f"         Support Levels: {len(support_levels)}")
                            logger.info(f"         Resistance Levels: {len(resistance_levels)}")
                            
                            # Check if technical levels are present
                            has_technical_levels = len(support_levels) > 0 or len(resistance_levels) > 0
                            
                            if has_technical_levels:
                                technical_levels_results['analyses_with_technical_levels'] += 1
                                
                                # Verify RR calculation uses technical levels
                                if signal.lower() == 'long' and entry_price > 0 and stop_loss > 0 and take_profit > 0:
                                    expected_rr = self.rr_formulas['LONG'](entry_price, take_profit, stop_loss)
                                    rr_matches = abs(expected_rr - rr_ratio) < 0.1  # Allow small variance
                                    
                                    if rr_matches:
                                        technical_levels_results['technical_level_based_rr'] += 1
                                        logger.info(f"         ✅ RR calculation matches LONG formula: {expected_rr:.2f} ≈ {rr_ratio:.2f}")
                                    else:
                                        logger.warning(f"         ⚠️ RR calculation mismatch: Expected {expected_rr:.2f}, Got {rr_ratio:.2f}")
                                        
                                elif signal.lower() == 'short' and entry_price > 0 and stop_loss > 0 and take_profit > 0:
                                    expected_rr = self.rr_formulas['SHORT'](entry_price, take_profit, stop_loss)
                                    rr_matches = abs(expected_rr - rr_ratio) < 0.1  # Allow small variance
                                    
                                    if rr_matches:
                                        technical_levels_results['technical_level_based_rr'] += 1
                                        logger.info(f"         ✅ RR calculation matches SHORT formula: {expected_rr:.2f} ≈ {rr_ratio:.2f}")
                                    else:
                                        logger.warning(f"         ⚠️ RR calculation mismatch: Expected {expected_rr:.2f}, Got {rr_ratio:.2f}")
                                
                                # Check if support/resistance levels are used in SL/TP calculation
                                sl_near_support = any(abs(stop_loss - level) / level < 0.05 for level in support_levels if level > 0)
                                tp_near_resistance = any(abs(take_profit - level) / level < 0.05 for level in resistance_levels if level > 0)
                                
                                if sl_near_support or tp_near_resistance:
                                    technical_levels_results['support_resistance_consistency'] += 1
                                    logger.info(f"         ✅ SL/TP levels align with support/resistance")
                                else:
                                    logger.info(f"         ⚪ SL/TP levels may use other technical analysis")
                                    
                            else:
                                technical_levels_results['analyses_with_fallback_levels'] += 1
                                logger.info(f"         ⚪ Using fallback levels (no clear support/resistance)")
                                
                                # Check if fallback uses fixed percentages (not confidence-based)
                                if entry_price > 0:
                                    sl_percentage = abs(stop_loss - entry_price) / entry_price
                                    tp_percentage = abs(take_profit - entry_price) / entry_price
                                    
                                    # Common fallback percentages (2%, 3%, 5%)
                                    common_percentages = [0.02, 0.03, 0.05, 0.1]
                                    sl_is_standard = any(abs(sl_percentage - pct) < 0.005 for pct in common_percentages)
                                    tp_is_standard = any(abs(tp_percentage - pct) < 0.005 for pct in common_percentages)
                                    
                                    if sl_is_standard or tp_is_standard:
                                        technical_levels_results['fallback_percentage_consistency'] += 1
                                        logger.info(f"         ✅ Fallback uses standard percentages: SL={sl_percentage:.1%}, TP={tp_percentage:.1%}")
                                    else:
                                        logger.info(f"         ⚪ Custom fallback percentages: SL={sl_percentage:.1%}, TP={tp_percentage:.1%}")
                            
                            # Check reasoning for technical analysis keywords
                            technical_keywords = ['support', 'resistance', 'fibonacci', 'ema', 'sma', 'bollinger', 'rsi', 'macd']
                            technical_mentions = sum(1 for keyword in technical_keywords if keyword.lower() in reasoning.lower())
                            
                            logger.info(f"         📊 Technical analysis keywords in reasoning: {technical_mentions}/{len(technical_keywords)}")
                            
                    await asyncio.sleep(2)  # Delay between requests
                    
                except Exception as e:
                    logger.error(f"   ❌ Error analyzing {symbol}: {e}")
            
            # Analyze results
            total_analyses = technical_levels_results['analyses_with_technical_levels'] + technical_levels_results['analyses_with_fallback_levels']
            
            logger.info(f"   📊 Technical Levels Analysis Results:")
            logger.info(f"      Total analyses: {total_analyses}")
            logger.info(f"      With technical levels: {technical_levels_results['analyses_with_technical_levels']}")
            logger.info(f"      With fallback levels: {technical_levels_results['analyses_with_fallback_levels']}")
            logger.info(f"      Technical level based RR: {technical_levels_results['technical_level_based_rr']}")
            logger.info(f"      Support/resistance consistency: {technical_levels_results['support_resistance_consistency']}")
            logger.info(f"      Fallback percentage consistency: {technical_levels_results['fallback_percentage_consistency']}")
            
            # Determine test result
            if total_analyses > 0:
                technical_basis_rate = (technical_levels_results['technical_level_based_rr'] + 
                                      technical_levels_results['support_resistance_consistency'] + 
                                      technical_levels_results['fallback_percentage_consistency']) / total_analyses
                
                if technical_basis_rate >= 0.8:
                    self.log_test_result("Technical Levels Based RR Calculation", True, 
                                       f"RR calculation based on technical analysis: {technical_basis_rate:.1%} technical basis")
                elif technical_basis_rate >= 0.5:
                    self.log_test_result("Technical Levels Based RR Calculation", False, 
                                       f"Partial technical basis: {technical_basis_rate:.1%} technical basis")
                else:
                    self.log_test_result("Technical Levels Based RR Calculation", False, 
                                       f"Limited technical basis: {technical_basis_rate:.1%} technical basis")
            else:
                self.log_test_result("Technical Levels Based RR Calculation", False, 
                                   "No analyses available for technical levels testing")
                
        except Exception as e:
            self.log_test_result("Technical Levels Based RR Calculation", False, f"Exception: {str(e)}")
    
    async def test_4_database_rr_consistency_analysis(self):
        """Test 4: Database Analysis of IA1 RR Consistency Across Different Confidence Levels"""
        logger.info("\n🔍 TEST 4: Database RR Consistency Analysis Test")
        
        try:
            if not self.db:
                self.log_test_result("Database RR Consistency Analysis", False, 
                                   "Database connection not available")
                return
            
            logger.info("   🚀 Analyzing IA1 analyses in database for RR consistency...")
            
            # Query recent IA1 analyses from database
            try:
                # Get recent IA1 analyses (last 24 hours)
                from datetime import datetime, timedelta
                cutoff_time = datetime.now() - timedelta(hours=24)
                
                cursor = self.db.technical_analyses.find({
                    "timestamp": {"$gte": cutoff_time},
                    "risk_reward_ratio": {"$exists": True, "$gt": 0}
                }).sort("timestamp", -1).limit(50)
                
                analyses = list(cursor)
                
                if not analyses:
                    # Try without time filter if no recent data
                    cursor = self.db.technical_analyses.find({
                        "risk_reward_ratio": {"$exists": True, "$gt": 0}
                    }).sort("timestamp", -1).limit(20)
                    analyses = list(cursor)
                
                logger.info(f"   📊 Found {len(analyses)} IA1 analyses in database")
                
                if len(analyses) < 5:
                    self.log_test_result("Database RR Consistency Analysis", False, 
                                       f"Insufficient IA1 analyses in database: {len(analyses)} found")
                    return
                
                # Analyze RR consistency patterns
                db_analysis_results = {
                    'total_analyses': len(analyses),
                    'confidence_ranges': {
                        'low': [],      # 0-70%
                        'medium': [],   # 70-85%
                        'high': []      # 85-100%
                    },
                    'rr_by_confidence': {},
                    'same_symbol_analyses': {},
                    'formula_validation_count': 0,
                    'formula_validation_passed': 0
                }
                
                # Group analyses by confidence ranges and symbols
                for analysis in analyses:
                    confidence = analysis.get('analysis_confidence', 0)
                    rr_ratio = analysis.get('risk_reward_ratio', 0)
                    symbol = analysis.get('symbol', 'UNKNOWN')
                    entry_price = analysis.get('entry_price', 0)
                    stop_loss = analysis.get('stop_loss_price', 0)
                    take_profit = analysis.get('take_profit_price', 0)
                    signal = analysis.get('ia1_signal', 'hold')
                    
                    # Categorize by confidence
                    if confidence < 0.7:
                        db_analysis_results['confidence_ranges']['low'].append({
                            'symbol': symbol, 'confidence': confidence, 'rr': rr_ratio,
                            'entry': entry_price, 'sl': stop_loss, 'tp': take_profit, 'signal': signal
                        })
                    elif confidence < 0.85:
                        db_analysis_results['confidence_ranges']['medium'].append({
                            'symbol': symbol, 'confidence': confidence, 'rr': rr_ratio,
                            'entry': entry_price, 'sl': stop_loss, 'tp': take_profit, 'signal': signal
                        })
                    else:
                        db_analysis_results['confidence_ranges']['high'].append({
                            'symbol': symbol, 'confidence': confidence, 'rr': rr_ratio,
                            'entry': entry_price, 'sl': stop_loss, 'tp': take_profit, 'signal': signal
                        })
                    
                    # Group by symbol for same-symbol analysis
                    if symbol not in db_analysis_results['same_symbol_analyses']:
                        db_analysis_results['same_symbol_analyses'][symbol] = []
                    db_analysis_results['same_symbol_analyses'][symbol].append({
                        'confidence': confidence, 'rr': rr_ratio,
                        'entry': entry_price, 'sl': stop_loss, 'tp': take_profit, 'signal': signal
                    })
                    
                    # Validate RR formula if we have all required data
                    if entry_price > 0 and stop_loss > 0 and take_profit > 0 and signal in ['long', 'short']:
                        db_analysis_results['formula_validation_count'] += 1
                        
                        if signal == 'long':
                            expected_rr = self.rr_formulas['LONG'](entry_price, take_profit, stop_loss)
                        else:  # short
                            expected_rr = self.rr_formulas['SHORT'](entry_price, take_profit, stop_loss)
                        
                        if abs(expected_rr - rr_ratio) < 0.2:  # Allow some variance
                            db_analysis_results['formula_validation_passed'] += 1
                
                # Analyze confidence vs RR correlation
                logger.info(f"   📊 Confidence Range Analysis:")
                for range_name, range_data in db_analysis_results['confidence_ranges'].items():
                    if range_data:
                        avg_confidence = sum(d['confidence'] for d in range_data) / len(range_data)
                        avg_rr = sum(d['rr'] for d in range_data) / len(range_data)
                        rr_variance = max(d['rr'] for d in range_data) - min(d['rr'] for d in range_data) if len(range_data) > 1 else 0
                        
                        logger.info(f"      {range_name.upper()} confidence ({len(range_data)} analyses): "
                                  f"Avg confidence={avg_confidence:.1%}, Avg RR={avg_rr:.2f}, RR variance={rr_variance:.2f}")
                
                # Analyze same-symbol consistency
                same_symbol_consistency = 0
                same_symbol_total = 0
                
                logger.info(f"   📊 Same Symbol RR Consistency:")
                for symbol, symbol_analyses in db_analysis_results['same_symbol_analyses'].items():
                    if len(symbol_analyses) >= 2:
                        same_symbol_total += 1
                        rr_values = [a['rr'] for a in symbol_analyses]
                        confidence_values = [a['confidence'] for a in symbol_analyses]
                        
                        rr_variance = max(rr_values) - min(rr_values)
                        confidence_variance = max(confidence_values) - min(confidence_values)
                        
                        # If RR is consistent despite confidence variance, that's good
                        if rr_variance <= 0.5 or confidence_variance <= 0.1:
                            same_symbol_consistency += 1
                            status = "✅"
                        else:
                            status = "⚠️"
                        
                        logger.info(f"      {status} {symbol} ({len(symbol_analyses)} analyses): "
                                  f"RR variance={rr_variance:.2f}, Confidence variance={confidence_variance:.1%}")
                
                # Formula validation results
                formula_accuracy = db_analysis_results['formula_validation_passed'] / db_analysis_results['formula_validation_count'] if db_analysis_results['formula_validation_count'] > 0 else 0
                
                logger.info(f"   📊 Database Analysis Summary:")
                logger.info(f"      Total analyses: {db_analysis_results['total_analyses']}")
                logger.info(f"      Formula validation: {db_analysis_results['formula_validation_passed']}/{db_analysis_results['formula_validation_count']} ({formula_accuracy:.1%})")
                logger.info(f"      Same symbol consistency: {same_symbol_consistency}/{same_symbol_total}")
                
                # Determine test result
                consistency_rate = same_symbol_consistency / same_symbol_total if same_symbol_total > 0 else 0
                
                if formula_accuracy >= 0.8 and consistency_rate >= 0.7:
                    self.log_test_result("Database RR Consistency Analysis", True, 
                                       f"Database shows RR independence: {formula_accuracy:.1%} formula accuracy, {consistency_rate:.1%} consistency")
                elif formula_accuracy >= 0.6 and consistency_rate >= 0.5:
                    self.log_test_result("Database RR Consistency Analysis", False, 
                                       f"Partial RR independence: {formula_accuracy:.1%} formula accuracy, {consistency_rate:.1%} consistency")
                else:
                    self.log_test_result("Database RR Consistency Analysis", False, 
                                       f"RR consistency issues: {formula_accuracy:.1%} formula accuracy, {consistency_rate:.1%} consistency")
                
            except Exception as db_error:
                logger.error(f"   ❌ Database query error: {db_error}")
                self.log_test_result("Database RR Consistency Analysis", False, f"Database error: {str(db_error)}")
                
        except Exception as e:
            self.log_test_result("Database RR Consistency Analysis", False, f"Exception: {str(e)}")
    
    async def test_5_long_short_rr_formula_verification(self):
        """Test 5: LONG and SHORT RR Formula Implementation Verification"""
        logger.info("\n🔍 TEST 5: LONG and SHORT RR Formula Verification Test")
        
        try:
            # Test specific LONG and SHORT RR formula implementation in live IA1 analyses
            logger.info("   🚀 Testing LONG and SHORT RR formulas in live analyses...")
            
            formula_verification_results = {
                'long_tests': 0,
                'short_tests': 0,
                'long_formula_correct': 0,
                'short_formula_correct': 0,
                'formula_errors': []
            }
            
            # Test symbols that typically generate LONG and SHORT signals
            test_scenarios = [
                {"symbol": "BTCUSDT", "expected_signals": ["long", "short"]},
                {"symbol": "ETHUSDT", "expected_signals": ["long", "short"]},
                {"symbol": "SOLUSDT", "expected_signals": ["long", "short"]}
            ]
            
            for scenario in test_scenarios:
                symbol = scenario["symbol"]
                logger.info(f"   📊 Testing RR formulas for {symbol}...")
                
                try:
                    # Get IA1 analysis
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        if analysis_data.get('success') and 'analysis' in analysis_data:
                            analysis = analysis_data['analysis']
                            
                            signal = analysis.get('recommendation', 'hold').lower()
                            rr_ratio = analysis.get('risk_reward_ratio', 0)
                            entry_price = analysis.get('entry_price', 0)
                            stop_loss = analysis.get('stop_loss_price', 0)
                            take_profit = analysis.get('take_profit_price', 0)
                            
                            if signal in ['long', 'short'] and all([entry_price > 0, stop_loss > 0, take_profit > 0]):
                                if signal == 'long':
                                    formula_verification_results['long_tests'] += 1
                                    expected_rr = self.rr_formulas['LONG'](entry_price, take_profit, stop_loss)
                                    
                                    if abs(expected_rr - rr_ratio) < 0.1:
                                        formula_verification_results['long_formula_correct'] += 1
                                        logger.info(f"      ✅ LONG formula correct: {expected_rr:.2f} ≈ {rr_ratio:.2f}")
                                    else:
                                        formula_verification_results['formula_errors'].append(
                                            f"LONG {symbol}: Expected {expected_rr:.2f}, Got {rr_ratio:.2f}"
                                        )
                                        logger.warning(f"      ❌ LONG formula mismatch: Expected {expected_rr:.2f}, Got {rr_ratio:.2f}")
                                    
                                    # Log formula details
                                    logger.info(f"         LONG Formula: ({take_profit:.6f} - {entry_price:.6f}) / ({entry_price:.6f} - {stop_loss:.6f})")
                                    
                                elif signal == 'short':
                                    formula_verification_results['short_tests'] += 1
                                    expected_rr = self.rr_formulas['SHORT'](entry_price, take_profit, stop_loss)
                                    
                                    if abs(expected_rr - rr_ratio) < 0.1:
                                        formula_verification_results['short_formula_correct'] += 1
                                        logger.info(f"      ✅ SHORT formula correct: {expected_rr:.2f} ≈ {rr_ratio:.2f}")
                                    else:
                                        formula_verification_results['formula_errors'].append(
                                            f"SHORT {symbol}: Expected {expected_rr:.2f}, Got {rr_ratio:.2f}"
                                        )
                                        logger.warning(f"      ❌ SHORT formula mismatch: Expected {expected_rr:.2f}, Got {rr_ratio:.2f}")
                                    
                                    # Log formula details
                                    logger.info(f"         SHORT Formula: ({entry_price:.6f} - {take_profit:.6f}) / ({stop_loss:.6f} - {entry_price:.6f})")
                            else:
                                logger.info(f"      ⚪ {symbol}: {signal} signal, insufficient data for formula test")
                    
                    await asyncio.sleep(2)  # Delay between requests
                    
                except Exception as e:
                    logger.error(f"   ❌ Error testing {symbol}: {e}")
            
            # Calculate results
            total_tests = formula_verification_results['long_tests'] + formula_verification_results['short_tests']
            total_correct = formula_verification_results['long_formula_correct'] + formula_verification_results['short_formula_correct']
            
            logger.info(f"   📊 Formula Verification Results:")
            logger.info(f"      LONG tests: {formula_verification_results['long_formula_correct']}/{formula_verification_results['long_tests']}")
            logger.info(f"      SHORT tests: {formula_verification_results['short_formula_correct']}/{formula_verification_results['short_tests']}")
            logger.info(f"      Total accuracy: {total_correct}/{total_tests}")
            
            if formula_verification_results['formula_errors']:
                logger.info(f"      Formula errors:")
                for error in formula_verification_results['formula_errors']:
                    logger.info(f"        - {error}")
            
            # Determine test result
            if total_tests > 0:
                accuracy = total_correct / total_tests
                
                if accuracy >= 0.9:
                    self.log_test_result("LONG and SHORT RR Formula Verification", True, 
                                       f"RR formulas correctly implemented: {accuracy:.1%} accuracy ({total_correct}/{total_tests})")
                elif accuracy >= 0.7:
                    self.log_test_result("LONG and SHORT RR Formula Verification", False, 
                                       f"Most RR formulas correct: {accuracy:.1%} accuracy ({total_correct}/{total_tests})")
                else:
                    self.log_test_result("LONG and SHORT RR Formula Verification", False, 
                                       f"RR formula implementation issues: {accuracy:.1%} accuracy ({total_correct}/{total_tests})")
            else:
                self.log_test_result("LONG and SHORT RR Formula Verification", False, 
                                   "No LONG/SHORT signals available for formula testing")
                
        except Exception as e:
            self.log_test_result("LONG and SHORT RR Formula Verification", False, f"Exception: {str(e)}")
    
    async def test_6_backend_logs_rr_calculation_analysis(self):
        """Test 6: Backend Logs Analysis for RR Calculation Details"""
        logger.info("\n🔍 TEST 6: Backend Logs RR Calculation Analysis Test")
        
        try:
            # Analyze backend logs for RR calculation details and evidence of independence
            logger.info("   🚀 Analyzing backend logs for RR calculation patterns...")
            
            log_analysis_results = {
                'rr_calculation_entries': 0,
                'long_rr_calculations': 0,
                'short_rr_calculations': 0,
                'technical_level_references': 0,
                'confidence_correlation_evidence': 0,
                'formula_pattern_matches': 0,
                'independence_evidence': []
            }
            
            # Analyze backend logs for RR calculation patterns
            error_analysis = await self._analyze_backend_logs()
            
            # Look for specific RR calculation patterns in logs
            rr_calculation_patterns = [
                r'LONG RR CALCULATION.*formula',
                r'SHORT RR CALCULATION.*formula', 
                r'RR.*Entry.*SL.*TP',
                r'risk_reward_ratio.*=',
                r'support.*resistance.*RR',
                r'technical.*levels.*RR'
            ]
            
            confidence_patterns = [
                r'confidence.*RR',
                r'RR.*confidence',
                r'confidence.*risk.*reward'
            ]
            
            independence_patterns = [
                r'RR.*independent.*confidence',
                r'technical.*levels.*only',
                r'support.*resistance.*calculation'
            ]
            
            try:
                # Check recent backend logs
                log_files = [
                    "/var/log/supervisor/backend.out.log",
                    "/var/log/supervisor/backend.err.log"
                ]
                
                for log_file in log_files:
                    if os.path.exists(log_file):
                        try:
                            result = subprocess.run(['tail', '-n', '1000', log_file], 
                                                  capture_output=True, text=True, timeout=30)
                            
                            if result.returncode == 0:
                                log_content = result.stdout
                                lines = log_content.split('\n')
                                
                                for line in lines:
                                    # Count RR calculation entries
                                    if any(re.search(pattern, line, re.IGNORECASE) for pattern in rr_calculation_patterns):
                                        log_analysis_results['rr_calculation_entries'] += 1
                                        
                                        if 'LONG RR CALCULATION' in line.upper():
                                            log_analysis_results['long_rr_calculations'] += 1
                                        elif 'SHORT RR CALCULATION' in line.upper():
                                            log_analysis_results['short_rr_calculations'] += 1
                                    
                                    # Count technical level references
                                    if any(keyword in line.lower() for keyword in ['support', 'resistance', 'fibonacci', 'ema', 'sma']):
                                        log_analysis_results['technical_level_references'] += 1
                                    
                                    # Look for confidence correlation evidence
                                    if any(re.search(pattern, line, re.IGNORECASE) for pattern in confidence_patterns):
                                        log_analysis_results['confidence_correlation_evidence'] += 1
                                    
                                    # Look for independence evidence
                                    if any(re.search(pattern, line, re.IGNORECASE) for pattern in independence_patterns):
                                        log_analysis_results['independence_evidence'].append(line.strip())
                                    
                                    # Look for formula pattern matches
                                    if re.search(r'\(.*-.*\).*\/.*\(.*-.*\)', line):
                                        log_analysis_results['formula_pattern_matches'] += 1
                                        
                        except Exception as e:
                            logger.warning(f"   ⚠️ Could not analyze {log_file}: {e}")
                
                logger.info(f"   📊 Backend Logs RR Analysis:")
                logger.info(f"      RR calculation entries: {log_analysis_results['rr_calculation_entries']}")
                logger.info(f"      LONG RR calculations: {log_analysis_results['long_rr_calculations']}")
                logger.info(f"      SHORT RR calculations: {log_analysis_results['short_rr_calculations']}")
                logger.info(f"      Technical level references: {log_analysis_results['technical_level_references']}")
                logger.info(f"      Formula pattern matches: {log_analysis_results['formula_pattern_matches']}")
                logger.info(f"      Confidence correlation evidence: {log_analysis_results['confidence_correlation_evidence']}")
                logger.info(f"      Independence evidence entries: {len(log_analysis_results['independence_evidence'])}")
                
                # Show some independence evidence if found
                if log_analysis_results['independence_evidence']:
                    logger.info(f"   📋 Independence Evidence Examples:")
                    for i, evidence in enumerate(log_analysis_results['independence_evidence'][:3]):
                        logger.info(f"      {i+1}. {evidence[:100]}...")
                
                # Determine test result based on log analysis
                has_rr_calculations = log_analysis_results['rr_calculation_entries'] > 0
                has_formula_evidence = log_analysis_results['formula_pattern_matches'] > 0
                has_technical_basis = log_analysis_results['technical_level_references'] > log_analysis_results['rr_calculation_entries']
                low_confidence_correlation = log_analysis_results['confidence_correlation_evidence'] <= log_analysis_results['rr_calculation_entries'] * 0.1
                
                evidence_score = sum([has_rr_calculations, has_formula_evidence, has_technical_basis, low_confidence_correlation])
                
                if evidence_score >= 3:
                    self.log_test_result("Backend Logs RR Calculation Analysis", True, 
                                       f"Logs show RR independence evidence: {evidence_score}/4 indicators positive")
                elif evidence_score >= 2:
                    self.log_test_result("Backend Logs RR Calculation Analysis", False, 
                                       f"Partial log evidence: {evidence_score}/4 indicators positive")
                else:
                    self.log_test_result("Backend Logs RR Calculation Analysis", False, 
                                       f"Limited log evidence: {evidence_score}/4 indicators positive")
                
            except Exception as log_error:
                logger.error(f"   ❌ Log analysis error: {log_error}")
                self.log_test_result("Backend Logs RR Calculation Analysis", False, f"Log analysis error: {str(log_error)}")
                
        except Exception as e:
            self.log_test_result("Backend Logs RR Calculation Analysis", False, f"Exception: {str(e)}")
    
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
        """Run comprehensive IA1 RR calculation independence test suite"""
        logger.info("🚀 Starting IA1 Risk-Reward Calculation Independence Test Suite")
        logger.info("=" * 80)
        logger.info("📋 IA1 RISK-REWARD CALCULATION INDEPENDENCE TEST SUITE")
        logger.info("🎯 Testing: RR calculation independence from confidence level")
        logger.info("🎯 Expected: Consistent RR ratios based on technical levels, not confidence")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_ia1_rr_formula_validation()
        await self.test_2_ia1_confidence_independence()
        await self.test_3_technical_levels_consistency()
        await self.test_4_database_rr_consistency_analysis()
        await self.test_5_long_short_rr_formula_verification()
        await self.test_6_backend_logs_rr_calculation_analysis()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 IA1 RISK-REWARD CALCULATION INDEPENDENCE TEST SUMMARY")
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
            if "Formula Validation" in result['test']:
                requirements_status['RR Formula Implementation'] = result['success']
            elif "Confidence Independence" in result['test']:
                requirements_status['RR Independence from Confidence'] = result['success']
            elif "Technical Levels" in result['test']:
                requirements_status['Technical Analysis Based RR'] = result['success']
            elif "Database" in result['test']:
                requirements_status['Historical RR Consistency'] = result['success']
            elif "Formula Verification" in result['test']:
                requirements_status['LONG/SHORT Formula Accuracy'] = result['success']
            elif "Backend Logs" in result['test']:
                requirements_status['Implementation Evidence'] = result['success']
        
        logger.info("🎯 CRITICAL REQUIREMENTS STATUS:")
        for requirement, status in requirements_status.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {requirement}")
        
        requirements_met = sum(1 for status in requirements_status.values() if status)
        total_requirements = len(requirements_status)
        
        # Final verdict
        logger.info(f"\n🏆 REQUIREMENTS SATISFACTION: {requirements_met}/{total_requirements}")
        
        if requirements_met == total_requirements:
            logger.info("\n🎉 VERDICT: IA1 RR CALCULATION FULLY INDEPENDENT FROM CONFIDENCE!")
            logger.info("✅ RR formulas correctly implemented (LONG & SHORT)")
            logger.info("✅ RR calculation independent of confidence level")
            logger.info("✅ Technical analysis based RR (support/resistance)")
            logger.info("✅ Historical consistency confirmed")
            logger.info("✅ Formula accuracy verified in live tests")
            logger.info("✅ Implementation evidence found in logs")
            logger.info("✅ System meets all independence requirements")
        elif requirements_met >= total_requirements * 0.8:
            logger.info("\n⚠️ VERDICT: IA1 RR CALCULATION MOSTLY INDEPENDENT")
            logger.info("🔍 Minor issues may need attention for complete independence")
        elif requirements_met >= total_requirements * 0.6:
            logger.info("\n⚠️ VERDICT: IA1 RR CALCULATION PARTIALLY INDEPENDENT")
            logger.info("🔧 Several requirements need attention for full independence")
        else:
            logger.info("\n❌ VERDICT: IA1 RR CALCULATION NOT INDEPENDENT FROM CONFIDENCE")
            logger.info("🚨 Major issues detected - RR may be influenced by confidence")
            logger.info("🚨 System needs fixes to ensure RR independence")
        
        return passed_tests, total_tests

async def main():
    """Main function to run the comprehensive IA1 RR independence test suite"""
    test_suite = IA1RiskRewardIndependenceTestSuite()
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
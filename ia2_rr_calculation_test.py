#!/usr/bin/env python3
"""
IA2 RR Calculation Fix Testing Suite
Focus: Test the simplified IA2 RR calculation to match IA1 support/resistance formula

TESTING REQUIREMENTS FROM REVIEW REQUEST:
1. IA2 RR Calculation Validation: Verify IA2 decisions show valid "calculated_rr" values instead of null
2. RR Calculation Consistency: Check that IA2 uses same simple support/resistance formula as IA1  
3. Formula Verification: LONG: RR = (TP-Entry)/(Entry-SL), SHORT: RR = (Entry-TP)/(SL-Entry)
4. Fallback RR Elimination: Ensure IA2 no longer falls back to "1.00:1 (IA1 R:R unavailable)"
5. RR Reasoning Field: Verify "rr_reasoning" field shows simple S/R calculation details

APPROACH:
- Trigger IA1 analyses that should escalate to IA2
- Examine IA2 decision responses for RR calculation validation
- Test both LONG and SHORT signals to validate both formula variants
- Check for elimination of null calculated_rr values
- Verify rr_reasoning field contains proper calculation details
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IA2RRCalculationTestSuite:
    """Test suite for IA2 RR calculation fix validation"""
    
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
        logger.info(f"Testing IA2 RR Calculation Fix at: {self.api_url}")
        
        # Test results
        self.test_results = []
        self.ia2_decisions_analyzed = []
        
        # Symbols to test (focus on active ones that might trigger IA2)
        self.test_symbols = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT',
            'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT'
        ]
        
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
    
    async def test_1_get_recent_ia2_decisions(self):
        """Test 1: Get Recent IA2 Decisions for RR Analysis"""
        logger.info("\n🔍 TEST 1: Get Recent IA2 Decisions")
        
        try:
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            
            if response.status_code == 200:
                decisions = response.json()
                logger.info(f"   📊 Retrieved {len(decisions)} total decisions")
                
                # Filter for IA2 decisions (recent ones)
                ia2_decisions = []
                for decision in decisions:
                    if isinstance(decision, dict):
                        # Look for IA2 characteristics
                        has_ia2_fields = any(field in decision for field in [
                            'calculated_rr', 'rr_reasoning', 'technical_indicators_analysis',
                            'intelligent_tp_strategy', 'strategy_type'
                        ])
                        
                        # Check if it's a recent decision (within last 24 hours)
                        timestamp = decision.get('timestamp', '')
                        if timestamp:
                            try:
                                decision_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                if (datetime.now() - decision_time.replace(tzinfo=None)).days < 1:
                                    if has_ia2_fields or decision.get('confidence', 0) >= 0.7:
                                        ia2_decisions.append(decision)
                            except:
                                # If timestamp parsing fails, include it anyway
                                if has_ia2_fields:
                                    ia2_decisions.append(decision)
                
                self.ia2_decisions_analyzed = ia2_decisions
                logger.info(f"   📊 Found {len(ia2_decisions)} potential IA2 decisions for analysis")
                
                if len(ia2_decisions) > 0:
                    self.log_test_result("Get Recent IA2 Decisions", True, 
                                       f"Found {len(ia2_decisions)} IA2 decisions to analyze")
                else:
                    self.log_test_result("Get Recent IA2 Decisions", False, 
                                       "No recent IA2 decisions found for RR analysis")
            else:
                self.log_test_result("Get Recent IA2 Decisions", False, 
                                   f"Failed to retrieve decisions: HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test_result("Get Recent IA2 Decisions", False, f"Exception: {str(e)}")
    
    async def test_2_trigger_new_ia2_decisions(self):
        """Test 2: Trigger New IA1 Analyses to Generate IA2 Decisions"""
        logger.info("\n🔍 TEST 2: Trigger New IA1 Analyses for IA2 Escalation")
        
        try:
            # Trigger market scan to generate new opportunities
            logger.info("   🚀 Triggering market scan to generate opportunities...")
            scan_response = requests.post(f"{self.api_url}/scan", timeout=60)
            
            if scan_response.status_code in [200, 201]:
                scan_result = scan_response.json()
                opportunities_count = len(scan_result) if isinstance(scan_result, list) else scan_result.get('count', 0)
                logger.info(f"   📊 Market scan generated {opportunities_count} opportunities")
                
                # Wait a moment for IA1 analyses to process
                logger.info("   ⏳ Waiting for IA1 analyses to process...")
                await asyncio.sleep(10)
                
                # Trigger IA1 analyses on some symbols
                ia1_triggered = 0
                for symbol in self.test_symbols[:5]:  # Test first 5 symbols
                    try:
                        logger.info(f"   🎯 Triggering IA1 analysis for {symbol}")
                        ia1_response = requests.post(f"{self.api_url}/analyze/{symbol}", timeout=45)
                        
                        if ia1_response.status_code in [200, 201]:
                            ia1_result = ia1_response.json()
                            confidence = ia1_result.get('confidence', 0)
                            rr_ratio = ia1_result.get('risk_reward_ratio', 0)
                            
                            logger.info(f"      📊 {symbol} IA1: Confidence={confidence:.2f}, RR={rr_ratio:.2f}")
                            
                            # Check if this should escalate to IA2 (confidence >= 70% and RR >= 2.0)
                            if confidence >= 0.7 and rr_ratio >= 2.0:
                                logger.info(f"      🚀 {symbol} should escalate to IA2 (C={confidence:.2f}, RR={rr_ratio:.2f})")
                                ia1_triggered += 1
                            else:
                                logger.info(f"      ⏸️ {symbol} won't escalate (C={confidence:.2f}, RR={rr_ratio:.2f})")
                        
                        # Small delay between requests
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        logger.info(f"      ❌ {symbol} IA1 analysis failed: {str(e)}")
                
                # Wait for IA2 decisions to be generated
                if ia1_triggered > 0:
                    logger.info(f"   ⏳ Waiting for IA2 decisions to be generated from {ia1_triggered} eligible IA1 analyses...")
                    await asyncio.sleep(15)
                
                self.log_test_result("Trigger New IA2 Decisions", True, 
                                   f"Triggered {ia1_triggered} IA1 analyses that should escalate to IA2")
            else:
                self.log_test_result("Trigger New IA2 Decisions", False, 
                                   f"Market scan failed: HTTP {scan_response.status_code}")
                
        except Exception as e:
            self.log_test_result("Trigger New IA2 Decisions", False, f"Exception: {str(e)}")
    
    async def test_3_validate_calculated_rr_not_null(self):
        """Test 3: Validate IA2 Decisions Have Non-Null calculated_rr Values"""
        logger.info("\n🔍 TEST 3: Validate calculated_rr Values Are Not Null")
        
        try:
            # Get fresh IA2 decisions after triggering
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            
            if response.status_code == 200:
                decisions = response.json()
                
                # Find recent IA2 decisions
                recent_ia2_decisions = []
                for decision in decisions:
                    if isinstance(decision, dict):
                        # Look for IA2 characteristics and recent timestamp
                        has_calculated_rr = 'calculated_rr' in decision
                        has_rr_reasoning = 'rr_reasoning' in decision
                        has_ia2_fields = has_calculated_rr or has_rr_reasoning or 'technical_indicators_analysis' in decision
                        
                        if has_ia2_fields:
                            recent_ia2_decisions.append(decision)
                
                logger.info(f"   📊 Analyzing {len(recent_ia2_decisions)} IA2 decisions for calculated_rr validation")
                
                null_rr_count = 0
                valid_rr_count = 0
                missing_rr_count = 0
                
                for decision in recent_ia2_decisions:
                    symbol = decision.get('symbol', 'UNKNOWN')
                    calculated_rr = decision.get('calculated_rr')
                    
                    if calculated_rr is None:
                        null_rr_count += 1
                        logger.info(f"      ❌ {symbol}: calculated_rr is null")
                    elif 'calculated_rr' not in decision:
                        missing_rr_count += 1
                        logger.info(f"      ⚠️ {symbol}: calculated_rr field missing")
                    else:
                        valid_rr_count += 1
                        logger.info(f"      ✅ {symbol}: calculated_rr = {calculated_rr}")
                
                total_decisions = len(recent_ia2_decisions)
                
                if total_decisions > 0:
                    success_rate = valid_rr_count / total_decisions
                    
                    if null_rr_count == 0 and success_rate >= 0.8:
                        self.log_test_result("Validate calculated_rr Not Null", True, 
                                           f"All IA2 decisions have valid calculated_rr: {valid_rr_count}/{total_decisions}")
                    else:
                        self.log_test_result("Validate calculated_rr Not Null", False, 
                                           f"Found null/missing calculated_rr: {null_rr_count} null, {missing_rr_count} missing, {valid_rr_count} valid")
                else:
                    self.log_test_result("Validate calculated_rr Not Null", False, 
                                       "No IA2 decisions found to validate calculated_rr")
            else:
                self.log_test_result("Validate calculated_rr Not Null", False, 
                                   f"Failed to retrieve decisions: HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test_result("Validate calculated_rr Not Null", False, f"Exception: {str(e)}")
    
    async def test_4_validate_rr_formula_consistency(self):
        """Test 4: Validate RR Formula Consistency (LONG vs SHORT)"""
        logger.info("\n🔍 TEST 4: Validate RR Formula Consistency")
        
        try:
            # Get recent decisions
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            
            if response.status_code == 200:
                decisions = response.json()
                
                # Find IA2 decisions with trading signals
                trading_decisions = []
                for decision in decisions:
                    if isinstance(decision, dict):
                        signal = decision.get('signal', '').upper()
                        if signal in ['LONG', 'SHORT'] and 'calculated_rr' in decision:
                            trading_decisions.append(decision)
                
                logger.info(f"   📊 Analyzing {len(trading_decisions)} IA2 trading decisions for formula validation")
                
                long_decisions = [d for d in trading_decisions if d.get('signal', '').upper() == 'LONG']
                short_decisions = [d for d in trading_decisions if d.get('signal', '').upper() == 'SHORT']
                
                logger.info(f"   📊 Found {len(long_decisions)} LONG and {len(short_decisions)} SHORT decisions")
                
                formula_validation_results = []
                
                # Validate LONG formula: RR = (TP-Entry)/(Entry-SL)
                for decision in long_decisions:
                    symbol = decision.get('symbol', 'UNKNOWN')
                    calculated_rr = decision.get('calculated_rr')
                    entry_price = decision.get('entry_price')
                    stop_loss = decision.get('stop_loss')
                    take_profit = decision.get('take_profit_1')  # Use TP1 as primary TP
                    
                    if all(v is not None for v in [calculated_rr, entry_price, stop_loss, take_profit]):
                        # Calculate expected RR using LONG formula
                        expected_rr = (take_profit - entry_price) / (entry_price - stop_loss)
                        rr_diff = abs(calculated_rr - expected_rr)
                        
                        if rr_diff < 0.1:  # Allow small tolerance
                            formula_validation_results.append(f"✅ {symbol} LONG: RR={calculated_rr:.2f} matches formula ({expected_rr:.2f})")
                        else:
                            formula_validation_results.append(f"❌ {symbol} LONG: RR={calculated_rr:.2f} doesn't match formula ({expected_rr:.2f})")
                    else:
                        formula_validation_results.append(f"⚠️ {symbol} LONG: Missing price data for formula validation")
                
                # Validate SHORT formula: RR = (Entry-TP)/(SL-Entry)
                for decision in short_decisions:
                    symbol = decision.get('symbol', 'UNKNOWN')
                    calculated_rr = decision.get('calculated_rr')
                    entry_price = decision.get('entry_price')
                    stop_loss = decision.get('stop_loss')
                    take_profit = decision.get('take_profit_1')  # Use TP1 as primary TP
                    
                    if all(v is not None for v in [calculated_rr, entry_price, stop_loss, take_profit]):
                        # Calculate expected RR using SHORT formula
                        expected_rr = (entry_price - take_profit) / (stop_loss - entry_price)
                        rr_diff = abs(calculated_rr - expected_rr)
                        
                        if rr_diff < 0.1:  # Allow small tolerance
                            formula_validation_results.append(f"✅ {symbol} SHORT: RR={calculated_rr:.2f} matches formula ({expected_rr:.2f})")
                        else:
                            formula_validation_results.append(f"❌ {symbol} SHORT: RR={calculated_rr:.2f} doesn't match formula ({expected_rr:.2f})")
                    else:
                        formula_validation_results.append(f"⚠️ {symbol} SHORT: Missing price data for formula validation")
                
                # Log all validation results
                for result in formula_validation_results:
                    logger.info(f"      {result}")
                
                # Evaluate overall formula consistency
                successful_validations = len([r for r in formula_validation_results if r.startswith("✅")])
                total_validations = len([r for r in formula_validation_results if not r.startswith("⚠️")])
                
                if total_validations > 0:
                    success_rate = successful_validations / total_validations
                    
                    if success_rate >= 0.8:
                        self.log_test_result("Validate RR Formula Consistency", True, 
                                           f"RR formulas consistent: {successful_validations}/{total_validations} ({success_rate:.1%})")
                    else:
                        self.log_test_result("Validate RR Formula Consistency", False, 
                                           f"RR formula inconsistencies: {successful_validations}/{total_validations} ({success_rate:.1%})")
                else:
                    self.log_test_result("Validate RR Formula Consistency", False, 
                                       "No trading decisions with complete price data found for formula validation")
            else:
                self.log_test_result("Validate RR Formula Consistency", False, 
                                   f"Failed to retrieve decisions: HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test_result("Validate RR Formula Consistency", False, f"Exception: {str(e)}")
    
    async def test_5_validate_rr_reasoning_field(self):
        """Test 5: Validate rr_reasoning Field Contains Calculation Details"""
        logger.info("\n🔍 TEST 5: Validate rr_reasoning Field")
        
        try:
            # Get recent decisions
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            
            if response.status_code == 200:
                decisions = response.json()
                
                # Find IA2 decisions with rr_reasoning
                decisions_with_reasoning = []
                for decision in decisions:
                    if isinstance(decision, dict) and 'rr_reasoning' in decision:
                        decisions_with_reasoning.append(decision)
                
                logger.info(f"   📊 Analyzing {len(decisions_with_reasoning)} IA2 decisions for rr_reasoning validation")
                
                reasoning_validation_results = []
                
                for decision in decisions_with_reasoning:
                    symbol = decision.get('symbol', 'UNKNOWN')
                    rr_reasoning = decision.get('rr_reasoning', '')
                    signal = decision.get('signal', '').upper()
                    
                    # Check for key elements in reasoning
                    has_support_resistance = any(term in rr_reasoning.lower() for term in ['support', 'resistance'])
                    has_formula_reference = any(term in rr_reasoning.lower() for term in ['formula', 'calculation', 'rr ='])
                    has_price_levels = any(char in rr_reasoning for char in ['$', '0.', '1.', '2.', '3.', '4.', '5.'])
                    has_signal_reference = signal.lower() in rr_reasoning.lower() if signal else False
                    
                    quality_score = sum([has_support_resistance, has_formula_reference, has_price_levels, has_signal_reference])
                    
                    if quality_score >= 3:
                        reasoning_validation_results.append(f"✅ {symbol}: High-quality rr_reasoning (score: {quality_score}/4)")
                        logger.info(f"      ✅ {symbol}: '{rr_reasoning[:100]}...'")
                    elif quality_score >= 2:
                        reasoning_validation_results.append(f"⚠️ {symbol}: Adequate rr_reasoning (score: {quality_score}/4)")
                        logger.info(f"      ⚠️ {symbol}: '{rr_reasoning[:100]}...'")
                    else:
                        reasoning_validation_results.append(f"❌ {symbol}: Poor rr_reasoning (score: {quality_score}/4)")
                        logger.info(f"      ❌ {symbol}: '{rr_reasoning[:100]}...'")
                
                # Evaluate reasoning quality
                high_quality = len([r for r in reasoning_validation_results if r.startswith("✅")])
                adequate_quality = len([r for r in reasoning_validation_results if r.startswith("⚠️")])
                total_reasoning = len(reasoning_validation_results)
                
                if total_reasoning > 0:
                    success_rate = (high_quality + adequate_quality) / total_reasoning
                    
                    if success_rate >= 0.8:
                        self.log_test_result("Validate rr_reasoning Field", True, 
                                           f"Good rr_reasoning quality: {high_quality} high + {adequate_quality} adequate / {total_reasoning}")
                    else:
                        self.log_test_result("Validate rr_reasoning Field", False, 
                                           f"Poor rr_reasoning quality: {high_quality} high + {adequate_quality} adequate / {total_reasoning}")
                else:
                    self.log_test_result("Validate rr_reasoning Field", False, 
                                       "No IA2 decisions with rr_reasoning field found")
            else:
                self.log_test_result("Validate rr_reasoning Field", False, 
                                   f"Failed to retrieve decisions: HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test_result("Validate rr_reasoning Field", False, f"Exception: {str(e)}")
    
    async def test_6_check_fallback_elimination(self):
        """Test 6: Check for Elimination of Fallback RR Messages"""
        logger.info("\n🔍 TEST 6: Check for Elimination of Fallback RR Messages")
        
        try:
            # Get recent decisions
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            
            if response.status_code == 200:
                decisions = response.json()
                
                # Look for fallback patterns in IA2 decisions
                fallback_patterns = [
                    "1.00:1 (IA1 R:R unavailable)",
                    "IA1 R:R unavailable",
                    "fallback",
                    "default R:R",
                    "unavailable"
                ]
                
                decisions_with_fallback = []
                total_ia2_decisions = 0
                
                for decision in decisions:
                    if isinstance(decision, dict):
                        # Check if this looks like an IA2 decision
                        has_ia2_fields = any(field in decision for field in [
                            'calculated_rr', 'rr_reasoning', 'technical_indicators_analysis'
                        ])
                        
                        if has_ia2_fields:
                            total_ia2_decisions += 1
                            
                            # Check for fallback patterns in various fields
                            decision_text = json.dumps(decision).lower()
                            
                            found_fallbacks = []
                            for pattern in fallback_patterns:
                                if pattern.lower() in decision_text:
                                    found_fallbacks.append(pattern)
                            
                            if found_fallbacks:
                                symbol = decision.get('symbol', 'UNKNOWN')
                                decisions_with_fallback.append({
                                    'symbol': symbol,
                                    'fallbacks': found_fallbacks
                                })
                                logger.info(f"      ❌ {symbol}: Found fallback patterns: {found_fallbacks}")
                
                logger.info(f"   📊 Analyzed {total_ia2_decisions} IA2 decisions for fallback patterns")
                
                if len(decisions_with_fallback) == 0:
                    self.log_test_result("Check Fallback Elimination", True, 
                                       f"No fallback RR patterns found in {total_ia2_decisions} IA2 decisions")
                else:
                    fallback_symbols = [d['symbol'] for d in decisions_with_fallback]
                    self.log_test_result("Check Fallback Elimination", False, 
                                       f"Found fallback patterns in {len(decisions_with_fallback)} decisions: {fallback_symbols}")
            else:
                self.log_test_result("Check Fallback Elimination", False, 
                                   f"Failed to retrieve decisions: HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test_result("Check Fallback Elimination", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_tests(self):
        """Run all IA2 RR calculation tests"""
        logger.info("🚀 Starting IA2 RR Calculation Fix Test Suite")
        logger.info("=" * 80)
        logger.info("📋 IA2 RR CALCULATION FIX COMPREHENSIVE TESTING")
        logger.info("🎯 Testing: RR calculation validation, formula consistency, reasoning field, fallback elimination")
        logger.info("🎯 Expected: IA2 uses simple S/R formula like IA1, no null calculated_rr, proper reasoning")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_get_recent_ia2_decisions()
        await self.test_2_trigger_new_ia2_decisions()
        await self.test_3_validate_calculated_rr_not_null()
        await self.test_4_validate_rr_formula_consistency()
        await self.test_5_validate_rr_reasoning_field()
        await self.test_6_check_fallback_elimination()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 IA2 RR CALCULATION FIX TEST SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # System analysis
        logger.info("\n" + "=" * 80)
        logger.info("📋 IA2 RR CALCULATION FIX STATUS")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - IA2 RR Calculation Fix FULLY WORKING!")
            logger.info("✅ IA2 decisions show valid calculated_rr values (no null)")
            logger.info("✅ RR calculation uses same simple S/R formula as IA1")
            logger.info("✅ LONG and SHORT formulas working correctly")
            logger.info("✅ No fallback to '1.00:1 (IA1 R:R unavailable)'")
            logger.info("✅ rr_reasoning field shows S/R calculation details")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY WORKING - IA2 RR calculation mostly fixed with minor issues")
            logger.info("🔍 Some aspects may need fine-tuning for complete functionality")
        elif passed_tests >= total_tests * 0.6:
            logger.info("⚠️ PARTIALLY WORKING - Core RR calculation improvements working")
            logger.info("🔧 Some formula aspects may need additional implementation")
        else:
            logger.info("❌ NOT WORKING - Critical issues with IA2 RR calculation fix")
            logger.info("🚨 Major problems preventing proper RR calculation")
        
        # Specific requirements check
        logger.info("\n📝 IA2 RR CALCULATION FIX REQUIREMENTS VERIFICATION:")
        
        requirements_met = []
        requirements_failed = []
        
        # Check each requirement based on test results
        for result in self.test_results:
            if result['success']:
                if "calculated_rr Not Null" in result['test']:
                    requirements_met.append("✅ IA2 decisions show valid calculated_rr values (not null)")
                elif "RR Formula Consistency" in result['test']:
                    requirements_met.append("✅ RR calculation uses same simple S/R formula as IA1")
                elif "rr_reasoning Field" in result['test']:
                    requirements_met.append("✅ rr_reasoning field shows S/R calculation details")
                elif "Fallback Elimination" in result['test']:
                    requirements_met.append("✅ No fallback to '1.00:1 (IA1 R:R unavailable)'")
            else:
                if "calculated_rr Not Null" in result['test']:
                    requirements_failed.append("❌ IA2 decisions still have null calculated_rr values")
                elif "RR Formula Consistency" in result['test']:
                    requirements_failed.append("❌ RR calculation formula inconsistent with IA1")
                elif "rr_reasoning Field" in result['test']:
                    requirements_failed.append("❌ rr_reasoning field lacks proper calculation details")
                elif "Fallback Elimination" in result['test']:
                    requirements_failed.append("❌ Still using fallback RR calculations")
        
        for req in requirements_met:
            logger.info(f"   {req}")
        
        for req in requirements_failed:
            logger.info(f"   {req}")
        
        logger.info(f"\n🏆 FINAL RESULT: {len(requirements_met)}/{len(requirements_met) + len(requirements_failed)} requirements satisfied")
        
        # Final verdict
        if len(requirements_failed) == 0:
            logger.info("\n🎉 VERDICT: IA2 RR Calculation Fix is FULLY WORKING!")
            logger.info("✅ All RR calculation requirements implemented and working correctly")
            logger.info("✅ IA2 now uses simple S/R formula like IA1 with proper reasoning")
            logger.info("✅ No more null calculated_rr or fallback calculations")
        elif len(requirements_failed) <= 1:
            logger.info("\n⚠️ VERDICT: IA2 RR Calculation Fix is MOSTLY WORKING")
            logger.info("🔍 Minor issues may need attention for complete functionality")
        else:
            logger.info("\n❌ VERDICT: IA2 RR Calculation Fix is NOT WORKING")
            logger.info("🚨 Major implementation gaps preventing proper RR calculation")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = IA2RRCalculationTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
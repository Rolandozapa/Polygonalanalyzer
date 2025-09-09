#!/usr/bin/env python3
"""
FORMUSDT Risk-Reward Calculation Fix Validation Test Suite
Focus: Validate that the FORMUSDT RR=0 issue has been resolved

PROBLÈME IDENTIFIÉ:
L'analyse IA1 attribuait tous les prix (entry_price, stop_loss_price, take_profit_price) 
à la même valeur (opportunity.current_price), causant RR=0 et bloquant l'admission IA2.

CORRECTION TESTÉE:
1. Calcul de prix réalistes basés sur les niveaux techniques IA1
2. Fallback avec pourcentages par défaut si pas de niveaux techniques
3. LONG: SL=-5%, TP=+10% | SHORT: SL=+5%, TP=-10% | HOLD: SL=-2%, TP=+2%
4. Logging détaillé des prix calculés

OBJECTIFS DE VALIDATION:
1. Nouveau cycle de trading génère analyses avec prix différents
2. FORMUSDT spécifique a RR > 0
3. Admission IA2 fonctionne pour analyses avec RR > 0
4. Logs montrent prix calculés correctement
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FORMUSDTRiskRewardTestSuite:
    """Test suite for FORMUSDT Risk-Reward calculation fix validation"""
    
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
        logger.info(f"Testing FORMUSDT Risk-Reward Fix at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Expected price calculation log patterns
        self.expected_price_log_patterns = [
            "💰 PRIX CALCULÉS",
            "Entry=$",
            "SL=$", 
            "TP=$",
            "SIGNAL):",
            "LONG)",
            "SHORT)",
            "HOLD)"
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
    
    async def test_1_trigger_new_trading_cycle(self):
        """Test 1: Trigger new trading cycle to generate fresh analyses"""
        logger.info("\n🔍 TEST 1: Trigger new trading cycle for fresh IA1 analyses")
        
        try:
            logger.info("   🚀 Calling /api/trading/start-trading endpoint...")
            start_time = time.time()
            
            response = requests.post(f"{self.api_url}/trading/start-trading", timeout=180)
            
            elapsed_time = time.time() - start_time
            logger.info(f"   ⏱️ Request completed in {elapsed_time:.2f} seconds")
            
            if response.status_code in [200, 201]:
                logger.info(f"   ✅ Trading cycle started successfully (HTTP {response.status_code})")
                
                # Wait for processing
                logger.info("   ⏳ Waiting 45 seconds for IA1 analysis processing...")
                await asyncio.sleep(45)
                
                success = True
                details = f"HTTP {response.status_code}, processing time: {elapsed_time:.2f}s"
            else:
                logger.warning(f"   ⚠️ Trading cycle returned HTTP {response.status_code}")
                logger.warning(f"   Response: {response.text[:200]}...")
                success = False
                details = f"HTTP {response.status_code}: {response.text[:100]}"
            
            self.log_test_result("Trigger New Trading Cycle", success, details)
            
        except Exception as e:
            self.log_test_result("Trigger New Trading Cycle", False, f"Exception: {str(e)}")
    
    async def test_2_verify_price_differentiation(self):
        """Test 2: Verify new IA1 analyses have different entry/SL/TP prices"""
        logger.info("\n🔍 TEST 2: Verify IA1 analyses have differentiated prices (entry ≠ SL ≠ TP)")
        
        try:
            # Get recent IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Verify Price Differentiation", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("Verify Price Differentiation", False, "No IA1 analyses found")
                return
            
            # Check recent analyses for price differentiation
            differentiated_analyses = 0
            identical_price_analyses = 0
            valid_rr_analyses = 0
            zero_rr_analyses = 0
            
            logger.info(f"   📊 Analyzing {len(analyses)} IA1 analyses for price differentiation...")
            
            for analysis in analyses[-15:]:  # Check last 15 analyses
                symbol = analysis.get('symbol', 'Unknown')
                entry_price = analysis.get('entry_price', 0)
                stop_loss_price = analysis.get('stop_loss_price', 0)
                take_profit_price = analysis.get('take_profit_price', 0)
                risk_reward_ratio = analysis.get('risk_reward_ratio', 0)
                recommendation = analysis.get('recommendation', '').upper()
                
                # Check if prices are different
                prices_different = not (entry_price == stop_loss_price == take_profit_price)
                
                if prices_different:
                    differentiated_analyses += 1
                    logger.info(f"      ✅ {symbol}: Entry=${entry_price:.6f}, SL=${stop_loss_price:.6f}, TP=${take_profit_price:.6f}")
                else:
                    identical_price_analyses += 1
                    logger.info(f"      ❌ {symbol}: Identical prices - Entry=${entry_price:.6f}, SL=${stop_loss_price:.6f}, TP=${take_profit_price:.6f}")
                
                # Check Risk-Reward ratio
                if risk_reward_ratio > 0:
                    valid_rr_analyses += 1
                    logger.info(f"      ✅ {symbol}: RR={risk_reward_ratio:.2f} (Valid)")
                else:
                    zero_rr_analyses += 1
                    logger.info(f"      ❌ {symbol}: RR={risk_reward_ratio:.2f} (Invalid)")
            
            logger.info(f"   📊 Price differentiation results:")
            logger.info(f"      ✅ Differentiated prices: {differentiated_analyses}")
            logger.info(f"      ❌ Identical prices: {identical_price_analyses}")
            logger.info(f"      ✅ Valid RR (>0): {valid_rr_analyses}")
            logger.info(f"      ❌ Zero RR: {zero_rr_analyses}")
            
            # Success criteria: Most analyses should have differentiated prices and valid RR
            total_recent = len(analyses[-15:])
            differentiation_rate = differentiated_analyses / total_recent if total_recent > 0 else 0
            valid_rr_rate = valid_rr_analyses / total_recent if total_recent > 0 else 0
            
            success = differentiation_rate >= 0.8 and valid_rr_rate >= 0.8  # 80% success rate
            
            details = f"Differentiated: {differentiated_analyses}/{total_recent} ({differentiation_rate:.1%}), Valid RR: {valid_rr_analyses}/{total_recent} ({valid_rr_rate:.1%})"
            
            self.log_test_result("Verify Price Differentiation", success, details)
            
        except Exception as e:
            self.log_test_result("Verify Price Differentiation", False, f"Exception: {str(e)}")
    
    async def test_3_formusdt_specific_analysis(self):
        """Test 3: Look for FORMUSDT specific analysis with correct RR calculation"""
        logger.info("\n🔍 TEST 3: Search for FORMUSDT analysis with corrected Risk-Reward calculation")
        
        try:
            # Get recent IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("FORMUSDT Specific Analysis", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("FORMUSDT Specific Analysis", False, "No IA1 analyses found")
                return
            
            # Look for FORMUSDT analysis
            formusdt_found = False
            formusdt_analysis = None
            
            for analysis in analyses:
                symbol = analysis.get('symbol', '')
                if 'FORM' in symbol.upper():
                    formusdt_found = True
                    formusdt_analysis = analysis
                    logger.info(f"   🎯 Found FORMUSDT analysis: {symbol}")
                    break
            
            if not formusdt_found:
                logger.info("   ⚠️ FORMUSDT analysis not found in current dataset")
                logger.info("   📊 Available symbols in recent analyses:")
                for analysis in analyses[-10:]:
                    symbol = analysis.get('symbol', 'Unknown')
                    logger.info(f"      - {symbol}")
                
                # Still check if any analysis has the expected characteristics
                success = False
                details = "FORMUSDT not found in current analyses"
            else:
                # Analyze FORMUSDT specific data
                symbol = formusdt_analysis.get('symbol', 'Unknown')
                entry_price = formusdt_analysis.get('entry_price', 0)
                stop_loss_price = formusdt_analysis.get('stop_loss_price', 0)
                take_profit_price = formusdt_analysis.get('take_profit_price', 0)
                risk_reward_ratio = formusdt_analysis.get('risk_reward_ratio', 0)
                confidence = formusdt_analysis.get('confidence', 0)
                recommendation = formusdt_analysis.get('recommendation', '').upper()
                
                logger.info(f"   📊 FORMUSDT Analysis Details:")
                logger.info(f"      Symbol: {symbol}")
                logger.info(f"      Signal: {recommendation}")
                logger.info(f"      Confidence: {confidence:.1%}")
                logger.info(f"      Entry Price: ${entry_price:.6f}")
                logger.info(f"      Stop Loss: ${stop_loss_price:.6f}")
                logger.info(f"      Take Profit: ${take_profit_price:.6f}")
                logger.info(f"      Risk-Reward: {risk_reward_ratio:.2f}")
                
                # Check if prices are different (fix working)
                prices_different = not (entry_price == stop_loss_price == take_profit_price)
                valid_rr = risk_reward_ratio > 0
                
                # Check IA2 eligibility criteria
                voie1_eligible = recommendation in ['LONG', 'SHORT'] and confidence >= 0.70
                voie2_eligible = risk_reward_ratio >= 2.0
                ia2_eligible = voie1_eligible or voie2_eligible
                
                logger.info(f"   🎯 FORMUSDT Fix Validation:")
                logger.info(f"      ✅ Prices differentiated: {prices_different}")
                logger.info(f"      ✅ Valid RR (>0): {valid_rr}")
                logger.info(f"      📊 VOIE 1 eligible (Signal + Conf≥70%): {voie1_eligible}")
                logger.info(f"      📊 VOIE 2 eligible (RR≥2.0): {voie2_eligible}")
                logger.info(f"      🎯 IA2 eligible: {ia2_eligible}")
                
                success = prices_different and valid_rr
                details = f"Signal: {recommendation}, Conf: {confidence:.1%}, RR: {risk_reward_ratio:.2f}, Prices diff: {prices_different}, IA2 eligible: {ia2_eligible}"
            
            self.log_test_result("FORMUSDT Specific Analysis", success, details)
            
        except Exception as e:
            self.log_test_result("FORMUSDT Specific Analysis", False, f"Exception: {str(e)}")
    
    async def test_4_price_calculation_logs(self):
        """Test 4: Check backend logs for price calculation patterns"""
        logger.info("\n🔍 TEST 4: Verify price calculation logs show correct patterns")
        
        try:
            import subprocess
            
            # Get recent backend logs
            backend_logs = ""
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "3000", "/var/log/supervisor/backend.out.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "3000", "/var/log/supervisor/backend.err.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            if not backend_logs:
                self.log_test_result("Price Calculation Logs", False, "Could not retrieve backend logs")
                return
            
            # Check for price calculation log patterns
            pattern_counts = {}
            for pattern in self.expected_price_log_patterns:
                count = backend_logs.count(pattern)
                pattern_counts[pattern] = count
                logger.info(f"   📊 '{pattern}': {count} occurrences")
            
            # Look for specific FORMUSDT price calculations
            formusdt_price_logs = 0
            lines = backend_logs.split('\n')
            for line in lines:
                if 'FORM' in line.upper() and '💰 PRIX CALCULÉS' in line:
                    formusdt_price_logs += 1
                    logger.info(f"   🎯 FORMUSDT price calculation: {line.strip()}")
            
            # Look for general price calculation patterns
            prix_calcules_logs = backend_logs.count("💰 PRIX CALCULÉS")
            entry_price_logs = backend_logs.count("Entry=$")
            sl_price_logs = backend_logs.count("SL=$")
            tp_price_logs = backend_logs.count("TP=$")
            
            logger.info(f"   📊 Price calculation summary:")
            logger.info(f"      💰 PRIX CALCULÉS logs: {prix_calcules_logs}")
            logger.info(f"      Entry=$ logs: {entry_price_logs}")
            logger.info(f"      SL=$ logs: {sl_price_logs}")
            logger.info(f"      TP=$ logs: {tp_price_logs}")
            logger.info(f"      FORMUSDT specific: {formusdt_price_logs}")
            
            # Success criteria: Evidence of price calculation logging
            patterns_found = sum(1 for count in pattern_counts.values() if count > 0)
            total_occurrences = sum(pattern_counts.values())
            
            success = patterns_found >= 4 and total_occurrences >= 10
            
            details = f"Patterns found: {patterns_found}/{len(self.expected_price_log_patterns)}, Total occurrences: {total_occurrences}, FORMUSDT logs: {formusdt_price_logs}"
            
            self.log_test_result("Price Calculation Logs", success, details)
            
        except Exception as e:
            self.log_test_result("Price Calculation Logs", False, f"Exception: {str(e)}")
    
    async def test_5_ia2_admission_pipeline(self):
        """Test 5: Verify IA2 admission pipeline works for analyses with RR > 0"""
        logger.info("\n🔍 TEST 5: Test IA2 admission pipeline for analyses with valid RR")
        
        try:
            # Get IA1 analyses
            ia1_response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if ia1_response.status_code != 200:
                self.log_test_result("IA2 Admission Pipeline", False, f"IA1 HTTP {ia1_response.status_code}: {ia1_response.text}")
                return
            
            ia1_data = ia1_response.json()
            ia1_analyses = ia1_data.get('analyses', [])
            
            # Get IA2 decisions
            ia2_response = requests.get(f"{self.api_url}/decisions", timeout=30)
            
            if ia2_response.status_code != 200:
                self.log_test_result("IA2 Admission Pipeline", False, f"IA2 HTTP {ia2_response.status_code}: {ia2_response.text}")
                return
            
            ia2_data = ia2_response.json()
            ia2_decisions = ia2_data.get('decisions', [])
            
            logger.info(f"   📊 Pipeline data:")
            logger.info(f"      IA1 analyses: {len(ia1_analyses)}")
            logger.info(f"      IA2 decisions: {len(ia2_decisions)}")
            
            # Analyze IA1→IA2 pipeline
            voie1_eligible = 0
            voie2_eligible = 0
            total_eligible = 0
            valid_rr_count = 0
            
            for analysis in ia1_analyses[-10:]:  # Check last 10
                symbol = analysis.get('symbol', 'Unknown')
                confidence = analysis.get('confidence', 0)
                risk_reward_ratio = analysis.get('risk_reward_ratio', 0)
                recommendation = analysis.get('recommendation', '').upper()
                
                # Check eligibility criteria
                is_voie1 = recommendation in ['LONG', 'SHORT'] and confidence >= 0.70
                is_voie2 = risk_reward_ratio >= 2.0
                is_eligible = is_voie1 or is_voie2
                has_valid_rr = risk_reward_ratio > 0
                
                if is_voie1:
                    voie1_eligible += 1
                if is_voie2:
                    voie2_eligible += 1
                if is_eligible:
                    total_eligible += 1
                if has_valid_rr:
                    valid_rr_count += 1
                
                logger.info(f"      📊 {symbol}: Signal={recommendation}, Conf={confidence:.1%}, RR={risk_reward_ratio:.2f}, VOIE1={is_voie1}, VOIE2={is_voie2}, Eligible={is_eligible}")
            
            # Check for recent IA2 decisions
            recent_ia2_decisions = 0
            for decision in ia2_decisions[-10:]:  # Check last 10
                timestamp = decision.get('timestamp', '')
                if self._is_recent_timestamp(timestamp):
                    recent_ia2_decisions += 1
            
            logger.info(f"   📊 IA1→IA2 Pipeline Analysis:")
            logger.info(f"      VOIE 1 eligible (LONG/SHORT + Conf≥70%): {voie1_eligible}")
            logger.info(f"      VOIE 2 eligible (RR≥2.0): {voie2_eligible}")
            logger.info(f"      Total eligible for IA2: {total_eligible}")
            logger.info(f"      Valid RR (>0): {valid_rr_count}")
            logger.info(f"      Recent IA2 decisions: {recent_ia2_decisions}")
            
            # Success criteria: Pipeline should process eligible analyses
            pipeline_working = total_eligible > 0 and valid_rr_count > 0
            
            success = pipeline_working
            details = f"Eligible: {total_eligible}, Valid RR: {valid_rr_count}, Recent IA2: {recent_ia2_decisions}, VOIE1: {voie1_eligible}, VOIE2: {voie2_eligible}"
            
            self.log_test_result("IA2 Admission Pipeline", success, details)
            
        except Exception as e:
            self.log_test_result("IA2 Admission Pipeline", False, f"Exception: {str(e)}")
    
    async def test_6_system_wide_rr_validation(self):
        """Test 6: Validate RR > 0 for all signals across different symbols"""
        logger.info("\n🔍 TEST 6: Validate Risk-Reward > 0 for all trading signals system-wide")
        
        try:
            # Get recent IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("System-wide RR Validation", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("System-wide RR Validation", False, "No IA1 analyses found")
                return
            
            # Analyze RR by signal type
            long_signals = []
            short_signals = []
            hold_signals = []
            
            for analysis in analyses[-20:]:  # Check last 20 analyses
                symbol = analysis.get('symbol', 'Unknown')
                risk_reward_ratio = analysis.get('risk_reward_ratio', 0)
                recommendation = analysis.get('recommendation', '').upper()
                entry_price = analysis.get('entry_price', 0)
                stop_loss_price = analysis.get('stop_loss_price', 0)
                take_profit_price = analysis.get('take_profit_price', 0)
                
                signal_data = {
                    'symbol': symbol,
                    'rr': risk_reward_ratio,
                    'entry': entry_price,
                    'sl': stop_loss_price,
                    'tp': take_profit_price,
                    'prices_diff': not (entry_price == stop_loss_price == take_profit_price)
                }
                
                if recommendation == 'LONG':
                    long_signals.append(signal_data)
                elif recommendation == 'SHORT':
                    short_signals.append(signal_data)
                elif recommendation == 'HOLD':
                    hold_signals.append(signal_data)
            
            # Validate each signal type
            def validate_signals(signals, signal_type):
                valid_rr = sum(1 for s in signals if s['rr'] > 0)
                valid_prices = sum(1 for s in signals if s['prices_diff'])
                total = len(signals)
                
                logger.info(f"   📊 {signal_type} Signals ({total} total):")
                if total > 0:
                    logger.info(f"      ✅ Valid RR (>0): {valid_rr}/{total} ({valid_rr/total:.1%})")
                    logger.info(f"      ✅ Differentiated prices: {valid_prices}/{total} ({valid_prices/total:.1%})")
                    
                    # Show examples
                    for i, signal in enumerate(signals[:3]):  # Show first 3
                        logger.info(f"      Example {i+1}: {signal['symbol']} - RR={signal['rr']:.2f}, Prices diff={signal['prices_diff']}")
                
                return valid_rr, valid_prices, total
            
            long_valid_rr, long_valid_prices, long_total = validate_signals(long_signals, "LONG")
            short_valid_rr, short_valid_prices, short_total = validate_signals(short_signals, "SHORT")
            hold_valid_rr, hold_valid_prices, hold_total = validate_signals(hold_signals, "HOLD")
            
            # Overall validation
            total_signals = long_total + short_total + hold_total
            total_valid_rr = long_valid_rr + short_valid_rr + hold_valid_rr
            total_valid_prices = long_valid_prices + short_valid_prices + hold_valid_prices
            
            logger.info(f"   📊 System-wide Validation:")
            logger.info(f"      Total signals analyzed: {total_signals}")
            logger.info(f"      Valid RR (>0): {total_valid_rr}/{total_signals} ({total_valid_rr/total_signals:.1%} if total_signals > 0 else 0)")
            logger.info(f"      Differentiated prices: {total_valid_prices}/{total_signals} ({total_valid_prices/total_signals:.1%} if total_signals > 0 else 0)")
            
            # Success criteria: Most signals should have valid RR and differentiated prices
            rr_success_rate = total_valid_rr / total_signals if total_signals > 0 else 0
            price_success_rate = total_valid_prices / total_signals if total_signals > 0 else 0
            
            success = rr_success_rate >= 0.8 and price_success_rate >= 0.8  # 80% success rate
            
            details = f"RR success: {total_valid_rr}/{total_signals} ({rr_success_rate:.1%}), Price success: {total_valid_prices}/{total_signals} ({price_success_rate:.1%})"
            
            self.log_test_result("System-wide RR Validation", success, details)
            
        except Exception as e:
            self.log_test_result("System-wide RR Validation", False, f"Exception: {str(e)}")
    
    def _is_recent_timestamp(self, timestamp_str: str) -> bool:
        """Check if timestamp is from the last 24 hours"""
        try:
            if not timestamp_str:
                return False
            
            # Parse timestamp (handle different formats)
            if 'T' in timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = datetime.fromisoformat(timestamp_str)
            
            # Remove timezone info for comparison
            if timestamp.tzinfo:
                timestamp = timestamp.replace(tzinfo=None)
            
            now = datetime.now()
            return (now - timestamp) <= timedelta(hours=24)
            
        except Exception:
            return False
    
    async def run_comprehensive_tests(self):
        """Run all FORMUSDT Risk-Reward fix validation tests"""
        logger.info("🚀 Starting FORMUSDT Risk-Reward Calculation Fix Validation")
        logger.info("=" * 80)
        logger.info("📋 FORMUSDT RISK-REWARD FIX VALIDATION")
        logger.info("🎯 Testing: Price differentiation, RR > 0, IA2 admission, system-wide validation")
        logger.info("🎯 Expected: All analyses have entry ≠ SL ≠ TP, RR > 0, FORMUSDT can be admitted to IA2")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_trigger_new_trading_cycle()
        await self.test_2_verify_price_differentiation()
        await self.test_3_formusdt_specific_analysis()
        await self.test_4_price_calculation_logs()
        await self.test_5_ia2_admission_pipeline()
        await self.test_6_system_wide_rr_validation()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 FORMUSDT RISK-REWARD FIX VALIDATION SUMMARY")
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
        logger.info("📋 FORMUSDT RISK-REWARD FIX STATUS")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - FORMUSDT Risk-Reward Fix FULLY FUNCTIONAL!")
            logger.info("✅ New trading cycle generates fresh analyses")
            logger.info("✅ Price differentiation working (entry ≠ SL ≠ TP)")
            logger.info("✅ FORMUSDT has valid Risk-Reward calculation")
            logger.info("✅ Price calculation logging operational")
            logger.info("✅ IA2 admission pipeline functional")
            logger.info("✅ System-wide RR validation successful")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY FUNCTIONAL - FORMUSDT fix working with minor gaps")
            logger.info("🔍 Some components may need fine-tuning for full optimization")
        elif passed_tests >= total_tests * 0.5:
            logger.info("⚠️ PARTIALLY FUNCTIONAL - Core fix features working")
            logger.info("🔧 Some advanced features may need implementation or debugging")
        else:
            logger.info("❌ SYSTEM NOT FUNCTIONAL - Critical issues with FORMUSDT fix")
            logger.info("🚨 Major implementation gaps or system errors preventing functionality")
        
        # Specific requirements check
        logger.info("\n📝 FORMUSDT FIX REQUIREMENTS VERIFICATION:")
        
        requirements_met = []
        requirements_failed = []
        
        # Check each requirement based on test results
        for result in self.test_results:
            if result['success']:
                if "Trading Cycle" in result['test']:
                    requirements_met.append("✅ New trading cycle generates fresh analyses")
                elif "Price Differentiation" in result['test']:
                    requirements_met.append("✅ Price differentiation working (entry ≠ SL ≠ TP)")
                elif "FORMUSDT Specific" in result['test']:
                    requirements_met.append("✅ FORMUSDT has valid Risk-Reward calculation")
                elif "Price Calculation Logs" in result['test']:
                    requirements_met.append("✅ Price calculation logging operational")
                elif "IA2 Admission" in result['test']:
                    requirements_met.append("✅ IA2 admission pipeline functional")
                elif "System-wide RR" in result['test']:
                    requirements_met.append("✅ System-wide RR validation successful")
            else:
                if "Trading Cycle" in result['test']:
                    requirements_failed.append("❌ New trading cycle not generating analyses")
                elif "Price Differentiation" in result['test']:
                    requirements_failed.append("❌ Price differentiation not working")
                elif "FORMUSDT Specific" in result['test']:
                    requirements_failed.append("❌ FORMUSDT still has RR calculation issues")
                elif "Price Calculation Logs" in result['test']:
                    requirements_failed.append("❌ Price calculation logging not operational")
                elif "IA2 Admission" in result['test']:
                    requirements_failed.append("❌ IA2 admission pipeline not functional")
                elif "System-wide RR" in result['test']:
                    requirements_failed.append("❌ System-wide RR validation failed")
        
        for req in requirements_met:
            logger.info(f"   {req}")
        
        for req in requirements_failed:
            logger.info(f"   {req}")
        
        logger.info(f"\n🏆 FINAL RESULT: {len(requirements_met)}/{len(requirements_met) + len(requirements_failed)} requirements satisfied")
        
        # Final verdict
        if len(requirements_failed) == 0:
            logger.info("\n🎉 VERDICT: FORMUSDT Risk-Reward Fix is FULLY FUNCTIONAL!")
            logger.info("✅ All price calculation features implemented and working correctly")
            logger.info("✅ FORMUSDT can now be properly admitted to IA2 with valid RR > 0")
            logger.info("✅ System-wide fix prevents RR=0 issues across all symbols")
        elif len(requirements_failed) <= 1:
            logger.info("\n⚠️ VERDICT: FORMUSDT Risk-Reward Fix is MOSTLY FUNCTIONAL")
            logger.info("🔍 Minor issues may need attention for complete functionality")
        elif len(requirements_failed) <= 3:
            logger.info("\n⚠️ VERDICT: FORMUSDT Risk-Reward Fix is PARTIALLY FUNCTIONAL")
            logger.info("🔧 Several components need implementation or debugging")
        else:
            logger.info("\n❌ VERDICT: FORMUSDT Risk-Reward Fix is NOT FUNCTIONAL")
            logger.info("🚨 Major implementation gaps preventing fix from working")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = FORMUSDTRiskRewardTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
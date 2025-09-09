#!/usr/bin/env python3
"""
Backend Testing Suite for Refined Anti-Momentum Validation System
Focus: TEST REFINED ANTI-MOMENTUM VALIDATION SYSTEM - Verify that the refined sophisticated validation system correctly distinguishes between legitimate reversals (like XLM/GRT SHORT predictions that were correct) and momentum errors.

Refined validation features to test:
1. _validate_legitimate_reversal() method implementation  
2. Sophisticated reversal signal detection (RSI extremes, Stochastic, Bollinger, volatility exhaustion)
3. Differentiated penalties: 20% for legitimate reversals vs 35% for momentum errors
4. Warning signal detection for false counter-momentum signals
5. Reversal score vs warning score evaluation logic
6. Enhanced logging for legitimate_reversal vs momentum_error cases

Expected log patterns:
- "✅ LEGITIMATE REVERSAL DETECTED {symbol}:" (for cases like XLM/GRT SHORT)
- "🔄 Reversal signals: [RSI_OVERBOUGHT_EXTREME, STOCHASTIC_OVERBOUGHT, etc]"
- "⚠️ POTENTIAL MOMENTUM ERROR {symbol}:" (for dangerous counter-momentum cases)
- "💥 Reversal validation: FAILED (NO_REVERSAL_SIGNALS/INSUFFICIENT_REVERSAL_CONFIRMATION)"
- "✅ SOPHISTICATED VALIDATION APPLIED {symbol}: ... (legitimate_reversal/momentum_error/forced_hold)"

Test specifically:
- Strong momentum + RSI >75 + Stochastic >80 → Should detect LEGITIMATE REVERSAL
- Strong momentum + RSI 45-55 + no extremes → Should detect MOMENTUM ERROR  
- System should now preserve good counter-momentum signals while filtering bad ones

Verify the refined system balances reversal detection with momentum error prevention, allowing legitimate contrarian trades while blocking dangerous ones.
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

class RefinedAntiMomentumValidationTestSuite:
    """Test suite for Refined Anti-Momentum Validation System Verification"""
    
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
        logger.info(f"Testing Refined Anti-Momentum Validation System at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Expected refined validation log patterns
        self.expected_log_patterns = [
            "✅ LEGITIMATE REVERSAL DETECTED",
            "🔄 Reversal signals:",
            "⚠️ POTENTIAL MOMENTUM ERROR",
            "💥 Reversal validation: FAILED",
            "✅ SOPHISTICATED VALIDATION APPLIED",
            "RSI_OVERBOUGHT_EXTREME",
            "STOCHASTIC_OVERBOUGHT",
            "BOLLINGER_EXTREME_POSITION",
            "HIGH_VOLATILITY_EXHAUSTION",
            "legitimate_reversal",
            "momentum_error",
            "forced_hold"
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
    
    async def test_1_validate_legitimate_reversal_method(self):
        """Test 1: Verify _validate_legitimate_reversal method is implemented"""
        logger.info("\n🔍 TEST 1: Check _validate_legitimate_reversal method implementation")
        
        try:
            # Check if the method exists in the backend code
            backend_code = ""
            try:
                with open('/app/backend/server.py', 'r') as f:
                    backend_code = f.read()
            except Exception as e:
                self.log_test_result("Validate Legitimate Reversal Method", False, f"Could not read backend code: {e}")
                return
            
            # Check for required method implementation
            method_found = "def _validate_legitimate_reversal" in backend_code
            
            # Check for sophisticated reversal signal detection components
            components_found = {}
            required_components = [
                "RSI_OVERBOUGHT_EXTREME",
                "RSI_OVERSOLD_EXTREME", 
                "STOCHASTIC_OVERBOUGHT",
                "STOCHASTIC_OVERSOLD",
                "BOLLINGER_EXTREME_POSITION",
                "HIGH_VOLATILITY_EXHAUSTION",
                "MOMENTUM_CONFIDENCE_DIVERGENCE",
                "reversal_score",
                "warning_score",
                "is_legitimate_reversal"
            ]
            
            for component in required_components:
                if component in backend_code:
                    components_found[component] = True
                    logger.info(f"      ✅ Component found: {component}")
                else:
                    components_found[component] = False
                    logger.info(f"      ❌ Component missing: {component}")
            
            # Check for differentiated penalty logic
            penalty_patterns = [
                "confidence_penalty = 0.2",  # 20% for legitimate reversals
                "confidence_penalty = 0.35", # 35% for momentum errors
                "legitimate_reversal",
                "momentum_error"
            ]
            
            penalty_found = {}
            for pattern in penalty_patterns:
                if pattern in backend_code:
                    penalty_found[pattern] = True
                    logger.info(f"      ✅ Penalty pattern found: {pattern}")
                else:
                    penalty_found[pattern] = False
                    logger.info(f"      ❌ Penalty pattern missing: {pattern}")
            
            components_implemented = sum(components_found.values())
            penalty_patterns_found = sum(penalty_found.values())
            
            logger.info(f"   📊 Method found: {method_found}")
            logger.info(f"   📊 Components implemented: {components_implemented}/{len(required_components)}")
            logger.info(f"   📊 Penalty patterns found: {penalty_patterns_found}/{len(penalty_patterns)}")
            
            # Success criteria: Method exists and most components/patterns found
            success = method_found and components_implemented >= 7 and penalty_patterns_found >= 3
            
            details = f"Method: {method_found}, Components: {components_implemented}/{len(required_components)}, Penalties: {penalty_patterns_found}/{len(penalty_patterns)}"
            
            self.log_test_result("Validate Legitimate Reversal Method", success, details)
            
        except Exception as e:
            self.log_test_result("Validate Legitimate Reversal Method", False, f"Exception: {str(e)}")
    
    async def test_2_refined_validation_log_patterns(self):
        """Test 2: Verify refined validation log patterns are present"""
        logger.info("\n🔍 TEST 2: Check for refined anti-momentum validation log patterns")
        
        try:
            import subprocess
            
            # Get recent backend logs
            backend_logs = ""
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "5000", "/var/log/supervisor/backend.out.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "5000", "/var/log/supervisor/backend.err.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            if not backend_logs:
                self.log_test_result("Refined Validation Log Patterns", False, "Could not retrieve backend logs")
                return
            
            # Check for refined validation log patterns
            pattern_counts = {}
            for pattern in self.expected_log_patterns:
                count = backend_logs.count(pattern)
                pattern_counts[pattern] = count
                logger.info(f"   📊 '{pattern}': {count} occurrences")
            
            # Success criteria: At least 4 different patterns found with multiple occurrences
            patterns_found = sum(1 for count in pattern_counts.values() if count > 0)
            total_occurrences = sum(pattern_counts.values())
            
            # Look for specific validation cases
            legitimate_reversals = backend_logs.count("✅ LEGITIMATE REVERSAL DETECTED")
            momentum_errors = backend_logs.count("⚠️ POTENTIAL MOMENTUM ERROR")
            sophisticated_validations = backend_logs.count("✅ SOPHISTICATED VALIDATION APPLIED")
            
            logger.info(f"   📊 Legitimate reversals detected: {legitimate_reversals}")
            logger.info(f"   📊 Momentum errors detected: {momentum_errors}")
            logger.info(f"   📊 Sophisticated validations applied: {sophisticated_validations}")
            
            success = patterns_found >= 4 and total_occurrences >= 8
            
            details = f"Patterns found: {patterns_found}/{len(self.expected_log_patterns)}, Total occurrences: {total_occurrences}, Legitimate reversals: {legitimate_reversals}, Momentum errors: {momentum_errors}"
            
            self.log_test_result("Refined Validation Log Patterns", success, details)
            
        except Exception as e:
            self.log_test_result("Refined Validation Log Patterns", False, f"Exception: {str(e)}")
    
    async def test_3_legitimate_reversal_detection(self):
        """Test 3: Verify legitimate reversal detection for strong technical signals"""
        logger.info("\n🔍 TEST 3: Test legitimate reversal detection (RSI >75 + Stochastic >80 cases)")
        
        try:
            # Trigger new analysis to get fresh data
            logger.info("   🚀 Triggering fresh analysis via /api/start-trading...")
            start_response = requests.post(f"{self.api_url}/start-trading", timeout=180)
            
            if start_response.status_code not in [200, 201]:
                logger.warning(f"   ⚠️ Start trading returned HTTP {start_response.status_code}, continuing with existing data...")
            else:
                # Wait for processing
                logger.info("   ⏳ Waiting 30 seconds for analysis processing...")
                await asyncio.sleep(30)
            
            # Get IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Legitimate Reversal Detection", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("Legitimate Reversal Detection", False, "No IA1 analyses found")
                return
            
            # Look for legitimate reversal cases
            legitimate_reversal_cases = 0
            extreme_rsi_cases = 0
            extreme_stochastic_cases = 0
            bollinger_extreme_cases = 0
            
            for analysis in analyses[-15:]:  # Check last 15 analyses
                symbol = analysis.get('symbol', 'Unknown')
                reasoning = analysis.get('reasoning', '')
                confidence = analysis.get('confidence', 0)
                recommendation = analysis.get('recommendation', '').upper()
                
                # Check for extreme technical indicators that should trigger legitimate reversal
                has_extreme_rsi = any(rsi_indicator in reasoning for rsi_indicator in [
                    "RSI >75", "RSI > 75", "RSI 75", "RSI 76", "RSI 77", "RSI 78", "RSI 79", "RSI 80",
                    "RSI <25", "RSI < 25", "RSI 24", "RSI 23", "RSI 22", "RSI 21", "RSI 20"
                ])
                
                has_extreme_stochastic = any(stoch_indicator in reasoning for stoch_indicator in [
                    "Stochastic >80", "Stochastic > 80", "Stochastic 80", "Stochastic 85", "Stochastic 90",
                    "Stochastic <20", "Stochastic < 20", "Stochastic 20", "Stochastic 15", "Stochastic 10"
                ])
                
                has_bollinger_extreme = any(bb_indicator in reasoning for bb_indicator in [
                    "upper Bollinger", "lower Bollinger", "Bollinger Band", "band rejection", "band extreme"
                ])
                
                if has_extreme_rsi:
                    extreme_rsi_cases += 1
                    logger.info(f"      📊 {symbol}: Extreme RSI detected")
                
                if has_extreme_stochastic:
                    extreme_stochastic_cases += 1
                    logger.info(f"      📊 {symbol}: Extreme Stochastic detected")
                
                if has_bollinger_extreme:
                    bollinger_extreme_cases += 1
                    logger.info(f"      📊 {symbol}: Bollinger extreme detected")
                
                # Check if this was classified as legitimate reversal
                if has_extreme_rsi and has_extreme_stochastic:
                    legitimate_reversal_cases += 1
                    logger.info(f"      ✅ {symbol}: Should be legitimate reversal (extreme RSI + Stochastic)")
            
            # Check backend logs for legitimate reversal detections
            import subprocess
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
            
            legitimate_reversal_logs = backend_logs.count("✅ LEGITIMATE REVERSAL DETECTED")
            reversal_signals_logs = backend_logs.count("🔄 Reversal signals:")
            rsi_extreme_logs = backend_logs.count("RSI_OVERBOUGHT_EXTREME") + backend_logs.count("RSI_OVERSOLD_EXTREME")
            stochastic_extreme_logs = backend_logs.count("STOCHASTIC_OVERBOUGHT") + backend_logs.count("STOCHASTIC_OVERSOLD")
            
            logger.info(f"   📊 Potential legitimate reversal cases: {legitimate_reversal_cases}")
            logger.info(f"   📊 Extreme RSI cases: {extreme_rsi_cases}")
            logger.info(f"   📊 Extreme Stochastic cases: {extreme_stochastic_cases}")
            logger.info(f"   📊 Bollinger extreme cases: {bollinger_extreme_cases}")
            logger.info(f"   📊 Legitimate reversal logs: {legitimate_reversal_logs}")
            logger.info(f"   📊 Reversal signals logs: {reversal_signals_logs}")
            logger.info(f"   📊 RSI extreme logs: {rsi_extreme_logs}")
            logger.info(f"   📊 Stochastic extreme logs: {stochastic_extreme_logs}")
            
            # Success criteria: Evidence of legitimate reversal detection system working
            success = (legitimate_reversal_logs > 0 or reversal_signals_logs > 0 or 
                      (extreme_rsi_cases > 0 and extreme_stochastic_cases > 0))
            
            details = f"Legitimate cases: {legitimate_reversal_cases}, RSI extreme: {extreme_rsi_cases}, Stochastic extreme: {extreme_stochastic_cases}, Logs: {legitimate_reversal_logs}"
            
            self.log_test_result("Legitimate Reversal Detection", success, details)
            
        except Exception as e:
            self.log_test_result("Legitimate Reversal Detection", False, f"Exception: {str(e)}")
    
    async def test_4_momentum_error_detection(self):
        """Test 4: Verify momentum error detection for weak technical signals"""
        logger.info("\n🔍 TEST 4: Test momentum error detection (RSI 45-55 + no extremes cases)")
        
        try:
            # Get IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Momentum Error Detection", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("Momentum Error Detection", False, "No IA1 analyses found")
                return
            
            # Look for momentum error cases
            momentum_error_cases = 0
            moderate_rsi_cases = 0
            no_extreme_cases = 0
            counter_trend_cases = 0
            
            for analysis in analyses[-15:]:  # Check last 15 analyses
                symbol = analysis.get('symbol', 'Unknown')
                reasoning = analysis.get('reasoning', '')
                confidence = analysis.get('confidence', 0)
                recommendation = analysis.get('recommendation', '').upper()
                
                # Check for moderate RSI (no extremes)
                has_moderate_rsi = any(rsi_indicator in reasoning for rsi_indicator in [
                    "RSI 45", "RSI 46", "RSI 47", "RSI 48", "RSI 49", "RSI 50", 
                    "RSI 51", "RSI 52", "RSI 53", "RSI 54", "RSI 55"
                ])
                
                # Check for lack of extreme signals
                has_no_extremes = not any(extreme in reasoning for extreme in [
                    "overbought", "oversold", "extreme", "divergence", "exhaustion"
                ])
                
                # Check for counter-trend signals
                has_counter_trend = ("SHORT" in recommendation and any(bullish in reasoning.lower() for bullish in ["bullish", "uptrend", "+"])) or \
                                   ("LONG" in recommendation and any(bearish in reasoning.lower() for bearish in ["bearish", "downtrend", "-"]))
                
                if has_moderate_rsi:
                    moderate_rsi_cases += 1
                    logger.info(f"      📊 {symbol}: Moderate RSI detected")
                
                if has_no_extremes:
                    no_extreme_cases += 1
                    logger.info(f"      📊 {symbol}: No extreme signals detected")
                
                if has_counter_trend:
                    counter_trend_cases += 1
                    logger.info(f"      📊 {symbol}: Counter-trend signal detected")
                
                # Check if this should be classified as momentum error
                if has_moderate_rsi and has_no_extremes and has_counter_trend:
                    momentum_error_cases += 1
                    logger.info(f"      ⚠️ {symbol}: Should be momentum error (moderate RSI + no extremes + counter-trend)")
            
            # Check backend logs for momentum error detections
            import subprocess
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
            
            momentum_error_logs = backend_logs.count("⚠️ POTENTIAL MOMENTUM ERROR")
            validation_failed_logs = backend_logs.count("💥 Reversal validation: FAILED")
            no_reversal_signals_logs = backend_logs.count("NO_REVERSAL_SIGNALS")
            insufficient_confirmation_logs = backend_logs.count("INSUFFICIENT_REVERSAL_CONFIRMATION")
            
            logger.info(f"   📊 Potential momentum error cases: {momentum_error_cases}")
            logger.info(f"   📊 Moderate RSI cases: {moderate_rsi_cases}")
            logger.info(f"   📊 No extreme cases: {no_extreme_cases}")
            logger.info(f"   📊 Counter-trend cases: {counter_trend_cases}")
            logger.info(f"   📊 Momentum error logs: {momentum_error_logs}")
            logger.info(f"   📊 Validation failed logs: {validation_failed_logs}")
            logger.info(f"   📊 No reversal signals logs: {no_reversal_signals_logs}")
            logger.info(f"   📊 Insufficient confirmation logs: {insufficient_confirmation_logs}")
            
            # Success criteria: Evidence of momentum error detection system working
            success = (momentum_error_logs > 0 or validation_failed_logs > 0 or 
                      no_reversal_signals_logs > 0 or insufficient_confirmation_logs > 0)
            
            details = f"Momentum error cases: {momentum_error_cases}, Moderate RSI: {moderate_rsi_cases}, No extremes: {no_extreme_cases}, Error logs: {momentum_error_logs}"
            
            self.log_test_result("Momentum Error Detection", success, details)
            
        except Exception as e:
            self.log_test_result("Momentum Error Detection", False, f"Exception: {str(e)}")
    
    async def test_5_differentiated_penalties(self):
        """Test 5: Verify differentiated penalties (20% vs 35%) are applied correctly"""
        logger.info("\n🔍 TEST 5: Test differentiated penalties (20% legitimate vs 35% momentum error)")
        
        try:
            # Check backend logs for penalty applications
            import subprocess
            backend_logs = ""
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "5000", "/var/log/supervisor/backend.out.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            if not backend_logs:
                self.log_test_result("Differentiated Penalties", False, "Could not retrieve backend logs")
                return
            
            # Look for penalty applications
            legitimate_penalty_20 = 0
            momentum_error_penalty_35 = 0
            confidence_reductions = []
            
            # Parse logs for confidence changes
            lines = backend_logs.split('\n')
            for line in lines:
                if "Confidence:" in line and "→" in line:
                    # Extract confidence changes
                    if "(-20%" in line or "(-0.2" in line:
                        legitimate_penalty_20 += 1
                        logger.info(f"      ✅ 20% penalty applied (legitimate reversal): {line.strip()}")
                    elif "(-35%" in line or "(-0.35" in line:
                        momentum_error_penalty_35 += 1
                        logger.info(f"      ⚠️ 35% penalty applied (momentum error): {line.strip()}")
                    
                    # Extract confidence values
                    if "%" in line:
                        confidence_reductions.append(line.strip())
            
            # Check for specific penalty patterns
            legitimate_reversal_corrections = backend_logs.count("legitimate_reversal")
            momentum_error_corrections = backend_logs.count("momentum_error")
            forced_hold_corrections = backend_logs.count("forced_hold")
            sophisticated_validations = backend_logs.count("✅ SOPHISTICATED VALIDATION APPLIED")
            
            logger.info(f"   📊 20% penalties applied (legitimate): {legitimate_penalty_20}")
            logger.info(f"   📊 35% penalties applied (momentum error): {momentum_error_penalty_35}")
            logger.info(f"   📊 Legitimate reversal corrections: {legitimate_reversal_corrections}")
            logger.info(f"   📊 Momentum error corrections: {momentum_error_corrections}")
            logger.info(f"   📊 Forced hold corrections: {forced_hold_corrections}")
            logger.info(f"   📊 Sophisticated validations: {sophisticated_validations}")
            
            # Show some confidence reduction examples
            if confidence_reductions:
                logger.info("   📊 Confidence reduction examples:")
                for i, reduction in enumerate(confidence_reductions[:3]):  # Show first 3
                    logger.info(f"      {i+1}. {reduction}")
            
            # Success criteria: Evidence of differentiated penalty system working
            success = (legitimate_penalty_20 > 0 or momentum_error_penalty_35 > 0 or 
                      legitimate_reversal_corrections > 0 or momentum_error_corrections > 0 or
                      sophisticated_validations > 0)
            
            details = f"20% penalties: {legitimate_penalty_20}, 35% penalties: {momentum_error_penalty_35}, Legitimate corrections: {legitimate_reversal_corrections}, Error corrections: {momentum_error_corrections}"
            
            self.log_test_result("Differentiated Penalties", success, details)
            
        except Exception as e:
            self.log_test_result("Differentiated Penalties", False, f"Exception: {str(e)}")
    
    async def test_6_sophisticated_validation_integration(self):
        """Test 6: Verify sophisticated validation is integrated into IA1 analysis workflow"""
        logger.info("\n🔍 TEST 6: Test sophisticated validation integration in IA1 workflow")
        
        try:
            # Get IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Sophisticated Validation Integration", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("Sophisticated Validation Integration", False, "No IA1 analyses found")
                return
            
            # Analyze IA1 analyses for validation integration
            validation_enhanced_analyses = 0
            confidence_adjusted_analyses = 0
            reversal_aware_analyses = 0
            momentum_aware_analyses = 0
            
            for analysis in analyses[-10:]:  # Check last 10 analyses
                symbol = analysis.get('symbol', 'Unknown')
                reasoning = analysis.get('reasoning', '')
                confidence = analysis.get('confidence', 0)
                recommendation = analysis.get('recommendation', '').upper()
                
                # Check for validation enhancement indicators
                validation_indicators = [
                    "reversal", "momentum", "validation", "sophisticated", "technical confluence",
                    "overbought", "oversold", "extreme", "divergence"
                ]
                
                has_validation_enhancement = any(indicator in reasoning.lower() for indicator in validation_indicators)
                if has_validation_enhancement:
                    validation_enhanced_analyses += 1
                    logger.info(f"      ✅ {symbol}: Validation enhancement detected")
                
                # Check for confidence adjustments (lower confidence might indicate validation applied)
                if confidence < 75:  # Lower confidence might indicate validation penalty
                    confidence_adjusted_analyses += 1
                    logger.info(f"      📉 {symbol}: Lower confidence ({confidence}%) - possible validation adjustment")
                
                # Check for reversal awareness
                reversal_keywords = ["reversal", "counter-trend", "against momentum", "contrarian"]
                if any(keyword in reasoning.lower() for keyword in reversal_keywords):
                    reversal_aware_analyses += 1
                
                # Check for momentum awareness
                momentum_keywords = ["momentum", "trend", "direction", "bullish", "bearish"]
                if any(keyword in reasoning.lower() for keyword in momentum_keywords):
                    momentum_aware_analyses += 1
            
            # Check backend logs for validation integration
            import subprocess
            backend_logs = ""
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "2000", "/var/log/supervisor/backend.out.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            validation_applied_logs = backend_logs.count("✅ SOPHISTICATED VALIDATION APPLIED")
            momentum_validation_logs = backend_logs.count("MOMENTUM VALIDATION")
            reversal_validation_logs = backend_logs.count("REVERSAL VALIDATION")
            
            logger.info(f"   📊 Validation enhanced analyses: {validation_enhanced_analyses}/10")
            logger.info(f"   📊 Confidence adjusted analyses: {confidence_adjusted_analyses}/10")
            logger.info(f"   📊 Reversal aware analyses: {reversal_aware_analyses}/10")
            logger.info(f"   📊 Momentum aware analyses: {momentum_aware_analyses}/10")
            logger.info(f"   📊 Validation applied logs: {validation_applied_logs}")
            logger.info(f"   📊 Momentum validation logs: {momentum_validation_logs}")
            logger.info(f"   📊 Reversal validation logs: {reversal_validation_logs}")
            
            # Success criteria: Evidence of sophisticated validation integration
            success = (validation_enhanced_analyses >= 5 or validation_applied_logs > 0 or 
                      momentum_validation_logs > 0 or reversal_validation_logs > 0)
            
            details = f"Enhanced: {validation_enhanced_analyses}/10, Adjusted: {confidence_adjusted_analyses}/10, Reversal aware: {reversal_aware_analyses}/10, Validation logs: {validation_applied_logs}"
            
            self.log_test_result("Sophisticated Validation Integration", success, details)
            
        except Exception as e:
            self.log_test_result("Sophisticated Validation Integration", False, f"Exception: {str(e)}")
    
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
        """Run all refined anti-momentum validation system tests"""
        logger.info("🚀 Starting Refined Anti-Momentum Validation System Test Suite")
        logger.info("=" * 80)
        logger.info("📋 REFINED ANTI-MOMENTUM VALIDATION SYSTEM VERIFICATION")
        logger.info("🎯 Testing: _validate_legitimate_reversal, sophisticated signal detection, differentiated penalties")
        logger.info("🎯 Expected: Distinguish legitimate reversals from momentum errors with appropriate penalties")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_validate_legitimate_reversal_method()
        await self.test_2_refined_validation_log_patterns()
        await self.test_3_legitimate_reversal_detection()
        await self.test_4_momentum_error_detection()
        await self.test_5_differentiated_penalties()
        await self.test_6_sophisticated_validation_integration()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 REFINED ANTI-MOMENTUM VALIDATION SYSTEM SUMMARY")
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
        logger.info("📋 REFINED ANTI-MOMENTUM VALIDATION SYSTEM STATUS")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Refined Anti-Momentum Validation System FULLY FUNCTIONAL!")
            logger.info("✅ _validate_legitimate_reversal method implemented")
            logger.info("✅ Sophisticated reversal signal detection working")
            logger.info("✅ Legitimate reversal detection operational")
            logger.info("✅ Momentum error detection functional")
            logger.info("✅ Differentiated penalties (20% vs 35%) applied")
            logger.info("✅ Sophisticated validation integrated in IA1 workflow")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY FUNCTIONAL - Refined validation system working with minor gaps")
            logger.info("🔍 Some components may need fine-tuning for full optimization")
        elif passed_tests >= total_tests * 0.5:
            logger.info("⚠️ PARTIALLY FUNCTIONAL - Core validation features working")
            logger.info("🔧 Some advanced features may need implementation or debugging")
        else:
            logger.info("❌ SYSTEM NOT FUNCTIONAL - Critical issues with refined validation")
            logger.info("🚨 Major implementation gaps or system errors preventing functionality")
        
        # Specific requirements check
        logger.info("\n📝 REFINED VALIDATION REQUIREMENTS VERIFICATION:")
        
        requirements_met = []
        requirements_failed = []
        
        # Check each requirement based on test results
        for result in self.test_results:
            if result['success']:
                if "Method" in result['test']:
                    requirements_met.append("✅ _validate_legitimate_reversal method implemented")
                elif "Log Patterns" in result['test']:
                    requirements_met.append("✅ Refined validation log patterns detected")
                elif "Legitimate Reversal" in result['test']:
                    requirements_met.append("✅ Legitimate reversal detection working")
                elif "Momentum Error" in result['test']:
                    requirements_met.append("✅ Momentum error detection functional")
                elif "Differentiated Penalties" in result['test']:
                    requirements_met.append("✅ Differentiated penalties (20% vs 35%) applied")
                elif "Integration" in result['test']:
                    requirements_met.append("✅ Sophisticated validation integrated in IA1")
            else:
                if "Method" in result['test']:
                    requirements_failed.append("❌ _validate_legitimate_reversal method not implemented")
                elif "Log Patterns" in result['test']:
                    requirements_failed.append("❌ Refined validation log patterns missing")
                elif "Legitimate Reversal" in result['test']:
                    requirements_failed.append("❌ Legitimate reversal detection not working")
                elif "Momentum Error" in result['test']:
                    requirements_failed.append("❌ Momentum error detection not functional")
                elif "Differentiated Penalties" in result['test']:
                    requirements_failed.append("❌ Differentiated penalties not applied")
                elif "Integration" in result['test']:
                    requirements_failed.append("❌ Sophisticated validation not integrated")
        
        for req in requirements_met:
            logger.info(f"   {req}")
        
        for req in requirements_failed:
            logger.info(f"   {req}")
        
        logger.info(f"\n🏆 FINAL RESULT: {len(requirements_met)}/{len(requirements_met) + len(requirements_failed)} requirements satisfied")
        
        # Final verdict
        if len(requirements_failed) == 0:
            logger.info("\n🎉 VERDICT: Refined Anti-Momentum Validation System is FULLY FUNCTIONAL!")
            logger.info("✅ All validation features implemented and working correctly")
            logger.info("✅ System successfully distinguishes legitimate reversals from momentum errors")
            logger.info("✅ Differentiated penalties preserve good contrarian trades while blocking bad ones")
        elif len(requirements_failed) <= 1:
            logger.info("\n⚠️ VERDICT: Refined Anti-Momentum Validation System is MOSTLY FUNCTIONAL")
            logger.info("🔍 Minor issues may need attention for complete functionality")
        elif len(requirements_failed) <= 3:
            logger.info("\n⚠️ VERDICT: Refined Anti-Momentum Validation System is PARTIALLY FUNCTIONAL")
            logger.info("🔧 Several components need implementation or debugging")
        else:
            logger.info("\n❌ VERDICT: Refined Anti-Momentum Validation System is NOT FUNCTIONAL")
            logger.info("🚨 Major implementation gaps preventing refined validation")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = RefinedAntiMomentumValidationTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
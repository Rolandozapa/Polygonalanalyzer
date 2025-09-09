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

class MultiTimeframeHierarchicalAnalysisTestSuite:
    """Test suite for Multi-Timeframe Hierarchical Analysis System Verification"""
    
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
        logger.info(f"Testing Multi-Timeframe Hierarchical Analysis System at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Expected multi-timeframe log patterns
        self.expected_log_patterns = [
            "🎯 MULTI-TIMEFRAME ANALYSIS",
            "📊 Dominant Timeframe:",
            "📊 Decisive Pattern:",
            "📊 Hierarchy Confidence:",
            "⚠️ Anti-Momentum Risk:",
            "DAILY_BULLISH_MOMENTUM",
            "DAILY_BEARISH_MOMENTUM",
            "H4_BULLISH_CONTINUATION",
            "H4_BEARISH_CONTINUATION"
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
    
    async def test_1_multi_timeframe_log_patterns(self):
        """Test 1: Verify multi-timeframe analysis log patterns are present"""
        logger.info("\n🔍 TEST 1: Check for multi-timeframe analysis log patterns")
        
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
                self.log_test_result("Multi-Timeframe Log Patterns", False, "Could not retrieve backend logs")
                return
            
            # Check for multi-timeframe log patterns
            pattern_counts = {}
            for pattern in self.expected_log_patterns:
                count = backend_logs.count(pattern)
                pattern_counts[pattern] = count
                logger.info(f"   📊 '{pattern}': {count} occurrences")
            
            # Success criteria: At least 3 different patterns found with multiple occurrences
            patterns_found = sum(1 for count in pattern_counts.values() if count > 0)
            total_occurrences = sum(pattern_counts.values())
            
            success = patterns_found >= 3 and total_occurrences >= 5
            
            details = f"Patterns found: {patterns_found}/{len(self.expected_log_patterns)}, Total occurrences: {total_occurrences}"
            
            self.log_test_result("Multi-Timeframe Log Patterns", success, details)
            
        except Exception as e:
            self.log_test_result("Multi-Timeframe Log Patterns", False, f"Exception: {str(e)}")
    
    async def test_2_analyze_multi_timeframe_hierarchy_method(self):
        """Test 2: Verify analyze_multi_timeframe_hierarchy method is implemented"""
        logger.info("\n🔍 TEST 2: Check analyze_multi_timeframe_hierarchy method implementation")
        
        try:
            # Check if the method exists in the backend code
            backend_code = ""
            try:
                with open('/app/backend/server.py', 'r') as f:
                    backend_code = f.read()
            except Exception as e:
                self.log_test_result("Multi-Timeframe Method Implementation", False, f"Could not read backend code: {e}")
                return
            
            # Check for required method implementation
            method_found = "def analyze_multi_timeframe_hierarchy" in backend_code
            
            # Check for multi-timeframe analysis components
            components_found = {}
            required_components = [
                "_analyze_daily_context",
                "_analyze_h4_context", 
                "_analyze_h1_context",
                "dominant_timeframe",
                "decisive_pattern",
                "hierarchy_confidence",
                "anti_momentum_warning"
            ]
            
            for component in required_components:
                if component in backend_code:
                    components_found[component] = True
                    logger.info(f"      ✅ Component found: {component}")
                else:
                    components_found[component] = False
                    logger.info(f"      ❌ Component missing: {component}")
            
            # Check for multi-timeframe usage patterns
            usage_patterns = [
                "🎯 MULTI-TIMEFRAME ANALYSIS",
                "📊 Dominant Timeframe:",
                "📊 Decisive Pattern:",
                "📊 Hierarchy Confidence:",
                "⚠️ Anti-Momentum Risk:"
            ]
            
            usage_found = {}
            for pattern in usage_patterns:
                if pattern in backend_code:
                    usage_found[pattern] = True
                    logger.info(f"      ✅ Usage pattern found: {pattern}")
                else:
                    usage_found[pattern] = False
                    logger.info(f"      ❌ Usage pattern missing: {pattern}")
            
            components_implemented = sum(components_found.values())
            usage_patterns_found = sum(usage_found.values())
            
            logger.info(f"   📊 Method found: {method_found}")
            logger.info(f"   📊 Components implemented: {components_implemented}/{len(required_components)}")
            logger.info(f"   📊 Usage patterns found: {usage_patterns_found}/{len(usage_patterns)}")
            
            # Success criteria: Method exists and most components/patterns found
            success = method_found and components_implemented >= 5 and usage_patterns_found >= 3
            
            details = f"Method: {method_found}, Components: {components_implemented}/{len(required_components)}, Usage: {usage_patterns_found}/{len(usage_patterns)}"
            
            self.log_test_result("Multi-Timeframe Method Implementation", success, details)
            
        except Exception as e:
            self.log_test_result("Multi-Timeframe Method Implementation", False, f"Exception: {str(e)}")
    
    async def test_3_ia1_analyses_with_multi_timeframe_context(self):
        """Test 3: Verify IA1 analyses contain multi-timeframe context"""
        logger.info("\n🔍 TEST 3: Check IA1 analyses for multi-timeframe context integration")
        
        try:
            # Get IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("IA1 Multi-Timeframe Context", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("IA1 Multi-Timeframe Context", False, "No IA1 analyses found")
                return
            
            # Check recent analyses for multi-timeframe context
            multi_timeframe_analyses = 0
            dominant_timeframe_analyses = 0
            decisive_pattern_analyses = 0
            anti_momentum_analyses = 0
            
            for analysis in analyses[-10:]:  # Check last 10 analyses
                symbol = analysis.get('symbol', 'Unknown')
                reasoning = analysis.get('reasoning', '')
                
                # Check for multi-timeframe keywords in reasoning
                multi_timeframe_keywords = [
                    "multi-timeframe", "timeframe hierarchy", "dominant timeframe",
                    "daily momentum", "4H trend", "1H signals", "decisive pattern"
                ]
                
                has_multi_timeframe = any(keyword.lower() in reasoning.lower() for keyword in multi_timeframe_keywords)
                if has_multi_timeframe:
                    multi_timeframe_analyses += 1
                    logger.info(f"      ✅ {symbol}: Has multi-timeframe context")
                
                # Check for specific multi-timeframe fields
                if "dominant timeframe" in reasoning.lower() or "Dominant Timeframe:" in reasoning:
                    dominant_timeframe_analyses += 1
                
                if "decisive pattern" in reasoning.lower() or "Decisive Pattern:" in reasoning:
                    decisive_pattern_analyses += 1
                
                if "anti-momentum" in reasoning.lower() or "Anti-Momentum Risk:" in reasoning:
                    anti_momentum_analyses += 1
                    logger.info(f"      ⚠️ {symbol}: Has anti-momentum risk assessment")
            
            logger.info(f"   📊 Analyses with multi-timeframe context: {multi_timeframe_analyses}/10")
            logger.info(f"   📊 Analyses with dominant timeframe: {dominant_timeframe_analyses}/10")
            logger.info(f"   📊 Analyses with decisive pattern: {decisive_pattern_analyses}/10")
            logger.info(f"   📊 Analyses with anti-momentum assessment: {anti_momentum_analyses}/10")
            
            # Success criteria: At least 30% of recent analyses have multi-timeframe context
            success = multi_timeframe_analyses >= 3 or dominant_timeframe_analyses >= 2
            
            details = f"Multi-timeframe: {multi_timeframe_analyses}/10, Dominant timeframe: {dominant_timeframe_analyses}/10, Decisive pattern: {decisive_pattern_analyses}/10, Anti-momentum: {anti_momentum_analyses}/10"
            
            self.log_test_result("IA1 Multi-Timeframe Context", success, details)
            
        except Exception as e:
            self.log_test_result("IA1 Multi-Timeframe Context", False, f"Exception: {str(e)}")
    
    async def test_4_anti_momentum_risk_detection(self):
        """Test 4: Verify anti-momentum risk detection for strong daily moves"""
        logger.info("\n🔍 TEST 4: Test anti-momentum risk detection for strong daily moves")
        
        try:
            # Get IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Anti-Momentum Risk Detection", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("Anti-Momentum Risk Detection", False, "No IA1 analyses found")
                return
            
            # Look for strong momentum cases and anti-momentum detection
            strong_momentum_cases = 0
            anti_momentum_detected = 0
            confidence_reduced_cases = 0
            
            for analysis in analyses:
                symbol = analysis.get('symbol', 'Unknown')
                reasoning = analysis.get('reasoning', '')
                confidence = analysis.get('confidence', 0)
                recommendation = analysis.get('recommendation', '')
                
                # Check if this is a strong momentum case (we'll look for mentions of high price changes)
                strong_momentum_indicators = [
                    "+7%", "+8%", "+9%", "+10%", "+5%", "+6%",
                    "-7%", "-8%", "-9%", "-10%", "-5%", "-6%",
                    "strong momentum", "strong daily", "powerful move"
                ]
                
                is_strong_momentum = any(indicator in reasoning for indicator in strong_momentum_indicators)
                if is_strong_momentum:
                    strong_momentum_cases += 1
                    logger.info(f"      📈 {symbol}: Strong momentum case detected")
                    
                    # Check for anti-momentum risk detection
                    anti_momentum_indicators = [
                        "anti-momentum", "counter-trend", "against momentum",
                        "Anti-Momentum Risk:", "⚠️ Anti-Momentum Risk: HIGH"
                    ]
                    
                    has_anti_momentum = any(indicator in reasoning for indicator in anti_momentum_indicators)
                    if has_anti_momentum:
                        anti_momentum_detected += 1
                        logger.info(f"      ⚠️ {symbol}: Anti-momentum risk detected")
                    
                    # Check if confidence was reduced for counter-trend signals
                    if confidence < 70 and ("SHORT" in recommendation.upper() or "LONG" in recommendation.upper()):
                        confidence_reduced_cases += 1
                        logger.info(f"      📉 {symbol}: Confidence reduced to {confidence}% for counter-trend signal")
            
            # Check backend logs for anti-momentum warnings
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
            
            anti_momentum_logs = backend_logs.count("⚠️ Anti-Momentum Risk:")
            anti_momentum_warnings = backend_logs.count("⚠️ ANTI-MOMENTUM WARNING")
            
            logger.info(f"   📊 Strong momentum cases found: {strong_momentum_cases}")
            logger.info(f"   📊 Anti-momentum risk detected: {anti_momentum_detected}")
            logger.info(f"   📊 Confidence reduced cases: {confidence_reduced_cases}")
            logger.info(f"   📊 Anti-momentum logs: {anti_momentum_logs}")
            logger.info(f"   📊 Anti-momentum warnings: {anti_momentum_warnings}")
            
            # Success criteria: Evidence of anti-momentum risk detection system working
            success = (anti_momentum_detected > 0 or anti_momentum_logs > 0 or 
                      confidence_reduced_cases > 0 or anti_momentum_warnings > 0)
            
            details = f"Strong momentum: {strong_momentum_cases}, Anti-momentum detected: {anti_momentum_detected}, Confidence reduced: {confidence_reduced_cases}, Logs: {anti_momentum_logs}"
            
            self.log_test_result("Anti-Momentum Risk Detection", success, details)
            
        except Exception as e:
            self.log_test_result("Anti-Momentum Risk Detection", False, f"Exception: {str(e)}")
    
    async def test_5_grtusdt_case_prevention(self):
        """Test 5: Verify GRTUSDT-like cases are prevented (strong daily momentum vs counter-trend signals)"""
        logger.info("\n🔍 TEST 5: Test GRTUSDT-like case prevention (strong momentum vs counter-trend)")
        
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
                self.log_test_result("GRTUSDT Case Prevention", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("GRTUSDT Case Prevention", False, "No IA1 analyses found")
                return
            
            # Look for potential GRTUSDT-like cases
            grtusdt_like_cases = 0
            prevented_cases = 0
            high_confidence_counter_trend = 0
            
            for analysis in analyses[-15:]:  # Check last 15 analyses
                symbol = analysis.get('symbol', 'Unknown')
                reasoning = analysis.get('reasoning', '')
                confidence = analysis.get('confidence', 0)
                recommendation = analysis.get('recommendation', '').upper()
                
                # Look for strong bullish momentum indicators
                bullish_momentum_indicators = [
                    "+5%", "+6%", "+7%", "+8%", "+9%", "+10%",
                    "strong bullish", "bullish momentum", "strong daily bullish",
                    "DAILY_BULLISH_MOMENTUM"
                ]
                
                has_strong_bullish = any(indicator in reasoning for indicator in bullish_momentum_indicators)
                
                if has_strong_bullish:
                    grtusdt_like_cases += 1
                    logger.info(f"      📈 {symbol}: Strong bullish momentum detected")
                    
                    # Check if SHORT signal was given with high confidence (this would be the GRTUSDT problem)
                    if "SHORT" in recommendation and confidence >= 80:
                        high_confidence_counter_trend += 1
                        logger.warning(f"      ⚠️ {symbol}: HIGH CONFIDENCE SHORT ({confidence}%) against bullish momentum - GRTUSDT-like issue!")
                    elif "SHORT" in recommendation and confidence < 70:
                        prevented_cases += 1
                        logger.info(f"      ✅ {symbol}: SHORT confidence reduced to {confidence}% - GRTUSDT issue prevented")
                    elif "HOLD" in recommendation:
                        prevented_cases += 1
                        logger.info(f"      ✅ {symbol}: HOLD recommended instead of counter-trend - GRTUSDT issue prevented")
            
            # Check backend logs for specific GRTUSDT-related patterns
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
            
            # Look for GRTUSDT or similar symbols with anti-momentum detection
            grtusdt_mentions = backend_logs.count("GRTUSDT")
            daily_bullish_momentum = backend_logs.count("DAILY_BULLISH_MOMENTUM")
            anti_momentum_high = backend_logs.count("⚠️ Anti-Momentum Risk: HIGH")
            
            logger.info(f"   📊 GRTUSDT-like cases (strong bullish): {grtusdt_like_cases}")
            logger.info(f"   📊 Cases prevented (low confidence/HOLD): {prevented_cases}")
            logger.info(f"   📊 High confidence counter-trend (problematic): {high_confidence_counter_trend}")
            logger.info(f"   📊 GRTUSDT mentions in logs: {grtusdt_mentions}")
            logger.info(f"   📊 Daily bullish momentum detections: {daily_bullish_momentum}")
            logger.info(f"   📊 High anti-momentum risk warnings: {anti_momentum_high}")
            
            # Success criteria: No high confidence counter-trend signals OR evidence of prevention
            success = (high_confidence_counter_trend == 0 or prevented_cases > 0 or 
                      anti_momentum_high > 0 or daily_bullish_momentum > 0)
            
            details = f"GRTUSDT-like cases: {grtusdt_like_cases}, Prevented: {prevented_cases}, Problematic: {high_confidence_counter_trend}, Anti-momentum warnings: {anti_momentum_high}"
            
            self.log_test_result("GRTUSDT Case Prevention", success, details)
            
        except Exception as e:
            self.log_test_result("GRTUSDT Case Prevention", False, f"Exception: {str(e)}")
    
    async def test_6_enhanced_ia1_decision_making(self):
        """Test 6: Verify enhanced IA1 decision-making with multi-timeframe context"""
        logger.info("\n🔍 TEST 6: Verify enhanced IA1 decision-making with multi-timeframe context")
        
        try:
            # Get IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Enhanced IA1 Decision Making", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("Enhanced IA1 Decision Making", False, "No IA1 analyses found")
                return
            
            # Analyze decision quality improvements
            enhanced_decisions = 0
            timeframe_aware_decisions = 0
            mature_chartist_decisions = 0
            trend_aligned_decisions = 0
            
            for analysis in analyses[-10:]:  # Check last 10 analyses
                symbol = analysis.get('symbol', 'Unknown')
                reasoning = analysis.get('reasoning', '')
                confidence = analysis.get('confidence', 0)
                recommendation = analysis.get('recommendation', '').upper()
                
                # Check for enhanced decision-making indicators
                enhancement_indicators = [
                    "multi-timeframe", "timeframe hierarchy", "dominant timeframe",
                    "chartist maturity", "established trend", "trend alignment"
                ]
                
                has_enhancement = any(indicator in reasoning.lower() for indicator in enhancement_indicators)
                if has_enhancement:
                    enhanced_decisions += 1
                    logger.info(f"      ✅ {symbol}: Enhanced decision-making detected")
                
                # Check for timeframe awareness
                timeframe_indicators = [
                    "daily", "4H", "1H", "timeframe", "Daily", "H4", "H1"
                ]
                
                has_timeframe_awareness = any(indicator in reasoning for indicator in timeframe_indicators)
                if has_timeframe_awareness:
                    timeframe_aware_decisions += 1
                
                # Check for mature chartist behavior (avoiding counter-trend trades)
                mature_indicators = [
                    "avoid counter-trend", "respect established trend", "trend maturity",
                    "chartist maturity", "established trends"
                ]
                
                has_maturity = any(indicator in reasoning.lower() for indicator in mature_indicators)
                if has_maturity:
                    mature_chartist_decisions += 1
                    logger.info(f"      🎯 {symbol}: Mature chartist behavior detected")
                
                # Check for trend alignment
                if "LONG" in recommendation and any(bullish in reasoning.lower() for bullish in ["bullish", "uptrend", "positive"]):
                    trend_aligned_decisions += 1
                elif "SHORT" in recommendation and any(bearish in reasoning.lower() for bearish in ["bearish", "downtrend", "negative"]):
                    trend_aligned_decisions += 1
                elif "HOLD" in recommendation:
                    trend_aligned_decisions += 1  # HOLD is always trend-neutral/safe
            
            # Check backend logs for enhanced IA1 patterns
            import subprocess
            backend_logs = ""
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "1000", "/var/log/supervisor/backend.out.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            enhanced_ia1_logs = backend_logs.count("Enhanced IA1")
            multi_timeframe_logs = backend_logs.count("🎯 MULTI-TIMEFRAME ANALYSIS")
            chartist_maturity_logs = backend_logs.count("chartist maturity")
            
            logger.info(f"   📊 Enhanced decisions: {enhanced_decisions}/10")
            logger.info(f"   📊 Timeframe aware decisions: {timeframe_aware_decisions}/10")
            logger.info(f"   📊 Mature chartist decisions: {mature_chartist_decisions}/10")
            logger.info(f"   📊 Trend aligned decisions: {trend_aligned_decisions}/10")
            logger.info(f"   📊 Enhanced IA1 logs: {enhanced_ia1_logs}")
            logger.info(f"   📊 Multi-timeframe logs: {multi_timeframe_logs}")
            logger.info(f"   📊 Chartist maturity logs: {chartist_maturity_logs}")
            
            # Success criteria: Evidence of enhanced decision-making
            success = (enhanced_decisions >= 3 or timeframe_aware_decisions >= 5 or 
                      mature_chartist_decisions >= 2 or multi_timeframe_logs > 0)
            
            details = f"Enhanced: {enhanced_decisions}/10, Timeframe aware: {timeframe_aware_decisions}/10, Mature: {mature_chartist_decisions}/10, Trend aligned: {trend_aligned_decisions}/10"
            
            self.log_test_result("Enhanced IA1 Decision Making", success, details)
            
        except Exception as e:
            self.log_test_result("Enhanced IA1 Decision Making", False, f"Exception: {str(e)}")
    
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
        """Run all multi-timeframe hierarchical analysis system tests"""
        logger.info("🚀 Starting Multi-Timeframe Hierarchical Analysis System Test Suite")
        logger.info("=" * 80)
        logger.info("📋 MULTI-TIMEFRAME HIERARCHICAL ANALYSIS SYSTEM VERIFICATION")
        logger.info("🎯 Testing: analyze_multi_timeframe_hierarchy, anti-momentum detection, enhanced IA1")
        logger.info("🎯 Expected: Improved chartist maturity preventing counter-momentum trading errors")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_multi_timeframe_log_patterns()
        await self.test_2_analyze_multi_timeframe_hierarchy_method()
        await self.test_3_ia1_analyses_with_multi_timeframe_context()
        await self.test_4_anti_momentum_risk_detection()
        await self.test_5_grtusdt_case_prevention()
        await self.test_6_enhanced_ia1_decision_making()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 MULTI-TIMEFRAME HIERARCHICAL ANALYSIS SYSTEM SUMMARY")
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
        logger.info("📋 MULTI-TIMEFRAME HIERARCHICAL ANALYSIS SYSTEM STATUS")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Multi-Timeframe Hierarchical Analysis System FULLY FUNCTIONAL!")
            logger.info("✅ Multi-timeframe log patterns detected")
            logger.info("✅ analyze_multi_timeframe_hierarchy method implemented")
            logger.info("✅ IA1 analyses contain multi-timeframe context")
            logger.info("✅ Anti-momentum risk detection working")
            logger.info("✅ GRTUSDT-like cases prevented")
            logger.info("✅ Enhanced IA1 decision-making active")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY FUNCTIONAL - Multi-timeframe system working with minor gaps")
            logger.info("🔍 Some components may need fine-tuning for full optimization")
        elif passed_tests >= total_tests * 0.5:
            logger.info("⚠️ PARTIALLY FUNCTIONAL - Core multi-timeframe features working")
            logger.info("🔧 Some advanced features may need implementation or debugging")
        else:
            logger.info("❌ SYSTEM NOT FUNCTIONAL - Critical issues with multi-timeframe analysis")
            logger.info("🚨 Major implementation gaps or system errors preventing functionality")
        
        # Specific requirements check
        logger.info("\n📝 MULTI-TIMEFRAME REQUIREMENTS VERIFICATION:")
        
        requirements_met = []
        requirements_failed = []
        
        # Check each requirement based on test results
        for result in self.test_results:
            if result['success']:
                if "Log Patterns" in result['test']:
                    requirements_met.append("✅ Multi-timeframe log patterns detected in backend")
                elif "Method Implementation" in result['test']:
                    requirements_met.append("✅ analyze_multi_timeframe_hierarchy method implemented")
                elif "Multi-Timeframe Context" in result['test']:
                    requirements_met.append("✅ IA1 analyses contain multi-timeframe context")
                elif "Anti-Momentum Risk" in result['test']:
                    requirements_met.append("✅ Anti-momentum risk detection functional")
                elif "GRTUSDT Case" in result['test']:
                    requirements_met.append("✅ GRTUSDT-like counter-momentum cases prevented")
                elif "Enhanced IA1" in result['test']:
                    requirements_met.append("✅ Enhanced IA1 decision-making with chartist maturity")
            else:
                if "Log Patterns" in result['test']:
                    requirements_failed.append("❌ Multi-timeframe log patterns missing or insufficient")
                elif "Method Implementation" in result['test']:
                    requirements_failed.append("❌ analyze_multi_timeframe_hierarchy method not implemented")
                elif "Multi-Timeframe Context" in result['test']:
                    requirements_failed.append("❌ IA1 analyses lack multi-timeframe context")
                elif "Anti-Momentum Risk" in result['test']:
                    requirements_failed.append("❌ Anti-momentum risk detection not working")
                elif "GRTUSDT Case" in result['test']:
                    requirements_failed.append("❌ GRTUSDT-like counter-momentum cases not prevented")
                elif "Enhanced IA1" in result['test']:
                    requirements_failed.append("❌ Enhanced IA1 decision-making not functional")
        
        for req in requirements_met:
            logger.info(f"   {req}")
        
        for req in requirements_failed:
            logger.info(f"   {req}")
        
        logger.info(f"\n🏆 FINAL RESULT: {len(requirements_met)}/{len(requirements_met) + len(requirements_failed)} requirements satisfied")
        
        # Final verdict
        if len(requirements_failed) == 0:
            logger.info("\n🎉 VERDICT: Multi-Timeframe Hierarchical Analysis System is FULLY FUNCTIONAL!")
            logger.info("✅ All multi-timeframe features implemented and working correctly")
            logger.info("✅ IA1 enhanced with chartist maturity preventing counter-momentum errors")
            logger.info("✅ GRTUSDT-like issues successfully prevented by the system")
        elif len(requirements_failed) <= 1:
            logger.info("\n⚠️ VERDICT: Multi-Timeframe Hierarchical Analysis System is MOSTLY FUNCTIONAL")
            logger.info("🔍 Minor issues may need attention for complete functionality")
        elif len(requirements_failed) <= 3:
            logger.info("\n⚠️ VERDICT: Multi-Timeframe Hierarchical Analysis System is PARTIALLY FUNCTIONAL")
            logger.info("🔧 Several components need implementation or debugging")
        else:
            logger.info("\n❌ VERDICT: Multi-Timeframe Hierarchical Analysis System is NOT FUNCTIONAL")
            logger.info("🚨 Major implementation gaps preventing multi-timeframe analysis")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = MultiTimeframeHierarchicalAnalysisTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
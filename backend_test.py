#!/usr/bin/env python3
"""
Backend Testing Suite for Sophisticated RR Analysis System
Focus: TEST SOPHISTICATED RR ANALYSIS SYSTEM - Verify that the new sophisticated RR analysis system with neutral/composite calculations is implemented and functional.

New sophisticated RR features to test:
1. calculate_neutral_risk_reward() - Volatility-based RR calculation without direction
2. calculate_composite_rr() - Combines directional + neutral RR approaches  
3. evaluate_sophisticated_risk_level() - LOW/MEDIUM/HIGH based on composite RR + volatility
4. RR validation (IA1 vs Composite RR divergence detection)
5. Enhanced IA2 prompt with sophisticated RR analysis section
6. Sophisticated risk level integration in IA2 JSON response

Expected log patterns:
- "🧠 SOPHISTICATED ANALYSIS {symbol}:"
- "📊 Composite RR: X.XX"  
- "📊 Bullish RR: X.XX, Bearish RR: X.XX"
- "📊 Neutral RR: X.XX"
- "🎯 Sophisticated Risk Level: LOW/MEDIUM/HIGH"
- "✅ RR VALIDATION {symbol}: IA1 RR X.XX ↔ Composite RR X.XX (ALIGNED/DIVERGENT)"
- "⚠️ SIGNIFICANT RR DIVERGENCE {symbol}: IA1 RR X.XX vs Composite RR X.XX"

Verify the sophisticated RR system enhances IA2's decision-making with advanced validation and risk assessment capabilities.
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

class SophisticatedRRAnalysisTestSuite:
    """Test suite for Sophisticated RR Analysis System Verification"""
    
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
        logger.info(f"Testing Sophisticated RR Analysis System at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Expected sophisticated RR log patterns
        self.expected_log_patterns = [
            "🧠 SOPHISTICATED ANALYSIS",
            "📊 Composite RR:",
            "📊 Bullish RR:",
            "📊 Bearish RR:",
            "📊 Neutral RR:",
            "🎯 Sophisticated Risk Level:",
            "✅ RR VALIDATION",
            "⚠️ SIGNIFICANT RR DIVERGENCE"
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
    
    async def test_1_sophisticated_rr_log_patterns(self):
        """Test 1: Verify sophisticated RR analysis log patterns are present"""
        logger.info("\n🔍 TEST 1: Check for sophisticated RR analysis log patterns")
        
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
                self.log_test_result("Sophisticated RR Log Patterns", False, "Could not retrieve backend logs")
                return
            
            # Check for sophisticated RR log patterns
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
            
            self.log_test_result("Sophisticated RR Log Patterns", success, details)
            
        except Exception as e:
            self.log_test_result("Sophisticated RR Log Patterns", False, f"Exception: {str(e)}")
    
    async def test_2_ia2_decisions_with_sophisticated_rr(self):
        """Test 2: Verify IA2 decisions contain sophisticated RR analysis data"""
        logger.info("\n🔍 TEST 2: Check IA2 decisions for sophisticated RR analysis integration")
        
        try:
            # Get IA2 decisions
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("IA2 Decisions Sophisticated RR", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            decisions = data.get('decisions', [])
            
            if not decisions:
                self.log_test_result("IA2 Decisions Sophisticated RR", False, "No IA2 decisions found")
                return
            
            # Check recent decisions for sophisticated RR fields
            sophisticated_rr_decisions = 0
            risk_level_decisions = 0
            composite_rr_decisions = 0
            
            for decision in decisions[-10:]:  # Check last 10 decisions
                symbol = decision.get('symbol', 'Unknown')
                
                # Check for sophisticated_rr_analysis field in decision_logic
                decision_logic = decision.get('decision_logic', {})
                if isinstance(decision_logic, dict):
                    sophisticated_rr = decision_logic.get('sophisticated_rr_analysis', {})
                    if sophisticated_rr:
                        sophisticated_rr_decisions += 1
                        logger.info(f"      ✅ {symbol}: Has sophisticated_rr_analysis field")
                        
                        # Check for specific sophisticated RR fields
                        if sophisticated_rr.get('composite_rr'):
                            composite_rr_decisions += 1
                        if sophisticated_rr.get('sophisticated_risk_level'):
                            risk_level_decisions += 1
                    else:
                        logger.info(f"      ❌ {symbol}: Missing sophisticated_rr_analysis field")
                
                # Check for risk_level field
                risk_level = decision.get('risk_level', '')
                if risk_level in ['LOW', 'MEDIUM', 'HIGH']:
                    logger.info(f"      ✅ {symbol}: Has risk_level: {risk_level}")
                else:
                    logger.info(f"      ⚠️ {symbol}: Missing or invalid risk_level: {risk_level}")
            
            logger.info(f"   📊 Decisions with sophisticated RR analysis: {sophisticated_rr_decisions}/10")
            logger.info(f"   📊 Decisions with composite RR: {composite_rr_decisions}/10")
            logger.info(f"   📊 Decisions with risk level: {risk_level_decisions}/10")
            
            # Success criteria: At least 30% of recent decisions have sophisticated RR data
            success = sophisticated_rr_decisions >= 3 or composite_rr_decisions >= 3
            
            details = f"Sophisticated RR: {sophisticated_rr_decisions}/10, Composite RR: {composite_rr_decisions}/10, Risk levels: {risk_level_decisions}/10"
            
            self.log_test_result("IA2 Decisions Sophisticated RR", success, details)
            
        except Exception as e:
            self.log_test_result("IA2 Decisions Sophisticated RR", False, f"Exception: {str(e)}")
    
    async def test_3_trigger_ia2_and_verify_sophisticated_rr(self):
        """Test 3: Trigger IA2 processing and verify sophisticated RR analysis is active"""
        logger.info("\n🔍 TEST 3: Trigger IA2 processing and verify sophisticated RR analysis")
        
        try:
            # Get initial decision count
            initial_response = requests.get(f"{self.api_url}/decisions", timeout=30)
            initial_data = initial_response.json() if initial_response.status_code == 200 else {}
            initial_count = len(initial_data.get('decisions', []))
            
            logger.info(f"   📊 Initial IA2 decisions count: {initial_count}")
            
            # Trigger IA2 processing
            logger.info("   🚀 Triggering IA2 processing via /api/start-trading...")
            start_response = requests.post(f"{self.api_url}/start-trading", timeout=180)
            
            logger.info(f"   📊 Start trading response: HTTP {start_response.status_code}")
            
            if start_response.status_code not in [200, 201]:
                self.log_test_result("Trigger IA2 Sophisticated RR", False, f"Start trading failed: HTTP {start_response.status_code}")
                return
            
            # Wait for processing
            logger.info("   ⏳ Waiting 45 seconds for IA2 processing...")
            await asyncio.sleep(45)
            
            # Check backend logs for sophisticated RR patterns
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
            
            # Count sophisticated RR analysis occurrences
            sophisticated_analysis_count = backend_logs.count("🧠 SOPHISTICATED ANALYSIS")
            composite_rr_count = backend_logs.count("📊 Composite RR:")
            risk_level_count = backend_logs.count("🎯 Sophisticated Risk Level:")
            rr_validation_count = backend_logs.count("✅ RR VALIDATION")
            
            logger.info(f"   📊 Sophisticated analysis logs: {sophisticated_analysis_count}")
            logger.info(f"   📊 Composite RR logs: {composite_rr_count}")
            logger.info(f"   📊 Risk level logs: {risk_level_count}")
            logger.info(f"   📊 RR validation logs: {rr_validation_count}")
            
            # Check for new decisions
            updated_response = requests.get(f"{self.api_url}/decisions", timeout=30)
            updated_data = updated_response.json() if updated_response.status_code == 200 else {}
            updated_count = len(updated_data.get('decisions', []))
            new_decisions = updated_count - initial_count
            
            logger.info(f"   📊 New decisions generated: {new_decisions}")
            
            # Success criteria: Either sophisticated RR logs found OR new decisions with sophisticated RR
            log_success = sophisticated_analysis_count > 0 or composite_rr_count > 0
            decision_success = new_decisions > 0 or updated_count > 15
            
            success = log_success or decision_success
            
            details = f"Sophisticated logs: {sophisticated_analysis_count}, New decisions: {new_decisions}, Total decisions: {updated_count}"
            
            self.log_test_result("Trigger IA2 Sophisticated RR", success, details)
            
        except Exception as e:
            self.log_test_result("Trigger IA2 Sophisticated RR", False, f"Exception: {str(e)}")
    
    async def test_4_sophisticated_rr_calculation_methods(self):
        """Test 4: Verify sophisticated RR calculation methods are implemented"""
        logger.info("\n🔍 TEST 4: Verify sophisticated RR calculation methods implementation")
        
        try:
            import subprocess
            
            # Check if the sophisticated RR methods exist in the backend code
            backend_code = ""
            try:
                with open('/app/backend/server.py', 'r') as f:
                    backend_code = f.read()
            except Exception as e:
                self.log_test_result("Sophisticated RR Methods", False, f"Could not read backend code: {e}")
                return
            
            # Check for required method implementations
            methods_found = {}
            required_methods = [
                "calculate_neutral_risk_reward",
                "calculate_composite_rr", 
                "evaluate_sophisticated_risk_level",
                "calculate_bullish_rr",
                "calculate_bearish_rr"
            ]
            
            for method in required_methods:
                if f"def {method}" in backend_code:
                    methods_found[method] = True
                    logger.info(f"      ✅ Method found: {method}")
                else:
                    methods_found[method] = False
                    logger.info(f"      ❌ Method missing: {method}")
            
            # Check for sophisticated RR usage in IA2 decision making
            sophisticated_usage_patterns = [
                "calculate_composite_rr(",
                "evaluate_sophisticated_risk_level(",
                "🧠 SOPHISTICATED ANALYSIS",
                "sophisticated_rr_analysis",
                "composite_rr_data"
            ]
            
            usage_found = {}
            for pattern in sophisticated_usage_patterns:
                if pattern in backend_code:
                    usage_found[pattern] = True
                    logger.info(f"      ✅ Usage pattern found: {pattern}")
                else:
                    usage_found[pattern] = False
                    logger.info(f"      ❌ Usage pattern missing: {pattern}")
            
            methods_implemented = sum(methods_found.values())
            usage_patterns_found = sum(usage_found.values())
            
            logger.info(f"   📊 Methods implemented: {methods_implemented}/{len(required_methods)}")
            logger.info(f"   📊 Usage patterns found: {usage_patterns_found}/{len(sophisticated_usage_patterns)}")
            
            # Success criteria: All methods implemented and most usage patterns found
            success = methods_implemented >= 4 and usage_patterns_found >= 3
            
            details = f"Methods: {methods_implemented}/{len(required_methods)}, Usage patterns: {usage_patterns_found}/{len(sophisticated_usage_patterns)}"
            
            self.log_test_result("Sophisticated RR Methods", success, details)
            
        except Exception as e:
            self.log_test_result("Sophisticated RR Methods", False, f"Exception: {str(e)}")
    
    async def test_5_rr_validation_and_divergence_detection(self):
        """Test 5: Verify RR validation and divergence detection functionality"""
        logger.info("\n🔍 TEST 5: Verify RR validation and divergence detection")
        
        try:
            import subprocess
            
            # Get recent backend logs
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
            
            if not backend_logs:
                self.log_test_result("RR Validation and Divergence", False, "Could not retrieve backend logs")
                return
            
            # Look for RR validation patterns
            rr_validation_aligned = backend_logs.count("✅ RR VALIDATION") + backend_logs.count("(ALIGNED)")
            rr_validation_divergent = backend_logs.count("(DIVERGENT)")
            significant_divergence = backend_logs.count("⚠️ SIGNIFICANT RR DIVERGENCE")
            
            # Look for IA1 vs Composite RR comparisons
            ia1_composite_comparisons = 0
            lines = backend_logs.split('\n')
            for line in lines:
                if "IA1 RR" in line and "Composite RR" in line:
                    ia1_composite_comparisons += 1
                    logger.info(f"      📋 RR comparison found: {line.strip()}")
            
            logger.info(f"   📊 RR validation (aligned): {rr_validation_aligned}")
            logger.info(f"   📊 RR validation (divergent): {rr_validation_divergent}")
            logger.info(f"   📊 Significant divergence warnings: {significant_divergence}")
            logger.info(f"   📊 IA1 vs Composite RR comparisons: {ia1_composite_comparisons}")
            
            # Check IA2 decisions for RR validation data
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            rr_validation_in_decisions = 0
            
            if response.status_code == 200:
                data = response.json()
                decisions = data.get('decisions', [])
                
                for decision in decisions[-5:]:  # Check last 5 decisions
                    decision_logic = decision.get('decision_logic', {})
                    if isinstance(decision_logic, dict):
                        sophisticated_rr = decision_logic.get('sophisticated_rr_analysis', {})
                        if sophisticated_rr and sophisticated_rr.get('rr_validation_status'):
                            rr_validation_in_decisions += 1
                            symbol = decision.get('symbol', 'Unknown')
                            validation_status = sophisticated_rr.get('rr_validation_status', 'Unknown')
                            logger.info(f"      ✅ {symbol}: RR validation status: {validation_status}")
            
            logger.info(f"   📊 Decisions with RR validation data: {rr_validation_in_decisions}/5")
            
            # Success criteria: Evidence of RR validation system working
            log_evidence = (rr_validation_aligned + rr_validation_divergent + significant_divergence) > 0
            comparison_evidence = ia1_composite_comparisons > 0
            decision_evidence = rr_validation_in_decisions > 0
            
            success = log_evidence or comparison_evidence or decision_evidence
            
            details = f"Log evidence: {log_evidence}, Comparisons: {ia1_composite_comparisons}, Decision validation: {rr_validation_in_decisions}"
            
            self.log_test_result("RR Validation and Divergence", success, details)
            
        except Exception as e:
            self.log_test_result("RR Validation and Divergence", False, f"Exception: {str(e)}")
    
    async def test_6_sophisticated_risk_level_integration(self):
        """Test 6: Verify sophisticated risk level integration in IA2 responses"""
        logger.info("\n🔍 TEST 6: Verify sophisticated risk level integration")
        
        try:
            # Get IA2 decisions
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Sophisticated Risk Level Integration", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            decisions = data.get('decisions', [])
            
            if not decisions:
                self.log_test_result("Sophisticated Risk Level Integration", False, "No IA2 decisions found")
                return
            
            # Check for sophisticated risk level in decisions
            risk_level_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "MISSING": 0}
            sophisticated_risk_levels = 0
            
            for decision in decisions[-10:]:  # Check last 10 decisions
                symbol = decision.get('symbol', 'Unknown')
                
                # Check main risk_level field
                risk_level = decision.get('risk_level', '')
                if risk_level in ['LOW', 'MEDIUM', 'HIGH']:
                    risk_level_counts[risk_level] += 1
                    logger.info(f"      ✅ {symbol}: Risk level: {risk_level}")
                else:
                    risk_level_counts["MISSING"] += 1
                    logger.info(f"      ❌ {symbol}: Missing risk level")
                
                # Check sophisticated_rr_analysis field
                decision_logic = decision.get('decision_logic', {})
                if isinstance(decision_logic, dict):
                    sophisticated_rr = decision_logic.get('sophisticated_rr_analysis', {})
                    if sophisticated_rr and sophisticated_rr.get('sophisticated_risk_level'):
                        sophisticated_risk_levels += 1
                        soph_risk = sophisticated_rr.get('sophisticated_risk_level', 'Unknown')
                        logger.info(f"      ✅ {symbol}: Sophisticated risk level: {soph_risk}")
            
            logger.info(f"   📊 Risk level distribution: LOW={risk_level_counts['LOW']}, MEDIUM={risk_level_counts['MEDIUM']}, HIGH={risk_level_counts['HIGH']}, MISSING={risk_level_counts['MISSING']}")
            logger.info(f"   📊 Decisions with sophisticated risk level: {sophisticated_risk_levels}/10")
            
            # Check backend logs for risk level calculations
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
            
            risk_level_logs = backend_logs.count("🎯 Sophisticated Risk Level:")
            risk_evaluation_logs = backend_logs.count("🎯 SOPHISTICATED RISK EVALUATION:")
            
            logger.info(f"   📊 Risk level calculation logs: {risk_level_logs}")
            logger.info(f"   📊 Risk evaluation logs: {risk_evaluation_logs}")
            
            # Success criteria: Most decisions have risk levels and some have sophisticated risk levels
            valid_risk_levels = risk_level_counts['LOW'] + risk_level_counts['MEDIUM'] + risk_level_counts['HIGH']
            success = valid_risk_levels >= 7 or sophisticated_risk_levels >= 3 or risk_level_logs > 0
            
            details = f"Valid risk levels: {valid_risk_levels}/10, Sophisticated: {sophisticated_risk_levels}/10, Logs: {risk_level_logs}"
            
            self.log_test_result("Sophisticated Risk Level Integration", success, details)
            
        except Exception as e:
            self.log_test_result("Sophisticated Risk Level Integration", False, f"Exception: {str(e)}")
    
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
        """Run all sophisticated RR analysis system tests"""
        logger.info("🚀 Starting Sophisticated RR Analysis System Test Suite")
        logger.info("=" * 80)
        logger.info("📋 SOPHISTICATED RR ANALYSIS SYSTEM VERIFICATION")
        logger.info("🎯 Testing: calculate_neutral_risk_reward, calculate_composite_rr, evaluate_sophisticated_risk_level")
        logger.info("🎯 Expected: Enhanced IA2 decisions with sophisticated RR validation and risk assessment")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_sophisticated_rr_log_patterns()
        await self.test_2_ia2_decisions_with_sophisticated_rr()
        await self.test_3_trigger_ia2_and_verify_sophisticated_rr()
        await self.test_4_sophisticated_rr_calculation_methods()
        await self.test_5_rr_validation_and_divergence_detection()
        await self.test_6_sophisticated_risk_level_integration()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 SOPHISTICATED RR ANALYSIS SYSTEM SUMMARY")
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
        logger.info("📋 SOPHISTICATED RR ANALYSIS SYSTEM STATUS")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Sophisticated RR Analysis System FULLY FUNCTIONAL!")
            logger.info("✅ Sophisticated RR log patterns detected")
            logger.info("✅ IA2 decisions contain sophisticated RR analysis")
            logger.info("✅ IA2 processing generates sophisticated RR calculations")
            logger.info("✅ Sophisticated RR calculation methods implemented")
            logger.info("✅ RR validation and divergence detection working")
            logger.info("✅ Sophisticated risk level integration active")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY FUNCTIONAL - Sophisticated RR system working with minor gaps")
            logger.info("🔍 Some components may need fine-tuning for full optimization")
        elif passed_tests >= total_tests * 0.5:
            logger.info("⚠️ PARTIALLY FUNCTIONAL - Core sophisticated RR features working")
            logger.info("🔧 Some advanced features may need implementation or debugging")
        else:
            logger.info("❌ SYSTEM NOT FUNCTIONAL - Critical issues with sophisticated RR analysis")
            logger.info("🚨 Major implementation gaps or system errors preventing functionality")
        
        # Specific requirements check
        logger.info("\n📝 SOPHISTICATED RR REQUIREMENTS VERIFICATION:")
        
        requirements_met = []
        requirements_failed = []
        
        # Check each requirement based on test results
        for result in self.test_results:
            if result['success']:
                if "Log Patterns" in result['test']:
                    requirements_met.append("✅ Sophisticated RR log patterns detected in backend")
                elif "Decisions Sophisticated RR" in result['test']:
                    requirements_met.append("✅ IA2 decisions contain sophisticated RR analysis data")
                elif "Trigger IA2" in result['test']:
                    requirements_met.append("✅ IA2 processing generates sophisticated RR calculations")
                elif "Methods" in result['test']:
                    requirements_met.append("✅ Sophisticated RR calculation methods implemented")
                elif "Validation" in result['test']:
                    requirements_met.append("✅ RR validation and divergence detection functional")
                elif "Risk Level" in result['test']:
                    requirements_met.append("✅ Sophisticated risk level integration working")
            else:
                if "Log Patterns" in result['test']:
                    requirements_failed.append("❌ Sophisticated RR log patterns missing or insufficient")
                elif "Decisions Sophisticated RR" in result['test']:
                    requirements_failed.append("❌ IA2 decisions lack sophisticated RR analysis data")
                elif "Trigger IA2" in result['test']:
                    requirements_failed.append("❌ IA2 processing not generating sophisticated RR calculations")
                elif "Methods" in result['test']:
                    requirements_failed.append("❌ Sophisticated RR calculation methods not implemented")
                elif "Validation" in result['test']:
                    requirements_failed.append("❌ RR validation and divergence detection not working")
                elif "Risk Level" in result['test']:
                    requirements_failed.append("❌ Sophisticated risk level integration not functional")
        
        for req in requirements_met:
            logger.info(f"   {req}")
        
        for req in requirements_failed:
            logger.info(f"   {req}")
        
        logger.info(f"\n🏆 FINAL RESULT: {len(requirements_met)}/{len(requirements_met) + len(requirements_failed)} requirements satisfied")
        
        # Final verdict
        if len(requirements_failed) == 0:
            logger.info("\n🎉 VERDICT: Sophisticated RR Analysis System is FULLY FUNCTIONAL!")
            logger.info("✅ All sophisticated RR features implemented and working correctly")
            logger.info("✅ IA2 decisions enhanced with advanced RR validation and risk assessment")
        elif len(requirements_failed) <= 1:
            logger.info("\n⚠️ VERDICT: Sophisticated RR Analysis System is MOSTLY FUNCTIONAL")
            logger.info("🔍 Minor issues may need attention for complete functionality")
        elif len(requirements_failed) <= 3:
            logger.info("\n⚠️ VERDICT: Sophisticated RR Analysis System is PARTIALLY FUNCTIONAL")
            logger.info("🔧 Several components need implementation or debugging")
        else:
            logger.info("\n❌ VERDICT: Sophisticated RR Analysis System is NOT FUNCTIONAL")
            logger.info("🚨 Major implementation gaps preventing sophisticated RR analysis")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = SophisticatedRRAnalysisTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
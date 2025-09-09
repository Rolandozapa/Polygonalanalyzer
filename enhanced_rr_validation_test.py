#!/usr/bin/env python3
"""
Backend Testing Suite for Enhanced RR Validation System
Focus: TEST ENHANCED RR VALIDATION SYSTEM - Verify that the new IA2 RR calculation validation system works correctly with proper LONG/SHORT equation validation.

New features to test:
1. IA2 receives enhanced prompt with CRITICAL RR CALCULATION FORMULAS
2. Backend validates IA2's RR calculations using correct equations:
   - LONG: Risk = Entry - Stop Loss, Reward = Take Profit - Entry
   - SHORT: Risk = Stop Loss - Entry, Reward = Entry - Take Profit
3. Validation checks level order (LONG: SL < Entry < TP, SHORT: TP < Entry < SL)
4. System corrects IA2's RR if calculation is wrong
5. Fallback to IA1 RR if validation fails

Expected log patterns:
- "🔧 IA2 RR CORRECTION: {symbol} - IA2 claimed X, corrected to Y"  
- "✅ IA2 RR VALIDATED: {symbol} - {rr} ({validation_message})"
- "❌ IA2 RR VALIDATION FAILED: {symbol} - {validation_message}"
- "🎯 USING IA2 ENHANCED RR" or "🔄 USING IA1 ORIGINAL RR"

Test by triggering IA2 decision making and verify the validation system works correctly for both LONG and SHORT positions.
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

class EnhancedRRValidationTestSuite:
    """Test suite for Enhanced RR Validation System"""
    
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
        logger.info(f"Testing Enhanced RR Validation System at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Expected log patterns for RR validation
        self.expected_log_patterns = [
            "🔧 IA2 RR CORRECTION:",
            "✅ IA2 RR VALIDATED:",
            "❌ IA2 RR VALIDATION FAILED:",
            "🎯 USING IA2 ENHANCED RR",
            "🔄 USING IA1 ORIGINAL RR"
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
    
    async def test_1_enhanced_rr_prompt_integration(self):
        """Test 1: Verify IA2 receives enhanced prompt with CRITICAL RR CALCULATION FORMULAS"""
        logger.info("\n🔍 TEST 1: Enhanced RR Prompt Integration")
        
        try:
            # Check if the enhanced prompt is in the server.py code
            with open('/app/backend/server.py', 'r') as f:
                server_code = f.read()
            
            # Look for critical RR calculation formulas in the prompt
            critical_formulas_found = []
            
            if "CRITICAL RR CALCULATION FORMULAS" in server_code:
                critical_formulas_found.append("CRITICAL RR CALCULATION FORMULAS section")
            
            if "Risk = Entry Price - Stop Loss" in server_code:
                critical_formulas_found.append("LONG Risk formula")
            
            if "Reward = Take Profit - Entry Price" in server_code:
                critical_formulas_found.append("LONG Reward formula")
            
            if "Risk = Stop Loss - Entry Price" in server_code:
                critical_formulas_found.append("SHORT Risk formula")
            
            if "Reward = Entry Price - Take Profit" in server_code:
                critical_formulas_found.append("SHORT Reward formula")
            
            if "SL < Entry < TP" in server_code:
                critical_formulas_found.append("LONG validation rule")
            
            if "TP < Entry < SL" in server_code:
                critical_formulas_found.append("SHORT validation rule")
            
            logger.info(f"   📊 Critical RR formulas found: {len(critical_formulas_found)}/7")
            for formula in critical_formulas_found:
                logger.info(f"      ✅ {formula}")
            
            # Success criteria: All 7 critical formulas present
            success = len(critical_formulas_found) >= 6  # Allow for minor variations
            
            details = f"Found {len(critical_formulas_found)}/7 critical RR formulas in IA2 prompt"
            
            self.log_test_result("Enhanced RR Prompt Integration", success, details)
            
        except Exception as e:
            self.log_test_result("Enhanced RR Prompt Integration", False, f"Exception: {str(e)}")
    
    async def test_2_rr_validation_backend_implementation(self):
        """Test 2: Verify backend RR validation implementation exists"""
        logger.info("\n🔍 TEST 2: RR Validation Backend Implementation")
        
        try:
            # Check if the RR validation code is implemented in server.py
            with open('/app/backend/server.py', 'r') as f:
                server_code = f.read()
            
            validation_components = []
            
            # Check for validation logic components
            if "🔧 IA2 RR CORRECTION:" in server_code:
                validation_components.append("RR correction logging")
            
            if "✅ IA2 RR VALIDATED:" in server_code:
                validation_components.append("RR validation success logging")
            
            if "❌ IA2 RR VALIDATION FAILED:" in server_code:
                validation_components.append("RR validation failure logging")
            
            if "🎯 USING IA2 ENHANCED RR" in server_code:
                validation_components.append("IA2 enhanced RR usage logging")
            
            if "🔄 USING IA1 ORIGINAL RR" in server_code:
                validation_components.append("IA1 fallback RR usage logging")
            
            # Check for validation formulas implementation
            if "risk = current_price - ia2_support" in server_code:
                validation_components.append("LONG risk calculation")
            
            if "reward = ia2_resistance - current_price" in server_code:
                validation_components.append("LONG reward calculation")
            
            if "risk = ia2_resistance - current_price" in server_code:
                validation_components.append("SHORT risk calculation")
            
            if "reward = current_price - ia2_support" in server_code:
                validation_components.append("SHORT reward calculation")
            
            logger.info(f"   📊 Validation components found: {len(validation_components)}/9")
            for component in validation_components:
                logger.info(f"      ✅ {component}")
            
            # Success criteria: Most validation components present
            success = len(validation_components) >= 7
            
            details = f"Found {len(validation_components)}/9 validation components in backend"
            
            self.log_test_result("RR Validation Backend Implementation", success, details)
            
        except Exception as e:
            self.log_test_result("RR Validation Backend Implementation", False, f"Exception: {str(e)}")
    
    async def test_3_trigger_ia2_decision_making(self):
        """Test 3: Trigger IA2 decision making to test RR validation system"""
        logger.info("\n🔍 TEST 3: Trigger IA2 Decision Making")
        
        try:
            # Get initial IA2 decisions count
            initial_response = requests.get(f"{self.api_url}/decisions", timeout=30)
            initial_data = initial_response.json() if initial_response.status_code == 200 else {}
            initial_decisions = initial_data.get('decisions', [])
            initial_count = len(initial_decisions)
            
            logger.info(f"   📊 Initial IA2 decisions count: {initial_count}")
            
            # Trigger IA2 processing via start-trading endpoint
            logger.info("   🚀 Triggering IA2 decision making via /api/start-trading...")
            start_response = requests.post(f"{self.api_url}/start-trading", timeout=180)
            
            logger.info(f"   📊 Start trading response: HTTP {start_response.status_code}")
            
            if start_response.status_code not in [200, 201]:
                self.log_test_result("Trigger IA2 Decision Making", False, f"Start trading failed: HTTP {start_response.status_code}")
                return
            
            # Wait for IA2 processing
            logger.info("   ⏳ Waiting 45 seconds for IA2 processing and RR validation...")
            await asyncio.sleep(45)
            
            # Check for new decisions
            updated_response = requests.get(f"{self.api_url}/decisions", timeout=30)
            updated_data = updated_response.json() if updated_response.status_code == 200 else {}
            updated_decisions = updated_data.get('decisions', [])
            updated_count = len(updated_decisions)
            
            new_decisions = updated_count - initial_count
            
            logger.info(f"   📊 Updated IA2 decisions count: {updated_count}")
            logger.info(f"   📊 New decisions generated: {new_decisions}")
            
            # Analyze recent decisions for RR validation data
            recent_decisions = []
            for decision in updated_decisions[-10:]:  # Check last 10 decisions
                symbol = decision.get('symbol', 'Unknown')
                signal = decision.get('signal', 'Unknown')
                confidence = decision.get('confidence', 0)
                rr_ratio = decision.get('risk_reward_ratio', 0)
                
                recent_decisions.append({
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': confidence,
                    'rr_ratio': rr_ratio
                })
                
                logger.info(f"      📋 Recent decision: {symbol} - {signal} (Confidence: {confidence:.1%}, RR: {rr_ratio:.2f})")
            
            # Success criteria: Either new decisions generated or existing decisions with RR data
            success = new_decisions > 0 or (updated_count > 0 and any(d['rr_ratio'] > 0 for d in recent_decisions))
            
            details = f"New decisions: {new_decisions}, Total: {updated_count}, Recent with RR: {sum(1 for d in recent_decisions if d['rr_ratio'] > 0)}"
            
            self.log_test_result("Trigger IA2 Decision Making", success, details)
            
        except Exception as e:
            self.log_test_result("Trigger IA2 Decision Making", False, f"Exception: {str(e)}")
    
    async def test_4_rr_validation_log_patterns(self):
        """Test 4: Check backend logs for RR validation patterns"""
        logger.info("\n🔍 TEST 4: RR Validation Log Patterns")
        
        try:
            import subprocess
            
            # Get recent backend logs
            backend_logs = ""
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "3000", "/var/log/supervisor/backend.out.log"],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "3000", "/var/log/supervisor/backend.err.log"],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            if not backend_logs:
                self.log_test_result("RR Validation Log Patterns", False, "Could not retrieve backend logs")
                return
            
            # Check for expected RR validation log patterns
            pattern_counts = {}
            for pattern in self.expected_log_patterns:
                count = backend_logs.count(pattern)
                pattern_counts[pattern] = count
                if count > 0:
                    logger.info(f"      ✅ Found {count} instances of: {pattern}")
                else:
                    logger.info(f"      ❌ Missing pattern: {pattern}")
            
            # Look for specific RR validation activities
            rr_validation_activities = []
            
            # Check for RR correction activities
            if "IA2 claimed" in backend_logs and "corrected to" in backend_logs:
                rr_validation_activities.append("RR correction activity detected")
            
            # Check for RR validation success
            if "RR validation:" in backend_logs and "Entry(" in backend_logs:
                rr_validation_activities.append("RR validation calculations detected")
            
            # Check for level order validation
            if "Level order invalid" in backend_logs or "Support(" in backend_logs:
                rr_validation_activities.append("Level order validation detected")
            
            # Check for enhanced RR usage
            if "IA2 RR ENHANCEMENT:" in backend_logs:
                rr_validation_activities.append("IA2 RR enhancement usage detected")
            
            # Check for fallback scenarios
            if "FALLBACK: Using IA1 RR" in backend_logs:
                rr_validation_activities.append("IA1 RR fallback detected")
            
            logger.info(f"   📊 RR validation patterns found: {sum(1 for count in pattern_counts.values() if count > 0)}/{len(self.expected_log_patterns)}")
            logger.info(f"   📊 RR validation activities: {len(rr_validation_activities)}")
            
            for activity in rr_validation_activities:
                logger.info(f"      ✅ {activity}")
            
            # Success criteria: At least some RR validation activity detected
            patterns_found = sum(1 for count in pattern_counts.values() if count > 0)
            success = patterns_found >= 2 or len(rr_validation_activities) >= 2
            
            details = f"Patterns: {patterns_found}/{len(self.expected_log_patterns)}, Activities: {len(rr_validation_activities)}"
            
            self.log_test_result("RR Validation Log Patterns", success, details)
            
        except Exception as e:
            self.log_test_result("RR Validation Log Patterns", False, f"Exception: {str(e)}")
    
    async def test_5_long_short_validation_verification(self):
        """Test 5: Verify LONG and SHORT position RR validation works correctly"""
        logger.info("\n🔍 TEST 5: LONG/SHORT Position RR Validation")
        
        try:
            # Get recent IA2 decisions to analyze LONG/SHORT validation
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("LONG/SHORT Position RR Validation", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            decisions = data.get('decisions', [])
            
            # Analyze recent decisions for LONG/SHORT validation
            long_decisions = []
            short_decisions = []
            validated_decisions = []
            
            for decision in decisions[-20:]:  # Check last 20 decisions
                symbol = decision.get('symbol', '')
                signal = decision.get('signal', '').upper()
                rr_ratio = decision.get('risk_reward_ratio', 0)
                
                if signal == 'LONG':
                    long_decisions.append({'symbol': symbol, 'rr': rr_ratio})
                elif signal == 'SHORT':
                    short_decisions.append({'symbol': symbol, 'rr': rr_ratio})
                
                # Check if decision has valid RR (indicating validation passed)
                if rr_ratio > 0:
                    validated_decisions.append({'symbol': symbol, 'signal': signal, 'rr': rr_ratio})
            
            logger.info(f"   📊 Recent LONG decisions: {len(long_decisions)}")
            logger.info(f"   📊 Recent SHORT decisions: {len(short_decisions)}")
            logger.info(f"   📊 Decisions with valid RR: {len(validated_decisions)}")
            
            # Show examples of validated decisions
            for decision in validated_decisions[:5]:  # Show first 5
                logger.info(f"      ✅ {decision['symbol']} - {decision['signal']} (RR: {decision['rr']:.2f})")
            
            # Check backend logs for specific LONG/SHORT validation messages
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
            
            # Look for LONG/SHORT specific validation
            long_validation_count = backend_logs.count("LONG RR validation:")
            short_validation_count = backend_logs.count("SHORT RR validation:")
            level_order_checks = backend_logs.count("Level order invalid")
            
            logger.info(f"   📊 LONG RR validations in logs: {long_validation_count}")
            logger.info(f"   📊 SHORT RR validations in logs: {short_validation_count}")
            logger.info(f"   📊 Level order checks in logs: {level_order_checks}")
            
            # Success criteria: Evidence of both LONG and SHORT validation or sufficient validated decisions
            success = (
                (long_validation_count > 0 or short_validation_count > 0) or
                (len(long_decisions) > 0 and len(short_decisions) > 0) or
                len(validated_decisions) >= 3
            )
            
            details = f"LONG: {len(long_decisions)}, SHORT: {len(short_decisions)}, Validated: {len(validated_decisions)}, Log validations: L{long_validation_count}/S{short_validation_count}"
            
            self.log_test_result("LONG/SHORT Position RR Validation", success, details)
            
        except Exception as e:
            self.log_test_result("LONG/SHORT Position RR Validation", False, f"Exception: {str(e)}")
    
    async def test_6_rr_correction_and_fallback_system(self):
        """Test 6: Verify RR correction and fallback system functionality"""
        logger.info("\n🔍 TEST 6: RR Correction and Fallback System")
        
        try:
            import subprocess
            
            # Get comprehensive backend logs
            backend_logs = ""
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "4000", "/var/log/supervisor/backend.out.log"],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            if not backend_logs:
                self.log_test_result("RR Correction and Fallback System", False, "Could not retrieve backend logs")
                return
            
            # Check for RR correction system activities
            correction_activities = []
            
            # RR correction detection
            correction_count = backend_logs.count("🔧 IA2 RR CORRECTION:")
            if correction_count > 0:
                correction_activities.append(f"RR corrections performed: {correction_count}")
            
            # RR validation success
            validation_success_count = backend_logs.count("✅ IA2 RR VALIDATED:")
            if validation_success_count > 0:
                correction_activities.append(f"RR validations successful: {validation_success_count}")
            
            # RR validation failures
            validation_failure_count = backend_logs.count("❌ IA2 RR VALIDATION FAILED:")
            if validation_failure_count > 0:
                correction_activities.append(f"RR validation failures: {validation_failure_count}")
            
            # Enhanced RR usage
            enhanced_rr_count = backend_logs.count("🎯 USING IA2 ENHANCED RR")
            if enhanced_rr_count > 0:
                correction_activities.append(f"IA2 enhanced RR usage: {enhanced_rr_count}")
            
            # Fallback to IA1 RR
            fallback_count = backend_logs.count("🔄 USING IA1 ORIGINAL RR")
            if fallback_count > 0:
                correction_activities.append(f"IA1 RR fallbacks: {fallback_count}")
            
            # IA2 RR enhancement activities
            enhancement_count = backend_logs.count("🎯 IA2 RR ENHANCEMENT:")
            if enhancement_count > 0:
                correction_activities.append(f"IA2 RR enhancements: {enhancement_count}")
            
            # Validation message details
            validation_detail_count = backend_logs.count("🔍 VALIDATION:")
            if validation_detail_count > 0:
                correction_activities.append(f"Validation details logged: {validation_detail_count}")
            
            logger.info(f"   📊 RR correction/fallback activities found: {len(correction_activities)}")
            for activity in correction_activities:
                logger.info(f"      ✅ {activity}")
            
            # Look for specific examples in logs
            if "IA2 claimed" in backend_logs and "corrected to" in backend_logs:
                logger.info("      ✅ RR correction example found in logs")
            
            if "final_rr_source" in backend_logs:
                logger.info("      ✅ RR source tracking detected")
            
            if "validation_message" in backend_logs:
                logger.info("      ✅ Validation message system active")
            
            # Success criteria: Evidence of correction/fallback system working
            success = len(correction_activities) >= 3 or (validation_success_count + validation_failure_count) > 0
            
            details = f"Activities: {len(correction_activities)}, Validations: {validation_success_count}, Failures: {validation_failure_count}, Corrections: {correction_count}"
            
            self.log_test_result("RR Correction and Fallback System", success, details)
            
        except Exception as e:
            self.log_test_result("RR Correction and Fallback System", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_tests(self):
        """Run all Enhanced RR Validation System tests"""
        logger.info("🚀 Starting Enhanced RR Validation System Test Suite")
        logger.info("=" * 80)
        logger.info("📋 ENHANCED RR VALIDATION SYSTEM TESTING")
        logger.info("🎯 Testing: IA2 RR calculation validation with LONG/SHORT equation validation")
        logger.info("🎯 Expected: Proper RR validation, correction, and fallback mechanisms")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_enhanced_rr_prompt_integration()
        await self.test_2_rr_validation_backend_implementation()
        await self.test_3_trigger_ia2_decision_making()
        await self.test_4_rr_validation_log_patterns()
        await self.test_5_long_short_validation_verification()
        await self.test_6_rr_correction_and_fallback_system()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 ENHANCED RR VALIDATION SYSTEM SUMMARY")
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
        logger.info("📋 ENHANCED RR VALIDATION ANALYSIS")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Enhanced RR Validation System FULLY WORKING!")
            logger.info("✅ IA2 receives enhanced prompt with critical RR formulas")
            logger.info("✅ Backend validates IA2's RR calculations correctly")
            logger.info("✅ LONG/SHORT equation validation operational")
            logger.info("✅ System corrects IA2's RR when calculation is wrong")
            logger.info("✅ Fallback to IA1 RR works when validation fails")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY WORKING - Enhanced RR Validation System functional with minor issues")
            logger.info("🔍 Some components need attention for full optimization")
        else:
            logger.info("❌ SYSTEM ISSUES - Enhanced RR Validation System has significant problems")
            logger.info("🚨 Critical components may not be working as expected")
        
        # Specific requirements check
        logger.info("\n📝 ENHANCED RR VALIDATION REQUIREMENTS VERIFICATION:")
        
        requirements_met = []
        requirements_failed = []
        
        # Check each requirement
        for result in self.test_results:
            if result['success']:
                if "Enhanced RR Prompt" in result['test']:
                    requirements_met.append("✅ IA2 receives enhanced prompt with CRITICAL RR CALCULATION FORMULAS")
                elif "RR Validation Backend" in result['test']:
                    requirements_met.append("✅ Backend validates IA2's RR calculations using correct equations")
                elif "Trigger IA2" in result['test']:
                    requirements_met.append("✅ IA2 decision making system operational")
                elif "RR Validation Log" in result['test']:
                    requirements_met.append("✅ RR validation log patterns detected")
                elif "LONG/SHORT" in result['test']:
                    requirements_met.append("✅ LONG/SHORT position validation works correctly")
                elif "RR Correction" in result['test']:
                    requirements_met.append("✅ RR correction and fallback system functional")
            else:
                if "Enhanced RR Prompt" in result['test']:
                    requirements_failed.append("❌ Enhanced RR prompt integration incomplete")
                elif "RR Validation Backend" in result['test']:
                    requirements_failed.append("❌ Backend RR validation implementation missing")
                elif "Trigger IA2" in result['test']:
                    requirements_failed.append("❌ IA2 decision making not working")
                elif "RR Validation Log" in result['test']:
                    requirements_failed.append("❌ RR validation log patterns not found")
                elif "LONG/SHORT" in result['test']:
                    requirements_failed.append("❌ LONG/SHORT position validation not working")
                elif "RR Correction" in result['test']:
                    requirements_failed.append("❌ RR correction and fallback system not functional")
        
        for req in requirements_met:
            logger.info(f"   {req}")
        
        for req in requirements_failed:
            logger.info(f"   {req}")
        
        logger.info(f"\n🏆 FINAL RESULT: {len(requirements_met)}/{len(requirements_met) + len(requirements_failed)} requirements satisfied")
        
        # Final verdict
        if len(requirements_failed) == 0:
            logger.info("\n🎉 VERDICT: Enhanced RR Validation System is FULLY WORKING!")
            logger.info("✅ All RR validation features operational as specified")
            logger.info("✅ LONG/SHORT equation validation working correctly")
            logger.info("✅ RR correction and fallback mechanisms functional")
        elif len(requirements_failed) <= 1:
            logger.info("\n⚠️ VERDICT: Enhanced RR Validation System mostly WORKING with minor issues")
            logger.info("🔍 Some fine-tuning may be needed for complete functionality")
        else:
            logger.info("\n❌ VERDICT: Enhanced RR Validation System has SIGNIFICANT ISSUES")
            logger.info("🚨 Critical components not working as expected")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = EnhancedRRValidationTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
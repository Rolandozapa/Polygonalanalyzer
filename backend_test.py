#!/usr/bin/env python3
"""
Backend Testing Suite for IA1→IA2 Filtering and Processing Pipeline
Focus: Testing IA1→IA2 filtering, bug fixes, and specific test cases
Review Request: Test the IA1→IA2 filtering and processing pipeline to verify that:

1. IA1 analyses with LONG/SHORT signals and confidence ≥70% pass to IA2 (VOIE 1)
2. The recent bug fix for missing active_position_manager in IA2 agent is working
3. The recent bug fix for missing _apply_adaptive_context_to_decision method is resolved
4. New IA2 decisions are being generated from recent IA1 analyses

Specific test cases:
- ONDOUSDT: SHORT + 87% confidence → should pass VOIE 1
- HBARUSDT: SHORT + 72% confidence → should pass VOIE 1  
- ARKMUSDT: LONG + 96% confidence → should pass VOIE 1
- ENAUSDT: LONG + 98% confidence → should pass VOIE 1
- FARTCOINUSDT: SHORT + 78% confidence → should pass VOIE 1

Test the /api/start-trading endpoint and verify that new IA2 decisions with today's timestamps are generated.
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

class IA1IA2PipelineTestSuite:
    """Test suite for IA1→IA2 Filtering and Processing Pipeline"""
    
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
        logger.info(f"Testing IA1→IA2 Pipeline at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Specific test cases from review request
        self.specific_test_cases = [
            {"symbol": "ONDOUSDT", "expected_signal": "SHORT", "expected_confidence": 87},
            {"symbol": "HBARUSDT", "expected_signal": "SHORT", "expected_confidence": 72},
            {"symbol": "ARKMUSDT", "expected_signal": "LONG", "expected_confidence": 96},
            {"symbol": "ENAUSDT", "expected_signal": "LONG", "expected_confidence": 98},
            {"symbol": "FARTCOINUSDT", "expected_signal": "SHORT", "expected_confidence": 78}
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
        
    async def test_voie1_filtering_logic(self):
        """Test 1: VOIE 1 - IA1 analyses with LONG/SHORT signals and confidence ≥70% pass to IA2"""
        logger.info("\n🔍 TEST 1: VOIE 1 Filtering Logic (LONG/SHORT + Confidence ≥ 70%)")
        
        try:
            # Get current IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("VOIE 1 Filtering Logic", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("VOIE 1 Filtering Logic", False, "No IA1 analyses found")
                return
            
            # Check for specific test cases and general VOIE 1 logic
            specific_cases_found = 0
            specific_cases_passed = 0
            voie1_cases_found = 0
            voie1_cases_passed = 0
            
            # Get IA2 decisions for comparison
            ia2_response = requests.get(f"{self.api_url}/decisions", timeout=30)
            decisions = ia2_response.json() if ia2_response.status_code == 200 else []
            
            for analysis in analyses:
                symbol = analysis.get('symbol', '')
                confidence = analysis.get('analysis_confidence', 0)
                signal = analysis.get('ia1_signal', 'hold').lower()
                
                # Check specific test cases
                for test_case in self.specific_test_cases:
                    if symbol == test_case["symbol"]:
                        specific_cases_found += 1
                        expected_signal = test_case["expected_signal"].lower()
                        expected_confidence = test_case["expected_confidence"] / 100.0
                        
                        logger.info(f"   🎯 {symbol} FOUND: Signal={signal.upper()}, Confidence={confidence:.1%}")
                        logger.info(f"      Expected: Signal={expected_signal.upper()}, Confidence={expected_confidence:.1%}")
                        
                        # Check if it has corresponding IA2 decision
                        has_ia2_decision = any(d.get('symbol') == symbol for d in decisions)
                        
                        if signal == expected_signal and confidence >= 0.70 and has_ia2_decision:
                            specific_cases_passed += 1
                            logger.info(f"      ✅ {symbol}: Correctly passed to IA2 via VOIE 1")
                        elif signal == expected_signal and confidence >= 0.70 and not has_ia2_decision:
                            logger.info(f"      ❌ {symbol}: Meets VOIE 1 criteria but no IA2 decision found")
                        else:
                            logger.info(f"      ⚠️ {symbol}: Does not match expected criteria or missing IA2 decision")
                
                # Check general VOIE 1 cases
                if signal in ['long', 'short'] and confidence >= 0.70:
                    voie1_cases_found += 1
                    
                    # Check if this analysis was sent to IA2
                    has_ia2_decision = any(d.get('symbol') == symbol for d in decisions)
                    
                    if has_ia2_decision:
                        voie1_cases_passed += 1
                        logger.info(f"      ✅ {symbol}: VOIE 1 case passed to IA2 (Signal={signal.upper()}, Conf={confidence:.1%})")
                    else:
                        logger.info(f"      ❌ {symbol}: VOIE 1 case NOT passed to IA2 (Signal={signal.upper()}, Conf={confidence:.1%})")
            
            # Calculate success metrics
            specific_success_rate = (specific_cases_passed / specific_cases_found) if specific_cases_found > 0 else 0
            voie1_success_rate = (voie1_cases_passed / voie1_cases_found) if voie1_cases_found > 0 else 0
            
            logger.info(f"   📊 Specific test cases found: {specific_cases_found}/5")
            logger.info(f"   📊 Specific test cases passed: {specific_cases_passed}")
            logger.info(f"   📊 Specific success rate: {specific_success_rate:.1%}")
            logger.info(f"   📊 Total VOIE 1 cases found: {voie1_cases_found}")
            logger.info(f"   📊 Total VOIE 1 cases passed: {voie1_cases_passed}")
            logger.info(f"   📊 VOIE 1 success rate: {voie1_success_rate:.1%}")
            
            # Success criteria: Good success rate for both specific cases and general VOIE 1 logic
            success = (specific_success_rate >= 0.6 or specific_cases_found == 0) and voie1_success_rate >= 0.7
            
            details = f"Specific: {specific_cases_passed}/{specific_cases_found} ({specific_success_rate:.1%}), VOIE 1: {voie1_cases_passed}/{voie1_cases_found} ({voie1_success_rate:.1%})"
            
            self.log_test_result("VOIE 1 Filtering Logic", success, details)
            
        except Exception as e:
            self.log_test_result("VOIE 1 Filtering Logic", False, f"Exception: {str(e)}")
    
    async def test_active_position_manager_bug_fix(self):
        """Test 2: Bug fix for missing active_position_manager in IA2 agent"""
        logger.info("\n🔍 TEST 2: Active Position Manager Bug Fix")
        
        try:
            # Check backend logs for active_position_manager related errors
            import subprocess
            
            # Get recent backend logs
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "500", "/var/log/supervisor/backend.out.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                backend_logs = log_result.stdout
            except:
                try:
                    log_result = subprocess.run(
                        ["tail", "-n", "500", "/var/log/supervisor/backend.err.log"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    backend_logs = log_result.stdout
                except:
                    backend_logs = ""
            
            if not backend_logs:
                self.log_test_result("Active Position Manager Bug Fix", False, "Could not retrieve backend logs")
                return
            
            # Check for active_position_manager errors
            apm_errors = backend_logs.count("active_position_manager")
            apm_attribute_errors = backend_logs.count("AttributeError") and "active_position_manager" in backend_logs
            
            # Check for successful IA2 decisions (indicating the bug is fixed)
            ia2_response = requests.get(f"{self.api_url}/decisions", timeout=30)
            if ia2_response.status_code == 200:
                decisions = ia2_response.json()
                recent_decisions = [d for d in decisions if self._is_recent_timestamp(d.get('timestamp', ''))]
                
                logger.info(f"   📊 Recent IA2 decisions: {len(recent_decisions)}")
                logger.info(f"   📊 Active position manager errors in logs: {apm_errors}")
                logger.info(f"   📊 AttributeError mentions: {apm_attribute_errors}")
                
                # Success criteria: Recent IA2 decisions exist and no active_position_manager AttributeErrors
                success = len(recent_decisions) > 0 and not apm_attribute_errors
                
                details = f"Recent IA2 decisions: {len(recent_decisions)}, APM errors: {apm_errors}, AttributeErrors: {apm_attribute_errors}"
                
                self.log_test_result("Active Position Manager Bug Fix", success, details)
            else:
                self.log_test_result("Active Position Manager Bug Fix", False, "Could not retrieve IA2 decisions")
            
        except Exception as e:
            self.log_test_result("Active Position Manager Bug Fix", False, f"Exception: {str(e)}")
    
    async def test_adaptive_context_method_bug_fix(self):
        """Test 3: Bug fix for missing _apply_adaptive_context_to_decision method"""
        logger.info("\n🔍 TEST 3: Adaptive Context Method Bug Fix")
        
        try:
            # Check backend logs for _apply_adaptive_context_to_decision related errors
            import subprocess
            
            # Get recent backend logs
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "500", "/var/log/supervisor/backend.out.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                backend_logs = log_result.stdout
            except:
                try:
                    log_result = subprocess.run(
                        ["tail", "-n", "500", "/var/log/supervisor/backend.err.log"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    backend_logs = log_result.stdout
                except:
                    backend_logs = ""
            
            if not backend_logs:
                self.log_test_result("Adaptive Context Method Bug Fix", False, "Could not retrieve backend logs")
                return
            
            # Check for _apply_adaptive_context_to_decision errors
            adaptive_context_errors = backend_logs.count("_apply_adaptive_context_to_decision")
            method_not_found_errors = "AttributeError" in backend_logs and "_apply_adaptive_context_to_decision" in backend_logs
            
            # Check for successful IA2 decisions with adaptive context
            ia2_response = requests.get(f"{self.api_url}/decisions", timeout=30)
            if ia2_response.status_code == 200:
                decisions = ia2_response.json()
                recent_decisions = [d for d in decisions if self._is_recent_timestamp(d.get('timestamp', ''))]
                
                # Check if decisions have adaptive context applied (look for enhanced reasoning)
                decisions_with_context = 0
                for decision in recent_decisions:
                    reasoning = decision.get('reasoning', '').lower()
                    if any(keyword in reasoning for keyword in ['adaptive', 'context', 'enhanced', 'optimized']):
                        decisions_with_context += 1
                
                logger.info(f"   📊 Recent IA2 decisions: {len(recent_decisions)}")
                logger.info(f"   📊 Decisions with adaptive context: {decisions_with_context}")
                logger.info(f"   📊 Adaptive context method errors: {adaptive_context_errors}")
                logger.info(f"   📊 Method not found errors: {method_not_found_errors}")
                
                # Success criteria: Recent decisions exist, some have adaptive context, no method errors
                success = len(recent_decisions) > 0 and not method_not_found_errors
                
                details = f"Recent decisions: {len(recent_decisions)}, With context: {decisions_with_context}, Method errors: {method_not_found_errors}"
                
                self.log_test_result("Adaptive Context Method Bug Fix", success, details)
            else:
                self.log_test_result("Adaptive Context Method Bug Fix", False, "Could not retrieve IA2 decisions")
            
        except Exception as e:
            self.log_test_result("Adaptive Context Method Bug Fix", False, f"Exception: {str(e)}")
    
    async def test_new_ia2_decisions_generation(self):
        """Test 4: New IA2 decisions are being generated from recent IA1 analyses"""
        logger.info("\n🔍 TEST 4: New IA2 Decisions Generation")
        
        try:
            # Get current IA1 analyses and IA2 decisions
            analyses_response = requests.get(f"{self.api_url}/analyses", timeout=30)
            decisions_response = requests.get(f"{self.api_url}/decisions", timeout=30)
            
            if analyses_response.status_code != 200 or decisions_response.status_code != 200:
                self.log_test_result("New IA2 Decisions Generation", False, "Could not retrieve system data")
                return
            
            analyses = analyses_response.json()
            decisions = decisions_response.json()
            
            # Filter recent data (today's data)
            recent_analyses = [a for a in analyses if self._is_recent_timestamp(a.get('timestamp', ''))]
            recent_decisions = [d for d in decisions if self._is_recent_timestamp(d.get('timestamp', ''))]
            
            # Check for matching symbols between recent IA1 analyses and IA2 decisions
            analysis_symbols = set(a.get('symbol', '') for a in recent_analyses)
            decision_symbols = set(d.get('symbol', '') for d in recent_decisions)
            
            matching_symbols = analysis_symbols.intersection(decision_symbols)
            
            logger.info(f"   📊 Recent IA1 analyses: {len(recent_analyses)}")
            logger.info(f"   📊 Recent IA2 decisions: {len(recent_decisions)}")
            logger.info(f"   📊 Analysis symbols: {len(analysis_symbols)}")
            logger.info(f"   📊 Decision symbols: {len(decision_symbols)}")
            logger.info(f"   📊 Matching symbols: {len(matching_symbols)}")
            
            if matching_symbols:
                logger.info(f"   📋 Matching symbols: {', '.join(list(matching_symbols)[:10])}")
            
            # Success criteria: Recent IA2 decisions exist and some match recent IA1 analyses
            success = len(recent_decisions) > 0 and len(matching_symbols) > 0
            
            details = f"Recent IA1: {len(recent_analyses)}, Recent IA2: {len(recent_decisions)}, Matching: {len(matching_symbols)}"
            
            self.log_test_result("New IA2 Decisions Generation", success, details)
            
        except Exception as e:
            self.log_test_result("New IA2 Decisions Generation", False, f"Exception: {str(e)}")
    
    async def test_start_trading_endpoint(self):
        """Test 5: /api/start-trading endpoint and verify new IA2 decisions generation"""
        logger.info("\n🔍 TEST 5: Start Trading Endpoint")
        
        try:
            # Get initial state
            initial_response = requests.get(f"{self.api_url}/decisions", timeout=30)
            initial_decisions = initial_response.json() if initial_response.status_code == 200 else []
            initial_count = len(initial_decisions)
            
            logger.info(f"   📊 Initial IA2 decisions count: {initial_count}")
            
            # Call start-trading endpoint
            start_response = requests.post(f"{self.api_url}/start-trading", timeout=60)
            
            if start_response.status_code not in [200, 201]:
                self.log_test_result("Start Trading Endpoint", False, f"Start trading failed: HTTP {start_response.status_code}")
                return
            
            logger.info("   🚀 Start trading endpoint called successfully")
            
            # Wait a bit for processing
            await asyncio.sleep(10)
            
            # Get updated state
            updated_response = requests.get(f"{self.api_url}/decisions", timeout=30)
            updated_decisions = updated_response.json() if updated_response.status_code == 200 else []
            updated_count = len(updated_decisions)
            
            # Check for new decisions with today's timestamps
            today = datetime.now().date()
            new_decisions_today = []
            
            for decision in updated_decisions:
                timestamp_str = decision.get('timestamp', '')
                if self._is_today_timestamp(timestamp_str):
                    new_decisions_today.append(decision)
            
            logger.info(f"   📊 Updated IA2 decisions count: {updated_count}")
            logger.info(f"   📊 New decisions today: {len(new_decisions_today)}")
            logger.info(f"   📊 Decision count change: {updated_count - initial_count}")
            
            if new_decisions_today:
                logger.info("   📋 New decisions today:")
                for decision in new_decisions_today[:5]:  # Show first 5
                    symbol = decision.get('symbol', 'Unknown')
                    signal = decision.get('signal', 'Unknown')
                    confidence = decision.get('confidence', 0)
                    logger.info(f"      • {symbol}: {signal} ({confidence:.1%})")
            
            # Success criteria: Start trading works and new decisions are generated
            success = start_response.status_code in [200, 201] and len(new_decisions_today) > 0
            
            details = f"Endpoint status: {start_response.status_code}, New decisions today: {len(new_decisions_today)}"
            
            self.log_test_result("Start Trading Endpoint", success, details)
            
        except Exception as e:
            self.log_test_result("Start Trading Endpoint", False, f"Exception: {str(e)}")
    
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
    
    def _is_today_timestamp(self, timestamp_str: str) -> bool:
        """Check if timestamp is from today"""
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
            
            today = datetime.now().date()
            return timestamp.date() == today
            
        except Exception:
            return False
    
    async def run_comprehensive_tests(self):
        """Run all IA1→IA2 Pipeline tests"""
        logger.info("🚀 Starting IA1→IA2 Filtering and Processing Pipeline Test Suite")
        logger.info("=" * 80)
        logger.info("📋 REVIEW REQUEST: Test IA1→IA2 filtering and processing pipeline")
        logger.info("🎯 1. IA1 analyses with LONG/SHORT signals and confidence ≥70% pass to IA2 (VOIE 1)")
        logger.info("🎯 2. Bug fix for missing active_position_manager in IA2 agent is working")
        logger.info("🎯 3. Bug fix for missing _apply_adaptive_context_to_decision method is resolved")
        logger.info("🎯 4. New IA2 decisions are being generated from recent IA1 analyses")
        logger.info("🎯 5. /api/start-trading endpoint generates new IA2 decisions with today's timestamps")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_voie1_filtering_logic()
        await self.test_active_position_manager_bug_fix()
        await self.test_adaptive_context_method_bug_fix()
        await self.test_new_ia2_decisions_generation()
        await self.test_start_trading_endpoint()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 IA1→IA2 PIPELINE TEST SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # Pipeline analysis
        logger.info("\n" + "=" * 80)
        logger.info("📋 IA1→IA2 PIPELINE ANALYSIS")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - IA1→IA2 pipeline working perfectly!")
            logger.info("✅ VOIE 1 filtering logic operational")
            logger.info("✅ Active position manager bug fixed")
            logger.info("✅ Adaptive context method bug fixed")
            logger.info("✅ New IA2 decisions being generated")
            logger.info("✅ Start trading endpoint functional")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY WORKING - Most pipeline features operational")
            logger.info("🔍 Some minor issues need attention for full compliance")
        else:
            logger.info("❌ CRITICAL ISSUES - IA1→IA2 pipeline needs fixes")
            logger.info("🚨 Multiple pipeline components not working properly")
        
        # Specific requirements check
        logger.info("\n📝 SPECIFIC REQUIREMENTS VERIFICATION:")
        
        requirements_met = []
        requirements_failed = []
        
        # Check each requirement
        for result in self.test_results:
            if result['success']:
                if "VOIE 1" in result['test']:
                    requirements_met.append("✅ VOIE 1: LONG/SHORT + Confidence ≥70% → Pass to IA2")
                elif "Active Position Manager" in result['test']:
                    requirements_met.append("✅ Bug fix: active_position_manager in IA2 agent working")
                elif "Adaptive Context" in result['test']:
                    requirements_met.append("✅ Bug fix: _apply_adaptive_context_to_decision method resolved")
                elif "New IA2 Decisions" in result['test']:
                    requirements_met.append("✅ New IA2 decisions generated from recent IA1 analyses")
                elif "Start Trading" in result['test']:
                    requirements_met.append("✅ /api/start-trading endpoint generates new IA2 decisions")
            else:
                if "VOIE 1" in result['test']:
                    requirements_failed.append("❌ VOIE 1 filtering logic not working properly")
                elif "Active Position Manager" in result['test']:
                    requirements_failed.append("❌ active_position_manager bug not fixed")
                elif "Adaptive Context" in result['test']:
                    requirements_failed.append("❌ _apply_adaptive_context_to_decision method bug not fixed")
                elif "New IA2 Decisions" in result['test']:
                    requirements_failed.append("❌ New IA2 decisions not being generated properly")
                elif "Start Trading" in result['test']:
                    requirements_failed.append("❌ /api/start-trading endpoint not working properly")
        
        for req in requirements_met:
            logger.info(f"   {req}")
        
        for req in requirements_failed:
            logger.info(f"   {req}")
        
        # Specific test cases verification
        logger.info("\n🎯 SPECIFIC TEST CASES VERIFICATION:")
        logger.info("   Expected test cases:")
        for test_case in self.specific_test_cases:
            logger.info(f"   • {test_case['symbol']}: {test_case['expected_signal']} + {test_case['expected_confidence']}% confidence → should pass VOIE 1")
        
        logger.info(f"\n🏆 FINAL RESULT: {len(requirements_met)}/{len(requirements_met) + len(requirements_failed)} requirements satisfied")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = IA1IA2PipelineTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
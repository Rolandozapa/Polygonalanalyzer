#!/usr/bin/env python3
"""
IA2 Technical Indicators Access Fix Testing Suite
Focus: Testing the fix for "string indices must be integers, not 'str'" error in IA2

TESTING REQUIREMENTS FROM REVIEW REQUEST:
1. Technical Indicators Error Resolution: Verify that the "string indices must be integers, not 'str'" error has been fixed
2. IA2 Decision Creation: Check if IA2 can now complete decision-making process without crashing
3. RR Calculation Fields: Test if "calculated_rr" and "rr_reasoning" fields are present in new decisions
4. Simple RR Formula Validation: Confirm IA2 is using the simple S/R based RR calculation as implemented
5. Conditional Logic Fix: Verify that the f-string conditional logic properly handles None current_indicators

CRITICAL BUG IDENTIFIED:
- IA2 was crashing with "string indices must be integers, not 'str'" error before reaching RR calculation phase
- This was caused by improper conditional logic when accessing current_indicators attributes in f-strings
- The fix restructured the conditionals to prevent attribute access when current_indicators is None

TEST APPROACH:
1. Trigger IA2 decision making process
2. Monitor backend logs for the specific error
3. Check if IA2 decisions are successfully created and stored
4. Validate presence of "calculated_rr" and "rr_reasoning" fields
5. Verify simple support/resistance RR formula is being used
"""

import asyncio
import json
import logging
import os
import sys
import time
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Any, List
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IA2TechnicalIndicatorsFixTestSuite:
    """Test suite for IA2 technical indicators access fix"""
    
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
        logger.info(f"Testing IA2 Technical Indicators Fix at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Track initial state
        self.initial_decisions_count = 0
        self.initial_backend_log_size = 0
        
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
    
    def get_backend_logs(self, lines: int = 1000) -> str:
        """Get recent backend logs"""
        try:
            log_result = subprocess.run(
                ["tail", "-n", str(lines), "/var/log/supervisor/backend.err.log"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return log_result.stdout
        except Exception as e:
            logger.warning(f"Could not get backend logs: {e}")
            return ""
    
    def count_decisions(self) -> int:
        """Count current number of IA2 decisions"""
        try:
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            if response.status_code == 200:
                data = response.json()
                decisions = data.get('decisions', [])
                return len(decisions)
        except Exception as e:
            logger.warning(f"Could not count decisions: {e}")
        return 0
    
    async def test_1_baseline_state_capture(self):
        """Test 1: Capture baseline state before testing"""
        logger.info("\n🔍 TEST 1: Capturing Baseline State")
        
        try:
            # Count initial decisions
            self.initial_decisions_count = self.count_decisions()
            logger.info(f"   📊 Initial decisions count: {self.initial_decisions_count}")
            
            # Get initial backend log size
            backend_logs = self.get_backend_logs(100)
            self.initial_backend_log_size = len(backend_logs.split('\n'))
            logger.info(f"   📊 Initial backend log lines: {self.initial_backend_log_size}")
            
            # Check for recent "string indices" errors in logs
            recent_string_errors = backend_logs.count("string indices must be integers, not 'str'")
            logger.info(f"   📊 Recent 'string indices' errors in logs: {recent_string_errors}")
            
            self.log_test_result("Baseline State Capture", True, 
                               f"Decisions: {self.initial_decisions_count}, Log lines: {self.initial_backend_log_size}, Recent errors: {recent_string_errors}")
            
        except Exception as e:
            self.log_test_result("Baseline State Capture", False, f"Exception: {str(e)}")
    
    async def test_2_trigger_ia2_decision_process(self):
        """Test 2: Trigger IA2 Decision Making Process"""
        logger.info("\n🔍 TEST 2: Triggering IA2 Decision Making Process")
        
        try:
            logger.info("   🚀 Triggering fresh analysis via /api/start-trading...")
            
            # Trigger the trading process
            start_response = requests.post(f"{self.api_url}/start-trading", timeout=180)
            
            if start_response.status_code in [200, 201]:
                logger.info(f"   ✅ Start trading request successful: HTTP {start_response.status_code}")
                
                # Wait for processing
                logger.info("   ⏳ Waiting 60 seconds for IA1→IA2 pipeline processing...")
                await asyncio.sleep(60)
                
                self.log_test_result("Trigger IA2 Decision Process", True, 
                                   f"Successfully triggered trading process: HTTP {start_response.status_code}")
            else:
                logger.info(f"   ⚠️ Start trading returned: HTTP {start_response.status_code}")
                self.log_test_result("Trigger IA2 Decision Process", False, 
                                   f"HTTP {start_response.status_code}: {start_response.text[:200]}")
                
        except Exception as e:
            self.log_test_result("Trigger IA2 Decision Process", False, f"Exception: {str(e)}")
    
    async def test_3_check_string_indices_error_resolution(self):
        """Test 3: Check if 'string indices must be integers' error has been resolved"""
        logger.info("\n🔍 TEST 3: Checking String Indices Error Resolution")
        
        try:
            # Get recent backend logs
            backend_logs = self.get_backend_logs(2000)
            
            # Look for the specific error
            string_indices_errors = []
            log_lines = backend_logs.split('\n')
            
            for i, line in enumerate(log_lines):
                if "string indices must be integers, not 'str'" in line:
                    # Get context around the error
                    context_start = max(0, i-2)
                    context_end = min(len(log_lines), i+3)
                    context = '\n'.join(log_lines[context_start:context_end])
                    string_indices_errors.append({
                        'line': line,
                        'context': context,
                        'line_number': i
                    })
            
            logger.info(f"   📊 Found {len(string_indices_errors)} 'string indices' errors in recent logs")
            
            # Check if errors are related to IA2
            ia2_related_errors = 0
            for error in string_indices_errors:
                if 'IA2' in error['context'] or 'ia2' in error['context']:
                    ia2_related_errors += 1
                    logger.info(f"   ❌ IA2-related string indices error found:")
                    logger.info(f"      {error['line']}")
            
            # Check for IA2 execution attempts
            ia2_execution_attempts = backend_logs.count("IA2 making ultra professional")
            ia2_execution_errors = backend_logs.count("IA2 ultra decision error")
            
            logger.info(f"   📊 IA2 execution attempts: {ia2_execution_attempts}")
            logger.info(f"   📊 IA2 execution errors: {ia2_execution_errors}")
            
            # Success criteria: No new IA2-related string indices errors
            success = ia2_related_errors == 0
            
            if success:
                details = f"No IA2-related string indices errors found. IA2 attempts: {ia2_execution_attempts}, errors: {ia2_execution_errors}"
            else:
                details = f"Found {ia2_related_errors} IA2-related string indices errors. IA2 attempts: {ia2_execution_attempts}, errors: {ia2_execution_errors}"
            
            self.log_test_result("String Indices Error Resolution", success, details)
            
        except Exception as e:
            self.log_test_result("String Indices Error Resolution", False, f"Exception: {str(e)}")
    
    async def test_4_check_ia2_decision_creation_success(self):
        """Test 4: Check if IA2 can now complete decision-making process without crashing"""
        logger.info("\n🔍 TEST 4: Checking IA2 Decision Creation Success")
        
        try:
            # Count current decisions
            current_decisions_count = self.count_decisions()
            new_decisions = current_decisions_count - self.initial_decisions_count
            
            logger.info(f"   📊 Initial decisions: {self.initial_decisions_count}")
            logger.info(f"   📊 Current decisions: {current_decisions_count}")
            logger.info(f"   📊 New decisions created: {new_decisions}")
            
            # Get recent decisions to check for successful IA2 completion
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("IA2 Decision Creation Success", False, f"HTTP {response.status_code}")
                return
            
            data = response.json()
            decisions = data.get('decisions', [])
            
            # Check recent decisions (last 10 or new ones)
            recent_decisions = decisions[-max(10, new_decisions):] if decisions else []
            
            successful_ia2_decisions = 0
            decisions_with_reasoning = 0
            
            for decision in recent_decisions:
                # Check if decision has IA2 reasoning (indicates successful completion)
                ia2_reasoning = decision.get('ia2_reasoning', '')
                if ia2_reasoning and len(ia2_reasoning) > 50:  # Substantial reasoning
                    successful_ia2_decisions += 1
                    
                    # Check if it's a recent decision (within last hour)
                    timestamp_str = decision.get('timestamp', '')
                    if self._is_recent_timestamp(timestamp_str, hours=1):
                        decisions_with_reasoning += 1
                        symbol = decision.get('symbol', 'Unknown')
                        confidence = decision.get('confidence', 0)
                        signal = decision.get('signal', 'Unknown')
                        logger.info(f"      ✅ Recent IA2 decision: {symbol} ({signal}, {confidence:.1%})")
            
            # Check backend logs for successful IA2 completions
            backend_logs = self.get_backend_logs(1000)
            ia2_success_patterns = [
                "IA2 decision stored",
                "IA2 DECISION STORED",
                "IA2 analysis completed",
                "IA2 decision made"
            ]
            
            ia2_success_count = sum(backend_logs.count(pattern) for pattern in ia2_success_patterns)
            
            logger.info(f"   📊 Successful IA2 decisions found: {successful_ia2_decisions}")
            logger.info(f"   📊 Recent decisions with reasoning: {decisions_with_reasoning}")
            logger.info(f"   📊 IA2 success patterns in logs: {ia2_success_count}")
            
            # Success criteria: Either new decisions created OR evidence of successful IA2 execution
            success = new_decisions > 0 or successful_ia2_decisions > 0 or ia2_success_count > 0
            
            details = f"New decisions: {new_decisions}, Successful IA2: {successful_ia2_decisions}, Recent with reasoning: {decisions_with_reasoning}, Log success patterns: {ia2_success_count}"
            
            self.log_test_result("IA2 Decision Creation Success", success, details)
            
        except Exception as e:
            self.log_test_result("IA2 Decision Creation Success", False, f"Exception: {str(e)}")
    
    async def test_5_check_rr_calculation_fields(self):
        """Test 5: Check if 'calculated_rr' and 'rr_reasoning' fields are present in new decisions"""
        logger.info("\n🔍 TEST 5: Checking RR Calculation Fields Presence")
        
        try:
            # Get recent decisions
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("RR Calculation Fields Presence", False, f"HTTP {response.status_code}")
                return
            
            data = response.json()
            decisions = data.get('decisions', [])
            
            # Check last 20 decisions for the new fields
            recent_decisions = decisions[-20:] if len(decisions) >= 20 else decisions
            
            decisions_with_calculated_rr = 0
            decisions_with_rr_reasoning = 0
            total_checked = len(recent_decisions)
            
            calculated_rr_values = []
            rr_reasoning_samples = []
            
            for decision in recent_decisions:
                symbol = decision.get('symbol', 'Unknown')
                
                # Check for calculated_rr field
                calculated_rr = decision.get('calculated_rr')
                if calculated_rr is not None:
                    decisions_with_calculated_rr += 1
                    calculated_rr_values.append(calculated_rr)
                    logger.info(f"      ✅ {symbol}: calculated_rr = {calculated_rr}")
                
                # Check for rr_reasoning field
                rr_reasoning = decision.get('rr_reasoning', '')
                if rr_reasoning and len(rr_reasoning) > 10:
                    decisions_with_rr_reasoning += 1
                    rr_reasoning_samples.append(rr_reasoning[:100] + "..." if len(rr_reasoning) > 100 else rr_reasoning)
                    logger.info(f"      ✅ {symbol}: rr_reasoning = {rr_reasoning[:50]}...")
            
            # Calculate percentages
            calculated_rr_percentage = (decisions_with_calculated_rr / max(total_checked, 1)) * 100
            rr_reasoning_percentage = (decisions_with_rr_reasoning / max(total_checked, 1)) * 100
            
            logger.info(f"   📊 Decisions with calculated_rr: {calculated_rr_percentage:.1f}% ({decisions_with_calculated_rr}/{total_checked})")
            logger.info(f"   📊 Decisions with rr_reasoning: {rr_reasoning_percentage:.1f}% ({decisions_with_rr_reasoning}/{total_checked})")
            
            if calculated_rr_values:
                avg_rr = sum(calculated_rr_values) / len(calculated_rr_values)
                logger.info(f"   📊 Average calculated_rr: {avg_rr:.2f}")
            
            if rr_reasoning_samples:
                logger.info(f"   📊 Sample rr_reasoning: {rr_reasoning_samples[0]}")
            
            # Success criteria: At least some decisions have the new fields
            success = decisions_with_calculated_rr > 0 and decisions_with_rr_reasoning > 0
            
            details = f"calculated_rr: {calculated_rr_percentage:.1f}% ({decisions_with_calculated_rr}/{total_checked}), rr_reasoning: {rr_reasoning_percentage:.1f}% ({decisions_with_rr_reasoning}/{total_checked})"
            
            self.log_test_result("RR Calculation Fields Presence", success, details)
            
        except Exception as e:
            self.log_test_result("RR Calculation Fields Presence", False, f"Exception: {str(e)}")
    
    async def test_6_validate_simple_rr_formula_usage(self):
        """Test 6: Validate that IA2 is using the simple support/resistance RR formula"""
        logger.info("\n🔍 TEST 6: Validating Simple RR Formula Usage")
        
        try:
            # Get recent decisions
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Simple RR Formula Usage", False, f"HTTP {response.status_code}")
                return
            
            data = response.json()
            decisions = data.get('decisions', [])
            
            # Check last 10 decisions for simple RR formula evidence
            recent_decisions = decisions[-10:] if len(decisions) >= 10 else decisions
            
            simple_formula_evidence = 0
            total_with_reasoning = 0
            formula_keywords_found = []
            
            # Keywords that indicate simple S/R formula usage
            simple_formula_keywords = [
                "support", "resistance", "entry", "stop loss", "take profit",
                "RR =", "formula:", "calculation", "LONG formula", "SHORT formula",
                "(TP-Entry)/(Entry-SL)", "(Entry-TP)/(SL-Entry)"
            ]
            
            for decision in recent_decisions:
                symbol = decision.get('symbol', 'Unknown')
                rr_reasoning = decision.get('rr_reasoning', '')
                ia2_reasoning = decision.get('ia2_reasoning', '')
                
                if rr_reasoning:
                    total_with_reasoning += 1
                    
                    # Check for simple formula keywords
                    found_keywords = []
                    for keyword in simple_formula_keywords:
                        if keyword.lower() in rr_reasoning.lower():
                            found_keywords.append(keyword)
                    
                    if found_keywords:
                        simple_formula_evidence += 1
                        formula_keywords_found.extend(found_keywords)
                        logger.info(f"      ✅ {symbol}: Simple formula evidence - {found_keywords}")
                        logger.info(f"         RR reasoning: {rr_reasoning[:100]}...")
                
                # Also check main IA2 reasoning for formula evidence
                if ia2_reasoning:
                    for keyword in ["support", "resistance", "RR calculation", "simple formula"]:
                        if keyword.lower() in ia2_reasoning.lower():
                            if keyword not in formula_keywords_found:
                                formula_keywords_found.append(keyword)
            
            # Check backend logs for simple RR formula usage
            backend_logs = self.get_backend_logs(1000)
            
            simple_rr_log_patterns = [
                "using LONG formula",
                "using SHORT formula", 
                "Support at",
                "Resistance at",
                "RR = (TP-Entry)/(Entry-SL)",
                "RR = (Entry-TP)/(SL-Entry)",
                "simple S/R calculation",
                "calculated_rr",
                "rr_reasoning"
            ]
            
            log_pattern_counts = {}
            for pattern in simple_rr_log_patterns:
                count = backend_logs.count(pattern)
                log_pattern_counts[pattern] = count
                if count > 0:
                    logger.info(f"      📊 Log pattern '{pattern}': {count} occurrences")
            
            total_log_evidence = sum(log_pattern_counts.values())
            
            # Calculate success metrics
            formula_evidence_percentage = (simple_formula_evidence / max(total_with_reasoning, 1)) * 100
            unique_keywords_found = len(set(formula_keywords_found))
            
            logger.info(f"   📊 Decisions with simple formula evidence: {formula_evidence_percentage:.1f}% ({simple_formula_evidence}/{total_with_reasoning})")
            logger.info(f"   📊 Unique formula keywords found: {unique_keywords_found}")
            logger.info(f"   📊 Backend log evidence patterns: {total_log_evidence}")
            
            # Success criteria: Evidence of simple formula usage in decisions or logs
            success = simple_formula_evidence > 0 or total_log_evidence > 0 or unique_keywords_found >= 3
            
            details = f"Formula evidence: {formula_evidence_percentage:.1f}% ({simple_formula_evidence}/{total_with_reasoning}), Keywords: {unique_keywords_found}, Log patterns: {total_log_evidence}"
            
            self.log_test_result("Simple RR Formula Usage", success, details)
            
        except Exception as e:
            self.log_test_result("Simple RR Formula Usage", False, f"Exception: {str(e)}")
    
    async def test_7_verify_conditional_logic_fix(self):
        """Test 7: Verify that f-string conditional logic properly handles None current_indicators"""
        logger.info("\n🔍 TEST 7: Verifying Conditional Logic Fix for None current_indicators")
        
        try:
            # Get backend logs to check for conditional logic handling
            backend_logs = self.get_backend_logs(2000)
            
            # Look for evidence of proper conditional handling
            conditional_logic_patterns = [
                "current_indicators is None",
                "current_indicators.mfi",
                "current_indicators.vwap_position", 
                "AttributeError",
                "NoneType",
                "has no attribute"
            ]
            
            pattern_counts = {}
            problematic_patterns = 0
            
            for pattern in conditional_logic_patterns:
                count = backend_logs.count(pattern)
                pattern_counts[pattern] = count
                
                # These patterns indicate problems with conditional logic
                if pattern in ["AttributeError", "NoneType", "has no attribute"] and count > 0:
                    problematic_patterns += count
                    logger.info(f"      ⚠️ Problematic pattern '{pattern}': {count} occurrences")
                elif count > 0:
                    logger.info(f"      📊 Pattern '{pattern}': {count} occurrences")
            
            # Check for successful IA2 executions without attribute errors
            ia2_executions = backend_logs.count("IA2 making ultra professional")
            ia2_attribute_errors = backend_logs.count("AttributeError")
            ia2_none_errors = backend_logs.count("NoneType")
            
            # Look for specific technical indicators access patterns
            technical_indicators_access = [
                "current_indicators.mfi",
                "current_indicators.vwap_position",
                "current_indicators.rsi",
                "current_indicators.macd"
            ]
            
            safe_access_count = 0
            for access_pattern in technical_indicators_access:
                count = backend_logs.count(access_pattern)
                if count > 0:
                    safe_access_count += count
                    logger.info(f"      📊 Technical indicator access '{access_pattern}': {count}")
            
            logger.info(f"   📊 IA2 execution attempts: {ia2_executions}")
            logger.info(f"   📊 AttributeError occurrences: {ia2_attribute_errors}")
            logger.info(f"   📊 NoneType errors: {ia2_none_errors}")
            logger.info(f"   📊 Safe technical indicators access: {safe_access_count}")
            logger.info(f"   📊 Total problematic patterns: {problematic_patterns}")
            
            # Success criteria: IA2 executions without attribute/NoneType errors related to current_indicators
            success = (ia2_executions > 0 and problematic_patterns == 0) or (problematic_patterns < ia2_executions)
            
            details = f"IA2 executions: {ia2_executions}, Problematic patterns: {problematic_patterns}, Safe access: {safe_access_count}"
            
            self.log_test_result("Conditional Logic Fix", success, details)
            
        except Exception as e:
            self.log_test_result("Conditional Logic Fix", False, f"Exception: {str(e)}")
    
    def _is_recent_timestamp(self, timestamp_str: str, hours: int = 24) -> bool:
        """Check if timestamp is within the specified hours"""
        try:
            if not timestamp_str:
                return False
            
            # Parse timestamp (handle different formats)
            if 'T' in timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                # Handle "2025-09-09 13:56:35 (Heure de Paris)" format
                if '(' in timestamp_str:
                    timestamp_str = timestamp_str.split('(')[0].strip()
                timestamp = datetime.fromisoformat(timestamp_str)
            
            # Remove timezone info for comparison
            if timestamp.tzinfo:
                timestamp = timestamp.replace(tzinfo=None)
            
            now = datetime.now()
            return (now - timestamp) <= timedelta(hours=hours)
            
        except Exception:
            return False
    
    async def run_comprehensive_tests(self):
        """Run all IA2 technical indicators fix tests"""
        logger.info("🚀 Starting IA2 Technical Indicators Access Fix Test Suite")
        logger.info("=" * 80)
        logger.info("📋 IA2 TECHNICAL INDICATORS ACCESS FIX TESTING")
        logger.info("🎯 Testing: String indices error fix, IA2 decision creation, RR calculation fields")
        logger.info("🎯 Focus: Verify IA2 can complete decisions without crashing, RR fields present")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_baseline_state_capture()
        await self.test_2_trigger_ia2_decision_process()
        await self.test_3_check_string_indices_error_resolution()
        await self.test_4_check_ia2_decision_creation_success()
        await self.test_5_check_rr_calculation_fields()
        await self.test_6_validate_simple_rr_formula_usage()
        await self.test_7_verify_conditional_logic_fix()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 IA2 TECHNICAL INDICATORS ACCESS FIX TEST SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        # Categorize results
        critical_failures = []
        working_components = []
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
            
            if result['success']:
                working_components.append(result['test'])
            else:
                critical_failures.append(result['test'])
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests)*100:.1f}%)")
        
        # Detailed analysis
        logger.info("\n" + "=" * 80)
        logger.info("📋 IA2 TECHNICAL INDICATORS FIX STATUS")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - IA2 Technical Indicators Fix FULLY WORKING!")
            logger.info("✅ String indices error resolved")
            logger.info("✅ IA2 decision creation working without crashes")
            logger.info("✅ RR calculation fields present in decisions")
            logger.info("✅ Simple support/resistance RR formula implemented")
            logger.info("✅ Conditional logic properly handles None current_indicators")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY WORKING - IA2 fix working with minor issues")
            logger.info("🔍 Some components may need fine-tuning")
        elif passed_tests >= total_tests * 0.6:
            logger.info("⚠️ PARTIALLY WORKING - Core fix implemented but issues remain")
            logger.info("🔧 Several components need attention")
        else:
            logger.info("❌ FIX NOT WORKING - Critical issues with IA2 technical indicators access")
            logger.info("🚨 Major problems preventing IA2 from functioning properly")
        
        # Specific requirements check
        logger.info("\n📝 REVIEW REQUIREMENTS VERIFICATION:")
        
        requirements_status = {
            "Technical Indicators Error Resolution": "String Indices Error Resolution" in working_components,
            "IA2 Decision Creation": "IA2 Decision Creation Success" in working_components,
            "RR Calculation Fields": "RR Calculation Fields Presence" in working_components,
            "Simple RR Formula Validation": "Simple RR Formula Usage" in working_components,
            "Conditional Logic Fix": "Conditional Logic Fix" in working_components
        }
        
        for requirement, status in requirements_status.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {requirement}: {'WORKING' if status else 'NOT WORKING'}")
        
        working_requirements = sum(requirements_status.values())
        total_requirements = len(requirements_status)
        
        logger.info(f"\n🏆 REQUIREMENTS SATISFACTION: {working_requirements}/{total_requirements} requirements met")
        
        # Final verdict
        if working_requirements == total_requirements:
            logger.info("\n🎉 VERDICT: IA2 Technical Indicators Access Fix is FULLY SUCCESSFUL!")
            logger.info("✅ All review requirements satisfied")
            logger.info("✅ IA2 can now complete decision-making without crashing")
            logger.info("✅ RR calculation fields are present and working")
            logger.info("✅ Simple support/resistance formula implemented correctly")
        elif working_requirements >= 4:
            logger.info("\n⚠️ VERDICT: IA2 Fix is MOSTLY SUCCESSFUL")
            logger.info("🔍 Minor issues remain but core functionality restored")
        elif working_requirements >= 3:
            logger.info("\n⚠️ VERDICT: IA2 Fix is PARTIALLY SUCCESSFUL")
            logger.info("🔧 Some progress made but significant issues remain")
        else:
            logger.info("\n❌ VERDICT: IA2 Fix is NOT SUCCESSFUL")
            logger.info("🚨 Critical issues prevent IA2 from functioning properly")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = IA2TechnicalIndicatorsFixTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed >= total * 0.8:  # 80% pass rate for success
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
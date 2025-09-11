#!/usr/bin/env python3
"""
IA2 RR Calculation Fix Testing Suite
Focus: Test IA2 RR calculation fix after resolving CPU issues

TESTING REQUIREMENTS FROM REVIEW REQUEST:
1. **CPU Performance Verification**: Confirm CPU usage is stable and not causing IA2 requests to fail
2. **IA2 RR Field Validation**: Check if new IA2 decisions now contain "calculated_rr" and "rr_reasoning" fields
3. **Session ID Change**: Verify that the new session_id "ia2_claude_simplified_rr_v2" is being used
4. **Simple RR Formula**: Confirm IA2 is using simple S/R based calculations instead of complex volatility-based approach
5. **Generate Fresh IA2 Decisions**: Trigger some new IA1 → IA2 escalations to test the updated prompt

EXPECTED OUTCOMES:
- Valid "calculated_rr" values (not null)
- "rr_reasoning" field with S/R level explanations  
- No more fallback RR patterns like "1.00:1 (IA1 R:R unavailable)"
- Session ID "ia2_claude_simplified_rr_v2" in use
- CPU usage stable
"""

import asyncio
import json
import logging
import os
import sys
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List
import requests
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IA2RRCalculationTestSuite:
    """Comprehensive test suite for IA2 RR calculation fix"""
    
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
        
        # Setup MongoDB connection
        ROOT_DIR = Path('/app/backend')
        load_dotenv(ROOT_DIR / '.env')
        self.mongo_url = os.environ['MONGO_URL']
        self.db_name = os.environ['DB_NAME']
        
        # Test results
        self.test_results = []
        
        # Expected session ID
        self.expected_session_id = "ia2_claude_simplified_rr_v2"
        
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
    
    async def test_1_cpu_performance_verification(self):
        """Test 1: CPU Performance Verification - Confirm CPU usage is stable"""
        logger.info("\n🔍 TEST 1: CPU Performance Verification")
        
        try:
            # Monitor CPU usage for 30 seconds
            cpu_readings = []
            memory_readings = []
            
            logger.info("   📊 Monitoring CPU and memory usage for 30 seconds...")
            
            for i in range(6):  # 6 readings over 30 seconds
                cpu_percent = psutil.cpu_percent(interval=5)
                memory_percent = psutil.virtual_memory().percent
                
                cpu_readings.append(cpu_percent)
                memory_readings.append(memory_percent)
                
                logger.info(f"   Reading {i+1}/6: CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%")
            
            # Calculate averages
            avg_cpu = sum(cpu_readings) / len(cpu_readings)
            avg_memory = sum(memory_readings) / len(memory_readings)
            max_cpu = max(cpu_readings)
            
            logger.info(f"   📊 CPU Usage - Average: {avg_cpu:.1f}%, Max: {max_cpu:.1f}%")
            logger.info(f"   📊 Memory Usage - Average: {avg_memory:.1f}%")
            
            # CPU is stable if average < 80% and max < 95%
            cpu_stable = avg_cpu < 80.0 and max_cpu < 95.0
            memory_stable = avg_memory < 90.0
            
            if cpu_stable and memory_stable:
                self.log_test_result("CPU Performance Verification", True, 
                                   f"CPU stable (avg: {avg_cpu:.1f}%, max: {max_cpu:.1f}%), Memory: {avg_memory:.1f}%")
            else:
                self.log_test_result("CPU Performance Verification", False, 
                                   f"System unstable - CPU avg: {avg_cpu:.1f}%, max: {max_cpu:.1f}%, Memory: {avg_memory:.1f}%")
                
        except Exception as e:
            self.log_test_result("CPU Performance Verification", False, f"Exception: {str(e)}")
    
    async def test_2_session_id_verification(self):
        """Test 2: Session ID Change - Verify new session_id is being used"""
        logger.info("\n🔍 TEST 2: Session ID Verification")
        
        try:
            # Check the server.py file for the session ID
            server_py_path = '/app/backend/server.py'
            
            if os.path.exists(server_py_path):
                with open(server_py_path, 'r') as f:
                    content = f.read()
                    
                    if self.expected_session_id in content:
                        # Find the exact line
                        lines = content.split('\n')
                        session_line = None
                        for i, line in enumerate(lines):
                            if self.expected_session_id in line and 'session_id' in line:
                                session_line = f"Line {i+1}: {line.strip()}"
                                break
                        
                        self.log_test_result("Session ID Verification", True, 
                                           f"New session ID '{self.expected_session_id}' found in server.py: {session_line}")
                    else:
                        self.log_test_result("Session ID Verification", False, 
                                           f"Session ID '{self.expected_session_id}' not found in server.py")
            else:
                self.log_test_result("Session ID Verification", False, 
                                   "server.py file not found")
                
        except Exception as e:
            self.log_test_result("Session ID Verification", False, f"Exception: {str(e)}")
    
    async def test_3_database_ia2_field_validation(self):
        """Test 3: IA2 RR Field Validation - Check existing decisions for new fields"""
        logger.info("\n🔍 TEST 3: Database IA2 Field Validation")
        
        try:
            client = AsyncIOMotorClient(self.mongo_url)
            db = client[self.db_name]
            
            # Get recent IA2 decisions
            decisions = await db.trading_decisions.find().sort('timestamp', -1).limit(20).to_list(length=20)
            
            logger.info(f"   📊 Analyzing {len(decisions)} recent IA2 decisions...")
            
            decisions_with_calculated_rr = 0
            decisions_with_rr_reasoning = 0
            fallback_patterns_found = 0
            
            for decision in decisions:
                symbol = decision.get('symbol', 'N/A')
                
                # Check for new fields
                has_calculated_rr = 'calculated_rr' in decision and decision['calculated_rr'] is not None
                has_rr_reasoning = 'rr_reasoning' in decision and decision['rr_reasoning']
                
                if has_calculated_rr:
                    decisions_with_calculated_rr += 1
                    logger.info(f"   ✅ {symbol}: calculated_rr = {decision['calculated_rr']}")
                
                if has_rr_reasoning:
                    decisions_with_rr_reasoning += 1
                    logger.info(f"   ✅ {symbol}: rr_reasoning present")
                
                # Check for fallback patterns in reasoning
                reasoning = decision.get('reasoning', '')
                if 'IA1 R:R unavailable' in reasoning or '1.00:1' in reasoning:
                    fallback_patterns_found += 1
                    logger.info(f"   ⚠️ {symbol}: Fallback pattern detected in reasoning")
            
            # Calculate percentages
            total_decisions = len(decisions)
            calculated_rr_percentage = (decisions_with_calculated_rr / total_decisions * 100) if total_decisions > 0 else 0
            rr_reasoning_percentage = (decisions_with_rr_reasoning / total_decisions * 100) if total_decisions > 0 else 0
            
            logger.info(f"   📊 Results:")
            logger.info(f"      calculated_rr field: {decisions_with_calculated_rr}/{total_decisions} ({calculated_rr_percentage:.1f}%)")
            logger.info(f"      rr_reasoning field: {decisions_with_rr_reasoning}/{total_decisions} ({rr_reasoning_percentage:.1f}%)")
            logger.info(f"      Fallback patterns: {fallback_patterns_found}/{total_decisions}")
            
            # Test passes if at least 50% of recent decisions have the new fields
            success = calculated_rr_percentage >= 50.0 and rr_reasoning_percentage >= 50.0 and fallback_patterns_found == 0
            
            if success:
                self.log_test_result("Database IA2 Field Validation", True, 
                                   f"New RR fields present in {calculated_rr_percentage:.1f}% of decisions, no fallback patterns")
            else:
                self.log_test_result("Database IA2 Field Validation", False, 
                                   f"New RR fields missing - calculated_rr: {calculated_rr_percentage:.1f}%, rr_reasoning: {rr_reasoning_percentage:.1f}%, fallbacks: {fallback_patterns_found}")
            
            await client.close()
            
        except Exception as e:
            self.log_test_result("Database IA2 Field Validation", False, f"Exception: {str(e)}")
    
    async def test_4_trigger_fresh_ia2_decisions(self):
        """Test 4: Generate Fresh IA2 Decisions - Trigger new IA1 → IA2 escalations"""
        logger.info("\n🔍 TEST 4: Generate Fresh IA2 Decisions")
        
        try:
            # First, trigger the scout to find opportunities
            logger.info("   🔍 Step 1: Triggering market scout...")
            scout_response = requests.get(f"{self.api_url}/scout", timeout=120)
            
            if scout_response.status_code == 200:
                scout_data = scout_response.json()
                opportunities_count = len(scout_data.get('opportunities', []))
                logger.info(f"   📊 Scout found {opportunities_count} opportunities")
                
                if opportunities_count > 0:
                    # Wait a bit for IA1 analysis to process
                    logger.info("   ⏳ Waiting 30 seconds for IA1 analysis...")
                    await asyncio.sleep(30)
                    
                    # Check for new IA1 analyses
                    ia1_response = requests.get(f"{self.api_url}/analyses", timeout=60)
                    
                    if ia1_response.status_code == 200:
                        ia1_data = ia1_response.json()
                        analyses_count = len(ia1_data.get('analyses', []))
                        logger.info(f"   📊 Found {analyses_count} IA1 analyses")
                        
                        if analyses_count > 0:
                            # Wait for IA2 decisions to be generated
                            logger.info("   ⏳ Waiting 60 seconds for IA2 decisions...")
                            await asyncio.sleep(60)
                            
                            # Check for new IA2 decisions
                            decisions_response = requests.get(f"{self.api_url}/decisions", timeout=60)
                            
                            if decisions_response.status_code == 200:
                                decisions_data = decisions_response.json()
                                decisions_count = len(decisions_data.get('decisions', []))
                                logger.info(f"   📊 Found {decisions_count} IA2 decisions")
                                
                                # Check if any recent decisions have the new fields
                                recent_decisions_with_new_fields = 0
                                
                                for decision in decisions_data.get('decisions', [])[:5]:  # Check last 5
                                    has_calculated_rr = 'calculated_rr' in decision and decision['calculated_rr'] is not None
                                    has_rr_reasoning = 'rr_reasoning' in decision and decision['rr_reasoning']
                                    
                                    if has_calculated_rr and has_rr_reasoning:
                                        recent_decisions_with_new_fields += 1
                                        logger.info(f"   ✅ {decision.get('symbol', 'N/A')}: New RR fields present")
                                
                                if recent_decisions_with_new_fields > 0:
                                    self.log_test_result("Generate Fresh IA2 Decisions", True, 
                                                       f"Successfully generated {recent_decisions_with_new_fields} new IA2 decisions with RR fields")
                                else:
                                    self.log_test_result("Generate Fresh IA2 Decisions", False, 
                                                       f"Generated {decisions_count} decisions but none have new RR fields")
                            else:
                                self.log_test_result("Generate Fresh IA2 Decisions", False, 
                                                   f"Failed to get IA2 decisions: HTTP {decisions_response.status_code}")
                        else:
                            self.log_test_result("Generate Fresh IA2 Decisions", False, 
                                               "No IA1 analyses generated")
                    else:
                        self.log_test_result("Generate Fresh IA2 Decisions", False, 
                                           f"Failed to get IA1 analyses: HTTP {ia1_response.status_code}")
                else:
                    self.log_test_result("Generate Fresh IA2 Decisions", False, 
                                       "Scout found no opportunities")
            else:
                self.log_test_result("Generate Fresh IA2 Decisions", False, 
                                   f"Scout failed: HTTP {scout_response.status_code}")
                
        except Exception as e:
            self.log_test_result("Generate Fresh IA2 Decisions", False, f"Exception: {str(e)}")
    
    async def test_5_simple_rr_formula_validation(self):
        """Test 5: Simple RR Formula - Confirm IA2 is using simple S/R based calculations"""
        logger.info("\n🔍 TEST 5: Simple RR Formula Validation")
        
        try:
            client = AsyncIOMotorClient(self.mongo_url)
            db = client[self.db_name]
            
            # Get recent IA2 decisions with trading signals (not HOLD)
            decisions = await db.trading_decisions.find({
                'signal': {'$in': ['long', 'short', 'LONG', 'SHORT']}
            }).sort('timestamp', -1).limit(10).to_list(length=10)
            
            logger.info(f"   📊 Analyzing {len(decisions)} recent trading decisions...")
            
            simple_formula_count = 0
            complex_formula_count = 0
            
            for decision in decisions:
                symbol = decision.get('symbol', 'N/A')
                signal = decision.get('signal', 'N/A')
                
                # Check if decision has the new RR reasoning field
                rr_reasoning = decision.get('rr_reasoning', '')
                calculated_rr = decision.get('calculated_rr')
                
                if rr_reasoning and calculated_rr is not None:
                    logger.info(f"   📊 {symbol} ({signal}): calculated_rr = {calculated_rr}")
                    logger.info(f"      rr_reasoning: {rr_reasoning[:100]}...")
                    
                    # Check for simple S/R formula indicators
                    simple_indicators = [
                        'Support at', 'Resistance at', 'using LONG formula', 'using SHORT formula',
                        'TP-Entry', 'Entry-SL', 'Entry-TP', 'SL-Entry'
                    ]
                    
                    # Check for complex formula indicators
                    complex_indicators = [
                        'volatility-based', 'advanced calculation', 'complex assessment',
                        'deemed suboptimal', 'IA1 R:R unavailable'
                    ]
                    
                    has_simple_indicators = any(indicator in rr_reasoning for indicator in simple_indicators)
                    has_complex_indicators = any(indicator in rr_reasoning for indicator in complex_indicators)
                    
                    if has_simple_indicators and not has_complex_indicators:
                        simple_formula_count += 1
                        logger.info(f"   ✅ {symbol}: Using simple S/R formula")
                    elif has_complex_indicators:
                        complex_formula_count += 1
                        logger.info(f"   ❌ {symbol}: Still using complex formula")
                    else:
                        logger.info(f"   ⚠️ {symbol}: Formula type unclear")
                else:
                    logger.info(f"   ❌ {symbol}: Missing new RR fields")
            
            total_analyzed = len(decisions)
            simple_percentage = (simple_formula_count / total_analyzed * 100) if total_analyzed > 0 else 0
            
            logger.info(f"   📊 Results:")
            logger.info(f"      Simple S/R formula: {simple_formula_count}/{total_analyzed} ({simple_percentage:.1f}%)")
            logger.info(f"      Complex formula: {complex_formula_count}/{total_analyzed}")
            
            # Test passes if at least 70% use simple formula and no complex formulas
            success = simple_percentage >= 70.0 and complex_formula_count == 0
            
            if success:
                self.log_test_result("Simple RR Formula Validation", True, 
                                   f"Simple S/R formula used in {simple_percentage:.1f}% of decisions, no complex formulas")
            else:
                self.log_test_result("Simple RR Formula Validation", False, 
                                   f"Simple formula: {simple_percentage:.1f}%, Complex formulas still present: {complex_formula_count}")
            
            await client.close()
            
        except Exception as e:
            self.log_test_result("Simple RR Formula Validation", False, f"Exception: {str(e)}")
    
    async def test_6_ia2_request_failure_check(self):
        """Test 6: IA2 Request Failure Check - Ensure IA2 requests are not failing due to CPU issues"""
        logger.info("\n🔍 TEST 6: IA2 Request Failure Check")
        
        try:
            # Check backend logs for IA2 failures
            backend_log_paths = [
                '/var/log/supervisor/backend.err.log',
                '/var/log/supervisor/backend.out.log'
            ]
            
            ia2_errors = []
            ia2_successes = []
            cpu_related_errors = []
            
            for log_path in backend_log_paths:
                if os.path.exists(log_path):
                    try:
                        # Read last 1000 lines
                        with open(log_path, 'r') as f:
                            lines = f.readlines()[-1000:]
                            
                        for line in lines:
                            line_lower = line.lower()
                            
                            # Check for IA2 related entries
                            if 'ia2' in line_lower:
                                if any(error_word in line_lower for error_word in ['error', 'failed', 'exception', 'timeout']):
                                    ia2_errors.append(line.strip())
                                elif any(success_word in line_lower for success_word in ['success', 'completed', 'decision']):
                                    ia2_successes.append(line.strip())
                            
                            # Check for CPU related errors
                            if any(cpu_word in line_lower for cpu_word in ['cpu', 'memory', 'timeout', 'resource']):
                                if any(error_word in line_lower for error_word in ['error', 'failed', 'exception']):
                                    cpu_related_errors.append(line.strip())
                                    
                    except Exception as e:
                        logger.info(f"   ⚠️ Could not read {log_path}: {e}")
            
            logger.info(f"   📊 Log Analysis Results:")
            logger.info(f"      IA2 errors found: {len(ia2_errors)}")
            logger.info(f"      IA2 successes found: {len(ia2_successes)}")
            logger.info(f"      CPU-related errors: {len(cpu_related_errors)}")
            
            # Show recent errors if any
            if ia2_errors:
                logger.info("   ❌ Recent IA2 errors:")
                for error in ia2_errors[-3:]:  # Show last 3 errors
                    logger.info(f"      {error}")
            
            if cpu_related_errors:
                logger.info("   ⚠️ Recent CPU-related errors:")
                for error in cpu_related_errors[-3:]:  # Show last 3 errors
                    logger.info(f"      {error}")
            
            # Test passes if there are more successes than errors and no recent CPU errors
            success_ratio = len(ia2_successes) / max(len(ia2_errors), 1)
            no_recent_cpu_errors = len(cpu_related_errors) == 0
            
            if success_ratio >= 2.0 and no_recent_cpu_errors:
                self.log_test_result("IA2 Request Failure Check", True, 
                                   f"IA2 requests stable - Success ratio: {success_ratio:.1f}, No CPU errors")
            else:
                self.log_test_result("IA2 Request Failure Check", False, 
                                   f"IA2 requests unstable - Success ratio: {success_ratio:.1f}, CPU errors: {len(cpu_related_errors)}")
                
        except Exception as e:
            self.log_test_result("IA2 Request Failure Check", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_tests(self):
        """Run all IA2 RR calculation fix tests"""
        logger.info("🚀 Starting IA2 RR Calculation Fix Test Suite")
        logger.info("=" * 80)
        logger.info("📋 IA2 RR CALCULATION FIX COMPREHENSIVE TESTING")
        logger.info("🎯 Testing: CPU stability, session ID, RR fields, simple formula, fresh decisions")
        logger.info("🎯 Expected: calculated_rr and rr_reasoning fields with simple S/R calculations")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_cpu_performance_verification()
        await self.test_2_session_id_verification()
        await self.test_3_database_ia2_field_validation()
        await self.test_4_trigger_fresh_ia2_decisions()
        await self.test_5_simple_rr_formula_validation()
        await self.test_6_ia2_request_failure_check()
        
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
        
        # Final analysis
        logger.info("\n" + "=" * 80)
        logger.info("📋 IA2 RR CALCULATION FIX STATUS")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - IA2 RR Calculation Fix FULLY WORKING!")
            logger.info("✅ CPU performance stable")
            logger.info("✅ New session ID in use")
            logger.info("✅ calculated_rr and rr_reasoning fields present")
            logger.info("✅ Simple S/R formula implemented")
            logger.info("✅ Fresh IA2 decisions generated successfully")
            logger.info("✅ No IA2 request failures")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY WORKING - IA2 RR fix functional with minor issues")
            logger.info("🔍 Some components may need fine-tuning")
        elif passed_tests >= total_tests * 0.5:
            logger.info("⚠️ PARTIALLY WORKING - Some IA2 RR fix components functional")
            logger.info("🔧 Several issues need to be addressed")
        else:
            logger.info("❌ NOT WORKING - IA2 RR calculation fix has major issues")
            logger.info("🚨 Critical problems preventing proper functionality")
        
        # Specific requirements check
        logger.info("\n📝 REVIEW REQUIREMENTS VERIFICATION:")
        
        requirements_status = {
            "CPU Performance Stable": False,
            "Session ID Updated": False,
            "RR Fields Present": False,
            "Simple Formula Used": False,
            "Fresh Decisions Generated": False,
            "No Request Failures": False
        }
        
        # Map test results to requirements
        for result in self.test_results:
            if result['success']:
                if "CPU Performance" in result['test']:
                    requirements_status["CPU Performance Stable"] = True
                elif "Session ID" in result['test']:
                    requirements_status["Session ID Updated"] = True
                elif "Database IA2 Field" in result['test']:
                    requirements_status["RR Fields Present"] = True
                elif "Simple RR Formula" in result['test']:
                    requirements_status["Simple Formula Used"] = True
                elif "Generate Fresh IA2" in result['test']:
                    requirements_status["Fresh Decisions Generated"] = True
                elif "Request Failure Check" in result['test']:
                    requirements_status["No Request Failures"] = True
        
        for requirement, status in requirements_status.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {requirement}")
        
        requirements_met = sum(requirements_status.values())
        total_requirements = len(requirements_status)
        
        logger.info(f"\n🏆 REQUIREMENTS SATISFIED: {requirements_met}/{total_requirements}")
        
        # Final verdict
        if requirements_met == total_requirements:
            logger.info("\n🎉 VERDICT: IA2 RR Calculation Fix is FULLY WORKING!")
            logger.info("✅ All review requirements satisfied")
            logger.info("✅ CPU stable, new session ID active, RR fields present")
            logger.info("✅ Simple S/R formula implemented, fresh decisions working")
        elif requirements_met >= total_requirements * 0.8:
            logger.info("\n⚠️ VERDICT: IA2 RR Calculation Fix is MOSTLY WORKING")
            logger.info("🔍 Minor issues may need attention")
        elif requirements_met >= total_requirements * 0.5:
            logger.info("\n⚠️ VERDICT: IA2 RR Calculation Fix is PARTIALLY WORKING")
            logger.info("🔧 Several requirements not met")
        else:
            logger.info("\n❌ VERDICT: IA2 RR Calculation Fix is NOT WORKING")
            logger.info("🚨 Major issues preventing proper functionality")
        
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
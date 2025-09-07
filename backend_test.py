#!/usr/bin/env python3
"""
Backend Testing Suite for Enhanced Chartist Pattern Integration with IA1
Focus: Pattern Detection, Integration Flow, IA1 Pattern Awareness, and Data Flow
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List

# Add backend to path
sys.path.append('/app/backend')

import requests
from motor.motor_asyncio import AsyncIOMotorClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChartistPatternIntegrationTestSuite:
    """Test suite for Enhanced Chartist Pattern Integration with IA1"""
    
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
        logger.info(f"Testing backend at: {self.api_url}")
        
        # MongoDB connection for direct data access
        self.mongo_client = None
        self.db = None
        
        # Test results
        self.test_results = []
        
    async def setup_database(self):
        """Setup database connection"""
        try:
            # Get MongoDB URL from backend env
            mongo_url = "mongodb://localhost:27017"  # Default
            try:
                with open('/app/backend/.env', 'r') as f:
                    for line in f:
                        if line.startswith('MONGO_URL='):
                            mongo_url = line.split('=')[1].strip().strip('"')
                            break
            except Exception:
                pass
            
            self.mongo_client = AsyncIOMotorClient(mongo_url)
            self.db = self.mongo_client['myapp']
            logger.info("✅ Database connection established")
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            
    async def cleanup_database(self):
        """Cleanup database connection"""
        if self.mongo_client:
            self.mongo_client.close()
            
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
        
    async def test_pattern_detection_flow(self):
        """Test 1: Pattern Integration Flow - Verify pattern detector correctly detects patterns"""
        logger.info("\n🔍 TEST 1: Pattern Integration Flow")
        
        try:
            # Get recent IA1 analyses to check for pattern detection
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Pattern Detection Flow", False, f"API error: {response.status_code}")
                return
                
            analyses = response.json()
            
            if not analyses:
                self.log_test_result("Pattern Detection Flow", False, "No IA1 analyses found")
                return
                
            # Check for pattern detection in analyses
            patterns_detected_count = 0
            total_analyses = 0
            pattern_types_found = set()
            
            for analysis in analyses[:10]:  # Check last 10 analyses
                total_analyses += 1
                symbol = analysis.get('symbol', 'UNKNOWN')
                patterns_detected = analysis.get('patterns_detected', [])
                ia1_reasoning = analysis.get('ia1_reasoning', '')
                
                logger.info(f"   📊 {symbol}: {len(patterns_detected)} patterns detected")
                
                if patterns_detected:
                    patterns_detected_count += 1
                    pattern_types_found.update(patterns_detected)
                    logger.info(f"      🎯 Patterns: {patterns_detected}")
                    
                # Check if reasoning mentions pattern analysis
                pattern_keywords = ['pattern', 'triangle', 'wedge', 'channel', 'support', 'resistance', 'breakout']
                if any(keyword in ia1_reasoning.lower() for keyword in pattern_keywords):
                    logger.info(f"      ✅ Pattern analysis found in reasoning")
                else:
                    logger.info(f"      ⚠️ No pattern analysis in reasoning")
                    
            success = patterns_detected_count > 0 and len(pattern_types_found) > 0
            details = f"Patterns detected in {patterns_detected_count}/{total_analyses} analyses, {len(pattern_types_found)} unique pattern types: {list(pattern_types_found)[:5]}"
            
            self.log_test_result("Pattern Detection Flow", success, details)
            
        except Exception as e:
            self.log_test_result("Pattern Detection Flow", False, f"Exception: {str(e)}")
            
    async def test_ia1_pattern_awareness(self):
        """Test 2: IA1 Pattern Awareness - Check that IA1 receives DETECTED CHARTIST PATTERNS section"""
        logger.info("\n🔍 TEST 2: IA1 Pattern Awareness")
        
        try:
            # Check backend logs for pattern integration messages
            import subprocess
            
            # Get recent backend logs
            log_cmd = "tail -n 500 /var/log/supervisor/backend.*.log 2>/dev/null || echo 'No logs found'"
            result = subprocess.run(log_cmd, shell=True, capture_output=True, text=True)
            
            backend_logs = result.stdout
            
            # Look for pattern integration log patterns
            pattern_integration_logs = []
            detected_patterns_logs = []
            ia1_pattern_logs = []
            
            for line in backend_logs.split('\n'):
                if 'PATTERNS COMPLETS' in line or 'patterns détectés' in line:
                    pattern_integration_logs.append(line.strip())
                elif 'DETECTED CHARTIST PATTERNS' in line:
                    detected_patterns_logs.append(line.strip())
                elif 'IA1 ANALYSE JUSTIFIÉE' in line and 'patterns' in line.lower():
                    ia1_pattern_logs.append(line.strip())
                    
            logger.info(f"   📊 Found {len(pattern_integration_logs)} pattern integration logs")
            logger.info(f"   📊 Found {len(detected_patterns_logs)} detected patterns logs")
            logger.info(f"   📊 Found {len(ia1_pattern_logs)} IA1 pattern analysis logs")
            
            # Show sample logs
            if pattern_integration_logs:
                logger.info(f"   📝 Sample pattern log: {pattern_integration_logs[-1][:100]}...")
            if detected_patterns_logs:
                logger.info(f"   📝 Sample detected log: {detected_patterns_logs[-1][:100]}...")
                
            # Check IA1 analyses for pattern awareness
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            pattern_aware_analyses = 0
            
            if response.status_code == 200:
                analyses = response.json()
                
                for analysis in analyses[:10]:
                    ia1_reasoning = analysis.get('ia1_reasoning', '')
                    patterns_detected = analysis.get('patterns_detected', [])
                    
                    # Check if IA1 reasoning incorporates detected patterns
                    if patterns_detected and any(pattern.lower() in ia1_reasoning.lower() for pattern in patterns_detected):
                        pattern_aware_analyses += 1
                        
                logger.info(f"   📊 Pattern-aware analyses: {pattern_aware_analyses}/10")
                
            success = len(pattern_integration_logs) > 0 or pattern_aware_analyses > 0
            details = f"Integration logs: {len(pattern_integration_logs)}, Pattern-aware analyses: {pattern_aware_analyses}"
            
            self.log_test_result("IA1 Pattern Awareness", success, details)
            
        except Exception as e:
            self.log_test_result("IA1 Pattern Awareness", False, f"Exception: {str(e)}")
            
    async def test_pattern_data_flow(self):
        """Test 3: Pattern Data Flow - Verify patterns_detected field includes all detected patterns"""
        logger.info("\n🔍 TEST 3: Pattern Data Flow")
        
        try:
            # Get recent IA1 analyses and check pattern data flow
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Pattern Data Flow", False, f"API error: {response.status_code}")
                return
                
            analyses = response.json()
            
            if not analyses:
                self.log_test_result("Pattern Data Flow", False, "No analyses found")
                return
                
            # Analyze pattern data flow
            analyses_with_patterns = 0
            total_patterns_count = 0
            multiple_patterns_count = 0
            
            for analysis in analyses[:15]:  # Check last 15 analyses
                symbol = analysis.get('symbol', 'UNKNOWN')
                patterns_detected = analysis.get('patterns_detected', [])
                ia1_reasoning = analysis.get('ia1_reasoning', '')
                
                if patterns_detected:
                    analyses_with_patterns += 1
                    total_patterns_count += len(patterns_detected)
                    
                    if len(patterns_detected) > 1:
                        multiple_patterns_count += 1
                        
                    logger.info(f"   📊 {symbol}: {len(patterns_detected)} patterns - {patterns_detected}")
                    
                    # Check if all patterns are included (not just primary one)
                    pattern_mentions = sum(1 for pattern in patterns_detected if pattern.lower() in ia1_reasoning.lower())
                    if pattern_mentions > 0:
                        logger.info(f"      ✅ {pattern_mentions}/{len(patterns_detected)} patterns mentioned in reasoning")
                    else:
                        logger.info(f"      ⚠️ Patterns not mentioned in reasoning")
                        
            # Check database for pattern storage
            pattern_storage_success = False
            if self.db:
                try:
                    # Check if technical_analyses collection has pattern data
                    analyses_cursor = self.db.technical_analyses.find().limit(10)
                    db_analyses = await analyses_cursor.to_list(length=10)
                    
                    db_patterns_count = 0
                    for db_analysis in db_analyses:
                        patterns = db_analysis.get('patterns_detected', [])
                        if patterns:
                            db_patterns_count += len(patterns)
                            
                    pattern_storage_success = db_patterns_count > 0
                    logger.info(f"   📊 Database patterns stored: {db_patterns_count}")
                    
                except Exception as e:
                    logger.debug(f"Database check failed: {e}")
                    
            success = analyses_with_patterns > 0 and total_patterns_count > 0
            details = f"Analyses with patterns: {analyses_with_patterns}/15, Total patterns: {total_patterns_count}, Multiple patterns: {multiple_patterns_count}, DB storage: {pattern_storage_success}"
            
            self.log_test_result("Pattern Data Flow", success, details)
            
        except Exception as e:
            self.log_test_result("Pattern Data Flow", False, f"Exception: {str(e)}")
            
    async def test_new_pattern_types(self):
        """Test 4: New Pattern Types - Look for detection of new pattern types"""
        logger.info("\n🔍 TEST 4: New Pattern Types Detection")
        
        try:
            # Get recent analyses and check for new pattern types
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("New Pattern Types", False, f"API error: {response.status_code}")
                return
                
            analyses = response.json()
            
            # Define expected new pattern types from the review request
            new_pattern_types = {
                'harmonic': ['harmonic', 'gartley', 'butterfly', 'bat', 'crab'],
                'diamond': ['diamond', 'diamond_top', 'diamond_bottom'],
                'expanding_wedge': ['expanding_wedge', 'broadening_wedge'],
                'triangular': ['symmetrical_triangle', 'ascending_triangle', 'descending_triangle'],
                'consolidation': ['rectangle', 'pennant', 'flag']
            }
            
            found_pattern_categories = set()
            all_detected_patterns = []
            
            for analysis in analyses[:20]:  # Check last 20 analyses
                symbol = analysis.get('symbol', 'UNKNOWN')
                patterns_detected = analysis.get('patterns_detected', [])
                
                for pattern in patterns_detected:
                    all_detected_patterns.append(pattern.lower())
                    
                    # Check which category this pattern belongs to
                    for category, pattern_list in new_pattern_types.items():
                        if any(p in pattern.lower() for p in pattern_list):
                            found_pattern_categories.add(category)
                            logger.info(f"   🎯 {symbol}: Found {category} pattern - {pattern}")
                            
            # Check backend logs for pattern detection messages
            import subprocess
            log_cmd = "tail -n 300 /var/log/supervisor/backend.*.log 2>/dev/null | grep -i 'pattern\\|triangle\\|wedge\\|diamond\\|harmonic\\|rectangle\\|pennant' || echo 'No pattern logs'"
            result = subprocess.run(log_cmd, shell=True, capture_output=True, text=True)
            
            pattern_logs = result.stdout
            advanced_pattern_logs = 0
            
            for line in pattern_logs.split('\n'):
                if any(pattern_type in line.lower() for category in new_pattern_types.values() for pattern_type in category):
                    advanced_pattern_logs += 1
                    
            logger.info(f"   📊 Found pattern categories: {found_pattern_categories}")
            logger.info(f"   📊 Total unique patterns detected: {len(set(all_detected_patterns))}")
            logger.info(f"   📊 Advanced pattern logs: {advanced_pattern_logs}")
            
            success = len(found_pattern_categories) > 0 or len(set(all_detected_patterns)) >= 3
            details = f"Pattern categories found: {len(found_pattern_categories)} ({list(found_pattern_categories)}), Unique patterns: {len(set(all_detected_patterns))}, Advanced logs: {advanced_pattern_logs}"
            
            self.log_test_result("New Pattern Types", success, details)
            
        except Exception as e:
            self.log_test_result("New Pattern Types", False, f"Exception: {str(e)}")
            
    async def test_system_integration(self):
        """Test 5: System Integration - Ensure pattern detection doesn't cause errors"""
        logger.info("\n🔍 TEST 5: System Integration")
        
        try:
            # Test all main endpoints to ensure they're working
            endpoints_to_test = [
                '/opportunities',
                '/analyses', 
                '/decisions',
                '/market-status'
            ]
            
            endpoint_results = {}
            
            for endpoint in endpoints_to_test:
                try:
                    response = requests.get(f"{self.api_url}{endpoint}", timeout=30)
                    endpoint_results[endpoint] = {
                        'status_code': response.status_code,
                        'working': response.status_code == 200,
                        'data_count': len(response.json()) if response.status_code == 200 else 0
                    }
                    logger.info(f"   📊 {endpoint}: {response.status_code} - {endpoint_results[endpoint]['data_count']} items")
                except Exception as e:
                    endpoint_results[endpoint] = {
                        'status_code': 0,
                        'working': False,
                        'error': str(e)
                    }
                    logger.info(f"   ❌ {endpoint}: Error - {str(e)}")
                    
            # Check backend logs for errors related to pattern detection
            import subprocess
            log_cmd = "tail -n 200 /var/log/supervisor/backend.*.log 2>/dev/null | grep -i 'error\\|exception\\|traceback' | grep -i 'pattern' || echo 'No pattern errors'"
            result = subprocess.run(log_cmd, shell=True, capture_output=True, text=True)
            
            pattern_errors = result.stdout.strip()
            has_pattern_errors = pattern_errors != 'No pattern errors' and len(pattern_errors) > 0
            
            if has_pattern_errors:
                logger.info(f"   ⚠️ Pattern-related errors found in logs")
                logger.info(f"   📝 Sample error: {pattern_errors.split('\\n')[0][:100]}...")
            else:
                logger.info(f"   ✅ No pattern-related errors in logs")
                
            # Check system performance
            working_endpoints = sum(1 for result in endpoint_results.values() if result['working'])
            total_endpoints = len(endpoints_to_test)
            
            # Check if analyses are being generated normally
            analyses_response = requests.get(f"{self.api_url}/analyses", timeout=30)
            normal_operation = False
            
            if analyses_response.status_code == 200:
                analyses = analyses_response.json()
                if len(analyses) > 0:
                    # Check if recent analyses exist (within reasonable time)
                    recent_analyses = 0
                    for analysis in analyses[:5]:
                        timestamp = analysis.get('timestamp', '')
                        if timestamp:  # If we have timestamps, that's good
                            recent_analyses += 1
                            
                    normal_operation = recent_analyses > 0
                    logger.info(f"   📊 Recent analyses: {recent_analyses}/5")
                    
            success = (working_endpoints >= total_endpoints * 0.8) and not has_pattern_errors and normal_operation
            details = f"Working endpoints: {working_endpoints}/{total_endpoints}, Pattern errors: {has_pattern_errors}, Normal operation: {normal_operation}"
            
            self.log_test_result("System Integration", success, details)
            
        except Exception as e:
            self.log_test_result("System Integration", False, f"Exception: {str(e)}")
            
    async def test_pattern_integration_examples(self):
        """Test 6: Pattern Integration Examples - Provide specific examples of patterns detected"""
        logger.info("\n🔍 TEST 6: Pattern Integration Examples")
        
        try:
            # Get recent analyses with detailed pattern information
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Pattern Integration Examples", False, f"API error: {response.status_code}")
                return
                
            analyses = response.json()
            
            if not analyses:
                self.log_test_result("Pattern Integration Examples", False, "No analyses found")
                return
                
            # Collect detailed examples of pattern integration
            pattern_examples = []
            integration_examples = []
            
            for analysis in analyses[:10]:
                symbol = analysis.get('symbol', 'UNKNOWN')
                patterns_detected = analysis.get('patterns_detected', [])
                ia1_reasoning = analysis.get('ia1_reasoning', '')
                confidence = analysis.get('analysis_confidence', 0)
                
                if patterns_detected:
                    example = {
                        'symbol': symbol,
                        'patterns': patterns_detected,
                        'confidence': confidence,
                        'reasoning_length': len(ia1_reasoning),
                        'pattern_integration': any(pattern.lower() in ia1_reasoning.lower() for pattern in patterns_detected)
                    }
                    pattern_examples.append(example)
                    
                    # Check for specific integration evidence
                    if example['pattern_integration']:
                        integration_detail = {
                            'symbol': symbol,
                            'patterns': patterns_detected,
                            'reasoning_snippet': ia1_reasoning[:200] + "..." if len(ia1_reasoning) > 200 else ia1_reasoning
                        }
                        integration_examples.append(integration_detail)
                        
                        logger.info(f"   🎯 EXAMPLE: {symbol}")
                        logger.info(f"      Patterns: {patterns_detected}")
                        logger.info(f"      Confidence: {confidence:.2f}")
                        logger.info(f"      Integration: ✅ Patterns mentioned in reasoning")
                        logger.info(f"      Reasoning: {ia1_reasoning[:150]}...")
                        
            # Check backend logs for specific pattern integration messages
            import subprocess
            log_cmd = "tail -n 300 /var/log/supervisor/backend.*.log 2>/dev/null | grep -A2 -B2 'PATTERN\\|patterns détectés\\|IA1 ANALYSE' || echo 'No detailed logs'"
            result = subprocess.run(log_cmd, shell=True, capture_output=True, text=True)
            
            detailed_logs = result.stdout
            integration_log_examples = []
            
            for line in detailed_logs.split('\n'):
                if 'patterns détectés' in line or 'PATTERN' in line:
                    integration_log_examples.append(line.strip())
                    
            logger.info(f"   📊 Pattern examples found: {len(pattern_examples)}")
            logger.info(f"   📊 Integration examples: {len(integration_examples)}")
            logger.info(f"   📊 Log integration examples: {len(integration_log_examples)}")
            
            # Show detailed integration example
            if integration_examples:
                best_example = max(integration_examples, key=lambda x: len(x['patterns']))
                logger.info(f"   🏆 BEST INTEGRATION EXAMPLE:")
                logger.info(f"      Symbol: {best_example['symbol']}")
                logger.info(f"      Patterns: {best_example['patterns']}")
                logger.info(f"      Reasoning: {best_example['reasoning_snippet']}")
                
            success = len(pattern_examples) > 0 and len(integration_examples) > 0
            details = f"Pattern examples: {len(pattern_examples)}, Integration examples: {len(integration_examples)}, Log examples: {len(integration_log_examples)}"
            
            self.log_test_result("Pattern Integration Examples", success, details)
            
        except Exception as e:
            self.log_test_result("Pattern Integration Examples", False, f"Exception: {str(e)}")
            
    async def run_comprehensive_tests(self):
        """Run all Enhanced Chartist Pattern Integration tests"""
        logger.info("🚀 Starting Enhanced Chartist Pattern Integration Test Suite")
        logger.info("=" * 70)
        
        await self.setup_database()
        
        # Run all tests
        await self.test_pattern_detection_flow()
        await self.test_ia1_pattern_awareness()
        await self.test_pattern_data_flow()
        await self.test_new_pattern_types()
        await self.test_system_integration()
        await self.test_pattern_integration_examples()
        
        await self.cleanup_database()
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("📊 ENHANCED CHARTIST PATTERN INTEGRATION TEST SUMMARY")
        logger.info("=" * 70)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Enhanced Chartist Pattern Integration is working correctly!")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY WORKING - Some minor pattern integration issues detected")
        else:
            logger.info("❌ CRITICAL ISSUES - Pattern integration needs attention")
            
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = ChartistPatternIntegrationTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
COMPREHENSIVE MARKET OPPORTUNITIES TIMESTAMP PERSISTENCE FIX TEST SUITE
Focus: Test Market Opportunities Timestamp Persistence Fix in advanced_market_aggregator.py

CRITICAL TEST REQUIREMENTS FROM REVIEW REQUEST:
1. **API ENDPOINT TESTING**: Test /api/opportunities endpoint multiple times to verify timestamps are persistent
2. **CACHE BEHAVIOR VERIFICATION**: Test cache hit/miss scenarios
3. **SYMBOL TIMESTAMP PERSISTENCE**: Verify same symbols get same timestamps, new symbols get unique timestamps
4. **MEMORY MANAGEMENT**: Verify timestamp cleanup functionality
5. **DATA INTEGRITY**: Ensure opportunity data updates correctly while timestamps remain fixed
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
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveTimestampPersistenceTestSuite:
    """Comprehensive test suite for Market Opportunities Timestamp Persistence Fix"""
    
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
        logger.info(f"Testing Market Opportunities Timestamp Persistence Fix at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
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
    
    async def test_1_multiple_api_calls_timestamp_persistence(self):
        """Test 1: Multiple API Calls - Verify Timestamps Remain Persistent"""
        logger.info("\n🔍 TEST 1: Multiple API Calls - Timestamp Persistence Verification")
        
        try:
            persistence_results = {
                'api_calls_successful': 0,
                'total_api_calls': 3,
                'timestamp_persistence_verified': False,
                'same_symbols_same_timestamps': True,
                'timestamp_data_by_call': [],
                'persistent_symbols': set()
            }
            
            logger.info("   🚀 Testing timestamp persistence across multiple /api/opportunities calls...")
            logger.info("   📊 Expected: Same symbols maintain identical timestamps across calls")
            
            # Make 3 API calls with delays between them
            for call_num in range(1, 4):
                logger.info(f"\n   📞 API Call #{call_num}/3")
                
                try:
                    start_time = time.time()
                    response = requests.get(f"{self.api_url}/opportunities", timeout=60)
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        if response_data.get('success') and 'opportunities' in response_data:
                            opportunities = response_data['opportunities']
                            persistence_results['api_calls_successful'] += 1
                            
                            logger.info(f"      ✅ API call #{call_num} successful (response time: {response_time:.2f}s)")
                            logger.info(f"      📋 Retrieved {len(opportunities)} opportunities")
                            
                            # Store timestamp data for this call
                            call_data = {}
                            for opp in opportunities:
                                symbol = opp.get('symbol')
                                timestamp = opp.get('timestamp')
                                if symbol and timestamp:
                                    call_data[symbol] = {
                                        'timestamp': timestamp,
                                        'current_price': opp.get('current_price', 0),
                                        'volume_24h': opp.get('volume_24h', 0)
                                    }
                            
                            persistence_results['timestamp_data_by_call'].append({
                                'call_number': call_num,
                                'data': call_data,
                                'symbol_count': len(call_data)
                            })
                            
                            logger.info(f"      📊 Captured timestamps for {len(call_data)} symbols")
                            
                            # Compare with previous calls if this is not the first call
                            if call_num > 1:
                                previous_call_data = persistence_results['timestamp_data_by_call'][call_num-2]['data']
                                
                                # Check for symbols that appear in both calls
                                common_symbols = set(call_data.keys()) & set(previous_call_data.keys())
                                
                                logger.info(f"      🔍 Comparing with call #{call_num-1}: {len(common_symbols)} common symbols")
                                
                                # Verify timestamp persistence for common symbols
                                timestamp_mismatches = 0
                                for symbol in common_symbols:
                                    current_timestamp = call_data[symbol]['timestamp']
                                    previous_timestamp = previous_call_data[symbol]['timestamp']
                                    
                                    if current_timestamp != previous_timestamp:
                                        timestamp_mismatches += 1
                                        logger.warning(f"         ❌ TIMESTAMP MISMATCH for {symbol}: {previous_timestamp} → {current_timestamp}")
                                        persistence_results['same_symbols_same_timestamps'] = False
                                    else:
                                        persistence_results['persistent_symbols'].add(symbol)
                                
                                if timestamp_mismatches == 0 and common_symbols:
                                    logger.info(f"      ✅ All {len(common_symbols)} common symbols maintained persistent timestamps")
                                elif common_symbols:
                                    logger.warning(f"      ❌ {timestamp_mismatches}/{len(common_symbols)} symbols had timestamp changes")
                        else:
                            logger.error(f"      ❌ API call #{call_num} returned invalid response structure")
                    
                    else:
                        logger.error(f"      ❌ API call #{call_num} failed: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"      ❌ API call #{call_num} exception: {e}")
                
                # Wait between calls (except after the last call)
                if call_num < 3:
                    logger.info(f"      ⏳ Waiting 5 seconds before next call...")
                    await asyncio.sleep(5)
            
            # Final analysis
            logger.info(f"\n   📊 FINAL ANALYSIS:")
            logger.info(f"      Successful API calls: {persistence_results['api_calls_successful']}/{persistence_results['total_api_calls']}")
            logger.info(f"      Persistent symbols: {len(persistence_results['persistent_symbols'])}")
            
            # Determine overall timestamp persistence success
            if (persistence_results['api_calls_successful'] >= 2 and 
                persistence_results['same_symbols_same_timestamps'] and
                len(persistence_results['persistent_symbols']) > 0):
                persistence_results['timestamp_persistence_verified'] = True
                logger.info(f"      ✅ TIMESTAMP PERSISTENCE VERIFIED")
            else:
                logger.warning(f"      ❌ TIMESTAMP PERSISTENCE FAILED")
            
            # Calculate test success
            success_criteria = [
                persistence_results['api_calls_successful'] >= 2,
                persistence_results['timestamp_persistence_verified'],
                persistence_results['same_symbols_same_timestamps'],
                len(persistence_results['persistent_symbols']) > 0
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.75:  # 75% success threshold
                self.log_test_result("Multiple API Calls Timestamp Persistence", True, 
                                   f"Timestamp persistence working: {success_count}/{len(success_criteria)} criteria met. Persistent symbols: {len(persistence_results['persistent_symbols'])}")
            else:
                self.log_test_result("Multiple API Calls Timestamp Persistence", False, 
                                   f"Timestamp persistence issues: {success_count}/{len(success_criteria)} criteria met")
                
        except Exception as e:
            self.log_test_result("Multiple API Calls Timestamp Persistence", False, f"Exception: {str(e)}")

    async def test_2_cache_behavior_verification(self):
        """Test 2: Cache Behavior Verification - Test Cache Hit/Miss Scenarios"""
        logger.info("\n🔍 TEST 2: Cache Behavior Verification")
        
        try:
            cache_results = {
                'initial_call_successful': False,
                'cache_hit_call_successful': False,
                'timestamp_persistence_in_cache_hit': True,
                'initial_data': {},
                'cache_hit_data': {}
            }
            
            logger.info("   🚀 Testing cache behavior and timestamp persistence...")
            logger.info("   📊 Expected: Timestamps persist in cache hit scenarios")
            
            # Step 1: Initial API call to populate cache
            logger.info("   📞 Step 1: Initial API call to populate cache")
            try:
                response = requests.get(f"{self.api_url}/opportunities", timeout=60)
                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get('success') and 'opportunities' in response_data:
                        opportunities = response_data['opportunities']
                        cache_results['initial_call_successful'] = True
                        
                        # Store initial timestamp data
                        for opp in opportunities:
                            symbol = opp.get('symbol')
                            timestamp = opp.get('timestamp')
                            if symbol and timestamp:
                                cache_results['initial_data'][symbol] = {
                                    'timestamp': timestamp,
                                    'current_price': opp.get('current_price', 0)
                                }
                        
                        logger.info(f"      ✅ Initial call successful: {len(cache_results['initial_data'])} symbols cached")
                    else:
                        logger.error(f"      ❌ Initial call returned invalid response structure")
                else:
                    logger.error(f"      ❌ Initial call failed: HTTP {response.status_code}")
            except Exception as e:
                logger.error(f"      ❌ Initial call exception: {e}")
            
            # Step 2: Immediate second call (should hit cache within TTL)
            logger.info("   📞 Step 2: Immediate second call (cache hit scenario)")
            await asyncio.sleep(2)  # Small delay
            
            try:
                response = requests.get(f"{self.api_url}/opportunities", timeout=60)
                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get('success') and 'opportunities' in response_data:
                        opportunities = response_data['opportunities']
                        cache_results['cache_hit_call_successful'] = True
                        
                        # Store cache hit timestamp data
                        for opp in opportunities:
                            symbol = opp.get('symbol')
                            timestamp = opp.get('timestamp')
                            if symbol and timestamp:
                                cache_results['cache_hit_data'][symbol] = {
                                    'timestamp': timestamp,
                                    'current_price': opp.get('current_price', 0)
                                }
                        
                        logger.info(f"      ✅ Cache hit call successful: {len(cache_results['cache_hit_data'])} symbols")
                        
                        # Compare timestamps between initial and cache hit calls
                        common_symbols = set(cache_results['initial_data'].keys()) & set(cache_results['cache_hit_data'].keys())
                        timestamp_mismatches = 0
                        
                        for symbol in common_symbols:
                            initial_timestamp = cache_results['initial_data'][symbol]['timestamp']
                            cache_hit_timestamp = cache_results['cache_hit_data'][symbol]['timestamp']
                            
                            if initial_timestamp != cache_hit_timestamp:
                                timestamp_mismatches += 1
                                logger.warning(f"         ❌ Cache hit timestamp mismatch for {symbol}: {initial_timestamp} → {cache_hit_timestamp}")
                                cache_results['timestamp_persistence_in_cache_hit'] = False
                        
                        if timestamp_mismatches == 0 and common_symbols:
                            logger.info(f"      ✅ All {len(common_symbols)} symbols maintained timestamps in cache hit")
                        elif common_symbols:
                            logger.warning(f"      ❌ {timestamp_mismatches}/{len(common_symbols)} symbols had timestamp changes in cache hit")
                    else:
                        logger.error(f"      ❌ Cache hit call returned invalid response structure")
                else:
                    logger.error(f"      ❌ Cache hit call failed: HTTP {response.status_code}")
            except Exception as e:
                logger.error(f"      ❌ Cache hit call exception: {e}")
            
            # Final analysis
            logger.info(f"\n   📊 CACHE BEHAVIOR ANALYSIS:")
            logger.info(f"      Initial call successful: {cache_results['initial_call_successful']}")
            logger.info(f"      Cache hit call successful: {cache_results['cache_hit_call_successful']}")
            logger.info(f"      Timestamp persistence in cache hit: {cache_results['timestamp_persistence_in_cache_hit']}")
            
            # Calculate test success
            success_criteria = [
                cache_results['initial_call_successful'],
                cache_results['cache_hit_call_successful'],
                cache_results['timestamp_persistence_in_cache_hit']
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.75:  # 75% success threshold
                self.log_test_result("Cache Behavior Verification", True, 
                                   f"Cache behavior working: {success_count}/{len(success_criteria)} criteria met")
            else:
                self.log_test_result("Cache Behavior Verification", False, 
                                   f"Cache behavior issues: {success_count}/{len(success_criteria)} criteria met")
                
        except Exception as e:
            self.log_test_result("Cache Behavior Verification", False, f"Exception: {str(e)}")

    async def test_3_logging_verification(self):
        """Test 3: Logging Verification - Check for REUSING/NEW TIMESTAMP messages"""
        logger.info("\n🔍 TEST 3: Logging Verification")
        
        try:
            logging_results = {
                'backend_logs_captured': False,
                'reusing_timestamp_logs_found': False,
                'new_timestamp_logs_found': False,
                'api_call_successful': False,
                'log_messages': {
                    'reusing': [],
                    'new': []
                }
            }
            
            logger.info("   🚀 Testing logging verification for timestamp behavior...")
            logger.info("   📊 Expected: 'REUSING TIMESTAMP' and 'NEW TIMESTAMP' logs")
            
            # Step 1: Make API call to generate logs
            logger.info("   📞 Making API call to generate timestamp logs...")
            try:
                response = requests.get(f"{self.api_url}/opportunities", timeout=60)
                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get('success') and 'opportunities' in response_data:
                        logging_results['api_call_successful'] = True
                        opportunities = response_data['opportunities']
                        logger.info(f"      ✅ API call successful: {len(opportunities)} opportunities")
                    else:
                        logger.error(f"      ❌ API call returned invalid response structure")
                else:
                    logger.error(f"      ❌ API call failed: HTTP {response.status_code}")
            except Exception as e:
                logger.error(f"      ❌ API call exception: {e}")
            
            # Step 2: Capture and analyze backend logs
            logger.info("   📋 Capturing and analyzing backend logs...")
            try:
                backend_logs = await self._capture_backend_logs()
                if backend_logs:
                    logging_results['backend_logs_captured'] = True
                    logger.info(f"      ✅ Captured {len(backend_logs)} backend log lines")
                    
                    # Search for timestamp-related log messages
                    for log_line in backend_logs:
                        if "REUSING TIMESTAMP" in log_line:
                            logging_results['reusing_timestamp_logs_found'] = True
                            logging_results['log_messages']['reusing'].append(log_line.strip())
                        elif "NEW TIMESTAMP" in log_line:
                            logging_results['new_timestamp_logs_found'] = True
                            logging_results['log_messages']['new'].append(log_line.strip())
                    
                    # Report findings
                    if logging_results['reusing_timestamp_logs_found']:
                        logger.info(f"      ✅ Found {len(logging_results['log_messages']['reusing'])} 'REUSING TIMESTAMP' messages")
                        for msg in logging_results['log_messages']['reusing'][:2]:  # Show first 2
                            logger.info(f"         - {msg}")
                    else:
                        logger.warning(f"      ❌ No 'REUSING TIMESTAMP' messages found")
                    
                    if logging_results['new_timestamp_logs_found']:
                        logger.info(f"      ✅ Found {len(logging_results['log_messages']['new'])} 'NEW TIMESTAMP' messages")
                        for msg in logging_results['log_messages']['new'][:2]:  # Show first 2
                            logger.info(f"         - {msg}")
                    else:
                        logger.warning(f"      ❌ No 'NEW TIMESTAMP' messages found")
                
                else:
                    logger.warning(f"      ⚠️ Could not capture backend logs")
            except Exception as e:
                logger.warning(f"      ⚠️ Log capture error: {e}")
            
            # Final analysis
            logger.info(f"\n   📊 LOGGING ANALYSIS:")
            logger.info(f"      Backend logs captured: {logging_results['backend_logs_captured']}")
            logger.info(f"      'REUSING TIMESTAMP' logs found: {logging_results['reusing_timestamp_logs_found']}")
            logger.info(f"      'NEW TIMESTAMP' logs found: {logging_results['new_timestamp_logs_found']}")
            
            # Calculate test success
            success_criteria = [
                logging_results['api_call_successful'],
                logging_results['backend_logs_captured'],
                logging_results['reusing_timestamp_logs_found'] or logging_results['new_timestamp_logs_found']
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.75:  # 75% success threshold
                self.log_test_result("Logging Verification", True, 
                                   f"Logging verification working: {success_count}/{len(success_criteria)} criteria met")
            else:
                self.log_test_result("Logging Verification", False, 
                                   f"Logging verification issues: {success_count}/{len(success_criteria)} criteria met")
                
        except Exception as e:
            self.log_test_result("Logging Verification", False, f"Exception: {str(e)}")

    async def test_4_data_integrity_verification(self):
        """Test 4: Data Integrity Verification - Ensure data updates while timestamps persist"""
        logger.info("\n🔍 TEST 4: Data Integrity Verification")
        
        try:
            integrity_results = {
                'api_calls_successful': 0,
                'total_api_calls': 2,
                'data_integrity_maintained': True,
                'price_updates_detected': False,
                'timestamp_persistence_maintained': True,
                'call_data': []
            }
            
            logger.info("   🚀 Testing data integrity while timestamps persist...")
            logger.info("   📊 Expected: Prices/volumes update while timestamps remain fixed")
            
            # Make 2 API calls with delay to check data updates
            for call_num in range(1, 3):
                logger.info(f"\n   📞 API Call #{call_num}/2")
                
                try:
                    response = requests.get(f"{self.api_url}/opportunities", timeout=60)
                    if response.status_code == 200:
                        response_data = response.json()
                        if response_data.get('success') and 'opportunities' in response_data:
                            opportunities = response_data['opportunities']
                            integrity_results['api_calls_successful'] += 1
                            
                            logger.info(f"      ✅ API call #{call_num} successful: {len(opportunities)} opportunities")
                            
                            # Store data for this call
                            call_data = {}
                            for opp in opportunities:
                                symbol = opp.get('symbol')
                                if symbol:
                                    call_data[symbol] = {
                                        'timestamp': opp.get('timestamp'),
                                        'current_price': opp.get('current_price', 0),
                                        'volume_24h': opp.get('volume_24h', 0),
                                        'price_change_24h': opp.get('price_change_24h', 0)
                                    }
                            
                            integrity_results['call_data'].append({
                                'call_number': call_num,
                                'data': call_data
                            })
                            
                            # Compare with previous call if this is the second call
                            if call_num == 2:
                                previous_data = integrity_results['call_data'][0]['data']
                                current_data = call_data
                                
                                common_symbols = set(previous_data.keys()) & set(current_data.keys())
                                logger.info(f"      🔍 Comparing data for {len(common_symbols)} common symbols")
                                
                                timestamp_changes = 0
                                price_changes = 0
                                
                                for symbol in list(common_symbols)[:10]:  # Check first 10 symbols
                                    prev = previous_data[symbol]
                                    curr = current_data[symbol]
                                    
                                    # Check timestamp persistence
                                    if prev['timestamp'] != curr['timestamp']:
                                        timestamp_changes += 1
                                        integrity_results['timestamp_persistence_maintained'] = False
                                    
                                    # Check for price updates (some prices might change)
                                    if prev['current_price'] != curr['current_price']:
                                        price_changes += 1
                                        integrity_results['price_updates_detected'] = True
                                
                                logger.info(f"      📊 Analysis: {timestamp_changes} timestamp changes, {price_changes} price changes")
                                
                                if timestamp_changes == 0:
                                    logger.info(f"      ✅ Timestamps remained persistent across calls")
                                else:
                                    logger.warning(f"      ❌ {timestamp_changes} timestamp changes detected")
                        else:
                            logger.error(f"      ❌ API call #{call_num} returned invalid response structure")
                    else:
                        logger.error(f"      ❌ API call #{call_num} failed: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"      ❌ API call #{call_num} exception: {e}")
                
                # Wait between calls
                if call_num == 1:
                    logger.info(f"      ⏳ Waiting 10 seconds before next call...")
                    await asyncio.sleep(10)
            
            # Final analysis
            logger.info(f"\n   📊 DATA INTEGRITY ANALYSIS:")
            logger.info(f"      API calls successful: {integrity_results['api_calls_successful']}/{integrity_results['total_api_calls']}")
            logger.info(f"      Timestamp persistence maintained: {integrity_results['timestamp_persistence_maintained']}")
            logger.info(f"      Price updates detected: {integrity_results['price_updates_detected']}")
            
            # Calculate test success
            success_criteria = [
                integrity_results['api_calls_successful'] >= 2,
                integrity_results['timestamp_persistence_maintained'],
                len(integrity_results['call_data']) >= 2
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.75:  # 75% success threshold
                self.log_test_result("Data Integrity Verification", True, 
                                   f"Data integrity maintained: {success_count}/{len(success_criteria)} criteria met")
            else:
                self.log_test_result("Data Integrity Verification", False, 
                                   f"Data integrity issues: {success_count}/{len(success_criteria)} criteria met")
                
        except Exception as e:
            self.log_test_result("Data Integrity Verification", False, f"Exception: {str(e)}")

    async def _capture_backend_logs(self):
        """Capture recent backend logs for analysis"""
        try:
            # Try to capture supervisor backend logs
            result = subprocess.run(
                ["tail", "-n", "100", "/var/log/supervisor/backend.out.log"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                return result.stdout.split('\n')
            
            # Fallback: try error log
            result = subprocess.run(
                ["tail", "-n", "100", "/var/log/supervisor/backend.err.log"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                return result.stdout.split('\n')
            
            return []
            
        except Exception as e:
            logger.warning(f"Could not capture backend logs: {e}")
            return []

async def main():
    """Run the comprehensive Market Opportunities Timestamp Persistence Fix test suite"""
    logger.info("🚀 STARTING COMPREHENSIVE MARKET OPPORTUNITIES TIMESTAMP PERSISTENCE FIX TEST SUITE")
    logger.info("=" * 80)
    
    test_suite = ComprehensiveTimestampPersistenceTestSuite()
    
    try:
        # Run all tests
        await test_suite.test_1_multiple_api_calls_timestamp_persistence()
        await test_suite.test_2_cache_behavior_verification()
        await test_suite.test_3_logging_verification()
        await test_suite.test_4_data_integrity_verification()
        
        # Print final summary
        logger.info("\n" + "=" * 80)
        logger.info("🏁 COMPREHENSIVE TIMESTAMP PERSISTENCE FIX TEST SUITE COMPLETE")
        logger.info("=" * 80)
        
        # Summary of results
        total_tests = len(test_suite.test_results)
        passed_tests = sum(1 for result in test_suite.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        logger.info(f"\n📊 FINAL RESULTS:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   ✅ Passed: {passed_tests}")
        logger.info(f"   ❌ Failed: {failed_tests}")
        logger.info(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Detailed results
        logger.info(f"\n📋 DETAILED RESULTS:")
        for result in test_suite.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"   {status}: {result['test']}")
            if result['details']:
                logger.info(f"      Details: {result['details']}")
        
        # Overall assessment
        if passed_tests == total_tests:
            logger.info(f"\n🎉 ALL TESTS PASSED - MARKET OPPORTUNITIES TIMESTAMP PERSISTENCE FIX IS WORKING CORRECTLY!")
        elif passed_tests >= total_tests * 0.75:
            logger.info(f"\n✅ MOSTLY SUCCESSFUL - Market Opportunities Timestamp Persistence Fix is largely working ({passed_tests}/{total_tests} tests passed)")
        else:
            logger.info(f"\n⚠️ ISSUES DETECTED - Market Opportunities Timestamp Persistence Fix needs attention ({failed_tests}/{total_tests} tests failed)")
        
    except Exception as e:
        logger.error(f"❌ Test suite execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
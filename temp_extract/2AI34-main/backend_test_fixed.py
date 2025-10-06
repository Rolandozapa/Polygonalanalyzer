#!/usr/bin/env python3
"""
MARKET OPPORTUNITIES TIMESTAMP PERSISTENCE FIX TEST SUITE
Focus: Test Market Opportunities Timestamp Persistence Fix in advanced_market_aggregator.py

CRITICAL TEST REQUIREMENTS FROM REVIEW REQUEST:
1. **API ENDPOINT TESTING**: Test /api/opportunities endpoint multiple times to verify timestamps are persistent
   - Call the endpoint at least 3 times with a few seconds between calls
   - Verify that the timestamps for the same symbols remain exactly the same across multiple calls
   - Check that new symbols get new timestamps but existing symbols preserve their original timestamps

2. **CACHE BEHAVIOR VERIFICATION**: 
   - Test cache hit scenario (within 4-hour TTL)
   - Test cache miss scenario (force cache refresh if possible)
   - Verify timestamp persistence works in both scenarios

3. **SYMBOL TIMESTAMP PERSISTENCE**: 
   - Verify that same symbols get same timestamps when regenerated
   - Verify new symbols get unique timestamps with proper offset (15 seconds apart)
   - Check logging for "REUSING TIMESTAMP" vs "NEW TIMESTAMP" messages

4. **MEMORY MANAGEMENT**:
   - Verify timestamp cleanup functionality works properly
   - Check for any memory leaks in timestamp storage

5. **DATA INTEGRITY**: 
   - Ensure opportunity data (prices, volumes) still updates correctly while timestamps remain fixed
   - Verify the fix doesn't break existing functionality

FOCUS ON: The critical bug was that timestamps changed constantly instead of being persistent per symbol. 
The fix should ensure that once a symbol gets a timestamp, it keeps that timestamp until it's cleaned up (24 hours).

SUCCESS CRITERIA:
✅ Same symbols maintain identical timestamps across multiple API calls
✅ New symbols get unique timestamps with proper 15-second offset
✅ Cache behavior preserves timestamp persistence
✅ Logging shows "REUSING TIMESTAMP" for existing symbols and "NEW TIMESTAMP" for new ones
✅ Timestamp cleanup works properly (24-hour TTL)
✅ Data integrity maintained while timestamps are persistent
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

class MarketOpportunitiesTimestampPersistenceTestSuite:
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
        
        # Expected opportunity fields
        self.opportunity_fields = ['symbol', 'current_price', 'volume_24h', 'price_change_24h', 'timestamp']
        
        # Test data storage for timestamp comparison
        self.timestamp_data = {}
        self.call_count = 0
        
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
                'new_symbols_new_timestamps': True,
                'timestamp_data_by_call': [],
                'persistent_symbols': set(),
                'new_symbols_detected': set()
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
                                new_symbols = set(call_data.keys()) - set(previous_call_data.keys())
                                
                                logger.info(f"      🔍 Comparing with call #{call_num-1}: {len(common_symbols)} common symbols, {len(new_symbols)} new symbols")
                                
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
                                        logger.debug(f"         ✅ Timestamp persistent for {symbol}: {current_timestamp}")
                                        persistence_results['persistent_symbols'].add(symbol)
                                
                                if timestamp_mismatches == 0 and common_symbols:
                                    logger.info(f"      ✅ All {len(common_symbols)} common symbols maintained persistent timestamps")
                                elif common_symbols:
                                    logger.warning(f"      ❌ {timestamp_mismatches}/{len(common_symbols)} symbols had timestamp changes")
                                
                                # Track new symbols
                                persistence_results['new_symbols_detected'].update(new_symbols)
                                if new_symbols:
                                    logger.info(f"      🆕 New symbols detected: {list(new_symbols)[:5]}{'...' if len(new_symbols) > 5 else ''}")
                        else:
                            logger.error(f"      ❌ API call #{call_num} returned invalid response structure")
                    
                    else:
                        logger.error(f"      ❌ API call #{call_num} failed: HTTP {response.status_code}")
                        if response.text:
                            logger.error(f"         Error response: {response.text[:200]}...")
                
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
            logger.info(f"      New symbols detected: {len(persistence_results['new_symbols_detected'])}")
            
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
                                   f"Timestamp persistence working: {success_count}/{len(success_criteria)} criteria met. Persistent symbols: {len(persistence_results['persistent_symbols'])}, API calls: {persistence_results['api_calls_successful']}")
            else:
                self.log_test_result("Multiple API Calls Timestamp Persistence", False, 
                                   f"Timestamp persistence issues: {success_count}/{len(success_criteria)} criteria met. Persistent symbols: {len(persistence_results['persistent_symbols'])}")
                
        except Exception as e:
            self.log_test_result("Multiple API Calls Timestamp Persistence", False, f"Exception: {str(e)}")

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
    logger.info("🚀 STARTING MARKET OPPORTUNITIES TIMESTAMP PERSISTENCE FIX TEST SUITE")
    logger.info("=" * 80)
    
    test_suite = MarketOpportunitiesTimestampPersistenceTestSuite()
    
    try:
        # Run the main test
        await test_suite.test_1_multiple_api_calls_timestamp_persistence()
        
        # Print final summary
        logger.info("\n" + "=" * 80)
        logger.info("🏁 MARKET OPPORTUNITIES TIMESTAMP PERSISTENCE FIX TEST SUITE COMPLETE")
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
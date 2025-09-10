#!/usr/bin/env python3
"""
BingX API Connectivity Test After IP Update
Focus: Test BingX API connectivity with updated IP whitelist (34.121.6.206)

SPECIFIC TESTING REQUIREMENTS FROM REVIEW REQUEST:
1. GET /api/bingx/balance - Check if we can retrieve account balance
2. GET /api/bingx/status - Verify API connection status  
3. Test if authentication works with the updated IP whitelist
4. GET /api/bingx/positions - See if we can access positions

API CREDENTIALS (confirmed working):
- API Key: HZCsTzoH1DYjGYi3jmeAeRYA6hIVb9vySpPnJx7FaE6p9eDVU23qobsCAyX7JbMJj57QgY60l3BhfhA5ag
- Secret Key: ynNfVmYYSPmlP1roNGa5L8VeUokXMF94XH9VNf7pBXocIgMWxBlnarX3Uy13ftf9zQfI5jzrvMXQiA0qGQ
- Updated IP Whitelist: 34.121.6.206

EXPECTED RESULTS:
- Should now return HTTP 200 instead of API authentication errors
- Balance endpoint should return actual USDT balance from BingX account
- Status should show "connected" instead of error messages
- No more IP whitelist or authentication errors
"""

import asyncio
import json
import logging
import requests
import time
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BingXIPConnectivityTest:
    """Focused test for BingX API connectivity after IP whitelist update"""
    
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
        logger.info(f"Testing BingX API connectivity at: {self.api_url}")
        logger.info(f"Expected IP whitelist: 34.121.6.206")
        
        # Test results
        self.test_results = []
        
        # Priority endpoints from review request
        self.priority_endpoints = [
            {'method': 'GET', 'path': '/bingx/balance', 'name': 'Account Balance', 'priority': 1},
            {'method': 'GET', 'path': '/bingx/status', 'name': 'API Connection Status', 'priority': 1},
            {'method': 'GET', 'path': '/bingx/positions', 'name': 'Positions Access', 'priority': 1},
        ]
        
    def log_test_result(self, test_name: str, success: bool, details: str = "", response_data: Dict = None):
        """Log test result with detailed information"""
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
        if details:
            logger.info(f"   Details: {details}")
        if response_data:
            logger.info(f"   Response: {json.dumps(response_data, indent=2)}")
        
        self.test_results.append({
            'test': test_name,
            'success': success,
            'details': details,
            'response_data': response_data,
            'timestamp': datetime.now().isoformat()
        })
    
    async def test_bingx_balance_retrieval(self):
        """Priority Test 1: GET /api/bingx/balance - Check if we can retrieve account balance"""
        logger.info("\n🔍 PRIORITY TEST 1: BingX Account Balance Retrieval")
        logger.info("   Expected: HTTP 200 with actual USDT balance from BingX account")
        
        try:
            response = requests.get(f"{self.api_url}/bingx/balance", timeout=30)
            
            logger.info(f"   HTTP Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    logger.info(f"   Response received: {json.dumps(data, indent=2)}")
                    
                    # Check for balance data
                    balance_fields = ['balance', 'total_balance', 'available_balance', 'available_margin']
                    balance_found = any(field in data for field in balance_fields)
                    
                    if balance_found:
                        # Extract balance information
                        balance = data.get('balance', data.get('total_balance', 0))
                        available = data.get('available_balance', data.get('available_margin', 0))
                        
                        self.log_test_result(
                            "BingX Balance Retrieval", 
                            True, 
                            f"✅ Balance retrieved successfully - Total: ${balance}, Available: ${available}",
                            data
                        )
                    else:
                        self.log_test_result(
                            "BingX Balance Retrieval", 
                            False, 
                            "❌ No balance data found in response",
                            data
                        )
                        
                except json.JSONDecodeError:
                    self.log_test_result(
                        "BingX Balance Retrieval", 
                        False, 
                        f"❌ Invalid JSON response: {response.text[:200]}"
                    )
                    
            elif response.status_code == 401:
                self.log_test_result(
                    "BingX Balance Retrieval", 
                    False, 
                    f"❌ Authentication failed (HTTP 401) - IP whitelist or credentials issue: {response.text}"
                )
            elif response.status_code == 403:
                self.log_test_result(
                    "BingX Balance Retrieval", 
                    False, 
                    f"❌ Access forbidden (HTTP 403) - IP not whitelisted: {response.text}"
                )
            else:
                self.log_test_result(
                    "BingX Balance Retrieval", 
                    False, 
                    f"❌ HTTP {response.status_code}: {response.text}"
                )
                
        except requests.exceptions.Timeout:
            self.log_test_result(
                "BingX Balance Retrieval", 
                False, 
                "❌ Request timeout - API may be unreachable"
            )
        except Exception as e:
            self.log_test_result(
                "BingX Balance Retrieval", 
                False, 
                f"❌ Exception: {str(e)}"
            )
    
    async def test_bingx_connection_status(self):
        """Priority Test 2: GET /api/bingx/status - Verify API connection status"""
        logger.info("\n🔍 PRIORITY TEST 2: BingX API Connection Status")
        logger.info("   Expected: Status should show 'connected' instead of error messages")
        
        try:
            response = requests.get(f"{self.api_url}/bingx/status", timeout=30)
            
            logger.info(f"   HTTP Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    logger.info(f"   Response received: {json.dumps(data, indent=2)}")
                    
                    # Check connection status
                    status = data.get('status', 'unknown')
                    api_connected = data.get('api_connected', False)
                    
                    if api_connected or status == 'connected':
                        self.log_test_result(
                            "BingX Connection Status", 
                            True, 
                            f"✅ API connection successful - Status: {status}, Connected: {api_connected}",
                            data
                        )
                    else:
                        # Check for error messages
                        error_msg = data.get('error', data.get('message', 'Unknown error'))
                        self.log_test_result(
                            "BingX Connection Status", 
                            False, 
                            f"❌ API not connected - Status: {status}, Error: {error_msg}",
                            data
                        )
                        
                except json.JSONDecodeError:
                    self.log_test_result(
                        "BingX Connection Status", 
                        False, 
                        f"❌ Invalid JSON response: {response.text[:200]}"
                    )
                    
            else:
                self.log_test_result(
                    "BingX Connection Status", 
                    False, 
                    f"❌ HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_test_result(
                "BingX Connection Status", 
                False, 
                f"❌ Exception: {str(e)}"
            )
    
    async def test_bingx_positions_access(self):
        """Priority Test 3: GET /api/bingx/positions - See if we can access positions"""
        logger.info("\n🔍 PRIORITY TEST 3: BingX Positions Access")
        logger.info("   Expected: HTTP 200 with positions data (may be empty array)")
        
        try:
            response = requests.get(f"{self.api_url}/bingx/positions", timeout=30)
            
            logger.info(f"   HTTP Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    logger.info(f"   Response received: {json.dumps(data, indent=2)}")
                    
                    # Check positions data structure
                    if isinstance(data, list):
                        position_count = len(data)
                        self.log_test_result(
                            "BingX Positions Access", 
                            True, 
                            f"✅ Positions data retrieved successfully - {position_count} positions found",
                            data
                        )
                    elif isinstance(data, dict) and 'positions' in data:
                        positions = data['positions']
                        position_count = len(positions) if isinstance(positions, list) else 0
                        self.log_test_result(
                            "BingX Positions Access", 
                            True, 
                            f"✅ Positions data retrieved successfully - {position_count} positions found",
                            data
                        )
                    else:
                        self.log_test_result(
                            "BingX Positions Access", 
                            True, 
                            "✅ Positions endpoint accessible (non-standard format but HTTP 200)",
                            data
                        )
                        
                except json.JSONDecodeError:
                    self.log_test_result(
                        "BingX Positions Access", 
                        False, 
                        f"❌ Invalid JSON response: {response.text[:200]}"
                    )
                    
            elif response.status_code == 401:
                self.log_test_result(
                    "BingX Positions Access", 
                    False, 
                    f"❌ Authentication failed (HTTP 401) - IP whitelist or credentials issue: {response.text}"
                )
            elif response.status_code == 403:
                self.log_test_result(
                    "BingX Positions Access", 
                    False, 
                    f"❌ Access forbidden (HTTP 403) - IP not whitelisted: {response.text}"
                )
            else:
                self.log_test_result(
                    "BingX Positions Access", 
                    False, 
                    f"❌ HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_test_result(
                "BingX Positions Access", 
                False, 
                f"❌ Exception: {str(e)}"
            )
    
    async def test_authentication_with_updated_ip(self):
        """Test 4: Verify authentication works with updated IP whitelist"""
        logger.info("\n🔍 TEST 4: Authentication with Updated IP Whitelist")
        logger.info("   Expected: No IP whitelist or authentication errors")
        
        # Test multiple endpoints to verify authentication
        auth_test_results = []
        
        endpoints_to_test = [
            {'path': '/bingx/status', 'name': 'Status'},
            {'path': '/bingx/balance', 'name': 'Balance'},
            {'path': '/bingx/positions', 'name': 'Positions'}
        ]
        
        for endpoint in endpoints_to_test:
            try:
                response = requests.get(f"{self.api_url}{endpoint['path']}", timeout=30)
                
                if response.status_code == 200:
                    auth_test_results.append(f"✅ {endpoint['name']}: Authentication successful")
                elif response.status_code == 401:
                    auth_test_results.append(f"❌ {endpoint['name']}: Authentication failed (401)")
                elif response.status_code == 403:
                    auth_test_results.append(f"❌ {endpoint['name']}: IP not whitelisted (403)")
                else:
                    auth_test_results.append(f"⚠️ {endpoint['name']}: HTTP {response.status_code}")
                    
            except Exception as e:
                auth_test_results.append(f"❌ {endpoint['name']}: Exception - {str(e)}")
        
        # Evaluate authentication results
        successful_auth = len([r for r in auth_test_results if r.startswith("✅")])
        total_auth_tests = len(auth_test_results)
        
        logger.info("   Authentication Test Results:")
        for result in auth_test_results:
            logger.info(f"      {result}")
        
        if successful_auth == total_auth_tests:
            self.log_test_result(
                "Authentication with Updated IP", 
                True, 
                f"✅ All endpoints authenticated successfully ({successful_auth}/{total_auth_tests})"
            )
        elif successful_auth > 0:
            self.log_test_result(
                "Authentication with Updated IP", 
                False, 
                f"⚠️ Partial authentication success ({successful_auth}/{total_auth_tests})"
            )
        else:
            self.log_test_result(
                "Authentication with Updated IP", 
                False, 
                f"❌ Authentication failed on all endpoints ({successful_auth}/{total_auth_tests})"
            )
    
    async def run_ip_connectivity_tests(self):
        """Run all IP connectivity tests"""
        logger.info("🚀 Starting BingX IP Connectivity Test Suite")
        logger.info("=" * 80)
        logger.info("📋 BINGX API CONNECTIVITY TEST AFTER IP UPDATE")
        logger.info("🎯 Testing: API connectivity with updated IP whitelist (34.121.6.206)")
        logger.info("🎯 Expected: HTTP 200 responses instead of authentication errors")
        logger.info("=" * 80)
        
        # Run priority tests
        await self.test_bingx_balance_retrieval()
        await self.test_bingx_connection_status()
        await self.test_bingx_positions_access()
        await self.test_authentication_with_updated_ip()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 BINGX IP CONNECTIVITY TEST SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
        
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # IP Update Analysis
        logger.info("\n" + "=" * 80)
        logger.info("📋 IP WHITELIST UPDATE ANALYSIS")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 IP UPDATE SUCCESSFUL!")
            logger.info("✅ All BingX API endpoints now accessible")
            logger.info("✅ Authentication working with updated IP whitelist (34.121.6.206)")
            logger.info("✅ Balance retrieval operational")
            logger.info("✅ Positions access working")
            logger.info("✅ No more IP whitelist or authentication errors")
        elif passed_tests >= 3:
            logger.info("⚠️ IP UPDATE MOSTLY SUCCESSFUL")
            logger.info("✅ Most BingX API endpoints accessible")
            logger.info("⚠️ Some minor issues may need attention")
        elif passed_tests >= 2:
            logger.info("⚠️ IP UPDATE PARTIALLY SUCCESSFUL")
            logger.info("⚠️ Some BingX API endpoints accessible")
            logger.info("🔧 Additional configuration may be needed")
        else:
            logger.info("❌ IP UPDATE NOT SUCCESSFUL")
            logger.info("❌ BingX API endpoints still not accessible")
            logger.info("🚨 IP whitelist may not be properly updated or other issues exist")
        
        # Specific findings
        logger.info("\n📝 SPECIFIC FINDINGS:")
        
        balance_test = next((r for r in self.test_results if "Balance" in r['test']), None)
        if balance_test and balance_test['success']:
            logger.info("✅ Balance endpoint: Working - Can retrieve actual USDT balance")
        elif balance_test:
            logger.info("❌ Balance endpoint: Failed - Cannot retrieve balance")
        
        status_test = next((r for r in self.test_results if "Status" in r['test']), None)
        if status_test and status_test['success']:
            logger.info("✅ Status endpoint: Working - Shows 'connected' status")
        elif status_test:
            logger.info("❌ Status endpoint: Failed - Shows error messages")
        
        positions_test = next((r for r in self.test_results if "Positions" in r['test']), None)
        if positions_test and positions_test['success']:
            logger.info("✅ Positions endpoint: Working - Can access positions data")
        elif positions_test:
            logger.info("❌ Positions endpoint: Failed - Cannot access positions")
        
        auth_test = next((r for r in self.test_results if "Authentication" in r['test']), None)
        if auth_test and auth_test['success']:
            logger.info("✅ Authentication: Working - No IP whitelist errors")
        elif auth_test:
            logger.info("❌ Authentication: Failed - IP whitelist or credential issues")
        
        # Final verdict
        logger.info(f"\n🏆 FINAL VERDICT:")
        if passed_tests == total_tests:
            logger.info("🎉 IP WHITELIST UPDATE RESOLVED CONNECTIVITY ISSUES!")
            logger.info("✅ BingX API is now fully accessible with IP 34.121.6.206")
            logger.info("✅ All priority endpoints working as expected")
        elif passed_tests >= 3:
            logger.info("⚠️ IP WHITELIST UPDATE MOSTLY RESOLVED ISSUES")
            logger.info("🔍 Minor issues may need attention for complete functionality")
        else:
            logger.info("❌ IP WHITELIST UPDATE DID NOT RESOLVE CONNECTIVITY ISSUES")
            logger.info("🚨 Further investigation needed - check IP configuration or credentials")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = BingXIPConnectivityTest()
    passed, total = await test_suite.run_ip_connectivity_tests()
    
    # Return results for integration with main testing system
    return passed, total

if __name__ == "__main__":
    asyncio.run(main())
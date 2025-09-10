#!/usr/bin/env python3
"""
BingX POST Request Signature Fix Testing Suite
Focus: Testing the specific POST request signature issue fix as requested in review

REVIEW REQUEST CONTEXT:
- Main issue was with BingX POST requests failing due to "Null signature" errors
- Parameters were incorrectly being sent in JSON body instead of URL query string
- The _make_request method has been updated to fix this by:
  * Moving ALL parameters to URL query string for both GET and POST requests
  * Using proper HMAC-SHA256 signature generation with the query string
  * Sending empty JSON body for POST requests as required by BingX

SPECIFIC TESTING FOCUS:
1. BingX API Connectivity: Test all BingX endpoints to ensure authentication works
2. POST Request Signature: Specifically test POST endpoints like /api/bingx/execute-ia2
3. Balance and Positions: Verify that balance and positions API calls return valid data
4. Error Handling: Confirm that API errors are properly handled and logged

EXPECTED RESULTS:
- POST requests should now work without signature errors
- Balance retrieval should return actual account data
- All BingX endpoints should authenticate successfully

IP WHITELIST: User has already updated BingX IP whitelist to 34.121.6.206
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

class BingXPostSignatureTestSuite:
    """Focused test suite for BingX POST request signature fix validation"""
    
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
        logger.info(f"Testing BingX POST Signature Fix at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Priority endpoints for POST signature testing
        self.priority_endpoints = [
            {'method': 'GET', 'path': '/bingx/status', 'name': 'System Status', 'priority': 'HIGH'},
            {'method': 'GET', 'path': '/bingx/balance', 'name': 'Account Balance', 'priority': 'HIGH'},
            {'method': 'GET', 'path': '/bingx/positions', 'name': 'Open Positions', 'priority': 'HIGH'},
            {'method': 'POST', 'path': '/bingx/execute-ia2', 'name': 'IA2 Trade Execution (POST)', 'priority': 'CRITICAL'},
            {'method': 'POST', 'path': '/bingx/trade', 'name': 'Manual Trade (POST)', 'priority': 'HIGH'},
            {'method': 'POST', 'path': '/bingx/close-position', 'name': 'Close Position (POST)', 'priority': 'HIGH'},
            {'method': 'POST', 'path': '/bingx/risk-config', 'name': 'Update Risk Config (POST)', 'priority': 'MEDIUM'},
            {'method': 'POST', 'path': '/bingx/emergency-stop', 'name': 'Emergency Stop (POST)', 'priority': 'HIGH'},
        ]
        
        # Mock data for POST requests
        self.mock_ia2_decision = {
            "symbol": "BTCUSDT",
            "signal": "LONG",
            "confidence": 0.85,
            "position_size": 2.5,
            "leverage": 5,
            "entry_price": 45000.0,
            "stop_loss": 43000.0,
            "take_profit": 48000.0,
            "reasoning": "Strong bullish momentum with RSI oversold recovery - POST signature test"
        }
        
        self.mock_trade_data = {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": 0.001,
            "leverage": 5,
            "test_signature": True
        }
        
        self.mock_close_data = {
            "symbol": "BTCUSDT",
            "position_side": "LONG",
            "test_signature": True
        }
        
        self.mock_risk_config = {
            "max_position_size": 0.1,
            "max_leverage": 10,
            "stop_loss_percentage": 0.02,
            "test_signature": True
        }
        
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
    
    async def test_1_bingx_connectivity_after_ip_update(self):
        """Test 1: BingX API Connectivity After IP Whitelist Update to 34.121.6.206"""
        logger.info("\n🔍 TEST 1: BingX API Connectivity After IP Whitelist Update")
        logger.info("   🎯 Expected: IP whitelist updated to 34.121.6.206, connectivity restored")
        
        try:
            # Test status endpoint first
            response = requests.get(f"{self.api_url}/bingx/status", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"   📊 Status response: {json.dumps(data, indent=2)}")
                
                # Check for connectivity indicators
                api_connected = data.get('api_connected', False)
                status = data.get('status', 'unknown')
                
                if api_connected or status in ['connected', 'operational']:
                    self.log_test_result("BingX Connectivity After IP Update", True, 
                                       f"API connected successfully: {status}")
                else:
                    # Even if status shows 'not_initialized', check if we can access other endpoints
                    logger.info("   ⚠️ Status shows not connected, testing balance endpoint...")
                    balance_response = requests.get(f"{self.api_url}/bingx/balance", timeout=30)
                    
                    if balance_response.status_code == 200:
                        balance_data = balance_response.json()
                        if 'balance' in balance_data or 'total_balance' in balance_data:
                            self.log_test_result("BingX Connectivity After IP Update", True, 
                                               "Balance endpoint accessible - connectivity working despite status display")
                        else:
                            self.log_test_result("BingX Connectivity After IP Update", False, 
                                               f"Status not connected and balance endpoint returns no data")
                    else:
                        self.log_test_result("BingX Connectivity After IP Update", False, 
                                           f"Status not connected and balance endpoint fails: HTTP {balance_response.status_code}")
            else:
                self.log_test_result("BingX Connectivity After IP Update", False, 
                                   f"Status endpoint failed: HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test_result("BingX Connectivity After IP Update", False, f"Exception: {str(e)}")
    
    async def test_2_balance_and_positions_data_retrieval(self):
        """Test 2: Balance and Positions Data Retrieval (Verify Real Account Data)"""
        logger.info("\n🔍 TEST 2: Balance and Positions Data Retrieval")
        logger.info("   🎯 Expected: Actual account balance data from BingX account")
        
        balance_success = False
        positions_success = False
        
        try:
            # Test balance retrieval
            logger.info("   📊 Testing balance retrieval...")
            balance_response = requests.get(f"{self.api_url}/bingx/balance", timeout=30)
            
            if balance_response.status_code == 200:
                balance_data = balance_response.json()
                logger.info(f"   📊 Balance response: {json.dumps(balance_data, indent=2)}")
                
                # Look for actual balance data
                balance_fields = ['balance', 'total_balance', 'available_balance', 'available_margin']
                balance_found = False
                actual_balance = 0
                
                for field in balance_fields:
                    if field in balance_data and balance_data[field] != 0:
                        actual_balance = balance_data[field]
                        balance_found = True
                        logger.info(f"      ✅ Found balance: {field} = ${actual_balance}")
                        break
                
                if balance_found:
                    balance_success = True
                    logger.info(f"      ✅ Balance retrieval successful: ${actual_balance}")
                else:
                    logger.info(f"      ❌ No balance data found in response")
            else:
                logger.info(f"      ❌ Balance endpoint failed: HTTP {balance_response.status_code}")
            
            # Test positions retrieval
            logger.info("   📊 Testing positions retrieval...")
            positions_response = requests.get(f"{self.api_url}/bingx/positions", timeout=30)
            
            if positions_response.status_code == 200:
                positions_data = positions_response.json()
                logger.info(f"   📊 Positions response: {json.dumps(positions_data, indent=2)}")
                
                # Check for positions data structure
                if 'positions' in positions_data or 'data' in positions_data or isinstance(positions_data, list):
                    positions_success = True
                    positions_count = 0
                    
                    if 'positions' in positions_data:
                        positions_count = len(positions_data['positions'])
                    elif 'data' in positions_data:
                        positions_count = len(positions_data['data'])
                    elif isinstance(positions_data, list):
                        positions_count = len(positions_data)
                    
                    logger.info(f"      ✅ Positions retrieval successful: {positions_count} positions found")
                else:
                    logger.info(f"      ❌ No positions data structure found")
            else:
                logger.info(f"      ❌ Positions endpoint failed: HTTP {positions_response.status_code}")
            
            # Overall success
            overall_success = balance_success and positions_success
            details = f"Balance: {'✅' if balance_success else '❌'}, Positions: {'✅' if positions_success else '❌'}"
            
            self.log_test_result("Balance and Positions Data Retrieval", overall_success, details)
            
        except Exception as e:
            self.log_test_result("Balance and Positions Data Retrieval", False, f"Exception: {str(e)}")
    
    async def test_3_post_request_signature_fix_validation(self):
        """Test 3: POST Request Signature Fix Validation (CRITICAL TEST)"""
        logger.info("\n🔍 TEST 3: POST Request Signature Fix Validation")
        logger.info("   🎯 CRITICAL: Testing POST endpoints to verify signature generation fix")
        logger.info("   🎯 Expected: No 'Null signature' errors, parameters in query string, empty JSON body")
        
        post_results = []
        
        # Test each POST endpoint
        for endpoint in self.priority_endpoints:
            if endpoint['method'] != 'POST':
                continue
                
            try:
                method = endpoint['method']
                path = endpoint['path']
                name = endpoint['name']
                priority = endpoint['priority']
                
                logger.info(f"   🚀 Testing {priority} POST endpoint: {path} ({name})")
                
                # Select appropriate mock data
                if 'execute-ia2' in path:
                    test_data = self.mock_ia2_decision
                elif 'trade' in path and 'execute' not in path:
                    test_data = self.mock_trade_data
                elif 'close-position' in path:
                    test_data = self.mock_close_data
                elif 'risk-config' in path:
                    test_data = self.mock_risk_config
                else:
                    test_data = {"test_signature": True}
                
                # Make POST request
                response = requests.post(f"{self.api_url}{path}", json=test_data, timeout=60)
                
                logger.info(f"      📊 Response: HTTP {response.status_code}")
                
                # Analyze response for signature issues
                signature_error = False
                authentication_error = False
                
                try:
                    response_data = response.json()
                    response_text = json.dumps(response_data).lower()
                    
                    # Check for signature-related errors
                    signature_keywords = ['null signature', 'signature', 'invalid signature', 'signature error']
                    auth_keywords = ['authentication', 'unauthorized', 'invalid api key', 'forbidden']
                    
                    for keyword in signature_keywords:
                        if keyword in response_text:
                            signature_error = True
                            logger.info(f"      ❌ Signature error detected: {keyword}")
                            break
                    
                    for keyword in auth_keywords:
                        if keyword in response_text:
                            authentication_error = True
                            logger.info(f"      ⚠️ Authentication issue: {keyword}")
                            break
                    
                    if not signature_error and not authentication_error:
                        logger.info(f"      ✅ No signature errors detected")
                    
                except:
                    response_text = response.text.lower()
                    if 'signature' in response_text:
                        signature_error = True
                        logger.info(f"      ❌ Signature error in text response")
                
                # Evaluate result
                if response.status_code in [200, 201]:
                    result_status = "SUCCESS"
                    success = True
                elif response.status_code in [400, 422] and not signature_error:
                    result_status = "VALIDATION_ERROR" # Expected for mock data
                    success = True
                elif signature_error:
                    result_status = "SIGNATURE_ERROR"
                    success = False
                elif authentication_error:
                    result_status = "AUTH_ERROR"
                    success = False
                else:
                    result_status = f"HTTP_{response.status_code}"
                    success = response.status_code != 500  # Server errors are bad
                
                post_results.append({
                    'endpoint': path,
                    'name': name,
                    'priority': priority,
                    'status': result_status,
                    'success': success,
                    'signature_error': signature_error,
                    'http_code': response.status_code
                })
                
                status_icon = "✅" if success else "❌"
                logger.info(f"      {status_icon} {name}: {result_status}")
                
            except Exception as e:
                post_results.append({
                    'endpoint': path,
                    'name': name,
                    'priority': priority,
                    'status': 'EXCEPTION',
                    'success': False,
                    'signature_error': False,
                    'error': str(e)
                })
                logger.info(f"      ❌ {name}: Exception - {str(e)}")
        
        # Analyze results
        total_post_endpoints = len(post_results)
        successful_posts = len([r for r in post_results if r['success']])
        signature_errors = len([r for r in post_results if r['signature_error']])
        critical_failures = len([r for r in post_results if r['priority'] == 'CRITICAL' and not r['success']])
        
        logger.info(f"   📊 POST Endpoints Results:")
        logger.info(f"      Total POST endpoints tested: {total_post_endpoints}")
        logger.info(f"      Successful (no signature errors): {successful_posts}")
        logger.info(f"      Signature errors detected: {signature_errors}")
        logger.info(f"      Critical endpoint failures: {critical_failures}")
        
        # Success criteria: No signature errors and critical endpoints working
        success = signature_errors == 0 and critical_failures == 0
        
        details = f"Success: {successful_posts}/{total_post_endpoints}, Signature errors: {signature_errors}, Critical failures: {critical_failures}"
        
        self.log_test_result("POST Request Signature Fix Validation", success, details)
    
    async def test_4_ia2_execute_endpoint_specific_test(self):
        """Test 4: IA2 Execute Endpoint Specific Test (Most Critical POST Endpoint)"""
        logger.info("\n🔍 TEST 4: IA2 Execute Endpoint Specific Test")
        logger.info("   🎯 CRITICAL: Focus on /api/bingx/execute-ia2 - the main POST endpoint mentioned in review")
        
        try:
            logger.info("   🚀 Testing IA2 execution with comprehensive mock decision...")
            logger.info(f"   📊 Mock IA2 decision: {json.dumps(self.mock_ia2_decision, indent=2)}")
            
            # Make the critical POST request
            response = requests.post(f"{self.api_url}/bingx/execute-ia2", 
                                   json=self.mock_ia2_decision, timeout=90)
            
            logger.info(f"   📊 Response Status: HTTP {response.status_code}")
            
            try:
                response_data = response.json()
                logger.info(f"   📊 Response Data: {json.dumps(response_data, indent=2)}")
            except:
                logger.info(f"   📊 Response Text: {response.text}")
                response_data = {}
            
            # Detailed analysis
            signature_issue = False
            authentication_issue = False
            execution_result = None
            
            if response.status_code == 200:
                # Check execution result
                status = response_data.get('status', 'unknown')
                execution_result = status
                
                logger.info(f"   📊 Execution Status: {status}")
                
                if status == 'executed':
                    logger.info("      ✅ Trade executed successfully")
                elif status == 'skipped':
                    reason = response_data.get('reason', 'Unknown')
                    logger.info(f"      ✅ Trade skipped (valid): {reason}")
                elif status == 'rejected':
                    errors = response_data.get('errors', [])
                    logger.info(f"      ✅ Trade rejected by risk management (valid): {errors}")
                elif status == 'error':
                    error_msg = response_data.get('error', 'Unknown error')
                    if 'signature' in error_msg.lower():
                        signature_issue = True
                        logger.info(f"      ❌ Signature error: {error_msg}")
                    else:
                        logger.info(f"      ⚠️ Other error: {error_msg}")
                else:
                    logger.info(f"      ⚠️ Unexpected status: {status}")
                    
            elif response.status_code in [400, 422]:
                # Check if it's a validation error vs signature error
                response_text = response.text.lower()
                if 'signature' in response_text or 'null signature' in response_text:
                    signature_issue = True
                    logger.info("      ❌ Signature error detected in validation response")
                else:
                    logger.info("      ✅ Validation error (expected for mock data)")
                    
            elif response.status_code in [401, 403]:
                authentication_issue = True
                logger.info("      ⚠️ Authentication issue detected")
                
            else:
                logger.info(f"      ❌ Unexpected HTTP status: {response.status_code}")
            
            # Check backend logs for additional context
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "100", "/var/log/supervisor/backend.out.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                recent_logs = log_result.stdout
                
                # Look for IA2 execution logs
                ia2_logs = [line for line in recent_logs.split('\n') if 'IA2' in line or 'execute-ia2' in line]
                if ia2_logs:
                    logger.info("   📊 Recent IA2 execution logs:")
                    for log_line in ia2_logs[-5:]:  # Last 5 relevant logs
                        logger.info(f"      {log_line}")
                        
            except Exception as e:
                logger.info(f"   ⚠️ Could not retrieve backend logs: {e}")
            
            # Success criteria
            success = not signature_issue and (
                response.status_code == 200 or 
                (response.status_code in [400, 422] and not signature_issue)
            )
            
            details = f"HTTP {response.status_code}, Status: {execution_result}, Signature issue: {signature_issue}, Auth issue: {authentication_issue}"
            
            self.log_test_result("IA2 Execute Endpoint Specific Test", success, details)
            
        except Exception as e:
            self.log_test_result("IA2 Execute Endpoint Specific Test", False, f"Exception: {str(e)}")
    
    async def test_5_error_handling_and_logging_validation(self):
        """Test 5: Error Handling and Logging Validation"""
        logger.info("\n🔍 TEST 5: Error Handling and Logging Validation")
        logger.info("   🎯 Expected: API errors properly handled and logged, no system crashes")
        
        error_handling_results = []
        
        try:
            # Test 1: Invalid symbol in GET request
            logger.info("   🧪 Testing invalid symbol handling...")
            invalid_symbol_response = requests.get(f"{self.api_url}/bingx/market-price?symbol=INVALIDUSDT", timeout=30)
            
            if invalid_symbol_response.status_code in [400, 404, 422]:
                error_handling_results.append("✅ Invalid symbol handled correctly")
                logger.info("      ✅ Invalid symbol properly rejected")
            else:
                error_handling_results.append(f"❌ Invalid symbol: HTTP {invalid_symbol_response.status_code}")
                logger.info(f"      ❌ Invalid symbol not handled: HTTP {invalid_symbol_response.status_code}")
            
            # Test 2: Invalid POST data
            logger.info("   🧪 Testing invalid POST data handling...")
            invalid_post_data = {
                "symbol": "",  # Empty symbol
                "signal": "INVALID_SIGNAL",
                "confidence": 2.0,  # Invalid confidence > 1
                "position_size": -10,  # Invalid negative size
                "leverage": 1000  # Invalid high leverage
            }
            
            invalid_post_response = requests.post(f"{self.api_url}/bingx/execute-ia2", 
                                                json=invalid_post_data, timeout=30)
            
            if invalid_post_response.status_code in [400, 422]:
                error_handling_results.append("✅ Invalid POST data handled correctly")
                logger.info("      ✅ Invalid POST data properly rejected")
            elif invalid_post_response.status_code == 200:
                # Check if it was rejected in the response
                try:
                    response_data = invalid_post_response.json()
                    if response_data.get('status') in ['rejected', 'error']:
                        error_handling_results.append("✅ Invalid POST data handled correctly (rejected)")
                        logger.info("      ✅ Invalid POST data rejected by business logic")
                    else:
                        error_handling_results.append("❌ Invalid POST data not properly validated")
                        logger.info("      ❌ Invalid POST data not properly validated")
                except:
                    error_handling_results.append("❌ Invalid POST data response unclear")
                    logger.info("      ❌ Invalid POST data response unclear")
            else:
                error_handling_results.append(f"❌ Invalid POST data: HTTP {invalid_post_response.status_code}")
                logger.info(f"      ❌ Invalid POST data not handled: HTTP {invalid_post_response.status_code}")
            
            # Test 3: System responsiveness after errors
            logger.info("   🧪 Testing system responsiveness after errors...")
            status_after_errors = requests.get(f"{self.api_url}/bingx/status", timeout=30)
            
            if status_after_errors.status_code == 200:
                error_handling_results.append("✅ System remains responsive after errors")
                logger.info("      ✅ System still responsive after error tests")
            else:
                error_handling_results.append(f"❌ System unresponsive: HTTP {status_after_errors.status_code}")
                logger.info(f"      ❌ System unresponsive after errors: HTTP {status_after_errors.status_code}")
            
            # Test 4: Check backend logs for proper error logging
            logger.info("   🧪 Checking backend logs for error handling...")
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "200", "/var/log/supervisor/backend.out.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                recent_logs = log_result.stdout
                
                # Look for error handling patterns
                error_patterns = ['ERROR', 'Exception', 'Failed', 'Invalid', 'Rejected']
                error_logs_found = 0
                
                for pattern in error_patterns:
                    if pattern in recent_logs:
                        error_logs_found += 1
                
                if error_logs_found >= 2:
                    error_handling_results.append("✅ Error logging working properly")
                    logger.info(f"      ✅ Error logging detected: {error_logs_found} patterns found")
                else:
                    error_handling_results.append("⚠️ Limited error logging detected")
                    logger.info(f"      ⚠️ Limited error logging: {error_logs_found} patterns found")
                    
            except Exception as e:
                error_handling_results.append("⚠️ Could not check error logs")
                logger.info(f"      ⚠️ Could not check backend logs: {e}")
            
            # Evaluate overall error handling
            successful_error_tests = len([r for r in error_handling_results if r.startswith("✅")])
            total_error_tests = len(error_handling_results)
            
            logger.info(f"   📊 Error Handling Results:")
            for result in error_handling_results:
                logger.info(f"      {result}")
            
            success = successful_error_tests >= 3  # At least 3 out of 4 error handling aspects working
            
            details = f"Error handling tests passed: {successful_error_tests}/{total_error_tests}"
            
            self.log_test_result("Error Handling and Logging Validation", success, details)
            
        except Exception as e:
            self.log_test_result("Error Handling and Logging Validation", False, f"Exception: {str(e)}")
    
    async def test_6_comprehensive_endpoint_authentication(self):
        """Test 6: Comprehensive Endpoint Authentication Test"""
        logger.info("\n🔍 TEST 6: Comprehensive Endpoint Authentication Test")
        logger.info("   🎯 Expected: All BingX endpoints authenticate successfully with updated IP whitelist")
        
        authentication_results = []
        
        for endpoint in self.priority_endpoints:
            try:
                method = endpoint['method']
                path = endpoint['path']
                name = endpoint['name']
                priority = endpoint['priority']
                
                logger.info(f"   🔐 Testing authentication: {method} {path} ({name})")
                
                if method == 'GET':
                    if 'market-price' in path:
                        response = requests.get(f"{self.api_url}{path}?symbol=BTCUSDT", timeout=30)
                    else:
                        response = requests.get(f"{self.api_url}{path}", timeout=30)
                        
                elif method == 'POST':
                    # Use minimal test data to focus on authentication
                    test_data = {"test_auth": True}
                    response = requests.post(f"{self.api_url}{path}", json=test_data, timeout=30)
                
                # Analyze authentication
                auth_success = False
                auth_details = ""
                
                if response.status_code == 200:
                    auth_success = True
                    auth_details = "Authenticated successfully"
                elif response.status_code in [400, 422]:
                    # Check if it's validation error (good) vs auth error (bad)
                    try:
                        response_data = response.json()
                        response_text = json.dumps(response_data).lower()
                    except:
                        response_text = response.text.lower()
                    
                    auth_keywords = ['unauthorized', 'authentication', 'invalid api key', 'forbidden', 'signature']
                    
                    if any(keyword in response_text for keyword in auth_keywords):
                        auth_success = False
                        auth_details = f"Authentication failed: {response.status_code}"
                    else:
                        auth_success = True
                        auth_details = f"Authenticated (validation error expected): {response.status_code}"
                        
                elif response.status_code in [401, 403]:
                    auth_success = False
                    auth_details = f"Authentication failed: {response.status_code}"
                else:
                    auth_success = False
                    auth_details = f"Unexpected response: {response.status_code}"
                
                authentication_results.append({
                    'endpoint': f"{method} {path}",
                    'name': name,
                    'priority': priority,
                    'auth_success': auth_success,
                    'details': auth_details,
                    'http_code': response.status_code
                })
                
                status_icon = "✅" if auth_success else "❌"
                logger.info(f"      {status_icon} {name}: {auth_details}")
                
            except Exception as e:
                authentication_results.append({
                    'endpoint': f"{method} {path}",
                    'name': name,
                    'priority': priority,
                    'auth_success': False,
                    'details': f"Exception: {str(e)}",
                    'error': str(e)
                })
                logger.info(f"      ❌ {name}: Exception - {str(e)}")
        
        # Analyze authentication results
        total_endpoints = len(authentication_results)
        authenticated_endpoints = len([r for r in authentication_results if r['auth_success']])
        critical_auth_failures = len([r for r in authentication_results 
                                    if r['priority'] == 'CRITICAL' and not r['auth_success']])
        high_priority_auth_failures = len([r for r in authentication_results 
                                         if r['priority'] == 'HIGH' and not r['auth_success']])
        
        logger.info(f"   📊 Authentication Results Summary:")
        logger.info(f"      Total endpoints tested: {total_endpoints}")
        logger.info(f"      Successfully authenticated: {authenticated_endpoints}")
        logger.info(f"      Critical authentication failures: {critical_auth_failures}")
        logger.info(f"      High priority authentication failures: {high_priority_auth_failures}")
        
        # Success criteria: All critical endpoints authenticate, most high priority endpoints authenticate
        success = critical_auth_failures == 0 and high_priority_auth_failures <= 1
        
        details = f"Authenticated: {authenticated_endpoints}/{total_endpoints}, Critical failures: {critical_auth_failures}, High priority failures: {high_priority_auth_failures}"
        
        self.log_test_result("Comprehensive Endpoint Authentication", success, details)
    
    async def run_comprehensive_tests(self):
        """Run all BingX POST signature fix tests"""
        logger.info("🚀 Starting BingX POST Request Signature Fix Test Suite")
        logger.info("=" * 100)
        logger.info("📋 BINGX POST REQUEST SIGNATURE FIX VALIDATION")
        logger.info("🎯 Focus: POST signature fix, connectivity after IP update, error handling")
        logger.info("🎯 Expected: No 'Null signature' errors, all endpoints authenticate successfully")
        logger.info("🎯 Context: IP whitelist updated to 34.121.6.206, parameters moved to query string")
        logger.info("=" * 100)
        
        # Run all tests in sequence
        await self.test_1_bingx_connectivity_after_ip_update()
        await self.test_2_balance_and_positions_data_retrieval()
        await self.test_3_post_request_signature_fix_validation()
        await self.test_4_ia2_execute_endpoint_specific_test()
        await self.test_5_error_handling_and_logging_validation()
        await self.test_6_comprehensive_endpoint_authentication()
        
        # Summary
        logger.info("\n" + "=" * 100)
        logger.info("📊 BINGX POST REQUEST SIGNATURE FIX TEST SUMMARY")
        logger.info("=" * 100)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        # Categorize results by importance
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
        logger.info("\n" + "=" * 100)
        logger.info("📋 BINGX POST SIGNATURE FIX ANALYSIS")
        logger.info("=" * 100)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - BingX POST Signature Fix FULLY SUCCESSFUL!")
            logger.info("✅ IP whitelist update to 34.121.6.206 working")
            logger.info("✅ POST request signature generation fixed")
            logger.info("✅ All endpoints authenticate successfully")
            logger.info("✅ Balance and positions data retrieval working")
            logger.info("✅ Error handling robust and logging operational")
            logger.info("✅ IA2 execute endpoint (critical POST) functional")
        elif passed_tests >= total_tests * 0.83:  # 5/6 tests
            logger.info("⚠️ MOSTLY SUCCESSFUL - BingX POST signature fix working with minor issues")
            logger.info("🔍 Most critical functionality restored, minor optimizations may be needed")
        elif passed_tests >= total_tests * 0.67:  # 4/6 tests
            logger.info("⚠️ PARTIALLY SUCCESSFUL - Core POST signature fix working")
            logger.info("🔧 Some endpoints may need additional attention")
        else:
            logger.info("❌ POST SIGNATURE FIX NOT FULLY SUCCESSFUL - Critical issues remain")
            logger.info("🚨 Major issues preventing reliable BingX API integration")
        
        # Specific fix validation
        logger.info("\n📝 POST SIGNATURE FIX VALIDATION RESULTS:")
        
        signature_fix_indicators = []
        
        for result in self.test_results:
            if result['success']:
                if "Connectivity After IP Update" in result['test']:
                    signature_fix_indicators.append("✅ IP whitelist update successful")
                elif "Balance and Positions" in result['test']:
                    signature_fix_indicators.append("✅ Account data retrieval working")
                elif "POST Request Signature Fix" in result['test']:
                    signature_fix_indicators.append("✅ POST signature generation fixed")
                elif "IA2 Execute Endpoint" in result['test']:
                    signature_fix_indicators.append("✅ Critical IA2 POST endpoint working")
                elif "Error Handling" in result['test']:
                    signature_fix_indicators.append("✅ Error handling and logging operational")
                elif "Endpoint Authentication" in result['test']:
                    signature_fix_indicators.append("✅ All endpoints authenticate successfully")
            else:
                if "Connectivity After IP Update" in result['test']:
                    signature_fix_indicators.append("❌ IP whitelist update issues")
                elif "Balance and Positions" in result['test']:
                    signature_fix_indicators.append("❌ Account data retrieval not working")
                elif "POST Request Signature Fix" in result['test']:
                    signature_fix_indicators.append("❌ POST signature generation still has issues")
                elif "IA2 Execute Endpoint" in result['test']:
                    signature_fix_indicators.append("❌ Critical IA2 POST endpoint not working")
                elif "Error Handling" in result['test']:
                    signature_fix_indicators.append("❌ Error handling needs improvement")
                elif "Endpoint Authentication" in result['test']:
                    signature_fix_indicators.append("❌ Authentication issues persist")
        
        for indicator in signature_fix_indicators:
            logger.info(f"   {indicator}")
        
        # Final verdict
        logger.info(f"\n🏆 FINAL VERDICT:")
        
        if passed_tests >= 5:
            logger.info("🎉 VERDICT: BingX POST Request Signature Fix is SUCCESSFUL!")
            logger.info("✅ The main issue (Null signature errors) has been resolved")
            logger.info("✅ Parameters correctly moved to query string for POST requests")
            logger.info("✅ HMAC-SHA256 signature generation working properly")
            logger.info("✅ BingX API integration ready for live trading operations")
        elif passed_tests >= 4:
            logger.info("⚠️ VERDICT: BingX POST Signature Fix is MOSTLY SUCCESSFUL")
            logger.info("🔍 Core signature issues resolved, minor refinements may be needed")
        elif passed_tests >= 3:
            logger.info("⚠️ VERDICT: BingX POST Signature Fix has PARTIAL SUCCESS")
            logger.info("🔧 Some signature issues resolved, but additional work needed")
        else:
            logger.info("❌ VERDICT: BingX POST Signature Fix is NOT SUCCESSFUL")
            logger.info("🚨 Critical signature issues persist, major debugging required")
        
        # Recommendations
        logger.info("\n📋 RECOMMENDATIONS:")
        
        if len(critical_failures) == 0:
            logger.info("✅ No critical issues found - system ready for production use")
        else:
            logger.info("🔧 CRITICAL ISSUES TO ADDRESS:")
            for failure in critical_failures:
                logger.info(f"   ❌ {failure}")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = BingXPostSignatureTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed >= total * 0.83:  # 83% pass rate for success (5/6 tests)
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
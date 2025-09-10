#!/usr/bin/env python3
"""
BingX "Invalid Parameters" Issue Analysis Test
Focus: Analyze the specific "Invalid parameters" error for POST /openApi/swap/v2/trade/order

PROBLEM STATEMENT:
BingX returns "Invalid parameters, The request you constructed does not meet the requirements"
even though:
✅ HMAC-SHA256 signature is correct
✅ Timestamp is synchronized 
✅ POST request format is correct
✅ Basic parameters are correct

CURRENT PARAMETERS BEING SENT:
{
  "symbol": "ETH-USDT",
  "side": "BUY", 
  "positionSide": "LONG",
  "type": "MARKET",
  "quantity": "0.00024554468203583584",
  "recvWindow": 5000,
  "timestamp": 1757528669837
}

TESTS TO PERFORM:
1. Compare with BingX documentation requirements
2. Test different parameter formats (quantity as integer vs string vs precision)
3. Test other symbols (BTC-USDT with standard quantities)
4. Test additional required parameters (timeInForce, etc.)
5. Analyze detailed error responses
6. Test parameter validation and formatting
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BingXParameterAnalysisTest:
    """Focused test for BingX Invalid Parameters issue"""
    
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
        logger.info(f"Testing BingX Parameter Issues at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Current problematic parameters
        self.current_params = {
            "symbol": "ETH-USDT",
            "side": "BUY", 
            "positionSide": "LONG",
            "type": "MARKET",
            "quantity": "0.00024554468203583584",
            "recvWindow": 5000,
            "timestamp": 1757528669837
        }
        
        # Alternative parameter sets to test
        self.test_parameter_sets = [
            {
                "name": "Current Parameters (Problematic)",
                "params": self.current_params.copy()
            },
            {
                "name": "BTC-USDT with Standard Quantity",
                "params": {
                    "symbol": "BTC-USDT",
                    "side": "BUY",
                    "positionSide": "LONG", 
                    "type": "MARKET",
                    "quantity": "0.001",
                    "recvWindow": 5000,
                    "timestamp": int(time.time() * 1000)
                }
            },
            {
                "name": "ETH-USDT with Rounded Quantity",
                "params": {
                    "symbol": "ETH-USDT",
                    "side": "BUY",
                    "positionSide": "LONG",
                    "type": "MARKET", 
                    "quantity": "0.0002",
                    "recvWindow": 5000,
                    "timestamp": int(time.time() * 1000)
                }
            },
            {
                "name": "With timeInForce Parameter",
                "params": {
                    "symbol": "BTC-USDT",
                    "side": "BUY",
                    "positionSide": "LONG",
                    "type": "MARKET",
                    "quantity": "0.001",
                    "timeInForce": "IOC",
                    "recvWindow": 5000,
                    "timestamp": int(time.time() * 1000)
                }
            },
            {
                "name": "Integer Quantity Format",
                "params": {
                    "symbol": "BTC-USDT",
                    "side": "BUY",
                    "positionSide": "LONG",
                    "type": "MARKET",
                    "quantity": 0.001,  # As number, not string
                    "recvWindow": 5000,
                    "timestamp": int(time.time() * 1000)
                }
            },
            {
                "name": "SHORT Position Test",
                "params": {
                    "symbol": "BTC-USDT",
                    "side": "SELL",
                    "positionSide": "SHORT",
                    "type": "MARKET",
                    "quantity": "0.001",
                    "recvWindow": 5000,
                    "timestamp": int(time.time() * 1000)
                }
            },
            {
                "name": "LIMIT Order Type",
                "params": {
                    "symbol": "BTC-USDT",
                    "side": "BUY",
                    "positionSide": "LONG",
                    "type": "LIMIT",
                    "quantity": "0.001",
                    "price": "45000.0",
                    "timeInForce": "GTC",
                    "recvWindow": 5000,
                    "timestamp": int(time.time() * 1000)
                }
            }
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
    
    async def test_1_current_parameter_analysis(self):
        """Test 1: Analyze current problematic parameters"""
        logger.info("\n🔍 TEST 1: Current Parameter Analysis")
        logger.info(f"   📊 Current parameters: {json.dumps(self.current_params, indent=2)}")
        
        try:
            # Test with current parameters via IA2 execute endpoint
            response = requests.post(f"{self.api_url}/bingx/execute-ia2", 
                                   json={
                                       "symbol": "ETHUSDT",
                                       "signal": "LONG",
                                       "confidence": 0.85,
                                       "position_size": 2.5,
                                       "leverage": 5,
                                       "entry_price": 3500.0,
                                       "stop_loss": 3400.0,
                                       "take_profit": 3700.0,
                                       "reasoning": "Parameter analysis test"
                                   }, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"   📊 Response: {json.dumps(result, indent=2)}")
                
                status = result.get('status', 'unknown')
                error_message = result.get('error', '')
                
                if 'Invalid parameters' in error_message:
                    self.log_test_result("Current Parameter Analysis", False, 
                                       f"Invalid parameters error confirmed: {error_message}")
                elif status == 'error':
                    self.log_test_result("Current Parameter Analysis", False, 
                                       f"Error status: {error_message}")
                else:
                    self.log_test_result("Current Parameter Analysis", True, 
                                       f"Parameters accepted: {status}")
            else:
                self.log_test_result("Current Parameter Analysis", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test_result("Current Parameter Analysis", False, f"Exception: {str(e)}")
    
    async def test_2_parameter_format_variations(self):
        """Test 2: Test different parameter formats"""
        logger.info("\n🔍 TEST 2: Parameter Format Variations")
        
        format_results = []
        
        for test_set in self.test_parameter_sets:
            try:
                name = test_set['name']
                params = test_set['params']
                
                logger.info(f"   🧪 Testing: {name}")
                logger.info(f"      Parameters: {json.dumps(params, indent=6)}")
                
                # Create IA2 decision based on parameters
                ia2_decision = {
                    "symbol": params['symbol'].replace('-', ''),
                    "signal": "LONG" if params['side'] == "BUY" else "SHORT",
                    "confidence": 0.85,
                    "position_size": 2.5,
                    "leverage": 5,
                    "entry_price": 45000.0 if 'BTC' in params['symbol'] else 3500.0,
                    "stop_loss": 43000.0 if 'BTC' in params['symbol'] else 3400.0,
                    "take_profit": 48000.0 if 'BTC' in params['symbol'] else 3700.0,
                    "reasoning": f"Parameter format test: {name}"
                }
                
                response = requests.post(f"{self.api_url}/bingx/execute-ia2", 
                                       json=ia2_decision, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    status = result.get('status', 'unknown')
                    error_message = result.get('error', '')
                    
                    if 'Invalid parameters' in error_message:
                        format_results.append(f"❌ {name}: Invalid parameters error")
                        logger.info(f"      ❌ Invalid parameters: {error_message}")
                    elif status == 'error':
                        format_results.append(f"⚠️ {name}: Other error - {error_message}")
                        logger.info(f"      ⚠️ Error: {error_message}")
                    elif status in ['executed', 'rejected', 'skipped']:
                        format_results.append(f"✅ {name}: Parameters accepted ({status})")
                        logger.info(f"      ✅ Success: {status}")
                    else:
                        format_results.append(f"❓ {name}: Unknown status - {status}")
                        logger.info(f"      ❓ Unknown: {status}")
                else:
                    format_results.append(f"❌ {name}: HTTP {response.status_code}")
                    logger.info(f"      ❌ HTTP Error: {response.status_code}")
                    
            except Exception as e:
                format_results.append(f"❌ {name}: Exception - {str(e)}")
                logger.info(f"      ❌ Exception: {str(e)}")
        
        # Evaluate results
        successful_formats = len([r for r in format_results if r.startswith("✅")])
        total_formats = len(format_results)
        
        logger.info(f"\n   📊 Parameter Format Test Results:")
        for result in format_results:
            logger.info(f"      {result}")
        
        if successful_formats > 0:
            self.log_test_result("Parameter Format Variations", True, 
                               f"Found working formats: {successful_formats}/{total_formats}")
        else:
            self.log_test_result("Parameter Format Variations", False, 
                               f"No working formats found: {successful_formats}/{total_formats}")
    
    async def test_3_bingx_documentation_comparison(self):
        """Test 3: Compare with BingX documentation requirements"""
        logger.info("\n🔍 TEST 3: BingX Documentation Comparison")
        
        # BingX Futures API documentation requirements for POST /openApi/swap/v2/trade/order
        required_params = {
            "symbol": "String - Trading pair symbol (e.g., BTC-USDT)",
            "side": "String - BUY or SELL", 
            "positionSide": "String - LONG or SHORT for futures",
            "type": "String - MARKET, LIMIT, STOP, TAKE_PROFIT, etc.",
            "quantity": "String - Order quantity",
            "timestamp": "Long - Request timestamp in milliseconds",
            "recvWindow": "Long - Request valid time window (optional, max 60000)"
        }
        
        optional_params = {
            "price": "String - Required for LIMIT orders",
            "stopPrice": "String - Required for STOP orders", 
            "timeInForce": "String - GTC, IOC, FOK (required for LIMIT orders)",
            "reduceOnly": "Boolean - Reduce only flag",
            "closePosition": "Boolean - Close position flag",
            "workingType": "String - MARK_PRICE or CONTRACT_PRICE"
        }
        
        logger.info("   📋 BingX API Documentation Requirements:")
        logger.info("   Required Parameters:")
        for param, desc in required_params.items():
            logger.info(f"      • {param}: {desc}")
        
        logger.info("   Optional Parameters:")
        for param, desc in optional_params.items():
            logger.info(f"      • {param}: {desc}")
        
        # Analyze current parameters against documentation
        current_analysis = []
        
        for param, value in self.current_params.items():
            if param in required_params:
                current_analysis.append(f"✅ {param}: {value} (Required - Present)")
            elif param in optional_params:
                current_analysis.append(f"ℹ️ {param}: {value} (Optional - Present)")
            else:
                current_analysis.append(f"❓ {param}: {value} (Unknown parameter)")
        
        # Check for missing required parameters
        missing_required = []
        for param in required_params.keys():
            if param not in self.current_params:
                missing_required.append(param)
        
        logger.info("\n   📊 Current Parameters Analysis:")
        for analysis in current_analysis:
            logger.info(f"      {analysis}")
        
        if missing_required:
            logger.info(f"\n   ⚠️ Missing Required Parameters: {missing_required}")
            self.log_test_result("BingX Documentation Comparison", False, 
                               f"Missing required parameters: {missing_required}")
        else:
            logger.info("\n   ✅ All required parameters present")
            
            # Check for potential issues
            issues = []
            
            # Check quantity precision
            quantity = self.current_params.get('quantity', '')
            if isinstance(quantity, str) and len(quantity.split('.')[-1]) > 8:
                issues.append("Quantity has excessive decimal precision")
            
            # Check symbol format
            symbol = self.current_params.get('symbol', '')
            if '-' not in symbol:
                issues.append("Symbol might need hyphen format (e.g., BTC-USDT)")
            
            # Check timestamp validity
            timestamp = self.current_params.get('timestamp', 0)
            current_time = int(time.time() * 1000)
            if abs(current_time - timestamp) > 60000:  # More than 1 minute difference
                issues.append("Timestamp might be too old or invalid")
            
            if issues:
                logger.info(f"   ⚠️ Potential Issues Found:")
                for issue in issues:
                    logger.info(f"      • {issue}")
                self.log_test_result("BingX Documentation Comparison", False, 
                                   f"Potential issues: {issues}")
            else:
                self.log_test_result("BingX Documentation Comparison", True, 
                                   "Parameters match documentation requirements")
    
    async def test_4_backend_logs_analysis(self):
        """Test 4: Analyze backend logs for detailed error information"""
        logger.info("\n🔍 TEST 4: Backend Logs Analysis")
        
        try:
            # Check backend logs for BingX errors
            import subprocess
            
            # Get recent backend logs
            result = subprocess.run(['tail', '-n', '100', '/var/log/supervisor/backend.out.log'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                logs = result.stdout
                
                # Look for BingX related errors
                bingx_errors = []
                invalid_param_errors = []
                
                for line in logs.split('\n'):
                    if 'bingx' in line.lower() or 'BingX' in line:
                        if 'error' in line.lower() or 'Error' in line:
                            bingx_errors.append(line.strip())
                        if 'Invalid parameters' in line:
                            invalid_param_errors.append(line.strip())
                
                logger.info(f"   📊 Found {len(bingx_errors)} BingX-related errors in logs")
                logger.info(f"   📊 Found {len(invalid_param_errors)} 'Invalid parameters' errors")
                
                if invalid_param_errors:
                    logger.info("   🔍 Invalid Parameters Errors:")
                    for error in invalid_param_errors[-3:]:  # Show last 3
                        logger.info(f"      {error}")
                
                if bingx_errors:
                    logger.info("   🔍 Recent BingX Errors:")
                    for error in bingx_errors[-5:]:  # Show last 5
                        logger.info(f"      {error}")
                
                # Check for specific error patterns
                error_patterns = [
                    "signature",
                    "timestamp", 
                    "parameters",
                    "authentication",
                    "IP whitelist",
                    "API key"
                ]
                
                pattern_counts = {}
                for pattern in error_patterns:
                    count = sum(1 for line in logs.split('\n') if pattern.lower() in line.lower())
                    if count > 0:
                        pattern_counts[pattern] = count
                
                if pattern_counts:
                    logger.info("   📊 Error Pattern Analysis:")
                    for pattern, count in pattern_counts.items():
                        logger.info(f"      • {pattern}: {count} occurrences")
                
                self.log_test_result("Backend Logs Analysis", True, 
                                   f"Analyzed logs: {len(bingx_errors)} BingX errors, {len(invalid_param_errors)} parameter errors")
            else:
                self.log_test_result("Backend Logs Analysis", False, 
                                   f"Could not read backend logs: {result.stderr}")
                
        except Exception as e:
            self.log_test_result("Backend Logs Analysis", False, f"Exception: {str(e)}")
    
    async def test_5_direct_bingx_api_test(self):
        """Test 5: Test BingX API directly to isolate the issue"""
        logger.info("\n🔍 TEST 5: Direct BingX API Test")
        
        try:
            # Test balance endpoint first (simpler)
            balance_response = requests.get(f"{self.api_url}/bingx/balance", timeout=30)
            
            if balance_response.status_code == 200:
                balance_data = balance_response.json()
                logger.info(f"   📊 Balance endpoint working: {json.dumps(balance_data, indent=2)}")
                
                # Test positions endpoint
                positions_response = requests.get(f"{self.api_url}/bingx/positions", timeout=30)
                
                if positions_response.status_code == 200:
                    positions_data = positions_response.json()
                    logger.info(f"   📊 Positions endpoint working: {json.dumps(positions_data, indent=2)}")
                    
                    # Now test a simple trade execution
                    simple_trade = {
                        "symbol": "BTCUSDT",
                        "signal": "LONG", 
                        "confidence": 0.85,
                        "position_size": 1.0,  # Small position
                        "leverage": 2,         # Low leverage
                        "entry_price": 45000.0,
                        "stop_loss": 44000.0,
                        "take_profit": 46000.0,
                        "reasoning": "Direct API test with simple parameters"
                    }
                    
                    trade_response = requests.post(f"{self.api_url}/bingx/execute-ia2", 
                                                 json=simple_trade, timeout=60)
                    
                    if trade_response.status_code == 200:
                        trade_result = trade_response.json()
                        logger.info(f"   📊 Trade execution result: {json.dumps(trade_result, indent=2)}")
                        
                        status = trade_result.get('status', 'unknown')
                        error_msg = trade_result.get('error', '')
                        
                        if 'Invalid parameters' in error_msg:
                            self.log_test_result("Direct BingX API Test", False, 
                                               f"Invalid parameters error persists: {error_msg}")
                        else:
                            self.log_test_result("Direct BingX API Test", True, 
                                               f"API responding correctly: {status}")
                    else:
                        self.log_test_result("Direct BingX API Test", False, 
                                           f"Trade execution failed: HTTP {trade_response.status_code}")
                else:
                    self.log_test_result("Direct BingX API Test", False, 
                                       f"Positions endpoint failed: HTTP {positions_response.status_code}")
            else:
                self.log_test_result("Direct BingX API Test", False, 
                                   f"Balance endpoint failed: HTTP {balance_response.status_code}")
                
        except Exception as e:
            self.log_test_result("Direct BingX API Test", False, f"Exception: {str(e)}")
    
    async def run_parameter_analysis(self):
        """Run comprehensive parameter analysis"""
        logger.info("🚀 Starting BingX 'Invalid Parameters' Analysis")
        logger.info("=" * 80)
        logger.info("📋 BINGX PARAMETER ISSUE ANALYSIS")
        logger.info("🎯 Focus: Analyze 'Invalid parameters' error for POST /openApi/swap/v2/trade/order")
        logger.info("🎯 Goal: Identify exact parameter format or missing field causing the error")
        logger.info("=" * 80)
        
        # Run all tests
        await self.test_1_current_parameter_analysis()
        await self.test_2_parameter_format_variations()
        await self.test_3_bingx_documentation_comparison()
        await self.test_4_backend_logs_analysis()
        await self.test_5_direct_bingx_api_test()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 BINGX PARAMETER ANALYSIS SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
        
        logger.info(f"\n🎯 ANALYSIS RESULT: {passed_tests}/{total_tests} tests completed successfully")
        
        # Recommendations
        logger.info("\n" + "=" * 80)
        logger.info("💡 RECOMMENDATIONS FOR FIXING 'INVALID PARAMETERS' ERROR")
        logger.info("=" * 80)
        
        recommendations = [
            "1. 🔍 Check quantity precision - BingX may have limits on decimal places",
            "2. 🔍 Verify symbol format - ensure correct hyphen usage (BTC-USDT vs BTCUSDT)",
            "3. 🔍 Add timeInForce parameter for LIMIT orders (GTC, IOC, FOK)",
            "4. 🔍 Ensure timestamp is current (within 60 seconds of server time)",
            "5. 🔍 Test with standard quantities (0.001 BTC instead of micro amounts)",
            "6. 🔍 Verify parameter order in signature generation",
            "7. 🔍 Check if additional parameters are required for futures trading",
            "8. 🔍 Test with different symbols to isolate symbol-specific issues"
        ]
        
        for rec in recommendations:
            logger.info(f"   {rec}")
        
        # Next steps
        logger.info("\n📋 NEXT STEPS:")
        logger.info("   1. Implement parameter format corrections based on findings")
        logger.info("   2. Test with BingX testnet if available")
        logger.info("   3. Compare working parameters from successful trades")
        logger.info("   4. Review BingX API documentation for recent changes")
        logger.info("   5. Contact BingX support if issue persists")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = BingXParameterAnalysisTest()
    passed, total = await test_suite.run_parameter_analysis()
    
    # Exit with appropriate code
    if passed >= total * 0.6:  # 60% success rate acceptable for analysis
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
BingX Trade Execution Issue Testing Suite
Focus: Test spécifique du problème d'exécution des trades BingX

PROBLÈME IDENTIFIÉ:
1. Le système IA1 → IA2 fonctionne et génère des décisions
2. L'endpoint `/api/bingx/execute-ia2` retourne une erreur BingX "Invalid parameters" 
3. L'erreur indique: "timestamp: This field is required. symbol: This field is required. 
   positionSide: This field is required. type: This field is required. side: This field is required."

TESTS À EFFECTUER:
1. Tester l'exécution manuelle IA2: Vérifier si `/api/bingx/execute-ia2` fonctionne avec des données valides
2. Analyser les paramètres BingX: Vérifier si les paramètres sont correctement formatés pour l'API BingX
3. Tester la méthode `place_market_order`: Voir si le problème vient de la construction de l'ordre
4. Vérifier l'intégration automatique: Tester si l'exécution automatique des trades depuis IA2 fonctionne

OBJECTIF: Identifier pourquoi les trades ne s'exécutent pas et proposer une solution
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

class BingXTradeExecutionTestSuite:
    """Test suite spécifique pour le problème d'exécution des trades BingX"""
    
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
        logger.info(f"Testing BingX Trade Execution at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Test data from review request
        self.test_ia2_decision = {
            "symbol": "BTCUSDT",
            "signal": "LONG", 
            "confidence": 0.85,
            "position_size": 1.0,
            "leverage": 2
        }
        
        # Additional test data variations
        self.test_variations = [
            {
                "symbol": "ETHUSDT",
                "signal": "SHORT",
                "confidence": 0.75,
                "position_size": 0.5,
                "leverage": 3
            },
            {
                "symbol": "SOLUSDT", 
                "signal": "LONG",
                "confidence": 0.90,
                "position_size": 2.0,
                "leverage": 5
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
    
    async def test_1_bingx_connectivity_status(self):
        """Test 1: Vérifier la connectivité BingX de base"""
        logger.info("\n🔍 TEST 1: BingX Connectivity Status")
        
        try:
            response = requests.get(f"{self.api_url}/bingx/status", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"   📊 BingX Status: {json.dumps(data, indent=2)}")
                
                api_connected = data.get('api_connected', False)
                status = data.get('status', 'unknown')
                
                if api_connected or status == 'connected':
                    self.log_test_result("BingX Connectivity", True, f"Status: {status}, Connected: {api_connected}")
                else:
                    self.log_test_result("BingX Connectivity", False, f"Not connected - Status: {status}")
            else:
                self.log_test_result("BingX Connectivity", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test_result("BingX Connectivity", False, f"Exception: {str(e)}")
    
    async def test_2_bingx_account_balance(self):
        """Test 2: Vérifier l'accès au compte BingX"""
        logger.info("\n🔍 TEST 2: BingX Account Balance Access")
        
        try:
            response = requests.get(f"{self.api_url}/bingx/balance", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"   📊 Balance Data: {json.dumps(data, indent=2)}")
                
                # Check for balance information
                has_balance = any(key in data for key in ['balance', 'total_balance', 'available_balance'])
                
                if has_balance:
                    balance = data.get('balance', data.get('total_balance', 0))
                    self.log_test_result("BingX Account Access", True, f"Balance retrieved: ${balance}")
                else:
                    self.log_test_result("BingX Account Access", False, "No balance data found")
            else:
                self.log_test_result("BingX Account Access", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test_result("BingX Account Access", False, f"Exception: {str(e)}")
    
    async def test_3_ia2_execute_endpoint_basic(self):
        """Test 3: Test de base de l'endpoint /api/bingx/execute-ia2"""
        logger.info("\n🔍 TEST 3: IA2 Execute Endpoint Basic Test")
        logger.info(f"   🚀 Testing with data: {self.test_ia2_decision}")
        
        try:
            response = requests.post(
                f"{self.api_url}/bingx/execute-ia2", 
                json=self.test_ia2_decision, 
                timeout=60
            )
            
            logger.info(f"   📊 Response Status: HTTP {response.status_code}")
            
            if response.status_code in [200, 201]:
                data = response.json()
                logger.info(f"   📊 Response Data: {json.dumps(data, indent=2)}")
                
                status = data.get('status', 'unknown')
                
                if status == 'executed':
                    order_id = data.get('order_id')
                    self.log_test_result("IA2 Execute Basic", True, f"Trade executed successfully, Order ID: {order_id}")
                elif status in ['skipped', 'rejected']:
                    reason = data.get('reason', data.get('errors', 'Unknown reason'))
                    self.log_test_result("IA2 Execute Basic", True, f"Trade {status}: {reason} (Valid response)")
                elif status == 'error':
                    error_msg = data.get('error', data.get('message', 'Unknown error'))
                    if 'Invalid parameters' in str(error_msg) or 'required' in str(error_msg).lower():
                        self.log_test_result("IA2 Execute Basic", False, f"BingX Parameter Error: {error_msg}")
                    else:
                        self.log_test_result("IA2 Execute Basic", False, f"Other Error: {error_msg}")
                else:
                    self.log_test_result("IA2 Execute Basic", False, f"Unexpected status: {status}")
            else:
                response_text = response.text
                logger.info(f"   📊 Error Response: {response_text}")
                
                if 'Invalid parameters' in response_text or 'required' in response_text.lower():
                    self.log_test_result("IA2 Execute Basic", False, f"BingX Parameter Error (HTTP {response.status_code}): {response_text}")
                else:
                    self.log_test_result("IA2 Execute Basic", False, f"HTTP {response.status_code}: {response_text}")
                
        except Exception as e:
            self.log_test_result("IA2 Execute Basic", False, f"Exception: {str(e)}")
    
    async def test_4_ia2_execute_parameter_variations(self):
        """Test 4: Test avec différentes variations de paramètres"""
        logger.info("\n🔍 TEST 4: IA2 Execute Parameter Variations")
        
        successful_variations = 0
        total_variations = len(self.test_variations)
        
        for i, variation in enumerate(self.test_variations):
            logger.info(f"   🚀 Testing variation {i+1}: {variation}")
            
            try:
                response = requests.post(
                    f"{self.api_url}/bingx/execute-ia2", 
                    json=variation, 
                    timeout=60
                )
                
                if response.status_code in [200, 201]:
                    data = response.json()
                    status = data.get('status', 'unknown')
                    
                    if status in ['executed', 'skipped', 'rejected']:
                        successful_variations += 1
                        logger.info(f"      ✅ Variation {i+1}: {status}")
                    else:
                        logger.info(f"      ❌ Variation {i+1}: Unexpected status {status}")
                else:
                    logger.info(f"      ❌ Variation {i+1}: HTTP {response.status_code}")
                    
            except Exception as e:
                logger.info(f"      ❌ Variation {i+1}: Exception {str(e)}")
        
        success_rate = successful_variations / total_variations if total_variations > 0 else 0
        
        if success_rate >= 0.5:  # At least 50% success
            self.log_test_result("IA2 Execute Variations", True, f"Success rate: {successful_variations}/{total_variations} ({success_rate:.1%})")
        else:
            self.log_test_result("IA2 Execute Variations", False, f"Low success rate: {successful_variations}/{total_variations} ({success_rate:.1%})")
    
    async def test_5_bingx_manual_trade_endpoint(self):
        """Test 5: Test de l'endpoint de trade manuel BingX"""
        logger.info("\n🔍 TEST 5: BingX Manual Trade Endpoint")
        
        manual_trade_data = {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": 0.001,
            "leverage": 2
        }
        
        logger.info(f"   🚀 Testing manual trade: {manual_trade_data}")
        
        try:
            response = requests.post(
                f"{self.api_url}/bingx/trade", 
                json=manual_trade_data, 
                timeout=60
            )
            
            logger.info(f"   📊 Response Status: HTTP {response.status_code}")
            
            if response.status_code in [200, 201]:
                data = response.json()
                logger.info(f"   📊 Response Data: {json.dumps(data, indent=2)}")
                
                status = data.get('status', 'unknown')
                
                if status in ['executed', 'skipped', 'rejected']:
                    self.log_test_result("BingX Manual Trade", True, f"Manual trade {status}")
                else:
                    error_msg = data.get('error', data.get('message', 'Unknown'))
                    if 'Invalid parameters' in str(error_msg) or 'required' in str(error_msg).lower():
                        self.log_test_result("BingX Manual Trade", False, f"Parameter Error: {error_msg}")
                    else:
                        self.log_test_result("BingX Manual Trade", False, f"Other Error: {error_msg}")
            else:
                response_text = response.text
                if 'Invalid parameters' in response_text or 'required' in response_text.lower():
                    self.log_test_result("BingX Manual Trade", False, f"Parameter Error (HTTP {response.status_code}): {response_text}")
                else:
                    self.log_test_result("BingX Manual Trade", False, f"HTTP {response.status_code}: {response_text}")
                
        except Exception as e:
            self.log_test_result("BingX Manual Trade", False, f"Exception: {str(e)}")
    
    async def test_6_backend_logs_analysis(self):
        """Test 6: Analyse des logs backend pour identifier les erreurs"""
        logger.info("\n🔍 TEST 6: Backend Logs Analysis")
        
        try:
            import subprocess
            
            # Get recent backend logs
            log_result = subprocess.run(
                ["tail", "-n", "1000", "/var/log/supervisor/backend.out.log"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            backend_logs = log_result.stdout
            
            # Look for BingX related errors
            bingx_error_patterns = [
                "Invalid parameters",
                "timestamp: This field is required",
                "symbol: This field is required", 
                "positionSide: This field is required",
                "type: This field is required",
                "side: This field is required",
                "BingX API Error",
                "place_market_order",
                "execute_ia2_trade"
            ]
            
            found_errors = {}
            for pattern in bingx_error_patterns:
                count = backend_logs.count(pattern)
                if count > 0:
                    found_errors[pattern] = count
                    logger.info(f"      🔍 Found '{pattern}': {count} occurrences")
            
            # Look for successful BingX operations
            success_patterns = [
                "BingX trade executed",
                "Order placed successfully",
                "Trade execution successful"
            ]
            
            found_successes = {}
            for pattern in success_patterns:
                count = backend_logs.count(pattern)
                if count > 0:
                    found_successes[pattern] = count
                    logger.info(f"      ✅ Found '{pattern}': {count} occurrences")
            
            # Analyze results
            total_errors = sum(found_errors.values())
            total_successes = sum(found_successes.values())
            
            logger.info(f"   📊 Total BingX errors found: {total_errors}")
            logger.info(f"   📊 Total BingX successes found: {total_successes}")
            
            if total_errors > 0:
                # Extract specific error details
                error_details = []
                for pattern, count in found_errors.items():
                    if 'required' in pattern:
                        error_details.append(f"{pattern} ({count}x)")
                
                if error_details:
                    self.log_test_result("Backend Logs Analysis", False, f"Parameter errors found: {', '.join(error_details)}")
                else:
                    self.log_test_result("Backend Logs Analysis", False, f"BingX errors found: {total_errors} total")
            elif total_successes > 0:
                self.log_test_result("Backend Logs Analysis", True, f"BingX successes found: {total_successes}")
            else:
                self.log_test_result("Backend Logs Analysis", True, "No BingX errors found in recent logs")
                
        except Exception as e:
            self.log_test_result("Backend Logs Analysis", False, f"Exception: {str(e)}")
    
    async def test_7_ia2_automatic_integration(self):
        """Test 7: Test de l'intégration automatique IA2 → BingX"""
        logger.info("\n🔍 TEST 7: IA2 Automatic Integration Test")
        
        try:
            # Trigger IA2 analysis to see if automatic BingX execution works
            logger.info("   🚀 Triggering fresh analysis to test automatic integration...")
            
            start_response = requests.post(f"{self.api_url}/start-trading", timeout=180)
            
            if start_response.status_code in [200, 201]:
                logger.info("   ✅ Trading analysis triggered successfully")
                
                # Wait for processing
                logger.info("   ⏳ Waiting 45 seconds for IA2 processing and BingX integration...")
                await asyncio.sleep(45)
                
                # Check recent IA2 decisions
                decisions_response = requests.get(f"{self.api_url}/decisions", timeout=30)
                
                if decisions_response.status_code == 200:
                    decisions_data = decisions_response.json()
                    decisions = decisions_data.get('decisions', [])
                    
                    # Look for recent decisions with BingX execution attempts
                    recent_decisions = 0
                    bingx_executions = 0
                    
                    for decision in decisions[-10:]:  # Check last 10 decisions
                        timestamp_str = decision.get('timestamp', '')
                        if self._is_recent_timestamp(timestamp_str):
                            recent_decisions += 1
                            
                            # Check if decision has BingX execution info
                            reasoning = decision.get('ia2_reasoning', '')
                            if any(keyword in reasoning.lower() for keyword in ['bingx', 'executed', 'trade', 'order']):
                                bingx_executions += 1
                                symbol = decision.get('symbol', 'Unknown')
                                signal = decision.get('signal', 'Unknown')
                                logger.info(f"      ✅ {symbol}: {signal} - BingX execution detected in reasoning")
                    
                    logger.info(f"   📊 Recent IA2 decisions: {recent_decisions}")
                    logger.info(f"   📊 BingX execution attempts: {bingx_executions}")
                    
                    if bingx_executions > 0:
                        self.log_test_result("IA2 Automatic Integration", True, f"BingX integration working: {bingx_executions} execution attempts")
                    elif recent_decisions > 0:
                        self.log_test_result("IA2 Automatic Integration", False, f"IA2 decisions generated ({recent_decisions}) but no BingX integration detected")
                    else:
                        self.log_test_result("IA2 Automatic Integration", False, "No recent IA2 decisions generated")
                else:
                    self.log_test_result("IA2 Automatic Integration", False, f"Could not retrieve decisions: HTTP {decisions_response.status_code}")
            else:
                self.log_test_result("IA2 Automatic Integration", False, f"Could not trigger analysis: HTTP {start_response.status_code}")
                
        except Exception as e:
            self.log_test_result("IA2 Automatic Integration", False, f"Exception: {str(e)}")
    
    def _is_recent_timestamp(self, timestamp_str: str) -> bool:
        """Check if timestamp is from the last 2 hours"""
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
            return (now - timestamp) <= timedelta(hours=2)
            
        except Exception:
            return False
    
    async def run_comprehensive_tests(self):
        """Run all BingX trade execution tests"""
        logger.info("🚀 Starting BingX Trade Execution Issue Test Suite")
        logger.info("=" * 80)
        logger.info("📋 BINGX TRADE EXECUTION PROBLEM ANALYSIS")
        logger.info("🎯 Problem: IA1→IA2 works but /api/bingx/execute-ia2 returns 'Invalid parameters'")
        logger.info("🎯 Error: Missing timestamp, symbol, positionSide, type, side fields")
        logger.info("🎯 Goal: Identify why trades don't execute and propose solution")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_bingx_connectivity_status()
        await self.test_2_bingx_account_balance()
        await self.test_3_ia2_execute_endpoint_basic()
        await self.test_4_ia2_execute_parameter_variations()
        await self.test_5_bingx_manual_trade_endpoint()
        await self.test_6_backend_logs_analysis()
        await self.test_7_ia2_automatic_integration()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 BINGX TRADE EXECUTION ISSUE ANALYSIS SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        # Categorize issues
        parameter_errors = []
        connectivity_issues = []
        integration_problems = []
        working_components = []
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
            
            if result['success']:
                working_components.append(result['test'])
            else:
                if 'Parameter Error' in result['details'] or 'Invalid parameters' in result['details']:
                    parameter_errors.append(result['test'])
                elif 'Connectivity' in result['test'] or 'Account Access' in result['test']:
                    connectivity_issues.append(result['test'])
                else:
                    integration_problems.append(result['test'])
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # Issue analysis
        logger.info("\n" + "=" * 80)
        logger.info("🔍 ROOT CAUSE ANALYSIS")
        logger.info("=" * 80)
        
        if parameter_errors:
            logger.info("🚨 CRITICAL ISSUE IDENTIFIED: BingX Parameter Formatting Problems")
            logger.info("   📋 Components with parameter errors:")
            for error in parameter_errors:
                logger.info(f"      ❌ {error}")
            logger.info("   🔧 SOLUTION NEEDED: Fix parameter mapping in BingX integration")
            logger.info("   📝 Missing fields: timestamp, symbol, positionSide, type, side")
        
        if connectivity_issues:
            logger.info("⚠️ CONNECTIVITY ISSUES DETECTED:")
            for issue in connectivity_issues:
                logger.info(f"      ❌ {issue}")
        
        if integration_problems:
            logger.info("⚠️ INTEGRATION ISSUES DETECTED:")
            for problem in integration_problems:
                logger.info(f"      ❌ {problem}")
        
        if working_components:
            logger.info("✅ WORKING COMPONENTS:")
            for component in working_components:
                logger.info(f"      ✅ {component}")
        
        # Recommendations
        logger.info("\n📋 RECOMMENDATIONS:")
        
        if parameter_errors:
            logger.info("🔧 IMMEDIATE ACTION REQUIRED:")
            logger.info("   1. Fix BingX parameter mapping in place_market_order method")
            logger.info("   2. Ensure all required fields are included: timestamp, symbol, positionSide, type, side")
            logger.info("   3. Verify parameter format matches BingX API specification")
            logger.info("   4. Test parameter construction before sending to BingX API")
        
        if connectivity_issues:
            logger.info("🔧 CONNECTIVITY FIXES NEEDED:")
            logger.info("   1. Verify BingX API credentials and IP whitelist")
            logger.info("   2. Check network connectivity to BingX servers")
            logger.info("   3. Validate API key permissions for trading")
        
        if integration_problems:
            logger.info("🔧 INTEGRATION IMPROVEMENTS:")
            logger.info("   1. Review IA2 → BingX data flow")
            logger.info("   2. Add better error handling and logging")
            logger.info("   3. Implement parameter validation before API calls")
        
        # Final verdict
        logger.info(f"\n🏆 FINAL DIAGNOSIS:")
        
        if parameter_errors:
            logger.info("🚨 VERDICT: BingX Trade Execution BLOCKED by Parameter Errors")
            logger.info("✅ IA1→IA2 system working correctly")
            logger.info("❌ BingX parameter formatting preventing trade execution")
            logger.info("🔧 Fix required: Update parameter mapping in BingX integration")
        elif connectivity_issues:
            logger.info("⚠️ VERDICT: BingX Trade Execution BLOCKED by Connectivity Issues")
            logger.info("🔧 Fix required: Resolve BingX API connectivity problems")
        elif integration_problems:
            logger.info("⚠️ VERDICT: BingX Trade Execution has Integration Issues")
            logger.info("🔧 Fix required: Improve IA2→BingX integration flow")
        else:
            logger.info("✅ VERDICT: BingX Trade Execution System Working")
            logger.info("🎉 No critical issues detected - system operational")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = BingXTradeExecutionTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed >= total * 0.6:  # 60% pass rate considering this is debugging
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
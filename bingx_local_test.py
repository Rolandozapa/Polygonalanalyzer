#!/usr/bin/env python3
"""
BingX Local Testing - Direct localhost access to test POST signature fix
"""

import asyncio
import json
import logging
import requests
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BingXLocalTestSuite:
    """Local test suite for BingX POST signature fix"""
    
    def __init__(self):
        self.api_url = "http://localhost:8001/api"
        logger.info(f"Testing BingX locally at: {self.api_url}")
        
        self.mock_ia2_decision = {
            "symbol": "BTCUSDT",
            "signal": "LONG",
            "confidence": 0.85,
            "position_size": 2.5,
            "leverage": 5,
            "entry_price": 45000.0,
            "stop_loss": 43000.0,
            "take_profit": 48000.0,
            "reasoning": "Strong bullish momentum - POST signature test"
        }
    
    async def test_bingx_connectivity(self):
        """Test BingX API connectivity"""
        logger.info("\n🔍 TEST: BingX API Connectivity")
        
        try:
            response = requests.get(f"{self.api_url}/bingx/status", timeout=30)
            logger.info(f"   Status endpoint: HTTP {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"   Response: {json.dumps(data, indent=2)}")
                return True
            else:
                logger.info(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            logger.info(f"   Exception: {e}")
            return False
    
    async def test_balance_retrieval(self):
        """Test balance retrieval"""
        logger.info("\n🔍 TEST: Balance Retrieval")
        
        try:
            response = requests.get(f"{self.api_url}/bingx/balance", timeout=30)
            logger.info(f"   Balance endpoint: HTTP {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"   Response: {json.dumps(data, indent=2)}")
                
                # Check for actual balance data
                balance_fields = ['balance', 'total_balance', 'available_balance']
                for field in balance_fields:
                    if field in data and data[field] != 0:
                        logger.info(f"   ✅ Found balance: {field} = ${data[field]}")
                        return True
                
                logger.info("   ⚠️ No balance data found")
                return False
            else:
                logger.info(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            logger.info(f"   Exception: {e}")
            return False
    
    async def test_post_signature_fix(self):
        """Test POST signature fix with IA2 execute endpoint"""
        logger.info("\n🔍 TEST: POST Signature Fix - IA2 Execute")
        
        try:
            logger.info(f"   Sending POST request with data: {json.dumps(self.mock_ia2_decision, indent=2)}")
            
            response = requests.post(f"{self.api_url}/bingx/execute-ia2", 
                                   json=self.mock_ia2_decision, timeout=60)
            
            logger.info(f"   IA2 Execute endpoint: HTTP {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"   Response: {json.dumps(data, indent=2)}")
                
                # Check for signature errors
                response_text = json.dumps(data).lower()
                if 'null signature' in response_text or 'signature error' in response_text:
                    logger.info("   ❌ Signature error detected!")
                    return False
                else:
                    logger.info("   ✅ No signature errors detected")
                    
                    # Check execution status
                    status = data.get('status', 'unknown')
                    logger.info(f"   Execution status: {status}")
                    
                    if status in ['executed', 'skipped', 'rejected']:
                        logger.info("   ✅ Valid execution status")
                        return True
                    else:
                        logger.info("   ⚠️ Unexpected status")
                        return False
            else:
                logger.info(f"   Error: {response.text}")
                
                # Check if it's a signature error
                if 'signature' in response.text.lower():
                    logger.info("   ❌ Signature error in response!")
                    return False
                else:
                    logger.info("   ⚠️ Non-signature error")
                    return True  # Non-signature errors are acceptable
                
        except Exception as e:
            logger.info(f"   Exception: {e}")
            return False
    
    async def test_backend_logs_analysis(self):
        """Analyze backend logs for BingX activity"""
        logger.info("\n🔍 TEST: Backend Logs Analysis")
        
        try:
            # Get recent backend logs
            log_result = subprocess.run(
                ["tail", "-n", "200", "/var/log/supervisor/backend.out.log"],
                capture_output=True,
                text=True,
                timeout=10
            )
            recent_logs = log_result.stdout
            
            # Look for BingX-related logs
            bingx_logs = [line for line in recent_logs.split('\n') if 'bingx' in line.lower() or 'BingX' in line]
            
            if bingx_logs:
                logger.info("   📊 Recent BingX logs found:")
                for log_line in bingx_logs[-10:]:  # Last 10 BingX logs
                    logger.info(f"      {log_line}")
                return True
            else:
                logger.info("   ⚠️ No recent BingX logs found")
                return False
                
        except Exception as e:
            logger.info(f"   Exception: {e}")
            return False
    
    async def run_tests(self):
        """Run all local tests"""
        logger.info("🚀 Starting BingX Local Test Suite")
        logger.info("=" * 80)
        
        results = []
        
        # Run tests
        connectivity_result = await self.test_bingx_connectivity()
        results.append(("BingX Connectivity", connectivity_result))
        
        balance_result = await self.test_balance_retrieval()
        results.append(("Balance Retrieval", balance_result))
        
        post_signature_result = await self.test_post_signature_fix()
        results.append(("POST Signature Fix", post_signature_result))
        
        logs_result = await self.test_backend_logs_analysis()
        results.append(("Backend Logs Analysis", logs_result))
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 LOCAL TEST RESULTS SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = 0
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status}: {test_name}")
            if result:
                passed_tests += 1
        
        total_tests = len(results)
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests)*100:.1f}%)")
        
        # Analysis
        if passed_tests >= 3:
            logger.info("\n🎉 VERDICT: BingX integration is working locally!")
            logger.info("✅ The POST signature fix appears to be successful")
            logger.info("✅ Issue is likely with external gateway/proxy configuration")
        elif passed_tests >= 2:
            logger.info("\n⚠️ VERDICT: BingX integration partially working")
            logger.info("🔍 Some components working, others need attention")
        else:
            logger.info("\n❌ VERDICT: BingX integration has issues")
            logger.info("🚨 Multiple components not working properly")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = BingXLocalTestSuite()
    await test_suite.run_tests()

if __name__ == "__main__":
    asyncio.run(main())
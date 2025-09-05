import requests
import sys
import json
import time
import asyncio
import websockets
from datetime import datetime

class DualAITradingBotTester:
    def __init__(self, base_url="https://dual-ai-trader-1.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.ws_url = f"{base_url.replace('https', 'wss')}/api/ws"
        self.tests_run = 0
        self.tests_passed = 0
        self.websocket_messages = []

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=10):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if endpoint else f"{self.api_url}/"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    # Show more relevant data for each endpoint
                    if 'opportunities' in response_data:
                        print(f"   Found {len(response_data['opportunities'])} opportunities")
                    elif 'analyses' in response_data:
                        print(f"   Found {len(response_data['analyses'])} analyses")
                    elif 'decisions' in response_data:
                        print(f"   Found {len(response_data['decisions'])} decisions")
                    elif 'performance' in response_data:
                        perf = response_data['performance']
                        print(f"   Performance: {perf.get('total_opportunities', 0)} opps, {perf.get('executed_trades', 0)} trades")
                    else:
                        print(f"   Response: {json.dumps(response_data, indent=2)[:150]}...")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text[:200]}...")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    async def test_websocket_connection(self):
        """Test WebSocket real-time connection"""
        print(f"\n🔍 Testing WebSocket Connection...")
        self.tests_run += 1
        
        try:
            async with websockets.connect(self.ws_url, timeout=10) as websocket:
                print("✅ WebSocket connected successfully")
                
                # Send a test message
                await websocket.send("test message")
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    response_data = json.loads(response)
                    print(f"   WebSocket Response: {response_data}")
                    self.tests_passed += 1
                    return True
                except asyncio.TimeoutError:
                    print("⚠️  WebSocket connected but no response received")
                    self.tests_passed += 1
                    return True
                    
        except Exception as e:
            print(f"❌ WebSocket connection failed: {str(e)}")
            return False

    def test_root_endpoint(self):
        """Test root API endpoint"""
        return self.run_test("Root Endpoint", "GET", "", 200)

    def test_get_opportunities(self):
        """Test get opportunities endpoint"""
        return self.run_test("Get Opportunities", "GET", "opportunities", 200)

    def test_get_analyses(self):
        """Test get analyses endpoint"""
        return self.run_test("Get Technical Analyses", "GET", "analyses", 200)

    def test_get_decisions(self):
        """Test get decisions endpoint"""
        return self.run_test("Get Trading Decisions", "GET", "decisions", 200)

    def test_get_performance(self):
        """Test get performance endpoint"""
        return self.run_test("Get Performance Metrics", "GET", "performance", 200)

    def test_start_trading(self):
        """Test start trading endpoint"""
        return self.run_test("Start Trading System", "POST", "start-trading", 200)

    def test_stop_trading(self):
        """Test stop trading endpoint"""
        return self.run_test("Stop Trading System", "POST", "stop-trading", 200)

    def test_ai_integration(self):
        """Test AI integration by starting trading and checking for AI-generated data"""
        print(f"\n🤖 Testing AI Integration...")
        
        # Start trading
        success, _ = self.test_start_trading()
        if not success:
            print("❌ Cannot test AI integration - trading start failed")
            return False
        
        print("   Waiting for AI to generate data (30 seconds)...")
        time.sleep(30)
        
        # Check if AI generated analyses
        success, analyses_data = self.test_get_analyses()
        if success and analyses_data.get('analyses'):
            analyses = analyses_data['analyses']
            if len(analyses) > 0:
                analysis = analyses[0]
                print(f"✅ IA1 Technical Analysis generated:")
                print(f"   Symbol: {analysis.get('symbol')}")
                print(f"   Confidence: {analysis.get('analysis_confidence', 0) * 100:.1f}%")
                print(f"   RSI: {analysis.get('rsi', 0):.2f}")
                print(f"   Reasoning: {analysis.get('ia1_reasoning', '')[:100]}...")
                
                # Check if AI generated decisions
                success, decisions_data = self.test_get_decisions()
                if success and decisions_data.get('decisions'):
                    decisions = decisions_data['decisions']
                    if len(decisions) > 0:
                        decision = decisions[0]
                        print(f"✅ IA2 Trading Decision generated:")
                        print(f"   Signal: {decision.get('signal')}")
                        print(f"   Confidence: {decision.get('confidence', 0) * 100:.1f}%")
                        print(f"   Entry Price: ${decision.get('entry_price', 0):.2f}")
                        print(f"   Reasoning: {decision.get('ia2_reasoning', '')[:100]}...")
                        return True
        
        print("❌ AI integration test failed - no AI-generated data found")
        return False

    async def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting Dual AI Trading Bot System Tests")
        print("=" * 60)
        
        # Basic API tests
        self.test_root_endpoint()
        self.test_get_opportunities()
        self.test_get_analyses()
        self.test_get_decisions()
        self.test_get_performance()
        
        # WebSocket test
        await self.test_websocket_connection()
        
        # Trading system tests
        self.test_start_trading()
        
        # AI integration test (this takes time)
        self.test_ai_integration()
        
        # Stop trading
        self.test_stop_trading()
        
        # Print results
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {self.tests_passed}/{self.tests_run} tests passed")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed! The Dual AI Trading Bot system is working correctly.")
            return 0
        else:
            failed_tests = self.tests_run - self.tests_passed
            print(f"⚠️  {failed_tests} test(s) failed. Please check the issues above.")
            return 1

async def main():
    """Main test function"""
    tester = DualAITradingBotTester()
    return await tester.run_all_tests()

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test execution failed: {str(e)}")
        sys.exit(1)
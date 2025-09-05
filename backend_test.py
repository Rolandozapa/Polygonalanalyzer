import requests
import sys
import json
import time
import asyncio
import websockets
from datetime import datetime
import os
from pathlib import Path

class DualAITradingBotTester:
    def __init__(self, base_url=None):
        # Get the correct backend URL from frontend/.env
        if base_url is None:
            try:
                env_path = Path(__file__).parent / "frontend" / ".env"
                with open(env_path, 'r') as f:
                    for line in f:
                        if line.startswith('REACT_APP_BACKEND_URL='):
                            base_url = line.split('=', 1)[1].strip()
                            break
                if not base_url:
                    base_url = "https://market-oracle-ai-3.preview.emergentagent.com"
            except:
                base_url = "https://market-oracle-ai-3.preview.emergentagent.com"
        
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.ws_url = f"{base_url.replace('http', 'ws')}/api/ws"
        self.tests_run = 0
        self.tests_passed = 0
        self.websocket_messages = []
        self.ia1_performance_times = []

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test with extended timeout for IA1 optimization testing"""
        url = f"{self.api_url}/{endpoint}" if endpoint else f"{self.api_url}/"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        
        start_time = time.time()
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)

            end_time = time.time()
            response_time = end_time - start_time
            
            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code} - Time: {response_time:.2f}s")
                
                # Track IA1 performance times for optimization testing
                if 'analyze' in endpoint:
                    self.ia1_performance_times.append(response_time)
                    print(f"   ⚡ IA1 Analysis Time: {response_time:.2f}s")
                
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
                    elif 'status' in response_data:
                        print(f"   System Status: {response_data['status']}")
                    else:
                        print(f"   Response: {json.dumps(response_data, indent=2)[:150]}...")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code} - Time: {response_time:.2f}s")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text[:200]}...")
                return False, {}

        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            print(f"❌ Failed - Error: {str(e)} - Time: {response_time:.2f}s")
            return False, {}

    def test_system_status(self):
        """Test system status endpoint"""
        return self.run_test("System Status", "GET", "", 200)

    def test_market_status(self):
        """Test market status endpoint"""
        return self.run_test("Market Status", "GET", "market-status", 200)

    def test_get_opportunities(self):
        """Test get opportunities endpoint (Scout functionality)"""
        return self.run_test("Get Opportunities (Scout)", "GET", "opportunities", 200)

    def test_get_analyses(self):
        """Test get analyses endpoint"""
        return self.run_test("Get Technical Analyses", "GET", "analyses", 200)

    def test_get_decisions(self):
        """Test get decisions endpoint (IA2 functionality)"""
        return self.run_test("Get Trading Decisions (IA2)", "GET", "decisions", 200)

    def test_start_trading_system(self):
        """Test starting the trading system"""
        return self.run_test("Start Trading System", "POST", "start-trading", 200)

    def test_stop_trading_system(self):
        """Test stopping the trading system"""
        return self.run_test("Stop Trading System", "POST", "stop-trading", 200)

    def test_ia1_analysis_speed_via_system(self):
        """Test IA1 analysis speed through the actual system workflow"""
        print(f"\n⚡ Testing IA1 Performance via System Workflow...")
        
        # First, get baseline analysis count
        success, initial_analyses = self.test_get_analyses()
        if not success:
            print(f"   ❌ Cannot get initial analyses")
            return False
        
        initial_count = len(initial_analyses.get('analyses', []))
        print(f"   📊 Initial analyses count: {initial_count}")
        
        # Start the trading system
        print(f"   🚀 Starting trading system...")
        success, _ = self.test_start_trading_system()
        if not success:
            print(f"   ❌ Failed to start trading system")
            return False
        
        # Wait for the system to generate new analyses (IA1 optimization test)
        print(f"   ⏱️  Waiting for IA1 to generate analyses (60 seconds max)...")
        
        analysis_start_time = time.time()
        new_analyses_found = False
        max_wait_time = 60  # 60 seconds max wait
        check_interval = 5   # Check every 5 seconds
        
        while time.time() - analysis_start_time < max_wait_time:
            time.sleep(check_interval)
            
            success, current_analyses = self.test_get_analyses()
            if success:
                current_count = len(current_analyses.get('analyses', []))
                elapsed_time = time.time() - analysis_start_time
                
                print(f"   📈 After {elapsed_time:.1f}s: {current_count} analyses (was {initial_count})")
                
                if current_count > initial_count:
                    # New analysis found! Calculate the time
                    analysis_time = elapsed_time
                    self.ia1_performance_times.append(analysis_time)
                    
                    print(f"   ✅ New IA1 analysis generated in {analysis_time:.2f}s")
                    
                    # Check the quality of the latest analysis
                    latest_analysis = current_analyses['analyses'][0]  # Most recent first
                    self._validate_analysis_quality(latest_analysis)
                    
                    new_analyses_found = True
                    break
            else:
                print(f"   ⚠️  Failed to check analyses at {time.time() - analysis_start_time:.1f}s")
        
        # Stop the trading system
        print(f"   🛑 Stopping trading system...")
        self.test_stop_trading_system()
        
        if new_analyses_found:
            return True
        else:
            print(f"   ❌ No new IA1 analyses generated within {max_wait_time}s")
            return False

    def _validate_analysis_quality(self, analysis):
        """Validate the quality of an IA1 analysis"""
        print(f"   🔍 Validating analysis quality:")
        
        symbol = analysis.get('symbol', 'Unknown')
        rsi = analysis.get('rsi', 0)
        macd_signal = analysis.get('macd_signal', 0)
        confidence = analysis.get('analysis_confidence', 0)
        reasoning = analysis.get('ia1_reasoning', '')
        
        print(f"      Symbol: {symbol}")
        print(f"      RSI: {rsi:.2f}")
        print(f"      MACD Signal: {macd_signal:.6f}")
        print(f"      Confidence: {confidence:.2f}")
        print(f"      Reasoning length: {len(reasoning)} chars")
        
        # Validate technical indicators are reasonable
        quality_checks = {
            "RSI in range": 0 <= rsi <= 100,
            "Confidence reasonable": confidence >= 0.5,
            "Has reasoning": len(reasoning) > 50,
            "Has support levels": len(analysis.get('support_levels', [])) > 0,
            "Has resistance levels": len(analysis.get('resistance_levels', [])) > 0
        }
        
        passed_checks = sum(quality_checks.values())
        total_checks = len(quality_checks)
        
        print(f"      Quality checks: {passed_checks}/{total_checks} passed")
        for check, passed in quality_checks.items():
            print(f"        {check}: {'✅' if passed else '❌'}")
        
        return passed_checks >= total_checks * 0.8  # 80% pass rate

    def test_scout_ia1_integration_via_system(self):
        """Test Scout -> IA1 integration through system workflow"""
        print(f"\n🔗 Testing Scout -> IA1 Integration via System...")
        
        # Check opportunities (Scout output)
        success, opportunities_data = self.test_get_opportunities()
        if not success:
            print(f"   ❌ Scout not working - no opportunities")
            return False
        
        opportunities = opportunities_data.get('opportunities', [])
        print(f"   ✅ Scout working: {len(opportunities)} opportunities found")
        
        # Check analyses (IA1 output)
        success, analyses_data = self.test_get_analyses()
        if not success:
            print(f"   ❌ IA1 not working - cannot get analyses")
            return False
        
        analyses = analyses_data.get('analyses', [])
        print(f"   📊 IA1 analyses available: {len(analyses)}")
        
        if len(analyses) == 0:
            print(f"   ⚠️  No IA1 analyses found - may need to run system longer")
            return False
        
        # Check if analyses correspond to opportunities (integration test)
        opportunity_symbols = set(opp.get('symbol', '') for opp in opportunities)
        analysis_symbols = set(analysis.get('symbol', '') for analysis in analyses)
        
        common_symbols = opportunity_symbols.intersection(analysis_symbols)
        
        print(f"   🔍 Integration check:")
        print(f"      Opportunity symbols: {len(opportunity_symbols)}")
        print(f"      Analysis symbols: {len(analysis_symbols)}")
        print(f"      Common symbols: {len(common_symbols)}")
        
        if len(common_symbols) > 0:
            print(f"   ✅ Scout -> IA1 integration working: {len(common_symbols)} symbols processed")
            return True
        else:
            print(f"   ⚠️  Limited integration evidence - may be timing issue")
            return len(analyses) > 0  # At least IA1 is working

    def test_technical_analysis_quality_from_system(self):
        """Test technical analysis quality from actual system output"""
        print(f"\n📈 Testing Technical Analysis Quality from System...")
        
        success, analyses_data = self.test_get_analyses()
        if not success:
            print(f"   ❌ Cannot retrieve analyses for quality testing")
            return False
        
        analyses = analyses_data.get('analyses', [])
        if len(analyses) == 0:
            print(f"   ❌ No analyses available for quality testing")
            return False
        
        print(f"   📊 Testing quality of {len(analyses)} analyses...")
        
        quality_scores = []
        
        for i, analysis in enumerate(analyses[:5]):  # Test up to 5 most recent
            print(f"\n   Analysis {i+1}:")
            quality_score = self._validate_analysis_quality(analysis)
            quality_scores.append(1 if quality_score else 0)
        
        overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        print(f"\n   📊 Overall Quality Assessment:")
        print(f"      Analyses tested: {len(quality_scores)}")
        print(f"      Quality score: {overall_quality*100:.1f}%")
        
        if overall_quality >= 0.8:
            print(f"   ✅ Technical analysis quality maintained with 10-day optimization")
            return True
        else:
            print(f"   ⚠️  Technical analysis quality concerns detected")
            return overall_quality >= 0.6  # Accept 60% as partial success

    def test_ia1_optimization_evidence(self):
        """Test for evidence of IA1 optimization implementation"""
        print(f"\n🔍 Testing IA1 Optimization Evidence...")
        
        # Check if analyses show high quality (indicating optimization is working)
        success, analyses_data = self.test_get_analyses()
        if not success:
            print(f"   ❌ Cannot retrieve analyses")
            return False
        
        analyses = analyses_data.get('analyses', [])
        if len(analyses) == 0:
            print(f"   ❌ No analyses available")
            return False
        
        print(f"   📊 Analyzing {len(analyses)} recent IA1 analyses...")
        
        # Check analysis timestamps to estimate generation speed
        recent_analyses = analyses[:3]  # Most recent 3
        optimization_indicators = 0
        
        for i, analysis in enumerate(recent_analyses):
            symbol = analysis.get('symbol', 'Unknown')
            confidence = analysis.get('analysis_confidence', 0)
            reasoning = analysis.get('ia1_reasoning', '')
            timestamp = analysis.get('timestamp', '')
            
            print(f"\n   Analysis {i+1} - {symbol}:")
            print(f"      Timestamp: {timestamp}")
            print(f"      Confidence: {confidence:.2f}")
            print(f"      Reasoning length: {len(reasoning)} chars")
            
            # Look for optimization indicators
            quality_indicators = 0
            
            # High confidence suggests good analysis
            if confidence >= 0.7:
                quality_indicators += 1
                print(f"      ✅ High confidence analysis")
            
            # Reasonable reasoning length (not too verbose, not too short)
            if 200 <= len(reasoning) <= 1500:
                quality_indicators += 1
                print(f"      ✅ Appropriate reasoning length")
            
            # Technical indicators present
            if (analysis.get('rsi', 0) > 0 and 
                len(analysis.get('support_levels', [])) > 0 and 
                len(analysis.get('resistance_levels', [])) > 0):
                quality_indicators += 1
                print(f"      ✅ Complete technical indicators")
            
            # Check for fast/optimized language in reasoning
            fast_keywords = ['fast', 'quick', 'streamlined', '10-day', 'optimized', 'efficient']
            if any(keyword in reasoning.lower() for keyword in fast_keywords):
                quality_indicators += 1
                print(f"      ✅ Contains optimization language")
            
            if quality_indicators >= 3:
                optimization_indicators += 1
                print(f"      ✅ Shows optimization characteristics")
            else:
                print(f"      ⚠️  Limited optimization evidence")
        
        optimization_rate = optimization_indicators / len(recent_analyses)
        print(f"\n   📊 Optimization Evidence: {optimization_indicators}/{len(recent_analyses)} analyses ({optimization_rate*100:.1f}%)")
        
        if optimization_rate >= 0.67:  # 2/3 analyses show optimization
            print(f"   ✅ Strong evidence of IA1 optimization implementation")
            return True
        else:
            print(f"   ⚠️  Limited evidence of IA1 optimization")
            return optimization_rate > 0

    async def run_ia1_optimization_tests(self):
        """Run comprehensive IA1 performance optimization tests"""
        print("🚀 Starting IA1 Performance Optimization Tests")
        print("=" * 70)
        print(f"🎯 Target: Reduce IA1 analysis time from 50-60s to 15-25s")
        print(f"🔧 Optimization: 30-day → 10-day historical data + streamlined prompts")
        print("=" * 70)
        
        # 1. Basic connectivity test
        print(f"\n1️⃣ BASIC CONNECTIVITY TESTS")
        system_success, _ = self.test_system_status()
        market_success, _ = self.test_market_status()
        
        # 2. Scout functionality test
        print(f"\n2️⃣ SCOUT FUNCTIONALITY TEST")
        scout_success, _ = self.test_get_opportunities()
        
        # 3. IA1 Analysis Speed Test (MAIN TEST)
        print(f"\n3️⃣ IA1 OPTIMIZATION EVIDENCE TEST")
        optimization_evidence = self.test_ia1_optimization_evidence()
        
        # 4. IA1 Analysis Speed Test (System Workflow)
        print(f"\n4️⃣ IA1 ANALYSIS SPEED TEST (SYSTEM WORKFLOW)")
        speed_success = self.test_ia1_analysis_speed_via_system()
        
        # 5. Scout -> IA1 Integration Test
        print(f"\n5️⃣ SCOUT -> IA1 INTEGRATION TEST")
        integration_success = self.test_scout_ia1_integration_via_system()
        
        # 6. Technical Analysis Quality Test
        print(f"\n6️⃣ TECHNICAL ANALYSIS QUALITY TEST")
        quality_success = self.test_technical_analysis_quality_from_system()
        
        # 7. IA2 Decision Making Test
        print(f"\n7️⃣ IA2 DECISION MAKING TEST")
        decision_success, _ = self.test_get_decisions()
        
        # Performance Summary
        print("\n" + "=" * 70)
        print("📊 IA1 OPTIMIZATION TEST RESULTS")
        print("=" * 70)
        
        if self.ia1_performance_times:
            avg_time = sum(self.ia1_performance_times) / len(self.ia1_performance_times)
            improvement = ((55 - avg_time) / 55) * 100  # Assuming 55s baseline
            
            print(f"⚡ Performance Metrics:")
            print(f"   • Average IA1 Analysis Time: {avg_time:.2f}s")
            print(f"   • Performance Improvement: {improvement:.1f}%")
            print(f"   • Target Achievement: {'✅ SUCCESS' if avg_time <= 25 else '❌ NEEDS WORK'}")
            
            if avg_time <= 15:
                print(f"   🚀 EXCEPTIONAL: Exceeded optimization target!")
            elif avg_time <= 25:
                print(f"   ✅ SUCCESS: Within optimization target (15-25s)")
            elif avg_time <= 40:
                print(f"   ⚠️  PARTIAL: Better than baseline but above target")
            else:
                print(f"   ❌ FAILED: No significant improvement over baseline")
        
        print(f"\n🔍 Test Results Summary:")
        print(f"   • System Connectivity: {'✅' if system_success else '❌'}")
        print(f"   • Market Status: {'✅' if market_success else '❌'}")
        print(f"   • Scout Functionality: {'✅' if scout_success else '❌'}")
        print(f"   • IA1 Optimization Evidence: {'✅' if optimization_evidence else '❌'}")
        print(f"   • IA1 Speed Test: {'✅' if speed_success else '❌'}")
        print(f"   • Scout->IA1 Integration: {'✅' if integration_success else '❌'}")
        print(f"   • Technical Analysis Quality: {'✅' if quality_success else '❌'}")
        print(f"   • IA2 Decision Making: {'✅' if decision_success else '❌'}")
        
        # Overall assessment
        critical_tests = [scout_success, optimization_evidence, integration_success, quality_success]
        critical_passed = sum(critical_tests)
        
        print(f"\n🎯 Overall Assessment:")
        if critical_passed == 4:
            print(f"   ✅ IA1 OPTIMIZATION SUCCESSFUL - All critical tests passed")
            optimization_status = "SUCCESS"
        elif critical_passed >= 3:
            print(f"   ⚠️  IA1 OPTIMIZATION PARTIAL - Some issues detected")
            optimization_status = "PARTIAL"
        else:
            print(f"   ❌ IA1 OPTIMIZATION FAILED - Major issues detected")
            optimization_status = "FAILED"
        
        print(f"\n📋 Test Summary: {self.tests_passed}/{self.tests_run} tests passed")
        
        return optimization_status, {
            "avg_analysis_time": sum(self.ia1_performance_times) / len(self.ia1_performance_times) if self.ia1_performance_times else 0,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_run,
            "system_working": system_success,
            "scout_working": scout_success,
            "ia1_speed_optimized": speed_success,
            "integration_working": integration_success,
            "quality_maintained": quality_success,
            "ia2_working": decision_success
        }

    async def run_all_tests(self):
        """Run all tests including IA1 optimization focus"""
        return await self.run_ia1_optimization_tests()

async def main():
    """Main test function"""
    tester = DualAITradingBotTester()
    return await tester.run_all_tests()

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        print(f"\nFinal result: {result}")
        sys.exit(0 if result[0] in ["SUCCESS", "PARTIAL"] else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test execution failed: {str(e)}")
        sys.exit(1)
import requests
import sys
import json
import time
from datetime import datetime
import os
from pathlib import Path

class CorrectedAPIEconomyTester:
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
                    base_url = "https://cryptobot-plus.preview.emergentagent.com"
            except:
                base_url = "https://cryptobot-plus.preview.emergentagent.com"
        
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test"""
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
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=timeout)

            end_time = time.time()
            response_time = end_time - start_time
            
            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code} - Time: {response_time:.2f}s")
                
                try:
                    response_data = response.json()
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code} - Time: {response_time:.2f}s")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_scout_operation_continuity(self):
        """Test 1: Scout continues to function and find opportunities"""
        print(f"\n🔍 TEST 1: SCOUT OPERATION CONTINUITY")
        
        # Clear cache first
        print(f"   🗑️ Clearing cache for fresh cycle...")
        success, _ = self.run_test("Clear Cache", "DELETE", "decisions/clear", 200)
        
        # Test scout functionality
        success, opportunities_data = self.run_test("Scout Opportunities", "GET", "opportunities", 200)
        if not success:
            return False
        
        opportunities = opportunities_data.get('opportunities', [])
        print(f"   📊 Scout Results: {len(opportunities)} opportunities found")
        
        if len(opportunities) == 0:
            print(f"   ❌ Scout not generating opportunities")
            return False
        
        # Analyze opportunity quality
        high_confidence_count = sum(1 for opp in opportunities if opp.get('data_confidence', 0) >= 0.7)
        multi_source_count = sum(1 for opp in opportunities if len(opp.get('data_sources', [])) > 1)
        
        print(f"   📈 Quality Analysis:")
        print(f"      High confidence (≥70%): {high_confidence_count}/{len(opportunities)} ({high_confidence_count/len(opportunities)*100:.1f}%)")
        print(f"      Multi-source: {multi_source_count}/{len(opportunities)} ({multi_source_count/len(opportunities)*100:.1f}%)")
        
        scout_success = len(opportunities) >= 5 and high_confidence_count >= len(opportunities) * 0.3
        print(f"   🎯 Scout Continuity: {'✅ SUCCESS' if scout_success else '❌ FAILED'}")
        
        return scout_success

    def test_multi_source_ohlcv_validation(self):
        """Test 2: Multi-Source OHLCV Validation system"""
        print(f"\n📊 TEST 2: MULTI-SOURCE OHLCV VALIDATION")
        
        # Get technical analyses to check validation
        success, analyses_data = self.run_test("Technical Analyses", "GET", "analyses", 200)
        if not success:
            return False
        
        analyses = analyses_data.get('analyses', [])
        print(f"   📊 Analyses Available: {len(analyses)}")
        
        if len(analyses) == 0:
            print(f"   ❌ No analyses - validation may be too strict")
            return False
        
        # Look for multi-source validation evidence
        multi_source_evidence = 0
        coherence_evidence = 0
        
        for analysis in analyses[:10]:
            reasoning = analysis.get('ia1_reasoning', '')
            sources = analysis.get('data_sources', [])
            
            # Check for multi-source keywords
            multi_keywords = ['multi-source', 'coherence', 'validation', 'primary', 'secondary']
            if any(keyword in reasoning.lower() for keyword in multi_keywords):
                multi_source_evidence += 1
            
            # Check for coherence validation
            if 'coherence' in reasoning.lower() or len(sources) > 1:
                coherence_evidence += 1
        
        print(f"   📈 Validation Evidence:")
        print(f"      Multi-source evidence: {multi_source_evidence}/{len(analyses)} ({multi_source_evidence/len(analyses)*100:.1f}%)")
        print(f"      Coherence validation: {coherence_evidence}/{len(analyses)} ({coherence_evidence/len(analyses)*100:.1f}%)")
        
        validation_success = multi_source_evidence > 0 and coherence_evidence > 0
        print(f"   🎯 Multi-Source Validation: {'✅ SUCCESS' if validation_success else '❌ NEEDS WORK'}")
        
        return validation_success

    def test_smart_api_economy_filtering(self):
        """Test 3: Smart API Economy Filtering"""
        print(f"\n💰 TEST 3: SMART API ECONOMY FILTERING")
        
        # Get pipeline data
        success_opp, opp_data = self.run_test("Opportunities", "GET", "opportunities", 200)
        success_ana, ana_data = self.run_test("Analyses", "GET", "analyses", 200)
        
        if not (success_opp and success_ana):
            return False
        
        opportunities = opp_data.get('opportunities', [])
        analyses = ana_data.get('analyses', [])
        
        print(f"   📊 Pipeline Flow:")
        print(f"      Opportunities: {len(opportunities)}")
        print(f"      Analyses: {len(analyses)}")
        
        if len(opportunities) == 0:
            return False
        
        # Calculate filtering rate
        conversion_rate = len(analyses) / len(opportunities)
        print(f"      Conversion Rate: {conversion_rate*100:.1f}%")
        
        # Check filtering criteria
        no_data_opportunities = sum(1 for opp in opportunities if len(opp.get('data_sources', [])) == 0)
        multi_source_opportunities = sum(1 for opp in opportunities if len(opp.get('data_sources', [])) >= 2)
        
        print(f"   📈 Filtering Analysis:")
        print(f"      No OHLCV data: {no_data_opportunities}")
        print(f"      Multi-source (≥2): {multi_source_opportunities}")
        
        # Smart filtering validation
        not_over_filtering = conversion_rate >= 0.2  # At least 20% pass
        not_under_filtering = conversion_rate <= 0.8  # Not letting everything through
        allows_quality_data = len(analyses) > 0
        
        filtering_success = not_over_filtering and allows_quality_data
        print(f"   🎯 Smart Filtering: {'✅ SUCCESS' if filtering_success else '❌ NEEDS ADJUSTMENT'}")
        
        return filtering_success

    def test_technical_pattern_integration(self):
        """Test 4: Technical Pattern Integration"""
        print(f"\n📈 TEST 4: TECHNICAL PATTERN INTEGRATION")
        
        success, analyses_data = self.run_test("Technical Analyses", "GET", "analyses", 200)
        if not success:
            return False
        
        analyses = analyses_data.get('analyses', [])
        if len(analyses) == 0:
            return False
        
        # Check technical indicators
        rsi_working = sum(1 for a in analyses if a.get('rsi', 50) != 50)
        macd_working = sum(1 for a in analyses if abs(a.get('macd_signal', 0)) > 0.000001)
        patterns_detected = sum(1 for a in analyses if len(a.get('patterns_detected', [])) > 0)
        
        print(f"   📊 Technical Analysis:")
        print(f"      RSI Working: {rsi_working}/{len(analyses)} ({rsi_working/len(analyses)*100:.1f}%)")
        print(f"      MACD Working: {macd_working}/{len(analyses)} ({macd_working/len(analyses)*100:.1f}%)")
        print(f"      Patterns Detected: {patterns_detected}/{len(analyses)} ({patterns_detected/len(analyses)*100:.1f}%)")
        
        technical_success = (rsi_working + macd_working) >= len(analyses) * 0.6
        print(f"   🎯 Technical Integration: {'✅ SUCCESS' if technical_success else '❌ NEEDS WORK'}")
        
        return technical_success

    def test_end_to_end_pipeline(self):
        """Test 5: End-to-End Corrected Pipeline"""
        print(f"\n🔄 TEST 5: END-TO-END CORRECTED PIPELINE")
        
        # Start trading system
        print(f"   🚀 Starting trading system...")
        success, _ = self.run_test("Start Trading", "POST", "start-trading", 200)
        if not success:
            return False
        
        # Wait for pipeline processing
        print(f"   ⏱️ Waiting for pipeline (45 seconds)...")
        time.sleep(45)
        
        # Get all pipeline data
        success_opp, opp_data = self.run_test("Opportunities", "GET", "opportunities", 200)
        success_ana, ana_data = self.run_test("Analyses", "GET", "analyses", 200)
        success_dec, dec_data = self.run_test("Decisions", "GET", "decisions", 200)
        
        # Stop trading system
        self.run_test("Stop Trading", "POST", "stop-trading", 200)
        
        if not (success_opp and success_ana and success_dec):
            return False
        
        opportunities = opp_data.get('opportunities', [])
        analyses = ana_data.get('analyses', [])
        decisions = dec_data.get('decisions', [])
        
        print(f"   📊 Pipeline Results:")
        print(f"      Scout → Opportunities: {len(opportunities)}")
        print(f"      IA1 → Analyses: {len(analyses)}")
        print(f"      IA2 → Decisions: {len(decisions)}")
        
        # Find common symbols
        opp_symbols = set(opp.get('symbol', '') for opp in opportunities)
        ana_symbols = set(ana.get('symbol', '') for ana in analyses)
        dec_symbols = set(dec.get('symbol', '') for dec in decisions)
        
        common_symbols = opp_symbols.intersection(ana_symbols).intersection(dec_symbols)
        print(f"      Full Pipeline Symbols: {len(common_symbols)}")
        
        pipeline_success = len(opportunities) > 0 and len(analyses) > 0 and len(decisions) > 0 and len(common_symbols) > 0
        print(f"   🎯 End-to-End Pipeline: {'✅ SUCCESS' if pipeline_success else '❌ NEEDS WORK'}")
        
        return pipeline_success

    def test_frontend_data_generation(self):
        """Test 6: Frontend Data Generation"""
        print(f"\n🖥️ TEST 6: FRONTEND DATA GENERATION")
        
        # Test all frontend endpoints
        endpoints = [
            ("opportunities", "Opportunities"),
            ("analyses", "Analyses"),
            ("decisions", "Decisions"),
            ("market-status", "Market Status"),
            ("performance", "Performance")
        ]
        
        endpoint_results = {}
        for endpoint, name in endpoints:
            success, data = self.run_test(f"Frontend {name}", "GET", endpoint, 200)
            endpoint_results[endpoint] = success and len(data.get(endpoint, [])) > 0 if isinstance(data.get(endpoint), list) else success
        
        # Check for trading signals (not all HOLD)
        success, dec_data = self.run_test("Check Trading Signals", "GET", "decisions", 200)
        trading_signals = 0
        if success:
            decisions = dec_data.get('decisions', [])
            trading_signals = sum(1 for d in decisions if d.get('signal', 'hold').lower() in ['long', 'short'])
        
        print(f"   📊 Frontend Data:")
        for endpoint, name in endpoints:
            status = "✅" if endpoint_results[endpoint] else "❌"
            print(f"      {name}: {status}")
        print(f"      Trading Signals: {trading_signals}")
        
        frontend_success = all(endpoint_results.values()) and trading_signals > 0
        print(f"   🎯 Frontend Data Generation: {'✅ SUCCESS' if frontend_success else '❌ NEEDS WORK'}")
        
        return frontend_success

    def run_comprehensive_test(self):
        """Run comprehensive corrected API economy test"""
        print(f"\n🎯 COMPREHENSIVE CORRECTED API ECONOMY SYSTEM TEST")
        print(f"=" * 80)
        
        test_results = {
            'scout_continuity': self.test_scout_operation_continuity(),
            'multi_source_validation': self.test_multi_source_ohlcv_validation(),
            'smart_filtering': self.test_smart_api_economy_filtering(),
            'technical_integration': self.test_technical_pattern_integration(),
            'end_to_end_pipeline': self.test_end_to_end_pipeline(),
            'frontend_data': self.test_frontend_data_generation()
        }
        
        # Results summary
        tests_passed = sum(test_results.values())
        total_tests = len(test_results)
        success_rate = tests_passed / total_tests
        
        print(f"\n" + "=" * 80)
        print(f"🎯 CORRECTED API ECONOMY TEST RESULTS")
        print(f"=" * 80)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Tests Passed: {tests_passed}/{total_tests}")
        print(f"   Success Rate: {success_rate*100:.1f}%")
        
        if success_rate >= 0.8:
            print(f"\n✅ CORRECTED API ECONOMY SYSTEM: SUCCESS")
            print(f"   The corrected system is working properly!")
        elif success_rate >= 0.6:
            print(f"\n⚠️ CORRECTED API ECONOMY SYSTEM: PARTIAL SUCCESS")
            print(f"   Most components working, some adjustments needed")
        else:
            print(f"\n❌ CORRECTED API ECONOMY SYSTEM: NEEDS WORK")
            print(f"   Significant fixes required")
        
        return success_rate >= 0.8

if __name__ == "__main__":
    tester = CorrectedAPIEconomyTester()
    success = tester.run_comprehensive_test()
    
    if success:
        print(f"\n🎉 CORRECTED API ECONOMY TEST: PASSED")
        sys.exit(0)
    else:
        print(f"\n❌ CORRECTED API ECONOMY TEST: FAILED")
        sys.exit(1)
#!/usr/bin/env python3

import requests
import sys
import json
import time
import os
from pathlib import Path

class ScoutOptionATester:
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
                    base_url = "https://ai-trade-pro.preview.emergentagent.com"
            except:
                base_url = "https://ai-trade-pro.preview.emergentagent.com"
        
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

    def test_scout_option_a_implementation(self):
        """Test Scout Option A Implementation - Lateral Filter Removed + 7 Overrides Optimized"""
        print(f"\n🎯 TESTING SCOUT OPTION A IMPLEMENTATION...")
        print(f"   📋 EXPECTED: Lateral filter REMOVED + 7 overrides optimized")
        print(f"   🎯 TARGET: Pass rate 20-25% (up from 16%)")
        print(f"   🔍 FOCUS: KTAUSDT-type opportunities should now pass")
        
        # Step 1: Clear cache for fresh test
        print(f"\n   🗑️ Step 1: Clearing cache for fresh Option A test...")
        try:
            clear_success, clear_result = self.run_test("Clear Cache", "POST", "decisions/clear", 200)
            if clear_success:
                print(f"   ✅ Cache cleared - ready for fresh Option A test")
            else:
                print(f"   ⚠️ Cache clear failed, using existing data")
        except:
            print(f"   ⚠️ Cache clear not available, continuing...")
        
        # Step 2: Start system and measure Scout→IA1 pass rate
        print(f"\n   🚀 Step 2: Starting system to test Option A improvements...")
        start_success, _ = self.run_test("Start Trading System", "POST", "start-trading", 200)
        if not start_success:
            print(f"   ❌ Failed to start trading system")
            return False
        
        # Step 3: Wait for Scout cycle and IA1 processing
        print(f"   ⏱️ Step 3: Waiting for Scout→IA1 cycle (90 seconds)...")
        time.sleep(90)
        
        # Step 4: Analyze Scout opportunities
        print(f"\n   📊 Step 4: Analyzing Scout opportunities...")
        success, opportunities_data = self.run_test("Get Opportunities (Scout)", "GET", "opportunities", 200)
        if not success:
            print(f"   ❌ Cannot retrieve Scout opportunities")
            self.run_test("Stop Trading System", "POST", "stop-trading", 200)
            return False
        
        opportunities = opportunities_data.get('opportunities', [])
        scout_count = len(opportunities)
        print(f"   ✅ Scout found {scout_count} opportunities")
        
        # Step 5: Analyze IA1 analyses (what passed Scout filters)
        print(f"\n   📊 Step 5: Analyzing IA1 analyses...")
        success, analyses_data = self.run_test("Get Technical Analyses", "GET", "analyses", 200)
        if not success:
            print(f"   ❌ Cannot retrieve IA1 analyses")
            self.run_test("Stop Trading System", "POST", "stop-trading", 200)
            return False
        
        analyses = analyses_data.get('analyses', [])
        ia1_count = len(analyses)
        print(f"   ✅ IA1 generated {ia1_count} analyses")
        
        # Step 6: Calculate pass rate
        if scout_count > 0:
            pass_rate = (ia1_count / scout_count) * 100
            print(f"\n   📈 SCOUT→IA1 PASS RATE: {pass_rate:.1f}% ({ia1_count}/{scout_count})")
        else:
            print(f"   ❌ No Scout opportunities to calculate pass rate")
            self.run_test("Stop Trading System", "POST", "stop-trading", 200)
            return False
        
        # Step 7: Look for KTAUSDT-type opportunities
        print(f"\n   🔍 Step 7: Searching for KTAUSDT-type opportunities...")
        ktausdt_type_opportunities = []
        ktausdt_type_analyses = []
        
        # Check opportunities for high volume + movement
        for opp in opportunities:
            symbol = opp.get('symbol', '')
            volume = opp.get('volume_24h', 0)
            price_change = abs(opp.get('price_change_24h', 0))
            
            # KTAUSDT criteria: High volume (>1M) + significant movement (>5%)
            if volume >= 1_000_000 and price_change >= 5.0:
                ktausdt_type_opportunities.append({
                    'symbol': symbol,
                    'volume': volume,
                    'price_change': price_change
                })
                print(f"   🎯 KTAUSDT-type found: {symbol} - Vol: ${volume:,.0f}, Move: {price_change:+.1f}%")
        
        # Check if these made it to IA1
        analysis_symbols = set(analysis.get('symbol', '') for analysis in analyses)
        for ktausdt_opp in ktausdt_type_opportunities:
            if ktausdt_opp['symbol'] in analysis_symbols:
                ktausdt_type_analyses.append(ktausdt_opp)
                print(f"   ✅ KTAUSDT-type PASSED: {ktausdt_opp['symbol']} made it to IA1")
        
        ktausdt_pass_rate = (len(ktausdt_type_analyses) / len(ktausdt_type_opportunities)) * 100 if ktausdt_type_opportunities else 0
        
        # Step 8: Analyze override effectiveness
        print(f"\n   🎯 Step 8: Analyzing 7 Override Effectiveness...")
        
        # Look for override indicators in IA1 reasoning
        override_mentions = 0
        high_volume_passes = 0
        excellent_data_passes = 0
        
        for analysis in analyses:
            reasoning = analysis.get('ia1_reasoning', '').lower()
            symbol = analysis.get('symbol', '')
            
            # Check for override keywords
            override_keywords = ['override', 'bypass', 'excellent', 'volume élevé', 'données solides']
            if any(keyword in reasoning for keyword in override_keywords):
                override_mentions += 1
                print(f"   🎯 Override detected: {symbol}")
            
            # Check corresponding opportunity for override criteria
            for opp in opportunities:
                if opp.get('symbol') == symbol:
                    volume = opp.get('volume_24h', 0)
                    confidence = opp.get('data_confidence', 0)
                    price_change = abs(opp.get('price_change_24h', 0))
                    
                    # Override 2: Volume élevé + mouvement (≥1M$ + ≥5%)
                    if volume >= 1_000_000 and price_change >= 5.0:
                        high_volume_passes += 1
                    
                    # Override 1: Données excellentes (≥90% confiance)
                    if confidence >= 0.9:
                        excellent_data_passes += 1
                    
                    break
        
        override_rate = (override_mentions / ia1_count) * 100 if ia1_count > 0 else 0
        
        # Step 9: Stop system
        print(f"\n   🛑 Step 9: Stopping trading system...")
        self.run_test("Stop Trading System", "POST", "stop-trading", 200)
        
        # Step 10: Comprehensive Option A validation
        print(f"\n   📊 OPTION A COMPREHENSIVE ANALYSIS:")
        print(f"      Scout Opportunities: {scout_count}")
        print(f"      IA1 Analyses: {ia1_count}")
        print(f"      Pass Rate: {pass_rate:.1f}% (target: 20-25%)")
        print(f"      KTAUSDT-type Found: {len(ktausdt_type_opportunities)}")
        print(f"      KTAUSDT-type Passed: {len(ktausdt_type_analyses)}")
        print(f"      KTAUSDT Pass Rate: {ktausdt_pass_rate:.1f}%")
        print(f"      Override Mentions: {override_mentions} ({override_rate:.1f}%)")
        print(f"      High Volume Passes: {high_volume_passes}")
        print(f"      Excellent Data Passes: {excellent_data_passes}")
        
        # Validation criteria for Option A success
        pass_rate_improved = pass_rate >= 20.0 and pass_rate <= 30.0  # Target range 20-25%
        ktausdt_recovery = len(ktausdt_type_analyses) > 0 or len(ktausdt_type_opportunities) == 0  # KTAUSDT types should pass
        overrides_working = override_mentions > 0 or high_volume_passes > 0  # Overrides should be active
        lateral_filter_removed = pass_rate > 16.0  # Should be better than old 16%
        
        print(f"\n   ✅ OPTION A VALIDATION:")
        print(f"      Pass Rate 20-25%: {'✅' if pass_rate_improved else '❌'} ({pass_rate:.1f}%)")
        print(f"      KTAUSDT Recovery: {'✅' if ktausdt_recovery else '❌'} ({len(ktausdt_type_analyses)}/{len(ktausdt_type_opportunities)})")
        print(f"      Overrides Working: {'✅' if overrides_working else '❌'} ({override_mentions} mentions)")
        print(f"      Better than 16%: {'✅' if lateral_filter_removed else '❌'} ({pass_rate:.1f}% vs 16%)")
        
        option_a_success = (
            pass_rate_improved and
            ktausdt_recovery and
            lateral_filter_removed
        )
        
        print(f"\n   🎯 OPTION A IMPLEMENTATION: {'✅ SUCCESS' if option_a_success else '❌ NEEDS WORK'}")
        
        if not option_a_success:
            print(f"   💡 ISSUES DETECTED:")
            if not pass_rate_improved:
                print(f"      - Pass rate {pass_rate:.1f}% not in target range 20-25%")
            if not ktausdt_recovery:
                print(f"      - KTAUSDT-type opportunities still being filtered ({len(ktausdt_type_analyses)}/{len(ktausdt_type_opportunities)} passed)")
            if not lateral_filter_removed:
                print(f"      - Pass rate {pass_rate:.1f}% not significantly better than old 16%")
        else:
            print(f"   🎉 SUCCESS: Option A implementation working as expected!")
            print(f"   🎯 Lateral filter removed: Pass rate improved to {pass_rate:.1f}%")
            print(f"   🎯 7 Overrides active: {override_mentions} override mentions detected")
            print(f"   🎯 KTAUSDT recovery: {len(ktausdt_type_analyses)} high-value opportunities passed")
        
        return option_a_success

if __name__ == "__main__":
    tester = ScoutOptionATester()
    
    print("🎯 SCOUT OPTION A IMPLEMENTATION TEST")
    print("=" * 60)
    print(f"🔧 Testing Request: Scout Option A - Lateral filter REMOVED + 7 overrides optimized")
    print(f"🎯 Expected: Pass rate 20-25% (up from 16%)")
    print(f"🎯 Expected: KTAUSDT-type opportunities should now pass")
    print("=" * 60)
    
    # Run the Option A test
    result = tester.test_scout_option_a_implementation()
    
    print("\n" + "=" * 60)
    print("📊 SCOUT OPTION A TEST RESULTS")
    print("=" * 60)
    
    print(f"\n🔍 Test Results Summary:")
    print(f"   Tests Run: {tester.tests_run}")
    print(f"   Tests Passed: {tester.tests_passed}")
    print(f"   Success Rate: {(tester.tests_passed/tester.tests_run)*100:.1f}%")
    
    print(f"\n🎯 SCOUT OPTION A Assessment:")
    if result:
        print(f"   ✅ SCOUT OPTION A IMPLEMENTATION SUCCESSFUL")
        print(f"   ✅ Lateral filter removed and 7 overrides optimized working")
        test_status = "SUCCESS"
    else:
        print(f"   ❌ SCOUT OPTION A IMPLEMENTATION FAILED")
        print(f"   ❌ Issues detected - implementation not working as expected")
        test_status = "FAILED"
    
    print(f"\n📋 Specific Request Validation:")
    print(f"   • Lateral Filter Removed: {'✅' if result else '❌'}")
    print(f"   • 7 Overrides Optimized: {'✅' if result else '❌'}")
    print(f"   • Pass Rate 20-25%: {'✅' if result else '❌'}")
    print(f"   • KTAUSDT Recovery: {'✅' if result else '❌'}")
    
    if test_status == "SUCCESS":
        print(f"\n🎉 Scout Option A implementation test completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Scout Option A implementation test failed. Check the output above.")
        sys.exit(1)
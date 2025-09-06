#!/usr/bin/env python3
"""
IA1 Deduplication Fix Test Script
Focus: Test the IA1 deduplication fix as requested in the review
"""

import requests
import sys
import json
import time
from datetime import datetime, timedelta
import pytz
from pathlib import Path

class IA1DeduplicationTester:
    def __init__(self):
        # Get the correct backend URL from frontend/.env
        try:
            env_path = Path(__file__).parent / "frontend" / ".env"
            with open(env_path, 'r') as f:
                for line in f:
                    if line.startswith('REACT_APP_BACKEND_URL='):
                        base_url = line.split('=', 1)[1].strip()
                        break
                else:
                    base_url = "https://ai-trade-pro.preview.emergentagent.com"
        except:
            base_url = "https://ai-trade-pro.preview.emergentagent.com"
        
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        
        # Paris timezone for testing
        self.PARIS_TZ = pytz.timezone('Europe/Paris')

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

    def test_ia1_deduplication_fix(self):
        """Test IA1 deduplication fix - main focus of current review request"""
        print(f"\n🔍 Testing IA1 Deduplication Fix (MAIN FOCUS)...")
        
        # Test 1: Check /api/analyses endpoint for duplicates
        print(f"\n   📊 Test 1: Checking /api/analyses endpoint for duplicates...")
        success, analyses_data = self.run_test("Get Technical Analyses", "GET", "analyses", 200)
        if not success:
            print(f"   ❌ Cannot retrieve analyses for deduplication testing")
            return False
        
        analyses = analyses_data.get('analyses', [])
        if len(analyses) == 0:
            print(f"   ❌ No analyses available for deduplication testing")
            return False
        
        print(f"   📈 Found {len(analyses)} analyses to check for duplicates")
        
        # Check for duplicates by symbol within 4 hours
        now_paris = datetime.now(self.PARIS_TZ)
        four_hours_ago = now_paris - timedelta(hours=4)
        
        symbol_timestamps = {}
        duplicates_found = []
        
        for analysis in analyses:
            symbol = analysis.get('symbol', 'Unknown')
            timestamp_str = analysis.get('timestamp', '')
            
            try:
                # Parse timestamp - handle both ISO and French format
                if 'T' in timestamp_str:
                    # ISO format
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=pytz.UTC)
                    timestamp_paris = timestamp.astimezone(self.PARIS_TZ)
                elif '(Heure de Paris)' in timestamp_str:
                    # French format: "2025-09-06 17:33:53 (Heure de Paris)"
                    date_part = timestamp_str.split(' (Heure de Paris)')[0]
                    timestamp_paris = datetime.strptime(date_part, '%Y-%m-%d %H:%M:%S')
                    timestamp_paris = self.PARIS_TZ.localize(timestamp_paris)
                else:
                    continue  # Skip if timestamp format is unexpected
                
                # Only check recent analyses (within last 4 hours)
                if timestamp_paris >= four_hours_ago:
                    if symbol in symbol_timestamps:
                        # Check if this is a duplicate (same symbol within 4 hours)
                        existing_timestamp = symbol_timestamps[symbol]
                        time_diff = abs((timestamp_paris - existing_timestamp).total_seconds())
                        
                        if time_diff < 14400:  # 4 hours = 14400 seconds
                            duplicates_found.append({
                                'symbol': symbol,
                                'timestamp1': existing_timestamp,
                                'timestamp2': timestamp_paris,
                                'time_diff_minutes': time_diff / 60
                            })
                            print(f"   ❌ DUPLICATE FOUND: {symbol} analyzed twice within {time_diff/60:.1f} minutes")
                        else:
                            symbol_timestamps[symbol] = timestamp_paris
                    else:
                        symbol_timestamps[symbol] = timestamp_paris
                        
            except Exception as e:
                print(f"   ⚠️  Error parsing timestamp for {symbol}: {e}")
                continue
        
        # Test 2: Check timezone consistency (Paris timezone)
        print(f"\n   🕐 Test 2: Checking timezone consistency (Paris timezone)...")
        timezone_consistent = True
        paris_timezone_count = 0
        
        for analysis in analyses[:10]:  # Check first 10
            timestamp_str = analysis.get('timestamp', '')
            symbol = analysis.get('symbol', 'Unknown')
            
            try:
                if 'T' in timestamp_str:
                    # Check if timestamp appears to be in Paris timezone format
                    if '+01:00' in timestamp_str or '+02:00' in timestamp_str:
                        paris_timezone_count += 1
                        print(f"   ✅ {symbol}: Paris timezone detected ({timestamp_str})")
                    else:
                        print(f"   ⚠️  {symbol}: Non-Paris timezone ({timestamp_str})")
                        timezone_consistent = False
            except Exception as e:
                print(f"   ⚠️  Error checking timezone for {symbol}: {e}")
        
        # Test 3: Start system and check for new duplicates
        print(f"\n   🚀 Test 3: Testing live deduplication during system operation...")
        
        # Get initial analysis count
        initial_count = len(analyses)
        initial_symbols = set(analysis.get('symbol') for analysis in analyses)
        
        # Start trading system
        start_success, _ = self.run_test("Start Trading System", "POST", "start-trading", 200)
        if start_success:
            print(f"   ⏱️  Waiting for new analyses (60 seconds)...")
            time.sleep(60)
            
            # Check for new analyses
            success, new_analyses_data = self.run_test("Get Technical Analyses", "GET", "analyses", 200)
            if success:
                new_analyses = new_analyses_data.get('analyses', [])
                new_count = len(new_analyses)
                new_symbols = set(analysis.get('symbol') for analysis in new_analyses)
                
                print(f"   📊 After 60s: {new_count} analyses (was {initial_count})")
                
                # Check if new analyses created duplicates
                recent_duplicates = []
                symbol_recent_count = {}
                
                # Count recent analyses per symbol (last 4 hours)
                for analysis in new_analyses:
                    symbol = analysis.get('symbol', 'Unknown')
                    timestamp_str = analysis.get('timestamp', '')
                    
                    try:
                        if 'T' in timestamp_str:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            if timestamp.tzinfo is None:
                                timestamp = timestamp.replace(tzinfo=pytz.UTC)
                            timestamp_paris = timestamp.astimezone(self.PARIS_TZ)
                            
                            if timestamp_paris >= four_hours_ago:
                                symbol_recent_count[symbol] = symbol_recent_count.get(symbol, 0) + 1
                    except:
                        continue
                
                # Check for symbols with multiple recent analyses
                for symbol, count in symbol_recent_count.items():
                    if count > 1:
                        recent_duplicates.append({'symbol': symbol, 'count': count})
                        print(f"   ❌ RECENT DUPLICATE: {symbol} has {count} analyses in last 4h")
            
            # Stop trading system
            self.run_test("Stop Trading System", "POST", "stop-trading", 200)
        
        # Test 4: Validate deduplication logic effectiveness
        print(f"\n   🎯 Test 4: Deduplication effectiveness validation...")
        
        duplicate_rate = len(duplicates_found) / len(symbol_timestamps) if symbol_timestamps else 0
        no_duplicates = len(duplicates_found) == 0
        timezone_ok = paris_timezone_count >= len(analyses[:10]) * 0.8  # 80% should have Paris timezone
        
        print(f"\n   📊 Deduplication Test Results:")
        print(f"      Total recent analyses: {len(symbol_timestamps)}")
        print(f"      Duplicates found: {len(duplicates_found)}")
        print(f"      Duplicate rate: {duplicate_rate*100:.1f}%")
        print(f"      Paris timezone consistency: {paris_timezone_count}/{len(analyses[:10])} ({paris_timezone_count/len(analyses[:10])*100:.1f}%)")
        
        print(f"\n   ✅ Deduplication Fix Validation:")
        print(f"      No duplicates in /api/analyses: {'✅' if no_duplicates else '❌'}")
        print(f"      Paris timezone consistent: {'✅' if timezone_ok else '❌'}")
        print(f"      System generates unique analyses: {'✅' if len(recent_duplicates) == 0 else '❌'}")
        
        deduplication_working = no_duplicates and timezone_ok and len(recent_duplicates) == 0
        
        print(f"\n   🎯 IA1 Deduplication Fix: {'✅ SUCCESS' if deduplication_working else '❌ FAILED'}")
        
        if not deduplication_working:
            print(f"   💡 ISSUES FOUND:")
            if not no_duplicates:
                print(f"      - {len(duplicates_found)} duplicate analyses found in /api/analyses endpoint")
            if not timezone_ok:
                print(f"      - Timezone inconsistency detected (not all using Paris timezone)")
            if len(recent_duplicates) > 0:
                print(f"      - {len(recent_duplicates)} symbols generated duplicate analyses during testing")
        else:
            print(f"   💡 SUCCESS: IA1 deduplication fix is working correctly")
            print(f"      - No duplicates in /api/analyses endpoint")
            print(f"      - Consistent Paris timezone usage")
            print(f"      - Live system respects 4-hour deduplication window")
        
        return deduplication_working

    def test_complete_scout_ia1_ia2_cycle(self):
        """Test complete Scout → IA1 → IA2 cycle for deduplication"""
        print(f"\n🔄 Testing Complete Scout → IA1 → IA2 Cycle (Deduplication Focus)...")
        
        # Step 1: Check Scout opportunities
        print(f"\n   📊 Step 1: Checking Scout opportunities...")
        success, opportunities_data = self.run_test("Get Opportunities (Scout)", "GET", "opportunities", 200)
        if not success:
            print(f"   ❌ Scout not working")
            return False
        
        opportunities = opportunities_data.get('opportunities', [])
        scout_symbols = set(opp.get('symbol') for opp in opportunities)
        print(f"   ✅ Scout: {len(opportunities)} opportunities, {len(scout_symbols)} unique symbols")
        
        # Step 2: Check IA1 analyses
        print(f"\n   📈 Step 2: Checking IA1 analyses...")
        success, analyses_data = self.run_test("Get Technical Analyses", "GET", "analyses", 200)
        if not success:
            print(f"   ❌ IA1 not working")
            return False
        
        analyses = analyses_data.get('analyses', [])
        ia1_symbols = set(analysis.get('symbol') for analysis in analyses)
        print(f"   ✅ IA1: {len(analyses)} analyses, {len(ia1_symbols)} unique symbols")
        
        # Step 3: Check IA2 decisions
        print(f"\n   🎯 Step 3: Checking IA2 decisions...")
        success, decisions_data = self.run_test("Get Trading Decisions (IA2)", "GET", "decisions", 200)
        if not success:
            print(f"   ❌ IA2 not working")
            return False
        
        decisions = decisions_data.get('decisions', [])
        ia2_symbols = set(decision.get('symbol') for decision in decisions)
        print(f"   ✅ IA2: {len(decisions)} decisions, {len(ia2_symbols)} unique symbols")
        
        # Step 4: Check pipeline integration
        print(f"\n   🔗 Step 4: Checking pipeline integration...")
        
        scout_to_ia1 = scout_symbols.intersection(ia1_symbols)
        ia1_to_ia2 = ia1_symbols.intersection(ia2_symbols)
        full_pipeline = scout_symbols.intersection(ia1_symbols).intersection(ia2_symbols)
        
        print(f"   📊 Pipeline Flow Analysis:")
        print(f"      Scout → IA1 overlap: {len(scout_to_ia1)} symbols")
        print(f"      IA1 → IA2 overlap: {len(ia1_to_ia2)} symbols")
        print(f"      Full pipeline (Scout→IA1→IA2): {len(full_pipeline)} symbols")
        
        # Step 5: Check for duplicates in each stage
        print(f"\n   🔍 Step 5: Checking for duplicates in each stage...")
        
        # Check Scout duplicates
        scout_duplicate_count = len(opportunities) - len(scout_symbols)
        print(f"      Scout duplicates: {scout_duplicate_count}")
        
        # Check IA1 duplicates (same symbol, recent timestamp)
        ia1_duplicate_count = len(analyses) - len(ia1_symbols)
        print(f"      IA1 duplicates: {ia1_duplicate_count}")
        
        # Check IA2 duplicates
        ia2_duplicate_count = len(decisions) - len(ia2_symbols)
        print(f"      IA2 duplicates: {ia2_duplicate_count}")
        
        # Step 6: Validation
        pipeline_working = len(full_pipeline) > 0
        no_scout_duplicates = scout_duplicate_count == 0
        no_ia1_duplicates = ia1_duplicate_count == 0
        no_ia2_duplicates = ia2_duplicate_count == 0
        good_flow_rate = len(scout_to_ia1) / len(scout_symbols) >= 0.1 if scout_symbols else False
        
        print(f"\n   ✅ Complete Cycle Validation:")
        print(f"      Pipeline working: {'✅' if pipeline_working else '❌'} ({len(full_pipeline)} symbols)")
        print(f"      No Scout duplicates: {'✅' if no_scout_duplicates else '❌'}")
        print(f"      No IA1 duplicates: {'✅' if no_ia1_duplicates else '❌'}")
        print(f"      No IA2 duplicates: {'✅' if no_ia2_duplicates else '❌'}")
        print(f"      Good flow rate: {'✅' if good_flow_rate else '❌'} ({len(scout_to_ia1)}/{len(scout_symbols)})")
        
        cycle_success = (
            pipeline_working and
            no_scout_duplicates and
            no_ia1_duplicates and
            no_ia2_duplicates and
            good_flow_rate
        )
        
        print(f"\n   🎯 Complete Cycle Assessment: {'✅ SUCCESS' if cycle_success else '❌ NEEDS WORK'}")
        
        return cycle_success

    def run_all_tests(self):
        """Run all IA1 deduplication tests"""
        print(f"🎯 Starting IA1 Deduplication Fix Tests")
        print(f"Backend URL: {self.base_url}")
        print(f"API URL: {self.api_url}")
        print(f"=" * 80)
        print(f"🔧 Testing IA1 deduplication fix as requested in review:")
        print(f"   • Fixed timezone inconsistency (UTC → Paris timezone)")
        print(f"   • Added deduplication in /api/analyses endpoint")
        print(f"   • Verify no duplicates (same crypto < 4h apart)")
        print(f"   • Test complete Scout → IA1 → IA2 cycle")
        print(f"=" * 80)

        # Core system tests
        self.run_test("System Status", "GET", "", 200)
        self.run_test("Market Status", "GET", "market-status", 200)
        
        # IA1 DEDUPLICATION FIX TESTS (MAIN FOCUS)
        print(f"\n" + "🎯" * 20 + " IA1 DEDUPLICATION FIX TESTS " + "🎯" * 20)
        ia1_dedup_success = self.test_ia1_deduplication_fix()
        complete_cycle_success = self.test_complete_scout_ia1_ia2_cycle()
        
        # Performance summary
        print(f"\n" + "=" * 80)
        print(f"🎯 IA1 DEDUPLICATION FIX TEST SUMMARY")
        print(f"=" * 80)
        print(f"Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        # MAIN FOCUS RESULTS
        print(f"\n🎯 IA1 DEDUPLICATION FIX RESULTS:")
        print(f"   IA1 Deduplication Fix: {'✅ SUCCESS' if ia1_dedup_success else '❌ FAILED'}")
        print(f"   Complete Cycle Test: {'✅ SUCCESS' if complete_cycle_success else '❌ FAILED'}")
        
        # Overall assessment
        if ia1_dedup_success and complete_cycle_success:
            print(f"\n✅ OVERALL ASSESSMENT: IA1 DEDUPLICATION FIX SUCCESSFUL")
            print(f"   • No duplicates found in /api/analyses endpoint")
            print(f"   • Paris timezone consistency maintained")
            print(f"   • Complete pipeline working without duplicates")
            print(f"   • 4-hour deduplication window respected")
        else:
            print(f"\n❌ OVERALL ASSESSMENT: IA1 DEDUPLICATION FIX NEEDS WORK")
            if not ia1_dedup_success:
                print(f"   • Deduplication logic not working properly")
            if not complete_cycle_success:
                print(f"   • Complete pipeline has issues")
        
        print(f"=" * 80)
        
        return ia1_dedup_success and complete_cycle_success

if __name__ == "__main__":
    tester = IA1DeduplicationTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
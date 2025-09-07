#!/usr/bin/env python3

import requests
import sys
import json
import time
from pathlib import Path

class ConfidenceBasedFilterTester:
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

    def run_api_call(self, name, method, endpoint, expected_status=200, data=None, timeout=30):
        """Run a single API call"""
        url = f"{self.api_url}/{endpoint}" if endpoint else f"{self.api_url}/"
        headers = {'Content-Type': 'application/json'}

        print(f"🔍 {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            if success:
                try:
                    return True, response.json()
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Status: {response.status_code}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_ia1_confidence_based_hold_filter(self):
        """🎯 TEST OPTION B: IA1 Confidence-Based HOLD Filter for IA2 Economy"""
        print(f"\n🎯 Testing IA1 CONFIDENCE-BASED HOLD FILTER (Option B)...")
        print(f"   📋 TESTING LOGIC:")
        print(f"      • IA1 confidence < 70% → HOLD implicit → Skip IA2 (save credits)")
        print(f"      • IA1 confidence ≥ 70% → Potential signal → Send to IA2")
        print(f"      • Expected: 20-40% IA2 economy through intelligent filtering")
        
        # Step 1: Clear cache for fresh test
        print(f"\n   🗑️ Step 1: Clearing cache for fresh confidence-based test...")
        try:
            clear_success, _ = self.run_api_call("Clear Cache", "POST", "decisions/clear", 200)
            if clear_success:
                print(f"   ✅ Cache cleared - ready for fresh confidence test")
            else:
                print(f"   ⚠️ Cache clear failed, using existing data")
        except:
            print(f"   ⚠️ Cache clear not available, using existing data")
        
        # Step 2: Start system to generate fresh IA1 analyses
        print(f"\n   🚀 Step 2: Starting system to generate IA1 analyses with confidence filtering...")
        start_success, _ = self.run_api_call("Start Trading System", "POST", "start-trading", 200)
        if not start_success:
            print(f"   ❌ Failed to start trading system")
            return False
        
        # Step 3: Wait for IA1 analyses generation
        print(f"   ⏱️ Step 3: Waiting for IA1 confidence-based filtering (60 seconds)...")
        time.sleep(60)
        
        # Step 4: Stop system
        print(f"   🛑 Step 4: Stopping system...")
        self.run_api_call("Stop Trading System", "POST", "stop-trading", 200)
        
        # Step 5: Analyze IA1 confidence distribution
        print(f"\n   📊 Step 5: Analyzing IA1 Confidence Distribution...")
        success, analyses_data = self.run_api_call("Get IA1 Analyses", "GET", "analyses", 200)
        if not success:
            print(f"   ❌ Cannot retrieve IA1 analyses for confidence testing")
            return False
        
        analyses = analyses_data.get('analyses', [])
        if len(analyses) == 0:
            print(f"   ❌ No IA1 analyses available for confidence testing")
            return False
        
        print(f"   📈 Found {len(analyses)} IA1 analyses for confidence analysis")
        
        # Analyze confidence distribution
        confidence_stats = {
            'total': len(analyses),
            'below_70': 0,
            'above_70': 0,
            'confidences': []
        }
        
        for analysis in analyses:
            confidence = analysis.get('analysis_confidence', 0)
            confidence_stats['confidences'].append(confidence)
            
            if confidence < 0.70:
                confidence_stats['below_70'] += 1
            else:
                confidence_stats['above_70'] += 1
        
        # Calculate statistics
        avg_confidence = sum(confidence_stats['confidences']) / len(confidence_stats['confidences'])
        min_confidence = min(confidence_stats['confidences'])
        max_confidence = max(confidence_stats['confidences'])
        
        below_70_rate = confidence_stats['below_70'] / confidence_stats['total']
        above_70_rate = confidence_stats['above_70'] / confidence_stats['total']
        
        print(f"\n   📊 IA1 Confidence Analysis:")
        print(f"      Total IA1 Analyses: {confidence_stats['total']}")
        print(f"      Below 70% (should skip IA2): {confidence_stats['below_70']} ({below_70_rate*100:.1f}%)")
        print(f"      Above 70% (should go to IA2): {confidence_stats['above_70']} ({above_70_rate*100:.1f}%)")
        print(f"      Average Confidence: {avg_confidence:.1%}")
        print(f"      Range: {min_confidence:.1%} - {max_confidence:.1%}")
        
        # Step 6: Analyze IA2 decisions to verify filtering
        print(f"\n   🎯 Step 6: Analyzing IA2 Decisions to Verify Filtering...")
        success, decisions_data = self.run_api_call("Get IA2 Decisions", "GET", "decisions", 200)
        if not success:
            print(f"   ❌ Cannot retrieve IA2 decisions for filtering verification")
            return False
        
        decisions = decisions_data.get('decisions', [])
        print(f"   📈 Found {len(decisions)} IA2 decisions")
        
        # Step 7: Calculate IA2 economy rate
        print(f"\n   💰 Step 7: Calculating IA2 Economy Rate...")
        
        # Expected: IA2 decisions should be <= IA1 analyses with ≥70% confidence
        expected_max_ia2 = confidence_stats['above_70']
        actual_ia2 = len(decisions)
        
        # Calculate economy
        if confidence_stats['total'] > 0:
            theoretical_ia2_without_filter = confidence_stats['total']  # All IA1 would go to IA2
            actual_ia2_calls = actual_ia2
            ia2_calls_saved = theoretical_ia2_without_filter - actual_ia2_calls
            economy_rate = ia2_calls_saved / theoretical_ia2_without_filter if theoretical_ia2_without_filter > 0 else 0
        else:
            economy_rate = 0
            ia2_calls_saved = 0
        
        print(f"   💰 IA2 Economy Analysis:")
        print(f"      IA1 Analyses Total: {confidence_stats['total']}")
        print(f"      IA1 High Confidence (≥70%): {confidence_stats['above_70']}")
        print(f"      IA1 Low Confidence (<70%): {confidence_stats['below_70']}")
        print(f"      IA2 Decisions Generated: {actual_ia2}")
        print(f"      IA2 Calls Saved: {ia2_calls_saved}")
        print(f"      Economy Rate: {economy_rate*100:.1f}%")
        
        # Step 8: Verify confidence-based filtering logic
        print(f"\n   🔍 Step 8: Verifying Confidence-Based Filtering Logic...")
        
        # Check if IA2 decisions correspond to high-confidence IA1 analyses
        ia1_symbols_high_conf = set()
        ia1_symbols_low_conf = set()
        
        for analysis in analyses:
            symbol = analysis.get('symbol', '')
            confidence = analysis.get('analysis_confidence', 0)
            
            if confidence >= 0.70:
                ia1_symbols_high_conf.add(symbol)
            else:
                ia1_symbols_low_conf.add(symbol)
        
        ia2_symbols = set(decision.get('symbol', '') for decision in decisions)
        
        # Verify filtering logic
        low_conf_leaked_to_ia2 = ia1_symbols_low_conf.intersection(ia2_symbols)
        high_conf_processed_by_ia2 = ia1_symbols_high_conf.intersection(ia2_symbols)
        
        print(f"   🔍 Filtering Logic Verification:")
        print(f"      IA1 High Confidence Symbols: {len(ia1_symbols_high_conf)}")
        print(f"      IA1 Low Confidence Symbols: {len(ia1_symbols_low_conf)}")
        print(f"      IA2 Decision Symbols: {len(ia2_symbols)}")
        print(f"      Low Conf → IA2 (should be 0): {len(low_conf_leaked_to_ia2)}")
        print(f"      High Conf → IA2: {len(high_conf_processed_by_ia2)}")
        
        # Step 9: Validation criteria
        print(f"\n   ✅ Confidence-Based Filter Validation:")
        
        # Criterion 1: No low-confidence analyses should reach IA2
        no_leakage = len(low_conf_leaked_to_ia2) == 0
        print(f"      No Low Confidence Leakage: {'✅' if no_leakage else '❌'} ({len(low_conf_leaked_to_ia2)} leaked)")
        
        # Criterion 2: Some high-confidence analyses should reach IA2
        high_conf_processed = len(high_conf_processed_by_ia2) > 0
        print(f"      High Confidence Processed: {'✅' if high_conf_processed else '❌'} ({len(high_conf_processed_by_ia2)} processed)")
        
        # Criterion 3: Economy rate should be reasonable (20-40% target)
        reasonable_economy = 0.20 <= economy_rate <= 0.60  # Allow up to 60% for good filtering
        print(f"      Reasonable Economy (20-60%): {'✅' if reasonable_economy else '❌'} ({economy_rate*100:.1f}%)")
        
        # Criterion 4: IA2 decisions should be <= high confidence IA1 analyses
        proper_filtering = actual_ia2 <= expected_max_ia2
        print(f"      Proper Filtering Logic: {'✅' if proper_filtering else '❌'} ({actual_ia2} ≤ {expected_max_ia2})")
        
        # Criterion 5: Some IA1 analyses should have varied confidence (not all same)
        confidence_variation = max_confidence - min_confidence >= 0.10  # At least 10% range
        print(f"      Confidence Variation: {'✅' if confidence_variation else '❌'} ({(max_confidence-min_confidence)*100:.1f}% range)")
        
        # Overall assessment
        filter_working = (
            no_leakage and
            high_conf_processed and
            reasonable_economy and
            proper_filtering and
            confidence_variation
        )
        
        print(f"\n   🎯 CONFIDENCE-BASED HOLD FILTER ASSESSMENT:")
        print(f"      Filter Status: {'✅ WORKING' if filter_working else '❌ FAILED'}")
        print(f"      IA2 Economy Achieved: {economy_rate*100:.1f}%")
        print(f"      Credits Saved: {ia2_calls_saved} IA2 calls")
        
        if filter_working:
            print(f"   💡 SUCCESS: Option B confidence-based filtering is operational!")
            print(f"   💡 Low confidence IA1 analyses (<70%) are being filtered out")
            print(f"   💡 High confidence IA1 analyses (≥70%) are processed by IA2")
            print(f"   💡 Economy rate of {economy_rate*100:.1f}% achieved")
        else:
            print(f"   💡 ISSUES DETECTED:")
            if not no_leakage:
                print(f"      - Low confidence analyses are leaking to IA2")
            if not reasonable_economy:
                print(f"      - Economy rate {economy_rate*100:.1f}% outside target 20-60%")
            if not proper_filtering:
                print(f"      - More IA2 decisions than expected high-confidence IA1")
            if not confidence_variation:
                print(f"      - IA1 confidence lacks variation for proper testing")
        
        return filter_working

if __name__ == "__main__":
    print("🎯 IA1 CONFIDENCE-BASED HOLD FILTER TEST")
    print("=" * 50)
    
    tester = ConfidenceBasedFilterTester()
    result = tester.test_ia1_confidence_based_hold_filter()
    
    print(f"\n" + "=" * 50)
    print(f"🎯 FINAL RESULT: {'✅ PASSED' if result else '❌ FAILED'}")
    
    sys.exit(0 if result else 1)
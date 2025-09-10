#!/usr/bin/env python3
"""
Quick test for IA1→IA2 Conditional Filtering Logic
"""

import requests
import json

def test_conditional_filtering():
    api_url = "https://bingx-fusion.preview.emergentagent.com/api"
    
    print("🔍 Testing IA1→IA2 Conditional Filtering Logic")
    print("=" * 60)
    
    # Get current analyses
    try:
        response = requests.get(f"{api_url}/analyses", timeout=30)
        analyses = response.json().get('analyses', [])
        print(f"📊 Total IA1 analyses: {len(analyses)}")
    except Exception as e:
        print(f"❌ Error getting analyses: {e}")
        return
    
    # Get current decisions
    try:
        response = requests.get(f"{api_url}/decisions", timeout=30)
        decisions = response.json().get('decisions', [])
        decision_symbols = [d.get('symbol') for d in decisions]
        print(f"📊 Total IA2 decisions: {len(decisions)}")
    except Exception as e:
        print(f"❌ Error getting decisions: {e}")
        return
    
    print("\n🎯 CONDITIONAL FILTERING ANALYSIS:")
    print("-" * 60)
    
    voie1_passed = 0
    voie1_total = 0
    voie2_passed = 0
    voie2_total = 0
    blocked_correctly = 0
    blocked_total = 0
    
    ai16zusdt_found = False
    
    for analysis in analyses:
        symbol = analysis.get('symbol', '')
        confidence = analysis.get('analysis_confidence', 0)
        signal = analysis.get('ia1_signal', 'hold').lower()
        rr = analysis.get('risk_reward_ratio', 0)
        has_ia2_decision = symbol in decision_symbols
        
        # Check AI16ZUSDT specific case
        if symbol == "AI16ZUSDT":
            ai16zusdt_found = True
            print(f"\n🎯 AI16ZUSDT CASE:")
            print(f"   Signal: {signal.upper()}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   RR: {rr:.2f}:1")
            
            meets_voie1 = signal in ['long', 'short'] and confidence >= 0.70
            meets_voie2 = rr >= 2.0
            should_pass = meets_voie1 or meets_voie2
            
            print(f"   VOIE 1 criteria: {meets_voie1} (LONG/SHORT + Conf≥70%)")
            print(f"   VOIE 2 criteria: {meets_voie2} (RR≥2.0)")
            print(f"   Should pass to IA2: {should_pass}")
            print(f"   Actually passed: {has_ia2_decision}")
            
            if should_pass and has_ia2_decision:
                print(f"   ✅ CORRECT: Passed as expected")
            elif should_pass and not has_ia2_decision:
                print(f"   ❌ ERROR: Should pass but didn't")
            elif not should_pass and has_ia2_decision:
                print(f"   ❌ ERROR: Shouldn't pass but did")
            else:
                print(f"   ✅ CORRECT: Blocked as expected")
        
        # VOIE 1 analysis: LONG/SHORT + Confidence ≥ 70%
        if signal in ['long', 'short'] and confidence >= 0.70:
            voie1_total += 1
            if has_ia2_decision:
                voie1_passed += 1
                print(f"✅ VOIE 1 PASS: {symbol} ({signal.upper()}, {confidence:.1%})")
            else:
                print(f"❌ VOIE 1 FAIL: {symbol} ({signal.upper()}, {confidence:.1%}) - Should pass but didn't")
        
        # VOIE 2 analysis: RR ≥ 2.0
        elif rr >= 2.0:
            voie2_total += 1
            if has_ia2_decision:
                voie2_passed += 1
                print(f"✅ VOIE 2 PASS: {symbol} (RR={rr:.2f}:1)")
            else:
                print(f"❌ VOIE 2 FAIL: {symbol} (RR={rr:.2f}:1) - Should pass but didn't")
        
        # Should be blocked
        else:
            blocked_total += 1
            if not has_ia2_decision:
                blocked_correctly += 1
                print(f"✅ BLOCKED: {symbol} ({signal.upper()}, {confidence:.1%}, RR={rr:.2f}) - Correctly blocked")
            else:
                print(f"❌ LEAKED: {symbol} ({signal.upper()}, {confidence:.1%}, RR={rr:.2f}) - Should be blocked but passed")
    
    print("\n📊 SUMMARY:")
    print("-" * 60)
    print(f"AI16ZUSDT found: {ai16zusdt_found}")
    print(f"VOIE 1 success rate: {voie1_passed}/{voie1_total} ({(voie1_passed/voie1_total*100) if voie1_total > 0 else 0:.1f}%)")
    print(f"VOIE 2 success rate: {voie2_passed}/{voie2_total} ({(voie2_passed/voie2_total*100) if voie2_total > 0 else 0:.1f}%)")
    print(f"Blocking success rate: {blocked_correctly}/{blocked_total} ({(blocked_correctly/blocked_total*100) if blocked_total > 0 else 100:.1f}%)")
    
    # Overall assessment
    total_correct = voie1_passed + voie2_passed + blocked_correctly
    total_cases = voie1_total + voie2_total + blocked_total
    overall_success = (total_correct / total_cases * 100) if total_cases > 0 else 0
    
    print(f"\n🏆 OVERALL SUCCESS RATE: {total_correct}/{total_cases} ({overall_success:.1f}%)")
    
    if overall_success >= 80:
        print("✅ CONDITIONAL FILTERING WORKING WELL")
    elif overall_success >= 60:
        print("⚠️ CONDITIONAL FILTERING PARTIALLY WORKING")
    else:
        print("❌ CONDITIONAL FILTERING NEEDS FIXES")

if __name__ == "__main__":
    test_conditional_filtering()
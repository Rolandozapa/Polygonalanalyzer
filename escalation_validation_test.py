#!/usr/bin/env python3
"""
Quick Escalation Validation Test
Verify that the IA1 to IA2 escalation system is working after the fix
"""

import requests
import json
import time
from pymongo import MongoClient

def test_escalation_fix():
    """Test that escalation system is working after the fix"""
    api_url = "https://dualai-trader.preview.emergentagent.com/api"
    
    print("🔍 Testing IA1 to IA2 Escalation System Fix...")
    
    # Connect to database to check decisions
    try:
        mongo_client = MongoClient("mongodb://localhost:27017")
        db = mongo_client["myapp"]
        
        # Get initial count
        initial_decisions = db.trading_decisions.count_documents({})
        print(f"📊 Initial IA2 decisions in database: {initial_decisions}")
        
        # Test escalation
        print("🚀 Running IA1 cycle to test escalation...")
        response = requests.post(
            f"{api_url}/run-ia1-cycle",
            json={"symbol": "SOLUSDT"},
            timeout=90
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Response received")
            print(f"   Success: {data.get('success', False)}")
            print(f"   Escalated to IA2: {data.get('escalated_to_ia2', False)}")
            
            if data.get('escalated_to_ia2'):
                ia2_decision = data.get('ia2_decision', {})
                print(f"   IA2 Decision: {ia2_decision.get('signal', 'N/A')} with {ia2_decision.get('confidence', 0):.1%} confidence")
                
                # Check database
                time.sleep(2)
                final_decisions = db.trading_decisions.count_documents({})
                new_decisions = final_decisions - initial_decisions
                print(f"📊 Final IA2 decisions in database: {final_decisions}")
                print(f"📊 New decisions added: {new_decisions}")
                
                if new_decisions > 0:
                    print("✅ ESCALATION SYSTEM WORKING: IA2 decision saved to database!")
                    return True
                else:
                    print("⚠️ Escalation occurred but no new decision in database")
                    return False
            else:
                analysis = data.get('analysis', {})
                confidence = analysis.get('confidence', 0)
                rr_ratio = analysis.get('risk_reward_ratio', 0)
                signal = analysis.get('recommendation', 'hold')
                
                print(f"   IA1 Analysis: {signal} signal, {confidence:.1%} confidence, RR={rr_ratio:.2f}")
                print("⚪ No escalation occurred - checking criteria...")
                
                # Check if it should have escalated
                should_escalate = (
                    (signal.lower() in ['long', 'short'] and confidence >= 0.70) or
                    (rr_ratio >= 2.0) or
                    (signal.lower() in ['long', 'short'] and confidence >= 0.95)
                )
                
                if should_escalate:
                    print("❌ Should have escalated but didn't - escalation logic issue")
                    return False
                else:
                    print("✅ Correctly did not escalate - criteria not met")
                    return True
        else:
            print(f"❌ API call failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_escalation_fix()
    if success:
        print("\n🎉 ESCALATION SYSTEM FIX VALIDATED!")
    else:
        print("\n❌ ESCALATION SYSTEM STILL HAS ISSUES")
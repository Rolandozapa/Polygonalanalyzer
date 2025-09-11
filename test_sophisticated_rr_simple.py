#!/usr/bin/env python3
"""
Simple test to verify sophisticated RR analysis system is working
"""

import requests
import json
import time
import subprocess

def test_sophisticated_rr_system():
    """Test the sophisticated RR analysis system"""
    
    print("🚀 Testing Sophisticated RR Analysis System")
    print("=" * 60)
    
    # Test 1: Check if methods are implemented in code
    print("\n🔍 TEST 1: Check method implementation")
    try:
        with open('/app/backend/server.py', 'r') as f:
            code = f.read()
        
        methods = [
            'calculate_neutral_risk_reward',
            'calculate_composite_rr', 
            'evaluate_sophisticated_risk_level'
        ]
        
        for method in methods:
            if f'def {method}' in code:
                print(f"   ✅ {method} - IMPLEMENTED")
            else:
                print(f"   ❌ {method} - MISSING")
                
        # Check for usage patterns
        patterns = [
            '🧠 SOPHISTICATED ANALYSIS',
            'composite_rr_data',
            'sophisticated_risk_level'
        ]
        
        for pattern in patterns:
            if pattern in code:
                print(f"   ✅ Usage pattern '{pattern}' - FOUND")
            else:
                print(f"   ❌ Usage pattern '{pattern}' - MISSING")
                
    except Exception as e:
        print(f"   ❌ Error checking code: {e}")
    
    # Test 2: Check recent backend logs for sophisticated RR patterns
    print("\n🔍 TEST 2: Check backend logs for sophisticated RR activity")
    try:
        # Get recent logs
        result = subprocess.run(
            ["tail", "-n", "2000", "/var/log/supervisor/backend.out.log"],
            capture_output=True,
            text=True,
            timeout=10
        )
        logs = result.stdout
        
        patterns = [
            "🧠 SOPHISTICATED ANALYSIS",
            "📊 Composite RR:",
            "📊 Neutral RR:",
            "🎯 Sophisticated Risk Level:",
            "⚠️ SIGNIFICANT RR DIVERGENCE"
        ]
        
        total_occurrences = 0
        for pattern in patterns:
            count = logs.count(pattern)
            total_occurrences += count
            if count > 0:
                print(f"   ✅ '{pattern}' - {count} occurrences")
            else:
                print(f"   ⚪ '{pattern}' - 0 occurrences")
        
        print(f"   📊 Total sophisticated RR log entries: {total_occurrences}")
        
    except Exception as e:
        print(f"   ❌ Error checking logs: {e}")
    
    # Test 3: Check API endpoints
    print("\n🔍 TEST 3: Check API endpoints")
    try:
        # Check decisions endpoint
        response = requests.get("https://dual-ai-trader-4.preview.emergentagent.com/api/decisions", timeout=10)
        if response.status_code == 200:
            data = response.json()
            decisions = data.get('decisions', [])
            print(f"   ✅ Decisions API - {len(decisions)} decisions available")
            
            # Check recent decisions for sophisticated RR fields
            sophisticated_count = 0
            risk_level_count = 0
            
            for decision in decisions[:5]:  # Check first 5
                # Check for risk_level field
                if decision.get('risk_level') in ['LOW', 'MEDIUM', 'HIGH']:
                    risk_level_count += 1
                
                # Check for sophisticated RR in decision_logic
                decision_logic = decision.get('decision_logic', {})
                if isinstance(decision_logic, dict) and decision_logic.get('sophisticated_rr_analysis'):
                    sophisticated_count += 1
            
            print(f"   📊 Decisions with risk_level: {risk_level_count}/5")
            print(f"   📊 Decisions with sophisticated RR: {sophisticated_count}/5")
            
        else:
            print(f"   ❌ Decisions API - HTTP {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error checking API: {e}")
    
    # Test 4: Trigger new analysis and check
    print("\n🔍 TEST 4: Trigger trading and check for sophisticated RR")
    try:
        # Trigger trading
        response = requests.post("https://dual-ai-trader-4.preview.emergentagent.com/api/start-trading", 
                               json={}, timeout=30)
        if response.status_code == 200:
            print("   ✅ Trading triggered successfully")
            
            # Wait a bit for processing
            print("   ⏳ Waiting 20 seconds for processing...")
            time.sleep(20)
            
            # Check logs again for new sophisticated RR activity
            result = subprocess.run(
                ["tail", "-n", "500", "/var/log/supervisor/backend.out.log"],
                capture_output=True,
                text=True,
                timeout=10
            )
            recent_logs = result.stdout
            
            sophisticated_activity = recent_logs.count("🧠 SOPHISTICATED ANALYSIS")
            composite_rr_activity = recent_logs.count("📊 Composite RR:")
            
            print(f"   📊 Recent sophisticated analysis activity: {sophisticated_activity}")
            print(f"   📊 Recent composite RR activity: {composite_rr_activity}")
            
            if sophisticated_activity > 0 or composite_rr_activity > 0:
                print("   ✅ Sophisticated RR analysis is ACTIVE")
            else:
                print("   ⚠️ No recent sophisticated RR analysis detected")
                
        else:
            print(f"   ❌ Trading trigger failed - HTTP {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error triggering trading: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Sophisticated RR Analysis System Test Complete")

if __name__ == "__main__":
    test_sophisticated_rr_system()
#!/usr/bin/env python3
"""
Quick Test for BingX Balance and IA2 Confidence Fixes
Focused testing with shorter timeouts
"""

import requests
import json
import time
from pathlib import Path

def test_balance_fix():
    """Test if balance shows $250 instead of $0"""
    print("🎯 Testing Enhanced Balance Fix...")
    
    try:
        # Use internal URL for faster response
        response = requests.get("http://localhost:8001/api/market-status", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Market status response received")
            
            # Search for balance fields
            balance_found = False
            balance_value = 0.0
            
            # Check common balance field names
            balance_fields = ['balance', 'bingx_balance', 'account_balance', 'available_balance', 'total_balance']
            
            for field in balance_fields:
                if field in data:
                    balance_value = data[field]
                    balance_found = True
                    print(f"📊 Found {field}: ${balance_value}")
                    break
            
            # Check nested account_info
            if not balance_found and 'account_info' in data:
                account_info = data['account_info']
                for field in balance_fields:
                    if field in account_info:
                        balance_value = account_info[field]
                        balance_found = True
                        print(f"📊 Found account_info.{field}: ${balance_value}")
                        break
            
            if balance_found:
                if balance_value == 250.0:
                    print(f"✅ BALANCE FIX SUCCESS: Shows $250.00")
                    return True
                elif balance_value == 0.0:
                    print(f"❌ BALANCE FIX FAILED: Still shows $0.00")
                    return False
                else:
                    print(f"⚠️ BALANCE FIX PARTIAL: Shows ${balance_value} (expected $250)")
                    return balance_value > 0
            else:
                print(f"❌ No balance field found in response")
                # Print response structure for debugging
                print(f"Response keys: {list(data.keys())}")
                return False
        else:
            print(f"❌ Market status failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Balance test error: {e}")
        return False

def test_confidence_variation():
    """Test if IA2 confidence varies by symbol instead of uniform 76%"""
    print("\n🎯 Testing IA2 Confidence Variation...")
    
    try:
        # Get decisions
        response = requests.get("http://localhost:8001/api/decisions", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            decisions = data.get('decisions', [])
            
            if len(decisions) == 0:
                print(f"❌ No decisions available for testing")
                return False
            
            print(f"📊 Analyzing {len(decisions)} decisions...")
            
            # Collect confidence values
            confidences = []
            confidence_by_symbol = {}
            
            for decision in decisions:
                symbol = decision.get('symbol', 'Unknown')
                confidence = decision.get('confidence', 0.0)
                
                confidences.append(confidence)
                
                if symbol not in confidence_by_symbol:
                    confidence_by_symbol[symbol] = []
                confidence_by_symbol[symbol].append(confidence)
            
            # Analyze variation
            unique_confidences = list(set(confidences))
            avg_confidence = sum(confidences) / len(confidences)
            min_confidence = min(confidences)
            max_confidence = max(confidences)
            confidence_range = max_confidence - min_confidence
            
            print(f"📊 Confidence Statistics:")
            print(f"   Unique values: {len(unique_confidences)}")
            print(f"   Average: {avg_confidence:.3f}")
            print(f"   Range: {min_confidence:.3f} - {max_confidence:.3f} (span: {confidence_range:.3f})")
            
            # Check for uniform 76% issue
            uniform_76_count = sum(1 for c in confidences if abs(c - 0.76) < 0.001)
            uniform_76_rate = uniform_76_count / len(confidences)
            
            print(f"🔍 Uniform 76% Check:")
            print(f"   Decisions at 76%: {uniform_76_count}/{len(confidences)} ({uniform_76_rate*100:.1f}%)")
            
            # Show confidence by symbol (first 5)
            print(f"📋 Confidence by Symbol:")
            for i, (symbol, confs) in enumerate(list(confidence_by_symbol.items())[:5]):
                symbol_avg = sum(confs) / len(confs)
                print(f"   {symbol}: {symbol_avg:.3f} (n={len(confs)})")
            
            # Validation
            has_variation = len(unique_confidences) > 1
            significant_range = confidence_range >= 0.05  # At least 5% range
            not_uniform_76 = uniform_76_rate < 0.8  # Less than 80% at 76%
            maintains_minimum = min_confidence >= 0.50  # Maintains 50% minimum
            
            print(f"✅ Validation Results:")
            print(f"   Has variation: {'✅' if has_variation else '❌'}")
            print(f"   Significant range (≥5%): {'✅' if significant_range else '❌'}")
            print(f"   Not uniform 76% (<80%): {'✅' if not_uniform_76 else '❌'}")
            print(f"   Maintains 50% minimum: {'✅' if maintains_minimum else '❌'}")
            
            variation_working = (
                has_variation and
                significant_range and
                not_uniform_76 and
                maintains_minimum
            )
            
            if variation_working:
                print(f"✅ CONFIDENCE VARIATION SUCCESS: Real variation detected")
                return True
            else:
                print(f"❌ CONFIDENCE VARIATION FAILED: Still showing uniform behavior")
                return False
                
        else:
            print(f"❌ Decisions endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Confidence test error: {e}")
        return False

def main():
    """Run quick fix tests"""
    print("🚀 Quick Fix Test Suite")
    print("="*50)
    
    # Test 1: Balance Fix
    balance_success = test_balance_fix()
    
    # Test 2: Confidence Variation Fix
    confidence_success = test_confidence_variation()
    
    print("\n" + "="*50)
    print("📊 FINAL RESULTS:")
    print(f"   Balance Fix: {'✅ PASS' if balance_success else '❌ FAIL'}")
    print(f"   Confidence Variation: {'✅ PASS' if confidence_success else '❌ FAIL'}")
    
    overall_success = balance_success and confidence_success
    
    if overall_success:
        print(f"✅ OVERALL: Both fixes are working!")
    else:
        print(f"❌ OVERALL: One or both fixes need attention")
        
        if not balance_success:
            print(f"   Issue: Balance not showing $250")
        if not confidence_success:
            print(f"   Issue: Confidence still uniform instead of varied")
    
    return overall_success

if __name__ == "__main__":
    main()
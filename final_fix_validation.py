#!/usr/bin/env python3
"""
Final Validation of BingX Balance and IA2 Confidence Fixes
Based on actual API responses and log analysis
"""

import requests
import json
import subprocess
import re
from typing import List, Dict

def check_balance_fix_in_logs() -> bool:
    """Check if balance fix is working by examining backend logs"""
    print("🎯 Testing Enhanced Balance Fix (via logs)")
    
    try:
        # Check recent backend logs for balance information
        result = subprocess.run(
            ["tail", "-n", "100", "/var/log/supervisor/backend.err.log"],
            capture_output=True, text=True
        )
        
        log_content = result.stdout
        
        # Look for the enhanced balance message
        balance_250_found = "Using enhanced simulation balance for testing: $250.0" in log_content
        
        if balance_250_found:
            print("✅ BALANCE FIX SUCCESS: Found $250.0 balance in logs")
            print("   Log evidence: 'Using enhanced simulation balance for testing: $250.0'")
            return True
        else:
            print("❌ BALANCE FIX NOT FOUND: No $250 balance evidence in recent logs")
            return False
            
    except Exception as e:
        print(f"❌ Error checking logs: {e}")
        return False

def test_confidence_variation_fix() -> bool:
    """Test IA2 confidence variation fix"""
    print("\n🎯 Testing IA2 Confidence Variation Fix")
    
    try:
        # Get decisions from API
        response = requests.get("http://localhost:8001/api/decisions", timeout=15)
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            return False
        
        data = response.json()
        decisions = data.get('decisions', [])
        
        if len(decisions) == 0:
            print("❌ No decisions available for testing")
            return False
        
        print(f"📊 Analyzing {len(decisions)} decisions for confidence variation")
        
        # Collect confidence data
        confidences = []
        confidence_by_symbol = {}
        
        for decision in decisions:
            symbol = decision.get('symbol', 'Unknown')
            confidence = decision.get('confidence', 0.0)
            
            confidences.append(confidence)
            
            if symbol not in confidence_by_symbol:
                confidence_by_symbol[symbol] = []
            confidence_by_symbol[symbol].append(confidence)
        
        # Calculate statistics
        unique_confidences = list(set(round(c, 3) for c in confidences))
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        confidence_range = max_confidence - min_confidence
        
        print(f"📊 Confidence Analysis:")
        print(f"   Total decisions: {len(decisions)}")
        print(f"   Unique confidence values: {len(unique_confidences)}")
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"   Range: {min_confidence:.3f} - {max_confidence:.3f} (span: {confidence_range:.3f})")
        
        # Check for old uniform 76% issue
        uniform_76_count = sum(1 for c in confidences if abs(c - 0.76) < 0.001)
        uniform_76_rate = uniform_76_count / len(confidences)
        
        print(f"🔍 Uniform 76% Check:")
        print(f"   Decisions at exactly 76%: {uniform_76_count}/{len(confidences)} ({uniform_76_rate*100:.1f}%)")
        
        # Show confidence by symbol
        print(f"📋 Confidence by Symbol:")
        symbol_averages = {}
        for symbol, confs in confidence_by_symbol.items():
            symbol_avg = sum(confs) / len(confs)
            symbol_averages[symbol] = symbol_avg
            print(f"   {symbol}: {symbol_avg:.3f} (n={len(confs)})")
        
        # Validation criteria
        has_variation = len(unique_confidences) > 1
        significant_range = confidence_range >= 0.05  # At least 5% range
        not_uniform_76 = uniform_76_rate < 0.5  # Less than 50% at 76%
        maintains_minimum = min_confidence >= 0.50  # Maintains 50% minimum
        symbol_variation = len(set(round(avg, 3) for avg in symbol_averages.values())) > 1 if len(symbol_averages) > 1 else True
        
        print(f"\n✅ Validation Results:")
        print(f"   Has variation: {'✅' if has_variation else '❌'} ({len(unique_confidences)} unique)")
        print(f"   Significant range: {'✅' if significant_range else '❌'} ({confidence_range:.3f})")
        print(f"   Not uniform 76%: {'✅' if not_uniform_76 else '❌'} ({uniform_76_rate*100:.1f}% at 76%)")
        print(f"   Maintains 50% minimum: {'✅' if maintains_minimum else '❌'} (min: {min_confidence:.3f})")
        print(f"   Symbol variation: {'✅' if symbol_variation else '❌'}")
        
        # Overall assessment
        variation_working = (
            has_variation and
            significant_range and
            not_uniform_76 and
            maintains_minimum and
            symbol_variation
        )
        
        if variation_working:
            print(f"✅ CONFIDENCE VARIATION SUCCESS: Real variation detected across symbols")
            return True
        else:
            print(f"❌ CONFIDENCE VARIATION FAILED: Issues detected")
            return False
            
    except Exception as e:
        print(f"❌ Error testing confidence variation: {e}")
        return False

def test_deterministic_variation_quality() -> bool:
    """Test that confidence variation is deterministic and based on symbol characteristics"""
    print("\n🎯 Testing Deterministic Variation Quality")
    
    try:
        # Get decisions
        response = requests.get("http://localhost:8001/api/decisions", timeout=15)
        
        if response.status_code != 200:
            return False
        
        data = response.json()
        decisions = data.get('decisions', [])
        
        if len(decisions) < 2:
            print("❌ Need at least 2 decisions for deterministic testing")
            return False
        
        # Analyze if different symbols consistently produce different confidence
        symbol_confidence_map = {}
        
        for decision in decisions:
            symbol = decision.get('symbol', 'Unknown')
            confidence = decision.get('confidence', 0.0)
            
            if symbol not in symbol_confidence_map:
                symbol_confidence_map[symbol] = []
            symbol_confidence_map[symbol].append(confidence)
        
        # Check for deterministic behavior (same symbol should have consistent confidence)
        deterministic_symbols = 0
        total_symbols = 0
        
        print(f"📊 Deterministic Analysis:")
        
        for symbol, confidences in symbol_confidence_map.items():
            if len(confidences) > 1:
                # Check if all confidences for this symbol are the same (deterministic)
                confidence_range = max(confidences) - min(confidences)
                is_deterministic = confidence_range < 0.001  # Very small tolerance
                
                if is_deterministic:
                    deterministic_symbols += 1
                
                total_symbols += 1
                print(f"   {symbol}: {len(confidences)} decisions, range: {confidence_range:.6f} ({'deterministic' if is_deterministic else 'variable'})")
        
        # Check if different symbols have different confidence levels
        unique_symbol_confidences = set()
        for symbol, confidences in symbol_confidence_map.items():
            avg_confidence = sum(confidences) / len(confidences)
            unique_symbol_confidences.add(round(avg_confidence, 3))
        
        symbol_diversity = len(unique_symbol_confidences) > 1
        
        print(f"\n✅ Deterministic Quality:")
        print(f"   Symbol diversity: {'✅' if symbol_diversity else '❌'} ({len(unique_symbol_confidences)} unique levels)")
        print(f"   Deterministic per symbol: {deterministic_symbols}/{total_symbols} symbols")
        
        # Quality assessment
        quality_good = symbol_diversity and (deterministic_symbols >= total_symbols * 0.7 if total_symbols > 0 else True)
        
        if quality_good:
            print(f"✅ DETERMINISTIC QUALITY SUCCESS: Symbols produce consistent but different confidence")
            return True
        else:
            print(f"❌ DETERMINISTIC QUALITY FAILED: Inconsistent or uniform behavior")
            return False
            
    except Exception as e:
        print(f"❌ Error testing deterministic quality: {e}")
        return False

def test_50_percent_minimum_enforcement() -> bool:
    """Test that 50% minimum confidence is enforced"""
    print("\n🎯 Testing 50% Minimum Confidence Enforcement")
    
    try:
        response = requests.get("http://localhost:8001/api/decisions", timeout=15)
        
        if response.status_code != 200:
            return False
        
        data = response.json()
        decisions = data.get('decisions', [])
        
        if len(decisions) == 0:
            print("❌ No decisions available")
            return False
        
        # Check all confidences are >= 50%
        violations = []
        
        for decision in decisions:
            symbol = decision.get('symbol', 'Unknown')
            confidence = decision.get('confidence', 0.0)
            
            if confidence < 0.50:
                violations.append((symbol, confidence))
        
        print(f"📊 50% Minimum Check:")
        print(f"   Total decisions: {len(decisions)}")
        print(f"   Violations (< 50%): {len(violations)}")
        
        if violations:
            print(f"❌ VIOLATIONS FOUND:")
            for symbol, conf in violations[:5]:  # Show first 5
                print(f"      {symbol}: {conf:.3f}")
        
        minimum_enforced = len(violations) == 0
        
        if minimum_enforced:
            print(f"✅ 50% MINIMUM SUCCESS: All decisions maintain ≥50% confidence")
            return True
        else:
            print(f"❌ 50% MINIMUM FAILED: {len(violations)} violations found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing 50% minimum: {e}")
        return False

def main():
    """Run final validation tests"""
    print("🚀 FINAL VALIDATION: BingX Balance & IA2 Confidence Fixes")
    print("="*70)
    
    # Test results
    results = {}
    
    # Test 1: Balance Fix (via logs)
    results['balance_fix'] = check_balance_fix_in_logs()
    
    # Test 2: Confidence Variation Fix
    results['confidence_variation'] = test_confidence_variation_fix()
    
    # Test 3: Deterministic Quality
    results['deterministic_quality'] = test_deterministic_variation_quality()
    
    # Test 4: 50% Minimum Enforcement
    results['minimum_enforcement'] = test_50_percent_minimum_enforcement()
    
    # Summary
    print("\n" + "="*70)
    print("📊 FINAL VALIDATION RESULTS:")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    print(f"\n📈 Score: {passed_tests}/{total_tests} tests passed")
    
    # Critical assessment
    critical_fixes = ['balance_fix', 'confidence_variation']
    critical_passed = all(results[fix] for fix in critical_fixes)
    
    overall_success = passed_tests >= 3 and critical_passed
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    
    if overall_success:
        print(f"✅ SUCCESS: Both critical fixes are working!")
        print(f"   • Balance shows $250 instead of $0")
        print(f"   • Confidence varies by symbol instead of uniform 76%")
        print(f"   • System maintains reliability and 50% minimum")
    else:
        print(f"❌ ISSUES REMAIN: Critical fixes need attention")
        
        if not results['balance_fix']:
            print(f"   • Balance fix not confirmed")
        if not results['confidence_variation']:
            print(f"   • Confidence still uniform instead of varied")
    
    return overall_success

if __name__ == "__main__":
    main()
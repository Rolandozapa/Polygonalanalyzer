#!/usr/bin/env python3
"""
Focused IA2 Decision Agent Testing
Tests the specific fixes for IA2: LLM parsing, confidence calculation, trading thresholds
"""

import requests
import json
import time
import sys
from pathlib import Path

class IA2FocusedTester:
    def __init__(self):
        # Get backend URL from frontend/.env
        try:
            env_path = Path(__file__).parent / "frontend" / ".env"
            with open(env_path, 'r') as f:
                for line in f:
                    if line.startswith('REACT_APP_BACKEND_URL='):
                        base_url = line.split('=', 1)[1].strip()
                        break
                else:
                    base_url = "https://smart-crypto-bot-14.preview.emergentagent.com"
        except:
            base_url = "https://smart-crypto-bot-14.preview.emergentagent.com"
        
        self.api_url = f"{base_url}/api"
        print(f"🎯 Testing IA2 Decision Agent at: {self.api_url}")

    def get_decisions_sample(self, limit=5, timeout=10):
        """Get a small sample of decisions for testing"""
        try:
            response = requests.get(f"{self.api_url}/decisions", timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                decisions = data.get('decisions', [])
                return decisions[:limit] if decisions else []
            else:
                print(f"❌ API Error: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Request Error: {e}")
            return []

    def test_ia2_llm_response_parsing(self):
        """Test IA2 LLM Response Parsing Fix"""
        print("\n🔍 Testing IA2 LLM Response Parsing...")
        
        decisions = self.get_decisions_sample(limit=3)
        if not decisions:
            print("   ❌ No decisions available for testing")
            return False
        
        parsing_success = 0
        total_tested = len(decisions)
        
        for i, decision in enumerate(decisions):
            symbol = decision.get('symbol', 'Unknown')
            reasoning = decision.get('ia2_reasoning', '')
            
            print(f"   Decision {i+1} - {symbol}:")
            
            # Check if reasoning is properly parsed (not null, not raw response)
            if reasoning is None or reasoning == "null" or reasoning == "None":
                print(f"      ❌ Reasoning is NULL - parsing failed")
            elif len(reasoning.strip()) == 0:
                print(f"      ❌ Reasoning is empty - parsing failed")
            elif reasoning.startswith('{') and 'confidence' in reasoning:
                print(f"      ✅ JSON-like reasoning detected - parsing working")
                parsing_success += 1
            elif len(reasoning) > 50 and any(word in reasoning.lower() for word in ['analysis', 'decision', 'confidence']):
                print(f"      ✅ Structured reasoning present - parsing working")
                parsing_success += 1
            else:
                print(f"      ⚠️  Basic reasoning present ({len(reasoning)} chars)")
                parsing_success += 0.5  # Partial credit
        
        success_rate = parsing_success / total_tested
        print(f"\n   📊 LLM Parsing Results: {parsing_success}/{total_tested} ({success_rate*100:.1f}%)")
        
        return success_rate >= 0.8  # 80% success rate

    def test_ia2_confidence_calculation(self):
        """Test IA2 Confidence Calculation Fix"""
        print("\n📊 Testing IA2 Confidence Calculation...")
        
        decisions = self.get_decisions_sample(limit=10)
        if not decisions:
            print("   ❌ No decisions available for testing")
            return False
        
        confidences = []
        for decision in decisions:
            conf = decision.get('confidence', 0)
            confidences.append(conf)
        
        if not confidences:
            print("   ❌ No confidence values found")
            return False
        
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        
        print(f"   📈 Confidence Statistics:")
        print(f"      Sample Size: {len(confidences)}")
        print(f"      Average: {avg_confidence:.3f}")
        print(f"      Range: {min_confidence:.3f} - {max_confidence:.3f}")
        
        # Check if confidence is improved from the problematic 37.3% (0.373)
        improved_from_baseline = avg_confidence > 0.40  # Should be significantly higher
        reasonable_range = 0.2 <= avg_confidence <= 0.9  # Should be in reasonable range
        
        print(f"\n   🎯 Confidence Fix Validation:")
        print(f"      Improved from 37.3%: {'✅' if improved_from_baseline else '❌'} (avg: {avg_confidence:.1%})")
        print(f"      Reasonable Range: {'✅' if reasonable_range else '❌'} (20%-90%)")
        
        return improved_from_baseline and reasonable_range

    def test_ia2_trading_signal_thresholds(self):
        """Test IA2 Trading Signal Thresholds Fix"""
        print("\n📈 Testing IA2 Trading Signal Thresholds...")
        
        decisions = self.get_decisions_sample(limit=15)
        if not decisions:
            print("   ❌ No decisions available for testing")
            return False
        
        signal_counts = {'long': 0, 'short': 0, 'hold': 0}
        confidence_by_signal = {'long': [], 'short': [], 'hold': []}
        
        for decision in decisions:
            signal = decision.get('signal', 'hold').lower()
            confidence = decision.get('confidence', 0)
            
            if signal in signal_counts:
                signal_counts[signal] += 1
                confidence_by_signal[signal].append(confidence)
        
        total = len(decisions)
        trading_rate = (signal_counts['long'] + signal_counts['short']) / total
        
        print(f"   📊 Signal Distribution:")
        for signal, count in signal_counts.items():
            percentage = (count / total) * 100
            avg_conf = sum(confidence_by_signal[signal]) / len(confidence_by_signal[signal]) if confidence_by_signal[signal] else 0
            print(f"      {signal.upper()}: {count} ({percentage:.1f}%) - avg conf: {avg_conf:.3f}")
        
        print(f"   📈 Trading Rate: {trading_rate:.1%}")
        
        # Analyze threshold effectiveness
        # With lowered thresholds, we should see some trading decisions
        # But in a conservative market, mostly HOLD is also acceptable
        
        # Check if any decisions meet the new thresholds (>0.65 moderate, >0.75 strong)
        moderate_threshold_met = any(conf >= 0.65 for conf in [decision.get('confidence', 0) for decision in decisions])
        strong_threshold_met = any(conf >= 0.75 for conf in [decision.get('confidence', 0) for decision in decisions])
        
        print(f"\n   🎯 Threshold Analysis:")
        print(f"      Moderate Threshold (≥0.65): {'✅' if moderate_threshold_met else '❌'}")
        print(f"      Strong Threshold (≥0.75): {'✅' if strong_threshold_met else '❌'}")
        print(f"      Trading Rate: {'✅' if trading_rate > 0 else '⚠️'} ({trading_rate:.1%})")
        
        # Success criteria: Either some trading activity OR confidence levels that could trigger trades
        thresholds_working = moderate_threshold_met or trading_rate > 0.05  # 5% trading rate
        
        return thresholds_working

    def test_ia2_reasoning_field_fix(self):
        """Test IA2 Reasoning Field Fix (not null)"""
        print("\n🧠 Testing IA2 Reasoning Field Fix...")
        
        decisions = self.get_decisions_sample(limit=5)
        if not decisions:
            print("   ❌ No decisions available for testing")
            return False
        
        reasoning_stats = {
            'total': len(decisions),
            'has_reasoning': 0,
            'null_or_empty': 0,
            'quality_reasoning': 0
        }
        
        for i, decision in enumerate(decisions):
            symbol = decision.get('symbol', 'Unknown')
            reasoning = decision.get('ia2_reasoning', '')
            
            print(f"   Decision {i+1} - {symbol}:")
            
            if reasoning is None or reasoning == "null" or reasoning == "None" or len(reasoning.strip()) == 0:
                reasoning_stats['null_or_empty'] += 1
                print(f"      ❌ Reasoning: NULL/EMPTY")
            else:
                reasoning_stats['has_reasoning'] += 1
                print(f"      ✅ Reasoning: Present ({len(reasoning)} chars)")
                
                # Check reasoning quality
                quality_indicators = 0
                if len(reasoning) >= 50: quality_indicators += 1
                if any(word in reasoning.lower() for word in ['analysis', 'confidence', 'decision']): quality_indicators += 1
                if any(word in reasoning.lower() for word in ['rsi', 'macd', 'technical', 'signal']): quality_indicators += 1
                
                if quality_indicators >= 2:
                    reasoning_stats['quality_reasoning'] += 1
                    print(f"      ✅ Quality: HIGH")
                else:
                    print(f"      ⚠️  Quality: MODERATE")
        
        reasoning_rate = reasoning_stats['has_reasoning'] / reasoning_stats['total']
        quality_rate = reasoning_stats['quality_reasoning'] / reasoning_stats['total']
        
        print(f"\n   📊 Reasoning Fix Results:")
        print(f"      Has Reasoning: {reasoning_stats['has_reasoning']}/{reasoning_stats['total']} ({reasoning_rate:.1%})")
        print(f"      Quality Reasoning: {reasoning_stats['quality_reasoning']}/{reasoning_stats['total']} ({quality_rate:.1%})")
        print(f"      Null/Empty: {reasoning_stats['null_or_empty']}")
        
        # Success: Most decisions should have reasoning (fix for null issue)
        reasoning_fixed = reasoning_rate >= 0.8 and reasoning_stats['null_or_empty'] == 0
        
        return reasoning_fixed

    def run_comprehensive_ia2_test(self):
        """Run all IA2 tests"""
        print("🤖 IA2 Decision Agent Comprehensive Test")
        print("=" * 60)
        print("🎯 Testing fixes: LLM parsing, confidence calc, thresholds, reasoning")
        print("=" * 60)
        
        # Run all tests
        test_results = {}
        
        test_results['llm_parsing'] = self.test_ia2_llm_response_parsing()
        test_results['confidence_calc'] = self.test_ia2_confidence_calculation()
        test_results['trading_thresholds'] = self.test_ia2_trading_signal_thresholds()
        test_results['reasoning_fix'] = self.test_ia2_reasoning_field_fix()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 IA2 COMPREHENSIVE TEST RESULTS")
        print("=" * 60)
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        print(f"\n🔍 Individual Test Results:")
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   • {test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\n🎯 Overall Assessment:")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        
        if passed_tests == total_tests:
            print(f"   ✅ IA2 FIXES FULLY SUCCESSFUL")
            return "SUCCESS"
        elif passed_tests >= total_tests * 0.75:
            print(f"   ⚠️  IA2 FIXES MOSTLY SUCCESSFUL")
            return "PARTIAL"
        else:
            print(f"   ❌ IA2 FIXES NEED MORE WORK")
            return "FAILED"

def main():
    tester = IA2FocusedTester()
    result = tester.run_comprehensive_ia2_test()
    
    print(f"\nFinal Result: {result}")
    return 0 if result in ["SUCCESS", "PARTIAL"] else 1

if __name__ == "__main__":
    sys.exit(main())
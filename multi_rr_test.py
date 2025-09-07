#!/usr/bin/env python3
"""
Multi-RR Decision Engine Test Script
Tests the newly implemented Multi-RR Decision Engine with enhanced formulas
"""

import requests
import time
import json
from pathlib import Path

class MultiRRTester:
    def __init__(self):
        # Get backend URL from frontend/.env
        try:
            env_path = Path(__file__).parent / "frontend" / ".env"
            with open(env_path, 'r') as f:
                for line in f:
                    if line.startswith('REACT_APP_BACKEND_URL='):
                        self.base_url = line.split('=', 1)[1].strip()
                        break
                else:
                    self.base_url = "https://smart-crypto-bot-14.preview.emergentagent.com"
        except:
            self.base_url = "https://smart-crypto-bot-14.preview.emergentagent.com"
        
        self.api_url = f"{self.base_url}/api"
        print(f"🎯 Multi-RR Decision Engine Tester")
        print(f"🌐 Backend URL: {self.base_url}")
        print(f"=" * 80)

    def get_analyses(self):
        """Get current IA1 analyses"""
        try:
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            if response.status_code == 200:
                return True, response.json()
            else:
                print(f"❌ Failed to get analyses: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                return False, {}
        except Exception as e:
            print(f"❌ Error getting analyses: {e}")
            return False, {}

    def start_trading_system(self):
        """Start the trading system"""
        try:
            response = requests.post(f"{self.api_url}/start-trading", timeout=30)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Error starting system: {e}")
            return False

    def stop_trading_system(self):
        """Stop the trading system"""
        try:
            response = requests.post(f"{self.api_url}/stop-trading", timeout=30)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Error stopping system: {e}")
            return False

    def test_contradiction_detection(self):
        """Test contradiction detection (RSI oversold vs MACD bearish)"""
        print(f"\n🔍 Testing Contradiction Detection...")
        
        success, data = self.get_analyses()
        if not success:
            print(f"   ❌ Cannot retrieve analyses")
            return False
        
        analyses = data.get('analyses', [])
        if not analyses:
            print(f"   ❌ No analyses available")
            return False
        
        print(f"   📊 Analyzing {len(analyses)} analyses for contradictions...")
        
        contradiction_cases = 0
        biousdt_found = False
        multi_rr_resolutions = 0
        
        for analysis in analyses[:10]:  # Check first 10
            symbol = analysis.get('symbol', 'Unknown')
            rsi = analysis.get('rsi', 50)
            macd_signal = analysis.get('macd_signal', 0)
            bb_position = analysis.get('bollinger_position', 0)
            reasoning = analysis.get('ia1_reasoning', '').lower()
            
            # Check for BIOUSDT specifically
            if 'biousdt' in symbol.lower():
                biousdt_found = True
                print(f"   🎯 BIOUSDT found: RSI {rsi:.1f}, MACD {macd_signal:.6f}")
            
            # Test RSI oversold vs MACD bearish contradiction
            if rsi < 30 and macd_signal < -0.001:
                contradiction_cases += 1
                print(f"   🤔 Contradiction detected - {symbol}: RSI {rsi:.1f} (oversold) vs MACD {macd_signal:.6f} (bearish)")
                
                # Check for Multi-RR resolution
                if 'multi-rr' in reasoning or 'contradiction' in reasoning:
                    multi_rr_resolutions += 1
                    print(f"      ✅ Multi-RR resolution found")
                else:
                    print(f"      ❌ No Multi-RR resolution")
        
        print(f"\n   📊 Contradiction Detection Results:")
        print(f"      BIOUSDT found: {'✅' if biousdt_found else '❌'}")
        print(f"      Contradiction cases: {contradiction_cases}")
        print(f"      Multi-RR resolutions: {multi_rr_resolutions}")
        
        return contradiction_cases > 0 and multi_rr_resolutions > 0

    def test_enhanced_rr_calculations(self):
        """Test the three enhanced RR calculation methods"""
        print(f"\n🧮 Testing Enhanced RR Calculation Methods...")
        
        success, data = self.get_analyses()
        if not success:
            print(f"   ❌ Cannot retrieve analyses")
            return False
        
        analyses = data.get('analyses', [])
        if not analyses:
            print(f"   ❌ No analyses available")
            return False
        
        print(f"   📊 Testing enhanced formulas in {len(analyses)} analyses...")
        
        # Test for evidence of the three calculation methods
        hold_opportunity_evidence = 0
        pattern_rr_evidence = 0
        technical_signal_evidence = 0
        actual_rr_ratios = 0
        
        for analysis in analyses[:10]:
            symbol = analysis.get('symbol', 'Unknown')
            reasoning = analysis.get('ia1_reasoning', '').lower()
            rr_ratio = analysis.get('risk_reward_ratio', 0.0)
            
            # Test 1: _calculate_hold_opportunity_rr (support/resistance based)
            if any(kw in reasoning for kw in ['support', 'resistance', 'breakout', 'hold opportunity']):
                hold_opportunity_evidence += 1
            
            # Test 2: _calculate_pattern_rr (dynamic ATR based on pattern strength)
            if any(kw in reasoning for kw in ['atr', 'pattern strength', 'dynamic', 'volatility']):
                pattern_rr_evidence += 1
            
            # Test 3: _calculate_technical_signal_rr (RSI/MACD/BB confluence)
            if any(kw in reasoning for kw in ['confluence', 'signal strength', 'technical']):
                technical_signal_evidence += 1
            
            # Test 4: Actual RR calculations (not 0.0)
            if rr_ratio > 0.0:
                actual_rr_ratios += 1
                print(f"      {symbol}: RR {rr_ratio:.2f}:1 ✅")
            else:
                print(f"      {symbol}: RR {rr_ratio:.2f}:1 ❌")
        
        print(f"\n   📊 Enhanced Formula Evidence:")
        print(f"      Hold Opportunity RR: {hold_opportunity_evidence}/10")
        print(f"      Pattern RR (ATR-based): {pattern_rr_evidence}/10")
        print(f"      Technical Signal RR: {technical_signal_evidence}/10")
        print(f"      Actual RR calculations: {actual_rr_ratios}/10")
        
        return (hold_opportunity_evidence >= 2 and 
                pattern_rr_evidence >= 2 and 
                technical_signal_evidence >= 3 and
                actual_rr_ratios >= 5)

    def test_technical_signal_recognition(self):
        """Test technical signal recognition patterns"""
        print(f"\n📊 Testing Technical Signal Recognition...")
        
        success, data = self.get_analyses()
        if not success:
            print(f"   ❌ Cannot retrieve analyses")
            return False
        
        analyses = data.get('analyses', [])
        if not analyses:
            print(f"   ❌ No analyses available")
            return False
        
        print(f"   📊 Testing signal patterns in {len(analyses)} analyses...")
        
        long_signals = 0
        short_signals = 0
        correct_recognition = 0
        
        for analysis in analyses[:10]:
            symbol = analysis.get('symbol', 'Unknown')
            rsi = analysis.get('rsi', 50)
            bb_position = analysis.get('bollinger_position', 0)
            ia1_signal = analysis.get('ia1_signal', 'hold').lower()
            reasoning = analysis.get('ia1_reasoning', '').lower()
            
            # Test LONG signal: RSI < 30 + BB < -0.5
            if rsi < 30 and bb_position < -0.5:
                long_signals += 1
                if 'long' in ia1_signal or 'long' in reasoning or 'oversold' in reasoning:
                    correct_recognition += 1
                    print(f"      ✅ LONG recognized: {symbol} (RSI {rsi:.1f}, BB {bb_position:.2f})")
                else:
                    print(f"      ❌ LONG missed: {symbol} (RSI {rsi:.1f}, BB {bb_position:.2f})")
            
            # Test SHORT signal: RSI > 70 + BB > 0.5
            elif rsi > 70 and bb_position > 0.5:
                short_signals += 1
                if 'short' in ia1_signal or 'short' in reasoning or 'overbought' in reasoning:
                    correct_recognition += 1
                    print(f"      ✅ SHORT recognized: {symbol} (RSI {rsi:.1f}, BB {bb_position:.2f})")
                else:
                    print(f"      ❌ SHORT missed: {symbol} (RSI {rsi:.1f}, BB {bb_position:.2f})")
        
        total_signals = long_signals + short_signals
        recognition_rate = correct_recognition / total_signals if total_signals > 0 else 0
        
        print(f"\n   📊 Signal Recognition Results:")
        print(f"      LONG opportunities: {long_signals}")
        print(f"      SHORT opportunities: {short_signals}")
        print(f"      Correct recognition: {correct_recognition}/{total_signals} ({recognition_rate*100:.1f}%)")
        
        return recognition_rate >= 0.6 if total_signals > 0 else True

    def test_multi_rr_vignettes(self):
        """Test Multi-RR vignette display in IA1 analyses"""
        print(f"\n🖼️ Testing Multi-RR Vignette Display...")
        
        success, data = self.get_analyses()
        if not success:
            print(f"   ❌ Cannot retrieve analyses")
            return False
        
        analyses = data.get('analyses', [])
        if not analyses:
            print(f"   ❌ No analyses available")
            return False
        
        print(f"   📊 Testing vignette display in {len(analyses)} analyses...")
        
        vignettes_found = 0
        actual_ratios = 0
        rr_mentions = 0
        
        for analysis in analyses[:10]:
            symbol = analysis.get('symbol', 'Unknown')
            reasoning = analysis.get('ia1_reasoning', '')
            rr_ratio = analysis.get('risk_reward_ratio', 0.0)
            
            # Check for Multi-RR vignette patterns
            has_vignette = any(pattern in reasoning.lower() for pattern in [
                'hold:', 'long:', 'short:', 'multi-rr', 'rr analysis'
            ])
            
            # Check for actual ratio mentions
            has_ratios = any(pattern in reasoning for pattern in [':1', 'rr ', 'ratio'])
            
            if has_vignette:
                vignettes_found += 1
                print(f"      ✅ Vignette found: {symbol}")
            
            if rr_ratio > 0.0:
                actual_ratios += 1
            
            if has_ratios:
                rr_mentions += 1
        
        print(f"\n   📊 Vignette Display Results:")
        print(f"      Vignettes with Multi-RR: {vignettes_found}/10")
        print(f"      Actual RR ratios (not 0.0): {actual_ratios}/10")
        print(f"      RR ratio mentions: {rr_mentions}/10")
        
        return vignettes_found >= 3 and actual_ratios >= 5

    def run_comprehensive_test(self):
        """Run comprehensive Multi-RR Decision Engine tests"""
        print(f"\n🚀 Starting Comprehensive Multi-RR Decision Engine Tests...")
        
        # Start trading system for fresh data
        print(f"\n🚀 Starting trading system...")
        if self.start_trading_system():
            print(f"✅ Trading system started")
            
            # Wait for fresh analyses
            print(f"⏱️  Waiting for fresh Multi-RR analyses (60 seconds)...")
            time.sleep(60)
        else:
            print(f"⚠️  Using existing data for testing...")
        
        # Run all tests
        test_results = {}
        
        print(f"\n" + "="*80)
        test_results['contradiction_detection'] = self.test_contradiction_detection()
        
        print(f"\n" + "="*80)
        test_results['enhanced_calculations'] = self.test_enhanced_rr_calculations()
        
        print(f"\n" + "="*80)
        test_results['signal_recognition'] = self.test_technical_signal_recognition()
        
        print(f"\n" + "="*80)
        test_results['vignette_display'] = self.test_multi_rr_vignettes()
        
        # Stop trading system
        print(f"\n🛑 Stopping trading system...")
        self.stop_trading_system()
        
        # Final results
        print(f"\n" + "="*80)
        print(f"🎯 MULTI-RR DECISION ENGINE TEST RESULTS")
        print(f"="*80)
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\n📊 Overall Results: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        
        if passed_tests >= 3:
            print(f"🎉 SUCCESS: Multi-RR Decision Engine is working!")
            print(f"💡 Contradiction detection: {'✅' if test_results['contradiction_detection'] else '❌'}")
            print(f"💡 Enhanced formulas: {'✅' if test_results['enhanced_calculations'] else '❌'}")
            print(f"💡 Signal recognition: {'✅' if test_results['signal_recognition'] else '❌'}")
            print(f"💡 Vignette display: {'✅' if test_results['vignette_display'] else '❌'}")
        else:
            print(f"⚠️  ISSUES DETECTED: Multi-RR system needs attention")
            print(f"💡 Focus on failed tests for improvement")
        
        return passed_tests >= 3

if __name__ == "__main__":
    tester = MultiRRTester()
    tester.run_comprehensive_test()
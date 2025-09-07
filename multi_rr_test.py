#!/usr/bin/env python3

import requests
import sys
import json
import time
import os
from pathlib import Path

class MultiRRTester:
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
                    base_url = "https://aitra-platform.preview.emergentagent.com"
            except:
                base_url = "https://aitra-platform.preview.emergentagent.com"
        
        self.base_url = base_url
        self.api_url = f"{base_url}/api"

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if endpoint else f"{self.api_url}/"
        headers = {'Content-Type': 'application/json'}

        print(f"\n🔍 Testing {name}...")
        
        # Retry logic for network issues
        max_retries = 3
        for attempt in range(max_retries):
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
                    print(f"✅ Passed - Status: {response.status_code} - Time: {response_time:.2f}s")
                    try:
                        response_data = response.json()
                        return True, response_data
                    except:
                        return True, {}
                else:
                    if attempt < max_retries - 1:
                        print(f"⚠️  Attempt {attempt + 1} failed - Status: {response.status_code}, retrying...")
                        time.sleep(2)
                        continue
                    else:
                        print(f"❌ Failed - Expected {expected_status}, got {response.status_code} - Time: {response_time:.2f}s")
                        return False, {}

            except Exception as e:
                end_time = time.time()
                response_time = end_time - start_time
                if attempt < max_retries - 1:
                    print(f"⚠️  Attempt {attempt + 1} failed - Error: {str(e)}, retrying...")
                    time.sleep(2)
                    continue
                else:
                    print(f"❌ Failed - Error: {str(e)} - Time: {response_time:.2f}s")
                    return False, {}
        
        return False, {}

    def test_start_trading_system(self):
        """Test starting the trading system"""
        return self.run_test("Start Trading System", "POST", "start-trading", 200)

    def test_stop_trading_system(self):
        """Test stopping the trading system"""
        return self.run_test("Stop Trading System", "POST", "stop-trading", 200)

    def test_get_analyses(self):
        """Test get analyses endpoint"""
        return self.run_test("Get Technical Analyses", "GET", "analyses", 200)

    def test_multi_rr_decision_engine_comprehensive(self):
        """Comprehensive test of the Multi-RR Decision Engine as requested in review"""
        print(f"\n🤖 COMPREHENSIVE MULTI-RR DECISION ENGINE TEST...")
        print(f"   Testing: Enhanced Multi-RR system with contradiction detection and vignette display")
        
        # Test 1: Start Trading System to Generate Fresh IA1 Analyses
        print(f"\n   🚀 STEP 1: Starting Trading System for Fresh Multi-RR Analyses...")
        success, _ = self.test_start_trading_system()
        if not success:
            print(f"   ❌ Failed to start trading system")
            return False
        
        # Wait for system to generate analyses with Multi-RR
        print(f"   ⏱️  Waiting for Multi-RR analyses generation (3-5 minutes as requested)...")
        
        start_time = time.time()
        max_wait_time = 300  # 5 minutes maximum
        check_interval = 15   # Check every 15 seconds
        
        multi_rr_analyses = []
        biousdt_found = False
        contradiction_cases = []
        
        while time.time() - start_time < max_wait_time:
            time.sleep(check_interval)
            elapsed = time.time() - start_time
            
            # Get current analyses
            success, analyses_data = self.test_get_analyses()
            if success:
                analyses = analyses_data.get('analyses', [])
                print(f"   📊 After {elapsed:.1f}s: {len(analyses)} analyses available")
                
                # Test 2: Search for BIOUSDT Contradiction Case
                for analysis in analyses:
                    symbol = analysis.get('symbol', '')
                    rsi = analysis.get('rsi', 0)
                    macd = analysis.get('macd_signal', 0)
                    reasoning = analysis.get('ia1_reasoning', '')
                    
                    # Check for BIOUSDT specific case (RSI 24.2 + MACD 0.013892)
                    if 'BIOUSDT' in symbol.upper():
                        biousdt_found = True
                        print(f"   🎯 BIOUSDT FOUND: RSI {rsi:.1f}, MACD {macd:.6f}")
                        
                        # Check if it matches the expected contradiction case
                        if abs(rsi - 24.2) < 5.0 and abs(macd - 0.013892) < 0.01:
                            print(f"   ✅ BIOUSDT EXACT MATCH: RSI {rsi:.1f} ≈ 24.2, MACD {macd:.6f} ≈ 0.013892")
                        
                        # Check for Multi-RR processing
                        if any(keyword in reasoning.lower() for keyword in ['multi-rr', 'contradiction', 'rr analysis']):
                            print(f"   ✅ BIOUSDT Multi-RR Processing Detected")
                        else:
                            print(f"   ⚠️  BIOUSDT Multi-RR Processing Not Detected")
                    
                    # Test 3: Check for Multi-RR Vignette Display Format
                    multi_rr_keywords = ['multi-rr analysis', 'hold:', 'long:', 'short:', 'winner:', '🤖', '🏆']
                    if any(keyword in reasoning.lower() for keyword in multi_rr_keywords):
                        multi_rr_analyses.append({
                            'symbol': symbol,
                            'rsi': rsi,
                            'macd': macd,
                            'reasoning': reasoning,
                            'has_vignette': True
                        })
                        print(f"   ✅ Multi-RR Vignette Found: {symbol}")
                    
                    # Test 4: Check for Enhanced RR Calculations (not 0.0:0.0)
                    rr_ratio = analysis.get('risk_reward_ratio', 0.0)
                    if rr_ratio > 0.0:
                        print(f"   ✅ Real RR Calculation: {symbol} = {rr_ratio:.2f}:1")
                    
                    # Test 5: Check for Technical Signal Recognition
                    # RSI<30+BB<-0.5=LONG and RSI>70+BB>0.5=SHORT
                    bb_position = analysis.get('bollinger_position', 0)
                    if rsi < 30 and bb_position < -0.5:
                        if 'long' in reasoning.lower():
                            print(f"   ✅ LONG Signal Recognition: {symbol} (RSI {rsi:.1f} < 30, BB {bb_position:.2f} < -0.5)")
                    elif rsi > 70 and bb_position > 0.5:
                        if 'short' in reasoning.lower():
                            print(f"   ✅ SHORT Signal Recognition: {symbol} (RSI {rsi:.1f} > 70, BB {bb_position:.2f} > 0.5)")
                    
                    # Detect contradiction cases
                    if rsi < 30 and macd > 0:  # RSI oversold + MACD bullish
                        contradiction_cases.append({
                            'symbol': symbol,
                            'type': 'RSI_OVERSOLD vs MACD_BULLISH',
                            'rsi': rsi,
                            'macd': macd,
                            'reasoning': reasoning
                        })
                    elif rsi > 70 and macd < 0:  # RSI overbought + MACD bearish
                        contradiction_cases.append({
                            'symbol': symbol,
                            'type': 'RSI_OVERBOUGHT vs MACD_BEARISH',
                            'rsi': rsi,
                            'macd': macd,
                            'reasoning': reasoning
                        })
                
                # Check if we have enough data for comprehensive testing
                if len(analyses) >= 10 and len(multi_rr_analyses) > 0:
                    print(f"   ✅ Sufficient data for comprehensive testing")
                    break
            
            if elapsed >= 180:  # 3 minutes minimum as requested
                print(f"   ⏱️  Minimum 3 minutes reached, proceeding with analysis...")
                break
        
        # Stop the trading system
        print(f"   🛑 Stopping trading system...")
        self.test_stop_trading_system()
        
        # Test 6: Comprehensive Analysis of Results
        print(f"\n   📊 COMPREHENSIVE MULTI-RR ANALYSIS RESULTS:")
        
        # Get final analyses for comprehensive testing
        success, final_analyses_data = self.test_get_analyses()
        if not success:
            print(f"   ❌ Cannot retrieve final analyses")
            return False
        
        final_analyses = final_analyses_data.get('analyses', [])
        
        # Analysis 1: BIOUSDT Case Verification
        biousdt_analyses = [a for a in final_analyses if 'BIOUSDT' in a.get('symbol', '').upper()]
        print(f"\n   🎯 BIOUSDT CONTRADICTION CASE ANALYSIS:")
        print(f"      BIOUSDT Analyses Found: {len(biousdt_analyses)}")
        
        biousdt_success = False
        if biousdt_analyses:
            for analysis in biousdt_analyses:
                rsi = analysis.get('rsi', 0)
                macd = analysis.get('macd_signal', 0)
                reasoning = analysis.get('ia1_reasoning', '')
                
                print(f"      BIOUSDT: RSI {rsi:.1f}, MACD {macd:.6f}")
                
                # Check for contradiction detection
                if rsi < 30 and macd > 0:  # RSI oversold + MACD bullish (as found in previous tests)
                    print(f"      ✅ Contradiction Detected: RSI oversold ({rsi:.1f}) + MACD bullish ({macd:.6f})")
                    
                    # Check for Multi-RR resolution
                    if any(keyword in reasoning.lower() for keyword in ['multi-rr', 'contradiction', 'rr analysis']):
                        print(f"      ✅ Multi-RR Resolution Applied")
                        biousdt_success = True
                    else:
                        print(f"      ❌ Multi-RR Resolution Missing")
        
        # Analysis 2: Multi-RR Vignette Display Format
        vignette_analyses = []
        for analysis in final_analyses:
            reasoning = analysis.get('ia1_reasoning', '')
            symbol = analysis.get('symbol', '')
            
            # Look for the specific format: "🤖 MULTI-RR ANALYSIS: • HOLD: 1.8:1 • LONG: 2.3:1 🏆 WINNER: LONG"
            if '🤖' in reasoning and 'multi-rr' in reasoning.lower():
                vignette_analyses.append({
                    'symbol': symbol,
                    'reasoning': reasoning
                })
        
        print(f"\n   🎨 MULTI-RR VIGNETTE DISPLAY ANALYSIS:")
        print(f"      Analyses with Multi-RR Vignettes: {len(vignette_analyses)}")
        
        vignette_success = len(vignette_analyses) > 0
        if vignette_analyses:
            for vignette in vignette_analyses[:3]:  # Show first 3
                print(f"      ✅ {vignette['symbol']}: Multi-RR vignette detected")
                # Look for specific format elements
                reasoning = vignette['reasoning']
                if 'hold:' in reasoning.lower() and 'long:' in reasoning.lower():
                    print(f"         Format elements found: HOLD and LONG ratios")
                if '🏆' in reasoning and 'winner' in reasoning.lower():
                    print(f"         Winner declaration found")
        
        # Analysis 3: Enhanced RR Calculations (Three New Methods)
        print(f"\n   🧮 ENHANCED RR CALCULATIONS ANALYSIS:")
        
        hold_opportunity_count = 0
        pattern_rr_count = 0
        technical_signal_count = 0
        real_ratios_count = 0
        
        for analysis in final_analyses:
            reasoning = analysis.get('ia1_reasoning', '').lower()
            rr_ratio = analysis.get('risk_reward_ratio', 0.0)
            
            # Check for the three new methods
            if any(keyword in reasoning for keyword in ['support', 'resistance', 'hold opportunity']):
                hold_opportunity_count += 1
            
            if any(keyword in reasoning for keyword in ['pattern', 'atr', 'dynamic']):
                pattern_rr_count += 1
            
            if any(keyword in reasoning for keyword in ['rsi', 'macd', 'bb', 'confluence', 'technical signal']):
                technical_signal_count += 1
            
            # Check for real ratios (not 0.0:0.0)
            if rr_ratio > 0.0:
                real_ratios_count += 1
        
        print(f"      Hold Opportunity RR (support/resistance): {hold_opportunity_count}/{len(final_analyses)} ({hold_opportunity_count/len(final_analyses)*100:.1f}%)")
        print(f"      Pattern RR (dynamic ATR): {pattern_rr_count}/{len(final_analyses)} ({pattern_rr_count/len(final_analyses)*100:.1f}%)")
        print(f"      Technical Signal RR (RSI/MACD/BB): {technical_signal_count}/{len(final_analyses)} ({technical_signal_count/len(final_analyses)*100:.1f}%)")
        print(f"      Real RR Ratios (not 0.0:0.0): {real_ratios_count}/{len(final_analyses)} ({real_ratios_count/len(final_analyses)*100:.1f}%)")
        
        enhanced_calculations_success = (
            hold_opportunity_count > 0 and
            real_ratios_count >= len(final_analyses) * 0.3  # At least 30% have real ratios
        )
        
        # Analysis 4: Technical Signal Recognition
        print(f"\n   🔍 TECHNICAL SIGNAL RECOGNITION ANALYSIS:")
        
        long_signals = 0
        short_signals = 0
        
        for analysis in final_analyses:
            rsi = analysis.get('rsi', 50)
            bb_position = analysis.get('bollinger_position', 0)
            reasoning = analysis.get('ia1_reasoning', '').lower()
            
            # RSI<30+BB<-0.5=LONG
            if rsi < 30 and bb_position < -0.5:
                if 'long' in reasoning:
                    long_signals += 1
                    print(f"      ✅ LONG Signal: {analysis.get('symbol', '')} (RSI {rsi:.1f}, BB {bb_position:.2f})")
            
            # RSI>70+BB>0.5=SHORT
            elif rsi > 70 and bb_position > 0.5:
                if 'short' in reasoning:
                    short_signals += 1
                    print(f"      ✅ SHORT Signal: {analysis.get('symbol', '')} (RSI {rsi:.1f}, BB {bb_position:.2f})")
        
        print(f"      LONG Signals Detected: {long_signals}")
        print(f"      SHORT Signals Detected: {short_signals}")
        
        signal_recognition_success = (long_signals + short_signals) > 0
        
        # Analysis 5: End-to-End Flow Validation
        print(f"\n   🔄 END-TO-END FLOW VALIDATION:")
        
        contradiction_detected = len(contradiction_cases) > 0
        multi_rr_applied = len(vignette_analyses) > 0
        user_visible = len(final_analyses) > 0
        
        print(f"      Contradictions Detected: {len(contradiction_cases)}")
        print(f"      Multi-RR Applied: {len(vignette_analyses)}")
        print(f"      User Visible Analyses: {len(final_analyses)}")
        
        # Show contradiction examples
        if contradiction_cases:
            print(f"      Contradiction Examples:")
            for case in contradiction_cases[:3]:
                print(f"         {case['symbol']}: {case['type']} (RSI {case['rsi']:.1f}, MACD {case['macd']:.6f})")
        
        end_to_end_success = contradiction_detected and user_visible
        
        # Final Assessment
        print(f"\n   🎯 COMPREHENSIVE MULTI-RR SYSTEM ASSESSMENT:")
        
        components = {
            "BIOUSDT Contradiction Case": biousdt_success,
            "Multi-RR Vignette Display": vignette_success,
            "Enhanced RR Calculations": enhanced_calculations_success,
            "Technical Signal Recognition": signal_recognition_success,
            "End-to-End Flow": end_to_end_success
        }
        
        passed_components = sum(components.values())
        total_components = len(components)
        
        for component, success in components.items():
            print(f"      {component}: {'✅' if success else '❌'}")
        
        overall_success = passed_components >= 3  # At least 3/5 components working
        
        print(f"\n   🏆 MULTI-RR DECISION ENGINE FINAL RESULT:")
        print(f"      Components Passed: {passed_components}/{total_components}")
        print(f"      Overall Status: {'✅ SUCCESS' if overall_success else '❌ NEEDS IMPROVEMENT'}")
        
        if not overall_success:
            print(f"\n   💡 RECOMMENDATIONS:")
            if not biousdt_success:
                print(f"      - Fix BIOUSDT contradiction detection logic")
            if not vignette_success:
                print(f"      - Implement Multi-RR vignette display format")
            if not enhanced_calculations_success:
                print(f"      - Ensure enhanced RR calculation methods produce real ratios")
            if not signal_recognition_success:
                print(f"      - Implement technical signal recognition (RSI+BB triggers)")
            if not end_to_end_success:
                print(f"      - Fix end-to-end flow from contradiction detection to user display")
        
        return overall_success

if __name__ == "__main__":
    tester = MultiRRTester()
    print(f"🚀 Starting Multi-RR Decision Engine Test...")
    print(f"🌐 Backend URL: {tester.base_url}")
    print(f"📡 API URL: {tester.api_url}")
    
    success = tester.test_multi_rr_decision_engine_comprehensive()
    
    print(f"\n🎯 Multi-RR Test Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if success:
        print("🎉 Multi-RR Decision Engine test passed!")
        sys.exit(0)
    else:
        print("⚠️  Multi-RR Decision Engine test failed - check logs above")
        sys.exit(1)
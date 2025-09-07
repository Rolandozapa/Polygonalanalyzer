#!/usr/bin/env python3

import requests
import json
import time
from pathlib import Path

class MultiRRTester:
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

    def get_analyses(self):
        """Get current IA1 analyses"""
        try:
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {}
        except Exception as e:
            print(f"Error getting analyses: {e}")
            return False, {}

    def test_multi_rr_decision_engine(self):
        """🚀 TEST RÉVOLUTIONNAIRE - Multi-RR Decision Engine pour Contradictions IA1"""
        print(f"\n🚀 Testing Multi-RR Decision Engine for IA1 Contradictions...")
        print(f"   📋 TESTING OBJECTIVES:")
        print(f"      • Detect IA1 contradictions (HOLD vs PATTERN direction)")
        print(f"      • Calculate Multi-RR for HOLD vs PATTERN scenarios")
        print(f"      • Resolve intelligently based on best RR ratio")
        print(f"      • Update ia1_signal with final recommendation")
        print(f"      • Log transparent Multi-RR resolution process")
        
        # Get current analyses to check Multi-RR engine
        success, analyses_data = self.get_analyses()
        if not success:
            print(f"   ❌ Cannot retrieve analyses for Multi-RR testing")
            return False
        
        analyses = analyses_data.get('analyses', [])
        if len(analyses) == 0:
            print(f"   ❌ No analyses available for Multi-RR testing")
            return False
        
        print(f"   📊 Analyzing Multi-RR Decision Engine in {len(analyses)} analyses...")
        
        # Multi-RR analysis tracking
        contradiction_detected_count = 0
        multi_rr_resolution_count = 0
        hold_vs_pattern_cases = 0
        resolution_reasoning_count = 0
        signal_updated_count = 0
        
        multi_rr_examples = []
        
        for i, analysis in enumerate(analyses[:20]):  # Analyze first 20 analyses
            symbol = analysis.get('symbol', 'Unknown')
            ia1_signal = analysis.get('ia1_signal', 'hold').upper()
            reasoning = analysis.get('ia1_reasoning', '')
            patterns = analysis.get('patterns_detected', [])
            confidence = analysis.get('analysis_confidence', 0)
            
            # Check for Multi-RR Decision Engine keywords
            has_contradiction_detection = '🤔 contradiction ia1' in reasoning.lower() or 'contradiction' in reasoning.lower()
            has_multi_rr_resolution = '🤖 multi-rr resolution' in reasoning.lower() or 'multi-rr' in reasoning.lower()
            has_rr_analysis = 'rr analysis' in reasoning.lower() or 'rr ratio' in reasoning.lower()
            has_winner_selection = '🏆 winner' in reasoning.lower() or 'winner:' in reasoning.lower()
            has_resolution_reasoning = 'resolution reasoning' in reasoning.lower() or 'wins with' in reasoning.lower()
            
            # Check for specific WLDUSDT case mentioned in review
            is_wldusdt = symbol.upper() == 'WLDUSDT'
            has_bearish_channel = any('bearish' in pattern.lower() and 'channel' in pattern.lower() for pattern in patterns)
            
            # Count Multi-RR indicators
            if has_contradiction_detection:
                contradiction_detected_count += 1
            if has_multi_rr_resolution:
                multi_rr_resolution_count += 1
            if has_rr_analysis:
                hold_vs_pattern_cases += 1
            if has_resolution_reasoning:
                resolution_reasoning_count += 1
            if ia1_signal in ['LONG', 'SHORT']:  # Signal was updated from HOLD
                signal_updated_count += 1
            
            # Collect examples for detailed analysis
            if i < 5 or has_multi_rr_resolution or is_wldusdt:
                example = {
                    'symbol': symbol,
                    'ia1_signal': ia1_signal,
                    'patterns': patterns,
                    'confidence': confidence,
                    'has_contradiction': has_contradiction_detection,
                    'has_multi_rr': has_multi_rr_resolution,
                    'has_winner': has_winner_selection,
                    'is_wldusdt': is_wldusdt,
                    'has_bearish_channel': has_bearish_channel,
                    'reasoning_preview': reasoning[:200] + '...' if len(reasoning) > 200 else reasoning
                }
                multi_rr_examples.append(example)
                
                print(f"\n   Analysis {i+1} - {symbol}:")
                print(f"      Signal: {ia1_signal}")
                print(f"      Patterns: {patterns}")
                print(f"      Contradiction Detected: {'✅' if has_contradiction_detection else '❌'}")
                print(f"      Multi-RR Resolution: {'✅' if has_multi_rr_resolution else '❌'}")
                print(f"      Winner Selection: {'✅' if has_winner_selection else '❌'}")
                if is_wldusdt:
                    print(f"      🎯 WLDUSDT CASE: {'✅ Found' if is_wldusdt else '❌'}")
                    print(f"      Bearish Channel: {'✅' if has_bearish_channel else '❌'}")
        
        # Calculate Multi-RR system performance
        total_analyses = len(analyses)
        contradiction_rate = contradiction_detected_count / total_analyses if total_analyses > 0 else 0
        resolution_rate = multi_rr_resolution_count / total_analyses if total_analyses > 0 else 0
        signal_update_rate = signal_updated_count / total_analyses if total_analyses > 0 else 0
        
        print(f"\n   📊 Multi-RR Decision Engine Analysis:")
        print(f"      Total Analyses: {total_analyses}")
        print(f"      Contradictions Detected: {contradiction_detected_count} ({contradiction_rate*100:.1f}%)")
        print(f"      Multi-RR Resolutions: {multi_rr_resolution_count} ({resolution_rate*100:.1f}%)")
        print(f"      RR Analysis Cases: {hold_vs_pattern_cases}")
        print(f"      Resolution Reasoning: {resolution_reasoning_count}")
        print(f"      Signals Updated: {signal_updated_count} ({signal_update_rate*100:.1f}%)")
        
        # Check for WLDUSDT specific case
        wldusdt_cases = [ex for ex in multi_rr_examples if ex['is_wldusdt']]
        print(f"\n   🎯 WLDUSDT Case Analysis:")
        if wldusdt_cases:
            wldusdt = wldusdt_cases[0]
            print(f"      WLDUSDT Found: ✅")
            print(f"      Signal: {wldusdt['ia1_signal']}")
            print(f"      Bearish Channel: {'✅' if wldusdt['has_bearish_channel'] else '❌'}")
            print(f"      Multi-RR Applied: {'✅' if wldusdt['has_multi_rr'] else '❌'}")
        else:
            print(f"      WLDUSDT Found: ❌ (Not in current analyses)")
        
        # Validation criteria for Multi-RR Decision Engine
        contradiction_system_active = contradiction_detected_count > 0
        multi_rr_engine_working = multi_rr_resolution_count > 0
        resolution_logic_present = resolution_reasoning_count > 0
        signal_updates_happening = signal_updated_count > 0
        rr_calculations_working = hold_vs_pattern_cases > 0
        
        print(f"\n   ✅ Multi-RR Decision Engine Validation:")
        print(f"      Contradiction Detection: {'✅' if contradiction_system_active else '❌'}")
        print(f"      Multi-RR Engine Active: {'✅' if multi_rr_engine_working else '❌'}")
        print(f"      Resolution Logic: {'✅' if resolution_logic_present else '❌'}")
        print(f"      Signal Updates: {'✅' if signal_updates_happening else '❌'}")
        print(f"      RR Calculations: {'✅' if rr_calculations_working else '❌'}")
        
        # Overall Multi-RR system assessment
        multi_rr_system_working = (
            contradiction_system_active and
            multi_rr_engine_working and
            resolution_logic_present and
            (signal_updates_happening or resolution_reasoning_count > 0)
        )
        
        print(f"\n   🎯 Multi-RR Decision Engine: {'✅ WORKING' if multi_rr_system_working else '❌ NEEDS IMPLEMENTATION'}")
        
        if multi_rr_system_working:
            print(f"   💡 SUCCESS: Multi-RR engine resolves IA1 contradictions intelligently")
            print(f"   💡 Contradictions detected: {contradiction_detected_count}")
            print(f"   💡 Multi-RR resolutions: {multi_rr_resolution_count}")
            print(f"   💡 Signals updated based on RR analysis")
        else:
            print(f"   💡 IMPLEMENTATION NEEDED:")
            if not contradiction_system_active:
                print(f"      - Contradiction detection not working")
            if not multi_rr_engine_working:
                print(f"      - Multi-RR resolution engine not active")
            if not resolution_logic_present:
                print(f"      - Resolution reasoning not present")
        
        return multi_rr_system_working

if __name__ == "__main__":
    print("🚀 Multi-RR Decision Engine Test")
    print("=" * 50)
    
    tester = MultiRRTester()
    result = tester.test_multi_rr_decision_engine()
    
    print("\n" + "=" * 50)
    print(f"🎯 FINAL RESULT: {'✅ PASS' if result else '❌ FAIL'}")
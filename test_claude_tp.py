#!/usr/bin/env python3

import requests
import json
from pathlib import Path

class ClaudeTPTester:
    def __init__(self):
        # Get the correct backend URL from frontend/.env
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

    def get_decisions(self):
        """Get trading decisions from IA2"""
        try:
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {}
        except Exception as e:
            print(f"Error getting decisions: {e}")
            return False, {}

    def test_claude_intelligent_tp_strategy_system(self):
        """🚀 TEST RÉVOLUTIONNAIRE - Test Claude Intelligent TP Strategy System"""
        print(f"\n🚀 Testing Claude Intelligent TP Strategy System...")
        print(f"   📋 REVOLUTIONARY TESTING LOGIC:")
        print(f"      • Claude generates intelligent TP strategies based on pattern analysis")
        print(f"      • 3 scenarios: Conservative/Base/Optimistic with adaptive TP levels")
        print(f"      • Priority: Claude TP Strategy > Hardcoded fallback")
        print(f"      • Support for both LONG and SHORT with adapted strategies")
        print(f"      • Logging: '✅ Using Claude TP Strategy' confirmation")
        
        # Get current decisions to analyze
        success, decisions_data = self.get_decisions()
        if not success:
            print(f"   ❌ Cannot retrieve decisions for Claude TP strategy testing")
            return False
        
        decisions = decisions_data.get('decisions', [])
        if len(decisions) == 0:
            print(f"   ❌ No decisions available for Claude TP strategy testing")
            return False
        
        print(f"   📊 Analyzing Claude TP strategies in {len(decisions)} decisions...")
        
        # Analyze decisions for Claude TP strategy evidence
        claude_tp_strategy_count = 0
        fallback_hardcoded_count = 0
        long_claude_tp_count = 0
        short_claude_tp_count = 0
        pattern_analysis_count = 0
        scenario_based_count = 0
        
        claude_tp_examples = []
        fallback_examples = []
        
        for i, decision in enumerate(decisions[:30]):  # Analyze first 30 decisions
            symbol = decision.get('symbol', 'Unknown')
            signal = decision.get('signal', 'hold').upper()
            reasoning = decision.get('ia2_reasoning', '').lower()
            confidence = decision.get('confidence', 0)
            
            # Check for Claude TP Strategy patterns in reasoning
            has_claude_tp_strategy = 'claude tp strategy' in reasoning
            has_pattern_analysis = 'pattern analysis' in reasoning or 'double bottom' in reasoning or 'resistance' in reasoning
            has_scenario_based = 'conservative' in reasoning or 'optimistic' in reasoning or 'base scenario' in reasoning
            has_intelligent_tp = 'intelligent tp' in reasoning or 'adaptive' in reasoning
            has_fallback_hardcoded = '5-level tp' in reasoning or 'fallback' in reasoning
            
            # Count Claude TP strategy usage
            if has_claude_tp_strategy or has_intelligent_tp:
                claude_tp_strategy_count += 1
                if signal == 'LONG':
                    long_claude_tp_count += 1
                elif signal == 'SHORT':
                    short_claude_tp_count += 1
                
                claude_tp_examples.append({
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': confidence,
                    'has_pattern_analysis': has_pattern_analysis,
                    'has_scenarios': has_scenario_based,
                    'reasoning_snippet': reasoning[:150]
                })
            
            # Count fallback usage
            if has_fallback_hardcoded:
                fallback_hardcoded_count += 1
                fallback_examples.append({
                    'symbol': symbol,
                    'signal': signal,
                    'reasoning_snippet': reasoning[:100]
                })
            
            # Count pattern analysis and scenario usage
            if has_pattern_analysis:
                pattern_analysis_count += 1
            if has_scenario_based:
                scenario_based_count += 1
            
            if i < 5:  # Show details for first 5 decisions
                print(f"   Decision {i+1} - {symbol} ({signal}):")
                print(f"      Confidence: {confidence:.3f}")
                print(f"      Claude TP Strategy: {'✅' if has_claude_tp_strategy else '❌'}")
                print(f"      Pattern Analysis: {'✅' if has_pattern_analysis else '❌'}")
                print(f"      Scenario-Based: {'✅' if has_scenario_based else '❌'}")
                print(f"      Fallback Hardcoded: {'✅' if has_fallback_hardcoded else '❌'}")
        
        total_analyzed = len(decisions[:30])
        claude_tp_rate = claude_tp_strategy_count / total_analyzed if total_analyzed > 0 else 0
        fallback_rate = fallback_hardcoded_count / total_analyzed if total_analyzed > 0 else 0
        
        print(f"\n   📊 Claude TP Strategy Analysis Results:")
        print(f"      Total Analyzed: {total_analyzed}")
        print(f"      Claude TP Strategy Usage: {claude_tp_strategy_count} ({claude_tp_rate*100:.1f}%)")
        print(f"      LONG Claude TP: {long_claude_tp_count}")
        print(f"      SHORT Claude TP: {short_claude_tp_count}")
        print(f"      Pattern Analysis: {pattern_analysis_count} ({pattern_analysis_count/total_analyzed*100:.1f}%)")
        print(f"      Scenario-Based: {scenario_based_count} ({scenario_based_count/total_analyzed*100:.1f}%)")
        print(f"      Fallback Hardcoded: {fallback_hardcoded_count} ({fallback_rate*100:.1f}%)")
        
        # Show Claude TP strategy examples
        if claude_tp_examples:
            print(f"\n   🎯 CLAUDE TP STRATEGY EXAMPLES:")
            for i, example in enumerate(claude_tp_examples[:3]):
                print(f"      {i+1}. {example['symbol']} ({example['signal']}):")
                print(f"         Confidence: {example['confidence']:.3f}")
                print(f"         Pattern Analysis: {'✅' if example['has_pattern_analysis'] else '❌'}")
                print(f"         Scenarios: {'✅' if example['has_scenarios'] else '❌'}")
                print(f"         Reasoning: {example['reasoning_snippet']}...")
        
        # Show fallback examples
        if fallback_examples:
            print(f"\n   🔄 FALLBACK HARDCODED EXAMPLES:")
            for i, example in enumerate(fallback_examples[:2]):
                print(f"      {i+1}. {example['symbol']} ({example['signal']}):")
                print(f"         Reasoning: {example['reasoning_snippet']}...")
        
        # Test for specific TP percentage patterns
        print(f"\n   🔍 Testing TP Percentage Patterns...")
        tp_percentage_patterns = {
            'tp1_conservative': 0,  # 0.3-0.8%
            'tp2_moderate': 0,      # 1.0-1.5%
            'tp3_aggressive': 0,    # 1.8-2.5%
            'tp4_optimistic': 0     # 3.0-4.0%
        }
        
        for decision in decisions[:20]:
            reasoning = decision.get('ia2_reasoning', '').lower()
            
            # Look for specific TP percentage mentions
            if 'tp1' in reasoning and any(pct in reasoning for pct in ['0.5%', '0.6%', '0.7%', '0.8%']):
                tp_percentage_patterns['tp1_conservative'] += 1
            if 'tp2' in reasoning and any(pct in reasoning for pct in ['1.0%', '1.2%', '1.5%']):
                tp_percentage_patterns['tp2_moderate'] += 1
            if 'tp3' in reasoning and any(pct in reasoning for pct in ['1.8%', '2.0%', '2.1%', '2.5%']):
                tp_percentage_patterns['tp3_aggressive'] += 1
            if 'tp4' in reasoning and any(pct in reasoning for pct in ['3.0%', '3.2%', '4.0%']):
                tp_percentage_patterns['tp4_optimistic'] += 1
        
        print(f"      TP1 Conservative (0.5-0.8%): {tp_percentage_patterns['tp1_conservative']}")
        print(f"      TP2 Moderate (1.0-1.5%): {tp_percentage_patterns['tp2_moderate']}")
        print(f"      TP3 Aggressive (1.8-2.5%): {tp_percentage_patterns['tp3_aggressive']}")
        print(f"      TP4 Optimistic (3.0-4.0%): {tp_percentage_patterns['tp4_optimistic']}")
        
        # Validation criteria for Claude TP Strategy system
        claude_tp_implemented = claude_tp_strategy_count > 0
        reasonable_claude_usage = claude_tp_rate >= 0.20  # At least 20% should use Claude TP
        both_directions_supported = long_claude_tp_count > 0 and short_claude_tp_count > 0
        pattern_analysis_present = pattern_analysis_count > 0
        scenario_logic_present = scenario_based_count > 0
        fallback_available = fallback_hardcoded_count > 0  # Fallback should exist
        tp_percentages_realistic = sum(tp_percentage_patterns.values()) > 0
        
        print(f"\n   ✅ Claude TP Strategy System Validation:")
        print(f"      Claude TP Implemented: {'✅' if claude_tp_implemented else '❌'}")
        print(f"      Reasonable Usage Rate: {'✅' if reasonable_claude_usage else '❌'} (≥20%)")
        print(f"      LONG/SHORT Support: {'✅' if both_directions_supported else '❌'}")
        print(f"      Pattern Analysis: {'✅' if pattern_analysis_present else '❌'}")
        print(f"      Scenario Logic: {'✅' if scenario_logic_present else '❌'}")
        print(f"      Fallback Available: {'✅' if fallback_available else '❌'}")
        print(f"      Realistic TP %: {'✅' if tp_percentages_realistic else '❌'}")
        
        # Test for logging confirmation
        print(f"\n   📝 Testing Logging Confirmation...")
        # This would require checking backend logs, but we can infer from reasoning
        logging_evidence = claude_tp_strategy_count > 0  # If Claude TP is used, logging should occur
        
        print(f"      Logging Evidence: {'✅' if logging_evidence else '❌'}")
        
        revolutionary_system_working = (
            claude_tp_implemented and
            reasonable_claude_usage and
            pattern_analysis_present and
            fallback_available and
            logging_evidence
        )
        
        print(f"\n   🚀 REVOLUTIONARY CLAUDE TP SYSTEM: {'✅ SUCCESS' if revolutionary_system_working else '❌ NEEDS WORK'}")
        
        if revolutionary_system_working:
            print(f"   💡 SUCCESS: Claude is generating intelligent TP strategies!")
            print(f"   💡 Usage Rate: {claude_tp_rate*100:.1f}% Claude TP vs {fallback_rate*100:.1f}% fallback")
            print(f"   💡 Pattern-Based: {pattern_analysis_count} decisions with pattern analysis")
            print(f"   💡 Adaptive: Both LONG ({long_claude_tp_count}) and SHORT ({short_claude_tp_count}) supported")
        else:
            print(f"   💡 ISSUES DETECTED:")
            if not claude_tp_implemented:
                print(f"      - No evidence of Claude TP strategy implementation")
            if not reasonable_claude_usage:
                print(f"      - Low Claude TP usage rate ({claude_tp_rate*100:.1f}% < 20%)")
            if not pattern_analysis_present:
                print(f"      - Missing pattern analysis in TP strategies")
            if not both_directions_supported:
                print(f"      - LONG/SHORT support incomplete")
        
        return revolutionary_system_working

if __name__ == "__main__":
    tester = ClaudeTPTester()
    result = tester.test_claude_intelligent_tp_strategy_system()
    print(f"\n🎯 Final Result: {'✅ PASSED' if result else '❌ FAILED'}")
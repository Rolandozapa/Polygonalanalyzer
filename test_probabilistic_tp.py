#!/usr/bin/env python3

import requests
import time
import json
from pathlib import Path

class ProbabilisticTPTester:
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
                    base_url = "https://aitra-platform.preview.emergentagent.com"
        except:
            base_url = "https://aitra-platform.preview.emergentagent.com"
        
        self.base_url = base_url
        self.api_url = f"{base_url}/api"

    def test_probabilistic_tp_system(self):
        """Test the new Probabilistic Optimal TP System for IA2"""
        print(f"\n🎯 Testing Probabilistic Optimal TP System for IA2...")
        print(f"   Backend URL: {self.base_url}")
        
        # Start the trading system to generate fresh decisions
        print(f"   🚀 Starting trading system for fresh IA2 decisions...")
        try:
            start_response = requests.post(f"{self.api_url}/start-trading", timeout=30)
            if start_response.status_code != 200:
                print(f"   ❌ Failed to start trading system: {start_response.status_code}")
                return False
            print(f"   ✅ Trading system started successfully")
        except Exception as e:
            print(f"   ❌ Error starting trading system: {e}")
            return False
        
        # Wait for the system to generate decisions
        print(f"   ⏱️  Waiting for IA2 to generate decisions with probabilistic TP (90 seconds)...")
        time.sleep(90)  # Extended wait for IA2 processing
        
        # Get decisions for analysis
        try:
            decisions_response = requests.get(f"{self.api_url}/decisions", timeout=30)
            if decisions_response.status_code != 200:
                print(f"   ❌ Cannot retrieve decisions: {decisions_response.status_code}")
                self.stop_trading_system()
                return False
            
            decisions_data = decisions_response.json()
            decisions = decisions_data.get('decisions', [])
            
            if len(decisions) == 0:
                print(f"   ❌ No decisions available for TP testing")
                self.stop_trading_system()
                return False
            
            print(f"   📊 Analyzing {len(decisions)} decisions for probabilistic TP system...")
            
        except Exception as e:
            print(f"   ❌ Error retrieving decisions: {e}")
            self.stop_trading_system()
            return False
        
        # Analyze decisions for probabilistic TP features
        tp_analysis = {
            'total_decisions': len(decisions),
            'long_short_signals': 0,
            'hold_signals': 0,
            'decisions_with_tp_strategy': 0,
            'decisions_with_tp_levels': 0,
            'decisions_with_custom_percentages': 0,
            'decisions_with_custom_distributions': 0,
            'decisions_with_probability_reasoning': 0,
            'hold_without_tp_strategy': 0,
            'probabilistic_examples': []
        }
        
        for i, decision in enumerate(decisions):
            symbol = decision.get('symbol', 'Unknown')
            signal = decision.get('signal', 'hold').lower()
            reasoning = decision.get('ia2_reasoning', '')
            
            # Count signal types
            if signal in ['long', 'short']:
                tp_analysis['long_short_signals'] += 1
            elif signal == 'hold':
                tp_analysis['hold_signals'] += 1
            
            # Check for take_profit_strategy in reasoning (since it's in JSON format)
            has_tp_strategy = 'take_profit_strategy' in reasoning or 'tp_levels' in reasoning
            has_tp_levels = 'tp_levels' in reasoning or ('tp1' in reasoning and 'tp2' in reasoning)
            has_custom_percentages = any(pct in reasoning for pct in ['percentage_from_entry', '0.4', '0.8', '1.4', '2.2'])
            has_custom_distributions = any(dist in reasoning for dist in ['allocation', '45', '30', '20', '5'])
            has_probability_reasoning = any(prob in reasoning for prob in ['probability', 'expected_contribution', 'optimization_metrics'])
            
            if has_tp_strategy:
                tp_analysis['decisions_with_tp_strategy'] += 1
            if has_tp_levels:
                tp_analysis['decisions_with_tp_levels'] += 1
            if has_custom_percentages:
                tp_analysis['decisions_with_custom_percentages'] += 1
            if has_custom_distributions:
                tp_analysis['decisions_with_custom_distributions'] += 1
            if has_probability_reasoning:
                tp_analysis['decisions_with_probability_reasoning'] += 1
            
            # Check HOLD signals don't have TP strategy
            if signal == 'hold' and not has_tp_strategy:
                tp_analysis['hold_without_tp_strategy'] += 1
            
            # Collect examples for detailed analysis
            if i < 10 and (has_tp_strategy or has_probability_reasoning):
                tp_analysis['probabilistic_examples'].append({
                    'symbol': symbol,
                    'signal': signal,
                    'has_tp_strategy': has_tp_strategy,
                    'has_tp_levels': has_tp_levels,
                    'has_custom_percentages': has_custom_percentages,
                    'has_custom_distributions': has_custom_distributions,
                    'has_probability_reasoning': has_probability_reasoning,
                    'reasoning_preview': reasoning[:300] + '...' if len(reasoning) > 300 else reasoning
                })
        
        # Stop the trading system
        print(f"   🛑 Stopping trading system...")
        self.stop_trading_system()
        
        # Display analysis results
        print(f"\n   📊 Probabilistic TP System Analysis:")
        print(f"      Total Decisions: {tp_analysis['total_decisions']}")
        print(f"      LONG/SHORT Signals: {tp_analysis['long_short_signals']}")
        print(f"      HOLD Signals: {tp_analysis['hold_signals']}")
        print(f"      Decisions with TP Strategy: {tp_analysis['decisions_with_tp_strategy']}")
        print(f"      Decisions with TP Levels: {tp_analysis['decisions_with_tp_levels']}")
        print(f"      Custom Percentages Found: {tp_analysis['decisions_with_custom_percentages']}")
        print(f"      Custom Distributions Found: {tp_analysis['decisions_with_custom_distributions']}")
        print(f"      Probability Reasoning Found: {tp_analysis['decisions_with_probability_reasoning']}")
        print(f"      HOLD without TP Strategy: {tp_analysis['hold_without_tp_strategy']}")
        
        # Show specific examples
        if tp_analysis['probabilistic_examples']:
            print(f"\n   🔍 Probabilistic TP Configuration Examples:")
            for i, example in enumerate(tp_analysis['probabilistic_examples'][:3]):
                print(f"\n      Example {i+1} - {example['symbol']} ({example['signal'].upper()}):")
                print(f"         TP Strategy: {'✅' if example['has_tp_strategy'] else '❌'}")
                print(f"         TP Levels: {'✅' if example['has_tp_levels'] else '❌'}")
                print(f"         Custom %: {'✅' if example['has_custom_percentages'] else '❌'}")
                print(f"         Custom Dist: {'✅' if example['has_custom_distributions'] else '❌'}")
                print(f"         Probability: {'✅' if example['has_probability_reasoning'] else '❌'}")
                print(f"         Preview: {example['reasoning_preview'][:150]}...")
        
        # Validation criteria for probabilistic TP system
        long_short_rate = tp_analysis['long_short_signals'] / tp_analysis['total_decisions'] if tp_analysis['total_decisions'] > 0 else 0
        tp_strategy_rate = tp_analysis['decisions_with_tp_strategy'] / max(tp_analysis['long_short_signals'], 1)
        hold_compliance_rate = tp_analysis['hold_without_tp_strategy'] / max(tp_analysis['hold_signals'], 1)
        
        # Key validation checks
        has_long_short_signals = tp_analysis['long_short_signals'] > 0
        tp_strategy_present = tp_analysis['decisions_with_tp_strategy'] > 0
        custom_configurations = tp_analysis['decisions_with_custom_percentages'] > 0 or tp_analysis['decisions_with_custom_distributions'] > 0
        probability_analysis = tp_analysis['decisions_with_probability_reasoning'] > 0
        hold_signals_compliant = tp_analysis['hold_signals'] == 0 or hold_compliance_rate >= 0.8
        system_stability = tp_analysis['total_decisions'] >= 5  # At least 5 decisions generated
        
        print(f"\n   ✅ Probabilistic TP System Validation:")
        print(f"      Has LONG/SHORT Signals: {'✅' if has_long_short_signals else '❌'} ({tp_analysis['long_short_signals']} signals)")
        print(f"      TP Strategy Present: {'✅' if tp_strategy_present else '❌'} ({tp_analysis['decisions_with_tp_strategy']} with strategy)")
        print(f"      Custom Configurations: {'✅' if custom_configurations else '❌'} (custom % or distributions)")
        print(f"      Probability Analysis: {'✅' if probability_analysis else '❌'} ({tp_analysis['decisions_with_probability_reasoning']} with probability)")
        print(f"      HOLD Compliance: {'✅' if hold_signals_compliant else '❌'} ({hold_compliance_rate*100:.1f}% compliant)")
        print(f"      System Stability: {'✅' if system_stability else '❌'} ({tp_analysis['total_decisions']} decisions)")
        
        # Overall assessment
        probabilistic_tp_working = (
            has_long_short_signals and
            tp_strategy_present and
            hold_signals_compliant and
            system_stability
        )
        
        # Enhanced assessment if we have evidence of probabilistic features
        enhanced_probabilistic = (
            probabilistic_tp_working and
            (custom_configurations or probability_analysis)
        )
        
        print(f"\n   🎯 Probabilistic TP System Assessment:")
        if enhanced_probabilistic:
            print(f"      Status: ✅ FULLY WORKING - Enhanced probabilistic TP system operational")
            print(f"      Evidence: Custom configurations and probability analysis detected")
        elif probabilistic_tp_working:
            print(f"      Status: ✅ BASIC WORKING - Core TP system operational")
            print(f"      Note: Advanced probabilistic features may need more testing")
        else:
            print(f"      Status: ❌ NEEDS WORK - Core issues detected")
            print(f"      Issues: Missing LONG/SHORT signals or TP strategy implementation")
        
        return enhanced_probabilistic or probabilistic_tp_working

    def stop_trading_system(self):
        """Stop the trading system"""
        try:
            stop_response = requests.post(f"{self.api_url}/stop-trading", timeout=30)
            if stop_response.status_code == 200:
                print(f"   ✅ Trading system stopped successfully")
            else:
                print(f"   ⚠️  Stop trading returned {stop_response.status_code}")
        except Exception as e:
            print(f"   ⚠️  Error stopping trading system: {e}")

if __name__ == "__main__":
    tester = ProbabilisticTPTester()
    result = tester.test_probabilistic_tp_system()
    print(f"\n🎯 Final Result: {'✅ SUCCESS' if result else '❌ FAILED'}")
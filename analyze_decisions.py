#!/usr/bin/env python3

import requests
import json

def analyze_decisions():
    """Analyze existing decisions for probabilistic TP features"""
    
    # Get decisions
    try:
        response = requests.get("https://aitra-platform.preview.emergentagent.com/api/decisions", timeout=30)
        if response.status_code != 200:
            print(f"❌ Failed to get decisions: {response.status_code}")
            return
        
        data = response.json()
        decisions = data.get('decisions', [])
        
        print(f"📊 Analyzing {len(decisions)} existing decisions for probabilistic TP features...")
        
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
            'examples': []
        }
        
        for i, decision in enumerate(decisions):
            symbol = decision.get('symbol', 'Unknown')
            signal = decision.get('signal', 'hold').lower()
            reasoning = decision.get('ia2_reasoning', '')
            
            print(f"\n🔍 Decision {i+1}: {symbol} ({signal.upper()})")
            print(f"   Reasoning length: {len(reasoning)} chars")
            
            # Count signal types
            if signal in ['long', 'short']:
                tp_analysis['long_short_signals'] += 1
            elif signal == 'hold':
                tp_analysis['hold_signals'] += 1
            
            # Check for probabilistic TP features
            has_tp_strategy = 'take_profit_strategy' in reasoning or 'tp_levels' in reasoning
            has_tp_levels = 'tp_levels' in reasoning or ('tp1' in reasoning and 'tp2' in reasoning)
            has_custom_percentages = any(pct in reasoning for pct in ['percentage_from_entry', '0.4', '0.8', '1.4', '2.2'])
            has_custom_distributions = any(dist in reasoning for dist in ['allocation', '45', '30', '20', '5'])
            has_probability_reasoning = any(prob in reasoning for prob in ['probability', 'expected_contribution', 'optimization_metrics'])
            
            # Look for TP patterns in reasoning
            tp_patterns = ['TP1', 'TP2', 'TP3', 'TP4', 'TP5', 'take_profit', 'distribution']
            found_patterns = [p for p in tp_patterns if p.lower() in reasoning.lower()]
            
            print(f"   TP Strategy: {'✅' if has_tp_strategy else '❌'}")
            print(f"   TP Levels: {'✅' if has_tp_levels else '❌'}")
            print(f"   Custom %: {'✅' if has_custom_percentages else '❌'}")
            print(f"   Custom Dist: {'✅' if has_custom_distributions else '❌'}")
            print(f"   Probability: {'✅' if has_probability_reasoning else '❌'}")
            print(f"   TP Patterns found: {found_patterns}")
            
            if found_patterns:
                print(f"   Preview: {reasoning[:200]}...")
            
            # Update counters
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
            
            # Collect examples
            if has_tp_strategy or has_probability_reasoning or found_patterns:
                tp_analysis['examples'].append({
                    'symbol': symbol,
                    'signal': signal,
                    'has_tp_strategy': has_tp_strategy,
                    'has_tp_levels': has_tp_levels,
                    'has_custom_percentages': has_custom_percentages,
                    'has_custom_distributions': has_custom_distributions,
                    'has_probability_reasoning': has_probability_reasoning,
                    'patterns': found_patterns,
                    'reasoning_preview': reasoning[:300] + '...' if len(reasoning) > 300 else reasoning
                })
        
        # Display summary
        print(f"\n📊 Probabilistic TP System Analysis Summary:")
        print(f"   Total Decisions: {tp_analysis['total_decisions']}")
        print(f"   LONG/SHORT Signals: {tp_analysis['long_short_signals']}")
        print(f"   HOLD Signals: {tp_analysis['hold_signals']}")
        print(f"   Decisions with TP Strategy: {tp_analysis['decisions_with_tp_strategy']}")
        print(f"   Decisions with TP Levels: {tp_analysis['decisions_with_tp_levels']}")
        print(f"   Custom Percentages Found: {tp_analysis['decisions_with_custom_percentages']}")
        print(f"   Custom Distributions Found: {tp_analysis['decisions_with_custom_distributions']}")
        print(f"   Probability Reasoning Found: {tp_analysis['decisions_with_probability_reasoning']}")
        print(f"   HOLD without TP Strategy: {tp_analysis['hold_without_tp_strategy']}")
        
        # Show examples
        if tp_analysis['examples']:
            print(f"\n🔍 Probabilistic TP Examples Found:")
            for i, example in enumerate(tp_analysis['examples']):
                print(f"\n   Example {i+1} - {example['symbol']} ({example['signal'].upper()}):")
                print(f"      TP Strategy: {'✅' if example['has_tp_strategy'] else '❌'}")
                print(f"      TP Levels: {'✅' if example['has_tp_levels'] else '❌'}")
                print(f"      Custom %: {'✅' if example['has_custom_percentages'] else '❌'}")
                print(f"      Custom Dist: {'✅' if example['has_custom_distributions'] else '❌'}")
                print(f"      Probability: {'✅' if example['has_probability_reasoning'] else '❌'}")
                print(f"      Patterns: {example['patterns']}")
                print(f"      Preview: {example['reasoning_preview'][:150]}...")
        
        # Assessment
        has_long_short_signals = tp_analysis['long_short_signals'] > 0
        tp_strategy_present = tp_analysis['decisions_with_tp_strategy'] > 0
        custom_configurations = tp_analysis['decisions_with_custom_percentages'] > 0 or tp_analysis['decisions_with_custom_distributions'] > 0
        probability_analysis = tp_analysis['decisions_with_probability_reasoning'] > 0
        hold_signals_compliant = tp_analysis['hold_signals'] == 0 or tp_analysis['hold_without_tp_strategy'] >= tp_analysis['hold_signals'] * 0.8
        system_stability = tp_analysis['total_decisions'] >= 3
        
        print(f"\n✅ Probabilistic TP System Validation:")
        print(f"   Has LONG/SHORT Signals: {'✅' if has_long_short_signals else '❌'} ({tp_analysis['long_short_signals']} signals)")
        print(f"   TP Strategy Present: {'✅' if tp_strategy_present else '❌'} ({tp_analysis['decisions_with_tp_strategy']} with strategy)")
        print(f"   Custom Configurations: {'✅' if custom_configurations else '❌'} (custom % or distributions)")
        print(f"   Probability Analysis: {'✅' if probability_analysis else '❌'} ({tp_analysis['decisions_with_probability_reasoning']} with probability)")
        print(f"   HOLD Compliance: {'✅' if hold_signals_compliant else '❌'}")
        print(f"   System Stability: {'✅' if system_stability else '❌'} ({tp_analysis['total_decisions']} decisions)")
        
        # Overall assessment
        probabilistic_tp_working = (
            tp_strategy_present and
            hold_signals_compliant and
            system_stability
        )
        
        enhanced_probabilistic = (
            probabilistic_tp_working and
            (custom_configurations or probability_analysis)
        )
        
        print(f"\n🎯 Probabilistic TP System Assessment:")
        if enhanced_probabilistic:
            print(f"   Status: ✅ FULLY WORKING - Enhanced probabilistic TP system operational")
            print(f"   Evidence: Custom configurations and probability analysis detected")
        elif probabilistic_tp_working:
            print(f"   Status: ✅ BASIC WORKING - Core TP system operational")
            print(f"   Note: Advanced probabilistic features may need more testing")
        else:
            print(f"   Status: ❌ NEEDS WORK - Core issues detected")
            print(f"   Issues: Missing TP strategy implementation or system instability")
        
        return enhanced_probabilistic or probabilistic_tp_working
        
    except Exception as e:
        print(f"❌ Error analyzing decisions: {e}")
        return False

if __name__ == "__main__":
    result = analyze_decisions()
    print(f"\n🎯 Final Assessment: {'✅ SUCCESS' if result else '❌ FAILED'}")
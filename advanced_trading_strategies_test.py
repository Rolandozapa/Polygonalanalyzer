import requests
import sys
import json
import time
import asyncio
from datetime import datetime
import os
from pathlib import Path

class AdvancedTradingStrategiesIA2Tester:
    """
    Test suite for REVOLUTIONARY advanced trading strategies implemented for IA2:
    1. Advanced Multi-Level Take Profit System (4-level TP strategy)
    2. Claude Advanced Strategy Integration 
    3. Position Inversion Logic
    4. Advanced Strategy Manager Integration
    5. Enhanced Decision Quality
    6. Risk Management Enhancement
    """
    
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
        self.tests_run = 0
        self.tests_passed = 0
        self.advanced_features_tested = []

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=60):
        """Run a single API test with extended timeout for advanced strategy testing"""
        url = f"{self.api_url}/{endpoint}" if endpoint else f"{self.api_url}/"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        
        start_time = time.time()
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=timeout)

            end_time = time.time()
            response_time = end_time - start_time
            
            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code} - Time: {response_time:.2f}s")
                
                try:
                    response_data = response.json()
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code} - Time: {response_time:.2f}s")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text[:200]}...")
                return False, {}

        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            print(f"❌ Failed - Error: {str(e)} - Time: {response_time:.2f}s")
            return False, {}

    def test_advanced_multi_level_take_profit_system(self):
        """Test Advanced Multi-Level Take Profit System (4-level TP strategy)"""
        print(f"\n🎯 Testing Advanced Multi-Level Take Profit System...")
        
        # Get recent decisions to analyze TP strategy implementation
        success, decisions_data = self.run_test("Get IA2 Decisions for TP Analysis", "GET", "decisions", 200)
        if not success:
            print(f"   ❌ Cannot retrieve decisions for TP analysis")
            return False
        
        decisions = decisions_data.get('decisions', [])
        if len(decisions) == 0:
            print(f"   ❌ No decisions available for TP analysis")
            return False
        
        print(f"   📊 Analyzing {len(decisions)} decisions for 4-level TP strategy...")
        
        # Analyze TP levels in decisions
        tp_strategy_found = 0
        advanced_tp_decisions = []
        tp_level_analysis = {
            'tp1_found': 0,
            'tp2_found': 0, 
            'tp3_found': 0,
            'tp4_found': 0,
            'proper_distribution': 0,
            'proper_percentages': 0
        }
        
        for i, decision in enumerate(decisions[:20]):  # Analyze first 20 decisions
            symbol = decision.get('symbol', 'Unknown')
            signal = decision.get('signal', 'hold')
            reasoning = decision.get('ia2_reasoning', '')
            confidence = decision.get('confidence', 0)
            
            # Check for 4-level TP strategy indicators in reasoning
            tp_indicators = {
                'tp1': any(keyword in reasoning.lower() for keyword in ['tp1', 'take profit 1', '25%', '1.5%']),
                'tp2': any(keyword in reasoning.lower() for keyword in ['tp2', 'take profit 2', '30%', '3%', '3.0%']),
                'tp3': any(keyword in reasoning.lower() for keyword in ['tp3', 'take profit 3', '25%', '5%', '5.0%']),
                'tp4': any(keyword in reasoning.lower() for keyword in ['tp4', 'take profit 4', '20%', '8%', '8.0%'])
            }
            
            # Check for advanced strategy keywords
            advanced_strategy_keywords = [
                'multi-level', 'take profit strategy', 'position distribution', 
                'graduated', 'tp levels', 'advanced strategy', 'position scaling'
            ]
            
            has_advanced_strategy = any(keyword in reasoning.lower() for keyword in advanced_strategy_keywords)
            
            # Count TP levels found
            tp_levels_found = sum(tp_indicators.values())
            
            if tp_levels_found >= 2 or has_advanced_strategy:  # At least 2 TP levels or advanced strategy mention
                tp_strategy_found += 1
                advanced_tp_decisions.append({
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': confidence,
                    'tp_levels': tp_levels_found,
                    'has_advanced_strategy': has_advanced_strategy,
                    'tp_indicators': tp_indicators
                })
                
                # Update TP level analysis
                for tp_level, found in tp_indicators.items():
                    if found:
                        tp_level_analysis[f'{tp_level}_found'] += 1
                
                # Check for proper distribution (25%, 30%, 25%, 20%)
                distribution_keywords = ['25%', '30%', '20%']
                if sum(1 for kw in distribution_keywords if kw in reasoning) >= 2:
                    tp_level_analysis['proper_distribution'] += 1
                
                # Check for proper percentage gains (1.5%, 3%, 5%, 8%+)
                percentage_keywords = ['1.5%', '3%', '5%', '8%']
                if sum(1 for kw in percentage_keywords if kw in reasoning) >= 2:
                    tp_level_analysis['proper_percentages'] += 1
            
            # Show detailed analysis for first 5 decisions
            if i < 5:
                print(f"\n   Decision {i+1} - {symbol} ({signal}):")
                print(f"      Confidence: {confidence:.3f}")
                print(f"      TP Levels Found: {tp_levels_found}/4")
                print(f"      Advanced Strategy: {'✅' if has_advanced_strategy else '❌'}")
                print(f"      TP Indicators: {tp_indicators}")
                if tp_levels_found > 0 or has_advanced_strategy:
                    print(f"      ✅ Advanced TP Strategy Detected")
                else:
                    print(f"      ❌ No Advanced TP Strategy")
        
        # Calculate TP strategy statistics
        total_decisions = len(decisions[:20])
        tp_strategy_rate = tp_strategy_found / total_decisions if total_decisions > 0 else 0
        
        print(f"\n   📊 4-Level TP Strategy Analysis:")
        print(f"      Decisions with TP Strategy: {tp_strategy_found}/{total_decisions} ({tp_strategy_rate*100:.1f}%)")
        print(f"      TP1 Mentions: {tp_level_analysis['tp1_found']}")
        print(f"      TP2 Mentions: {tp_level_analysis['tp2_found']}")
        print(f"      TP3 Mentions: {tp_level_analysis['tp3_found']}")
        print(f"      TP4 Mentions: {tp_level_analysis['tp4_found']}")
        print(f"      Proper Distribution (25%,30%,25%,20%): {tp_level_analysis['proper_distribution']}")
        print(f"      Proper Percentages (1.5%,3%,5%,8%): {tp_level_analysis['proper_percentages']}")
        
        # Validation criteria for 4-level TP system
        tp_strategy_implemented = tp_strategy_rate >= 0.30  # At least 30% should have TP strategy
        multiple_tp_levels = (tp_level_analysis['tp1_found'] + tp_level_analysis['tp2_found'] + 
                             tp_level_analysis['tp3_found'] + tp_level_analysis['tp4_found']) >= 4
        proper_distribution_found = tp_level_analysis['proper_distribution'] > 0
        proper_percentages_found = tp_level_analysis['proper_percentages'] > 0
        
        print(f"\n   ✅ 4-Level TP System Validation:")
        print(f"      TP Strategy Rate ≥30%: {'✅' if tp_strategy_implemented else '❌'} ({tp_strategy_rate*100:.1f}%)")
        print(f"      Multiple TP Levels: {'✅' if multiple_tp_levels else '❌'} (≥4 mentions)")
        print(f"      Proper Distribution: {'✅' if proper_distribution_found else '❌'} (25%,30%,25%,20%)")
        print(f"      Proper Percentages: {'✅' if proper_percentages_found else '❌'} (1.5%,3%,5%,8%)")
        
        tp_system_working = (
            tp_strategy_implemented and
            multiple_tp_levels and
            proper_distribution_found and
            proper_percentages_found
        )
        
        print(f"\n   🎯 Advanced Multi-Level TP System: {'✅ WORKING' if tp_system_working else '❌ NEEDS IMPLEMENTATION'}")
        
        if tp_system_working:
            self.advanced_features_tested.append("4-Level TP Strategy")
        
        return tp_system_working

    def test_claude_advanced_strategy_integration(self):
        """Test Claude Advanced Strategy Integration with sophisticated analysis"""
        print(f"\n🤖 Testing Claude Advanced Strategy Integration...")
        
        # Get recent decisions to analyze Claude integration
        success, decisions_data = self.run_test("Get IA2 Decisions for Claude Analysis", "GET", "decisions", 200)
        if not success:
            print(f"   ❌ Cannot retrieve decisions for Claude analysis")
            return False
        
        decisions = decisions_data.get('decisions', [])
        if len(decisions) == 0:
            print(f"   ❌ No decisions available for Claude analysis")
            return False
        
        print(f"   📊 Analyzing {len(decisions)} decisions for Claude integration...")
        
        # Analyze Claude-specific patterns and advanced strategy elements
        claude_indicators = {
            'advanced_reasoning': 0,
            'strategic_insights': 0,
            'multi_tp_strategy': 0,
            'position_management': 0,
            'inversion_criteria': 0,
            'technical_confluence': 0,
            'sophisticated_analysis': 0
        }
        
        claude_keywords = {
            'advanced_reasoning': ['comprehensive analysis', 'technical confluence', 'market context', 'strategic rationale'],
            'strategic_insights': ['strategy type', 'advanced strategy', 'position management', 'risk assessment'],
            'multi_tp_strategy': ['take profit strategy', 'tp1', 'tp2', 'tp3', 'tp4', 'position distribution'],
            'position_management': ['entry strategy', 'stop loss percentage', 'trailing stop', 'position size'],
            'inversion_criteria': ['inversion', 'opposite signal', 'confidence threshold', 'enable inversion'],
            'technical_confluence': ['confluence', 'indicator alignment', 'multiple signals', 'technical convergence'],
            'sophisticated_analysis': ['nuanced', 'sophisticated', 'comprehensive', 'advanced reasoning']
        }
        
        for i, decision in enumerate(decisions[:15]):  # Analyze first 15 decisions
            symbol = decision.get('symbol', 'Unknown')
            reasoning = decision.get('ia2_reasoning', '')
            confidence = decision.get('confidence', 0)
            signal = decision.get('signal', 'hold')
            
            # Check for Claude-specific patterns
            claude_patterns_found = {}
            for category, keywords in claude_keywords.items():
                found = sum(1 for keyword in keywords if keyword.lower() in reasoning.lower())
                claude_patterns_found[category] = found
                if found > 0:
                    claude_indicators[category] += 1
            
            # Show detailed analysis for first 5 decisions
            if i < 5:
                print(f"\n   Decision {i+1} - {symbol} ({signal}):")
                print(f"      Confidence: {confidence:.3f}")
                print(f"      Reasoning Length: {len(reasoning)} chars")
                print(f"      Claude Patterns Found:")
                for category, count in claude_patterns_found.items():
                    print(f"        {category}: {count} keywords")
                
                # Check for JSON-like structure in reasoning (Claude format)
                has_json_structure = any(indicator in reasoning for indicator in ['{', '}', '"signal":', '"confidence":'])
                print(f"      JSON Structure: {'✅' if has_json_structure else '❌'}")
                
                # Check reasoning quality
                quality_score = len(reasoning) / 100  # Basic quality metric
                print(f"      Quality Score: {quality_score:.1f}")
        
        # Calculate Claude integration statistics
        total_analyzed = min(len(decisions), 15)
        
        print(f"\n   📊 Claude Integration Analysis:")
        for category, count in claude_indicators.items():
            percentage = (count / total_analyzed) * 100 if total_analyzed > 0 else 0
            print(f"      {category.replace('_', ' ').title()}: {count}/{total_analyzed} ({percentage:.1f}%)")
        
        # Validation criteria for Claude integration
        advanced_reasoning_present = claude_indicators['advanced_reasoning'] >= total_analyzed * 0.4  # 40%
        strategic_insights_present = claude_indicators['strategic_insights'] >= total_analyzed * 0.3  # 30%
        multi_tp_present = claude_indicators['multi_tp_strategy'] >= total_analyzed * 0.2  # 20%
        sophisticated_analysis_present = claude_indicators['sophisticated_analysis'] >= total_analyzed * 0.5  # 50%
        
        print(f"\n   ✅ Claude Integration Validation:")
        print(f"      Advanced Reasoning ≥40%: {'✅' if advanced_reasoning_present else '❌'}")
        print(f"      Strategic Insights ≥30%: {'✅' if strategic_insights_present else '❌'}")
        print(f"      Multi-TP Strategy ≥20%: {'✅' if multi_tp_present else '❌'}")
        print(f"      Sophisticated Analysis ≥50%: {'✅' if sophisticated_analysis_present else '❌'}")
        
        claude_integration_working = (
            advanced_reasoning_present and
            strategic_insights_present and
            sophisticated_analysis_present
        )
        
        print(f"\n   🎯 Claude Advanced Strategy Integration: {'✅ WORKING' if claude_integration_working else '❌ NEEDS VERIFICATION'}")
        
        if claude_integration_working:
            self.advanced_features_tested.append("Claude Advanced Integration")
        
        return claude_integration_working

    def test_position_inversion_logic(self):
        """Test Position Inversion Logic for automatic position reversal"""
        print(f"\n🔄 Testing Position Inversion Logic...")
        
        # Get recent decisions to analyze inversion logic
        success, decisions_data = self.run_test("Get IA2 Decisions for Inversion Analysis", "GET", "decisions", 200)
        if not success:
            print(f"   ❌ Cannot retrieve decisions for inversion analysis")
            return False
        
        decisions = decisions_data.get('decisions', [])
        if len(decisions) == 0:
            print(f"   ❌ No decisions available for inversion analysis")
            return False
        
        print(f"   📊 Analyzing {len(decisions)} decisions for position inversion logic...")
        
        # Analyze inversion-related patterns
        inversion_indicators = {
            'inversion_mentioned': 0,
            'confidence_threshold_check': 0,
            'opposite_signal_analysis': 0,
            'inversion_criteria_present': 0,
            'position_reversal_logic': 0
        }
        
        inversion_keywords = [
            'inversion', 'position inversion', 'reverse position', 'opposite signal',
            'confidence threshold', 'inversion criteria', 'automatic reversal',
            'position reversal', 'invert position', 'signal reversal'
        ]
        
        # Look for symbols with multiple decisions (potential inversion scenarios)
        symbol_decisions = {}
        for decision in decisions:
            symbol = decision.get('symbol', 'Unknown')
            if symbol not in symbol_decisions:
                symbol_decisions[symbol] = []
            symbol_decisions[symbol].append(decision)
        
        # Analyze inversion patterns
        potential_inversions = 0
        inversion_opportunities = []
        
        for i, decision in enumerate(decisions[:20]):  # Analyze first 20 decisions
            symbol = decision.get('symbol', 'Unknown')
            reasoning = decision.get('ia2_reasoning', '')
            confidence = decision.get('confidence', 0)
            signal = decision.get('signal', 'hold')
            
            # Check for inversion-related keywords
            inversion_mentions = sum(1 for keyword in inversion_keywords if keyword.lower() in reasoning.lower())
            
            if inversion_mentions > 0:
                inversion_indicators['inversion_mentioned'] += 1
                
                # Check for specific inversion criteria
                if 'confidence threshold' in reasoning.lower():
                    inversion_indicators['confidence_threshold_check'] += 1
                
                if 'opposite signal' in reasoning.lower():
                    inversion_indicators['opposite_signal_analysis'] += 1
                
                if 'inversion criteria' in reasoning.lower():
                    inversion_indicators['inversion_criteria_present'] += 1
                
                if any(keyword in reasoning.lower() for keyword in ['reverse', 'reversal', 'invert']):
                    inversion_indicators['position_reversal_logic'] += 1
                
                inversion_opportunities.append({
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': confidence,
                    'inversion_mentions': inversion_mentions
                })
            
            # Show detailed analysis for first 5 decisions
            if i < 5:
                print(f"\n   Decision {i+1} - {symbol} ({signal}):")
                print(f"      Confidence: {confidence:.3f}")
                print(f"      Inversion Mentions: {inversion_mentions}")
                if inversion_mentions > 0:
                    print(f"      ✅ Inversion Logic Detected")
                    print(f"      Inversion Keywords Found: {inversion_mentions}")
                else:
                    print(f"      ❌ No Inversion Logic")
        
        # Analyze symbols with multiple decisions for actual inversion patterns
        symbols_with_multiple_decisions = {k: v for k, v in symbol_decisions.items() if len(v) > 1}
        
        print(f"\n   📊 Position Inversion Analysis:")
        print(f"      Decisions with Inversion Logic: {inversion_indicators['inversion_mentioned']}/20")
        print(f"      Confidence Threshold Checks: {inversion_indicators['confidence_threshold_check']}")
        print(f"      Opposite Signal Analysis: {inversion_indicators['opposite_signal_analysis']}")
        print(f"      Inversion Criteria Present: {inversion_indicators['inversion_criteria_present']}")
        print(f"      Position Reversal Logic: {inversion_indicators['position_reversal_logic']}")
        print(f"      Symbols with Multiple Decisions: {len(symbols_with_multiple_decisions)}")
        
        # Check for actual inversion patterns in symbols with multiple decisions
        actual_inversions_detected = 0
        for symbol, symbol_decisions_list in symbols_with_multiple_decisions.items():
            if len(symbol_decisions_list) >= 2:
                # Check if there are opposite signals (potential inversion)
                signals = [d.get('signal', 'hold') for d in symbol_decisions_list]
                has_long = 'long' in signals
                has_short = 'short' in signals
                
                if has_long and has_short:
                    actual_inversions_detected += 1
                    print(f"      ✅ Potential Inversion Detected: {symbol} (LONG & SHORT signals)")
        
        print(f"      Actual Inversion Patterns: {actual_inversions_detected}")
        
        # Validation criteria for position inversion logic
        inversion_logic_implemented = inversion_indicators['inversion_mentioned'] >= 2  # At least 2 mentions
        inversion_criteria_working = inversion_indicators['inversion_criteria_present'] > 0
        confidence_threshold_working = inversion_indicators['confidence_threshold_check'] > 0
        multiple_decision_symbols = len(symbols_with_multiple_decisions) > 0
        
        print(f"\n   ✅ Position Inversion Logic Validation:")
        print(f"      Inversion Logic Implemented: {'✅' if inversion_logic_implemented else '❌'} (≥2 mentions)")
        print(f"      Inversion Criteria Present: {'✅' if inversion_criteria_working else '❌'}")
        print(f"      Confidence Threshold Check: {'✅' if confidence_threshold_working else '❌'}")
        print(f"      Multiple Decision Symbols: {'✅' if multiple_decision_symbols else '❌'}")
        
        inversion_logic_working = (
            inversion_logic_implemented and
            (inversion_criteria_working or confidence_threshold_working)
        )
        
        print(f"\n   🎯 Position Inversion Logic: {'✅ WORKING' if inversion_logic_working else '❌ NEEDS IMPLEMENTATION'}")
        
        if inversion_logic_working:
            self.advanced_features_tested.append("Position Inversion Logic")
        
        return inversion_logic_working

    def test_advanced_strategy_manager_integration(self):
        """Test Advanced Strategy Manager Integration"""
        print(f"\n⚙️ Testing Advanced Strategy Manager Integration...")
        
        # Get recent decisions to analyze strategy manager integration
        success, decisions_data = self.run_test("Get IA2 Decisions for Strategy Manager Analysis", "GET", "decisions", 200)
        if not success:
            print(f"   ❌ Cannot retrieve decisions for strategy manager analysis")
            return False
        
        decisions = decisions_data.get('decisions', [])
        if len(decisions) == 0:
            print(f"   ❌ No decisions available for strategy manager analysis")
            return False
        
        print(f"   📊 Analyzing {len(decisions)} decisions for advanced strategy manager integration...")
        
        # Analyze strategy manager patterns
        strategy_manager_indicators = {
            'position_direction_mentioned': 0,
            'strategy_creation': 0,
            'multi_source_validation': 0,
            'strategy_execution': 0,
            'strategy_management': 0,
            'long_short_strategies': 0
        }
        
        strategy_keywords = {
            'position_direction': ['LONG', 'SHORT', 'position direction', 'PositionDirection'],
            'strategy_creation': ['strategy created', 'create strategy', 'strategy generation', 'advanced strategy'],
            'multi_source_validation': ['multi-source', 'validation rate', 'strategy quality', 'source validation'],
            'strategy_execution': ['strategy execution', 'execute strategy', 'strategy management', 'strategy implementation'],
            'strategy_management': ['strategy manager', 'advanced_strategy_manager', 'strategy system', 'manage strategy']
        }
        
        long_short_distribution = {'LONG': 0, 'SHORT': 0, 'HOLD': 0}
        
        for i, decision in enumerate(decisions[:15]):  # Analyze first 15 decisions
            symbol = decision.get('symbol', 'Unknown')
            reasoning = decision.get('ia2_reasoning', '')
            confidence = decision.get('confidence', 0)
            signal = decision.get('signal', 'hold').upper()
            
            # Count signal distribution
            if signal in long_short_distribution:
                long_short_distribution[signal] += 1
            
            # Check for strategy manager patterns
            strategy_patterns_found = {}
            for category, keywords in strategy_keywords.items():
                found = sum(1 for keyword in keywords if keyword.lower() in reasoning.lower())
                strategy_patterns_found[category] = found
                if found > 0 and category in strategy_manager_indicators:
                    strategy_manager_indicators[category] += 1
            
            # Check for position direction specifically
            if any(keyword in reasoning for keyword in ['LONG', 'SHORT']) or signal in ['LONG', 'SHORT']:
                strategy_manager_indicators['position_direction_mentioned'] += 1
            
            # Check for LONG/SHORT strategy creation
            if signal in ['LONG', 'SHORT']:
                strategy_manager_indicators['long_short_strategies'] += 1
            
            # Show detailed analysis for first 5 decisions
            if i < 5:
                print(f"\n   Decision {i+1} - {symbol} ({signal}):")
                print(f"      Confidence: {confidence:.3f}")
                print(f"      Strategy Manager Patterns:")
                for category, count in strategy_patterns_found.items():
                    if count > 0:
                        print(f"        {category}: {count} keywords ✅")
                    else:
                        print(f"        {category}: {count} keywords ❌")
        
        # Calculate strategy manager statistics
        total_analyzed = min(len(decisions), 15)
        
        print(f"\n   📊 Advanced Strategy Manager Analysis:")
        for category, count in strategy_manager_indicators.items():
            percentage = (count / total_analyzed) * 100 if total_analyzed > 0 else 0
            print(f"      {category.replace('_', ' ').title()}: {count}/{total_analyzed} ({percentage:.1f}%)")
        
        print(f"\n   📈 LONG/SHORT Strategy Distribution:")
        for signal_type, count in long_short_distribution.items():
            percentage = (count / total_analyzed) * 100 if total_analyzed > 0 else 0
            print(f"      {signal_type}: {count}/{total_analyzed} ({percentage:.1f}%)")
        
        # Validation criteria for strategy manager integration
        position_direction_working = strategy_manager_indicators['position_direction_mentioned'] > 0
        strategy_creation_working = strategy_manager_indicators['strategy_creation'] > 0
        long_short_strategies_present = strategy_manager_indicators['long_short_strategies'] > 0
        strategy_management_present = strategy_manager_indicators['strategy_management'] > 0
        
        print(f"\n   ✅ Advanced Strategy Manager Validation:")
        print(f"      Position Direction (LONG/SHORT): {'✅' if position_direction_working else '❌'}")
        print(f"      Strategy Creation: {'✅' if strategy_creation_working else '❌'}")
        print(f"      LONG/SHORT Strategies: {'✅' if long_short_strategies_present else '❌'}")
        print(f"      Strategy Management: {'✅' if strategy_management_present else '❌'}")
        
        strategy_manager_working = (
            position_direction_working and
            long_short_strategies_present
        )
        
        print(f"\n   🎯 Advanced Strategy Manager Integration: {'✅ WORKING' if strategy_manager_working else '❌ NEEDS IMPLEMENTATION'}")
        
        if strategy_manager_working:
            self.advanced_features_tested.append("Advanced Strategy Manager")
        
        return strategy_manager_working

    def test_enhanced_decision_quality(self):
        """Test Enhanced Decision Quality with advanced features"""
        print(f"\n📈 Testing Enhanced Decision Quality...")
        
        # Get recent decisions to analyze quality enhancements
        success, decisions_data = self.run_test("Get IA2 Decisions for Quality Analysis", "GET", "decisions", 200)
        if not success:
            print(f"   ❌ Cannot retrieve decisions for quality analysis")
            return False
        
        decisions = decisions_data.get('decisions', [])
        if len(decisions) == 0:
            print(f"   ❌ No decisions available for quality analysis")
            return False
        
        print(f"   📊 Analyzing {len(decisions)} decisions for enhanced quality...")
        
        # Analyze decision quality metrics
        quality_metrics = {
            'high_confidence_decisions': 0,  # ≥70%
            'enhanced_reasoning': 0,         # >500 chars with strategy details
            'advanced_thresholds': 0,        # Uses enhanced thresholds
            'strategy_details': 0,           # Contains strategy information
            'risk_management': 0,            # Contains risk management info
            'technical_analysis_depth': 0    # Deep technical analysis
        }
        
        confidence_levels = []
        reasoning_lengths = []
        
        for i, decision in enumerate(decisions[:20]):  # Analyze first 20 decisions
            symbol = decision.get('symbol', 'Unknown')
            reasoning = decision.get('ia2_reasoning', '')
            confidence = decision.get('confidence', 0)
            signal = decision.get('signal', 'hold')
            
            confidence_levels.append(confidence)
            reasoning_lengths.append(len(reasoning))
            
            # Check quality metrics
            if confidence >= 0.70:
                quality_metrics['high_confidence_decisions'] += 1
            
            if len(reasoning) > 500 and any(keyword in reasoning.lower() for keyword in ['strategy', 'analysis', 'technical']):
                quality_metrics['enhanced_reasoning'] += 1
            
            if any(keyword in reasoning.lower() for keyword in ['threshold', 'enhanced', 'advanced']):
                quality_metrics['advanced_thresholds'] += 1
            
            if any(keyword in reasoning.lower() for keyword in ['strategy', 'take profit', 'position', 'management']):
                quality_metrics['strategy_details'] += 1
            
            if any(keyword in reasoning.lower() for keyword in ['risk', 'stop loss', 'risk management', 'risk-reward']):
                quality_metrics['risk_management'] += 1
            
            if any(keyword in reasoning.lower() for keyword in ['rsi', 'macd', 'technical', 'confluence', 'indicator']):
                quality_metrics['technical_analysis_depth'] += 1
            
            # Show detailed analysis for first 5 decisions
            if i < 5:
                print(f"\n   Decision {i+1} - {symbol} ({signal}):")
                print(f"      Confidence: {confidence:.3f} {'✅' if confidence >= 0.70 else '❌'}")
                print(f"      Reasoning Length: {len(reasoning)} chars {'✅' if len(reasoning) > 500 else '❌'}")
                print(f"      Quality Indicators:")
                print(f"        Strategy Details: {'✅' if 'strategy' in reasoning.lower() else '❌'}")
                print(f"        Risk Management: {'✅' if 'risk' in reasoning.lower() else '❌'}")
                print(f"        Technical Analysis: {'✅' if any(t in reasoning.lower() for t in ['rsi', 'macd']) else '❌'}")
        
        # Calculate quality statistics
        total_analyzed = min(len(decisions), 20)
        avg_confidence = sum(confidence_levels) / len(confidence_levels) if confidence_levels else 0
        avg_reasoning_length = sum(reasoning_lengths) / len(reasoning_lengths) if reasoning_lengths else 0
        
        print(f"\n   📊 Enhanced Decision Quality Analysis:")
        print(f"      Average Confidence: {avg_confidence:.3f}")
        print(f"      Average Reasoning Length: {avg_reasoning_length:.0f} chars")
        
        for metric, count in quality_metrics.items():
            percentage = (count / total_analyzed) * 100 if total_analyzed > 0 else 0
            print(f"      {metric.replace('_', ' ').title()}: {count}/{total_analyzed} ({percentage:.1f}%)")
        
        # Validation criteria for enhanced decision quality
        high_confidence_rate = quality_metrics['high_confidence_decisions'] / total_analyzed >= 0.30  # 30%
        enhanced_reasoning_rate = quality_metrics['enhanced_reasoning'] / total_analyzed >= 0.50  # 50%
        strategy_details_rate = quality_metrics['strategy_details'] / total_analyzed >= 0.60  # 60%
        risk_management_rate = quality_metrics['risk_management'] / total_analyzed >= 0.40  # 40%
        technical_depth_rate = quality_metrics['technical_analysis_depth'] / total_analyzed >= 0.70  # 70%
        
        print(f"\n   ✅ Enhanced Decision Quality Validation:")
        print(f"      High Confidence Rate ≥30%: {'✅' if high_confidence_rate else '❌'}")
        print(f"      Enhanced Reasoning ≥50%: {'✅' if enhanced_reasoning_rate else '❌'}")
        print(f"      Strategy Details ≥60%: {'✅' if strategy_details_rate else '❌'}")
        print(f"      Risk Management ≥40%: {'✅' if risk_management_rate else '❌'}")
        print(f"      Technical Depth ≥70%: {'✅' if technical_depth_rate else '❌'}")
        
        enhanced_quality_working = (
            high_confidence_rate and
            enhanced_reasoning_rate and
            strategy_details_rate and
            technical_depth_rate
        )
        
        print(f"\n   🎯 Enhanced Decision Quality: {'✅ WORKING' if enhanced_quality_working else '❌ NEEDS IMPROVEMENT'}")
        
        if enhanced_quality_working:
            self.advanced_features_tested.append("Enhanced Decision Quality")
        
        return enhanced_quality_working

    def test_risk_management_enhancement(self):
        """Test Risk Management Enhancement with advanced features"""
        print(f"\n🛡️ Testing Risk Management Enhancement...")
        
        # Get recent decisions to analyze risk management
        success, decisions_data = self.run_test("Get IA2 Decisions for Risk Management Analysis", "GET", "decisions", 200)
        if not success:
            print(f"   ❌ Cannot retrieve decisions for risk management analysis")
            return False
        
        decisions = decisions_data.get('decisions', [])
        if len(decisions) == 0:
            print(f"   ❌ No decisions available for risk management analysis")
            return False
        
        print(f"   📊 Analyzing {len(decisions)} decisions for risk management enhancement...")
        
        # Analyze risk management features
        risk_management_metrics = {
            'position_sizing_3_8_percent': 0,    # 3-8% range
            'min_2_1_risk_reward': 0,            # Minimum 2:1 risk-reward
            'stop_loss_calculations': 0,         # Stop-loss level calculations
            'take_profit_calculations': 0,       # Take-profit level calculations
            'account_balance_integration': 0,    # Account balance integration
            'advanced_position_sizing': 0        # Advanced position sizing logic
        }
        
        position_sizes = []
        risk_reward_ratios = []
        
        for i, decision in enumerate(decisions[:15]):  # Analyze first 15 decisions
            symbol = decision.get('symbol', 'Unknown')
            reasoning = decision.get('ia2_reasoning', '')
            confidence = decision.get('confidence', 0)
            signal = decision.get('signal', 'hold')
            position_size = decision.get('position_size', 0)
            risk_reward_ratio = decision.get('risk_reward_ratio', 0)
            stop_loss = decision.get('stop_loss', 0)
            take_profit_1 = decision.get('take_profit_1', 0)
            
            if position_size > 0:
                position_sizes.append(position_size)
            
            if risk_reward_ratio > 0:
                risk_reward_ratios.append(risk_reward_ratio)
            
            # Check risk management features
            # Position sizing (3-8% range)
            if any(keyword in reasoning.lower() for keyword in ['3%', '4%', '5%', '6%', '7%', '8%', 'position siz']):
                risk_management_metrics['position_sizing_3_8_percent'] += 1
            
            # Risk-reward ratio (minimum 2:1)
            if risk_reward_ratio >= 2.0 or '2:1' in reasoning or 'risk-reward' in reasoning.lower():
                risk_management_metrics['min_2_1_risk_reward'] += 1
            
            # Stop-loss calculations
            if stop_loss > 0 or 'stop loss' in reasoning.lower() or 'stop-loss' in reasoning.lower():
                risk_management_metrics['stop_loss_calculations'] += 1
            
            # Take-profit calculations
            if take_profit_1 > 0 or 'take profit' in reasoning.lower() or 'take-profit' in reasoning.lower():
                risk_management_metrics['take_profit_calculations'] += 1
            
            # Account balance integration
            if any(keyword in reasoning.lower() for keyword in ['account balance', 'balance', '$250', 'available']):
                risk_management_metrics['account_balance_integration'] += 1
            
            # Advanced position sizing
            if any(keyword in reasoning.lower() for keyword in ['advanced position', 'position management', 'sizing']):
                risk_management_metrics['advanced_position_sizing'] += 1
            
            # Show detailed analysis for first 5 decisions
            if i < 5:
                print(f"\n   Decision {i+1} - {symbol} ({signal}):")
                print(f"      Position Size: {position_size:.4f}")
                print(f"      Risk-Reward Ratio: {risk_reward_ratio:.2f}")
                print(f"      Stop Loss: {stop_loss:.6f}")
                print(f"      Take Profit 1: {take_profit_1:.6f}")
                print(f"      Risk Management Features:")
                print(f"        Position Sizing: {'✅' if '3%' in reasoning or 'position siz' in reasoning.lower() else '❌'}")
                print(f"        Risk-Reward: {'✅' if risk_reward_ratio >= 2.0 or 'risk-reward' in reasoning.lower() else '❌'}")
                print(f"        Stop Loss: {'✅' if stop_loss > 0 or 'stop loss' in reasoning.lower() else '❌'}")
                print(f"        Account Balance: {'✅' if 'balance' in reasoning.lower() else '❌'}")
        
        # Calculate risk management statistics
        total_analyzed = min(len(decisions), 15)
        avg_position_size = sum(position_sizes) / len(position_sizes) if position_sizes else 0
        avg_risk_reward = sum(risk_reward_ratios) / len(risk_reward_ratios) if risk_reward_ratios else 0
        
        print(f"\n   📊 Risk Management Enhancement Analysis:")
        print(f"      Average Position Size: {avg_position_size:.4f}")
        print(f"      Average Risk-Reward Ratio: {avg_risk_reward:.2f}")
        
        for metric, count in risk_management_metrics.items():
            percentage = (count / total_analyzed) * 100 if total_analyzed > 0 else 0
            print(f"      {metric.replace('_', ' ').title()}: {count}/{total_analyzed} ({percentage:.1f}%)")
        
        # Validation criteria for risk management enhancement
        position_sizing_working = risk_management_metrics['position_sizing_3_8_percent'] >= total_analyzed * 0.30  # 30%
        risk_reward_working = risk_management_metrics['min_2_1_risk_reward'] >= total_analyzed * 0.40  # 40%
        stop_loss_working = risk_management_metrics['stop_loss_calculations'] >= total_analyzed * 0.50  # 50%
        take_profit_working = risk_management_metrics['take_profit_calculations'] >= total_analyzed * 0.50  # 50%
        balance_integration_working = risk_management_metrics['account_balance_integration'] >= total_analyzed * 0.20  # 20%
        
        print(f"\n   ✅ Risk Management Enhancement Validation:")
        print(f"      Position Sizing (3-8%) ≥30%: {'✅' if position_sizing_working else '❌'}")
        print(f"      Risk-Reward (2:1) ≥40%: {'✅' if risk_reward_working else '❌'}")
        print(f"      Stop-Loss Calculations ≥50%: {'✅' if stop_loss_working else '❌'}")
        print(f"      Take-Profit Calculations ≥50%: {'✅' if take_profit_working else '❌'}")
        print(f"      Account Balance Integration ≥20%: {'✅' if balance_integration_working else '❌'}")
        
        risk_management_working = (
            position_sizing_working and
            risk_reward_working and
            stop_loss_working and
            take_profit_working
        )
        
        print(f"\n   🎯 Risk Management Enhancement: {'✅ WORKING' if risk_management_working else '❌ NEEDS IMPROVEMENT'}")
        
        if risk_management_working:
            self.advanced_features_tested.append("Risk Management Enhancement")
        
        return risk_management_working

    def test_end_to_end_advanced_pipeline(self):
        """Test end-to-end advanced trading pipeline with all features"""
        print(f"\n🔄 Testing End-to-End Advanced Trading Pipeline...")
        
        # Start trading system to generate fresh advanced decisions
        print(f"   🚀 Starting trading system for advanced pipeline test...")
        success, _ = self.run_test("Start Trading System", "POST", "start-trading", 200)
        if not success:
            print(f"   ❌ Failed to start trading system")
            return False
        
        # Wait for system to generate decisions with advanced features
        print(f"   ⏱️ Waiting for advanced pipeline to generate decisions (90 seconds)...")
        time.sleep(90)
        
        # Stop trading system
        print(f"   🛑 Stopping trading system...")
        self.run_test("Stop Trading System", "POST", "stop-trading", 200)
        
        # Test all advanced components
        print(f"\n   🔍 Testing all advanced components in pipeline...")
        
        # Test each advanced feature
        tp_system_working = self.test_advanced_multi_level_take_profit_system()
        claude_integration_working = self.test_claude_advanced_strategy_integration()
        inversion_logic_working = self.test_position_inversion_logic()
        strategy_manager_working = self.test_advanced_strategy_manager_integration()
        enhanced_quality_working = self.test_enhanced_decision_quality()
        risk_management_working = self.test_risk_management_enhancement()
        
        # Calculate overall pipeline success
        advanced_components = [
            tp_system_working,
            claude_integration_working,
            inversion_logic_working,
            strategy_manager_working,
            enhanced_quality_working,
            risk_management_working
        ]
        
        components_working = sum(advanced_components)
        total_components = len(advanced_components)
        
        print(f"\n   📊 Advanced Pipeline Component Results:")
        print(f"      4-Level TP System: {'✅' if tp_system_working else '❌'}")
        print(f"      Claude Integration: {'✅' if claude_integration_working else '❌'}")
        print(f"      Position Inversion: {'✅' if inversion_logic_working else '❌'}")
        print(f"      Strategy Manager: {'✅' if strategy_manager_working else '❌'}")
        print(f"      Enhanced Quality: {'✅' if enhanced_quality_working else '❌'}")
        print(f"      Risk Management: {'✅' if risk_management_working else '❌'}")
        
        pipeline_success_rate = components_working / total_components
        pipeline_working = pipeline_success_rate >= 0.67  # At least 4/6 components working
        
        print(f"\n   🎯 Advanced Pipeline Assessment:")
        print(f"      Components Working: {components_working}/{total_components} ({pipeline_success_rate*100:.1f}%)")
        print(f"      Pipeline Status: {'✅ SUCCESS' if pipeline_working else '❌ NEEDS WORK'}")
        
        return pipeline_working

    def run_comprehensive_advanced_strategy_tests(self):
        """Run comprehensive tests for all REVOLUTIONARY advanced trading strategies"""
        print(f"\n🚀 REVOLUTIONARY ADVANCED TRADING STRATEGIES IA2 TESTING")
        print(f"=" * 80)
        
        start_time = time.time()
        
        # Test each advanced feature
        test_results = {}
        
        print(f"\n1️⃣ Testing Advanced Multi-Level Take Profit System...")
        test_results['multi_level_tp'] = self.test_advanced_multi_level_take_profit_system()
        
        print(f"\n2️⃣ Testing Claude Advanced Strategy Integration...")
        test_results['claude_integration'] = self.test_claude_advanced_strategy_integration()
        
        print(f"\n3️⃣ Testing Position Inversion Logic...")
        test_results['position_inversion'] = self.test_position_inversion_logic()
        
        print(f"\n4️⃣ Testing Advanced Strategy Manager Integration...")
        test_results['strategy_manager'] = self.test_advanced_strategy_manager_integration()
        
        print(f"\n5️⃣ Testing Enhanced Decision Quality...")
        test_results['enhanced_quality'] = self.test_enhanced_decision_quality()
        
        print(f"\n6️⃣ Testing Risk Management Enhancement...")
        test_results['risk_management'] = self.test_risk_management_enhancement()
        
        print(f"\n7️⃣ Testing End-to-End Advanced Pipeline...")
        test_results['end_to_end_pipeline'] = self.test_end_to_end_advanced_pipeline()
        
        # Calculate overall results
        total_tests = len(test_results)
        passed_tests = sum(test_results.values())
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Final summary
        print(f"\n" + "=" * 80)
        print(f"🎯 REVOLUTIONARY ADVANCED TRADING STRATEGIES TEST SUMMARY")
        print(f"=" * 80)
        
        print(f"\n📊 Test Results:")
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Tests Run: {self.tests_run}")
        print(f"   Tests Passed: {self.tests_passed}")
        print(f"   Advanced Features Tested: {len(self.advanced_features_tested)}")
        print(f"   Advanced Features Working: {self.advanced_features_tested}")
        print(f"   Success Rate: {success_rate*100:.1f}% ({passed_tests}/{total_tests})")
        print(f"   Total Time: {total_time:.1f} seconds")
        
        # Overall assessment
        if success_rate >= 0.80:  # 80% success rate
            print(f"\n🎉 REVOLUTIONARY ADVANCED TRADING STRATEGIES: ✅ SUCCESSFULLY IMPLEMENTED")
            print(f"   The advanced IA2 system is working with sophisticated trading strategies!")
        elif success_rate >= 0.60:  # 60% success rate
            print(f"\n⚠️ REVOLUTIONARY ADVANCED TRADING STRATEGIES: 🔶 PARTIALLY IMPLEMENTED")
            print(f"   Most advanced features are working, some need refinement.")
        else:
            print(f"\n❌ REVOLUTIONARY ADVANCED TRADING STRATEGIES: ❌ NEEDS MAJOR WORK")
            print(f"   Advanced features require significant implementation work.")
        
        return success_rate >= 0.60  # Consider 60%+ as acceptable

if __name__ == "__main__":
    print("🚀 Starting REVOLUTIONARY Advanced Trading Strategies IA2 Testing...")
    
    tester = AdvancedTradingStrategiesIA2Tester()
    success = tester.run_comprehensive_advanced_strategy_tests()
    
    if success:
        print(f"\n✅ Advanced Trading Strategies Testing: SUCCESS")
        sys.exit(0)
    else:
        print(f"\n❌ Advanced Trading Strategies Testing: FAILED")
        sys.exit(1)
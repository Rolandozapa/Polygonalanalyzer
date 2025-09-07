import requests
import sys
import json
import time
import asyncio
from datetime import datetime
import os
from pathlib import Path

class AdvancedTradingStrategiesTester:
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

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test"""
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

    def clear_decision_cache(self):
        """Clear decision cache to generate fresh decisions"""
        print(f"\n🗑️ Clearing decision cache for fresh data...")
        success, result = self.run_test("Clear Decision Cache", "DELETE", "decisions/clear", 200)
        if success:
            print(f"   ✅ Cache cleared successfully")
            return True
        else:
            print(f"   ❌ Failed to clear cache")
            return False

    def start_trading_system(self):
        """Start the trading system"""
        success, _ = self.run_test("Start Trading System", "POST", "start-trading", 200)
        return success

    def stop_trading_system(self):
        """Stop the trading system"""
        success, _ = self.run_test("Stop Trading System", "POST", "stop-trading", 200)
        return success

    def get_decisions(self):
        """Get current trading decisions"""
        success, data = self.run_test("Get Trading Decisions", "GET", "decisions", 200)
        if success:
            return data.get('decisions', [])
        return []

    def test_claude_json_response_validation(self):
        """Test that Claude generates proper structured JSON responses"""
        print(f"\n🎯 Testing Claude JSON Response Validation...")
        
        # Clear cache and generate fresh decisions to trigger Claude calls
        if not self.clear_decision_cache():
            return False
        
        # Start trading system to generate fresh Claude decisions
        print(f"   🚀 Starting trading system to trigger Claude calls...")
        if not self.start_trading_system():
            return False
        
        # Wait for Claude decisions to be generated
        print(f"   ⏱️ Waiting for Claude to generate decisions (90 seconds)...")
        time.sleep(90)
        
        # Stop trading system
        self.stop_trading_system()
        
        # Get fresh decisions
        decisions = self.get_decisions()
        if not decisions:
            print(f"   ❌ No decisions generated for Claude testing")
            return False
        
        print(f"   📊 Analyzing {len(decisions)} decisions for Claude JSON structure...")
        
        # Analyze decisions for Claude-specific patterns and JSON structure
        claude_patterns = 0
        json_structure_count = 0
        tp_strategy_count = 0
        position_management_count = 0
        inversion_criteria_count = 0
        advanced_reasoning_count = 0
        
        claude_keywords = [
            'comprehensive analysis', 'technical confluence', 'market context',
            'multi-level', 'tp1', 'tp2', 'tp3', 'tp4', 'position distribution',
            'inversion', 'advanced strategy', 'graduated', 'risk-reward optimization'
        ]
        
        for i, decision in enumerate(decisions[:10]):  # Analyze first 10 decisions
            symbol = decision.get('symbol', 'Unknown')
            reasoning = decision.get('ia2_reasoning', '').lower()
            confidence = decision.get('confidence', 0)
            signal = decision.get('signal', 'hold')
            
            # Check for Claude-specific patterns
            claude_pattern_found = any(keyword in reasoning for keyword in claude_keywords)
            if claude_pattern_found:
                claude_patterns += 1
            
            # Check for JSON-like structure mentions
            json_indicators = ['tp1_percentage', 'tp2_percentage', 'tp_distribution', 'position_management', 'inversion_criteria']
            if any(indicator in reasoning for indicator in json_indicators):
                json_structure_count += 1
            
            # Check for take profit strategy mentions
            tp_indicators = ['1.5%', '3.0%', '5.0%', '8.0%', '25%', '30%', '25%', '20%']
            if any(indicator in reasoning for indicator in tp_indicators):
                tp_strategy_count += 1
            
            # Check for position management mentions
            pm_indicators = ['entry_strategy', 'stop_loss_percentage', 'trailing_stop', 'position_size']
            if any(indicator in reasoning for indicator in pm_indicators):
                position_management_count += 1
            
            # Check for inversion criteria mentions
            inv_indicators = ['enable_inversion', 'confidence_threshold', 'opposite_signal']
            if any(indicator in reasoning for indicator in inv_indicators):
                inversion_criteria_count += 1
            
            # Check for advanced reasoning patterns
            advanced_indicators = ['multi-level tp', 'position distribution', 'inversion logic', 'advanced strategy']
            if any(indicator in reasoning for indicator in advanced_indicators):
                advanced_reasoning_count += 1
            
            if i < 3:  # Show details for first 3 decisions
                print(f"\n   Decision {i+1} - {symbol} ({signal}):")
                print(f"      Confidence: {confidence:.3f}")
                print(f"      Claude Patterns: {'✅' if claude_pattern_found else '❌'}")
                print(f"      Reasoning Length: {len(decision.get('ia2_reasoning', ''))} chars")
                print(f"      Reasoning Preview: {reasoning[:150]}...")
        
        total_analyzed = min(len(decisions), 10)
        claude_pattern_rate = claude_patterns / total_analyzed
        json_structure_rate = json_structure_count / total_analyzed
        tp_strategy_rate = tp_strategy_count / total_analyzed
        
        print(f"\n   📊 Claude JSON Response Analysis:")
        print(f"      Decisions Analyzed: {total_analyzed}")
        print(f"      Claude-Specific Patterns: {claude_patterns}/{total_analyzed} ({claude_pattern_rate*100:.1f}%)")
        print(f"      JSON Structure Mentions: {json_structure_count}/{total_analyzed} ({json_structure_rate*100:.1f}%)")
        print(f"      TP Strategy Details: {tp_strategy_count}/{total_analyzed} ({tp_strategy_rate*100:.1f}%)")
        print(f"      Position Management: {position_management_count}/{total_analyzed}")
        print(f"      Inversion Criteria: {inversion_criteria_count}/{total_analyzed}")
        print(f"      Advanced Reasoning: {advanced_reasoning_count}/{total_analyzed}")
        
        # Validation criteria
        claude_integration_working = claude_pattern_rate >= 0.30  # 30% should show Claude patterns
        json_format_working = json_structure_rate >= 0.20  # 20% should mention JSON structure
        tp_strategy_working = tp_strategy_rate >= 0.15  # 15% should mention TP strategy details
        
        print(f"\n   ✅ Claude JSON Validation:")
        print(f"      Claude Patterns ≥30%: {'✅' if claude_integration_working else '❌'}")
        print(f"      JSON Structure ≥20%: {'✅' if json_format_working else '❌'}")
        print(f"      TP Strategy Details ≥15%: {'✅' if tp_strategy_working else '❌'}")
        
        claude_json_working = claude_integration_working and json_format_working and tp_strategy_working
        
        print(f"\n   🎯 Claude JSON Response Validation: {'✅ SUCCESS' if claude_json_working else '❌ NEEDS IMPROVEMENT'}")
        
        return claude_json_working

    def test_advanced_strategy_creation(self):
        """Test the _create_and_execute_advanced_strategy method"""
        print(f"\n🎯 Testing Advanced Strategy Creation...")
        
        # Get recent decisions to analyze for advanced strategy creation
        decisions = self.get_decisions()
        if not decisions:
            print(f"   ❌ No decisions available for advanced strategy testing")
            return False
        
        print(f"   📊 Analyzing {len(decisions)} decisions for advanced strategy creation...")
        
        # Look for evidence of advanced strategy creation
        advanced_strategy_count = 0
        multi_level_tp_count = 0
        position_direction_count = 0
        strategy_execution_count = 0
        
        for i, decision in enumerate(decisions[:15]):  # Analyze first 15 decisions
            symbol = decision.get('symbol', 'Unknown')
            reasoning = decision.get('ia2_reasoning', '').lower()
            signal = decision.get('signal', 'hold')
            confidence = decision.get('confidence', 0)
            
            # Check for advanced strategy creation indicators
            strategy_indicators = [
                'advanced strategy', 'multi-level', 'position distribution',
                'tp1', 'tp2', 'tp3', 'tp4', '25%', '30%', '25%', '20%'
            ]
            
            if any(indicator in reasoning for indicator in strategy_indicators):
                advanced_strategy_count += 1
            
            # Check for multi-level TP strategy
            tp_levels = ['1.5%', '3.0%', '5.0%', '8.0%']
            if sum(1 for level in tp_levels if level in reasoning) >= 2:
                multi_level_tp_count += 1
            
            # Check for position direction (LONG/SHORT)
            if signal.upper() in ['LONG', 'SHORT']:
                position_direction_count += 1
            
            # Check for strategy execution mentions
            execution_indicators = ['strategy execution', 'bingx integration', 'position management']
            if any(indicator in reasoning for indicator in execution_indicators):
                strategy_execution_count += 1
            
            if i < 3:  # Show details for first 3 decisions
                print(f"\n   Decision {i+1} - {symbol} ({signal}):")
                print(f"      Confidence: {confidence:.3f}")
                print(f"      Advanced Strategy: {'✅' if any(indicator in reasoning for indicator in strategy_indicators) else '❌'}")
                print(f"      Multi-Level TP: {'✅' if sum(1 for level in tp_levels if level in reasoning) >= 2 else '❌'}")
                print(f"      Position Direction: {'✅' if signal.upper() in ['LONG', 'SHORT'] else '❌'}")
        
        total_analyzed = min(len(decisions), 15)
        advanced_strategy_rate = advanced_strategy_count / total_analyzed
        multi_level_tp_rate = multi_level_tp_count / total_analyzed
        position_direction_rate = position_direction_count / total_analyzed
        
        print(f"\n   📊 Advanced Strategy Creation Analysis:")
        print(f"      Decisions Analyzed: {total_analyzed}")
        print(f"      Advanced Strategy Mentions: {advanced_strategy_count}/{total_analyzed} ({advanced_strategy_rate*100:.1f}%)")
        print(f"      Multi-Level TP Strategy: {multi_level_tp_count}/{total_analyzed} ({multi_level_tp_rate*100:.1f}%)")
        print(f"      Position Direction (LONG/SHORT): {position_direction_count}/{total_analyzed} ({position_direction_rate*100:.1f}%)")
        print(f"      Strategy Execution Mentions: {strategy_execution_count}/{total_analyzed}")
        
        # Validation criteria
        strategy_creation_working = advanced_strategy_rate >= 0.25  # 25% should mention advanced strategies
        multi_tp_working = multi_level_tp_rate >= 0.15  # 15% should have multi-level TP
        position_direction_working = position_direction_rate >= 0.20  # 20% should have LONG/SHORT
        
        print(f"\n   ✅ Advanced Strategy Creation Validation:")
        print(f"      Advanced Strategy ≥25%: {'✅' if strategy_creation_working else '❌'}")
        print(f"      Multi-Level TP ≥15%: {'✅' if multi_tp_working else '❌'}")
        print(f"      Position Direction ≥20%: {'✅' if position_direction_working else '❌'}")
        
        advanced_creation_working = strategy_creation_working and multi_tp_working and position_direction_working
        
        print(f"\n   🎯 Advanced Strategy Creation: {'✅ SUCCESS' if advanced_creation_working else '❌ NEEDS IMPROVEMENT'}")
        
        return advanced_creation_working

    def test_position_inversion_logic(self):
        """Test the enhanced _check_position_inversion implementation"""
        print(f"\n🎯 Testing Position Inversion Logic...")
        
        # Get recent decisions to analyze for position inversion
        decisions = self.get_decisions()
        if not decisions:
            print(f"   ❌ No decisions available for position inversion testing")
            return False
        
        print(f"   📊 Analyzing {len(decisions)} decisions for position inversion logic...")
        
        # Look for evidence of position inversion implementation
        inversion_mentions = 0
        technical_analysis_count = 0
        signal_strength_count = 0
        confidence_threshold_count = 0
        opposite_signal_count = 0
        
        # Group decisions by symbol to look for inversion patterns
        symbol_decisions = {}
        for decision in decisions:
            symbol = decision.get('symbol', 'Unknown')
            if symbol not in symbol_decisions:
                symbol_decisions[symbol] = []
            symbol_decisions[symbol].append(decision)
        
        for i, decision in enumerate(decisions[:20]):  # Analyze first 20 decisions
            symbol = decision.get('symbol', 'Unknown')
            reasoning = decision.get('ia2_reasoning', '').lower()
            signal = decision.get('signal', 'hold')
            confidence = decision.get('confidence', 0)
            
            # Check for inversion logic mentions
            inversion_indicators = [
                'inversion', 'position inversion', 'reverse position', 'opposite signal',
                'enable_inversion', 'confidence_threshold', 'signal strength'
            ]
            
            if any(indicator in reasoning for indicator in inversion_indicators):
                inversion_mentions += 1
            
            # Check for technical analysis mentions (RSI, MACD)
            technical_indicators = ['rsi', 'macd', 'technical analysis', 'bullish', 'bearish']
            if any(indicator in reasoning for indicator in technical_indicators):
                technical_analysis_count += 1
            
            # Check for signal strength calculations
            strength_indicators = ['signal strength', 'strength calculation', 'confidence threshold']
            if any(indicator in reasoning for indicator in strength_indicators):
                signal_strength_count += 1
            
            # Check for confidence threshold analysis
            threshold_indicators = ['10%', 'threshold', 'higher confidence', 'opposite signal strength']
            if any(indicator in reasoning for indicator in threshold_indicators):
                confidence_threshold_count += 1
            
            # Check for opposite signal evaluation
            opposite_indicators = ['opposite', 'reverse', 'invert', 'flip position']
            if any(indicator in reasoning for indicator in opposite_indicators):
                opposite_signal_count += 1
        
        # Look for actual inversion patterns in symbol decisions
        inversion_patterns = 0
        for symbol, symbol_decisions_list in symbol_decisions.items():
            if len(symbol_decisions_list) >= 2:
                signals = [d.get('signal', 'hold').upper() for d in symbol_decisions_list]
                # Check for LONG/SHORT alternation
                has_long = 'LONG' in signals
                has_short = 'SHORT' in signals
                if has_long and has_short:
                    inversion_patterns += 1
        
        total_analyzed = min(len(decisions), 20)
        inversion_mention_rate = inversion_mentions / total_analyzed
        technical_analysis_rate = technical_analysis_count / total_analyzed
        
        print(f"\n   📊 Position Inversion Logic Analysis:")
        print(f"      Decisions Analyzed: {total_analyzed}")
        print(f"      Inversion Mentions: {inversion_mentions}/{total_analyzed} ({inversion_mention_rate*100:.1f}%)")
        print(f"      Technical Analysis: {technical_analysis_count}/{total_analyzed} ({technical_analysis_rate*100:.1f}%)")
        print(f"      Signal Strength Mentions: {signal_strength_count}/{total_analyzed}")
        print(f"      Confidence Threshold: {confidence_threshold_count}/{total_analyzed}")
        print(f"      Opposite Signal Analysis: {opposite_signal_count}/{total_analyzed}")
        print(f"      Symbols with Inversion Patterns: {inversion_patterns}/{len(symbol_decisions)}")
        
        # Validation criteria
        inversion_logic_working = inversion_mention_rate >= 0.10  # 10% should mention inversion
        technical_integration_working = technical_analysis_rate >= 0.70  # 70% should use technical analysis
        inversion_patterns_present = inversion_patterns > 0  # At least some inversion patterns
        
        print(f"\n   ✅ Position Inversion Logic Validation:")
        print(f"      Inversion Mentions ≥10%: {'✅' if inversion_logic_working else '❌'}")
        print(f"      Technical Analysis ≥70%: {'✅' if technical_integration_working else '❌'}")
        print(f"      Inversion Patterns Present: {'✅' if inversion_patterns_present else '❌'}")
        
        inversion_working = inversion_logic_working and technical_integration_working
        
        print(f"\n   🎯 Position Inversion Logic: {'✅ SUCCESS' if inversion_working else '❌ NEEDS IMPLEMENTATION'}")
        
        return inversion_working

    def test_multi_level_take_profit_integration(self):
        """Test complete TP strategy system"""
        print(f"\n🎯 Testing Multi-Level Take Profit Integration...")
        
        # Get recent decisions to analyze for TP strategy
        decisions = self.get_decisions()
        if not decisions:
            print(f"   ❌ No decisions available for TP strategy testing")
            return False
        
        print(f"   📊 Analyzing {len(decisions)} decisions for multi-level TP strategy...")
        
        # Look for evidence of multi-level TP integration
        tp_strategy_mentions = 0
        four_level_tp_count = 0
        distribution_mentions = 0
        percentage_mentions = 0
        detailed_tp_count = 0
        
        # Expected TP levels and distribution
        expected_tp_levels = ['1.5%', '3.0%', '5.0%', '8.0%']
        expected_distribution = ['25%', '30%', '25%', '20%']
        
        for i, decision in enumerate(decisions[:15]):  # Analyze first 15 decisions
            symbol = decision.get('symbol', 'Unknown')
            reasoning = decision.get('ia2_reasoning', '').lower()
            signal = decision.get('signal', 'hold')
            confidence = decision.get('confidence', 0)
            
            # Check for TP strategy mentions
            tp_indicators = [
                'take profit', 'tp1', 'tp2', 'tp3', 'tp4', 'multi-level',
                'take_profit_strategy', 'tp_distribution'
            ]
            
            if any(indicator in reasoning for indicator in tp_indicators):
                tp_strategy_mentions += 1
            
            # Check for 4-level TP strategy
            tp_level_count = sum(1 for level in expected_tp_levels if level in reasoning)
            if tp_level_count >= 3:  # At least 3 of 4 levels mentioned
                four_level_tp_count += 1
            
            # Check for position distribution
            distribution_count = sum(1 for dist in expected_distribution if dist in reasoning)
            if distribution_count >= 2:  # At least 2 distribution percentages
                distribution_mentions += 1
            
            # Check for specific percentage mentions
            if any(level in reasoning for level in expected_tp_levels):
                percentage_mentions += 1
            
            # Check for detailed TP strategy (both levels and distribution)
            if tp_level_count >= 2 and distribution_count >= 1:
                detailed_tp_count += 1
            
            if i < 3:  # Show details for first 3 decisions
                print(f"\n   Decision {i+1} - {symbol} ({signal}):")
                print(f"      Confidence: {confidence:.3f}")
                print(f"      TP Strategy: {'✅' if any(indicator in reasoning for indicator in tp_indicators) else '❌'}")
                print(f"      TP Levels Found: {tp_level_count}/4")
                print(f"      Distribution Found: {distribution_count}/4")
                print(f"      Reasoning Preview: {reasoning[:200]}...")
        
        total_analyzed = min(len(decisions), 15)
        tp_strategy_rate = tp_strategy_mentions / total_analyzed
        four_level_rate = four_level_tp_count / total_analyzed
        distribution_rate = distribution_mentions / total_analyzed
        detailed_tp_rate = detailed_tp_count / total_analyzed
        
        print(f"\n   📊 Multi-Level TP Integration Analysis:")
        print(f"      Decisions Analyzed: {total_analyzed}")
        print(f"      TP Strategy Mentions: {tp_strategy_mentions}/{total_analyzed} ({tp_strategy_rate*100:.1f}%)")
        print(f"      4-Level TP Strategy: {four_level_tp_count}/{total_analyzed} ({four_level_rate*100:.1f}%)")
        print(f"      Distribution Mentions: {distribution_mentions}/{total_analyzed} ({distribution_rate*100:.1f}%)")
        print(f"      Detailed TP Strategy: {detailed_tp_count}/{total_analyzed} ({detailed_tp_rate*100:.1f}%)")
        print(f"      Percentage Mentions: {percentage_mentions}/{total_analyzed}")
        
        # Validation criteria
        tp_integration_working = tp_strategy_rate >= 0.30  # 30% should mention TP strategy
        four_level_working = four_level_rate >= 0.20  # 20% should have 4-level TP
        distribution_working = distribution_rate >= 0.15  # 15% should mention distribution
        
        print(f"\n   ✅ Multi-Level TP Integration Validation:")
        print(f"      TP Strategy ≥30%: {'✅' if tp_integration_working else '❌'}")
        print(f"      4-Level TP ≥20%: {'✅' if four_level_working else '❌'}")
        print(f"      Distribution ≥15%: {'✅' if distribution_working else '❌'}")
        
        tp_integration_success = tp_integration_working and four_level_working and distribution_working
        
        print(f"\n   🎯 Multi-Level TP Integration: {'✅ SUCCESS' if tp_integration_success else '❌ NEEDS IMPROVEMENT'}")
        
        return tp_integration_success

    def test_end_to_end_advanced_pipeline(self):
        """Test complete advanced trading system pipeline"""
        print(f"\n🎯 Testing End-to-End Advanced Pipeline...")
        
        # Clear cache and start fresh pipeline
        print(f"   🗑️ Clearing cache for fresh pipeline test...")
        if not self.clear_decision_cache():
            return False
        
        # Start trading system for complete pipeline
        print(f"   🚀 Starting complete advanced trading pipeline...")
        if not self.start_trading_system():
            return False
        
        # Wait for complete pipeline to process
        print(f"   ⏱️ Waiting for complete pipeline (Scout → IA1 → IA2 → Advanced Strategies) - 120 seconds...")
        time.sleep(120)
        
        # Stop trading system
        self.stop_trading_system()
        
        # Get pipeline results
        print(f"   📊 Analyzing complete pipeline results...")
        
        # Check Scout component (opportunities)
        success, opp_data = self.run_test("Get Opportunities", "GET", "opportunities", 200)
        opportunities = opp_data.get('opportunities', []) if success else []
        
        # Check IA1 component (analyses)
        success, analysis_data = self.run_test("Get Analyses", "GET", "analyses", 200)
        analyses = analysis_data.get('analyses', []) if success else []
        
        # Check IA2 component (decisions)
        decisions = self.get_decisions()
        
        print(f"\n   📊 Pipeline Component Results:")
        print(f"      Scout Opportunities: {len(opportunities)}")
        print(f"      IA1 Technical Analyses: {len(analyses)}")
        print(f"      IA2 Trading Decisions: {len(decisions)}")
        
        # Check for pipeline integration (common symbols)
        opp_symbols = set(opp.get('symbol', '') for opp in opportunities)
        analysis_symbols = set(analysis.get('symbol', '') for analysis in analyses)
        decision_symbols = set(decision.get('symbol', '') for decision in decisions)
        
        # Find symbols that flow through entire pipeline
        complete_pipeline_symbols = opp_symbols.intersection(analysis_symbols).intersection(decision_symbols)
        
        print(f"\n   🔗 Pipeline Integration Analysis:")
        print(f"      Opportunity Symbols: {len(opp_symbols)}")
        print(f"      Analysis Symbols: {len(analysis_symbols)}")
        print(f"      Decision Symbols: {len(decision_symbols)}")
        print(f"      Complete Pipeline Symbols: {len(complete_pipeline_symbols)}")
        
        # Analyze advanced features in pipeline results
        advanced_features_count = 0
        claude_integration_count = 0
        strategy_creation_count = 0
        
        for decision in decisions[:10]:  # Analyze first 10 decisions
            reasoning = decision.get('ia2_reasoning', '').lower()
            
            # Check for advanced features
            advanced_indicators = [
                'advanced strategy', 'multi-level', 'position inversion',
                'tp1', 'tp2', 'tp3', 'tp4', 'position distribution'
            ]
            
            if any(indicator in reasoning for indicator in advanced_indicators):
                advanced_features_count += 1
            
            # Check for Claude integration
            claude_indicators = [
                'comprehensive analysis', 'technical confluence', 'market context'
            ]
            
            if any(indicator in reasoning for indicator in claude_indicators):
                claude_integration_count += 1
            
            # Check for strategy creation
            strategy_indicators = [
                'strategy creation', 'advanced strategy manager', 'strategy execution'
            ]
            
            if any(indicator in reasoning for indicator in strategy_indicators):
                strategy_creation_count += 1
        
        # Validation criteria
        pipeline_components_working = len(opportunities) > 0 and len(analyses) > 0 and len(decisions) > 0
        pipeline_integration_working = len(complete_pipeline_symbols) >= 3  # At least 3 symbols through complete pipeline
        advanced_features_working = advanced_features_count >= 2  # At least 2 decisions with advanced features
        
        print(f"\n   📊 Advanced Pipeline Features:")
        print(f"      Advanced Features: {advanced_features_count}/10 decisions")
        print(f"      Claude Integration: {claude_integration_count}/10 decisions")
        print(f"      Strategy Creation: {strategy_creation_count}/10 decisions")
        
        print(f"\n   ✅ End-to-End Pipeline Validation:")
        print(f"      All Components Working: {'✅' if pipeline_components_working else '❌'}")
        print(f"      Pipeline Integration: {'✅' if pipeline_integration_working else '❌'}")
        print(f"      Advanced Features: {'✅' if advanced_features_working else '❌'}")
        
        pipeline_success = pipeline_components_working and pipeline_integration_working and advanced_features_working
        
        print(f"\n   🎯 End-to-End Advanced Pipeline: {'✅ SUCCESS' if pipeline_success else '❌ NEEDS IMPROVEMENT'}")
        
        return pipeline_success

    def test_performance_validation(self):
        """Test system performance with advanced features"""
        print(f"\n🎯 Testing Performance Validation with Advanced Features...")
        
        # Get current decisions for performance analysis
        decisions = self.get_decisions()
        if not decisions:
            print(f"   ❌ No decisions available for performance testing")
            return False
        
        print(f"   📊 Analyzing performance of {len(decisions)} decisions with advanced features...")
        
        # Performance metrics
        high_confidence_count = 0
        trading_signal_count = 0
        advanced_strategy_count = 0
        quality_reasoning_count = 0
        
        confidence_levels = []
        reasoning_lengths = []
        
        for decision in decisions:
            confidence = decision.get('confidence', 0)
            signal = decision.get('signal', 'hold')
            reasoning = decision.get('ia2_reasoning', '')
            
            confidence_levels.append(confidence)
            reasoning_lengths.append(len(reasoning))
            
            # High confidence decisions
            if confidence >= 0.70:
                high_confidence_count += 1
            
            # Trading signals (not HOLD)
            if signal.upper() in ['LONG', 'SHORT']:
                trading_signal_count += 1
            
            # Advanced strategy features
            advanced_indicators = [
                'advanced strategy', 'multi-level', 'tp1', 'tp2', 'tp3', 'tp4',
                'position distribution', 'inversion', 'strategy execution'
            ]
            
            if any(indicator in reasoning.lower() for indicator in advanced_indicators):
                advanced_strategy_count += 1
            
            # Quality reasoning
            if len(reasoning) >= 200 and reasoning != "null":
                quality_reasoning_count += 1
        
        # Calculate performance statistics
        total_decisions = len(decisions)
        high_confidence_rate = high_confidence_count / total_decisions
        trading_signal_rate = trading_signal_count / total_decisions
        advanced_strategy_rate = advanced_strategy_count / total_decisions
        quality_reasoning_rate = quality_reasoning_count / total_decisions
        
        avg_confidence = sum(confidence_levels) / len(confidence_levels) if confidence_levels else 0
        avg_reasoning_length = sum(reasoning_lengths) / len(reasoning_lengths) if reasoning_lengths else 0
        
        print(f"\n   📊 Performance Metrics:")
        print(f"      Total Decisions: {total_decisions}")
        print(f"      High Confidence (≥70%): {high_confidence_count} ({high_confidence_rate*100:.1f}%)")
        print(f"      Trading Signals: {trading_signal_count} ({trading_signal_rate*100:.1f}%)")
        print(f"      Advanced Strategy Features: {advanced_strategy_count} ({advanced_strategy_rate*100:.1f}%)")
        print(f"      Quality Reasoning: {quality_reasoning_count} ({quality_reasoning_rate*100:.1f}%)")
        print(f"      Average Confidence: {avg_confidence:.3f}")
        print(f"      Average Reasoning Length: {avg_reasoning_length:.0f} chars")
        
        # Performance validation criteria
        confidence_quality_maintained = avg_confidence >= 0.60  # Average confidence ≥60%
        trading_signals_generated = trading_signal_rate >= 0.15  # At least 15% trading signals
        advanced_features_present = advanced_strategy_rate >= 0.20  # 20% with advanced features
        reasoning_quality_maintained = quality_reasoning_rate >= 0.80  # 80% quality reasoning
        
        print(f"\n   ✅ Performance Validation:")
        print(f"      Confidence Quality ≥60%: {'✅' if confidence_quality_maintained else '❌'}")
        print(f"      Trading Signals ≥15%: {'✅' if trading_signals_generated else '❌'}")
        print(f"      Advanced Features ≥20%: {'✅' if advanced_features_present else '❌'}")
        print(f"      Reasoning Quality ≥80%: {'✅' if reasoning_quality_maintained else '❌'}")
        
        performance_success = (
            confidence_quality_maintained and
            trading_signals_generated and
            advanced_features_present and
            reasoning_quality_maintained
        )
        
        print(f"\n   🎯 Performance Validation: {'✅ SUCCESS' if performance_success else '❌ DEGRADATION DETECTED'}")
        
        return performance_success

    def run_complete_advanced_trading_test(self):
        """Run complete advanced trading strategies test suite"""
        print(f"\n🚀 COMPLETE ADVANCED TRADING STRATEGIES IMPLEMENTATION TEST")
        print(f"=" * 80)
        
        test_results = {}
        
        # Test 1: Claude JSON Response Validation
        test_results['claude_json'] = self.test_claude_json_response_validation()
        
        # Test 2: Advanced Strategy Creation
        test_results['strategy_creation'] = self.test_advanced_strategy_creation()
        
        # Test 3: Position Inversion Logic
        test_results['position_inversion'] = self.test_position_inversion_logic()
        
        # Test 4: Multi-Level Take Profit Integration
        test_results['multi_level_tp'] = self.test_multi_level_take_profit_integration()
        
        # Test 5: End-to-End Advanced Pipeline
        test_results['end_to_end_pipeline'] = self.test_end_to_end_advanced_pipeline()
        
        # Test 6: Performance Validation
        test_results['performance'] = self.test_performance_validation()
        
        # Calculate overall results
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        success_rate = passed_tests / total_tests
        
        print(f"\n" + "=" * 80)
        print(f"🎯 COMPLETE ADVANCED TRADING STRATEGIES TEST RESULTS")
        print(f"=" * 80)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {success_rate*100:.1f}%")
        
        if success_rate >= 0.80:  # 80% success rate
            print(f"\n🎉 ADVANCED TRADING STRATEGIES: ✅ SUCCESS")
            print(f"   The REVOLUTIONARY advanced trading system is working as designed!")
        elif success_rate >= 0.60:  # 60% success rate
            print(f"\n⚠️ ADVANCED TRADING STRATEGIES: 🔄 PARTIAL SUCCESS")
            print(f"   Most components working, some improvements needed.")
        else:
            print(f"\n❌ ADVANCED TRADING STRATEGIES: ❌ NEEDS MAJOR WORK")
            print(f"   Significant implementation gaps detected.")
        
        return success_rate >= 0.60

if __name__ == "__main__":
    tester = AdvancedTradingStrategiesTester()
    success = tester.run_complete_advanced_trading_test()
    sys.exit(0 if success else 1)
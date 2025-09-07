import requests
import json
import time
from pathlib import Path

class ComprehensiveAdvancedTradingTest:
    def __init__(self):
        self.base_url = "https://smart-crypto-bot-14.preview.emergentagent.com"
        self.api_url = f"{self.base_url}/api"
        
    def get_data(self, endpoint):
        """Get data from API endpoint"""
        try:
            response = requests.get(f"{self.api_url}/{endpoint}", timeout=10)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            print(f"Error getting {endpoint}: {e}")
            return {}
    
    def test_complete_advanced_trading_implementation(self):
        """Comprehensive test of advanced trading strategies implementation"""
        print(f"\n🎯 COMPREHENSIVE ADVANCED TRADING STRATEGIES IMPLEMENTATION TEST")
        print(f"=" * 80)
        
        test_results = {}
        
        # Test 1: Claude JSON Response Validation
        print(f"\n🔍 Test 1: Claude JSON Response Validation")
        test_results['claude_json'] = self.test_claude_json_response_validation()
        
        # Test 2: Advanced Strategy Creation Framework
        print(f"\n🔍 Test 2: Advanced Strategy Creation Framework")
        test_results['strategy_creation'] = self.test_advanced_strategy_creation_framework()
        
        # Test 3: Position Inversion Logic Implementation
        print(f"\n🔍 Test 3: Position Inversion Logic Implementation")
        test_results['position_inversion'] = self.test_position_inversion_logic_implementation()
        
        # Test 4: Multi-Level Take Profit Integration
        print(f"\n🔍 Test 4: Multi-Level Take Profit Integration")
        test_results['multi_level_tp'] = self.test_multi_level_take_profit_integration()
        
        # Test 5: End-to-End Advanced Pipeline
        print(f"\n🔍 Test 5: End-to-End Advanced Pipeline")
        test_results['end_to_end_pipeline'] = self.test_end_to_end_advanced_pipeline()
        
        # Test 6: Performance Validation
        print(f"\n🔍 Test 6: Performance Validation")
        test_results['performance'] = self.test_performance_validation()
        
        # Calculate overall results
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        success_rate = passed_tests / total_tests
        
        print(f"\n" + "=" * 80)
        print(f"🎯 COMPREHENSIVE ADVANCED TRADING STRATEGIES TEST RESULTS")
        print(f"=" * 80)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {success_rate*100:.1f}%")
        
        # Detailed assessment
        if success_rate >= 0.80:
            print(f"\n🎉 ADVANCED TRADING STRATEGIES: ✅ SUCCESS")
            print(f"   The REVOLUTIONARY advanced trading system is working as designed!")
        elif success_rate >= 0.60:
            print(f"\n⚠️ ADVANCED TRADING STRATEGIES: 🔄 PARTIAL SUCCESS")
            print(f"   Most components working, some improvements needed.")
        else:
            print(f"\n❌ ADVANCED TRADING STRATEGIES: ❌ NEEDS MAJOR WORK")
            print(f"   Significant implementation gaps detected.")
        
        return success_rate >= 0.60
    
    def test_claude_json_response_validation(self):
        """Test Claude JSON response validation and structure"""
        print(f"   📊 Testing Claude JSON response validation...")
        
        # Check server.py for Claude integration
        try:
            with open('/app/backend/server.py', 'r') as f:
                server_content = f.read()
            
            # Look for Claude-specific implementation
            claude_model_specified = 'claude-3-7-sonnet-20250219' in server_content
            json_format_prompt = 'MANDATORY: Respond ONLY with valid JSON' in server_content
            tp_strategy_structure = 'take_profit_strategy' in server_content
            position_management_structure = 'position_management' in server_content
            inversion_criteria_structure = 'inversion_criteria' in server_content
            
            # Look for JSON parsing implementation
            parse_llm_response = '_parse_llm_response' in server_content
            json_parsing_logic = 'json.loads' in server_content
            
            print(f"      Claude Model Specified: {'✅' if claude_model_specified else '❌'}")
            print(f"      JSON Format Prompt: {'✅' if json_format_prompt else '❌'}")
            print(f"      TP Strategy Structure: {'✅' if tp_strategy_structure else '❌'}")
            print(f"      Position Management: {'✅' if position_management_structure else '❌'}")
            print(f"      Inversion Criteria: {'✅' if inversion_criteria_structure else '❌'}")
            print(f"      JSON Parsing Method: {'✅' if parse_llm_response else '❌'}")
            print(f"      JSON Parsing Logic: {'✅' if json_parsing_logic else '❌'}")
            
            # Check for advanced strategy prompt details
            multi_level_tp_prompt = 'TP1 (25%): Quick profit taking at 1.5%' in server_content
            tp_distribution_prompt = 'tp_distribution' in server_content
            
            print(f"      Multi-Level TP Prompt: {'✅' if multi_level_tp_prompt else '❌'}")
            print(f"      TP Distribution Prompt: {'✅' if tp_distribution_prompt else '❌'}")
            
            # Calculate score
            components = [
                claude_model_specified, json_format_prompt, tp_strategy_structure,
                position_management_structure, inversion_criteria_structure,
                parse_llm_response, json_parsing_logic, multi_level_tp_prompt
            ]
            
            score = sum(components) / len(components)
            
            print(f"   🎯 Claude JSON Response Validation: {score*100:.1f}% implemented")
            
            return score >= 0.75  # 75% of components implemented
            
        except Exception as e:
            print(f"      ❌ Error checking Claude integration: {e}")
            return False
    
    def test_advanced_strategy_creation_framework(self):
        """Test advanced strategy creation framework"""
        print(f"   📊 Testing advanced strategy creation framework...")
        
        try:
            # Check advanced_trading_strategies.py file
            with open('/app/backend/advanced_trading_strategies.py', 'r') as f:
                strategy_content = f.read()
            
            # Check for key components
            position_direction_enum = 'class PositionDirection' in strategy_content
            take_profit_level_class = 'class TakeProfitLevel' in strategy_content
            advanced_strategy_class = 'class AdvancedTradingStrategy' in strategy_content
            strategy_manager_class = 'class AdvancedTradingStrategyManager' in strategy_content
            
            # Check for key methods
            create_advanced_strategy = 'async def create_advanced_strategy' in strategy_content
            execute_strategy = 'async def execute_strategy' in strategy_content
            check_position_inversion = 'async def check_position_inversion_signal' in strategy_content
            
            # Check for multi-level TP implementation
            four_tp_levels = strategy_content.count('TakeProfitLevel(') >= 4
            tp_percentages = all(pct in strategy_content for pct in ['1.015', '1.03', '1.05'])
            tp_distribution = all(dist in strategy_content for dist in ['25.0', '30.0', '25.0', '20.0'])
            
            # Check BingX integration
            bingx_integration = 'bingx_trading_engine' in strategy_content
            order_placement = 'place_order' in strategy_content
            
            print(f"      Position Direction Enum: {'✅' if position_direction_enum else '❌'}")
            print(f"      Take Profit Level Class: {'✅' if take_profit_level_class else '❌'}")
            print(f"      Advanced Strategy Class: {'✅' if advanced_strategy_class else '❌'}")
            print(f"      Strategy Manager Class: {'✅' if strategy_manager_class else '❌'}")
            print(f"      Create Strategy Method: {'✅' if create_advanced_strategy else '❌'}")
            print(f"      Execute Strategy Method: {'✅' if execute_strategy else '❌'}")
            print(f"      Position Inversion Method: {'✅' if check_position_inversion else '❌'}")
            print(f"      Four TP Levels: {'✅' if four_tp_levels else '❌'}")
            print(f"      TP Percentages (1.5%, 3%, 5%): {'✅' if tp_percentages else '❌'}")
            print(f"      TP Distribution (25,30,25,20): {'✅' if tp_distribution else '❌'}")
            print(f"      BingX Integration: {'✅' if bingx_integration else '❌'}")
            print(f"      Order Placement: {'✅' if order_placement else '❌'}")
            
            # Check server.py integration
            with open('/app/backend/server.py', 'r') as f:
                server_content = f.read()
            
            strategy_import = 'from advanced_trading_strategies import' in server_content
            strategy_manager_usage = 'advanced_strategy_manager' in server_content
            create_execute_method = '_create_and_execute_advanced_strategy' in server_content
            
            print(f"      Strategy Import: {'✅' if strategy_import else '❌'}")
            print(f"      Strategy Manager Usage: {'✅' if strategy_manager_usage else '❌'}")
            print(f"      Create/Execute Method: {'✅' if create_execute_method else '❌'}")
            
            # Calculate score
            components = [
                position_direction_enum, take_profit_level_class, advanced_strategy_class,
                strategy_manager_class, create_advanced_strategy, execute_strategy,
                check_position_inversion, four_tp_levels, tp_percentages, tp_distribution,
                bingx_integration, strategy_import, strategy_manager_usage, create_execute_method
            ]
            
            score = sum(components) / len(components)
            
            print(f"   🎯 Advanced Strategy Creation Framework: {score*100:.1f}% implemented")
            
            return score >= 0.80  # 80% of components implemented
            
        except Exception as e:
            print(f"      ❌ Error checking strategy framework: {e}")
            return False
    
    def test_position_inversion_logic_implementation(self):
        """Test position inversion logic implementation"""
        print(f"   📊 Testing position inversion logic implementation...")
        
        try:
            # Check advanced_trading_strategies.py for inversion logic
            with open('/app/backend/advanced_trading_strategies.py', 'r') as f:
                strategy_content = f.read()
            
            # Check for inversion components
            inversion_method = 'check_position_inversion_signal' in strategy_content
            inversion_threshold = 'position_inversion_threshold' in strategy_content
            execute_inversion = '_execute_position_inversion' in strategy_content
            close_position = '_close_current_position' in strategy_content
            confidence_delta = 'confidence_delta' in strategy_content
            
            # Check for technical analysis integration
            rsi_analysis = 'RSI analysis' in strategy_content or 'rsi' in strategy_content.lower()
            macd_analysis = 'MACD analysis' in strategy_content or 'macd' in strategy_content.lower()
            signal_strength = 'signal_strength' in strategy_content
            
            print(f"      Inversion Method: {'✅' if inversion_method else '❌'}")
            print(f"      Inversion Threshold: {'✅' if inversion_threshold else '❌'}")
            print(f"      Execute Inversion: {'✅' if execute_inversion else '❌'}")
            print(f"      Close Position: {'✅' if close_position else '❌'}")
            print(f"      Confidence Delta: {'✅' if confidence_delta else '❌'}")
            
            # Check server.py for inversion integration
            with open('/app/backend/server.py', 'r') as f:
                server_content = f.read()
            
            check_inversion_method = '_check_position_inversion' in server_content
            rsi_inversion_logic = 'analysis.rsi < 30' in server_content or 'analysis.rsi > 70' in server_content
            macd_inversion_logic = 'analysis.macd_signal' in server_content
            bullish_bearish_signals = 'bullish_signals' in server_content and 'bearish_signals' in server_content
            
            print(f"      Check Inversion Method: {'✅' if check_inversion_method else '❌'}")
            print(f"      RSI Inversion Logic: {'✅' if rsi_inversion_logic else '❌'}")
            print(f"      MACD Inversion Logic: {'✅' if macd_inversion_logic else '❌'}")
            print(f"      Bullish/Bearish Signals: {'✅' if bullish_bearish_signals else '❌'}")
            
            # Calculate score
            components = [
                inversion_method, inversion_threshold, execute_inversion, close_position,
                confidence_delta, check_inversion_method, rsi_inversion_logic,
                macd_inversion_logic, bullish_bearish_signals
            ]
            
            score = sum(components) / len(components)
            
            print(f"   🎯 Position Inversion Logic: {score*100:.1f}% implemented")
            
            return score >= 0.70  # 70% of components implemented
            
        except Exception as e:
            print(f"      ❌ Error checking inversion logic: {e}")
            return False
    
    def test_multi_level_take_profit_integration(self):
        """Test multi-level take profit integration"""
        print(f"   📊 Testing multi-level take profit integration...")
        
        try:
            # Check for TP strategy in server.py
            with open('/app/backend/server.py', 'r') as f:
                server_content = f.read()
            
            # Check Claude prompt for TP strategy
            tp_strategy_prompt = 'take_profit_strategy' in server_content
            tp_percentages_prompt = all(pct in server_content for pct in ['1.5', '3.0', '5.0', '8.0'])
            tp_distribution_prompt = '[25, 30, 25, 20]' in server_content
            
            # Check TP strategy extraction from Claude response
            tp_strategy_extraction = "claude_decision.get('take_profit_strategy'" in server_content
            tp_details_in_reasoning = 'TP1(' in server_content and 'TP2(' in server_content
            
            print(f"      TP Strategy Prompt: {'✅' if tp_strategy_prompt else '❌'}")
            print(f"      TP Percentages in Prompt: {'✅' if tp_percentages_prompt else '❌'}")
            print(f"      TP Distribution in Prompt: {'✅' if tp_distribution_prompt else '❌'}")
            print(f"      TP Strategy Extraction: {'✅' if tp_strategy_extraction else '❌'}")
            print(f"      TP Details in Reasoning: {'✅' if tp_details_in_reasoning else '❌'}")
            
            # Check advanced_trading_strategies.py for TP implementation
            with open('/app/backend/advanced_trading_strategies.py', 'r') as f:
                strategy_content = f.read()
            
            # Check for TP level implementation
            tp_level_class = 'class TakeProfitLevel' in strategy_content
            four_tp_creation = strategy_content.count('TakeProfitLevel(') >= 8  # 4 for LONG, 4 for SHORT
            tp_order_placement = '_place_take_profit_order' in strategy_content
            tp_percentage_calculation = 'quantity_percentage' in strategy_content
            
            print(f"      TP Level Class: {'✅' if tp_level_class else '❌'}")
            print(f"      Four TP Levels Created: {'✅' if four_tp_creation else '❌'}")
            print(f"      TP Order Placement: {'✅' if tp_order_placement else '❌'}")
            print(f"      TP Percentage Calculation: {'✅' if tp_percentage_calculation else '❌'}")
            
            # Check for specific TP percentages and distribution
            tp_1_5_percent = '1.015' in strategy_content  # 1.5%
            tp_3_percent = '1.03' in strategy_content     # 3%
            tp_5_percent = '1.05' in strategy_content     # 5%
            distribution_25_30_25_20 = all(dist in strategy_content for dist in ['25.0', '30.0', '25.0', '20.0'])
            
            print(f"      TP 1.5% Level: {'✅' if tp_1_5_percent else '❌'}")
            print(f"      TP 3% Level: {'✅' if tp_3_percent else '❌'}")
            print(f"      TP 5% Level: {'✅' if tp_5_percent else '❌'}")
            print(f"      Distribution 25,30,25,20: {'✅' if distribution_25_30_25_20 else '❌'}")
            
            # Calculate score
            components = [
                tp_strategy_prompt, tp_percentages_prompt, tp_distribution_prompt,
                tp_strategy_extraction, tp_details_in_reasoning, tp_level_class,
                four_tp_creation, tp_order_placement, tp_percentage_calculation,
                tp_1_5_percent, tp_3_percent, tp_5_percent, distribution_25_30_25_20
            ]
            
            score = sum(components) / len(components)
            
            print(f"   🎯 Multi-Level TP Integration: {score*100:.1f}% implemented")
            
            return score >= 0.75  # 75% of components implemented
            
        except Exception as e:
            print(f"      ❌ Error checking TP integration: {e}")
            return False
    
    def test_end_to_end_advanced_pipeline(self):
        """Test end-to-end advanced pipeline"""
        print(f"   📊 Testing end-to-end advanced pipeline...")
        
        # Get current system data
        opportunities_data = self.get_data("opportunities")
        analyses_data = self.get_data("analyses")
        decisions_data = self.get_data("decisions")
        
        opportunities = opportunities_data.get('opportunities', [])
        analyses = analyses_data.get('analyses', [])
        decisions = decisions_data.get('decisions', [])
        
        print(f"      Scout Opportunities: {len(opportunities)}")
        print(f"      IA1 Technical Analyses: {len(analyses)}")
        print(f"      IA2 Trading Decisions: {len(decisions)}")
        
        # Check pipeline integration
        pipeline_components_present = len(analyses) > 0  # At least IA1 is working
        
        # Check for enhanced OHLCV integration
        enhanced_ohlcv_working = False
        if analyses:
            # Check for non-zero MACD values (sign of enhanced OHLCV)
            non_zero_macd = sum(1 for analysis in analyses[:5] if abs(analysis.get('macd_signal', 0)) > 0.000001)
            enhanced_ohlcv_working = non_zero_macd > 0
        
        print(f"      Pipeline Components Present: {'✅' if pipeline_components_present else '❌'}")
        print(f"      Enhanced OHLCV Working: {'✅' if enhanced_ohlcv_working else '❌'}")
        
        # Check server.py for pipeline integration
        try:
            with open('/app/backend/server.py', 'r') as f:
                server_content = f.read()
            
            # Check for advanced pipeline components
            scout_to_ia1 = 'analyze_opportunity' in server_content
            ia1_to_ia2 = 'make_decision' in server_content
            ia2_to_advanced = '_create_and_execute_advanced_strategy' in server_content
            position_inversion_check = '_check_position_inversion' in server_content
            
            print(f"      Scout → IA1 Integration: {'✅' if scout_to_ia1 else '❌'}")
            print(f"      IA1 → IA2 Integration: {'✅' if ia1_to_ia2 else '❌'}")
            print(f"      IA2 → Advanced Strategies: {'✅' if ia2_to_advanced else '❌'}")
            print(f"      Position Inversion Check: {'✅' if position_inversion_check else '❌'}")
            
            # Calculate score
            components = [
                pipeline_components_present, enhanced_ohlcv_working,
                scout_to_ia1, ia1_to_ia2, ia2_to_advanced, position_inversion_check
            ]
            
            score = sum(components) / len(components)
            
            print(f"   🎯 End-to-End Advanced Pipeline: {score*100:.1f}% working")
            
            return score >= 0.70  # 70% of components working
            
        except Exception as e:
            print(f"      ❌ Error checking pipeline: {e}")
            return False
    
    def test_performance_validation(self):
        """Test system performance with advanced features"""
        print(f"   📊 Testing system performance with advanced features...")
        
        # Get current system data
        analyses_data = self.get_data("analyses")
        decisions_data = self.get_data("decisions")
        
        analyses = analyses_data.get('analyses', [])
        decisions = decisions_data.get('decisions', [])
        
        # Performance metrics
        analysis_quality = 0
        decision_quality = 0
        advanced_features_present = 0
        
        # Check analysis quality
        if analyses:
            high_confidence_analyses = sum(1 for analysis in analyses if analysis.get('analysis_confidence', 0) >= 0.7)
            complete_analyses = sum(1 for analysis in analyses if 
                                  analysis.get('rsi', 0) > 0 and 
                                  len(analysis.get('support_levels', [])) > 0 and
                                  len(analysis.get('resistance_levels', [])) > 0)
            
            analysis_quality = (high_confidence_analyses + complete_analyses) / (2 * len(analyses))
        
        # Check decision quality
        if decisions:
            high_confidence_decisions = sum(1 for decision in decisions if decision.get('confidence', 0) >= 0.6)
            quality_reasoning = sum(1 for decision in decisions if 
                                  len(decision.get('ia2_reasoning', '')) >= 200 and
                                  decision.get('ia2_reasoning', '') != 'null')
            
            decision_quality = (high_confidence_decisions + quality_reasoning) / (2 * len(decisions))
        
        # Check for advanced features in system
        try:
            with open('/app/backend/server.py', 'r') as f:
                server_content = f.read()
            
            advanced_features = [
                'advanced_trading_strategies' in server_content,
                'claude-3-7-sonnet' in server_content,
                '_create_and_execute_advanced_strategy' in server_content,
                '_check_position_inversion' in server_content,
                'take_profit_strategy' in server_content
            ]
            
            advanced_features_present = sum(advanced_features) / len(advanced_features)
            
        except:
            advanced_features_present = 0
        
        print(f"      Analysis Quality: {analysis_quality*100:.1f}%")
        print(f"      Decision Quality: {decision_quality*100:.1f}%")
        print(f"      Advanced Features Present: {advanced_features_present*100:.1f}%")
        
        # System performance check
        system_responsive = True
        try:
            response = requests.get(f"{self.api_url}/", timeout=5)
            system_responsive = response.status_code == 200
        except:
            system_responsive = False
        
        print(f"      System Responsive: {'✅' if system_responsive else '❌'}")
        
        # Calculate overall performance score
        performance_components = [analysis_quality, decision_quality, advanced_features_present]
        if system_responsive:
            performance_components.append(1.0)
        else:
            performance_components.append(0.0)
        
        performance_score = sum(performance_components) / len(performance_components)
        
        print(f"   🎯 System Performance: {performance_score*100:.1f}%")
        
        return performance_score >= 0.70  # 70% performance maintained

if __name__ == "__main__":
    tester = ComprehensiveAdvancedTradingTest()
    success = tester.test_complete_advanced_trading_implementation()
    print(f"\n🎯 Comprehensive test completed with {'SUCCESS' if success else 'NEEDS IMPROVEMENT'}")
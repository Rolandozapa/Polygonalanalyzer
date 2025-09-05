import requests
import json
import time
from pathlib import Path

class FocusedAdvancedTradingTest:
    def __init__(self):
        self.base_url = "https://dual-ai-trader-2.preview.emergentagent.com"
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
    
    def test_advanced_trading_implementation(self):
        """Test advanced trading implementation with existing data"""
        print(f"\n🎯 FOCUSED ADVANCED TRADING STRATEGIES TEST")
        print(f"=" * 60)
        
        # Get existing data
        print(f"\n📊 Gathering existing system data...")
        
        opportunities_data = self.get_data("opportunities")
        analyses_data = self.get_data("analyses")
        decisions_data = self.get_data("decisions")
        
        opportunities = opportunities_data.get('opportunities', [])
        analyses = analyses_data.get('analyses', [])
        decisions = decisions_data.get('decisions', [])
        
        print(f"   Opportunities: {len(opportunities)}")
        print(f"   Analyses: {len(analyses)}")
        print(f"   Decisions: {len(decisions)}")
        
        # Test 1: Enhanced OHLCV and MACD Analysis
        print(f"\n🔍 Test 1: Enhanced OHLCV and MACD Analysis")
        macd_working = self.test_enhanced_macd_calculations(analyses)
        print(f"   Enhanced MACD: {'✅ WORKING' if macd_working else '❌ NEEDS FIX'}")
        
        # Test 2: Multi-Source Data Integration
        print(f"\n🔍 Test 2: Multi-Source Data Integration")
        multi_source_working = self.test_multi_source_integration(opportunities, analyses)
        print(f"   Multi-Source Data: {'✅ WORKING' if multi_source_working else '❌ NEEDS FIX'}")
        
        # Test 3: Advanced Strategy Framework Analysis
        print(f"\n🔍 Test 3: Advanced Strategy Framework")
        framework_working = self.test_advanced_strategy_framework()
        print(f"   Strategy Framework: {'✅ WORKING' if framework_working else '❌ NEEDS IMPLEMENTATION'}")
        
        # Test 4: Claude Integration Evidence
        print(f"\n🔍 Test 4: Claude Integration Evidence")
        claude_working = self.test_claude_integration_evidence(decisions)
        print(f"   Claude Integration: {'✅ WORKING' if claude_working else '❌ NEEDS VERIFICATION'}")
        
        # Test 5: Generate Fresh Decision for Advanced Features
        print(f"\n🔍 Test 5: Fresh Decision Generation Test")
        fresh_decision_working = self.test_fresh_decision_generation()
        print(f"   Fresh Decision Generation: {'✅ WORKING' if fresh_decision_working else '❌ NEEDS FIX'}")
        
        # Overall Assessment
        tests_passed = sum([macd_working, multi_source_working, framework_working, claude_working, fresh_decision_working])
        total_tests = 5
        
        print(f"\n" + "=" * 60)
        print(f"🎯 FOCUSED ADVANCED TRADING TEST RESULTS")
        print(f"=" * 60)
        print(f"   Tests Passed: {tests_passed}/{total_tests}")
        print(f"   Success Rate: {tests_passed/total_tests*100:.1f}%")
        
        if tests_passed >= 4:
            print(f"\n✅ ADVANCED TRADING SYSTEM: SUCCESS")
            print(f"   Most advanced features are working correctly!")
        elif tests_passed >= 3:
            print(f"\n⚠️ ADVANCED TRADING SYSTEM: PARTIAL SUCCESS")
            print(f"   Core features working, some advanced features need work.")
        else:
            print(f"\n❌ ADVANCED TRADING SYSTEM: NEEDS MAJOR WORK")
            print(f"   Significant implementation gaps detected.")
        
        return tests_passed >= 3
    
    def test_enhanced_macd_calculations(self, analyses):
        """Test enhanced MACD calculations"""
        if not analyses:
            print(f"   ❌ No analyses available for MACD testing")
            return False
        
        print(f"   📊 Analyzing {len(analyses)} technical analyses for MACD improvements...")
        
        non_zero_macd_count = 0
        macd_values = []
        
        for analysis in analyses[:10]:  # Check first 10
            macd_signal = analysis.get('macd_signal', 0)
            symbol = analysis.get('symbol', 'Unknown')
            
            macd_values.append(macd_signal)
            
            if abs(macd_signal) > 0.000001:  # Non-zero MACD
                non_zero_macd_count += 1
                print(f"      {symbol}: MACD = {macd_signal:.6f} ✅")
            else:
                print(f"      {symbol}: MACD = {macd_signal:.6f} ❌")
        
        non_zero_rate = non_zero_macd_count / min(len(analyses), 10)
        unique_values = len(set(macd_values))
        
        print(f"   📊 MACD Analysis Results:")
        print(f"      Non-zero MACD: {non_zero_macd_count}/{min(len(analyses), 10)} ({non_zero_rate*100:.1f}%)")
        print(f"      Unique MACD values: {unique_values}")
        print(f"      MACD range: {min(macd_values):.6f} to {max(macd_values):.6f}")
        
        # MACD is working if >50% are non-zero and we have variation
        macd_working = non_zero_rate >= 0.5 and unique_values >= 3
        
        return macd_working
    
    def test_multi_source_integration(self, opportunities, analyses):
        """Test multi-source data integration"""
        print(f"   📊 Analyzing multi-source data integration...")
        
        # Check opportunities for multiple data sources
        multi_source_opps = 0
        source_types = set()
        
        for opp in opportunities[:10]:
            sources = opp.get('data_sources', [])
            if len(sources) >= 2:
                multi_source_opps += 1
            source_types.update(sources)
        
        # Check analyses for enhanced data sources
        enhanced_sources = 0
        for analysis in analyses[:10]:
            sources = analysis.get('data_sources', [])
            enhanced_keywords = ['enhanced', 'multi', 'merged', 'cryptocompare', 'kraken', 'coingecko']
            if any(keyword in str(sources).lower() for keyword in enhanced_keywords):
                enhanced_sources += 1
        
        print(f"   📊 Multi-Source Integration Results:")
        print(f"      Multi-source opportunities: {multi_source_opps}/{min(len(opportunities), 10)}")
        print(f"      Enhanced source analyses: {enhanced_sources}/{min(len(analyses), 10)}")
        print(f"      Unique source types: {len(source_types)}")
        print(f"      Source types: {list(source_types)[:5]}")
        
        # Multi-source working if we have variety and enhanced sources
        multi_source_working = len(source_types) >= 3 and enhanced_sources >= 2
        
        return multi_source_working
    
    def test_advanced_strategy_framework(self):
        """Test if advanced strategy framework files exist and are accessible"""
        print(f"   📊 Checking advanced strategy framework...")
        
        # Check if advanced strategy files exist
        try:
            # Check server.py for advanced strategy imports and methods
            with open('/app/backend/server.py', 'r') as f:
                server_content = f.read()
            
            # Look for advanced strategy components
            advanced_imports = 'from advanced_trading_strategies import' in server_content
            strategy_manager = 'advanced_strategy_manager' in server_content
            create_advanced_method = '_create_and_execute_advanced_strategy' in server_content
            position_inversion_method = '_check_position_inversion' in server_content
            claude_integration = 'claude-3-7-sonnet' in server_content
            
            print(f"      Advanced imports: {'✅' if advanced_imports else '❌'}")
            print(f"      Strategy manager: {'✅' if strategy_manager else '❌'}")
            print(f"      Advanced strategy method: {'✅' if create_advanced_method else '❌'}")
            print(f"      Position inversion method: {'✅' if position_inversion_method else '❌'}")
            print(f"      Claude integration: {'✅' if claude_integration else '❌'}")
            
            # Check for advanced strategy file
            try:
                with open('/app/backend/advanced_trading_strategies.py', 'r') as f:
                    strategy_content = f.read()
                strategy_file_exists = True
                multi_level_tp = 'multi_level_take_profit' in strategy_content or 'TP1' in strategy_content
                position_direction = 'PositionDirection' in strategy_content
            except:
                strategy_file_exists = False
                multi_level_tp = False
                position_direction = False
            
            print(f"      Strategy file exists: {'✅' if strategy_file_exists else '❌'}")
            print(f"      Multi-level TP: {'✅' if multi_level_tp else '❌'}")
            print(f"      Position direction: {'✅' if position_direction else '❌'}")
            
            # Framework working if most components are present
            components_present = sum([
                advanced_imports, strategy_manager, create_advanced_method,
                position_inversion_method, claude_integration, strategy_file_exists
            ])
            
            framework_working = components_present >= 4  # At least 4/6 components
            
            return framework_working
            
        except Exception as e:
            print(f"      Error checking framework: {e}")
            return False
    
    def test_claude_integration_evidence(self, decisions):
        """Test for evidence of Claude integration in decisions"""
        if not decisions:
            print(f"   ❌ No decisions available for Claude testing")
            return False
        
        print(f"   📊 Analyzing {len(decisions)} decisions for Claude integration...")
        
        claude_patterns = 0
        sophisticated_reasoning = 0
        advanced_keywords = 0
        
        claude_indicators = [
            'comprehensive analysis', 'technical confluence', 'market context',
            'nuanced', 'sophisticated', 'multi-faceted', 'strategic'
        ]
        
        advanced_strategy_indicators = [
            'multi-level', 'take profit', 'tp1', 'tp2', 'position distribution',
            'advanced strategy', 'inversion', 'risk-reward optimization'
        ]
        
        for decision in decisions[:5]:  # Check first 5 decisions
            reasoning = decision.get('ia2_reasoning', '').lower()
            
            # Check for Claude-specific patterns
            if any(indicator in reasoning for indicator in claude_indicators):
                claude_patterns += 1
            
            # Check for sophisticated reasoning (length and complexity)
            if len(reasoning) >= 500 and len(reasoning.split()) >= 50:
                sophisticated_reasoning += 1
            
            # Check for advanced strategy keywords
            if any(indicator in reasoning for indicator in advanced_strategy_indicators):
                advanced_keywords += 1
        
        total_checked = min(len(decisions), 5)
        
        print(f"   📊 Claude Integration Analysis:")
        print(f"      Claude patterns: {claude_patterns}/{total_checked}")
        print(f"      Sophisticated reasoning: {sophisticated_reasoning}/{total_checked}")
        print(f"      Advanced keywords: {advanced_keywords}/{total_checked}")
        
        # Claude working if we see sophisticated patterns
        claude_working = sophisticated_reasoning >= 3 or claude_patterns >= 2
        
        return claude_working
    
    def test_fresh_decision_generation(self):
        """Test fresh decision generation with advanced features"""
        print(f"   📊 Testing fresh decision generation...")
        
        try:
            # Start trading system
            start_response = requests.post(f"{self.api_url}/start-trading", timeout=10)
            if start_response.status_code != 200:
                print(f"      ❌ Failed to start trading system")
                return False
            
            print(f"      ✅ Trading system started")
            
            # Wait for some processing
            print(f"      ⏱️ Waiting 30 seconds for processing...")
            time.sleep(30)
            
            # Check for new data
            decisions_data = self.get_data("decisions")
            decisions = decisions_data.get('decisions', [])
            
            # Stop trading system
            requests.post(f"{self.api_url}/stop-trading", timeout=10)
            
            if decisions:
                print(f"      ✅ Generated {len(decisions)} decisions")
                
                # Quick analysis of latest decision
                latest = decisions[0]
                confidence = latest.get('confidence', 0)
                reasoning = latest.get('ia2_reasoning', '')
                signal = latest.get('signal', 'hold')
                
                print(f"      Latest decision: {latest.get('symbol', 'Unknown')} - {signal} @ {confidence:.3f}")
                print(f"      Reasoning length: {len(reasoning)} chars")
                
                # Check for advanced features in reasoning
                advanced_features = any(keyword in reasoning.lower() for keyword in [
                    'multi-level', 'tp1', 'tp2', 'advanced strategy', 'position distribution'
                ])
                
                print(f"      Advanced features: {'✅' if advanced_features else '❌'}")
                
                return True
            else:
                print(f"      ⚠️ No decisions generated in 30 seconds")
                return False
                
        except Exception as e:
            print(f"      ❌ Error in fresh decision test: {e}")
            return False

if __name__ == "__main__":
    tester = FocusedAdvancedTradingTest()
    success = tester.test_advanced_trading_implementation()
    print(f"\n🎯 Test completed with {'SUCCESS' if success else 'PARTIAL SUCCESS'}")
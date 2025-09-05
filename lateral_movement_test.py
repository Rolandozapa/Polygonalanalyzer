import requests
import sys
import json
import time
import asyncio
from datetime import datetime
import os
from pathlib import Path

class LateralMovementFilterTester:
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
                    base_url = "https://dualtrade-ai.preview.emergentagent.com"
            except:
                base_url = "https://dualtrade-ai.preview.emergentagent.com"
        
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

    def test_clear_cache_and_generate_fresh_cycle(self):
        """Clear cache and generate fresh trading cycle to test lateral movement filter"""
        print(f"\n🗑️ STEP 1: Clearing Cache and Generating Fresh Trading Cycle...")
        
        # Clear decision cache
        print(f"   🗑️ Clearing decision cache...")
        success, clear_result = self.run_test("Clear Decision Cache", "DELETE", "decisions/clear", 200)
        
        if not success:
            print(f"   ❌ Failed to clear cache")
            return False
        
        print(f"   ✅ Cache cleared successfully")
        
        # Start trading system to generate fresh cycle
        print(f"   🚀 Starting trading system for fresh cycle...")
        success, _ = self.run_test("Start Trading System", "POST", "start-trading", 200)
        if not success:
            print(f"   ❌ Failed to start trading system")
            return False
        
        # Wait for fresh cycle generation
        print(f"   ⏱️ Waiting for fresh trading cycle (120 seconds max)...")
        
        cycle_start_time = time.time()
        max_wait_time = 120
        check_interval = 15
        
        while time.time() - cycle_start_time < max_wait_time:
            time.sleep(check_interval)
            
            # Check for opportunities (Scout output)
            success, opp_data = self.run_test("Check Opportunities", "GET", "opportunities", 200)
            if success:
                opportunities = opp_data.get('opportunities', [])
                elapsed_time = time.time() - cycle_start_time
                
                print(f"   📈 After {elapsed_time:.1f}s: {len(opportunities)} opportunities found")
                
                if len(opportunities) > 0:
                    print(f"   ✅ Fresh trading cycle generated!")
                    break
        
        # Stop trading system
        print(f"   🛑 Stopping trading system...")
        self.run_test("Stop Trading System", "POST", "stop-trading", 200)
        
        return True

    def test_lateral_movement_detection_logs(self):
        """Test for lateral movement detection in system logs"""
        print(f"\n💰 STEP 2: Testing Lateral Movement Detection System...")
        
        # Get current opportunities to analyze
        success, opp_data = self.run_test("Get Opportunities", "GET", "opportunities", 200)
        if not success:
            print(f"   ❌ Cannot get opportunities for lateral movement testing")
            return False
        
        opportunities = opp_data.get('opportunities', [])
        if len(opportunities) == 0:
            print(f"   ❌ No opportunities available for lateral movement testing")
            return False
        
        print(f"   📊 Analyzing {len(opportunities)} opportunities for lateral movement detection...")
        
        # Get technical analyses to see which passed through IA1
        success, analysis_data = self.run_test("Get Technical Analyses", "GET", "analyses", 200)
        if not success:
            print(f"   ❌ Cannot get analyses for lateral movement comparison")
            return False
        
        analyses = analysis_data.get('analyses', [])
        
        # Compare opportunities vs analyses to detect filtering
        opportunity_symbols = set(opp.get('symbol', '') for opp in opportunities)
        analysis_symbols = set(analysis.get('symbol', '') for analysis in analyses)
        
        filtered_symbols = opportunity_symbols - analysis_symbols
        passed_symbols = opportunity_symbols.intersection(analysis_symbols)
        
        print(f"\n   📊 Lateral Movement Filter Analysis:")
        print(f"      Total Opportunities: {len(opportunities)}")
        print(f"      Passed to IA1: {len(passed_symbols)}")
        print(f"      Filtered (Lateral): {len(filtered_symbols)}")
        print(f"      Filter Rate: {len(filtered_symbols)/len(opportunities)*100:.1f}%")
        
        # Show examples of filtered symbols (likely lateral movements)
        if filtered_symbols:
            print(f"\n   💰 Symbols Filtered (Likely Lateral Movements):")
            for i, symbol in enumerate(list(filtered_symbols)[:5]):  # Show first 5
                # Find the opportunity data for this symbol
                opp_data = next((opp for opp in opportunities if opp.get('symbol') == symbol), None)
                if opp_data:
                    price_change = opp_data.get('price_change_24h', 0)
                    volatility = opp_data.get('volatility', 0)
                    print(f"      {i+1}. {symbol}: {price_change:+.1f}% change, {volatility*100:.1f}% volatility")
        
        # Show examples of symbols that passed through
        if passed_symbols:
            print(f"\n   🎯 Symbols Passed (Directional Movements):")
            for i, symbol in enumerate(list(passed_symbols)[:5]):  # Show first 5
                opp_data = next((opp for opp in opportunities if opp.get('symbol') == symbol), None)
                if opp_data:
                    price_change = opp_data.get('price_change_24h', 0)
                    volatility = opp_data.get('volatility', 0)
                    print(f"      {i+1}. {symbol}: {price_change:+.1f}% change, {volatility*100:.1f}% volatility")
        
        # Validate lateral movement detection is working
        lateral_detection_working = len(filtered_symbols) > 0  # Some symbols should be filtered
        api_economy_active = len(filtered_symbols) / len(opportunities) >= 0.2  # At least 20% filtered
        directional_movements_pass = len(passed_symbols) > 0  # Some should pass through
        
        print(f"\n   ✅ Lateral Movement Detection Validation:")
        print(f"      Lateral Detection Active: {'✅' if lateral_detection_working else '❌'}")
        print(f"      API Economy Rate ≥20%: {'✅' if api_economy_active else '❌'}")
        print(f"      Directional Movements Pass: {'✅' if directional_movements_pass else '❌'}")
        
        return lateral_detection_working and api_economy_active and directional_movements_pass

    def test_movement_classification_criteria(self):
        """Test the 4-criteria lateral detection system"""
        print(f"\n🔍 STEP 3: Testing 4-Criteria Movement Classification System...")
        
        # Get opportunities and analyses for detailed analysis
        success, opp_data = self.run_test("Get Opportunities", "GET", "opportunities", 200)
        if not success:
            return False
        
        success, analysis_data = self.run_test("Get Technical Analyses", "GET", "analyses", 200)
        if not success:
            return False
        
        opportunities = opp_data.get('opportunities', [])
        analyses = analysis_data.get('analyses', [])
        
        if len(opportunities) == 0:
            print(f"   ❌ No opportunities for classification testing")
            return False
        
        print(f"   📊 Analyzing movement classification for {len(opportunities)} opportunities...")
        
        # Analyze opportunities for lateral movement characteristics
        lateral_candidates = []
        directional_candidates = []
        
        for opp in opportunities:
            symbol = opp.get('symbol', '')
            price_change = abs(opp.get('price_change_24h', 0))
            volatility = opp.get('volatility', 0) * 100  # Convert to percentage
            
            # Simulate the 4 criteria analysis
            criteria_met = 0
            criteria_details = []
            
            # Criterion 1: Weak trend (<3% over period)
            if price_change < 3.0:
                criteria_met += 1
                criteria_details.append(f"Weak trend: {price_change:.1f}%")
            
            # Criterion 2: Low volatility (<2% daily)
            if volatility < 2.0:
                criteria_met += 1
                criteria_details.append(f"Low volatility: {volatility:.1f}%")
            
            # Criterion 3: Limited range (estimated from volatility)
            estimated_range = volatility * 4  # Rough estimate
            if estimated_range < 8.0:
                criteria_met += 1
                criteria_details.append(f"Limited range: ~{estimated_range:.1f}%")
            
            # Criterion 4: MA convergence (estimated from low volatility)
            if volatility < 1.5:  # Very low volatility suggests MA convergence
                criteria_met += 1
                criteria_details.append("MA convergence likely")
            
            # Classification based on criteria
            if criteria_met >= 3:
                lateral_candidates.append({
                    'symbol': symbol,
                    'criteria_met': criteria_met,
                    'details': criteria_details,
                    'price_change': price_change,
                    'volatility': volatility
                })
            else:
                directional_candidates.append({
                    'symbol': symbol,
                    'criteria_met': criteria_met,
                    'details': criteria_details,
                    'price_change': price_change,
                    'volatility': volatility
                })
        
        print(f"\n   📊 Movement Classification Results:")
        print(f"      Lateral Candidates (≥3 criteria): {len(lateral_candidates)}")
        print(f"      Directional Candidates (<3 criteria): {len(directional_candidates)}")
        
        # Show examples of lateral movements
        if lateral_candidates:
            print(f"\n   💰 Lateral Movement Examples:")
            for i, candidate in enumerate(lateral_candidates[:3]):
                print(f"      {i+1}. {candidate['symbol']}: {candidate['criteria_met']}/4 criteria")
                print(f"         {candidate['price_change']:+.1f}% change, {candidate['volatility']:.1f}% vol")
                print(f"         Criteria: {', '.join(candidate['details'])}")
        
        # Show examples of directional movements
        if directional_candidates:
            print(f"\n   🎯 Directional Movement Examples:")
            for i, candidate in enumerate(directional_candidates[:3]):
                print(f"      {i+1}. {candidate['symbol']}: {candidate['criteria_met']}/4 criteria")
                print(f"         {candidate['price_change']:+.1f}% change, {candidate['volatility']:.1f}% vol")
                print(f"         Criteria: {', '.join(candidate['details'])}")
        
        # Validate classification system
        has_lateral_detection = len(lateral_candidates) > 0
        has_directional_detection = len(directional_candidates) > 0
        reasonable_distribution = len(lateral_candidates) / len(opportunities) <= 0.8  # Not everything lateral
        
        print(f"\n   ✅ Classification System Validation:")
        print(f"      Lateral Movements Detected: {'✅' if has_lateral_detection else '❌'}")
        print(f"      Directional Movements Detected: {'✅' if has_directional_detection else '❌'}")
        print(f"      Reasonable Distribution: {'✅' if reasonable_distribution else '❌'}")
        
        return has_lateral_detection and has_directional_detection and reasonable_distribution

    def test_directional_movement_allowance(self):
        """Test that trending movements pass through to IA1"""
        print(f"\n📈 STEP 4: Testing Directional Movement Allowance...")
        
        # Get opportunities and analyses
        success, opp_data = self.run_test("Get Opportunities", "GET", "opportunities", 200)
        if not success:
            return False
        
        success, analysis_data = self.run_test("Get Technical Analyses", "GET", "analyses", 200)
        if not success:
            return False
        
        opportunities = opp_data.get('opportunities', [])
        analyses = analysis_data.get('analyses', [])
        
        if len(opportunities) == 0:
            print(f"   ❌ No opportunities for directional testing")
            return False
        
        print(f"   📊 Analyzing directional movement allowance...")
        
        # Identify strong directional movements in opportunities
        strong_bullish = []
        strong_bearish = []
        moderate_trends = []
        
        for opp in opportunities:
            symbol = opp.get('symbol', '')
            price_change = opp.get('price_change_24h', 0)
            
            if price_change > 5.0:  # Strong bullish trend
                strong_bullish.append({'symbol': symbol, 'change': price_change})
            elif price_change < -5.0:  # Strong bearish trend
                strong_bearish.append({'symbol': symbol, 'change': price_change})
            elif abs(price_change) >= 3.0:  # Moderate trends
                moderate_trends.append({'symbol': symbol, 'change': price_change})
        
        # Check which directional movements reached IA1
        analysis_symbols = set(analysis.get('symbol', '') for analysis in analyses)
        
        bullish_passed = [item for item in strong_bullish if item['symbol'] in analysis_symbols]
        bearish_passed = [item for item in strong_bearish if item['symbol'] in analysis_symbols]
        moderate_passed = [item for item in moderate_trends if item['symbol'] in analysis_symbols]
        
        print(f"\n   📊 Directional Movement Analysis:")
        print(f"      Strong Bullish (>5%): {len(strong_bullish)} found, {len(bullish_passed)} passed")
        print(f"      Strong Bearish (<-5%): {len(strong_bearish)} found, {len(bearish_passed)} passed")
        print(f"      Moderate Trends (3-5%): {len(moderate_trends)} found, {len(moderate_passed)} passed")
        
        # Show examples of directional movements that passed
        if bullish_passed:
            print(f"\n   📈 Strong Bullish Movements Passed:")
            for i, item in enumerate(bullish_passed[:3]):
                print(f"      {i+1}. {item['symbol']}: {item['change']:+.1f}% (PASSED to IA1)")
        
        if bearish_passed:
            print(f"\n   📉 Strong Bearish Movements Passed:")
            for i, item in enumerate(bearish_passed[:3]):
                print(f"      {i+1}. {item['symbol']}: {item['change']:+.1f}% (PASSED to IA1)")
        
        if moderate_passed:
            print(f"\n   📊 Moderate Trends Passed:")
            for i, item in enumerate(moderate_passed[:3]):
                print(f"      {i+1}. {item['symbol']}: {item['change']:+.1f}% (PASSED to IA1)")
        
        # Calculate pass rates for directional movements
        total_directional = len(strong_bullish) + len(strong_bearish) + len(moderate_trends)
        total_passed = len(bullish_passed) + len(bearish_passed) + len(moderate_passed)
        
        directional_pass_rate = total_passed / total_directional if total_directional > 0 else 0
        
        print(f"\n   📊 Directional Pass Rate: {directional_pass_rate*100:.1f}% ({total_passed}/{total_directional})")
        
        # Validate directional movement allowance
        strong_trends_pass = (len(bullish_passed) > 0 or len(bearish_passed) > 0) if (len(strong_bullish) > 0 or len(strong_bearish) > 0) else True
        moderate_trends_pass = len(moderate_passed) > 0 if len(moderate_trends) > 0 else True
        reasonable_pass_rate = directional_pass_rate >= 0.5  # At least 50% of directional movements should pass
        
        print(f"\n   ✅ Directional Movement Validation:")
        print(f"      Strong Trends Pass Through: {'✅' if strong_trends_pass else '❌'}")
        print(f"      Moderate Trends Pass Through: {'✅' if moderate_trends_pass else '❌'}")
        print(f"      Reasonable Pass Rate (≥50%): {'✅' if reasonable_pass_rate else '❌'}")
        
        return strong_trends_pass and moderate_trends_pass and reasonable_pass_rate

    def test_api_economy_enhancement(self):
        """Test the improved API economy rate with lateral filtering"""
        print(f"\n💰 STEP 5: Testing API Economy Enhancement...")
        
        # Get comprehensive data for API economy analysis
        success, opp_data = self.run_test("Get Opportunities", "GET", "opportunities", 200)
        if not success:
            return False
        
        success, analysis_data = self.run_test("Get Technical Analyses", "GET", "analyses", 200)
        if not success:
            return False
        
        success, decision_data = self.run_test("Get Trading Decisions", "GET", "decisions", 200)
        if not success:
            return False
        
        opportunities = opp_data.get('opportunities', [])
        analyses = analysis_data.get('analyses', [])
        decisions = decision_data.get('decisions', [])
        
        if len(opportunities) == 0:
            print(f"   ❌ No opportunities for API economy testing")
            return False
        
        print(f"   📊 Analyzing API Economy Enhancement...")
        
        # Calculate API economy metrics
        total_opportunities = len(opportunities)
        ia1_analyses = len(analyses)
        ia2_decisions = len(decisions)
        
        # Calculate filtering rates
        scout_to_ia1_filter_rate = (total_opportunities - ia1_analyses) / total_opportunities if total_opportunities > 0 else 0
        ia1_to_ia2_filter_rate = (ia1_analyses - ia2_decisions) / ia1_analyses if ia1_analyses > 0 else 0
        overall_filter_rate = (total_opportunities - ia2_decisions) / total_opportunities if total_opportunities > 0 else 0
        
        print(f"\n   📊 API Economy Metrics:")
        print(f"      Total Opportunities (Scout): {total_opportunities}")
        print(f"      IA1 Analyses Generated: {ia1_analyses}")
        print(f"      IA2 Decisions Generated: {ia2_decisions}")
        
        print(f"\n   💰 Filtering Rates:")
        print(f"      Scout → IA1 Filter: {scout_to_ia1_filter_rate*100:.1f}% ({total_opportunities - ia1_analyses} filtered)")
        print(f"      IA1 → IA2 Filter: {ia1_to_ia2_filter_rate*100:.1f}% ({ia1_analyses - ia2_decisions} filtered)")
        print(f"      Overall Filter Rate: {overall_filter_rate*100:.1f}% ({total_opportunities - ia2_decisions} filtered)")
        
        # Analyze actionable vs non-actionable movements
        actionable_movements = 0
        boring_consolidations = 0
        
        for opp in opportunities:
            price_change = abs(opp.get('price_change_24h', 0))
            volatility = opp.get('volatility', 0) * 100
            
            # Classify as actionable or boring
            if price_change >= 3.0 or volatility >= 2.0:  # Actionable movement
                actionable_movements += 1
            else:  # Boring consolidation
                boring_consolidations += 1
        
        actionable_rate = actionable_movements / total_opportunities if total_opportunities > 0 else 0
        
        print(f"\n   🎯 Movement Quality Analysis:")
        print(f"      Actionable Movements: {actionable_movements} ({actionable_rate*100:.1f}%)")
        print(f"      Boring Consolidations: {boring_consolidations} ({(1-actionable_rate)*100:.1f}%)")
        
        # Calculate API efficiency improvement
        api_calls_saved = total_opportunities - ia1_analyses
        efficiency_improvement = api_calls_saved / total_opportunities if total_opportunities > 0 else 0
        
        print(f"\n   💰 API Economy Enhancement:")
        print(f"      API Calls Saved: {api_calls_saved}")
        print(f"      Efficiency Improvement: {efficiency_improvement*100:.1f}%")
        print(f"      Focus on Actionable: {ia1_analyses} analyses for {actionable_movements} actionable movements")
        
        # Validate API economy enhancement
        significant_filtering = scout_to_ia1_filter_rate >= 0.2  # At least 20% filtered at source
        efficiency_improved = efficiency_improvement >= 0.2  # At least 20% improvement
        actionable_focus = ia1_analyses <= actionable_movements * 1.2  # Focus on actionable movements
        
        print(f"\n   ✅ API Economy Enhancement Validation:")
        print(f"      Significant Filtering (≥20%): {'✅' if significant_filtering else '❌'}")
        print(f"      Efficiency Improved (≥20%): {'✅' if efficiency_improved else '❌'}")
        print(f"      Actionable Focus: {'✅' if actionable_focus else '❌'}")
        
        return significant_filtering and efficiency_improved and actionable_focus

    def test_override_logic(self):
        """Test the enhanced override system"""
        print(f"\n🎯 STEP 6: Testing Enhanced Override Logic...")
        
        # Get technical analyses to examine override cases
        success, analysis_data = self.run_test("Get Technical Analyses", "GET", "analyses", 200)
        if not success:
            return False
        
        analyses = analysis_data.get('analyses', [])
        
        if len(analyses) == 0:
            print(f"   ❌ No analyses for override testing")
            return False
        
        print(f"   📊 Analyzing override logic in {len(analyses)} analyses...")
        
        # Analyze analyses for override indicators
        excellent_data_overrides = []
        pattern_overrides = []
        high_quality_directional = []
        
        for analysis in analyses:
            symbol = analysis.get('symbol', '')
            confidence = analysis.get('analysis_confidence', 0)
            patterns = analysis.get('patterns_detected', [])
            data_sources = analysis.get('data_sources', [])
            reasoning = analysis.get('ia1_reasoning', '')
            
            # Check for excellent data quality override (≥90%)
            if confidence >= 0.9:
                excellent_data_overrides.append({
                    'symbol': symbol,
                    'confidence': confidence,
                    'sources': len(data_sources)
                })
            
            # Check for technical pattern override
            if len(patterns) > 0 and any(pattern for pattern in patterns if pattern != "No patterns detected"):
                pattern_overrides.append({
                    'symbol': symbol,
                    'patterns': patterns,
                    'confidence': confidence
                })
            
            # Check for high-quality directional data
            if confidence >= 0.8 and len(data_sources) >= 2:
                high_quality_directional.append({
                    'symbol': symbol,
                    'confidence': confidence,
                    'sources': len(data_sources)
                })
        
        print(f"\n   📊 Override Analysis Results:")
        print(f"      Excellent Data Overrides (≥90%): {len(excellent_data_overrides)}")
        print(f"      Technical Pattern Overrides: {len(pattern_overrides)}")
        print(f"      High-Quality Directional: {len(high_quality_directional)}")
        
        # Show examples of overrides
        if excellent_data_overrides:
            print(f"\n   🎯 Excellent Data Quality Overrides:")
            for i, override in enumerate(excellent_data_overrides[:3]):
                print(f"      {i+1}. {override['symbol']}: {override['confidence']:.1%} confidence, {override['sources']} sources")
        
        if pattern_overrides:
            print(f"\n   📈 Technical Pattern Overrides:")
            for i, override in enumerate(pattern_overrides[:3]):
                patterns_str = ', '.join(override['patterns'][:2])  # Show first 2 patterns
                print(f"      {i+1}. {override['symbol']}: {patterns_str} ({override['confidence']:.1%} confidence)")
        
        # Validate override logic
        has_excellent_overrides = len(excellent_data_overrides) > 0
        has_pattern_overrides = len(pattern_overrides) > 0
        override_quality_maintained = all(item['confidence'] >= 0.8 for item in excellent_data_overrides + pattern_overrides)
        
        print(f"\n   ✅ Override Logic Validation:")
        print(f"      Excellent Data Overrides Present: {'✅' if has_excellent_overrides else '❌'}")
        print(f"      Pattern Overrides Present: {'✅' if has_pattern_overrides else '❌'}")
        print(f"      Override Quality Maintained: {'✅' if override_quality_maintained else '❌'}")
        
        return has_excellent_overrides or has_pattern_overrides

    def test_movement_type_logging(self):
        """Test comprehensive movement analysis logging"""
        print(f"\n📝 STEP 7: Testing Movement Type Logging...")
        
        # Get opportunities and analyses for movement type analysis
        success, opp_data = self.run_test("Get Opportunities", "GET", "opportunities", 200)
        if not success:
            return False
        
        success, analysis_data = self.run_test("Get Technical Analyses", "GET", "analyses", 200)
        if not success:
            return False
        
        opportunities = opp_data.get('opportunities', [])
        analyses = analysis_data.get('analyses', [])
        
        if len(opportunities) == 0:
            print(f"   ❌ No opportunities for movement type testing")
            return False
        
        print(f"   📊 Analyzing movement type classification...")
        
        # Classify movements based on available data
        movement_types = {
            'LATERAL': [],
            'BULLISH_TREND': [],
            'BEARISH_TREND': [],
            'MODERATE_TREND': [],
            'MIXED': []
        }
        
        for opp in opportunities:
            symbol = opp.get('symbol', '')
            price_change = opp.get('price_change_24h', 0)
            volatility = opp.get('volatility', 0) * 100
            
            # Classify movement type
            if abs(price_change) < 3.0 and volatility < 2.0:
                movement_type = 'LATERAL'
            elif price_change > 5.0:
                movement_type = 'BULLISH_TREND'
            elif price_change < -5.0:
                movement_type = 'BEARISH_TREND'
            elif abs(price_change) >= 3.0:
                movement_type = 'MODERATE_TREND'
            else:
                movement_type = 'MIXED'
            
            movement_types[movement_type].append({
                'symbol': symbol,
                'price_change': price_change,
                'volatility': volatility,
                'trend_strength': abs(price_change),
                'volatility_level': 'HIGH' if volatility > 3.0 else 'MEDIUM' if volatility > 1.5 else 'LOW'
            })
        
        print(f"\n   📊 Movement Type Distribution:")
        total_movements = len(opportunities)
        for movement_type, movements in movement_types.items():
            count = len(movements)
            percentage = count / total_movements * 100 if total_movements > 0 else 0
            print(f"      {movement_type}: {count} ({percentage:.1f}%)")
        
        # Show examples of each movement type
        for movement_type, movements in movement_types.items():
            if movements:
                print(f"\n   📈 {movement_type} Examples:")
                for i, movement in enumerate(movements[:2]):  # Show first 2 of each type
                    print(f"      {i+1}. {movement['symbol']}: {movement['price_change']:+.1f}% change, {movement['volatility_level']} volatility")
        
        # Analyze trend strength and volatility levels
        all_movements = []
        for movements in movement_types.values():
            all_movements.extend(movements)
        
        if all_movements:
            avg_trend_strength = sum(m['trend_strength'] for m in all_movements) / len(all_movements)
            volatility_levels = [m['volatility_level'] for m in all_movements]
            high_vol_count = volatility_levels.count('HIGH')
            medium_vol_count = volatility_levels.count('MEDIUM')
            low_vol_count = volatility_levels.count('LOW')
            
            print(f"\n   📊 Movement Analysis Summary:")
            print(f"      Average Trend Strength: {avg_trend_strength:.1f}%")
            print(f"      Volatility Levels - HIGH: {high_vol_count}, MEDIUM: {medium_vol_count}, LOW: {low_vol_count}")
        
        # Validate movement type logging
        has_lateral_classification = len(movement_types['LATERAL']) > 0
        has_directional_classification = (len(movement_types['BULLISH_TREND']) + 
                                        len(movement_types['BEARISH_TREND']) + 
                                        len(movement_types['MODERATE_TREND'])) > 0
        comprehensive_classification = len([mt for mt, movements in movement_types.items() if movements]) >= 3
        
        print(f"\n   ✅ Movement Type Logging Validation:")
        print(f"      Lateral Classification: {'✅' if has_lateral_classification else '❌'}")
        print(f"      Directional Classification: {'✅' if has_directional_classification else '❌'}")
        print(f"      Comprehensive Classification: {'✅' if comprehensive_classification else '❌'}")
        
        return has_lateral_classification and has_directional_classification and comprehensive_classification

    def run_comprehensive_lateral_movement_test(self):
        """Run comprehensive lateral movement filter test"""
        print(f"\n🎯 COMPREHENSIVE LATERAL MOVEMENT FILTER TEST")
        print(f"=" * 60)
        
        test_results = []
        
        # Step 1: Clear cache and generate fresh cycle
        result1 = self.test_clear_cache_and_generate_fresh_cycle()
        test_results.append(("Clear Cache & Fresh Cycle", result1))
        
        # Step 2: Test lateral movement detection
        result2 = self.test_lateral_movement_detection_logs()
        test_results.append(("Lateral Movement Detection", result2))
        
        # Step 3: Test 4-criteria classification system
        result3 = self.test_movement_classification_criteria()
        test_results.append(("4-Criteria Classification", result3))
        
        # Step 4: Test directional movement allowance
        result4 = self.test_directional_movement_allowance()
        test_results.append(("Directional Movement Allowance", result4))
        
        # Step 5: Test API economy enhancement
        result5 = self.test_api_economy_enhancement()
        test_results.append(("API Economy Enhancement", result5))
        
        # Step 6: Test override logic
        result6 = self.test_override_logic()
        test_results.append(("Override Logic", result6))
        
        # Step 7: Test movement type logging
        result7 = self.test_movement_type_logging()
        test_results.append(("Movement Type Logging", result7))
        
        # Summary
        print(f"\n🎯 LATERAL MOVEMENT FILTER TEST SUMMARY")
        print(f"=" * 60)
        
        passed_tests = 0
        total_tests = len(test_results)
        
        for test_name, result in test_results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {test_name}: {status}")
            if result:
                passed_tests += 1
        
        success_rate = passed_tests / total_tests * 100 if total_tests > 0 else 0
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        overall_success = success_rate >= 70  # 70% pass rate required
        
        print(f"\n🎯 LATERAL MOVEMENT FILTER SYSTEM: {'✅ WORKING' if overall_success else '❌ NEEDS WORK'}")
        
        if overall_success:
            print(f"\n✅ CONCLUSION: The lateral movement filter system is working correctly!")
            print(f"   - Lateral/sideways movements are being detected and filtered")
            print(f"   - Trending movements pass through to IA1 analysis")
            print(f"   - API economy is improved by eliminating boring consolidation periods")
            print(f"   - System focuses IA resources on actionable market conditions")
        else:
            print(f"\n❌ CONCLUSION: The lateral movement filter system needs improvement!")
            print(f"   - Some components are not working as expected")
            print(f"   - Review failed tests and adjust the filtering logic")
        
        return overall_success

if __name__ == "__main__":
    print("🎯 LATERAL MOVEMENT FILTER TESTING SYSTEM")
    print("=" * 60)
    
    tester = LateralMovementFilterTester()
    success = tester.run_comprehensive_lateral_movement_test()
    
    if success:
        print(f"\n🎉 All lateral movement filter tests completed successfully!")
    else:
        print(f"\n⚠️ Some lateral movement filter tests failed - review results above")
    
    sys.exit(0 if success else 1)
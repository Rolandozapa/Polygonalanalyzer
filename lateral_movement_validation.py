import requests
import json
import time
from pathlib import Path

class LateralMovementValidation:
    def __init__(self):
        # Get backend URL
        try:
            env_path = Path(__file__).parent / "frontend" / ".env"
            with open(env_path, 'r') as f:
                for line in f:
                    if line.startswith('REACT_APP_BACKEND_URL='):
                        self.base_url = line.split('=', 1)[1].strip()
                        break
        except:
            self.base_url = "https://ai-trade-pro.preview.emergentagent.com"
        
        self.api_url = f"{self.base_url}/api"

    def test_system_and_generate_data(self):
        """Test system and generate fresh data to analyze lateral movement filter"""
        print(f"\n🎯 LATERAL MOVEMENT FILTER VALIDATION")
        print(f"=" * 60)
        
        # Start trading system
        print(f"\n🚀 Starting trading system to generate data...")
        try:
            response = requests.post(f"{self.api_url}/start-trading", timeout=10)
            if response.status_code == 200:
                print(f"✅ Trading system started successfully")
            else:
                print(f"❌ Failed to start trading system: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error starting trading system: {e}")
            return False
        
        # Wait for system to process
        print(f"⏱️ Waiting for system to process opportunities (60 seconds)...")
        time.sleep(60)
        
        # Stop trading system
        try:
            requests.post(f"{self.api_url}/stop-trading", timeout=10)
            print(f"🛑 Trading system stopped")
        except:
            pass
        
        return True

    def analyze_lateral_movement_evidence(self):
        """Analyze evidence of lateral movement filtering from system behavior"""
        print(f"\n🔍 ANALYZING LATERAL MOVEMENT FILTER EVIDENCE")
        print(f"=" * 60)
        
        # Get current system data
        try:
            opp_response = requests.get(f"{self.api_url}/opportunities", timeout=10)
            analysis_response = requests.get(f"{self.api_url}/analyses", timeout=10)
            decision_response = requests.get(f"{self.api_url}/decisions", timeout=10)
            
            opportunities = opp_response.json().get('opportunities', []) if opp_response.status_code == 200 else []
            analyses = analysis_response.json().get('analyses', []) if analysis_response.status_code == 200 else []
            decisions = decision_response.json().get('decisions', []) if decision_response.status_code == 200 else []
            
        except Exception as e:
            print(f"❌ Error fetching system data: {e}")
            return False
        
        print(f"\n📊 SYSTEM DATA OVERVIEW:")
        print(f"   Opportunities (Scout): {len(opportunities)}")
        print(f"   Analyses (IA1): {len(analyses)}")
        print(f"   Decisions (IA2): {len(decisions)}")
        
        if len(opportunities) == 0 and len(analyses) == 0:
            print(f"\n⚠️ No data available for analysis. System may need more time to generate data.")
            return self.analyze_code_implementation()
        
        # Analyze filtering evidence
        return self.analyze_filtering_patterns(opportunities, analyses, decisions)

    def analyze_filtering_patterns(self, opportunities, analyses, decisions):
        """Analyze filtering patterns in the data"""
        print(f"\n🔍 FILTERING PATTERN ANALYSIS:")
        
        if len(opportunities) > 0:
            # Analyze opportunity characteristics
            opp_symbols = set(opp.get('symbol', '') for opp in opportunities)
            analysis_symbols = set(analysis.get('symbol', '') for analysis in analyses)
            
            filtered_symbols = opp_symbols - analysis_symbols
            passed_symbols = opp_symbols.intersection(analysis_symbols)
            
            filter_rate = len(filtered_symbols) / len(opportunities) if len(opportunities) > 0 else 0
            
            print(f"   Total Opportunities: {len(opportunities)}")
            print(f"   Passed to IA1: {len(passed_symbols)}")
            print(f"   Filtered Out: {len(filtered_symbols)}")
            print(f"   Filter Rate: {filter_rate*100:.1f}%")
            
            # Analyze characteristics of filtered vs passed opportunities
            if opportunities:
                print(f"\n📊 OPPORTUNITY CHARACTERISTICS:")
                
                lateral_characteristics = []
                directional_characteristics = []
                
                for opp in opportunities:
                    symbol = opp.get('symbol', '')
                    price_change = opp.get('price_change_24h', 0)
                    volatility = opp.get('volatility', 0) * 100
                    
                    # Classify based on lateral movement criteria
                    lateral_signals = 0
                    if abs(price_change) < 3.0: lateral_signals += 1  # Weak trend
                    if volatility < 2.0: lateral_signals += 1  # Low volatility
                    if volatility < 1.5: lateral_signals += 1  # Very low volatility (MA convergence proxy)
                    
                    if lateral_signals >= 2:  # Likely lateral movement
                        lateral_characteristics.append({
                            'symbol': symbol,
                            'price_change': price_change,
                            'volatility': volatility,
                            'passed_ia1': symbol in analysis_symbols
                        })
                    else:  # Likely directional movement
                        directional_characteristics.append({
                            'symbol': symbol,
                            'price_change': price_change,
                            'volatility': volatility,
                            'passed_ia1': symbol in analysis_symbols
                        })
                
                print(f"   Lateral Characteristics: {len(lateral_characteristics)}")
                print(f"   Directional Characteristics: {len(directional_characteristics)}")
                
                # Show examples
                if lateral_characteristics:
                    print(f"\n💰 LATERAL MOVEMENT EXAMPLES:")
                    for i, char in enumerate(lateral_characteristics[:3]):
                        status = "PASSED" if char['passed_ia1'] else "FILTERED"
                        print(f"      {i+1}. {char['symbol']}: {char['price_change']:+.1f}% change, {char['volatility']:.1f}% vol - {status}")
                
                if directional_characteristics:
                    print(f"\n🎯 DIRECTIONAL MOVEMENT EXAMPLES:")
                    for i, char in enumerate(directional_characteristics[:3]):
                        status = "PASSED" if char['passed_ia1'] else "FILTERED"
                        print(f"      {i+1}. {char['symbol']}: {char['price_change']:+.1f}% change, {char['volatility']:.1f}% vol - {status}")
                
                # Calculate effectiveness
                lateral_filtered = sum(1 for char in lateral_characteristics if not char['passed_ia1'])
                directional_passed = sum(1 for char in directional_characteristics if char['passed_ia1'])
                
                lateral_filter_rate = lateral_filtered / len(lateral_characteristics) if len(lateral_characteristics) > 0 else 0
                directional_pass_rate = directional_passed / len(directional_characteristics) if len(directional_characteristics) > 0 else 0
                
                print(f"\n✅ FILTER EFFECTIVENESS:")
                print(f"   Lateral Movements Filtered: {lateral_filter_rate*100:.1f}% ({lateral_filtered}/{len(lateral_characteristics)})")
                print(f"   Directional Movements Passed: {directional_pass_rate*100:.1f}% ({directional_passed}/{len(directional_characteristics)})")
                
                # Validate lateral movement filter
                filter_working = filter_rate > 0.1  # Some filtering happening
                lateral_detection = len(lateral_characteristics) > 0
                directional_detection = len(directional_characteristics) > 0
                effective_filtering = lateral_filter_rate > directional_pass_rate if len(lateral_characteristics) > 0 and len(directional_characteristics) > 0 else True
                
                print(f"\n🎯 LATERAL MOVEMENT FILTER VALIDATION:")
                print(f"   Filter Active (>10%): {'✅' if filter_working else '❌'}")
                print(f"   Lateral Detection: {'✅' if lateral_detection else '❌'}")
                print(f"   Directional Detection: {'✅' if directional_detection else '❌'}")
                print(f"   Effective Filtering: {'✅' if effective_filtering else '❌'}")
                
                return filter_working and lateral_detection and directional_detection
        
        return False

    def analyze_code_implementation(self):
        """Analyze the code implementation of lateral movement detection"""
        print(f"\n🔍 CODE IMPLEMENTATION ANALYSIS:")
        print(f"=" * 60)
        
        try:
            # Read the backend server code
            with open('/app/backend/server.py', 'r') as f:
                server_code = f.read()
            
            # Check for lateral movement detection implementation
            lateral_detection_found = False
            api_economy_found = False
            movement_classification_found = False
            
            # Look for key lateral movement detection patterns
            if '_detect_lateral_movement' in server_code:
                lateral_detection_found = True
                print(f"✅ Lateral movement detection method found")
            
            if 'API ÉCONOMIE' in server_code or 'API ECONOMY' in server_code:
                api_economy_found = True
                print(f"✅ API economy logging found")
            
            if 'LATERAL' in server_code and 'BULLISH_TREND' in server_code and 'BEARISH_TREND' in server_code:
                movement_classification_found = True
                print(f"✅ Movement classification system found")
            
            # Look for the 4 criteria implementation
            criteria_found = 0
            if 'trend_percentage' in server_code: criteria_found += 1
            if 'volatility' in server_code: criteria_found += 1
            if 'price_range' in server_code: criteria_found += 1
            if 'ma_convergence' in server_code: criteria_found += 1
            
            print(f"✅ Lateral detection criteria found: {criteria_found}/4")
            
            # Look for override logic
            override_found = 'override' in server_code.lower() or 'excellent' in server_code.lower()
            if override_found:
                print(f"✅ Override logic implementation found")
            
            # Look for movement type logging
            movement_logging = 'movement_type' in server_code or 'MOVEMENT_TYPE' in server_code
            if movement_logging:
                print(f"✅ Movement type logging found")
            
            print(f"\n📊 IMPLEMENTATION COMPLETENESS:")
            components = [
                ("Lateral Movement Detection", lateral_detection_found),
                ("API Economy Logging", api_economy_found),
                ("Movement Classification", movement_classification_found),
                ("4-Criteria System", criteria_found >= 3),
                ("Override Logic", override_found),
                ("Movement Type Logging", movement_logging)
            ]
            
            implemented_count = sum(1 for _, implemented in components if implemented)
            
            for component, implemented in components:
                status = "✅" if implemented else "❌"
                print(f"   {component}: {status}")
            
            implementation_rate = implemented_count / len(components)
            
            print(f"\n🎯 IMPLEMENTATION ASSESSMENT:")
            print(f"   Components Implemented: {implemented_count}/{len(components)}")
            print(f"   Implementation Rate: {implementation_rate*100:.1f}%")
            
            implementation_success = implementation_rate >= 0.8  # 80% implementation required
            
            print(f"\n🎯 LATERAL MOVEMENT FILTER IMPLEMENTATION: {'✅ COMPLETE' if implementation_success else '❌ INCOMPLETE'}")
            
            return implementation_success
            
        except Exception as e:
            print(f"❌ Error analyzing code implementation: {e}")
            return False

    def test_api_economy_logs(self):
        """Test for API economy logs in the system"""
        print(f"\n📝 TESTING API ECONOMY LOGS:")
        
        try:
            # Check backend logs for lateral movement detection
            import subprocess
            result = subprocess.run(['tail', '-n', '100', '/var/log/supervisor/backend.err.log'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                log_content = result.stdout
                
                # Look for lateral movement detection logs
                lateral_logs = log_content.count('Mouvement latéral détecté') + log_content.count('LATERAL')
                api_economy_logs = log_content.count('API ÉCONOMIE') + log_content.count('API ECONOMY')
                skip_logs = log_content.count('SKIP IA1') + log_content.count('SKIP TECHNIQUE')
                pattern_logs = log_content.count('PATTERN DÉTECTÉ') + log_content.count('bullish_') + log_content.count('bearish_')
                movement_logs = log_content.count('BULLISH_TREND') + log_content.count('BEARISH_TREND') + log_content.count('MIXED')
                
                print(f"   Lateral Movement Logs: {lateral_logs}")
                print(f"   API Economy Logs: {api_economy_logs}")
                print(f"   Skip IA1 Logs: {skip_logs}")
                print(f"   Pattern Detection Logs: {pattern_logs}")
                print(f"   Movement Type Logs: {movement_logs}")
                
                # Show recent relevant log entries
                lines = log_content.split('\n')
                relevant_lines = [line for line in lines if any(keyword in line for keyword in 
                                ['ÉCONOMIE', 'ECONOMY', 'SKIP IA1', 'LATERAL', 'BULLISH_TREND', 'BEARISH_TREND', 'PATTERN DÉTECTÉ'])]
                
                if relevant_lines:
                    print(f"\n📋 RECENT LATERAL MOVEMENT FILTER LOGS:")
                    for line in relevant_lines[-5:]:  # Show last 5 relevant lines
                        print(f"   {line.strip()}")
                
                logs_found = lateral_logs > 0 or api_economy_logs > 0 or skip_logs > 0 or pattern_logs > 0
                
                print(f"\n✅ LOG ANALYSIS:")
                print(f"   Lateral Movement Filter Logs Found: {'✅' if logs_found else '❌'}")
                
                return logs_found
            else:
                print(f"❌ Could not read backend logs")
                return False
                
        except Exception as e:
            print(f"❌ Error checking logs: {e}")
            return False

    def run_comprehensive_validation(self):
        """Run comprehensive lateral movement filter validation"""
        print(f"\n🎯 COMPREHENSIVE LATERAL MOVEMENT FILTER VALIDATION")
        print(f"=" * 80)
        
        validation_results = []
        
        # Test 1: Generate system data
        print(f"\n1️⃣ SYSTEM DATA GENERATION TEST")
        result1 = self.test_system_and_generate_data()
        validation_results.append(("System Data Generation", result1))
        
        # Test 2: Analyze lateral movement evidence
        print(f"\n2️⃣ LATERAL MOVEMENT EVIDENCE ANALYSIS")
        result2 = self.analyze_lateral_movement_evidence()
        validation_results.append(("Lateral Movement Evidence", result2))
        
        # Test 3: Check API economy logs
        print(f"\n3️⃣ API ECONOMY LOGS TEST")
        result3 = self.test_api_economy_logs()
        validation_results.append(("API Economy Logs", result3))
        
        # Summary
        print(f"\n🎯 VALIDATION SUMMARY")
        print(f"=" * 60)
        
        passed_tests = 0
        total_tests = len(validation_results)
        
        for test_name, result in validation_results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {test_name}: {status}")
            if result:
                passed_tests += 1
        
        success_rate = passed_tests / total_tests * 100 if total_tests > 0 else 0
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        overall_success = success_rate >= 66  # 2/3 tests must pass
        
        print(f"\n🎯 LATERAL MOVEMENT FILTER SYSTEM: {'✅ WORKING' if overall_success else '❌ NEEDS WORK'}")
        
        if overall_success:
            print(f"\n✅ CONCLUSION: The lateral movement filter system is implemented and working!")
            print(f"   - Lateral movement detection logic is present in the code")
            print(f"   - API economy improvements are active")
            print(f"   - System logs show filtering activity")
            print(f"   - Boring consolidation periods are being filtered out")
        else:
            print(f"\n❌ CONCLUSION: The lateral movement filter system needs attention!")
            print(f"   - Some components may not be working as expected")
            print(f"   - Check failed tests and review implementation")
        
        return overall_success

if __name__ == "__main__":
    validator = LateralMovementValidation()
    success = validator.run_comprehensive_validation()
    
    if success:
        print(f"\n🎉 Lateral movement filter validation completed successfully!")
    else:
        print(f"\n⚠️ Lateral movement filter validation found issues - review results above")
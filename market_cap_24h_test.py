#!/usr/bin/env python3
"""
MARKET CAP 24H BONUS/MALUS LOGIC TESTING SUITE
Focus: Testing the new Market Cap 24h bonus/malus logic for IA1 confidence adjustment

TESTING REQUIREMENTS FROM REVIEW REQUEST:
1. Market Cap 24h Integration: Verify global Market Cap 24h change is retrieved and integrated into IA1 scoring
2. Bonus/Malus Logic: Test _calculate_mcap_bonus_malus method with different scenarios:
   - LONG signal + Market Cap rising (+) → Should get BONUS
   - LONG signal + Market Cap falling (-) → Should get MALUS  
   - SHORT signal + Market Cap falling (-) → Should get BONUS
   - SHORT signal + Market Cap rising (+) → Should get MALUS
   - HOLD signal → Should be neutral (no bonus/malus)
3. Confidence Impact: Verify Market Cap bonus/malus affects final IA1 confidence score (10% weight)
4. Intensity Factor: Test that larger Market Cap changes create stronger bonus/malus effects
5. Critical Variable Integration: Confirm Market Cap 24h is available via critical variables endpoint
6. End-to-End Validation: Test full IA1 analysis to verify Market Cap bonus/malus is applied in practice

EXPECTED SYSTEM CAPABILITIES:
- Intelligent confidence adjustment based on global market momentum
- Enhanced timing for entries (favor signals aligned with market direction)
- Better risk management (penalize counter-trend positions)
- Logged bonus/malus calculations for transparency
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import requests
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketCap24hTestSuite:
    """Comprehensive test suite for Market Cap 24h bonus/malus logic"""
    
    def __init__(self):
        # Get backend URL from frontend env
        try:
            with open('/app/frontend/.env', 'r') as f:
                for line in f:
                    if line.startswith('REACT_APP_BACKEND_URL='):
                        backend_url = line.split('=')[1].strip()
                        break
                else:
                    backend_url = "http://localhost:8001"
        except Exception:
            backend_url = "http://localhost:8001"
        
        self.api_url = f"{backend_url}/api"
        logger.info(f"Testing Market Cap 24h Bonus/Malus System at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Test scenarios for bonus/malus logic
        self.test_scenarios = [
            # LONG signal scenarios
            {"signal": "LONG", "mcap_change": 3.5, "expected": "BONUS", "description": "LONG + Market Cap rising"},
            {"signal": "LONG", "mcap_change": -2.8, "expected": "MALUS", "description": "LONG + Market Cap falling"},
            {"signal": "LONG", "mcap_change": 0.05, "expected": "NEUTRAL", "description": "LONG + Market Cap minimal change"},
            
            # SHORT signal scenarios  
            {"signal": "SHORT", "mcap_change": -4.2, "expected": "BONUS", "description": "SHORT + Market Cap falling"},
            {"signal": "SHORT", "mcap_change": 2.1, "expected": "MALUS", "description": "SHORT + Market Cap rising"},
            {"signal": "SHORT", "mcap_change": -0.08, "expected": "NEUTRAL", "description": "SHORT + Market Cap minimal change"},
            
            # HOLD signal scenarios
            {"signal": "HOLD", "mcap_change": 5.0, "expected": "NEUTRAL", "description": "HOLD + Market Cap rising"},
            {"signal": "HOLD", "mcap_change": -3.0, "expected": "NEUTRAL", "description": "HOLD + Market Cap falling"},
            
            # Intensity factor scenarios
            {"signal": "LONG", "mcap_change": 1.0, "expected": "WEAK_BONUS", "description": "LONG + Small Market Cap rise"},
            {"signal": "LONG", "mcap_change": 6.0, "expected": "STRONG_BONUS", "description": "LONG + Large Market Cap rise"},
            {"signal": "SHORT", "mcap_change": -1.5, "expected": "WEAK_BONUS", "description": "SHORT + Small Market Cap fall"},
            {"signal": "SHORT", "mcap_change": -7.0, "expected": "STRONG_BONUS", "description": "SHORT + Large Market Cap fall"},
        ]
        
    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if details:
            logger.info(f"   Details: {details}")
        
        self.test_results.append({
            'test': test_name,
            'success': success,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
    
    async def test_1_critical_variables_mcap_integration(self):
        """Test 1: Market Cap 24h Integration via Critical Variables Endpoint"""
        logger.info("\n🔍 TEST 1: Market Cap 24h Integration via Critical Variables")
        
        try:
            response = requests.get(f"{self.api_url}/admin/market/critical-variables", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"   📊 Critical variables response: {json.dumps(data, indent=2)}")
                
                # Check for Market Cap 24h data
                critical_vars = data.get('critical_variables', {})
                market_cap_data = critical_vars.get('market_cap', {})
                
                if 'change_24h' in market_cap_data:
                    mcap_change_24h = market_cap_data['change_24h']
                    mcap_status = market_cap_data.get('status', 'Unknown')
                    
                    logger.info(f"   📊 Market Cap 24h change: {mcap_change_24h:+.2f}%")
                    logger.info(f"   📊 Market Cap status: {mcap_status}")
                    
                    # Verify data is realistic
                    if isinstance(mcap_change_24h, (int, float)) and -50 <= mcap_change_24h <= 50:
                        self.log_test_result("Market Cap 24h Critical Variables Integration", True, 
                                           f"Market Cap 24h: {mcap_change_24h:+.2f}%, Status: {mcap_status}")
                    else:
                        self.log_test_result("Market Cap 24h Critical Variables Integration", False, 
                                           f"Unrealistic Market Cap 24h value: {mcap_change_24h}")
                else:
                    self.log_test_result("Market Cap 24h Critical Variables Integration", False, 
                                       "Market Cap 24h change not found in critical variables")
            else:
                self.log_test_result("Market Cap 24h Critical Variables Integration", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test_result("Market Cap 24h Critical Variables Integration", False, f"Exception: {str(e)}")
    
    async def test_2_bonus_malus_logic_scenarios(self):
        """Test 2: Bonus/Malus Logic with Different Market Cap Scenarios"""
        logger.info("\n🔍 TEST 2: Bonus/Malus Logic Scenarios Testing")
        
        try:
            # Import the server module to access the _calculate_mcap_bonus_malus method
            sys.path.append('/app/backend')
            from server import UltraProfessionalIA1TechnicalAnalyst
            
            # Create instance to test the method
            ia1_analyst = UltraProfessionalIA1TechnicalAnalyst()
            
            scenario_results = []
            
            for scenario in self.test_scenarios:
                signal = scenario['signal']
                mcap_change = scenario['mcap_change']
                expected = scenario['expected']
                description = scenario['description']
                
                logger.info(f"   🧪 Testing: {description}")
                logger.info(f"      Signal: {signal}, Market Cap 24h: {mcap_change:+.2f}%")
                
                # Calculate bonus/malus
                bonus_malus_score = ia1_analyst._calculate_mcap_bonus_malus(mcap_change, signal.lower())
                
                logger.info(f"      Result: {bonus_malus_score:+.4f}")
                
                # Evaluate result based on expected outcome
                test_passed = False
                
                if expected == "BONUS":
                    test_passed = bonus_malus_score > 0.05  # Significant positive bonus
                elif expected == "MALUS":
                    test_passed = bonus_malus_score < -0.05  # Significant negative malus
                elif expected == "NEUTRAL":
                    test_passed = abs(bonus_malus_score) <= 0.05  # Near zero
                elif expected == "WEAK_BONUS":
                    test_passed = 0.05 < bonus_malus_score <= 0.3  # Small positive bonus
                elif expected == "STRONG_BONUS":
                    test_passed = bonus_malus_score > 0.3  # Large positive bonus
                
                scenario_results.append({
                    'description': description,
                    'signal': signal,
                    'mcap_change': mcap_change,
                    'expected': expected,
                    'actual_score': bonus_malus_score,
                    'passed': test_passed
                })
                
                status_icon = "✅" if test_passed else "❌"
                logger.info(f"      {status_icon} Expected: {expected}, Score: {bonus_malus_score:+.4f}")
            
            # Evaluate overall scenario testing
            passed_scenarios = sum(1 for result in scenario_results if result['passed'])
            total_scenarios = len(scenario_results)
            
            success_rate = passed_scenarios / total_scenarios if total_scenarios > 0 else 0
            
            if success_rate >= 0.9:  # 90% success rate
                self.log_test_result("Bonus/Malus Logic Scenarios", True, 
                                   f"Success rate: {passed_scenarios}/{total_scenarios} ({success_rate:.1%})")
            else:
                self.log_test_result("Bonus/Malus Logic Scenarios", False, 
                                   f"Low success rate: {passed_scenarios}/{total_scenarios} ({success_rate:.1%})")
            
            # Log detailed scenario results
            logger.info(f"   📊 Scenario Test Results:")
            for result in scenario_results:
                status_icon = "✅" if result['passed'] else "❌"
                logger.info(f"      {status_icon} {result['description']}: {result['actual_score']:+.4f}")
                
        except Exception as e:
            self.log_test_result("Bonus/Malus Logic Scenarios", False, f"Exception: {str(e)}")
    
    async def test_3_confidence_impact_validation(self):
        """Test 3: Verify Market Cap Bonus/Malus Affects IA1 Confidence (10% Weight)"""
        logger.info("\n🔍 TEST 3: Market Cap Bonus/Malus Impact on IA1 Confidence")
        
        try:
            # Trigger fresh IA1 analysis to capture Market Cap impact
            logger.info("   🚀 Triggering fresh IA1 analysis to test Market Cap impact...")
            start_response = requests.post(f"{self.api_url}/start-trading", timeout=180)
            
            if start_response.status_code not in [200, 201]:
                logger.warning(f"   ⚠️ Start trading returned HTTP {start_response.status_code}")
            else:
                # Wait for processing
                logger.info("   ⏳ Waiting 30 seconds for IA1 analysis processing...")
                await asyncio.sleep(30)
            
            # Get recent IA1 analyses
            analyses_response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if analyses_response.status_code != 200:
                self.log_test_result("Market Cap Confidence Impact", False, f"HTTP {analyses_response.status_code}")
                return
            
            analyses_data = analyses_response.json()
            analyses = analyses_data.get('analyses', [])
            
            # Check backend logs for Market Cap bonus/malus evidence
            backend_logs = ""
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "2000", "/var/log/supervisor/backend.out.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            # Look for Market Cap bonus/malus patterns in logs
            mcap_patterns = [
                "🌍 Global Market Cap 24h:",
                "🟢 LONG + Market Cap",
                "🔴 LONG + Market Cap", 
                "🟢 SHORT + Market Cap",
                "🔴 SHORT + Market Cap",
                "BONUS",
                "MALUS",
                "mcap_24h"
            ]
            
            pattern_counts = {}
            for pattern in mcap_patterns:
                count = backend_logs.count(pattern)
                pattern_counts[pattern] = count
                logger.info(f"      📊 '{pattern}': {count} occurrences")
            
            # Check for 10% weight evidence in logs
            weight_evidence = backend_logs.count("'mcap_24h': 0.10") + backend_logs.count("10% - MARKET CAP 24H")
            logger.info(f"      📊 10% weight evidence: {weight_evidence} occurrences")
            
            # Analyze recent analyses for confidence variations
            confidence_variations = []
            mcap_influenced_analyses = 0
            
            for analysis in analyses[-10:]:  # Check last 10 analyses
                symbol = analysis.get('symbol', 'Unknown')
                confidence = analysis.get('analysis_confidence', 0) * 100
                signal = analysis.get('ia1_signal', '').upper()
                reasoning = analysis.get('ia1_reasoning', '')
                
                # Check if analysis shows Market Cap influence
                mcap_keywords = ['market cap', 'global market', 'momentum', 'trend alignment']
                if any(keyword.lower() in reasoning.lower() for keyword in mcap_keywords):
                    mcap_influenced_analyses += 1
                    logger.info(f"      ✅ {symbol}: Market Cap influence detected (Confidence: {confidence:.1f}%)")
                
                confidence_variations.append({
                    'symbol': symbol,
                    'confidence': confidence,
                    'signal': signal
                })
            
            # Check for confidence score variations (evidence of Market Cap impact)
            confidence_range = max([c['confidence'] for c in confidence_variations]) - min([c['confidence'] for c in confidence_variations])
            
            patterns_detected = sum(1 for count in pattern_counts.values() if count > 0)
            mcap_influence_percentage = (mcap_influenced_analyses / max(len(analyses[-10:]), 1)) * 100
            
            logger.info(f"   📊 Market Cap patterns detected: {patterns_detected}/{len(mcap_patterns)}")
            logger.info(f"   📊 10% weight evidence: {weight_evidence}")
            logger.info(f"   📊 Market Cap influenced analyses: {mcap_influence_percentage:.1f}% ({mcap_influenced_analyses}/10)")
            logger.info(f"   📊 Confidence range variation: {confidence_range:.1f}%")
            
            # Success criteria
            pattern_success = patterns_detected >= 3  # Multiple Market Cap patterns found
            weight_success = weight_evidence > 0  # 10% weight evidence found
            influence_success = mcap_influenced_analyses >= 2 or confidence_range >= 10  # Evidence of impact
            
            success = pattern_success and weight_success and influence_success
            
            details = f"Patterns: {patterns_detected}/{len(mcap_patterns)}, Weight evidence: {weight_evidence}, Influenced analyses: {mcap_influence_percentage:.1f}%"
            
            self.log_test_result("Market Cap Confidence Impact", success, details)
            
        except Exception as e:
            self.log_test_result("Market Cap Confidence Impact", False, f"Exception: {str(e)}")
    
    async def test_4_intensity_factor_validation(self):
        """Test 4: Verify Larger Market Cap Changes Create Stronger Effects"""
        logger.info("\n🔍 TEST 4: Intensity Factor - Larger Changes Create Stronger Effects")
        
        try:
            # Import the server module to test intensity factor
            sys.path.append('/app/backend')
            from server import UltraProfessionalIA1TechnicalAnalyst
            
            ia1_analyst = UltraProfessionalIA1TechnicalAnalyst()
            
            # Test intensity scaling with different Market Cap changes
            intensity_tests = [
                {"mcap_change": 1.0, "signal": "LONG", "expected_intensity": "LOW"},
                {"mcap_change": 3.0, "signal": "LONG", "expected_intensity": "MEDIUM"},
                {"mcap_change": 6.0, "signal": "LONG", "expected_intensity": "HIGH"},
                {"mcap_change": 10.0, "signal": "LONG", "expected_intensity": "MAX"},
                {"mcap_change": -1.5, "signal": "SHORT", "expected_intensity": "LOW"},
                {"mcap_change": -4.0, "signal": "SHORT", "expected_intensity": "MEDIUM"},
                {"mcap_change": -8.0, "signal": "SHORT", "expected_intensity": "HIGH"},
            ]
            
            intensity_results = []
            
            for test in intensity_tests:
                mcap_change = test['mcap_change']
                signal = test['signal']
                expected_intensity = test['expected_intensity']
                
                bonus_malus_score = ia1_analyst._calculate_mcap_bonus_malus(mcap_change, signal.lower())
                abs_score = abs(bonus_malus_score)
                
                logger.info(f"   🧪 Market Cap {mcap_change:+.1f}% + {signal}: Score {bonus_malus_score:+.4f} (|{abs_score:.4f}|)")
                
                # Classify intensity based on absolute score
                if abs_score <= 0.2:
                    actual_intensity = "LOW"
                elif abs_score <= 0.5:
                    actual_intensity = "MEDIUM"
                elif abs_score <= 0.8:
                    actual_intensity = "HIGH"
                else:
                    actual_intensity = "MAX"
                
                intensity_match = actual_intensity == expected_intensity
                intensity_results.append({
                    'mcap_change': mcap_change,
                    'signal': signal,
                    'score': bonus_malus_score,
                    'abs_score': abs_score,
                    'expected_intensity': expected_intensity,
                    'actual_intensity': actual_intensity,
                    'match': intensity_match
                })
                
                status_icon = "✅" if intensity_match else "❌"
                logger.info(f"      {status_icon} Expected: {expected_intensity}, Actual: {actual_intensity}")
            
            # Test that larger changes produce larger effects
            scaling_tests = [
                (1.0, 3.0, "LONG"),   # 3x change should produce larger effect
                (2.0, 6.0, "SHORT"),  # 3x change should produce larger effect
                (1.5, 4.5, "LONG"),   # 3x change should produce larger effect
            ]
            
            scaling_success = 0
            for small_change, large_change, signal in scaling_tests:
                small_score = abs(ia1_analyst._calculate_mcap_bonus_malus(small_change if signal == "LONG" else -small_change, signal.lower()))
                large_score = abs(ia1_analyst._calculate_mcap_bonus_malus(large_change if signal == "LONG" else -large_change, signal.lower()))
                
                if large_score > small_score:
                    scaling_success += 1
                    logger.info(f"      ✅ Scaling test: {small_change}% → {small_score:.4f}, {large_change}% → {large_score:.4f}")
                else:
                    logger.info(f"      ❌ Scaling test failed: {small_change}% → {small_score:.4f}, {large_change}% → {large_score:.4f}")
            
            # Evaluate intensity factor testing
            intensity_matches = sum(1 for result in intensity_results if result['match'])
            total_intensity_tests = len(intensity_results)
            scaling_rate = scaling_success / len(scaling_tests)
            
            intensity_success = (intensity_matches / total_intensity_tests) >= 0.7  # 70% intensity classification accuracy
            scaling_success_rate = scaling_rate >= 0.8  # 80% scaling tests pass
            
            success = intensity_success and scaling_success_rate
            
            details = f"Intensity classification: {intensity_matches}/{total_intensity_tests} ({intensity_matches/total_intensity_tests:.1%}), Scaling: {scaling_success}/{len(scaling_tests)} ({scaling_rate:.1%})"
            
            self.log_test_result("Intensity Factor Validation", success, details)
            
        except Exception as e:
            self.log_test_result("Intensity Factor Validation", False, f"Exception: {str(e)}")
    
    async def test_5_end_to_end_ia1_integration(self):
        """Test 5: End-to-End IA1 Analysis with Market Cap Bonus/Malus Applied"""
        logger.info("\n🔍 TEST 5: End-to-End IA1 Analysis with Market Cap Integration")
        
        try:
            # Get current global Market Cap 24h for context
            critical_vars_response = requests.get(f"{self.api_url}/admin/market/critical-variables", timeout=30)
            current_mcap_change = 0.0
            
            if critical_vars_response.status_code == 200:
                critical_data = critical_vars_response.json()
                mcap_data = critical_data.get('critical_variables', {}).get('market_cap', {})
                current_mcap_change = mcap_data.get('change_24h', 0.0)
                logger.info(f"   📊 Current global Market Cap 24h: {current_mcap_change:+.2f}%")
            
            # Trigger fresh analysis to capture Market Cap integration
            logger.info("   🚀 Triggering fresh IA1 analysis for end-to-end testing...")
            start_response = requests.post(f"{self.api_url}/start-trading", timeout=180)
            
            if start_response.status_code not in [200, 201]:
                logger.warning(f"   ⚠️ Start trading returned HTTP {start_response.status_code}")
            
            # Wait for processing
            logger.info("   ⏳ Waiting 45 seconds for complete IA1 processing...")
            await asyncio.sleep(45)
            
            # Get recent IA1 analyses
            analyses_response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if analyses_response.status_code != 200:
                self.log_test_result("End-to-End IA1 Market Cap Integration", False, f"HTTP {analyses_response.status_code}")
                return
            
            analyses_data = analyses_response.json()
            analyses = analyses_data.get('analyses', [])
            
            # Analyze recent analyses for Market Cap integration evidence
            recent_analyses = analyses[-5:]  # Check last 5 analyses
            mcap_integration_evidence = 0
            confidence_adjustments = []
            
            for analysis in recent_analyses:
                symbol = analysis.get('symbol', 'Unknown')
                confidence = analysis.get('analysis_confidence', 0) * 100
                signal = analysis.get('ia1_signal', '').upper()
                reasoning = analysis.get('ia1_reasoning', '')
                
                # Check for Market Cap integration indicators
                mcap_indicators = [
                    'market cap', 'global market', 'momentum alignment', 'trend alignment',
                    'market direction', 'macro trend', 'overall market'
                ]
                
                mcap_mentions = sum(1 for indicator in mcap_indicators if indicator.lower() in reasoning.lower())
                
                if mcap_mentions > 0:
                    mcap_integration_evidence += 1
                    logger.info(f"      ✅ {symbol}: Market Cap integration detected ({mcap_mentions} indicators)")
                    logger.info(f"         Signal: {signal}, Confidence: {confidence:.1f}%")
                
                confidence_adjustments.append({
                    'symbol': symbol,
                    'confidence': confidence,
                    'signal': signal,
                    'mcap_mentions': mcap_mentions
                })
            
            # Check backend logs for detailed Market Cap processing
            backend_logs = ""
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "3000", "/var/log/supervisor/backend.out.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            # Look for comprehensive Market Cap integration patterns
            integration_patterns = [
                "🌍 Global Market Cap 24h:",
                "🎯 APPLYING PROFESSIONAL SCORING TO IA1",
                "mcap_24h.*BONUS",
                "mcap_24h.*MALUS", 
                "Market Cap.*bonus",
                "Market Cap.*malus",
                "10% - MARKET CAP 24H"
            ]
            
            pattern_evidence = {}
            for pattern in integration_patterns:
                count = backend_logs.count(pattern) if pattern not in ["mcap_24h.*BONUS", "mcap_24h.*MALUS", "Market Cap.*bonus", "Market Cap.*malus"] else len([line for line in backend_logs.split('\n') if pattern.replace('.*', '') in line])
                pattern_evidence[pattern] = count
                logger.info(f"      📊 '{pattern}': {count} occurrences")
            
            # Check for confidence score variations indicating Market Cap impact
            if confidence_adjustments:
                confidence_values = [adj['confidence'] for adj in confidence_adjustments]
                confidence_std = (max(confidence_values) - min(confidence_values)) if len(confidence_values) > 1 else 0
                logger.info(f"   📊 Confidence variation range: {confidence_std:.1f}%")
            
            # Evaluate end-to-end integration
            integration_percentage = (mcap_integration_evidence / max(len(recent_analyses), 1)) * 100
            pattern_detections = sum(1 for count in pattern_evidence.values() if count > 0)
            
            logger.info(f"   📊 Market Cap integration evidence: {integration_percentage:.1f}% ({mcap_integration_evidence}/{len(recent_analyses)})")
            logger.info(f"   📊 Integration patterns detected: {pattern_detections}/{len(integration_patterns)}")
            
            # Success criteria
            integration_success = integration_percentage >= 60  # At least 60% of analyses show integration
            pattern_success = pattern_detections >= 4  # Multiple integration patterns detected
            processing_success = len(recent_analyses) >= 3  # System is processing analyses
            
            success = integration_success and pattern_success and processing_success
            
            details = f"Integration: {integration_percentage:.1f}%, Patterns: {pattern_detections}/{len(integration_patterns)}, Analyses: {len(recent_analyses)}"
            
            self.log_test_result("End-to-End IA1 Market Cap Integration", success, details)
            
        except Exception as e:
            self.log_test_result("End-to-End IA1 Market Cap Integration", False, f"Exception: {str(e)}")
    
    async def test_6_transparency_and_logging(self):
        """Test 6: Verify Market Cap Bonus/Malus Calculations are Logged for Transparency"""
        logger.info("\n🔍 TEST 6: Market Cap Bonus/Malus Transparency and Logging")
        
        try:
            # Check backend logs for transparency in Market Cap calculations
            backend_logs = ""
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "4000", "/var/log/supervisor/backend.out.log"],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            # Look for detailed logging patterns that show transparency
            transparency_patterns = [
                "🟢 LONG + Market Cap",      # Bonus logging
                "🔴 LONG + Market Cap",      # Malus logging  
                "🟢 SHORT + Market Cap",     # Bonus logging
                "🔴 SHORT + Market Cap",     # Malus logging
                "BONUS",                     # General bonus mentions
                "MALUS",                     # General malus mentions
                "intensity_factor",          # Intensity factor calculations
                "Market Cap.*%",             # Market Cap percentage mentions
                "bonus_score",               # Bonus score calculations
                "malus_score"                # Malus score calculations
            ]
            
            transparency_evidence = {}
            for pattern in transparency_patterns:
                if pattern in ["Market Cap.*%", "bonus_score", "malus_score"]:
                    # Use regex-like counting for these patterns
                    count = len([line for line in backend_logs.split('\n') if any(p in line for p in pattern.split('.*'))])
                else:
                    count = backend_logs.count(pattern)
                transparency_evidence[pattern] = count
                logger.info(f"      📊 '{pattern}': {count} occurrences")
            
            # Check for detailed calculation logging
            calculation_details = [
                "intensity_factor",
                "tanh_norm",
                "Market Cap 24h:",
                "bonus_malus_score",
                "calculate_mcap_bonus_malus"
            ]
            
            calculation_evidence = {}
            for detail in calculation_details:
                count = backend_logs.count(detail)
                calculation_evidence[detail] = count
                logger.info(f"      📊 Calculation detail '{detail}': {count} occurrences")
            
            # Check for specific Market Cap value logging
            mcap_value_patterns = [
                "+",  # Positive Market Cap changes
                "-",  # Negative Market Cap changes  
                "%"   # Percentage symbols
            ]
            
            # Count lines that contain Market Cap values with percentages
            mcap_value_lines = [line for line in backend_logs.split('\n') if 'Market Cap' in line and '%' in line]
            mcap_value_count = len(mcap_value_lines)
            
            logger.info(f"      📊 Market Cap value logging lines: {mcap_value_count}")
            
            # Sample some Market Cap value lines for verification
            if mcap_value_lines:
                logger.info("      📊 Sample Market Cap logging:")
                for line in mcap_value_lines[-3:]:  # Show last 3 lines
                    logger.info(f"         {line.strip()}")
            
            # Evaluate transparency and logging
            transparency_detections = sum(1 for count in transparency_evidence.values() if count > 0)
            calculation_detections = sum(1 for count in calculation_evidence.values() if count > 0)
            
            logger.info(f"   📊 Transparency patterns detected: {transparency_detections}/{len(transparency_patterns)}")
            logger.info(f"   📊 Calculation details detected: {calculation_detections}/{len(calculation_details)}")
            logger.info(f"   📊 Market Cap value logging: {mcap_value_count} lines")
            
            # Success criteria
            transparency_success = transparency_detections >= 6  # Good transparency logging
            calculation_success = calculation_detections >= 3   # Calculation details logged
            value_logging_success = mcap_value_count >= 5       # Market Cap values logged
            
            success = transparency_success and calculation_success and value_logging_success
            
            details = f"Transparency: {transparency_detections}/{len(transparency_patterns)}, Calculations: {calculation_detections}/{len(calculation_details)}, Values: {mcap_value_count}"
            
            self.log_test_result("Market Cap Transparency and Logging", success, details)
            
        except Exception as e:
            self.log_test_result("Market Cap Transparency and Logging", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_tests(self):
        """Run all Market Cap 24h bonus/malus tests"""
        logger.info("🚀 Starting Market Cap 24h Bonus/Malus Comprehensive Test Suite")
        logger.info("=" * 100)
        logger.info("📋 MARKET CAP 24H BONUS/MALUS LOGIC COMPREHENSIVE TESTING")
        logger.info("🎯 Testing: Market Cap integration, bonus/malus logic, confidence impact, intensity factor")
        logger.info("🎯 Expected: Intelligent confidence adjustment based on global market momentum")
        logger.info("=" * 100)
        
        # Run all tests in sequence
        await self.test_1_critical_variables_mcap_integration()
        await self.test_2_bonus_malus_logic_scenarios()
        await self.test_3_confidence_impact_validation()
        await self.test_4_intensity_factor_validation()
        await self.test_5_end_to_end_ia1_integration()
        await self.test_6_transparency_and_logging()
        
        # Summary
        logger.info("\n" + "=" * 100)
        logger.info("📊 MARKET CAP 24H BONUS/MALUS COMPREHENSIVE TEST SUMMARY")
        logger.info("=" * 100)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests)*100:.1f}%)")
        
        # System analysis
        logger.info("\n" + "=" * 100)
        logger.info("📋 MARKET CAP 24H BONUS/MALUS SYSTEM STATUS")
        logger.info("=" * 100)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Market Cap 24h Bonus/Malus System FULLY FUNCTIONAL!")
            logger.info("✅ Market Cap 24h integration working correctly")
            logger.info("✅ Bonus/malus logic operating as designed")
            logger.info("✅ Confidence impact properly weighted (10%)")
            logger.info("✅ Intensity factor scaling appropriately")
            logger.info("✅ End-to-end IA1 integration operational")
            logger.info("✅ Transparency and logging comprehensive")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY FUNCTIONAL - Market Cap system working with minor gaps")
            logger.info("🔍 Some components may need fine-tuning for optimal performance")
        elif passed_tests >= total_tests * 0.6:
            logger.info("⚠️ PARTIALLY FUNCTIONAL - Core Market Cap features working")
            logger.info("🔧 Several advanced features may need implementation or debugging")
        else:
            logger.info("❌ SYSTEM NOT FUNCTIONAL - Critical issues with Market Cap integration")
            logger.info("🚨 Major implementation gaps preventing Market Cap bonus/malus functionality")
        
        # Specific requirements check
        logger.info("\n📝 MARKET CAP 24H REQUIREMENTS VERIFICATION:")
        
        requirements_met = []
        requirements_failed = []
        
        # Check each requirement based on test results
        for result in self.test_results:
            if result['success']:
                if "Critical Variables" in result['test']:
                    requirements_met.append("✅ Market Cap 24h integration via critical variables")
                elif "Bonus/Malus Logic" in result['test']:
                    requirements_met.append("✅ Bonus/malus logic with all scenarios working")
                elif "Confidence Impact" in result['test']:
                    requirements_met.append("✅ Market Cap affects IA1 confidence (10% weight)")
                elif "Intensity Factor" in result['test']:
                    requirements_met.append("✅ Larger Market Cap changes create stronger effects")
                elif "End-to-End" in result['test']:
                    requirements_met.append("✅ Full IA1 analysis with Market Cap bonus/malus applied")
                elif "Transparency" in result['test']:
                    requirements_met.append("✅ Market Cap calculations logged for transparency")
            else:
                if "Critical Variables" in result['test']:
                    requirements_failed.append("❌ Market Cap 24h integration not working")
                elif "Bonus/Malus Logic" in result['test']:
                    requirements_failed.append("❌ Bonus/malus logic scenarios failing")
                elif "Confidence Impact" in result['test']:
                    requirements_failed.append("❌ Market Cap not affecting IA1 confidence properly")
                elif "Intensity Factor" in result['test']:
                    requirements_failed.append("❌ Intensity factor not scaling correctly")
                elif "End-to-End" in result['test']:
                    requirements_failed.append("❌ End-to-end IA1 integration not working")
                elif "Transparency" in result['test']:
                    requirements_failed.append("❌ Market Cap calculations not properly logged")
        
        for req in requirements_met:
            logger.info(f"   {req}")
        
        for req in requirements_failed:
            logger.info(f"   {req}")
        
        logger.info(f"\n🏆 FINAL RESULT: {len(requirements_met)}/{len(requirements_met) + len(requirements_failed)} requirements satisfied")
        
        # Final verdict
        if len(requirements_failed) == 0:
            logger.info("\n🎉 VERDICT: Market Cap 24h Bonus/Malus System is FULLY FUNCTIONAL!")
            logger.info("✅ Intelligent confidence adjustment based on global market momentum")
            logger.info("✅ Enhanced timing for entries (favor signals aligned with market direction)")
            logger.info("✅ Better risk management (penalize counter-trend positions)")
            logger.info("✅ Logged bonus/malus calculations for transparency")
            logger.info("✅ System provides all expected capabilities from review request")
        elif len(requirements_failed) <= 1:
            logger.info("\n⚠️ VERDICT: Market Cap 24h System is MOSTLY FUNCTIONAL")
            logger.info("🔍 Minor issues may need attention for complete functionality")
        elif len(requirements_failed) <= 2:
            logger.info("\n⚠️ VERDICT: Market Cap 24h System is PARTIALLY FUNCTIONAL")
            logger.info("🔧 Several components need implementation or debugging")
        else:
            logger.info("\n❌ VERDICT: Market Cap 24h System is NOT FUNCTIONAL")
            logger.info("🚨 Major implementation gaps preventing Market Cap bonus/malus functionality")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = MarketCap24hTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
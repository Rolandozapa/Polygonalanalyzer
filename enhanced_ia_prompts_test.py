#!/usr/bin/env python3
"""
Enhanced IA1 and IA2 Prompts Testing Suite with Technical Indicators Integration
Focus: Testing the newly enhanced IA1 and IA2 prompts with RSI, MACD, Stochastic, Bollinger Bands
Review Request: Test enhanced technical indicators integration and confidence/RR thresholds
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedIAPromptsTestSuite:
    """Test suite for Enhanced IA1 and IA2 Prompts with Technical Indicators Integration"""
    
    def __init__(self):
        # Use localhost for testing (external URL has routing issues)
        backend_url = "http://localhost:8001"
        
        self.api_url = f"{backend_url}/api"
        logger.info(f"Testing Enhanced IA1 and IA2 Prompts at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Test symbols for analysis
        self.test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT"]
        
        # Expected technical indicators
        self.expected_indicators = ["RSI", "MACD", "Stochastic", "Bollinger Bands"]
        
        # IA1 confidence and RR thresholds
        self.ia1_confidence_threshold = 70.0  # ≥70% for IA2 escalation
        self.ia1_rr_threshold = 2.0  # ≥2:1 Risk-Reward for IA2 escalation
        
        # IA2 execution threshold
        self.ia2_execution_threshold = 80.0  # ≥80% for trade execution
        
        # Store analysis data for cross-test validation
        self.ia1_analyses = []
        self.ia2_decisions = []
        
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
    
    async def test_ia1_enhanced_technical_analysis(self):
        """Test 1: IA1 Enhanced Technical Analysis with RSI, MACD, Stochastic, Bollinger Bands"""
        logger.info("\n🔍 TEST 1: IA1 Enhanced Technical Analysis with Technical Indicators")
        
        try:
            # Get recent IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("IA1 Enhanced Technical Analysis", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            analyses = response.json()
            
            if not analyses or 'analyses' not in analyses:
                self.log_test_result("IA1 Enhanced Technical Analysis", False, "No IA1 analyses found")
                return
            
            # Store for later tests
            self.ia1_analyses = analyses.get('analyses', [])
            
            # Analyze technical indicators integration
            indicators_found = {"RSI": 0, "MACD": 0, "Stochastic": 0, "Bollinger Bands": 0}
            confluence_analyses = 0
            enhanced_precision_count = 0
            total_analyses = len(analyses.get('analyses', []))
            
            for analysis in analyses.get('analyses', []):
                # Check for RSI values (should be present as numeric field)
                if 'rsi' in analysis and analysis['rsi'] != 50.0:  # 50.0 is default fallback
                    indicators_found["RSI"] += 1
                
                # Check for MACD values (should be present as numeric field)
                if 'macd_signal' in analysis and analysis['macd_signal'] != 0.0:  # 0.0 is default fallback
                    indicators_found["MACD"] += 1
                
                # Check for Bollinger Bands (bollinger_position field)
                if 'bollinger_position' in analysis and analysis['bollinger_position'] != 0.0:
                    indicators_found["Bollinger Bands"] += 1
                
                # Check for Stochastic in reasoning text (may not be separate field yet)
                analysis_text = str(analysis).lower()
                if 'stochastic' in analysis_text:
                    indicators_found["Stochastic"] += 1
                
                # Check for confluence analysis in reasoning
                reasoning = analysis.get('ia1_reasoning', '').lower()
                confluence_keywords = ["confluence", "align", "contradict", "mixed signals", "multi-rr", "technical precision"]
                if any(keyword in reasoning for keyword in confluence_keywords):
                    confluence_analyses += 1
                
                # Check for enhanced precision keywords
                precision_keywords = ["enhanced", "advanced", "technical indicators", "oscillator", "momentum", "volatility", "multi-rr"]
                if any(keyword in reasoning for keyword in precision_keywords):
                    enhanced_precision_count += 1
            
            # Calculate success metrics
            indicators_coverage = sum(1 for count in indicators_found.values() if count > 0)
            confluence_rate = (confluence_analyses / total_analyses) * 100
            precision_rate = (enhanced_precision_count / total_analyses) * 100
            
            logger.info(f"   📊 Total IA1 analyses: {total_analyses}")
            logger.info(f"   📊 Technical indicators coverage: {indicators_coverage}/{len(self.expected_indicators)}")
            for indicator, count in indicators_found.items():
                logger.info(f"      {indicator}: {count}/{total_analyses} analyses ({(count/total_analyses)*100:.1f}%)")
            logger.info(f"   📊 Confluence analysis rate: {confluence_analyses}/{total_analyses} ({confluence_rate:.1f}%)")
            logger.info(f"   📊 Enhanced precision rate: {enhanced_precision_count}/{total_analyses} ({precision_rate:.1f}%)")
            
            # Success criteria: At least 3/4 indicators present AND confluence analysis ≥50% AND precision ≥70%
            success = (indicators_coverage >= 3 and confluence_rate >= 50 and precision_rate >= 70)
            
            details = f"Indicators: {indicators_coverage}/4, Confluence: {confluence_rate:.1f}%, Precision: {precision_rate:.1f}%"
            
            self.log_test_result("IA1 Enhanced Technical Analysis", success, details)
            
        except Exception as e:
            self.log_test_result("IA1 Enhanced Technical Analysis", False, f"Exception: {str(e)}")
    
    async def test_ia1_confidence_rr_thresholds(self):
        """Test 2: IA1 Confidence & RR Thresholds (≥70% confidence AND ≥2:1 RR for IA2 escalation)"""
        logger.info("\n🔍 TEST 2: IA1 Confidence & RR Thresholds for IA2 Escalation")
        
        try:
            if not self.ia1_analyses:
                self.log_test_result("IA1 Confidence & RR Thresholds", False, "No IA1 analyses available from previous test")
                return
            
            # Analyze confidence and RR filtering
            high_confidence_analyses = []
            high_rr_analyses = []
            escalated_to_ia2 = []
            
            for analysis in self.ia1_analyses:
                # Extract confidence
                confidence = analysis.get('analysis_confidence')
                rr_ratio = analysis.get('risk_reward_ratio')
                
                if confidence is not None and confidence >= (self.ia1_confidence_threshold / 100.0):
                    high_confidence_analyses.append(analysis)
                
                if rr_ratio is not None and rr_ratio >= self.ia1_rr_threshold:
                    high_rr_analyses.append(analysis)
                
                # Check if both thresholds met (should escalate to IA2)
                if (confidence is not None and confidence >= (self.ia1_confidence_threshold / 100.0) and
                    rr_ratio is not None and rr_ratio >= self.ia1_rr_threshold):
                    escalated_to_ia2.append(analysis)
            
            # Get IA2 decisions to verify escalation
            try:
                response = requests.get(f"{self.api_url}/decisions", timeout=30)
                if response.status_code == 200:
                    decisions_data = response.json()
                    ia2_decisions = decisions_data.get('decisions', [])
                    self.ia2_decisions = ia2_decisions
                    actual_ia2_count = len(ia2_decisions)
                else:
                    actual_ia2_count = 0
            except:
                actual_ia2_count = 0
            
            total_analyses = len(self.ia1_analyses)
            high_confidence_count = len(high_confidence_analyses)
            high_rr_count = len(high_rr_analyses)
            expected_escalations = len(escalated_to_ia2)
            
            logger.info(f"   📊 Total IA1 analyses: {total_analyses}")
            logger.info(f"   📊 High confidence (≥{self.ia1_confidence_threshold}%): {high_confidence_count}/{total_analyses}")
            logger.info(f"   📊 High RR (≥{self.ia1_rr_threshold}:1): {high_rr_count}/{total_analyses}")
            logger.info(f"   📊 Expected IA2 escalations: {expected_escalations}")
            logger.info(f"   📊 Actual IA2 decisions: {actual_ia2_count}")
            
            # Success criteria: Filtering logic working (escalations match expectations within reasonable range)
            escalation_efficiency = abs(actual_ia2_count - expected_escalations) <= max(2, expected_escalations * 0.3)
            threshold_compliance = high_confidence_count > 0 and high_rr_count > 0
            
            success = escalation_efficiency and threshold_compliance
            
            details = f"Expected escalations: {expected_escalations}, Actual: {actual_ia2_count}, Efficiency: {escalation_efficiency}"
            
            self.log_test_result("IA1 Confidence & RR Thresholds", success, details)
            
        except Exception as e:
            self.log_test_result("IA1 Confidence & RR Thresholds", False, f"Exception: {str(e)}")
    
    async def test_ia2_enhanced_decision_making(self):
        """Test 3: IA2 Enhanced Decision Making with Technical Indicators Analysis"""
        logger.info("\n🔍 TEST 3: IA2 Enhanced Decision Making with Technical Indicators")
        
        try:
            if not self.ia2_decisions:
                # Try to get IA2 decisions
                response = requests.get(f"{self.api_url}/decisions", timeout=30)
                if response.status_code != 200:
                    self.log_test_result("IA2 Enhanced Decision Making", False, f"HTTP {response.status_code}: {response.text}")
                    return
                
                decisions_data = response.json()
                self.ia2_decisions = decisions_data.get('decisions', [])
            
            if not self.ia2_decisions or len(self.ia2_decisions) == 0:
                self.log_test_result("IA2 Enhanced Decision Making", False, "No IA2 decisions found")
                return
            
            # Analyze technical indicators integration in IA2 decisions
            technical_indicators_analysis_count = 0
            rsi_impact_count = 0
            macd_influence_count = 0
            stochastic_timing_count = 0
            bollinger_volatility_count = 0
            confluence_score_count = 0
            
            total_decisions = len(self.ia2_decisions)
            
            for decision in self.ia2_decisions:
                reasoning = decision.get('ia2_reasoning', '').lower()
                
                # Check for technical_indicators_analysis section
                if "technical" in reasoning and ("indicators" in reasoning or "rsi" in reasoning or "macd" in reasoning):
                    technical_indicators_analysis_count += 1
                
                # Check for specific indicator impacts
                if "rsi" in reasoning and ("impact" in reasoning or "overbought" in reasoning or "oversold" in reasoning or "rsi at" in reasoning):
                    rsi_impact_count += 1
                
                if "macd" in reasoning and ("influence" in reasoning or "crossover" in reasoning or "momentum" in reasoning or "macd" in reasoning):
                    macd_influence_count += 1
                
                if "stochastic" in reasoning and ("timing" in reasoning or "%k" in reasoning or "%d" in reasoning):
                    stochastic_timing_count += 1
                
                if "bollinger" in reasoning and ("volatility" in reasoning or "band" in reasoning or "squeeze" in reasoning or "position" in reasoning):
                    bollinger_volatility_count += 1
                
                if "confluence" in reasoning and ("score" in reasoning or "align" in reasoning or "technical" in reasoning):
                    confluence_score_count += 1
            
            # Calculate success metrics
            technical_analysis_rate = (technical_indicators_analysis_count / total_decisions) * 100
            rsi_rate = (rsi_impact_count / total_decisions) * 100
            macd_rate = (macd_influence_count / total_decisions) * 100
            stochastic_rate = (stochastic_timing_count / total_decisions) * 100
            bollinger_rate = (bollinger_volatility_count / total_decisions) * 100
            confluence_rate = (confluence_score_count / total_decisions) * 100
            
            logger.info(f"   📊 Total IA2 decisions: {total_decisions}")
            logger.info(f"   📊 Technical indicators analysis: {technical_indicators_analysis_count}/{total_decisions} ({technical_analysis_rate:.1f}%)")
            logger.info(f"   📊 RSI impact analysis: {rsi_impact_count}/{total_decisions} ({rsi_rate:.1f}%)")
            logger.info(f"   📊 MACD influence analysis: {macd_influence_count}/{total_decisions} ({macd_rate:.1f}%)")
            logger.info(f"   📊 Stochastic timing analysis: {stochastic_timing_count}/{total_decisions} ({stochastic_rate:.1f}%)")
            logger.info(f"   📊 Bollinger volatility analysis: {bollinger_volatility_count}/{total_decisions} ({bollinger_rate:.1f}%)")
            logger.info(f"   📊 Confluence score analysis: {confluence_score_count}/{total_decisions} ({confluence_rate:.1f}%)")
            
            # Success criteria: Technical analysis ≥70% AND at least 3/4 specific indicators ≥50%
            indicator_rates = [rsi_rate, macd_rate, stochastic_rate, bollinger_rate]
            indicators_above_threshold = sum(1 for rate in indicator_rates if rate >= 50)
            
            success = technical_analysis_rate >= 70 and indicators_above_threshold >= 3
            
            details = f"Technical analysis: {technical_analysis_rate:.1f}%, Indicators ≥50%: {indicators_above_threshold}/4"
            
            self.log_test_result("IA2 Enhanced Decision Making", success, details)
            
        except Exception as e:
            self.log_test_result("IA2 Enhanced Decision Making", False, f"Exception: {str(e)}")
    
    async def test_ia2_execution_threshold(self):
        """Test 4: IA2 Execution Threshold (≥80% confidence for execution)"""
        logger.info("\n🔍 TEST 4: IA2 Execution Threshold (≥80% confidence)")
        
        try:
            if not self.ia2_decisions:
                self.log_test_result("IA2 Execution Threshold", False, "No IA2 decisions available from previous test")
                return
            
            # Analyze execution decisions based on confidence
            high_confidence_decisions = []
            low_confidence_decisions = []
            executed_trades = []
            hold_decisions = []
            
            for decision in self.ia2_decisions:
                confidence = decision.get('confidence')
                signal = decision.get('signal', '').upper()
                
                if confidence is not None:
                    if confidence >= (self.ia2_execution_threshold / 100.0):
                        high_confidence_decisions.append(decision)
                        if signal in ['LONG', 'SHORT', 'BUY', 'SELL']:
                            executed_trades.append(decision)
                    else:
                        low_confidence_decisions.append(decision)
                        if signal == 'HOLD':
                            hold_decisions.append(decision)
            
            total_decisions = len(self.ia2_decisions)
            high_confidence_count = len(high_confidence_decisions)
            low_confidence_count = len(low_confidence_decisions)
            executed_count = len(executed_trades)
            hold_count = len(hold_decisions)
            
            logger.info(f"   📊 Total IA2 decisions: {total_decisions}")
            logger.info(f"   📊 High confidence (≥{self.ia2_execution_threshold}%): {high_confidence_count}/{total_decisions}")
            logger.info(f"   📊 Low confidence (<{self.ia2_execution_threshold}%): {low_confidence_count}/{total_decisions}")
            logger.info(f"   📊 Executed trades: {executed_count}")
            logger.info(f"   📊 HOLD decisions: {hold_count}")
            
            # Success criteria: High confidence leads to execution, low confidence leads to HOLD
            execution_compliance = True
            hold_compliance = True
            
            # Check if high confidence decisions are being executed
            if high_confidence_count > 0:
                execution_rate = (executed_count / high_confidence_count) * 100
                execution_compliance = execution_rate >= 70  # At least 70% of high confidence should execute
                logger.info(f"   📊 Execution rate for high confidence: {execution_rate:.1f}%")
            
            # Check if low confidence decisions are being held
            if low_confidence_count > 0:
                hold_rate = (hold_count / low_confidence_count) * 100
                hold_compliance = hold_rate >= 70  # At least 70% of low confidence should hold
                logger.info(f"   📊 Hold rate for low confidence: {hold_rate:.1f}%")
            
            success = execution_compliance and hold_compliance and (high_confidence_count > 0 or low_confidence_count > 0)
            
            details = f"High conf: {high_confidence_count}, Low conf: {low_confidence_count}, Exec compliance: {execution_compliance}, Hold compliance: {hold_compliance}"
            
            self.log_test_result("IA2 Execution Threshold", success, details)
            
        except Exception as e:
            self.log_test_result("IA2 Execution Threshold", False, f"Exception: {str(e)}")
    
    async def test_enhanced_adaptive_context(self):
        """Test 5: Enhanced Adaptive Context System with Technical Indicators Fields"""
        logger.info("\n🔍 TEST 5: Enhanced Adaptive Context System")
        
        try:
            # Test adaptive context endpoints
            context_endpoints = [
                "/adaptive-context/status",
                "/adaptive-context/current",
                "/adaptive-context/update"
            ]
            
            context_data = {}
            successful_endpoints = 0
            
            for endpoint in context_endpoints:
                try:
                    response = requests.get(f"{self.api_url}{endpoint}", timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        context_data[endpoint] = data
                        successful_endpoints += 1
                        logger.info(f"   ✅ {endpoint}: Success")
                    else:
                        logger.info(f"   ❌ {endpoint}: HTTP {response.status_code}")
                except Exception as e:
                    logger.info(f"   ❌ {endpoint}: Exception {str(e)}")
            
            # Check for enhanced technical indicators fields
            enhanced_fields = [
                "stochastic_environment",
                "bollinger_environment", 
                "technical_confluence",
                "indicators_divergence",
                "momentum_regime",
                "volatility_regime"
            ]
            
            fields_found = 0
            context_text = str(context_data).lower()
            
            for field in enhanced_fields:
                if field.lower() in context_text:
                    fields_found += 1
                    logger.info(f"   ✅ Enhanced field found: {field}")
            
            logger.info(f"   📊 Successful endpoints: {successful_endpoints}/{len(context_endpoints)}")
            logger.info(f"   📊 Enhanced fields found: {fields_found}/{len(enhanced_fields)}")
            
            # Success criteria: At least 2/3 endpoints working AND at least 4/6 enhanced fields present
            success = successful_endpoints >= 2 and fields_found >= 4
            
            details = f"Endpoints: {successful_endpoints}/3, Enhanced fields: {fields_found}/6"
            
            self.log_test_result("Enhanced Adaptive Context System", success, details)
            
        except Exception as e:
            self.log_test_result("Enhanced Adaptive Context System", False, f"Exception: {str(e)}")
    
    async def test_end_to_end_enhanced_flow(self):
        """Test 6: End-to-End Enhanced Flow with Multiple Crypto Symbols"""
        logger.info("\n🔍 TEST 6: End-to-End Enhanced Flow Testing")
        
        try:
            # Test the complete flow: Scout → IA1 (enhanced) → IA2 (enhanced) → Execute
            flow_results = {}
            
            # 1. Test Scout opportunities
            try:
                response = requests.get(f"{self.api_url}/opportunities", timeout=30)
                if response.status_code == 200:
                    opportunities_data = response.json()
                    opportunities = opportunities_data.get('opportunities', [])
                    flow_results['scout'] = len(opportunities)
                    logger.info(f"   📊 Scout opportunities: {len(opportunities)}")
                else:
                    flow_results['scout'] = 0
            except:
                flow_results['scout'] = 0
            
            # 2. Test IA1 analyses (already have from previous tests)
            flow_results['ia1'] = len(self.ia1_analyses) if self.ia1_analyses else 0
            logger.info(f"   📊 IA1 analyses: {flow_results['ia1']}")
            
            # 3. Test IA2 decisions (already have from previous tests)
            flow_results['ia2'] = len(self.ia2_decisions) if self.ia2_decisions else 0
            logger.info(f"   📊 IA2 decisions: {flow_results['ia2']}")
            
            # 4. Test active positions (execution)
            try:
                response = requests.get(f"{self.api_url}/active-positions", timeout=30)
                if response.status_code == 200:
                    positions_data = response.json()
                    if isinstance(positions_data, dict):
                        active_positions = positions_data.get('active_positions', [])
                        flow_results['execution'] = len(active_positions)
                    else:
                        flow_results['execution'] = len(positions_data) if isinstance(positions_data, list) else 0
                    logger.info(f"   📊 Active positions: {flow_results['execution']}")
                else:
                    flow_results['execution'] = 0
            except:
                flow_results['execution'] = 0
            
            # 5. Test multiple crypto symbols consistency
            symbols_tested = set()
            
            # Extract symbols from IA1 analyses
            for analysis in self.ia1_analyses:
                if isinstance(analysis, dict) and 'symbol' in analysis:
                    symbols_tested.add(analysis['symbol'])
            
            # Extract symbols from IA2 decisions
            for decision in self.ia2_decisions:
                if isinstance(decision, dict) and 'symbol' in decision:
                    symbols_tested.add(decision['symbol'])
            
            symbols_count = len(symbols_tested)
            logger.info(f"   📊 Unique symbols tested: {symbols_count}")
            logger.info(f"   📊 Symbols: {list(symbols_tested)[:5]}{'...' if symbols_count > 5 else ''}")
            
            # Calculate flow efficiency
            total_flow_steps = sum(1 for step in flow_results.values() if step > 0)
            flow_continuity = flow_results['ia1'] > 0 and flow_results['ia2'] > 0
            symbol_diversity = symbols_count >= 3  # At least 3 different symbols
            
            logger.info(f"   📊 Flow steps active: {total_flow_steps}/4")
            logger.info(f"   📊 Flow continuity: {flow_continuity}")
            logger.info(f"   📊 Symbol diversity: {symbol_diversity}")
            
            # Success criteria: At least 3/4 flow steps active AND flow continuity AND symbol diversity
            success = total_flow_steps >= 3 and flow_continuity and symbol_diversity
            
            details = f"Flow steps: {total_flow_steps}/4, Continuity: {flow_continuity}, Symbols: {symbols_count}"
            
            self.log_test_result("End-to-End Enhanced Flow", success, details)
            
        except Exception as e:
            self.log_test_result("End-to-End Enhanced Flow", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_tests(self):
        """Run all Enhanced IA Prompts tests"""
        logger.info("🚀 Starting Enhanced IA1 and IA2 Prompts Test Suite")
        logger.info("=" * 80)
        logger.info("📋 REVIEW REQUEST: Test enhanced technical indicators integration")
        logger.info("🎯 OBJECTIVE: Verify RSI, MACD, Stochastic, Bollinger Bands integration")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_ia1_enhanced_technical_analysis()
        await self.test_ia1_confidence_rr_thresholds()
        await self.test_ia2_enhanced_decision_making()
        await self.test_ia2_execution_threshold()
        await self.test_enhanced_adaptive_context()
        await self.test_end_to_end_enhanced_flow()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 ENHANCED IA PROMPTS TEST SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # Review request analysis
        logger.info("\n" + "=" * 80)
        logger.info("📋 ENHANCED IA PROMPTS REVIEW ANALYSIS")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Enhanced IA1 and IA2 prompts working perfectly!")
            logger.info("✅ IA1 enhanced technical analysis with RSI, MACD, Stochastic, Bollinger Bands")
            logger.info("✅ IA1 confidence ≥70% and RR ≥2:1 thresholds for IA2 escalation")
            logger.info("✅ IA2 enhanced decision making with technical indicators analysis")
            logger.info("✅ IA2 execution threshold ≥80% confidence working")
            logger.info("✅ Enhanced Adaptive Context System with technical indicators fields")
            logger.info("✅ End-to-end enhanced flow working across multiple crypto symbols")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY WORKING - Most enhanced features are operational")
            logger.info("🔍 Some minor improvements needed for full compliance")
        else:
            logger.info("❌ CRITICAL ISSUES - Enhanced IA prompts need significant fixes")
            logger.info("🚨 Multiple enhanced features not meeting requirements")
        
        # Specific requirements check
        logger.info("\n📝 ENHANCED FEATURES VERIFICATION:")
        
        requirements_met = []
        requirements_failed = []
        
        # Check each test requirement
        test_mapping = {
            "IA1 Enhanced Technical Analysis": "✅ IA1 analyzes RSI, MACD, Stochastic, Bollinger Bands with confluence",
            "IA1 Confidence & RR Thresholds": "✅ IA1 filters with confidence ≥70% AND RR ≥2:1 for IA2 escalation", 
            "IA2 Enhanced Decision Making": "✅ IA2 incorporates technical indicators analysis in reasoning",
            "IA2 Execution Threshold": "✅ IA2 executes only when confidence ≥80%",
            "Enhanced Adaptive Context System": "✅ Enhanced Adaptive Context with technical indicators fields",
            "End-to-End Enhanced Flow": "✅ Complete enhanced flow working across multiple symbols"
        }
        
        for result in self.test_results:
            test_name = result['test']
            if test_name in test_mapping:
                if result['success']:
                    requirements_met.append(test_mapping[test_name])
                else:
                    requirements_failed.append(test_mapping[test_name].replace("✅", "❌"))
        
        for req in requirements_met:
            logger.info(f"   {req}")
        
        for req in requirements_failed:
            logger.info(f"   {req}")
        
        # Technical indicators verification
        logger.info("\n🎯 TECHNICAL INDICATORS INTEGRATION:")
        for indicator in self.expected_indicators:
            logger.info(f"   📊 {indicator}: Enhanced analysis and decision integration")
        
        logger.info(f"\n🏆 FINAL RESULT: {len(requirements_met)}/{len(requirements_met) + len(requirements_failed)} enhanced features working")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = EnhancedIAPromptsTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
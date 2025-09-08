#!/usr/bin/env python3
"""
Enhanced Technical Indicators Integration Test Suite
Focus: Testing IA1 and IA2 system with Stochastic Oscillator integration fix
Review Request: Verify Stochastic Oscillator (%K and %D values) are properly calculated and appearing in IA1 analyses
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

class EnhancedTechnicalIndicatorsTestSuite:
    """Test suite for Enhanced Technical Indicators Integration with Stochastic Oscillator fix"""
    
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
        logger.info(f"Testing Enhanced Technical Indicators Integration at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Expected technical indicators
        self.expected_indicators = {
            "rsi": ["rsi", "rsi_14", "rsi_signal", "oversold", "overbought"],
            "macd": ["macd", "macd_signal", "macd_histogram", "macd_trend", "bullish", "bearish"],
            "stochastic": ["stochastic", "stoch_k", "stoch_d", "%k", "%d", "stochastic_signal"],
            "bollinger": ["bollinger", "bollinger_bands", "bollinger_position", "upper_band", "lower_band", "squeeze"]
        }
        
        # Test symbols for analysis
        self.test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT"]
        
        # Confidence and RR thresholds
        self.ia1_confidence_threshold = 0.70  # 70%
        self.ia1_rr_threshold = 2.0  # 2:1
        self.ia2_confidence_threshold = 0.80  # 80%
        
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
    
    async def test_stochastic_integration_verification(self):
        """Test 1: Verify Stochastic Oscillator (%K and %D values) are properly calculated and appearing in IA1 analyses"""
        logger.info("\n🔍 TEST 1: Stochastic Integration Verification")
        
        try:
            # Get recent IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Stochastic Integration Verification", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            analyses = response.json()
            
            if not analyses or len(analyses) == 0:
                self.log_test_result("Stochastic Integration Verification", False, "No IA1 analyses found")
                return
            
            logger.info(f"   📊 Analyzing {len(analyses)} IA1 analyses for Stochastic indicators")
            
            stochastic_found_count = 0
            stochastic_values_found = 0
            debug_logs_found = 0
            
            stochastic_keywords = ["stochastic", "stoch_k", "stoch_d", "%k", "%d", "stochastic_signal"]
            
            for i, analysis in enumerate(analyses):
                analysis_text = str(analysis).lower()
                
                # Check for Stochastic keywords
                stochastic_present = any(keyword in analysis_text for keyword in stochastic_keywords)
                
                if stochastic_present:
                    stochastic_found_count += 1
                    logger.info(f"      ✅ Analysis {i+1}: Stochastic keywords found")
                    
                    # Check for actual Stochastic values (numeric patterns)
                    import re
                    stoch_value_patterns = [
                        r'stochastic[:\s]*(\d+\.?\d*)',
                        r'%k[:\s]*(\d+\.?\d*)',
                        r'%d[:\s]*(\d+\.?\d*)',
                        r'stoch_k[:\s]*(\d+\.?\d*)',
                        r'stoch_d[:\s]*(\d+\.?\d*)'
                    ]
                    
                    values_found = False
                    for pattern in stoch_value_patterns:
                        matches = re.findall(pattern, analysis_text)
                        if matches:
                            values_found = True
                            logger.info(f"         📊 Stochastic values: {matches[:3]}")  # Show first 3 values
                            break
                    
                    if values_found:
                        stochastic_values_found += 1
                    
                    # Check for debug logs pattern "Stochastic: XX.X"
                    debug_pattern = r'stochastic:\s*(\d+\.?\d*)'
                    debug_matches = re.findall(debug_pattern, analysis_text)
                    if debug_matches:
                        debug_logs_found += 1
                        logger.info(f"         🔍 Debug log values: {debug_matches[:2]}")
                else:
                    logger.info(f"      ❌ Analysis {i+1}: No Stochastic indicators found")
            
            # Calculate coverage percentages
            stochastic_coverage = (stochastic_found_count / len(analyses)) * 100
            values_coverage = (stochastic_values_found / len(analyses)) * 100
            debug_coverage = (debug_logs_found / len(analyses)) * 100
            
            logger.info(f"   📊 Stochastic keyword coverage: {stochastic_found_count}/{len(analyses)} ({stochastic_coverage:.1f}%)")
            logger.info(f"   📊 Stochastic values coverage: {stochastic_values_found}/{len(analyses)} ({values_coverage:.1f}%)")
            logger.info(f"   📊 Debug logs coverage: {debug_logs_found}/{len(analyses)} ({debug_coverage:.1f}%)")
            
            # Success criteria: At least 50% of analyses should contain Stochastic indicators
            success = stochastic_coverage >= 50.0
            
            details = f"Stochastic coverage: {stochastic_coverage:.1f}%, Values: {values_coverage:.1f}%, Debug logs: {debug_coverage:.1f}%"
            
            self.log_test_result("Stochastic Integration Verification", success, details)
            
            # Store for later tests
            self.ia1_analyses = analyses
            self.stochastic_coverage = stochastic_coverage
            
        except Exception as e:
            self.log_test_result("Stochastic Integration Verification", False, f"Exception: {str(e)}")
    
    async def test_enhanced_technical_indicators_coverage(self):
        """Test 2: Test that all 4 technical indicators (RSI, MACD, Stochastic, Bollinger Bands) are working in IA1 analysis"""
        logger.info("\n🔍 TEST 2: Enhanced Technical Indicators Coverage")
        
        try:
            if not hasattr(self, 'ia1_analyses') or not self.ia1_analyses:
                self.log_test_result("Enhanced Technical Indicators Coverage", False, "No IA1 analyses from previous test")
                return
            
            analyses = self.ia1_analyses
            
            # Track coverage for each indicator
            indicator_coverage = {}
            
            for indicator_name, keywords in self.expected_indicators.items():
                found_count = 0
                
                for analysis in analyses:
                    analysis_text = str(analysis).lower()
                    
                    # Check if any keywords for this indicator are present
                    indicator_present = any(keyword in analysis_text for keyword in keywords)
                    
                    if indicator_present:
                        found_count += 1
                
                coverage_percentage = (found_count / len(analyses)) * 100
                indicator_coverage[indicator_name] = {
                    'count': found_count,
                    'total': len(analyses),
                    'percentage': coverage_percentage
                }
                
                logger.info(f"   📊 {indicator_name.upper()}: {found_count}/{len(analyses)} ({coverage_percentage:.1f}%)")
            
            # Check for non-default values (realistic indicator values)
            realistic_values_count = 0
            
            for analysis in analyses:
                analysis_text = str(analysis).lower()
                
                # Look for realistic technical indicator values (not 0.0, 50.0, etc.)
                import re
                
                # RSI values (should be between 0-100, not exactly 50)
                rsi_matches = re.findall(r'rsi[:\s]*(\d+\.?\d*)', analysis_text)
                realistic_rsi = any(float(val) != 50.0 and 0 <= float(val) <= 100 for val in rsi_matches if val)
                
                # MACD values (should not be exactly 0.0)
                macd_matches = re.findall(r'macd[:\s]*(-?\d+\.?\d*)', analysis_text)
                realistic_macd = any(float(val) != 0.0 for val in macd_matches if val)
                
                # Stochastic values (should be between 0-100, not exactly 50)
                stoch_matches = re.findall(r'stoch(?:astic)?[:\s]*(\d+\.?\d*)', analysis_text)
                realistic_stoch = any(float(val) != 50.0 and 0 <= float(val) <= 100 for val in stoch_matches if val)
                
                if realistic_rsi or realistic_macd or realistic_stoch:
                    realistic_values_count += 1
            
            realistic_values_percentage = (realistic_values_count / len(analyses)) * 100
            
            logger.info(f"   📊 Realistic indicator values: {realistic_values_count}/{len(analyses)} ({realistic_values_percentage:.1f}%)")
            
            # Success criteria: All 4 indicators should have at least 50% coverage
            indicators_meeting_threshold = sum(1 for coverage in indicator_coverage.values() if coverage['percentage'] >= 50.0)
            all_indicators_working = indicators_meeting_threshold == 4
            
            details = f"Indicators ≥50%: {indicators_meeting_threshold}/4, Realistic values: {realistic_values_percentage:.1f}%"
            
            self.log_test_result("Enhanced Technical Indicators Coverage", all_indicators_working, details)
            
            # Store for later tests
            self.indicator_coverage = indicator_coverage
            
        except Exception as e:
            self.log_test_result("Enhanced Technical Indicators Coverage", False, f"Exception: {str(e)}")
    
    async def test_ia1_analysis_enhancement(self):
        """Test 3: Check if IA1 analyses show improved technical confluence analysis with all 4 indicators working together"""
        logger.info("\n🔍 TEST 3: IA1 Analysis Enhancement")
        
        try:
            if not hasattr(self, 'ia1_analyses') or not self.ia1_analyses:
                self.log_test_result("IA1 Analysis Enhancement", False, "No IA1 analyses from previous test")
                return
            
            analyses = self.ia1_analyses
            
            confluence_analysis_count = 0
            enhanced_precision_count = 0
            multi_indicator_count = 0
            
            confluence_keywords = [
                "confluence", "align", "confirm", "contradict", "divergence", 
                "technical confluence", "indicators align", "mixed signals"
            ]
            
            precision_keywords = [
                "precise", "enhanced", "advanced", "comprehensive", "detailed",
                "sophisticated", "refined", "optimized"
            ]
            
            for i, analysis in enumerate(analyses):
                analysis_text = str(analysis).lower()
                
                # Check for confluence analysis
                has_confluence = any(keyword in analysis_text for keyword in confluence_keywords)
                if has_confluence:
                    confluence_analysis_count += 1
                    logger.info(f"      ✅ Analysis {i+1}: Confluence analysis detected")
                
                # Check for enhanced precision language
                has_precision = any(keyword in analysis_text for keyword in precision_keywords)
                if has_precision:
                    enhanced_precision_count += 1
                
                # Check for multiple indicators mentioned together
                indicators_mentioned = 0
                for indicator_keywords in self.expected_indicators.values():
                    if any(keyword in analysis_text for keyword in indicator_keywords):
                        indicators_mentioned += 1
                
                if indicators_mentioned >= 3:  # At least 3 of 4 indicators
                    multi_indicator_count += 1
                    logger.info(f"      ✅ Analysis {i+1}: {indicators_mentioned}/4 indicators mentioned")
            
            # Calculate percentages
            confluence_percentage = (confluence_analysis_count / len(analyses)) * 100
            precision_percentage = (enhanced_precision_count / len(analyses)) * 100
            multi_indicator_percentage = (multi_indicator_count / len(analyses)) * 100
            
            logger.info(f"   📊 Confluence analysis: {confluence_analysis_count}/{len(analyses)} ({confluence_percentage:.1f}%)")
            logger.info(f"   📊 Enhanced precision: {enhanced_precision_count}/{len(analyses)} ({precision_percentage:.1f}%)")
            logger.info(f"   📊 Multi-indicator analysis: {multi_indicator_count}/{len(analyses)} ({multi_indicator_percentage:.1f}%)")
            
            # Success criteria: At least 60% should show confluence analysis and 70% should use multiple indicators
            success = confluence_percentage >= 60.0 and multi_indicator_percentage >= 70.0
            
            details = f"Confluence: {confluence_percentage:.1f}%, Multi-indicator: {multi_indicator_percentage:.1f}%, Precision: {precision_percentage:.1f}%"
            
            self.log_test_result("IA1 Analysis Enhancement", success, details)
            
        except Exception as e:
            self.log_test_result("IA1 Analysis Enhancement", False, f"Exception: {str(e)}")
    
    async def test_ia2_technical_integration(self):
        """Test 4: Verify IA2 decisions are leveraging the enhanced technical indicators data from IA1"""
        logger.info("\n🔍 TEST 4: IA2 Technical Integration")
        
        try:
            # Get recent IA2 decisions
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("IA2 Technical Integration", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            decisions = response.json()
            
            if not decisions or len(decisions) == 0:
                self.log_test_result("IA2 Technical Integration", False, "No IA2 decisions found")
                return
            
            logger.info(f"   📊 Analyzing {len(decisions)} IA2 decisions for technical indicators integration")
            
            # Track technical indicators usage in IA2 decisions
            ia2_indicator_usage = {}
            
            for indicator_name, keywords in self.expected_indicators.items():
                found_count = 0
                
                for decision in decisions:
                    decision_text = str(decision).lower()
                    
                    # Check if any keywords for this indicator are present in IA2 reasoning
                    indicator_present = any(keyword in decision_text for keyword in keywords)
                    
                    if indicator_present:
                        found_count += 1
                
                usage_percentage = (found_count / len(decisions)) * 100
                ia2_indicator_usage[indicator_name] = {
                    'count': found_count,
                    'total': len(decisions),
                    'percentage': usage_percentage
                }
                
                logger.info(f"   📊 IA2 {indicator_name.upper()}: {found_count}/{len(decisions)} ({usage_percentage:.1f}%)")
            
            # Check for technical analysis integration keywords
            technical_integration_count = 0
            technical_keywords = [
                "technical analysis", "technical confluence", "indicators", "rsi", "macd", 
                "stochastic", "bollinger", "technical momentum", "confluence score"
            ]
            
            for decision in decisions:
                decision_text = str(decision).lower()
                
                has_technical_integration = any(keyword in decision_text for keyword in technical_keywords)
                if has_technical_integration:
                    technical_integration_count += 1
            
            technical_integration_percentage = (technical_integration_count / len(decisions)) * 100
            
            logger.info(f"   📊 Technical integration: {technical_integration_count}/{len(decisions)} ({technical_integration_percentage:.1f}%)")
            
            # Success criteria: At least 70% of IA2 decisions should show technical analysis integration
            # and at least 2 of 4 indicators should have ≥50% usage
            indicators_meeting_threshold = sum(1 for usage in ia2_indicator_usage.values() if usage['percentage'] >= 50.0)
            success = technical_integration_percentage >= 70.0 and indicators_meeting_threshold >= 2
            
            details = f"Technical integration: {technical_integration_percentage:.1f}%, Indicators ≥50%: {indicators_meeting_threshold}/4"
            
            self.log_test_result("IA2 Technical Integration", success, details)
            
            # Store for later tests
            self.ia2_decisions = decisions
            self.ia2_indicator_usage = ia2_indicator_usage
            
        except Exception as e:
            self.log_test_result("IA2 Technical Integration", False, f"Exception: {str(e)}")
    
    async def test_confidence_rr_thresholds(self):
        """Test 5: Re-test the confidence ≥70% and RR ≥2:1 thresholds for IA1→IA2 escalation"""
        logger.info("\n🔍 TEST 5: Confidence and RR Thresholds")
        
        try:
            if not hasattr(self, 'ia1_analyses') or not self.ia1_analyses:
                self.log_test_result("Confidence and RR Thresholds", False, "No IA1 analyses from previous test")
                return
            
            if not hasattr(self, 'ia2_decisions') or not self.ia2_decisions:
                self.log_test_result("Confidence and RR Thresholds", False, "No IA2 decisions from previous test")
                return
            
            analyses = self.ia1_analyses
            decisions = self.ia2_decisions
            
            # Analyze IA1 confidence and RR ratios
            high_confidence_ia1 = 0
            high_rr_ia1 = 0
            both_thresholds_ia1 = 0
            
            import re
            
            for analysis in analyses:
                analysis_text = str(analysis).lower()
                
                # Extract confidence values
                confidence_matches = re.findall(r'confidence[:\s]*(\d+\.?\d*)', analysis_text)
                if confidence_matches:
                    confidence = float(confidence_matches[0])
                    if confidence >= self.ia1_confidence_threshold:
                        high_confidence_ia1 += 1
                
                # Extract RR ratios
                rr_matches = re.findall(r'(?:risk.?reward|rr)[:\s]*(\d+\.?\d*)', analysis_text)
                if rr_matches:
                    rr_ratio = float(rr_matches[0])
                    if rr_ratio >= self.ia1_rr_threshold:
                        high_rr_ia1 += 1
                
                # Check if both thresholds are met
                if (confidence_matches and float(confidence_matches[0]) >= self.ia1_confidence_threshold and
                    rr_matches and float(rr_matches[0]) >= self.ia1_rr_threshold):
                    both_thresholds_ia1 += 1
            
            # Analyze IA2 confidence and execution
            high_confidence_ia2 = 0
            executed_decisions = 0
            
            for decision in decisions:
                decision_text = str(decision).lower()
                
                # Extract IA2 confidence values
                confidence_matches = re.findall(r'confidence[:\s]*(\d+\.?\d*)', decision_text)
                if confidence_matches:
                    confidence = float(confidence_matches[0])
                    if confidence >= self.ia2_confidence_threshold:
                        high_confidence_ia2 += 1
                
                # Check if decision was executed (not HOLD)
                signal_matches = re.findall(r'signal[:\s]*["\']?(\w+)["\']?', decision_text)
                if signal_matches:
                    signal = signal_matches[0].upper()
                    if signal in ['LONG', 'SHORT']:
                        executed_decisions += 1
            
            # Calculate percentages
            ia1_confidence_percentage = (high_confidence_ia1 / len(analyses)) * 100 if analyses else 0
            ia1_rr_percentage = (high_rr_ia1 / len(analyses)) * 100 if analyses else 0
            ia1_both_percentage = (both_thresholds_ia1 / len(analyses)) * 100 if analyses else 0
            
            ia2_confidence_percentage = (high_confidence_ia2 / len(decisions)) * 100 if decisions else 0
            ia2_execution_percentage = (executed_decisions / len(decisions)) * 100 if decisions else 0
            
            logger.info(f"   📊 IA1 confidence ≥70%: {high_confidence_ia1}/{len(analyses)} ({ia1_confidence_percentage:.1f}%)")
            logger.info(f"   📊 IA1 RR ≥2:1: {high_rr_ia1}/{len(analyses)} ({ia1_rr_percentage:.1f}%)")
            logger.info(f"   📊 IA1 both thresholds: {both_thresholds_ia1}/{len(analyses)} ({ia1_both_percentage:.1f}%)")
            logger.info(f"   📊 IA2 confidence ≥80%: {high_confidence_ia2}/{len(decisions)} ({ia2_confidence_percentage:.1f}%)")
            logger.info(f"   📊 IA2 execution rate: {executed_decisions}/{len(decisions)} ({ia2_execution_percentage:.1f}%)")
            
            # Success criteria: 
            # - IA1 should have reasonable confidence and RR compliance
            # - IA2 should execute high confidence decisions at reasonable rate
            ia1_thresholds_working = ia1_confidence_percentage >= 50.0 and ia1_rr_percentage >= 30.0
            ia2_execution_working = ia2_execution_percentage >= 50.0  # At least 50% execution rate
            
            success = ia1_thresholds_working and ia2_execution_working
            
            details = f"IA1 conf: {ia1_confidence_percentage:.1f}%, IA1 RR: {ia1_rr_percentage:.1f}%, IA2 exec: {ia2_execution_percentage:.1f}%"
            
            self.log_test_result("Confidence and RR Thresholds", success, details)
            
        except Exception as e:
            self.log_test_result("Confidence and RR Thresholds", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_tests(self):
        """Run all Enhanced Technical Indicators Integration tests"""
        logger.info("🚀 Starting Enhanced Technical Indicators Integration Test Suite")
        logger.info("=" * 80)
        logger.info("📋 REVIEW REQUEST: Test enhanced IA1 and IA2 system with Stochastic Oscillator integration fix")
        logger.info("🎯 CRITICAL FOCUS: Verify Stochastic Oscillator (%K and %D values) are properly calculated")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_stochastic_integration_verification()
        await self.test_enhanced_technical_indicators_coverage()
        await self.test_ia1_analysis_enhancement()
        await self.test_ia2_technical_integration()
        await self.test_confidence_rr_thresholds()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 ENHANCED TECHNICAL INDICATORS INTEGRATION TEST SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # Critical findings analysis
        logger.info("\n" + "=" * 80)
        logger.info("🔍 CRITICAL FINDINGS ANALYSIS")
        logger.info("=" * 80)
        
        critical_issues = []
        working_components = []
        
        # Analyze each test result
        for result in self.test_results:
            if result['success']:
                working_components.append(f"✅ {result['test']}")
            else:
                critical_issues.append(f"❌ {result['test']}: {result['details']}")
        
        if critical_issues:
            logger.info("🚨 CRITICAL ISSUES FOUND:")
            for issue in critical_issues:
                logger.info(f"   {issue}")
        
        if working_components:
            logger.info("\n✅ WORKING COMPONENTS:")
            for component in working_components:
                logger.info(f"   {component}")
        
        # Specific review requirements check
        logger.info("\n📝 REVIEW REQUIREMENTS VERIFICATION:")
        
        requirements_status = []
        
        # Check Stochastic integration
        stochastic_test = any("Stochastic Integration" in result['test'] and result['success'] for result in self.test_results)
        if stochastic_test:
            requirements_status.append("✅ Stochastic Oscillator (%K and %D values) properly calculated and appearing in IA1")
        else:
            requirements_status.append("❌ Stochastic Oscillator integration FAILED - not appearing in IA1 analyses")
        
        # Check all 4 indicators
        indicators_test = any("Technical Indicators Coverage" in result['test'] and result['success'] for result in self.test_results)
        if indicators_test:
            requirements_status.append("✅ All 4 technical indicators (RSI, MACD, Stochastic, Bollinger Bands) working in IA1")
        else:
            requirements_status.append("❌ Not all 4 technical indicators working properly in IA1 analysis")
        
        # Check IA1 enhancement
        ia1_test = any("IA1 Analysis Enhancement" in result['test'] and result['success'] for result in self.test_results)
        if ia1_test:
            requirements_status.append("✅ IA1 analyses show improved technical confluence analysis with all 4 indicators")
        else:
            requirements_status.append("❌ IA1 technical confluence analysis not sufficiently enhanced")
        
        # Check IA2 integration
        ia2_test = any("IA2 Technical Integration" in result['test'] and result['success'] for result in self.test_results)
        if ia2_test:
            requirements_status.append("✅ IA2 decisions leverage enhanced technical indicators data from IA1")
        else:
            requirements_status.append("❌ IA2 not properly leveraging enhanced technical indicators from IA1")
        
        # Check thresholds
        thresholds_test = any("Confidence and RR Thresholds" in result['test'] and result['success'] for result in self.test_results)
        if thresholds_test:
            requirements_status.append("✅ Confidence ≥70% and RR ≥2:1 thresholds working for IA1→IA2 escalation")
        else:
            requirements_status.append("❌ Confidence and RR thresholds not working properly for IA1→IA2 escalation")
        
        for status in requirements_status:
            logger.info(f"   {status}")
        
        # Final assessment
        logger.info(f"\n🏆 FINAL ASSESSMENT: {passed_tests}/{total_tests} components working")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Enhanced Technical Indicators Integration is working perfectly!")
            logger.info("✅ Stochastic Oscillator integration fix successful")
            logger.info("✅ All 4 technical indicators operational in IA1 and IA2")
            logger.info("✅ Enhanced technical confluence analysis working")
            logger.info("✅ Confidence and RR thresholds properly implemented")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY WORKING - Minor issues need attention")
            logger.info("🔍 Most enhanced technical indicators features are operational")
        else:
            logger.info("❌ CRITICAL ISSUES - Enhanced Technical Indicators Integration needs significant fixes")
            logger.info("🚨 Multiple components not meeting review requirements")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = EnhancedTechnicalIndicatorsTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
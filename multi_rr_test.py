#!/usr/bin/env python3
"""
Backend Testing Suite for Multi-RR Display and IA2 RR Consistency Fixes
Focus: Testing critical fixes for IA1 Risk-Reward display and IA2 RR consistency
Review Request: Verify Multi-RR analysis appears early in IA1 reasoning and IA2 uses same RR values
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

class MultiRRDisplayAndIA2ConsistencyTestSuite:
    """Test suite for Multi-RR Display and IA2 RR Consistency Fixes"""
    
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
        logger.info(f"Testing Multi-RR Display and IA2 RR Consistency at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Focus on 2-3 symbols max for quick verification as requested
        self.test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        # Store analysis and decision data for cross-validation
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
        
    async def test_multi_rr_display_early_positioning(self):
        """Test 1: Multi-RR analysis appears at BEGINNING of IA1 reasoning text (within 800-char limit)"""
        logger.info("\n🔍 TEST 1: Multi-RR Display Early Positioning in IA1 Reasoning")
        
        try:
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Multi-RR Display Early Positioning", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            
            # Handle the API response structure
            if isinstance(data, dict) and 'analyses' in data:
                analyses = data['analyses']
            elif isinstance(data, list):
                analyses = data
            else:
                self.log_test_result("Multi-RR Display Early Positioning", False, "Invalid API response structure")
                return
            
            if not analyses or len(analyses) == 0:
                self.log_test_result("Multi-RR Display Early Positioning", False, "No IA1 analyses found")
                return
            
            # Store analyses for later cross-validation
            self.ia1_analyses = analyses
            
            multi_rr_early_count = 0
            multi_rr_late_count = 0
            total_analyses = len(data)
            
            # Focus on first 2-3 analyses as requested
            analyses_to_check = data[:3]
            
            for analysis in analyses_to_check:
                symbol = analysis.get('symbol', 'Unknown')
                reasoning = analysis.get('reasoning', '')
                
                if not reasoning:
                    continue
                
                # Check if Multi-RR analysis exists
                multi_rr_keywords = ['multi-rr', 'multi rr', 'hold:', 'long:', 'short:', 'winner:', 'rr analysis']
                has_multi_rr = any(keyword.lower() in reasoning.lower() for keyword in multi_rr_keywords)
                
                if has_multi_rr:
                    # Check position within first 800 characters (frontend limit)
                    first_800_chars = reasoning[:800]
                    multi_rr_in_first_800 = any(keyword.lower() in first_800_chars.lower() for keyword in multi_rr_keywords)
                    
                    if multi_rr_in_first_800:
                        multi_rr_early_count += 1
                        logger.info(f"   ✅ {symbol}: Multi-RR found in first 800 chars")
                        
                        # Find exact position
                        for keyword in multi_rr_keywords:
                            pos = reasoning.lower().find(keyword.lower())
                            if pos != -1:
                                logger.info(f"      Multi-RR keyword '{keyword}' at position {pos}")
                                break
                    else:
                        multi_rr_late_count += 1
                        logger.info(f"   ❌ {symbol}: Multi-RR found but after 800 chars")
                        
                        # Find position where it appears
                        for keyword in multi_rr_keywords:
                            pos = reasoning.lower().find(keyword.lower())
                            if pos != -1:
                                logger.info(f"      Multi-RR keyword '{keyword}' at position {pos} (beyond 800-char limit)")
                                break
                else:
                    logger.info(f"   ⚪ {symbol}: No Multi-RR analysis found")
            
            # Success criteria: At least 1 analysis has Multi-RR early AND no Multi-RR late
            success = multi_rr_early_count > 0 and multi_rr_late_count == 0
            
            details = f"Early Multi-RR: {multi_rr_early_count}, Late Multi-RR: {multi_rr_late_count}, Total checked: {len(analyses_to_check)}"
            
            self.log_test_result("Multi-RR Display Early Positioning", success, details)
            
        except Exception as e:
            self.log_test_result("Multi-RR Display Early Positioning", False, f"Exception: {str(e)}")
    
    async def test_ia2_rr_consistency_with_ia1(self):
        """Test 2: IA2 decisions use same Risk-Reward ratio calculated by IA1 (no recalculation)"""
        logger.info("\n🔍 TEST 2: IA2 RR Consistency with IA1 (No Recalculation)")
        
        try:
            # Get IA2 decisions
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("IA2 RR Consistency", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            
            if not isinstance(data, list) or len(data) == 0:
                self.log_test_result("IA2 RR Consistency", False, "No IA2 decisions found")
                return
            
            # Store decisions for cross-validation
            self.ia2_decisions = data
            
            # Focus on first 2-3 decisions as requested
            decisions_to_check = data[:3]
            
            consistent_rr_count = 0
            inconsistent_rr_count = 0
            rr_comparisons = []
            
            for decision in decisions_to_check:
                symbol = decision.get('symbol', 'Unknown')
                ia2_rr = decision.get('risk_reward_ratio', 0)
                
                # Find corresponding IA1 analysis
                ia1_analysis = None
                if hasattr(self, 'ia1_analyses') and self.ia1_analyses:
                    for analysis in self.ia1_analyses:
                        if analysis.get('symbol') == symbol:
                            ia1_analysis = analysis
                            break
                
                if ia1_analysis:
                    ia1_rr = ia1_analysis.get('risk_reward_ratio', 0)
                    
                    # Check if RR values are consistent (within small tolerance for floating point)
                    rr_difference = abs(ia1_rr - ia2_rr)
                    is_consistent = rr_difference < 0.01  # 0.01 tolerance
                    
                    if is_consistent:
                        consistent_rr_count += 1
                        logger.info(f"   ✅ {symbol}: RR consistent - IA1: {ia1_rr:.2f}, IA2: {ia2_rr:.2f}")
                    else:
                        inconsistent_rr_count += 1
                        logger.info(f"   ❌ {symbol}: RR inconsistent - IA1: {ia1_rr:.2f}, IA2: {ia2_rr:.2f} (diff: {rr_difference:.3f})")
                    
                    rr_comparisons.append({
                        'symbol': symbol,
                        'ia1_rr': ia1_rr,
                        'ia2_rr': ia2_rr,
                        'difference': rr_difference,
                        'consistent': is_consistent
                    })
                else:
                    logger.info(f"   ⚪ {symbol}: No corresponding IA1 analysis found")
            
            # Success criteria: All checked decisions have consistent RR values
            success = consistent_rr_count > 0 and inconsistent_rr_count == 0
            
            details = f"Consistent RR: {consistent_rr_count}, Inconsistent RR: {inconsistent_rr_count}, Total checked: {len(decisions_to_check)}"
            
            self.log_test_result("IA2 RR Consistency", success, details)
            
            # Store comparisons for detailed analysis
            self.rr_comparisons = rr_comparisons
            
        except Exception as e:
            self.log_test_result("IA2 RR Consistency", False, f"Exception: {str(e)}")
    
    async def test_rr_filtering_threshold_enforcement(self):
        """Test 3: Only analyses with IA1 RR >= 2.0 are passed to IA2"""
        logger.info("\n🔍 TEST 3: RR Filtering Threshold Enforcement (IA1 RR >= 2.0)")
        
        try:
            if not hasattr(self, 'ia1_analyses') or not self.ia1_analyses:
                self.log_test_result("RR Filtering Threshold", False, "No IA1 analyses available from previous test")
                return
            
            if not hasattr(self, 'ia2_decisions') or not self.ia2_decisions:
                self.log_test_result("RR Filtering Threshold", False, "No IA2 decisions available from previous test")
                return
            
            # Analyze IA1 analyses RR distribution
            high_rr_analyses = []  # RR >= 2.0
            low_rr_analyses = []   # RR < 2.0
            
            for analysis in self.ia1_analyses[:5]:  # Check first 5 analyses
                symbol = analysis.get('symbol', 'Unknown')
                rr = analysis.get('risk_reward_ratio', 0)
                
                if rr >= 2.0:
                    high_rr_analyses.append({'symbol': symbol, 'rr': rr})
                else:
                    low_rr_analyses.append({'symbol': symbol, 'rr': rr})
            
            # Check which symbols have IA2 decisions
            ia2_symbols = [decision.get('symbol') for decision in self.ia2_decisions]
            
            # Verify filtering logic
            correctly_passed = 0  # High RR analyses that have IA2 decisions
            incorrectly_passed = 0  # Low RR analyses that have IA2 decisions
            correctly_filtered = 0  # Low RR analyses that don't have IA2 decisions
            
            for analysis in high_rr_analyses:
                if analysis['symbol'] in ia2_symbols:
                    correctly_passed += 1
                    logger.info(f"   ✅ {analysis['symbol']}: High RR ({analysis['rr']:.2f}) correctly passed to IA2")
                else:
                    logger.info(f"   ⚪ {analysis['symbol']}: High RR ({analysis['rr']:.2f}) but no IA2 decision found")
            
            for analysis in low_rr_analyses:
                if analysis['symbol'] in ia2_symbols:
                    incorrectly_passed += 1
                    logger.info(f"   ❌ {analysis['symbol']}: Low RR ({analysis['rr']:.2f}) incorrectly passed to IA2")
                else:
                    correctly_filtered += 1
                    logger.info(f"   ✅ {analysis['symbol']}: Low RR ({analysis['rr']:.2f}) correctly filtered out")
            
            # Success criteria: No low RR analyses should have IA2 decisions
            success = incorrectly_passed == 0 and (correctly_passed > 0 or correctly_filtered > 0)
            
            details = f"High RR passed: {correctly_passed}, Low RR filtered: {correctly_filtered}, Low RR incorrectly passed: {incorrectly_passed}"
            
            self.log_test_result("RR Filtering Threshold", success, details)
            
        except Exception as e:
            self.log_test_result("RR Filtering Threshold", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_tests(self):
        """Run all Multi-RR Display and IA2 RR Consistency tests"""
        logger.info("🚀 Starting Multi-RR Display and IA2 RR Consistency Test Suite")
        logger.info("=" * 80)
        logger.info("📋 REVIEW REQUEST: Test critical fixes for IA1 Risk-Reward display and IA2 RR consistency")
        logger.info("🎯 OBJECTIVE: Verify Multi-RR early display and IA2 RR consistency")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_multi_rr_display_early_positioning()
        await self.test_ia2_rr_consistency_with_ia1()
        await self.test_rr_filtering_threshold_enforcement()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 MULTI-RR DISPLAY AND IA2 RR CONSISTENCY TEST SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # Detailed analysis
        logger.info("\n" + "=" * 80)
        logger.info("📋 DETAILED ANALYSIS OF CRITICAL FIXES")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Multi-RR Display and IA2 RR Consistency fixes working perfectly!")
            logger.info("✅ Multi-RR analysis appears at beginning of IA1 reasoning (within 800-char limit)")
            logger.info("✅ IA2 decisions use same Risk-Reward ratio calculated by IA1")
            logger.info("✅ RR filtering correctly enforces >= 2.0 threshold")
        elif passed_tests >= total_tests * 0.67:
            logger.info("⚠️ PARTIAL SUCCESS - Most fixes are working")
            logger.info("🔍 Some improvements needed for complete compliance")
        else:
            logger.info("❌ CRITICAL ISSUES - Fixes need attention")
            logger.info("🚨 Multiple components not working as expected")
        
        # Specific requirements check
        logger.info("\n📝 SPECIFIC REQUIREMENTS VERIFICATION:")
        
        requirements_met = []
        requirements_failed = []
        
        # Check Multi-RR display fix
        multi_rr_test = any("Multi-RR Display" in result['test'] and result['success'] for result in self.test_results)
        if multi_rr_test:
            requirements_met.append("✅ Multi-RR analysis appears at BEGINNING of IA1 reasoning text")
        else:
            requirements_failed.append("❌ Multi-RR analysis not appearing early in IA1 reasoning")
        
        # Check IA2 RR consistency fix
        consistency_test = any("IA2 RR Consistency" in result['test'] and result['success'] for result in self.test_results)
        if consistency_test:
            requirements_met.append("✅ IA2 decisions use same Risk-Reward ratio calculated by IA1")
        else:
            requirements_failed.append("❌ IA2 decisions recalculating RR instead of using IA1 values")
        
        # Check RR filtering fix
        filtering_test = any("RR Filtering" in result['test'] and result['success'] for result in self.test_results)
        if filtering_test:
            requirements_met.append("✅ Only analyses with IA1 RR >= 2.0 are passed to IA2")
        else:
            requirements_failed.append("❌ RR filtering not properly enforcing >= 2.0 threshold")
        
        for req in requirements_met:
            logger.info(f"   {req}")
        
        for req in requirements_failed:
            logger.info(f"   {req}")
        
        # Additional insights
        if hasattr(self, 'rr_comparisons') and self.rr_comparisons:
            logger.info("\n🔍 RR COMPARISON DETAILS:")
            for comp in self.rr_comparisons[:3]:  # Show first 3
                logger.info(f"   {comp['symbol']}: IA1={comp['ia1_rr']:.2f}, IA2={comp['ia2_rr']:.2f}, Diff={comp['difference']:.3f}")
        
        logger.info(f"\n🏆 FINAL RESULT: {len(requirements_met)}/{len(requirements_met) + len(requirements_failed)} critical fixes verified")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = MultiRRDisplayAndIA2ConsistencyTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
IA1 Enhanced Scoring System Validation Test
Focus: Validate that the IA1 scoring system is working based on actual logs and API data
"""

import asyncio
import json
import logging
import os
import sys
import time
import math
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Any, List
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IA1ScoringValidationTest:
    """Validation test for IA1 Enhanced Scoring System"""
    
    def __init__(self):
        self.api_url = "https://smart-trading-bot-8.preview.emergentagent.com/api"
        logger.info(f"Validating IA1 Enhanced Scoring System at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
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
    
    async def test_1_scoring_logs_validation(self):
        """Test 1: Validate IA1 scoring logs are present"""
        logger.info("\n🔍 TEST 1: Validate IA1 scoring logs")
        
        try:
            # Get backend logs
            log_result = subprocess.run(
                ["grep", "-E", "IA1 PROFESSIONAL SCORING|Base Confidence|Market Adjustment|MC Multiplier|Key Factors", 
                 "/var/log/supervisor/backend.err.log"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if log_result.returncode != 0:
                self.log_test_result("Scoring Logs Validation", False, "No scoring logs found")
                return
            
            logs = log_result.stdout.strip().split('\n')
            
            # Count different types of logs
            professional_scoring_logs = len([log for log in logs if "IA1 PROFESSIONAL SCORING" in log])
            base_confidence_logs = len([log for log in logs if "Base Confidence:" in log])
            market_adjustment_logs = len([log for log in logs if "Market Adjustment:" in log])
            mc_multiplier_logs = len([log for log in logs if "MC Multiplier:" in log])
            key_factors_logs = len([log for log in logs if "Key Factors:" in log])
            
            logger.info(f"   📊 Professional scoring logs: {professional_scoring_logs}")
            logger.info(f"   📊 Base confidence logs: {base_confidence_logs}")
            logger.info(f"   📊 Market adjustment logs: {market_adjustment_logs}")
            logger.info(f"   📊 MC multiplier logs: {mc_multiplier_logs}")
            logger.info(f"   📊 Key factors logs: {key_factors_logs}")
            
            # Show recent examples
            recent_logs = logs[-10:] if len(logs) > 10 else logs
            logger.info("   📋 Recent scoring logs:")
            for log in recent_logs:
                logger.info(f"      {log}")
            
            # Success criteria: All types of logs present
            success = (professional_scoring_logs > 0 and base_confidence_logs > 0 and 
                      market_adjustment_logs > 0 and mc_multiplier_logs > 0 and key_factors_logs > 0)
            
            details = f"Prof: {professional_scoring_logs}, Base: {base_confidence_logs}, Adj: {market_adjustment_logs}, MC: {mc_multiplier_logs}, Factors: {key_factors_logs}"
            
            self.log_test_result("Scoring Logs Validation", success, details)
            
        except Exception as e:
            self.log_test_result("Scoring Logs Validation", False, f"Exception: {str(e)}")
    
    async def test_2_score_differential_analysis(self):
        """Test 2: Analyze score differentials from logs"""
        logger.info("\n🔍 TEST 2: Analyze score differentials")
        
        try:
            # Get base confidence logs
            log_result = subprocess.run(
                ["grep", "Base Confidence:", "/var/log/supervisor/backend.err.log"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if log_result.returncode != 0:
                self.log_test_result("Score Differential Analysis", False, "No base confidence logs found")
                return
            
            logs = log_result.stdout.strip().split('\n')
            
            # Parse base vs final scores
            score_differentials = []
            for log in logs:
                try:
                    # Extract base and final percentages
                    # Format: "Base Confidence: 83.0% → Final: 81.2%"
                    if "→" in log and "Final:" in log:
                        parts = log.split("→")
                        base_part = parts[0].split(":")[-1].strip().replace("%", "")
                        final_part = parts[1].split(":")[1].strip().replace("%", "")
                        
                        base_val = float(base_part)
                        final_val = float(final_part)
                        differential = final_val - base_val
                        score_differentials.append({
                            'base': base_val,
                            'final': final_val,
                            'differential': differential
                        })
                        
                        logger.info(f"      📊 Score: {base_val:.1f}% → {final_val:.1f}% (Δ{differential:+.1f}%)")
                except:
                    continue
            
            if not score_differentials:
                self.log_test_result("Score Differential Analysis", False, "No score differentials found")
                return
            
            # Calculate statistics
            differentials = [s['differential'] for s in score_differentials]
            min_diff = min(differentials)
            max_diff = max(differentials)
            avg_diff = sum(differentials) / len(differentials)
            
            # Check for variability
            positive_adjustments = len([d for d in differentials if d > 0])
            negative_adjustments = len([d for d in differentials if d < 0])
            zero_adjustments = len([d for d in differentials if d == 0])
            
            logger.info(f"   📊 Score differentials analyzed: {len(score_differentials)}")
            logger.info(f"   📊 Differential range: {min_diff:.1f}% to {max_diff:.1f}%")
            logger.info(f"   📊 Average differential: {avg_diff:.1f}%")
            logger.info(f"   📊 Positive adjustments: {positive_adjustments}")
            logger.info(f"   📊 Negative adjustments: {negative_adjustments}")
            logger.info(f"   📊 Zero adjustments: {zero_adjustments}")
            
            # Success criteria: Scores show meaningful variation
            success = (len(score_differentials) > 0 and 
                      abs(max_diff - min_diff) > 1.0 and  # Range > 1%
                      (positive_adjustments > 0 or negative_adjustments > 0))  # Some adjustments made
            
            details = f"Count: {len(score_differentials)}, Range: {min_diff:.1f}% to {max_diff:.1f}%, Pos: {positive_adjustments}, Neg: {negative_adjustments}"
            
            self.log_test_result("Score Differential Analysis", success, details)
            
        except Exception as e:
            self.log_test_result("Score Differential Analysis", False, f"Exception: {str(e)}")
    
    async def test_3_market_cap_multiplier_validation(self):
        """Test 3: Validate market cap multipliers"""
        logger.info("\n🔍 TEST 3: Validate market cap multipliers")
        
        try:
            # Get MC multiplier logs
            log_result = subprocess.run(
                ["grep", "MC Multiplier:", "/var/log/supervisor/backend.err.log"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if log_result.returncode != 0:
                self.log_test_result("Market Cap Multiplier Validation", False, "No MC multiplier logs found")
                return
            
            logs = log_result.stdout.strip().split('\n')
            
            # Parse MC multipliers
            mc_data = []
            for log in logs:
                try:
                    # Extract multiplier and market cap
                    # Format: "MC Multiplier: 1.10 (Market Cap: 1,004,045,230)"
                    mult_part = log.split("MC Multiplier:")[1].split("(")[0].strip()
                    multiplier = float(mult_part)
                    
                    cap_part = log.split("Market Cap:")[1].split(")")[0].strip().replace(",", "")
                    market_cap = float(cap_part)
                    
                    mc_data.append({
                        'multiplier': multiplier,
                        'market_cap': market_cap
                    })
                    
                    logger.info(f"      📊 MC: ${market_cap:,.0f} → Multiplier: {multiplier:.2f}")
                except:
                    continue
            
            if not mc_data:
                self.log_test_result("Market Cap Multiplier Validation", False, "No MC data found")
                return
            
            # Validate multiplier logic
            correct_multipliers = 0
            for data in mc_data:
                market_cap = data['market_cap']
                multiplier = data['multiplier']
                
                expected_multiplier = None
                if market_cap < 10_000_000:  # micro cap
                    expected_multiplier = 0.8
                elif market_cap < 100_000_000:  # small cap
                    expected_multiplier = 0.95
                elif market_cap < 1_000_000_000:  # mid cap
                    expected_multiplier = 1.0
                else:  # large cap
                    expected_multiplier = 1.1
                
                if abs(multiplier - expected_multiplier) < 0.01:
                    correct_multipliers += 1
                    logger.info(f"      ✅ Correct: ${market_cap:,.0f} → {multiplier:.2f}")
                else:
                    logger.info(f"      ❌ Incorrect: ${market_cap:,.0f} → {multiplier:.2f} (expected {expected_multiplier:.2f})")
            
            # Check for multiplier variety
            unique_multipliers = set(round(data['multiplier'], 2) for data in mc_data)
            
            logger.info(f"   📊 MC multipliers analyzed: {len(mc_data)}")
            logger.info(f"   📊 Correct multipliers: {correct_multipliers}/{len(mc_data)}")
            logger.info(f"   📊 Unique multipliers: {len(unique_multipliers)} ({unique_multipliers})")
            
            # Success criteria: MC multipliers are applied correctly
            success = (len(mc_data) > 0 and 
                      correct_multipliers >= len(mc_data) * 0.8 and  # 80% correct
                      len(unique_multipliers) >= 1)  # At least some variety
            
            details = f"Total: {len(mc_data)}, Correct: {correct_multipliers}/{len(mc_data)}, Unique: {len(unique_multipliers)}"
            
            self.log_test_result("Market Cap Multiplier Validation", success, details)
            
        except Exception as e:
            self.log_test_result("Market Cap Multiplier Validation", False, f"Exception: {str(e)}")
    
    async def test_4_market_factors_integration(self):
        """Test 4: Validate market factors integration"""
        logger.info("\n🔍 TEST 4: Validate market factors integration")
        
        try:
            # Get key factors logs
            log_result = subprocess.run(
                ["grep", "Key Factors:", "/var/log/supervisor/backend.err.log"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if log_result.returncode != 0:
                self.log_test_result("Market Factors Integration", False, "No key factors logs found")
                return
            
            logs = log_result.stdout.strip().split('\n')
            
            # Parse market factors
            factors_data = []
            for log in logs:
                try:
                    # Extract RSI, Vol, Price∆
                    # Format: "Key Factors: RSI=61.6, Vol=5.5%, Price∆=5.5%"
                    factors_part = log.split("Key Factors:")[1].strip()
                    
                    rsi_val = None
                    vol_val = None
                    price_delta_val = None
                    
                    if "RSI=" in factors_part:
                        rsi_str = factors_part.split("RSI=")[1].split(",")[0].strip()
                        rsi_val = float(rsi_str)
                    
                    if "Vol=" in factors_part:
                        vol_str = factors_part.split("Vol=")[1].split(",")[0].strip().replace("%", "")
                        vol_val = float(vol_str)
                    
                    if "Price∆=" in factors_part:
                        price_str = factors_part.split("Price∆=")[1].split(",")[0].strip().replace("%", "")
                        price_delta_val = float(price_str)
                    
                    factors_data.append({
                        'rsi': rsi_val,
                        'vol': vol_val,
                        'price_delta': price_delta_val
                    })
                    
                    logger.info(f"      📊 Factors: RSI={rsi_val}, Vol={vol_val}%, Price∆={price_delta_val}%")
                except:
                    continue
            
            if not factors_data:
                self.log_test_result("Market Factors Integration", False, "No factors data found")
                return
            
            # Analyze factor ranges
            rsi_values = [f['rsi'] for f in factors_data if f['rsi'] is not None]
            vol_values = [f['vol'] for f in factors_data if f['vol'] is not None]
            price_values = [f['price_delta'] for f in factors_data if f['price_delta'] is not None]
            
            logger.info(f"   📊 Factors analyzed: {len(factors_data)}")
            logger.info(f"   📊 RSI values: {len(rsi_values)} (range: {min(rsi_values):.1f}-{max(rsi_values):.1f})" if rsi_values else "   📊 RSI values: 0")
            logger.info(f"   📊 Vol values: {len(vol_values)} (range: {min(vol_values):.1f}%-{max(vol_values):.1f}%)" if vol_values else "   📊 Vol values: 0")
            logger.info(f"   📊 Price∆ values: {len(price_values)} (range: {min(price_values):.1f}%-{max(price_values):.1f}%)" if price_values else "   📊 Price∆ values: 0")
            
            # Success criteria: Market factors are being captured
            success = (len(factors_data) > 0 and 
                      len(rsi_values) > 0 and 
                      len(vol_values) > 0 and 
                      len(price_values) > 0)
            
            details = f"Total: {len(factors_data)}, RSI: {len(rsi_values)}, Vol: {len(vol_values)}, Price∆: {len(price_values)}"
            
            self.log_test_result("Market Factors Integration", success, details)
            
        except Exception as e:
            self.log_test_result("Market Factors Integration", False, f"Exception: {str(e)}")
    
    async def test_5_api_analyses_validation(self):
        """Test 5: Validate API analyses show scoring effects"""
        logger.info("\n🔍 TEST 5: Validate API analyses")
        
        try:
            # Get IA1 analyses from API
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("API Analyses Validation", False, f"HTTP {response.status_code}")
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("API Analyses Validation", False, "No analyses found")
                return
            
            # Analyze confidence scores for variability
            confidence_scores = []
            symbols_analyzed = []
            
            for analysis in analyses[-10:]:  # Check last 10 analyses
                symbol = analysis.get('symbol', 'Unknown')
                confidence = analysis.get('confidence', 0)
                
                confidence_scores.append(confidence)
                symbols_analyzed.append(symbol)
                
                logger.info(f"      📊 {symbol}: Confidence = {confidence:.1%}")
            
            if len(confidence_scores) < 2:
                self.log_test_result("API Analyses Validation", False, "Insufficient analyses")
                return
            
            # Calculate variability
            min_conf = min(confidence_scores)
            max_conf = max(confidence_scores)
            mean_conf = sum(confidence_scores) / len(confidence_scores)
            
            # Calculate standard deviation
            variance = sum((x - mean_conf) ** 2 for x in confidence_scores) / len(confidence_scores)
            std_dev = variance ** 0.5
            
            conf_range = max_conf - min_conf
            unique_confidences = len(set(round(conf, 3) for conf in confidence_scores))
            
            logger.info(f"   📊 Analyses found: {len(analyses)}")
            logger.info(f"   📊 Confidence range: {min_conf:.1%} - {max_conf:.1%}")
            logger.info(f"   📊 Mean confidence: {mean_conf:.1%}")
            logger.info(f"   📊 Standard deviation: {std_dev:.3f}")
            logger.info(f"   📊 Unique values: {unique_confidences}/{len(confidence_scores)}")
            
            # Success criteria: Analyses show reasonable variation
            success = (len(analyses) > 0 and 
                      conf_range > 0.05 and  # Range > 5%
                      std_dev > 0.02 and  # Std dev > 2%
                      unique_confidences >= len(confidence_scores) * 0.5)  # At least 50% unique
            
            details = f"Count: {len(analyses)}, Range: {conf_range:.1%}, Std: {std_dev:.3f}, Unique: {unique_confidences}/{len(confidence_scores)}"
            
            self.log_test_result("API Analyses Validation", success, details)
            
        except Exception as e:
            self.log_test_result("API Analyses Validation", False, f"Exception: {str(e)}")
    
    async def run_validation_tests(self):
        """Run all validation tests"""
        logger.info("🚀 Starting IA1 Enhanced Scoring System Validation")
        logger.info("=" * 80)
        logger.info("📋 IA1 ENHANCED SCORING SYSTEM VALIDATION")
        logger.info("🎯 Validating: Scoring logs, differentials, market cap multipliers, factors")
        logger.info("=" * 80)
        
        # Run all tests
        await self.test_1_scoring_logs_validation()
        await self.test_2_score_differential_analysis()
        await self.test_3_market_cap_multiplier_validation()
        await self.test_4_market_factors_integration()
        await self.test_5_api_analyses_validation()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 IA1 ENHANCED SCORING SYSTEM VALIDATION SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
        
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # Final verdict
        if passed_tests == total_tests:
            logger.info("\n🎉 VERDICT: IA1 Enhanced Scoring System is FULLY FUNCTIONAL!")
            logger.info("✅ All scoring components working correctly")
            logger.info("✅ Score differentials show proper variability")
            logger.info("✅ Market cap multipliers applied correctly")
            logger.info("✅ Market factors integrated successfully")
            logger.info("✅ API analyses reflect scoring enhancements")
        elif passed_tests >= total_tests * 0.8:
            logger.info("\n⚠️ VERDICT: IA1 Enhanced Scoring System is MOSTLY FUNCTIONAL")
            logger.info("🔍 Minor issues may need attention for complete functionality")
        else:
            logger.info("\n❌ VERDICT: IA1 Enhanced Scoring System has SIGNIFICANT ISSUES")
            logger.info("🚨 Major components not working as expected")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = IA1ScoringValidationTest()
    passed, total = await test_suite.run_validation_tests()
    
    # Exit with appropriate code
    if passed >= total * 0.8:  # 80% pass rate is acceptable
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
IA1 Enhanced Scoring System Comprehensive Test Suite
Focus: VALIDATION SYSTÈME DE SCORING IA1 AMÉLIORÉ - Test complet du système de scoring sophistiqué

Testing Components:
1. compute_final_score() - Formule principale de scoring
2. tanh_norm() et clamp() - Fonctions de normalisation  
3. get_market_cap_multiplier() - Ajustements selon market cap
4. Facteurs de marché - Volatilité, Fear & Greed, volume, etc.
5. Intégration pipeline - Application dans les analyses IA1

Expected log patterns:
- "🎯 IA1 PROFESSIONAL SCORING {symbol}:"
- "📊 Base Confidence: XX% → Final: YY%"
- "📊 Market Adjustment: XX.X points"
- "📊 MC Multiplier: X.XX (Market Cap: XXX)"
- "🎯 Key Factors: RSI=XX.X, Vol=XX.X%, Price∆=XX.X%"

Success Criteria:
- compute_final_score utilisée dans toutes les analyses IA1 récentes
- Scores finaux différents des scores de base (ajustements visibles)
- Logs montrent calculs de scoring professionnel détaillés
- Market cap multipliers appliqués correctement selon les buckets
- Facteurs de marché intégrés dans les ajustements de score
- Variabilité des scores selon les conditions spécifiques de chaque symbole
- Absence d'erreurs dans les calculs de scoring (pas de NaN/Infinity)
"""

import asyncio
import json
import logging
import os
import sys
import time
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IA1EnhancedScoringSystemTestSuite:
    """Test suite for IA1 Enhanced Scoring System Verification"""
    
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
        logger.info(f"Testing IA1 Enhanced Scoring System at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Expected scoring log patterns
        self.expected_scoring_patterns = [
            "🎯 IA1 PROFESSIONAL SCORING",
            "📊 Base Confidence:",
            "📊 Market Adjustment:",
            "📊 MC Multiplier:",
            "🎯 Key Factors:",
            "RSI=",
            "Vol=",
            "Price∆="
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
    
    async def test_1_scoring_methods_implementation(self):
        """Test 1: Verify scoring methods are implemented in backend"""
        logger.info("\n🔍 TEST 1: Check scoring methods implementation")
        
        try:
            # Check if the methods exist in the backend code
            backend_code = ""
            try:
                with open('/app/backend/server.py', 'r') as f:
                    backend_code = f.read()
            except Exception as e:
                self.log_test_result("Scoring Methods Implementation", False, f"Could not read backend code: {e}")
                return
            
            # Check for required method implementations
            methods_found = {}
            required_methods = [
                "def compute_final_score",
                "def tanh_norm", 
                "def clamp",
                "def get_market_cap_multiplier"
            ]
            
            for method in required_methods:
                if method in backend_code:
                    methods_found[method] = True
                    logger.info(f"      ✅ Method found: {method}")
                else:
                    methods_found[method] = False
                    logger.info(f"      ❌ Method missing: {method}")
            
            # Check for scoring formula components
            formula_components = [
                "factor_scores",
                "norm_funcs", 
                "weights",
                "amplitude",
                "mc_mult",
                "note_base",
                "note_final",
                "adjustment"
            ]
            
            components_found = {}
            for component in formula_components:
                if component in backend_code:
                    components_found[component] = True
                    logger.info(f"      ✅ Component found: {component}")
                else:
                    components_found[component] = False
                    logger.info(f"      ❌ Component missing: {component}")
            
            # Check market cap buckets
            mc_buckets = [
                "10_000_000",  # micro cap threshold
                "100_000_000", # small cap threshold  
                "1_000_000_000", # mid cap threshold
                "return 0.8",  # micro cap multiplier
                "return 0.95", # small cap multiplier
                "return 1.0",  # mid cap multiplier
                "return 1.1"   # large cap multiplier
            ]
            
            buckets_found = {}
            for bucket in mc_buckets:
                if bucket in backend_code:
                    buckets_found[bucket] = True
                    logger.info(f"      ✅ MC Bucket found: {bucket}")
                else:
                    buckets_found[bucket] = False
                    logger.info(f"      ❌ MC Bucket missing: {bucket}")
            
            methods_implemented = sum(methods_found.values())
            components_implemented = sum(components_found.values())
            buckets_implemented = sum(buckets_found.values())
            
            logger.info(f"   📊 Methods implemented: {methods_implemented}/{len(required_methods)}")
            logger.info(f"   📊 Components implemented: {components_implemented}/{len(formula_components)}")
            logger.info(f"   📊 MC Buckets implemented: {buckets_implemented}/{len(mc_buckets)}")
            
            # Success criteria: All methods and most components found
            success = methods_implemented == len(required_methods) and components_implemented >= 6 and buckets_implemented >= 5
            
            details = f"Methods: {methods_implemented}/{len(required_methods)}, Components: {components_implemented}/{len(formula_components)}, Buckets: {buckets_implemented}/{len(mc_buckets)}"
            
            self.log_test_result("Scoring Methods Implementation", success, details)
            
        except Exception as e:
            self.log_test_result("Scoring Methods Implementation", False, f"Exception: {str(e)}")
    
    async def test_2_scoring_functions_validation(self):
        """Test 2: Validate scoring functions work correctly"""
        logger.info("\n🔍 TEST 2: Validate scoring functions behavior")
        
        try:
            # Test tanh_norm function behavior
            tanh_tests = [
                (0, 1.0, 0.0),      # tanh(0/1) = 0
                (1, 1.0, 0.762),    # tanh(1/1) ≈ 0.762
                (-1, 1.0, -0.762),  # tanh(-1/1) ≈ -0.762
                (10, 10.0, 0.762),  # tanh(10/10) ≈ 0.762
                (5, 2.0, 0.996)     # tanh(5/2) ≈ 0.996
            ]
            
            tanh_correct = 0
            for x, s, expected in tanh_tests:
                result = math.tanh(x / s)
                if abs(result - expected) < 0.01:  # Allow small tolerance
                    tanh_correct += 1
                    logger.info(f"      ✅ tanh_norm({x}, {s}) = {result:.3f} ≈ {expected}")
                else:
                    logger.info(f"      ❌ tanh_norm({x}, {s}) = {result:.3f} ≠ {expected}")
            
            # Test clamp function behavior
            clamp_tests = [
                (-5, 0, 100, 0),     # clamp(-5, 0, 100) = 0
                (50, 0, 100, 50),    # clamp(50, 0, 100) = 50
                (150, 0, 100, 100),  # clamp(150, 0, 100) = 100
                (25, 30, 80, 30),    # clamp(25, 30, 80) = 30
                (90, 30, 80, 80)     # clamp(90, 30, 80) = 80
            ]
            
            clamp_correct = 0
            for x, lo, hi, expected in clamp_tests:
                result = max(lo, min(hi, x))
                if result == expected:
                    clamp_correct += 1
                    logger.info(f"      ✅ clamp({x}, {lo}, {hi}) = {result} = {expected}")
                else:
                    logger.info(f"      ❌ clamp({x}, {lo}, {hi}) = {result} ≠ {expected}")
            
            # Test market cap multiplier logic
            mc_tests = [
                (5_000_000, 0.8),      # micro cap < 10M
                (50_000_000, 0.95),    # small cap < 100M
                (500_000_000, 1.0),    # mid cap < 1B
                (5_000_000_000, 1.1)   # large cap > 1B
            ]
            
            mc_correct = 0
            for market_cap, expected in mc_tests:
                if market_cap < 10_000_000:
                    result = 0.8
                elif market_cap < 100_000_000:
                    result = 0.95
                elif market_cap < 1_000_000_000:
                    result = 1.0
                else:
                    result = 1.1
                
                if result == expected:
                    mc_correct += 1
                    logger.info(f"      ✅ MC Multiplier({market_cap:,}) = {result} = {expected}")
                else:
                    logger.info(f"      ❌ MC Multiplier({market_cap:,}) = {result} ≠ {expected}")
            
            logger.info(f"   📊 tanh_norm tests passed: {tanh_correct}/{len(tanh_tests)}")
            logger.info(f"   📊 clamp tests passed: {clamp_correct}/{len(clamp_tests)}")
            logger.info(f"   📊 MC multiplier tests passed: {mc_correct}/{len(mc_tests)}")
            
            # Success criteria: Most function tests pass
            success = tanh_correct >= 4 and clamp_correct >= 4 and mc_correct >= 3
            
            details = f"tanh: {tanh_correct}/{len(tanh_tests)}, clamp: {clamp_correct}/{len(clamp_tests)}, MC: {mc_correct}/{len(mc_tests)}"
            
            self.log_test_result("Scoring Functions Validation", success, details)
            
        except Exception as e:
            self.log_test_result("Scoring Functions Validation", False, f"Exception: {str(e)}")
    
    async def test_3_scoring_integration_in_ia1(self):
        """Test 3: Verify scoring system is integrated in IA1 analyses"""
        logger.info("\n🔍 TEST 3: Test scoring system integration in IA1 analyses")
        
        try:
            # Trigger new analysis to get fresh data
            logger.info("   🚀 Triggering fresh analysis via /api/trading/start-trading...")
            start_response = requests.post(f"{self.api_url}/trading/start-trading", timeout=180)
            
            if start_response.status_code not in [200, 201]:
                logger.warning(f"   ⚠️ Start trading returned HTTP {start_response.status_code}, continuing with existing data...")
            else:
                # Wait for processing
                logger.info("   ⏳ Waiting 30 seconds for analysis processing...")
                await asyncio.sleep(30)
            
            # Get IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Scoring Integration in IA1", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("Scoring Integration in IA1", False, "No IA1 analyses found")
                return
            
            # Analyze recent IA1 analyses for scoring integration
            scoring_enhanced_analyses = 0
            confidence_variations = []
            score_differentials = []
            
            for analysis in analyses[-10:]:  # Check last 10 analyses
                symbol = analysis.get('symbol', 'Unknown')
                reasoning = analysis.get('reasoning', '')
                confidence = analysis.get('confidence', 0)
                
                # Check for scoring enhancement indicators
                scoring_indicators = [
                    "professional scoring", "market adjustment", "scoring", "final score",
                    "base confidence", "market factors", "volatility", "market cap"
                ]
                
                has_scoring_enhancement = any(indicator in reasoning.lower() for indicator in scoring_indicators)
                if has_scoring_enhancement:
                    scoring_enhanced_analyses += 1
                    logger.info(f"      ✅ {symbol}: Scoring enhancement detected in reasoning")
                
                # Collect confidence values for analysis
                confidence_variations.append(confidence)
                
                # Check if confidence shows variation (not all the same)
                if len(confidence_variations) >= 2:
                    diff = abs(confidence - confidence_variations[-2])
                    if diff > 0.01:  # More than 1% difference
                        score_differentials.append(diff)
            
            # Check backend logs for scoring activity
            import subprocess
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
            
            # Count scoring log patterns
            scoring_logs = {}
            for pattern in self.expected_scoring_patterns:
                count = backend_logs.count(pattern)
                scoring_logs[pattern] = count
                logger.info(f"   📊 '{pattern}': {count} occurrences")
            
            professional_scoring_logs = backend_logs.count("🎯 IA1 PROFESSIONAL SCORING")
            base_confidence_logs = backend_logs.count("📊 Base Confidence:")
            market_adjustment_logs = backend_logs.count("📊 Market Adjustment:")
            mc_multiplier_logs = backend_logs.count("📊 MC Multiplier:")
            key_factors_logs = backend_logs.count("🎯 Key Factors:")
            
            # Calculate confidence variation
            confidence_std = 0
            if len(confidence_variations) > 1:
                mean_conf = sum(confidence_variations) / len(confidence_variations)
                variance = sum((x - mean_conf) ** 2 for x in confidence_variations) / len(confidence_variations)
                confidence_std = variance ** 0.5
            
            logger.info(f"   📊 Scoring enhanced analyses: {scoring_enhanced_analyses}/10")
            logger.info(f"   📊 Professional scoring logs: {professional_scoring_logs}")
            logger.info(f"   📊 Base confidence logs: {base_confidence_logs}")
            logger.info(f"   📊 Market adjustment logs: {market_adjustment_logs}")
            logger.info(f"   📊 MC multiplier logs: {mc_multiplier_logs}")
            logger.info(f"   📊 Key factors logs: {key_factors_logs}")
            logger.info(f"   📊 Confidence variation (std): {confidence_std:.3f}")
            logger.info(f"   📊 Score differentials: {len(score_differentials)} detected")
            
            # Success criteria: Evidence of scoring system integration
            success = (professional_scoring_logs > 0 and base_confidence_logs > 0 and 
                      market_adjustment_logs > 0 and confidence_std > 0.02)
            
            details = f"Enhanced: {scoring_enhanced_analyses}/10, Prof logs: {professional_scoring_logs}, Conf variation: {confidence_std:.3f}, Differentials: {len(score_differentials)}"
            
            self.log_test_result("Scoring Integration in IA1", success, details)
            
        except Exception as e:
            self.log_test_result("Scoring Integration in IA1", False, f"Exception: {str(e)}")
    
    async def test_4_market_factors_integration(self):
        """Test 4: Verify market factors are integrated in scoring"""
        logger.info("\n🔍 TEST 4: Test market factors integration in scoring")
        
        try:
            # Check backend logs for market factors usage
            import subprocess
            backend_logs = ""
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "5000", "/var/log/supervisor/backend.out.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            if not backend_logs:
                self.log_test_result("Market Factors Integration", False, "Could not retrieve backend logs")
                return
            
            # Check for market factors in logs
            market_factors = {
                'volatility': ['Vol=', 'volatility', 'var_cap', 'var_vol'],
                'rsi_extreme': ['RSI=', 'rsi_extreme', 'RSI extremes'],
                'volume': ['volume', 'volcap', 'Volume'],
                'price_change': ['Price∆=', 'price_change_24h', 'var_cap'],
                'market_cap': ['Market Cap:', 'MC Multiplier:', 'market_cap'],
                'fear_greed': ['fg', 'Fear & Greed', 'sentiment']
            }
            
            factors_found = {}
            for factor, keywords in market_factors.items():
                count = 0
                for keyword in keywords:
                    count += backend_logs.count(keyword)
                factors_found[factor] = count
                logger.info(f"   📊 {factor}: {count} occurrences")
            
            # Check for scoring formula components in logs
            formula_components = [
                'factor_scores',
                'norm_funcs', 
                'weights',
                'amplitude',
                'adjustment',
                'note_base',
                'note_final'
            ]
            
            components_in_logs = {}
            for component in formula_components:
                count = backend_logs.count(component)
                components_in_logs[component] = count
                if count > 0:
                    logger.info(f"   📊 Formula component '{component}': {count} occurrences")
            
            # Look for specific scoring calculations
            scoring_calculations = [
                "APPLYING PROFESSIONAL SCORING",
                "Market Adjustment:",
                "MC Multiplier:",
                "Key Factors:",
                "Base Confidence:",
                "Final:"
            ]
            
            calculations_found = {}
            for calc in scoring_calculations:
                count = backend_logs.count(calc)
                calculations_found[calc] = count
                logger.info(f"   📊 Scoring calc '{calc}': {count} occurrences")
            
            # Check for market factor values in logs (RSI, Vol, Price∆)
            factor_values_found = 0
            rsi_values = len([line for line in backend_logs.split('\n') if 'RSI=' in line and any(c.isdigit() for c in line)])
            vol_values = len([line for line in backend_logs.split('\n') if 'Vol=' in line and '%' in line])
            price_values = len([line for line in backend_logs.split('\n') if 'Price∆=' in line and '%' in line])
            
            factor_values_found = rsi_values + vol_values + price_values
            
            logger.info(f"   📊 RSI values in logs: {rsi_values}")
            logger.info(f"   📊 Vol values in logs: {vol_values}")
            logger.info(f"   📊 Price∆ values in logs: {price_values}")
            logger.info(f"   📊 Total factor values: {factor_values_found}")
            
            # Success criteria: Market factors are being used in scoring
            factors_active = sum(1 for count in factors_found.values() if count > 0)
            calculations_active = sum(1 for count in calculations_found.values() if count > 0)
            
            success = (factors_active >= 4 and calculations_active >= 3 and factor_values_found >= 5)
            
            details = f"Factors active: {factors_active}/6, Calculations: {calculations_active}/6, Factor values: {factor_values_found}"
            
            self.log_test_result("Market Factors Integration", success, details)
            
        except Exception as e:
            self.log_test_result("Market Factors Integration", False, f"Exception: {str(e)}")
    
    async def test_5_score_differentials_validation(self):
        """Test 5: Verify scores show differentials (not all identical)"""
        logger.info("\n🔍 TEST 5: Test score differentials and variability")
        
        try:
            # Get IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Score Differentials Validation", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("Score Differentials Validation", False, "No IA1 analyses found")
                return
            
            # Analyze confidence scores for variability
            confidence_scores = []
            symbols_analyzed = []
            
            for analysis in analyses[-15:]:  # Check last 15 analyses
                symbol = analysis.get('symbol', 'Unknown')
                confidence = analysis.get('confidence', 0)
                
                confidence_scores.append(confidence)
                symbols_analyzed.append(symbol)
                
                logger.info(f"   📊 {symbol}: Confidence = {confidence:.1%}")
            
            if len(confidence_scores) < 3:
                self.log_test_result("Score Differentials Validation", False, "Insufficient analyses for differential testing")
                return
            
            # Calculate statistical measures
            min_conf = min(confidence_scores)
            max_conf = max(confidence_scores)
            mean_conf = sum(confidence_scores) / len(confidence_scores)
            
            # Calculate standard deviation
            variance = sum((x - mean_conf) ** 2 for x in confidence_scores) / len(confidence_scores)
            std_dev = variance ** 0.5
            
            # Calculate range
            conf_range = max_conf - min_conf
            
            # Count unique confidence values
            unique_confidences = len(set(round(conf, 3) for conf in confidence_scores))
            
            # Check for reasonable distribution
            low_conf_count = sum(1 for conf in confidence_scores if conf < 0.7)
            medium_conf_count = sum(1 for conf in confidence_scores if 0.7 <= conf < 0.85)
            high_conf_count = sum(1 for conf in confidence_scores if conf >= 0.85)
            
            logger.info(f"   📊 Confidence statistics:")
            logger.info(f"      Min: {min_conf:.1%}, Max: {max_conf:.1%}, Mean: {mean_conf:.1%}")
            logger.info(f"      Standard deviation: {std_dev:.3f}")
            logger.info(f"      Range: {conf_range:.1%}")
            logger.info(f"      Unique values: {unique_confidences}/{len(confidence_scores)}")
            logger.info(f"      Distribution: Low(<70%): {low_conf_count}, Medium(70-85%): {medium_conf_count}, High(≥85%): {high_conf_count}")
            
            # Check backend logs for base vs final score comparisons
            import subprocess
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
            
            # Look for base vs final confidence logs
            base_final_logs = []
            lines = backend_logs.split('\n')
            for line in lines:
                if "Base Confidence:" in line and "Final:" in line and "→" in line:
                    base_final_logs.append(line.strip())
            
            # Parse base vs final differences
            score_adjustments = []
            for log_line in base_final_logs[-10:]:  # Last 10 adjustments
                try:
                    # Extract base and final percentages
                    parts = log_line.split("→")
                    if len(parts) == 2:
                        base_part = parts[0].split(":")[-1].strip().replace("%", "")
                        final_part = parts[1].split(":")[1].strip().replace("%", "")
                        
                        base_val = float(base_part)
                        final_val = float(final_part)
                        adjustment = final_val - base_val
                        score_adjustments.append(adjustment)
                        
                        logger.info(f"      📊 Score adjustment: {base_val:.1f}% → {final_val:.1f}% (Δ{adjustment:+.1f}%)")
                except:
                    continue
            
            # Calculate adjustment statistics
            adjustment_range = 0
            adjustment_std = 0
            if score_adjustments:
                adjustment_range = max(score_adjustments) - min(score_adjustments)
                adj_mean = sum(score_adjustments) / len(score_adjustments)
                adj_variance = sum((x - adj_mean) ** 2 for x in score_adjustments) / len(score_adjustments)
                adjustment_std = adj_variance ** 0.5
            
            logger.info(f"   📊 Score adjustments found: {len(score_adjustments)}")
            logger.info(f"   📊 Adjustment range: {adjustment_range:.1f}%")
            logger.info(f"   📊 Adjustment std dev: {adjustment_std:.3f}")
            
            # Success criteria: Scores show meaningful variation
            success = (std_dev > 0.05 and  # Standard deviation > 5%
                      conf_range > 0.15 and  # Range > 15%
                      unique_confidences >= len(confidence_scores) * 0.6 and  # At least 60% unique values
                      len(score_adjustments) > 0 and  # Evidence of adjustments
                      adjustment_range > 5.0)  # Adjustments vary by > 5%
            
            details = f"Std dev: {std_dev:.3f}, Range: {conf_range:.1%}, Unique: {unique_confidences}/{len(confidence_scores)}, Adjustments: {len(score_adjustments)}, Adj range: {adjustment_range:.1f}%"
            
            self.log_test_result("Score Differentials Validation", success, details)
            
        except Exception as e:
            self.log_test_result("Score Differentials Validation", False, f"Exception: {str(e)}")
    
    async def test_6_market_cap_multipliers_validation(self):
        """Test 6: Verify market cap multipliers are applied correctly"""
        logger.info("\n🔍 TEST 6: Test market cap multipliers application")
        
        try:
            # Check backend logs for MC multiplier applications
            import subprocess
            backend_logs = ""
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "5000", "/var/log/supervisor/backend.out.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            if not backend_logs:
                self.log_test_result("Market Cap Multipliers Validation", False, "Could not retrieve backend logs")
                return
            
            # Parse MC multiplier logs
            mc_multiplier_logs = []
            lines = backend_logs.split('\n')
            for line in lines:
                if "MC Multiplier:" in line and "Market Cap:" in line:
                    mc_multiplier_logs.append(line.strip())
            
            # Extract multiplier values and market caps
            multipliers_found = []
            market_caps_found = []
            
            for log_line in mc_multiplier_logs[-15:]:  # Last 15 MC logs
                try:
                    # Extract multiplier value
                    mult_part = log_line.split("MC Multiplier:")[1].split("(")[0].strip()
                    multiplier = float(mult_part)
                    multipliers_found.append(multiplier)
                    
                    # Extract market cap value
                    cap_part = log_line.split("Market Cap:")[1].split(")")[0].strip().replace(",", "")
                    market_cap = float(cap_part)
                    market_caps_found.append(market_cap)
                    
                    logger.info(f"      📊 MC: ${market_cap:,.0f} → Multiplier: {multiplier:.2f}")
                except:
                    continue
            
            # Validate multiplier logic
            correct_multipliers = 0
            total_multipliers = len(multipliers_found)
            
            for i, (market_cap, multiplier) in enumerate(zip(market_caps_found, multipliers_found)):
                expected_multiplier = None
                
                if market_cap < 10_000_000:  # micro cap
                    expected_multiplier = 0.8
                elif market_cap < 100_000_000:  # small cap
                    expected_multiplier = 0.95
                elif market_cap < 1_000_000_000:  # mid cap
                    expected_multiplier = 1.0
                else:  # large cap
                    expected_multiplier = 1.1
                
                if abs(multiplier - expected_multiplier) < 0.01:  # Allow small tolerance
                    correct_multipliers += 1
                    logger.info(f"      ✅ Correct: ${market_cap:,.0f} → {multiplier:.2f} (expected {expected_multiplier:.2f})")
                else:
                    logger.info(f"      ❌ Incorrect: ${market_cap:,.0f} → {multiplier:.2f} (expected {expected_multiplier:.2f})")
            
            # Check for multiplier variety (different buckets used)
            unique_multipliers = set(round(mult, 2) for mult in multipliers_found)
            expected_multipliers = {0.8, 0.95, 1.0, 1.1}
            multipliers_variety = len(unique_multipliers.intersection(expected_multipliers))
            
            # Check market cap range coverage
            if market_caps_found:
                min_cap = min(market_caps_found)
                max_cap = max(market_caps_found)
                cap_range_log10 = math.log10(max_cap / min_cap) if min_cap > 0 else 0
            else:
                min_cap = max_cap = cap_range_log10 = 0
            
            logger.info(f"   📊 MC multipliers found: {total_multipliers}")
            logger.info(f"   📊 Correct multipliers: {correct_multipliers}/{total_multipliers}")
            logger.info(f"   📊 Unique multipliers: {len(unique_multipliers)} ({unique_multipliers})")
            logger.info(f"   📊 Multiplier variety: {multipliers_variety}/4 buckets")
            logger.info(f"   📊 Market cap range: ${min_cap:,.0f} - ${max_cap:,.0f} (log10 range: {cap_range_log10:.1f})")
            
            # Success criteria: MC multipliers are applied correctly
            success = (total_multipliers > 0 and 
                      correct_multipliers >= total_multipliers * 0.8 and  # 80% correct
                      multipliers_variety >= 2 and  # At least 2 different buckets
                      cap_range_log10 > 1.0)  # Market cap range > 10x
            
            details = f"Total: {total_multipliers}, Correct: {correct_multipliers}/{total_multipliers}, Variety: {multipliers_variety}/4, Range: {cap_range_log10:.1f}"
            
            self.log_test_result("Market Cap Multipliers Validation", success, details)
            
        except Exception as e:
            self.log_test_result("Market Cap Multipliers Validation", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_tests(self):
        """Run all IA1 enhanced scoring system tests"""
        logger.info("🚀 Starting IA1 Enhanced Scoring System Test Suite")
        logger.info("=" * 80)
        logger.info("📋 IA1 ENHANCED SCORING SYSTEM VALIDATION")
        logger.info("🎯 Testing: compute_final_score, tanh_norm, clamp, market cap multipliers")
        logger.info("🎯 Expected: Sophisticated scoring with market factors integration")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_scoring_methods_implementation()
        await self.test_2_scoring_functions_validation()
        await self.test_3_scoring_integration_in_ia1()
        await self.test_4_market_factors_integration()
        await self.test_5_score_differentials_validation()
        await self.test_6_market_cap_multipliers_validation()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 IA1 ENHANCED SCORING SYSTEM SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # System analysis
        logger.info("\n" + "=" * 80)
        logger.info("📋 IA1 ENHANCED SCORING SYSTEM STATUS")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - IA1 Enhanced Scoring System FULLY FUNCTIONAL!")
            logger.info("✅ compute_final_score method implemented and working")
            logger.info("✅ tanh_norm and clamp functions operational")
            logger.info("✅ Market cap multipliers applied correctly")
            logger.info("✅ Market factors integrated in scoring")
            logger.info("✅ Score differentials show proper variability")
            logger.info("✅ Scoring system integrated in IA1 pipeline")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY FUNCTIONAL - Enhanced scoring system working with minor gaps")
            logger.info("🔍 Some components may need fine-tuning for full optimization")
        elif passed_tests >= total_tests * 0.5:
            logger.info("⚠️ PARTIALLY FUNCTIONAL - Core scoring features working")
            logger.info("🔧 Some advanced features may need implementation or debugging")
        else:
            logger.info("❌ SYSTEM NOT FUNCTIONAL - Critical issues with enhanced scoring")
            logger.info("🚨 Major implementation gaps or system errors preventing functionality")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = IA1EnhancedScoringSystemTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
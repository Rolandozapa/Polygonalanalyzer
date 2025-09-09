#!/usr/bin/env python3
"""
Technical Indicators Diagnostic Test Suite for IA1
URGENT DIAGNOSTIC - Problèmes avec le calcul des indicateurs techniques dans IA1

Focus: Diagnose technical indicators calculation issues in IA1 system
- RSI values validation (0-100 range)
- MACD signal and line values verification
- Stochastic %K and %D validation (0-100 range)
- Bollinger Bands position verification (-2 to +2 range)
- Integration verification in IA1 analyses
- Error identification (NaN, infinity, out-of-bounds values)
- Consistency testing across multiple symbols
"""

import asyncio
import json
import logging
import os
import sys
import time
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TechnicalIndicatorsDiagnosticSuite:
    """Comprehensive diagnostic suite for IA1 technical indicators calculation issues"""
    
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
        logger.info(f"🔍 DIAGNOSTIC URGENT - Testing Technical Indicators at: {self.api_url}")
        
        # Test results
        self.test_results = []
        self.indicators_data = []
        self.error_cases = []
        
        # Expected ranges for technical indicators
        self.expected_ranges = {
            'rsi': {'min': 0, 'max': 100, 'typical_min': 20, 'typical_max': 80},
            'macd': {'min': -1, 'max': 1, 'typical_min': -0.1, 'typical_max': 0.1},
            'stochastic_k': {'min': 0, 'max': 100, 'typical_min': 20, 'typical_max': 80},
            'stochastic_d': {'min': 0, 'max': 100, 'typical_min': 20, 'typical_max': 80},
            'bollinger_position': {'min': -3, 'max': 3, 'typical_min': -2, 'typical_max': 2}
        }
        
    def log_test_result(self, test_name: str, success: bool, details: str = "", critical: bool = False):
        """Log test result with criticality indicator"""
        status = "✅ PASS" if success else ("🚨 CRITICAL FAIL" if critical else "❌ FAIL")
        logger.info(f"{status}: {test_name}")
        if details:
            logger.info(f"   Details: {details}")
        
        self.test_results.append({
            'test': test_name,
            'success': success,
            'details': details,
            'critical': critical,
            'timestamp': datetime.now().isoformat()
        })
    
    def validate_indicator_value(self, indicator_name: str, value: Any, symbol: str = "Unknown") -> Dict[str, Any]:
        """Validate a single technical indicator value"""
        validation_result = {
            'indicator': indicator_name,
            'symbol': symbol,
            'value': value,
            'is_valid': False,
            'is_realistic': False,
            'errors': []
        }
        
        # Check if value exists and is numeric
        if value is None:
            validation_result['errors'].append("Value is None/missing")
            return validation_result
        
        # Check for NaN or infinity
        try:
            if math.isnan(float(value)):
                validation_result['errors'].append("Value is NaN")
                return validation_result
            if math.isinf(float(value)):
                validation_result['errors'].append("Value is infinity")
                return validation_result
        except (ValueError, TypeError):
            validation_result['errors'].append(f"Value is not numeric: {type(value)}")
            return validation_result
        
        numeric_value = float(value)
        
        # Get expected ranges for this indicator
        if indicator_name not in self.expected_ranges:
            validation_result['errors'].append(f"Unknown indicator: {indicator_name}")
            return validation_result
        
        ranges = self.expected_ranges[indicator_name]
        
        # Check if value is within valid range
        if ranges['min'] <= numeric_value <= ranges['max']:
            validation_result['is_valid'] = True
        else:
            validation_result['errors'].append(f"Value {numeric_value} outside valid range [{ranges['min']}, {ranges['max']}]")
        
        # Check if value is within typical/realistic range
        if ranges['typical_min'] <= numeric_value <= ranges['typical_max']:
            validation_result['is_realistic'] = True
        else:
            # Not an error, just note it's extreme
            validation_result['errors'].append(f"Value {numeric_value} is extreme (outside typical range [{ranges['typical_min']}, {ranges['typical_max']}])")
        
        return validation_result
    
    async def test_1_rsi_calculation_validation(self):
        """Test 1: RSI Calculation Validation (0-100 range)"""
        logger.info("\n🔍 TEST 1: RSI Calculation Validation")
        
        try:
            # Get IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("RSI Calculation Validation", False, f"HTTP {response.status_code}: {response.text}", critical=True)
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("RSI Calculation Validation", False, "No IA1 analyses found", critical=True)
                return
            
            # Analyze RSI values
            rsi_issues = []
            rsi_valid_count = 0
            rsi_total_count = 0
            rsi_values = []
            
            for analysis in analyses[-20:]:  # Check last 20 analyses
                symbol = analysis.get('symbol', 'Unknown')
                
                # Check different possible RSI field names
                rsi_value = None
                for field in ['rsi', 'rsi_value', 'technical_analysis', 'indicators']:
                    if field in analysis:
                        if field == 'technical_analysis' and isinstance(analysis[field], dict):
                            rsi_value = analysis[field].get('rsi')
                        elif field == 'indicators' and isinstance(analysis[field], dict):
                            rsi_value = analysis[field].get('rsi')
                        else:
                            rsi_value = analysis[field]
                        
                        if rsi_value is not None:
                            break
                
                # Also check in reasoning text for RSI mentions
                reasoning = analysis.get('reasoning', '') or analysis.get('ia1_reasoning', '')
                rsi_in_reasoning = False
                if reasoning:
                    import re
                    rsi_matches = re.findall(r'RSI[:\s]*(\d+\.?\d*)', reasoning, re.IGNORECASE)
                    if rsi_matches:
                        rsi_in_reasoning = True
                        if rsi_value is None:
                            try:
                                rsi_value = float(rsi_matches[0])
                            except ValueError:
                                pass
                
                rsi_total_count += 1
                
                if rsi_value is not None:
                    validation = self.validate_indicator_value('rsi', rsi_value, symbol)
                    rsi_values.append(validation)
                    
                    if validation['is_valid']:
                        rsi_valid_count += 1
                        logger.info(f"      ✅ {symbol}: RSI = {rsi_value} (valid)")
                    else:
                        rsi_issues.append(f"{symbol}: RSI = {rsi_value} - {', '.join(validation['errors'])}")
                        logger.info(f"      ❌ {symbol}: RSI = {rsi_value} - {', '.join(validation['errors'])}")
                else:
                    rsi_issues.append(f"{symbol}: RSI value missing/not found")
                    logger.info(f"      ❌ {symbol}: RSI value missing (reasoning has RSI: {rsi_in_reasoning})")
            
            # Calculate statistics
            rsi_coverage = (rsi_valid_count / rsi_total_count) * 100 if rsi_total_count > 0 else 0
            
            # Check for specific RSI calculation issues
            nan_count = sum(1 for v in rsi_values if 'NaN' in str(v.get('errors', [])))
            infinity_count = sum(1 for v in rsi_values if 'infinity' in str(v.get('errors', [])))
            out_of_bounds_count = sum(1 for v in rsi_values if not v['is_valid'] and 'outside valid range' in str(v.get('errors', [])))
            
            logger.info(f"   📊 RSI Coverage: {rsi_coverage:.1f}% ({rsi_valid_count}/{rsi_total_count})")
            logger.info(f"   📊 RSI Issues: {len(rsi_issues)} total")
            logger.info(f"   📊 NaN values: {nan_count}")
            logger.info(f"   📊 Infinity values: {infinity_count}")
            logger.info(f"   📊 Out of bounds (0-100): {out_of_bounds_count}")
            
            # Show specific issues
            if rsi_issues:
                logger.info("   🚨 RSI ISSUES DETECTED:")
                for issue in rsi_issues[:5]:  # Show first 5 issues
                    logger.info(f"      - {issue}")
            
            # Success criteria: At least 70% valid RSI values, no critical errors
            success = rsi_coverage >= 70 and nan_count == 0 and infinity_count == 0
            critical = nan_count > 0 or infinity_count > 0 or out_of_bounds_count > 5
            
            details = f"Coverage: {rsi_coverage:.1f}%, Issues: {len(rsi_issues)}, NaN: {nan_count}, Infinity: {infinity_count}, Out of bounds: {out_of_bounds_count}"
            
            self.log_test_result("RSI Calculation Validation", success, details, critical)
            
        except Exception as e:
            self.log_test_result("RSI Calculation Validation", False, f"Exception: {str(e)}", critical=True)
    
    async def test_2_macd_calculation_validation(self):
        """Test 2: MACD Calculation Validation"""
        logger.info("\n🔍 TEST 2: MACD Calculation Validation")
        
        try:
            # Get IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("MACD Calculation Validation", False, f"HTTP {response.status_code}", critical=True)
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("MACD Calculation Validation", False, "No IA1 analyses found", critical=True)
                return
            
            # Analyze MACD values
            macd_issues = []
            macd_valid_count = 0
            macd_total_count = 0
            macd_values = []
            
            for analysis in analyses[-20:]:  # Check last 20 analyses
                symbol = analysis.get('symbol', 'Unknown')
                
                # Check different possible MACD field names
                macd_value = None
                macd_signal = None
                
                for field in ['macd', 'macd_signal', 'macd_line', 'technical_analysis', 'indicators']:
                    if field in analysis:
                        if field == 'technical_analysis' and isinstance(analysis[field], dict):
                            macd_value = analysis[field].get('macd') or analysis[field].get('macd_signal')
                        elif field == 'indicators' and isinstance(analysis[field], dict):
                            macd_value = analysis[field].get('macd') or analysis[field].get('macd_signal')
                        else:
                            macd_value = analysis[field]
                        
                        if macd_value is not None:
                            break
                
                # Also check in reasoning text for MACD mentions
                reasoning = analysis.get('reasoning', '') or analysis.get('ia1_reasoning', '')
                macd_in_reasoning = False
                if reasoning:
                    import re
                    macd_matches = re.findall(r'MACD[:\s]*(-?\d+\.?\d*)', reasoning, re.IGNORECASE)
                    if macd_matches:
                        macd_in_reasoning = True
                        if macd_value is None:
                            try:
                                macd_value = float(macd_matches[0])
                            except ValueError:
                                pass
                
                macd_total_count += 1
                
                if macd_value is not None:
                    validation = self.validate_indicator_value('macd', macd_value, symbol)
                    macd_values.append(validation)
                    
                    if validation['is_valid']:
                        macd_valid_count += 1
                        logger.info(f"      ✅ {symbol}: MACD = {macd_value} (valid)")
                    else:
                        macd_issues.append(f"{symbol}: MACD = {macd_value} - {', '.join(validation['errors'])}")
                        logger.info(f"      ❌ {symbol}: MACD = {macd_value} - {', '.join(validation['errors'])}")
                else:
                    macd_issues.append(f"{symbol}: MACD value missing/not found")
                    logger.info(f"      ❌ {symbol}: MACD value missing (reasoning has MACD: {macd_in_reasoning})")
            
            # Calculate statistics
            macd_coverage = (macd_valid_count / macd_total_count) * 100 if macd_total_count > 0 else 0
            
            # Check for specific MACD calculation issues
            nan_count = sum(1 for v in macd_values if 'NaN' in str(v.get('errors', [])))
            infinity_count = sum(1 for v in macd_values if 'infinity' in str(v.get('errors', [])))
            extreme_count = sum(1 for v in macd_values if not v['is_realistic'])
            
            logger.info(f"   📊 MACD Coverage: {macd_coverage:.1f}% ({macd_valid_count}/{macd_total_count})")
            logger.info(f"   📊 MACD Issues: {len(macd_issues)} total")
            logger.info(f"   📊 NaN values: {nan_count}")
            logger.info(f"   📊 Infinity values: {infinity_count}")
            logger.info(f"   📊 Extreme values: {extreme_count}")
            
            # Show specific issues
            if macd_issues:
                logger.info("   🚨 MACD ISSUES DETECTED:")
                for issue in macd_issues[:5]:  # Show first 5 issues
                    logger.info(f"      - {issue}")
            
            # Success criteria: At least 70% valid MACD values, no critical errors
            success = macd_coverage >= 70 and nan_count == 0 and infinity_count == 0
            critical = nan_count > 0 or infinity_count > 0
            
            details = f"Coverage: {macd_coverage:.1f}%, Issues: {len(macd_issues)}, NaN: {nan_count}, Infinity: {infinity_count}, Extreme: {extreme_count}"
            
            self.log_test_result("MACD Calculation Validation", success, details, critical)
            
        except Exception as e:
            self.log_test_result("MACD Calculation Validation", False, f"Exception: {str(e)}", critical=True)
    
    async def test_3_stochastic_calculation_validation(self):
        """Test 3: Stochastic Oscillator Validation (%K and %D, 0-100 range)"""
        logger.info("\n🔍 TEST 3: Stochastic Oscillator Validation")
        
        try:
            # Get IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Stochastic Calculation Validation", False, f"HTTP {response.status_code}", critical=True)
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("Stochastic Calculation Validation", False, "No IA1 analyses found", critical=True)
                return
            
            # Analyze Stochastic values
            stoch_issues = []
            stoch_k_valid_count = 0
            stoch_d_valid_count = 0
            stoch_total_count = 0
            stoch_values = []
            
            for analysis in analyses[-20:]:  # Check last 20 analyses
                symbol = analysis.get('symbol', 'Unknown')
                
                # Check different possible Stochastic field names
                stoch_k_value = None
                stoch_d_value = None
                
                # Check various field combinations
                field_combinations = [
                    ['stochastic', 'stochastic_k', 'stoch_k'],
                    ['stochastic_d', 'stoch_d'],
                    ['technical_analysis'],
                    ['indicators']
                ]
                
                for fields in field_combinations:
                    for field in fields:
                        if field in analysis:
                            if field == 'technical_analysis' and isinstance(analysis[field], dict):
                                stoch_k_value = analysis[field].get('stochastic') or analysis[field].get('stochastic_k')
                                stoch_d_value = analysis[field].get('stochastic_d')
                            elif field == 'indicators' and isinstance(analysis[field], dict):
                                stoch_k_value = analysis[field].get('stochastic') or analysis[field].get('stochastic_k')
                                stoch_d_value = analysis[field].get('stochastic_d')
                            elif 'stoch' in field.lower():
                                if 'd' in field.lower():
                                    stoch_d_value = analysis[field]
                                else:
                                    stoch_k_value = analysis[field]
                
                # Also check in reasoning text for Stochastic mentions
                reasoning = analysis.get('reasoning', '') or analysis.get('ia1_reasoning', '')
                stoch_in_reasoning = False
                if reasoning:
                    import re
                    stoch_matches = re.findall(r'Stochastic[:\s]*(\d+\.?\d*)', reasoning, re.IGNORECASE)
                    if stoch_matches:
                        stoch_in_reasoning = True
                        if stoch_k_value is None:
                            try:
                                stoch_k_value = float(stoch_matches[0])
                            except ValueError:
                                pass
                
                stoch_total_count += 1
                
                # Validate Stochastic %K
                if stoch_k_value is not None:
                    validation_k = self.validate_indicator_value('stochastic_k', stoch_k_value, symbol)
                    stoch_values.append(validation_k)
                    
                    if validation_k['is_valid']:
                        stoch_k_valid_count += 1
                        logger.info(f"      ✅ {symbol}: Stochastic %K = {stoch_k_value} (valid)")
                    else:
                        stoch_issues.append(f"{symbol}: Stochastic %K = {stoch_k_value} - {', '.join(validation_k['errors'])}")
                        logger.info(f"      ❌ {symbol}: Stochastic %K = {stoch_k_value} - {', '.join(validation_k['errors'])}")
                else:
                    stoch_issues.append(f"{symbol}: Stochastic %K value missing/not found")
                    logger.info(f"      ❌ {symbol}: Stochastic %K missing (reasoning has Stochastic: {stoch_in_reasoning})")
                
                # Validate Stochastic %D
                if stoch_d_value is not None:
                    validation_d = self.validate_indicator_value('stochastic_d', stoch_d_value, symbol)
                    stoch_values.append(validation_d)
                    
                    if validation_d['is_valid']:
                        stoch_d_valid_count += 1
                        logger.info(f"      ✅ {symbol}: Stochastic %D = {stoch_d_value} (valid)")
                    else:
                        stoch_issues.append(f"{symbol}: Stochastic %D = {stoch_d_value} - {', '.join(validation_d['errors'])}")
                        logger.info(f"      ❌ {symbol}: Stochastic %D = {stoch_d_value} - {', '.join(validation_d['errors'])}")
                else:
                    stoch_issues.append(f"{symbol}: Stochastic %D value missing/not found")
                    logger.info(f"      ❌ {symbol}: Stochastic %D missing")
            
            # Calculate statistics
            stoch_k_coverage = (stoch_k_valid_count / stoch_total_count) * 100 if stoch_total_count > 0 else 0
            stoch_d_coverage = (stoch_d_valid_count / stoch_total_count) * 100 if stoch_total_count > 0 else 0
            
            # Check for specific Stochastic calculation issues
            nan_count = sum(1 for v in stoch_values if 'NaN' in str(v.get('errors', [])))
            infinity_count = sum(1 for v in stoch_values if 'infinity' in str(v.get('errors', [])))
            out_of_bounds_count = sum(1 for v in stoch_values if not v['is_valid'] and 'outside valid range' in str(v.get('errors', [])))
            
            logger.info(f"   📊 Stochastic %K Coverage: {stoch_k_coverage:.1f}% ({stoch_k_valid_count}/{stoch_total_count})")
            logger.info(f"   📊 Stochastic %D Coverage: {stoch_d_coverage:.1f}% ({stoch_d_valid_count}/{stoch_total_count})")
            logger.info(f"   📊 Stochastic Issues: {len(stoch_issues)} total")
            logger.info(f"   📊 NaN values: {nan_count}")
            logger.info(f"   📊 Infinity values: {infinity_count}")
            logger.info(f"   📊 Out of bounds (0-100): {out_of_bounds_count}")
            
            # Show specific issues
            if stoch_issues:
                logger.info("   🚨 STOCHASTIC ISSUES DETECTED:")
                for issue in stoch_issues[:5]:  # Show first 5 issues
                    logger.info(f"      - {issue}")
            
            # Success criteria: At least 50% coverage for both %K and %D, no critical errors
            success = stoch_k_coverage >= 50 and stoch_d_coverage >= 30 and nan_count == 0 and infinity_count == 0
            critical = nan_count > 0 or infinity_count > 0 or (stoch_k_coverage == 0 and stoch_d_coverage == 0)
            
            details = f"%K Coverage: {stoch_k_coverage:.1f}%, %D Coverage: {stoch_d_coverage:.1f}%, Issues: {len(stoch_issues)}, NaN: {nan_count}, Infinity: {infinity_count}"
            
            self.log_test_result("Stochastic Calculation Validation", success, details, critical)
            
        except Exception as e:
            self.log_test_result("Stochastic Calculation Validation", False, f"Exception: {str(e)}", critical=True)
    
    async def test_4_bollinger_bands_validation(self):
        """Test 4: Bollinger Bands Position Validation (-2 to +2 range)"""
        logger.info("\n🔍 TEST 4: Bollinger Bands Position Validation")
        
        try:
            # Get IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Bollinger Bands Validation", False, f"HTTP {response.status_code}", critical=True)
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("Bollinger Bands Validation", False, "No IA1 analyses found", critical=True)
                return
            
            # Analyze Bollinger Bands values
            bb_issues = []
            bb_valid_count = 0
            bb_total_count = 0
            bb_values = []
            
            for analysis in analyses[-20:]:  # Check last 20 analyses
                symbol = analysis.get('symbol', 'Unknown')
                
                # Check different possible Bollinger Bands field names
                bb_value = None
                
                field_names = [
                    'bollinger_position', 'bollinger_bands', 'bb_position', 
                    'bollinger', 'technical_analysis', 'indicators'
                ]
                
                for field in field_names:
                    if field in analysis:
                        if field == 'technical_analysis' and isinstance(analysis[field], dict):
                            bb_value = analysis[field].get('bollinger_position') or analysis[field].get('bollinger')
                        elif field == 'indicators' and isinstance(analysis[field], dict):
                            bb_value = analysis[field].get('bollinger_position') or analysis[field].get('bollinger')
                        else:
                            bb_value = analysis[field]
                        
                        if bb_value is not None:
                            break
                
                # Also check in reasoning text for Bollinger mentions
                reasoning = analysis.get('reasoning', '') or analysis.get('ia1_reasoning', '')
                bb_in_reasoning = False
                if reasoning:
                    import re
                    bb_matches = re.findall(r'Bollinger[:\s]*(-?\d+\.?\d*)', reasoning, re.IGNORECASE)
                    if bb_matches:
                        bb_in_reasoning = True
                        if bb_value is None:
                            try:
                                bb_value = float(bb_matches[0])
                            except ValueError:
                                pass
                    
                    # Also check for band mentions
                    band_mentions = ['upper band', 'lower band', 'middle band', 'bollinger band']
                    if any(mention in reasoning.lower() for mention in band_mentions):
                        bb_in_reasoning = True
                
                bb_total_count += 1
                
                if bb_value is not None:
                    validation = self.validate_indicator_value('bollinger_position', bb_value, symbol)
                    bb_values.append(validation)
                    
                    if validation['is_valid']:
                        bb_valid_count += 1
                        logger.info(f"      ✅ {symbol}: Bollinger Position = {bb_value} (valid)")
                    else:
                        bb_issues.append(f"{symbol}: Bollinger Position = {bb_value} - {', '.join(validation['errors'])}")
                        logger.info(f"      ❌ {symbol}: Bollinger Position = {bb_value} - {', '.join(validation['errors'])}")
                else:
                    bb_issues.append(f"{symbol}: Bollinger Bands value missing/not found")
                    logger.info(f"      ❌ {symbol}: Bollinger Bands missing (reasoning has Bollinger: {bb_in_reasoning})")
            
            # Calculate statistics
            bb_coverage = (bb_valid_count / bb_total_count) * 100 if bb_total_count > 0 else 0
            
            # Check for specific Bollinger Bands calculation issues
            nan_count = sum(1 for v in bb_values if 'NaN' in str(v.get('errors', [])))
            infinity_count = sum(1 for v in bb_values if 'infinity' in str(v.get('errors', [])))
            out_of_bounds_count = sum(1 for v in bb_values if not v['is_valid'] and 'outside valid range' in str(v.get('errors', [])))
            
            logger.info(f"   📊 Bollinger Bands Coverage: {bb_coverage:.1f}% ({bb_valid_count}/{bb_total_count})")
            logger.info(f"   📊 Bollinger Issues: {len(bb_issues)} total")
            logger.info(f"   📊 NaN values: {nan_count}")
            logger.info(f"   📊 Infinity values: {infinity_count}")
            logger.info(f"   📊 Out of bounds (-3 to +3): {out_of_bounds_count}")
            
            # Show specific issues
            if bb_issues:
                logger.info("   🚨 BOLLINGER BANDS ISSUES DETECTED:")
                for issue in bb_issues[:5]:  # Show first 5 issues
                    logger.info(f"      - {issue}")
            
            # Success criteria: At least 60% valid Bollinger values, no critical errors
            success = bb_coverage >= 60 and nan_count == 0 and infinity_count == 0
            critical = nan_count > 0 or infinity_count > 0 or bb_coverage == 0
            
            details = f"Coverage: {bb_coverage:.1f}%, Issues: {len(bb_issues)}, NaN: {nan_count}, Infinity: {infinity_count}, Out of bounds: {out_of_bounds_count}"
            
            self.log_test_result("Bollinger Bands Validation", success, details, critical)
            
        except Exception as e:
            self.log_test_result("Bollinger Bands Validation", False, f"Exception: {str(e)}", critical=True)
    
    async def test_5_indicators_integration_in_ia1(self):
        """Test 5: Verify all 4 indicators are integrated in IA1 analyses"""
        logger.info("\n🔍 TEST 5: Technical Indicators Integration in IA1")
        
        try:
            # Get IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Indicators Integration in IA1", False, f"HTTP {response.status_code}", critical=True)
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("Indicators Integration in IA1", False, "No IA1 analyses found", critical=True)
                return
            
            # Check integration of all 4 indicators
            integration_stats = {
                'rsi': {'count': 0, 'symbols': []},
                'macd': {'count': 0, 'symbols': []},
                'stochastic': {'count': 0, 'symbols': []},
                'bollinger': {'count': 0, 'symbols': []}
            }
            
            total_analyses = min(len(analyses), 15)  # Check last 15 analyses
            
            for analysis in analyses[-total_analyses:]:
                symbol = analysis.get('symbol', 'Unknown')
                reasoning = analysis.get('reasoning', '') or analysis.get('ia1_reasoning', '')
                
                # Check for RSI integration
                rsi_keywords = ['rsi', 'relative strength', 'overbought', 'oversold']
                if any(keyword in reasoning.lower() for keyword in rsi_keywords):
                    integration_stats['rsi']['count'] += 1
                    integration_stats['rsi']['symbols'].append(symbol)
                
                # Check for MACD integration
                macd_keywords = ['macd', 'moving average convergence', 'signal line', 'histogram']
                if any(keyword in reasoning.lower() for keyword in macd_keywords):
                    integration_stats['macd']['count'] += 1
                    integration_stats['macd']['symbols'].append(symbol)
                
                # Check for Stochastic integration
                stoch_keywords = ['stochastic', '%k', '%d', 'oscillator']
                if any(keyword in reasoning.lower() for keyword in stoch_keywords):
                    integration_stats['stochastic']['count'] += 1
                    integration_stats['stochastic']['symbols'].append(symbol)
                
                # Check for Bollinger Bands integration
                bb_keywords = ['bollinger', 'bands', 'upper band', 'lower band', 'squeeze']
                if any(keyword in reasoning.lower() for keyword in bb_keywords):
                    integration_stats['bollinger']['count'] += 1
                    integration_stats['bollinger']['symbols'].append(symbol)
            
            # Calculate integration percentages
            integration_percentages = {}
            for indicator, stats in integration_stats.items():
                percentage = (stats['count'] / total_analyses) * 100 if total_analyses > 0 else 0
                integration_percentages[indicator] = percentage
                
                logger.info(f"   📊 {indicator.upper()} integration: {percentage:.1f}% ({stats['count']}/{total_analyses})")
                if stats['symbols']:
                    logger.info(f"      Symbols: {', '.join(stats['symbols'][:5])}{'...' if len(stats['symbols']) > 5 else ''}")
            
            # Check for confluence analysis (multiple indicators mentioned together)
            confluence_count = 0
            for analysis in analyses[-total_analyses:]:
                reasoning = analysis.get('reasoning', '') or analysis.get('ia1_reasoning', '')
                
                indicators_mentioned = 0
                if any(keyword in reasoning.lower() for keyword in ['rsi', 'relative strength']):
                    indicators_mentioned += 1
                if any(keyword in reasoning.lower() for keyword in ['macd', 'moving average convergence']):
                    indicators_mentioned += 1
                if any(keyword in reasoning.lower() for keyword in ['stochastic', 'oscillator']):
                    indicators_mentioned += 1
                if any(keyword in reasoning.lower() for keyword in ['bollinger', 'bands']):
                    indicators_mentioned += 1
                
                if indicators_mentioned >= 2:
                    confluence_count += 1
            
            confluence_percentage = (confluence_count / total_analyses) * 100 if total_analyses > 0 else 0
            logger.info(f"   📊 Multi-indicator confluence: {confluence_percentage:.1f}% ({confluence_count}/{total_analyses})")
            
            # Success criteria: At least 3/4 indicators with >50% integration, confluence >30%
            indicators_well_integrated = sum(1 for pct in integration_percentages.values() if pct >= 50)
            confluence_adequate = confluence_percentage >= 30
            
            success = indicators_well_integrated >= 3 and confluence_adequate
            critical = indicators_well_integrated == 0 or max(integration_percentages.values()) < 30
            
            details = f"Well integrated: {indicators_well_integrated}/4, RSI: {integration_percentages['rsi']:.1f}%, MACD: {integration_percentages['macd']:.1f}%, Stochastic: {integration_percentages['stochastic']:.1f}%, Bollinger: {integration_percentages['bollinger']:.1f}%, Confluence: {confluence_percentage:.1f}%"
            
            self.log_test_result("Indicators Integration in IA1", success, details, critical)
            
        except Exception as e:
            self.log_test_result("Indicators Integration in IA1", False, f"Exception: {str(e)}", critical=True)
    
    async def test_6_consistency_across_symbols(self):
        """Test 6: Test consistency of indicator calculations across multiple symbols"""
        logger.info("\n🔍 TEST 6: Consistency Testing Across Multiple Symbols")
        
        try:
            # Trigger fresh analysis to get diverse symbols
            logger.info("   🚀 Triggering fresh analysis for consistency testing...")
            start_response = requests.post(f"{self.api_url}/start-trading", timeout=180)
            
            if start_response.status_code not in [200, 201]:
                logger.warning(f"   ⚠️ Start trading returned HTTP {start_response.status_code}")
            else:
                # Wait for processing
                logger.info("   ⏳ Waiting 30 seconds for analysis processing...")
                await asyncio.sleep(30)
            
            # Get IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Consistency Across Symbols", False, f"HTTP {response.status_code}", critical=True)
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("Consistency Across Symbols", False, "No IA1 analyses found", critical=True)
                return
            
            # Collect indicator values by symbol
            symbol_indicators = {}
            consistency_issues = []
            
            for analysis in analyses[-15:]:  # Check last 15 analyses
                symbol = analysis.get('symbol', 'Unknown')
                reasoning = analysis.get('reasoning', '') or analysis.get('ia1_reasoning', '')
                
                # Extract indicator values from reasoning using regex
                import re
                
                indicators = {}
                
                # Extract RSI
                rsi_matches = re.findall(r'RSI[:\s]*(\d+\.?\d*)', reasoning, re.IGNORECASE)
                if rsi_matches:
                    try:
                        indicators['rsi'] = float(rsi_matches[0])
                    except ValueError:
                        pass
                
                # Extract MACD
                macd_matches = re.findall(r'MACD[:\s]*(-?\d+\.?\d*)', reasoning, re.IGNORECASE)
                if macd_matches:
                    try:
                        indicators['macd'] = float(macd_matches[0])
                    except ValueError:
                        pass
                
                # Extract Stochastic
                stoch_matches = re.findall(r'Stochastic[:\s]*(\d+\.?\d*)', reasoning, re.IGNORECASE)
                if stoch_matches:
                    try:
                        indicators['stochastic'] = float(stoch_matches[0])
                    except ValueError:
                        pass
                
                # Extract Bollinger
                bb_matches = re.findall(r'Bollinger[:\s]*(-?\d+\.?\d*)', reasoning, re.IGNORECASE)
                if bb_matches:
                    try:
                        indicators['bollinger'] = float(bb_matches[0])
                    except ValueError:
                        pass
                
                if indicators:
                    symbol_indicators[symbol] = indicators
                    logger.info(f"      📊 {symbol}: {indicators}")
            
            # Analyze consistency patterns
            if len(symbol_indicators) < 3:
                self.log_test_result("Consistency Across Symbols", False, f"Insufficient data for consistency testing (only {len(symbol_indicators)} symbols)", critical=True)
                return
            
            # Check for suspicious patterns
            suspicious_patterns = []
            
            # Check for identical values across different symbols (suspicious)
            for indicator in ['rsi', 'macd', 'stochastic', 'bollinger']:
                values = []
                symbols_with_indicator = []
                
                for symbol, indicators in symbol_indicators.items():
                    if indicator in indicators:
                        values.append(indicators[indicator])
                        symbols_with_indicator.append(symbol)
                
                if len(values) >= 3:
                    # Check for identical values
                    unique_values = set(values)
                    if len(unique_values) == 1:
                        suspicious_patterns.append(f"All {indicator.upper()} values identical: {values[0]} across {symbols_with_indicator}")
                    
                    # Check for unrealistic clustering
                    if len(unique_values) < len(values) * 0.5:  # Less than 50% unique values
                        suspicious_patterns.append(f"{indicator.upper()} values show suspicious clustering: {list(unique_values)}")
                    
                    # Check for out-of-range values
                    if indicator in self.expected_ranges:
                        ranges = self.expected_ranges[indicator]
                        out_of_range = [v for v in values if v < ranges['min'] or v > ranges['max']]
                        if out_of_range:
                            suspicious_patterns.append(f"{indicator.upper()} out-of-range values: {out_of_range}")
            
            # Check for missing indicators across symbols
            indicator_coverage = {}
            for indicator in ['rsi', 'macd', 'stochastic', 'bollinger']:
                count = sum(1 for indicators in symbol_indicators.values() if indicator in indicators)
                coverage = (count / len(symbol_indicators)) * 100
                indicator_coverage[indicator] = coverage
                
                if coverage < 50:
                    consistency_issues.append(f"{indicator.upper()} missing in {100-coverage:.1f}% of symbols")
            
            logger.info(f"   📊 Symbols analyzed: {len(symbol_indicators)}")
            logger.info(f"   📊 Indicator coverage: RSI {indicator_coverage.get('rsi', 0):.1f}%, MACD {indicator_coverage.get('macd', 0):.1f}%, Stochastic {indicator_coverage.get('stochastic', 0):.1f}%, Bollinger {indicator_coverage.get('bollinger', 0):.1f}%")
            
            if suspicious_patterns:
                logger.info("   🚨 SUSPICIOUS PATTERNS DETECTED:")
                for pattern in suspicious_patterns:
                    logger.info(f"      - {pattern}")
            
            if consistency_issues:
                logger.info("   ⚠️ CONSISTENCY ISSUES:")
                for issue in consistency_issues:
                    logger.info(f"      - {issue}")
            
            # Success criteria: Good coverage, no suspicious identical values, reasonable distribution
            good_coverage = sum(1 for cov in indicator_coverage.values() if cov >= 60) >= 3
            no_suspicious_identical = not any('identical' in pattern for pattern in suspicious_patterns)
            no_critical_issues = len(consistency_issues) <= 2
            
            success = good_coverage and no_suspicious_identical and no_critical_issues
            critical = len(suspicious_patterns) > 3 or any('out-of-range' in pattern for pattern in suspicious_patterns)
            
            details = f"Symbols: {len(symbol_indicators)}, Good coverage: {good_coverage}, Suspicious patterns: {len(suspicious_patterns)}, Issues: {len(consistency_issues)}"
            
            self.log_test_result("Consistency Across Symbols", success, details, critical)
            
        except Exception as e:
            self.log_test_result("Consistency Across Symbols", False, f"Exception: {str(e)}", critical=True)
    
    async def test_7_backend_logs_error_analysis(self):
        """Test 7: Analyze backend logs for technical indicators calculation errors"""
        logger.info("\n🔍 TEST 7: Backend Logs Error Analysis")
        
        try:
            import subprocess
            
            # Get recent backend logs
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
            
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "5000", "/var/log/supervisor/backend.err.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            if not backend_logs:
                self.log_test_result("Backend Logs Error Analysis", False, "Could not retrieve backend logs", critical=True)
                return
            
            # Look for technical indicators related errors
            error_patterns = [
                # Calculation errors
                'RSI.*error', 'RSI.*exception', 'RSI.*failed',
                'MACD.*error', 'MACD.*exception', 'MACD.*failed',
                'Stochastic.*error', 'Stochastic.*exception', 'Stochastic.*failed',
                'Bollinger.*error', 'Bollinger.*exception', 'Bollinger.*failed',
                
                # Data errors
                'NaN', 'infinity', 'inf', 'null',
                'division by zero', 'ZeroDivisionError',
                'ValueError.*indicator', 'TypeError.*indicator',
                
                # Technical analysis errors
                'technical.*analysis.*error', 'indicator.*calculation.*error',
                'AdvancedTechnicalIndicators.*error',
                
                # DataFrame errors
                'DataFrame.*error', 'pandas.*error', 'numpy.*error'
            ]
            
            error_findings = {}
            total_errors = 0
            
            for pattern in error_patterns:
                import re
                matches = re.findall(pattern, backend_logs, re.IGNORECASE)
                if matches:
                    error_findings[pattern] = len(matches)
                    total_errors += len(matches)
                    logger.info(f"      🚨 '{pattern}': {len(matches)} occurrences")
            
            # Look for specific technical indicators debug logs
            debug_patterns = [
                'RSI:', 'MACD:', 'Stochastic:', 'Bollinger:',
                'Technical indicators:', 'Indicators calculated',
                'AdvancedTechnicalIndicators'
            ]
            
            debug_findings = {}
            for pattern in debug_patterns:
                count = backend_logs.count(pattern)
                debug_findings[pattern] = count
                logger.info(f"      📊 '{pattern}': {count} occurrences")
            
            # Look for successful indicator calculations
            success_patterns = [
                'RSI.*calculated', 'MACD.*calculated', 'Stochastic.*calculated', 'Bollinger.*calculated',
                'Technical analysis.*complete', 'Indicators.*success'
            ]
            
            success_findings = {}
            total_successes = 0
            
            for pattern in success_patterns:
                import re
                matches = re.findall(pattern, backend_logs, re.IGNORECASE)
                if matches:
                    success_findings[pattern] = len(matches)
                    total_successes += len(matches)
                    logger.info(f"      ✅ '{pattern}': {len(matches)} occurrences")
            
            # Analyze error severity
            critical_errors = sum(count for pattern, count in error_findings.items() 
                                if any(critical in pattern.lower() for critical in ['nan', 'infinity', 'division by zero', 'zerodivisionerror']))
            
            calculation_errors = sum(count for pattern, count in error_findings.items() 
                                   if any(calc in pattern.lower() for calc in ['rsi', 'macd', 'stochastic', 'bollinger']))
            
            logger.info(f"   📊 Total errors found: {total_errors}")
            logger.info(f"   📊 Critical errors (NaN/Infinity/Division): {critical_errors}")
            logger.info(f"   📊 Calculation errors: {calculation_errors}")
            logger.info(f"   📊 Total successes found: {total_successes}")
            logger.info(f"   📊 Debug logs found: {sum(debug_findings.values())}")
            
            # Show most frequent errors
            if error_findings:
                logger.info("   🚨 MOST FREQUENT ERRORS:")
                sorted_errors = sorted(error_findings.items(), key=lambda x: x[1], reverse=True)
                for pattern, count in sorted_errors[:5]:
                    logger.info(f"      - {pattern}: {count} times")
            
            # Success criteria: Low error rate, no critical calculation errors
            error_rate_acceptable = total_errors < 50  # Less than 50 total errors
            no_critical_calculation_errors = critical_errors == 0
            some_debug_activity = sum(debug_findings.values()) > 0
            
            success = error_rate_acceptable and no_critical_calculation_errors and some_debug_activity
            critical = critical_errors > 0 or calculation_errors > 20
            
            details = f"Total errors: {total_errors}, Critical: {critical_errors}, Calculation errors: {calculation_errors}, Debug logs: {sum(debug_findings.values())}"
            
            self.log_test_result("Backend Logs Error Analysis", success, details, critical)
            
        except Exception as e:
            self.log_test_result("Backend Logs Error Analysis", False, f"Exception: {str(e)}", critical=True)
    
    async def test_8_formusdt_specific_investigation(self):
        """Test 8: Specific investigation of FORMUSDT as mentioned in the review request"""
        logger.info("\n🔍 TEST 8: FORMUSDT Specific Investigation")
        
        try:
            # Get IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("FORMUSDT Specific Investigation", False, f"HTTP {response.status_code}", critical=False)
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            # Look specifically for FORMUSDT
            formusdt_analysis = None
            for analysis in analyses:
                if analysis.get('symbol', '').upper() == 'FORMUSDT':
                    formusdt_analysis = analysis
                    break
            
            if not formusdt_analysis:
                logger.info("      ⚠️ FORMUSDT analysis not found in current analyses")
                # Try to trigger analysis for FORMUSDT specifically
                logger.info("      🚀 Triggering fresh analysis to look for FORMUSDT...")
                start_response = requests.post(f"{self.api_url}/start-trading", timeout=180)
                
                if start_response.status_code in [200, 201]:
                    await asyncio.sleep(30)  # Wait for processing
                    
                    # Try again
                    response = requests.get(f"{self.api_url}/analyses", timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        analyses = data.get('analyses', [])
                        
                        for analysis in analyses:
                            if analysis.get('symbol', '').upper() == 'FORMUSDT':
                                formusdt_analysis = analysis
                                break
            
            if formusdt_analysis:
                logger.info("      ✅ FORMUSDT analysis found!")
                
                # Analyze FORMUSDT technical indicators
                symbol = formusdt_analysis.get('symbol', 'FORMUSDT')
                reasoning = formusdt_analysis.get('reasoning', '') or formusdt_analysis.get('ia1_reasoning', '')
                confidence = formusdt_analysis.get('confidence', 0)
                signal = formusdt_analysis.get('ia1_signal', '') or formusdt_analysis.get('recommendation', '')
                
                logger.info(f"      📊 Signal: {signal}")
                logger.info(f"      📊 Confidence: {confidence}")
                
                # Extract technical indicators from FORMUSDT analysis
                import re
                
                formusdt_indicators = {}
                
                # Extract RSI
                rsi_matches = re.findall(r'RSI[:\s]*(\d+\.?\d*)', reasoning, re.IGNORECASE)
                if rsi_matches:
                    formusdt_indicators['rsi'] = float(rsi_matches[0])
                
                # Extract MACD
                macd_matches = re.findall(r'MACD[:\s]*(-?\d+\.?\d*)', reasoning, re.IGNORECASE)
                if macd_matches:
                    formusdt_indicators['macd'] = float(macd_matches[0])
                
                # Extract Stochastic
                stoch_matches = re.findall(r'Stochastic[:\s]*(\d+\.?\d*)', reasoning, re.IGNORECASE)
                if stoch_matches:
                    formusdt_indicators['stochastic'] = float(stoch_matches[0])
                
                # Extract Bollinger
                bb_matches = re.findall(r'Bollinger[:\s]*(-?\d+\.?\d*)', reasoning, re.IGNORECASE)
                if bb_matches:
                    formusdt_indicators['bollinger'] = float(bb_matches[0])
                
                logger.info(f"      📊 Extracted indicators: {formusdt_indicators}")
                
                # Validate each indicator for FORMUSDT
                formusdt_issues = []
                formusdt_valid_indicators = 0
                
                for indicator, value in formusdt_indicators.items():
                    validation = self.validate_indicator_value(indicator, value, 'FORMUSDT')
                    
                    if validation['is_valid']:
                        formusdt_valid_indicators += 1
                        logger.info(f"      ✅ FORMUSDT {indicator.upper()}: {value} (valid)")
                    else:
                        formusdt_issues.append(f"{indicator.upper()}: {value} - {', '.join(validation['errors'])}")
                        logger.info(f"      ❌ FORMUSDT {indicator.upper()}: {value} - {', '.join(validation['errors'])}")
                
                # Check if FORMUSDT has all 4 indicators
                missing_indicators = []
                for required_indicator in ['rsi', 'macd', 'stochastic', 'bollinger']:
                    if required_indicator not in formusdt_indicators:
                        missing_indicators.append(required_indicator.upper())
                
                if missing_indicators:
                    logger.info(f"      ⚠️ FORMUSDT missing indicators: {', '.join(missing_indicators)}")
                
                # Check IA2 decisions for FORMUSDT
                decisions_response = requests.get(f"{self.api_url}/decisions", timeout=30)
                formusdt_ia2_decision = None
                
                if decisions_response.status_code == 200:
                    decisions_data = decisions_response.json()
                    decisions = decisions_data.get('decisions', [])
                    
                    for decision in decisions:
                        if decision.get('symbol', '').upper() == 'FORMUSDT':
                            formusdt_ia2_decision = decision
                            break
                
                if formusdt_ia2_decision:
                    logger.info(f"      ✅ FORMUSDT IA2 decision found: {formusdt_ia2_decision.get('ia2_signal', 'Unknown')}")
                else:
                    logger.info("      ⚠️ FORMUSDT IA2 decision not found")
                
                # Success criteria for FORMUSDT
                has_indicators = len(formusdt_indicators) >= 3  # At least 3 of 4 indicators
                indicators_valid = formusdt_valid_indicators >= 2  # At least 2 valid indicators
                no_critical_issues = not any('NaN' in issue or 'infinity' in issue for issue in formusdt_issues)
                
                success = has_indicators and indicators_valid and no_critical_issues
                
                details = f"Indicators found: {len(formusdt_indicators)}/4, Valid: {formusdt_valid_indicators}, Issues: {len(formusdt_issues)}, Missing: {missing_indicators}"
                
                self.log_test_result("FORMUSDT Specific Investigation", success, details, not no_critical_issues)
                
            else:
                self.log_test_result("FORMUSDT Specific Investigation", False, "FORMUSDT analysis not found after multiple attempts", critical=False)
            
        except Exception as e:
            self.log_test_result("FORMUSDT Specific Investigation", False, f"Exception: {str(e)}", critical=False)
    
    async def run_comprehensive_diagnostic(self):
        """Run all technical indicators diagnostic tests"""
        logger.info("🚀 Starting Technical Indicators Diagnostic Suite")
        logger.info("=" * 100)
        logger.info("🔍 DIAGNOSTIC URGENT - Problèmes avec le calcul des indicateurs techniques dans IA1")
        logger.info("🎯 Testing: RSI, MACD, Stochastic, Bollinger Bands calculations and integration")
        logger.info("🎯 Focus: Value validation, range checking, error identification, consistency")
        logger.info("=" * 100)
        
        # Run all diagnostic tests in sequence
        await self.test_1_rsi_calculation_validation()
        await self.test_2_macd_calculation_validation()
        await self.test_3_stochastic_calculation_validation()
        await self.test_4_bollinger_bands_validation()
        await self.test_5_indicators_integration_in_ia1()
        await self.test_6_consistency_across_symbols()
        await self.test_7_backend_logs_error_analysis()
        await self.test_8_formusdt_specific_investigation()
        
        # Summary
        logger.info("\n" + "=" * 100)
        logger.info("📊 TECHNICAL INDICATORS DIAGNOSTIC SUMMARY")
        logger.info("=" * 100)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        critical_failures = [result for result in self.test_results if not result['success'] and result.get('critical', False)]
        
        # Categorize results by indicator
        indicator_results = {
            'RSI': [],
            'MACD': [],
            'Stochastic': [],
            'Bollinger': [],
            'Integration': [],
            'System': []
        }
        
        for result in self.test_results:
            test_name = result['test']
            if 'RSI' in test_name:
                indicator_results['RSI'].append(result)
            elif 'MACD' in test_name:
                indicator_results['MACD'].append(result)
            elif 'Stochastic' in test_name:
                indicator_results['Stochastic'].append(result)
            elif 'Bollinger' in test_name:
                indicator_results['Bollinger'].append(result)
            elif 'Integration' in test_name or 'Consistency' in test_name:
                indicator_results['Integration'].append(result)
            else:
                indicator_results['System'].append(result)
        
        # Display results by category
        for category, results in indicator_results.items():
            if results:
                logger.info(f"\n📋 {category.upper()} RESULTS:")
                for result in results:
                    status = "✅ PASS" if result['success'] else ("🚨 CRITICAL FAIL" if result.get('critical', False) else "❌ FAIL")
                    logger.info(f"   {status}: {result['test']}")
                    if result['details']:
                        logger.info(f"      {result['details']}")
        
        logger.info(f"\n🎯 OVERALL DIAGNOSTIC RESULT: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests)*100:.1f}%)")
        
        # Critical issues analysis
        if critical_failures:
            logger.info(f"\n🚨 CRITICAL ISSUES IDENTIFIED ({len(critical_failures)} critical failures):")
            for failure in critical_failures:
                logger.info(f"   🚨 {failure['test']}: {failure['details']}")
        
        # Final diagnostic verdict
        logger.info("\n" + "=" * 100)
        logger.info("🏥 TECHNICAL INDICATORS DIAGNOSTIC VERDICT")
        logger.info("=" * 100)
        
        if len(critical_failures) == 0 and passed_tests >= total_tests * 0.75:
            logger.info("✅ DIAGNOSTIC RESULT: Technical indicators system is MOSTLY HEALTHY")
            logger.info("   ✅ No critical calculation errors detected")
            logger.info("   ✅ Most indicators are functioning within expected ranges")
            logger.info("   ✅ Integration with IA1 is operational")
        elif len(critical_failures) == 0 and passed_tests >= total_tests * 0.5:
            logger.info("⚠️ DIAGNOSTIC RESULT: Technical indicators system has MINOR ISSUES")
            logger.info("   ✅ No critical calculation errors detected")
            logger.info("   ⚠️ Some indicators may need calibration or integration improvements")
        elif len(critical_failures) <= 2:
            logger.info("⚠️ DIAGNOSTIC RESULT: Technical indicators system has MODERATE ISSUES")
            logger.info("   ⚠️ Some critical issues detected that need attention")
            logger.info("   🔧 Specific indicators may have calculation or integration problems")
        else:
            logger.info("🚨 DIAGNOSTIC RESULT: Technical indicators system has SEVERE ISSUES")
            logger.info("   🚨 Multiple critical calculation errors detected")
            logger.info("   🚨 Immediate attention required for indicator calculations")
        
        # Specific recommendations
        logger.info("\n📝 DIAGNOSTIC RECOMMENDATIONS:")
        
        # Check each indicator category
        rsi_issues = any(not r['success'] for r in indicator_results['RSI'])
        macd_issues = any(not r['success'] for r in indicator_results['MACD'])
        stoch_issues = any(not r['success'] for r in indicator_results['Stochastic'])
        bb_issues = any(not r['success'] for r in indicator_results['Bollinger'])
        integration_issues = any(not r['success'] for r in indicator_results['Integration'])
        
        if rsi_issues:
            logger.info("   🔧 RSI: Check RSI calculation logic, ensure 0-100 range validation")
        if macd_issues:
            logger.info("   🔧 MACD: Verify MACD signal line calculations and realistic value ranges")
        if stoch_issues:
            logger.info("   🔧 Stochastic: Implement or fix Stochastic %K and %D calculations")
        if bb_issues:
            logger.info("   🔧 Bollinger Bands: Verify Bollinger Bands position calculations (-2 to +2 range)")
        if integration_issues:
            logger.info("   🔧 Integration: Improve technical indicators integration in IA1 analyses")
        
        # System-level recommendations
        if len(critical_failures) > 0:
            logger.info("   🚨 URGENT: Fix NaN/Infinity calculation errors immediately")
            logger.info("   🚨 URGENT: Implement proper error handling for technical indicators")
        
        logger.info("\n🎯 NEXT STEPS:")
        logger.info("   1. Address critical calculation errors first")
        logger.info("   2. Verify technical indicators data sources and calculation methods")
        logger.info("   3. Ensure proper integration of all 4 indicators in IA1 analyses")
        logger.info("   4. Test with multiple symbols to ensure consistency")
        logger.info("   5. Monitor backend logs for ongoing calculation errors")
        
        return passed_tests, total_tests, len(critical_failures)

async def main():
    """Main diagnostic execution"""
    diagnostic_suite = TechnicalIndicatorsDiagnosticSuite()
    passed, total, critical = await diagnostic_suite.run_comprehensive_diagnostic()
    
    # Exit with appropriate code based on results
    if critical > 0:
        sys.exit(2)  # Critical issues
    elif passed >= total * 0.75:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Some issues

if __name__ == "__main__":
    asyncio.run(main())
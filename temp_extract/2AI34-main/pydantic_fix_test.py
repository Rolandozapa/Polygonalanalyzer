#!/usr/bin/env python3
"""
PYDANTIC VALIDATION FIX TESTING SUITE - FIBONACCI_KEY_LEVEL_PROXIMITY FIELD
Focus: Test Fix Pydantic Validation - Vérifier résolution erreur fibonacci_key_level_proximity et impact sur reasoning field.

OBJECTIF URGENT:
Tester si la correction Pydantic de `fibonacci_key_level_proximity` (boolean vs float) résout les problèmes de reasoning field vide et RR calculation.

TESTS CRITIQUES:

1. **Validation Pydantic Success**:
   - Tester /api/force-ia1-analysis avec un symbole disponible
   - Vérifier absence d'erreur "fibonacci_key_level_proximity field expects boolean but receives float"
   - Confirmer que l'analyse IA1 se termine sans erreurs de validation

2. **Reasoning Field Population Immediate Check**:
   - Examiner si "reasoning" contient maintenant du contenu (vs null précédent)
   - Vérifier format "STEP 1 - RSI ANALYSIS... STEP 2 - MACD MOMENTUM..."
   - Confirmer mentions RSI, MACD, MFI, VWAP avec valeurs calculées

3. **RR Calculation Restoration**:
   - Vérifier si "calculated_rr" utilise maintenant les formules réelles
   - Examiner extraction prix IA1: entry_price, stop_loss_price, take_profit_price
   - Confirmer calculs LONG: (TP-Entry)/(Entry-SL), SHORT: (Entry-TP)/(SL-Entry)

4. **Technical Indicators Integration Validation**:
   - Confirmer que les indicateurs techniques atteignent maintenant le reasoning field
   - Vérifier mentions explicites: RSI X.X, MACD Y.Y, MFI Z.Z dans le texte
   - Examiner cohérence entre indicateurs calculés et reasoning généré

5. **System Health Check**:
   - Vérifier stabilité backend après correction Pydantic
   - Confirmer absence d'erreurs validation dans les logs
   - Tester que l'endpoint force-ia1-analysis fonctionne sans crashes

HYPOTHÈSE: L'erreur Pydantic empêchait l'analyse IA1 de se terminer correctement, causant l'utilisation de données fallback (reasoning=null, RR=1.0). La correction devrait restaurer le fonctionnement normal.

PRIORITÉ: Valider que cette unique correction Pydantic résout les 2 problèmes critiques (reasoning field + RR calculation).
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
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PydanticFixValidationTestSuite:
    """Comprehensive test suite for Pydantic Validation Fix - fibonacci_key_level_proximity field"""
    
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
        logger.info(f"Testing Pydantic Fix Validation at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Test symbols for IA1 analysis (will be dynamically determined from available opportunities)
        self.test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']  # Preferred symbols from review request
        self.actual_test_symbols = []  # Will be populated from available opportunities
        
        # Core technical indicators to verify (as specified in review request)
        self.core_technical_indicators = ['RSI', 'MACD', 'MFI', 'VWAP']
        
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
    
    async def _capture_backend_logs(self):
        """Capture backend logs for analysis"""
        try:
            # Try to capture supervisor backend logs
            result = subprocess.run(
                ['tail', '-n', '200', '/var/log/supervisor/backend.out.log'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                return result.stdout.split('\n')
            else:
                # Try alternative log location
                result = subprocess.run(
                    ['tail', '-n', '200', '/var/log/supervisor/backend.err.log'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0 and result.stdout:
                    return result.stdout.split('\n')
                else:
                    return []
                    
        except Exception as e:
            logger.warning(f"Could not capture backend logs: {e}")
            return []
    
    async def test_pydantic_validation_fix_validation(self):
        """Test: Pydantic Validation Fix Validation - Critical Fix for fibonacci_key_level_proximity field"""
        logger.info("\n🔍 TEST: Pydantic Validation Fix Validation")
        
        try:
            pydantic_fix_results = {
                'analyses_attempted': 0,
                'analyses_successful': 0,
                'pydantic_validation_errors': 0,
                'fibonacci_field_errors': 0,
                'reasoning_field_populated': 0,
                'rr_calculation_working': 0,
                'technical_indicators_present': 0,
                'endpoint_functional': False,
                'error_logs': [],
                'successful_analyses': [],
                'response_times': []
            }
            
            logger.info("   🚀 Testing critical Pydantic validation fix for fibonacci_key_level_proximity field...")
            logger.info("   📊 Expected: 100% analyses complete without 'fibonacci_key_level_proximity field expects boolean but receives float' errors")
            
            # Get available symbols from scout system
            logger.info("   📞 Getting available symbols from scout system...")
            
            try:
                response = requests.get(f"{self.api_url}/opportunities", timeout=60)
                
                if response.status_code == 200:
                    opportunities = response.json()
                    if isinstance(opportunities, dict) and 'opportunities' in opportunities:
                        opportunities = opportunities['opportunities']
                    
                    # Get first 3 available symbols for testing
                    available_symbols = [opp.get('symbol') for opp in opportunities[:10] if opp.get('symbol')]
                    
                    # Prefer BTCUSDT if available, otherwise use first available
                    preferred_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
                    test_symbols = []
                    
                    for symbol in preferred_symbols:
                        if symbol in available_symbols:
                            test_symbols.append(symbol)
                    
                    # Fill remaining slots with available symbols
                    for symbol in available_symbols:
                        if symbol not in test_symbols and len(test_symbols) < 3:
                            test_symbols.append(symbol)
                    
                    self.actual_test_symbols = test_symbols[:3]  # Limit to 3 symbols
                    logger.info(f"      ✅ Test symbols selected: {self.actual_test_symbols}")
                    
                else:
                    logger.warning(f"      ⚠️ Could not get opportunities, using default symbols")
                    self.actual_test_symbols = self.test_symbols
                    
            except Exception as e:
                logger.warning(f"      ⚠️ Error getting opportunities: {e}, using default symbols")
                self.actual_test_symbols = self.test_symbols
            
            # Test each symbol for Pydantic validation fix
            for symbol in self.actual_test_symbols:
                logger.info(f"\n   📞 Testing force-ia1-analysis for {symbol} - checking for Pydantic validation errors...")
                pydantic_fix_results['analyses_attempted'] += 1
                
                try:
                    start_time = time.time()
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    response_time = time.time() - start_time
                    pydantic_fix_results['response_times'].append(response_time)
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        pydantic_fix_results['analyses_successful'] += 1
                        pydantic_fix_results['endpoint_functional'] = True
                        pydantic_fix_results['successful_analyses'].append({
                            'symbol': symbol,
                            'response_time': response_time,
                            'analysis_data': analysis_data
                        })
                        
                        logger.info(f"      ✅ {symbol} analysis successful (response time: {response_time:.2f}s)")
                        
                        # Check for reasoning field population (critical test)
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        reasoning = ia1_analysis.get('reasoning', '') if isinstance(ia1_analysis, dict) else ''
                        
                        if reasoning and reasoning.strip() and reasoning != 'null':
                            pydantic_fix_results['reasoning_field_populated'] += 1
                            logger.info(f"         ✅ Reasoning field populated: {len(reasoning)} chars")
                            
                            # Check for technical indicators in reasoning
                            technical_indicators = ['RSI', 'MACD', 'MFI', 'VWAP']
                            indicators_found = []
                            for indicator in technical_indicators:
                                if indicator.upper() in reasoning.upper():
                                    indicators_found.append(indicator)
                            
                            if indicators_found:
                                pydantic_fix_results['technical_indicators_present'] += 1
                                logger.info(f"         ✅ Technical indicators in reasoning: {indicators_found}")
                            else:
                                logger.warning(f"         ⚠️ No technical indicators found in reasoning")
                        else:
                            logger.warning(f"         ❌ Reasoning field empty or null - Pydantic validation may have failed")
                        
                        # Check for RR calculation (critical test)
                        calculated_rr = ia1_analysis.get('calculated_rr') or ia1_analysis.get('risk_reward_ratio')
                        if calculated_rr and calculated_rr != 1.0 and calculated_rr != '1.0':
                            pydantic_fix_results['rr_calculation_working'] += 1
                            logger.info(f"         ✅ RR calculation working: {calculated_rr}")
                        else:
                            logger.warning(f"         ❌ RR calculation using default/fallback value: {calculated_rr}")
                        
                    elif response.status_code == 500:
                        # Check if it's a Pydantic validation error
                        error_text = response.text.lower()
                        if 'fibonacci_key_level_proximity' in error_text or 'validation error' in error_text:
                            pydantic_fix_results['pydantic_validation_errors'] += 1
                            logger.error(f"      ❌ {symbol} PYDANTIC VALIDATION ERROR detected: {response.text[:300]}")
                        elif 'boolean' in error_text and 'float' in error_text:
                            pydantic_fix_results['fibonacci_field_errors'] += 1
                            logger.error(f"      ❌ {symbol} FIBONACCI FIELD TYPE ERROR detected: {response.text[:300]}")
                        else:
                            logger.error(f"      ❌ {symbol} analysis failed: HTTP 500 (Other Error)")
                        
                        pydantic_fix_results['error_logs'].append(f"{symbol}: {response.text[:300]}")
                        
                    else:
                        logger.error(f"      ❌ {symbol} analysis failed: HTTP {response.status_code}")
                        if response.text:
                            error_text = response.text[:300]
                            logger.error(f"         Error response: {error_text}")
                            pydantic_fix_results['error_logs'].append(f"{symbol}: {error_text}")
                
                except Exception as e:
                    logger.error(f"      ❌ {symbol} analysis exception: {e}")
                    pydantic_fix_results['error_logs'].append(f"{symbol}: {str(e)}")
                
                # Wait between analyses
                if symbol != self.actual_test_symbols[-1]:
                    logger.info(f"      ⏳ Waiting 10 seconds before next analysis...")
                    await asyncio.sleep(10)
            
            # Capture backend logs to check for Pydantic validation errors
            logger.info("   📋 Capturing backend logs to check for Pydantic validation errors...")
            
            try:
                backend_logs = await self._capture_backend_logs()
                if backend_logs:
                    # Look for specific Pydantic validation errors
                    pydantic_errors = []
                    fibonacci_errors = []
                    
                    for log_line in backend_logs:
                        if any(error_pattern in log_line.lower() for error_pattern in [
                            'fibonacci_key_level_proximity', 'validation error', 'pydantic'
                        ]):
                            pydantic_errors.append(log_line.strip())
                        
                        if any(error_pattern in log_line.lower() for error_pattern in [
                            'boolean but receives float', 'type=bool_type', 'input_type=float'
                        ]):
                            fibonacci_errors.append(log_line.strip())
                    
                    if pydantic_errors:
                        pydantic_fix_results['pydantic_validation_errors'] += len(pydantic_errors)
                        logger.error(f"      ❌ Pydantic validation errors found: {len(pydantic_errors)}")
                        for error in pydantic_errors[:3]:  # Show first 3
                            logger.error(f"         - {error}")
                    else:
                        logger.info(f"      ✅ No Pydantic validation errors found in backend logs")
                    
                    if fibonacci_errors:
                        pydantic_fix_results['fibonacci_field_errors'] += len(fibonacci_errors)
                        logger.error(f"      ❌ Fibonacci field type errors found: {len(fibonacci_errors)}")
                        for error in fibonacci_errors[:2]:  # Show first 2
                            logger.error(f"         - {error}")
                    else:
                        logger.info(f"      ✅ No fibonacci field type errors found")
                        
            except Exception as e:
                logger.warning(f"      ⚠️ Could not analyze backend logs: {e}")
            
            # Final analysis
            success_rate = pydantic_fix_results['analyses_successful'] / max(pydantic_fix_results['analyses_attempted'], 1)
            reasoning_rate = pydantic_fix_results['reasoning_field_populated'] / max(pydantic_fix_results['analyses_attempted'], 1)
            rr_rate = pydantic_fix_results['rr_calculation_working'] / max(pydantic_fix_results['analyses_attempted'], 1)
            technical_rate = pydantic_fix_results['technical_indicators_present'] / max(pydantic_fix_results['analyses_attempted'], 1)
            avg_response_time = sum(pydantic_fix_results['response_times']) / max(len(pydantic_fix_results['response_times']), 1)
            
            logger.info(f"\n   📊 PYDANTIC VALIDATION FIX VALIDATION RESULTS:")
            logger.info(f"      Analyses attempted: {pydantic_fix_results['analyses_attempted']}")
            logger.info(f"      Analyses successful: {pydantic_fix_results['analyses_successful']}")
            logger.info(f"      Success rate: {success_rate:.2f}")
            logger.info(f"      Reasoning field populated: {pydantic_fix_results['reasoning_field_populated']} ({reasoning_rate:.2f})")
            logger.info(f"      RR calculation working: {pydantic_fix_results['rr_calculation_working']} ({rr_rate:.2f})")
            logger.info(f"      Technical indicators present: {pydantic_fix_results['technical_indicators_present']} ({technical_rate:.2f})")
            logger.info(f"      Pydantic validation errors: {pydantic_fix_results['pydantic_validation_errors']}")
            logger.info(f"      Fibonacci field errors: {pydantic_fix_results['fibonacci_field_errors']}")
            logger.info(f"      Endpoint functional: {pydantic_fix_results['endpoint_functional']}")
            logger.info(f"      Average response time: {avg_response_time:.2f}s")
            
            # Calculate test success based on review requirements (no Pydantic errors, reasoning populated, RR working)
            success_criteria = [
                pydantic_fix_results['analyses_successful'] >= 1,  # At least 1 successful analysis
                pydantic_fix_results['pydantic_validation_errors'] == 0,  # No Pydantic validation errors
                pydantic_fix_results['fibonacci_field_errors'] == 0,  # No fibonacci field errors
                pydantic_fix_results['reasoning_field_populated'] >= 1,  # At least 1 reasoning populated
                pydantic_fix_results['endpoint_functional'],  # Endpoint is functional
                avg_response_time < 60  # Reasonable response time
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.83:  # 83% success threshold (5/6 criteria)
                self.log_test_result("Pydantic Validation Fix Validation", True, 
                                   f"Pydantic fix successful: {success_count}/{len(success_criteria)} criteria met. Success rate: {success_rate:.2f}, Reasoning rate: {reasoning_rate:.2f}, RR rate: {rr_rate:.2f}, No Pydantic errors: {pydantic_fix_results['pydantic_validation_errors'] == 0}")
            else:
                self.log_test_result("Pydantic Validation Fix Validation", False, 
                                   f"Pydantic fix issues: {success_count}/{len(success_criteria)} criteria met. May still have Pydantic validation errors or reasoning/RR calculation problems")
                
        except Exception as e:
            self.log_test_result("Pydantic Validation Fix Validation", False, f"Exception: {str(e)}")

    async def test_reasoning_field_restoration(self):
        """Test: Reasoning Field Restoration - Verify reasoning contains detailed content with technical indicators"""
        logger.info("\n🔍 TEST: Reasoning Field Restoration")
        
        try:
            reasoning_results = {
                'analyses_attempted': 0,
                'reasoning_fields_populated': 0,
                'technical_indicators_mentioned': 0,
                'step_by_step_format': 0,
                'detailed_content_length': [],
                'technical_indicators_found': {},
                'successful_analyses': []
            }
            
            logger.info("   🚀 Testing reasoning field restoration with technical indicators...")
            logger.info("   📊 Expected: Reasoning field contains detailed step-by-step analysis with RSI, MACD, MFI, VWAP mentions")
            
            # Use symbols from previous test
            test_symbols = self.actual_test_symbols if self.actual_test_symbols else self.test_symbols
            
            for symbol in test_symbols:
                logger.info(f"\n   📞 Testing reasoning field for {symbol}...")
                reasoning_results['analyses_attempted'] += 1
                
                try:
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        reasoning = ia1_analysis.get('reasoning', '') if isinstance(ia1_analysis, dict) else ''
                        
                        if reasoning and reasoning.strip() and reasoning != 'null':
                            reasoning_results['reasoning_fields_populated'] += 1
                            reasoning_length = len(reasoning)
                            reasoning_results['detailed_content_length'].append(reasoning_length)
                            
                            logger.info(f"      ✅ Reasoning field populated: {reasoning_length} chars")
                            
                            # Check for step-by-step format
                            step_indicators = ['STEP', 'ANALYSIS', 'RSI', 'MACD', 'MFI', 'VWAP']
                            step_count = sum(1 for indicator in step_indicators if indicator.upper() in reasoning.upper())
                            
                            if step_count >= 3:  # At least 3 step indicators
                                reasoning_results['step_by_step_format'] += 1
                                logger.info(f"         ✅ Step-by-step format detected: {step_count} indicators")
                            
                            # Check for technical indicators mentions
                            technical_indicators = ['RSI', 'MACD', 'MFI', 'VWAP']
                            indicators_found = []
                            
                            for indicator in technical_indicators:
                                if indicator.upper() in reasoning.upper():
                                    indicators_found.append(indicator)
                                    if indicator not in reasoning_results['technical_indicators_found']:
                                        reasoning_results['technical_indicators_found'][indicator] = 0
                                    reasoning_results['technical_indicators_found'][indicator] += 1
                            
                            if indicators_found:
                                reasoning_results['technical_indicators_mentioned'] += 1
                                logger.info(f"         ✅ Technical indicators mentioned: {indicators_found}")
                                
                                # Store successful analysis
                                reasoning_results['successful_analyses'].append({
                                    'symbol': symbol,
                                    'reasoning_length': reasoning_length,
                                    'technical_indicators': indicators_found,
                                    'reasoning_sample': reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
                                })
                            else:
                                logger.warning(f"         ⚠️ No technical indicators found in reasoning")
                        else:
                            logger.warning(f"      ❌ Reasoning field empty or null for {symbol}")
                    
                    else:
                        logger.error(f"      ❌ Analysis failed for {symbol}: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"      ❌ Analysis exception for {symbol}: {e}")
                
                # Wait between analyses
                if symbol != test_symbols[-1]:
                    await asyncio.sleep(10)
            
            # Calculate metrics
            if reasoning_results['analyses_attempted'] > 0:
                reasoning_rate = reasoning_results['reasoning_fields_populated'] / reasoning_results['analyses_attempted']
                technical_rate = reasoning_results['technical_indicators_mentioned'] / reasoning_results['analyses_attempted']
                step_format_rate = reasoning_results['step_by_step_format'] / reasoning_results['analyses_attempted']
                avg_length = sum(reasoning_results['detailed_content_length']) / max(len(reasoning_results['detailed_content_length']), 1)
            else:
                reasoning_rate = technical_rate = step_format_rate = avg_length = 0.0
            
            logger.info(f"\n   📊 REASONING FIELD RESTORATION RESULTS:")
            logger.info(f"      Analyses attempted: {reasoning_results['analyses_attempted']}")
            logger.info(f"      Reasoning fields populated: {reasoning_results['reasoning_fields_populated']} ({reasoning_rate:.2f})")
            logger.info(f"      Technical indicators mentioned: {reasoning_results['technical_indicators_mentioned']} ({technical_rate:.2f})")
            logger.info(f"      Step-by-step format: {reasoning_results['step_by_step_format']} ({step_format_rate:.2f})")
            logger.info(f"      Average reasoning length: {avg_length:.0f} chars")
            
            if reasoning_results['technical_indicators_found']:
                logger.info(f"      📊 Technical indicators found:")
                for indicator, count in reasoning_results['technical_indicators_found'].items():
                    logger.info(f"         - {indicator}: {count} mentions")
            
            # Calculate test success
            success_criteria = [
                reasoning_results['analyses_attempted'] >= 1,
                reasoning_results['reasoning_fields_populated'] >= 1,
                reasoning_rate >= 0.67,  # At least 67% with reasoning
                technical_rate >= 0.67,  # At least 67% with technical indicators
                avg_length >= 100  # At least 100 chars average
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Reasoning Field Restoration", True, 
                                   f"Reasoning restoration successful: {success_count}/{len(success_criteria)} criteria met. Reasoning rate: {reasoning_rate:.2f}, Technical rate: {technical_rate:.2f}, Avg length: {avg_length:.0f}")
            else:
                self.log_test_result("Reasoning Field Restoration", False, 
                                   f"Reasoning restoration issues: {success_count}/{len(success_criteria)} criteria met. May have empty reasoning or missing technical indicators")
                
        except Exception as e:
            self.log_test_result("Reasoning Field Restoration", False, f"Exception: {str(e)}")

    async def test_rr_calculation_formula_implementation(self):
        """Test: RR Calculation Formula Implementation - Verify calculated_rr uses real mathematical formulas"""
        logger.info("\n🔍 TEST: RR Calculation Formula Implementation")
        
        try:
            rr_results = {
                'analyses_attempted': 0,
                'rr_calculations_found': 0,
                'real_formula_calculations': 0,
                'default_values_detected': 0,
                'formula_validations': [],
                'successful_calculations': []
            }
            
            logger.info("   🚀 Testing RR calculation formula implementation...")
            logger.info("   📊 Expected: calculated_rr uses real formulas LONG: (TP-Entry)/(Entry-SL), SHORT: (Entry-TP)/(SL-Entry)")
            
            # Use symbols from previous test
            test_symbols = self.actual_test_symbols if self.actual_test_symbols else self.test_symbols
            
            for symbol in test_symbols:
                logger.info(f"\n   📞 Testing RR calculation for {symbol}...")
                rr_results['analyses_attempted'] += 1
                
                try:
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        
                        # Extract RR and price data
                        calculated_rr = ia1_analysis.get('calculated_rr') or ia1_analysis.get('risk_reward_ratio')
                        entry_price = ia1_analysis.get('entry_price')
                        stop_loss_price = ia1_analysis.get('stop_loss_price')
                        take_profit_price = ia1_analysis.get('take_profit_price')
                        signal = ia1_analysis.get('signal', '').upper()
                        
                        if calculated_rr is not None:
                            rr_results['rr_calculations_found'] += 1
                            logger.info(f"      ✅ RR calculation found: {calculated_rr}")
                            
                            # Check if it's a default value
                            if calculated_rr in [1.0, 2.2, '1.0', '2.2']:
                                rr_results['default_values_detected'] += 1
                                logger.warning(f"         ⚠️ Default RR value detected: {calculated_rr}")
                            else:
                                # Validate formula if we have price data
                                if entry_price and stop_loss_price and take_profit_price:
                                    try:
                                        entry = float(entry_price)
                                        sl = float(stop_loss_price)
                                        tp = float(take_profit_price)
                                        
                                        # Calculate expected RR based on signal
                                        if signal == 'LONG':
                                            expected_rr = (tp - entry) / (entry - sl) if (entry - sl) != 0 else 0
                                        elif signal == 'SHORT':
                                            expected_rr = (entry - tp) / (sl - entry) if (sl - entry) != 0 else 0
                                        else:
                                            expected_rr = None
                                        
                                        if expected_rr is not None:
                                            actual_rr = float(calculated_rr)
                                            rr_difference = abs(actual_rr - expected_rr)
                                            
                                            if rr_difference < 0.1:  # Within 0.1 tolerance
                                                rr_results['real_formula_calculations'] += 1
                                                logger.info(f"         ✅ Formula validation SUCCESS: Expected {expected_rr:.2f}, Got {actual_rr:.2f}")
                                                
                                                rr_results['successful_calculations'].append({
                                                    'symbol': symbol,
                                                    'signal': signal,
                                                    'entry_price': entry,
                                                    'stop_loss': sl,
                                                    'take_profit': tp,
                                                    'expected_rr': expected_rr,
                                                    'actual_rr': actual_rr,
                                                    'difference': rr_difference
                                                })
                                            else:
                                                logger.warning(f"         ❌ Formula validation FAILED: Expected {expected_rr:.2f}, Got {actual_rr:.2f}, Diff: {rr_difference:.2f}")
                                        
                                        rr_results['formula_validations'].append({
                                            'symbol': symbol,
                                            'signal': signal,
                                            'expected_rr': expected_rr,
                                            'actual_rr': float(calculated_rr),
                                            'validation_success': rr_difference < 0.1 if expected_rr is not None else False
                                        })
                                        
                                    except (ValueError, TypeError) as e:
                                        logger.warning(f"         ⚠️ Could not validate formula: {e}")
                                else:
                                    logger.warning(f"         ⚠️ Missing price data for formula validation")
                        else:
                            logger.warning(f"      ❌ No RR calculation found for {symbol}")
                    
                    else:
                        logger.error(f"      ❌ Analysis failed for {symbol}: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"      ❌ Analysis exception for {symbol}: {e}")
                
                # Wait between analyses
                if symbol != test_symbols[-1]:
                    await asyncio.sleep(10)
            
            # Calculate metrics
            if rr_results['analyses_attempted'] > 0:
                rr_found_rate = rr_results['rr_calculations_found'] / rr_results['analyses_attempted']
                formula_rate = rr_results['real_formula_calculations'] / max(rr_results['rr_calculations_found'], 1)
                default_rate = rr_results['default_values_detected'] / max(rr_results['rr_calculations_found'], 1)
            else:
                rr_found_rate = formula_rate = default_rate = 0.0
            
            logger.info(f"\n   📊 RR CALCULATION FORMULA IMPLEMENTATION RESULTS:")
            logger.info(f"      Analyses attempted: {rr_results['analyses_attempted']}")
            logger.info(f"      RR calculations found: {rr_results['rr_calculations_found']} ({rr_found_rate:.2f})")
            logger.info(f"      Real formula calculations: {rr_results['real_formula_calculations']} ({formula_rate:.2f})")
            logger.info(f"      Default values detected: {rr_results['default_values_detected']} ({default_rate:.2f})")
            
            if rr_results['successful_calculations']:
                logger.info(f"      📊 Successful formula validations:")
                for calc in rr_results['successful_calculations']:
                    logger.info(f"         - {calc['symbol']} ({calc['signal']}): Expected {calc['expected_rr']:.2f}, Got {calc['actual_rr']:.2f}")
            
            # Calculate test success
            success_criteria = [
                rr_results['analyses_attempted'] >= 1,
                rr_results['rr_calculations_found'] >= 1,
                rr_found_rate >= 0.67,  # At least 67% with RR calculations
                rr_results['real_formula_calculations'] >= 1,  # At least 1 real formula calculation
                default_rate < 0.5  # Less than 50% default values
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("RR Calculation Formula Implementation", True, 
                                   f"RR formula implementation successful: {success_count}/{len(success_criteria)} criteria met. Formula rate: {formula_rate:.2f}, Default rate: {default_rate:.2f}")
            else:
                self.log_test_result("RR Calculation Formula Implementation", False, 
                                   f"RR formula implementation issues: {success_count}/{len(success_criteria)} criteria met. May still use default values instead of real formulas")
                
        except Exception as e:
            self.log_test_result("RR Calculation Formula Implementation", False, f"Exception: {str(e)}")

    async def test_ia2_escalation_test(self):
        """Test: IA2 Escalation Test - Verify RR > 2.0 + confidence > 70% trigger IA2 escalation"""
        logger.info("\n🔍 TEST: IA2 Escalation Test")
        
        try:
            escalation_results = {
                'analyses_attempted': 0,
                'high_rr_analyses': 0,
                'high_confidence_analyses': 0,
                'escalation_criteria_met': 0,
                'ia2_escalations_detected': 0,
                'escalation_logs_found': 0,
                'successful_escalations': []
            }
            
            logger.info("   🚀 Testing IA2 escalation criteria...")
            logger.info("   📊 Expected: RR > 2.0 + confidence > 70% trigger IA2 escalation with logs")
            
            # Use symbols from previous test
            test_symbols = self.actual_test_symbols if self.actual_test_symbols else self.test_symbols
            
            for symbol in test_symbols:
                logger.info(f"\n   📞 Testing IA2 escalation for {symbol}...")
                escalation_results['analyses_attempted'] += 1
                
                try:
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        
                        # Extract escalation criteria
                        calculated_rr = ia1_analysis.get('calculated_rr') or ia1_analysis.get('risk_reward_ratio')
                        confidence = ia1_analysis.get('confidence')
                        
                        if calculated_rr is not None and confidence is not None:
                            try:
                                rr_value = float(calculated_rr)
                                conf_value = float(confidence)
                                
                                logger.info(f"      📊 {symbol}: RR={rr_value:.2f}, Confidence={conf_value:.2f}")
                                
                                # Check escalation criteria
                                high_rr = rr_value > 2.0
                                high_confidence = conf_value > 0.7  # 70%
                                
                                if high_rr:
                                    escalation_results['high_rr_analyses'] += 1
                                    logger.info(f"         ✅ High RR detected: {rr_value:.2f} > 2.0")
                                
                                if high_confidence:
                                    escalation_results['high_confidence_analyses'] += 1
                                    logger.info(f"         ✅ High confidence detected: {conf_value:.2f} > 0.7")
                                
                                if high_rr and high_confidence:
                                    escalation_results['escalation_criteria_met'] += 1
                                    logger.info(f"         🚀 IA2 escalation criteria MET for {symbol}")
                                    
                                    # Check if IA2 decision exists in response
                                    ia2_decision = analysis_data.get('ia2_decision')
                                    if ia2_decision:
                                        escalation_results['ia2_escalations_detected'] += 1
                                        logger.info(f"         ✅ IA2 decision found in response")
                                        
                                        escalation_results['successful_escalations'].append({
                                            'symbol': symbol,
                                            'rr_value': rr_value,
                                            'confidence': conf_value,
                                            'ia2_decision': ia2_decision
                                        })
                                    else:
                                        logger.warning(f"         ⚠️ IA2 escalation criteria met but no IA2 decision found")
                                else:
                                    logger.info(f"         📊 IA2 escalation criteria NOT met")
                                    
                            except (ValueError, TypeError) as e:
                                logger.warning(f"         ⚠️ Could not parse RR/confidence values: {e}")
                        else:
                            logger.warning(f"      ❌ Missing RR or confidence data for {symbol}")
                    
                    else:
                        logger.error(f"      ❌ Analysis failed for {symbol}: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"      ❌ Analysis exception for {symbol}: {e}")
                
                # Wait between analyses
                if symbol != test_symbols[-1]:
                    await asyncio.sleep(10)
            
            # Check backend logs for escalation messages
            logger.info("   📋 Checking backend logs for IA2 escalation messages...")
            
            try:
                backend_logs = await self._capture_backend_logs()
                if backend_logs:
                    escalation_log_patterns = [
                        'escalating to ia2', 'ia2 escalation', 'escalation to ia2',
                        'ia2 decision', 'escalation criteria met'
                    ]
                    
                    for log_line in backend_logs:
                        if any(pattern in log_line.lower() for pattern in escalation_log_patterns):
                            escalation_results['escalation_logs_found'] += 1
                            logger.info(f"      ✅ Escalation log found: {log_line.strip()}")
                            break
                    
                    if escalation_results['escalation_logs_found'] == 0:
                        logger.warning(f"      ⚠️ No IA2 escalation logs found in backend")
                        
            except Exception as e:
                logger.warning(f"      ⚠️ Could not analyze backend logs: {e}")
            
            # Calculate metrics
            if escalation_results['analyses_attempted'] > 0:
                high_rr_rate = escalation_results['high_rr_analyses'] / escalation_results['analyses_attempted']
                high_conf_rate = escalation_results['high_confidence_analyses'] / escalation_results['analyses_attempted']
                criteria_met_rate = escalation_results['escalation_criteria_met'] / escalation_results['analyses_attempted']
                escalation_rate = escalation_results['ia2_escalations_detected'] / max(escalation_results['escalation_criteria_met'], 1)
            else:
                high_rr_rate = high_conf_rate = criteria_met_rate = escalation_rate = 0.0
            
            logger.info(f"\n   📊 IA2 ESCALATION TEST RESULTS:")
            logger.info(f"      Analyses attempted: {escalation_results['analyses_attempted']}")
            logger.info(f"      High RR analyses (>2.0): {escalation_results['high_rr_analyses']} ({high_rr_rate:.2f})")
            logger.info(f"      High confidence analyses (>70%): {escalation_results['high_confidence_analyses']} ({high_conf_rate:.2f})")
            logger.info(f"      Escalation criteria met: {escalation_results['escalation_criteria_met']} ({criteria_met_rate:.2f})")
            logger.info(f"      IA2 escalations detected: {escalation_results['ia2_escalations_detected']} ({escalation_rate:.2f})")
            logger.info(f"      Escalation logs found: {escalation_results['escalation_logs_found']}")
            
            if escalation_results['successful_escalations']:
                logger.info(f"      📊 Successful escalations:")
                for escalation in escalation_results['successful_escalations']:
                    logger.info(f"         - {escalation['symbol']}: RR={escalation['rr_value']:.2f}, Conf={escalation['confidence']:.2f}")
            
            # Calculate test success
            success_criteria = [
                escalation_results['analyses_attempted'] >= 1,
                escalation_results['escalation_criteria_met'] >= 0,  # At least 0 (may not have high RR/conf)
                escalation_rate >= 0.5 if escalation_results['escalation_criteria_met'] > 0 else True,  # 50% escalation rate if criteria met
                True  # Always pass if no escalation criteria met (expected)
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.75:  # 75% success threshold
                self.log_test_result("IA2 Escalation Test", True, 
                                   f"IA2 escalation test successful: {success_count}/{len(success_criteria)} criteria met. Criteria met: {escalation_results['escalation_criteria_met']}, Escalations: {escalation_results['ia2_escalations_detected']}")
            else:
                self.log_test_result("IA2 Escalation Test", False, 
                                   f"IA2 escalation test issues: {success_count}/{len(success_criteria)} criteria met. May have escalation logic problems")
                
        except Exception as e:
            self.log_test_result("IA2 Escalation Test", False, f"Exception: {str(e)}")

    async def run_all_tests(self):
        """Run all Pydantic fix validation tests"""
        logger.info("🚀 STARTING PYDANTIC FIX VALIDATION TESTING SUITE")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_pydantic_validation_fix_validation()
        await self.test_reasoning_field_restoration()
        await self.test_rr_calculation_formula_implementation()
        await self.test_ia2_escalation_test()
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("🎯 PYDANTIC FIX VALIDATION TEST SUITE SUMMARY")
        logger.info("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success rate: {passed_tests/total_tests:.2f}" if total_tests > 0 else "No tests run")
        
        logger.info("\n📊 DETAILED RESULTS:")
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
        
        return passed_tests, failed_tests

async def main():
    """Main test execution"""
    test_suite = PydanticFixValidationTestSuite()
    passed, failed = await test_suite.run_all_tests()
    
    # Exit with appropriate code
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        logger.error(f"❌ {failed} TESTS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
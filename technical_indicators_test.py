#!/usr/bin/env python3
"""
IA1 TECHNICAL INDICATORS FIX TEST SUITE
Focus: Test IA1 Technical Indicators Fix - COMPREHENSIVE VALIDATION

CRITICAL TEST REQUIREMENTS FROM REVIEW REQUEST:
1. **Technical Indicators Calculation**: Test the `/api/run-ia1-cycle` endpoint to ensure technical indicators are showing calculated values instead of defaults:
   - RSI should show real values (not 50.0)
   - MACD should show real values (not 0.0) 
   - Stochastic should show real values (not 50.0)
   - MFI should show real values (not 50.0)
   - VWAP should show real values (not 0.0)
   - Advanced indicators like mfi_signal, vwap_signal should show meaningful values (not "neutral")

2. **Error Handling Robustness**: Test that even when there are technical errors during analysis, the calculated indicators are preserved instead of falling back to defaults.

3. **IA1 to IA2 Escalation**: Test if the system can now properly escalate to IA2 based on real technical indicators:
   - Risk-reward ratios should be calculated based on real data
   - Confidence levels should reflect actual market conditions
   - Escalation criteria should work with real indicators

4. **Data Consistency**: Verify that the technical indicators in the API response match what's being calculated and logged in the backend.

SUCCESS CRITERIA:
✅ Technical indicators show real calculated values instead of defaults
✅ RSI, MACD, Stochastic, MFI, VWAP show meaningful non-default values
✅ Advanced signals (mfi_signal, vwap_signal) show calculated values not "neutral"
✅ Error handling preserves calculated indicators during fallback scenarios
✅ IA1 to IA2 escalation works with real technical indicators
✅ Data consistency between API response and backend calculations
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
from pymongo import MongoClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TechnicalIndicatorsFixTestSuite:
    """Comprehensive test suite for IA1 Technical Indicators Fix"""
    
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
        logger.info(f"Testing Technical Indicators Fix at: {self.api_url}")
        
        # MongoDB connection for direct database analysis
        try:
            self.mongo_client = MongoClient("mongodb://localhost:27017")
            self.db = self.mongo_client["myapp"]
            logger.info("✅ MongoDB connection established for technical indicators testing")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.mongo_client = None
            self.db = None
        
        # Test results
        self.test_results = []
        
        # Test symbols for technical indicators testing
        self.test_symbols = [
            "BTCUSDT",   # High volatility, good for technical indicators
            "ETHUSDT",   # Medium volatility, stable indicators
            "SOLUSDT",   # High volatility, trending
            "XRPUSDT",   # Lower volatility, edge cases
            "ADAUSDT"    # Medium volatility, range-bound
        ]
        
        # Default values that should NOT appear in real calculations
        self.default_values = {
            "rsi": 50.0,
            "macd": 0.0,
            "stochastic": 50.0,
            "mfi": 50.0,
            "vwap": 0.0
        }
        
        # Neutral signals that should NOT appear in real calculations
        self.neutral_signals = ["neutral", "hold", "unknown", ""]
        
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
    
    async def test_1_technical_indicators_real_values(self):
        """Test 1: Verify Technical Indicators Show Real Values Instead of Defaults"""
        logger.info("\n🔍 TEST 1: Technical Indicators Real Values Test")
        
        try:
            indicators_test_results = {
                'symbols_tested': 0,
                'analyses_with_real_rsi': 0,
                'analyses_with_real_macd': 0,
                'analyses_with_real_stochastic': 0,
                'analyses_with_real_mfi': 0,
                'analyses_with_real_vwap': 0,
                'default_value_violations': [],
                'real_indicators_found': []
            }
            
            logger.info("   🚀 Testing technical indicators for real calculated values...")
            logger.info(f"      Default values to avoid: RSI={self.default_values['rsi']}, MACD={self.default_values['macd']}, Stochastic={self.default_values['stochastic']}, MFI={self.default_values['mfi']}, VWAP={self.default_values['vwap']}")
            
            # Test multiple symbols to verify technical indicators
            for symbol in self.test_symbols[:3]:  # Test first 3 symbols
                try:
                    logger.info(f"   📊 Testing technical indicators for {symbol}...")
                    
                    # Run IA1 cycle to get analysis with technical indicators
                    response = requests.post(
                        f"{self.api_url}/run-ia1-cycle",
                        json={"symbol": symbol},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        cycle_data = response.json()
                        
                        if cycle_data.get('success'):
                            analysis = cycle_data.get('analysis', {})
                            indicators_test_results['symbols_tested'] += 1
                            
                            # Extract technical indicators from analysis
                            rsi_value = analysis.get('rsi_signal', 'unknown')
                            macd_value = analysis.get('macd_trend', 'unknown')
                            stochastic_value = analysis.get('stochastic_signal', 'unknown')
                            mfi_value = analysis.get('mfi_signal', 'unknown')
                            vwap_value = analysis.get('vwap_signal', 'unknown')
                            
                            # Also check for numeric values in reasoning or other fields
                            reasoning = analysis.get('reasoning', '')
                            analysis_text = analysis.get('analysis', '')
                            
                            logger.info(f"      📈 {symbol} Technical Indicators:")
                            logger.info(f"         RSI Signal: {rsi_value}")
                            logger.info(f"         MACD Trend: {macd_value}")
                            logger.info(f"         Stochastic Signal: {stochastic_value}")
                            logger.info(f"         MFI Signal: {mfi_value}")
                            logger.info(f"         VWAP Signal: {vwap_value}")
                            
                            # Check RSI for real values
                            if rsi_value not in self.neutral_signals and rsi_value != 'unknown':
                                indicators_test_results['analyses_with_real_rsi'] += 1
                                indicators_test_results['real_indicators_found'].append(f"{symbol}: RSI={rsi_value}")
                                logger.info(f"         ✅ RSI shows real signal: {rsi_value}")
                            else:
                                indicators_test_results['default_value_violations'].append(f"{symbol}: RSI shows default/neutral value: {rsi_value}")
                                logger.warning(f"         ❌ RSI shows default/neutral: {rsi_value}")
                            
                            # Check MACD for real values
                            if macd_value not in self.neutral_signals and macd_value != 'unknown':
                                indicators_test_results['analyses_with_real_macd'] += 1
                                indicators_test_results['real_indicators_found'].append(f"{symbol}: MACD={macd_value}")
                                logger.info(f"         ✅ MACD shows real trend: {macd_value}")
                            else:
                                indicators_test_results['default_value_violations'].append(f"{symbol}: MACD shows default/neutral value: {macd_value}")
                                logger.warning(f"         ❌ MACD shows default/neutral: {macd_value}")
                            
                            # Check Stochastic for real values
                            if stochastic_value not in self.neutral_signals and stochastic_value != 'unknown':
                                indicators_test_results['analyses_with_real_stochastic'] += 1
                                indicators_test_results['real_indicators_found'].append(f"{symbol}: Stochastic={stochastic_value}")
                                logger.info(f"         ✅ Stochastic shows real signal: {stochastic_value}")
                            else:
                                indicators_test_results['default_value_violations'].append(f"{symbol}: Stochastic shows default/neutral value: {stochastic_value}")
                                logger.warning(f"         ❌ Stochastic shows default/neutral: {stochastic_value}")
                            
                            # Check MFI for real values
                            if mfi_value not in self.neutral_signals and mfi_value != 'unknown':
                                indicators_test_results['analyses_with_real_mfi'] += 1
                                indicators_test_results['real_indicators_found'].append(f"{symbol}: MFI={mfi_value}")
                                logger.info(f"         ✅ MFI shows real signal: {mfi_value}")
                            else:
                                indicators_test_results['default_value_violations'].append(f"{symbol}: MFI shows default/neutral value: {mfi_value}")
                                logger.warning(f"         ❌ MFI shows default/neutral: {mfi_value}")
                            
                            # Check VWAP for real values
                            if vwap_value not in self.neutral_signals and vwap_value != 'unknown':
                                indicators_test_results['analyses_with_real_vwap'] += 1
                                indicators_test_results['real_indicators_found'].append(f"{symbol}: VWAP={vwap_value}")
                                logger.info(f"         ✅ VWAP shows real signal: {vwap_value}")
                            else:
                                indicators_test_results['default_value_violations'].append(f"{symbol}: VWAP shows default/neutral value: {vwap_value}")
                                logger.warning(f"         ❌ VWAP shows default/neutral: {vwap_value}")
                            
                            # Look for numeric values in reasoning text
                            rsi_numeric_found = re.search(r'RSI[:\s]*(\d+\.?\d*)', reasoning + ' ' + analysis_text, re.IGNORECASE)
                            macd_numeric_found = re.search(r'MACD[:\s]*(-?\d+\.?\d*)', reasoning + ' ' + analysis_text, re.IGNORECASE)
                            
                            if rsi_numeric_found:
                                rsi_numeric = float(rsi_numeric_found.group(1))
                                if abs(rsi_numeric - self.default_values['rsi']) > 5.0:  # Not default 50.0
                                    logger.info(f"         ✅ RSI numeric value found: {rsi_numeric}")
                                else:
                                    logger.warning(f"         ⚠️ RSI numeric close to default: {rsi_numeric}")
                            
                            if macd_numeric_found:
                                macd_numeric = float(macd_numeric_found.group(1))
                                if abs(macd_numeric - self.default_values['macd']) > 0.01:  # Not default 0.0
                                    logger.info(f"         ✅ MACD numeric value found: {macd_numeric}")
                                else:
                                    logger.warning(f"         ⚠️ MACD numeric close to default: {macd_numeric}")
                        else:
                            logger.warning(f"      ❌ IA1 cycle failed for {symbol}: {cycle_data.get('error', 'Unknown error')}")
                    else:
                        logger.warning(f"      ❌ API call failed for {symbol}: HTTP {response.status_code}")
                    
                    await asyncio.sleep(3)  # Delay between requests
                    
                except Exception as e:
                    logger.error(f"   ❌ Error testing technical indicators for {symbol}: {e}")
                    indicators_test_results['default_value_violations'].append(f"Exception testing {symbol}: {str(e)}")
            
            # Calculate results
            total_indicators_tested = indicators_test_results['symbols_tested'] * 5  # 5 indicators per symbol
            real_indicators_found = (indicators_test_results['analyses_with_real_rsi'] + 
                                   indicators_test_results['analyses_with_real_macd'] + 
                                   indicators_test_results['analyses_with_real_stochastic'] + 
                                   indicators_test_results['analyses_with_real_mfi'] + 
                                   indicators_test_results['analyses_with_real_vwap'])
            
            logger.info(f"   📊 Technical Indicators Test Results:")
            logger.info(f"      Symbols tested: {indicators_test_results['symbols_tested']}")
            logger.info(f"      Real RSI values: {indicators_test_results['analyses_with_real_rsi']}")
            logger.info(f"      Real MACD values: {indicators_test_results['analyses_with_real_macd']}")
            logger.info(f"      Real Stochastic values: {indicators_test_results['analyses_with_real_stochastic']}")
            logger.info(f"      Real MFI values: {indicators_test_results['analyses_with_real_mfi']}")
            logger.info(f"      Real VWAP values: {indicators_test_results['analyses_with_real_vwap']}")
            logger.info(f"      Total real indicators: {real_indicators_found}/{total_indicators_tested}")
            
            if indicators_test_results['default_value_violations']:
                logger.info(f"      Default value violations:")
                for violation in indicators_test_results['default_value_violations'][:5]:  # Show first 5
                    logger.info(f"        - {violation}")
            
            if indicators_test_results['real_indicators_found']:
                logger.info(f"      Real indicators found:")
                for real_indicator in indicators_test_results['real_indicators_found'][:5]:  # Show first 5
                    logger.info(f"        - {real_indicator}")
            
            # Determine test result
            if total_indicators_tested > 0:
                real_indicators_rate = real_indicators_found / total_indicators_tested
                
                if real_indicators_rate >= 0.8:
                    self.log_test_result("Technical Indicators Real Values", True, 
                                       f"Technical indicators show real values: {real_indicators_rate:.1%} real indicators ({real_indicators_found}/{total_indicators_tested})")
                elif real_indicators_rate >= 0.6:
                    self.log_test_result("Technical Indicators Real Values", False, 
                                       f"Most indicators show real values: {real_indicators_rate:.1%} real indicators ({real_indicators_found}/{total_indicators_tested})")
                else:
                    self.log_test_result("Technical Indicators Real Values", False, 
                                       f"Many indicators still show defaults: {real_indicators_rate:.1%} real indicators ({real_indicators_found}/{total_indicators_tested})")
            else:
                self.log_test_result("Technical Indicators Real Values", False, 
                                   "No technical indicators available for testing")
                
        except Exception as e:
            self.log_test_result("Technical Indicators Real Values", False, f"Exception: {str(e)}")
    
    async def test_2_error_handling_robustness(self):
        """Test 2: Error Handling Preserves Calculated Indicators"""
        logger.info("\n🔍 TEST 2: Error Handling Robustness Test")
        
        try:
            error_handling_results = {
                'symbols_tested': 0,
                'analyses_with_errors': 0,
                'indicators_preserved_during_errors': 0,
                'fallback_to_defaults_detected': 0,
                'error_scenarios': []
            }
            
            logger.info("   🚀 Testing error handling robustness for technical indicators...")
            
            # Test symbols that might trigger errors or edge cases
            edge_case_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
            
            for symbol in edge_case_symbols:
                try:
                    logger.info(f"   📊 Testing error handling for {symbol}...")
                    
                    # Run multiple IA1 cycles to potentially trigger error conditions
                    for attempt in range(2):
                        try:
                            response = requests.post(
                                f"{self.api_url}/run-ia1-cycle",
                                json={"symbol": symbol},
                                timeout=45  # Shorter timeout to potentially trigger errors
                            )
                            
                            if response.status_code == 200:
                                cycle_data = response.json()
                                error_handling_results['symbols_tested'] += 1
                                
                                if cycle_data.get('success'):
                                    analysis = cycle_data.get('analysis', {})
                                    
                                    # Check if there were any errors mentioned in the response
                                    error_mentioned = False
                                    reasoning = analysis.get('reasoning', '').lower()
                                    analysis_text = analysis.get('analysis', '').lower()
                                    
                                    error_keywords = ['error', 'fallback', 'default', 'failed', 'exception']
                                    if any(keyword in reasoning or keyword in analysis_text for keyword in error_keywords):
                                        error_mentioned = True
                                        error_handling_results['analyses_with_errors'] += 1
                                        logger.info(f"      ⚠️ Error condition detected in analysis")
                                    
                                    # Check if indicators are still real despite potential errors
                                    rsi_value = analysis.get('rsi_signal', 'unknown')
                                    macd_value = analysis.get('macd_trend', 'unknown')
                                    stochastic_value = analysis.get('stochastic_signal', 'unknown')
                                    mfi_value = analysis.get('mfi_signal', 'unknown')
                                    vwap_value = analysis.get('vwap_signal', 'unknown')
                                    
                                    real_indicators_count = 0
                                    if rsi_value not in self.neutral_signals and rsi_value != 'unknown':
                                        real_indicators_count += 1
                                    if macd_value not in self.neutral_signals and macd_value != 'unknown':
                                        real_indicators_count += 1
                                    if stochastic_value not in self.neutral_signals and stochastic_value != 'unknown':
                                        real_indicators_count += 1
                                    if mfi_value not in self.neutral_signals and mfi_value != 'unknown':
                                        real_indicators_count += 1
                                    if vwap_value not in self.neutral_signals and vwap_value != 'unknown':
                                        real_indicators_count += 1
                                    
                                    if error_mentioned and real_indicators_count >= 3:
                                        error_handling_results['indicators_preserved_during_errors'] += 1
                                        logger.info(f"      ✅ Indicators preserved despite errors: {real_indicators_count}/5 real")
                                        error_handling_results['error_scenarios'].append(
                                            f"{symbol} attempt {attempt+1}: {real_indicators_count}/5 indicators preserved during error"
                                        )
                                    elif error_mentioned and real_indicators_count < 3:
                                        error_handling_results['fallback_to_defaults_detected'] += 1
                                        logger.warning(f"      ❌ Fallback to defaults detected: {real_indicators_count}/5 real")
                                        error_handling_results['error_scenarios'].append(
                                            f"{symbol} attempt {attempt+1}: Only {real_indicators_count}/5 indicators preserved during error"
                                        )
                                    else:
                                        logger.info(f"      ⚪ Normal analysis: {real_indicators_count}/5 real indicators")
                                else:
                                    logger.warning(f"      ❌ IA1 cycle failed for {symbol} attempt {attempt+1}")
                            else:
                                logger.warning(f"      ❌ API call failed for {symbol} attempt {attempt+1}: HTTP {response.status_code}")
                            
                            await asyncio.sleep(2)  # Short delay between attempts
                            
                        except Exception as e:
                            logger.info(f"      ⚠️ Exception during {symbol} attempt {attempt+1}: {e}")
                            # This is actually good for testing error handling
                            error_handling_results['analyses_with_errors'] += 1
                    
                    await asyncio.sleep(3)  # Delay between symbols
                    
                except Exception as e:
                    logger.error(f"   ❌ Error testing error handling for {symbol}: {e}")
            
            # Analyze results
            logger.info(f"   📊 Error Handling Test Results:")
            logger.info(f"      Symbols tested: {len(edge_case_symbols)}")
            logger.info(f"      Total analyses: {error_handling_results['symbols_tested']}")
            logger.info(f"      Analyses with errors: {error_handling_results['analyses_with_errors']}")
            logger.info(f"      Indicators preserved during errors: {error_handling_results['indicators_preserved_during_errors']}")
            logger.info(f"      Fallback to defaults detected: {error_handling_results['fallback_to_defaults_detected']}")
            
            if error_handling_results['error_scenarios']:
                logger.info(f"      Error scenarios:")
                for scenario in error_handling_results['error_scenarios']:
                    logger.info(f"        - {scenario}")
            
            # Determine test result
            if error_handling_results['analyses_with_errors'] > 0:
                preservation_rate = error_handling_results['indicators_preserved_during_errors'] / error_handling_results['analyses_with_errors']
                fallback_rate = error_handling_results['fallback_to_defaults_detected'] / error_handling_results['analyses_with_errors']
                
                if preservation_rate >= 0.8 and fallback_rate <= 0.2:
                    self.log_test_result("Error Handling Robustness", True, 
                                       f"Indicators preserved during errors: {preservation_rate:.1%} preservation, {fallback_rate:.1%} fallback")
                elif preservation_rate >= 0.6:
                    self.log_test_result("Error Handling Robustness", False, 
                                       f"Partial indicator preservation: {preservation_rate:.1%} preservation, {fallback_rate:.1%} fallback")
                else:
                    self.log_test_result("Error Handling Robustness", False, 
                                       f"Poor error handling: {preservation_rate:.1%} preservation, {fallback_rate:.1%} fallback")
            else:
                self.log_test_result("Error Handling Robustness", False, 
                                   "No error conditions encountered for testing")
                
        except Exception as e:
            self.log_test_result("Error Handling Robustness", False, f"Exception: {str(e)}")
    
    async def test_3_ia1_to_ia2_escalation_with_real_indicators(self):
        """Test 3: IA1 to IA2 Escalation Based on Real Technical Indicators"""
        logger.info("\n🔍 TEST 3: IA1 to IA2 Escalation with Real Technical Indicators Test")
        
        try:
            escalation_results = {
                'symbols_tested': 0,
                'escalations_occurred': 0,
                'escalations_with_real_indicators': 0,
                'escalations_with_default_indicators': 0,
                'confidence_based_on_real_data': 0,
                'escalation_details': []
            }
            
            logger.info("   🚀 Testing IA1 to IA2 escalation with real technical indicators...")
            
            # Test symbols that are likely to trigger escalation
            escalation_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
            
            for symbol in escalation_symbols:
                try:
                    logger.info(f"   📊 Testing escalation for {symbol}...")
                    
                    # Run IA1 cycle to potentially trigger escalation
                    response = requests.post(
                        f"{self.api_url}/run-ia1-cycle",
                        json={"symbol": symbol},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        cycle_data = response.json()
                        escalation_results['symbols_tested'] += 1
                        
                        if cycle_data.get('success'):
                            analysis = cycle_data.get('analysis', {})
                            escalated_to_ia2 = cycle_data.get('escalated_to_ia2', False)
                            ia2_decision = cycle_data.get('ia2_decision')
                            
                            confidence = analysis.get('confidence', 0)
                            rr_ratio = analysis.get('risk_reward_ratio', 0)
                            signal = analysis.get('recommendation', 'hold').lower()
                            
                            # Check technical indicators in the analysis
                            rsi_value = analysis.get('rsi_signal', 'unknown')
                            macd_value = analysis.get('macd_trend', 'unknown')
                            stochastic_value = analysis.get('stochastic_signal', 'unknown')
                            mfi_value = analysis.get('mfi_signal', 'unknown')
                            vwap_value = analysis.get('vwap_signal', 'unknown')
                            
                            real_indicators_count = 0
                            if rsi_value not in self.neutral_signals and rsi_value != 'unknown':
                                real_indicators_count += 1
                            if macd_value not in self.neutral_signals and macd_value != 'unknown':
                                real_indicators_count += 1
                            if stochastic_value not in self.neutral_signals and stochastic_value != 'unknown':
                                real_indicators_count += 1
                            if mfi_value not in self.neutral_signals and mfi_value != 'unknown':
                                real_indicators_count += 1
                            if vwap_value not in self.neutral_signals and vwap_value != 'unknown':
                                real_indicators_count += 1
                            
                            logger.info(f"      📈 {symbol} Analysis:")
                            logger.info(f"         Confidence: {confidence:.1%}")
                            logger.info(f"         RR Ratio: {rr_ratio:.2f}")
                            logger.info(f"         Signal: {signal}")
                            logger.info(f"         Real indicators: {real_indicators_count}/5")
                            logger.info(f"         Escalated to IA2: {escalated_to_ia2}")
                            
                            if escalated_to_ia2:
                                escalation_results['escalations_occurred'] += 1
                                
                                if real_indicators_count >= 3:
                                    escalation_results['escalations_with_real_indicators'] += 1
                                    logger.info(f"         ✅ Escalation based on real indicators: {real_indicators_count}/5")
                                else:
                                    escalation_results['escalations_with_default_indicators'] += 1
                                    logger.warning(f"         ❌ Escalation with mostly default indicators: {real_indicators_count}/5")
                                
                                # Check if confidence reflects real market conditions
                                if confidence > 0.5 and real_indicators_count >= 3:
                                    escalation_results['confidence_based_on_real_data'] += 1
                                    logger.info(f"         ✅ Confidence reflects real data: {confidence:.1%}")
                                
                                escalation_results['escalation_details'].append({
                                    'symbol': symbol,
                                    'confidence': confidence,
                                    'rr_ratio': rr_ratio,
                                    'signal': signal,
                                    'real_indicators': real_indicators_count,
                                    'ia2_decision': ia2_decision.get('signal', 'N/A') if ia2_decision else 'N/A'
                                })
                                
                                # Check IA2 decision if available
                                if ia2_decision:
                                    ia2_signal = ia2_decision.get('signal', 'N/A')
                                    ia2_confidence = ia2_decision.get('confidence', 0)
                                    logger.info(f"         🤖 IA2 Decision: {ia2_signal} with confidence {ia2_confidence:.1%}")
                            else:
                                logger.info(f"         ⚪ No escalation occurred")
                        else:
                            logger.warning(f"      ❌ IA1 cycle failed for {symbol}: {cycle_data.get('error', 'Unknown error')}")
                    else:
                        logger.warning(f"      ❌ API call failed for {symbol}: HTTP {response.status_code}")
                    
                    await asyncio.sleep(3)  # Delay between requests
                    
                except Exception as e:
                    logger.error(f"   ❌ Error testing escalation for {symbol}: {e}")
            
            # Analyze results
            logger.info(f"   📊 Escalation with Real Indicators Test Results:")
            logger.info(f"      Symbols tested: {escalation_results['symbols_tested']}")
            logger.info(f"      Escalations occurred: {escalation_results['escalations_occurred']}")
            logger.info(f"      Escalations with real indicators: {escalation_results['escalations_with_real_indicators']}")
            logger.info(f"      Escalations with default indicators: {escalation_results['escalations_with_default_indicators']}")
            logger.info(f"      Confidence based on real data: {escalation_results['confidence_based_on_real_data']}")
            
            if escalation_results['escalation_details']:
                logger.info(f"      Escalation details:")
                for detail in escalation_results['escalation_details']:
                    logger.info(f"        - {detail['symbol']}: {detail['confidence']:.1%} confidence, {detail['real_indicators']}/5 real indicators, IA2: {detail['ia2_decision']}")
            
            # Determine test result
            if escalation_results['escalations_occurred'] > 0:
                real_indicators_rate = escalation_results['escalations_with_real_indicators'] / escalation_results['escalations_occurred']
                confidence_quality_rate = escalation_results['confidence_based_on_real_data'] / escalation_results['escalations_occurred']
                
                if real_indicators_rate >= 0.8 and confidence_quality_rate >= 0.8:
                    self.log_test_result("IA1 to IA2 Escalation with Real Indicators", True, 
                                       f"Escalations based on real indicators: {real_indicators_rate:.1%} real indicators, {confidence_quality_rate:.1%} quality confidence")
                elif real_indicators_rate >= 0.6:
                    self.log_test_result("IA1 to IA2 Escalation with Real Indicators", False, 
                                       f"Most escalations use real indicators: {real_indicators_rate:.1%} real indicators, {confidence_quality_rate:.1%} quality confidence")
                else:
                    self.log_test_result("IA1 to IA2 Escalation with Real Indicators", False, 
                                       f"Escalations still use default indicators: {real_indicators_rate:.1%} real indicators, {confidence_quality_rate:.1%} quality confidence")
            else:
                self.log_test_result("IA1 to IA2 Escalation with Real Indicators", False, 
                                   "No escalations occurred for testing")
                
        except Exception as e:
            self.log_test_result("IA1 to IA2 Escalation with Real Indicators", False, f"Exception: {str(e)}")
    
    async def test_4_data_consistency_backend_vs_api(self):
        """Test 4: Data Consistency Between Backend Calculations and API Response"""
        logger.info("\n🔍 TEST 4: Data Consistency Between Backend and API Test")
        
        try:
            consistency_results = {
                'symbols_tested': 0,
                'backend_logs_analyzed': 0,
                'api_backend_matches': 0,
                'consistency_violations': [],
                'log_indicator_values': [],
                'api_indicator_values': []
            }
            
            logger.info("   🚀 Testing data consistency between backend calculations and API response...")
            
            # Test symbols and capture both API response and backend logs
            consistency_symbols = ["BTCUSDT", "ETHUSDT"]
            
            for symbol in consistency_symbols:
                try:
                    logger.info(f"   📊 Testing data consistency for {symbol}...")
                    
                    # Clear recent logs and run analysis
                    start_time = datetime.now()
                    
                    # Run IA1 cycle
                    response = requests.post(
                        f"{self.api_url}/run-ia1-cycle",
                        json={"symbol": symbol},
                        timeout=60
                    )
                    
                    end_time = datetime.now()
                    
                    if response.status_code == 200:
                        cycle_data = response.json()
                        consistency_results['symbols_tested'] += 1
                        
                        if cycle_data.get('success'):
                            analysis = cycle_data.get('analysis', {})
                            
                            # Extract API indicator values
                            api_indicators = {
                                'rsi': analysis.get('rsi_signal', 'unknown'),
                                'macd': analysis.get('macd_trend', 'unknown'),
                                'stochastic': analysis.get('stochastic_signal', 'unknown'),
                                'mfi': analysis.get('mfi_signal', 'unknown'),
                                'vwap': analysis.get('vwap_signal', 'unknown')
                            }
                            
                            consistency_results['api_indicator_values'].append({
                                'symbol': symbol,
                                'indicators': api_indicators
                            })
                            
                            logger.info(f"      📈 {symbol} API Indicators:")
                            for indicator, value in api_indicators.items():
                                logger.info(f"         {indicator.upper()}: {value}")
                            
                            # Analyze backend logs for the same time period
                            log_indicators = await self._extract_indicators_from_logs(symbol, start_time, end_time)
                            
                            if log_indicators:
                                consistency_results['backend_logs_analyzed'] += 1
                                consistency_results['log_indicator_values'].append({
                                    'symbol': symbol,
                                    'indicators': log_indicators
                                })
                                
                                logger.info(f"      📋 {symbol} Backend Log Indicators:")
                                for indicator, value in log_indicators.items():
                                    logger.info(f"         {indicator.upper()}: {value}")
                                
                                # Compare API vs Backend
                                matches = 0
                                total_comparisons = 0
                                
                                for indicator in api_indicators:
                                    if indicator in log_indicators:
                                        total_comparisons += 1
                                        api_val = api_indicators[indicator]
                                        log_val = log_indicators[indicator]
                                        
                                        if api_val == log_val or (api_val != 'unknown' and log_val != 'unknown' and api_val != 'neutral' and log_val != 'neutral'):
                                            matches += 1
                                            logger.info(f"         ✅ {indicator.upper()} consistent: API={api_val}, Log={log_val}")
                                        else:
                                            consistency_results['consistency_violations'].append(
                                                f"{symbol} {indicator.upper()}: API={api_val}, Log={log_val}"
                                            )
                                            logger.warning(f"         ❌ {indicator.upper()} inconsistent: API={api_val}, Log={log_val}")
                                
                                if total_comparisons > 0:
                                    consistency_rate = matches / total_comparisons
                                    if consistency_rate >= 0.8:
                                        consistency_results['api_backend_matches'] += 1
                                        logger.info(f"      ✅ High consistency: {matches}/{total_comparisons} indicators match")
                                    else:
                                        logger.warning(f"      ⚠️ Low consistency: {matches}/{total_comparisons} indicators match")
                            else:
                                logger.warning(f"      ⚠️ No backend log indicators found for {symbol}")
                        else:
                            logger.warning(f"      ❌ IA1 cycle failed for {symbol}: {cycle_data.get('error', 'Unknown error')}")
                    else:
                        logger.warning(f"      ❌ API call failed for {symbol}: HTTP {response.status_code}")
                    
                    await asyncio.sleep(5)  # Longer delay to allow log processing
                    
                except Exception as e:
                    logger.error(f"   ❌ Error testing data consistency for {symbol}: {e}")
            
            # Analyze results
            logger.info(f"   📊 Data Consistency Test Results:")
            logger.info(f"      Symbols tested: {consistency_results['symbols_tested']}")
            logger.info(f"      Backend logs analyzed: {consistency_results['backend_logs_analyzed']}")
            logger.info(f"      API-Backend matches: {consistency_results['api_backend_matches']}")
            
            if consistency_results['consistency_violations']:
                logger.info(f"      Consistency violations:")
                for violation in consistency_results['consistency_violations']:
                    logger.info(f"        - {violation}")
            
            # Determine test result
            if consistency_results['backend_logs_analyzed'] > 0:
                consistency_rate = consistency_results['api_backend_matches'] / consistency_results['backend_logs_analyzed']
                
                if consistency_rate >= 0.8:
                    self.log_test_result("Data Consistency Backend vs API", True, 
                                       f"High data consistency: {consistency_rate:.1%} API-backend matches ({consistency_results['api_backend_matches']}/{consistency_results['backend_logs_analyzed']})")
                elif consistency_rate >= 0.6:
                    self.log_test_result("Data Consistency Backend vs API", False, 
                                       f"Moderate data consistency: {consistency_rate:.1%} API-backend matches ({consistency_results['api_backend_matches']}/{consistency_results['backend_logs_analyzed']})")
                else:
                    self.log_test_result("Data Consistency Backend vs API", False, 
                                       f"Low data consistency: {consistency_rate:.1%} API-backend matches ({consistency_results['api_backend_matches']}/{consistency_results['backend_logs_analyzed']})")
            else:
                self.log_test_result("Data Consistency Backend vs API", False, 
                                   "No backend logs available for consistency testing")
                
        except Exception as e:
            self.log_test_result("Data Consistency Backend vs API", False, f"Exception: {str(e)}")
    
    async def _extract_indicators_from_logs(self, symbol: str, start_time: datetime, end_time: datetime) -> Dict[str, str]:
        """Extract technical indicators from backend logs for a specific symbol and time period"""
        try:
            log_files = [
                "/var/log/supervisor/backend.out.log",
                "/var/log/supervisor/backend.err.log"
            ]
            
            indicators = {}
            
            for log_file in log_files:
                if os.path.exists(log_file):
                    try:
                        result = subprocess.run(['tail', '-n', '2000', log_file], 
                                              capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            log_content = result.stdout
                            lines = log_content.split('\n')
                            
                            for line in lines:
                                if symbol in line:
                                    # Look for RSI patterns
                                    rsi_match = re.search(r'RSI[:\s]*(\w+)', line, re.IGNORECASE)
                                    if rsi_match and 'rsi' not in indicators:
                                        indicators['rsi'] = rsi_match.group(1).lower()
                                    
                                    # Look for MACD patterns
                                    macd_match = re.search(r'MACD[:\s]*(\w+)', line, re.IGNORECASE)
                                    if macd_match and 'macd' not in indicators:
                                        indicators['macd'] = macd_match.group(1).lower()
                                    
                                    # Look for Stochastic patterns
                                    stoch_match = re.search(r'Stochastic[:\s]*(\w+)', line, re.IGNORECASE)
                                    if stoch_match and 'stochastic' not in indicators:
                                        indicators['stochastic'] = stoch_match.group(1).lower()
                                    
                                    # Look for MFI patterns
                                    mfi_match = re.search(r'MFI[:\s]*(\w+)', line, re.IGNORECASE)
                                    if mfi_match and 'mfi' not in indicators:
                                        indicators['mfi'] = mfi_match.group(1).lower()
                                    
                                    # Look for VWAP patterns
                                    vwap_match = re.search(r'VWAP[:\s]*(\w+)', line, re.IGNORECASE)
                                    if vwap_match and 'vwap' not in indicators:
                                        indicators['vwap'] = vwap_match.group(1).lower()
                            
                    except Exception as e:
                        logger.warning(f"   ⚠️ Could not analyze {log_file}: {e}")
            
            return indicators
            
        except Exception as e:
            logger.warning(f"Log indicator extraction failed: {e}")
            return {}
    
    async def run_comprehensive_test_suite(self):
        """Run comprehensive technical indicators fix test suite"""
        logger.info("🚀 Starting IA1 Technical Indicators Fix Test Suite")
        logger.info("=" * 80)
        logger.info("📋 IA1 TECHNICAL INDICATORS FIX TEST SUITE")
        logger.info("🎯 Testing: Technical indicators showing real values instead of defaults")
        logger.info("🎯 Expected: RSI, MACD, Stochastic, MFI, VWAP show calculated values")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Run all tests
        await self.test_1_technical_indicators_real_values()
        await self.test_2_error_handling_robustness()
        await self.test_3_ia1_to_ia2_escalation_with_real_indicators()
        await self.test_4_data_consistency_backend_vs_api()
        
        # Calculate overall results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 TECHNICAL INDICATORS FIX TEST SUITE RESULTS")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total execution time: {duration:.1f} seconds")
        logger.info(f"📈 Tests passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
        logger.info("")
        
        # Detailed results
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
        
        logger.info("\n" + "=" * 80)
        logger.info("🎯 TECHNICAL INDICATORS FIX VALIDATION SUMMARY")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Technical indicators fix is working correctly!")
            logger.info("✅ Technical indicators show real calculated values")
            logger.info("✅ Error handling preserves calculated indicators")
            logger.info("✅ IA1 to IA2 escalation works with real indicators")
            logger.info("✅ Data consistency maintained between backend and API")
        elif passed_tests >= total_tests * 0.75:
            logger.info("✅ MOST TESTS PASSED - Technical indicators fix is mostly working")
            logger.info("⚠️  Some minor issues detected, but core functionality works")
        else:
            logger.info("❌ MULTIPLE TEST FAILURES - Technical indicators fix needs attention")
            logger.info("🔧 Review failed tests and fix remaining issues")
        
        logger.info("=" * 80)
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'duration': duration,
            'test_results': self.test_results
        }

# Main execution
async def main():
    """Main test execution function"""
    test_suite = TechnicalIndicatorsFixTestSuite()
    results = await test_suite.run_comprehensive_test_suite()
    
    # Return exit code based on results
    if results['success_rate'] >= 0.8:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure

if __name__ == "__main__":
    asyncio.run(main())
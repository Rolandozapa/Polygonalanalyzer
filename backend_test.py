#!/usr/bin/env python3
"""
IA1 TECHNICAL INDICATORS FIX TEST SUITE
Focus: Test IA1 Technical Indicators Fix - COMPREHENSIVE VALIDATION

CRITICAL TEST REQUIREMENTS FROM REVIEW REQUEST:
1. **All Technical Indicators Working**: Verify RSI, MACD, Stochastic, MFI, VWAP all show calculated values with meaningful signals
2. **Signal Quality**: Confirm all signals show proper categorization (extreme_overbought, overbought, bullish, bearish, etc.) instead of 'unknown' or 'neutral'
3. **Error Handling**: Test that calculated indicators are preserved during fallback scenarios
4. **Data Consistency**: Verify backend calculations match API responses

EXPECTED RESULTS FROM PREVIOUS TEST:
- RSI: 100.0 with rsi_signal: "extreme_overbought" ✅
- MACD: 7.51e-08 with macd_trend: "bullish" ✅  
- Stochastic: 89.0% with stochastic_signal: "overbought" ✅
- MFI: 83.1% with mfi_signal: "overbought" ✅
- VWAP: 13.47% with vwap_signal: "extreme_overbought" ✅

SUCCESS CRITERIA:
✅ Technical indicators show real calculated values instead of defaults
✅ RSI, MACD, Stochastic, MFI, VWAP show meaningful non-default values
✅ Advanced signals show calculated values not "unknown" or "neutral"
✅ Error handling preserves calculated indicators during fallback scenarios
✅ 100% success rate for all technical indicators
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

class TechnicalIndicatorsTestSuite:
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
        logger.info(f"Testing Technical Indicators at: {self.api_url}")
        
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
            "BTCUSDT",   # High volatility, good for testing all indicators
            "ETHUSDT",   # Medium volatility, stable for testing
            "SOLUSDT",   # High volatility, trending
            "XRPUSDT",   # Lower volatility, edge cases
            "ADAUSDT"    # Medium volatility, range-bound
        ]
        
        # Expected technical indicators and their default values to avoid
        self.technical_indicators = {
            "RSI": {
                "field": "rsi_signal",
                "default_values": ["unknown", "neutral", "50.0", 50.0],
                "expected_values": ["oversold", "overbought", "extreme_oversold", "extreme_overbought"],
                "numeric_field": None  # RSI value is usually in analysis text
            },
            "MACD": {
                "field": "macd_trend", 
                "default_values": ["unknown", "neutral", "0.0", 0.0],
                "expected_values": ["bullish", "bearish"],
                "numeric_field": None  # MACD value is usually in analysis text
            },
            "Stochastic": {
                "field": "stochastic_signal",
                "default_values": ["unknown", "neutral", "50.0", 50.0],
                "expected_values": ["oversold", "overbought", "extreme_oversold", "extreme_overbought"],
                "numeric_field": None  # Stochastic value is usually in analysis text
            },
            "MFI": {
                "field": "mfi_signal",
                "default_values": ["unknown", "neutral", "50.0", 50.0],
                "expected_values": ["oversold", "overbought", "extreme_oversold", "extreme_overbought"],
                "numeric_field": None  # MFI value is usually in analysis text
            },
            "VWAP": {
                "field": "vwap_signal",
                "default_values": ["unknown", "neutral", "0.0", 0.0],
                "expected_values": ["oversold", "overbought", "extreme_oversold", "extreme_overbought"],
                "numeric_field": None  # VWAP value is usually in analysis text
            }
        }
        
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
    
    async def test_1_technical_indicators_calculation(self):
        """Test 1: Technical Indicators Calculation - Real Values vs Defaults"""
        logger.info("\n🔍 TEST 1: Technical Indicators Calculation Test")
        
        try:
            indicators_results = {
                'total_tests': 0,
                'indicators_working': 0,
                'indicators_failing': 0,
                'indicator_details': {},
                'failed_indicators': []
            }
            
            logger.info("   🚀 Testing all technical indicators for real calculated values...")
            logger.info("   📊 Expected Results:")
            logger.info("      - RSI: Real values with meaningful signals (not 'unknown')")
            logger.info("      - MACD: Real values with bullish/bearish trends (not 'unknown')")
            logger.info("      - Stochastic: Real values with overbought/oversold signals (not 'unknown')")
            logger.info("      - MFI: Real values with institutional flow signals (not 'unknown')")
            logger.info("      - VWAP: Real values with precision signals (not 'unknown')")
            
            # Test multiple symbols to get comprehensive results
            for symbol in self.test_symbols[:3]:  # Test first 3 symbols
                try:
                    logger.info(f"   📈 Testing technical indicators for {symbol}...")
                    
                    # Run IA1 cycle to get technical indicators
                    response = requests.post(
                        f"{self.api_url}/run-ia1-cycle",
                        json={"symbol": symbol},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        cycle_data = response.json()
                        
                        if cycle_data.get('success'):
                            analysis = cycle_data.get('analysis', {})
                            
                            # Test each technical indicator
                            for indicator_name, indicator_config in self.technical_indicators.items():
                                indicators_results['total_tests'] += 1
                                field_name = indicator_config['field']
                                default_values = indicator_config['default_values']
                                expected_values = indicator_config['expected_values']
                                
                                # Get the indicator value from analysis
                                indicator_value = analysis.get(field_name, "not_found")
                                
                                # Initialize indicator details if not exists
                                if indicator_name not in indicators_results['indicator_details']:
                                    indicators_results['indicator_details'][indicator_name] = {
                                        'tests': 0,
                                        'working': 0,
                                        'failing': 0,
                                        'values_found': [],
                                        'symbols_tested': []
                                    }
                                
                                indicators_results['indicator_details'][indicator_name]['tests'] += 1
                                indicators_results['indicator_details'][indicator_name]['symbols_tested'].append(symbol)
                                
                                # Check if indicator is working (not default/unknown values)
                                is_working = (
                                    indicator_value != "not_found" and
                                    indicator_value not in default_values and
                                    str(indicator_value).lower() not in [str(v).lower() for v in default_values]
                                )
                                
                                if is_working:
                                    indicators_results['indicators_working'] += 1
                                    indicators_results['indicator_details'][indicator_name]['working'] += 1
                                    indicators_results['indicator_details'][indicator_name]['values_found'].append(indicator_value)
                                    logger.info(f"      ✅ {indicator_name}: {field_name} = '{indicator_value}' (WORKING)")
                                else:
                                    indicators_results['indicators_failing'] += 1
                                    indicators_results['indicator_details'][indicator_name]['failing'] += 1
                                    indicators_results['failed_indicators'].append(f"{symbol}:{indicator_name}={indicator_value}")
                                    logger.warning(f"      ❌ {indicator_name}: {field_name} = '{indicator_value}' (DEFAULT/UNKNOWN)")
                            
                            # Also check analysis text for numeric values
                            analysis_text = analysis.get('analysis', '')
                            reasoning = analysis.get('reasoning', '')
                            full_text = f"{analysis_text} {reasoning}".lower()
                            
                            # Look for numeric indicator values in text
                            rsi_match = re.search(r'rsi[:\s]*(\d+\.?\d*)', full_text)
                            macd_match = re.search(r'macd[:\s]*(-?\d+\.?\d*e?-?\d*)', full_text)
                            stochastic_match = re.search(r'stochastic[:\s]*(\d+\.?\d*)', full_text)
                            mfi_match = re.search(r'mfi[:\s]*(\d+\.?\d*)', full_text)
                            vwap_match = re.search(r'vwap[:\s]*(-?\d+\.?\d*)', full_text)
                            
                            logger.info(f"      📊 Numeric values found in analysis:")
                            if rsi_match:
                                logger.info(f"         RSI: {rsi_match.group(1)}")
                            if macd_match:
                                logger.info(f"         MACD: {macd_match.group(1)}")
                            if stochastic_match:
                                logger.info(f"         Stochastic: {stochastic_match.group(1)}%")
                            if mfi_match:
                                logger.info(f"         MFI: {mfi_match.group(1)}%")
                            if vwap_match:
                                logger.info(f"         VWAP: {vwap_match.group(1)}%")
                        else:
                            logger.warning(f"      ❌ IA1 cycle failed for {symbol}: {cycle_data.get('error', 'Unknown error')}")
                    else:
                        logger.warning(f"      ❌ API call failed for {symbol}: HTTP {response.status_code}")
                    
                    await asyncio.sleep(3)  # Delay between requests
                    
                except Exception as e:
                    logger.error(f"   ❌ Error testing {symbol}: {e}")
            
            # Calculate overall results
            logger.info(f"   📊 Technical Indicators Test Results:")
            logger.info(f"      Total indicator tests: {indicators_results['total_tests']}")
            logger.info(f"      Working indicators: {indicators_results['indicators_working']}")
            logger.info(f"      Failing indicators: {indicators_results['indicators_failing']}")
            
            # Detailed results per indicator
            logger.info(f"   📋 Detailed Results by Indicator:")
            for indicator_name, details in indicators_results['indicator_details'].items():
                success_rate = details['working'] / details['tests'] if details['tests'] > 0 else 0
                status = "✅" if success_rate >= 0.8 else "⚠️" if success_rate >= 0.5 else "❌"
                logger.info(f"      {status} {indicator_name}: {details['working']}/{details['tests']} ({success_rate:.1%})")
                if details['values_found']:
                    unique_values = list(set(details['values_found']))
                    logger.info(f"         Values found: {unique_values}")
            
            if indicators_results['failed_indicators']:
                logger.info(f"   ❌ Failed Indicators:")
                for failure in indicators_results['failed_indicators'][:10]:  # Show first 10
                    logger.info(f"      - {failure}")
            
            # Determine test result
            if indicators_results['total_tests'] > 0:
                success_rate = indicators_results['indicators_working'] / indicators_results['total_tests']
                
                if success_rate >= 0.9:
                    self.log_test_result("Technical Indicators Calculation", True, 
                                       f"Technical indicators working correctly: {success_rate:.1%} success rate ({indicators_results['indicators_working']}/{indicators_results['total_tests']})")
                elif success_rate >= 0.7:
                    self.log_test_result("Technical Indicators Calculation", False, 
                                       f"Most technical indicators working: {success_rate:.1%} success rate ({indicators_results['indicators_working']}/{indicators_results['total_tests']})")
                else:
                    self.log_test_result("Technical Indicators Calculation", False, 
                                       f"Technical indicators issues: {success_rate:.1%} success rate ({indicators_results['indicators_working']}/{indicators_results['total_tests']})")
            else:
                self.log_test_result("Technical Indicators Calculation", False, 
                                   "No technical indicators tests could be performed")
                
        except Exception as e:
            self.log_test_result("Technical Indicators Calculation", False, f"Exception: {str(e)}")
    
    async def test_2_signal_quality_validation(self):
        """Test 2: Signal Quality - Meaningful Values vs Unknown/Neutral"""
        logger.info("\n🔍 TEST 2: Signal Quality Validation Test")
        
        try:
            signal_quality_results = {
                'total_signals_tested': 0,
                'meaningful_signals': 0,
                'unknown_neutral_signals': 0,
                'signal_categories': {
                    'RSI': {'meaningful': 0, 'unknown': 0, 'values': []},
                    'MACD': {'meaningful': 0, 'unknown': 0, 'values': []},
                    'Stochastic': {'meaningful': 0, 'unknown': 0, 'values': []},
                    'MFI': {'meaningful': 0, 'unknown': 0, 'values': []},
                    'VWAP': {'meaningful': 0, 'unknown': 0, 'values': []}
                }
            }
            
            logger.info("   🚀 Testing signal quality for meaningful categorization...")
            logger.info("   📊 Expected Signal Categories:")
            logger.info("      - RSI: oversold, overbought, extreme_oversold, extreme_overbought")
            logger.info("      - MACD: bullish, bearish")
            logger.info("      - Stochastic: oversold, overbought, extreme_oversold, extreme_overbought")
            logger.info("      - MFI: oversold, overbought (institutional flow)")
            logger.info("      - VWAP: oversold, overbought, extreme_oversold, extreme_overbought")
            
            # Test multiple symbols for signal quality
            for symbol in self.test_symbols[:3]:  # Test first 3 symbols
                try:
                    logger.info(f"   📈 Testing signal quality for {symbol}...")
                    
                    # Run IA1 cycle to get signals
                    response = requests.post(
                        f"{self.api_url}/run-ia1-cycle",
                        json={"symbol": symbol},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        cycle_data = response.json()
                        
                        if cycle_data.get('success'):
                            analysis = cycle_data.get('analysis', {})
                            
                            # Test signal quality for each indicator
                            for indicator_name, indicator_config in self.technical_indicators.items():
                                signal_quality_results['total_signals_tested'] += 1
                                field_name = indicator_config['field']
                                expected_values = indicator_config['expected_values']
                                
                                signal_value = analysis.get(field_name, "not_found")
                                
                                # Check if signal is meaningful
                                is_meaningful = (
                                    signal_value != "not_found" and
                                    signal_value not in ["unknown", "neutral", ""] and
                                    str(signal_value).lower() not in ["unknown", "neutral", ""]
                                )
                                
                                if is_meaningful:
                                    signal_quality_results['meaningful_signals'] += 1
                                    signal_quality_results['signal_categories'][indicator_name]['meaningful'] += 1
                                    signal_quality_results['signal_categories'][indicator_name]['values'].append(signal_value)
                                    logger.info(f"      ✅ {indicator_name}: '{signal_value}' (MEANINGFUL)")
                                else:
                                    signal_quality_results['unknown_neutral_signals'] += 1
                                    signal_quality_results['signal_categories'][indicator_name]['unknown'] += 1
                                    logger.warning(f"      ❌ {indicator_name}: '{signal_value}' (UNKNOWN/NEUTRAL)")
                        else:
                            logger.warning(f"      ❌ IA1 cycle failed for {symbol}: {cycle_data.get('error', 'Unknown error')}")
                    else:
                        logger.warning(f"      ❌ API call failed for {symbol}: HTTP {response.status_code}")
                    
                    await asyncio.sleep(3)  # Delay between requests
                    
                except Exception as e:
                    logger.error(f"   ❌ Error testing signal quality for {symbol}: {e}")
            
            # Calculate signal quality results
            logger.info(f"   📊 Signal Quality Test Results:")
            logger.info(f"      Total signals tested: {signal_quality_results['total_signals_tested']}")
            logger.info(f"      Meaningful signals: {signal_quality_results['meaningful_signals']}")
            logger.info(f"      Unknown/neutral signals: {signal_quality_results['unknown_neutral_signals']}")
            
            # Detailed results per indicator
            logger.info(f"   📋 Signal Quality by Indicator:")
            for indicator_name, details in signal_quality_results['signal_categories'].items():
                total = details['meaningful'] + details['unknown']
                if total > 0:
                    quality_rate = details['meaningful'] / total
                    status = "✅" if quality_rate >= 0.8 else "⚠️" if quality_rate >= 0.5 else "❌"
                    logger.info(f"      {status} {indicator_name}: {details['meaningful']}/{total} meaningful ({quality_rate:.1%})")
                    if details['values']:
                        unique_values = list(set(details['values']))
                        logger.info(f"         Signal values: {unique_values}")
            
            # Determine test result
            if signal_quality_results['total_signals_tested'] > 0:
                quality_rate = signal_quality_results['meaningful_signals'] / signal_quality_results['total_signals_tested']
                
                if quality_rate >= 0.9:
                    self.log_test_result("Signal Quality Validation", True, 
                                       f"Signal quality excellent: {quality_rate:.1%} meaningful signals ({signal_quality_results['meaningful_signals']}/{signal_quality_results['total_signals_tested']})")
                elif quality_rate >= 0.7:
                    self.log_test_result("Signal Quality Validation", False, 
                                       f"Signal quality good: {quality_rate:.1%} meaningful signals ({signal_quality_results['meaningful_signals']}/{signal_quality_results['total_signals_tested']})")
                else:
                    self.log_test_result("Signal Quality Validation", False, 
                                       f"Signal quality issues: {quality_rate:.1%} meaningful signals ({signal_quality_results['meaningful_signals']}/{signal_quality_results['total_signals_tested']})")
            else:
                self.log_test_result("Signal Quality Validation", False, 
                                   "No signals available for quality testing")
                
        except Exception as e:
            self.log_test_result("Signal Quality Validation", False, f"Exception: {str(e)}")
    
    async def test_3_error_handling_robustness(self):
        """Test 3: Error Handling - Calculated Indicators Preserved During Fallback"""
        logger.info("\n🔍 TEST 3: Error Handling Robustness Test")
        
        try:
            error_handling_results = {
                'total_tests': 0,
                'indicators_preserved': 0,
                'indicators_lost': 0,
                'fallback_scenarios': 0,
                'preservation_details': {}
            }
            
            logger.info("   🚀 Testing error handling robustness...")
            logger.info("   📊 Expected: Calculated indicators preserved even during fallback scenarios")
            
            # Test multiple symbols to trigger potential error scenarios
            for symbol in self.test_symbols[:2]:  # Test 2 symbols for error handling
                try:
                    logger.info(f"   📈 Testing error handling for {symbol}...")
                    
                    # Run multiple IA1 cycles to potentially trigger error conditions
                    for attempt in range(2):
                        try:
                            response = requests.post(
                                f"{self.api_url}/run-ia1-cycle",
                                json={"symbol": symbol},
                                timeout=30  # Shorter timeout to potentially trigger errors
                            )
                            
                            if response.status_code == 200:
                                cycle_data = response.json()
                                
                                if cycle_data.get('success'):
                                    analysis = cycle_data.get('analysis', {})
                                    
                                    # Check if any indicators show calculated values despite potential errors
                                    for indicator_name, indicator_config in self.technical_indicators.items():
                                        error_handling_results['total_tests'] += 1
                                        field_name = indicator_config['field']
                                        default_values = indicator_config['default_values']
                                        
                                        indicator_value = analysis.get(field_name, "not_found")
                                        
                                        # Check if indicator has calculated value (not default)
                                        has_calculated_value = (
                                            indicator_value != "not_found" and
                                            indicator_value not in default_values and
                                            str(indicator_value).lower() not in [str(v).lower() for v in default_values]
                                        )
                                        
                                        if has_calculated_value:
                                            error_handling_results['indicators_preserved'] += 1
                                            logger.info(f"      ✅ {indicator_name}: Preserved value '{indicator_value}'")
                                        else:
                                            error_handling_results['indicators_lost'] += 1
                                            logger.warning(f"      ⚠️ {indicator_name}: Default/unknown value '{indicator_value}'")
                                        
                                        # Track preservation details
                                        if indicator_name not in error_handling_results['preservation_details']:
                                            error_handling_results['preservation_details'][indicator_name] = {
                                                'preserved': 0, 'lost': 0, 'values': []
                                            }
                                        
                                        if has_calculated_value:
                                            error_handling_results['preservation_details'][indicator_name]['preserved'] += 1
                                            error_handling_results['preservation_details'][indicator_name]['values'].append(indicator_value)
                                        else:
                                            error_handling_results['preservation_details'][indicator_name]['lost'] += 1
                                
                                # Check for fallback scenarios in logs or response
                                if 'fallback' in str(cycle_data).lower() or 'error' in str(cycle_data).lower():
                                    error_handling_results['fallback_scenarios'] += 1
                                    logger.info(f"      📋 Fallback scenario detected for {symbol} attempt {attempt + 1}")
                            
                            await asyncio.sleep(2)  # Short delay between attempts
                            
                        except Exception as e:
                            logger.info(f"      📋 Error scenario triggered for {symbol} attempt {attempt + 1}: {e}")
                            error_handling_results['fallback_scenarios'] += 1
                    
                except Exception as e:
                    logger.error(f"   ❌ Error testing error handling for {symbol}: {e}")
            
            # Calculate error handling results
            logger.info(f"   📊 Error Handling Test Results:")
            logger.info(f"      Total indicator tests: {error_handling_results['total_tests']}")
            logger.info(f"      Indicators preserved: {error_handling_results['indicators_preserved']}")
            logger.info(f"      Indicators lost to defaults: {error_handling_results['indicators_lost']}")
            logger.info(f"      Fallback scenarios detected: {error_handling_results['fallback_scenarios']}")
            
            # Detailed preservation results
            logger.info(f"   📋 Preservation Details by Indicator:")
            for indicator_name, details in error_handling_results['preservation_details'].items():
                total = details['preserved'] + details['lost']
                if total > 0:
                    preservation_rate = details['preserved'] / total
                    status = "✅" if preservation_rate >= 0.8 else "⚠️" if preservation_rate >= 0.5 else "❌"
                    logger.info(f"      {status} {indicator_name}: {details['preserved']}/{total} preserved ({preservation_rate:.1%})")
            
            # Determine test result
            if error_handling_results['total_tests'] > 0:
                preservation_rate = error_handling_results['indicators_preserved'] / error_handling_results['total_tests']
                
                if preservation_rate >= 0.8:
                    self.log_test_result("Error Handling Robustness", True, 
                                       f"Error handling robust: {preservation_rate:.1%} indicators preserved ({error_handling_results['indicators_preserved']}/{error_handling_results['total_tests']})")
                elif preservation_rate >= 0.6:
                    self.log_test_result("Error Handling Robustness", False, 
                                       f"Error handling partial: {preservation_rate:.1%} indicators preserved ({error_handling_results['indicators_preserved']}/{error_handling_results['total_tests']})")
                else:
                    self.log_test_result("Error Handling Robustness", False, 
                                       f"Error handling issues: {preservation_rate:.1%} indicators preserved ({error_handling_results['indicators_preserved']}/{error_handling_results['total_tests']})")
            else:
                self.log_test_result("Error Handling Robustness", False, 
                                   "No error handling tests could be performed")
                
        except Exception as e:
            self.log_test_result("Error Handling Robustness", False, f"Exception: {str(e)}")
    
    async def test_4_data_consistency_validation(self):
        """Test 4: Data Consistency - Backend Calculations Match API Responses"""
        logger.info("\n🔍 TEST 4: Data Consistency Validation Test")
        
        try:
            consistency_results = {
                'total_consistency_checks': 0,
                'consistent_indicators': 0,
                'inconsistent_indicators': 0,
                'backend_log_matches': 0,
                'api_response_matches': 0,
                'consistency_details': {}
            }
            
            logger.info("   🚀 Testing data consistency between backend calculations and API responses...")
            logger.info("   📊 Expected: Backend logs should match API response values")
            
            # Test symbols for data consistency
            for symbol in self.test_symbols[:2]:  # Test 2 symbols for consistency
                try:
                    logger.info(f"   📈 Testing data consistency for {symbol}...")
                    
                    # Capture backend logs before API call
                    log_content_before = await self._capture_backend_logs()
                    
                    # Run IA1 cycle
                    response = requests.post(
                        f"{self.api_url}/run-ia1-cycle",
                        json={"symbol": symbol},
                        timeout=60
                    )
                    
                    # Capture backend logs after API call
                    await asyncio.sleep(2)  # Wait for logs to be written
                    log_content_after = await self._capture_backend_logs()
                    
                    if response.status_code == 200:
                        cycle_data = response.json()
                        
                        if cycle_data.get('success'):
                            analysis = cycle_data.get('analysis', {})
                            
                            # Extract new log entries
                            new_log_entries = self._extract_new_log_entries(log_content_before, log_content_after)
                            
                            # Check consistency for each indicator
                            for indicator_name, indicator_config in self.technical_indicators.items():
                                consistency_results['total_consistency_checks'] += 1
                                field_name = indicator_config['field']
                                
                                api_value = analysis.get(field_name, "not_found")
                                
                                # Look for indicator values in backend logs
                                log_values = self._extract_indicator_from_logs(indicator_name, new_log_entries)
                                
                                # Check consistency
                                is_consistent = self._check_consistency(api_value, log_values, indicator_name)
                                
                                if is_consistent:
                                    consistency_results['consistent_indicators'] += 1
                                    logger.info(f"      ✅ {indicator_name}: API='{api_value}' matches backend logs")
                                else:
                                    consistency_results['inconsistent_indicators'] += 1
                                    logger.warning(f"      ❌ {indicator_name}: API='{api_value}' vs Backend={log_values}")
                                
                                # Track consistency details
                                if indicator_name not in consistency_results['consistency_details']:
                                    consistency_results['consistency_details'][indicator_name] = {
                                        'consistent': 0, 'inconsistent': 0, 'api_values': [], 'log_values': []
                                    }
                                
                                if is_consistent:
                                    consistency_results['consistency_details'][indicator_name]['consistent'] += 1
                                else:
                                    consistency_results['consistency_details'][indicator_name]['inconsistent'] += 1
                                
                                consistency_results['consistency_details'][indicator_name]['api_values'].append(api_value)
                                consistency_results['consistency_details'][indicator_name]['log_values'].extend(log_values)
                    
                    await asyncio.sleep(3)  # Delay between requests
                    
                except Exception as e:
                    logger.error(f"   ❌ Error testing data consistency for {symbol}: {e}")
            
            # Calculate consistency results
            logger.info(f"   📊 Data Consistency Test Results:")
            logger.info(f"      Total consistency checks: {consistency_results['total_consistency_checks']}")
            logger.info(f"      Consistent indicators: {consistency_results['consistent_indicators']}")
            logger.info(f"      Inconsistent indicators: {consistency_results['inconsistent_indicators']}")
            
            # Detailed consistency results
            logger.info(f"   📋 Consistency Details by Indicator:")
            for indicator_name, details in consistency_results['consistency_details'].items():
                total = details['consistent'] + details['inconsistent']
                if total > 0:
                    consistency_rate = details['consistent'] / total
                    status = "✅" if consistency_rate >= 0.8 else "⚠️" if consistency_rate >= 0.5 else "❌"
                    logger.info(f"      {status} {indicator_name}: {details['consistent']}/{total} consistent ({consistency_rate:.1%})")
            
            # Determine test result
            if consistency_results['total_consistency_checks'] > 0:
                consistency_rate = consistency_results['consistent_indicators'] / consistency_results['total_consistency_checks']
                
                if consistency_rate >= 0.8:
                    self.log_test_result("Data Consistency Validation", True, 
                                       f"Data consistency excellent: {consistency_rate:.1%} consistent ({consistency_results['consistent_indicators']}/{consistency_results['total_consistency_checks']})")
                elif consistency_rate >= 0.6:
                    self.log_test_result("Data Consistency Validation", False, 
                                       f"Data consistency partial: {consistency_rate:.1%} consistent ({consistency_results['consistent_indicators']}/{consistency_results['total_consistency_checks']})")
                else:
                    self.log_test_result("Data Consistency Validation", False, 
                                       f"Data consistency issues: {consistency_rate:.1%} consistent ({consistency_results['consistent_indicators']}/{consistency_results['total_consistency_checks']})")
            else:
                self.log_test_result("Data Consistency Validation", False, 
                                   "No data consistency tests could be performed")
                
        except Exception as e:
            self.log_test_result("Data Consistency Validation", False, f"Exception: {str(e)}")
    
    async def test_5_comprehensive_indicators_validation(self):
        """Test 5: Comprehensive Validation - All Expected Results from Review Request"""
        logger.info("\n🔍 TEST 5: Comprehensive Indicators Validation Test")
        
        try:
            comprehensive_results = {
                'expected_results_found': 0,
                'expected_results_missing': 0,
                'total_expected': 5,  # RSI, MACD, Stochastic, MFI, VWAP
                'detailed_findings': {},
                'success_rate': 0.0
            }
            
            logger.info("   🚀 Testing for specific expected results from review request...")
            logger.info("   📊 Expected Results to Validate:")
            logger.info("      - RSI: 100.0 with rsi_signal: 'extreme_overbought' ✅")
            logger.info("      - MACD: 7.51e-08 with macd_trend: 'bullish' ✅")
            logger.info("      - Stochastic: 89.0% with stochastic_signal: 'overbought' ✅")
            logger.info("      - MFI: 83.1% with mfi_signal: 'overbought' ✅")
            logger.info("      - VWAP: 13.47% with vwap_signal: 'extreme_overbought' ✅")
            
            # Expected results mapping
            expected_results = {
                'RSI': {'signal_field': 'rsi_signal', 'expected_signals': ['extreme_overbought', 'overbought'], 'numeric_range': [80, 100]},
                'MACD': {'signal_field': 'macd_trend', 'expected_signals': ['bullish'], 'numeric_range': [0, 1]},
                'Stochastic': {'signal_field': 'stochastic_signal', 'expected_signals': ['overbought'], 'numeric_range': [80, 100]},
                'MFI': {'signal_field': 'mfi_signal', 'expected_signals': ['overbought'], 'numeric_range': [80, 100]},
                'VWAP': {'signal_field': 'vwap_signal', 'expected_signals': ['extreme_overbought', 'overbought'], 'numeric_range': [10, 20]}
            }
            
            # Test multiple symbols to find expected results
            for symbol in self.test_symbols:  # Test all symbols
                try:
                    logger.info(f"   📈 Testing comprehensive validation for {symbol}...")
                    
                    # Run IA1 cycle
                    response = requests.post(
                        f"{self.api_url}/run-ia1-cycle",
                        json={"symbol": symbol},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        cycle_data = response.json()
                        
                        if cycle_data.get('success'):
                            analysis = cycle_data.get('analysis', {})
                            analysis_text = analysis.get('analysis', '')
                            reasoning = analysis.get('reasoning', '')
                            full_text = f"{analysis_text} {reasoning}".lower()
                            
                            # Check each expected result
                            for indicator_name, expected in expected_results.items():
                                signal_field = expected['signal_field']
                                expected_signals = expected['expected_signals']
                                numeric_range = expected['numeric_range']
                                
                                # Get signal value
                                signal_value = analysis.get(signal_field, "not_found")
                                
                                # Extract numeric value from text
                                numeric_value = self._extract_numeric_value(indicator_name, full_text)
                                
                                # Check if results match expectations
                                signal_matches = signal_value in expected_signals
                                numeric_in_range = (numeric_value is not None and 
                                                  numeric_range[0] <= numeric_value <= numeric_range[1])
                                
                                # Initialize detailed findings
                                if indicator_name not in comprehensive_results['detailed_findings']:
                                    comprehensive_results['detailed_findings'][indicator_name] = {
                                        'signal_matches': 0, 'numeric_matches': 0, 'total_tests': 0,
                                        'signals_found': [], 'numeric_values': []
                                    }
                                
                                comprehensive_results['detailed_findings'][indicator_name]['total_tests'] += 1
                                
                                if signal_matches:
                                    comprehensive_results['detailed_findings'][indicator_name]['signal_matches'] += 1
                                    comprehensive_results['detailed_findings'][indicator_name]['signals_found'].append(signal_value)
                                
                                if numeric_in_range:
                                    comprehensive_results['detailed_findings'][indicator_name]['numeric_matches'] += 1
                                    comprehensive_results['detailed_findings'][indicator_name]['numeric_values'].append(numeric_value)
                                
                                # Log findings
                                if signal_matches or numeric_in_range:
                                    logger.info(f"      ✅ {indicator_name}: Signal='{signal_value}', Numeric={numeric_value}")
                                else:
                                    logger.info(f"      ⚪ {indicator_name}: Signal='{signal_value}', Numeric={numeric_value}")
                    
                    await asyncio.sleep(3)  # Delay between requests
                    
                except Exception as e:
                    logger.error(f"   ❌ Error in comprehensive validation for {symbol}: {e}")
            
            # Calculate comprehensive results
            logger.info(f"   📊 Comprehensive Validation Results:")
            
            indicators_meeting_expectations = 0
            for indicator_name, findings in comprehensive_results['detailed_findings'].items():
                total_tests = findings['total_tests']
                signal_rate = findings['signal_matches'] / total_tests if total_tests > 0 else 0
                numeric_rate = findings['numeric_matches'] / total_tests if total_tests > 0 else 0
                
                # Consider indicator successful if either signal or numeric criteria are met
                meets_expectations = signal_rate >= 0.3 or numeric_rate >= 0.3  # At least 30% success
                
                if meets_expectations:
                    indicators_meeting_expectations += 1
                    status = "✅"
                else:
                    status = "❌"
                
                logger.info(f"      {status} {indicator_name}: Signal matches={signal_rate:.1%}, Numeric matches={numeric_rate:.1%}")
                if findings['signals_found']:
                    logger.info(f"         Signals found: {list(set(findings['signals_found']))}")
                if findings['numeric_values']:
                    logger.info(f"         Numeric values: {findings['numeric_values'][:5]}")  # Show first 5
            
            comprehensive_results['expected_results_found'] = indicators_meeting_expectations
            comprehensive_results['expected_results_missing'] = comprehensive_results['total_expected'] - indicators_meeting_expectations
            comprehensive_results['success_rate'] = indicators_meeting_expectations / comprehensive_results['total_expected']
            
            logger.info(f"      Expected results found: {comprehensive_results['expected_results_found']}/{comprehensive_results['total_expected']}")
            logger.info(f"      Overall success rate: {comprehensive_results['success_rate']:.1%}")
            
            # Determine test result
            if comprehensive_results['success_rate'] >= 0.8:
                self.log_test_result("Comprehensive Indicators Validation", True, 
                                   f"Comprehensive validation successful: {comprehensive_results['success_rate']:.1%} indicators meeting expectations ({comprehensive_results['expected_results_found']}/{comprehensive_results['total_expected']})")
            elif comprehensive_results['success_rate'] >= 0.6:
                self.log_test_result("Comprehensive Indicators Validation", False, 
                                   f"Partial comprehensive validation: {comprehensive_results['success_rate']:.1%} indicators meeting expectations ({comprehensive_results['expected_results_found']}/{comprehensive_results['total_expected']})")
            else:
                self.log_test_result("Comprehensive Indicators Validation", False, 
                                   f"Comprehensive validation issues: {comprehensive_results['success_rate']:.1%} indicators meeting expectations ({comprehensive_results['expected_results_found']}/{comprehensive_results['total_expected']})")
                
        except Exception as e:
            self.log_test_result("Comprehensive Indicators Validation", False, f"Exception: {str(e)}")
    
    async def _capture_backend_logs(self):
        """Capture recent backend logs"""
        try:
            log_files = [
                "/var/log/supervisor/backend.out.log",
                "/var/log/supervisor/backend.err.log"
            ]
            
            all_logs = []
            for log_file in log_files:
                if os.path.exists(log_file):
                    try:
                        result = subprocess.run(['tail', '-n', '100', log_file], 
                                              capture_output=True, text=True, timeout=10)
                        if result.returncode == 0:
                            all_logs.extend(result.stdout.split('\n'))
                    except Exception:
                        pass
            
            return all_logs
        except Exception:
            return []
    
    def _extract_new_log_entries(self, before_logs, after_logs):
        """Extract new log entries"""
        try:
            before_set = set(before_logs)
            new_entries = [log for log in after_logs if log not in before_set]
            return new_entries
        except Exception:
            return after_logs  # Return all if comparison fails
    
    def _extract_indicator_from_logs(self, indicator_name, log_entries):
        """Extract indicator values from log entries"""
        values = []
        try:
            indicator_patterns = {
                'RSI': [r'rsi[:\s]*(\d+\.?\d*)', r'rsi.*signal[:\s]*(\w+)'],
                'MACD': [r'macd[:\s]*(-?\d+\.?\d*e?-?\d*)', r'macd.*trend[:\s]*(\w+)'],
                'Stochastic': [r'stochastic[:\s]*(\d+\.?\d*)', r'stochastic.*signal[:\s]*(\w+)'],
                'MFI': [r'mfi[:\s]*(\d+\.?\d*)', r'mfi.*signal[:\s]*(\w+)'],
                'VWAP': [r'vwap[:\s]*(-?\d+\.?\d*)', r'vwap.*signal[:\s]*(\w+)']
            }
            
            patterns = indicator_patterns.get(indicator_name, [])
            for log_entry in log_entries:
                for pattern in patterns:
                    matches = re.findall(pattern, log_entry.lower())
                    values.extend(matches)
            
            return values
        except Exception:
            return []
    
    def _check_consistency(self, api_value, log_values, indicator_name):
        """Check consistency between API value and log values"""
        try:
            if not log_values:
                return False  # No log values to compare
            
            # Convert API value to string for comparison
            api_str = str(api_value).lower()
            
            # Check if API value appears in log values
            for log_value in log_values:
                log_str = str(log_value).lower()
                if api_str == log_str or api_str in log_str or log_str in api_str:
                    return True
            
            return False
        except Exception:
            return False
    
    def _extract_numeric_value(self, indicator_name, text):
        """Extract numeric value for indicator from text"""
        try:
            patterns = {
                'RSI': r'rsi[:\s]*(\d+\.?\d*)',
                'MACD': r'macd[:\s]*(-?\d+\.?\d*e?-?\d*)',
                'Stochastic': r'stochastic[:\s]*(\d+\.?\d*)',
                'MFI': r'mfi[:\s]*(\d+\.?\d*)',
                'VWAP': r'vwap[:\s]*(-?\d+\.?\d*)'
            }
            
            pattern = patterns.get(indicator_name)
            if pattern:
                match = re.search(pattern, text)
                if match:
                    return float(match.group(1))
            
            return None
        except Exception:
            return None
    
    async def run_comprehensive_test_suite(self):
        """Run comprehensive technical indicators test suite"""
        logger.info("🚀 Starting Technical Indicators Fix Validation Test Suite")
        logger.info("=" * 80)
        logger.info("📋 TECHNICAL INDICATORS FIX VALIDATION TEST SUITE")
        logger.info("🎯 Testing: Complete technical indicators fix validation")
        logger.info("🎯 Expected: 100% success rate for all technical indicators")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_technical_indicators_calculation()
        await self.test_2_signal_quality_validation()
        await self.test_3_error_handling_robustness()
        await self.test_4_data_consistency_validation()
        await self.test_5_comprehensive_indicators_validation()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 TECHNICAL INDICATORS FIX VALIDATION TEST SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # Critical requirements analysis
        logger.info("\n" + "=" * 80)
        logger.info("📋 CRITICAL REQUIREMENTS VERIFICATION")
        logger.info("=" * 80)
        
        requirements_status = {}
        
        for result in self.test_results:
            if "Technical Indicators Calculation" in result['test']:
                requirements_status['All Technical Indicators Working'] = result['success']
            elif "Signal Quality" in result['test']:
                requirements_status['Signal Quality - Meaningful Values'] = result['success']
            elif "Error Handling" in result['test']:
                requirements_status['Error Handling Robustness'] = result['success']
            elif "Data Consistency" in result['test']:
                requirements_status['Backend-API Data Consistency'] = result['success']
            elif "Comprehensive" in result['test']:
                requirements_status['Expected Results Validation'] = result['success']
        
        logger.info("🎯 CRITICAL REQUIREMENTS STATUS:")
        for requirement, status in requirements_status.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {requirement}")
        
        requirements_met = sum(1 for status in requirements_status.values() if status)
        total_requirements = len(requirements_status)
        
        # Final verdict
        logger.info(f"\n🏆 REQUIREMENTS SATISFACTION: {requirements_met}/{total_requirements}")
        
        if requirements_met == total_requirements:
            logger.info("\n🎉 VERDICT: TECHNICAL INDICATORS FIX 100% SUCCESSFUL!")
            logger.info("✅ All technical indicators working with calculated values")
            logger.info("✅ Signal quality excellent - meaningful categorization")
            logger.info("✅ Error handling robust - indicators preserved during fallback")
            logger.info("✅ Data consistency confirmed - backend matches API responses")
            logger.info("✅ Expected results validated - RSI, MACD, Stochastic, MFI, VWAP")
            logger.info("✅ System meets all technical indicators requirements")
        elif requirements_met >= total_requirements * 0.8:
            logger.info("\n⚠️ VERDICT: TECHNICAL INDICATORS FIX MOSTLY SUCCESSFUL")
            logger.info("🔍 Minor issues may need attention for complete fix")
        elif requirements_met >= total_requirements * 0.6:
            logger.info("\n⚠️ VERDICT: TECHNICAL INDICATORS FIX PARTIALLY SUCCESSFUL")
            logger.info("🔧 Several requirements need attention for complete fix")
        else:
            logger.info("\n❌ VERDICT: TECHNICAL INDICATORS FIX NOT SUCCESSFUL")
            logger.info("🚨 Major issues detected - technical indicators still not working properly")
            logger.info("🚨 System needs significant fixes for technical indicators")
        
        return passed_tests, total_tests

async def main():
    """Main function to run the comprehensive technical indicators test suite"""
    test_suite = TechnicalIndicatorsTestSuite()
    passed_tests, total_tests = await test_suite.run_comprehensive_test_suite()
    
    # Exit with appropriate code
    if passed_tests == total_tests:
        sys.exit(0)  # All tests passed
    elif passed_tests >= total_tests * 0.8:
        sys.exit(1)  # Mostly successful
    else:
        sys.exit(2)  # Major issues

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
MFI AND STOCHASTIC INDICATORS REMOVAL TESTING SUITE
Focus: Test the trading bot system after removal of MFI and Stochastic indicators.

CRITICAL VALIDATION POINTS:

1. **API Force IA1 Analysis**:
   - Test with 3-4 symbols (BTCUSDT, ETHUSDT, SOLUSDT, LINKUSDT) to verify:
   - No "NameError: 'mfi' is not defined" errors
   - No "NameError: 'stochastic_k' is not defined" errors
   - IA1 analyses complete successfully
   - Valid JSON responses with all required fields

2. **API Opportunities**:
   - Verify scout system works: /api/opportunities returns data
   - Opportunities contain correct technical indicators
   - No MFI/stochastic references in responses

3. **Validation of Indicators**:
   - In IA1 responses, verify RSI, MACD, ATR are present and calculated
   - No MFI or stochastic fields in responses
   - Values are realistic (not fallbacks like 50.0 everywhere)

4. **Backend Logs**:
   - Verify no more MFI errors in logs
   - No stochastic errors in logs
   - No NameError in recent logs

5. **Test IA2 Escalation**:
   - If possible, test that high confidence or good RR triggers IA2

FOCUS: Confirm all systems work correctly after MFI and stochastic removal.
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

class MFIStochasticRemovalTestSuite:
    """Comprehensive test suite for MFI and Stochastic indicators removal validation"""
    
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
        logger.info(f"Testing MFI and Stochastic Indicators Removal at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Test symbols for analysis (from review request)
        self.test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LINKUSDT']  # Specific symbols from review request
        self.actual_test_symbols = []  # Will be populated from available opportunities
        
        # Expected indicators that should be present (after MFI/Stochastic removal)
        self.expected_indicators = ['RSI', 'MACD', 'ATR', 'VWAP', 'ADX', 'BB']
        
        # Indicators that should NOT be present (removed indicators)
        self.removed_indicators = ['MFI', 'stochastic', 'stoch_k', 'stoch_d', 'mfi']
        
        # Error patterns to check for in logs
        self.error_patterns = [
            "NameError: name 'mfi' is not defined",
            "NameError: name 'stochastic_k' is not defined", 
            "NameError: name 'stochastic_d' is not defined",
            "NameError: 'mfi' is not defined",
            "NameError: 'stochastic_k' is not defined",
            "mfi is not defined",
            "stochastic_k is not defined"
        ]
        
        # Technical analysis data storage
        self.technical_analyses = []
        self.backend_logs = []
        
        # Database connection info
        self.mongo_url = "mongodb://localhost:27017"
        self.db_name = "myapp"
        
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
    
    async def test_1_api_force_ia1_analysis(self):
        """Test 1: API Force IA1 Analysis - Test with 3-4 symbols to verify no MFI/Stochastic errors"""
        logger.info("\n🔍 TEST 1: API Force IA1 Analysis - MFI/Stochastic Removal Validation")
        
        try:
            analysis_results = {
                'analyses_attempted': 0,
                'analyses_successful': 0,
                'mfi_errors_found': 0,
                'stochastic_errors_found': 0,
                'valid_json_responses': 0,
                'required_indicators_present': 0,
                'removed_indicators_absent': 0,
                'successful_analyses': [],
                'error_details': [],
                'response_times': []
            }
            
            logger.info("   🚀 Testing IA1 analysis with specific symbols to verify no MFI/Stochastic errors...")
            logger.info("   📊 Expected: No NameError for 'mfi' or 'stochastic_k', successful analyses with valid JSON")
            
            # Get available symbols from scout system
            logger.info("   📞 Getting available symbols from scout system...")
            
            try:
                response = requests.get(f"{self.api_url}/opportunities", timeout=60)
                
                if response.status_code == 200:
                    opportunities = response.json()
                    if isinstance(opportunities, dict) and 'opportunities' in opportunities:
                        opportunities = opportunities['opportunities']
                    
                    # Get available symbols for testing
                    available_symbols = [opp.get('symbol') for opp in opportunities[:15] if opp.get('symbol')]
                    
                    # Prefer specific symbols from review request
                    preferred_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LINKUSDT']
                    test_symbols = []
                    
                    for symbol in preferred_symbols:
                        if symbol in available_symbols:
                            test_symbols.append(symbol)
                    
                    # Fill remaining slots with available symbols if needed
                    for symbol in available_symbols:
                        if symbol not in test_symbols and len(test_symbols) < 4:
                            test_symbols.append(symbol)
                    
                    self.actual_test_symbols = test_symbols[:4]  # Limit to 4 symbols as per review request
                    logger.info(f"      ✅ Test symbols selected: {self.actual_test_symbols}")
                    
                else:
                    logger.warning(f"      ⚠️ Could not get opportunities, using default symbols")
                    self.actual_test_symbols = self.test_symbols
                    
            except Exception as e:
                logger.warning(f"      ⚠️ Error getting opportunities: {e}, using default symbols")
                self.actual_test_symbols = self.test_symbols
            
            # Test each symbol for MFI/Stochastic removal validation
            for symbol in self.actual_test_symbols:
                logger.info(f"\n   📞 Testing IA1 analysis for {symbol} - checking for MFI/Stochastic errors...")
                analysis_results['analyses_attempted'] += 1
                
                try:
                    start_time = time.time()
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    response_time = time.time() - start_time
                    analysis_results['response_times'].append(response_time)
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        analysis_results['analyses_successful'] += 1
                        analysis_results['valid_json_responses'] += 1
                        
                        logger.info(f"      ✅ {symbol} analysis successful (response time: {response_time:.2f}s)")
                        
                        # Extract IA1 analysis data
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        if not isinstance(ia1_analysis, dict):
                            ia1_analysis = {}
                        
                        # Check for required indicators (should be present)
                        required_indicators_found = []
                        for indicator in self.expected_indicators:
                            # Check various field patterns for each indicator
                            if indicator == 'RSI':
                                if any(key for key in ia1_analysis.keys() if 'rsi' in key.lower()):
                                    required_indicators_found.append(indicator)
                            elif indicator == 'MACD':
                                if any(key for key in ia1_analysis.keys() if 'macd' in key.lower()):
                                    required_indicators_found.append(indicator)
                            elif indicator == 'ATR':
                                if any(key for key in ia1_analysis.keys() if 'atr' in key.lower()):
                                    required_indicators_found.append(indicator)
                            elif indicator == 'VWAP':
                                if any(key for key in ia1_analysis.keys() if 'vwap' in key.lower()):
                                    required_indicators_found.append(indicator)
                            elif indicator == 'ADX':
                                if any(key for key in ia1_analysis.keys() if 'adx' in key.lower()):
                                    required_indicators_found.append(indicator)
                            elif indicator == 'BB':
                                if any(key for key in ia1_analysis.keys() if 'bb' in key.lower() or 'bollinger' in key.lower()):
                                    required_indicators_found.append(indicator)
                        
                        if len(required_indicators_found) >= 4:  # At least 4 of 6 expected indicators
                            analysis_results['required_indicators_present'] += 1
                            logger.info(f"         ✅ Required indicators present: {required_indicators_found}")
                        else:
                            logger.warning(f"         ⚠️ Missing required indicators. Found: {required_indicators_found}")
                        
                        # Check for removed indicators (should NOT be present)
                        removed_indicators_found = []
                        for indicator in self.removed_indicators:
                            if any(key for key in ia1_analysis.keys() if indicator.lower() in key.lower()):
                                removed_indicators_found.append(indicator)
                        
                        if len(removed_indicators_found) == 0:
                            analysis_results['removed_indicators_absent'] += 1
                            logger.info(f"         ✅ Removed indicators correctly absent")
                        else:
                            logger.warning(f"         ❌ Removed indicators still present: {removed_indicators_found}")
                        
                        # Store successful analysis details
                        analysis_results['successful_analyses'].append({
                            'symbol': symbol,
                            'response_time': response_time,
                            'required_indicators': required_indicators_found,
                            'removed_indicators': removed_indicators_found,
                            'analysis_data': ia1_analysis
                        })
                        
                    elif response.status_code == 500:
                        # Check for specific MFI/Stochastic errors
                        error_text = response.text
                        logger.error(f"      ❌ {symbol} analysis failed: HTTP 500")
                        logger.error(f"         Error response: {error_text[:500]}")
                        
                        # Check for MFI errors
                        if any(pattern in error_text for pattern in ["mfi", "MFI"]):
                            analysis_results['mfi_errors_found'] += 1
                            logger.error(f"         🚨 MFI ERROR DETECTED in {symbol}")
                        
                        # Check for Stochastic errors  
                        if any(pattern in error_text for pattern in ["stochastic", "stoch_k", "stoch_d"]):
                            analysis_results['stochastic_errors_found'] += 1
                            logger.error(f"         🚨 STOCHASTIC ERROR DETECTED in {symbol}")
                        
                        analysis_results['error_details'].append({
                            'symbol': symbol,
                            'error_type': 'HTTP_500',
                            'error_text': error_text[:500],
                            'has_mfi_error': any(pattern in error_text for pattern in ["mfi", "MFI"]),
                            'has_stochastic_error': any(pattern in error_text for pattern in ["stochastic", "stoch_k", "stoch_d"])
                        })
                        
                    else:
                        logger.error(f"      ❌ {symbol} analysis failed: HTTP {response.status_code}")
                        if response.text:
                            error_text = response.text[:300]
                            logger.error(f"         Error response: {error_text}")
                            analysis_results['error_details'].append({
                                'symbol': symbol,
                                'error_type': f'HTTP_{response.status_code}',
                                'error_text': error_text
                            })
                
                except Exception as e:
                    logger.error(f"      ❌ {symbol} analysis exception: {e}")
                    analysis_results['error_details'].append({
                        'symbol': symbol,
                        'error_type': 'EXCEPTION',
                        'error_text': str(e)
                    })
                
                # Wait between analyses
                if symbol != self.actual_test_symbols[-1]:
                    logger.info(f"      ⏳ Waiting 5 seconds before next analysis...")
                    await asyncio.sleep(5)
                            logger.warning(f"         ⚠️ Old field names still present: recommended_trade_type={old_trade_type}, minimum_rr_for_trade_type={old_rr_threshold}")
                        
                        # Store analysis details
                        field_validation_results['successful_analyses'].append({
                            'symbol': symbol,
                            'trade_type': trade_type,
                            'minimum_rr_threshold': minimum_rr_threshold,
                            'old_trade_type': old_trade_type,
                            'old_rr_threshold': old_rr_threshold,
                            'response_time': response_time
                        })
                        
                        # Store field validation details
                        field_validation_results['field_validation_details'].append({
                            'symbol': symbol,
                            'new_fields_present': trade_type is not None and minimum_rr_threshold is not None,
                            'old_fields_absent': old_trade_type is None and old_rr_threshold is None,
                            'trade_type_valid': trade_type in self.valid_trade_types if trade_type else False,
                            'rr_threshold_valid': isinstance(minimum_rr_threshold, (int, float)) and minimum_rr_threshold > 0 if minimum_rr_threshold else False
                        })
                        
                    else:
                        logger.error(f"      ❌ {symbol} analysis failed: HTTP {response.status_code}")
                        if response.text:
                            logger.error(f"         Error response: {response.text[:300]}")
                
                except Exception as e:
                    logger.error(f"      ❌ {symbol} analysis exception: {e}")
                
                # Wait between analyses
                if symbol != self.actual_test_symbols[-1]:
                    logger.info(f"      ⏳ Waiting 10 seconds before next analysis...")
                    await asyncio.sleep(10)
            
            # Capture backend logs to check for MFI/Stochastic errors
            logger.info("   📋 Capturing backend logs to check for MFI/Stochastic errors...")
            
            try:
                backend_logs = await self._capture_backend_logs()
                if backend_logs:
                    # Look for MFI/Stochastic error patterns
                    mfi_error_logs = []
                    stochastic_error_logs = []
                    nameerror_logs = []
                    success_logs = []
                    
                    for log_line in backend_logs:
                        log_lower = log_line.lower()
                        
                        # Check for MFI errors
                        if any(pattern in log_line for pattern in ["NameError: name 'mfi'", "mfi is not defined", "NameError: 'mfi'"]):
                            mfi_error_logs.append(log_line.strip())
                            analysis_results['mfi_errors_found'] += 1
                        
                        # Check for Stochastic errors
                        if any(pattern in log_line for pattern in ["NameError: name 'stochastic_k'", "stochastic_k is not defined", "NameError: 'stochastic_k'"]):
                            stochastic_error_logs.append(log_line.strip())
                            analysis_results['stochastic_errors_found'] += 1
                        
                        # Check for any NameError
                        if 'nameerror' in log_lower:
                            nameerror_logs.append(log_line.strip())
                        
                        # Check for successful IA1 completions
                        if any(pattern in log_lower for pattern in ['ia1 analysis completed', 'ia1 ultra analysis', 'analysis successful']):
                            success_logs.append(log_line.strip())
                    
                    logger.info(f"      📊 Backend logs analysis:")
                    logger.info(f"         - MFI error logs: {len(mfi_error_logs)}")
                    logger.info(f"         - Stochastic error logs: {len(stochastic_error_logs)}")
                    logger.info(f"         - NameError logs: {len(nameerror_logs)}")
                    logger.info(f"         - Success logs: {len(success_logs)}")
                    
                    # Show critical error logs
                    if mfi_error_logs:
                        logger.error(f"      🚨 MFI ERROR FOUND: {mfi_error_logs[0]}")
                    if stochastic_error_logs:
                        logger.error(f"      🚨 STOCHASTIC ERROR FOUND: {stochastic_error_logs[0]}")
                    if nameerror_logs and not mfi_error_logs and not stochastic_error_logs:
                        logger.warning(f"      ⚠️ Other NameError found: {nameerror_logs[0]}")
                    if success_logs:
                        logger.info(f"      ✅ Sample success log: {success_logs[0]}")
                        
            except Exception as e:
                logger.warning(f"      ⚠️ Could not analyze backend logs: {e}")
            
            # Check database for field name persistence
            logger.info("   🗄️ Checking database for new field name persistence...")
            
            try:
                # Connect to MongoDB to check recent analyses
                from pymongo import MongoClient
                client = MongoClient(self.mongo_url)
                db = client[self.db_name]
                
                # Get recent analyses
                recent_analyses = list(db.technical_analyses.find().sort("timestamp", -1).limit(10))
                
                new_fields_in_db = 0
                old_fields_in_db = 0
                
                for analysis in recent_analyses:
                    # Check for new field names
                    if 'trade_type' in analysis and 'minimum_rr_threshold' in analysis:
                        new_fields_in_db += 1
                    
                    # Check for old field names (should not be present)
                    if 'recommended_trade_type' in analysis or 'minimum_rr_for_trade_type' in analysis:
                        old_fields_in_db += 1
                
                field_validation_results['database_field_validation'] = new_fields_in_db
                logger.info(f"      ✅ Database field validation: {new_fields_in_db}/10 recent analyses have new field names")
                if old_fields_in_db > 0:
                    logger.warning(f"      ⚠️ Database old fields: {old_fields_in_db}/10 recent analyses still have old field names")
                else:
                    logger.info(f"      ✅ Database clean: 0/10 recent analyses have old field names")
                
                # Show sample analysis with new fields
                if recent_analyses:
                    sample_analysis = recent_analyses[0]
                    trade_type = sample_analysis.get('trade_type')
                    min_rr = sample_analysis.get('minimum_rr_threshold')
                    symbol = sample_analysis.get('symbol', 'UNKNOWN')
                    logger.info(f"      📋 Sample DB analysis ({symbol}): trade_type={trade_type}, minimum_rr_threshold={min_rr}")
                
                client.close()
                
            except Exception as e:
                logger.warning(f"      ⚠️ Could not check database: {e}")
            
            # Final analysis
            success_rate = field_validation_results['analyses_successful'] / max(field_validation_results['analyses_attempted'], 1)
            new_fields_rate = field_validation_results['new_field_names_present'] / max(field_validation_results['analyses_successful'] * 2, 1)  # 2 new fields per analysis
            old_fields_absent_rate = field_validation_results['old_field_names_absent'] / max(field_validation_results['analyses_successful'], 1)
            
            logger.info(f"\n   📊 FIELD NAME VALIDATION RESULTS:")
            logger.info(f"      Analyses attempted: {field_validation_results['analyses_attempted']}")
            logger.info(f"      Analyses successful: {field_validation_results['analyses_successful']}")
            logger.info(f"      Success rate: {success_rate:.2f}")
            logger.info(f"      New field names present: {field_validation_results['new_field_names_present']}")
            logger.info(f"      New fields rate: {new_fields_rate:.2f}")
            logger.info(f"      Old field names absent: {field_validation_results['old_field_names_absent']}")
            logger.info(f"      Old fields absent rate: {old_fields_absent_rate:.2f}")
            logger.info(f"      Trade type valid: {field_validation_results['trade_type_valid']}")
            logger.info(f"      Minimum RR threshold valid: {field_validation_results['minimum_rr_threshold_valid']}")
            logger.info(f"      Database field validation: {field_validation_results['database_field_validation']}")
            
            # Show field validation details
            if field_validation_results['field_validation_details']:
                logger.info(f"      📊 Field Validation Details:")
                for detail in field_validation_results['field_validation_details']:
                    logger.info(f"         - {detail['symbol']}: new_fields={detail['new_fields_present']}, old_fields_absent={detail['old_fields_absent']}, trade_type_valid={detail['trade_type_valid']}, rr_valid={detail['rr_threshold_valid']}")
            
            # Calculate test success based on review requirements
            success_criteria = [
                field_validation_results['analyses_successful'] >= 2,  # At least 2 successful analyses
                field_validation_results['new_field_names_present'] >= 4,  # At least 4 new field instances (2 per analysis)
                field_validation_results['old_field_names_absent'] >= 2,  # At least 2 analyses without old fields
                field_validation_results['trade_type_valid'] >= 2,  # At least 2 valid trade types
                field_validation_results['minimum_rr_threshold_valid'] >= 2,  # At least 2 valid RR thresholds
                success_rate >= 0.67  # At least 67% success rate
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.83:  # 83% success threshold (5/6 criteria)
                self.log_test_result("Field Name Validation", True, 
                                   f"Field name validation successful: {success_count}/{len(success_criteria)} criteria met. New fields rate: {new_fields_rate:.2f}, Old fields absent rate: {old_fields_absent_rate:.2f}, Success rate: {success_rate:.2f}")
            else:
                self.log_test_result("Field Name Validation", False, 
                                   f"Field name validation issues: {success_count}/{len(success_criteria)} criteria met. May still be using old field names or missing new field names")
                
        except Exception as e:
            self.log_test_result("Field Name Validation", False, f"Exception: {str(e)}")

    async def test_2_dynamic_rr_escalation_logic(self):
        """Test 2: Dynamic RR Escalation Logic - Test _should_send_to_ia2 method uses new field names and dynamic thresholds"""
        logger.info("\n🔍 TEST 2: Dynamic RR Escalation Logic")
        
        try:
            escalation_results = {
                'analyses_attempted': 0,
                'analyses_successful': 0,
                'ia2_escalations_triggered': 0,
                'trade_type_scalp_count': 0,
                'trade_type_intraday_count': 0,
                'trade_type_swing_count': 0,
                'trade_type_position_count': 0,
                'dynamic_threshold_usage': 0,
                'escalation_details': [],
                'backend_escalation_logs': []
            }
            
            logger.info("   🚀 Testing dynamic RR escalation logic with trade type adaptation...")
            logger.info("   📊 Expected: Different RR thresholds based on trade type (SCALP: 1.0, INTRADAY: 1.5, SWING: 2.0)")
            
            # Get available symbols from scout system
            logger.info("   📞 Getting available symbols from scout system...")
            
            try:
                response = requests.get(f"{self.api_url}/opportunities", timeout=60)
                
                if response.status_code == 200:
                    opportunities = response.json()
                    if isinstance(opportunities, dict) and 'opportunities' in opportunities:
                        opportunities = opportunities['opportunities']
                    
                    # Get more symbols to increase chance of different trade types
                    available_symbols = [opp.get('symbol') for opp in opportunities[:15] if opp.get('symbol')]
                    
                    # Select diverse symbols that might generate different trade types
                    test_symbols = []
                    preferred_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT']
                    
                    for symbol in preferred_symbols:
                        if symbol in available_symbols:
                            test_symbols.append(symbol)
                    
                    # Fill remaining slots with available symbols
                    for symbol in available_symbols:
                        if symbol not in test_symbols and len(test_symbols) < 5:
                            test_symbols.append(symbol)
                    
                    self.actual_test_symbols = test_symbols[:5]  # Test up to 5 symbols
                    logger.info(f"      ✅ Test symbols selected: {self.actual_test_symbols}")
                    
                else:
                    logger.warning(f"      ⚠️ Could not get opportunities, using default symbols")
                    self.actual_test_symbols = self.test_symbols
                    
            except Exception as e:
                logger.warning(f"      ⚠️ Error getting opportunities: {e}, using default symbols")
                self.actual_test_symbols = self.test_symbols
            
            # Test each symbol for dynamic RR escalation
            for symbol in self.actual_test_symbols:
                logger.info(f"\n   📞 Testing dynamic RR escalation for {symbol}...")
                escalation_results['analyses_attempted'] += 1
                
                try:
                    start_time = time.time()
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        escalation_results['analyses_successful'] += 1
                        
                        logger.info(f"      ✅ {symbol} analysis successful (response time: {response_time:.2f}s)")
                        
                        # Extract IA1 analysis data
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        if not isinstance(ia1_analysis, dict):
                            ia1_analysis = {}
                        
                        # Extract key fields for escalation logic
                        trade_type = ia1_analysis.get('trade_type', 'SWING')
                        minimum_rr_threshold = ia1_analysis.get('minimum_rr_threshold', 2.0)
                        risk_reward_ratio = ia1_analysis.get('risk_reward_ratio', 1.0)
                        ia1_signal = ia1_analysis.get('ia1_signal', 'hold')
                        confidence = ia1_analysis.get('analysis_confidence', 0.5)
                        
                        logger.info(f"         📊 Escalation Data: trade_type={trade_type}, min_rr_threshold={minimum_rr_threshold}, actual_rr={risk_reward_ratio}, signal={ia1_signal}, confidence={confidence:.2f}")
                        
                        # Count trade types
                        if trade_type == 'SCALP':
                            escalation_results['trade_type_scalp_count'] += 1
                        elif trade_type == 'INTRADAY':
                            escalation_results['trade_type_intraday_count'] += 1
                        elif trade_type == 'SWING':
                            escalation_results['trade_type_swing_count'] += 1
                        elif trade_type == 'POSITION':
                            escalation_results['trade_type_position_count'] += 1
                        
                        # Check if dynamic threshold is being used correctly
                        expected_threshold = self.trade_type_rr_mapping.get(trade_type, 2.0)
                        if abs(minimum_rr_threshold - expected_threshold) < 0.5:
                            escalation_results['dynamic_threshold_usage'] += 1
                            logger.info(f"         ✅ Dynamic threshold correct: {minimum_rr_threshold} ≈ {expected_threshold} for {trade_type}")
                        else:
                            logger.warning(f"         ⚠️ Dynamic threshold incorrect: {minimum_rr_threshold} vs expected {expected_threshold} for {trade_type}")
                        
                        # Determine if IA2 escalation should occur based on logic
                        should_escalate_confidence = confidence > 0.95
                        should_escalate_rr = risk_reward_ratio > minimum_rr_threshold
                        should_escalate_signal = ia1_signal.lower() in ['long', 'short']
                        should_escalate = should_escalate_signal and (should_escalate_confidence or should_escalate_rr)
                        
                        # Check if IA2 decision was actually created (check response for IA2 data)
                        ia2_decision = analysis_data.get('ia2_decision')
                        ia2_escalated = ia2_decision is not None and isinstance(ia2_decision, dict)
                        
                        if ia2_escalated:
                            escalation_results['ia2_escalations_triggered'] += 1
                            logger.info(f"         🚀 IA2 escalation triggered for {symbol}")
                        
                        # Store escalation details
                        escalation_results['escalation_details'].append({
                            'symbol': symbol,
                            'trade_type': trade_type,
                            'minimum_rr_threshold': minimum_rr_threshold,
                            'expected_threshold': expected_threshold,
                            'risk_reward_ratio': risk_reward_ratio,
                            'ia1_signal': ia1_signal,
                            'confidence': confidence,
                            'should_escalate': should_escalate,
                            'ia2_escalated': ia2_escalated,
                            'escalation_match': should_escalate == ia2_escalated,
                            'response_time': response_time
                        })
                        
                        logger.info(f"         📊 Escalation Logic: should_escalate={should_escalate}, ia2_escalated={ia2_escalated}, match={should_escalate == ia2_escalated}")
                        
                    else:
                        logger.error(f"      ❌ {symbol} analysis failed: HTTP {response.status_code}")
                        if response.text:
                            logger.error(f"         Error response: {response.text[:300]}")
                
                except Exception as e:
                    logger.error(f"      ❌ {symbol} analysis exception: {e}")
                
                # Wait between analyses
                if symbol != self.actual_test_symbols[-1]:
                    logger.info(f"      ⏳ Waiting 10 seconds before next analysis...")
                    await asyncio.sleep(10)
            
            # Capture backend logs to check for escalation logic
            logger.info("   📋 Capturing backend logs to check for escalation logic...")
            
            try:
                backend_logs = await self._capture_backend_logs()
                if backend_logs:
                    # Look for escalation-related logs
                    escalation_logs = []
                    should_send_logs = []
                    dynamic_threshold_logs = []
                    
                    for log_line in backend_logs:
                        log_lower = log_line.lower()
                        
                        if 'ia2 accepted' in log_lower or 'ia2 rejected' in log_lower:
                            escalation_logs.append(log_line.strip())
                        
                        if '_should_send_to_ia2' in log_lower:
                            should_send_logs.append(log_line.strip())
                        
                        if any(term in log_lower for term in ['minimum_rr_threshold', 'trade_type', 'scalp', 'intraday', 'swing']):
                            dynamic_threshold_logs.append(log_line.strip())
                    
                    escalation_results['backend_escalation_logs'] = escalation_logs
                    
                    logger.info(f"      📊 Backend escalation logs analysis:")
                    logger.info(f"         - Escalation decision logs: {len(escalation_logs)}")
                    logger.info(f"         - _should_send_to_ia2 logs: {len(should_send_logs)}")
                    logger.info(f"         - Dynamic threshold logs: {len(dynamic_threshold_logs)}")
                    
                    # Show sample logs
                    if escalation_logs:
                        logger.info(f"      📋 Sample escalation log: {escalation_logs[0]}")
                    if dynamic_threshold_logs:
                        logger.info(f"      📋 Sample dynamic threshold log: {dynamic_threshold_logs[0]}")
                        
            except Exception as e:
                logger.warning(f"      ⚠️ Could not analyze backend logs: {e}")
            
            # Final analysis
            success_rate = escalation_results['analyses_successful'] / max(escalation_results['analyses_attempted'], 1)
            dynamic_threshold_rate = escalation_results['dynamic_threshold_usage'] / max(escalation_results['analyses_successful'], 1)
            escalation_rate = escalation_results['ia2_escalations_triggered'] / max(escalation_results['analyses_successful'], 1)
            
            logger.info(f"\n   📊 DYNAMIC RR ESCALATION LOGIC RESULTS:")
            logger.info(f"      Analyses attempted: {escalation_results['analyses_attempted']}")
            logger.info(f"      Analyses successful: {escalation_results['analyses_successful']}")
            logger.info(f"      Success rate: {success_rate:.2f}")
            logger.info(f"      IA2 escalations triggered: {escalation_results['ia2_escalations_triggered']}")
            logger.info(f"      Escalation rate: {escalation_rate:.2f}")
            logger.info(f"      Dynamic threshold usage: {escalation_results['dynamic_threshold_usage']}")
            logger.info(f"      Dynamic threshold rate: {dynamic_threshold_rate:.2f}")
            logger.info(f"      Trade type distribution:")
            logger.info(f"         - SCALP: {escalation_results['trade_type_scalp_count']}")
            logger.info(f"         - INTRADAY: {escalation_results['trade_type_intraday_count']}")
            logger.info(f"         - SWING: {escalation_results['trade_type_swing_count']}")
            logger.info(f"         - POSITION: {escalation_results['trade_type_position_count']}")
            
            # Show escalation details
            if escalation_results['escalation_details']:
                logger.info(f"      📊 Escalation Logic Details:")
                for detail in escalation_results['escalation_details']:
                    logger.info(f"         - {detail['symbol']}: {detail['trade_type']} (threshold={detail['minimum_rr_threshold']:.1f}, rr={detail['risk_reward_ratio']:.2f}, escalated={detail['ia2_escalated']}, match={detail['escalation_match']})")
            
            # Calculate test success based on review requirements
            success_criteria = [
                escalation_results['analyses_successful'] >= 3,  # At least 3 successful analyses
                escalation_results['dynamic_threshold_usage'] >= 2,  # At least 2 correct dynamic thresholds
                dynamic_threshold_rate >= 0.6,  # At least 60% correct dynamic threshold usage
                len(set([detail['trade_type'] for detail in escalation_results['escalation_details']])) >= 2,  # At least 2 different trade types
                success_rate >= 0.6  # At least 60% success rate
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Dynamic RR Escalation Logic", True, 
                                   f"Dynamic RR escalation successful: {success_count}/{len(success_criteria)} criteria met. Dynamic threshold rate: {dynamic_threshold_rate:.2f}, Escalation rate: {escalation_rate:.2f}, Success rate: {success_rate:.2f}")
            else:
                self.log_test_result("Dynamic RR Escalation Logic", False, 
                                   f"Dynamic RR escalation issues: {success_count}/{len(success_criteria)} criteria met. May not be using dynamic thresholds correctly or missing trade type diversity")
                
        except Exception as e:
            self.log_test_result("Dynamic RR Escalation Logic", False, f"Exception: {str(e)}")

    async def test_3_database_persistence_validation(self):
        """Test 3: Database Persistence - Verify analyses stored in MongoDB with new field names and sensible values"""
        logger.info("\n🔍 TEST 3: Database Persistence Validation")
        
        try:
            db_persistence_results = {
                'database_connection_success': False,
                'total_analyses_found': 0,
                'analyses_with_new_fields': 0,
                'analyses_with_old_fields': 0,
                'trade_type_distribution': {},
                'rr_threshold_analysis': {},
                'field_value_validation': {},
                'sample_analyses': []
            }
            
            logger.info("   🚀 Testing database persistence of new field names and values...")
            logger.info("   📊 Expected: Analyses stored with trade_type and minimum_rr_threshold fields")
            
            try:
                # Connect to MongoDB
                from pymongo import MongoClient
                client = MongoClient(self.mongo_url)
                db = client[self.db_name]
                db_persistence_results['database_connection_success'] = True
                
                logger.info("      ✅ Database connection successful")
                
                # Get recent technical analyses
                recent_analyses = list(db.technical_analyses.find().sort("timestamp", -1).limit(20))
                db_persistence_results['total_analyses_found'] = len(recent_analyses)
                
                logger.info(f"      📊 Found {len(recent_analyses)} recent analyses in database")
                
                if recent_analyses:
                    # Analyze each analysis for field presence and values
                    trade_type_counts = {}
                    rr_threshold_values = []
                    
                    for i, analysis in enumerate(recent_analyses):
                        symbol = analysis.get('symbol', 'UNKNOWN')
                        timestamp = analysis.get('timestamp', 'UNKNOWN')
                        
                        # Check for new field names
                        trade_type = analysis.get('trade_type')
                        minimum_rr_threshold = analysis.get('minimum_rr_threshold')
                        
                        # Check for old field names (should not be present)
                        old_trade_type = analysis.get('recommended_trade_type')
                        old_rr_threshold = analysis.get('minimum_rr_for_trade_type')
                        
                        # Count analyses with new fields
                        if trade_type is not None and minimum_rr_threshold is not None:
                            db_persistence_results['analyses_with_new_fields'] += 1
                            
                            # Count trade type distribution
                            if trade_type in trade_type_counts:
                                trade_type_counts[trade_type] += 1
                            else:
                                trade_type_counts[trade_type] = 1
                            
                            # Collect RR threshold values
                            if isinstance(minimum_rr_threshold, (int, float)):
                                rr_threshold_values.append(minimum_rr_threshold)
                        
                        # Count analyses with old fields (should be 0)
                        if old_trade_type is not None or old_rr_threshold is not None:
                            db_persistence_results['analyses_with_old_fields'] += 1
                        
                        # Store sample analyses for detailed inspection
                        if i < 5:  # First 5 analyses
                            db_persistence_results['sample_analyses'].append({
                                'symbol': symbol,
                                'timestamp': str(timestamp),
                                'trade_type': trade_type,
                                'minimum_rr_threshold': minimum_rr_threshold,
                                'old_trade_type': old_trade_type,
                                'old_rr_threshold': old_rr_threshold,
                                'has_new_fields': trade_type is not None and minimum_rr_threshold is not None,
                                'has_old_fields': old_trade_type is not None or old_rr_threshold is not None
                            })
                            
                            logger.info(f"         📋 Sample {i+1} ({symbol}): trade_type={trade_type}, min_rr={minimum_rr_threshold}, old_fields={old_trade_type is not None or old_rr_threshold is not None}")
                    
                    # Analyze trade type distribution
                    db_persistence_results['trade_type_distribution'] = trade_type_counts
                    logger.info(f"      📊 Trade type distribution: {trade_type_counts}")
                    
                    # Analyze RR threshold values
                    if rr_threshold_values:
                        rr_analysis = {
                            'count': len(rr_threshold_values),
                            'min': min(rr_threshold_values),
                            'max': max(rr_threshold_values),
                            'avg': sum(rr_threshold_values) / len(rr_threshold_values),
                            'unique_values': list(set(rr_threshold_values))
                        }
                        db_persistence_results['rr_threshold_analysis'] = rr_analysis
                        logger.info(f"      📊 RR threshold analysis: count={rr_analysis['count']}, range={rr_analysis['min']:.1f}-{rr_analysis['max']:.1f}, avg={rr_analysis['avg']:.2f}")
                        logger.info(f"      📊 Unique RR thresholds: {rr_analysis['unique_values']}")
                    
                    # Validate field values make sense
                    field_validation = {
                        'valid_trade_types': 0,
                        'valid_rr_thresholds': 0,
                        'sensible_mappings': 0
                    }
                    
                    for analysis in recent_analyses:
                        trade_type = analysis.get('trade_type')
                        minimum_rr_threshold = analysis.get('minimum_rr_threshold')
                        
                        if trade_type in self.valid_trade_types:
                            field_validation['valid_trade_types'] += 1
                        
                        if isinstance(minimum_rr_threshold, (int, float)) and 0.5 <= minimum_rr_threshold <= 3.0:
                            field_validation['valid_rr_thresholds'] += 1
                        
                        # Check if mapping makes sense
                        if trade_type and minimum_rr_threshold:
                            expected_rr = self.trade_type_rr_mapping.get(trade_type, 2.0)
                            if abs(minimum_rr_threshold - expected_rr) < 0.5:
                                field_validation['sensible_mappings'] += 1
                    
                    db_persistence_results['field_value_validation'] = field_validation
                    logger.info(f"      📊 Field validation: valid_trade_types={field_validation['valid_trade_types']}, valid_rr_thresholds={field_validation['valid_rr_thresholds']}, sensible_mappings={field_validation['sensible_mappings']}")
                
                else:
                    logger.warning("      ⚠️ No analyses found in database")
                
                client.close()
                
            except Exception as e:
                logger.error(f"      ❌ Database analysis failed: {e}")
            
            # Final analysis
            new_fields_rate = db_persistence_results['analyses_with_new_fields'] / max(db_persistence_results['total_analyses_found'], 1)
            old_fields_rate = db_persistence_results['analyses_with_old_fields'] / max(db_persistence_results['total_analyses_found'], 1)
            
            logger.info(f"\n   📊 DATABASE PERSISTENCE VALIDATION RESULTS:")
            logger.info(f"      Database connection: {db_persistence_results['database_connection_success']}")
            logger.info(f"      Total analyses found: {db_persistence_results['total_analyses_found']}")
            logger.info(f"      Analyses with new fields: {db_persistence_results['analyses_with_new_fields']}")
            logger.info(f"      New fields rate: {new_fields_rate:.2f}")
            logger.info(f"      Analyses with old fields: {db_persistence_results['analyses_with_old_fields']}")
            logger.info(f"      Old fields rate: {old_fields_rate:.2f}")
            logger.info(f"      Trade type distribution: {db_persistence_results['trade_type_distribution']}")
            
            # Calculate test success based on review requirements
            success_criteria = [
                db_persistence_results['database_connection_success'],
                db_persistence_results['total_analyses_found'] >= 5,  # At least 5 analyses in database
                db_persistence_results['analyses_with_new_fields'] >= 3,  # At least 3 analyses with new fields
                new_fields_rate >= 0.6,  # At least 60% of analyses have new fields
                old_fields_rate <= 0.2,  # At most 20% of analyses have old fields
                len(db_persistence_results['trade_type_distribution']) >= 2  # At least 2 different trade types
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.83:  # 83% success threshold (5/6 criteria)
                self.log_test_result("Database Persistence Validation", True, 
                                   f"Database persistence successful: {success_count}/{len(success_criteria)} criteria met. New fields rate: {new_fields_rate:.2f}, Old fields rate: {old_fields_rate:.2f}, Trade types: {len(db_persistence_results['trade_type_distribution'])}")
            else:
                self.log_test_result("Database Persistence Validation", False, 
                                   f"Database persistence issues: {success_count}/{len(success_criteria)} criteria met. May have missing new fields or persistent old fields in database")
                
        except Exception as e:
            self.log_test_result("Database Persistence Validation", False, f"Exception: {str(e)}")

    async def test_0_pydantic_validation_fix_validation(self):
        """Test 0: Pydantic Validation Fix Validation - Critical Fix for fibonacci_key_level_proximity field"""
        logger.info("\n🔍 TEST 0: Pydantic Validation Fix Validation")
        
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

    async def test_1_detected_pattern_names_crash_fix_validation(self):
        """Test 0: detected_pattern_names Crash Fix Validation - Critical Bug Fix Test"""
        logger.info("\n🔍 TEST 0: detected_pattern_names Crash Fix Validation")
        
        try:
            crash_fix_results = {
                'analyses_attempted': 0,
                'analyses_successful': 0,
                'crash_errors_detected': 0,
                'variable_initialization_errors': 0,
                'backend_stability': True,
                'error_logs': [],
                'successful_analyses': [],
                'system_exceptions': 0
            }
            
            logger.info("   🚀 Testing critical detected_pattern_names variable initialization fix...")
            logger.info("   📊 Expected: 100% analyses complete without 'cannot access local variable detected_pattern_names' errors")
            
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
                    
                    # Prefer BTCUSDT, ETHUSDT, SOLUSDT if available, otherwise use first 3
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
            
            # Test each symbol for detected_pattern_names crash
            for symbol in self.actual_test_symbols:
                logger.info(f"\n   📞 Testing IA1 analysis for {symbol} - checking for detected_pattern_names crash...")
                crash_fix_results['analyses_attempted'] += 1
                
                try:
                    start_time = time.time()
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        crash_fix_results['analyses_successful'] += 1
                        crash_fix_results['successful_analyses'].append({
                            'symbol': symbol,
                            'response_time': response_time,
                            'analysis_data': analysis_data
                        })
                        
                        logger.info(f"      ✅ {symbol} analysis successful (response time: {response_time:.2f}s)")
                        
                        # Check for fallback analysis indicators (sign of crash recovery)
                        reasoning = ""
                        if 'ia1_analysis' in analysis_data and isinstance(analysis_data['ia1_analysis'], dict):
                            reasoning = analysis_data['ia1_analysis'].get('reasoning', '')
                        elif 'reasoning' in analysis_data:
                            reasoning = analysis_data.get('reasoning', '')
                        
                        if reasoning:
                            # Check for fallback indicators
                            fallback_indicators = [
                                'fallback analysis', 'json parsing failed', 'pattern-only analysis',
                                'technical analysis fallback', 'using detected pattern'
                            ]
                            
                            is_fallback = any(indicator in reasoning.lower() for indicator in fallback_indicators)
                            
                            if is_fallback:
                                logger.warning(f"         ⚠️ Fallback analysis detected - may indicate underlying issues")
                            else:
                                logger.info(f"         ✅ Full analysis completed - no fallback detected")
                        
                    elif response.status_code == 502:
                        logger.error(f"      ❌ {symbol} analysis failed: HTTP 502 (Backend Error)")
                        crash_fix_results['system_exceptions'] += 1
                        crash_fix_results['backend_stability'] = False
                        
                    else:
                        logger.error(f"      ❌ {symbol} analysis failed: HTTP {response.status_code}")
                        if response.text:
                            error_text = response.text[:300]
                            logger.error(f"         Error response: {error_text}")
                            crash_fix_results['error_logs'].append(f"{symbol}: {error_text}")
                
                except Exception as e:
                    logger.error(f"      ❌ {symbol} analysis exception: {e}")
                    crash_fix_results['system_exceptions'] += 1
                    crash_fix_results['error_logs'].append(f"{symbol}: {str(e)}")
                
                # Wait between analyses
                if symbol != self.actual_test_symbols[-1]:
                    logger.info(f"      ⏳ Waiting 10 seconds before next analysis...")
                    await asyncio.sleep(10)
            
            # Capture backend logs to check for detected_pattern_names errors
            logger.info("   📋 Capturing backend logs to check for detected_pattern_names errors...")
            
            try:
                backend_logs = await self._capture_backend_logs()
                if backend_logs:
                    # Look for specific detected_pattern_names errors
                    pattern_name_errors = []
                    variable_errors = []
                    
                    for log_line in backend_logs:
                        if 'detected_pattern_names' in log_line.lower() and 'error' in log_line.lower():
                            pattern_name_errors.append(log_line.strip())
                        
                        if any(error_pattern in log_line.lower() for error_pattern in [
                            'cannot access local variable', 'variable referenced before assignment',
                            'unbound local error', 'name error'
                        ]):
                            variable_errors.append(log_line.strip())
                    
                    crash_fix_results['crash_errors_detected'] = len(pattern_name_errors)
                    crash_fix_results['variable_initialization_errors'] = len(variable_errors)
                    
                    if pattern_name_errors:
                        logger.error(f"      ❌ detected_pattern_names errors found: {len(pattern_name_errors)}")
                        for error in pattern_name_errors[:3]:  # Show first 3
                            logger.error(f"         - {error}")
                    else:
                        logger.info(f"      ✅ No detected_pattern_names errors found in backend logs")
                    
                    if variable_errors:
                        logger.warning(f"      ⚠️ Variable initialization errors found: {len(variable_errors)}")
                        for error in variable_errors[:2]:  # Show first 2
                            logger.warning(f"         - {error}")
                    else:
                        logger.info(f"      ✅ No variable initialization errors found")
                        
            except Exception as e:
                logger.warning(f"      ⚠️ Could not analyze backend logs: {e}")
            
            # Final analysis
            success_rate = crash_fix_results['analyses_successful'] / max(crash_fix_results['analyses_attempted'], 1)
            
            logger.info(f"\n   📊 DETECTED_PATTERN_NAMES CRASH FIX VALIDATION RESULTS:")
            logger.info(f"      Analyses attempted: {crash_fix_results['analyses_attempted']}")
            logger.info(f"      Analyses successful: {crash_fix_results['analyses_successful']}")
            logger.info(f"      Success rate: {success_rate:.2f}")
            logger.info(f"      detected_pattern_names errors: {crash_fix_results['crash_errors_detected']}")
            logger.info(f"      Variable initialization errors: {crash_fix_results['variable_initialization_errors']}")
            logger.info(f"      Backend stability: {crash_fix_results['backend_stability']}")
            logger.info(f"      System exceptions: {crash_fix_results['system_exceptions']}")
            
            # Calculate test success based on review requirements (100% analyses without crash)
            success_criteria = [
                crash_fix_results['analyses_successful'] >= 2,  # At least 2 successful analyses
                crash_fix_results['crash_errors_detected'] == 0,  # No detected_pattern_names errors
                crash_fix_results['variable_initialization_errors'] == 0,  # No variable errors
                success_rate >= 0.67,  # At least 67% success rate (2/3)
                crash_fix_results['backend_stability']  # Backend stable
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("detected_pattern_names Crash Fix Validation", True, 
                                   f"Critical fix successful: {success_count}/{len(success_criteria)} criteria met. Success rate: {success_rate:.2f}, No crashes: {crash_fix_results['crash_errors_detected'] == 0}, Backend stable: {crash_fix_results['backend_stability']}")
            else:
                self.log_test_result("detected_pattern_names Crash Fix Validation", False, 
                                   f"Critical fix issues: {success_count}/{len(success_criteria)} criteria met. May still have detected_pattern_names crashes or system instability")
                
        except Exception as e:
            self.log_test_result("detected_pattern_names Crash Fix Validation", False, f"Exception: {str(e)}")

    async def test_ia2_decisions_database_analysis(self):
        """Test IA2: Database Analysis - Examine IA2 decisions in trading_decisions collection"""
        logger.info("\n🔍 TEST IA2: Database Analysis - IA2 Decisions Structure")
        
        try:
            ia2_db_results = {
                'total_ia2_decisions': 0,
                'ia2_reasoning_populated': 0,
                'strategic_reasoning_populated': 0,
                'calculated_rr_present': 0,
                'ia2_calculated_rr_present': 0,
                'technical_levels_complete': 0,
                'recent_ia2_decisions': [],
                'reasoning_field_analysis': {},
                'rr_calculation_analysis': {},
                'database_connection_success': False
            }
            
            logger.info("   🚀 Analyzing IA2 decisions in trading_decisions collection...")
            logger.info("   📊 Expected: IA2 decisions with populated reasoning and calculated RR fields")
            
            try:
                # Connect to MongoDB to analyze IA2 decisions
                from pymongo import MongoClient
                mongo_url = "mongodb://localhost:27017"
                client = MongoClient(mongo_url)
                db = client["myapp"]
                ia2_db_results['database_connection_success'] = True
                
                logger.info("      ✅ Database connection successful")
                
                # Get recent IA2 decisions (last 50)
                ia2_decisions = list(db.trading_decisions.find({
                    "$or": [
                        {"ia2_reasoning": {"$exists": True}},
                        {"strategic_reasoning": {"$exists": True}},
                        {"ia2_calculated_rr": {"$exists": True}},
                        {"signal": {"$in": ["LONG", "SHORT"]}}  # IA2 typically uses LONG/SHORT
                    ]
                }).sort("timestamp", -1).limit(50))
                
                ia2_db_results['total_ia2_decisions'] = len(ia2_decisions)
                logger.info(f"      📊 Found {len(ia2_decisions)} potential IA2 decisions")
                
                if ia2_decisions:
                    # Analyze each IA2 decision
                    for i, decision in enumerate(ia2_decisions[:10]):  # Analyze first 10 in detail
                        symbol = decision.get('symbol', 'UNKNOWN')
                        timestamp = decision.get('timestamp', 'UNKNOWN')
                        
                        # Check reasoning fields
                        ia2_reasoning = decision.get('ia2_reasoning', '')
                        strategic_reasoning = decision.get('strategic_reasoning', '')
                        reasoning = decision.get('reasoning', '')
                        
                        # Check RR calculation fields
                        calculated_rr = decision.get('calculated_rr')
                        ia2_calculated_rr = decision.get('ia2_calculated_rr')
                        risk_reward_ratio = decision.get('risk_reward_ratio')
                        
                        # Check technical levels
                        ia2_entry_price = decision.get('ia2_entry_price')
                        ia2_stop_loss = decision.get('ia2_stop_loss')
                        ia2_take_profit_1 = decision.get('ia2_take_profit_1')
                        
                        # Count populated fields
                        if ia2_reasoning and len(str(ia2_reasoning).strip()) > 10:
                            ia2_db_results['ia2_reasoning_populated'] += 1
                        
                        if strategic_reasoning and len(str(strategic_reasoning).strip()) > 10:
                            ia2_db_results['strategic_reasoning_populated'] += 1
                        
                        if calculated_rr and calculated_rr not in [None, 0, 1.0]:
                            ia2_db_results['calculated_rr_present'] += 1
                        
                        if ia2_calculated_rr and ia2_calculated_rr not in [None, 0, 1.0]:
                            ia2_db_results['ia2_calculated_rr_present'] += 1
                        
                        if all([ia2_entry_price, ia2_stop_loss, ia2_take_profit_1]):
                            ia2_db_results['technical_levels_complete'] += 1
                        
                        # Store detailed analysis for first 5 decisions
                        if i < 5:
                            ia2_db_results['recent_ia2_decisions'].append({
                                'symbol': symbol,
                                'timestamp': str(timestamp),
                                'ia2_reasoning_length': len(str(ia2_reasoning)) if ia2_reasoning else 0,
                                'strategic_reasoning_length': len(str(strategic_reasoning)) if strategic_reasoning else 0,
                                'reasoning_length': len(str(reasoning)) if reasoning else 0,
                                'calculated_rr': calculated_rr,
                                'ia2_calculated_rr': ia2_calculated_rr,
                                'risk_reward_ratio': risk_reward_ratio,
                                'technical_levels': {
                                    'entry': ia2_entry_price,
                                    'sl': ia2_stop_loss,
                                    'tp1': ia2_take_profit_1
                                }
                            })
                            
                            logger.info(f"         📋 {symbol}: ia2_reasoning={len(str(ia2_reasoning)) if ia2_reasoning else 0}chars, strategic_reasoning={len(str(strategic_reasoning)) if strategic_reasoning else 0}chars, calculated_rr={calculated_rr}")
                    
                    # Analyze reasoning field patterns
                    reasoning_analysis = {
                        'ia2_reasoning_avg_length': 0,
                        'strategic_reasoning_avg_length': 0,
                        'reasoning_avg_length': 0,
                        'empty_ia2_reasoning_count': 0,
                        'empty_strategic_reasoning_count': 0,
                        'empty_reasoning_count': 0
                    }
                    
                    total_ia2_reasoning_length = 0
                    total_strategic_reasoning_length = 0
                    total_reasoning_length = 0
                    
                    for decision in ia2_decisions[:20]:  # Analyze first 20
                        ia2_reasoning = decision.get('ia2_reasoning', '')
                        strategic_reasoning = decision.get('strategic_reasoning', '')
                        reasoning = decision.get('reasoning', '')
                        
                        ia2_len = len(str(ia2_reasoning)) if ia2_reasoning else 0
                        strategic_len = len(str(strategic_reasoning)) if strategic_reasoning else 0
                        reasoning_len = len(str(reasoning)) if reasoning else 0
                        
                        total_ia2_reasoning_length += ia2_len
                        total_strategic_reasoning_length += strategic_len
                        total_reasoning_length += reasoning_len
                        
                        if ia2_len <= 10:
                            reasoning_analysis['empty_ia2_reasoning_count'] += 1
                        if strategic_len <= 10:
                            reasoning_analysis['empty_strategic_reasoning_count'] += 1
                        if reasoning_len <= 10:
                            reasoning_analysis['empty_reasoning_count'] += 1
                    
                    reasoning_analysis['ia2_reasoning_avg_length'] = total_ia2_reasoning_length / min(len(ia2_decisions), 20)
                    reasoning_analysis['strategic_reasoning_avg_length'] = total_strategic_reasoning_length / min(len(ia2_decisions), 20)
                    reasoning_analysis['reasoning_avg_length'] = total_reasoning_length / min(len(ia2_decisions), 20)
                    
                    ia2_db_results['reasoning_field_analysis'] = reasoning_analysis
                    
                    # Analyze RR calculation patterns
                    rr_analysis = {
                        'calculated_rr_values': [],
                        'ia2_calculated_rr_values': [],
                        'risk_reward_ratio_values': [],
                        'null_calculated_rr_count': 0,
                        'fixed_rr_values_count': 0,
                        'dynamic_rr_values_count': 0
                    }
                    
                    for decision in ia2_decisions[:20]:  # Analyze first 20
                        calculated_rr = decision.get('calculated_rr')
                        ia2_calculated_rr = decision.get('ia2_calculated_rr')
                        risk_reward_ratio = decision.get('risk_reward_ratio')
                        
                        if calculated_rr is not None:
                            rr_analysis['calculated_rr_values'].append(calculated_rr)
                            if calculated_rr in [1.0, 2.0, 2.2]:
                                rr_analysis['fixed_rr_values_count'] += 1
                            elif calculated_rr > 0:
                                rr_analysis['dynamic_rr_values_count'] += 1
                        else:
                            rr_analysis['null_calculated_rr_count'] += 1
                        
                        if ia2_calculated_rr is not None:
                            rr_analysis['ia2_calculated_rr_values'].append(ia2_calculated_rr)
                        
                        if risk_reward_ratio is not None:
                            rr_analysis['risk_reward_ratio_values'].append(risk_reward_ratio)
                    
                    ia2_db_results['rr_calculation_analysis'] = rr_analysis
                    
                    logger.info(f"      📊 Reasoning Analysis:")
                    logger.info(f"         - IA2 reasoning populated: {ia2_db_results['ia2_reasoning_populated']}/{len(ia2_decisions)}")
                    logger.info(f"         - Strategic reasoning populated: {ia2_db_results['strategic_reasoning_populated']}/{len(ia2_decisions)}")
                    logger.info(f"         - Average ia2_reasoning length: {reasoning_analysis['ia2_reasoning_avg_length']:.1f} chars")
                    logger.info(f"         - Average strategic_reasoning length: {reasoning_analysis['strategic_reasoning_avg_length']:.1f} chars")
                    
                    logger.info(f"      📊 RR Calculation Analysis:")
                    logger.info(f"         - Calculated RR present: {ia2_db_results['calculated_rr_present']}/{len(ia2_decisions)}")
                    logger.info(f"         - IA2 calculated RR present: {ia2_db_results['ia2_calculated_rr_present']}/{len(ia2_decisions)}")
                    logger.info(f"         - Null calculated RR: {rr_analysis['null_calculated_rr_count']}")
                    logger.info(f"         - Fixed RR values: {rr_analysis['fixed_rr_values_count']}")
                    logger.info(f"         - Dynamic RR values: {rr_analysis['dynamic_rr_values_count']}")
                
                else:
                    logger.warning("      ⚠️ No IA2 decisions found in database")
                
                client.close()
                
            except Exception as e:
                logger.error(f"      ❌ Database analysis failed: {e}")
            
            # Final analysis
            logger.info(f"\n   📊 IA2 DATABASE ANALYSIS RESULTS:")
            logger.info(f"      Total IA2 decisions: {ia2_db_results['total_ia2_decisions']}")
            logger.info(f"      IA2 reasoning populated: {ia2_db_results['ia2_reasoning_populated']}")
            logger.info(f"      Strategic reasoning populated: {ia2_db_results['strategic_reasoning_populated']}")
            logger.info(f"      Calculated RR present: {ia2_db_results['calculated_rr_present']}")
            logger.info(f"      Technical levels complete: {ia2_db_results['technical_levels_complete']}")
            
            # Calculate success based on IA2 reasoning and RR calculation presence
            success_criteria = [
                ia2_db_results['database_connection_success'],
                ia2_db_results['total_ia2_decisions'] > 0,
                ia2_db_results['ia2_reasoning_populated'] > 0 or ia2_db_results['strategic_reasoning_populated'] > 0,
                ia2_db_results['calculated_rr_present'] > 0 or ia2_db_results['ia2_calculated_rr_present'] > 0
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.75:  # 75% success threshold
                self.log_test_result("IA2 Database Analysis", True, 
                                   f"IA2 database analysis successful: {success_count}/{len(success_criteria)} criteria met. IA2 decisions: {ia2_db_results['total_ia2_decisions']}, Reasoning populated: {ia2_db_results['ia2_reasoning_populated'] + ia2_db_results['strategic_reasoning_populated']}")
            else:
                self.log_test_result("IA2 Database Analysis", False, 
                                   f"IA2 database analysis issues: {success_count}/{len(success_criteria)} criteria met. May have missing IA2 decisions or empty reasoning fields")
                
        except Exception as e:
            self.log_test_result("IA2 Database Analysis", False, f"Exception: {str(e)}")

    async def test_ia2_api_decisions_structure(self):
        """Test IA2: API Decisions Structure - Test /api/decisions endpoint for IA2 data exposure"""
        logger.info("\n🔍 TEST IA2: API Decisions Structure")
        
        try:
            api_results = {
                'decisions_api_accessible': False,
                'total_decisions_returned': 0,
                'ia2_decisions_found': 0,
                'ia2_reasoning_exposed': 0,
                'calculated_rr_exposed': 0,
                'technical_levels_exposed': 0,
                'api_response_structure': {},
                'sample_ia2_decisions': []
            }
            
            logger.info("   🚀 Testing /api/decisions endpoint for IA2 data structure...")
            logger.info("   📊 Expected: API exposes IA2 reasoning and calculated RR fields")
            
            try:
                # Test /api/decisions endpoint
                response = requests.get(f"{self.api_url}/decisions", timeout=60)
                
                if response.status_code == 200:
                    decisions_data = response.json()
                    api_results['decisions_api_accessible'] = True
                    
                    # Handle different response structures
                    if isinstance(decisions_data, dict) and 'decisions' in decisions_data:
                        decisions = decisions_data['decisions']
                    elif isinstance(decisions_data, list):
                        decisions = decisions_data
                    else:
                        decisions = []
                    
                    api_results['total_decisions_returned'] = len(decisions)
                    logger.info(f"      ✅ Decisions API accessible: {len(decisions)} decisions returned")
                    
                    if decisions:
                        # Analyze API response structure
                        sample_decision = decisions[0]
                        api_results['api_response_structure'] = {
                            'fields_present': list(sample_decision.keys()) if isinstance(sample_decision, dict) else [],
                            'has_ia2_reasoning': 'ia2_reasoning' in sample_decision if isinstance(sample_decision, dict) else False,
                            'has_strategic_reasoning': 'strategic_reasoning' in sample_decision if isinstance(sample_decision, dict) else False,
                            'has_calculated_rr': 'calculated_rr' in sample_decision if isinstance(sample_decision, dict) else False,
                            'has_ia2_calculated_rr': 'ia2_calculated_rr' in sample_decision if isinstance(sample_decision, dict) else False
                        }
                        
                        logger.info(f"      📊 API Response Structure:")
                        logger.info(f"         - Fields present: {len(api_results['api_response_structure']['fields_present'])}")
                        logger.info(f"         - Has ia2_reasoning: {api_results['api_response_structure']['has_ia2_reasoning']}")
                        logger.info(f"         - Has strategic_reasoning: {api_results['api_response_structure']['has_strategic_reasoning']}")
                        logger.info(f"         - Has calculated_rr: {api_results['api_response_structure']['has_calculated_rr']}")
                        
                        # Analyze IA2 decisions in API response
                        for i, decision in enumerate(decisions[:20]):  # Check first 20
                            if not isinstance(decision, dict):
                                continue
                            
                            # Identify potential IA2 decisions
                            is_ia2_decision = False
                            if any(field in decision for field in ['ia2_reasoning', 'strategic_reasoning', 'ia2_calculated_rr']):
                                is_ia2_decision = True
                            elif decision.get('signal') in ['LONG', 'SHORT']:  # IA2 typically uses LONG/SHORT
                                is_ia2_decision = True
                            
                            if is_ia2_decision:
                                api_results['ia2_decisions_found'] += 1
                                
                                # Check reasoning exposure
                                ia2_reasoning = decision.get('ia2_reasoning', '')
                                strategic_reasoning = decision.get('strategic_reasoning', '')
                                reasoning = decision.get('reasoning', '')
                                
                                if (ia2_reasoning and len(str(ia2_reasoning).strip()) > 10) or \
                                   (strategic_reasoning and len(str(strategic_reasoning).strip()) > 10):
                                    api_results['ia2_reasoning_exposed'] += 1
                                
                                # Check RR calculation exposure
                                calculated_rr = decision.get('calculated_rr')
                                ia2_calculated_rr = decision.get('ia2_calculated_rr')
                                
                                if (calculated_rr and calculated_rr not in [None, 0, 1.0]) or \
                                   (ia2_calculated_rr and ia2_calculated_rr not in [None, 0, 1.0]):
                                    api_results['calculated_rr_exposed'] += 1
                                
                                # Check technical levels exposure
                                ia2_entry = decision.get('ia2_entry_price')
                                ia2_sl = decision.get('ia2_stop_loss')
                                ia2_tp1 = decision.get('ia2_take_profit_1')
                                
                                if all([ia2_entry, ia2_sl, ia2_tp1]):
                                    api_results['technical_levels_exposed'] += 1
                                
                                # Store sample for first 3 IA2 decisions
                                if len(api_results['sample_ia2_decisions']) < 3:
                                    api_results['sample_ia2_decisions'].append({
                                        'symbol': decision.get('symbol', 'UNKNOWN'),
                                        'signal': decision.get('signal', 'UNKNOWN'),
                                        'ia2_reasoning_length': len(str(ia2_reasoning)) if ia2_reasoning else 0,
                                        'strategic_reasoning_length': len(str(strategic_reasoning)) if strategic_reasoning else 0,
                                        'calculated_rr': calculated_rr,
                                        'ia2_calculated_rr': ia2_calculated_rr,
                                        'technical_levels_present': all([ia2_entry, ia2_sl, ia2_tp1])
                                    })
                        
                        logger.info(f"      📊 IA2 Decisions Analysis:")
                        logger.info(f"         - IA2 decisions found: {api_results['ia2_decisions_found']}")
                        logger.info(f"         - IA2 reasoning exposed: {api_results['ia2_reasoning_exposed']}")
                        logger.info(f"         - Calculated RR exposed: {api_results['calculated_rr_exposed']}")
                        logger.info(f"         - Technical levels exposed: {api_results['technical_levels_exposed']}")
                        
                        # Show sample IA2 decisions
                        for i, sample in enumerate(api_results['sample_ia2_decisions']):
                            logger.info(f"         📋 Sample IA2 #{i+1}: {sample['symbol']} {sample['signal']}, reasoning={sample['ia2_reasoning_length']}+{sample['strategic_reasoning_length']}chars, rr={sample['calculated_rr']}")
                    
                    else:
                        logger.warning("      ⚠️ No decisions returned from API")
                
                else:
                    logger.error(f"      ❌ Decisions API failed: HTTP {response.status_code}")
                    if response.text:
                        logger.error(f"         Error response: {response.text[:200]}...")
            
            except Exception as e:
                logger.error(f"      ❌ Decisions API exception: {e}")
            
            # Final analysis
            logger.info(f"\n   📊 IA2 API DECISIONS STRUCTURE RESULTS:")
            logger.info(f"      Decisions API accessible: {api_results['decisions_api_accessible']}")
            logger.info(f"      Total decisions returned: {api_results['total_decisions_returned']}")
            logger.info(f"      IA2 decisions found: {api_results['ia2_decisions_found']}")
            logger.info(f"      IA2 reasoning exposed: {api_results['ia2_reasoning_exposed']}")
            logger.info(f"      Calculated RR exposed: {api_results['calculated_rr_exposed']}")
            
            # Calculate success based on API exposure of IA2 data
            success_criteria = [
                api_results['decisions_api_accessible'],
                api_results['total_decisions_returned'] > 0,
                api_results['ia2_decisions_found'] > 0,
                api_results['ia2_reasoning_exposed'] > 0 or api_results['calculated_rr_exposed'] > 0
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.75:  # 75% success threshold
                self.log_test_result("IA2 API Decisions Structure", True, 
                                   f"IA2 API structure analysis successful: {success_count}/{len(success_criteria)} criteria met. IA2 decisions: {api_results['ia2_decisions_found']}, Reasoning exposed: {api_results['ia2_reasoning_exposed']}")
            else:
                self.log_test_result("IA2 API Decisions Structure", False, 
                                   f"IA2 API structure issues: {success_count}/{len(success_criteria)} criteria met. May have missing IA2 data exposure or API problems")
                
        except Exception as e:
            self.log_test_result("IA2 API Decisions Structure", False, f"Exception: {str(e)}")

    async def test_ia2_rr_calculation_formulas(self):
        """Test IA2: RR Calculation Formulas - Verify IA2 uses correct RR formulas"""
        logger.info("\n🔍 TEST IA2: RR Calculation Formulas Verification")
        
        try:
            rr_formula_results = {
                'ia2_decisions_analyzed': 0,
                'correct_rr_formulas': 0,
                'incorrect_rr_formulas': 0,
                'missing_technical_levels': 0,
                'formula_verification_details': [],
                'rr_calculation_patterns': {},
                'database_connection_success': False
            }
            
            logger.info("   🚀 Verifying IA2 RR calculation formulas...")
            logger.info("   📊 Expected: LONG: (TP1-Entry)/(Entry-SL), SHORT: (Entry-TP1)/(SL-Entry)")
            
            try:
                # Connect to MongoDB to analyze IA2 RR calculations
                from pymongo import MongoClient
                mongo_url = "mongodb://localhost:27017"
                client = MongoClient(mongo_url)
                db = client["myapp"]
                rr_formula_results['database_connection_success'] = True
                
                # Get recent IA2 decisions with technical levels
                ia2_decisions = list(db.trading_decisions.find({
                    "$and": [
                        {"$or": [
                            {"ia2_entry_price": {"$exists": True}},
                            {"entry_price": {"$exists": True}}
                        ]},
                        {"$or": [
                            {"ia2_stop_loss": {"$exists": True}},
                            {"stop_loss": {"$exists": True}}
                        ]},
                        {"$or": [
                            {"ia2_take_profit_1": {"$exists": True}},
                            {"take_profit_1": {"$exists": True}}
                        ]},
                        {"signal": {"$in": ["LONG", "SHORT"]}}
                    ]
                }).sort("timestamp", -1).limit(20))
                
                rr_formula_results['ia2_decisions_analyzed'] = len(ia2_decisions)
                logger.info(f"      📊 Found {len(ia2_decisions)} IA2 decisions with technical levels")
                
                if ia2_decisions:
                    for decision in ia2_decisions:
                        symbol = decision.get('symbol', 'UNKNOWN')
                        signal = decision.get('signal', '').upper()
                        
                        # Get technical levels (try IA2-specific fields first, then general fields)
                        entry_price = decision.get('ia2_entry_price') or decision.get('entry_price')
                        stop_loss = decision.get('ia2_stop_loss') or decision.get('stop_loss')
                        take_profit_1 = decision.get('ia2_take_profit_1') or decision.get('take_profit_1')
                        
                        # Get RR values
                        calculated_rr = decision.get('calculated_rr')
                        ia2_calculated_rr = decision.get('ia2_calculated_rr')
                        risk_reward_ratio = decision.get('risk_reward_ratio')
                        
                        # Use the most specific RR value available
                        actual_rr = ia2_calculated_rr or calculated_rr or risk_reward_ratio
                        
                        if all([entry_price, stop_loss, take_profit_1]) and all(isinstance(x, (int, float)) for x in [entry_price, stop_loss, take_profit_1]):
                            # Calculate expected RR based on formulas
                            expected_rr = None
                            formula_used = ""
                            
                            if signal == 'LONG' and entry_price > stop_loss and take_profit_1 > entry_price:
                                expected_rr = (take_profit_1 - entry_price) / (entry_price - stop_loss)
                                formula_used = "LONG: (TP1-Entry)/(Entry-SL)"
                            elif signal == 'SHORT' and stop_loss > entry_price and take_profit_1 < entry_price:
                                expected_rr = (entry_price - take_profit_1) / (stop_loss - entry_price)
                                formula_used = "SHORT: (Entry-TP1)/(SL-Entry)"
                            
                            if expected_rr is not None and actual_rr is not None:
                                # Check if calculated RR matches expected RR (within 10% tolerance)
                                rr_difference = abs(actual_rr - expected_rr)
                                rr_tolerance = expected_rr * 0.1  # 10% tolerance
                                
                                is_correct = rr_difference <= rr_tolerance
                                
                                if is_correct:
                                    rr_formula_results['correct_rr_formulas'] += 1
                                else:
                                    rr_formula_results['incorrect_rr_formulas'] += 1
                                
                                # Store verification details
                                rr_formula_results['formula_verification_details'].append({
                                    'symbol': symbol,
                                    'signal': signal,
                                    'entry': entry_price,
                                    'sl': stop_loss,
                                    'tp1': take_profit_1,
                                    'actual_rr': actual_rr,
                                    'expected_rr': expected_rr,
                                    'formula_used': formula_used,
                                    'is_correct': is_correct,
                                    'difference': rr_difference
                                })
                                
                                logger.info(f"         📊 {symbol} {signal}: actual_rr={actual_rr:.3f}, expected_rr={expected_rr:.3f}, correct={is_correct}")
                            
                            else:
                                logger.warning(f"         ⚠️ {symbol}: Could not calculate expected RR - signal={signal}, levels=({entry_price}, {stop_loss}, {take_profit_1})")
                        
                        else:
                            rr_formula_results['missing_technical_levels'] += 1
                            logger.warning(f"         ❌ {symbol}: Missing or invalid technical levels")
                    
                    # Analyze RR calculation patterns
                    if rr_formula_results['formula_verification_details']:
                        rr_values = [detail['actual_rr'] for detail in rr_formula_results['formula_verification_details']]
                        expected_rr_values = [detail['expected_rr'] for detail in rr_formula_results['formula_verification_details']]
                        
                        rr_formula_results['rr_calculation_patterns'] = {
                            'avg_actual_rr': sum(rr_values) / len(rr_values),
                            'avg_expected_rr': sum(expected_rr_values) / len(expected_rr_values),
                            'min_actual_rr': min(rr_values),
                            'max_actual_rr': max(rr_values),
                            'rr_values_distribution': {
                                'below_1': sum(1 for rr in rr_values if rr < 1.0),
                                '1_to_2': sum(1 for rr in rr_values if 1.0 <= rr < 2.0),
                                '2_to_3': sum(1 for rr in rr_values if 2.0 <= rr < 3.0),
                                'above_3': sum(1 for rr in rr_values if rr >= 3.0)
                            }
                        }
                        
                        logger.info(f"      📊 RR Calculation Patterns:")
                        logger.info(f"         - Average actual RR: {rr_formula_results['rr_calculation_patterns']['avg_actual_rr']:.3f}")
                        logger.info(f"         - Average expected RR: {rr_formula_results['rr_calculation_patterns']['avg_expected_rr']:.3f}")
                        logger.info(f"         - RR distribution: <1={rr_formula_results['rr_calculation_patterns']['rr_values_distribution']['below_1']}, 1-2={rr_formula_results['rr_calculation_patterns']['rr_values_distribution']['1_to_2']}, 2-3={rr_formula_results['rr_calculation_patterns']['rr_values_distribution']['2_to_3']}, >3={rr_formula_results['rr_calculation_patterns']['rr_values_distribution']['above_3']}")
                
                else:
                    logger.warning("      ⚠️ No IA2 decisions with technical levels found")
                
                client.close()
                
            except Exception as e:
                logger.error(f"      ❌ RR formula verification failed: {e}")
            
            # Final analysis
            total_verified = rr_formula_results['correct_rr_formulas'] + rr_formula_results['incorrect_rr_formulas']
            formula_accuracy = rr_formula_results['correct_rr_formulas'] / max(total_verified, 1)
            
            logger.info(f"\n   📊 IA2 RR CALCULATION FORMULAS RESULTS:")
            logger.info(f"      IA2 decisions analyzed: {rr_formula_results['ia2_decisions_analyzed']}")
            logger.info(f"      Correct RR formulas: {rr_formula_results['correct_rr_formulas']}")
            logger.info(f"      Incorrect RR formulas: {rr_formula_results['incorrect_rr_formulas']}")
            logger.info(f"      Missing technical levels: {rr_formula_results['missing_technical_levels']}")
            logger.info(f"      Formula accuracy: {formula_accuracy:.2f}")
            
            # Calculate success based on formula accuracy
            success_criteria = [
                rr_formula_results['database_connection_success'],
                rr_formula_results['ia2_decisions_analyzed'] > 0,
                total_verified > 0,
                formula_accuracy >= 0.7  # At least 70% formula accuracy
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.75:  # 75% success threshold
                self.log_test_result("IA2 RR Calculation Formulas", True, 
                                   f"IA2 RR formulas verification successful: {success_count}/{len(success_criteria)} criteria met. Formula accuracy: {formula_accuracy:.2f}, Correct: {rr_formula_results['correct_rr_formulas']}/{total_verified}")
            else:
                self.log_test_result("IA2 RR Calculation Formulas", False, 
                                   f"IA2 RR formulas issues: {success_count}/{len(success_criteria)} criteria met. May be using incorrect formulas or missing technical levels")
                
        except Exception as e:
            self.log_test_result("IA2 RR Calculation Formulas", False, f"Exception: {str(e)}")

    async def test_ia2_json_parsing_logs_analysis(self):
        """Test IA2: JSON Parsing Logs Analysis - Check for IA2 JSON parsing errors in logs"""
        logger.info("\n🔍 TEST IA2: JSON Parsing Logs Analysis")
        
        try:
            json_parsing_results = {
                'backend_logs_captured': False,
                'ia2_raw_responses_found': 0,
                'ia2_cleaned_json_found': 0,
                'ia2_json_parsing_errors': 0,
                'ia2_fallback_mode_detected': 0,
                'json_parsing_error_details': [],
                'ia2_response_patterns': {},
                'backend_log_analysis': {}
            }
            
            logger.info("   🚀 Analyzing backend logs for IA2 JSON parsing issues...")
            logger.info("   📊 Expected: IA2 generates valid JSON without parsing errors")
            
            try:
                # Capture backend logs
                backend_logs = await self._capture_backend_logs()
                if backend_logs:
                    json_parsing_results['backend_logs_captured'] = True
                    logger.info(f"      ✅ Backend logs captured: {len(backend_logs)} lines")
                    
                    # Analyze logs for IA2-specific patterns
                    ia2_raw_responses = []
                    ia2_cleaned_json = []
                    ia2_json_errors = []
                    ia2_fallback_indicators = []
                    
                    for log_line in backend_logs:
                        log_lower = log_line.lower()
                        
                        # Look for IA2 raw strategic responses
                        if 'ia2: raw strategic response' in log_lower or 'ia2 raw response' in log_lower:
                            ia2_raw_responses.append(log_line.strip())
                        
                        # Look for cleaned IA2 JSON
                        if 'cleaned ia2 json' in log_lower or 'ia2 cleaned json' in log_lower:
                            ia2_cleaned_json.append(log_line.strip())
                        
                        # Look for IA2 JSON parsing errors
                        if any(error_pattern in log_lower for error_pattern in [
                            'ia2 json parsing failed', 'ia2 json error', 'ia2 parsing error',
                            'failed to parse ia2', 'ia2 json decode error'
                        ]):
                            ia2_json_errors.append(log_line.strip())
                        
                        # Look for IA2 fallback mode indicators
                        if any(fallback_pattern in log_lower for fallback_pattern in [
                            'ia2 fallback', 'ia2 using fallback', 'ia2 fallback mode',
                            'ia2 simplified response', 'ia2 emergency response'
                        ]):
                            ia2_fallback_indicators.append(log_line.strip())
                    
                    json_parsing_results['ia2_raw_responses_found'] = len(ia2_raw_responses)
                    json_parsing_results['ia2_cleaned_json_found'] = len(ia2_cleaned_json)
                    json_parsing_results['ia2_json_parsing_errors'] = len(ia2_json_errors)
                    json_parsing_results['ia2_fallback_mode_detected'] = len(ia2_fallback_indicators)
                    
                    logger.info(f"      📊 IA2 Log Analysis:")
                    logger.info(f"         - IA2 raw responses: {len(ia2_raw_responses)}")
                    logger.info(f"         - IA2 cleaned JSON: {len(ia2_cleaned_json)}")
                    logger.info(f"         - IA2 JSON parsing errors: {len(ia2_json_errors)}")
                    logger.info(f"         - IA2 fallback mode detected: {len(ia2_fallback_indicators)}")
                    
                    # Show sample logs
                    if ia2_raw_responses:
                        logger.info(f"      📋 Sample IA2 raw response log: {ia2_raw_responses[0][:200]}...")
                    
                    if ia2_json_errors:
                        logger.error(f"      ❌ Sample IA2 JSON error: {ia2_json_errors[0][:200]}...")
                        json_parsing_results['json_parsing_error_details'] = ia2_json_errors[:5]  # Store first 5 errors
                    
                    if ia2_fallback_indicators:
                        logger.warning(f"      ⚠️ Sample IA2 fallback indicator: {ia2_fallback_indicators[0][:200]}...")
                    
                    # Analyze IA2 response patterns
                    if ia2_raw_responses or ia2_cleaned_json:
                        json_parsing_results['ia2_response_patterns'] = {
                            'total_ia2_responses': len(ia2_raw_responses),
                            'successful_json_cleaning': len(ia2_cleaned_json),
                            'json_success_rate': len(ia2_cleaned_json) / max(len(ia2_raw_responses), 1),
                            'error_rate': len(ia2_json_errors) / max(len(ia2_raw_responses), 1),
                            'fallback_rate': len(ia2_fallback_indicators) / max(len(ia2_raw_responses), 1)
                        }
                        
                        logger.info(f"      📊 IA2 Response Patterns:")
                        logger.info(f"         - JSON success rate: {json_parsing_results['ia2_response_patterns']['json_success_rate']:.2f}")
                        logger.info(f"         - Error rate: {json_parsing_results['ia2_response_patterns']['error_rate']:.2f}")
                        logger.info(f"         - Fallback rate: {json_parsing_results['ia2_response_patterns']['fallback_rate']:.2f}")
                    
                    # General backend log health analysis
                    total_errors = sum(1 for log in backend_logs if 'error' in log.lower())
                    total_warnings = sum(1 for log in backend_logs if 'warning' in log.lower())
                    
                    json_parsing_results['backend_log_analysis'] = {
                        'total_log_lines': len(backend_logs),
                        'total_errors': total_errors,
                        'total_warnings': total_warnings,
                        'error_rate': total_errors / max(len(backend_logs), 1),
                        'backend_health': 'Good' if total_errors < 10 else 'Poor'
                    }
                    
                    logger.info(f"      📊 Backend Log Health:")
                    logger.info(f"         - Total errors: {total_errors}")
                    logger.info(f"         - Total warnings: {total_warnings}")
                    logger.info(f"         - Backend health: {json_parsing_results['backend_log_analysis']['backend_health']}")
                
                else:
                    logger.warning("      ⚠️ Could not capture backend logs")
            
            except Exception as e:
                logger.error(f"      ❌ Log analysis failed: {e}")
            
            # Final analysis
            logger.info(f"\n   📊 IA2 JSON PARSING LOGS ANALYSIS RESULTS:")
            logger.info(f"      Backend logs captured: {json_parsing_results['backend_logs_captured']}")
            logger.info(f"      IA2 raw responses found: {json_parsing_results['ia2_raw_responses_found']}")
            logger.info(f"      IA2 cleaned JSON found: {json_parsing_results['ia2_cleaned_json_found']}")
            logger.info(f"      IA2 JSON parsing errors: {json_parsing_results['ia2_json_parsing_errors']}")
            logger.info(f"      IA2 fallback mode detected: {json_parsing_results['ia2_fallback_mode_detected']}")
            
            # Calculate success based on JSON parsing health
            success_criteria = [
                json_parsing_results['backend_logs_captured'],
                json_parsing_results['ia2_json_parsing_errors'] <= 2,  # Allow up to 2 parsing errors
                json_parsing_results['ia2_fallback_mode_detected'] <= 1,  # Allow up to 1 fallback
                json_parsing_results['ia2_raw_responses_found'] > 0 or json_parsing_results['ia2_cleaned_json_found'] > 0  # Some IA2 activity
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.75:  # 75% success threshold
                self.log_test_result("IA2 JSON Parsing Logs Analysis", True, 
                                   f"IA2 JSON parsing analysis successful: {success_count}/{len(success_criteria)} criteria met. Parsing errors: {json_parsing_results['ia2_json_parsing_errors']}, Fallback mode: {json_parsing_results['ia2_fallback_mode_detected']}")
            else:
                self.log_test_result("IA2 JSON Parsing Logs Analysis", False, 
                                   f"IA2 JSON parsing issues: {success_count}/{len(success_criteria)} criteria met. May have JSON parsing errors or excessive fallback mode usage")
                
        except Exception as e:
            self.log_test_result("IA2 JSON Parsing Logs Analysis", False, f"Exception: {str(e)}")

    async def test_1_scout_system_3_cryptos_test(self):
        """Test 1: Scout System 3 Cryptos Test - Test scout system with 3 cryptos"""
        logger.info("\n🔍 TEST 1: Scout System 3 Cryptos Test")
        
        try:
            scout_results = {
                'opportunities_api_accessible': False,
                'opportunities_found': 0,
                'target_cryptos_present': {},
                'timestamp_persistence_working': False,
                'scout_data_quality': {},
                'technical_data_available': False,
                'opportunities_data': []
            }
            
            logger.info("   🚀 Testing Scout System with 3 cryptos (BTCUSDT, ETHUSDT, SOLUSDT or alternatives)...")
            logger.info("   📊 Expected: Scout system returns opportunities for target cryptos with persistent timestamps")
            
            # Step 1: Test /api/opportunities endpoint
            logger.info("   📞 Testing /api/opportunities endpoint...")
            
            try:
                response = requests.get(f"{self.api_url}/opportunities", timeout=60)
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Handle both direct list and wrapped response
                    if isinstance(response_data, dict) and 'opportunities' in response_data:
                        opportunities = response_data['opportunities']
                    elif isinstance(response_data, list):
                        opportunities = response_data
                    else:
                        opportunities = []
                    
                    scout_results['opportunities_api_accessible'] = True
                    scout_results['opportunities_found'] = len(opportunities)
                    scout_results['opportunities_data'] = opportunities
                    
                    logger.info(f"      ✅ Opportunities API accessible: {len(opportunities)} opportunities found")
                    
                    # Check for target cryptos (preferred) and get available symbols
                    target_cryptos = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
                    found_cryptos = {}
                    available_symbols = []
                    
                    for opportunity in opportunities:
                        symbol = opportunity.get('symbol', '')
                        available_symbols.append(symbol)
                        
                        if symbol in target_cryptos:
                            found_cryptos[symbol] = {
                                'current_price': opportunity.get('current_price'),
                                'volume_24h': opportunity.get('volume_24h'),
                                'price_change_24h': opportunity.get('price_change_24h'),
                                'volatility': opportunity.get('volatility'),
                                'timestamp': opportunity.get('timestamp'),
                                'market_cap': opportunity.get('market_cap')
                            }
                            logger.info(f"         ✅ {symbol} found: Price=${opportunity.get('current_price')}, Vol=${opportunity.get('volume_24h')}")
                    
                    # If preferred symbols not found, use first 3 available symbols for testing
                    if len(found_cryptos) < 2 and len(available_symbols) >= 2:
                        logger.info(f"      📋 Preferred symbols not available, using first available symbols: {available_symbols[:3]}")
                        self.actual_test_symbols = available_symbols[:3]
                        
                        # Add first 3 available symbols to found_cryptos for testing
                        for symbol in self.actual_test_symbols:
                            for opportunity in opportunities:
                                if opportunity.get('symbol') == symbol:
                                    found_cryptos[symbol] = {
                                        'current_price': opportunity.get('current_price'),
                                        'volume_24h': opportunity.get('volume_24h'),
                                        'price_change_24h': opportunity.get('price_change_24h'),
                                        'volatility': opportunity.get('volatility'),
                                        'timestamp': opportunity.get('timestamp'),
                                        'market_cap': opportunity.get('market_cap')
                                    }
                                    break
                    else:
                        self.actual_test_symbols = list(found_cryptos.keys())
                    
                    scout_results['target_cryptos_present'] = found_cryptos
                    
                    if len(found_cryptos) >= 2:
                        logger.info(f"      ✅ Target cryptos found: {list(found_cryptos.keys())}")
                    else:
                        logger.warning(f"      ⚠️ Limited target cryptos found: {list(found_cryptos.keys())}")
                    
                    # Test timestamp persistence by calling API twice
                    logger.info("   🕐 Testing timestamp persistence...")
                    await asyncio.sleep(5)  # Wait 5 seconds
                    
                    response2 = requests.get(f"{self.api_url}/opportunities", timeout=60)
                    if response2.status_code == 200:
                        opportunities2 = response2.json()
                        
                        # Compare timestamps for same symbols
                        timestamp_matches = 0
                        timestamp_total = 0
                        
                        for opp1 in opportunities:
                            symbol1 = opp1.get('symbol')
                            timestamp1 = opp1.get('timestamp')
                            
                            for opp2 in opportunities2:
                                symbol2 = opp2.get('symbol')
                                timestamp2 = opp2.get('timestamp')
                                
                                if symbol1 == symbol2:
                                    timestamp_total += 1
                                    if timestamp1 == timestamp2:
                                        timestamp_matches += 1
                                    break
                        
                        if timestamp_total > 0:
                            persistence_rate = timestamp_matches / timestamp_total
                            scout_results['timestamp_persistence_working'] = persistence_rate >= 0.8
                            logger.info(f"      📊 Timestamp persistence: {timestamp_matches}/{timestamp_total} ({persistence_rate:.2f})")
                        
                    # Analyze data quality
                    quality_metrics = {
                        'with_price': sum(1 for opp in opportunities if opp.get('current_price')),
                        'with_volume': sum(1 for opp in opportunities if opp.get('volume_24h')),
                        'with_change': sum(1 for opp in opportunities if opp.get('price_change_24h') is not None),
                        'with_volatility': sum(1 for opp in opportunities if opp.get('volatility')),
                        'with_timestamp': sum(1 for opp in opportunities if opp.get('timestamp'))
                    }
                    
                    scout_results['scout_data_quality'] = quality_metrics
                    
                    # Check if technical data is available
                    technical_fields = ['rsi_signal', 'macd_trend', 'mfi_signal', 'vwap_signal']
                    technical_count = 0
                    for opp in opportunities[:10]:  # Check first 10
                        for field in technical_fields:
                            if field in opp and opp[field] not in [None, '', 'unknown']:
                                technical_count += 1
                                break
                    
                    scout_results['technical_data_available'] = technical_count > 0
                    
                    logger.info(f"      📊 Data quality: Price={quality_metrics['with_price']}, Volume={quality_metrics['with_volume']}, Technical={technical_count}")
                
                else:
                    logger.error(f"      ❌ Opportunities API failed: HTTP {response.status_code}")
                    if response.text:
                        logger.error(f"         Error response: {response.text[:200]}...")
            
            except Exception as e:
                logger.error(f"      ❌ Opportunities API exception: {e}")
            
            # Final analysis
            logger.info(f"\n   📊 SCOUT SYSTEM 3 CRYPTOS TEST RESULTS:")
            logger.info(f"      Opportunities API accessible: {scout_results['opportunities_api_accessible']}")
            logger.info(f"      Total opportunities found: {scout_results['opportunities_found']}")
            logger.info(f"      Target cryptos present: {len(scout_results['target_cryptos_present'])}/3")
            logger.info(f"      Timestamp persistence working: {scout_results['timestamp_persistence_working']}")
            logger.info(f"      Technical data available: {scout_results['technical_data_available']}")
            
            # Calculate test success
            success_criteria = [
                scout_results['opportunities_api_accessible'],
                scout_results['opportunities_found'] > 0,
                len(scout_results['target_cryptos_present']) >= 2,  # At least 2 target cryptos
                scout_results['timestamp_persistence_working']
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.75:  # 75% success threshold
                self.log_test_result("Scout System 3 Cryptos Test", True, 
                                   f"Scout system working: {success_count}/{len(success_criteria)} criteria met. Target cryptos: {len(scout_results['target_cryptos_present'])}, Opportunities: {scout_results['opportunities_found']}")
            else:
                self.log_test_result("Scout System 3 Cryptos Test", False, 
                                   f"Scout system issues: {success_count}/{len(success_criteria)} criteria met. May have API or timestamp persistence problems")
                
        except Exception as e:
            self.log_test_result("Scout System 3 Cryptos Test", False, f"Exception: {str(e)}")

    async def test_2_technical_indicators_integration_verification(self):
        """Test 2: Technical Indicators Integration Verification - Verify IA1 mentions RSI, MACD, MFI, VWAP with values"""
        logger.info("\n🔍 TEST 2: Technical Indicators Integration Verification")
        
        try:
            integration_results = {
                'analyses_with_technical_indicators': 0,
                'total_analyses': 0,
                'technical_indicators_mentioned': {},
                'specific_values_found': {},
                'reasoning_quality_score': 0.0,
                'balance_technical_vs_patterns': 0.0,
                'successful_analyses': []
            }
            
            logger.info("   🚀 Testing technical indicators integration in IA1 reasoning...")
            logger.info("   📊 Expected: >70% analyses mention RSI, MACD, MFI, VWAP with calculated values")
            
            # Use symbols from previous test or defaults
            test_symbols = self.actual_test_symbols if hasattr(self, 'actual_test_symbols') and self.actual_test_symbols else self.test_symbols
            logger.info(f"   📊 Testing symbols: {test_symbols}")
            
            # Test each symbol for technical indicators integration
            for symbol in test_symbols:
                logger.info(f"\n   📞 Testing IA1 technical indicators integration for {symbol}...")
                integration_results['total_analyses'] += 1
                
                try:
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        
                        # Extract reasoning from IA1 response
                        reasoning = ""
                        if 'ia1_analysis' in analysis_data and isinstance(analysis_data['ia1_analysis'], dict):
                            reasoning = analysis_data['ia1_analysis'].get('reasoning', '')
                        elif 'reasoning' in analysis_data:
                            reasoning = analysis_data.get('reasoning', '')
                        
                        if reasoning:
                            logger.info(f"      ✅ Reasoning field found: {len(reasoning)} chars")
                            
                            # Check for specific technical indicator mentions with values
                            technical_indicators_found = {}
                            specific_values_found = {}
                            
                            # Core technical indicators to check
                            core_indicators = ['RSI', 'MACD', 'MFI', 'VWAP']
                            
                            for indicator in core_indicators:
                                # Count mentions
                                mentions = reasoning.upper().count(indicator.upper())
                                if mentions > 0:
                                    technical_indicators_found[indicator] = mentions
                                    
                                    # Look for specific values (e.g., "RSI 32.1", "MACD -0.015")
                                    import re
                                    value_pattern = rf'{indicator}\s*[:\s]*[\d\.\-]+%?'
                                    matches = re.findall(value_pattern, reasoning, re.IGNORECASE)
                                    if matches:
                                        specific_values_found[indicator] = matches
                                        logger.info(f"         ✅ {indicator} with values: {matches}")
                                    else:
                                        logger.info(f"         📊 {indicator} mentioned {mentions} times (no specific values)")
                            
                            # Check for patterns mentions for balance assessment
                            pattern_keywords = ['pattern', 'support', 'resistance', 'trend', 'channel']
                            pattern_mentions = sum(reasoning.lower().count(keyword) for keyword in pattern_keywords)
                            
                            technical_mentions = sum(technical_indicators_found.values())
                            
                            # Calculate balance
                            total_mentions = technical_mentions + pattern_mentions
                            if total_mentions > 0:
                                technical_ratio = technical_mentions / total_mentions
                                integration_results['balance_technical_vs_patterns'] += technical_ratio
                                logger.info(f"         📊 Technical/Patterns balance: {technical_ratio:.2f} ({technical_mentions}/{total_mentions})")
                            
                            # Store results
                            if technical_indicators_found:
                                integration_results['analyses_with_technical_indicators'] += 1
                                integration_results['successful_analyses'].append({
                                    'symbol': symbol,
                                    'technical_indicators': technical_indicators_found,
                                    'specific_values': specific_values_found,
                                    'reasoning_sample': reasoning[:300] + "..." if len(reasoning) > 300 else reasoning
                                })
                                
                                # Update global counters
                                for indicator, count in technical_indicators_found.items():
                                    if indicator not in integration_results['technical_indicators_mentioned']:
                                        integration_results['technical_indicators_mentioned'][indicator] = 0
                                    integration_results['technical_indicators_mentioned'][indicator] += count
                                
                                # Update specific values
                                for indicator, values in specific_values_found.items():
                                    if indicator not in integration_results['specific_values_found']:
                                        integration_results['specific_values_found'][indicator] = []
                                    integration_results['specific_values_found'][indicator].extend(values)
                                
                                logger.info(f"      ✅ Technical indicators integration SUCCESS for {symbol}")
                            else:
                                logger.warning(f"      ❌ No technical indicators found in {symbol} reasoning")
                        else:
                            logger.warning(f"      ❌ No reasoning field found for {symbol}")
                    
                    else:
                        logger.error(f"      ❌ IA1 analysis for {symbol} failed: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"      ❌ IA1 analysis for {symbol} exception: {e}")
                
                # Wait between analyses
                if symbol != test_symbols[-1]:
                    await asyncio.sleep(10)
            
            # Calculate final metrics
            if integration_results['total_analyses'] > 0:
                technical_integration_rate = integration_results['analyses_with_technical_indicators'] / integration_results['total_analyses']
                avg_balance = integration_results['balance_technical_vs_patterns'] / integration_results['total_analyses']
            else:
                technical_integration_rate = 0.0
                avg_balance = 0.0
            
            logger.info(f"\n   📊 TECHNICAL INDICATORS INTEGRATION VERIFICATION RESULTS:")
            logger.info(f"      Total analyses: {integration_results['total_analyses']}")
            logger.info(f"      Analyses with technical indicators: {integration_results['analyses_with_technical_indicators']}")
            logger.info(f"      Technical integration rate: {technical_integration_rate:.2f}")
            logger.info(f"      Average technical/patterns balance: {avg_balance:.2f}")
            
            if integration_results['technical_indicators_mentioned']:
                logger.info(f"      📊 Technical indicators mentioned:")
                for indicator, count in integration_results['technical_indicators_mentioned'].items():
                    logger.info(f"         - {indicator}: {count} mentions")
            
            if integration_results['specific_values_found']:
                logger.info(f"      📊 Specific values found:")
                for indicator, values in integration_results['specific_values_found'].items():
                    logger.info(f"         - {indicator}: {len(values)} specific values")
            
            # Calculate test success based on review requirements (>70% technical indicators usage)
            success_criteria = [
                integration_results['total_analyses'] >= 2,  # At least 2 analyses
                integration_results['analyses_with_technical_indicators'] >= 1,  # At least 1 with indicators
                technical_integration_rate >= 0.7,  # >70% usage rate (target from review)
                len(integration_results['specific_values_found']) >= 2,  # At least 2 indicators with values
                avg_balance >= 0.3  # At least 30% technical mentions vs patterns
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Technical Indicators Integration Verification", True, 
                                   f"Technical indicators integration successful: {success_count}/{len(success_criteria)} criteria met. Integration rate: {technical_integration_rate:.2f}, Balance: {avg_balance:.2f}, Specific values: {len(integration_results['specific_values_found'])}")
            else:
                self.log_test_result("Technical Indicators Integration Verification", False, 
                                   f"Technical indicators integration issues: {success_count}/{len(success_criteria)} criteria met. Integration rate may be below 70% target or insufficient specific values")
                
        except Exception as e:
            self.log_test_result("Technical Indicators Integration Verification", False, f"Exception: {str(e)}")

    async def test_3_json_schema_validation_fix(self):
        """Test 3: JSON Schema Validation Fix - Verify IA1 uses 'signal' field correctly without fallback"""
        logger.info("\n🔍 TEST 3: JSON Schema Validation Fix")
        
        try:
            json_results = {
                'total_analyses': 0,
                'valid_json_responses': 0,
                'fallback_analyses': 0,
                'signal_field_present': 0,
                'complete_json_structure': 0,
                'json_parsing_errors': 0,
                'successful_responses': []
            }
            
            logger.info("   🚀 Testing JSON schema validation fix - 'signal' field usage...")
            logger.info("   📊 Expected: 100% JSON valid responses without 'JSON parsing failed' fallback")
            
            # Use symbols from previous tests
            test_symbols = self.actual_test_symbols if hasattr(self, 'actual_test_symbols') and self.actual_test_symbols else self.test_symbols
            logger.info(f"   📊 Testing symbols: {test_symbols}")
            
            # Test each symbol for JSON schema validation
            for symbol in test_symbols:
                logger.info(f"\n   📞 Testing JSON schema validation for {symbol}...")
                json_results['total_analyses'] += 1
                
                try:
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        
                        # Check for valid JSON structure
                        is_valid_json = True
                        has_signal_field = False
                        is_fallback = False
                        has_complete_structure = False
                        
                        # Extract IA1 analysis
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        
                        if isinstance(ia1_analysis, dict):
                            json_results['valid_json_responses'] += 1
                            logger.info(f"      ✅ Valid JSON response received")
                            
                            # Check for 'signal' field
                            if 'signal' in ia1_analysis:
                                has_signal_field = True
                                json_results['signal_field_present'] += 1
                                signal_value = ia1_analysis['signal']
                                logger.info(f"         ✅ 'signal' field present: {signal_value}")
                            else:
                                logger.warning(f"         ❌ 'signal' field missing")
                            
                            # Check for complete JSON structure
                            required_fields = ['signal', 'confidence', 'reasoning']
                            optional_fields = ['technical_indicators', 'patterns', 'entry_price']
                            
                            required_present = sum(1 for field in required_fields if field in ia1_analysis)
                            optional_present = sum(1 for field in optional_fields if field in ia1_analysis)
                            
                            if required_present == len(required_fields):
                                has_complete_structure = True
                                json_results['complete_json_structure'] += 1
                                logger.info(f"         ✅ Complete JSON structure: {required_present}/{len(required_fields)} required, {optional_present}/{len(optional_fields)} optional")
                            else:
                                logger.warning(f"         ⚠️ Incomplete JSON structure: {required_present}/{len(required_fields)} required fields")
                            
                            # Check for fallback analysis indicators
                            reasoning = ia1_analysis.get('reasoning', '')
                            if reasoning:
                                fallback_indicators = [
                                    'fallback analysis', 'json parsing failed', 'pattern-only analysis',
                                    'technical analysis fallback', 'using detected pattern'
                                ]
                                
                                is_fallback = any(indicator in reasoning.lower() for indicator in fallback_indicators)
                                
                                if is_fallback:
                                    json_results['fallback_analyses'] += 1
                                    logger.warning(f"         ❌ Fallback analysis detected")
                                else:
                                    logger.info(f"         ✅ No fallback analysis - full JSON processing")
                            
                            # Store successful response
                            if has_signal_field and has_complete_structure and not is_fallback:
                                json_results['successful_responses'].append({
                                    'symbol': symbol,
                                    'signal': ia1_analysis.get('signal'),
                                    'confidence': ia1_analysis.get('confidence'),
                                    'has_technical_indicators': 'technical_indicators' in ia1_analysis,
                                    'has_patterns': 'patterns' in ia1_analysis,
                                    'reasoning_length': len(reasoning)
                                })
                                logger.info(f"      ✅ JSON schema validation SUCCESS for {symbol}")
                            else:
                                logger.warning(f"      ❌ JSON schema validation issues for {symbol}")
                        
                        else:
                            json_results['json_parsing_errors'] += 1
                            logger.error(f"      ❌ Invalid JSON structure for {symbol}")
                    
                    else:
                        logger.error(f"      ❌ IA1 analysis for {symbol} failed: HTTP {response.status_code}")
                        json_results['json_parsing_errors'] += 1
                
                except Exception as e:
                    logger.error(f"      ❌ IA1 analysis for {symbol} exception: {e}")
                    json_results['json_parsing_errors'] += 1
                
                # Wait between analyses
                if symbol != test_symbols[-1]:
                    await asyncio.sleep(10)
            
            # Calculate final metrics
            if json_results['total_analyses'] > 0:
                valid_json_rate = json_results['valid_json_responses'] / json_results['total_analyses']
                signal_field_rate = json_results['signal_field_present'] / json_results['total_analyses']
                fallback_rate = json_results['fallback_analyses'] / json_results['total_analyses']
                complete_structure_rate = json_results['complete_json_structure'] / json_results['total_analyses']
            else:
                valid_json_rate = signal_field_rate = fallback_rate = complete_structure_rate = 0.0
            
            logger.info(f"\n   📊 JSON SCHEMA VALIDATION FIX RESULTS:")
            logger.info(f"      Total analyses: {json_results['total_analyses']}")
            logger.info(f"      Valid JSON responses: {json_results['valid_json_responses']} ({valid_json_rate:.2f})")
            logger.info(f"      'signal' field present: {json_results['signal_field_present']} ({signal_field_rate:.2f})")
            logger.info(f"      Complete JSON structure: {json_results['complete_json_structure']} ({complete_structure_rate:.2f})")
            logger.info(f"      Fallback analyses: {json_results['fallback_analyses']} ({fallback_rate:.2f})")
            logger.info(f"      JSON parsing errors: {json_results['json_parsing_errors']}")
            
            if json_results['successful_responses']:
                logger.info(f"      📊 Successful responses:")
                for response in json_results['successful_responses']:
                    logger.info(f"         - {response['symbol']}: signal={response['signal']}, confidence={response['confidence']}")
            
            # Calculate test success based on review requirements (100% JSON valid without fallback)
            success_criteria = [
                json_results['total_analyses'] >= 2,  # At least 2 analyses
                json_results['valid_json_responses'] >= 2,  # At least 2 valid JSON
                valid_json_rate >= 0.67,  # At least 67% valid JSON (2/3)
                json_results['fallback_analyses'] == 0,  # No fallback analyses
                signal_field_rate >= 0.67  # At least 67% with signal field
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("JSON Schema Validation Fix", True, 
                                   f"JSON schema validation successful: {success_count}/{len(success_criteria)} criteria met. Valid JSON rate: {valid_json_rate:.2f}, No fallbacks: {json_results['fallback_analyses'] == 0}, Signal field rate: {signal_field_rate:.2f}")
            else:
                self.log_test_result("JSON Schema Validation Fix", False, 
                                   f"JSON schema validation issues: {success_count}/{len(success_criteria)} criteria met. May still have JSON parsing failures or fallback analyses")
                
        except Exception as e:
            self.log_test_result("JSON Schema Validation Fix", False, f"Exception: {str(e)}")

    async def test_4_enhanced_rr_calculation_verification(self):
        """Test 4: Enhanced RR Calculation Verification - Test new mathematical RR formulas vs fixed values"""
        logger.info("\n🔍 TEST 4: Enhanced RR Calculation Verification")
        
        try:
            rr_results = {
                'total_analyses': 0,
                'analyses_with_calculated_rr': 0,
                'analyses_with_rr_reasoning': 0,
                'mathematical_formulas_used': 0,
                'technical_levels_coherent': 0,
                'rr_values_found': [],
                'rr_reasoning_samples': [],
                'escalation_criteria_met': 0,
                'successful_analyses': []
            }
            
            logger.info("   🚀 Testing enhanced RR calculation system with mathematical formulas...")
            logger.info("   📊 Expected: calculated_rr uses real formulas, rr_reasoning explains technical levels")
            
            # Use symbols from previous tests or get from opportunities
            test_symbols = self.actual_test_symbols if hasattr(self, 'actual_test_symbols') and self.actual_test_symbols else ['BTCUSDT', 'ETHUSDT']
            logger.info(f"   📊 Testing symbols: {test_symbols}")
            
            # Test each symbol for enhanced RR calculation
            for symbol in test_symbols:
                logger.info(f"\n   📞 Testing enhanced RR calculation for {symbol}...")
                rr_results['total_analyses'] += 1
                
                try:
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        
                        # Extract IA1 analysis data
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        
                        # Check for calculated_rr field
                        calculated_rr = ia1_analysis.get('calculated_rr') or ia1_analysis.get('risk_reward_ratio')
                        if calculated_rr is not None:
                            rr_results['analyses_with_calculated_rr'] += 1
                            rr_results['rr_values_found'].append({
                                'symbol': symbol,
                                'calculated_rr': calculated_rr
                            })
                            logger.info(f"      ✅ calculated_rr found: {calculated_rr}")
                            
                            # Check if RR is not a fixed value (1.0, 2.2, etc.)
                            fixed_values = [1.0, 2.0, 2.2, 3.0]
                            is_calculated = not any(abs(float(calculated_rr) - fixed) < 0.01 for fixed in fixed_values)
                            
                            if is_calculated:
                                rr_results['mathematical_formulas_used'] += 1
                                logger.info(f"         ✅ RR appears calculated (not fixed): {calculated_rr}")
                            else:
                                logger.warning(f"         ⚠️ RR appears to be fixed value: {calculated_rr}")
                        else:
                            logger.warning(f"      ❌ No calculated_rr field found for {symbol}")
                        
                        # Check for rr_reasoning field
                        rr_reasoning = ia1_analysis.get('rr_reasoning', '')
                        if rr_reasoning:
                            rr_results['analyses_with_rr_reasoning'] += 1
                            rr_results['rr_reasoning_samples'].append({
                                'symbol': symbol,
                                'rr_reasoning': rr_reasoning[:200] + "..." if len(rr_reasoning) > 200 else rr_reasoning
                            })
                            logger.info(f"      ✅ rr_reasoning found: {len(rr_reasoning)} chars")
                            
                            # Check for technical level mentions (VWAP, EMA21, SMA50)
                            technical_levels = ['VWAP', 'EMA21', 'EMA', 'SMA50', 'SMA', 'support', 'resistance']
                            technical_mentions = sum(1 for level in technical_levels if level.lower() in rr_reasoning.lower())
                            
                            if technical_mentions >= 2:
                                rr_results['technical_levels_coherent'] += 1
                                logger.info(f"         ✅ Technical levels mentioned: {technical_mentions} references")
                            else:
                                logger.warning(f"         ⚠️ Limited technical level mentions: {technical_mentions}")
                        else:
                            logger.warning(f"      ❌ No rr_reasoning field found for {symbol}")
                        
                        # Check technical levels coherence
                        entry_price = ia1_analysis.get('entry_price')
                        stop_loss = ia1_analysis.get('stop_loss_price') or ia1_analysis.get('stop_loss')
                        take_profit = ia1_analysis.get('take_profit_price') or ia1_analysis.get('take_profit')
                        signal = ia1_analysis.get('signal', '').upper()
                        
                        if all([entry_price, stop_loss, take_profit]):
                            logger.info(f"      📊 Technical levels: Entry={entry_price}, SL={stop_loss}, TP={take_profit}, Signal={signal}")
                            
                            # Verify logical order for LONG/SHORT
                            levels_coherent = False
                            if signal == 'LONG' and stop_loss < entry_price < take_profit:
                                levels_coherent = True
                                logger.info(f"         ✅ LONG levels coherent: SL < Entry < TP")
                            elif signal == 'SHORT' and take_profit < entry_price < stop_loss:
                                levels_coherent = True
                                logger.info(f"         ✅ SHORT levels coherent: TP < Entry < SL")
                            else:
                                logger.warning(f"         ⚠️ Levels may not be coherent for {signal} signal")
                            
                            # Calculate expected RR manually to verify
                            if levels_coherent:
                                if signal == 'LONG':
                                    expected_rr = (take_profit - entry_price) / (entry_price - stop_loss)
                                else:  # SHORT
                                    expected_rr = (entry_price - take_profit) / (stop_loss - entry_price)
                                
                                if calculated_rr:
                                    rr_difference = abs(float(calculated_rr) - expected_rr)
                                    if rr_difference < 0.1:  # Within 0.1 tolerance
                                        logger.info(f"         ✅ RR calculation verified: Expected={expected_rr:.2f}, Got={calculated_rr}")
                                    else:
                                        logger.warning(f"         ⚠️ RR calculation mismatch: Expected={expected_rr:.2f}, Got={calculated_rr}")
                        
                        # Check escalation criteria (RR > 2.0 or confidence > 70%)
                        confidence = ia1_analysis.get('confidence', 0)
                        if calculated_rr and (float(calculated_rr) > 2.0 or confidence > 0.7):
                            rr_results['escalation_criteria_met'] += 1
                            logger.info(f"      🚀 Escalation criteria met: RR={calculated_rr}, Confidence={confidence}")
                        
                        # Store successful analysis
                        if calculated_rr and rr_reasoning:
                            rr_results['successful_analyses'].append({
                                'symbol': symbol,
                                'calculated_rr': calculated_rr,
                                'rr_reasoning_length': len(rr_reasoning),
                                'technical_levels_present': all([entry_price, stop_loss, take_profit]),
                                'escalation_eligible': calculated_rr and float(calculated_rr) > 2.0
                            })
                            logger.info(f"      ✅ Enhanced RR calculation SUCCESS for {symbol}")
                    
                    else:
                        logger.error(f"      ❌ IA1 analysis for {symbol} failed: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"      ❌ IA1 analysis for {symbol} exception: {e}")
                
                # Wait between analyses
                if symbol != test_symbols[-1]:
                    await asyncio.sleep(10)
            
            # Calculate final metrics
            if rr_results['total_analyses'] > 0:
                calculated_rr_rate = rr_results['analyses_with_calculated_rr'] / rr_results['total_analyses']
                rr_reasoning_rate = rr_results['analyses_with_rr_reasoning'] / rr_results['total_analyses']
                mathematical_formula_rate = rr_results['mathematical_formulas_used'] / rr_results['total_analyses']
                technical_coherence_rate = rr_results['technical_levels_coherent'] / rr_results['total_analyses']
            else:
                calculated_rr_rate = rr_reasoning_rate = mathematical_formula_rate = technical_coherence_rate = 0.0
            
            logger.info(f"\n   📊 ENHANCED RR CALCULATION VERIFICATION RESULTS:")
            logger.info(f"      Total analyses: {rr_results['total_analyses']}")
            logger.info(f"      Analyses with calculated_rr: {rr_results['analyses_with_calculated_rr']} ({calculated_rr_rate:.2f})")
            logger.info(f"      Analyses with rr_reasoning: {rr_results['analyses_with_rr_reasoning']} ({rr_reasoning_rate:.2f})")
            logger.info(f"      Mathematical formulas used: {rr_results['mathematical_formulas_used']} ({mathematical_formula_rate:.2f})")
            logger.info(f"      Technical levels coherent: {rr_results['technical_levels_coherent']} ({technical_coherence_rate:.2f})")
            logger.info(f"      Escalation criteria met: {rr_results['escalation_criteria_met']}")
            
            if rr_results['rr_values_found']:
                logger.info(f"      📊 RR values found:")
                for rr_data in rr_results['rr_values_found']:
                    logger.info(f"         - {rr_data['symbol']}: {rr_data['calculated_rr']}")
            
            if rr_results['rr_reasoning_samples']:
                logger.info(f"      📊 RR reasoning samples:")
                for reasoning_data in rr_results['rr_reasoning_samples'][:2]:  # Show first 2
                    logger.info(f"         - {reasoning_data['symbol']}: {reasoning_data['rr_reasoning']}")
            
            # Calculate test success based on review requirements
            success_criteria = [
                rr_results['total_analyses'] >= 1,  # At least 1 analysis
                rr_results['analyses_with_calculated_rr'] >= 1,  # At least 1 with calculated_rr
                calculated_rr_rate >= 0.5,  # At least 50% with calculated_rr
                rr_results['analyses_with_rr_reasoning'] >= 1,  # At least 1 with rr_reasoning
                mathematical_formula_rate >= 0.5  # At least 50% using mathematical formulas
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Enhanced RR Calculation Verification", True, 
                                   f"Enhanced RR calculation successful: {success_count}/{len(success_criteria)} criteria met. Calculated RR rate: {calculated_rr_rate:.2f}, Mathematical formulas: {mathematical_formula_rate:.2f}, RR reasoning: {rr_reasoning_rate:.2f}")
            else:
                self.log_test_result("Enhanced RR Calculation Verification", False, 
                                   f"Enhanced RR calculation issues: {success_count}/{len(success_criteria)} criteria met. May still be using fixed RR values or missing rr_reasoning")
                
        except Exception as e:
            self.log_test_result("Enhanced RR Calculation Verification", False, f"Exception: {str(e)}")

    async def test_5_confluence_6_indicators_reasoning_verification(self):
        """Test 5: Confluence 6 Indicators Reasoning Verification - Verify explicit mentions of RSI, MACD, MFI, VWAP, EMA, patterns"""
        logger.info("\n🔍 TEST 5: Confluence 6 Indicators Reasoning Verification")
        
        try:
            confluence_results = {
                'total_analyses': 0,
                'analyses_with_confluence_reasoning': 0,
                'indicators_mentioned': {},
                'confluence_format_found': 0,
                'technical_indicators_analysis_present': 0,
                'reasoning_quality_scores': [],
                'successful_analyses': []
            }
            
            logger.info("   🚀 Testing confluence 6 indicators reasoning with explicit mentions...")
            logger.info("   📊 Expected: 'CONFLUENCE ANALYSIS: RSI X, MACD Y, MFI Z...' format")
            
            # Core 6 indicators to verify
            core_6_indicators = ['RSI', 'MACD', 'MFI', 'VWAP', 'EMA', 'patterns']
            
            # Use symbols from previous tests
            test_symbols = self.actual_test_symbols if hasattr(self, 'actual_test_symbols') and self.actual_test_symbols else ['BTCUSDT', 'ETHUSDT']
            logger.info(f"   📊 Testing symbols: {test_symbols}")
            
            # Test each symbol for confluence reasoning
            for symbol in test_symbols:
                logger.info(f"\n   📞 Testing confluence 6 indicators reasoning for {symbol}...")
                confluence_results['total_analyses'] += 1
                
                try:
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        
                        # Extract reasoning from IA1 analysis
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        reasoning = ia1_analysis.get('reasoning', '')
                        
                        if reasoning:
                            logger.info(f"      ✅ Reasoning field found: {len(reasoning)} chars")
                            
                            # Check for explicit indicator mentions with states/values
                            indicators_found = {}
                            for indicator in core_6_indicators:
                                # Count mentions and look for state descriptions
                                mentions = reasoning.upper().count(indicator.upper())
                                if mentions > 0:
                                    indicators_found[indicator] = mentions
                                    
                                    # Look for state descriptions (overbought, oversold, bullish, bearish, etc.)
                                    import re
                                    if indicator == 'RSI':
                                        states = re.findall(r'RSI[^.]*?(overbought|oversold|neutral|bullish|bearish)', reasoning, re.IGNORECASE)
                                    elif indicator == 'MACD':
                                        states = re.findall(r'MACD[^.]*?(bullish|bearish|crossover|divergence|histogram)', reasoning, re.IGNORECASE)
                                    elif indicator == 'MFI':
                                        states = re.findall(r'MFI[^.]*?(accumulation|distribution|neutral|institutional)', reasoning, re.IGNORECASE)
                                    elif indicator == 'VWAP':
                                        states = re.findall(r'VWAP[^.]*?(above|below|precision|overbought|oversold)', reasoning, re.IGNORECASE)
                                    elif indicator == 'EMA':
                                        states = re.findall(r'EMA[^.]*?(hierarchy|cross|golden|death|bullish|bearish)', reasoning, re.IGNORECASE)
                                    else:
                                        states = []
                                    
                                    if states:
                                        logger.info(f"         ✅ {indicator} with states: {states}")
                                    else:
                                        logger.info(f"         📊 {indicator} mentioned {mentions} times")
                            
                            # Check for confluence format
                            confluence_patterns = [
                                r'confluence[^.]*?rsi[^.]*?macd[^.]*?mfi',
                                r'6[/-]6|5[/-]6|4[/-]6',
                                r'confluence.*?analysis',
                                r'indicators.*?align'
                            ]
                            
                            confluence_format_found = False
                            for pattern in confluence_patterns:
                                if re.search(pattern, reasoning, re.IGNORECASE):
                                    confluence_format_found = True
                                    confluence_results['confluence_format_found'] += 1
                                    logger.info(f"         ✅ Confluence format detected")
                                    break
                            
                            if not confluence_format_found:
                                logger.info(f"         📊 No explicit confluence format found")
                            
                            # Check for technical_indicators_analysis field
                            technical_indicators_analysis = ia1_analysis.get('technical_indicators_analysis', {})
                            if technical_indicators_analysis:
                                confluence_results['technical_indicators_analysis_present'] += 1
                                logger.info(f"      ✅ technical_indicators_analysis field present")
                                
                                # Check for specific indicator analysis
                                indicator_fields = ['rsi_impact', 'macd_influence', 'mfi_score', 'vwap_score', 'ema_hierarchy_analysis']
                                present_fields = sum(1 for field in indicator_fields if field in technical_indicators_analysis)
                                logger.info(f"         📊 Indicator analysis fields: {present_fields}/{len(indicator_fields)}")
                            
                            # Calculate reasoning quality score
                            quality_factors = [
                                len(indicators_found) >= 4,  # At least 4 of 6 indicators mentioned
                                confluence_format_found,  # Confluence format present
                                len(reasoning) >= 200,  # Substantial reasoning
                                'technical' in reasoning.lower(),  # Technical analysis mentioned
                                any(state in reasoning.lower() for state in ['overbought', 'oversold', 'bullish', 'bearish'])  # Technical states
                            ]
                            quality_score = sum(quality_factors) / len(quality_factors)
                            confluence_results['reasoning_quality_scores'].append(quality_score)
                            
                            # Store results
                            if len(indicators_found) >= 3:  # At least 3 indicators mentioned
                                confluence_results['analyses_with_confluence_reasoning'] += 1
                                confluence_results['successful_analyses'].append({
                                    'symbol': symbol,
                                    'indicators_found': indicators_found,
                                    'confluence_format': confluence_format_found,
                                    'quality_score': quality_score,
                                    'reasoning_sample': reasoning[:300] + "..." if len(reasoning) > 300 else reasoning
                                })
                                
                                # Update global indicator counters
                                for indicator, count in indicators_found.items():
                                    if indicator not in confluence_results['indicators_mentioned']:
                                        confluence_results['indicators_mentioned'][indicator] = 0
                                    confluence_results['indicators_mentioned'][indicator] += count
                                
                                logger.info(f"      ✅ Confluence reasoning SUCCESS for {symbol} (quality: {quality_score:.2f})")
                            else:
                                logger.warning(f"      ❌ Insufficient confluence reasoning for {symbol}")
                        else:
                            logger.warning(f"      ❌ No reasoning field found for {symbol}")
                    
                    else:
                        logger.error(f"      ❌ IA1 analysis for {symbol} failed: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"      ❌ IA1 analysis for {symbol} exception: {e}")
                
                # Wait between analyses
                if symbol != test_symbols[-1]:
                    await asyncio.sleep(10)
            
            # Calculate final metrics
            if confluence_results['total_analyses'] > 0:
                confluence_reasoning_rate = confluence_results['analyses_with_confluence_reasoning'] / confluence_results['total_analyses']
                confluence_format_rate = confluence_results['confluence_format_found'] / confluence_results['total_analyses']
                technical_analysis_rate = confluence_results['technical_indicators_analysis_present'] / confluence_results['total_analyses']
                avg_quality_score = sum(confluence_results['reasoning_quality_scores']) / len(confluence_results['reasoning_quality_scores']) if confluence_results['reasoning_quality_scores'] else 0.0
            else:
                confluence_reasoning_rate = confluence_format_rate = technical_analysis_rate = avg_quality_score = 0.0
            
            logger.info(f"\n   📊 CONFLUENCE 6 INDICATORS REASONING VERIFICATION RESULTS:")
            logger.info(f"      Total analyses: {confluence_results['total_analyses']}")
            logger.info(f"      Analyses with confluence reasoning: {confluence_results['analyses_with_confluence_reasoning']} ({confluence_reasoning_rate:.2f})")
            logger.info(f"      Confluence format found: {confluence_results['confluence_format_found']} ({confluence_format_rate:.2f})")
            logger.info(f"      Technical indicators analysis present: {confluence_results['technical_indicators_analysis_present']} ({technical_analysis_rate:.2f})")
            logger.info(f"      Average reasoning quality score: {avg_quality_score:.2f}")
            
            if confluence_results['indicators_mentioned']:
                logger.info(f"      📊 Indicators mentioned:")
                for indicator, count in confluence_results['indicators_mentioned'].items():
                    logger.info(f"         - {indicator}: {count} mentions")
            
            # Calculate test success based on review requirements
            success_criteria = [
                confluence_results['total_analyses'] >= 1,  # At least 1 analysis
                confluence_results['analyses_with_confluence_reasoning'] >= 1,  # At least 1 with confluence reasoning
                confluence_reasoning_rate >= 0.5,  # At least 50% with confluence reasoning
                len(confluence_results['indicators_mentioned']) >= 4,  # At least 4 different indicators mentioned
                avg_quality_score >= 0.6  # Average quality score >= 60%
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Confluence 6 Indicators Reasoning Verification", True, 
                                   f"Confluence reasoning successful: {success_count}/{len(success_criteria)} criteria met. Confluence rate: {confluence_reasoning_rate:.2f}, Quality score: {avg_quality_score:.2f}, Indicators: {len(confluence_results['indicators_mentioned'])}")
            else:
                self.log_test_result("Confluence 6 Indicators Reasoning Verification", False, 
                                   f"Confluence reasoning issues: {success_count}/{len(success_criteria)} criteria met. May be missing explicit indicator mentions or confluence format")
                
        except Exception as e:
            self.log_test_result("Confluence 6 Indicators Reasoning Verification", False, f"Exception: {str(e)}")

    async def test_6_system_quality_assessment(self):
        """Test 6: System Quality Assessment - Overall system stability and quality"""
        logger.info("\n🔍 TEST 6: System Quality Assessment")
        
        try:
            quality_results = {
                'backend_stability': True,
                'api_response_times': [],
                'system_errors': 0,
                'http_502_errors': 0,
                'analyses_quality_score': 0.0,
                'overall_system_health': 'Unknown'
            }
            
            logger.info("   🚀 Assessing overall system quality and stability...")
            logger.info("   📊 Expected: Stable backend, no HTTP 502 errors, quality analyses")
            
            # Test system stability with multiple API calls
            logger.info("   📞 Testing system stability with multiple API calls...")
            
            test_endpoints = [
                '/opportunities',
                '/analyses',
                '/debug-anti-doublon'
            ]
            
            for endpoint in test_endpoints:
                try:
                    start_time = time.time()
                    response = requests.get(f"{self.api_url}{endpoint}", timeout=30)
                    response_time = time.time() - start_time
                    
                    quality_results['api_response_times'].append({
                        'endpoint': endpoint,
                        'response_time': response_time,
                        'status_code': response.status_code
                    })
                    
                    if response.status_code == 200:
                        logger.info(f"      ✅ {endpoint}: HTTP 200 ({response_time:.2f}s)")
                    elif response.status_code == 502:
                        quality_results['http_502_errors'] += 1
                        quality_results['backend_stability'] = False
                        logger.error(f"      ❌ {endpoint}: HTTP 502 - Backend Error")
                    else:
                        quality_results['system_errors'] += 1
                        logger.warning(f"      ⚠️ {endpoint}: HTTP {response.status_code}")
                
                except Exception as e:
                    quality_results['system_errors'] += 1
                    quality_results['backend_stability'] = False
                    logger.error(f"      ❌ {endpoint}: Exception - {e}")
                
                await asyncio.sleep(2)  # Brief pause between calls
            
            # Check backend logs for system health
            logger.info("   📋 Checking backend logs for system health...")
            
            try:
                backend_logs = await self._capture_backend_logs()
                if backend_logs:
                    # Look for error patterns
                    error_patterns = [
                        'error', 'exception', 'failed', 'crash', 'timeout'
                    ]
                    
                    error_count = 0
                    for log_line in backend_logs:
                        if any(pattern in log_line.lower() for pattern in error_patterns):
                            error_count += 1
                    
                    total_logs = len(backend_logs)
                    error_rate = error_count / max(total_logs, 1)
                    
                    logger.info(f"      📊 Backend logs: {total_logs} lines, {error_count} errors ({error_rate:.2f} error rate)")
                    
                    if error_rate > 0.1:  # More than 10% error rate
                        quality_results['backend_stability'] = False
                        logger.warning(f"      ⚠️ High error rate in backend logs")
                    else:
                        logger.info(f"      ✅ Backend logs show good stability")
                        
            except Exception as e:
                logger.warning(f"      ⚠️ Could not analyze backend logs: {e}")
            
            # Calculate average response times
            if quality_results['api_response_times']:
                avg_response_time = sum(rt['response_time'] for rt in quality_results['api_response_times']) / len(quality_results['api_response_times'])
                logger.info(f"      📊 Average API response time: {avg_response_time:.2f}s")
            
            # Determine overall system health
            health_factors = [
                quality_results['backend_stability'],
                quality_results['http_502_errors'] == 0,
                quality_results['system_errors'] <= 1,
                len(quality_results['api_response_times']) >= 2
            ]
            
            health_score = sum(health_factors) / len(health_factors)
            
            if health_score >= 0.75:
                quality_results['overall_system_health'] = 'Good'
            elif health_score >= 0.5:
                quality_results['overall_system_health'] = 'Fair'
            else:
                quality_results['overall_system_health'] = 'Poor'
            
            logger.info(f"\n   📊 SYSTEM QUALITY ASSESSMENT RESULTS:")
            logger.info(f"      Backend stability: {quality_results['backend_stability']}")
            logger.info(f"      HTTP 502 errors: {quality_results['http_502_errors']}")
            logger.info(f"      System errors: {quality_results['system_errors']}")
            logger.info(f"      API endpoints tested: {len(quality_results['api_response_times'])}")
            logger.info(f"      Overall system health: {quality_results['overall_system_health']}")
            
            # Calculate test success
            success_criteria = [
                quality_results['backend_stability'],
                quality_results['http_502_errors'] == 0,
                quality_results['system_errors'] <= 1,
                quality_results['overall_system_health'] in ['Good', 'Fair']
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.75:  # 75% success threshold
                self.log_test_result("System Quality Assessment", True, 
                                   f"System quality good: {success_count}/{len(success_criteria)} criteria met. Health: {quality_results['overall_system_health']}, Stability: {quality_results['backend_stability']}, HTTP 502 errors: {quality_results['http_502_errors']}")
            else:
                self.log_test_result("System Quality Assessment", False, 
                                   f"System quality issues: {success_count}/{len(success_criteria)} criteria met. May have stability problems or HTTP 502 errors")
                
        except Exception as e:
            self.log_test_result("System Quality Assessment", False, f"Exception: {str(e)}")
        
        try:
            scout_results = {
                'opportunities_api_accessible': False,
                'opportunities_found': 0,
                'target_cryptos_present': {},
                'timestamp_persistence_working': False,
                'scout_data_quality': {},
                'technical_data_available': False,
                'opportunities_data': []
            }
            
            logger.info("   🚀 Testing Scout System with limited cryptos (BTCUSDT, ETHUSDT, ADAUSDT)...")
            logger.info("   📊 Expected: Scout system returns opportunities for target cryptos with persistent timestamps")
            
            # Step 1: Test /api/opportunities endpoint
            logger.info("   📞 Testing /api/opportunities endpoint...")
            
            try:
                response = requests.get(f"{self.api_url}/opportunities", timeout=60)
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Handle both direct list and wrapped response
                    if isinstance(response_data, dict) and 'opportunities' in response_data:
                        opportunities = response_data['opportunities']
                    elif isinstance(response_data, list):
                        opportunities = response_data
                    else:
                        opportunities = []
                    
                    scout_results['opportunities_api_accessible'] = True
                    scout_results['opportunities_found'] = len(opportunities)
                    scout_results['opportunities_data'] = opportunities
                    
                    logger.info(f"      ✅ Opportunities API accessible: {len(opportunities)} opportunities found")
                    
                    # Check for target cryptos (preferred) and get available symbols
                    target_cryptos = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
                    found_cryptos = {}
                    available_symbols = []
                    
                    for opportunity in opportunities:
                        symbol = opportunity.get('symbol', '')
                        available_symbols.append(symbol)
                        
                        if symbol in target_cryptos:
                            found_cryptos[symbol] = {
                                'current_price': opportunity.get('current_price'),
                                'volume_24h': opportunity.get('volume_24h'),
                                'price_change_24h': opportunity.get('price_change_24h'),
                                'volatility': opportunity.get('volatility'),
                                'timestamp': opportunity.get('timestamp'),
                                'market_cap': opportunity.get('market_cap')
                            }
                            logger.info(f"         ✅ {symbol} found: Price=${opportunity.get('current_price')}, Vol=${opportunity.get('volume_24h')}")
                    
                    # If preferred symbols not found, use first 3 available symbols for testing
                    if len(found_cryptos) < 2 and len(available_symbols) >= 2:
                        logger.info(f"      📋 Preferred symbols not available, using first available symbols: {available_symbols[:3]}")
                        self.actual_test_symbols = available_symbols[:3]
                        
                        # Add first 3 available symbols to found_cryptos for testing
                        for symbol in self.actual_test_symbols:
                            for opportunity in opportunities:
                                if opportunity.get('symbol') == symbol:
                                    found_cryptos[symbol] = {
                                        'current_price': opportunity.get('current_price'),
                                        'volume_24h': opportunity.get('volume_24h'),
                                        'price_change_24h': opportunity.get('price_change_24h'),
                                        'volatility': opportunity.get('volatility'),
                                        'timestamp': opportunity.get('timestamp'),
                                        'market_cap': opportunity.get('market_cap')
                                    }
                                    break
                    else:
                        self.actual_test_symbols = list(found_cryptos.keys())
                    
                    scout_results['target_cryptos_present'] = found_cryptos
                    
                    if len(found_cryptos) >= 2:
                        logger.info(f"      ✅ Target cryptos found: {list(found_cryptos.keys())}")
                    else:
                        logger.warning(f"      ⚠️ Limited target cryptos found: {list(found_cryptos.keys())}")
                    
                    # Test timestamp persistence by calling API twice
                    logger.info("   🕐 Testing timestamp persistence...")
                    await asyncio.sleep(5)  # Wait 5 seconds
                    
                    response2 = requests.get(f"{self.api_url}/opportunities", timeout=60)
                    if response2.status_code == 200:
                        opportunities2 = response2.json()
                        
                        # Compare timestamps for same symbols
                        timestamp_matches = 0
                        timestamp_total = 0
                        
                        for opp1 in opportunities:
                            symbol1 = opp1.get('symbol')
                            timestamp1 = opp1.get('timestamp')
                            
                            for opp2 in opportunities2:
                                symbol2 = opp2.get('symbol')
                                timestamp2 = opp2.get('timestamp')
                                
                                if symbol1 == symbol2:
                                    timestamp_total += 1
                                    if timestamp1 == timestamp2:
                                        timestamp_matches += 1
                                    break
                        
                        if timestamp_total > 0:
                            persistence_rate = timestamp_matches / timestamp_total
                            scout_results['timestamp_persistence_working'] = persistence_rate >= 0.8
                            logger.info(f"      📊 Timestamp persistence: {timestamp_matches}/{timestamp_total} ({persistence_rate:.2f})")
                        
                    # Analyze data quality
                    quality_metrics = {
                        'with_price': sum(1 for opp in opportunities if opp.get('current_price')),
                        'with_volume': sum(1 for opp in opportunities if opp.get('volume_24h')),
                        'with_change': sum(1 for opp in opportunities if opp.get('price_change_24h') is not None),
                        'with_volatility': sum(1 for opp in opportunities if opp.get('volatility')),
                        'with_timestamp': sum(1 for opp in opportunities if opp.get('timestamp'))
                    }
                    
                    scout_results['scout_data_quality'] = quality_metrics
                    
                    # Check if technical data is available
                    technical_fields = ['rsi_signal', 'macd_trend', 'mfi_signal', 'vwap_signal']
                    technical_count = 0
                    for opp in opportunities[:10]:  # Check first 10
                        for field in technical_fields:
                            if field in opp and opp[field] not in [None, '', 'unknown']:
                                technical_count += 1
                                break
                    
                    scout_results['technical_data_available'] = technical_count > 0
                    
                    logger.info(f"      📊 Data quality: Price={quality_metrics['with_price']}, Volume={quality_metrics['with_volume']}, Technical={technical_count}")
                
                else:
                    logger.error(f"      ❌ Opportunities API failed: HTTP {response.status_code}")
                    if response.text:
                        logger.error(f"         Error response: {response.text[:200]}...")
            
            except Exception as e:
                logger.error(f"      ❌ Opportunities API exception: {e}")
            
            # Final analysis
            logger.info(f"\n   📊 SCOUT SYSTEM LIMITED TEST RESULTS:")
            logger.info(f"      Opportunities API accessible: {scout_results['opportunities_api_accessible']}")
            logger.info(f"      Total opportunities found: {scout_results['opportunities_found']}")
            logger.info(f"      Target cryptos present: {len(scout_results['target_cryptos_present'])}/3")
            logger.info(f"      Timestamp persistence working: {scout_results['timestamp_persistence_working']}")
            logger.info(f"      Technical data available: {scout_results['technical_data_available']}")
            
            # Calculate test success
            success_criteria = [
                scout_results['opportunities_api_accessible'],
                scout_results['opportunities_found'] > 0,
                len(scout_results['target_cryptos_present']) >= 2,  # At least 2 target cryptos
                scout_results['timestamp_persistence_working']
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.75:  # 75% success threshold
                self.log_test_result("Scout System Limited Test", True, 
                                   f"Scout system working: {success_count}/{len(success_criteria)} criteria met. Target cryptos: {len(scout_results['target_cryptos_present'])}, Opportunities: {scout_results['opportunities_found']}")
            else:
                self.log_test_result("Scout System Limited Test", False, 
                                   f"Scout system issues: {success_count}/{len(success_criteria)} criteria met. May have API or timestamp persistence problems")
                
        except Exception as e:
            self.log_test_result("Scout System Limited Test", False, f"Exception: {str(e)}")

    async def test_1_json_parsing_recovery_test(self):
        """Test 1: JSON Parsing Recovery Test - Verify simplified approach fixes fundamental parsing issues"""
        logger.info("\n🔍 TEST 1: JSON Parsing Recovery Test")
        
        try:
            parsing_results = {
                'successful_analyses': 0,
                'total_analyses_attempted': len(self.test_symbols),
                'json_parsing_success': 0,
                'json_parsing_failures': 0,
                'fallback_analyses': 0,
                'valid_json_responses': [],
                'parsing_error_logs': [],
                'datetime_errors': 0,
                'system_stability_issues': 0,
                'backend_logs_analysis': {}
            }
            
            logger.info("   🚀 Testing SIMPLIFIED IA1 Prompt approach for JSON parsing recovery...")
            
            # Use actual available symbols from scout test
            symbols_to_test = self.actual_test_symbols if hasattr(self, 'actual_test_symbols') and self.actual_test_symbols else self.test_symbols
            logger.info(f"   📊 Testing symbols: {symbols_to_test}")
            logger.info("   📊 Expected: IA1 returns valid JSON without 'JSON parsing failed' errors, no fallback to pattern-only analysis")
            
            # Update parsing results with actual symbol count
            parsing_results['total_analyses_attempted'] = len(symbols_to_test)
            
            # Test each symbol
            for symbol in symbols_to_test:
                logger.info(f"\n   📞 Testing IA1 simplified JSON analysis for {symbol}")
                
                try:
                    start_time = time.time()
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120  # Longer timeout for IA1 analysis
                    )
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        parsing_results['successful_analyses'] += 1
                        
                        logger.info(f"      ✅ IA1 analysis for {symbol} successful (response time: {response_time:.2f}s)")
                        
                        # Check for JSON parsing success indicators
                        is_valid_json = True
                        is_fallback_analysis = False
                        has_simplified_fields = False
                        
                        # Look for fallback analysis indicators
                        if 'reasoning' in analysis_data:
                            reasoning_text = str(analysis_data['reasoning']).lower()
                            if any(fallback_indicator in reasoning_text for fallback_indicator in [
                                'fallback analysis', 'json parsing failed', 'pattern-only analysis',
                                'technical analysis fallback', 'using detected pattern'
                            ]):
                                is_fallback_analysis = True
                                parsing_results['fallback_analyses'] += 1
                                logger.warning(f"      ❌ Fallback analysis detected for {symbol}")
                        
                        # Check for simplified JSON schema fields in IA1 response
                        ia1_response = analysis_data.get('ia1_analysis', {})
                        if isinstance(ia1_response, dict):
                            simplified_fields_found = []
                            for field in self.simplified_json_fields:
                                if field in ia1_response:
                                    simplified_fields_found.append(field)
                            
                            if len(simplified_fields_found) >= 4:  # At least 4 simplified fields
                                has_simplified_fields = True
                                logger.info(f"      ✅ Simplified JSON fields found: {simplified_fields_found}")
                            else:
                                logger.warning(f"      ⚠️ Limited simplified fields: {simplified_fields_found}")
                        
                        # Determine JSON parsing success
                        if not is_fallback_analysis and has_simplified_fields:
                            parsing_results['json_parsing_success'] += 1
                            parsing_results['valid_json_responses'].append({
                                'symbol': symbol,
                                'response_time': response_time,
                                'simplified_fields': simplified_fields_found,
                                'analysis_data': analysis_data
                            })
                            logger.info(f"      ✅ JSON parsing SUCCESS for {symbol}")
                        else:
                            parsing_results['json_parsing_failures'] += 1
                            logger.warning(f"      ❌ JSON parsing FAILURE for {symbol} (fallback: {is_fallback_analysis}, simplified: {has_simplified_fields})")
                        
                        # Store full analysis for later use
                        self.ia1_analyses.append({
                            'symbol': symbol,
                            'analysis_data': analysis_data,
                            'is_valid_json': is_valid_json and not is_fallback_analysis,
                            'has_simplified_fields': has_simplified_fields
                        })
                    
                    else:
                        logger.error(f"      ❌ IA1 analysis for {symbol} failed: HTTP {response.status_code}")
                        if response.text:
                            logger.error(f"         Error response: {response.text[:200]}...")
                
                except Exception as e:
                    logger.error(f"      ❌ IA1 analysis for {symbol} exception: {e}")
                
                # Wait between analyses to avoid overwhelming the system
                if symbol != self.test_symbols[-1]:
                    logger.info(f"      ⏳ Waiting 10 seconds before next analysis...")
                    await asyncio.sleep(10)
            
            # Capture backend logs for JSON parsing analysis
            logger.info("   📋 Capturing backend logs for JSON parsing analysis...")
            try:
                backend_logs = await self._capture_backend_logs()
                if backend_logs:
                    self.backend_logs = backend_logs  # Store for later use
                    
                    # Look for JSON parsing errors
                    json_error_patterns = [
                        'json parsing failed', 'fallback analysis', 'json decode error',
                        'invalid json', 'parsing error', 'malformed json'
                    ]
                    
                    json_errors_found = 0
                    for log_line in backend_logs:
                        for pattern in json_error_patterns:
                            if pattern.lower() in log_line.lower():
                                json_errors_found += 1
                                parsing_results['parsing_error_logs'].append(log_line.strip())
                                break
                    
                    # Look for datetime errors
                    datetime_error_patterns = [
                        'datetime', 'timezone', 'offset-naive', 'offset-aware',
                        'can\'t subtract', 'timezone error'
                    ]
                    
                    datetime_errors_found = 0
                    for log_line in backend_logs:
                        for pattern in datetime_error_patterns:
                            if pattern.lower() in log_line.lower() and 'error' in log_line.lower():
                                datetime_errors_found += 1
                                break
                    
                    parsing_results['datetime_errors'] = datetime_errors_found
                    parsing_results['backend_logs_analysis'] = {
                        'total_logs': len(backend_logs),
                        'json_errors': json_errors_found,
                        'datetime_errors': datetime_errors_found
                    }
                    
                    logger.info(f"      📊 Backend logs analysis: {len(backend_logs)} lines, {json_errors_found} JSON errors, {datetime_errors_found} datetime errors")
                    
            except Exception as e:
                logger.warning(f"      ⚠️ Could not analyze backend logs: {e}")
            
            # Final analysis
            logger.info(f"\n   📊 JSON PARSING RECOVERY TEST RESULTS:")
            logger.info(f"      Successful analyses: {parsing_results['successful_analyses']}/{parsing_results['total_analyses_attempted']}")
            logger.info(f"      JSON parsing successes: {parsing_results['json_parsing_success']}")
            logger.info(f"      JSON parsing failures: {parsing_results['json_parsing_failures']}")
            logger.info(f"      Fallback analyses: {parsing_results['fallback_analyses']}")
            logger.info(f"      Datetime errors: {parsing_results['datetime_errors']}")
            
            # Calculate JSON parsing success rate
            json_success_rate = parsing_results['json_parsing_success'] / max(parsing_results['successful_analyses'], 1)
            fallback_rate = parsing_results['fallback_analyses'] / max(parsing_results['successful_analyses'], 1)
            
            logger.info(f"      📊 Performance metrics:")
            logger.info(f"         JSON parsing success rate: {json_success_rate:.2f} ({parsing_results['json_parsing_success']}/{parsing_results['successful_analyses']})")
            logger.info(f"         Fallback analysis rate: {fallback_rate:.2f}")
            logger.info(f"         System stability: {'Good' if parsing_results['datetime_errors'] == 0 else 'Issues detected'}")
            
            # Show valid JSON responses
            if parsing_results['valid_json_responses']:
                logger.info(f"      ✅ Valid JSON responses:")
                for response in parsing_results['valid_json_responses']:
                    logger.info(f"         - {response['symbol']}: {len(response['simplified_fields'])} simplified fields")
            
            # Calculate test success based on simplified approach requirements
            success_criteria = [
                parsing_results['successful_analyses'] >= 1,  # At least 1 successful analysis
                parsing_results['json_parsing_success'] >= 1,  # At least 1 JSON parsing success
                parsing_results['fallback_analyses'] == 0,  # No fallback analyses
                parsing_results['datetime_errors'] == 0,  # No datetime errors
                json_success_rate >= 0.5  # 50% JSON success rate (target from review)
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold for simplified approach
                self.log_test_result("JSON Parsing Recovery Test", True, 
                                   f"Simplified approach successful: {success_count}/{len(success_criteria)} criteria met. JSON success rate: {json_success_rate:.2f}, No fallbacks: {parsing_results['fallback_analyses'] == 0}, No datetime errors: {parsing_results['datetime_errors'] == 0}")
            else:
                self.log_test_result("JSON Parsing Recovery Test", False, 
                                   f"Simplified approach issues: {success_count}/{len(success_criteria)} criteria met. JSON parsing may still be failing or system has stability issues")
                
        except Exception as e:
            self.log_test_result("JSON Parsing Recovery Test", False, f"Exception: {str(e)}")
    
    async def test_2_technical_indicators_integration_final_check(self):
        """Test 2: Technical Indicators Integration Final Check - Test if simplified format promotes indicators usage"""
        logger.info("\n🔍 TEST 2: Technical Indicators Integration Final Check")
        
        try:
            integration_results = {
                'reasoning_analysis': {},
                'technical_indicators_mentioned': {},
                'specific_values_found': {},
                'technical_indicators_object_present': False,
                'patterns_vs_indicators_ratio': 0.0,
                'simplified_format_compliance': False,
                'analyses_with_indicators': 0,
                'total_valid_analyses': 0,
                'backend_calculations_verified': False
            }
            
            logger.info("   🚀 Testing technical indicators integration in simplified JSON format...")
            logger.info("   📊 Expected: IA1 reasoning mentions specific calculated values: RSI 32.1, MACD -0.015, MFI 20.2, VWAP -3.9%")
            
            # Step 1: Analyze valid JSON responses from previous test
            logger.info("   📊 Analyzing valid JSON responses for technical indicators integration...")
            
            valid_analyses = []
            if hasattr(self, 'ia1_analyses') and self.ia1_analyses:
                for analysis in self.ia1_analyses:
                    if analysis.get('is_valid_json', False):
                        valid_analyses.append(analysis)
                        integration_results['total_valid_analyses'] += 1
                
                logger.info(f"      ✅ Found {len(valid_analyses)} valid JSON analyses to examine")
            else:
                logger.warning(f"      ⚠️ No valid JSON analyses available from previous test")
                return
            
            # Step 2: Analyze reasoning field for specific technical indicator values
            logger.info("   📋 Analyzing reasoning field for specific technical indicator values...")
            
            for analysis in valid_analyses:
                symbol = analysis['symbol']
                analysis_data = analysis['analysis_data']
                ia1_response = analysis_data.get('ia1_analysis', {})
                
                logger.info(f"      📊 Analyzing {symbol} reasoning for technical indicators...")
                
                # Check reasoning field for specific values
                reasoning = ia1_response.get('reasoning', '')
                if reasoning:
                    logger.info(f"         ✅ Reasoning field found: {len(reasoning)} chars")
                    
                    # Look for specific technical indicator values (RSI 32.1, MACD -0.015, etc.)
                    specific_values_found = {}
                    for pattern in self.specific_value_patterns:
                        matches = re.findall(pattern, reasoning, re.IGNORECASE)
                        if matches:
                            indicator = pattern.split('\\')[0]  # Extract indicator name
                            specific_values_found[indicator] = matches
                            logger.info(f"            ✅ {indicator} specific values found: {matches}")
                    
                    integration_results['specific_values_found'][symbol] = specific_values_found
                    
                    # Count technical indicators vs patterns mentions
                    technical_count = 0
                    patterns_count = 0
                    
                    for indicator in self.core_technical_indicators:
                        count = reasoning.lower().count(indicator.lower())
                        if count > 0:
                            technical_count += count
                            if indicator not in integration_results['technical_indicators_mentioned']:
                                integration_results['technical_indicators_mentioned'][indicator] = 0
                            integration_results['technical_indicators_mentioned'][indicator] += count
                    
                    for pattern in self.chartist_patterns:
                        count = reasoning.lower().count(pattern.lower())
                        patterns_count += count
                    
                    # Calculate ratio
                    total_mentions = technical_count + patterns_count
                    if total_mentions > 0:
                        tech_ratio = technical_count / total_mentions
                        integration_results['reasoning_analysis'][symbol] = {
                            'technical_count': technical_count,
                            'patterns_count': patterns_count,
                            'technical_ratio': tech_ratio,
                            'specific_values': len(specific_values_found),
                            'reasoning_sample': reasoning[:300] + "..." if len(reasoning) > 300 else reasoning
                        }
                        
                        logger.info(f"            📊 Technical: {technical_count}, Patterns: {patterns_count}, Ratio: {tech_ratio:.2f}")
                
                # Check technical_indicators object
                technical_indicators_obj = ia1_response.get('technical_indicators', {})
                if isinstance(technical_indicators_obj, dict) and technical_indicators_obj:
                    integration_results['technical_indicators_object_present'] = True
                    logger.info(f"         ✅ technical_indicators object found with {len(technical_indicators_obj)} fields")
                    
                    # Check for specific fields
                    found_fields = []
                    for field in self.technical_indicators_fields:
                        if field in technical_indicators_obj:
                            found_fields.append(field)
                    
                    logger.info(f"            📊 Technical indicator fields: {found_fields}")
                
                # Check if this analysis has technical indicators
                if (symbol in integration_results['specific_values_found'] and 
                    integration_results['specific_values_found'][symbol]) or \
                   any(indicator in integration_results['technical_indicators_mentioned'] 
                       for indicator in self.core_technical_indicators):
                    integration_results['analyses_with_indicators'] += 1
            
            # Step 3: Verify backend calculations
            logger.info("   🔍 Verifying backend technical indicator calculations...")
            
            try:
                if hasattr(self, 'backend_logs') and self.backend_logs:
                    backend_logs = self.backend_logs
                else:
                    backend_logs = await self._capture_backend_logs()
                
                if backend_logs:
                    # Look for technical indicator calculations
                    calculation_patterns = [
                        r'RSI[:\s]+[\d\.]+',
                        r'MACD[:\s]+[\d\.\-]+',
                        r'MFI[:\s]+[\d\.]+',
                        r'VWAP[:\s]+[\d\.\-]+%?'
                    ]
                    
                    calculations_found = {}
                    for log_line in backend_logs:
                        for pattern in calculation_patterns:
                            matches = re.findall(pattern, log_line, re.IGNORECASE)
                            if matches:
                                indicator = pattern.split('[')[0]
                                if indicator not in calculations_found:
                                    calculations_found[indicator] = []
                                calculations_found[indicator].extend(matches)
                    
                    if calculations_found:
                        integration_results['backend_calculations_verified'] = True
                        logger.info(f"      ✅ Backend calculations found:")
                        for indicator, values in calculations_found.items():
                            logger.info(f"         - {indicator}: {values[:3]}...")  # Show first 3
                    else:
                        logger.warning(f"      ❌ No backend technical indicator calculations found")
                        
            except Exception as e:
                logger.warning(f"      ⚠️ Backend calculations check error: {e}")
            
            # Step 4: Calculate overall technical indicators usage
            logger.info("   📊 Calculating overall technical indicators usage...")
            
            total_technical_mentions = sum(integration_results['technical_indicators_mentioned'].values())
            total_specific_values = sum(len(values) for values in integration_results['specific_values_found'].values())
            
            # Calculate improvement over previous 0% usage
            indicators_usage_rate = integration_results['analyses_with_indicators'] / max(integration_results['total_valid_analyses'], 1)
            
            logger.info(f"      📊 Technical indicators usage rate: {indicators_usage_rate:.2f} ({integration_results['analyses_with_indicators']}/{integration_results['total_valid_analyses']})")
            logger.info(f"      📊 Total technical mentions: {total_technical_mentions}")
            logger.info(f"      📊 Specific values found: {total_specific_values}")
            
            # Final analysis
            logger.info(f"\n   📊 TECHNICAL INDICATORS INTEGRATION FINAL CHECK RESULTS:")
            logger.info(f"      Total valid analyses: {integration_results['total_valid_analyses']}")
            logger.info(f"      Analyses with indicators: {integration_results['analyses_with_indicators']}")
            logger.info(f"      Technical indicators object present: {integration_results['technical_indicators_object_present']}")
            logger.info(f"      Backend calculations verified: {integration_results['backend_calculations_verified']}")
            logger.info(f"      Indicators usage rate: {indicators_usage_rate:.2f}")
            
            if integration_results['technical_indicators_mentioned']:
                logger.info(f"      📊 Technical indicators mentioned:")
                for indicator, count in integration_results['technical_indicators_mentioned'].items():
                    logger.info(f"         - {indicator}: {count} mentions")
            
            if integration_results['specific_values_found']:
                logger.info(f"      📊 Specific values found in reasoning:")
                for symbol, values in integration_results['specific_values_found'].items():
                    if values:
                        logger.info(f"         - {symbol}: {len(values)} indicators with specific values")
            
            # Calculate test success based on review requirements (>50% technical indicators usage)
            success_criteria = [
                integration_results['total_valid_analyses'] > 0,  # Have valid analyses
                integration_results['analyses_with_indicators'] > 0,  # Some analyses have indicators
                indicators_usage_rate >= 0.5,  # >50% usage rate (target from review)
                total_specific_values > 0,  # Some specific values found
                integration_results['backend_calculations_verified']  # Backend calculations working
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Technical Indicators Integration Final Check", True, 
                                   f"Technical indicators integration successful: {success_count}/{len(success_criteria)} criteria met. Usage rate: {indicators_usage_rate:.2f}, Specific values: {total_specific_values}, Backend verified: {integration_results['backend_calculations_verified']}")
            else:
                self.log_test_result("Technical Indicators Integration Final Check", False, 
                                   f"Technical indicators integration issues: {success_count}/{len(success_criteria)} criteria met. Usage rate may still be below 50% target or backend calculations not working")
                
        except Exception as e:
            self.log_test_result("Technical Indicators Integration Final Check", False, f"Exception: {str(e)}")
    
    async def test_3_prompt_effectiveness_check(self):
        """Test 3: Prompt Effectiveness Check - Analyze if prompt instructions are clear"""
        logger.info("\n🔍 TEST 3: Prompt Effectiveness Check")
        
        try:
            prompt_results = {
                'server_code_analyzed': False,
                'prompt_instructions_found': False,
                'technical_indicators_in_prompt': {},
                'confluence_matrix_mentioned': False,
                'six_indicator_system_found': False,
                'prompt_analysis_successful': False,
                'prompt_samples': [],
                'instruction_clarity_score': 0
            }
            
            logger.info("   🚀 Analyzing IA1 prompt instructions for technical indicators emphasis...")
            logger.info("   📊 Expected: Prompt explicitly asks IA1 to use RSI, MACD, MFI, VWAP, 6-indicator confluence")
            
            # Step 1: Analyze server.py code for IA1 prompt construction
            logger.info("   📋 Analyzing server.py code for IA1 prompt instructions...")
            
            try:
                with open('/app/backend/server.py', 'r') as f:
                    server_code = f.read()
                
                prompt_results['server_code_analyzed'] = True
                logger.info(f"      ✅ Server code loaded: {len(server_code)} characters")
                
                # Look for IA1 chat initialization and system message
                ia1_patterns = [
                    'get_ia1_chat', 'system_message', 'technical analyst',
                    'RSI', 'MACD', 'MFI', 'VWAP', 'Bollinger', 'Stochastic',
                    'confluence', '6-indicator', 'technical indicators'
                ]
                
                found_patterns = {}
                for pattern in ia1_patterns:
                    count = server_code.count(pattern)
                    if count > 0:
                        found_patterns[pattern] = count
                        if pattern in ['RSI', 'MACD', 'MFI', 'VWAP', 'Bollinger', 'Stochastic']:
                            prompt_results['technical_indicators_in_prompt'][pattern] = count
                
                if found_patterns:
                    prompt_results['prompt_instructions_found'] = True
                    logger.info(f"      ✅ IA1 prompt patterns found:")
                    for pattern, count in found_patterns.items():
                        logger.info(f"         - {pattern}: {count} occurrences")
                else:
                    logger.warning(f"      ❌ No IA1 prompt patterns found")
                
                # Look for confluence matrix instructions
                confluence_patterns = [
                    'confluence matrix', '6-indicator', 'CONFLUENCE', 'voting system',
                    'indicator confluence', 'technical confluence'
                ]
                
                confluence_found = False
                for pattern in confluence_patterns:
                    if pattern.lower() in server_code.lower():
                        confluence_found = True
                        logger.info(f"      ✅ Confluence pattern found: {pattern}")
                        break
                
                prompt_results['confluence_matrix_mentioned'] = confluence_found
                
                # Look for 6-indicator system
                six_indicator_patterns = [
                    '6-indicator', 'six indicator', 'MFI', 'VWAP', 'RSI', 'Multi-Timeframe', 'Volume', 'EMA'
                ]
                
                six_indicator_count = 0
                for pattern in six_indicator_patterns:
                    if pattern.lower() in server_code.lower():
                        six_indicator_count += 1
                
                if six_indicator_count >= 4:  # At least 4/8 patterns found
                    prompt_results['six_indicator_system_found'] = True
                    logger.info(f"      ✅ 6-indicator system found: {six_indicator_count}/{len(six_indicator_patterns)} patterns")
                else:
                    logger.warning(f"      ❌ 6-indicator system incomplete: {six_indicator_count}/{len(six_indicator_patterns)} patterns")
                
                # Extract sample prompt instructions
                try:
                    # Look for system message content
                    system_message_pattern = r'system_message="""(.*?)"""'
                    matches = re.findall(system_message_pattern, server_code, re.DOTALL)
                    
                    if matches:
                        for i, match in enumerate(matches[:2]):  # First 2 matches
                            sample = match[:500] + "..." if len(match) > 500 else match
                            prompt_results['prompt_samples'].append(f"Sample {i+1}: {sample}")
                            logger.info(f"      📝 Prompt sample {i+1}: {sample[:150]}...")
                    
                except Exception as e:
                    logger.debug(f"      Could not extract prompt samples: {e}")
                
                prompt_results['prompt_analysis_successful'] = True
                
            except Exception as e:
                logger.error(f"      ❌ Server code analysis error: {e}")
            
            # Step 2: Check backend logs for prompt construction
            logger.info("   📋 Checking backend logs for prompt construction evidence...")
            
            try:
                if hasattr(self, 'backend_logs') and self.backend_logs:
                    backend_logs = self.backend_logs
                else:
                    backend_logs = await self._capture_backend_logs()
                
                if backend_logs:
                    # Look for prompt-related logs
                    prompt_logs = []
                    prompt_patterns = [
                        'prompt', 'system message', 'IA1 analysis', 'technical indicators',
                        'confluence', 'sending to IA1'
                    ]
                    
                    for log_line in backend_logs:
                        for pattern in prompt_patterns:
                            if pattern.lower() in log_line.lower():
                                prompt_logs.append(log_line.strip())
                                break
                    
                    if prompt_logs:
                        logger.info(f"      ✅ Prompt construction logs found: {len(prompt_logs)} entries")
                        for log in prompt_logs[:2]:  # Show first 2
                            logger.info(f"         - {log[:100]}...")
                    else:
                        logger.info(f"      📋 No specific prompt construction logs found")
                        
                else:
                    logger.warning(f"      ⚠️ Could not access backend logs")
                    
            except Exception as e:
                logger.warning(f"      ⚠️ Log analysis error: {e}")
            
            # Step 3: Calculate instruction clarity score
            logger.info("   📊 Calculating prompt instruction clarity score...")
            
            clarity_factors = [
                prompt_results['prompt_instructions_found'],  # Basic prompt found
                len(prompt_results['technical_indicators_in_prompt']) >= 4,  # At least 4 indicators mentioned
                prompt_results['confluence_matrix_mentioned'],  # Confluence matrix mentioned
                prompt_results['six_indicator_system_found'],  # 6-indicator system found
                len(prompt_results['prompt_samples']) > 0  # Prompt samples extracted
            ]
            
            clarity_score = sum(clarity_factors) / len(clarity_factors)
            prompt_results['instruction_clarity_score'] = clarity_score
            
            logger.info(f"      📊 Instruction clarity score: {clarity_score:.2f} ({sum(clarity_factors)}/{len(clarity_factors)} factors)")
            
            # Final analysis
            logger.info(f"\n   📊 PROMPT EFFECTIVENESS ANALYSIS:")
            logger.info(f"      Server code analyzed: {prompt_results['server_code_analyzed']}")
            logger.info(f"      Prompt instructions found: {prompt_results['prompt_instructions_found']}")
            logger.info(f"      Technical indicators in prompt: {len(prompt_results['technical_indicators_in_prompt'])}")
            logger.info(f"      Confluence matrix mentioned: {prompt_results['confluence_matrix_mentioned']}")
            logger.info(f"      6-indicator system found: {prompt_results['six_indicator_system_found']}")
            logger.info(f"      Instruction clarity score: {prompt_results['instruction_clarity_score']:.2f}")
            
            if prompt_results['technical_indicators_in_prompt']:
                logger.info(f"      📊 Technical indicators in prompt:")
                for indicator, count in prompt_results['technical_indicators_in_prompt'].items():
                    logger.info(f"         - {indicator}: {count} mentions")
            
            # Calculate test success
            success_criteria = [
                prompt_results['server_code_analyzed'],
                prompt_results['prompt_instructions_found'],
                len(prompt_results['technical_indicators_in_prompt']) >= 3,  # At least 3 indicators
                prompt_results['confluence_matrix_mentioned'] or prompt_results['six_indicator_system_found']
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.75:  # 75% success threshold
                self.log_test_result("Prompt Effectiveness Check", True, 
                                   f"Prompt effectiveness good: {success_count}/{len(success_criteria)} criteria met. Clarity score: {prompt_results['instruction_clarity_score']:.2f}, Indicators: {len(prompt_results['technical_indicators_in_prompt'])}")
            else:
                self.log_test_result("Prompt Effectiveness Check", False, 
                                   f"Prompt effectiveness issues: {success_count}/{len(success_criteria)} criteria met. May need clearer technical indicator instructions")
                
        except Exception as e:
            self.log_test_result("Prompt Effectiveness Check", False, f"Exception: {str(e)}")
    
    async def test_4_database_storage_verification(self):
        """Test 4: Database Storage Verification - Check stored IA1 analyses"""
        logger.info("\n🔍 TEST 4: Database Storage Verification")
        
        try:
            storage_results = {
                'analyses_api_accessible': False,
                'recent_analyses_found': False,
                'technical_indicators_in_db': {},
                'reasoning_fields_present': False,
                'database_integration_working': False,
                'analyses_data': [],
                'technical_references_preserved': False,
                'sample_analyses': []
            }
            
            logger.info("   🚀 Testing database storage of IA1 analyses with technical indicators...")
            logger.info("   📊 Expected: Database stores IA1 analyses with technical indicator references")
            
            # Step 1: Query /api/analyses to get recent IA1 analyses
            logger.info("   📞 Querying /api/analyses for recent IA1 analyses...")
            
            try:
                response = requests.get(f"{self.api_url}/analyses", timeout=60)
                
                if response.status_code == 200:
                    analyses = response.json()
                    storage_results['analyses_api_accessible'] = True
                    
                    logger.info(f"      ✅ Analyses API accessible: {len(analyses)} analyses retrieved")
                    
                    if len(analyses) > 0:
                        storage_results['recent_analyses_found'] = True
                        storage_results['analyses_data'] = analyses
                        
                        # Analyze each analysis for technical indicators
                        technical_indicators_found = {}
                        reasoning_count = 0
                        
                        for i, analysis in enumerate(analyses[:10]):  # Check first 10 analyses
                            # Check if reasoning field is present
                            reasoning = ""
                            if 'reasoning' in analysis:
                                reasoning = analysis['reasoning']
                                reasoning_count += 1
                            elif 'ia1_reasoning' in analysis:
                                reasoning = analysis['ia1_reasoning']
                                reasoning_count += 1
                            
                            if reasoning:
                                # Look for technical indicators in stored reasoning
                                for indicator in self.core_technical_indicators:
                                    if indicator.lower() in reasoning.lower():
                                        if indicator not in technical_indicators_found:
                                            technical_indicators_found[indicator] = 0
                                        technical_indicators_found[indicator] += 1
                                
                                # Store sample analysis
                                if len(storage_results['sample_analyses']) < 3:
                                    storage_results['sample_analyses'].append({
                                        'symbol': analysis.get('symbol', 'Unknown'),
                                        'reasoning_length': len(reasoning),
                                        'reasoning_sample': reasoning[:200] + "..." if len(reasoning) > 200 else reasoning,
                                        'timestamp': analysis.get('timestamp', 'Unknown')
                                    })
                        
                        storage_results['technical_indicators_in_db'] = technical_indicators_found
                        
                        if reasoning_count > 0:
                            storage_results['reasoning_fields_present'] = True
                            logger.info(f"      ✅ Reasoning fields found in {reasoning_count}/{len(analyses[:10])} analyses")
                        else:
                            logger.warning(f"      ❌ No reasoning fields found in stored analyses")
                        
                        if technical_indicators_found:
                            storage_results['technical_references_preserved'] = True
                            logger.info(f"      ✅ Technical indicators found in database:")
                            for indicator, count in sorted(technical_indicators_found.items(), key=lambda x: x[1], reverse=True):
                                logger.info(f"         - {indicator}: {count} references")
                        else:
                            logger.warning(f"      ❌ No technical indicators found in stored analyses")
                        
                        # Show sample analyses
                        logger.info(f"      📝 Sample stored analyses:")
                        for sample in storage_results['sample_analyses']:
                            logger.info(f"         - {sample['symbol']}: {sample['reasoning_sample'][:100]}...")
                    
                    else:
                        logger.warning(f"      ⚠️ No analyses found in database")
                
                else:
                    logger.error(f"      ❌ Analyses API failed: HTTP {response.status_code}")
                    if response.text:
                        logger.error(f"         Error response: {response.text[:200]}...")
            
            except Exception as e:
                logger.error(f"      ❌ Analyses API exception: {e}")
            
            # Step 2: Check database integration with recent IA1 analyses
            logger.info("   📊 Checking database integration with recent IA1 analyses...")
            
            try:
                # Compare with our IA1 analyses from test 1
                if hasattr(self, 'ia1_analyses') and self.ia1_analyses:
                    logger.info(f"      📋 Comparing with {len(self.ia1_analyses)} IA1 analyses from earlier tests...")
                    
                    # Check if recent analyses match our test analyses
                    test_symbols = [analysis['symbol'] for analysis in self.ia1_analyses]
                    db_symbols = [analysis.get('symbol') for analysis in storage_results['analyses_data'][:20]]
                    
                    matching_symbols = set(test_symbols) & set(db_symbols)
                    
                    if matching_symbols:
                        storage_results['database_integration_working'] = True
                        logger.info(f"      ✅ Database integration working: {len(matching_symbols)} matching symbols found")
                        logger.info(f"         Matching symbols: {list(matching_symbols)}")
                    else:
                        logger.warning(f"      ⚠️ No matching symbols between test analyses and database")
                        logger.info(f"         Test symbols: {test_symbols}")
                        logger.info(f"         DB symbols: {db_symbols[:5]}...")
                
                else:
                    logger.info(f"      📋 No previous IA1 analyses to compare with")
                    
            except Exception as e:
                logger.warning(f"      ⚠️ Database integration check error: {e}")
            
            # Step 3: Verify technical indicator data persistence
            logger.info("   🔍 Verifying technical indicator data persistence...")
            
            try:
                # Look for specific technical indicator fields in analyses
                indicator_fields = ['rsi_signal', 'macd_trend', 'mfi_signal', 'vwap_signal', 'bollinger_position']
                
                field_counts = {}
                for analysis in storage_results['analyses_data'][:10]:
                    for field in indicator_fields:
                        if field in analysis and analysis[field] not in [None, '', 'unknown']:
                            if field not in field_counts:
                                field_counts[field] = 0
                            field_counts[field] += 1
                
                if field_counts:
                    logger.info(f"      ✅ Technical indicator fields found in database:")
                    for field, count in field_counts.items():
                        logger.info(f"         - {field}: {count} non-null values")
                else:
                    logger.warning(f"      ❌ No technical indicator fields with values found")
                
            except Exception as e:
                logger.warning(f"      ⚠️ Technical indicator field check error: {e}")
            
            # Final analysis
            logger.info(f"\n   📊 DATABASE STORAGE VERIFICATION ANALYSIS:")
            logger.info(f"      Analyses API accessible: {storage_results['analyses_api_accessible']}")
            logger.info(f"      Recent analyses found: {storage_results['recent_analyses_found']}")
            logger.info(f"      Reasoning fields present: {storage_results['reasoning_fields_present']}")
            logger.info(f"      Technical references preserved: {storage_results['technical_references_preserved']}")
            logger.info(f"      Database integration working: {storage_results['database_integration_working']}")
            logger.info(f"      Technical indicators in DB: {len(storage_results['technical_indicators_in_db'])}")
            
            # Calculate test success
            success_criteria = [
                storage_results['analyses_api_accessible'],
                storage_results['recent_analyses_found'],
                storage_results['reasoning_fields_present'],
                storage_results['technical_references_preserved'] or len(storage_results['technical_indicators_in_db']) >= 2
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.75:  # 75% success threshold
                self.log_test_result("Database Storage Verification", True, 
                                   f"Database storage working: {success_count}/{len(success_criteria)} criteria met. Technical indicators: {len(storage_results['technical_indicators_in_db'])}, Reasoning preserved: {storage_results['reasoning_fields_present']}")
            else:
                self.log_test_result("Database Storage Verification", False, 
                                   f"Database storage issues: {success_count}/{len(success_criteria)} criteria met. Technical references may not be preserved properly")
                
        except Exception as e:
            self.log_test_result("Database Storage Verification", False, f"Exception: {str(e)}")
    
    async def test_5_ia1_response_analysis_comparison(self):
        """Test 5: IA1 Response Analysis - Compare chartist vs technical indicator usage"""
        logger.info("\n🔍 TEST 5: IA1 Response Analysis Comparison")
        
        try:
            comparison_results = {
                'analyses_available': False,
                'technical_vs_chartist_analysis': {},
                'balance_assessment': {},
                'usage_patterns': {},
                'recommendation_analysis': {},
                'overall_balance_score': 0.0,
                'detailed_breakdown': []
            }
            
            logger.info("   🚀 Analyzing IA1 responses for technical indicators vs chartist patterns balance...")
            logger.info("   📊 Expected: IA1 balances both technical indicators and chartist patterns")
            
            # Use analyses from previous tests
            if hasattr(self, 'ia1_analyses') and self.ia1_analyses:
                comparison_results['analyses_available'] = True
                analyses_to_analyze = self.ia1_analyses
                logger.info(f"      ✅ Using {len(analyses_to_analyze)} IA1 analyses from previous tests")
            else:
                logger.warning(f"      ⚠️ No previous IA1 analyses available, will analyze stored analyses")
                analyses_to_analyze = []
            
            # If no previous analyses, try to get some from database
            if not analyses_to_analyze:
                try:
                    response = requests.get(f"{self.api_url}/analyses", timeout=60)
                    if response.status_code == 200:
                        db_analyses = response.json()[:5]  # Get first 5
                        for analysis in db_analyses:
                            if 'reasoning' in analysis or 'ia1_reasoning' in analysis:
                                reasoning = analysis.get('reasoning') or analysis.get('ia1_reasoning', '')
                                analyses_to_analyze.append({
                                    'symbol': analysis.get('symbol', 'Unknown'),
                                    'reasoning': reasoning,
                                    'analysis_data': analysis
                                })
                        logger.info(f"      📋 Retrieved {len(analyses_to_analyze)} analyses from database")
                except Exception as e:
                    logger.warning(f"      ⚠️ Could not retrieve analyses from database: {e}")
            
            if analyses_to_analyze:
                comparison_results['analyses_available'] = True
                
                # Analyze each IA1 response
                total_technical_mentions = 0
                total_chartist_mentions = 0
                analysis_breakdown = []
                
                for analysis in analyses_to_analyze:
                    symbol = analysis['symbol']
                    reasoning = analysis['reasoning']
                    
                    if not reasoning:
                        continue
                    
                    # Count technical indicators
                    technical_count = 0
                    technical_found = []
                    for indicator in self.core_technical_indicators:
                        count = reasoning.lower().count(indicator.lower())
                        if count > 0:
                            technical_count += count
                            technical_found.append(f"{indicator}({count})")
                    
                    # Count chartist patterns
                    chartist_count = 0
                    chartist_found = []
                    for pattern in self.chartist_patterns:
                        count = reasoning.lower().count(pattern.lower())
                        if count > 0:
                            chartist_count += count
                            chartist_found.append(f"{pattern}({count})")
                    
                    # Calculate balance for this analysis
                    total_mentions = technical_count + chartist_count
                    technical_ratio = technical_count / total_mentions if total_mentions > 0 else 0
                    
                    analysis_result = {
                        'symbol': symbol,
                        'technical_count': technical_count,
                        'chartist_count': chartist_count,
                        'technical_ratio': technical_ratio,
                        'technical_found': technical_found,
                        'chartist_found': chartist_found,
                        'reasoning_length': len(reasoning),
                        'balance_category': self._categorize_balance(technical_ratio)
                    }
                    
                    analysis_breakdown.append(analysis_result)
                    total_technical_mentions += technical_count
                    total_chartist_mentions += chartist_count
                    
                    logger.info(f"      📊 {symbol}: Technical={technical_count}, Chartist={chartist_count}, Ratio={technical_ratio:.2f}")
                
                comparison_results['detailed_breakdown'] = analysis_breakdown
                
                # Overall analysis
                total_mentions = total_technical_mentions + total_chartist_mentions
                overall_technical_ratio = total_technical_mentions / total_mentions if total_mentions > 0 else 0
                
                comparison_results['technical_vs_chartist_analysis'] = {
                    'total_technical_mentions': total_technical_mentions,
                    'total_chartist_mentions': total_chartist_mentions,
                    'overall_technical_ratio': overall_technical_ratio,
                    'total_analyses': len(analysis_breakdown)
                }
                
                # Balance assessment
                balance_categories = {}
                for analysis in analysis_breakdown:
                    category = analysis['balance_category']
                    if category not in balance_categories:
                        balance_categories[category] = 0
                    balance_categories[category] += 1
                
                comparison_results['balance_assessment'] = balance_categories
                
                # Usage patterns
                technical_usage = sum(1 for a in analysis_breakdown if a['technical_count'] > 0)
                chartist_usage = sum(1 for a in analysis_breakdown if a['chartist_count'] > 0)
                both_usage = sum(1 for a in analysis_breakdown if a['technical_count'] > 0 and a['chartist_count'] > 0)
                
                comparison_results['usage_patterns'] = {
                    'analyses_with_technical': technical_usage,
                    'analyses_with_chartist': chartist_usage,
                    'analyses_with_both': both_usage,
                    'technical_only': technical_usage - both_usage,
                    'chartist_only': chartist_usage - both_usage
                }
                
                # Calculate overall balance score (0.5 = perfect balance)
                balance_score = 1.0 - abs(0.5 - overall_technical_ratio)
                comparison_results['overall_balance_score'] = balance_score
                
                # Detailed reporting
                logger.info(f"\n      📊 DETAILED ANALYSIS RESULTS:")
                logger.info(f"         Total technical mentions: {total_technical_mentions}")
                logger.info(f"         Total chartist mentions: {total_chartist_mentions}")
                logger.info(f"         Overall technical ratio: {overall_technical_ratio:.2f}")
                logger.info(f"         Balance score: {balance_score:.2f} (1.0 = perfect)")
                
                logger.info(f"      📊 Usage patterns:")
                logger.info(f"         Analyses with technical indicators: {technical_usage}/{len(analysis_breakdown)}")
                logger.info(f"         Analyses with chartist patterns: {chartist_usage}/{len(analysis_breakdown)}")
                logger.info(f"         Analyses with both: {both_usage}/{len(analysis_breakdown)}")
                
                logger.info(f"      📊 Balance categories:")
                for category, count in balance_categories.items():
                    logger.info(f"         {category}: {count} analyses")
                
            else:
                logger.warning(f"      ❌ No IA1 analyses available for comparison")
            
            # Final analysis
            logger.info(f"\n   📊 IA1 RESPONSE ANALYSIS COMPARISON:")
            logger.info(f"      Analyses available: {comparison_results['analyses_available']}")
            if comparison_results['analyses_available']:
                logger.info(f"      Overall balance score: {comparison_results['overall_balance_score']:.2f}")
                logger.info(f"      Technical ratio: {comparison_results['technical_vs_chartist_analysis'].get('overall_technical_ratio', 0):.2f}")
                logger.info(f"      Analyses with both types: {comparison_results['usage_patterns'].get('analyses_with_both', 0)}")
            
            # Calculate test success
            success_criteria = [
                comparison_results['analyses_available'],
                comparison_results['overall_balance_score'] >= 0.3,  # At least some balance
                comparison_results['usage_patterns'].get('analyses_with_technical', 0) > 0,  # Some technical usage
                comparison_results['usage_patterns'].get('analyses_with_both', 0) > 0  # Some analyses use both
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.75:  # 75% success threshold
                self.log_test_result("IA1 Response Analysis Comparison", True, 
                                   f"Response analysis successful: {success_count}/{len(success_criteria)} criteria met. Balance score: {comparison_results['overall_balance_score']:.2f}, Both types used: {comparison_results['usage_patterns'].get('analyses_with_both', 0)}")
            else:
                self.log_test_result("IA1 Response Analysis Comparison", False, 
                                   f"Response analysis issues: {success_count}/{len(success_criteria)} criteria met. May be focusing too much on one type of analysis")
                
        except Exception as e:
            self.log_test_result("IA1 Response Analysis Comparison", False, f"Exception: {str(e)}")
    
    def _categorize_balance(self, technical_ratio):
        """Categorize the balance between technical and chartist analysis"""
        if technical_ratio == 0:
            return "Chartist Only"
        elif technical_ratio == 1:
            return "Technical Only"
        elif 0.4 <= technical_ratio <= 0.6:
            return "Balanced"
        elif technical_ratio > 0.6:
            return "Technical Heavy"
        else:
            return "Chartist Heavy"
    
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
            
            return [log for log in all_logs if log.strip()]
        except Exception:
            return []
    
    async def _capture_backend_logs(self):
        """Capture recent backend logs for analysis"""
        try:
            # Try to capture supervisor backend logs
            result = subprocess.run(
                ["tail", "-n", "100", "/var/log/supervisor/backend.out.log"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                return result.stdout.split('\n')
            
            # Fallback: try error log
            result = subprocess.run(
                ["tail", "-n", "100", "/var/log/supervisor/backend.err.log"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                return result.stdout.split('\n')
            
            return []
            
        except Exception as e:
            logger.warning(f"Could not capture backend logs: {e}")
            return []
    
    async def run_all_tests(self):
        """Run all tests focusing on Dynamic RR Integration Phase 1 validation"""
        logger.info("🚀 STARTING DYNAMIC RR INTEGRATION TESTING SUITE - PHASE 1")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Run Dynamic RR Integration tests (Phase 1 focus)
        await self.test_1_field_name_validation()
        await self.test_2_dynamic_rr_escalation_logic()
        await self.test_3_database_persistence_validation()
        
        # Calculate overall results
        total_time = time.time() - start_time
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        logger.info("\n" + "=" * 80)
        logger.info("🎯 DYNAMIC RR INTEGRATION PHASE 1 VALIDATION SUMMARY")
        logger.info("=" * 80)
        
        # Check specific test results
        field_name_result = next((r for r in self.test_results if 'field name' in r['test'].lower()), None)
        escalation_result = next((r for r in self.test_results if 'escalation' in r['test'].lower()), None)
        persistence_result = next((r for r in self.test_results if 'persistence' in r['test'].lower()), None)
        
        if field_name_result:
            status = "✅ SUCCESSFUL" if field_name_result['success'] else "❌ FAILED"
            logger.info(f"{status} FIELD NAME VALIDATION: {field_name_result['details']}")
        
        if escalation_result:
            status = "✅ SUCCESSFUL" if escalation_result['success'] else "❌ FAILED"
            logger.info(f"{status} DYNAMIC RR ESCALATION: {escalation_result['details']}")
        
        if persistence_result:
            status = "✅ SUCCESSFUL" if persistence_result['success'] else "❌ FAILED"
            logger.info(f"{status} DATABASE PERSISTENCE: {persistence_result['details']}")
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   Details: {result['details']}")
        
        logger.info(f"\n📊 OVERALL RESULTS:")
        logger.info(f"   Tests passed: {passed_tests}/{total_tests}")
        logger.info(f"   Success rate: {passed_tests/total_tests:.2f}")
        logger.info(f"   Total time: {total_time:.2f} seconds")
        
        # Determine overall success based on Phase 1 requirements
        field_name_success = field_name_result['success'] if field_name_result else False
        escalation_success = escalation_result['success'] if escalation_result else False
        persistence_success = persistence_result['success'] if persistence_result else False
        
        phase1_success = field_name_success and escalation_success and persistence_success
        overall_success = phase1_success and (passed_tests >= (total_tests * 0.67))  # 67% pass rate + all core tests
        
        if phase1_success:
            logger.info("🎉 DYNAMIC RR INTEGRATION PHASE 1 SUCCESSFUL:")
            logger.info("   ✅ Field names updated: trade_type and minimum_rr_threshold")
            logger.info("   ✅ Dynamic RR escalation logic working")
            logger.info("   ✅ Database persistence with new field names")
            if overall_success:
                logger.info("✅ READY FOR PHASE 2: Advanced technical indicators integration")
            else:
                logger.info("⚠️ PARTIAL SUCCESS: Core functionality working but some issues need attention")
        else:
            logger.info("❌ DYNAMIC RR INTEGRATION PHASE 1 FAILED:")
            if not field_name_success:
                logger.info("   ❌ Field name validation failed")
            if not escalation_success:
                logger.info("   ❌ Dynamic RR escalation logic failed")
            if not persistence_success:
                logger.info("   ❌ Database persistence failed")
            logger.info("🚨 REQUIRES IMMEDIATE ATTENTION: Phase 1 implementation not working correctly")
        
        return overall_success

async def main():
    """Main test execution function"""
    logger.info("🚀 Starting Dynamic RR Integration Testing Suite - Phase 1")
    logger.info("=" * 80)
    
    # Initialize test suite
    test_suite = DynamicRRIntegrationTestSuite()
    
    try:
        # Run all Dynamic RR Integration tests
        success = await test_suite.run_all_tests()
        
        if success:
            logger.info("🎉 DYNAMIC RR INTEGRATION PHASE 1 TESTING COMPLETED SUCCESSFULLY")
        else:
            logger.info("❌ DYNAMIC RR INTEGRATION PHASE 1 TESTING FAILED")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Test suite execution failed: {e}")
        return False
    
    # Summary is handled by run_all_tests method


if __name__ == "__main__":
    asyncio.run(main())
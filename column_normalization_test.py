#!/usr/bin/env python3
"""
COLUMN NORMALIZATION MULTI-TIMEFRAME SYSTEM VALIDATION

TESTING FOCUS: Validate that the column normalization fixes have resolved the multi-timeframe analysis pipeline issues and system is working properly.

KEY AREAS TO TEST:
1. **Multi-Timeframe Analysis Stability**: Test /api/force-ia1-analysis endpoint with multiple symbols (BTCUSDT, ETHUSDT, SOLUSDT) to ensure multi-timeframe analysis completes without column errors
2. **Column Error Resolution**: Verify backend logs show NO "KeyError: 'Close'" or similar column normalization errors during analysis
3. **Risk-Reward Calculation**: Confirm RR calculations work without column access errors and return proper values
4. **Data Processing Pipeline**: Validate that OHLCV data flows properly through all processing stages with consistent column naming
5. **System Performance**: Ensure API response times remain reasonable and system stability is maintained

TESTING REQUIREMENTS:
- Test at least 3 different symbols to validate consistency
- Check backend logs for any column-related errors
- Verify multi-timeframe data is being saved to database properly 
- Confirm API responses contain valid multi-timeframe analysis results
- Validate that fallback activation due to column errors has been eliminated

EXPECTED SUCCESS CRITERIA:
- All API calls complete successfully without column errors
- Backend logs show clean multi-timeframe analysis completion
- Risk-reward calculations function properly
- Multi-timeframe data persists to database
- No column normalization errors in logs
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

class ColumnNormalizationTestSuite:
    """Comprehensive test suite for Column Normalization Multi-Timeframe System validation"""
    
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
        logger.info(f"Testing Column Normalization Multi-Timeframe System at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Test symbols for multi-timeframe analysis (from review request)
        self.test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        self.actual_test_symbols = []
        
        # Column error patterns to check for
        self.column_error_patterns = [
            "KeyError: 'Close'",
            "KeyError: 'Open'", 
            "KeyError: 'High'",
            "KeyError: 'Low'",
            "KeyError: 'Volume'",
            "column normalization error",
            "OHLCV column access error",
            "fallback activation due to column",
            "column inconsistency"
        ]
        
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
                ['tail', '-n', '500', '/var/log/supervisor/backend.out.log'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                return result.stdout.split('\n')
            else:
                # Try alternative log location
                result = subprocess.run(
                    ['tail', '-n', '500', '/var/log/supervisor/backend.err.log'],
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
    
    async def test_1_multi_timeframe_analysis_stability(self):
        """Test 1: Multi-Timeframe Analysis Stability - Test /api/force-ia1-analysis with multiple symbols"""
        logger.info("\n🔍 TEST 1: Multi-Timeframe Analysis Stability")
        
        try:
            stability_results = {
                'analyses_attempted': 0,
                'analyses_successful': 0,
                'column_errors_detected': 0,
                'multi_timeframe_data_present': 0,
                'rr_calculations_working': 0,
                'response_times': [],
                'successful_symbols': [],
                'failed_symbols': [],
                'analysis_details': []
            }
            
            logger.info("   🚀 Testing multi-timeframe analysis stability with column normalization fixes...")
            logger.info(f"   📊 Testing symbols: {self.test_symbols}")
            
            # Get available symbols from scout system
            logger.info("   📞 Getting available symbols from scout system...")
            
            try:
                response = requests.get(f"{self.api_url}/opportunities", timeout=60)
                
                if response.status_code == 200:
                    opportunities = response.json()
                    if isinstance(opportunities, dict) and 'opportunities' in opportunities:
                        opportunities = opportunities['opportunities']
                    
                    # Get available symbols
                    available_symbols = [opp.get('symbol') for opp in opportunities[:20] if opp.get('symbol')]
                    
                    # Prefer test symbols if available
                    test_symbols = []
                    for symbol in self.test_symbols:
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
            
            # Test each symbol for multi-timeframe analysis stability
            for symbol in self.actual_test_symbols:
                logger.info(f"\n   📞 Testing multi-timeframe analysis for {symbol}...")
                stability_results['analyses_attempted'] += 1
                
                try:
                    start_time = time.time()
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=180  # Extended timeout for multi-timeframe analysis
                    )
                    response_time = time.time() - start_time
                    stability_results['response_times'].append(response_time)
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        stability_results['analyses_successful'] += 1
                        stability_results['successful_symbols'].append(symbol)
                        
                        logger.info(f"      ✅ {symbol} analysis successful (response time: {response_time:.2f}s)")
                        
                        # Extract analysis data
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        if not isinstance(ia1_analysis, dict):
                            ia1_analysis = {}
                        
                        # Check for multi-timeframe data presence
                        multi_tf_indicators = [
                            'current_price', 'entry_price', 'stop_loss_price', 'take_profit_price',
                            'rsi', 'macd_line', 'macd_histogram', 'adx', 'mfi', 'vwap'
                        ]
                        
                        multi_tf_present = 0
                        for indicator in multi_tf_indicators:
                            if indicator in ia1_analysis and ia1_analysis[indicator] is not None:
                                multi_tf_present += 1
                        
                        if multi_tf_present >= 6:  # At least 6 indicators present
                            stability_results['multi_timeframe_data_present'] += 1
                            logger.info(f"         ✅ Multi-timeframe data present: {multi_tf_present}/{len(multi_tf_indicators)} indicators")
                        else:
                            logger.warning(f"         ⚠️ Limited multi-timeframe data: {multi_tf_present}/{len(multi_tf_indicators)} indicators")
                        
                        # Check RR calculations
                        risk_reward_ratio = ia1_analysis.get('risk_reward_ratio') or ia1_analysis.get('calculated_rr')
                        entry_price = ia1_analysis.get('entry_price')
                        stop_loss = ia1_analysis.get('stop_loss_price')
                        take_profit = ia1_analysis.get('take_profit_price')
                        
                        if (risk_reward_ratio and risk_reward_ratio != 1.0 and 
                            entry_price and stop_loss and take_profit):
                            stability_results['rr_calculations_working'] += 1
                            logger.info(f"         ✅ RR calculation working: {risk_reward_ratio:.2f} (Entry: {entry_price}, SL: {stop_loss}, TP: {take_profit})")
                        else:
                            logger.warning(f"         ⚠️ RR calculation issues: RR={risk_reward_ratio}, Entry={entry_price}, SL={stop_loss}, TP={take_profit}")
                        
                        # Store analysis details
                        stability_results['analysis_details'].append({
                            'symbol': symbol,
                            'response_time': response_time,
                            'multi_tf_indicators': multi_tf_present,
                            'rr_working': risk_reward_ratio is not None and risk_reward_ratio != 1.0,
                            'entry_price': entry_price,
                            'risk_reward_ratio': risk_reward_ratio
                        })
                        
                    else:
                        stability_results['failed_symbols'].append(symbol)
                        logger.error(f"      ❌ {symbol} analysis failed: HTTP {response.status_code}")
                        if response.text:
                            error_text = response.text[:500]
                            logger.error(f"         Error response: {error_text}")
                            
                            # Check for column errors in response
                            for error_pattern in self.column_error_patterns:
                                if error_pattern.lower() in error_text.lower():
                                    stability_results['column_errors_detected'] += 1
                                    logger.error(f"         🚨 COLUMN ERROR DETECTED: {error_pattern}")
                                    break
                
                except Exception as e:
                    stability_results['failed_symbols'].append(symbol)
                    logger.error(f"      ❌ {symbol} analysis exception: {e}")
                    
                    # Check for column errors in exception
                    error_str = str(e).lower()
                    for error_pattern in self.column_error_patterns:
                        if error_pattern.lower() in error_str:
                            stability_results['column_errors_detected'] += 1
                            logger.error(f"         🚨 COLUMN ERROR IN EXCEPTION: {error_pattern}")
                            break
                
                # Wait between analyses
                if symbol != self.actual_test_symbols[-1]:
                    logger.info(f"      ⏳ Waiting 15 seconds before next analysis...")
                    await asyncio.sleep(15)
            
            # Final analysis
            success_rate = stability_results['analyses_successful'] / max(stability_results['analyses_attempted'], 1)
            multi_tf_rate = stability_results['multi_timeframe_data_present'] / max(stability_results['analyses_successful'], 1)
            rr_rate = stability_results['rr_calculations_working'] / max(stability_results['analyses_successful'], 1)
            avg_response_time = sum(stability_results['response_times']) / max(len(stability_results['response_times']), 1)
            
            logger.info(f"\n   📊 MULTI-TIMEFRAME ANALYSIS STABILITY RESULTS:")
            logger.info(f"      Analyses attempted: {stability_results['analyses_attempted']}")
            logger.info(f"      Analyses successful: {stability_results['analyses_successful']}")
            logger.info(f"      Success rate: {success_rate:.2f}")
            logger.info(f"      Multi-timeframe data present: {stability_results['multi_timeframe_data_present']} ({multi_tf_rate:.2f})")
            logger.info(f"      RR calculations working: {stability_results['rr_calculations_working']} ({rr_rate:.2f})")
            logger.info(f"      Column errors detected: {stability_results['column_errors_detected']}")
            logger.info(f"      Average response time: {avg_response_time:.2f}s")
            logger.info(f"      Successful symbols: {stability_results['successful_symbols']}")
            logger.info(f"      Failed symbols: {stability_results['failed_symbols']}")
            
            # Calculate test success
            success_criteria = [
                stability_results['analyses_successful'] >= 3,  # At least 3 successful analyses
                success_rate >= 0.6,  # At least 60% success rate
                stability_results['column_errors_detected'] == 0,  # No column errors
                stability_results['multi_timeframe_data_present'] >= 2,  # At least 2 with multi-TF data
                avg_response_time <= 60.0  # Reasonable response times
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Multi-Timeframe Analysis Stability", True, 
                                   f"Multi-timeframe analysis stable: {success_count}/{len(success_criteria)} criteria met. Success rate: {success_rate:.2f}, No column errors: {stability_results['column_errors_detected'] == 0}, Avg response time: {avg_response_time:.2f}s")
            else:
                self.log_test_result("Multi-Timeframe Analysis Stability", False, 
                                   f"Multi-timeframe analysis issues: {success_count}/{len(success_criteria)} criteria met. Column errors: {stability_results['column_errors_detected']}, Success rate: {success_rate:.2f}")
                
        except Exception as e:
            self.log_test_result("Multi-Timeframe Analysis Stability", False, f"Exception: {str(e)}")

    async def test_2_column_error_resolution(self):
        """Test 2: Column Error Resolution - Verify backend logs show NO column normalization errors"""
        logger.info("\n🔍 TEST 2: Column Error Resolution")
        
        try:
            column_error_results = {
                'backend_logs_captured': False,
                'total_log_lines': 0,
                'column_errors_found': 0,
                'keyerror_close_found': 0,
                'keyerror_ohlcv_found': 0,
                'fallback_activation_found': 0,
                'normalization_success_logs': 0,
                'multi_timeframe_completion_logs': 0,
                'error_details': [],
                'success_indicators': []
            }
            
            logger.info("   🚀 Analyzing backend logs for column normalization errors...")
            logger.info("   📊 Expected: NO 'KeyError: Close' or similar column access errors")
            
            # Capture backend logs
            logger.info("   📋 Capturing backend logs...")
            
            try:
                backend_logs = await self._capture_backend_logs()
                if backend_logs:
                    column_error_results['backend_logs_captured'] = True
                    column_error_results['total_log_lines'] = len(backend_logs)
                    logger.info(f"      ✅ Backend logs captured: {len(backend_logs)} lines")
                    
                    # Analyze logs for column errors
                    for log_line in backend_logs:
                        log_lower = log_line.lower()
                        
                        # Check for column error patterns
                        for error_pattern in self.column_error_patterns:
                            if error_pattern.lower() in log_lower:
                                column_error_results['column_errors_found'] += 1
                                column_error_results['error_details'].append({
                                    'pattern': error_pattern,
                                    'log_line': log_line.strip()
                                })
                                logger.error(f"         🚨 COLUMN ERROR FOUND: {error_pattern}")
                                logger.error(f"            Log: {log_line.strip()}")
                                
                                # Specific error counting
                                if "keyerror: 'close'" in log_lower:
                                    column_error_results['keyerror_close_found'] += 1
                                elif any(col in log_lower for col in ["keyerror: 'open'", "keyerror: 'high'", "keyerror: 'low'", "keyerror: 'volume'"]):
                                    column_error_results['keyerror_ohlcv_found'] += 1
                                elif "fallback activation" in log_lower:
                                    column_error_results['fallback_activation_found'] += 1
                                break
                        
                        # Check for success indicators
                        success_patterns = [
                            "normalized columns:",
                            "column normalization successful",
                            "multi-timeframe analysis completed",
                            "ohlcv data normalized",
                            "column helper function used"
                        ]
                        
                        for success_pattern in success_patterns:
                            if success_pattern in log_lower:
                                if "normalized columns:" in log_lower:
                                    column_error_results['normalization_success_logs'] += 1
                                elif "multi-timeframe analysis completed" in log_lower:
                                    column_error_results['multi_timeframe_completion_logs'] += 1
                                
                                column_error_results['success_indicators'].append({
                                    'pattern': success_pattern,
                                    'log_line': log_line.strip()
                                })
                                break
                    
                    logger.info(f"      📊 Log analysis results:")
                    logger.info(f"         - Total column errors: {column_error_results['column_errors_found']}")
                    logger.info(f"         - KeyError 'Close' errors: {column_error_results['keyerror_close_found']}")
                    logger.info(f"         - Other OHLCV KeyErrors: {column_error_results['keyerror_ohlcv_found']}")
                    logger.info(f"         - Fallback activations: {column_error_results['fallback_activation_found']}")
                    logger.info(f"         - Normalization success logs: {column_error_results['normalization_success_logs']}")
                    logger.info(f"         - Multi-TF completion logs: {column_error_results['multi_timeframe_completion_logs']}")
                    
                    # Show sample success indicators
                    if column_error_results['success_indicators']:
                        logger.info(f"      ✅ Sample success indicators:")
                        for indicator in column_error_results['success_indicators'][:3]:
                            logger.info(f"         - {indicator['pattern']}: {indicator['log_line'][:100]}...")
                    
                    # Show error details if any
                    if column_error_results['error_details']:
                        logger.error(f"      ❌ Column error details:")
                        for error in column_error_results['error_details'][:5]:  # Show first 5
                            logger.error(f"         - {error['pattern']}: {error['log_line'][:100]}...")
                    
                else:
                    logger.warning("      ⚠️ Could not capture backend logs")
                    
            except Exception as e:
                logger.warning(f"      ⚠️ Error analyzing backend logs: {e}")
            
            # Additional check: Look for recent multi-timeframe analyses in database
            logger.info("   🗄️ Checking database for recent multi-timeframe analyses...")
            
            try:
                from pymongo import MongoClient
                client = MongoClient(self.mongo_url)
                db = client[self.db_name]
                
                # Get recent analyses
                recent_analyses = list(db.technical_analyses.find().sort("timestamp", -1).limit(10))
                
                multi_tf_analyses = 0
                complete_analyses = 0
                
                for analysis in recent_analyses:
                    # Check if analysis has multi-timeframe indicators
                    multi_tf_indicators = ['rsi', 'macd_line', 'adx', 'mfi', 'vwap', 'entry_price']
                    present_indicators = sum(1 for indicator in multi_tf_indicators if analysis.get(indicator) is not None)
                    
                    if present_indicators >= 4:
                        multi_tf_analyses += 1
                        
                        # Check if analysis is complete (has reasoning, prices, etc.)
                        if (analysis.get('reasoning') and 
                            analysis.get('entry_price') and 
                            analysis.get('stop_loss_price')):
                            complete_analyses += 1
                
                logger.info(f"      📊 Database analysis: {multi_tf_analyses}/10 recent analyses have multi-TF data")
                logger.info(f"      📊 Complete analyses: {complete_analyses}/10 recent analyses are complete")
                
                client.close()
                
            except Exception as e:
                logger.warning(f"      ⚠️ Could not check database: {e}")
            
            # Final analysis
            logger.info(f"\n   📊 COLUMN ERROR RESOLUTION RESULTS:")
            logger.info(f"      Backend logs captured: {column_error_results['backend_logs_captured']}")
            logger.info(f"      Total log lines analyzed: {column_error_results['total_log_lines']}")
            logger.info(f"      Column errors found: {column_error_results['column_errors_found']}")
            logger.info(f"      KeyError 'Close' found: {column_error_results['keyerror_close_found']}")
            logger.info(f"      Other OHLCV KeyErrors: {column_error_results['keyerror_ohlcv_found']}")
            logger.info(f"      Fallback activations: {column_error_results['fallback_activation_found']}")
            logger.info(f"      Normalization success logs: {column_error_results['normalization_success_logs']}")
            
            # Calculate test success
            success_criteria = [
                column_error_results['backend_logs_captured'],  # Logs captured
                column_error_results['total_log_lines'] >= 100,  # Sufficient logs
                column_error_results['column_errors_found'] == 0,  # No column errors
                column_error_results['keyerror_close_found'] == 0,  # No KeyError 'Close'
                column_error_results['fallback_activation_found'] == 0  # No fallback activations
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Column Error Resolution", True, 
                                   f"Column errors resolved: {success_count}/{len(success_criteria)} criteria met. Column errors: {column_error_results['column_errors_found']}, KeyError 'Close': {column_error_results['keyerror_close_found']}")
            else:
                self.log_test_result("Column Error Resolution", False, 
                                   f"Column error issues remain: {success_count}/{len(success_criteria)} criteria met. Column errors found: {column_error_results['column_errors_found']}")
                
        except Exception as e:
            self.log_test_result("Column Error Resolution", False, f"Exception: {str(e)}")

    async def test_3_risk_reward_calculation_validation(self):
        """Test 3: Risk-Reward Calculation - Confirm RR calculations work without column access errors"""
        logger.info("\n🔍 TEST 3: Risk-Reward Calculation Validation")
        
        try:
            rr_validation_results = {
                'analyses_attempted': 0,
                'analyses_successful': 0,
                'rr_calculations_present': 0,
                'rr_calculations_valid': 0,
                'price_data_complete': 0,
                'column_errors_in_rr': 0,
                'rr_details': [],
                'calculation_methods': set()
            }
            
            logger.info("   🚀 Testing risk-reward calculations without column access errors...")
            logger.info("   📊 Expected: RR calculations complete with proper entry/SL/TP prices")
            
            # Test RR calculations with multiple symbols
            for symbol in self.actual_test_symbols[:3]:  # Test first 3 symbols
                logger.info(f"\n   📞 Testing RR calculation for {symbol}...")
                rr_validation_results['analyses_attempted'] += 1
                
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
                        rr_validation_results['analyses_successful'] += 1
                        
                        logger.info(f"      ✅ {symbol} analysis successful (response time: {response_time:.2f}s)")
                        
                        # Extract analysis data
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        if not isinstance(ia1_analysis, dict):
                            ia1_analysis = {}
                        
                        # Check RR calculation presence
                        risk_reward_ratio = ia1_analysis.get('risk_reward_ratio') or ia1_analysis.get('calculated_rr')
                        entry_price = ia1_analysis.get('entry_price')
                        stop_loss_price = ia1_analysis.get('stop_loss_price')
                        take_profit_price = ia1_analysis.get('take_profit_price')
                        current_price = ia1_analysis.get('current_price')
                        
                        logger.info(f"         📊 Price Data: Current={current_price}, Entry={entry_price}, SL={stop_loss_price}, TP={take_profit_price}")
                        logger.info(f"         📊 RR Ratio: {risk_reward_ratio}")
                        
                        # Check if RR calculation is present
                        if risk_reward_ratio is not None:
                            rr_validation_results['rr_calculations_present'] += 1
                            logger.info(f"         ✅ RR calculation present: {risk_reward_ratio}")
                            
                            # Check if RR calculation is valid (not default values)
                            if (isinstance(risk_reward_ratio, (int, float)) and 
                                risk_reward_ratio > 0 and 
                                risk_reward_ratio != 1.0):
                                rr_validation_results['rr_calculations_valid'] += 1
                                logger.info(f"         ✅ RR calculation valid: {risk_reward_ratio:.2f}")
                            else:
                                logger.warning(f"         ⚠️ RR calculation may be default/fallback: {risk_reward_ratio}")
                        else:
                            logger.warning(f"         ❌ RR calculation missing")
                        
                        # Check if price data is complete
                        if (entry_price and stop_loss_price and take_profit_price and 
                            all(isinstance(price, (int, float)) and price > 0 
                                for price in [entry_price, stop_loss_price, take_profit_price])):
                            rr_validation_results['price_data_complete'] += 1
                            logger.info(f"         ✅ Price data complete")
                            
                            # Manual RR calculation verification
                            if entry_price and stop_loss_price and take_profit_price:
                                # Assume LONG position for calculation
                                risk = abs(entry_price - stop_loss_price)
                                reward = abs(take_profit_price - entry_price)
                                manual_rr = reward / risk if risk > 0 else 0
                                
                                logger.info(f"         📊 Manual RR calculation: {manual_rr:.2f} (Risk: {risk:.6f}, Reward: {reward:.6f})")
                                
                                # Check if calculated RR is reasonable
                                if risk_reward_ratio and abs(risk_reward_ratio - manual_rr) < 0.5:
                                    logger.info(f"         ✅ RR calculation matches manual: {risk_reward_ratio:.2f} ≈ {manual_rr:.2f}")
                                elif risk_reward_ratio:
                                    logger.warning(f"         ⚠️ RR calculation differs from manual: {risk_reward_ratio:.2f} vs {manual_rr:.2f}")
                        else:
                            logger.warning(f"         ❌ Price data incomplete")
                        
                        # Check for calculation method indicators
                        rr_reasoning = ia1_analysis.get('rr_reasoning', '')
                        if rr_reasoning:
                            if 'atr' in rr_reasoning.lower():
                                rr_validation_results['calculation_methods'].add('ATR-based')
                            if 'support' in rr_reasoning.lower() or 'resistance' in rr_reasoning.lower():
                                rr_validation_results['calculation_methods'].add('Support/Resistance')
                            if 'fibonacci' in rr_reasoning.lower():
                                rr_validation_results['calculation_methods'].add('Fibonacci')
                        
                        # Store RR details
                        rr_validation_results['rr_details'].append({
                            'symbol': symbol,
                            'risk_reward_ratio': risk_reward_ratio,
                            'entry_price': entry_price,
                            'stop_loss_price': stop_loss_price,
                            'take_profit_price': take_profit_price,
                            'current_price': current_price,
                            'rr_present': risk_reward_ratio is not None,
                            'rr_valid': risk_reward_ratio is not None and risk_reward_ratio > 0 and risk_reward_ratio != 1.0,
                            'price_complete': all(price is not None for price in [entry_price, stop_loss_price, take_profit_price]),
                            'response_time': response_time
                        })
                        
                    else:
                        logger.error(f"      ❌ {symbol} analysis failed: HTTP {response.status_code}")
                        if response.text:
                            error_text = response.text[:500]
                            logger.error(f"         Error response: {error_text}")
                            
                            # Check for column errors in RR calculation
                            for error_pattern in self.column_error_patterns:
                                if error_pattern.lower() in error_text.lower():
                                    rr_validation_results['column_errors_in_rr'] += 1
                                    logger.error(f"         🚨 COLUMN ERROR IN RR CALCULATION: {error_pattern}")
                                    break
                
                except Exception as e:
                    logger.error(f"      ❌ {symbol} RR calculation exception: {e}")
                    
                    # Check for column errors in exception
                    error_str = str(e).lower()
                    for error_pattern in self.column_error_patterns:
                        if error_pattern.lower() in error_str:
                            rr_validation_results['column_errors_in_rr'] += 1
                            logger.error(f"         🚨 COLUMN ERROR IN RR EXCEPTION: {error_pattern}")
                            break
                
                # Wait between analyses
                if symbol != self.actual_test_symbols[2]:  # Not the last symbol
                    logger.info(f"      ⏳ Waiting 10 seconds before next analysis...")
                    await asyncio.sleep(10)
            
            # Final analysis
            success_rate = rr_validation_results['analyses_successful'] / max(rr_validation_results['analyses_attempted'], 1)
            rr_present_rate = rr_validation_results['rr_calculations_present'] / max(rr_validation_results['analyses_successful'], 1)
            rr_valid_rate = rr_validation_results['rr_calculations_valid'] / max(rr_validation_results['analyses_successful'], 1)
            price_complete_rate = rr_validation_results['price_data_complete'] / max(rr_validation_results['analyses_successful'], 1)
            
            logger.info(f"\n   📊 RISK-REWARD CALCULATION VALIDATION RESULTS:")
            logger.info(f"      Analyses attempted: {rr_validation_results['analyses_attempted']}")
            logger.info(f"      Analyses successful: {rr_validation_results['analyses_successful']}")
            logger.info(f"      Success rate: {success_rate:.2f}")
            logger.info(f"      RR calculations present: {rr_validation_results['rr_calculations_present']} ({rr_present_rate:.2f})")
            logger.info(f"      RR calculations valid: {rr_validation_results['rr_calculations_valid']} ({rr_valid_rate:.2f})")
            logger.info(f"      Price data complete: {rr_validation_results['price_data_complete']} ({price_complete_rate:.2f})")
            logger.info(f"      Column errors in RR: {rr_validation_results['column_errors_in_rr']}")
            logger.info(f"      Calculation methods detected: {list(rr_validation_results['calculation_methods'])}")
            
            # Show RR details
            if rr_validation_results['rr_details']:
                logger.info(f"      📊 RR Calculation Details:")
                for detail in rr_validation_results['rr_details']:
                    logger.info(f"         - {detail['symbol']}: RR={detail['risk_reward_ratio']:.2f if detail['risk_reward_ratio'] else 'None'}, Entry={detail['entry_price']:.6f if detail['entry_price'] else 'None'}, Valid={detail['rr_valid']}")
            
            # Calculate test success
            success_criteria = [
                rr_validation_results['analyses_successful'] >= 2,  # At least 2 successful analyses
                rr_validation_results['rr_calculations_present'] >= 2,  # At least 2 with RR calculations
                rr_validation_results['column_errors_in_rr'] == 0,  # No column errors in RR
                rr_present_rate >= 0.67,  # At least 67% have RR calculations
                price_complete_rate >= 0.67  # At least 67% have complete price data
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Risk-Reward Calculation Validation", True, 
                                   f"RR calculations working: {success_count}/{len(success_criteria)} criteria met. RR present rate: {rr_present_rate:.2f}, Valid rate: {rr_valid_rate:.2f}, No column errors: {rr_validation_results['column_errors_in_rr'] == 0}")
            else:
                self.log_test_result("Risk-Reward Calculation Validation", False, 
                                   f"RR calculation issues: {success_count}/{len(success_criteria)} criteria met. Column errors: {rr_validation_results['column_errors_in_rr']}, Present rate: {rr_present_rate:.2f}")
                
        except Exception as e:
            self.log_test_result("Risk-Reward Calculation Validation", False, f"Exception: {str(e)}")

    async def test_4_data_processing_pipeline_validation(self):
        """Test 4: Data Processing Pipeline - Validate OHLCV data flows properly with consistent column naming"""
        logger.info("\n🔍 TEST 4: Data Processing Pipeline Validation")
        
        try:
            pipeline_results = {
                'ohlcv_data_sources_tested': 0,
                'consistent_column_naming': 0,
                'data_flow_successful': 0,
                'technical_indicators_calculated': 0,
                'multi_timeframe_processing': 0,
                'pipeline_errors': 0,
                'data_source_details': [],
                'processing_stages': []
            }
            
            logger.info("   🚀 Testing OHLCV data processing pipeline with consistent column naming...")
            logger.info("   📊 Expected: Data flows through all stages without column inconsistencies")
            
            # Test data processing pipeline by analyzing backend logs during analysis
            logger.info("   📞 Triggering analysis to observe data processing pipeline...")
            
            # Clear any existing logs and trigger fresh analysis
            test_symbol = self.actual_test_symbols[0] if self.actual_test_symbols else 'BTCUSDT'
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_url}/force-ia1-analysis",
                    json={"symbol": test_symbol},
                    timeout=120
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    analysis_data = response.json()
                    logger.info(f"      ✅ {test_symbol} analysis successful for pipeline testing")
                    
                    # Wait a moment for logs to be written
                    await asyncio.sleep(5)
                    
                    # Capture fresh backend logs
                    backend_logs = await self._capture_backend_logs()
                    
                    if backend_logs:
                        logger.info(f"      📋 Analyzing {len(backend_logs)} log lines for pipeline validation...")
                        
                        # Look for data processing stages
                        processing_stages = [
                            'enhanced ohlcv fetcher',
                            'bingx enhanced provided',
                            'kraken enhanced provided',
                            'multi-source validation',
                            'normalized columns:',
                            'column normalization',
                            'technical indicators',
                            'multi-timeframe analysis',
                            'risk-reward calculation',
                            'fibonacci calculation'
                        ]
                        
                        stage_found = {}
                        for stage in processing_stages:
                            stage_found[stage] = 0
                        
                        data_sources_found = set()
                        column_operations = []
                        pipeline_errors_found = []
                        
                        for log_line in backend_logs:
                            log_lower = log_line.lower()
                            
                            # Check for processing stages
                            for stage in processing_stages:
                                if stage in log_lower:
                                    stage_found[stage] += 1
                                    
                                    # Extract data source information
                                    if 'enhanced provided' in stage:
                                        if 'bingx' in log_lower:
                                            data_sources_found.add('BingX')
                                        elif 'kraken' in log_lower:
                                            data_sources_found.add('Kraken')
                                        elif 'yahoo' in log_lower:
                                            data_sources_found.add('Yahoo Finance')
                            
                            # Check for column operations
                            if any(col_op in log_lower for col_op in ['normalized columns:', 'column helper', 'get_ohlcv_column']):
                                column_operations.append(log_line.strip())
                            
                            # Check for pipeline errors
                            if any(error in log_lower for error in ['pipeline error', 'data flow error', 'processing failed']):
                                pipeline_errors_found.append(log_line.strip())
                        
                        pipeline_results['ohlcv_data_sources_tested'] = len(data_sources_found)
                        pipeline_results['consistent_column_naming'] = len(column_operations)
                        pipeline_results['pipeline_errors'] = len(pipeline_errors_found)
                        
                        logger.info(f"      📊 Processing stages found:")
                        for stage, count in stage_found.items():
                            if count > 0:
                                logger.info(f"         - {stage}: {count} occurrences")
                                pipeline_results['processing_stages'].append({
                                    'stage': stage,
                                    'count': count
                                })
                        
                        logger.info(f"      📊 Data sources detected: {list(data_sources_found)}")
                        logger.info(f"      📊 Column operations: {len(column_operations)}")
                        logger.info(f"      📊 Pipeline errors: {len(pipeline_errors_found)}")
                        
                        # Show sample column operations
                        if column_operations:
                            logger.info(f"      ✅ Sample column operations:")
                            for op in column_operations[:3]:
                                logger.info(f"         - {op[:100]}...")
                        
                        # Show pipeline errors if any
                        if pipeline_errors_found:
                            logger.error(f"      ❌ Pipeline errors found:")
                            for error in pipeline_errors_found[:3]:
                                logger.error(f"         - {error[:100]}...")
                        
                        # Check if multi-timeframe processing occurred
                        multi_tf_indicators = ['4h analysis', '1h analysis', '15m analysis', 'multi-timeframe']
                        multi_tf_found = sum(1 for indicator in multi_tf_indicators 
                                           if any(indicator in log.lower() for log in backend_logs))
                        
                        if multi_tf_found > 0:
                            pipeline_results['multi_timeframe_processing'] = 1
                            logger.info(f"      ✅ Multi-timeframe processing detected: {multi_tf_found} indicators")
                        else:
                            logger.warning(f"      ⚠️ Multi-timeframe processing not clearly detected")
                        
                        # Check if technical indicators were calculated
                        tech_indicators = ['rsi calculated', 'macd calculated', 'adx calculated', 'mfi calculated']
                        tech_found = sum(1 for indicator in tech_indicators 
                                       if any(indicator in log.lower() for log in backend_logs))
                        
                        if tech_found > 0:
                            pipeline_results['technical_indicators_calculated'] = 1
                            logger.info(f"      ✅ Technical indicators calculated: {tech_found} indicators")
                        else:
                            logger.warning(f"      ⚠️ Technical indicator calculations not clearly detected")
                        
                        # Determine data flow success
                        critical_stages = ['enhanced ohlcv fetcher', 'technical indicators', 'multi-timeframe analysis']
                        stages_present = sum(1 for stage in critical_stages if stage_found.get(stage, 0) > 0)
                        
                        if stages_present >= 2:
                            pipeline_results['data_flow_successful'] = 1
                            logger.info(f"      ✅ Data flow successful: {stages_present}/{len(critical_stages)} critical stages present")
                        else:
                            logger.warning(f"      ⚠️ Data flow may have issues: {stages_present}/{len(critical_stages)} critical stages present")
                        
                        # Store data source details
                        pipeline_results['data_source_details'] = [
                            {
                                'source': source,
                                'detected': True
                            } for source in data_sources_found
                        ]
                    
                    else:
                        logger.warning("      ⚠️ Could not capture backend logs for pipeline analysis")
                
                else:
                    logger.error(f"      ❌ Analysis failed for pipeline testing: HTTP {response.status_code}")
            
            except Exception as e:
                logger.error(f"      ❌ Pipeline testing exception: {e}")
            
            # Additional validation: Check database for consistent data structure
            logger.info("   🗄️ Checking database for consistent data structure...")
            
            try:
                from pymongo import MongoClient
                client = MongoClient(self.mongo_url)
                db = client[self.db_name]
                
                # Get recent analyses to check data consistency
                recent_analyses = list(db.technical_analyses.find().sort("timestamp", -1).limit(5))
                
                consistent_structure = 0
                for analysis in recent_analyses:
                    # Check if analysis has expected OHLCV-derived fields
                    ohlcv_fields = ['current_price', 'entry_price', 'rsi', 'macd_line', 'adx']
                    present_fields = sum(1 for field in ohlcv_fields if analysis.get(field) is not None)
                    
                    if present_fields >= 3:  # At least 3 OHLCV-derived fields
                        consistent_structure += 1
                
                logger.info(f"      📊 Database consistency: {consistent_structure}/5 recent analyses have consistent OHLCV-derived data")
                
                client.close()
                
            except Exception as e:
                logger.warning(f"      ⚠️ Could not check database consistency: {e}")
            
            # Final analysis
            logger.info(f"\n   📊 DATA PROCESSING PIPELINE VALIDATION RESULTS:")
            logger.info(f"      OHLCV data sources tested: {pipeline_results['ohlcv_data_sources_tested']}")
            logger.info(f"      Consistent column naming operations: {pipeline_results['consistent_column_naming']}")
            logger.info(f"      Data flow successful: {pipeline_results['data_flow_successful']}")
            logger.info(f"      Technical indicators calculated: {pipeline_results['technical_indicators_calculated']}")
            logger.info(f"      Multi-timeframe processing: {pipeline_results['multi_timeframe_processing']}")
            logger.info(f"      Pipeline errors: {pipeline_results['pipeline_errors']}")
            logger.info(f"      Processing stages detected: {len(pipeline_results['processing_stages'])}")
            
            # Calculate test success
            success_criteria = [
                pipeline_results['ohlcv_data_sources_tested'] >= 1,  # At least 1 data source
                pipeline_results['consistent_column_naming'] >= 1,  # Column operations detected
                pipeline_results['data_flow_successful'] == 1,  # Data flow successful
                pipeline_results['pipeline_errors'] == 0,  # No pipeline errors
                len(pipeline_results['processing_stages']) >= 3  # At least 3 processing stages
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Data Processing Pipeline Validation", True, 
                                   f"Pipeline validation successful: {success_count}/{len(success_criteria)} criteria met. Data sources: {pipeline_results['ohlcv_data_sources_tested']}, Column ops: {pipeline_results['consistent_column_naming']}, No errors: {pipeline_results['pipeline_errors'] == 0}")
            else:
                self.log_test_result("Data Processing Pipeline Validation", False, 
                                   f"Pipeline validation issues: {success_count}/{len(success_criteria)} criteria met. Pipeline errors: {pipeline_results['pipeline_errors']}, Data flow: {pipeline_results['data_flow_successful']}")
                
        except Exception as e:
            self.log_test_result("Data Processing Pipeline Validation", False, f"Exception: {str(e)}")

    async def test_5_system_performance_validation(self):
        """Test 5: System Performance - Ensure API response times remain reasonable and system stability"""
        logger.info("\n🔍 TEST 5: System Performance Validation")
        
        try:
            performance_results = {
                'response_times': [],
                'successful_requests': 0,
                'failed_requests': 0,
                'timeout_requests': 0,
                'avg_response_time': 0,
                'max_response_time': 0,
                'min_response_time': 0,
                'system_stability': True,
                'memory_usage_stable': True,
                'performance_details': []
            }
            
            logger.info("   🚀 Testing system performance and stability with column normalization fixes...")
            logger.info("   📊 Expected: Reasonable response times (<60s) and stable system performance")
            
            # Test performance with multiple rapid requests
            test_symbols = self.actual_test_symbols[:3] if len(self.actual_test_symbols) >= 3 else ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
            
            for i, symbol in enumerate(test_symbols):
                logger.info(f"\n   📞 Performance test {i+1}/3 for {symbol}...")
                
                try:
                    start_time = time.time()
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=90  # Reasonable timeout
                    )
                    response_time = time.time() - start_time
                    performance_results['response_times'].append(response_time)
                    
                    if response.status_code == 200:
                        performance_results['successful_requests'] += 1
                        logger.info(f"      ✅ {symbol} successful (response time: {response_time:.2f}s)")
                        
                        # Check response quality
                        analysis_data = response.json()
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        
                        # Performance quality indicators
                        quality_indicators = {
                            'has_reasoning': bool(ia1_analysis.get('reasoning')),
                            'has_prices': bool(ia1_analysis.get('entry_price')),
                            'has_technical_data': bool(ia1_analysis.get('rsi')),
                            'response_complete': len(str(analysis_data)) > 1000
                        }
                        
                        quality_score = sum(quality_indicators.values())
                        logger.info(f"         📊 Response quality: {quality_score}/4 indicators present")
                        
                        performance_results['performance_details'].append({
                            'symbol': symbol,
                            'response_time': response_time,
                            'quality_score': quality_score,
                            'success': True
                        })
                        
                    else:
                        performance_results['failed_requests'] += 1
                        logger.error(f"      ❌ {symbol} failed: HTTP {response.status_code} (response time: {response_time:.2f}s)")
                        
                        performance_results['performance_details'].append({
                            'symbol': symbol,
                            'response_time': response_time,
                            'quality_score': 0,
                            'success': False
                        })
                
                except requests.exceptions.Timeout:
                    performance_results['timeout_requests'] += 1
                    logger.error(f"      ⏰ {symbol} timed out (>90s)")
                    
                    performance_results['performance_details'].append({
                        'symbol': symbol,
                        'response_time': 90.0,
                        'quality_score': 0,
                        'success': False,
                        'timeout': True
                    })
                
                except Exception as e:
                    performance_results['failed_requests'] += 1
                    logger.error(f"      ❌ {symbol} exception: {e}")
                    
                    performance_results['performance_details'].append({
                        'symbol': symbol,
                        'response_time': 0,
                        'quality_score': 0,
                        'success': False,
                        'exception': str(e)
                    })
                
                # Wait between requests to avoid overwhelming system
                if i < len(test_symbols) - 1:
                    logger.info(f"      ⏳ Waiting 10 seconds before next performance test...")
                    await asyncio.sleep(10)
            
            # Calculate performance metrics
            if performance_results['response_times']:
                performance_results['avg_response_time'] = sum(performance_results['response_times']) / len(performance_results['response_times'])
                performance_results['max_response_time'] = max(performance_results['response_times'])
                performance_results['min_response_time'] = min(performance_results['response_times'])
            
            # Check system stability indicators
            logger.info("   📊 Checking system stability indicators...")
            
            try:
                # Check if system is responsive
                health_response = requests.get(f"{self.api_url}/opportunities", timeout=30)
                if health_response.status_code == 200:
                    logger.info("      ✅ System health check passed")
                else:
                    performance_results['system_stability'] = False
                    logger.warning(f"      ⚠️ System health check failed: HTTP {health_response.status_code}")
            
            except Exception as e:
                performance_results['system_stability'] = False
                logger.warning(f"      ⚠️ System health check exception: {e}")
            
            # Check backend logs for performance issues
            try:
                backend_logs = await self._capture_backend_logs()
                if backend_logs:
                    performance_issues = []
                    memory_issues = []
                    
                    for log_line in backend_logs:
                        log_lower = log_line.lower()
                        
                        # Check for performance issues
                        if any(issue in log_lower for issue in ['timeout', 'slow', 'performance', 'memory error', 'out of memory']):
                            performance_issues.append(log_line.strip())
                        
                        # Check for memory issues
                        if any(mem_issue in log_lower for mem_issue in ['memory', 'ram', 'heap', 'gc']):
                            memory_issues.append(log_line.strip())
                    
                    if performance_issues:
                        logger.warning(f"      ⚠️ Performance issues found in logs: {len(performance_issues)}")
                        for issue in performance_issues[:2]:
                            logger.warning(f"         - {issue[:100]}...")
                    else:
                        logger.info(f"      ✅ No performance issues found in logs")
                    
                    if len(memory_issues) > 10:  # Too many memory-related logs might indicate issues
                        performance_results['memory_usage_stable'] = False
                        logger.warning(f"      ⚠️ High memory activity detected: {len(memory_issues)} memory-related logs")
                    else:
                        logger.info(f"      ✅ Memory usage appears stable")
            
            except Exception as e:
                logger.warning(f"      ⚠️ Could not analyze performance logs: {e}")
            
            # Final analysis
            total_requests = performance_results['successful_requests'] + performance_results['failed_requests'] + performance_results['timeout_requests']
            success_rate = performance_results['successful_requests'] / max(total_requests, 1)
            
            logger.info(f"\n   📊 SYSTEM PERFORMANCE VALIDATION RESULTS:")
            logger.info(f"      Total requests: {total_requests}")
            logger.info(f"      Successful requests: {performance_results['successful_requests']}")
            logger.info(f"      Failed requests: {performance_results['failed_requests']}")
            logger.info(f"      Timeout requests: {performance_results['timeout_requests']}")
            logger.info(f"      Success rate: {success_rate:.2f}")
            logger.info(f"      Average response time: {performance_results['avg_response_time']:.2f}s")
            logger.info(f"      Max response time: {performance_results['max_response_time']:.2f}s")
            logger.info(f"      Min response time: {performance_results['min_response_time']:.2f}s")
            logger.info(f"      System stability: {performance_results['system_stability']}")
            logger.info(f"      Memory usage stable: {performance_results['memory_usage_stable']}")
            
            # Show performance details
            if performance_results['performance_details']:
                logger.info(f"      📊 Performance Details:")
                for detail in performance_results['performance_details']:
                    status = "✅" if detail['success'] else "❌"
                    logger.info(f"         {status} {detail['symbol']}: {detail['response_time']:.2f}s, quality: {detail['quality_score']}/4")
            
            # Calculate test success
            success_criteria = [
                performance_results['successful_requests'] >= 2,  # At least 2 successful requests
                success_rate >= 0.67,  # At least 67% success rate
                performance_results['avg_response_time'] <= 60.0,  # Average response time <= 60s
                performance_results['timeout_requests'] == 0,  # No timeouts
                performance_results['system_stability']  # System stable
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("System Performance Validation", True, 
                                   f"Performance validation successful: {success_count}/{len(success_criteria)} criteria met. Success rate: {success_rate:.2f}, Avg response time: {performance_results['avg_response_time']:.2f}s, Stable: {performance_results['system_stability']}")
            else:
                self.log_test_result("System Performance Validation", False, 
                                   f"Performance validation issues: {success_count}/{len(success_criteria)} criteria met. Success rate: {success_rate:.2f}, Timeouts: {performance_results['timeout_requests']}, Stable: {performance_results['system_stability']}")
                
        except Exception as e:
            self.log_test_result("System Performance Validation", False, f"Exception: {str(e)}")

    async def run_all_tests(self):
        """Run all column normalization validation tests"""
        logger.info("🚀 STARTING COLUMN NORMALIZATION MULTI-TIMEFRAME SYSTEM VALIDATION")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Run all tests
        await self.test_1_multi_timeframe_analysis_stability()
        await self.test_2_column_error_resolution()
        await self.test_3_risk_reward_calculation_validation()
        await self.test_4_data_processing_pipeline_validation()
        await self.test_5_system_performance_validation()
        
        total_time = time.time() - start_time
        
        # Generate final report
        logger.info("\n" + "=" * 80)
        logger.info("📊 COLUMN NORMALIZATION VALIDATION FINAL REPORT")
        logger.info("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {passed_tests/total_tests:.2%}")
        logger.info(f"Total Time: {total_time:.2f} seconds")
        
        logger.info("\nTest Results Summary:")
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
        
        # Overall assessment
        if passed_tests == total_tests:
            logger.info("\n🎉 ALL TESTS PASSED - Column normalization fixes are working correctly!")
        elif passed_tests >= total_tests * 0.8:
            logger.info(f"\n✅ MOSTLY SUCCESSFUL - {passed_tests}/{total_tests} tests passed. Minor issues may remain.")
        else:
            logger.info(f"\n⚠️ SIGNIFICANT ISSUES - Only {passed_tests}/{total_tests} tests passed. Column normalization may need further fixes.")
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests/total_tests,
            'total_time': total_time,
            'test_results': self.test_results
        }

async def main():
    """Main test execution"""
    test_suite = ColumnNormalizationTestSuite()
    results = await test_suite.run_all_tests()
    return results

if __name__ == "__main__":
    asyncio.run(main())
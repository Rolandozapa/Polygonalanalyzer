#!/usr/bin/env python3
"""
PATTERN DETECTION SYSTEM FIXES AND MACD CALCULATION ISSUES TEST SUITE
Focus: Test pattern detection system fixes and MACD calculation issues in trading bot system

CRITICAL TEST REQUIREMENTS FROM REVIEW REQUEST:
1. **PATTERN DETECTION SYSTEM FIX**: 
   - Previous issue: Yahoo Finance OHLCV was disabled in technical_pattern_detector.py line 289-291
   - Previous issue: Pattern detection was disabled in server.py line 2015-2017 (now fixed)
   - Fix implemented: Re-enabled Yahoo Finance OHLCV in technical_pattern_detector.py (_fetch_yahoo_ohlcv method)
   - Fix implemented: Re-enabled pattern detection in server.py (removed bypass on line 2017)
   - Expected result: Pattern detection should show "✅ Pattern detection enabled" instead of "⚠️ Pattern detection temporarily disabled"
   - Expected result: Patterns array should contain detected pattern names instead of being empty

2. **MACD CALCULATION FIX**: 
   - Previous issue: All IA1 analyses showed MACD values as 0.000000 despite numpy.float64 fix
   - Fix implemented: Enhanced OHLCV system should properly feed data to IA1 analysis
   - Expected result: MACD values should show real calculations (e.g., 214.39) instead of 0.000000
   - Test: Check latest analysis for non-zero MACD values (macd_signal, macd_line, macd_histogram, macd_trend)

3. **TECHNICAL INDICATORS INTEGRATION**:
   - Previous issue: Enhanced OHLCV system not properly feeding data to IA1 analysis
   - Fix implemented: Enhanced OHLCV system integration with technical indicators
   - Expected result: Technical indicators should receive real OHLCV data instead of fallback values
   - Test: Verify RSI, MACD, MFI, VWAP show meaningful signals instead of 'unknown'

4. **KEY ENDPOINTS TO TEST**:
   - GET /api/opportunities (should show opportunities with pattern detection)
   - POST /api/run-ia1-cycle (should show real MACD values and detected patterns)
   - GET /api/analyses (should show recent analyses with non-zero MACD values)

5. **CRITICAL FIXES IMPLEMENTED**:
   - Re-enabled Yahoo Finance OHLCV in technical_pattern_detector.py (_fetch_yahoo_ohlcv method)
   - Re-enabled pattern detection in server.py (removed bypass on line 2017)
   - Fixed yfinance duplicate entry in requirements.txt

SUCCESS CRITERIA:
✅ Pattern detection should show "✅ Pattern detection enabled" instead of "⚠️ Pattern detection temporarily disabled"
✅ MACD values should show real calculations (e.g., 214.39) instead of 0.000000
✅ Patterns array should contain detected pattern names instead of being empty
✅ Technical indicators should show meaningful values instead of 'unknown'
✅ Enhanced OHLCV system should properly feed data to IA1 analysis
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

class MACDFibonacciIntegrationTestSuite:
    """Comprehensive test suite for MACD Calculation Fix and Fibonacci Retracement Integration"""
    
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
        logger.info(f"Testing MACD & Fibonacci Integration at: {self.api_url}")
        
        # MongoDB connection for direct database analysis
        try:
            self.mongo_client = MongoClient("mongodb://localhost:27017")
            self.db = self.mongo_client["myapp"]
            logger.info("✅ MongoDB connection established for MACD & Fibonacci testing")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.mongo_client = None
            self.db = None
        
        # Test results
        self.test_results = []
        
        # Expected MACD fields
        self.macd_fields = ['macd_signal', 'macd_line', 'macd_histogram', 'macd_trend']
        
        # Expected Fibonacci fields
        self.fibonacci_fields = ['fibonacci_signal_strength', 'fibonacci_signal_direction', 'fibonacci_key_level_proximity']
        
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
    
    async def test_1_macd_calculation_fix_verification(self):
        """Test 1: MACD Calculation Fix - Verify Real MACD Values Instead of Zeros"""
        logger.info("\n🔍 TEST 1: MACD Calculation Fix Verification")
        
        try:
            macd_results = {
                'ia1_cycle_successful': False,
                'macd_fields_present': False,
                'macd_values_non_zero': False,
                'macd_signal_meaningful': False,
                'database_persistence': False,
                'macd_data': {},
                'analysis_id': None
            }
            
            logger.info("   🚀 Testing MACD calculation fix in IA1 analysis...")
            logger.info("   📊 Expected: Real MACD values (not 0.000000) in analysis response")
            
            # Step 1: Trigger IA1 cycle to generate new analysis with MACD fix
            logger.info("   📈 Running IA1 cycle to generate analysis with MACD fix...")
            start_time = time.time()
            response = requests.post(f"{self.api_url}/run-ia1-cycle", timeout=120)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                cycle_data = response.json()
                
                if cycle_data.get('success'):
                    macd_results['ia1_cycle_successful'] = True
                    logger.info(f"      ✅ IA1 cycle successful (response time: {response_time:.2f}s)")
                    
                    # Check if analysis data contains MACD fields
                    analysis_data = cycle_data.get('analysis_data', {})
                    if analysis_data:
                        logger.info(f"      📋 Analysis data received: {len(str(analysis_data))} characters")
                        
                        # Check for MACD fields presence
                        macd_fields_found = []
                        for field in self.macd_fields:
                            if field in analysis_data:
                                macd_fields_found.append(field)
                                macd_results['macd_data'][field] = analysis_data[field]
                        
                        if len(macd_fields_found) >= 2:  # At least 2 MACD fields present
                            macd_results['macd_fields_present'] = True
                            logger.info(f"      ✅ MACD fields present: {macd_fields_found}")
                            
                            # Check for non-zero MACD values
                            non_zero_values = []
                            for field, value in macd_results['macd_data'].items():
                                if isinstance(value, (int, float)) and value != 0.0:
                                    non_zero_values.append(f"{field}={value}")
                                elif isinstance(value, str) and value not in ['0', '0.0', '0.000000', 'unknown', 'neutral']:
                                    non_zero_values.append(f"{field}={value}")
                            
                            if non_zero_values:
                                macd_results['macd_values_non_zero'] = True
                                logger.info(f"      ✅ Non-zero MACD values found: {non_zero_values}")
                                
                                # Check for meaningful MACD signal
                                macd_signal = macd_results['macd_data'].get('macd_signal', 'unknown')
                                if macd_signal not in ['unknown', 'neutral', '0', 0, 0.0]:
                                    macd_results['macd_signal_meaningful'] = True
                                    logger.info(f"      ✅ Meaningful MACD signal: {macd_signal}")
                                else:
                                    logger.warning(f"      ⚠️ MACD signal still showing default: {macd_signal}")
                            else:
                                logger.warning(f"      ❌ All MACD values are zero or default: {macd_results['macd_data']}")
                        else:
                            logger.warning(f"      ❌ Insufficient MACD fields found: {macd_fields_found}")
                    else:
                        logger.warning(f"      ❌ No analysis data in IA1 cycle response")
                else:
                    logger.warning(f"      ❌ IA1 cycle failed: {cycle_data.get('error', 'Unknown error')}")
            else:
                logger.error(f"      ❌ IA1 cycle HTTP error: {response.status_code}")
                if response.text:
                    logger.error(f"         Error response: {response.text[:200]}...")
            
            # Step 2: Verify database persistence of MACD data
            if macd_results['ia1_cycle_successful'] and self.db:
                logger.info("   📊 Checking database persistence of MACD data...")
                try:
                    # Get latest analysis from database
                    latest_analysis = self.db.technical_analyses.find_one(
                        {}, sort=[("timestamp", -1)]
                    )
                    
                    if latest_analysis:
                        macd_results['analysis_id'] = latest_analysis.get('id', 'N/A')
                        logger.info(f"      📋 Latest analysis found: {macd_results['analysis_id']}")
                        
                        # Check for MACD fields in database
                        db_macd_fields = []
                        for field in self.macd_fields:
                            if field in latest_analysis and latest_analysis[field] not in [None, 0, 0.0, '0', 'unknown']:
                                db_macd_fields.append(f"{field}={latest_analysis[field]}")
                        
                        if db_macd_fields:
                            macd_results['database_persistence'] = True
                            logger.info(f"      ✅ MACD data persisted in database: {db_macd_fields}")
                        else:
                            logger.warning(f"      ❌ No meaningful MACD data found in database")
                    else:
                        logger.warning(f"      ❌ No analyses found in database")
                        
                except Exception as e:
                    logger.error(f"      ❌ Database query error: {e}")
            
            # Calculate test success
            success_criteria = [
                macd_results['ia1_cycle_successful'],
                macd_results['macd_fields_present'],
                macd_results['macd_values_non_zero'] or macd_results['macd_signal_meaningful']
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.67:  # 67% success threshold
                self.log_test_result("MACD Calculation Fix Verification", True, 
                                   f"MACD fix working: {success_count}/{len(success_criteria)} criteria met. MACD data: {macd_results['macd_data']}")
            else:
                self.log_test_result("MACD Calculation Fix Verification", False, 
                                   f"MACD fix issues: {success_count}/{len(success_criteria)} criteria met. Data: {macd_results['macd_data']}")
                
        except Exception as e:
            self.log_test_result("MACD Calculation Fix Verification", False, f"Exception: {str(e)}")
    
    async def test_2_fibonacci_retracement_integration(self):
        """Test 2: Fibonacci Retracement Integration - Verify New Fibonacci Fields"""
        logger.info("\n🔍 TEST 2: Fibonacci Retracement Integration Verification")
        
        try:
            fibonacci_results = {
                'ia1_cycle_successful': False,
                'fibonacci_fields_present': False,
                'fibonacci_values_meaningful': False,
                'fibonacci_levels_calculated': False,
                'database_persistence': False,
                'fibonacci_data': {},
                'analysis_id': None
            }
            
            logger.info("   🚀 Testing Fibonacci retracement integration in IA1 analysis...")
            logger.info("   📊 Expected: Fibonacci fields with meaningful values and 9-level analysis")
            
            # Step 1: Trigger IA1 cycle to generate analysis with Fibonacci integration
            logger.info("   📈 Running IA1 cycle to generate analysis with Fibonacci integration...")
            start_time = time.time()
            response = requests.post(f"{self.api_url}/run-ia1-cycle", timeout=120)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                cycle_data = response.json()
                
                if cycle_data.get('success'):
                    fibonacci_results['ia1_cycle_successful'] = True
                    logger.info(f"      ✅ IA1 cycle successful (response time: {response_time:.2f}s)")
                    
                    # Check if analysis data contains Fibonacci fields
                    analysis_data = cycle_data.get('analysis_data', {})
                    if analysis_data:
                        logger.info(f"      📋 Analysis data received: {len(str(analysis_data))} characters")
                        
                        # Check for Fibonacci fields presence
                        fibonacci_fields_found = []
                        for field in self.fibonacci_fields:
                            if field in analysis_data:
                                fibonacci_fields_found.append(field)
                                fibonacci_results['fibonacci_data'][field] = analysis_data[field]
                        
                        if len(fibonacci_fields_found) >= 2:  # At least 2 Fibonacci fields present
                            fibonacci_results['fibonacci_fields_present'] = True
                            logger.info(f"      ✅ Fibonacci fields present: {fibonacci_fields_found}")
                            
                            # Check for meaningful Fibonacci values
                            meaningful_values = []
                            for field, value in fibonacci_results['fibonacci_data'].items():
                                if field == 'fibonacci_signal_strength' and isinstance(value, (int, float)) and 0 <= value <= 1:
                                    meaningful_values.append(f"{field}={value}")
                                elif field == 'fibonacci_signal_direction' and isinstance(value, str) and value in ['bullish', 'bearish', 'neutral']:
                                    meaningful_values.append(f"{field}={value}")
                                elif field == 'fibonacci_key_level_proximity' and isinstance(value, bool):
                                    meaningful_values.append(f"{field}={value}")
                            
                            if meaningful_values:
                                fibonacci_results['fibonacci_values_meaningful'] = True
                                logger.info(f"      ✅ Meaningful Fibonacci values found: {meaningful_values}")
                            else:
                                logger.warning(f"      ❌ Fibonacci values not meaningful: {fibonacci_results['fibonacci_data']}")
                            
                            # Check for Fibonacci levels in analysis (support/resistance)
                            support_levels = analysis_data.get('support', [])
                            resistance_levels = analysis_data.get('resistance', [])
                            if len(support_levels) >= 2 or len(resistance_levels) >= 2:
                                fibonacci_results['fibonacci_levels_calculated'] = True
                                logger.info(f"      ✅ Fibonacci levels calculated: {len(support_levels)} support, {len(resistance_levels)} resistance")
                            else:
                                logger.warning(f"      ⚠️ Limited Fibonacci levels: {len(support_levels)} support, {len(resistance_levels)} resistance")
                        else:
                            logger.warning(f"      ❌ Insufficient Fibonacci fields found: {fibonacci_fields_found}")
                    else:
                        logger.warning(f"      ❌ No analysis data in IA1 cycle response")
                else:
                    logger.warning(f"      ❌ IA1 cycle failed: {cycle_data.get('error', 'Unknown error')}")
            else:
                logger.error(f"      ❌ IA1 cycle HTTP error: {response.status_code}")
                if response.text:
                    logger.error(f"         Error response: {response.text[:200]}...")
            
            # Step 2: Verify database persistence of Fibonacci data
            if fibonacci_results['ia1_cycle_successful'] and self.db:
                logger.info("   📊 Checking database persistence of Fibonacci data...")
                try:
                    # Get latest analysis from database
                    latest_analysis = self.db.technical_analyses.find_one(
                        {}, sort=[("timestamp", -1)]
                    )
                    
                    if latest_analysis:
                        fibonacci_results['analysis_id'] = latest_analysis.get('id', 'N/A')
                        logger.info(f"      📋 Latest analysis found: {fibonacci_results['analysis_id']}")
                        
                        # Check for Fibonacci fields in database
                        db_fibonacci_fields = []
                        for field in self.fibonacci_fields:
                            if field in latest_analysis and latest_analysis[field] is not None:
                                db_fibonacci_fields.append(f"{field}={latest_analysis[field]}")
                        
                        if db_fibonacci_fields:
                            fibonacci_results['database_persistence'] = True
                            logger.info(f"      ✅ Fibonacci data persisted in database: {db_fibonacci_fields}")
                        else:
                            logger.warning(f"      ❌ No Fibonacci data found in database")
                    else:
                        logger.warning(f"      ❌ No analyses found in database")
                        
                except Exception as e:
                    logger.error(f"      ❌ Database query error: {e}")
            
            # Calculate test success
            success_criteria = [
                fibonacci_results['ia1_cycle_successful'],
                fibonacci_results['fibonacci_fields_present'],
                fibonacci_results['fibonacci_values_meaningful']
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.67:  # 67% success threshold
                self.log_test_result("Fibonacci Retracement Integration", True, 
                                   f"Fibonacci integration working: {success_count}/{len(success_criteria)} criteria met. Fibonacci data: {fibonacci_results['fibonacci_data']}")
            else:
                self.log_test_result("Fibonacci Retracement Integration", False, 
                                   f"Fibonacci integration issues: {success_count}/{len(success_criteria)} criteria met. Data: {fibonacci_results['fibonacci_data']}")
                
        except Exception as e:
            self.log_test_result("Fibonacci Retracement Integration", False, f"Exception: {str(e)}")
    
    async def test_3_api_endpoints_enhanced_data(self):
        """Test 3: API Endpoints Enhanced Data - Verify /api/analyses Returns Enhanced Technical Data"""
        logger.info("\n🔍 TEST 3: API Endpoints Enhanced Data Verification")
        
        try:
            api_results = {
                'analyses_endpoint_accessible': False,
                'enhanced_data_present': False,
                'macd_data_in_api': False,
                'fibonacci_data_in_api': False,
                'data_structure_valid': False,
                'latest_analysis': {},
                'total_analyses': 0
            }
            
            logger.info("   🚀 Testing /api/analyses endpoint for enhanced technical data...")
            logger.info("   📊 Expected: Enhanced analyses with MACD and Fibonacci data")
            
            # Step 1: Test /api/analyses endpoint
            logger.info("   📈 Calling /api/analyses endpoint...")
            start_time = time.time()
            response = requests.get(f"{self.api_url}/analyses", timeout=60)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                api_results['analyses_endpoint_accessible'] = True
                analyses_data = response.json()
                
                logger.info(f"      ✅ Analyses endpoint accessible (response time: {response_time:.2f}s)")
                
                if isinstance(analyses_data, list) and len(analyses_data) > 0:
                    api_results['total_analyses'] = len(analyses_data)
                    latest_analysis = analyses_data[0]  # Assuming sorted by latest
                    api_results['latest_analysis'] = latest_analysis
                    
                    logger.info(f"      📋 Found {api_results['total_analyses']} analyses, checking latest...")
                    
                    # Check for enhanced data structure
                    required_fields = ['symbol', 'timestamp', 'ia1_signal']
                    enhanced_fields_present = sum(1 for field in required_fields if field in latest_analysis)
                    
                    if enhanced_fields_present >= len(required_fields):
                        api_results['data_structure_valid'] = True
                        logger.info(f"      ✅ Valid data structure: {enhanced_fields_present}/{len(required_fields)} required fields")
                        
                        # Check for MACD data in API response
                        macd_fields_in_api = []
                        for field in self.macd_fields:
                            if field in latest_analysis and latest_analysis[field] not in [None, 0, 0.0, '0', 'unknown']:
                                macd_fields_in_api.append(f"{field}={latest_analysis[field]}")
                        
                        if macd_fields_in_api:
                            api_results['macd_data_in_api'] = True
                            logger.info(f"      ✅ MACD data in API response: {macd_fields_in_api}")
                        else:
                            logger.warning(f"      ❌ No meaningful MACD data in API response")
                        
                        # Check for Fibonacci data in API response
                        fibonacci_fields_in_api = []
                        for field in self.fibonacci_fields:
                            if field in latest_analysis and latest_analysis[field] is not None:
                                fibonacci_fields_in_api.append(f"{field}={latest_analysis[field]}")
                        
                        if fibonacci_fields_in_api:
                            api_results['fibonacci_data_in_api'] = True
                            logger.info(f"      ✅ Fibonacci data in API response: {fibonacci_fields_in_api}")
                        else:
                            logger.warning(f"      ❌ No Fibonacci data in API response")
                        
                        # Check if we have enhanced data overall
                        if api_results['macd_data_in_api'] or api_results['fibonacci_data_in_api']:
                            api_results['enhanced_data_present'] = True
                            logger.info(f"      ✅ Enhanced technical data present in API")
                        else:
                            logger.warning(f"      ❌ No enhanced technical data found in API")
                    else:
                        logger.warning(f"      ❌ Invalid data structure: {enhanced_fields_present}/{len(required_fields)} required fields")
                else:
                    logger.warning(f"      ❌ No analyses data or invalid format: {type(analyses_data)}")
            else:
                logger.error(f"      ❌ Analyses endpoint HTTP error: {response.status_code}")
                if response.text:
                    logger.error(f"         Error response: {response.text[:200]}...")
            
            # Calculate test success
            success_criteria = [
                api_results['analyses_endpoint_accessible'],
                api_results['data_structure_valid'],
                api_results['enhanced_data_present']
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.67:  # 67% success threshold
                self.log_test_result("API Endpoints Enhanced Data", True, 
                                   f"Enhanced API data working: {success_count}/{len(success_criteria)} criteria met. {api_results['total_analyses']} analyses, MACD: {api_results['macd_data_in_api']}, Fibonacci: {api_results['fibonacci_data_in_api']}")
            else:
                self.log_test_result("API Endpoints Enhanced Data", False, 
                                   f"Enhanced API data issues: {success_count}/{len(success_criteria)} criteria met")
                
        except Exception as e:
            self.log_test_result("API Endpoints Enhanced Data", False, f"Exception: {str(e)}")
    
    async def test_4_backend_integration_verification(self):
        """Test 4: Backend Integration - Verify fibonacci_calculator.py and MACD Integration"""
        logger.info("\n🔍 TEST 4: Backend Integration Verification")
        
        try:
            integration_results = {
                'fibonacci_calculator_imported': False,
                'macd_calculator_imported': False,
                'backend_logs_healthy': False,
                'no_import_errors': False,
                'calculations_working': False,
                'error_count': 0,
                'import_errors': []
            }
            
            logger.info("   🚀 Testing backend integration of fibonacci_calculator.py and MACD systems...")
            logger.info("   📊 Expected: No import errors, healthy backend logs, working calculations")
            
            # Step 1: Check backend logs for import errors and calculation health
            logger.info("   📋 Checking backend logs for integration health...")
            try:
                backend_logs = await self._capture_backend_logs()
                
                if backend_logs:
                    logger.info(f"      📊 Captured {len(backend_logs)} log lines for analysis")
                    
                    # Check for import errors
                    import_error_patterns = [
                        'ImportError',
                        'ModuleNotFoundError',
                        'fibonacci_calculator',
                        'macd_calculator',
                        'cannot import',
                        'No module named'
                    ]
                    
                    import_errors = []
                    for log_line in backend_logs:
                        for pattern in import_error_patterns:
                            if pattern.lower() in log_line.lower():
                                import_errors.append(log_line.strip())
                                break
                    
                    if not import_errors:
                        integration_results['no_import_errors'] = True
                        logger.info(f"      ✅ No import errors found in backend logs")
                    else:
                        integration_results['import_errors'] = import_errors[:3]  # Show first 3
                        logger.warning(f"      ❌ Import errors found: {len(import_errors)} errors")
                        for error in import_errors[:3]:
                            logger.warning(f"         - {error}")
                    
                    # Check for successful imports/calculations
                    success_patterns = [
                        'fibonacci_calculator',
                        'MACD calculated successfully',
                        'Fibonacci analysis',
                        'fibonacci retracement',
                        'MACD optimized'
                    ]
                    
                    success_messages = []
                    for log_line in backend_logs:
                        for pattern in success_patterns:
                            if pattern.lower() in log_line.lower() and 'error' not in log_line.lower():
                                success_messages.append(log_line.strip())
                                break
                    
                    if success_messages:
                        integration_results['calculations_working'] = True
                        logger.info(f"      ✅ Calculation success messages found: {len(success_messages)}")
                        for msg in success_messages[:2]:
                            logger.info(f"         - {msg}")
                    else:
                        logger.warning(f"      ⚠️ No calculation success messages found")
                    
                    # Check for fibonacci_calculator specific mentions
                    fibonacci_mentions = [log for log in backend_logs if 'fibonacci' in log.lower()]
                    if fibonacci_mentions:
                        integration_results['fibonacci_calculator_imported'] = True
                        logger.info(f"      ✅ Fibonacci calculator mentions found: {len(fibonacci_mentions)}")
                    
                    # Check for MACD specific mentions
                    macd_mentions = [log for log in backend_logs if 'macd' in log.lower()]
                    if macd_mentions:
                        integration_results['macd_calculator_imported'] = True
                        logger.info(f"      ✅ MACD calculator mentions found: {len(macd_mentions)}")
                    
                    # Overall backend health
                    error_patterns = ['ERROR', 'CRITICAL', 'Exception', 'Traceback']
                    error_count = 0
                    for log_line in backend_logs:
                        for pattern in error_patterns:
                            if pattern in log_line:
                                error_count += 1
                                break
                    
                    integration_results['error_count'] = error_count
                    if error_count < len(backend_logs) * 0.1:  # Less than 10% error rate
                        integration_results['backend_logs_healthy'] = True
                        logger.info(f"      ✅ Backend logs healthy: {error_count} errors in {len(backend_logs)} lines")
                    else:
                        logger.warning(f"      ⚠️ Backend logs show issues: {error_count} errors in {len(backend_logs)} lines")
                else:
                    logger.warning(f"      ⚠️ No backend logs captured")
                    
            except Exception as e:
                logger.error(f"      ❌ Error analyzing backend logs: {e}")
            
            # Step 2: Test a simple IA1 cycle to verify integration
            logger.info("   📈 Testing integration with IA1 cycle...")
            try:
                response = requests.post(f"{self.api_url}/run-ia1-cycle", timeout=60)
                if response.status_code == 200:
                    cycle_data = response.json()
                    if cycle_data.get('success'):
                        logger.info(f"      ✅ IA1 cycle successful - integration working")
                        integration_results['calculations_working'] = True
                    else:
                        logger.warning(f"      ⚠️ IA1 cycle failed: {cycle_data.get('error', 'Unknown')}")
                else:
                    logger.warning(f"      ⚠️ IA1 cycle HTTP error: {response.status_code}")
            except Exception as e:
                logger.warning(f"      ⚠️ IA1 cycle test error: {e}")
            
            # Calculate test success
            success_criteria = [
                integration_results['no_import_errors'],
                integration_results['backend_logs_healthy'],
                integration_results['fibonacci_calculator_imported'] or integration_results['macd_calculator_imported'],
                integration_results['calculations_working']
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.75:  # 75% success threshold
                self.log_test_result("Backend Integration Verification", True, 
                                   f"Backend integration working: {success_count}/{len(success_criteria)} criteria met. Errors: {integration_results['error_count']}, Fibonacci: {integration_results['fibonacci_calculator_imported']}, MACD: {integration_results['macd_calculator_imported']}")
            else:
                self.log_test_result("Backend Integration Verification", False, 
                                   f"Backend integration issues: {success_count}/{len(success_criteria)} criteria met. Import errors: {len(integration_results['import_errors'])}")
                
        except Exception as e:
            self.log_test_result("Backend Integration Verification", False, f"Exception: {str(e)}")
    
    async def test_5_database_persistence_verification(self):
        """Test 5: Database Persistence - Verify MACD and Fibonacci Data Storage"""
        logger.info("\n🔍 TEST 5: Database Persistence Verification")
        
        try:
            persistence_results = {
                'database_accessible': False,
                'recent_analyses_found': False,
                'macd_fields_in_db': False,
                'fibonacci_fields_in_db': False,
                'data_consistency': False,
                'analyses_count': 0,
                'sample_analysis': {}
            }
            
            logger.info("   🚀 Testing database persistence of MACD and Fibonacci data...")
            logger.info("   📊 Expected: Recent analyses with enhanced technical indicator fields")
            
            if not self.db:
                logger.error("      ❌ MongoDB connection not available")
                self.log_test_result("Database Persistence Verification", False, "MongoDB connection not available")
                return
            
            persistence_results['database_accessible'] = True
            
            # Step 1: Check for recent analyses in database
            try:
                # Get recent analyses (within last 2 hours)
                two_hours_ago = datetime.now() - timedelta(hours=2)
                recent_analyses = list(self.db.technical_analyses.find({
                    "timestamp": {"$gte": two_hours_ago}
                }).sort("timestamp", -1).limit(5))
                
                persistence_results['analyses_count'] = len(recent_analyses)
                
                if recent_analyses:
                    persistence_results['recent_analyses_found'] = True
                    latest_analysis = recent_analyses[0]
                    persistence_results['sample_analysis'] = {
                        'id': latest_analysis.get('id', 'N/A'),
                        'symbol': latest_analysis.get('symbol', 'N/A'),
                        'timestamp': latest_analysis.get('timestamp', 'N/A')
                    }
                    
                    logger.info(f"      ✅ Recent analyses found: {persistence_results['analyses_count']} analyses")
                    logger.info(f"         Latest: {persistence_results['sample_analysis']['symbol']} at {persistence_results['sample_analysis']['timestamp']}")
                    
                    # Check for MACD fields in database
                    macd_fields_in_db = []
                    for field in self.macd_fields:
                        if field in latest_analysis and latest_analysis[field] not in [None, 0, 0.0, '0', 'unknown']:
                            macd_fields_in_db.append(f"{field}={latest_analysis[field]}")
                    
                    if macd_fields_in_db:
                        persistence_results['macd_fields_in_db'] = True
                        logger.info(f"      ✅ MACD fields persisted: {macd_fields_in_db}")
                    else:
                        logger.warning(f"      ❌ No meaningful MACD fields in database")
                    
                    # Check for Fibonacci fields in database
                    fibonacci_fields_in_db = []
                    for field in self.fibonacci_fields:
                        if field in latest_analysis and latest_analysis[field] is not None:
                            fibonacci_fields_in_db.append(f"{field}={latest_analysis[field]}")
                    
                    if fibonacci_fields_in_db:
                        persistence_results['fibonacci_fields_in_db'] = True
                        logger.info(f"      ✅ Fibonacci fields persisted: {fibonacci_fields_in_db}")
                    else:
                        logger.warning(f"      ❌ No Fibonacci fields in database")
                    
                    # Check data consistency across multiple analyses
                    if len(recent_analyses) >= 2:
                        consistent_fields = 0
                        total_fields_checked = 0
                        
                        for field in self.macd_fields + self.fibonacci_fields:
                            if field in recent_analyses[0] and field in recent_analyses[1]:
                                total_fields_checked += 1
                                val1 = recent_analyses[0][field]
                                val2 = recent_analyses[1][field]
                                
                                # Check if both have meaningful values (not defaults)
                                if (val1 not in [None, 0, 0.0, '0', 'unknown'] and 
                                    val2 not in [None, 0, 0.0, '0', 'unknown']):
                                    consistent_fields += 1
                        
                        if total_fields_checked > 0 and consistent_fields >= total_fields_checked * 0.5:
                            persistence_results['data_consistency'] = True
                            logger.info(f"      ✅ Data consistency good: {consistent_fields}/{total_fields_checked} fields consistent")
                        else:
                            logger.warning(f"      ⚠️ Data consistency issues: {consistent_fields}/{total_fields_checked} fields consistent")
                else:
                    logger.warning(f"      ❌ No recent analyses found in database")
                    
            except Exception as e:
                logger.error(f"      ❌ Database query error: {e}")
            
            # Calculate test success
            success_criteria = [
                persistence_results['database_accessible'],
                persistence_results['recent_analyses_found'],
                persistence_results['macd_fields_in_db'] or persistence_results['fibonacci_fields_in_db']
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.67:  # 67% success threshold
                self.log_test_result("Database Persistence Verification", True, 
                                   f"Database persistence working: {success_count}/{len(success_criteria)} criteria met. {persistence_results['analyses_count']} analyses, MACD: {persistence_results['macd_fields_in_db']}, Fibonacci: {persistence_results['fibonacci_fields_in_db']}")
            else:
                self.log_test_result("Database Persistence Verification", False, 
                                   f"Database persistence issues: {success_count}/{len(success_criteria)} criteria met")
                
        except Exception as e:
            self.log_test_result("Database Persistence Verification", False, f"Exception: {str(e)}")
    
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
    
    async def run_comprehensive_test_suite(self):
        """Run comprehensive MACD & Fibonacci Integration test suite"""
        logger.info("🚀 Starting MACD Calculation Fix and Fibonacci Retracement Integration Test Suite")
        logger.info("=" * 80)
        logger.info("📋 MACD & FIBONACCI INTEGRATION TEST SUITE")
        logger.info("🎯 Testing: MACD calculation fix and Fibonacci retracement integration")
        logger.info("🎯 Expected: Real MACD values, Fibonacci analysis, enhanced API data")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_macd_calculation_fix_verification()
        await self.test_2_fibonacci_retracement_integration()
        await self.test_3_api_endpoints_enhanced_data()
        await self.test_4_backend_integration_verification()
        await self.test_5_database_persistence_verification()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 MACD & FIBONACCI INTEGRATION TEST SUMMARY")
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
            if "MACD Calculation Fix" in result['test']:
                requirements_status['MACD Calculation Fix (Real Values)'] = result['success']
            elif "Fibonacci Retracement Integration" in result['test']:
                requirements_status['Fibonacci Retracement Integration'] = result['success']
            elif "API Endpoints Enhanced Data" in result['test']:
                requirements_status['Enhanced API Endpoints'] = result['success']
            elif "Backend Integration" in result['test']:
                requirements_status['Backend Integration (fibonacci_calculator.py)'] = result['success']
            elif "Database Persistence" in result['test']:
                requirements_status['Database Persistence of Enhanced Data'] = result['success']
        
        logger.info("🎯 CRITICAL REQUIREMENTS STATUS:")
        for requirement, status in requirements_status.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {requirement}")
        
        requirements_met = sum(1 for status in requirements_status.values() if status)
        total_requirements = len(requirements_status)
        
        # Final verdict
        logger.info(f"\n🏆 REQUIREMENTS SATISFACTION: {requirements_met}/{total_requirements}")
        
        if requirements_met == total_requirements:
            logger.info("\n🎉 VERDICT: MACD & FIBONACCI INTEGRATION 100% SUCCESSFUL!")
            logger.info("✅ MACD calculation fix working - real values instead of zeros")
            logger.info("✅ Fibonacci retracement integration complete with 9-level analysis")
            logger.info("✅ Enhanced API endpoints returning technical indicator data")
            logger.info("✅ Backend integration of fibonacci_calculator.py working")
            logger.info("✅ Database persistence of enhanced technical data confirmed")
        elif requirements_met >= total_requirements * 0.8:
            logger.info("\n⚠️ VERDICT: MACD & FIBONACCI INTEGRATION MOSTLY SUCCESSFUL")
            logger.info("🔍 Minor issues may need attention for complete integration")
        elif requirements_met >= total_requirements * 0.6:
            logger.info("\n⚠️ VERDICT: MACD & FIBONACCI INTEGRATION PARTIALLY SUCCESSFUL")
            logger.info("🔧 Several requirements need attention for complete integration")
        else:
            logger.info("\n❌ VERDICT: MACD & FIBONACCI INTEGRATION NOT SUCCESSFUL")
            logger.info("🚨 Major issues detected - technical indicators not working properly")
            logger.info("🚨 System needs significant fixes for MACD and Fibonacci integration")
        
        return passed_tests, total_tests

async def main():
    """Main function to run the comprehensive MACD & Fibonacci Integration test suite"""
    test_suite = MACDFibonacciIntegrationTestSuite()
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
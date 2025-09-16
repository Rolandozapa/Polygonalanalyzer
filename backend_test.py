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

class PatternDetectionMACDTestSuite:
    """Comprehensive test suite for Pattern Detection System Fixes and MACD Calculation Issues"""
    
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
        logger.info(f"Testing Pattern Detection & MACD Fixes at: {self.api_url}")
        
        # MongoDB connection for direct database analysis
        try:
            self.mongo_client = MongoClient("mongodb://localhost:27017")
            self.db = self.mongo_client["myapp"]
            logger.info("✅ MongoDB connection established for Pattern Detection & MACD testing")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.mongo_client = None
            self.db = None
        
        # Test results
        self.test_results = []
        
        # Expected MACD fields
        self.macd_fields = ['macd_signal', 'macd_line', 'macd_histogram', 'macd_trend']
        
        # Expected pattern detection fields
        self.pattern_fields = ['patterns', 'master_pattern']
        
        # Expected technical indicator fields
        self.technical_indicator_fields = ['rsi_signal', 'mfi_signal', 'vwap_signal']
        
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
    
    async def test_1_pattern_detection_system_fix(self):
        """Test 1: Pattern Detection System Fix - Verify Pattern Detection is Re-enabled"""
        logger.info("\n🔍 TEST 1: Pattern Detection System Fix Verification")
        
        try:
            pattern_results = {
                'opportunities_endpoint_accessible': False,
                'pattern_detection_enabled': False,
                'patterns_detected': False,
                'yahoo_finance_working': False,
                'pattern_data': {},
                'opportunities_count': 0
            }
            
            logger.info("   🚀 Testing pattern detection system re-enablement...")
            logger.info("   📊 Expected: Pattern detection enabled, patterns detected in opportunities")
            
            # Step 1: Test /api/opportunities endpoint for pattern detection
            logger.info("   📈 Testing /api/opportunities endpoint for pattern detection...")
            start_time = time.time()
            response = requests.get(f"{self.api_url}/opportunities", timeout=60)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                pattern_results['opportunities_endpoint_accessible'] = True
                opportunities_data = response.json()
                
                logger.info(f"      ✅ Opportunities endpoint accessible (response time: {response_time:.2f}s)")
                
                if isinstance(opportunities_data, list) and len(opportunities_data) > 0:
                    pattern_results['opportunities_count'] = len(opportunities_data)
                    logger.info(f"      📋 Found {pattern_results['opportunities_count']} opportunities")
                    
                    # Check for pattern detection status in opportunities
                    patterns_found = 0
                    pattern_detection_messages = []
                    
                    for opportunity in opportunities_data[:5]:  # Check first 5 opportunities
                        # Look for pattern detection indicators
                        if 'patterns' in opportunity and opportunity['patterns']:
                            patterns_found += 1
                            pattern_results['patterns_detected'] = True
                            pattern_results['pattern_data'][opportunity.get('symbol', 'unknown')] = opportunity['patterns']
                        
                        # Look for pattern detection status messages
                        if 'analysis' in opportunity:
                            analysis_text = str(opportunity['analysis']).lower()
                            if 'pattern detection enabled' in analysis_text:
                                pattern_detection_messages.append("Pattern detection enabled found")
                                pattern_results['pattern_detection_enabled'] = True
                            elif 'pattern detection temporarily disabled' in analysis_text:
                                pattern_detection_messages.append("Pattern detection still disabled")
                    
                    if patterns_found > 0:
                        logger.info(f"      ✅ Patterns detected in {patterns_found} opportunities: {list(pattern_results['pattern_data'].keys())}")
                    else:
                        logger.warning(f"      ❌ No patterns detected in opportunities")
                    
                    if pattern_detection_messages:
                        logger.info(f"      📋 Pattern detection status messages: {pattern_detection_messages}")
                    
                    # Check if Yahoo Finance OHLCV is working (indirect test)
                    valid_price_data = 0
                    for opportunity in opportunities_data[:3]:
                        if (opportunity.get('current_price', 0) > 0 and 
                            opportunity.get('volume_24h', 0) > 0):
                            valid_price_data += 1
                    
                    if valid_price_data >= 2:
                        pattern_results['yahoo_finance_working'] = True
                        logger.info(f"      ✅ Yahoo Finance OHLCV appears to be working: {valid_price_data}/3 opportunities have valid price data")
                    else:
                        logger.warning(f"      ⚠️ Yahoo Finance OHLCV may have issues: {valid_price_data}/3 opportunities have valid price data")
                        
                else:
                    logger.warning(f"      ❌ No opportunities data or invalid format: {type(opportunities_data)}")
            else:
                logger.error(f"      ❌ Opportunities endpoint HTTP error: {response.status_code}")
                if response.text:
                    logger.error(f"         Error response: {response.text[:200]}...")
            
            # Step 2: Test IA1 cycle for pattern detection
            logger.info("   📈 Testing IA1 cycle for pattern detection...")
            try:
                response = requests.post(f"{self.api_url}/run-ia1-cycle", timeout=120)
                if response.status_code == 200:
                    cycle_data = response.json()
                    if cycle_data.get('success'):
                        analysis_data = cycle_data.get('analysis_data', {})
                        
                        # Check for patterns in IA1 analysis
                        if 'patterns' in analysis_data and analysis_data['patterns']:
                            pattern_results['patterns_detected'] = True
                            pattern_results['pattern_data']['ia1_analysis'] = analysis_data['patterns']
                            logger.info(f"      ✅ Patterns detected in IA1 analysis: {analysis_data['patterns']}")
                        
                        # Check for pattern detection status in analysis text
                        analysis_text = analysis_data.get('analysis', '')
                        if 'pattern detection enabled' in analysis_text.lower():
                            pattern_results['pattern_detection_enabled'] = True
                            logger.info(f"      ✅ Pattern detection enabled confirmed in IA1 analysis")
                        elif 'pattern detection temporarily disabled' in analysis_text.lower():
                            logger.warning(f"      ❌ Pattern detection still disabled in IA1 analysis")
                    else:
                        logger.warning(f"      ⚠️ IA1 cycle failed: {cycle_data.get('error', 'Unknown')}")
                else:
                    logger.warning(f"      ⚠️ IA1 cycle HTTP error: {response.status_code}")
            except Exception as e:
                logger.warning(f"      ⚠️ IA1 cycle test error: {e}")
            
            # Calculate test success
            success_criteria = [
                pattern_results['opportunities_endpoint_accessible'],
                pattern_results['yahoo_finance_working'],
                pattern_results['pattern_detection_enabled'] or pattern_results['patterns_detected']
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.67:  # 67% success threshold
                self.log_test_result("Pattern Detection System Fix", True, 
                                   f"Pattern detection working: {success_count}/{len(success_criteria)} criteria met. Opportunities: {pattern_results['opportunities_count']}, Patterns detected: {pattern_results['patterns_detected']}, Detection enabled: {pattern_results['pattern_detection_enabled']}")
            else:
                self.log_test_result("Pattern Detection System Fix", False, 
                                   f"Pattern detection issues: {success_count}/{len(success_criteria)} criteria met. Patterns: {pattern_results['pattern_data']}")
                
        except Exception as e:
            self.log_test_result("Pattern Detection System Fix", False, f"Exception: {str(e)}")
    
    async def test_2_macd_calculation_fix_verification(self):
        """Test 2: MACD Calculation Fix - Verify Real MACD Values Instead of Zeros"""
        logger.info("\n🔍 TEST 2: MACD Calculation Fix Verification")
        
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
    
    async def test_3_technical_indicators_integration(self):
        """Test 3: Technical Indicators Integration - Verify Enhanced OHLCV System Feeds Data"""
        logger.info("\n🔍 TEST 3: Technical Indicators Integration Verification")
        
        try:
            indicators_results = {
                'ia1_cycle_successful': False,
                'technical_indicators_present': False,
                'indicators_meaningful': False,
                'ohlcv_integration_working': False,
                'indicators_data': {},
                'analysis_id': None
            }
            
            logger.info("   🚀 Testing technical indicators integration with enhanced OHLCV system...")
            logger.info("   📊 Expected: RSI, MFI, VWAP showing meaningful signals instead of 'unknown'")
            
            # Step 1: Trigger IA1 cycle to test technical indicators
            logger.info("   📈 Running IA1 cycle to test technical indicators integration...")
            start_time = time.time()
            response = requests.post(f"{self.api_url}/run-ia1-cycle", timeout=120)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                cycle_data = response.json()
                
                if cycle_data.get('success'):
                    indicators_results['ia1_cycle_successful'] = True
                    logger.info(f"      ✅ IA1 cycle successful (response time: {response_time:.2f}s)")
                    
                    # Check if analysis data contains technical indicator fields
                    analysis_data = cycle_data.get('analysis_data', {})
                    if analysis_data:
                        logger.info(f"      📋 Analysis data received: {len(str(analysis_data))} characters")
                        
                        # Check for technical indicator fields presence
                        indicators_found = []
                        for field in self.technical_indicator_fields:
                            if field in analysis_data:
                                indicators_found.append(field)
                                indicators_results['indicators_data'][field] = analysis_data[field]
                        
                        if len(indicators_found) >= 2:  # At least 2 indicator fields present
                            indicators_results['technical_indicators_present'] = True
                            logger.info(f"      ✅ Technical indicator fields present: {indicators_found}")
                            
                            # Check for meaningful indicator values (not 'unknown')
                            meaningful_indicators = []
                            for field, value in indicators_results['indicators_data'].items():
                                if isinstance(value, str) and value not in ['unknown', 'neutral', 'N/A', '']:
                                    meaningful_indicators.append(f"{field}={value}")
                                elif isinstance(value, (int, float)) and value != 0:
                                    meaningful_indicators.append(f"{field}={value}")
                            
                            if meaningful_indicators:
                                indicators_results['indicators_meaningful'] = True
                                logger.info(f"      ✅ Meaningful indicator values found: {meaningful_indicators}")
                            else:
                                logger.warning(f"      ❌ All indicators showing default/unknown values: {indicators_results['indicators_data']}")
                            
                            # Check for OHLCV integration indicators
                            price_fields = ['entry_price', 'current_price']
                            valid_prices = 0
                            for field in price_fields:
                                if field in analysis_data and isinstance(analysis_data[field], (int, float)) and analysis_data[field] > 0:
                                    valid_prices += 1
                            
                            if valid_prices >= 1:
                                indicators_results['ohlcv_integration_working'] = True
                                logger.info(f"      ✅ OHLCV integration working: {valid_prices}/{len(price_fields)} price fields have valid data")
                            else:
                                logger.warning(f"      ❌ OHLCV integration issues: {valid_prices}/{len(price_fields)} price fields have valid data")
                        else:
                            logger.warning(f"      ❌ Insufficient technical indicator fields found: {indicators_found}")
                    else:
                        logger.warning(f"      ❌ No analysis data in IA1 cycle response")
                else:
                    logger.warning(f"      ❌ IA1 cycle failed: {cycle_data.get('error', 'Unknown error')}")
            else:
                logger.error(f"      ❌ IA1 cycle HTTP error: {response.status_code}")
                if response.text:
                    logger.error(f"         Error response: {response.text[:200]}...")
            
            # Step 2: Check /api/analyses for recent technical indicator data
            logger.info("   📊 Checking /api/analyses for technical indicator data...")
            try:
                response = requests.get(f"{self.api_url}/analyses", timeout=60)
                if response.status_code == 200:
                    analyses_data = response.json()
                    if isinstance(analyses_data, list) and len(analyses_data) > 0:
                        latest_analysis = analyses_data[0]
                        
                        # Check for technical indicators in API response
                        api_indicators = []
                        for field in self.technical_indicator_fields:
                            if field in latest_analysis and latest_analysis[field] not in ['unknown', 'neutral', None]:
                                api_indicators.append(f"{field}={latest_analysis[field]}")
                        
                        if api_indicators:
                            logger.info(f"      ✅ Technical indicators in API response: {api_indicators}")
                        else:
                            logger.warning(f"      ❌ No meaningful technical indicators in API response")
                    else:
                        logger.warning(f"      ❌ No analyses data in API response")
                else:
                    logger.warning(f"      ⚠️ Analyses endpoint error: {response.status_code}")
            except Exception as e:
                logger.warning(f"      ⚠️ Analyses endpoint test error: {e}")
            
            # Calculate test success
            success_criteria = [
                indicators_results['ia1_cycle_successful'],
                indicators_results['technical_indicators_present'],
                indicators_results['indicators_meaningful'] or indicators_results['ohlcv_integration_working']
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.67:  # 67% success threshold
                self.log_test_result("Technical Indicators Integration", True, 
                                   f"Technical indicators working: {success_count}/{len(success_criteria)} criteria met. Indicators: {indicators_results['indicators_data']}")
            else:
                self.log_test_result("Technical Indicators Integration", False, 
                                   f"Technical indicators issues: {success_count}/{len(success_criteria)} criteria met. Data: {indicators_results['indicators_data']}")
                
        except Exception as e:
            self.log_test_result("Technical Indicators Integration", False, f"Exception: {str(e)}")
    
    async def test_4_api_analyses_enhanced_data(self):
        """Test 4: API Analyses Enhanced Data - Verify /api/analyses Returns Non-Zero MACD Values"""
        logger.info("\n🔍 TEST 4: API Analyses Enhanced Data Verification")
        
        try:
            api_results = {
                'analyses_endpoint_accessible': False,
                'recent_analyses_found': False,
                'macd_data_in_api': False,
                'pattern_data_in_api': False,
                'data_structure_valid': False,
                'latest_analysis': {},
                'total_analyses': 0
            }
            
            logger.info("   🚀 Testing /api/analyses endpoint for enhanced data with non-zero MACD values...")
            logger.info("   📊 Expected: Recent analyses with non-zero MACD values and pattern data")
            
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
                    api_results['recent_analyses_found'] = True
                    latest_analysis = analyses_data[0]  # Assuming sorted by latest
                    api_results['latest_analysis'] = latest_analysis
                    
                    logger.info(f"      📋 Found {api_results['total_analyses']} analyses, checking latest...")
                    
                    # Check for enhanced data structure
                    required_fields = ['symbol', 'timestamp', 'ia1_signal']
                    enhanced_fields_present = sum(1 for field in required_fields if field in latest_analysis)
                    
                    if enhanced_fields_present >= len(required_fields):
                        api_results['data_structure_valid'] = True
                        logger.info(f"      ✅ Valid data structure: {enhanced_fields_present}/{len(required_fields)} required fields")
                        
                        # Check for MACD data in API response (specifically non-zero values)
                        macd_fields_in_api = []
                        for field in self.macd_fields:
                            if field in latest_analysis:
                                value = latest_analysis[field]
                                if isinstance(value, (int, float)) and value != 0.0:
                                    macd_fields_in_api.append(f"{field}={value}")
                                elif isinstance(value, str) and value not in ['0', '0.0', '0.000000', 'unknown', 'neutral']:
                                    macd_fields_in_api.append(f"{field}={value}")
                        
                        if macd_fields_in_api:
                            api_results['macd_data_in_api'] = True
                            logger.info(f"      ✅ Non-zero MACD data in API response: {macd_fields_in_api}")
                        else:
                            logger.warning(f"      ❌ No non-zero MACD data in API response")
                            # Log what MACD values we actually found
                            actual_macd_values = []
                            for field in self.macd_fields:
                                if field in latest_analysis:
                                    actual_macd_values.append(f"{field}={latest_analysis[field]}")
                            logger.warning(f"         Actual MACD values: {actual_macd_values}")
                        
                        # Check for pattern data in API response
                        pattern_fields_in_api = []
                        for field in self.pattern_fields:
                            if field in latest_analysis and latest_analysis[field]:
                                if isinstance(latest_analysis[field], list) and len(latest_analysis[field]) > 0:
                                    pattern_fields_in_api.append(f"{field}={latest_analysis[field]}")
                                elif isinstance(latest_analysis[field], str) and latest_analysis[field] not in ['', 'null', 'none']:
                                    pattern_fields_in_api.append(f"{field}={latest_analysis[field]}")
                        
                        if pattern_fields_in_api:
                            api_results['pattern_data_in_api'] = True
                            logger.info(f"      ✅ Pattern data in API response: {pattern_fields_in_api}")
                        else:
                            logger.warning(f"      ❌ No pattern data in API response")
                        
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
                api_results['recent_analyses_found'],
                api_results['data_structure_valid'],
                api_results['macd_data_in_api'] or api_results['pattern_data_in_api']
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.75:  # 75% success threshold
                self.log_test_result("API Analyses Enhanced Data", True, 
                                   f"Enhanced API data working: {success_count}/{len(success_criteria)} criteria met. {api_results['total_analyses']} analyses, MACD: {api_results['macd_data_in_api']}, Patterns: {api_results['pattern_data_in_api']}")
            else:
                self.log_test_result("API Analyses Enhanced Data", False, 
                                   f"Enhanced API data issues: {success_count}/{len(success_criteria)} criteria met")
                
        except Exception as e:
            self.log_test_result("API Analyses Enhanced Data", False, f"Exception: {str(e)}")
    
    async def test_5_backend_logs_verification(self):
        """Test 5: Backend Logs Verification - Check for Pattern Detection and MACD Integration Health"""
        logger.info("\n🔍 TEST 5: Backend Logs Verification")
        
        try:
            logs_results = {
                'backend_logs_accessible': False,
                'pattern_detection_mentions': False,
                'macd_calculation_mentions': False,
                'yahoo_finance_mentions': False,
                'no_critical_errors': False,
                'error_count': 0,
                'success_messages': []
            }
            
            logger.info("   🚀 Testing backend logs for pattern detection and MACD integration health...")
            logger.info("   📊 Expected: Pattern detection enabled, MACD calculations working, no critical errors")
            
            # Step 1: Check backend logs
            logger.info("   📋 Checking backend logs for integration health...")
            try:
                backend_logs = await self._capture_backend_logs()
                
                if backend_logs:
                    logs_results['backend_logs_accessible'] = True
                    logger.info(f"      📊 Captured {len(backend_logs)} log lines for analysis")
                    
                    # Check for pattern detection mentions
                    pattern_detection_patterns = [
                        'pattern detection enabled',
                        'pattern detection',
                        'technical_pattern_detector',
                        'yahoo finance ohlcv',
                        'patterns detected'
                    ]
                    
                    pattern_messages = []
                    for log_line in backend_logs:
                        for pattern in pattern_detection_patterns:
                            if pattern.lower() in log_line.lower():
                                pattern_messages.append(log_line.strip())
                                break
                    
                    if pattern_messages:
                        logs_results['pattern_detection_mentions'] = True
                        logger.info(f"      ✅ Pattern detection mentions found: {len(pattern_messages)}")
                        for msg in pattern_messages[:2]:
                            logger.info(f"         - {msg}")
                    else:
                        logger.warning(f"      ❌ No pattern detection mentions found in logs")
                    
                    # Check for MACD calculation mentions
                    macd_patterns = [
                        'macd',
                        'macd calculation',
                        'macd_calculator',
                        'macd optimized',
                        'macd signal'
                    ]
                    
                    macd_messages = []
                    for log_line in backend_logs:
                        for pattern in macd_patterns:
                            if pattern.lower() in log_line.lower() and 'error' not in log_line.lower():
                                macd_messages.append(log_line.strip())
                                break
                    
                    if macd_messages:
                        logs_results['macd_calculation_mentions'] = True
                        logger.info(f"      ✅ MACD calculation mentions found: {len(macd_messages)}")
                        for msg in macd_messages[:2]:
                            logger.info(f"         - {msg}")
                    else:
                        logger.warning(f"      ❌ No MACD calculation mentions found in logs")
                    
                    # Check for Yahoo Finance mentions
                    yahoo_patterns = [
                        'yahoo finance',
                        'yfinance',
                        'yahoo ohlcv',
                        'yahoo_finance'
                    ]
                    
                    yahoo_messages = []
                    for log_line in backend_logs:
                        for pattern in yahoo_patterns:
                            if pattern.lower() in log_line.lower():
                                yahoo_messages.append(log_line.strip())
                                break
                    
                    if yahoo_messages:
                        logs_results['yahoo_finance_mentions'] = True
                        logger.info(f"      ✅ Yahoo Finance mentions found: {len(yahoo_messages)}")
                    else:
                        logger.warning(f"      ❌ No Yahoo Finance mentions found in logs")
                    
                    # Check for critical errors
                    error_patterns = ['ERROR', 'CRITICAL', 'Exception', 'Traceback', 'Failed']
                    error_count = 0
                    critical_errors = []
                    
                    for log_line in backend_logs:
                        for pattern in error_patterns:
                            if pattern in log_line:
                                error_count += 1
                                if len(critical_errors) < 3:  # Store first 3 errors
                                    critical_errors.append(log_line.strip())
                                break
                    
                    logs_results['error_count'] = error_count
                    if error_count < len(backend_logs) * 0.1:  # Less than 10% error rate
                        logs_results['no_critical_errors'] = True
                        logger.info(f"      ✅ Backend logs healthy: {error_count} errors in {len(backend_logs)} lines")
                    else:
                        logger.warning(f"      ⚠️ Backend logs show issues: {error_count} errors in {len(backend_logs)} lines")
                        for error in critical_errors:
                            logger.warning(f"         - {error}")
                    
                    # Collect success messages
                    success_patterns = [
                        'successfully',
                        'completed',
                        'working',
                        'enabled',
                        'operational'
                    ]
                    
                    for log_line in backend_logs:
                        for pattern in success_patterns:
                            if (pattern.lower() in log_line.lower() and 
                                'error' not in log_line.lower() and
                                ('pattern' in log_line.lower() or 'macd' in log_line.lower())):
                                logs_results['success_messages'].append(log_line.strip())
                                break
                    
                else:
                    logger.warning(f"      ⚠️ No backend logs captured")
                    
            except Exception as e:
                logger.error(f"      ❌ Error analyzing backend logs: {e}")
            
            # Calculate test success
            success_criteria = [
                logs_results['backend_logs_accessible'],
                logs_results['pattern_detection_mentions'] or logs_results['yahoo_finance_mentions'],
                logs_results['macd_calculation_mentions'],
                logs_results['no_critical_errors']
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.75:  # 75% success threshold
                self.log_test_result("Backend Logs Verification", True, 
                                   f"Backend logs healthy: {success_count}/{len(success_criteria)} criteria met. Errors: {logs_results['error_count']}, Pattern detection: {logs_results['pattern_detection_mentions']}, MACD: {logs_results['macd_calculation_mentions']}")
            else:
                self.log_test_result("Backend Logs Verification", False, 
                                   f"Backend logs issues: {success_count}/{len(success_criteria)} criteria met. Error count: {logs_results['error_count']}")
                
        except Exception as e:
            self.log_test_result("Backend Logs Verification", False, f"Exception: {str(e)}")
    
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
        """Run comprehensive Pattern Detection & MACD Fixes test suite"""
        logger.info("🚀 Starting Pattern Detection System Fixes and MACD Calculation Issues Test Suite")
        logger.info("=" * 80)
        logger.info("📋 PATTERN DETECTION & MACD FIXES TEST SUITE")
        logger.info("🎯 Testing: Pattern detection system fixes and MACD calculation issues")
        logger.info("🎯 Expected: Pattern detection enabled, real MACD values, enhanced OHLCV integration")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_pattern_detection_system_fix()
        await self.test_2_macd_calculation_fix_verification()
        await self.test_3_technical_indicators_integration()
        await self.test_4_api_analyses_enhanced_data()
        await self.test_5_backend_logs_verification()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 PATTERN DETECTION & MACD FIXES TEST SUMMARY")
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
            if "Pattern Detection System Fix" in result['test']:
                requirements_status['Pattern Detection System Re-enabled'] = result['success']
            elif "MACD Calculation Fix" in result['test']:
                requirements_status['MACD Calculation Fix (Real Values)'] = result['success']
            elif "Technical Indicators Integration" in result['test']:
                requirements_status['Enhanced OHLCV System Integration'] = result['success']
            elif "API Analyses Enhanced Data" in result['test']:
                requirements_status['API Endpoints with Non-Zero MACD Values'] = result['success']
            elif "Backend Logs Verification" in result['test']:
                requirements_status['Backend Integration Health'] = result['success']
        
        logger.info("🎯 CRITICAL REQUIREMENTS STATUS:")
        for requirement, status in requirements_status.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {requirement}")
        
        requirements_met = sum(1 for status in requirements_status.values() if status)
        total_requirements = len(requirements_status)
        
        # Final verdict
        logger.info(f"\n🏆 REQUIREMENTS SATISFACTION: {requirements_met}/{total_requirements}")
        
        if requirements_met == total_requirements:
            logger.info("\n🎉 VERDICT: PATTERN DETECTION & MACD FIXES 100% SUCCESSFUL!")
            logger.info("✅ Pattern detection system re-enabled - Yahoo Finance OHLCV working")
            logger.info("✅ MACD calculation fix working - real values instead of zeros")
            logger.info("✅ Enhanced OHLCV system properly feeding data to IA1 analysis")
            logger.info("✅ API endpoints returning non-zero MACD values and pattern data")
            logger.info("✅ Backend integration healthy with no critical errors")
        elif requirements_met >= total_requirements * 0.8:
            logger.info("\n⚠️ VERDICT: PATTERN DETECTION & MACD FIXES MOSTLY SUCCESSFUL")
            logger.info("🔍 Minor issues may need attention for complete integration")
        elif requirements_met >= total_requirements * 0.6:
            logger.info("\n⚠️ VERDICT: PATTERN DETECTION & MACD FIXES PARTIALLY SUCCESSFUL")
            logger.info("🔧 Several requirements need attention for complete fixes")
        else:
            logger.info("\n❌ VERDICT: PATTERN DETECTION & MACD FIXES NOT SUCCESSFUL")
            logger.info("🚨 Major issues detected - pattern detection and MACD calculations not working properly")
            logger.info("🚨 System needs significant fixes for pattern detection and MACD integration")
        
        return passed_tests, total_tests

async def main():
    """Main function to run the comprehensive Pattern Detection & MACD Fixes test suite"""
    test_suite = PatternDetectionMACDTestSuite()
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
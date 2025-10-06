#!/usr/bin/env python3
"""
COMPREHENSIVE TALIB INDICATORS SYSTEM INTEGRATION TESTING SUITE

TESTING OBJECTIVE: Validate that the new TALib-based indicators system is fully integrated and working correctly in the trading bot backend.

INTEGRATION CONTEXT:
- Successfully integrated new TALib indicators system into server.py 
- Replaced previous indicator system with modular /app/core/indicators/talib_indicators.py
- Added automatic OHLCV column normalization for enhanced_ohlcv_fetcher compatibility
- All indicators now calculated using TA-Lib library for maximum accuracy

KEY TESTING AREAS:

1. TALIB SYSTEM HEALTH:
   - Test that TALib indicators load correctly at startup
   - Verify no import errors or module loading issues 
   - Check backend logs for successful TALib initialization

2. INDICATOR CALCULATIONS:
   - Test IA1 analysis with real market data to verify indicators work
   - Validate RSI, MACD, ADX, Bollinger Bands, Stochastic, MFI, VWAP calculations
   - Ensure no NaN, None, or default fallback values 
   - Verify regime detection and confluence grading functionality

3. DATA COMPATIBILITY:
   - Test that enhanced_ohlcv_fetcher data (capitalized columns) works correctly
   - Verify column normalization (Open,High,Low,Close,Volume → open,high,low,close,volume)
   - Check handling of various data formats and edge cases

4. API ENDPOINTS:
   - Test /api/run-ia1-cycle for TALib integration
   - Test /api/force-ia1-analysis (when opportunities available)
   - Validate that indicators appear in API responses correctly
   - Check /api/analyses endpoint for TALib-calculated values

5. PERFORMANCE & STABILITY:
   - Monitor CPU usage during indicator calculations
   - Test system stability with TALib calculations
   - Verify no memory leaks or performance degradation

CRITICAL SUCCESS CRITERIA:
- Backend starts successfully with TALib system loaded
- IA1 analysis produces real indicator values (not defaults)
- All major indicators (RSI, MACD, ADX, etc.) calculate correctly
- API endpoints return valid technical analysis data
- No errors in backend logs related to TALib integration
- System performance remains stable
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
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TALibIntegrationTestSuite:
    """Comprehensive test suite for TALib Indicators System Integration validation"""
    
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
        logger.info(f"Testing TALib Integration at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Test symbols for analysis (will be dynamically determined from available opportunities)
        self.test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']  # Preferred symbols from review request
        self.actual_test_symbols = []  # Will be populated from available opportunities
        
        # Expected TALib indicators to validate
        self.expected_indicators = [
            'rsi_14', 'macd_line', 'macd_signal', 'macd_histogram', 'adx', 'plus_di', 'minus_di',
            'bb_upper', 'bb_middle', 'bb_lower', 'stoch_k', 'stoch_d', 'mfi', 'atr', 'vwap',
            'ema_9', 'ema_21', 'sma_20', 'sma_50', 'volume_ratio'
        ]
        
        # TALib-specific validation criteria
        self.talib_validation_criteria = {
            'rsi_14': {'min': 0, 'max': 100, 'default': 50.0},
            'macd_line': {'min': -10, 'max': 10, 'default': 0.0},
            'macd_histogram': {'min': -5, 'max': 5, 'default': 0.0},
            'adx': {'min': 0, 'max': 100, 'default': 25.0},
            'bb_position': {'min': 0, 'max': 1, 'default': 0.5},
            'stoch_k': {'min': 0, 'max': 100, 'default': 50.0},
            'mfi': {'min': 0, 'max': 100, 'default': 50.0},
            'atr': {'min': 0, 'max': 1000, 'default': 0.02},
            'volume_ratio': {'min': 0, 'max': 50, 'default': 1.0}
        }
        
        # Database connection info
        self.mongo_url = "mongodb://localhost:27017"
        self.db_name = "myapp"
        
        # Performance monitoring
        self.initial_cpu_usage = None
        self.initial_memory_usage = None
        
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
    
    async def _capture_backend_logs(self, lines: int = 200):
        """Capture backend logs for analysis"""
        try:
            # Try to capture supervisor backend logs
            result = subprocess.run(
                ['tail', '-n', str(lines), '/var/log/supervisor/backend.out.log'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                return result.stdout.split('\n')
            else:
                # Try alternative log location
                result = subprocess.run(
                    ['tail', '-n', str(lines), '/var/log/supervisor/backend.err.log'],
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
    
    def _monitor_system_performance(self):
        """Monitor system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available / (1024**3),  # GB
                'memory_used': memory.used / (1024**3)  # GB
            }
        except Exception as e:
            logger.warning(f"Could not monitor system performance: {e}")
            return {}
    
    async def test_1_talib_system_health(self):
        """Test 1: TALib System Health - Verify TALib indicators load correctly at startup"""
        logger.info("\n🔍 TEST 1: TALib System Health")
        
        try:
            health_results = {
                'backend_responsive': False,
                'talib_import_success': False,
                'talib_initialization_logs': [],
                'import_error_logs': [],
                'system_performance': {},
                'backend_startup_time': 0.0
            }
            
            logger.info("   🚀 Testing TALib system health and initialization...")
            logger.info("   📊 Expected: Backend responsive, TALib imports successful, no initialization errors")
            
            # Monitor initial system performance
            health_results['system_performance'] = self._monitor_system_performance()
            self.initial_cpu_usage = health_results['system_performance'].get('cpu_percent', 0)
            self.initial_memory_usage = health_results['system_performance'].get('memory_percent', 0)
            
            logger.info(f"      📊 Initial system performance: CPU {self.initial_cpu_usage:.1f}%, Memory {self.initial_memory_usage:.1f}%")
            
            # Test backend responsiveness
            logger.info("   📞 Testing backend responsiveness...")
            
            try:
                start_time = time.time()
                response = requests.get(f"{self.api_url}/opportunities", timeout=30)
                response_time = time.time() - start_time
                health_results['backend_startup_time'] = response_time
                
                if response.status_code == 200:
                    health_results['backend_responsive'] = True
                    logger.info(f"      ✅ Backend responsive (response time: {response_time:.2f}s)")
                else:
                    logger.error(f"      ❌ Backend not responsive: HTTP {response.status_code}")
                    
            except Exception as e:
                logger.error(f"      ❌ Backend connection failed: {e}")
            
            # Capture backend logs to check for TALib initialization
            logger.info("   📋 Capturing backend logs to check TALib initialization...")
            
            try:
                backend_logs = await self._capture_backend_logs(300)  # More logs for startup analysis
                if backend_logs:
                    # Look for TALib-related logs
                    talib_logs = []
                    import_error_logs = []
                    initialization_logs = []
                    
                    for log_line in backend_logs:
                        log_lower = log_line.lower()
                        
                        # TALib import and initialization success
                        if any(pattern in log_lower for pattern in [
                            'talib', 'ta-lib', 'technical indicators', 'indicators initialized',
                            'talib_indicators', 'get_talib_indicators'
                        ]):
                            talib_logs.append(log_line.strip())
                            
                            if any(success_pattern in log_lower for success_pattern in [
                                'initialized', 'loaded', 'success', 'complete', 'ready'
                            ]):
                                initialization_logs.append(log_line.strip())
                        
                        # Import errors
                        if any(error_pattern in log_lower for error_pattern in [
                            'import error', 'modulenotfounderror', 'importerror', 'failed to import'
                        ]) and 'talib' in log_lower:
                            import_error_logs.append(log_line.strip())
                    
                    health_results['talib_initialization_logs'] = initialization_logs
                    health_results['import_error_logs'] = import_error_logs
                    
                    logger.info(f"      📊 Backend logs analysis:")
                    logger.info(f"         - TALib-related logs: {len(talib_logs)}")
                    logger.info(f"         - Initialization success logs: {len(initialization_logs)}")
                    logger.info(f"         - Import error logs: {len(import_error_logs)}")
                    
                    # Show sample logs
                    if initialization_logs:
                        health_results['talib_import_success'] = True
                        logger.info(f"      ✅ TALib initialization successful")
                        for log in initialization_logs[:3]:  # Show first 3
                            logger.info(f"         📋 {log}")
                    else:
                        logger.warning(f"      ⚠️ No TALib initialization success logs found")
                    
                    if import_error_logs:
                        logger.error(f"      ❌ TALib import errors detected:")
                        for error in import_error_logs[:2]:  # Show first 2
                            logger.error(f"         - {error}")
                    else:
                        logger.info(f"      ✅ No TALib import errors found")
                        
            except Exception as e:
                logger.warning(f"      ⚠️ Could not analyze backend logs: {e}")
            
            # Test direct TALib import (if possible)
            logger.info("   🔬 Testing direct TALib import availability...")
            
            try:
                # Try to import TALib directly to verify it's available
                import talib
                logger.info(f"      ✅ TALib direct import successful: version available")
                health_results['talib_import_success'] = True
            except ImportError as e:
                logger.error(f"      ❌ TALib direct import failed: {e}")
            except Exception as e:
                logger.warning(f"      ⚠️ TALib import test error: {e}")
            
            # Final system performance check
            final_performance = self._monitor_system_performance()
            cpu_change = final_performance.get('cpu_percent', 0) - self.initial_cpu_usage
            memory_change = final_performance.get('memory_percent', 0) - self.initial_memory_usage
            
            logger.info(f"      📊 Final system performance: CPU {final_performance.get('cpu_percent', 0):.1f}% ({cpu_change:+.1f}%), Memory {final_performance.get('memory_percent', 0):.1f}% ({memory_change:+.1f}%)")
            
            # Calculate test success based on review requirements
            success_criteria = [
                health_results['backend_responsive'],  # Backend is responsive
                health_results['talib_import_success'],  # TALib imports successfully
                len(health_results['import_error_logs']) == 0,  # No import errors
                health_results['backend_startup_time'] < 60,  # Reasonable startup time
                final_performance.get('cpu_percent', 100) < 90,  # CPU usage reasonable
                final_performance.get('memory_percent', 100) < 90  # Memory usage reasonable
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.83:  # 83% success threshold (5/6 criteria)
                self.log_test_result("TALib System Health", True, 
                                   f"TALib system health excellent: {success_count}/{len(success_criteria)} criteria met. Backend responsive: {health_results['backend_responsive']}, TALib import: {health_results['talib_import_success']}, No errors: {len(health_results['import_error_logs']) == 0}, Startup time: {health_results['backend_startup_time']:.2f}s")
            else:
                self.log_test_result("TALib System Health", False, 
                                   f"TALib system health issues: {success_count}/{len(success_criteria)} criteria met. May have import errors or performance issues")
                
        except Exception as e:
            self.log_test_result("TALib System Health", False, f"Exception: {str(e)}")

    async def test_2_indicator_calculations(self):
        """Test 2: Indicator Calculations - Test IA1 analysis with real market data to verify indicators work"""
        logger.info("\n🔍 TEST 2: Indicator Calculations")
        
        try:
            calculation_results = {
                'analyses_attempted': 0,
                'analyses_successful': 0,
                'indicators_calculated': {},
                'real_values_count': 0,
                'default_values_count': 0,
                'regime_detection_working': 0,
                'confluence_grading_working': 0,
                'successful_analyses': [],
                'indicator_validation_details': []
            }
            
            logger.info("   🚀 Testing TALib indicator calculations with real market data...")
            logger.info("   📊 Expected: Real indicator values (not defaults), regime detection, confluence grading")
            
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
                    
                    # Prefer symbols from review request
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
            
            # Test each symbol for TALib indicator calculations
            for symbol in self.actual_test_symbols:
                logger.info(f"\n   📞 Testing TALib indicator calculations for {symbol}...")
                calculation_results['analyses_attempted'] += 1
                
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
                        calculation_results['analyses_successful'] += 1
                        
                        logger.info(f"      ✅ {symbol} analysis successful (response time: {response_time:.2f}s)")
                        
                        # Extract IA1 analysis data
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        if not isinstance(ia1_analysis, dict):
                            ia1_analysis = {}
                        
                        # Validate TALib indicators
                        indicator_validation = {
                            'symbol': symbol,
                            'indicators_found': 0,
                            'real_values': 0,
                            'default_values': 0,
                            'out_of_range_values': 0,
                            'indicator_details': {}
                        }
                        
                        for indicator in self.expected_indicators:
                            value = ia1_analysis.get(indicator)
                            
                            if value is not None:
                                indicator_validation['indicators_found'] += 1
                                
                                # Check if it's a real value or default
                                if indicator in self.talib_validation_criteria:
                                    criteria = self.talib_validation_criteria[indicator]
                                    default_value = criteria['default']
                                    min_val = criteria['min']
                                    max_val = criteria['max']
                                    
                                    # Check if it's exactly the default value
                                    if abs(float(value) - default_value) < 0.001:
                                        indicator_validation['default_values'] += 1
                                        calculation_results['default_values_count'] += 1
                                        logger.warning(f"         ⚠️ {indicator} using default value: {value}")
                                    else:
                                        indicator_validation['real_values'] += 1
                                        calculation_results['real_values_count'] += 1
                                        logger.info(f"         ✅ {indicator} real value: {value}")
                                    
                                    # Check if value is in reasonable range
                                    if not (min_val <= float(value) <= max_val):
                                        indicator_validation['out_of_range_values'] += 1
                                        logger.warning(f"         ⚠️ {indicator} out of range: {value} (expected {min_val}-{max_val})")
                                    
                                    indicator_validation['indicator_details'][indicator] = {
                                        'value': value,
                                        'is_default': abs(float(value) - default_value) < 0.001,
                                        'in_range': min_val <= float(value) <= max_val
                                    }
                                else:
                                    # For indicators without specific criteria, just check it's not None
                                    indicator_validation['real_values'] += 1
                                    calculation_results['real_values_count'] += 1
                                    logger.info(f"         ✅ {indicator} calculated: {value}")
                                    
                                    indicator_validation['indicator_details'][indicator] = {
                                        'value': value,
                                        'is_default': False,
                                        'in_range': True
                                    }
                            else:
                                logger.warning(f"         ❌ {indicator} missing from analysis")
                        
                        # Check regime detection
                        regime = ia1_analysis.get('regime', 'UNKNOWN')
                        confidence = ia1_analysis.get('confidence', 0.0)
                        
                        if regime != 'UNKNOWN' and regime != 'CONSOLIDATION' and confidence > 0.5:
                            calculation_results['regime_detection_working'] += 1
                            logger.info(f"         ✅ Regime detection working: {regime} (confidence: {confidence:.2f})")
                        elif regime == 'CONSOLIDATION' and confidence >= 0.5:
                            calculation_results['regime_detection_working'] += 1
                            logger.info(f"         ✅ Regime detection working (consolidation): {regime} (confidence: {confidence:.2f})")
                        else:
                            logger.warning(f"         ⚠️ Regime detection may not be working: {regime} (confidence: {confidence:.2f})")
                        
                        # Check confluence grading
                        confluence_grade = ia1_analysis.get('confluence_grade', 'UNKNOWN')
                        confluence_score = ia1_analysis.get('confluence_score', 0)
                        
                        if confluence_grade in ['A++', 'A+', 'A', 'B+', 'B', 'C', 'D'] and confluence_score > 0:
                            calculation_results['confluence_grading_working'] += 1
                            logger.info(f"         ✅ Confluence grading working: {confluence_grade} (score: {confluence_score})")
                        else:
                            logger.warning(f"         ⚠️ Confluence grading may not be working: {confluence_grade} (score: {confluence_score})")
                        
                        # Store analysis details
                        calculation_results['successful_analyses'].append({
                            'symbol': symbol,
                            'response_time': response_time,
                            'indicators_found': indicator_validation['indicators_found'],
                            'real_values': indicator_validation['real_values'],
                            'default_values': indicator_validation['default_values'],
                            'regime': regime,
                            'confidence': confidence,
                            'confluence_grade': confluence_grade,
                            'confluence_score': confluence_score
                        })
                        
                        calculation_results['indicator_validation_details'].append(indicator_validation)
                        
                        # Update indicators calculated count
                        for indicator in self.expected_indicators:
                            if indicator in ia1_analysis:
                                if indicator not in calculation_results['indicators_calculated']:
                                    calculation_results['indicators_calculated'][indicator] = 0
                                calculation_results['indicators_calculated'][indicator] += 1
                        
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
            
            # Final analysis
            success_rate = calculation_results['analyses_successful'] / max(calculation_results['analyses_attempted'], 1)
            real_values_rate = calculation_results['real_values_count'] / max(calculation_results['real_values_count'] + calculation_results['default_values_count'], 1)
            regime_detection_rate = calculation_results['regime_detection_working'] / max(calculation_results['analyses_successful'], 1)
            confluence_rate = calculation_results['confluence_grading_working'] / max(calculation_results['analyses_successful'], 1)
            
            logger.info(f"\n   📊 INDICATOR CALCULATIONS RESULTS:")
            logger.info(f"      Analyses attempted: {calculation_results['analyses_attempted']}")
            logger.info(f"      Analyses successful: {calculation_results['analyses_successful']}")
            logger.info(f"      Success rate: {success_rate:.2f}")
            logger.info(f"      Real values count: {calculation_results['real_values_count']}")
            logger.info(f"      Default values count: {calculation_results['default_values_count']}")
            logger.info(f"      Real values rate: {real_values_rate:.2f}")
            logger.info(f"      Regime detection working: {calculation_results['regime_detection_working']} ({regime_detection_rate:.2f})")
            logger.info(f"      Confluence grading working: {calculation_results['confluence_grading_working']} ({confluence_rate:.2f})")
            logger.info(f"      Indicators calculated: {len(calculation_results['indicators_calculated'])}")
            
            # Show indicators calculated
            if calculation_results['indicators_calculated']:
                logger.info(f"      📊 Indicators successfully calculated:")
                for indicator, count in calculation_results['indicators_calculated'].items():
                    logger.info(f"         - {indicator}: {count}/{calculation_results['analyses_successful']}")
            
            # Calculate test success based on review requirements
            success_criteria = [
                calculation_results['analyses_successful'] >= 2,  # At least 2 successful analyses
                success_rate >= 0.67,  # At least 67% success rate
                calculation_results['real_values_count'] >= 10,  # At least 10 real indicator values
                real_values_rate >= 0.7,  # At least 70% real values (not defaults)
                calculation_results['regime_detection_working'] >= 2,  # At least 2 working regime detections
                calculation_results['confluence_grading_working'] >= 2,  # At least 2 working confluence gradings
                len(calculation_results['indicators_calculated']) >= 10  # At least 10 different indicators calculated
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.86:  # 86% success threshold (6/7 criteria)
                self.log_test_result("Indicator Calculations", True, 
                                   f"TALib indicator calculations successful: {success_count}/{len(success_criteria)} criteria met. Success rate: {success_rate:.2f}, Real values rate: {real_values_rate:.2f}, Regime detection: {regime_detection_rate:.2f}, Confluence: {confluence_rate:.2f}")
            else:
                self.log_test_result("Indicator Calculations", False, 
                                   f"TALib indicator calculations issues: {success_count}/{len(success_criteria)} criteria met. May be using default values or missing calculations")
                
        except Exception as e:
            self.log_test_result("Indicator Calculations", False, f"Exception: {str(e)}")

    async def test_3_data_compatibility(self):
        """Test 3: Data Compatibility - Test enhanced_ohlcv_fetcher data compatibility and column normalization"""
        logger.info("\n🔍 TEST 3: Data Compatibility")
        
        try:
            compatibility_results = {
                'backend_logs_captured': False,
                'column_normalization_logs': [],
                'ohlcv_data_logs': [],
                'enhanced_ohlcv_logs': [],
                'data_format_errors': [],
                'successful_normalizations': 0,
                'data_sources_working': [],
                'compatibility_score': 0.0
            }
            
            logger.info("   🚀 Testing data compatibility and column normalization...")
            logger.info("   📊 Expected: Enhanced OHLCV data works, column normalization successful, no data format errors")
            
            # Capture backend logs to check for data compatibility
            logger.info("   📋 Capturing backend logs to check data compatibility...")
            
            try:
                backend_logs = await self._capture_backend_logs(400)  # More logs for data analysis
                if backend_logs:
                    compatibility_results['backend_logs_captured'] = True
                    
                    # Look for data compatibility logs
                    column_normalization_logs = []
                    ohlcv_data_logs = []
                    enhanced_ohlcv_logs = []
                    data_format_errors = []
                    
                    for log_line in backend_logs:
                        log_lower = log_line.lower()
                        
                        # Column normalization
                        if any(pattern in log_lower for pattern in [
                            'normalize_data', 'normalized columns', 'column normalization',
                            'open,high,low,close,volume', 'ohlcv normalization'
                        ]):
                            column_normalization_logs.append(log_line.strip())
                        
                        # OHLCV data processing
                        if any(pattern in log_lower for pattern in [
                            'ohlcv data', 'enhanced_ohlcv', 'ohlcv fetcher', 'market data',
                            'bingx enhanced', 'kraken enhanced', 'yahoo finance'
                        ]):
                            ohlcv_data_logs.append(log_line.strip())
                        
                        # Enhanced OHLCV specific
                        if any(pattern in log_lower for pattern in [
                            'enhanced_ohlcv_fetcher', 'multi-source validation', 'enhanced provided',
                            'capitalized columns', 'open,high,low,close,volume'
                        ]):
                            enhanced_ohlcv_logs.append(log_line.strip())
                        
                        # Data format errors
                        if any(error_pattern in log_lower for error_pattern in [
                            'data format error', 'column missing', 'normalization failed',
                            'invalid ohlcv', 'data validation failed'
                        ]):
                            data_format_errors.append(log_line.strip())
                    
                    compatibility_results['column_normalization_logs'] = column_normalization_logs
                    compatibility_results['ohlcv_data_logs'] = ohlcv_data_logs
                    compatibility_results['enhanced_ohlcv_logs'] = enhanced_ohlcv_logs
                    compatibility_results['data_format_errors'] = data_format_errors
                    
                    logger.info(f"      📊 Backend logs analysis:")
                    logger.info(f"         - Column normalization logs: {len(column_normalization_logs)}")
                    logger.info(f"         - OHLCV data logs: {len(ohlcv_data_logs)}")
                    logger.info(f"         - Enhanced OHLCV logs: {len(enhanced_ohlcv_logs)}")
                    logger.info(f"         - Data format errors: {len(data_format_errors)}")
                    
                    # Show sample logs
                    if column_normalization_logs:
                        compatibility_results['successful_normalizations'] = len(column_normalization_logs)
                        logger.info(f"      ✅ Column normalization working")
                        for log in column_normalization_logs[:2]:  # Show first 2
                            logger.info(f"         📋 {log}")
                    else:
                        logger.warning(f"      ⚠️ No column normalization logs found")
                    
                    if enhanced_ohlcv_logs:
                        logger.info(f"      ✅ Enhanced OHLCV integration working")
                        for log in enhanced_ohlcv_logs[:2]:  # Show first 2
                            logger.info(f"         📋 {log}")
                    else:
                        logger.warning(f"      ⚠️ No enhanced OHLCV logs found")
                    
                    if data_format_errors:
                        logger.error(f"      ❌ Data format errors detected:")
                        for error in data_format_errors[:2]:  # Show first 2
                            logger.error(f"         - {error}")
                    else:
                        logger.info(f"      ✅ No data format errors found")
                    
                    # Check for data sources working
                    data_sources = []
                    for log_line in backend_logs:
                        if 'enhanced provided' in log_line.lower():
                            # Extract data source name
                            if 'bingx enhanced' in log_line.lower():
                                data_sources.append('BingX')
                            elif 'kraken enhanced' in log_line.lower():
                                data_sources.append('Kraken')
                            elif 'yahoo finance' in log_line.lower():
                                data_sources.append('Yahoo Finance')
                            elif 'cryptocompare enhanced' in log_line.lower():
                                data_sources.append('CryptoCompare')
                    
                    compatibility_results['data_sources_working'] = list(set(data_sources))
                    logger.info(f"      📊 Data sources working: {compatibility_results['data_sources_working']}")
                        
            except Exception as e:
                logger.warning(f"      ⚠️ Could not analyze backend logs: {e}")
            
            # Test API endpoints for data compatibility
            logger.info("   📞 Testing API endpoints for data compatibility...")
            
            try:
                # Test opportunities endpoint (should show enhanced OHLCV data)
                response = requests.get(f"{self.api_url}/opportunities", timeout=30)
                
                if response.status_code == 200:
                    opportunities = response.json()
                    if isinstance(opportunities, dict) and 'opportunities' in opportunities:
                        opportunities = opportunities['opportunities']
                    
                    if opportunities and len(opportunities) > 0:
                        logger.info(f"      ✅ Opportunities endpoint working: {len(opportunities)} opportunities")
                        
                        # Check data quality
                        sample_opp = opportunities[0]
                        required_fields = ['symbol', 'current_price', 'volume_24h', 'price_change_24h']
                        missing_fields = [field for field in required_fields if field not in sample_opp or sample_opp[field] is None]
                        
                        if not missing_fields:
                            logger.info(f"      ✅ Data quality good: all required fields present")
                            compatibility_results['compatibility_score'] += 0.3
                        else:
                            logger.warning(f"      ⚠️ Data quality issues: missing fields {missing_fields}")
                    else:
                        logger.warning(f"      ⚠️ No opportunities available for data testing")
                else:
                    logger.error(f"      ❌ Opportunities endpoint failed: HTTP {response.status_code}")
                    
            except Exception as e:
                logger.error(f"      ❌ API endpoint test failed: {e}")
            
            # Calculate compatibility score
            score_components = [
                len(compatibility_results['column_normalization_logs']) > 0,  # Column normalization working
                len(compatibility_results['enhanced_ohlcv_logs']) > 0,  # Enhanced OHLCV working
                len(compatibility_results['data_format_errors']) == 0,  # No data format errors
                len(compatibility_results['data_sources_working']) >= 2,  # At least 2 data sources working
                compatibility_results['backend_logs_captured']  # Backend logs captured successfully
            ]
            
            compatibility_results['compatibility_score'] = sum(score_components) / len(score_components)
            
            logger.info(f"\n   📊 DATA COMPATIBILITY RESULTS:")
            logger.info(f"      Backend logs captured: {compatibility_results['backend_logs_captured']}")
            logger.info(f"      Column normalization logs: {len(compatibility_results['column_normalization_logs'])}")
            logger.info(f"      Enhanced OHLCV logs: {len(compatibility_results['enhanced_ohlcv_logs'])}")
            logger.info(f"      Data format errors: {len(compatibility_results['data_format_errors'])}")
            logger.info(f"      Data sources working: {len(compatibility_results['data_sources_working'])}")
            logger.info(f"      Compatibility score: {compatibility_results['compatibility_score']:.2f}")
            
            # Calculate test success based on review requirements
            success_criteria = [
                compatibility_results['backend_logs_captured'],  # Backend logs captured
                len(compatibility_results['column_normalization_logs']) > 0,  # Column normalization working
                len(compatibility_results['data_format_errors']) == 0,  # No data format errors
                len(compatibility_results['data_sources_working']) >= 1,  # At least 1 data source working
                compatibility_results['compatibility_score'] >= 0.6  # Overall compatibility score good
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Data Compatibility", True, 
                                   f"Data compatibility excellent: {success_count}/{len(success_criteria)} criteria met. Compatibility score: {compatibility_results['compatibility_score']:.2f}, Data sources: {len(compatibility_results['data_sources_working'])}, No format errors: {len(compatibility_results['data_format_errors']) == 0}")
            else:
                self.log_test_result("Data Compatibility", False, 
                                   f"Data compatibility issues: {success_count}/{len(success_criteria)} criteria met. May have data format errors or missing normalization")
                
        except Exception as e:
            self.log_test_result("Data Compatibility", False, f"Exception: {str(e)}")

    async def test_4_api_endpoints(self):
        """Test 4: API Endpoints - Test API endpoints for TALib integration"""
        logger.info("\n🔍 TEST 4: API Endpoints")
        
        try:
            api_results = {
                'endpoints_tested': 0,
                'endpoints_successful': 0,
                'run_ia1_cycle_working': False,
                'force_ia1_analysis_working': False,
                'analyses_endpoint_working': False,
                'talib_values_in_responses': 0,
                'response_times': [],
                'endpoint_details': []
            }
            
            logger.info("   🚀 Testing API endpoints for TALib integration...")
            logger.info("   📊 Expected: All endpoints working, TALib values in responses, reasonable response times")
            
            # Test /api/run-ia1-cycle endpoint
            logger.info("   📞 Testing /api/run-ia1-cycle endpoint...")
            api_results['endpoints_tested'] += 1
            
            try:
                start_time = time.time()
                response = requests.post(f"{self.api_url}/run-ia1-cycle", timeout=120)
                response_time = time.time() - start_time
                api_results['response_times'].append(response_time)
                
                if response.status_code == 200:
                    cycle_data = response.json()
                    api_results['endpoints_successful'] += 1
                    api_results['run_ia1_cycle_working'] = True
                    
                    logger.info(f"      ✅ /api/run-ia1-cycle working (response time: {response_time:.2f}s)")
                    
                    # Check for TALib values in response
                    if isinstance(cycle_data, dict):
                        talib_indicators_found = 0
                        for indicator in self.expected_indicators:
                            if self._find_nested_key(cycle_data, indicator):
                                talib_indicators_found += 1
                        
                        if talib_indicators_found > 0:
                            api_results['talib_values_in_responses'] += talib_indicators_found
                            logger.info(f"         ✅ TALib indicators found in response: {talib_indicators_found}")
                        else:
                            logger.warning(f"         ⚠️ No TALib indicators found in response")
                    
                    api_results['endpoint_details'].append({
                        'endpoint': '/api/run-ia1-cycle',
                        'success': True,
                        'response_time': response_time,
                        'talib_indicators': talib_indicators_found
                    })
                    
                else:
                    logger.error(f"      ❌ /api/run-ia1-cycle failed: HTTP {response.status_code}")
                    if response.text:
                        logger.error(f"         Error: {response.text[:200]}")
                    
                    api_results['endpoint_details'].append({
                        'endpoint': '/api/run-ia1-cycle',
                        'success': False,
                        'response_time': response_time,
                        'error': response.text[:200] if response.text else 'Unknown error'
                    })
                    
            except Exception as e:
                logger.error(f"      ❌ /api/run-ia1-cycle exception: {e}")
                api_results['endpoint_details'].append({
                    'endpoint': '/api/run-ia1-cycle',
                    'success': False,
                    'error': str(e)
                })
            
            # Test /api/force-ia1-analysis endpoint
            logger.info("   📞 Testing /api/force-ia1-analysis endpoint...")
            api_results['endpoints_tested'] += 1
            
            # Get a test symbol
            test_symbol = self.actual_test_symbols[0] if self.actual_test_symbols else 'BTCUSDT'
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_url}/force-ia1-analysis",
                    json={"symbol": test_symbol},
                    timeout=120
                )
                response_time = time.time() - start_time
                api_results['response_times'].append(response_time)
                
                if response.status_code == 200:
                    analysis_data = response.json()
                    api_results['endpoints_successful'] += 1
                    api_results['force_ia1_analysis_working'] = True
                    
                    logger.info(f"      ✅ /api/force-ia1-analysis working (response time: {response_time:.2f}s)")
                    
                    # Check for TALib values in response
                    if isinstance(analysis_data, dict):
                        talib_indicators_found = 0
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        
                        for indicator in self.expected_indicators:
                            if indicator in ia1_analysis and ia1_analysis[indicator] is not None:
                                talib_indicators_found += 1
                        
                        if talib_indicators_found > 0:
                            api_results['talib_values_in_responses'] += talib_indicators_found
                            logger.info(f"         ✅ TALib indicators found in response: {talib_indicators_found}")
                        else:
                            logger.warning(f"         ⚠️ No TALib indicators found in response")
                    
                    api_results['endpoint_details'].append({
                        'endpoint': '/api/force-ia1-analysis',
                        'success': True,
                        'response_time': response_time,
                        'talib_indicators': talib_indicators_found,
                        'symbol': test_symbol
                    })
                    
                else:
                    logger.error(f"      ❌ /api/force-ia1-analysis failed: HTTP {response.status_code}")
                    if response.text:
                        logger.error(f"         Error: {response.text[:200]}")
                    
                    api_results['endpoint_details'].append({
                        'endpoint': '/api/force-ia1-analysis',
                        'success': False,
                        'response_time': response_time,
                        'error': response.text[:200] if response.text else 'Unknown error',
                        'symbol': test_symbol
                    })
                    
            except Exception as e:
                logger.error(f"      ❌ /api/force-ia1-analysis exception: {e}")
                api_results['endpoint_details'].append({
                    'endpoint': '/api/force-ia1-analysis',
                    'success': False,
                    'error': str(e),
                    'symbol': test_symbol
                })
            
            # Test /api/analyses endpoint
            logger.info("   📞 Testing /api/analyses endpoint...")
            api_results['endpoints_tested'] += 1
            
            try:
                start_time = time.time()
                response = requests.get(f"{self.api_url}/analyses", timeout=60)
                response_time = time.time() - start_time
                api_results['response_times'].append(response_time)
                
                if response.status_code == 200:
                    analyses_data = response.json()
                    api_results['endpoints_successful'] += 1
                    api_results['analyses_endpoint_working'] = True
                    
                    logger.info(f"      ✅ /api/analyses working (response time: {response_time:.2f}s)")
                    
                    # Check for TALib values in analyses
                    if isinstance(analyses_data, list) and len(analyses_data) > 0:
                        talib_indicators_found = 0
                        sample_analysis = analyses_data[0]
                        
                        for indicator in self.expected_indicators:
                            if indicator in sample_analysis and sample_analysis[indicator] is not None:
                                talib_indicators_found += 1
                        
                        if talib_indicators_found > 0:
                            api_results['talib_values_in_responses'] += talib_indicators_found
                            logger.info(f"         ✅ TALib indicators found in analyses: {talib_indicators_found}")
                            logger.info(f"         📊 Total analyses available: {len(analyses_data)}")
                        else:
                            logger.warning(f"         ⚠️ No TALib indicators found in analyses")
                    else:
                        logger.warning(f"         ⚠️ No analyses data available")
                    
                    api_results['endpoint_details'].append({
                        'endpoint': '/api/analyses',
                        'success': True,
                        'response_time': response_time,
                        'talib_indicators': talib_indicators_found,
                        'analyses_count': len(analyses_data) if isinstance(analyses_data, list) else 0
                    })
                    
                else:
                    logger.error(f"      ❌ /api/analyses failed: HTTP {response.status_code}")
                    if response.text:
                        logger.error(f"         Error: {response.text[:200]}")
                    
                    api_results['endpoint_details'].append({
                        'endpoint': '/api/analyses',
                        'success': False,
                        'response_time': response_time,
                        'error': response.text[:200] if response.text else 'Unknown error'
                    })
                    
            except Exception as e:
                logger.error(f"      ❌ /api/analyses exception: {e}")
                api_results['endpoint_details'].append({
                    'endpoint': '/api/analyses',
                    'success': False,
                    'error': str(e)
                })
            
            # Final analysis
            success_rate = api_results['endpoints_successful'] / max(api_results['endpoints_tested'], 1)
            avg_response_time = sum(api_results['response_times']) / max(len(api_results['response_times']), 1)
            
            logger.info(f"\n   📊 API ENDPOINTS RESULTS:")
            logger.info(f"      Endpoints tested: {api_results['endpoints_tested']}")
            logger.info(f"      Endpoints successful: {api_results['endpoints_successful']}")
            logger.info(f"      Success rate: {success_rate:.2f}")
            logger.info(f"      Average response time: {avg_response_time:.2f}s")
            logger.info(f"      TALib values in responses: {api_results['talib_values_in_responses']}")
            logger.info(f"      run-ia1-cycle working: {api_results['run_ia1_cycle_working']}")
            logger.info(f"      force-ia1-analysis working: {api_results['force_ia1_analysis_working']}")
            logger.info(f"      analyses endpoint working: {api_results['analyses_endpoint_working']}")
            
            # Show endpoint details
            if api_results['endpoint_details']:
                logger.info(f"      📊 Endpoint Details:")
                for detail in api_results['endpoint_details']:
                    if detail['success']:
                        logger.info(f"         ✅ {detail['endpoint']}: {detail['response_time']:.2f}s, TALib indicators: {detail.get('talib_indicators', 0)}")
                    else:
                        logger.info(f"         ❌ {detail['endpoint']}: {detail.get('error', 'Unknown error')}")
            
            # Calculate test success based on review requirements
            success_criteria = [
                api_results['endpoints_successful'] >= 2,  # At least 2 endpoints working
                success_rate >= 0.67,  # At least 67% success rate
                api_results['force_ia1_analysis_working'],  # force-ia1-analysis working (critical)
                api_results['talib_values_in_responses'] >= 10,  # At least 10 TALib values in responses
                avg_response_time < 60,  # Reasonable response times
                api_results['analyses_endpoint_working']  # analyses endpoint working
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.83:  # 83% success threshold (5/6 criteria)
                self.log_test_result("API Endpoints", True, 
                                   f"API endpoints excellent: {success_count}/{len(success_criteria)} criteria met. Success rate: {success_rate:.2f}, TALib values: {api_results['talib_values_in_responses']}, Avg response time: {avg_response_time:.2f}s")
            else:
                self.log_test_result("API Endpoints", False, 
                                   f"API endpoints issues: {success_count}/{len(success_criteria)} criteria met. May have endpoint failures or missing TALib values")
                
        except Exception as e:
            self.log_test_result("API Endpoints", False, f"Exception: {str(e)}")
    
    def _find_nested_key(self, data: dict, key: str) -> bool:
        """Recursively find a key in nested dictionary"""
        if isinstance(data, dict):
            if key in data:
                return True
            for value in data.values():
                if self._find_nested_key(value, key):
                    return True
        elif isinstance(data, list):
            for item in data:
                if self._find_nested_key(item, key):
                    return True
        return False

    async def test_5_performance_stability(self):
        """Test 5: Performance & Stability - Monitor system performance and stability"""
        logger.info("\n🔍 TEST 5: Performance & Stability")
        
        try:
            performance_results = {
                'initial_performance': {},
                'final_performance': {},
                'cpu_usage_stable': False,
                'memory_usage_stable': False,
                'no_memory_leaks': False,
                'system_stable': False,
                'performance_degradation': 0.0,
                'backend_errors': 0,
                'stability_score': 0.0
            }
            
            logger.info("   🚀 Testing system performance and stability...")
            logger.info("   📊 Expected: Stable CPU/memory usage, no memory leaks, system stability")
            
            # Get initial performance metrics
            performance_results['initial_performance'] = self._monitor_system_performance()
            initial_cpu = performance_results['initial_performance'].get('cpu_percent', 0)
            initial_memory = performance_results['initial_performance'].get('memory_percent', 0)
            
            logger.info(f"      📊 Initial performance: CPU {initial_cpu:.1f}%, Memory {initial_memory:.1f}%")
            
            # Run some load tests to check stability
            logger.info("   🔄 Running load tests to check system stability...")
            
            load_test_results = []
            for i in range(3):
                logger.info(f"      📞 Load test {i+1}/3...")
                
                try:
                    # Test multiple endpoints quickly
                    start_time = time.time()
                    
                    # Test opportunities endpoint
                    response1 = requests.get(f"{self.api_url}/opportunities", timeout=30)
                    
                    # Test analyses endpoint
                    response2 = requests.get(f"{self.api_url}/analyses", timeout=30)
                    
                    total_time = time.time() - start_time
                    
                    # Check performance during load
                    load_performance = self._monitor_system_performance()
                    
                    load_test_results.append({
                        'test_number': i+1,
                        'total_time': total_time,
                        'opportunities_status': response1.status_code,
                        'analyses_status': response2.status_code,
                        'cpu_percent': load_performance.get('cpu_percent', 0),
                        'memory_percent': load_performance.get('memory_percent', 0)
                    })
                    
                    logger.info(f"         ✅ Load test {i+1} completed: {total_time:.2f}s, CPU {load_performance.get('cpu_percent', 0):.1f}%, Memory {load_performance.get('memory_percent', 0):.1f}%")
                    
                except Exception as e:
                    logger.error(f"         ❌ Load test {i+1} failed: {e}")
                    load_test_results.append({
                        'test_number': i+1,
                        'error': str(e)
                    })
                
                # Wait between load tests
                if i < 2:
                    await asyncio.sleep(5)
            
            # Get final performance metrics
            await asyncio.sleep(10)  # Wait for system to stabilize
            performance_results['final_performance'] = self._monitor_system_performance()
            final_cpu = performance_results['final_performance'].get('cpu_percent', 0)
            final_memory = performance_results['final_performance'].get('memory_percent', 0)
            
            logger.info(f"      📊 Final performance: CPU {final_cpu:.1f}%, Memory {final_memory:.1f}%")
            
            # Analyze performance stability
            cpu_change = final_cpu - initial_cpu
            memory_change = final_memory - initial_memory
            
            # CPU stability check
            if abs(cpu_change) < 20:  # Less than 20% CPU change
                performance_results['cpu_usage_stable'] = True
                logger.info(f"      ✅ CPU usage stable: {cpu_change:+.1f}% change")
            else:
                logger.warning(f"      ⚠️ CPU usage unstable: {cpu_change:+.1f}% change")
            
            # Memory stability check
            if abs(memory_change) < 10:  # Less than 10% memory change
                performance_results['memory_usage_stable'] = True
                logger.info(f"      ✅ Memory usage stable: {memory_change:+.1f}% change")
            else:
                logger.warning(f"      ⚠️ Memory usage unstable: {memory_change:+.1f}% change")
            
            # Memory leak detection
            if memory_change < 5:  # Memory increase less than 5%
                performance_results['no_memory_leaks'] = True
                logger.info(f"      ✅ No memory leaks detected")
            else:
                logger.warning(f"      ⚠️ Potential memory leak: {memory_change:+.1f}% memory increase")
            
            # Overall performance degradation
            performance_results['performance_degradation'] = max(abs(cpu_change), abs(memory_change))
            
            # Check backend logs for errors
            logger.info("   📋 Checking backend logs for stability issues...")
            
            try:
                backend_logs = await self._capture_backend_logs(200)
                if backend_logs:
                    error_count = 0
                    stability_issues = []
                    
                    for log_line in backend_logs:
                        log_lower = log_line.lower()
                        
                        # Count errors
                        if any(error_pattern in log_lower for error_pattern in [
                            'error', 'exception', 'failed', 'crash', 'timeout'
                        ]):
                            error_count += 1
                            if len(stability_issues) < 5:  # Keep first 5 errors
                                stability_issues.append(log_line.strip())
                    
                    performance_results['backend_errors'] = error_count
                    
                    if error_count == 0:
                        logger.info(f"      ✅ No backend errors found")
                    elif error_count < 5:
                        logger.warning(f"      ⚠️ Few backend errors found: {error_count}")
                        for issue in stability_issues:
                            logger.warning(f"         - {issue}")
                    else:
                        logger.error(f"      ❌ Many backend errors found: {error_count}")
                        for issue in stability_issues[:3]:  # Show first 3
                            logger.error(f"         - {issue}")
                        
            except Exception as e:
                logger.warning(f"      ⚠️ Could not check backend logs: {e}")
            
            # System stability assessment
            stability_factors = [
                performance_results['cpu_usage_stable'],
                performance_results['memory_usage_stable'],
                performance_results['no_memory_leaks'],
                performance_results['backend_errors'] < 10,  # Less than 10 errors
                performance_results['performance_degradation'] < 15  # Less than 15% degradation
            ]
            
            performance_results['system_stable'] = sum(stability_factors) >= 4  # At least 4/5 factors good
            performance_results['stability_score'] = sum(stability_factors) / len(stability_factors)
            
            logger.info(f"\n   📊 PERFORMANCE & STABILITY RESULTS:")
            logger.info(f"      CPU usage stable: {performance_results['cpu_usage_stable']}")
            logger.info(f"      Memory usage stable: {performance_results['memory_usage_stable']}")
            logger.info(f"      No memory leaks: {performance_results['no_memory_leaks']}")
            logger.info(f"      Backend errors: {performance_results['backend_errors']}")
            logger.info(f"      Performance degradation: {performance_results['performance_degradation']:.1f}%")
            logger.info(f"      System stable: {performance_results['system_stable']}")
            logger.info(f"      Stability score: {performance_results['stability_score']:.2f}")
            
            # Show load test results
            if load_test_results:
                logger.info(f"      📊 Load Test Results:")
                for result in load_test_results:
                    if 'error' not in result:
                        logger.info(f"         - Test {result['test_number']}: {result['total_time']:.2f}s, CPU {result['cpu_percent']:.1f}%, Memory {result['memory_percent']:.1f}%")
                    else:
                        logger.info(f"         - Test {result['test_number']}: Error - {result['error']}")
            
            # Calculate test success based on review requirements
            success_criteria = [
                performance_results['cpu_usage_stable'],  # CPU usage stable
                performance_results['memory_usage_stable'],  # Memory usage stable
                performance_results['no_memory_leaks'],  # No memory leaks
                performance_results['backend_errors'] < 5,  # Few backend errors
                performance_results['system_stable'],  # Overall system stable
                performance_results['stability_score'] >= 0.8  # High stability score
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.83:  # 83% success threshold (5/6 criteria)
                self.log_test_result("Performance & Stability", True, 
                                   f"Performance & stability excellent: {success_count}/{len(success_criteria)} criteria met. Stability score: {performance_results['stability_score']:.2f}, Performance degradation: {performance_results['performance_degradation']:.1f}%, Backend errors: {performance_results['backend_errors']}")
            else:
                self.log_test_result("Performance & Stability", False, 
                                   f"Performance & stability issues: {success_count}/{len(success_criteria)} criteria met. May have performance degradation or stability issues")
                
        except Exception as e:
            self.log_test_result("Performance & Stability", False, f"Exception: {str(e)}")

    async def run_all_tests(self):
        """Run all TALib integration tests"""
        logger.info("🚀 STARTING COMPREHENSIVE TALIB INDICATORS SYSTEM INTEGRATION TESTING")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Run all tests in sequence
        await self.test_1_talib_system_health()
        await self.test_2_indicator_calculations()
        await self.test_3_data_compatibility()
        await self.test_4_api_endpoints()
        await self.test_5_performance_stability()
        
        total_time = time.time() - start_time
        
        # Generate final summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 TALIB INTEGRATION TESTING SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        logger.info(f"Total tests run: {total_tests}")
        logger.info(f"Tests passed: {passed_tests}")
        logger.info(f"Tests failed: {total_tests - passed_tests}")
        logger.info(f"Success rate: {success_rate:.2%}")
        logger.info(f"Total testing time: {total_time:.2f} seconds")
        
        logger.info("\n📋 DETAILED TEST RESULTS:")
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
        
        # Overall assessment
        if success_rate >= 0.8:
            logger.info(f"\n🎉 OVERALL ASSESSMENT: TALib Integration SUCCESSFUL")
            logger.info(f"   The TALib indicators system is fully integrated and working correctly.")
        elif success_rate >= 0.6:
            logger.info(f"\n⚠️ OVERALL ASSESSMENT: TALib Integration PARTIALLY SUCCESSFUL")
            logger.info(f"   Most TALib functionality is working, but some issues need attention.")
        else:
            logger.info(f"\n❌ OVERALL ASSESSMENT: TALib Integration NEEDS WORK")
            logger.info(f"   Significant issues found that need to be addressed.")
        
        return success_rate

async def main():
    """Main test execution"""
    test_suite = TALibIntegrationTestSuite()
    success_rate = await test_suite.run_all_tests()
    
    # Exit with appropriate code
    if success_rate >= 0.8:
        exit(0)  # Success
    else:
        exit(1)  # Failure

if __name__ == "__main__":
    asyncio.run(main())
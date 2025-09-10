#!/usr/bin/env python3
"""
Trading System Startup Test Suite
Focus: Complete system startup, mathematical error validation, and trading functionality
Based on review request: Test le démarrage complet du système de trading après correction des erreurs mathématiques
"""

import asyncio
import json
import logging
import os
import sys
import time
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional
import re
import math

# Add backend to path
sys.path.append('/app/backend')

import requests
import websockets
from motor.motor_asyncio import AsyncIOMotorClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingSystemStartupTestSuite:
    """Test suite for Trading System Startup and Mathematical Error Validation"""
    
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
        self.ws_url = backend_url.replace('http', 'ws')
        logger.info(f"Testing backend at: {self.api_url}")
        logger.info(f"WebSocket URL: {self.ws_url}")
        
        # MongoDB connection for direct data access
        self.mongo_client = None
        self.db = None
        
        # Test results
        self.test_results = []
        
        # Trading system state
        self.trading_started = False
        
    async def setup_database(self):
        """Setup database connection"""
        try:
            # Get MongoDB URL from backend env
            mongo_url = "mongodb://localhost:27017"  # Default
            try:
                with open('/app/backend/.env', 'r') as f:
                    for line in f:
                        if line.startswith('MONGO_URL='):
                            mongo_url = line.split('=')[1].strip().strip('"')
                            break
            except Exception:
                pass
            
            self.mongo_client = AsyncIOMotorClient(mongo_url)
            self.db = self.mongo_client['myapp']
            logger.info("✅ Database connection established")
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            
    async def cleanup_database(self):
        """Cleanup database connection"""
        if self.mongo_client:
            self.mongo_client.close()
            
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
        
    def get_backend_logs(self, lines: int = 500) -> str:
        """Get recent backend logs"""
        try:
            log_cmd = f"tail -n {lines} /var/log/supervisor/backend.*.log 2>/dev/null || echo 'No logs found'"
            result = subprocess.run(log_cmd, shell=True, capture_output=True, text=True)
            return result.stdout
        except Exception as e:
            logger.error(f"Failed to get backend logs: {e}")
            return ""
            
    def check_mathematical_errors(self, logs: str) -> Dict[str, Any]:
        """Check for mathematical errors in logs"""
        errors = {
            'runtime_warnings': [],
            'divide_by_zero': [],
            'nan_values': [],
            'inf_values': [],
            'math_errors': []
        }
        
        # Patterns to look for
        patterns = {
            'runtime_warnings': [r'RuntimeWarning', r'Warning.*math', r'Warning.*divide'],
            'divide_by_zero': [r'divide by zero', r'division by zero', r'ZeroDivisionError'],
            'nan_values': [r'nan', r'NaN', r'not a number'],
            'inf_values': [r'inf', r'infinity', r'Infinity'],
            'math_errors': [r'ValueError.*math', r'OverflowError', r'ArithmeticError']
        }
        
        for line in logs.split('\n'):
            for error_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    if re.search(pattern, line, re.IGNORECASE):
                        errors[error_type].append(line.strip())
                        
        return errors
        
    async def test_trading_system_startup(self):
        """Test 1: Trading System Startup - POST /api/start-trading"""
        logger.info("\n🔍 TEST 1: Trading System Startup")
        
        try:
            # First check if trading is already running
            status_response = requests.get(f"{self.api_url}/market-status", timeout=30)
            if status_response.status_code == 200:
                status_data = status_response.json()
                logger.info(f"   📊 Current system status: {status_data}")
                
            # Start trading system
            logger.info("   🚀 Starting trading system...")
            start_response = requests.post(f"{self.api_url}/start-trading", timeout=60)
            
            if start_response.status_code == 200:
                start_data = start_response.json()
                logger.info(f"   ✅ Trading system started: {start_data}")
                self.trading_started = True
                
                # Wait a moment for system to initialize
                await asyncio.sleep(5)
                
                # Verify system is running
                verify_response = requests.get(f"{self.api_url}/market-status", timeout=30)
                if verify_response.status_code == 200:
                    verify_data = verify_response.json()
                    is_running = verify_data.get('trading_active', False)
                    
                    success = is_running
                    details = f"Start response: {start_response.status_code}, Trading active: {is_running}"
                    
                else:
                    success = False
                    details = f"Start response: {start_response.status_code}, Verification failed: {verify_response.status_code}"
                    
            else:
                success = False
                details = f"Start trading failed: {start_response.status_code} - {start_response.text}"
                
            self.log_test_result("Trading System Startup", success, details)
            
        except Exception as e:
            self.log_test_result("Trading System Startup", False, f"Exception: {str(e)}")
            
    async def test_mathematical_error_absence(self):
        """Test 2: Absence of Mathematical Errors - Check logs for RuntimeWarning, divide by zero"""
        logger.info("\n🔍 TEST 2: Mathematical Error Absence")
        
        try:
            # Get recent backend logs
            logs = self.get_backend_logs(1000)
            
            if not logs or logs.strip() == 'No logs found':
                self.log_test_result("Mathematical Error Absence", False, "No backend logs available")
                return
                
            # Check for mathematical errors
            errors = self.check_mathematical_errors(logs)
            
            # Count total errors
            total_errors = sum(len(error_list) for error_list in errors.values())
            
            # Log findings
            logger.info(f"   📊 Mathematical error analysis:")
            for error_type, error_list in errors.items():
                if error_list:
                    logger.info(f"      ❌ {error_type}: {len(error_list)} occurrences")
                    # Show first few examples
                    for i, error in enumerate(error_list[:3]):
                        logger.info(f"         {i+1}. {error[:100]}...")
                else:
                    logger.info(f"      ✅ {error_type}: No errors found")
                    
            # Check specifically for pattern-related mathematical operations
            pattern_math_logs = []
            macd_calculation_logs = []
            rsi_calculation_logs = []
            
            for line in logs.split('\n'):
                if 'MACD' in line and any(keyword in line.lower() for keyword in ['calculate', 'rsi', 'macd']):
                    macd_calculation_logs.append(line.strip())
                elif 'RSI' in line and 'calculate' in line.lower():
                    rsi_calculation_logs.append(line.strip())
                elif 'pattern' in line.lower() and any(keyword in line.lower() for keyword in ['calculate', 'math', 'error']):
                    pattern_math_logs.append(line.strip())
                    
            logger.info(f"   📊 Pattern math operations: {len(pattern_math_logs)}")
            logger.info(f"   📊 MACD calculations: {len(macd_calculation_logs)}")
            logger.info(f"   📊 RSI calculations: {len(rsi_calculation_logs)}")
            
            # Success if no critical mathematical errors
            critical_errors = errors['runtime_warnings'] + errors['divide_by_zero'] + errors['math_errors']
            success = len(critical_errors) == 0
            
            details = f"Total errors: {total_errors}, Critical errors: {len(critical_errors)}, Pattern math ops: {len(pattern_math_logs)}"
            
            self.log_test_result("Mathematical Error Absence", success, details)
            
        except Exception as e:
            self.log_test_result("Mathematical Error Absence", False, f"Exception: {str(e)}")
            
    async def test_pattern_functionality(self):
        """Test 3: Pattern Functionality - Verify pattern detection and IA1 analyses work without errors"""
        logger.info("\n🔍 TEST 3: Pattern Functionality")
        
        try:
            # Wait for system to generate some analyses
            logger.info("   ⏳ Waiting for pattern detection and IA1 analyses...")
            await asyncio.sleep(30)  # Give system time to work
            
            # Check IA1 analyses
            analyses_response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if analyses_response.status_code != 200:
                self.log_test_result("Pattern Functionality", False, f"Analyses API failed: {analyses_response.status_code}")
                return
                
            analyses_data = analyses_response.json()
            analyses = analyses_data if isinstance(analyses_data, list) else analyses_data.get('analyses', [])
            
            logger.info(f"   📊 Found {len(analyses)} IA1 analyses")
            
            # Check pattern detection
            patterns_detected = 0
            analyses_with_patterns = 0
            pattern_types = set()
            
            for analysis in analyses:
                symbol = analysis.get('symbol', 'UNKNOWN')
                patterns = analysis.get('patterns_detected', [])
                confidence = analysis.get('analysis_confidence', 0)
                reasoning = analysis.get('ia1_reasoning', '')
                
                if patterns:
                    analyses_with_patterns += 1
                    patterns_detected += len(patterns)
                    pattern_types.update(patterns)
                    
                    logger.info(f"      🎯 {symbol}: {len(patterns)} patterns, confidence: {confidence:.2f}")
                    logger.info(f"         Patterns: {patterns}")
                    
                # Check for mathematical indicators in reasoning
                has_rsi = 'rsi' in reasoning.lower()
                has_macd = 'macd' in reasoning.lower()
                has_technical = any(term in reasoning.lower() for term in ['support', 'resistance', 'bollinger'])
                
                if has_rsi or has_macd or has_technical:
                    logger.info(f"      ✅ {symbol}: Technical indicators present (RSI: {has_rsi}, MACD: {has_macd}, Technical: {has_technical})")
                    
            # Check IA2 decisions
            decisions_response = requests.get(f"{self.api_url}/decisions", timeout=30)
            ia2_decisions = 0
            
            if decisions_response.status_code == 200:
                decisions_data = decisions_response.json()
                decisions = decisions_data if isinstance(decisions_data, list) else decisions_data.get('decisions', [])
                ia2_decisions = len(decisions)
                logger.info(f"   📊 Found {ia2_decisions} IA2 decisions")
                
            # Check backend logs for pattern processing
            logs = self.get_backend_logs(500)
            pattern_processing_logs = 0
            ia1_generation_logs = 0
            
            for line in logs.split('\n'):
                if 'pattern' in line.lower() and any(keyword in line.lower() for keyword in ['detect', 'found', 'analysis']):
                    pattern_processing_logs += 1
                elif 'ia1' in line.lower() and any(keyword in line.lower() for keyword in ['analysis', 'generate', 'complete']):
                    ia1_generation_logs += 1
                    
            logger.info(f"   📊 Pattern processing logs: {pattern_processing_logs}")
            logger.info(f"   📊 IA1 generation logs: {ia1_generation_logs}")
            
            # Success criteria
            success = (
                len(analyses) > 0 and  # IA1 analyses generated
                analyses_with_patterns > 0 and  # Some patterns detected
                len(pattern_types) > 0 and  # Pattern variety
                pattern_processing_logs > 0  # System processing patterns
            )
            
            details = f"IA1 analyses: {len(analyses)}, Patterns detected: {patterns_detected}, Pattern types: {len(pattern_types)}, IA2 decisions: {ia2_decisions}"
            
            self.log_test_result("Pattern Functionality", success, details)
            
        except Exception as e:
            self.log_test_result("Pattern Functionality", False, f"Exception: {str(e)}")
            
    async def test_background_trading_loop(self):
        """Test 4: Background Trading Loop - Verify continuous operation"""
        logger.info("\n🔍 TEST 4: Background Trading Loop")
        
        try:
            # Monitor system for continuous operation
            logger.info("   ⏳ Monitoring background trading loop for 60 seconds...")
            
            initial_time = time.time()
            monitoring_duration = 60  # seconds
            
            # Get initial counts
            initial_response = requests.get(f"{self.api_url}/market-status", timeout=30)
            initial_data = initial_response.json() if initial_response.status_code == 200 else {}
            
            # Monitor for activity
            activity_checks = []
            
            for i in range(6):  # Check every 10 seconds
                await asyncio.sleep(10)
                
                # Check system status
                status_response = requests.get(f"{self.api_url}/market-status", timeout=30)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    activity_checks.append({
                        'timestamp': time.time(),
                        'trading_active': status_data.get('trading_active', False),
                        'opportunities': len(requests.get(f"{self.api_url}/opportunities", timeout=30).json() or []),
                        'analyses': len(requests.get(f"{self.api_url}/analyses", timeout=30).json() or [])
                    })
                    
                    logger.info(f"      Check {i+1}/6: Trading active: {status_data.get('trading_active', False)}")
                    
            # Check logs for continuous activity
            logs = self.get_backend_logs(300)
            recent_activity_logs = 0
            
            current_time = time.time()
            for line in logs.split('\n'):
                if any(keyword in line.lower() for keyword in ['scout', 'ia1', 'ia2', 'analysis', 'opportunity']):
                    recent_activity_logs += 1
                    
            logger.info(f"   📊 Recent activity logs: {recent_activity_logs}")
            
            # Check for system consistency
            trading_active_count = sum(1 for check in activity_checks if check['trading_active'])
            opportunities_generated = any(check['opportunities'] > 0 for check in activity_checks)
            analyses_generated = any(check['analyses'] > 0 for check in activity_checks)
            
            logger.info(f"   📊 Trading active checks: {trading_active_count}/6")
            logger.info(f"   📊 Opportunities generated: {opportunities_generated}")
            logger.info(f"   📊 Analyses generated: {analyses_generated}")
            
            # Success criteria
            success = (
                trading_active_count >= 4 and  # Trading mostly active
                recent_activity_logs > 10 and  # System activity in logs
                (opportunities_generated or analyses_generated)  # Some data generation
            )
            
            details = f"Active checks: {trading_active_count}/6, Activity logs: {recent_activity_logs}, Data generation: {opportunities_generated or analyses_generated}"
            
            self.log_test_result("Background Trading Loop", success, details)
            
        except Exception as e:
            self.log_test_result("Background Trading Loop", False, f"Exception: {str(e)}")
            
    async def test_system_performance(self):
        """Test 5: System Performance - Response times, memory stability, no crashes"""
        logger.info("\n🔍 TEST 5: System Performance")
        
        try:
            # Test API response times
            endpoints = [
                '/market-status',
                '/opportunities', 
                '/analyses',
                '/decisions'
            ]
            
            response_times = {}
            
            for endpoint in endpoints:
                start_time = time.time()
                try:
                    response = requests.get(f"{self.api_url}{endpoint}", timeout=30)
                    end_time = time.time()
                    
                    response_times[endpoint] = {
                        'time': end_time - start_time,
                        'status': response.status_code,
                        'success': response.status_code == 200
                    }
                    
                    logger.info(f"   📊 {endpoint}: {response_times[endpoint]['time']:.2f}s - {response_times[endpoint]['status']}")
                    
                except Exception as e:
                    response_times[endpoint] = {
                        'time': 30.0,  # timeout
                        'status': 0,
                        'success': False,
                        'error': str(e)
                    }
                    logger.info(f"   ❌ {endpoint}: Error - {str(e)}")
                    
            # Check for system crashes in logs
            logs = self.get_backend_logs(500)
            crash_indicators = []
            
            crash_patterns = [
                r'crash', r'fatal', r'critical error', r'segmentation fault',
                r'memory error', r'out of memory', r'killed', r'terminated unexpectedly'
            ]
            
            for line in logs.split('\n'):
                for pattern in crash_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        crash_indicators.append(line.strip())
                        
            logger.info(f"   📊 Crash indicators found: {len(crash_indicators)}")
            
            # Check memory usage patterns
            memory_logs = []
            for line in logs.split('\n'):
                if any(keyword in line.lower() for keyword in ['memory', 'ram', 'heap', 'gc']):
                    memory_logs.append(line.strip())
                    
            logger.info(f"   📊 Memory-related logs: {len(memory_logs)}")
            
            # Calculate performance metrics
            avg_response_time = sum(rt['time'] for rt in response_times.values()) / len(response_times)
            successful_endpoints = sum(1 for rt in response_times.values() if rt['success'])
            
            logger.info(f"   📊 Average response time: {avg_response_time:.2f}s")
            logger.info(f"   📊 Successful endpoints: {successful_endpoints}/{len(endpoints)}")
            
            # Success criteria
            success = (
                avg_response_time < 10.0 and  # Reasonable response times
                successful_endpoints >= len(endpoints) * 0.8 and  # Most endpoints working
                len(crash_indicators) == 0  # No crashes
            )
            
            details = f"Avg response: {avg_response_time:.2f}s, Working endpoints: {successful_endpoints}/{len(endpoints)}, Crashes: {len(crash_indicators)}"
            
            self.log_test_result("System Performance", success, details)
            
        except Exception as e:
            self.log_test_result("System Performance", False, f"Exception: {str(e)}")
            
    async def test_api_endpoints_integration(self):
        """Test 6: API Endpoints Integration - All endpoints functional"""
        logger.info("\n🔍 TEST 6: API Endpoints Integration")
        
        try:
            # Comprehensive endpoint testing
            endpoints_to_test = [
                {'path': '/market-status', 'method': 'GET', 'expected_fields': ['trading_active']},
                {'path': '/opportunities', 'method': 'GET', 'expected_type': list},
                {'path': '/analyses', 'method': 'GET', 'expected_type': list},
                {'path': '/decisions', 'method': 'GET', 'expected_type': list},
                {'path': '/start-trading', 'method': 'POST', 'expected_fields': ['message']},
                {'path': '/stop-trading', 'method': 'POST', 'expected_fields': ['message']},
            ]
            
            endpoint_results = {}
            
            for endpoint_config in endpoints_to_test:
                path = endpoint_config['path']
                method = endpoint_config['method']
                
                try:
                    if method == 'GET':
                        response = requests.get(f"{self.api_url}{path}", timeout=30)
                    elif method == 'POST':
                        response = requests.post(f"{self.api_url}{path}", timeout=30)
                    else:
                        continue
                        
                    success = response.status_code == 200
                    data = response.json() if success else None
                    
                    # Validate response structure
                    structure_valid = True
                    if success and data:
                        if 'expected_fields' in endpoint_config:
                            for field in endpoint_config['expected_fields']:
                                if field not in data:
                                    structure_valid = False
                                    break
                        elif 'expected_type' in endpoint_config:
                            expected_type = endpoint_config['expected_type']
                            if not isinstance(data, expected_type):
                                # Handle wrapped responses
                                if isinstance(data, dict) and len(data) == 1:
                                    inner_data = list(data.values())[0]
                                    structure_valid = isinstance(inner_data, expected_type)
                                else:
                                    structure_valid = False
                                    
                    endpoint_results[path] = {
                        'status_code': response.status_code,
                        'success': success,
                        'structure_valid': structure_valid,
                        'data_size': len(data) if isinstance(data, (list, dict)) else 0
                    }
                    
                    logger.info(f"   📊 {method} {path}: {response.status_code} - Structure: {structure_valid} - Size: {endpoint_results[path]['data_size']}")
                    
                except Exception as e:
                    endpoint_results[path] = {
                        'status_code': 0,
                        'success': False,
                        'structure_valid': False,
                        'error': str(e)
                    }
                    logger.info(f"   ❌ {method} {path}: Error - {str(e)}")
                    
            # Calculate success metrics
            working_endpoints = sum(1 for result in endpoint_results.values() if result['success'])
            valid_structures = sum(1 for result in endpoint_results.values() if result.get('structure_valid', False))
            total_endpoints = len(endpoints_to_test)
            
            logger.info(f"   📊 Working endpoints: {working_endpoints}/{total_endpoints}")
            logger.info(f"   📊 Valid structures: {valid_structures}/{total_endpoints}")
            
            # Success criteria
            success = (
                working_endpoints >= total_endpoints * 0.8 and  # Most endpoints working
                valid_structures >= total_endpoints * 0.7  # Most structures valid
            )
            
            details = f"Working: {working_endpoints}/{total_endpoints}, Valid structures: {valid_structures}/{total_endpoints}"
            
            self.log_test_result("API Endpoints Integration", success, details)
            
        except Exception as e:
            self.log_test_result("API Endpoints Integration", False, f"Exception: {str(e)}")
            
    async def test_websocket_functionality(self):
        """Test 7: WebSocket Functionality - Real-time communication"""
        logger.info("\n🔍 TEST 7: WebSocket Functionality")
        
        try:
            # Test WebSocket connection
            ws_url = f"{self.ws_url}/ws"
            logger.info(f"   🔌 Testing WebSocket connection to: {ws_url}")
            
            websocket_working = False
            messages_received = []
            
            try:
                # Try to connect to WebSocket
                async with websockets.connect(ws_url, timeout=10) as websocket:
                    logger.info("   ✅ WebSocket connection established")
                    websocket_working = True
                    
                    # Listen for messages for a short time
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        messages_received.append(message)
                        logger.info(f"   📨 Received message: {message[:100]}...")
                    except asyncio.TimeoutError:
                        logger.info("   ⏳ No messages received within timeout (normal)")
                        
            except Exception as ws_error:
                logger.info(f"   ❌ WebSocket connection failed: {str(ws_error)}")
                websocket_working = False
                
            # Check backend logs for WebSocket activity
            logs = self.get_backend_logs(200)
            websocket_logs = 0
            
            for line in logs.split('\n'):
                if any(keyword in line.lower() for keyword in ['websocket', 'ws', 'connection', 'broadcast']):
                    websocket_logs += 1
                    
            logger.info(f"   📊 WebSocket-related logs: {websocket_logs}")
            logger.info(f"   📊 Messages received: {len(messages_received)}")
            
            # Success criteria (WebSocket may not be critical for basic functionality)
            success = websocket_working or websocket_logs > 0
            
            details = f"Connection: {websocket_working}, Messages: {len(messages_received)}, Logs: {websocket_logs}"
            
            self.log_test_result("WebSocket Functionality", success, details)
            
        except Exception as e:
            self.log_test_result("WebSocket Functionality", False, f"Exception: {str(e)}")
            
    async def test_frontend_communication(self):
        """Test 8: Frontend Communication - Verify frontend can access backend"""
        logger.info("\n🔍 TEST 8: Frontend Communication")
        
        try:
            # Test CORS and basic connectivity
            headers = {
                'Origin': 'https://dual-ai-trader-3.preview.emergentagent.com',
                'Access-Control-Request-Method': 'GET',
                'Access-Control-Request-Headers': 'Content-Type'
            }
            
            # Test preflight request
            preflight_response = requests.options(f"{self.api_url}/market-status", headers=headers, timeout=30)
            preflight_success = preflight_response.status_code in [200, 204]
            
            logger.info(f"   📊 CORS preflight: {preflight_response.status_code} - {preflight_success}")
            
            # Test actual requests with CORS headers
            cors_headers = {'Origin': 'https://dual-ai-trader-3.preview.emergentagent.com'}
            
            cors_tests = [
                '/market-status',
                '/opportunities',
                '/analyses'
            ]
            
            cors_results = {}
            
            for endpoint in cors_tests:
                try:
                    response = requests.get(f"{self.api_url}{endpoint}", headers=cors_headers, timeout=30)
                    cors_results[endpoint] = {
                        'status': response.status_code,
                        'success': response.status_code == 200,
                        'cors_headers': 'Access-Control-Allow-Origin' in response.headers
                    }
                    
                    logger.info(f"   📊 {endpoint}: {response.status_code} - CORS: {cors_results[endpoint]['cors_headers']}")
                    
                except Exception as e:
                    cors_results[endpoint] = {
                        'status': 0,
                        'success': False,
                        'cors_headers': False,
                        'error': str(e)
                    }
                    
            # Check backend configuration for CORS
            logs = self.get_backend_logs(200)
            cors_config_logs = 0
            
            for line in logs.split('\n'):
                if any(keyword in line.lower() for keyword in ['cors', 'origin', 'access-control']):
                    cors_config_logs += 1
                    
            # Calculate success metrics
            successful_cors = sum(1 for result in cors_results.values() if result['success'])
            cors_headers_present = sum(1 for result in cors_results.values() if result['cors_headers'])
            
            logger.info(f"   📊 Successful CORS requests: {successful_cors}/{len(cors_tests)}")
            logger.info(f"   📊 CORS headers present: {cors_headers_present}/{len(cors_tests)}")
            logger.info(f"   📊 CORS config logs: {cors_config_logs}")
            
            # Success criteria
            success = (
                preflight_success and
                successful_cors >= len(cors_tests) * 0.8
            )
            
            details = f"Preflight: {preflight_success}, Successful: {successful_cors}/{len(cors_tests)}, Headers: {cors_headers_present}/{len(cors_tests)}"
            
            self.log_test_result("Frontend Communication", success, details)
            
        except Exception as e:
            self.log_test_result("Frontend Communication", False, f"Exception: {str(e)}")
            
    async def run_comprehensive_tests(self):
        """Run all Trading System Startup tests"""
        logger.info("🚀 Starting Trading System Startup Test Suite")
        logger.info("Focus: Complete system startup after mathematical error corrections")
        logger.info("=" * 80)
        
        await self.setup_database()
        
        # Run all tests in sequence
        await self.test_trading_system_startup()
        await self.test_mathematical_error_absence()
        await self.test_pattern_functionality()
        await self.test_background_trading_loop()
        await self.test_system_performance()
        await self.test_api_endpoints_integration()
        await self.test_websocket_functionality()
        await self.test_frontend_communication()
        
        await self.cleanup_database()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 TRADING SYSTEM STARTUP TEST SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Trading system startup is working correctly!")
            logger.info("✅ Mathematical errors have been successfully corrected")
            logger.info("✅ System is stable and ready for trading operations")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY WORKING - Minor issues detected but system is functional")
        else:
            logger.info("❌ CRITICAL ISSUES - Trading system needs attention before production use")
            
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = TradingSystemStartupTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
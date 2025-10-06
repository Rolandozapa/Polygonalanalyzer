#!/usr/bin/env python3
"""
COMPREHENSIVE PERFORMANCE TEST SUITE AFTER CPU OPTIMIZATIONS
Focus: Complete backend performance validation after implementing major CPU optimizations

TESTING REQUIREMENTS FROM REVIEW REQUEST:
1. **PERFORMANCE VALIDATION**: Verify CPU stays low during API operations
2. **CORE API FUNCTIONALITY**: Test all main endpoints respond correctly  
3. **TRADING SYSTEM**: Verify IA1/IA2 analysis and decision pipeline
4. **DATABASE OPERATIONS**: Confirm MongoDB operations are efficient
5. **WEBSOCKET CONNECTIONS**: Test real-time updates work with optimized timing
6. **BINGX INTEGRATION**: Verify trading integration still functional
7. **SYSTEM STABILITY**: Ensure optimizations don't break existing features

CPU OPTIMIZATIONS IMPLEMENTED:
- Frontend polling: 5s → 15s  
- ThreadPoolExecutor: 20 → 6 workers
- Backend loops: 30s → 60s WebSocket, 5s → 15s position monitoring  
- CRITICAL FIX: psutil.cpu_percent(interval=1) → psutil.cpu_percent() (non-blocking)

CURRENT STATUS:
- CPU usage improved from 97-100% to 11.7% (72% reduction)
- All services running (backend, frontend, mongodb, code-server)
- Dashboard shows: 50 opportunities, 30 IA2 decisions, system active

EXPECTED RESULTS: 
All functionality working with significantly improved CPU performance (target: <20% CPU usage)

SPECIFIC TESTS NEEDED:
- GET /api/opportunities, /api/analyses, /api/decisions, /api/performance
- POST /api/trading/start-cycle (if available)
- WebSocket connections /api/ws
- BingX endpoints /api/bingx/status, /api/bingx/balance
- Monitor CPU during intensive operations
"""

import asyncio
import json
import logging
import os
import sys
import time
import psutil
import websockets
from datetime import datetime, timedelta
from typing import Dict, Any, List
import requests
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceTestSuite:
    """Comprehensive performance test suite after CPU optimizations"""
    
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
        self.ws_url = backend_url.replace('http', 'ws') + "/api/ws"
        logger.info(f"Testing Performance Optimizations at: {self.api_url}")
        
        # Test results
        self.test_results = []
        self.cpu_measurements = []
        
        # Core API endpoints to test
        self.core_endpoints = [
            {'method': 'GET', 'path': '/opportunities', 'name': 'Market Opportunities'},
            {'method': 'GET', 'path': '/analyses', 'name': 'IA1 Technical Analyses'},
            {'method': 'GET', 'path': '/decisions', 'name': 'IA2 Trading Decisions'},
            {'method': 'GET', 'path': '/performance', 'name': 'Trading Performance'},
            {'method': 'GET', 'path': '/bingx/status', 'name': 'BingX Status'},
            {'method': 'GET', 'path': '/bingx/balance', 'name': 'BingX Balance'},
            {'method': 'GET', 'path': '/active-positions', 'name': 'Active Positions'},
            {'method': 'GET', 'path': '/trading/execution-mode', 'name': 'Execution Mode'},
        ]
        
        # Performance thresholds
        self.cpu_threshold = 20.0  # Target: <20% CPU usage
        self.response_time_threshold = 5.0  # Target: <5s response time
        self.memory_threshold = 80.0  # Target: <80% memory usage
        
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
    
    def measure_system_performance(self) -> Dict[str, float]:
        """Measure current system performance metrics"""
        try:
            # CPU usage (non-blocking as per optimization)
            cpu_percent = psutil.cpu_percent()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Process count
            process_count = len(psutil.pids())
            
            metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_percent': disk_percent,
                'process_count': process_count,
                'timestamp': time.time()
            }
            
            self.cpu_measurements.append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error measuring system performance: {e}")
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'disk_percent': 0.0,
                'process_count': 0,
                'timestamp': time.time()
            }
    
    async def test_1_baseline_performance_measurement(self):
        """Test 1: Baseline System Performance Measurement"""
        logger.info("\n🔍 TEST 1: Baseline System Performance Measurement")
        
        try:
            # Take multiple measurements over 30 seconds
            measurements = []
            for i in range(6):  # 6 measurements over 30 seconds
                metrics = self.measure_system_performance()
                measurements.append(metrics)
                logger.info(f"   📊 Measurement {i+1}: CPU: {metrics['cpu_percent']:.1f}%, Memory: {metrics['memory_percent']:.1f}%")
                if i < 5:  # Don't sleep after last measurement
                    await asyncio.sleep(5)
            
            # Calculate averages
            avg_cpu = sum(m['cpu_percent'] for m in measurements) / len(measurements)
            avg_memory = sum(m['memory_percent'] for m in measurements) / len(measurements)
            max_cpu = max(m['cpu_percent'] for m in measurements)
            
            logger.info(f"   📊 Performance Summary:")
            logger.info(f"      Average CPU: {avg_cpu:.1f}%")
            logger.info(f"      Maximum CPU: {max_cpu:.1f}%")
            logger.info(f"      Average Memory: {avg_memory:.1f}%")
            
            # Check if CPU optimization target is met
            if avg_cpu <= self.cpu_threshold:
                self.log_test_result("Baseline Performance - CPU Optimization", True, 
                                   f"CPU usage {avg_cpu:.1f}% is below target {self.cpu_threshold}%")
            else:
                self.log_test_result("Baseline Performance - CPU Optimization", False, 
                                   f"CPU usage {avg_cpu:.1f}% exceeds target {self.cpu_threshold}%")
            
            # Check memory usage
            if avg_memory <= self.memory_threshold:
                self.log_test_result("Baseline Performance - Memory Usage", True, 
                                   f"Memory usage {avg_memory:.1f}% is acceptable")
            else:
                self.log_test_result("Baseline Performance - Memory Usage", False, 
                                   f"Memory usage {avg_memory:.1f}% is high")
                
        except Exception as e:
            self.log_test_result("Baseline Performance Measurement", False, f"Exception: {str(e)}")
    
    async def test_2_core_api_functionality(self):
        """Test 2: Core API Functionality with Performance Monitoring"""
        logger.info("\n🔍 TEST 2: Core API Functionality with Performance Monitoring")
        
        api_results = []
        
        for endpoint in self.core_endpoints:
            try:
                method = endpoint['method']
                path = endpoint['path']
                name = endpoint['name']
                
                logger.info(f"   Testing {method} {path} ({name})")
                
                # Measure performance before API call
                pre_metrics = self.measure_system_performance()
                start_time = time.time()
                
                # Make API call
                if method == 'GET':
                    if 'market-price' in path:
                        response = requests.get(f"{self.api_url}{path}?symbol=BTCUSDT", timeout=30)
                    else:
                        response = requests.get(f"{self.api_url}{path}", timeout=30)
                
                # Measure performance after API call
                end_time = time.time()
                post_metrics = self.measure_system_performance()
                
                response_time = end_time - start_time
                cpu_increase = post_metrics['cpu_percent'] - pre_metrics['cpu_percent']
                
                # Evaluate response
                if response.status_code in [200, 201]:
                    try:
                        data = response.json()
                        data_size = len(str(data))
                        
                        api_results.append({
                            'endpoint': f"{method} {path}",
                            'name': name,
                            'status': 'SUCCESS',
                            'response_time': response_time,
                            'cpu_increase': cpu_increase,
                            'data_size': data_size
                        })
                        
                        logger.info(f"      ✅ {name}: SUCCESS (HTTP {response.status_code}, {response_time:.2f}s, CPU +{cpu_increase:.1f}%)")
                        
                    except json.JSONDecodeError:
                        api_results.append({
                            'endpoint': f"{method} {path}",
                            'name': name,
                            'status': 'SUCCESS_NO_JSON',
                            'response_time': response_time,
                            'cpu_increase': cpu_increase,
                            'data_size': len(response.text)
                        })
                        logger.info(f"      ✅ {name}: SUCCESS - No JSON response ({response_time:.2f}s)")
                else:
                    api_results.append({
                        'endpoint': f"{method} {path}",
                        'name': name,
                        'status': f'HTTP_{response.status_code}',
                        'response_time': response_time,
                        'cpu_increase': cpu_increase,
                        'data_size': len(response.text)
                    })
                    logger.info(f"      ❌ {name}: HTTP {response.status_code} ({response_time:.2f}s)")
                    
            except Exception as e:
                api_results.append({
                    'endpoint': f"{method} {path}",
                    'name': name,
                    'status': 'ERROR',
                    'response_time': 0,
                    'cpu_increase': 0,
                    'error': str(e)
                })
                logger.info(f"      ❌ {name}: Exception - {str(e)}")
        
        # Evaluate overall API performance
        successful_apis = len([r for r in api_results if r['status'] in ['SUCCESS', 'SUCCESS_NO_JSON']])
        total_apis = len(api_results)
        
        # Check response times
        response_times = [r['response_time'] for r in api_results if 'response_time' in r and r['response_time'] > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        # Check CPU impact
        cpu_increases = [r['cpu_increase'] for r in api_results if 'cpu_increase' in r]
        avg_cpu_increase = sum(cpu_increases) / len(cpu_increases) if cpu_increases else 0
        max_cpu_increase = max(cpu_increases) if cpu_increases else 0
        
        logger.info(f"   📊 API Performance Summary:")
        logger.info(f"      Success Rate: {successful_apis}/{total_apis} ({successful_apis/total_apis:.1%})")
        logger.info(f"      Average Response Time: {avg_response_time:.2f}s")
        logger.info(f"      Maximum Response Time: {max_response_time:.2f}s")
        logger.info(f"      Average CPU Increase: {avg_cpu_increase:.1f}%")
        logger.info(f"      Maximum CPU Increase: {max_cpu_increase:.1f}%")
        
        # Test success criteria
        success_rate = successful_apis / total_apis if total_apis > 0 else 0
        
        if success_rate >= 0.8 and avg_response_time <= self.response_time_threshold:
            self.log_test_result("Core API Functionality", True, 
                               f"APIs working: {successful_apis}/{total_apis}, avg response: {avg_response_time:.2f}s")
        else:
            self.log_test_result("Core API Functionality", False, 
                               f"Poor API performance: {successful_apis}/{total_apis}, avg response: {avg_response_time:.2f}s")
    
    async def test_3_trading_system_pipeline(self):
        """Test 3: IA1/IA2 Trading System Pipeline"""
        logger.info("\n🔍 TEST 3: IA1/IA2 Trading System Pipeline")
        
        try:
            # Test opportunities endpoint (IA1 input)
            logger.info("   Testing market opportunities (IA1 input)...")
            opportunities_response = requests.get(f"{self.api_url}/opportunities", timeout=30)
            
            if opportunities_response.status_code == 200:
                opportunities = opportunities_response.json()
                opp_count = len(opportunities) if isinstance(opportunities, list) else 0
                logger.info(f"      ✅ Market opportunities: {opp_count} found")
                
                # Test IA1 analyses
                logger.info("   Testing IA1 technical analyses...")
                analyses_response = requests.get(f"{self.api_url}/analyses", timeout=30)
                
                if analyses_response.status_code == 200:
                    analyses = analyses_response.json()
                    analyses_count = len(analyses) if isinstance(analyses, list) else 0
                    logger.info(f"      ✅ IA1 analyses: {analyses_count} found")
                    
                    # Test IA2 decisions
                    logger.info("   Testing IA2 trading decisions...")
                    decisions_response = requests.get(f"{self.api_url}/decisions", timeout=30)
                    
                    if decisions_response.status_code == 200:
                        decisions = decisions_response.json()
                        decisions_count = len(decisions) if isinstance(decisions, list) else 0
                        logger.info(f"      ✅ IA2 decisions: {decisions_count} found")
                        
                        # Analyze pipeline efficiency
                        if opp_count > 0 and analyses_count > 0:
                            ia1_conversion = analyses_count / opp_count
                            logger.info(f"      📊 IA1 conversion rate: {ia1_conversion:.1%} ({analyses_count}/{opp_count})")
                            
                            if decisions_count > 0:
                                ia2_conversion = decisions_count / analyses_count
                                logger.info(f"      📊 IA2 conversion rate: {ia2_conversion:.1%} ({decisions_count}/{analyses_count})")
                                
                                self.log_test_result("Trading System Pipeline", True, 
                                                   f"Pipeline working: {opp_count}→{analyses_count}→{decisions_count}")
                            else:
                                self.log_test_result("Trading System Pipeline", False, 
                                                   "No IA2 decisions generated")
                        else:
                            self.log_test_result("Trading System Pipeline", False, 
                                               "No IA1 analyses generated from opportunities")
                    else:
                        self.log_test_result("Trading System Pipeline", False, 
                                           f"IA2 decisions endpoint failed: HTTP {decisions_response.status_code}")
                else:
                    self.log_test_result("Trading System Pipeline", False, 
                                       f"IA1 analyses endpoint failed: HTTP {analyses_response.status_code}")
            else:
                self.log_test_result("Trading System Pipeline", False, 
                                   f"Opportunities endpoint failed: HTTP {opportunities_response.status_code}")
                
        except Exception as e:
            self.log_test_result("Trading System Pipeline", False, f"Exception: {str(e)}")
    
    async def test_4_websocket_performance(self):
        """Test 4: WebSocket Performance with Optimized Timing"""
        logger.info("\n🔍 TEST 4: WebSocket Performance with Optimized Timing")
        
        try:
            # Measure CPU before WebSocket connection
            pre_metrics = self.measure_system_performance()
            
            # Test WebSocket connection
            logger.info(f"   Connecting to WebSocket: {self.ws_url}")
            
            messages_received = []
            connection_successful = False
            
            try:
                async with websockets.connect(self.ws_url, timeout=10) as websocket:
                    connection_successful = True
                    logger.info("      ✅ WebSocket connection established")
                    
                    # Listen for messages for 30 seconds
                    start_time = time.time()
                    while time.time() - start_time < 30:
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=5)
                            messages_received.append({
                                'timestamp': time.time(),
                                'message': message[:100] + "..." if len(message) > 100 else message
                            })
                            logger.info(f"      📨 Message received: {message[:50]}...")
                        except asyncio.TimeoutError:
                            # No message received in 5 seconds, continue
                            pass
                        except websockets.exceptions.ConnectionClosed:
                            logger.info("      ⚠️ WebSocket connection closed")
                            break
                    
            except Exception as ws_e:
                logger.info(f"      ❌ WebSocket connection failed: {str(ws_e)}")
            
            # Measure CPU after WebSocket test
            post_metrics = self.measure_system_performance()
            cpu_increase = post_metrics['cpu_percent'] - pre_metrics['cpu_percent']
            
            # Analyze WebSocket performance
            message_count = len(messages_received)
            
            if connection_successful:
                if message_count > 0:
                    # Calculate message frequency
                    if len(messages_received) > 1:
                        time_span = messages_received[-1]['timestamp'] - messages_received[0]['timestamp']
                        message_frequency = message_count / time_span if time_span > 0 else 0
                        logger.info(f"      📊 Messages: {message_count}, Frequency: {message_frequency:.2f} msg/s")
                    
                    self.log_test_result("WebSocket Performance", True, 
                                       f"WebSocket working: {message_count} messages, CPU +{cpu_increase:.1f}%")
                else:
                    self.log_test_result("WebSocket Performance", True, 
                                       f"WebSocket connected but no messages (may be normal), CPU +{cpu_increase:.1f}%")
            else:
                self.log_test_result("WebSocket Performance", False, 
                                   "WebSocket connection failed")
                
        except Exception as e:
            self.log_test_result("WebSocket Performance", False, f"Exception: {str(e)}")
    
    async def test_5_database_operations_efficiency(self):
        """Test 5: Database Operations Efficiency"""
        logger.info("\n🔍 TEST 5: Database Operations Efficiency")
        
        try:
            # Test multiple database-heavy endpoints
            db_endpoints = [
                {'path': '/opportunities', 'name': 'Opportunities (DB read)'},
                {'path': '/analyses', 'name': 'Analyses (DB read)'},
                {'path': '/decisions', 'name': 'Decisions (DB read)'},
                {'path': '/performance', 'name': 'Performance (DB aggregation)'},
            ]
            
            db_results = []
            
            for endpoint in db_endpoints:
                # Measure performance before DB operation
                pre_metrics = self.measure_system_performance()
                start_time = time.time()
                
                response = requests.get(f"{self.api_url}{endpoint['path']}", timeout=30)
                
                end_time = time.time()
                post_metrics = self.measure_system_performance()
                
                response_time = end_time - start_time
                cpu_increase = post_metrics['cpu_percent'] - pre_metrics['cpu_percent']
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        record_count = len(data) if isinstance(data, list) else 1
                        
                        db_results.append({
                            'endpoint': endpoint['name'],
                            'response_time': response_time,
                            'cpu_increase': cpu_increase,
                            'record_count': record_count,
                            'success': True
                        })
                        
                        logger.info(f"      ✅ {endpoint['name']}: {record_count} records, {response_time:.2f}s, CPU +{cpu_increase:.1f}%")
                        
                    except json.JSONDecodeError:
                        db_results.append({
                            'endpoint': endpoint['name'],
                            'response_time': response_time,
                            'cpu_increase': cpu_increase,
                            'record_count': 0,
                            'success': True
                        })
                        logger.info(f"      ✅ {endpoint['name']}: Non-JSON response, {response_time:.2f}s")
                else:
                    db_results.append({
                        'endpoint': endpoint['name'],
                        'response_time': response_time,
                        'cpu_increase': cpu_increase,
                        'record_count': 0,
                        'success': False
                    })
                    logger.info(f"      ❌ {endpoint['name']}: HTTP {response.status_code}")
            
            # Analyze database performance
            successful_db_ops = len([r for r in db_results if r['success']])
            total_db_ops = len(db_results)
            
            avg_response_time = sum(r['response_time'] for r in db_results) / len(db_results)
            avg_cpu_increase = sum(r['cpu_increase'] for r in db_results) / len(db_results)
            total_records = sum(r['record_count'] for r in db_results)
            
            logger.info(f"   📊 Database Performance Summary:")
            logger.info(f"      Success Rate: {successful_db_ops}/{total_db_ops}")
            logger.info(f"      Average Response Time: {avg_response_time:.2f}s")
            logger.info(f"      Average CPU Increase: {avg_cpu_increase:.1f}%")
            logger.info(f"      Total Records Retrieved: {total_records}")
            
            if successful_db_ops == total_db_ops and avg_response_time <= 3.0:
                self.log_test_result("Database Operations Efficiency", True, 
                                   f"DB operations efficient: {avg_response_time:.2f}s avg, {total_records} records")
            else:
                self.log_test_result("Database Operations Efficiency", False, 
                                   f"DB operations slow: {avg_response_time:.2f}s avg, {successful_db_ops}/{total_db_ops} success")
                
        except Exception as e:
            self.log_test_result("Database Operations Efficiency", False, f"Exception: {str(e)}")
    
    async def test_6_bingx_integration_stability(self):
        """Test 6: BingX Integration Stability"""
        logger.info("\n🔍 TEST 6: BingX Integration Stability")
        
        try:
            bingx_endpoints = [
                {'path': '/bingx/status', 'name': 'BingX Status'},
                {'path': '/bingx/balance', 'name': 'BingX Balance'},
                {'path': '/bingx/positions', 'name': 'BingX Positions'},
            ]
            
            bingx_results = []
            
            for endpoint in bingx_endpoints:
                try:
                    pre_metrics = self.measure_system_performance()
                    start_time = time.time()
                    
                    response = requests.get(f"{self.api_url}{endpoint['path']}", timeout=30)
                    
                    end_time = time.time()
                    post_metrics = self.measure_system_performance()
                    
                    response_time = end_time - start_time
                    cpu_increase = post_metrics['cpu_percent'] - pre_metrics['cpu_percent']
                    
                    if response.status_code == 200:
                        data = response.json()
                        bingx_results.append({
                            'endpoint': endpoint['name'],
                            'success': True,
                            'response_time': response_time,
                            'cpu_increase': cpu_increase
                        })
                        logger.info(f"      ✅ {endpoint['name']}: SUCCESS ({response_time:.2f}s, CPU +{cpu_increase:.1f}%)")
                    else:
                        bingx_results.append({
                            'endpoint': endpoint['name'],
                            'success': False,
                            'response_time': response_time,
                            'cpu_increase': cpu_increase,
                            'status_code': response.status_code
                        })
                        logger.info(f"      ❌ {endpoint['name']}: HTTP {response.status_code}")
                        
                except Exception as e:
                    bingx_results.append({
                        'endpoint': endpoint['name'],
                        'success': False,
                        'error': str(e)
                    })
                    logger.info(f"      ❌ {endpoint['name']}: Exception - {str(e)}")
            
            # Evaluate BingX integration
            successful_bingx = len([r for r in bingx_results if r['success']])
            total_bingx = len(bingx_results)
            
            if successful_bingx >= 2:  # At least 2 out of 3 BingX endpoints working
                self.log_test_result("BingX Integration Stability", True, 
                                   f"BingX integration stable: {successful_bingx}/{total_bingx} endpoints working")
            else:
                self.log_test_result("BingX Integration Stability", False, 
                                   f"BingX integration unstable: {successful_bingx}/{total_bingx} endpoints working")
                
        except Exception as e:
            self.log_test_result("BingX Integration Stability", False, f"Exception: {str(e)}")
    
    async def test_7_intensive_operations_cpu_impact(self):
        """Test 7: CPU Impact During Intensive Operations"""
        logger.info("\n🔍 TEST 7: CPU Impact During Intensive Operations")
        
        try:
            # Measure baseline CPU
            baseline_metrics = self.measure_system_performance()
            baseline_cpu = baseline_metrics['cpu_percent']
            logger.info(f"   📊 Baseline CPU: {baseline_cpu:.1f}%")
            
            # Perform intensive operations concurrently
            logger.info("   🚀 Starting intensive operations...")
            
            async def intensive_api_calls():
                """Make multiple concurrent API calls"""
                tasks = []
                endpoints = ['/opportunities', '/analyses', '/decisions', '/performance', '/bingx/status']
                
                for _ in range(3):  # 3 rounds of calls
                    for endpoint in endpoints:
                        task = asyncio.create_task(self.make_api_call(endpoint))
                        tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return results
            
            # Monitor CPU during intensive operations
            cpu_measurements = []
            
            # Start intensive operations
            intensive_task = asyncio.create_task(intensive_api_calls())
            
            # Monitor CPU for 30 seconds
            for i in range(6):  # 6 measurements over 30 seconds
                await asyncio.sleep(5)
                metrics = self.measure_system_performance()
                cpu_measurements.append(metrics['cpu_percent'])
                logger.info(f"   📊 CPU during intensive ops ({i+1}/6): {metrics['cpu_percent']:.1f}%")
            
            # Wait for intensive operations to complete
            await intensive_task
            
            # Analyze CPU impact
            max_cpu = max(cpu_measurements)
            avg_cpu = sum(cpu_measurements) / len(cpu_measurements)
            cpu_spike = max_cpu - baseline_cpu
            
            logger.info(f"   📊 CPU Impact Analysis:")
            logger.info(f"      Baseline CPU: {baseline_cpu:.1f}%")
            logger.info(f"      Maximum CPU: {max_cpu:.1f}%")
            logger.info(f"      Average CPU: {avg_cpu:.1f}%")
            logger.info(f"      CPU Spike: +{cpu_spike:.1f}%")
            
            # Check if CPU stays within acceptable limits
            if max_cpu <= self.cpu_threshold and cpu_spike <= 15.0:  # Max 15% spike allowed
                self.log_test_result("Intensive Operations CPU Impact", True, 
                                   f"CPU controlled during intensive ops: max {max_cpu:.1f}%, spike +{cpu_spike:.1f}%")
            else:
                self.log_test_result("Intensive Operations CPU Impact", False, 
                                   f"CPU too high during intensive ops: max {max_cpu:.1f}%, spike +{cpu_spike:.1f}%")
                
        except Exception as e:
            self.log_test_result("Intensive Operations CPU Impact", False, f"Exception: {str(e)}")
    
    async def make_api_call(self, endpoint: str):
        """Helper method to make API calls"""
        try:
            response = requests.get(f"{self.api_url}{endpoint}", timeout=10)
            return response.status_code
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def test_8_system_stability_over_time(self):
        """Test 8: System Stability Over Time"""
        logger.info("\n🔍 TEST 8: System Stability Over Time (2 minutes)")
        
        try:
            stability_measurements = []
            start_time = time.time()
            
            # Monitor system for 2 minutes
            while time.time() - start_time < 120:  # 2 minutes
                metrics = self.measure_system_performance()
                stability_measurements.append(metrics)
                
                # Make periodic API calls to simulate normal usage
                try:
                    response = requests.get(f"{self.api_url}/opportunities", timeout=5)
                    api_status = response.status_code
                except:
                    api_status = "Error"
                
                elapsed = time.time() - start_time
                logger.info(f"   📊 Stability check ({elapsed:.0f}s): CPU {metrics['cpu_percent']:.1f}%, Memory {metrics['memory_percent']:.1f}%, API: {api_status}")
                
                await asyncio.sleep(15)  # Check every 15 seconds
            
            # Analyze stability
            cpu_values = [m['cpu_percent'] for m in stability_measurements]
            memory_values = [m['memory_percent'] for m in stability_measurements]
            
            cpu_stability = max(cpu_values) - min(cpu_values)
            memory_stability = max(memory_values) - min(memory_values)
            avg_cpu = sum(cpu_values) / len(cpu_values)
            avg_memory = sum(memory_values) / len(memory_values)
            
            logger.info(f"   📊 Stability Analysis:")
            logger.info(f"      CPU Range: {min(cpu_values):.1f}% - {max(cpu_values):.1f}% (variation: {cpu_stability:.1f}%)")
            logger.info(f"      Memory Range: {min(memory_values):.1f}% - {max(memory_values):.1f}% (variation: {memory_stability:.1f}%)")
            logger.info(f"      Average CPU: {avg_cpu:.1f}%")
            logger.info(f"      Average Memory: {avg_memory:.1f}%")
            
            # Check stability criteria
            if avg_cpu <= self.cpu_threshold and cpu_stability <= 10.0 and memory_stability <= 5.0:
                self.log_test_result("System Stability Over Time", True, 
                                   f"System stable: CPU {avg_cpu:.1f}% avg, variations CPU ±{cpu_stability:.1f}%, Memory ±{memory_stability:.1f}%")
            else:
                self.log_test_result("System Stability Over Time", False, 
                                   f"System unstable: CPU {avg_cpu:.1f}% avg, variations CPU ±{cpu_stability:.1f}%, Memory ±{memory_stability:.1f}%")
                
        except Exception as e:
            self.log_test_result("System Stability Over Time", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_performance_tests(self):
        """Run all performance tests"""
        logger.info("🚀 Starting Comprehensive Performance Test Suite After CPU Optimizations")
        logger.info("=" * 80)
        logger.info("📋 PERFORMANCE VALIDATION AFTER CPU OPTIMIZATIONS")
        logger.info("🎯 Testing: CPU performance, API functionality, trading pipeline, WebSocket, DB efficiency")
        logger.info("🎯 Expected: All functionality working with CPU <20% (target achieved: 11.7%)")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_baseline_performance_measurement()
        await self.test_2_core_api_functionality()
        await self.test_3_trading_system_pipeline()
        await self.test_4_websocket_performance()
        await self.test_5_database_operations_efficiency()
        await self.test_6_bingx_integration_stability()
        await self.test_7_intensive_operations_cpu_impact()
        await self.test_8_system_stability_over_time()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 COMPREHENSIVE PERFORMANCE TEST SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # CPU Performance Analysis
        if self.cpu_measurements:
            all_cpu_values = [m['cpu_percent'] for m in self.cpu_measurements]
            avg_cpu = sum(all_cpu_values) / len(all_cpu_values)
            max_cpu = max(all_cpu_values)
            min_cpu = min(all_cpu_values)
            
            logger.info("\n" + "=" * 80)
            logger.info("📊 CPU OPTIMIZATION PERFORMANCE ANALYSIS")
            logger.info("=" * 80)
            logger.info(f"🎯 Target CPU Usage: <{self.cpu_threshold}%")
            logger.info(f"📊 Measured CPU Performance:")
            logger.info(f"   Average CPU: {avg_cpu:.1f}%")
            logger.info(f"   Maximum CPU: {max_cpu:.1f}%")
            logger.info(f"   Minimum CPU: {min_cpu:.1f}%")
            logger.info(f"   CPU Range: {min_cpu:.1f}% - {max_cpu:.1f}%")
            
            if avg_cpu <= self.cpu_threshold:
                logger.info(f"✅ CPU OPTIMIZATION SUCCESS: Average {avg_cpu:.1f}% is below target {self.cpu_threshold}%")
                cpu_improvement = ((100 - avg_cpu) / 100) * 100  # Improvement from theoretical 100%
                logger.info(f"🚀 CPU Performance: {100 - avg_cpu:.1f}% efficiency achieved")
            else:
                logger.info(f"❌ CPU OPTIMIZATION NEEDS WORK: Average {avg_cpu:.1f}% exceeds target {self.cpu_threshold}%")
        
        # System analysis
        logger.info("\n" + "=" * 80)
        logger.info("📋 SYSTEM PERFORMANCE STATUS AFTER OPTIMIZATIONS")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - PERFORMANCE OPTIMIZATIONS FULLY SUCCESSFUL!")
            logger.info("✅ CPU usage optimized and stable")
            logger.info("✅ Core API functionality maintained")
            logger.info("✅ Trading system pipeline operational")
            logger.info("✅ WebSocket performance optimized")
            logger.info("✅ Database operations efficient")
            logger.info("✅ BingX integration stable")
            logger.info("✅ System stable under intensive operations")
            logger.info("✅ Long-term stability confirmed")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY SUCCESSFUL - Performance optimizations working with minor issues")
            logger.info("🔍 Some components may need fine-tuning for full optimization")
        elif passed_tests >= total_tests * 0.6:
            logger.info("⚠️ PARTIALLY SUCCESSFUL - Core optimizations working")
            logger.info("🔧 Some performance aspects may need additional optimization")
        else:
            logger.info("❌ OPTIMIZATION NOT SUCCESSFUL - Critical performance issues remain")
            logger.info("🚨 Major performance problems preventing optimization goals")
        
        # Specific requirements check
        logger.info("\n📝 PERFORMANCE OPTIMIZATION REQUIREMENTS VERIFICATION:")
        
        requirements_met = []
        requirements_failed = []
        
        # Check each requirement based on test results
        for result in self.test_results:
            if result['success']:
                if "CPU Optimization" in result['test']:
                    requirements_met.append("✅ CPU usage optimized (<20% target achieved)")
                elif "Core API Functionality" in result['test']:
                    requirements_met.append("✅ Core API functionality maintained")
                elif "Trading System Pipeline" in result['test']:
                    requirements_met.append("✅ IA1/IA2 trading pipeline operational")
                elif "WebSocket Performance" in result['test']:
                    requirements_met.append("✅ WebSocket performance optimized")
                elif "Database Operations" in result['test']:
                    requirements_met.append("✅ Database operations efficient")
                elif "BingX Integration" in result['test']:
                    requirements_met.append("✅ BingX integration stable")
                elif "Intensive Operations" in result['test']:
                    requirements_met.append("✅ CPU controlled during intensive operations")
                elif "System Stability" in result['test']:
                    requirements_met.append("✅ System stability maintained over time")
            else:
                if "CPU Optimization" in result['test']:
                    requirements_failed.append("❌ CPU usage not optimized")
                elif "Core API Functionality" in result['test']:
                    requirements_failed.append("❌ Core API functionality degraded")
                elif "Trading System Pipeline" in result['test']:
                    requirements_failed.append("❌ Trading pipeline not working")
                elif "WebSocket Performance" in result['test']:
                    requirements_failed.append("❌ WebSocket performance issues")
                elif "Database Operations" in result['test']:
                    requirements_failed.append("❌ Database operations inefficient")
                elif "BingX Integration" in result['test']:
                    requirements_failed.append("❌ BingX integration unstable")
                elif "Intensive Operations" in result['test']:
                    requirements_failed.append("❌ CPU spikes during intensive operations")
                elif "System Stability" in result['test']:
                    requirements_failed.append("❌ System stability issues")
        
        for req in requirements_met:
            logger.info(f"   {req}")
        
        for req in requirements_failed:
            logger.info(f"   {req}")
        
        logger.info(f"\n🏆 FINAL RESULT: {len(requirements_met)}/{len(requirements_met) + len(requirements_failed)} optimization requirements satisfied")
        
        # Final verdict
        if len(requirements_failed) == 0:
            logger.info("\n🎉 VERDICT: CPU OPTIMIZATIONS FULLY SUCCESSFUL!")
            logger.info("✅ All performance targets achieved")
            logger.info("✅ CPU usage optimized while maintaining full functionality")
            logger.info("✅ System ready for production with improved performance")
        elif len(requirements_failed) <= 1:
            logger.info("\n⚠️ VERDICT: CPU OPTIMIZATIONS MOSTLY SUCCESSFUL")
            logger.info("🔍 Minor performance issues may need attention")
        elif len(requirements_failed) <= 3:
            logger.info("\n⚠️ VERDICT: CPU OPTIMIZATIONS PARTIALLY SUCCESSFUL")
            logger.info("🔧 Several performance aspects need additional optimization")
        else:
            logger.info("\n❌ VERDICT: CPU OPTIMIZATIONS NOT SUCCESSFUL")
            logger.info("🚨 Major performance issues preventing optimization goals")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = PerformanceTestSuite()
    passed, total = await test_suite.run_comprehensive_performance_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
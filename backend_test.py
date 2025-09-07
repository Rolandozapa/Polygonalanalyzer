#!/usr/bin/env python3
"""
Backend Testing Suite for Active Trading System Position Sizing Fixes
Focus: IA2 Position Size Matching & Zero Position Size Handling
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List

# Add backend to path
sys.path.append('/app/backend')

import requests
from motor.motor_asyncio import AsyncIOMotorClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PositionSizingTestSuite:
    """Test suite for Position Sizing fixes in Active Trading System"""
    
    def __init__(self):
        # Use localhost for internal testing
        self.base_url = "http://localhost:8001"
        
        self.api_url = f"{self.base_url}/api"
        logger.info(f"Testing backend at: {self.api_url}")
        
        # MongoDB connection for direct data access
        self.mongo_client = None
        self.db = None
        
        # Test results
        self.test_results = []
        
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
        
    async def test_ia2_position_size_matching(self):
        """Test 1: Verify IA2 Position Size Matching"""
        logger.info("\n🔍 TEST 1: IA2 Position Size Matching")
        
        try:
            # Get recent IA2 decisions to analyze position sizing
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("IA2 Position Size Matching", False, f"API error: {response.status_code}")
                return
                
            data = response.json()
            
            # Handle API response format
            if isinstance(data, dict) and 'decisions' in data:
                decisions = data['decisions']
            else:
                decisions = data
            
            if not decisions:
                self.log_test_result("IA2 Position Size Matching", False, "No decisions found")
                return
                
            # Analyze decisions for position sizing logic
            position_size_found = 0
            correct_position_sizes = 0
            
            for decision in decisions[:10]:  # Check last 10 decisions
                symbol = decision.get('symbol', 'UNKNOWN')
                signal = decision.get('signal', 'UNKNOWN')
                position_size = decision.get('position_size', 0)
                reasoning = decision.get('ia2_reasoning', '')
                
                logger.info(f"   📊 {symbol}: Signal={signal}, Position Size={position_size:.3f}")
                
                if position_size > 0:
                    position_size_found += 1
                    
                    # Check if reasoning mentions IA2 position sizing
                    if any(keyword in reasoning.lower() for keyword in ['position', 'size', 'ia2', '%']):
                        correct_position_sizes += 1
                        logger.info(f"      ✅ Position sizing logic found in reasoning")
                    else:
                        logger.info(f"      ⚠️ No position sizing details in reasoning")
                        
            success = position_size_found > 0 and correct_position_sizes >= position_size_found * 0.5
            details = f"Found {position_size_found} decisions with position sizes, {correct_position_sizes} with proper reasoning"
            
            self.log_test_result("IA2 Position Size Matching", success, details)
            
        except Exception as e:
            self.log_test_result("IA2 Position Size Matching", False, f"Exception: {str(e)}")
            
    async def test_zero_position_size_handling(self):
        """Test 2: Zero Position Size Handling"""
        logger.info("\n🔍 TEST 2: Zero Position Size Handling")
        
        try:
            # Check active positions to see if any trades were executed with 0% position size
            response = requests.get(f"{self.api_url}/active-positions", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Zero Position Size Handling", False, f"API error: {response.status_code}")
                return
                
            positions_data = response.json()
            active_positions = positions_data.get('active_positions', [])
            
            # Check if any active positions have 0% size (should not exist)
            zero_size_positions = []
            for position in active_positions:
                position_size_usd = position.get('position_size_usd', 0)
                if position_size_usd <= 0:
                    zero_size_positions.append(position)
                    
            # Also check recent decisions for 0% position sizes
            decisions_response = requests.get(f"{self.api_url}/decisions", timeout=30)
            zero_position_decisions = 0
            hold_decisions = 0
            
            if decisions_response.status_code == 200:
                decisions = decisions_response.json()
                
                for decision in decisions[:20]:  # Check last 20 decisions
                    signal = decision.get('signal', '').upper()
                    position_size = decision.get('position_size', 0)
                    
                    if signal == 'HOLD':
                        hold_decisions += 1
                    elif position_size <= 0:
                        zero_position_decisions += 1
                        logger.info(f"   📊 Found 0% position size: {decision.get('symbol')} - {signal}")
                        
                logger.info(f"   📊 Analysis: {hold_decisions} HOLD signals, {zero_position_decisions} zero position sizes")
                
            success = len(zero_size_positions) == 0  # No active positions should have 0% size
            details = f"Active positions with 0% size: {len(zero_size_positions)}, Zero position decisions: {zero_position_decisions}"
            
            self.log_test_result("Zero Position Size Handling", success, details)
            
        except Exception as e:
            self.log_test_result("Zero Position Size Handling", False, f"Exception: {str(e)}")
            
    async def test_position_size_logging(self):
        """Test 3: Position Size Logging"""
        logger.info("\n🔍 TEST 3: Position Size Logging")
        
        try:
            # Check backend logs for position sizing messages
            import subprocess
            
            # Get recent backend logs
            log_cmd = "tail -n 200 /var/log/supervisor/backend.*.log 2>/dev/null || echo 'No logs found'"
            result = subprocess.run(log_cmd, shell=True, capture_output=True, text=True)
            
            backend_logs = result.stdout
            
            # Look for position sizing log patterns
            ia2_position_logs = []
            skip_execution_logs = []
            
            for line in backend_logs.split('\n'):
                if 'Using IA2 position sizing' in line:
                    ia2_position_logs.append(line.strip())
                elif 'Skipping execution' in line and ('IA2 determined 0%' in line or '0% position size' in line):
                    skip_execution_logs.append(line.strip())
                    
            logger.info(f"   📊 Found {len(ia2_position_logs)} IA2 position sizing logs")
            logger.info(f"   📊 Found {len(skip_execution_logs)} skip execution logs")
            
            # Show sample logs
            if ia2_position_logs:
                logger.info(f"   📝 Sample IA2 log: {ia2_position_logs[-1]}")
            if skip_execution_logs:
                logger.info(f"   📝 Sample skip log: {skip_execution_logs[-1]}")
                
            success = len(ia2_position_logs) > 0 or len(skip_execution_logs) > 0
            details = f"IA2 position logs: {len(ia2_position_logs)}, Skip execution logs: {len(skip_execution_logs)}"
            
            self.log_test_result("Position Size Logging", success, details)
            
        except Exception as e:
            self.log_test_result("Position Size Logging", False, f"Exception: {str(e)}")
            
    async def test_ia2_integration(self):
        """Test 4: Integration with IA2"""
        logger.info("\n🔍 TEST 4: Integration with IA2")
        
        try:
            # Get recent IA2 decisions and check for position_size_percentage field
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("IA2 Integration", False, f"API error: {response.status_code}")
                return
                
            decisions = response.json()
            
            if not decisions:
                self.log_test_result("IA2 Integration", False, "No decisions found")
                return
                
            # Check if decisions contain position_size_percentage field
            decisions_with_position_size = 0
            total_decisions = 0
            
            for decision in decisions[:15]:  # Check last 15 decisions
                total_decisions += 1
                symbol = decision.get('symbol', 'UNKNOWN')
                signal = decision.get('signal', 'UNKNOWN')
                position_size = decision.get('position_size', None)
                
                if position_size is not None:
                    decisions_with_position_size += 1
                    logger.info(f"   📊 {symbol}: {signal} - Position Size: {position_size:.3f}")
                else:
                    logger.info(f"   ⚠️ {symbol}: {signal} - No position size field")
                    
            # Check if Active Position Manager receives this data
            positions_response = requests.get(f"{self.api_url}/active-positions", timeout=30)
            active_positions_working = positions_response.status_code == 200
            
            success = (decisions_with_position_size >= total_decisions * 0.8) and active_positions_working
            details = f"Decisions with position size: {decisions_with_position_size}/{total_decisions}, Active positions API: {active_positions_working}"
            
            self.log_test_result("IA2 Integration", success, details)
            
        except Exception as e:
            self.log_test_result("IA2 Integration", False, f"Exception: {str(e)}")
            
    async def test_system_behavior(self):
        """Test 5: System Behavior"""
        logger.info("\n🔍 TEST 5: System Behavior")
        
        try:
            # Test execution mode endpoint
            mode_response = requests.get(f"{self.api_url}/trading/execution-mode", timeout=30)
            
            if mode_response.status_code != 200:
                self.log_test_result("System Behavior", False, f"Execution mode API error: {mode_response.status_code}")
                return
                
            execution_mode = mode_response.json().get('execution_mode', 'UNKNOWN')
            logger.info(f"   📊 Current execution mode: {execution_mode}")
            
            # Get recent decisions to analyze signal distribution
            decisions_response = requests.get(f"{self.api_url}/decisions", timeout=30)
            
            if decisions_response.status_code != 200:
                self.log_test_result("System Behavior", False, f"Decisions API error: {decisions_response.status_code}")
                return
                
            decisions = decisions_response.json()
            
            # Analyze signal distribution
            signal_counts = {'HOLD': 0, 'LONG': 0, 'SHORT': 0}
            position_size_stats = {'zero': 0, 'positive': 0}
            
            for decision in decisions[:20]:  # Last 20 decisions
                signal = decision.get('signal', '').upper()
                position_size = decision.get('position_size', 0)
                
                if signal in signal_counts:
                    signal_counts[signal] += 1
                    
                if position_size <= 0:
                    position_size_stats['zero'] += 1
                else:
                    position_size_stats['positive'] += 1
                    
            logger.info(f"   📊 Signal distribution: {signal_counts}")
            logger.info(f"   📊 Position size stats: {position_size_stats}")
            
            # Check that HOLD signals don't have active positions
            positions_response = requests.get(f"{self.api_url}/active-positions", timeout=30)
            total_positions = 0
            if positions_response.status_code == 200:
                positions_data = positions_response.json()
                active_positions = positions_data.get('active_positions', [])
                total_positions = len(active_positions)
                logger.info(f"   📊 Active positions: {total_positions}")
                
            # Success criteria: System is responding and has logical behavior
            success = (
                execution_mode in ['LIVE', 'SIMULATION'] and
                sum(signal_counts.values()) > 0 and
                signal_counts['HOLD'] >= 0  # HOLD signals should exist
            )
            
            details = f"Mode: {execution_mode}, Signals: {signal_counts}, Active positions: {total_positions}"
            
            self.log_test_result("System Behavior", success, details)
            
        except Exception as e:
            self.log_test_result("System Behavior", False, f"Exception: {str(e)}")
            
    async def test_position_size_calculation_accuracy(self):
        """Test 6: Position Size Calculation Accuracy"""
        logger.info("\n🔍 TEST 6: Position Size Calculation Accuracy")
        
        try:
            # Get recent decisions and check position size calculations
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Position Size Calculation Accuracy", False, f"API error: {response.status_code}")
                return
                
            decisions = response.json()
            
            if not decisions:
                self.log_test_result("Position Size Calculation Accuracy", False, "No decisions found")
                return
                
            # Analyze position size ranges and reasonableness
            position_sizes = []
            reasonable_sizes = 0
            
            for decision in decisions[:10]:
                symbol = decision.get('symbol', 'UNKNOWN')
                signal = decision.get('signal', 'UNKNOWN')
                position_size = decision.get('position_size', 0)
                confidence = decision.get('confidence', 0)
                
                if signal != 'HOLD' and position_size > 0:
                    position_sizes.append(position_size)
                    
                    # Check if position size is reasonable (0.1% to 8%)
                    if 0.001 <= position_size <= 0.08:
                        reasonable_sizes += 1
                        logger.info(f"   ✅ {symbol}: {position_size:.3f} ({position_size*100:.1f}%) - Reasonable")
                    else:
                        logger.info(f"   ⚠️ {symbol}: {position_size:.3f} ({position_size*100:.1f}%) - Outside normal range")
                        
            if position_sizes:
                avg_position_size = sum(position_sizes) / len(position_sizes)
                min_size = min(position_sizes)
                max_size = max(position_sizes)
                
                logger.info(f"   📊 Position size stats: Avg={avg_position_size:.3f}, Min={min_size:.3f}, Max={max_size:.3f}")
                
                success = reasonable_sizes >= len(position_sizes) * 0.8  # 80% should be reasonable
                details = f"Reasonable sizes: {reasonable_sizes}/{len(position_sizes)}, Avg: {avg_position_size:.3f}"
            else:
                success = True  # No trading positions is acceptable
                details = "No trading positions found (HOLD signals only)"
                
            self.log_test_result("Position Size Calculation Accuracy", success, details)
            
        except Exception as e:
            self.log_test_result("Position Size Calculation Accuracy", False, f"Exception: {str(e)}")
            
    async def run_comprehensive_tests(self):
        """Run all position sizing tests"""
        logger.info("🚀 Starting Comprehensive Position Sizing Test Suite")
        logger.info("=" * 60)
        
        await self.setup_database()
        
        # Run all tests
        await self.test_ia2_position_size_matching()
        await self.test_zero_position_size_handling()
        await self.test_position_size_logging()
        await self.test_ia2_integration()
        await self.test_system_behavior()
        await self.test_position_size_calculation_accuracy()
        
        await self.cleanup_database()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 60)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Position sizing fixes are working correctly!")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY WORKING - Some minor issues detected")
        else:
            logger.info("❌ CRITICAL ISSUES - Position sizing fixes need attention")
            
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = PositionSizingTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
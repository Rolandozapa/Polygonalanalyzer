#!/usr/bin/env python3
"""
Backend Testing Suite for Enhanced Technical Pattern Detection System
Focus: New Pattern Types, Advanced Patterns, Pattern Quality, System Integration
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List
import subprocess

# Add backend to path
sys.path.append('/app/backend')

import requests
from motor.motor_asyncio import AsyncIOMotorClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedPatternDetectionTestSuite:
    """Test suite for Enhanced Technical Pattern Detection System"""
    
    def __init__(self):
        # Use localhost for internal testing
        self.base_url = "http://localhost:8001"
        
        self.api_url = f"{self.base_url}/api"
        logger.info(f"Testing Enhanced Pattern Detection at: {self.api_url}")
        
        # MongoDB connection for direct data access
        self.mongo_client = None
        self.db = None
        
        # Test results
        self.test_results = []
        
        # Expected new pattern types from review request
        self.new_pattern_types = [
            'triple_top', 'triple_bottom', 'symmetrical_triangle', 
            'pennant_bullish', 'pennant_bearish', 'rectangle_consolidation',
            'rounding_top', 'rounding_bottom'
        ]
        
        # Expected advanced pattern types
        self.advanced_pattern_types = [
            'gartley_bullish', 'gartley_bearish', 'bat_bullish', 'bat_bearish',
            'butterfly_bullish', 'butterfly_bearish', 'crab_bullish', 'crab_bearish',
            'diamond_top', 'diamond_bottom', 'expanding_wedge'
        ]
        
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
        
    async def test_new_pattern_types(self):
        """Test 1: Verify New Pattern Types Detection"""
        logger.info("\n🔍 TEST 1: New Pattern Types Detection")
        
        try:
            # Get recent IA1 analyses to check for new patterns
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("New Pattern Types Detection", False, f"API error: {response.status_code}")
                return
                
            data = response.json()
            
            # Handle API response format
            if isinstance(data, dict) and 'analyses' in data:
                analyses = data['analyses']
            else:
                analyses = data
            
            if not analyses:
                self.log_test_result("New Pattern Types Detection", False, "No analyses found")
                return
                
            # Check for new pattern types in analyses
            new_patterns_found = set()
            total_analyses = len(analyses)
            
            for analysis in analyses[:20]:  # Check last 20 analyses
                symbol = analysis.get('symbol', 'UNKNOWN')
                patterns_detected = analysis.get('patterns_detected', [])
                
                logger.info(f"   📊 {symbol}: Patterns detected: {patterns_detected}")
                
                # Check for new pattern types
                for pattern in patterns_detected:
                    pattern_lower = pattern.lower().replace(' ', '_').replace('-', '_')
                    for new_pattern in self.new_pattern_types:
                        if new_pattern in pattern_lower or pattern_lower in new_pattern:
                            new_patterns_found.add(new_pattern)
                            logger.info(f"      ✅ Found new pattern: {pattern} -> {new_pattern}")
                            
            success = len(new_patterns_found) >= 3  # At least 3 new pattern types should be detected
            details = f"New patterns found: {list(new_patterns_found)} ({len(new_patterns_found)}/{len(self.new_pattern_types)})"
            
            self.log_test_result("New Pattern Types Detection", success, details)
            
        except Exception as e:
            self.log_test_result("New Pattern Types Detection", False, f"Exception: {str(e)}")
            
    async def test_advanced_patterns(self):
        """Test 2: Advanced Patterns (Harmonic, Diamond, Expanding Wedge)"""
        logger.info("\n🔍 TEST 2: Advanced Patterns Detection")
        
        try:
            # Get recent analyses and check for advanced patterns
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Advanced Patterns Detection", False, f"API error: {response.status_code}")
                return
                
            data = response.json()
            
            # Handle API response format
            if isinstance(data, dict) and 'analyses' in data:
                analyses = data['analyses']
            else:
                analyses = data
            
            if not analyses:
                self.log_test_result("Advanced Patterns Detection", False, "No analyses found")
                return
                
            # Check for advanced pattern types
            advanced_patterns_found = set()
            harmonic_patterns_found = 0
            
            for analysis in analyses[:25]:  # Check last 25 analyses
                symbol = analysis.get('symbol', 'UNKNOWN')
                patterns_detected = analysis.get('patterns_detected', [])
                reasoning = analysis.get('ia1_reasoning', '').lower()
                
                # Check for advanced pattern keywords in reasoning and patterns
                for pattern in patterns_detected:
                    pattern_lower = pattern.lower().replace(' ', '_').replace('-', '_')
                    for advanced_pattern in self.advanced_pattern_types:
                        if advanced_pattern in pattern_lower or pattern_lower in advanced_pattern:
                            advanced_patterns_found.add(advanced_pattern)
                            if 'gartley' in advanced_pattern or 'bat' in advanced_pattern or 'butterfly' in advanced_pattern or 'crab' in advanced_pattern:
                                harmonic_patterns_found += 1
                            logger.info(f"      ✅ Found advanced pattern: {pattern} -> {advanced_pattern}")
                
                # Check reasoning for harmonic pattern keywords
                harmonic_keywords = ['gartley', 'bat', 'butterfly', 'crab', 'harmonic', 'fibonacci retracement']
                for keyword in harmonic_keywords:
                    if keyword in reasoning:
                        logger.info(f"      🎯 Found harmonic keyword in {symbol}: {keyword}")
                        
            success = len(advanced_patterns_found) >= 2 or harmonic_patterns_found >= 1
            details = f"Advanced patterns: {list(advanced_patterns_found)}, Harmonic patterns: {harmonic_patterns_found}"
            
            self.log_test_result("Advanced Patterns Detection", success, details)
            
        except Exception as e:
            self.log_test_result("Advanced Patterns Detection", False, f"Exception: {str(e)}")
            
    async def test_pattern_quality(self):
        """Test 3: Pattern Quality (Confidence, Strength, Trading Direction)"""
        logger.info("\n🔍 TEST 3: Pattern Quality Verification")
        
        try:
            # Get recent analyses to check pattern quality metrics
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Pattern Quality Verification", False, f"API error: {response.status_code}")
                return
                
            analyses = response.json()
            
            if not analyses:
                self.log_test_result("Pattern Quality Verification", False, "No analyses found")
                return
                
            # Analyze pattern quality metrics
            confidence_scores = []
            reasonable_confidences = 0
            trading_directions_found = 0
            
            for analysis in analyses[:15]:  # Check last 15 analyses
                symbol = analysis.get('symbol', 'UNKNOWN')
                confidence = analysis.get('analysis_confidence', 0)
                ia1_signal = analysis.get('ia1_signal', 'hold')
                patterns_detected = analysis.get('patterns_detected', [])
                
                confidence_scores.append(confidence)
                
                # Check if confidence is in reasonable range (0.7-0.9)
                if 0.7 <= confidence <= 0.9:
                    reasonable_confidences += 1
                    logger.info(f"   ✅ {symbol}: Confidence {confidence:.2f} - Reasonable")
                else:
                    logger.info(f"   ⚠️ {symbol}: Confidence {confidence:.2f} - Outside 0.7-0.9 range")
                
                # Check trading direction assignment
                if ia1_signal in ['long', 'short']:
                    trading_directions_found += 1
                    logger.info(f"   📊 {symbol}: Trading direction: {ia1_signal.upper()}")
                    
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                min_confidence = min(confidence_scores)
                max_confidence = max(confidence_scores)
                
                logger.info(f"   📊 Confidence stats: Avg={avg_confidence:.2f}, Min={min_confidence:.2f}, Max={max_confidence:.2f}")
                
                success = (
                    reasonable_confidences >= len(confidence_scores) * 0.6 and  # 60% should be in 0.7-0.9 range
                    trading_directions_found >= len(confidence_scores) * 0.3 and  # 30% should have trading directions
                    avg_confidence >= 0.7  # Average confidence should be >= 0.7
                )
                details = f"Reasonable confidences: {reasonable_confidences}/{len(confidence_scores)}, Trading directions: {trading_directions_found}, Avg confidence: {avg_confidence:.2f}"
            else:
                success = False
                details = "No confidence scores found"
                
            self.log_test_result("Pattern Quality Verification", success, details)
            
        except Exception as e:
            self.log_test_result("Pattern Quality Verification", False, f"Exception: {str(e)}")
            
    async def test_system_integration(self):
        """Test 4: System Integration (TechnicalPatternDetector initialization, filtering)"""
        logger.info("\n🔍 TEST 4: System Integration")
        
        try:
            # Test if system is generating opportunities and analyses
            opportunities_response = requests.get(f"{self.api_url}/opportunities", timeout=30)
            analyses_response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            opportunities_working = opportunities_response.status_code == 200
            analyses_working = analyses_response.status_code == 200
            
            if not opportunities_working:
                self.log_test_result("System Integration", False, f"Opportunities API error: {opportunities_response.status_code}")
                return
                
            if not analyses_working:
                self.log_test_result("System Integration", False, f"Analyses API error: {analyses_response.status_code}")
                return
                
            opportunities = opportunities_response.json()
            analyses = analyses_response.json()
            
            # Check pattern filtering and deduplication
            total_opportunities = len(opportunities)
            total_analyses = len(analyses)
            
            logger.info(f"   📊 Total opportunities: {total_opportunities}")
            logger.info(f"   📊 Total analyses: {total_analyses}")
            
            # Check if maximum 5 strongest patterns are returned per symbol
            symbol_pattern_counts = {}
            for analysis in analyses[:10]:
                symbol = analysis.get('symbol', 'UNKNOWN')
                patterns = analysis.get('patterns_detected', [])
                symbol_pattern_counts[symbol] = len(patterns)
                
            max_patterns_per_symbol = max(symbol_pattern_counts.values()) if symbol_pattern_counts else 0
            symbols_with_reasonable_patterns = sum(1 for count in symbol_pattern_counts.values() if count <= 5)
            
            logger.info(f"   📊 Max patterns per symbol: {max_patterns_per_symbol}")
            logger.info(f"   📊 Symbols with ≤5 patterns: {symbols_with_reasonable_patterns}/{len(symbol_pattern_counts)}")
            
            # Check if TechnicalPatternDetector is working (should have some analyses with patterns)
            analyses_with_patterns = sum(1 for analysis in analyses if analysis.get('patterns_detected', []))
            
            success = (
                total_opportunities > 0 and
                total_analyses > 0 and
                max_patterns_per_symbol <= 5 and  # Max 5 patterns per symbol
                analyses_with_patterns >= total_analyses * 0.5  # At least 50% should have patterns
            )
            
            details = f"Opportunities: {total_opportunities}, Analyses: {total_analyses}, Max patterns/symbol: {max_patterns_per_symbol}, Analyses with patterns: {analyses_with_patterns}"
            
            self.log_test_result("System Integration", success, details)
            
        except Exception as e:
            self.log_test_result("System Integration", False, f"Exception: {str(e)}")
            
    async def test_backend_logs(self):
        """Test 5: Backend Logs for Pattern Detection"""
        logger.info("\n🔍 TEST 5: Backend Logs Analysis")
        
        try:
            # Check backend logs for pattern detection messages
            log_cmd = "tail -n 500 /var/log/supervisor/backend.*.log 2>/dev/null || echo 'No logs found'"
            result = subprocess.run(log_cmd, shell=True, capture_output=True, text=True)
            
            backend_logs = result.stdout
            
            # Look for pattern detection log patterns
            pattern_detection_logs = []
            strong_pattern_logs = []
            no_import_errors = True
            
            for line in backend_logs.split('\n'):
                if 'Detected' in line and 'strong patterns for' in line:
                    pattern_detection_logs.append(line.strip())
                elif 'TECHNICAL FILTER' in line and 'SENDING TO IA1' in line:
                    strong_pattern_logs.append(line.strip())
                elif 'ImportError' in line or 'ModuleNotFoundError' in line:
                    no_import_errors = False
                    logger.error(f"   ❌ Import error found: {line.strip()}")
                    
            logger.info(f"   📊 Found {len(pattern_detection_logs)} pattern detection logs")
            logger.info(f"   📊 Found {len(strong_pattern_logs)} strong pattern logs")
            
            # Show sample logs
            if pattern_detection_logs:
                logger.info(f"   📝 Sample pattern log: {pattern_detection_logs[-1]}")
            if strong_pattern_logs:
                logger.info(f"   📝 Sample strong pattern log: {strong_pattern_logs[-1]}")
                
            # Check for specific pattern types in logs
            new_pattern_mentions = 0
            for log_line in backend_logs.split('\n'):
                for pattern_type in self.new_pattern_types + self.advanced_pattern_types:
                    if pattern_type in log_line.lower():
                        new_pattern_mentions += 1
                        break
                        
            success = (
                len(pattern_detection_logs) > 0 and
                no_import_errors and
                new_pattern_mentions >= 3  # At least 3 mentions of new/advanced patterns
            )
            
            details = f"Pattern detection logs: {len(pattern_detection_logs)}, Strong pattern logs: {len(strong_pattern_logs)}, No import errors: {no_import_errors}, New pattern mentions: {new_pattern_mentions}"
            
            self.log_test_result("Backend Logs Analysis", success, details)
            
        except Exception as e:
            self.log_test_result("Backend Logs Analysis", False, f"Exception: {str(e)}")
            
    async def test_pattern_variety_multiple_symbols(self):
        """Test 6: Pattern Variety Across Multiple Symbols"""
        logger.info("\n🔍 TEST 6: Pattern Variety Across Multiple Symbols")
        
        try:
            # Get analyses for multiple symbols
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Pattern Variety Test", False, f"API error: {response.status_code}")
                return
                
            analyses = response.json()
            
            if not analyses:
                self.log_test_result("Pattern Variety Test", False, "No analyses found")
                return
                
            # Analyze pattern variety across symbols
            unique_symbols = set()
            all_patterns_found = set()
            symbol_pattern_examples = {}
            
            for analysis in analyses[:30]:  # Check last 30 analyses
                symbol = analysis.get('symbol', 'UNKNOWN')
                patterns_detected = analysis.get('patterns_detected', [])
                confidence = analysis.get('analysis_confidence', 0)
                
                unique_symbols.add(symbol)
                
                for pattern in patterns_detected:
                    all_patterns_found.add(pattern.lower())
                    if symbol not in symbol_pattern_examples:
                        symbol_pattern_examples[symbol] = []
                    symbol_pattern_examples[symbol].append({
                        'pattern': pattern,
                        'confidence': confidence
                    })
                    
            logger.info(f"   📊 Unique symbols analyzed: {len(unique_symbols)}")
            logger.info(f"   📊 Total unique patterns found: {len(all_patterns_found)}")
            
            # Show examples of patterns found per symbol
            examples_shown = 0
            for symbol, patterns in symbol_pattern_examples.items():
                if examples_shown < 5:  # Show first 5 symbols
                    pattern_names = [p['pattern'] for p in patterns]
                    avg_confidence = sum(p['confidence'] for p in patterns) / len(patterns)
                    logger.info(f"   🎯 {symbol}: {pattern_names} (avg confidence: {avg_confidence:.2f})")
                    examples_shown += 1
                    
            # Check for variety in pattern types
            new_patterns_in_results = sum(1 for pattern in all_patterns_found 
                                        for new_pattern in self.new_pattern_types 
                                        if new_pattern.replace('_', ' ') in pattern or pattern in new_pattern)
            
            advanced_patterns_in_results = sum(1 for pattern in all_patterns_found 
                                             for advanced_pattern in self.advanced_pattern_types 
                                             if advanced_pattern.replace('_', ' ') in pattern or pattern in advanced_pattern)
            
            success = (
                len(unique_symbols) >= 5 and  # At least 5 different symbols
                len(all_patterns_found) >= 8 and  # At least 8 different pattern types
                new_patterns_in_results >= 2 and  # At least 2 new pattern types
                advanced_patterns_in_results >= 1  # At least 1 advanced pattern type
            )
            
            details = f"Symbols: {len(unique_symbols)}, Pattern types: {len(all_patterns_found)}, New patterns: {new_patterns_in_results}, Advanced patterns: {advanced_patterns_in_results}"
            
            self.log_test_result("Pattern Variety Test", success, details)
            
        except Exception as e:
            self.log_test_result("Pattern Variety Test", False, f"Exception: {str(e)}")
            
    async def run_comprehensive_tests(self):
        """Run all enhanced pattern detection tests"""
        logger.info("🚀 Starting Enhanced Technical Pattern Detection Test Suite")
        logger.info("=" * 70)
        
        await self.setup_database()
        
        # Run all tests
        await self.test_new_pattern_types()
        await self.test_advanced_patterns()
        await self.test_pattern_quality()
        await self.test_system_integration()
        await self.test_backend_logs()
        await self.test_pattern_variety_multiple_symbols()
        
        await self.cleanup_database()
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("📊 ENHANCED PATTERN DETECTION TEST SUMMARY")
        logger.info("=" * 70)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Enhanced Pattern Detection System is working correctly!")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY WORKING - Some minor issues detected")
        else:
            logger.info("❌ CRITICAL ISSUES - Enhanced Pattern Detection System needs attention")
            
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = EnhancedPatternDetectionTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
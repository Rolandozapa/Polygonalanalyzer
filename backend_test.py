#!/usr/bin/env python3
"""
AUTONOMOUS TREND DETECTION SYSTEM COMPREHENSIVE TEST SUITE
Focus: Test the new autonomous trend detection system with 4h frequency and advanced filters

CRITICAL TEST REQUIREMENTS FROM REVIEW REQUEST:
1. **Trending Auto-Updater**: Test 4h frequency (14400s), min var volume daily 5%, min var price 1%
2. **Lateral Pattern Detector**: Test sophisticated multi-criteria analysis and TrendType classification
3. **Advanced Market Aggregator**: Test get_current_opportunities() with BingX data and 4h cache TTL
4. **Integration Testing**: Verify trending_auto_updater → pattern detector → market aggregator flow
5. **Filter Validation**: Ensure only real trends pass filters (no lateral patterns)
6. **BingX Data Source**: Verify top 50 market cap futures BingX via official API

TESTING APPROACH:
- Test trending_auto_updater.fetch_trending_cryptos() returns 50 cryptos with filters applied
- Test lateral_pattern_detector with edge cases (low volumes, lateral prices)
- Validate advanced_market_aggregator.get_current_opportunities() uses filtered data
- Test complete integration: trending_auto_updater → pattern detector → market aggregator
- Test 4h frequency and cache TTL alignment
- Validate that only real trends pass filters (no lateral patterns)

EXPECTED RESULTS:
- 50 cryptos max with price change ≥1% and sufficient volume
- Effective filtering of lateral figures
- Fresh BingX API data used
- Stable performance with 4h frequency
- Top 50 market cap futures correctly retrieved
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

class AutonomousTrendDetectionTestSuite:
    """Comprehensive test suite for Autonomous Trend Detection System"""
    
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
        logger.info(f"Testing Autonomous Trend Detection System at: {self.api_url}")
        
        # MongoDB connection for direct database analysis
        try:
            self.mongo_client = MongoClient("mongodb://localhost:27017")
            self.db = self.mongo_client["myapp"]
            logger.info("✅ MongoDB connection established for trend analysis")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.mongo_client = None
            self.db = None
        
        # Test results
        self.test_results = []
        
        # Import modules for direct testing
        try:
            sys.path.append('/app/backend')
            from trending_auto_updater import trending_auto_updater
            from lateral_pattern_detector import lateral_pattern_detector, TrendType
            from advanced_market_aggregator import advanced_market_aggregator
            
            self.trending_updater = trending_auto_updater
            self.pattern_detector = lateral_pattern_detector
            self.market_aggregator = advanced_market_aggregator
            logger.info("✅ Successfully imported trend detection modules")
        except Exception as e:
            logger.error(f"❌ Failed to import trend detection modules: {e}")
            self.trending_updater = None
            self.pattern_detector = None
            self.market_aggregator = None
        
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
    
    async def test_1_trending_auto_updater_configuration(self):
        """Test 1: Verify Trending Auto-Updater Configuration (4h frequency, filters)"""
        logger.info("\n🔍 TEST 1: Trending Auto-Updater Configuration Test")
        
        try:
            if not self.trending_updater:
                self.log_test_result("Trending Auto-Updater Configuration", False, 
                                   "Trending updater module not available")
                return
            
            # Check configuration
            config_analysis = {
                'update_interval': self.trending_updater.update_interval,
                'expected_interval': 14400,  # 4 hours
                'bingx_api_configured': bool(self.trending_updater.bingx_api_base),
                'top_futures_count': len(self.trending_updater.bingx_top_futures),
                'expected_futures_count': 50,
                'has_pattern_detector': hasattr(self.trending_updater, 'pattern_detector')
            }
            
            logger.info(f"   📊 Configuration Analysis:")
            logger.info(f"      Update interval: {config_analysis['update_interval']}s (expected: {config_analysis['expected_interval']}s)")
            logger.info(f"      BingX API configured: {config_analysis['bingx_api_configured']}")
            logger.info(f"      Top futures symbols: {config_analysis['top_futures_count']} (expected: {config_analysis['expected_futures_count']})")
            logger.info(f"      Pattern detector integration: {config_analysis['has_pattern_detector']}")
            
            # Verify 4h frequency
            frequency_correct = config_analysis['update_interval'] == config_analysis['expected_interval']
            
            # Verify BingX integration
            bingx_integration = (
                config_analysis['bingx_api_configured'] and 
                config_analysis['top_futures_count'] >= 40  # At least 40 symbols
            )
            
            # Overall configuration score
            config_score = sum([
                frequency_correct,
                bingx_integration,
                config_analysis['has_pattern_detector']
            ])
            
            if config_score >= 3:
                self.log_test_result("Trending Auto-Updater Configuration", True, 
                                   f"Configuration correct: 4h frequency, BingX integration, pattern detector")
            elif config_score >= 2:
                self.log_test_result("Trending Auto-Updater Configuration", False, 
                                   f"Partial configuration: {config_score}/3 requirements met")
            else:
                self.log_test_result("Trending Auto-Updater Configuration", False, 
                                   f"Configuration issues: {config_score}/3 requirements met")
                
        except Exception as e:
            self.log_test_result("Trending Auto-Updater Configuration", False, f"Exception: {str(e)}")
    
    async def test_2_bingx_trending_data_fetch(self):
        """Test 2: BingX Trending Data Fetch with Filters"""
        logger.info("\n🔍 TEST 2: BingX Trending Data Fetch Test")
        
        try:
            if not self.trending_updater:
                self.log_test_result("BingX Trending Data Fetch", False, 
                                   "Trending updater module not available")
                return
            
            logger.info("   🚀 Fetching trending cryptos from BingX...")
            
            # Fetch trending cryptos
            trending_cryptos = await self.trending_updater.fetch_trending_cryptos()
            
            if not trending_cryptos:
                self.log_test_result("BingX Trending Data Fetch", False, 
                                   "No trending cryptos returned")
                return
            
            # Analyze results
            fetch_analysis = {
                'total_cryptos': len(trending_cryptos),
                'expected_max': 50,
                'bingx_api_count': len([c for c in trending_cryptos if c.source == 'bingx_api']),
                'bingx_page_count': len([c for c in trending_cryptos if c.source == 'bingx_page']),
                'fallback_count': len([c for c in trending_cryptos if c.source == 'bingx_fallback']),
                'with_price_change': len([c for c in trending_cryptos if c.price_change and abs(c.price_change) >= 1.0]),
                'with_volume': len([c for c in trending_cryptos if c.volume and c.volume >= 500000]),
                'unique_symbols': len(set(c.symbol for c in trending_cryptos))
            }
            
            logger.info(f"   📊 Fetch Analysis:")
            logger.info(f"      Total cryptos: {fetch_analysis['total_cryptos']} (max expected: {fetch_analysis['expected_max']})")
            logger.info(f"      BingX API source: {fetch_analysis['bingx_api_count']}")
            logger.info(f"      BingX page source: {fetch_analysis['bingx_page_count']}")
            logger.info(f"      Fallback source: {fetch_analysis['fallback_count']}")
            logger.info(f"      With price change ≥1%: {fetch_analysis['with_price_change']}")
            logger.info(f"      With volume ≥500K: {fetch_analysis['with_volume']}")
            logger.info(f"      Unique symbols: {fetch_analysis['unique_symbols']}")
            
            # Log top 10 trending cryptos
            logger.info(f"   📈 Top 10 Trending Cryptos:")
            for i, crypto in enumerate(trending_cryptos[:10]):
                price_str = f", {crypto.price_change:+.1f}%" if crypto.price_change else ""
                volume_str = f", Vol: {crypto.volume/1_000_000:.1f}M" if crypto.volume else ""
                logger.info(f"      {i+1}. {crypto.symbol} ({crypto.source}{price_str}{volume_str})")
            
            # Determine test result
            filters_working = (
                fetch_analysis['with_price_change'] >= fetch_analysis['total_cryptos'] * 0.7 and  # 70% have ≥1% change
                fetch_analysis['with_volume'] >= fetch_analysis['total_cryptos'] * 0.7  # 70% have sufficient volume
            )
            
            data_quality = (
                fetch_analysis['total_cryptos'] >= 10 and  # At least 10 cryptos
                fetch_analysis['unique_symbols'] == fetch_analysis['total_cryptos'] and  # No duplicates
                (fetch_analysis['bingx_api_count'] > 0 or fetch_analysis['bingx_page_count'] > 0)  # Real BingX data
            )
            
            if filters_working and data_quality:
                self.log_test_result("BingX Trending Data Fetch", True, 
                                   f"Filters working: {fetch_analysis['total_cryptos']} cryptos, {fetch_analysis['with_price_change']} with ≥1% change")
            elif data_quality:
                self.log_test_result("BingX Trending Data Fetch", False, 
                                   f"Data quality good but filters need improvement: {fetch_analysis['with_price_change']}/{fetch_analysis['total_cryptos']} with ≥1% change")
            else:
                self.log_test_result("BingX Trending Data Fetch", False, 
                                   f"Data quality issues: {fetch_analysis['total_cryptos']} cryptos, {fetch_analysis['unique_symbols']} unique")
                
        except Exception as e:
            self.log_test_result("BingX Trending Data Fetch", False, f"Exception: {str(e)}")
    
    async def test_3_lateral_pattern_detector_analysis(self):
        """Test 3: Lateral Pattern Detector Multi-Criteria Analysis"""
        logger.info("\n🔍 TEST 3: Lateral Pattern Detector Analysis Test")
        
        try:
            if not self.pattern_detector:
                self.log_test_result("Lateral Pattern Detector Analysis", False, 
                                   "Pattern detector module not available")
                return
            
            # Test cases for pattern detection
            test_cases = [
                # Strong bullish (should NOT be filtered)
                {"symbol": "BTCUSDT", "price_change": 8.5, "volume": 5000000, "expected_lateral": False, "expected_type": TrendType.STRONG_BULLISH},
                # Bullish (should NOT be filtered)
                {"symbol": "ETHUSDT", "price_change": 3.2, "volume": 2000000, "expected_lateral": False, "expected_type": TrendType.BULLISH},
                # Lateral pattern (should be filtered)
                {"symbol": "XRPUSDT", "price_change": 0.5, "volume": 100000, "expected_lateral": True, "expected_type": TrendType.LATERAL},
                # Strong bearish (should NOT be filtered)
                {"symbol": "SOLUSDT", "price_change": -7.8, "volume": 3000000, "expected_lateral": False, "expected_type": TrendType.STRONG_BEARISH},
                # Bearish (should NOT be filtered)
                {"symbol": "ADAUSDT", "price_change": -2.5, "volume": 1500000, "expected_lateral": False, "expected_type": TrendType.BEARISH},
                # Edge case: High volume but low price change (might be lateral)
                {"symbol": "DOGEUSDT", "price_change": 0.8, "volume": 10000000, "expected_lateral": True, "expected_type": TrendType.LATERAL}
            ]
            
            analysis_results = {
                'total_tests': len(test_cases),
                'correct_classifications': 0,
                'correct_lateral_detections': 0,
                'correct_filter_decisions': 0,
                'trend_strength_scores': [],
                'confidence_scores': []
            }
            
            logger.info(f"   📊 Testing {len(test_cases)} pattern detection scenarios:")
            
            for i, test_case in enumerate(test_cases):
                try:
                    # Analyze pattern
                    analysis = self.pattern_detector.analyze_trend_pattern(
                        symbol=test_case['symbol'],
                        price_change_pct=test_case['price_change'],
                        volume=test_case['volume']
                    )
                    
                    # Check classification accuracy
                    classification_correct = analysis.trend_type == test_case['expected_type']
                    lateral_detection_correct = analysis.is_lateral == test_case['expected_lateral']
                    
                    # Check filter decision
                    should_filter = self.pattern_detector.should_filter_opportunity(analysis)
                    filter_decision_correct = should_filter == test_case['expected_lateral']
                    
                    # Update results
                    if classification_correct:
                        analysis_results['correct_classifications'] += 1
                    if lateral_detection_correct:
                        analysis_results['correct_lateral_detections'] += 1
                    if filter_decision_correct:
                        analysis_results['correct_filter_decisions'] += 1
                    
                    analysis_results['trend_strength_scores'].append(analysis.trend_strength)
                    analysis_results['confidence_scores'].append(analysis.confidence)
                    
                    # Log result
                    status = "✅" if (classification_correct and lateral_detection_correct and filter_decision_correct) else "❌"
                    logger.info(f"      {status} {test_case['symbol']}: {analysis.trend_type.value}, "
                              f"lateral={analysis.is_lateral}, filter={should_filter}, "
                              f"strength={analysis.trend_strength:.2f}, conf={analysis.confidence:.2f}")
                    logger.info(f"         Reasoning: {analysis.reasoning}")
                    
                except Exception as e:
                    logger.warning(f"      ❌ {test_case['symbol']}: Analysis failed - {e}")
            
            # Calculate performance metrics
            classification_accuracy = analysis_results['correct_classifications'] / analysis_results['total_tests']
            lateral_detection_accuracy = analysis_results['correct_lateral_detections'] / analysis_results['total_tests']
            filter_accuracy = analysis_results['correct_filter_decisions'] / analysis_results['total_tests']
            avg_trend_strength = sum(analysis_results['trend_strength_scores']) / len(analysis_results['trend_strength_scores']) if analysis_results['trend_strength_scores'] else 0
            avg_confidence = sum(analysis_results['confidence_scores']) / len(analysis_results['confidence_scores']) if analysis_results['confidence_scores'] else 0
            
            logger.info(f"   📊 Pattern Detection Performance:")
            logger.info(f"      Classification accuracy: {classification_accuracy:.1%}")
            logger.info(f"      Lateral detection accuracy: {lateral_detection_accuracy:.1%}")
            logger.info(f"      Filter decision accuracy: {filter_accuracy:.1%}")
            logger.info(f"      Average trend strength: {avg_trend_strength:.2f}")
            logger.info(f"      Average confidence: {avg_confidence:.2f}")
            
            # Determine test result
            if classification_accuracy >= 0.8 and lateral_detection_accuracy >= 0.8 and filter_accuracy >= 0.8:
                self.log_test_result("Lateral Pattern Detector Analysis", True, 
                                   f"Pattern detection working: {classification_accuracy:.1%} classification, {filter_accuracy:.1%} filter accuracy")
            elif classification_accuracy >= 0.6 and filter_accuracy >= 0.6:
                self.log_test_result("Lateral Pattern Detector Analysis", False, 
                                   f"Partial pattern detection: {classification_accuracy:.1%} classification, {filter_accuracy:.1%} filter accuracy")
            else:
                self.log_test_result("Lateral Pattern Detector Analysis", False, 
                                   f"Pattern detection issues: {classification_accuracy:.1%} classification, {filter_accuracy:.1%} filter accuracy")
                
        except Exception as e:
            self.log_test_result("Lateral Pattern Detector Analysis", False, f"Exception: {str(e)}")
    
    async def test_4_advanced_market_aggregator_integration(self):
        """Test 4: Advanced Market Aggregator Integration with BingX Data"""
        logger.info("\n🔍 TEST 4: Advanced Market Aggregator Integration Test")
        
        try:
            if not self.market_aggregator:
                self.log_test_result("Advanced Market Aggregator Integration", False, 
                                   "Market aggregator module not available")
                return
            
            logger.info("   🚀 Testing get_current_opportunities() with BingX integration...")
            
            # Test get_current_opportunities
            opportunities = self.market_aggregator.get_current_opportunities()
            
            if not opportunities:
                self.log_test_result("Advanced Market Aggregator Integration", False, 
                                   "No opportunities returned from market aggregator")
                return
            
            # Analyze opportunities
            integration_analysis = {
                'total_opportunities': len(opportunities),
                'expected_range': (10, 50),
                'bingx_sources': len([opp for opp in opportunities if 'bingx' in str(opp.data_sources).lower()]),
                'with_volume': len([opp for opp in opportunities if opp.volume_24h > 0]),
                'with_price_change': len([opp for opp in opportunities if abs(opp.price_change_24h) > 0]),
                'high_confidence': len([opp for opp in opportunities if opp.data_confidence >= 0.7]),
                'unique_symbols': len(set(opp.symbol for opp in opportunities))
            }
            
            logger.info(f"   📊 Integration Analysis:")
            logger.info(f"      Total opportunities: {integration_analysis['total_opportunities']} (expected: {integration_analysis['expected_range'][0]}-{integration_analysis['expected_range'][1]})")
            logger.info(f"      BingX sources: {integration_analysis['bingx_sources']}")
            logger.info(f"      With volume data: {integration_analysis['with_volume']}")
            logger.info(f"      With price change: {integration_analysis['with_price_change']}")
            logger.info(f"      High confidence (≥0.7): {integration_analysis['high_confidence']}")
            logger.info(f"      Unique symbols: {integration_analysis['unique_symbols']}")
            
            # Log top 10 opportunities
            logger.info(f"   📈 Top 10 Market Opportunities:")
            for i, opp in enumerate(opportunities[:10]):
                sources_str = f", Sources: {opp.data_sources}" if opp.data_sources else ""
                confidence_str = f", Conf: {opp.data_confidence:.1f}" if opp.data_confidence else ""
                logger.info(f"      {i+1}. {opp.symbol}: ${opp.current_price:.6f}, "
                          f"{opp.price_change_24h:+.1f}%, Vol: {opp.volume_24h/1_000_000:.1f}M{confidence_str}{sources_str}")
            
            # Test cache TTL alignment (check if cache is working)
            logger.info("   🔄 Testing cache TTL alignment...")
            start_time = time.time()
            opportunities_2 = self.market_aggregator.get_current_opportunities()
            cache_time = time.time() - start_time
            
            cache_working = cache_time < 0.1  # Should be very fast if cached
            same_data = len(opportunities) == len(opportunities_2)  # Should be same data if cached
            
            logger.info(f"      Cache response time: {cache_time:.3f}s")
            logger.info(f"      Same data returned: {same_data}")
            logger.info(f"      Cache working: {cache_working}")
            
            # Determine test result
            data_quality = (
                integration_analysis['expected_range'][0] <= integration_analysis['total_opportunities'] <= integration_analysis['expected_range'][1] and
                integration_analysis['unique_symbols'] == integration_analysis['total_opportunities'] and
                integration_analysis['with_volume'] >= integration_analysis['total_opportunities'] * 0.8
            )
            
            bingx_integration = integration_analysis['bingx_sources'] > 0 or integration_analysis['high_confidence'] >= integration_analysis['total_opportunities'] * 0.5
            
            if data_quality and bingx_integration and cache_working:
                self.log_test_result("Advanced Market Aggregator Integration", True, 
                                   f"Integration working: {integration_analysis['total_opportunities']} opportunities, {integration_analysis['bingx_sources']} BingX sources, cache functional")
            elif data_quality and bingx_integration:
                self.log_test_result("Advanced Market Aggregator Integration", False, 
                                   f"Integration mostly working: {integration_analysis['total_opportunities']} opportunities, cache issues")
            else:
                self.log_test_result("Advanced Market Aggregator Integration", False, 
                                   f"Integration issues: {integration_analysis['total_opportunities']} opportunities, {integration_analysis['bingx_sources']} BingX sources")
                
        except Exception as e:
            self.log_test_result("Advanced Market Aggregator Integration", False, f"Exception: {str(e)}")
    
    async def test_5_complete_system_integration_flow(self):
        """Test 5: Complete System Integration Flow (trending → pattern detector → market aggregator)"""
        logger.info("\n🔍 TEST 5: Complete System Integration Flow Test")
        
        try:
            if not all([self.trending_updater, self.pattern_detector, self.market_aggregator]):
                self.log_test_result("Complete System Integration Flow", False, 
                                   "One or more system modules not available")
                return
            
            logger.info("   🚀 Testing complete integration flow...")
            
            # Step 1: Get trending data
            logger.info("   📈 Step 1: Fetching trending cryptos...")
            trending_cryptos = await self.trending_updater.fetch_trending_cryptos()
            
            if not trending_cryptos:
                self.log_test_result("Complete System Integration Flow", False, 
                                   "No trending cryptos from step 1")
                return
            
            # Step 2: Apply pattern detection filters
            logger.info("   🔍 Step 2: Applying pattern detection filters...")
            filtered_cryptos = []
            filter_stats = {'total': len(trending_cryptos), 'filtered_out': 0, 'passed': 0}
            
            for crypto in trending_cryptos:
                if crypto.price_change is not None and crypto.volume is not None:
                    analysis = self.pattern_detector.analyze_trend_pattern(
                        symbol=crypto.symbol,
                        price_change_pct=crypto.price_change,
                        volume=crypto.volume
                    )
                    
                    if not self.pattern_detector.should_filter_opportunity(analysis):
                        filtered_cryptos.append(crypto)
                        filter_stats['passed'] += 1
                    else:
                        filter_stats['filtered_out'] += 1
                else:
                    # Keep cryptos without price/volume data for now
                    filtered_cryptos.append(crypto)
                    filter_stats['passed'] += 1
            
            logger.info(f"      Filter results: {filter_stats['passed']}/{filter_stats['total']} passed, {filter_stats['filtered_out']} filtered out")
            
            # Step 3: Get market opportunities
            logger.info("   📊 Step 3: Getting market opportunities...")
            opportunities = self.market_aggregator.get_current_opportunities()
            
            # Step 4: Analyze integration
            logger.info("   🔗 Step 4: Analyzing integration...")
            
            integration_analysis = {
                'trending_count': len(trending_cryptos),
                'filtered_count': len(filtered_cryptos),
                'opportunities_count': len(opportunities),
                'filter_effectiveness': filter_stats['filtered_out'] / filter_stats['total'] if filter_stats['total'] > 0 else 0,
                'data_flow_working': len(filtered_cryptos) > 0 and len(opportunities) > 0
            }
            
            # Check symbol overlap between trending and opportunities
            trending_symbols = set(crypto.symbol for crypto in trending_cryptos)
            opportunity_symbols = set(opp.symbol for opp in opportunities)
            symbol_overlap = len(trending_symbols.intersection(opportunity_symbols))
            overlap_percentage = symbol_overlap / len(trending_symbols) if trending_symbols else 0
            
            integration_analysis['symbol_overlap'] = symbol_overlap
            integration_analysis['overlap_percentage'] = overlap_percentage
            
            logger.info(f"   📊 Integration Flow Analysis:")
            logger.info(f"      Trending cryptos: {integration_analysis['trending_count']}")
            logger.info(f"      After filtering: {integration_analysis['filtered_count']}")
            logger.info(f"      Market opportunities: {integration_analysis['opportunities_count']}")
            logger.info(f"      Filter effectiveness: {integration_analysis['filter_effectiveness']:.1%}")
            logger.info(f"      Symbol overlap: {integration_analysis['symbol_overlap']} ({integration_analysis['overlap_percentage']:.1%})")
            logger.info(f"      Data flow working: {integration_analysis['data_flow_working']}")
            
            # Show some examples of the flow
            logger.info(f"   📋 Integration Flow Examples:")
            for i, crypto in enumerate(filtered_cryptos[:5]):
                price_str = f", {crypto.price_change:+.1f}%" if crypto.price_change else ""
                volume_str = f", Vol: {crypto.volume/1_000_000:.1f}M" if crypto.volume else ""
                logger.info(f"      {i+1}. {crypto.symbol} (trending → filtered → opportunities{price_str}{volume_str})")
            
            # Determine test result
            flow_working = (
                integration_analysis['data_flow_working'] and
                integration_analysis['filter_effectiveness'] > 0.1 and  # At least 10% filtering
                integration_analysis['overlap_percentage'] > 0.3  # At least 30% symbol overlap
            )
            
            if flow_working:
                self.log_test_result("Complete System Integration Flow", True, 
                                   f"Integration flow working: {integration_analysis['trending_count']} → {integration_analysis['filtered_count']} → {integration_analysis['opportunities_count']}, {integration_analysis['overlap_percentage']:.1%} overlap")
            else:
                self.log_test_result("Complete System Integration Flow", False, 
                                   f"Integration flow issues: {integration_analysis['trending_count']} → {integration_analysis['filtered_count']} → {integration_analysis['opportunities_count']}, {integration_analysis['overlap_percentage']:.1%} overlap")
                
        except Exception as e:
            self.log_test_result("Complete System Integration Flow", False, f"Exception: {str(e)}")
    
    async def test_6_system_performance_and_stability(self):
        """Test 6: System Performance & Stability with 4h Frequency"""
        logger.info("\n🔍 TEST 6: System Performance & Stability Test")
        
        try:
            # Check CPU usage
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                logger.info(f"   📊 System Resources:")
                logger.info(f"      CPU Usage: {cpu_percent:.1f}%")
                logger.info(f"      Memory Usage: {memory_percent:.1f}%")
                
                cpu_stable = cpu_percent < 90.0  # CPU under 90%
                memory_stable = memory_percent < 85.0  # Memory under 85%
                
            except ImportError:
                logger.info("   ⚠️ psutil not available, skipping resource monitoring")
                cpu_stable = True
                memory_stable = True
            
            # Test trending updater performance
            performance_analysis = {
                'cpu_stable': cpu_stable,
                'memory_stable': memory_stable,
                'trending_updater_responsive': False,
                'pattern_detector_responsive': False,
                'market_aggregator_responsive': False
            }
            
            # Test trending updater responsiveness
            if self.trending_updater:
                try:
                    start_time = time.time()
                    trending_info = self.trending_updater.get_trending_info()
                    response_time = time.time() - start_time
                    
                    performance_analysis['trending_updater_responsive'] = response_time < 1.0
                    logger.info(f"      Trending updater response time: {response_time:.3f}s")
                    logger.info(f"      Trending info: {trending_info.get('trending_count', 0)} symbols, "
                              f"auto-update: {trending_info.get('auto_update_active', False)}")
                except Exception as e:
                    logger.warning(f"      Trending updater test failed: {e}")
            
            # Test pattern detector responsiveness
            if self.pattern_detector:
                try:
                    start_time = time.time()
                    test_analysis = self.pattern_detector.analyze_trend_pattern("TESTUSDT", 2.5, 1000000)
                    response_time = time.time() - start_time
                    
                    performance_analysis['pattern_detector_responsive'] = response_time < 0.1
                    logger.info(f"      Pattern detector response time: {response_time:.3f}s")
                except Exception as e:
                    logger.warning(f"      Pattern detector test failed: {e}")
            
            # Test market aggregator responsiveness
            if self.market_aggregator:
                try:
                    start_time = time.time()
                    opportunities = self.market_aggregator.get_current_opportunities()
                    response_time = time.time() - start_time
                    
                    performance_analysis['market_aggregator_responsive'] = response_time < 2.0
                    logger.info(f"      Market aggregator response time: {response_time:.3f}s")
                    logger.info(f"      Opportunities returned: {len(opportunities)}")
                except Exception as e:
                    logger.warning(f"      Market aggregator test failed: {e}")
            
            # Check backend logs for errors
            error_analysis = await self._analyze_backend_logs()
            
            # Overall stability assessment
            stable_components = sum(1 for component, stable in performance_analysis.items() if stable)
            total_components = len(performance_analysis)
            
            logger.info(f"   📊 Performance Analysis:")
            for component, stable in performance_analysis.items():
                status = "✅" if stable else "❌"
                logger.info(f"      {status} {component.replace('_', ' ').title()}")
            
            logger.info(f"   📊 Error Analysis:")
            logger.info(f"      Total log entries: {error_analysis['total_entries']}")
            logger.info(f"      Error entries: {error_analysis['error_entries']}")
            logger.info(f"      Error rate: {error_analysis['error_rate']:.1%}")
            logger.info(f"      Critical errors: {error_analysis['critical_errors']}")
            
            # Determine test result
            if stable_components >= total_components * 0.8 and error_analysis['error_rate'] < 0.1:
                self.log_test_result("System Performance & Stability", True, 
                                   f"System stable: {stable_components}/{total_components} components responsive, {error_analysis['error_rate']:.1%} error rate")
            elif stable_components >= total_components * 0.6:
                self.log_test_result("System Performance & Stability", False, 
                                   f"System mostly stable: {stable_components}/{total_components} components responsive, {error_analysis['error_rate']:.1%} error rate")
            else:
                self.log_test_result("System Performance & Stability", False, 
                                   f"System stability issues: {stable_components}/{total_components} components responsive, {error_analysis['error_rate']:.1%} error rate")
                
        except Exception as e:
            self.log_test_result("System Performance & Stability", False, f"Exception: {str(e)}")
    
    async def _analyze_backend_logs(self):
        """Analyze backend logs for error patterns and quality"""
        try:
            log_files = [
                "/var/log/supervisor/backend.out.log",
                "/var/log/supervisor/backend.err.log"
            ]
            
            analysis = {
                'total_entries': 0,
                'error_entries': 0,
                'critical_errors': 0,
                'error_rate': 0.0,
                'trending_related_entries': 0,
                'recent_errors': []
            }
            
            error_patterns = [
                r'ERROR',
                r'CRITICAL',
                r'Exception',
                r'Traceback',
                r'Failed to fetch',
                r'Connection error'
            ]
            
            trending_patterns = [
                r'trending',
                r'BingX',
                r'pattern.*detector',
                r'market.*aggregator',
                r'lateral.*pattern'
            ]
            
            for log_file in log_files:
                try:
                    if os.path.exists(log_file):
                        result = subprocess.run(['tail', '-n', '500', log_file], 
                                              capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            log_content = result.stdout
                            lines = log_content.split('\n')
                            analysis['total_entries'] += len([line for line in lines if line.strip()])
                            
                            # Count errors
                            for line in lines:
                                if any(re.search(pattern, line, re.IGNORECASE) for pattern in error_patterns):
                                    analysis['error_entries'] += 1
                                    if 'CRITICAL' in line.upper():
                                        analysis['critical_errors'] += 1
                                    if len(analysis['recent_errors']) < 3:
                                        analysis['recent_errors'].append(line.strip())
                                
                                # Count trending-related entries
                                if any(re.search(pattern, line, re.IGNORECASE) for pattern in trending_patterns):
                                    analysis['trending_related_entries'] += 1
                            
                except Exception as e:
                    logger.warning(f"   ⚠️ Could not analyze {log_file}: {e}")
            
            # Calculate final metrics
            if analysis['total_entries'] > 0:
                analysis['error_rate'] = analysis['error_entries'] / analysis['total_entries']
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Log analysis failed: {e}")
            return {
                'total_entries': 0,
                'error_entries': 0,
                'critical_errors': 0,
                'error_rate': 0.0,
                'trending_related_entries': 0,
                'recent_errors': []
            }
    
    async def run_comprehensive_test_suite(self):
        """Run comprehensive autonomous trend detection system test suite"""
        logger.info("🚀 Starting Autonomous Trend Detection System Comprehensive Test Suite")
        logger.info("=" * 80)
        logger.info("📋 AUTONOMOUS TREND DETECTION SYSTEM TEST SUITE")
        logger.info("🎯 Testing: 4h frequency, advanced filters, BingX integration, lateral pattern detection")
        logger.info("🎯 Expected: Complete autonomous trend detection working with real BingX data")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_trending_auto_updater_configuration()
        await self.test_2_bingx_trending_data_fetch()
        await self.test_3_lateral_pattern_detector_analysis()
        await self.test_4_advanced_market_aggregator_integration()
        await self.test_5_complete_system_integration_flow()
        await self.test_6_system_performance_and_stability()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 AUTONOMOUS TREND DETECTION SYSTEM COMPREHENSIVE TEST SUMMARY")
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
            if "Configuration" in result['test']:
                requirements_status['4h Frequency & Configuration'] = result['success']
            elif "BingX Trending Data" in result['test']:
                requirements_status['BingX Data Fetch & Filters'] = result['success']
            elif "Lateral Pattern Detector" in result['test']:
                requirements_status['Lateral Pattern Detection'] = result['success']
            elif "Market Aggregator" in result['test']:
                requirements_status['Market Aggregator Integration'] = result['success']
            elif "Integration Flow" in result['test']:
                requirements_status['Complete System Integration'] = result['success']
            elif "Performance" in result['test']:
                requirements_status['System Performance & Stability'] = result['success']
        
        logger.info("🎯 CRITICAL REQUIREMENTS STATUS:")
        for requirement, status in requirements_status.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {requirement}")
        
        requirements_met = sum(1 for status in requirements_status.values() if status)
        total_requirements = len(requirements_status)
        
        # Final verdict
        logger.info(f"\n🏆 REQUIREMENTS SATISFACTION: {requirements_met}/{total_requirements}")
        
        if requirements_met == total_requirements:
            logger.info("\n🎉 VERDICT: AUTONOMOUS TREND DETECTION SYSTEM FULLY FUNCTIONAL!")
            logger.info("✅ 4h frequency configuration working")
            logger.info("✅ BingX API integration with advanced filters operational")
            logger.info("✅ Lateral pattern detection filtering effectively")
            logger.info("✅ Market aggregator using filtered BingX data")
            logger.info("✅ Complete integration flow working")
            logger.info("✅ System performance stable")
            logger.info("✅ Ready for production use")
        elif requirements_met >= total_requirements * 0.8:
            logger.info("\n⚠️ VERDICT: AUTONOMOUS TREND DETECTION SYSTEM MOSTLY FUNCTIONAL")
            logger.info("🔍 Minor issues may need attention for complete functionality")
        elif requirements_met >= total_requirements * 0.6:
            logger.info("\n⚠️ VERDICT: AUTONOMOUS TREND DETECTION SYSTEM PARTIALLY FUNCTIONAL")
            logger.info("🔧 Several critical requirements need implementation or debugging")
        else:
            logger.info("\n❌ VERDICT: AUTONOMOUS TREND DETECTION SYSTEM NOT FUNCTIONAL")
            logger.info("🚨 Major issues preventing autonomous trend detection from working correctly")
            logger.info("🚨 System needs significant debugging and fixes")
        
        return passed_tests, total_tests

async def main():
    """Main function to run the comprehensive autonomous trend detection test suite"""
    test_suite = AutonomousTrendDetectionTestSuite()
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
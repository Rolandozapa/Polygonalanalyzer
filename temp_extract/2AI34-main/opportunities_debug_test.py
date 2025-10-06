#!/usr/bin/env python3
"""
OPPORTUNITIES GENERATION AND REFRESH DEBUG TEST SUITE
Focus: Debug why opportunities list hasn't been updated for 6 hours and shows the same 14 tokens

CRITICAL DEBUG OBJECTIVES FROM REVIEW REQUEST:
1. **Check Trending Auto-Updater**: Why "No trending cryptos found in page content"?
2. **Verify Cache Behavior**: Is the cache preventing refresh despite 5min TTL?
3. **Test Real Data Sources**: Are APIs returning new trending tokens?
4. **Force Cache Clear**: Try to generate fresh opportunities

TESTING APPROACH:
- Monitor trending_auto_updater logs in detail
- Check if Readdy.link source is working: https://readdy.link/preview/917833d5-a5d5-4425-867f-4fe110fa36f2/1956022
- Verify AdvancedMarketAggregator cache behavior
- Test if opportunities can be forced to refresh
- Examine why same 14 symbols persist for 6 hours

FOCUS AREAS:
- Trending crypto source reliability (Readdy.link)
- Cache expiration logic in get_current_opportunities
- Fallback symbol list vs real trending data
- API rate limits affecting data fetching
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

class OpportunitiesDebugTestSuite:
    """Debug test suite for opportunities generation and refresh issues"""
    
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
        logger.info(f"Testing Opportunities Debug at: {self.api_url}")
        
        # MongoDB connection for direct database analysis
        try:
            self.mongo_client = MongoClient("mongodb://localhost:27017")
            self.db = self.mongo_client["myapp"]
            logger.info("✅ MongoDB connection established for opportunities analysis")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.mongo_client = None
            self.db = None
        
        # Test results
        self.test_results = []
        
        # Readdy.link URL from review request
        self.readdy_url = "https://readdy.link/preview/917833d5-a5d5-4425-867f-4fe110fa36f2/1956022"
        
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
    
    async def test_1_current_opportunities_analysis(self):
        """Test 1: Analyze current opportunities to understand staleness"""
        logger.info("\n🔍 TEST 1: Current Opportunities Analysis")
        
        try:
            # Get current opportunities from API
            response = requests.get(f"{self.api_url}/opportunities", timeout=30)
            
            if response.status_code == 200:
                opportunities = response.json()
                logger.info(f"   📊 API returned {len(opportunities)} opportunities")
                
                # Analyze opportunities data
                if opportunities:
                    # Check timestamps
                    timestamps = []
                    symbols = []
                    
                    for opp in opportunities:
                        if 'timestamp' in opp:
                            timestamps.append(opp['timestamp'])
                        if 'symbol' in opp:
                            symbols.append(opp['symbol'])
                    
                    logger.info(f"   📊 Symbols found: {symbols}")
                    logger.info(f"   📊 Total unique symbols: {len(set(symbols))}")
                    
                    if timestamps:
                        # Parse timestamps and check age
                        oldest_timestamp = None
                        newest_timestamp = None
                        
                        for ts in timestamps:
                            try:
                                if isinstance(ts, str):
                                    # Try different timestamp formats
                                    try:
                                        parsed_ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                                    except:
                                        parsed_ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                                else:
                                    parsed_ts = ts
                                
                                if oldest_timestamp is None or parsed_ts < oldest_timestamp:
                                    oldest_timestamp = parsed_ts
                                if newest_timestamp is None or parsed_ts > newest_timestamp:
                                    newest_timestamp = parsed_ts
                            except Exception as e:
                                logger.warning(f"   ⚠️ Could not parse timestamp {ts}: {e}")
                        
                        if oldest_timestamp and newest_timestamp:
                            now = datetime.now()
                            if oldest_timestamp.tzinfo:
                                now = now.replace(tzinfo=oldest_timestamp.tzinfo)
                            
                            age_hours = (now - newest_timestamp).total_seconds() / 3600
                            logger.info(f"   📊 Newest opportunity age: {age_hours:.1f} hours")
                            logger.info(f"   📊 Oldest timestamp: {oldest_timestamp}")
                            logger.info(f"   📊 Newest timestamp: {newest_timestamp}")
                            
                            # Check if opportunities are stale (>6 hours as mentioned in review)
                            if age_hours > 6:
                                self.log_test_result("Current Opportunities Analysis", False, 
                                                   f"Opportunities are stale: {age_hours:.1f} hours old, {len(set(symbols))} unique symbols")
                            else:
                                self.log_test_result("Current Opportunities Analysis", True, 
                                                   f"Opportunities are fresh: {age_hours:.1f} hours old, {len(set(symbols))} unique symbols")
                        else:
                            self.log_test_result("Current Opportunities Analysis", False, 
                                               "Could not parse opportunity timestamps")
                    else:
                        self.log_test_result("Current Opportunities Analysis", False, 
                                           "No timestamps found in opportunities")
                else:
                    self.log_test_result("Current Opportunities Analysis", False, 
                                       "No opportunities returned from API")
            else:
                self.log_test_result("Current Opportunities Analysis", False, 
                                   f"API returned HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test_result("Current Opportunities Analysis", False, f"Exception: {str(e)}")
    
    async def test_2_trending_auto_updater_logs(self):
        """Test 2: Analyze trending auto-updater logs for issues"""
        logger.info("\n🔍 TEST 2: Trending Auto-Updater Logs Analysis")
        
        try:
            # Check backend logs for trending auto-updater activity
            log_files = [
                "/var/log/supervisor/backend.out.log",
                "/var/log/supervisor/backend.err.log"
            ]
            
            trending_analysis = {
                'total_trending_entries': 0,
                'no_trending_cryptos_errors': 0,
                'readdy_link_attempts': 0,
                'readdy_link_failures': 0,
                'cache_related_entries': 0,
                'recent_trending_activity': [],
                'error_patterns': []
            }
            
            trending_patterns = [
                r'trending.*auto.*updater',
                r'No trending cryptos found',
                r'readdy\.link',
                r'trending.*symbols',
                r'cache.*trending',
                r'trending.*update',
                r'trending.*crawler'
            ]
            
            error_patterns = [
                r'No trending cryptos found in page content',
                r'trending.*error',
                r'trending.*failed',
                r'readdy.*error',
                r'cache.*error'
            ]
            
            for log_file in log_files:
                try:
                    if os.path.exists(log_file):
                        result = subprocess.run(['tail', '-n', '2000', log_file], 
                                              capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            log_content = result.stdout
                            lines = log_content.split('\n')
                            
                            for line in lines:
                                # Check for trending-related entries
                                if any(re.search(pattern, line, re.IGNORECASE) for pattern in trending_patterns):
                                    trending_analysis['total_trending_entries'] += 1
                                    
                                    # Check for specific error patterns
                                    if 'No trending cryptos found' in line:
                                        trending_analysis['no_trending_cryptos_errors'] += 1
                                    
                                    if 'readdy.link' in line.lower():
                                        trending_analysis['readdy_link_attempts'] += 1
                                        if any(re.search(pattern, line, re.IGNORECASE) for pattern in error_patterns):
                                            trending_analysis['readdy_link_failures'] += 1
                                    
                                    if 'cache' in line.lower():
                                        trending_analysis['cache_related_entries'] += 1
                                    
                                    # Store recent trending activity
                                    if len(trending_analysis['recent_trending_activity']) < 10:
                                        trending_analysis['recent_trending_activity'].append(line.strip())
                                
                                # Check for error patterns
                                if any(re.search(pattern, line, re.IGNORECASE) for pattern in error_patterns):
                                    if len(trending_analysis['error_patterns']) < 5:
                                        trending_analysis['error_patterns'].append(line.strip())
                        
                except Exception as e:
                    logger.warning(f"   ⚠️ Could not analyze {log_file}: {e}")
            
            # Log analysis results
            logger.info(f"   📊 Trending Auto-Updater Log Analysis:")
            logger.info(f"      Total trending entries: {trending_analysis['total_trending_entries']}")
            logger.info(f"      'No trending cryptos found' errors: {trending_analysis['no_trending_cryptos_errors']}")
            logger.info(f"      Readdy.link attempts: {trending_analysis['readdy_link_attempts']}")
            logger.info(f"      Readdy.link failures: {trending_analysis['readdy_link_failures']}")
            logger.info(f"      Cache-related entries: {trending_analysis['cache_related_entries']}")
            
            if trending_analysis['recent_trending_activity']:
                logger.info(f"   📊 Recent Trending Activity:")
                for activity in trending_analysis['recent_trending_activity'][-5:]:
                    logger.info(f"      {activity}")
            
            if trending_analysis['error_patterns']:
                logger.info(f"   📊 Error Patterns Found:")
                for error in trending_analysis['error_patterns']:
                    logger.info(f"      {error}")
            
            # Determine if trending auto-updater is working
            if trending_analysis['no_trending_cryptos_errors'] > 0:
                self.log_test_result("Trending Auto-Updater Logs Analysis", False, 
                                   f"Found {trending_analysis['no_trending_cryptos_errors']} 'No trending cryptos found' errors")
            elif trending_analysis['total_trending_entries'] > 0:
                self.log_test_result("Trending Auto-Updater Logs Analysis", True, 
                                   f"Trending auto-updater active: {trending_analysis['total_trending_entries']} entries")
            else:
                self.log_test_result("Trending Auto-Updater Logs Analysis", False, 
                                   "No trending auto-updater activity found in logs")
                
        except Exception as e:
            self.log_test_result("Trending Auto-Updater Logs Analysis", False, f"Exception: {str(e)}")
    
    async def test_3_readdy_link_source_verification(self):
        """Test 3: Verify Readdy.link source is working"""
        logger.info("\n🔍 TEST 3: Readdy.link Source Verification")
        
        try:
            logger.info(f"   🌐 Testing Readdy.link URL: {self.readdy_url}")
            
            # Test direct access to Readdy.link
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(self.readdy_url, headers=headers, timeout=30)
            
            logger.info(f"   📊 Readdy.link Response:")
            logger.info(f"      Status Code: {response.status_code}")
            logger.info(f"      Content Length: {len(response.content)} bytes")
            logger.info(f"      Content Type: {response.headers.get('content-type', 'Unknown')}")
            
            if response.status_code == 200:
                content = response.text
                
                # Look for crypto-related content
                crypto_indicators = [
                    'BTC', 'ETH', 'crypto', 'bitcoin', 'ethereum', 'USDT', 'trending',
                    'price', 'market', 'coin', 'token', 'trading'
                ]
                
                found_indicators = []
                for indicator in crypto_indicators:
                    if indicator.lower() in content.lower():
                        found_indicators.append(indicator)
                
                logger.info(f"      Crypto indicators found: {found_indicators}")
                logger.info(f"      Content preview: {content[:500]}...")
                
                # Check if content looks like it contains trending crypto data
                if len(found_indicators) >= 3:
                    self.log_test_result("Readdy.link Source Verification", True, 
                                       f"Readdy.link accessible with crypto content: {len(found_indicators)} indicators")
                else:
                    self.log_test_result("Readdy.link Source Verification", False, 
                                       f"Readdy.link accessible but limited crypto content: {len(found_indicators)} indicators")
            else:
                self.log_test_result("Readdy.link Source Verification", False, 
                                   f"Readdy.link returned HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test_result("Readdy.link Source Verification", False, f"Exception: {str(e)}")
    
    async def test_4_cache_behavior_analysis(self):
        """Test 4: Analyze cache behavior and TTL"""
        logger.info("\n🔍 TEST 4: Cache Behavior Analysis")
        
        try:
            # Test multiple requests to see if cache is working
            cache_test_results = []
            
            for i in range(3):
                logger.info(f"   🔄 Cache test request {i+1}/3")
                
                start_time = time.time()
                response = requests.get(f"{self.api_url}/opportunities", timeout=30)
                end_time = time.time()
                
                response_time = end_time - start_time
                
                if response.status_code == 200:
                    opportunities = response.json()
                    cache_test_results.append({
                        'request_num': i+1,
                        'response_time': response_time,
                        'opportunity_count': len(opportunities),
                        'success': True
                    })
                    logger.info(f"      Response time: {response_time:.2f}s, Opportunities: {len(opportunities)}")
                else:
                    cache_test_results.append({
                        'request_num': i+1,
                        'response_time': response_time,
                        'opportunity_count': 0,
                        'success': False
                    })
                    logger.info(f"      Failed: HTTP {response.status_code}")
                
                # Wait 2 seconds between requests
                if i < 2:
                    await asyncio.sleep(2)
            
            # Analyze cache behavior
            successful_requests = [r for r in cache_test_results if r['success']]
            
            if len(successful_requests) >= 2:
                # Check if response times suggest caching
                response_times = [r['response_time'] for r in successful_requests]
                avg_response_time = sum(response_times) / len(response_times)
                
                # Check if opportunity counts are identical (suggesting cache)
                opportunity_counts = [r['opportunity_count'] for r in successful_requests]
                identical_counts = len(set(opportunity_counts)) == 1
                
                logger.info(f"   📊 Cache Analysis:")
                logger.info(f"      Average response time: {avg_response_time:.2f}s")
                logger.info(f"      Identical opportunity counts: {identical_counts}")
                logger.info(f"      Opportunity counts: {opportunity_counts}")
                
                # Check backend logs for cache-related messages
                cache_log_analysis = await self._analyze_cache_logs()
                
                logger.info(f"   📊 Cache Log Analysis:")
                logger.info(f"      Cache hit entries: {cache_log_analysis['cache_hits']}")
                logger.info(f"      Cache miss entries: {cache_log_analysis['cache_misses']}")
                logger.info(f"      Cache TTL entries: {cache_log_analysis['ttl_entries']}")
                logger.info(f"      Cache clear entries: {cache_log_analysis['cache_clears']}")
                
                # Determine if cache is preventing refresh
                if identical_counts and cache_log_analysis['cache_hits'] > cache_log_analysis['cache_misses']:
                    self.log_test_result("Cache Behavior Analysis", False, 
                                       f"Cache may be preventing refresh: identical counts, {cache_log_analysis['cache_hits']} hits vs {cache_log_analysis['cache_misses']} misses")
                elif avg_response_time < 1.0 and identical_counts:
                    self.log_test_result("Cache Behavior Analysis", False, 
                                       f"Fast responses with identical data suggest aggressive caching: {avg_response_time:.2f}s avg")
                else:
                    self.log_test_result("Cache Behavior Analysis", True, 
                                       f"Cache behavior appears normal: {avg_response_time:.2f}s avg, varied counts: {not identical_counts}")
            else:
                self.log_test_result("Cache Behavior Analysis", False, 
                                   "Insufficient successful requests to analyze cache behavior")
                
        except Exception as e:
            self.log_test_result("Cache Behavior Analysis", False, f"Exception: {str(e)}")
    
    async def _analyze_cache_logs(self):
        """Analyze backend logs for cache-related entries"""
        cache_analysis = {
            'cache_hits': 0,
            'cache_misses': 0,
            'ttl_entries': 0,
            'cache_clears': 0,
            'recent_cache_activity': []
        }
        
        try:
            log_files = [
                "/var/log/supervisor/backend.out.log",
                "/var/log/supervisor/backend.err.log"
            ]
            
            cache_patterns = [
                r'cache.*hit',
                r'cache.*miss',
                r'ttl',
                r'cache.*clear',
                r'cache.*expire',
                r'AdvancedMarketAggregator.*cache'
            ]
            
            for log_file in log_files:
                try:
                    if os.path.exists(log_file):
                        result = subprocess.run(['tail', '-n', '1000', log_file], 
                                              capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            log_content = result.stdout
                            lines = log_content.split('\n')
                            
                            for line in lines:
                                if any(re.search(pattern, line, re.IGNORECASE) for pattern in cache_patterns):
                                    if 'hit' in line.lower():
                                        cache_analysis['cache_hits'] += 1
                                    if 'miss' in line.lower():
                                        cache_analysis['cache_misses'] += 1
                                    if 'ttl' in line.lower():
                                        cache_analysis['ttl_entries'] += 1
                                    if 'clear' in line.lower():
                                        cache_analysis['cache_clears'] += 1
                                    
                                    if len(cache_analysis['recent_cache_activity']) < 5:
                                        cache_analysis['recent_cache_activity'].append(line.strip())
                        
                except Exception as e:
                    logger.warning(f"Could not analyze cache logs in {log_file}: {e}")
            
        except Exception as e:
            logger.warning(f"Cache log analysis failed: {e}")
        
        return cache_analysis
    
    async def test_5_force_cache_clear_and_refresh(self):
        """Test 5: Force cache clear and try to generate fresh opportunities"""
        logger.info("\n🔍 TEST 5: Force Cache Clear and Refresh")
        
        try:
            # Get initial opportunities
            logger.info("   📊 Getting initial opportunities...")
            initial_response = requests.get(f"{self.api_url}/opportunities", timeout=30)
            
            if initial_response.status_code == 200:
                initial_opportunities = initial_response.json()
                initial_count = len(initial_opportunities)
                initial_symbols = [opp.get('symbol', 'Unknown') for opp in initial_opportunities]
                logger.info(f"      Initial opportunities: {initial_count}")
                logger.info(f"      Initial symbols: {initial_symbols}")
            else:
                logger.info(f"      Initial request failed: HTTP {initial_response.status_code}")
                initial_count = 0
                initial_symbols = []
            
            # Try to trigger a fresh scan/update
            logger.info("   🔄 Triggering fresh opportunities scan...")
            
            # Try different endpoints that might refresh opportunities
            refresh_endpoints = [
                '/run-ia1-cycle',
                '/admin/refresh-opportunities',
                '/opportunities/refresh',
                '/scan-opportunities'
            ]
            
            refresh_success = False
            for endpoint in refresh_endpoints:
                try:
                    logger.info(f"      Trying {endpoint}...")
                    response = requests.post(f"{self.api_url}{endpoint}", json={}, timeout=60)
                    
                    if response.status_code in [200, 201]:
                        logger.info(f"         ✅ {endpoint}: HTTP {response.status_code}")
                        refresh_success = True
                        break
                    else:
                        logger.info(f"         ❌ {endpoint}: HTTP {response.status_code}")
                        
                except Exception as e:
                    logger.info(f"         ❌ {endpoint}: Exception - {str(e)}")
            
            # Wait for potential refresh
            if refresh_success:
                logger.info("   ⏳ Waiting 30 seconds for refresh to complete...")
                await asyncio.sleep(30)
            
            # Get opportunities after refresh attempt
            logger.info("   📊 Getting opportunities after refresh attempt...")
            final_response = requests.get(f"{self.api_url}/opportunities", timeout=30)
            
            if final_response.status_code == 200:
                final_opportunities = final_response.json()
                final_count = len(final_opportunities)
                final_symbols = [opp.get('symbol', 'Unknown') for opp in final_opportunities]
                
                logger.info(f"      Final opportunities: {final_count}")
                logger.info(f"      Final symbols: {final_symbols}")
                
                # Compare initial vs final
                symbols_changed = set(initial_symbols) != set(final_symbols)
                count_changed = initial_count != final_count
                
                logger.info(f"   📊 Refresh Analysis:")
                logger.info(f"      Count changed: {count_changed} ({initial_count} → {final_count})")
                logger.info(f"      Symbols changed: {symbols_changed}")
                
                if symbols_changed or count_changed:
                    self.log_test_result("Force Cache Clear and Refresh", True, 
                                       f"Refresh successful: count {initial_count}→{final_count}, symbols changed: {symbols_changed}")
                else:
                    self.log_test_result("Force Cache Clear and Refresh", False, 
                                       f"No changes after refresh: same {final_count} opportunities, same symbols")
            else:
                self.log_test_result("Force Cache Clear and Refresh", False, 
                                   f"Final request failed: HTTP {final_response.status_code}")
                
        except Exception as e:
            self.log_test_result("Force Cache Clear and Refresh", False, f"Exception: {str(e)}")
    
    async def test_6_database_opportunities_analysis(self):
        """Test 6: Analyze opportunities in database for staleness patterns"""
        logger.info("\n🔍 TEST 6: Database Opportunities Analysis")
        
        try:
            if self.db is None:
                self.log_test_result("Database Opportunities Analysis", False, 
                                   "MongoDB connection not available")
                return
            
            # Get opportunities from database
            opportunities = list(self.db.market_opportunities.find({}).sort("timestamp", -1).limit(50))
            
            logger.info(f"   📊 Found {len(opportunities)} opportunities in database")
            
            if len(opportunities) == 0:
                self.log_test_result("Database Opportunities Analysis", False, 
                                   "No opportunities found in database")
                return
            
            # Analyze timestamps and symbols
            timestamp_analysis = {}
            symbol_analysis = {}
            
            for opp in opportunities:
                # Analyze timestamps
                timestamp = opp.get('timestamp')
                if timestamp:
                    try:
                        if isinstance(timestamp, str):
                            parsed_ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        else:
                            parsed_ts = timestamp
                        
                        hour_key = parsed_ts.strftime('%Y-%m-%d %H:00')
                        timestamp_analysis[hour_key] = timestamp_analysis.get(hour_key, 0) + 1
                        
                    except Exception as e:
                        logger.warning(f"Could not parse timestamp {timestamp}: {e}")
                
                # Analyze symbols
                symbol = opp.get('symbol', 'Unknown')
                symbol_analysis[symbol] = symbol_analysis.get(symbol, 0) + 1
            
            # Find most recent and oldest timestamps
            recent_opportunities = opportunities[:10]  # Most recent 10
            oldest_opportunities = opportunities[-10:]  # Oldest 10
            
            logger.info(f"   📊 Timestamp Distribution (by hour):")
            for hour, count in sorted(timestamp_analysis.items(), reverse=True)[:10]:
                logger.info(f"      {hour}: {count} opportunities")
            
            logger.info(f"   📊 Symbol Distribution (top 15):")
            sorted_symbols = sorted(symbol_analysis.items(), key=lambda x: x[1], reverse=True)
            for symbol, count in sorted_symbols[:15]:
                logger.info(f"      {symbol}: {count} occurrences")
            
            # Check for staleness patterns
            unique_symbols = len(symbol_analysis)
            total_opportunities = len(opportunities)
            
            # Check if we have the "same 14 tokens" issue mentioned in review
            if unique_symbols <= 15:
                logger.info(f"   ⚠️ Limited symbol diversity: only {unique_symbols} unique symbols")
                
                # Check if these symbols are repeating frequently
                high_frequency_symbols = [symbol for symbol, count in sorted_symbols if count > 3]
                logger.info(f"   📊 High-frequency symbols (>3 occurrences): {high_frequency_symbols}")
                
                if len(high_frequency_symbols) >= 10:
                    self.log_test_result("Database Opportunities Analysis", False, 
                                       f"Symbol staleness detected: {unique_symbols} unique symbols, {len(high_frequency_symbols)} high-frequency repeats")
                else:
                    self.log_test_result("Database Opportunities Analysis", True, 
                                       f"Symbol diversity acceptable: {unique_symbols} unique symbols")
            else:
                self.log_test_result("Database Opportunities Analysis", True, 
                                   f"Good symbol diversity: {unique_symbols} unique symbols")
                
        except Exception as e:
            self.log_test_result("Database Opportunities Analysis", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_debug(self):
        """Run comprehensive opportunities debug analysis"""
        logger.info("🚀 Starting Opportunities Generation and Refresh Debug")
        logger.info("=" * 80)
        logger.info("📋 OPPORTUNITIES DEBUG ANALYSIS")
        logger.info("🎯 Objective: Debug why opportunities list hasn't been updated for 6 hours")
        logger.info("🎯 Focus: Trending auto-updater, cache behavior, data sources, refresh mechanism")
        logger.info("=" * 80)
        
        # Run all debug tests in sequence
        await self.test_1_current_opportunities_analysis()
        await self.test_2_trending_auto_updater_logs()
        await self.test_3_readdy_link_source_verification()
        await self.test_4_cache_behavior_analysis()
        await self.test_5_force_cache_clear_and_refresh()
        await self.test_6_database_opportunities_analysis()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 OPPORTUNITIES DEBUG COMPREHENSIVE ANALYSIS SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # Critical issues analysis
        logger.info("\n" + "=" * 80)
        logger.info("📋 CRITICAL ISSUES IDENTIFIED")
        logger.info("=" * 80)
        
        critical_issues = []
        recommendations = []
        
        for result in self.test_results:
            if not result['success']:
                if "stale" in result['details'].lower() or "6 hours" in result['details']:
                    critical_issues.append("🚨 STALE OPPORTUNITIES: " + result['details'])
                    recommendations.append("• Force refresh opportunities generation system")
                
                if "no trending cryptos found" in result['details'].lower():
                    critical_issues.append("🚨 TRENDING AUTO-UPDATER FAILURE: " + result['details'])
                    recommendations.append("• Debug trending auto-updater and Readdy.link integration")
                
                if "cache" in result['details'].lower() and "preventing" in result['details'].lower():
                    critical_issues.append("🚨 CACHE BLOCKING REFRESH: " + result['details'])
                    recommendations.append("• Clear cache and adjust TTL settings")
                
                if "readdy.link" in result['details'].lower():
                    critical_issues.append("🚨 DATA SOURCE ISSUE: " + result['details'])
                    recommendations.append("• Verify Readdy.link accessibility and content parsing")
                
                if "14 tokens" in result['details'] or "symbol staleness" in result['details'].lower():
                    critical_issues.append("🚨 SYMBOL DIVERSITY ISSUE: " + result['details'])
                    recommendations.append("• Expand symbol sources and improve trending detection")
        
        if critical_issues:
            logger.info("🚨 CRITICAL ISSUES FOUND:")
            for issue in critical_issues:
                logger.info(f"   {issue}")
        else:
            logger.info("✅ No critical issues identified")
        
        if recommendations:
            logger.info("\n💡 RECOMMENDATIONS:")
            for rec in set(recommendations):  # Remove duplicates
                logger.info(f"   {rec}")
        
        # Final verdict
        logger.info(f"\n🏆 DEBUG ANALYSIS RESULT: {passed_tests}/{total_tests}")
        
        if len(critical_issues) == 0:
            logger.info("\n🎉 VERDICT: OPPORTUNITIES SYSTEM APPEARS HEALTHY")
            logger.info("✅ No major issues detected with opportunities generation")
        elif len(critical_issues) <= 2:
            logger.info("\n⚠️ VERDICT: MINOR OPPORTUNITIES ISSUES DETECTED")
            logger.info("🔍 Some issues found but system may be partially functional")
        else:
            logger.info("\n❌ VERDICT: MAJOR OPPORTUNITIES SYSTEM ISSUES")
            logger.info("🚨 Multiple critical issues preventing proper opportunities refresh")
            logger.info("🚨 Immediate attention required to restore functionality")
        
        return passed_tests, total_tests, critical_issues, recommendations

async def main():
    """Main function to run the comprehensive opportunities debug analysis"""
    test_suite = OpportunitiesDebugTestSuite()
    passed_tests, total_tests, critical_issues, recommendations = await test_suite.run_comprehensive_debug()
    
    # Exit with appropriate code
    if len(critical_issues) == 0:
        sys.exit(0)  # No critical issues
    elif len(critical_issues) <= 2:
        sys.exit(1)  # Minor issues
    else:
        sys.exit(2)  # Major issues

if __name__ == "__main__":
    asyncio.run(main())
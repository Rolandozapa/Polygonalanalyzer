#!/usr/bin/env python3
"""
BINGX OPPORTUNITIES REFRESH COMPREHENSIVE TEST SUITE
Focus: Force complete opportunities refresh with BingX data to resolve stale data issues

CRITICAL TEST REQUIREMENTS FROM REVIEW REQUEST:
1. **Identify Real Opportunities Source**: Where do the 14 opportunities actually come from?
2. **Force Cache Invalidation**: Clear all caches that might be preventing BingX data
3. **Test BingX Integration**: Verify trending_auto_updater has BingX data available
4. **Force Scout Refresh**: Make scout use new BingX opportunities instead of cached ones

CRITICAL ISSUES TO RESOLVE:
- Why get_current_opportunities logs don't appear (method not called?)
- Old timestamps (11:54 and 11:20 from this morning) persist despite cache TTL=10s
- Data sources still show ["cryptocompare", "coingecko"] instead of BingX
- DOGEUSDT (top BingX gainer +8.87%) missing from opportunities

TESTING APPROACH:
- Check if scout.scan_opportunities uses advanced_market_aggregator.get_current_opportunities
- Verify if trending_auto_updater.current_trending has BingX data
- Force clear all caches (opportunities, market data, etc.)
- Test multiple entry points: /api/start-scout, direct opportunity calls
- Monitor logs to see if get_current_opportunities is actually called

GOAL: Get fresh BingX trending opportunities (DOGEUSDT, ONDOUSDT with high gains) to replace the stale 4.8h old data.
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

class BingXOpportunitiesRefreshTestSuite:
    """Comprehensive test suite for BingX Opportunities Refresh Issues"""
    
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
        logger.info(f"Testing BingX Opportunities Refresh at: {self.api_url}")
        
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
        
        # Expected BingX trending symbols (from review request)
        self.expected_bingx_symbols = ["DOGEUSDT", "ONDOUSDT"]
        
        # Track cache clearing attempts
        self.cache_clear_attempts = []
        
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
    
    async def test_1_identify_opportunities_source(self):
        """Test 1: Identify where the 14 opportunities actually come from"""
        logger.info("\n🔍 TEST 1: Identify Real Opportunities Source")
        
        try:
            # Get current opportunities from API
            response = requests.get(f"{self.api_url}/opportunities", timeout=30)
            
            if response.status_code == 200:
                opportunities = response.json()
                logger.info(f"   📊 Found {len(opportunities)} opportunities from API")
                
                # Analyze data sources
                data_sources = {}
                timestamps = []
                symbols = []
                
                for opp in opportunities:
                    # Extract data sources
                    sources = opp.get('data_sources', [])
                    for source in sources:
                        data_sources[source] = data_sources.get(source, 0) + 1
                    
                    # Extract timestamps
                    timestamp = opp.get('timestamp')
                    if timestamp:
                        timestamps.append(timestamp)
                    
                    # Extract symbols
                    symbol = opp.get('symbol')
                    if symbol:
                        symbols.append(symbol)
                
                logger.info(f"   📊 Data Sources Analysis:")
                for source, count in data_sources.items():
                    logger.info(f"      {source}: {count} opportunities")
                
                logger.info(f"   📊 Symbols Found: {symbols[:10]}...")  # First 10 symbols
                
                # Check for expected BingX symbols
                bingx_symbols_found = [sym for sym in self.expected_bingx_symbols if sym in symbols]
                logger.info(f"   📊 Expected BingX symbols found: {bingx_symbols_found}")
                
                # Analyze timestamps for staleness
                if timestamps:
                    latest_timestamp = max(timestamps)
                    oldest_timestamp = min(timestamps)
                    logger.info(f"   📊 Timestamp Range:")
                    logger.info(f"      Latest: {latest_timestamp}")
                    logger.info(f"      Oldest: {oldest_timestamp}")
                    
                    # Check if data is stale (older than 1 hour)
                    try:
                        latest_dt = datetime.fromisoformat(latest_timestamp.replace('Z', '+00:00'))
                        age_hours = (datetime.now(latest_dt.tzinfo) - latest_dt).total_seconds() / 3600
                        logger.info(f"      Data age: {age_hours:.1f} hours")
                        
                        is_stale = age_hours > 1.0
                        has_bingx_sources = any('bingx' in source.lower() for source in data_sources.keys())
                        has_expected_symbols = len(bingx_symbols_found) > 0
                        
                        if not is_stale and has_bingx_sources and has_expected_symbols:
                            self.log_test_result("Identify Opportunities Source", True, 
                                               f"Fresh BingX data found: {len(opportunities)} opportunities, {len(bingx_symbols_found)} expected symbols")
                        else:
                            issues = []
                            if is_stale:
                                issues.append(f"Data is stale ({age_hours:.1f}h old)")
                            if not has_bingx_sources:
                                issues.append("No BingX sources found")
                            if not has_expected_symbols:
                                issues.append("Expected BingX symbols missing")
                            
                            self.log_test_result("Identify Opportunities Source", False, 
                                               f"Issues found: {', '.join(issues)}")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Could not parse timestamp: {e}")
                        self.log_test_result("Identify Opportunities Source", False, 
                                           f"Timestamp parsing error: {e}")
                else:
                    self.log_test_result("Identify Opportunities Source", False, 
                                       "No timestamps found in opportunities")
            else:
                self.log_test_result("Identify Opportunities Source", False, 
                                   f"API returned HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test_result("Identify Opportunities Source", False, f"Exception: {str(e)}")
    
    async def test_2_force_cache_invalidation(self):
        """Test 2: Force clear all caches that might be preventing BingX data"""
        logger.info("\n🔍 TEST 2: Force Cache Invalidation")
        
        try:
            # Method 1: Restart backend service to clear in-memory caches
            logger.info("   🔄 Method 1: Restarting backend service to clear caches")
            try:
                result = subprocess.run(['sudo', 'supervisorctl', 'restart', 'backend'], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    logger.info("   ✅ Backend service restarted successfully")
                    self.cache_clear_attempts.append("backend_restart_success")
                    # Wait for service to come back up
                    await asyncio.sleep(10)
                else:
                    logger.warning(f"   ⚠️ Backend restart failed: {result.stderr}")
                    self.cache_clear_attempts.append("backend_restart_failed")
            except Exception as e:
                logger.warning(f"   ⚠️ Could not restart backend: {e}")
                self.cache_clear_attempts.append("backend_restart_error")
            
            # Method 2: Check if there are any cache clearing endpoints
            logger.info("   🔄 Method 2: Looking for cache clearing endpoints")
            cache_endpoints = [
                "/admin/clear-cache",
                "/cache/clear",
                "/clear-cache",
                "/admin/refresh-data"
            ]
            
            for endpoint in cache_endpoints:
                try:
                    response = requests.post(f"{self.api_url}{endpoint}", timeout=10)
                    if response.status_code in [200, 201]:
                        logger.info(f"   ✅ Cache cleared via {endpoint}")
                        self.cache_clear_attempts.append(f"endpoint_{endpoint}_success")
                        break
                except Exception:
                    continue
            
            # Method 3: Force new opportunities fetch by calling scout
            logger.info("   🔄 Method 3: Forcing new opportunities fetch via scout")
            try:
                response = requests.post(f"{self.api_url}/start-scout", json={}, timeout=60)
                if response.status_code in [200, 201]:
                    logger.info("   ✅ Scout started successfully")
                    self.cache_clear_attempts.append("scout_start_success")
                    # Wait for scout to complete
                    await asyncio.sleep(30)
                else:
                    logger.warning(f"   ⚠️ Scout start failed: HTTP {response.status_code}")
                    self.cache_clear_attempts.append("scout_start_failed")
            except Exception as e:
                logger.warning(f"   ⚠️ Could not start scout: {e}")
                self.cache_clear_attempts.append("scout_start_error")
            
            # Verify cache clearing worked by checking opportunities again
            logger.info("   🔍 Verifying cache clearing effectiveness")
            try:
                response = requests.get(f"{self.api_url}/opportunities", timeout=30)
                if response.status_code == 200:
                    opportunities = response.json()
                    
                    # Check if we have fresh data
                    timestamps = [opp.get('timestamp') for opp in opportunities if opp.get('timestamp')]
                    if timestamps:
                        latest_timestamp = max(timestamps)
                        latest_dt = datetime.fromisoformat(latest_timestamp.replace('Z', '+00:00'))
                        age_minutes = (datetime.now(latest_dt.tzinfo) - latest_dt).total_seconds() / 60
                        
                        if age_minutes < 30:  # Data less than 30 minutes old
                            self.log_test_result("Force Cache Invalidation", True, 
                                               f"Cache cleared successfully: data is {age_minutes:.1f} minutes old")
                        else:
                            self.log_test_result("Force Cache Invalidation", False, 
                                               f"Cache clearing may have failed: data is {age_minutes:.1f} minutes old")
                    else:
                        self.log_test_result("Force Cache Invalidation", False, 
                                           "No timestamps in opportunities after cache clearing")
                else:
                    self.log_test_result("Force Cache Invalidation", False, 
                                       f"Could not verify cache clearing: HTTP {response.status_code}")
            except Exception as e:
                self.log_test_result("Force Cache Invalidation", False, 
                                   f"Cache clearing verification failed: {e}")
                
        except Exception as e:
            self.log_test_result("Force Cache Invalidation", False, f"Exception: {str(e)}")
    
    async def test_3_bingx_integration_verification(self):
        """Test 3: Verify trending_auto_updater has BingX data available"""
        logger.info("\n🔍 TEST 3: BingX Integration Verification")
        
        try:
            # Check backend logs for BingX integration activity
            logger.info("   🔍 Checking backend logs for BingX activity")
            
            bingx_log_patterns = [
                r'bingx.*trending',
                r'BingX.*API',
                r'trending.*bingx',
                r'get_current_opportunities.*called',
                r'BINGX.*data',
                r'trending_auto_updater'
            ]
            
            bingx_activity_found = False
            log_files = [
                "/var/log/supervisor/backend.out.log",
                "/var/log/supervisor/backend.err.log"
            ]
            
            for log_file in log_files:
                try:
                    if os.path.exists(log_file):
                        result = subprocess.run(['tail', '-n', '500', log_file], 
                                              capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            log_content = result.stdout.lower()
                            
                            for pattern in bingx_log_patterns:
                                matches = re.findall(pattern, log_content, re.IGNORECASE)
                                if matches:
                                    logger.info(f"   ✅ Found BingX activity in {log_file}: {len(matches)} matches for '{pattern}'")
                                    bingx_activity_found = True
                                    
                except Exception as e:
                    logger.warning(f"   ⚠️ Could not analyze {log_file}: {e}")
            
            # Check if trending_auto_updater endpoint exists
            logger.info("   🔍 Checking for trending updater endpoints")
            trending_endpoints = [
                "/admin/trending/status",
                "/trending/status", 
                "/admin/trending/update",
                "/trending/update"
            ]
            
            trending_endpoint_found = False
            for endpoint in trending_endpoints:
                try:
                    response = requests.get(f"{self.api_url}{endpoint}", timeout=10)
                    if response.status_code in [200, 201]:
                        logger.info(f"   ✅ Trending endpoint found: {endpoint}")
                        trending_endpoint_found = True
                        
                        # Try to get trending data
                        data = response.json()
                        if isinstance(data, dict):
                            trending_count = data.get('trending_count', 0)
                            trending_symbols = data.get('trending_symbols', [])
                            logger.info(f"   📊 Trending data: {trending_count} symbols, {trending_symbols[:5]}...")
                        break
                except Exception:
                    continue
            
            # Force trending update if endpoint exists
            if trending_endpoint_found:
                logger.info("   🔄 Forcing trending update")
                try:
                    response = requests.post(f"{self.api_url}/admin/trending/update", timeout=60)
                    if response.status_code in [200, 201]:
                        logger.info("   ✅ Trending update triggered successfully")
                        await asyncio.sleep(30)  # Wait for update to complete
                except Exception as e:
                    logger.warning(f"   ⚠️ Could not trigger trending update: {e}")
            
            # Check database for trending data
            logger.info("   🔍 Checking database for BingX trending data")
            db_trending_found = False
            if self.db is not None:
                try:
                    # Look for any collections that might contain trending data
                    collections = self.db.list_collection_names()
                    trending_collections = [col for col in collections if 'trend' in col.lower()]
                    
                    if trending_collections:
                        logger.info(f"   📊 Found trending collections: {trending_collections}")
                        
                        for collection in trending_collections:
                            count = self.db[collection].count_documents({})
                            if count > 0:
                                logger.info(f"   📊 {collection}: {count} documents")
                                db_trending_found = True
                    
                    # Also check market_opportunities for BingX sources
                    bingx_opportunities = self.db.market_opportunities.count_documents({
                        "data_sources": {"$in": ["bingx", "bingx_api", "bingx_trending", "bingx_fallback"]}
                    })
                    
                    if bingx_opportunities > 0:
                        logger.info(f"   📊 Found {bingx_opportunities} opportunities with BingX sources")
                        db_trending_found = True
                        
                except Exception as e:
                    logger.warning(f"   ⚠️ Database trending check failed: {e}")
            
            # Determine test result
            integration_indicators = {
                'bingx_logs': bingx_activity_found,
                'trending_endpoint': trending_endpoint_found,
                'database_data': db_trending_found
            }
            
            working_indicators = sum(1 for indicator in integration_indicators.values() if indicator)
            total_indicators = len(integration_indicators)
            
            logger.info(f"   📊 BingX Integration Indicators:")
            for indicator, status in integration_indicators.items():
                status_icon = "✅" if status else "❌"
                logger.info(f"      {status_icon} {indicator.replace('_', ' ').title()}")
            
            if working_indicators >= 2:
                self.log_test_result("BingX Integration Verification", True, 
                                   f"BingX integration working: {working_indicators}/{total_indicators} indicators positive")
            elif working_indicators >= 1:
                self.log_test_result("BingX Integration Verification", False, 
                                   f"Partial BingX integration: {working_indicators}/{total_indicators} indicators positive")
            else:
                self.log_test_result("BingX Integration Verification", False, 
                                   f"BingX integration not detected: {working_indicators}/{total_indicators} indicators positive")
                
        except Exception as e:
            self.log_test_result("BingX Integration Verification", False, f"Exception: {str(e)}")
    
    async def test_4_force_scout_refresh(self):
        """Test 4: Force scout to use new BingX opportunities instead of cached ones"""
        logger.info("\n🔍 TEST 4: Force Scout Refresh")
        
        try:
            # Get opportunities before scout refresh
            logger.info("   📊 Getting opportunities before scout refresh")
            before_opportunities = []
            try:
                response = requests.get(f"{self.api_url}/opportunities", timeout=30)
                if response.status_code == 200:
                    before_opportunities = response.json()
                    logger.info(f"   📊 Before: {len(before_opportunities)} opportunities")
            except Exception as e:
                logger.warning(f"   ⚠️ Could not get before opportunities: {e}")
            
            # Force scout refresh using multiple methods
            logger.info("   🔄 Method 1: Direct scout start")
            scout_methods = []
            
            try:
                response = requests.post(f"{self.api_url}/start-scout", json={}, timeout=120)
                if response.status_code in [200, 201]:
                    logger.info("   ✅ Scout started successfully")
                    scout_methods.append("start_scout_success")
                    await asyncio.sleep(60)  # Wait for scout to complete
                else:
                    logger.warning(f"   ⚠️ Scout start failed: HTTP {response.status_code}")
                    scout_methods.append("start_scout_failed")
            except Exception as e:
                logger.warning(f"   ⚠️ Scout start error: {e}")
                scout_methods.append("start_scout_error")
            
            # Method 2: Try IA1 cycle which should trigger scout
            logger.info("   🔄 Method 2: Triggering IA1 cycle (should trigger scout)")
            try:
                response = requests.post(f"{self.api_url}/run-ia1-cycle", json={}, timeout=120)
                if response.status_code in [200, 201]:
                    logger.info("   ✅ IA1 cycle triggered successfully")
                    scout_methods.append("ia1_cycle_success")
                    await asyncio.sleep(60)  # Wait for cycle to complete
                else:
                    logger.warning(f"   ⚠️ IA1 cycle failed: HTTP {response.status_code}")
                    scout_methods.append("ia1_cycle_failed")
            except Exception as e:
                logger.warning(f"   ⚠️ IA1 cycle error: {e}")
                scout_methods.append("ia1_cycle_error")
            
            # Method 3: Direct opportunities refresh if endpoint exists
            logger.info("   🔄 Method 3: Direct opportunities refresh")
            refresh_endpoints = [
                "/admin/opportunities/refresh",
                "/opportunities/refresh",
                "/refresh-opportunities"
            ]
            
            for endpoint in refresh_endpoints:
                try:
                    response = requests.post(f"{self.api_url}{endpoint}", timeout=60)
                    if response.status_code in [200, 201]:
                        logger.info(f"   ✅ Opportunities refreshed via {endpoint}")
                        scout_methods.append(f"refresh_{endpoint}_success")
                        await asyncio.sleep(30)
                        break
                except Exception:
                    continue
            
            # Get opportunities after scout refresh
            logger.info("   📊 Getting opportunities after scout refresh")
            after_opportunities = []
            try:
                response = requests.get(f"{self.api_url}/opportunities", timeout=30)
                if response.status_code == 200:
                    after_opportunities = response.json()
                    logger.info(f"   📊 After: {len(after_opportunities)} opportunities")
            except Exception as e:
                logger.warning(f"   ⚠️ Could not get after opportunities: {e}")
            
            # Compare before and after
            logger.info("   🔍 Comparing opportunities before and after refresh")
            
            # Check for changes in data sources
            before_sources = set()
            after_sources = set()
            
            for opp in before_opportunities:
                sources = opp.get('data_sources', [])
                before_sources.update(sources)
            
            for opp in after_opportunities:
                sources = opp.get('data_sources', [])
                after_sources.update(sources)
            
            logger.info(f"   📊 Data sources before: {list(before_sources)}")
            logger.info(f"   📊 Data sources after: {list(after_sources)}")
            
            # Check for expected BingX symbols
            before_symbols = set(opp.get('symbol') for opp in before_opportunities)
            after_symbols = set(opp.get('symbol') for opp in after_opportunities)
            
            bingx_symbols_before = [sym for sym in self.expected_bingx_symbols if sym in before_symbols]
            bingx_symbols_after = [sym for sym in self.expected_bingx_symbols if sym in after_symbols]
            
            logger.info(f"   📊 Expected BingX symbols before: {bingx_symbols_before}")
            logger.info(f"   📊 Expected BingX symbols after: {bingx_symbols_after}")
            
            # Check timestamps for freshness
            after_timestamps = [opp.get('timestamp') for opp in after_opportunities if opp.get('timestamp')]
            fresh_data = False
            
            if after_timestamps:
                latest_timestamp = max(after_timestamps)
                try:
                    latest_dt = datetime.fromisoformat(latest_timestamp.replace('Z', '+00:00'))
                    age_minutes = (datetime.now(latest_dt.tzinfo) - latest_dt).total_seconds() / 60
                    logger.info(f"   📊 Latest data age: {age_minutes:.1f} minutes")
                    fresh_data = age_minutes < 30  # Less than 30 minutes old
                except Exception as e:
                    logger.warning(f"   ⚠️ Could not parse timestamp: {e}")
            
            # Determine test result
            refresh_indicators = {
                'scout_methods_worked': len([m for m in scout_methods if 'success' in m]) > 0,
                'data_sources_changed': after_sources != before_sources,
                'bingx_symbols_added': len(bingx_symbols_after) > len(bingx_symbols_before),
                'fresh_data': fresh_data,
                'bingx_sources_present': any('bingx' in source.lower() for source in after_sources)
            }
            
            working_indicators = sum(1 for indicator in refresh_indicators.values() if indicator)
            total_indicators = len(refresh_indicators)
            
            logger.info(f"   📊 Scout Refresh Indicators:")
            for indicator, status in refresh_indicators.items():
                status_icon = "✅" if status else "❌"
                logger.info(f"      {status_icon} {indicator.replace('_', ' ').title()}")
            
            if working_indicators >= 3:
                self.log_test_result("Force Scout Refresh", True, 
                                   f"Scout refresh successful: {working_indicators}/{total_indicators} indicators positive")
            elif working_indicators >= 2:
                self.log_test_result("Force Scout Refresh", False, 
                                   f"Partial scout refresh: {working_indicators}/{total_indicators} indicators positive")
            else:
                self.log_test_result("Force Scout Refresh", False, 
                                   f"Scout refresh failed: {working_indicators}/{total_indicators} indicators positive")
                
        except Exception as e:
            self.log_test_result("Force Scout Refresh", False, f"Exception: {str(e)}")
    
    async def test_5_monitor_get_current_opportunities_logs(self):
        """Test 5: Monitor logs to see if get_current_opportunities is actually called"""
        logger.info("\n🔍 TEST 5: Monitor get_current_opportunities Logs")
        
        try:
            # Check recent logs for get_current_opportunities calls
            logger.info("   🔍 Checking logs for get_current_opportunities calls")
            
            log_patterns = [
                r'get_current_opportunities.*called',
                r'BRIDGE.*get_current_opportunities',
                r'get_current_opportunities.*fetching',
                r'current_opportunities.*cache',
                r'opportunities.*bingx.*trending'
            ]
            
            log_files = [
                "/var/log/supervisor/backend.out.log",
                "/var/log/supervisor/backend.err.log"
            ]
            
            opportunities_calls_found = 0
            bingx_integration_logs = 0
            cache_logs = 0
            
            for log_file in log_files:
                try:
                    if os.path.exists(log_file):
                        result = subprocess.run(['tail', '-n', '1000', log_file], 
                                              capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            log_content = result.stdout
                            
                            for pattern in log_patterns:
                                matches = re.findall(pattern, log_content, re.IGNORECASE)
                                if matches:
                                    logger.info(f"   ✅ Found pattern '{pattern}' in {log_file}: {len(matches)} matches")
                                    
                                    if 'get_current_opportunities' in pattern:
                                        opportunities_calls_found += len(matches)
                                    elif 'bingx' in pattern:
                                        bingx_integration_logs += len(matches)
                                    elif 'cache' in pattern:
                                        cache_logs += len(matches)
                            
                            # Also look for specific log messages that should appear
                            specific_messages = [
                                "get_current_opportunities() called",
                                "BRIDGE: get_current_opportunities",
                                "USING BINGX TRENDING",
                                "BINGX OPPORTUNITIES: Generated",
                                "FINAL OPPORTUNITIES: Generated"
                            ]
                            
                            for message in specific_messages:
                                if message.lower() in log_content.lower():
                                    logger.info(f"   ✅ Found specific message: '{message}'")
                                    opportunities_calls_found += 1
                                    
                except Exception as e:
                    logger.warning(f"   ⚠️ Could not analyze {log_file}: {e}")
            
            # Force a new opportunities call and monitor logs
            logger.info("   🔄 Forcing new opportunities call to generate logs")
            
            # Clear recent logs by noting current time
            start_time = datetime.now()
            
            # Make opportunities call
            try:
                response = requests.get(f"{self.api_url}/opportunities", timeout=30)
                if response.status_code == 200:
                    logger.info("   ✅ Opportunities call successful")
                    
                    # Wait a moment for logs to be written
                    await asyncio.sleep(5)
                    
                    # Check for new logs after our call
                    new_logs_found = False
                    for log_file in log_files:
                        try:
                            if os.path.exists(log_file):
                                result = subprocess.run(['tail', '-n', '100', log_file], 
                                                      capture_output=True, text=True, timeout=30)
                                
                                if result.returncode == 0:
                                    recent_content = result.stdout
                                    
                                    # Look for our specific patterns in recent logs
                                    for pattern in log_patterns:
                                        if re.search(pattern, recent_content, re.IGNORECASE):
                                            logger.info(f"   ✅ Found recent activity: '{pattern}'")
                                            new_logs_found = True
                                            
                        except Exception as e:
                            logger.warning(f"   ⚠️ Could not check recent logs in {log_file}: {e}")
                    
                    if not new_logs_found:
                        logger.warning("   ⚠️ No recent get_current_opportunities activity detected")
                        
            except Exception as e:
                logger.warning(f"   ⚠️ Could not make opportunities call: {e}")
            
            # Analyze findings
            logger.info(f"   📊 Log Analysis Results:")
            logger.info(f"      get_current_opportunities calls found: {opportunities_calls_found}")
            logger.info(f"      BingX integration logs found: {bingx_integration_logs}")
            logger.info(f"      Cache-related logs found: {cache_logs}")
            
            # Determine test result
            if opportunities_calls_found > 0 and bingx_integration_logs > 0:
                self.log_test_result("Monitor get_current_opportunities Logs", True, 
                                   f"Method is being called with BingX integration: {opportunities_calls_found} calls, {bingx_integration_logs} BingX logs")
            elif opportunities_calls_found > 0:
                self.log_test_result("Monitor get_current_opportunities Logs", False, 
                                   f"Method is called but BingX integration unclear: {opportunities_calls_found} calls, {bingx_integration_logs} BingX logs")
            else:
                self.log_test_result("Monitor get_current_opportunities Logs", False, 
                                   f"Method may not be called: {opportunities_calls_found} calls found")
                
        except Exception as e:
            self.log_test_result("Monitor get_current_opportunities Logs", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_bingx_test(self):
        """Run comprehensive BingX opportunities refresh test"""
        logger.info("🚀 Starting BingX Opportunities Refresh Comprehensive Test")
        logger.info("=" * 80)
        logger.info("📋 BINGX OPPORTUNITIES REFRESH COMPREHENSIVE TEST SUITE")
        logger.info("🎯 Goal: Get fresh BingX trending opportunities to replace stale data")
        logger.info("🎯 Expected: DOGEUSDT, ONDOUSDT with high gains from BingX")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_identify_opportunities_source()
        await self.test_2_force_cache_invalidation()
        await self.test_3_bingx_integration_verification()
        await self.test_4_force_scout_refresh()
        await self.test_5_monitor_get_current_opportunities_logs()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 BINGX OPPORTUNITIES REFRESH COMPREHENSIVE TEST SUMMARY")
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
        logger.info("📋 CRITICAL ISSUES RESOLUTION STATUS")
        logger.info("=" * 80)
        
        issues_status = {}
        
        for result in self.test_results:
            if "Opportunities Source" in result['test']:
                issues_status['Identify Real Opportunities Source'] = result['success']
            elif "Cache Invalidation" in result['test']:
                issues_status['Force Cache Invalidation'] = result['success']
            elif "BingX Integration" in result['test']:
                issues_status['Test BingX Integration'] = result['success']
            elif "Scout Refresh" in result['test']:
                issues_status['Force Scout Refresh'] = result['success']
            elif "get_current_opportunities" in result['test']:
                issues_status['Monitor get_current_opportunities Calls'] = result['success']
        
        logger.info("🎯 CRITICAL ISSUES STATUS:")
        for issue, status in issues_status.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {issue}")
        
        issues_resolved = sum(1 for status in issues_status.values() if status)
        total_issues = len(issues_status)
        
        # Final verdict
        logger.info(f"\n🏆 ISSUES RESOLUTION: {issues_resolved}/{total_issues}")
        
        if issues_resolved == total_issues:
            logger.info("\n🎉 VERDICT: BINGX OPPORTUNITIES REFRESH FULLY SUCCESSFUL!")
            logger.info("✅ All critical issues resolved")
            logger.info("✅ BingX trending data integration working")
            logger.info("✅ Cache invalidation successful")
            logger.info("✅ Scout refresh operational")
            logger.info("✅ Fresh opportunities with BingX data available")
            logger.info("✅ Expected symbols (DOGEUSDT, ONDOUSDT) should be present")
        elif issues_resolved >= total_issues * 0.8:
            logger.info("\n⚠️ VERDICT: BINGX OPPORTUNITIES REFRESH MOSTLY SUCCESSFUL")
            logger.info("🔍 Minor issues may need attention for complete BingX integration")
        elif issues_resolved >= total_issues * 0.6:
            logger.info("\n⚠️ VERDICT: BINGX OPPORTUNITIES REFRESH PARTIALLY SUCCESSFUL")
            logger.info("🔧 Several critical issues need resolution for proper BingX data flow")
        else:
            logger.info("\n❌ VERDICT: BINGX OPPORTUNITIES REFRESH NOT SUCCESSFUL")
            logger.info("🚨 Major issues preventing BingX data from reaching opportunities")
            logger.info("🚨 System still using stale data from old sources")
        
        # Specific recommendations
        logger.info("\n" + "=" * 80)
        logger.info("📋 RECOMMENDATIONS FOR MAIN AGENT")
        logger.info("=" * 80)
        
        recommendations = []
        
        if not issues_status.get('Identify Real Opportunities Source', False):
            recommendations.append("🔧 Check advanced_market_aggregator.get_current_opportunities method implementation")
            recommendations.append("🔧 Verify trending_auto_updater integration in get_current_opportunities")
        
        if not issues_status.get('Force Cache Invalidation', False):
            recommendations.append("🔧 Implement cache clearing mechanism or reduce cache TTL")
            recommendations.append("🔧 Ensure cache_ttl = 10s is actually being respected")
        
        if not issues_status.get('Test BingX Integration', False):
            recommendations.append("🔧 Fix trending_auto_updater BingX API integration")
            recommendations.append("🔧 Ensure BingX trending data is being fetched and stored")
        
        if not issues_status.get('Force Scout Refresh', False):
            recommendations.append("🔧 Fix scout.scan_opportunities to use get_current_opportunities")
            recommendations.append("🔧 Ensure scout refresh triggers new data fetch")
        
        if not issues_status.get('Monitor get_current_opportunities Calls', False):
            recommendations.append("🔧 Add logging to get_current_opportunities method")
            recommendations.append("🔧 Verify method is being called by scout or other components")
        
        if recommendations:
            for rec in recommendations:
                logger.info(f"   {rec}")
        else:
            logger.info("   🎉 No specific recommendations - system appears to be working correctly!")
        
        return passed_tests, total_tests

async def main():
    """Main function to run the comprehensive BingX opportunities refresh test"""
    test_suite = BingXOpportunitiesRefreshTestSuite()
    passed_tests, total_tests = await test_suite.run_comprehensive_bingx_test()
    
    # Exit with appropriate code
    if passed_tests == total_tests:
        sys.exit(0)  # All tests passed
    elif passed_tests >= total_tests * 0.8:
        sys.exit(1)  # Mostly successful
    else:
        sys.exit(2)  # Major issues

if __name__ == "__main__":
    asyncio.run(main())
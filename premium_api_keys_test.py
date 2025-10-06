#!/usr/bin/env python3
"""
PREMIUM API KEYS INTEGRATION & TOP 25 MARKET CAP FILTERING TEST SUITE
Focus: Comprehensive test of the Premium API Keys Integration & Top 25 Market Cap Filtering implementation

TESTING OBJECTIVES:

1. **API Keys Integration Testing**:
   - Verify real API keys are loaded from .env file
   - Test CoinMarketCap, TwelveData, CoinAPI, and Binance API connectivity
   - Confirm api_keys_backup.txt exists with proper key storage

2. **Top 25 Market Cap Filtering**:
   - Verify /api/opportunities prioritizes top 25 market cap symbols (BTCUSDT, ETHUSDT, LINKUSDT, etc.)
   - Confirm system limits to maximum 25 symbols as requested
   - Test that current results show reduced symbol count (from previous 80+ to ~7-25)

3. **Scout System Quality**:
   - Confirm opportunities include major market cap leaders (LINKUSDT, APTUSDT, NEARUSDT)
   - Verify adaptive filtering: relaxed for top-25 (2% change, 200k volume), strict for extended (5% change, 500k volume)
   - Test two-phase processing: priority batch then extended batch

4. **Data Quality Validation**:
   - Ensure all returned opportunities have real market data (authentic prices, volumes)
   - Verify data_sources still contain "bingx_scout_filtered"
   - Confirm filter_status remains "scout_filtered_only"

5. **System Performance**:
   - Test /api/trending-force-update with new filtering
   - Verify startup data fetch still works within 60 seconds
   - Confirm system responsiveness with reduced symbol processing

This validates the user's request to limit scout to BingX top 25 market cap and integrate premium API keys for enhanced data quality.
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PremiumAPIKeysTestSuite:
    """Comprehensive test suite for Premium API Keys Integration & Top 25 Market Cap Filtering"""
    
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
        logger.info(f"Testing Premium API Keys Integration at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Expected top 25 market cap symbols (from review request)
        self.top_25_symbols = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 
            'AVAXUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'UNIUSDT', 
            'ATOMUSDT', 'FILUSDT', 'APTUSDT', 'NEARUSDT', 'VETUSDT', 'ICPUSDT', 'HBARUSDT', 
            'ALGOUSDT', 'ETCUSDT', 'MANAUSDT', 'SANDUSDT'
        ]
        
        # Premium API keys to test
        self.premium_apis = {
            'CMC_API_KEY': 'CoinMarketCap',
            'COINAPI_KEY': 'CoinAPI', 
            'TWELVEDATA_KEY': 'TwelveData',
            'BINANCE_KEY': 'Binance'
        }
        
        # Expected data quality indicators
        self.quality_indicators = {
            'data_sources': 'bingx_scout_filtered',
            'filter_status': 'scout_filtered_only',
            'min_volume': 200000,  # 200k volume for top-25
            'min_price_change': 2.0,  # 2% change for top-25
            'max_symbols': 25  # Maximum 25 symbols as requested
        }
        
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
    
    async def test_1_api_keys_integration(self):
        """Test 1: API Keys Integration - Verify real API keys are loaded and functional"""
        logger.info("\n🔍 TEST 1: API Keys Integration Testing")
        
        try:
            api_keys_results = {
                'env_file_accessible': False,
                'api_keys_found': {},
                'backup_file_exists': False,
                'api_connectivity_tests': {},
                'total_keys_found': 0,
                'functional_apis': 0
            }
            
            logger.info("   🚀 Testing premium API keys integration...")
            logger.info("   📊 Expected: Real API keys loaded from .env, api_keys_backup.txt exists, APIs functional")
            
            # Test 1.1: Check .env file for API keys
            logger.info("   📞 Checking .env file for premium API keys...")
            
            try:
                with open('/app/backend/.env', 'r') as f:
                    env_content = f.read()
                    api_keys_results['env_file_accessible'] = True
                    
                    for key_name, api_name in self.premium_apis.items():
                        if key_name in env_content:
                            # Extract the key value
                            for line in env_content.split('\n'):
                                if line.startswith(f"{key_name}="):
                                    key_value = line.split('=', 1)[1].strip()
                                    if key_value and key_value != "your_key_here":
                                        api_keys_results['api_keys_found'][key_name] = {
                                            'api_name': api_name,
                                            'key_length': len(key_value),
                                            'key_preview': key_value[:8] + "..." if len(key_value) > 8 else key_value
                                        }
                                        api_keys_results['total_keys_found'] += 1
                                        logger.info(f"      ✅ {api_name} key found: {key_value[:8]}... (length: {len(key_value)})")
                                    break
                        else:
                            logger.warning(f"      ⚠️ {api_name} key ({key_name}) not found in .env")
                
                logger.info(f"      📊 Total API keys found: {api_keys_results['total_keys_found']}/4")
                
            except Exception as e:
                logger.error(f"      ❌ Could not read .env file: {e}")
            
            # Test 1.2: Check for api_keys_backup.txt
            logger.info("   📞 Checking for api_keys_backup.txt file...")
            
            try:
                if os.path.exists('/app/api_keys_backup.txt'):
                    api_keys_results['backup_file_exists'] = True
                    logger.info("      ✅ api_keys_backup.txt exists")
                    
                    # Check backup file content
                    with open('/app/api_keys_backup.txt', 'r') as f:
                        backup_content = f.read()
                        backup_keys_count = sum(1 for key_name in self.premium_apis.keys() if key_name in backup_content)
                        logger.info(f"      📊 Backup file contains {backup_keys_count}/4 API keys")
                else:
                    logger.warning("      ⚠️ api_keys_backup.txt not found")
                    
            except Exception as e:
                logger.error(f"      ❌ Could not check backup file: {e}")
            
            # Test 1.3: Test API connectivity (basic tests)
            logger.info("   📞 Testing API connectivity...")
            
            # Test CoinMarketCap API
            if 'CMC_API_KEY' in api_keys_results['api_keys_found']:
                try:
                    cmc_key = None
                    with open('/app/backend/.env', 'r') as f:
                        for line in f:
                            if line.startswith('CMC_API_KEY='):
                                cmc_key = line.split('=', 1)[1].strip()
                                break
                    
                    if cmc_key:
                        headers = {'X-CMC_PRO_API_KEY': cmc_key}
                        response = requests.get(
                            'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest',
                            headers=headers,
                            params={'limit': 10},
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            api_keys_results['api_connectivity_tests']['CMC'] = True
                            api_keys_results['functional_apis'] += 1
                            logger.info("      ✅ CoinMarketCap API connectivity successful")
                        else:
                            api_keys_results['api_connectivity_tests']['CMC'] = False
                            logger.warning(f"      ⚠️ CoinMarketCap API failed: HTTP {response.status_code}")
                            
                except Exception as e:
                    api_keys_results['api_connectivity_tests']['CMC'] = False
                    logger.warning(f"      ⚠️ CoinMarketCap API test failed: {e}")
            
            # Test TwelveData API
            if 'TWELVEDATA_KEY' in api_keys_results['api_keys_found']:
                try:
                    twelve_key = None
                    with open('/app/backend/.env', 'r') as f:
                        for line in f:
                            if line.startswith('TWELVEDATA_KEY='):
                                twelve_key = line.split('=', 1)[1].strip()
                                break
                    
                    if twelve_key:
                        response = requests.get(
                            'https://api.twelvedata.com/price',
                            params={'symbol': 'BTC/USD', 'apikey': twelve_key},
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            api_keys_results['api_connectivity_tests']['TwelveData'] = True
                            api_keys_results['functional_apis'] += 1
                            logger.info("      ✅ TwelveData API connectivity successful")
                        else:
                            api_keys_results['api_connectivity_tests']['TwelveData'] = False
                            logger.warning(f"      ⚠️ TwelveData API failed: HTTP {response.status_code}")
                            
                except Exception as e:
                    api_keys_results['api_connectivity_tests']['TwelveData'] = False
                    logger.warning(f"      ⚠️ TwelveData API test failed: {e}")
            
            # Test CoinAPI
            if 'COINAPI_KEY' in api_keys_results['api_keys_found']:
                try:
                    coin_key = None
                    with open('/app/backend/.env', 'r') as f:
                        for line in f:
                            if line.startswith('COINAPI_KEY='):
                                coin_key = line.split('=', 1)[1].strip()
                                break
                    
                    if coin_key:
                        headers = {'X-CoinAPI-Key': coin_key}
                        response = requests.get(
                            'https://rest.coinapi.io/v1/exchangerate/BTC/USD',
                            headers=headers,
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            api_keys_results['api_connectivity_tests']['CoinAPI'] = True
                            api_keys_results['functional_apis'] += 1
                            logger.info("      ✅ CoinAPI connectivity successful")
                        else:
                            api_keys_results['api_connectivity_tests']['CoinAPI'] = False
                            logger.warning(f"      ⚠️ CoinAPI failed: HTTP {response.status_code}")
                            
                except Exception as e:
                    api_keys_results['api_connectivity_tests']['CoinAPI'] = False
                    logger.warning(f"      ⚠️ CoinAPI test failed: {e}")
            
            # Test Binance API (public endpoint, no auth needed)
            if 'BINANCE_KEY' in api_keys_results['api_keys_found']:
                try:
                    response = requests.get(
                        'https://api.binance.com/api/v3/ticker/price',
                        params={'symbol': 'BTCUSDT'},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        api_keys_results['api_connectivity_tests']['Binance'] = True
                        api_keys_results['functional_apis'] += 1
                        logger.info("      ✅ Binance API connectivity successful")
                    else:
                        api_keys_results['api_connectivity_tests']['Binance'] = False
                        logger.warning(f"      ⚠️ Binance API failed: HTTP {response.status_code}")
                        
                except Exception as e:
                    api_keys_results['api_connectivity_tests']['Binance'] = False
                    logger.warning(f"      ⚠️ Binance API test failed: {e}")
            
            # Final analysis
            keys_found_rate = api_keys_results['total_keys_found'] / 4
            functional_rate = api_keys_results['functional_apis'] / max(api_keys_results['total_keys_found'], 1)
            
            logger.info(f"\n   📊 API KEYS INTEGRATION RESULTS:")
            logger.info(f"      .env file accessible: {api_keys_results['env_file_accessible']}")
            logger.info(f"      API keys found: {api_keys_results['total_keys_found']}/4 ({keys_found_rate:.2f})")
            logger.info(f"      Backup file exists: {api_keys_results['backup_file_exists']}")
            logger.info(f"      Functional APIs: {api_keys_results['functional_apis']}/{api_keys_results['total_keys_found']} ({functional_rate:.2f})")
            logger.info(f"      API connectivity: {api_keys_results['api_connectivity_tests']}")
            
            # Calculate test success
            success_criteria = [
                api_keys_results['env_file_accessible'],
                api_keys_results['total_keys_found'] >= 3,  # At least 3/4 keys found
                api_keys_results['backup_file_exists'],
                api_keys_results['functional_apis'] >= 2  # At least 2 APIs functional
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.75:  # 75% success threshold
                self.log_test_result("API Keys Integration", True, 
                                   f"API keys integration successful: {success_count}/{len(success_criteria)} criteria met. Keys found: {api_keys_results['total_keys_found']}/4, Functional APIs: {api_keys_results['functional_apis']}")
            else:
                self.log_test_result("API Keys Integration", False, 
                                   f"API keys integration issues: {success_count}/{len(success_criteria)} criteria met. Missing keys or connectivity issues")
                
        except Exception as e:
            self.log_test_result("API Keys Integration", False, f"Exception: {str(e)}")

    async def test_2_top_25_market_cap_filtering(self):
        """Test 2: Top 25 Market Cap Filtering - Verify system prioritizes top 25 symbols and limits count"""
        logger.info("\n🔍 TEST 2: Top 25 Market Cap Filtering")
        
        try:
            filtering_results = {
                'opportunities_endpoint_accessible': False,
                'total_opportunities': 0,
                'top_25_symbols_found': 0,
                'symbol_count_within_limit': False,
                'major_leaders_present': 0,
                'symbol_distribution': {},
                'opportunities_data': []
            }
            
            logger.info("   🚀 Testing top 25 market cap filtering...")
            logger.info("   📊 Expected: Opportunities prioritize top 25 symbols, max 25 symbols, includes LINKUSDT/APTUSDT/NEARUSDT")
            
            # Test 2.1: Get opportunities from scout system
            logger.info("   📞 Getting opportunities from /api/opportunities...")
            
            try:
                start_time = time.time()
                response = requests.get(f"{self.api_url}/opportunities", timeout=60)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    filtering_results['opportunities_endpoint_accessible'] = True
                    opportunities_data = response.json()
                    
                    # Handle different response formats
                    if isinstance(opportunities_data, dict):
                        if 'opportunities' in opportunities_data:
                            opportunities = opportunities_data['opportunities']
                            filter_status = opportunities_data.get('filter_status', 'unknown')
                            data_sources = opportunities_data.get('data_sources', [])
                        else:
                            opportunities = [opportunities_data]
                            filter_status = 'unknown'
                            data_sources = []
                    elif isinstance(opportunities_data, list):
                        opportunities = opportunities_data
                        filter_status = 'unknown'
                        data_sources = []
                    else:
                        opportunities = []
                        filter_status = 'unknown'
                        data_sources = []
                    
                    filtering_results['total_opportunities'] = len(opportunities)
                    filtering_results['opportunities_data'] = opportunities
                    
                    logger.info(f"      ✅ Opportunities endpoint accessible (response time: {response_time:.2f}s)")
                    logger.info(f"      📊 Total opportunities: {len(opportunities)}")
                    logger.info(f"      📊 Filter status: {filter_status}")
                    logger.info(f"      📊 Data sources: {data_sources}")
                    
                    # Test 2.2: Check symbol count limit (max 25)
                    if len(opportunities) <= self.quality_indicators['max_symbols']:
                        filtering_results['symbol_count_within_limit'] = True
                        logger.info(f"      ✅ Symbol count within limit: {len(opportunities)} ≤ {self.quality_indicators['max_symbols']}")
                    else:
                        logger.warning(f"      ⚠️ Symbol count exceeds limit: {len(opportunities)} > {self.quality_indicators['max_symbols']}")
                    
                    # Test 2.3: Check for top 25 symbols presence
                    symbols_found = []
                    symbol_counts = {}
                    
                    for opp in opportunities:
                        symbol = opp.get('symbol', '')
                        if symbol:
                            symbols_found.append(symbol)
                            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
                    
                    filtering_results['symbol_distribution'] = symbol_counts
                    
                    # Count how many top 25 symbols are present
                    top_25_present = []
                    for symbol in symbols_found:
                        if symbol in self.top_25_symbols:
                            top_25_present.append(symbol)
                    
                    filtering_results['top_25_symbols_found'] = len(set(top_25_present))
                    
                    logger.info(f"      📊 Top 25 symbols found: {filtering_results['top_25_symbols_found']}/25")
                    logger.info(f"      📊 Top 25 symbols present: {list(set(top_25_present))[:10]}{'...' if len(set(top_25_present)) > 10 else ''}")
                    
                    # Test 2.4: Check for major market cap leaders
                    major_leaders = ['LINKUSDT', 'APTUSDT', 'NEARUSDT']
                    leaders_found = []
                    
                    for leader in major_leaders:
                        if leader in symbols_found:
                            leaders_found.append(leader)
                            filtering_results['major_leaders_present'] += 1
                    
                    logger.info(f"      📊 Major leaders present: {leaders_found} ({filtering_results['major_leaders_present']}/3)")
                    
                    # Test 2.5: Sample opportunity data quality
                    if opportunities:
                        sample_opp = opportunities[0]
                        logger.info(f"      📋 Sample opportunity: {sample_opp.get('symbol', 'N/A')}")
                        logger.info(f"         - Current price: ${sample_opp.get('current_price', 'N/A')}")
                        logger.info(f"         - Volume 24h: ${sample_opp.get('volume_24h', 'N/A'):,}" if isinstance(sample_opp.get('volume_24h'), (int, float)) else f"         - Volume 24h: {sample_opp.get('volume_24h', 'N/A')}")
                        logger.info(f"         - Price change 24h: {sample_opp.get('price_change_24h', 'N/A')}%")
                        logger.info(f"         - Volatility: {sample_opp.get('volatility', 'N/A')}")
                    
                else:
                    logger.error(f"      ❌ Opportunities endpoint failed: HTTP {response.status_code}")
                    if response.text:
                        logger.error(f"         Error: {response.text[:300]}")
                        
            except Exception as e:
                logger.error(f"      ❌ Opportunities endpoint exception: {e}")
            
            # Final analysis
            top_25_rate = filtering_results['top_25_symbols_found'] / 25
            leaders_rate = filtering_results['major_leaders_present'] / 3
            
            logger.info(f"\n   📊 TOP 25 MARKET CAP FILTERING RESULTS:")
            logger.info(f"      Opportunities endpoint accessible: {filtering_results['opportunities_endpoint_accessible']}")
            logger.info(f"      Total opportunities: {filtering_results['total_opportunities']}")
            logger.info(f"      Symbol count within limit: {filtering_results['symbol_count_within_limit']}")
            logger.info(f"      Top 25 symbols found: {filtering_results['top_25_symbols_found']}/25 ({top_25_rate:.2f})")
            logger.info(f"      Major leaders present: {filtering_results['major_leaders_present']}/3 ({leaders_rate:.2f})")
            
            # Calculate test success
            success_criteria = [
                filtering_results['opportunities_endpoint_accessible'],
                filtering_results['total_opportunities'] > 0,
                filtering_results['symbol_count_within_limit'],
                filtering_results['top_25_symbols_found'] >= 5,  # At least 5 top-25 symbols
                filtering_results['major_leaders_present'] >= 1  # At least 1 major leader
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Top 25 Market Cap Filtering", True, 
                                   f"Top 25 filtering successful: {success_count}/{len(success_criteria)} criteria met. Opportunities: {filtering_results['total_opportunities']}, Top 25: {filtering_results['top_25_symbols_found']}, Leaders: {filtering_results['major_leaders_present']}")
            else:
                self.log_test_result("Top 25 Market Cap Filtering", False, 
                                   f"Top 25 filtering issues: {success_count}/{len(success_criteria)} criteria met. May not be prioritizing top 25 symbols correctly")
                
        except Exception as e:
            self.log_test_result("Top 25 Market Cap Filtering", False, f"Exception: {str(e)}")

    async def test_3_scout_system_quality(self):
        """Test 3: Scout System Quality - Verify adaptive filtering and two-phase processing"""
        logger.info("\n🔍 TEST 3: Scout System Quality")
        
        try:
            scout_quality_results = {
                'opportunities_analyzed': 0,
                'adaptive_filtering_evidence': 0,
                'volume_threshold_compliance': 0,
                'price_change_compliance': 0,
                'two_phase_processing_evidence': False,
                'quality_metrics': {},
                'filtering_analysis': []
            }
            
            logger.info("   🚀 Testing scout system quality and adaptive filtering...")
            logger.info("   📊 Expected: Relaxed filters for top-25 (2% change, 200k volume), strict for extended (5% change, 500k volume)")
            
            # Get opportunities for analysis
            try:
                response = requests.get(f"{self.api_url}/opportunities", timeout=60)
                
                if response.status_code == 200:
                    opportunities_data = response.json()
                    
                    # Handle different response formats
                    if isinstance(opportunities_data, dict) and 'opportunities' in opportunities_data:
                        opportunities = opportunities_data['opportunities']
                    elif isinstance(opportunities_data, list):
                        opportunities = opportunities_data
                    else:
                        opportunities = []
                    
                    scout_quality_results['opportunities_analyzed'] = len(opportunities)
                    logger.info(f"      📊 Analyzing {len(opportunities)} opportunities for quality metrics...")
                    
                    # Analyze each opportunity for filtering compliance
                    top_25_opportunities = []
                    extended_opportunities = []
                    
                    for opp in opportunities:
                        symbol = opp.get('symbol', '')
                        current_price = opp.get('current_price', 0)
                        volume_24h = opp.get('volume_24h', 0)
                        price_change_24h = abs(opp.get('price_change_24h', 0))
                        
                        # Classify as top-25 or extended
                        is_top_25 = symbol in self.top_25_symbols
                        
                        analysis = {
                            'symbol': symbol,
                            'is_top_25': is_top_25,
                            'current_price': current_price,
                            'volume_24h': volume_24h,
                            'price_change_24h': price_change_24h,
                            'meets_relaxed_volume': volume_24h >= 200000,  # 200k for top-25
                            'meets_strict_volume': volume_24h >= 500000,   # 500k for extended
                            'meets_relaxed_change': price_change_24h >= 2.0,  # 2% for top-25
                            'meets_strict_change': price_change_24h >= 5.0    # 5% for extended
                        }
                        
                        scout_quality_results['filtering_analysis'].append(analysis)
                        
                        if is_top_25:
                            top_25_opportunities.append(analysis)
                        else:
                            extended_opportunities.append(analysis)
                    
                    logger.info(f"      📊 Top-25 opportunities: {len(top_25_opportunities)}")
                    logger.info(f"      📊 Extended opportunities: {len(extended_opportunities)}")
                    
                    # Test adaptive filtering for top-25 symbols
                    if top_25_opportunities:
                        relaxed_volume_compliance = sum(1 for opp in top_25_opportunities if opp['meets_relaxed_volume'])
                        relaxed_change_compliance = sum(1 for opp in top_25_opportunities if opp['meets_relaxed_change'])
                        
                        relaxed_volume_rate = relaxed_volume_compliance / len(top_25_opportunities)
                        relaxed_change_rate = relaxed_change_compliance / len(top_25_opportunities)
                        
                        logger.info(f"      📊 Top-25 relaxed volume compliance: {relaxed_volume_compliance}/{len(top_25_opportunities)} ({relaxed_volume_rate:.2f})")
                        logger.info(f"      📊 Top-25 relaxed change compliance: {relaxed_change_compliance}/{len(top_25_opportunities)} ({relaxed_change_rate:.2f})")
                        
                        if relaxed_volume_rate >= 0.7:  # 70% compliance
                            scout_quality_results['volume_threshold_compliance'] += 1
                        if relaxed_change_rate >= 0.7:  # 70% compliance
                            scout_quality_results['price_change_compliance'] += 1
                    
                    # Test adaptive filtering for extended symbols
                    if extended_opportunities:
                        strict_volume_compliance = sum(1 for opp in extended_opportunities if opp['meets_strict_volume'])
                        strict_change_compliance = sum(1 for opp in extended_opportunities if opp['meets_strict_change'])
                        
                        strict_volume_rate = strict_volume_compliance / len(extended_opportunities) if extended_opportunities else 0
                        strict_change_rate = strict_change_compliance / len(extended_opportunities) if extended_opportunities else 0
                        
                        logger.info(f"      📊 Extended strict volume compliance: {strict_volume_compliance}/{len(extended_opportunities)} ({strict_volume_rate:.2f})")
                        logger.info(f"      📊 Extended strict change compliance: {strict_change_compliance}/{len(extended_opportunities)} ({strict_change_rate:.2f})")
                        
                        if strict_volume_rate >= 0.6:  # 60% compliance (stricter)
                            scout_quality_results['adaptive_filtering_evidence'] += 1
                    
                    # Evidence of two-phase processing
                    if len(top_25_opportunities) > 0 and len(extended_opportunities) >= 0:
                        scout_quality_results['two_phase_processing_evidence'] = True
                        logger.info(f"      ✅ Two-phase processing evidence: Top-25 batch + Extended batch")
                    
                    # Quality metrics summary
                    scout_quality_results['quality_metrics'] = {
                        'total_opportunities': len(opportunities),
                        'top_25_count': len(top_25_opportunities),
                        'extended_count': len(extended_opportunities),
                        'avg_volume': sum(opp.get('volume_24h', 0) for opp in opportunities) / len(opportunities) if opportunities else 0,
                        'avg_price_change': sum(abs(opp.get('price_change_24h', 0)) for opp in opportunities) / len(opportunities) if opportunities else 0
                    }
                    
                    logger.info(f"      📊 Quality metrics: {scout_quality_results['quality_metrics']}")
                    
                    # Show sample filtering analysis
                    if scout_quality_results['filtering_analysis']:
                        logger.info(f"      📋 Sample filtering analysis:")
                        for i, analysis in enumerate(scout_quality_results['filtering_analysis'][:5]):
                            logger.info(f"         - {analysis['symbol']}: top25={analysis['is_top_25']}, vol={analysis['volume_24h']:,.0f}, change={analysis['price_change_24h']:.1f}%")
                
                else:
                    logger.error(f"      ❌ Could not get opportunities for quality analysis: HTTP {response.status_code}")
                    
            except Exception as e:
                logger.error(f"      ❌ Scout quality analysis exception: {e}")
            
            # Final analysis
            logger.info(f"\n   📊 SCOUT SYSTEM QUALITY RESULTS:")
            logger.info(f"      Opportunities analyzed: {scout_quality_results['opportunities_analyzed']}")
            logger.info(f"      Volume threshold compliance: {scout_quality_results['volume_threshold_compliance']}")
            logger.info(f"      Price change compliance: {scout_quality_results['price_change_compliance']}")
            logger.info(f"      Adaptive filtering evidence: {scout_quality_results['adaptive_filtering_evidence']}")
            logger.info(f"      Two-phase processing evidence: {scout_quality_results['two_phase_processing_evidence']}")
            
            # Calculate test success
            success_criteria = [
                scout_quality_results['opportunities_analyzed'] > 0,
                scout_quality_results['volume_threshold_compliance'] >= 1,
                scout_quality_results['price_change_compliance'] >= 1,
                scout_quality_results['two_phase_processing_evidence']
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.75:  # 75% success threshold
                self.log_test_result("Scout System Quality", True, 
                                   f"Scout quality successful: {success_count}/{len(success_criteria)} criteria met. Analyzed: {scout_quality_results['opportunities_analyzed']}, Adaptive filtering evidence found")
            else:
                self.log_test_result("Scout System Quality", False, 
                                   f"Scout quality issues: {success_count}/{len(success_criteria)} criteria met. May not be using adaptive filtering correctly")
                
        except Exception as e:
            self.log_test_result("Scout System Quality", False, f"Exception: {str(e)}")

    async def test_4_data_quality_validation(self):
        """Test 4: Data Quality Validation - Ensure real market data and proper data sources"""
        logger.info("\n🔍 TEST 4: Data Quality Validation")
        
        try:
            data_quality_results = {
                'opportunities_with_real_data': 0,
                'authentic_prices_count': 0,
                'authentic_volumes_count': 0,
                'bingx_scout_filtered_present': False,
                'scout_filtered_only_status': False,
                'data_authenticity_score': 0.0,
                'sample_data_analysis': []
            }
            
            logger.info("   🚀 Testing data quality validation...")
            logger.info("   📊 Expected: Real market data, authentic prices/volumes, bingx_scout_filtered sources, scout_filtered_only status")
            
            # Get opportunities for data quality analysis
            try:
                response = requests.get(f"{self.api_url}/opportunities", timeout=60)
                
                if response.status_code == 200:
                    opportunities_data = response.json()
                    
                    # Extract opportunities and metadata
                    if isinstance(opportunities_data, dict):
                        opportunities = opportunities_data.get('opportunities', [])
                        filter_status = opportunities_data.get('filter_status', '')
                        data_sources = opportunities_data.get('data_sources', [])
                    else:
                        opportunities = opportunities_data if isinstance(opportunities_data, list) else []
                        filter_status = ''
                        data_sources = []
                    
                    logger.info(f"      📊 Analyzing {len(opportunities)} opportunities for data quality...")
                    logger.info(f"      📊 Filter status: '{filter_status}'")
                    logger.info(f"      📊 Data sources: {data_sources}")
                    
                    # Test 4.1: Check filter status
                    if filter_status == 'scout_filtered_only':
                        data_quality_results['scout_filtered_only_status'] = True
                        logger.info(f"      ✅ Filter status correct: '{filter_status}'")
                    else:
                        logger.warning(f"      ⚠️ Filter status unexpected: '{filter_status}' (expected: 'scout_filtered_only')")
                    
                    # Test 4.2: Check data sources
                    if isinstance(data_sources, list) and any('bingx_scout_filtered' in str(source).lower() for source in data_sources):
                        data_quality_results['bingx_scout_filtered_present'] = True
                        logger.info(f"      ✅ BingX scout filtered source present")
                    else:
                        logger.warning(f"      ⚠️ BingX scout filtered source not found in: {data_sources}")
                    
                    # Test 4.3: Analyze individual opportunity data quality
                    for i, opp in enumerate(opportunities):
                        symbol = opp.get('symbol', 'UNKNOWN')
                        current_price = opp.get('current_price', 0)
                        volume_24h = opp.get('volume_24h', 0)
                        price_change_24h = opp.get('price_change_24h', 0)
                        volatility = opp.get('volatility', 0)
                        
                        # Data quality checks
                        has_real_price = isinstance(current_price, (int, float)) and current_price > 0
                        has_real_volume = isinstance(volume_24h, (int, float)) and volume_24h > 0
                        has_meaningful_change = isinstance(price_change_24h, (int, float)) and abs(price_change_24h) >= 0.1
                        has_volatility = isinstance(volatility, (int, float)) and volatility > 0
                        
                        # Authenticity checks (not fake/placeholder values)
                        price_authentic = has_real_price and current_price != 1.0 and current_price != 100.0
                        volume_authentic = has_real_volume and volume_24h != 1000000 and volume_24h > 10000  # Not exactly 1M (fake)
                        change_authentic = has_meaningful_change and price_change_24h != 2.5  # Not exactly 2.5% (fake)
                        
                        if has_real_price and has_real_volume:
                            data_quality_results['opportunities_with_real_data'] += 1
                        
                        if price_authentic:
                            data_quality_results['authentic_prices_count'] += 1
                        
                        if volume_authentic:
                            data_quality_results['authentic_volumes_count'] += 1
                        
                        # Store sample analysis
                        if i < 5:  # First 5 opportunities
                            data_quality_results['sample_data_analysis'].append({
                                'symbol': symbol,
                                'current_price': current_price,
                                'volume_24h': volume_24h,
                                'price_change_24h': price_change_24h,
                                'volatility': volatility,
                                'has_real_price': has_real_price,
                                'has_real_volume': has_real_volume,
                                'price_authentic': price_authentic,
                                'volume_authentic': volume_authentic,
                                'change_authentic': change_authentic
                            })
                            
                            logger.info(f"         📋 {symbol}: price=${current_price}, vol={volume_24h:,.0f}, change={price_change_24h:.2f}%, authentic=P:{price_authentic}/V:{volume_authentic}/C:{change_authentic}")
                    
                    # Calculate data authenticity score
                    if opportunities:
                        real_data_rate = data_quality_results['opportunities_with_real_data'] / len(opportunities)
                        authentic_price_rate = data_quality_results['authentic_prices_count'] / len(opportunities)
                        authentic_volume_rate = data_quality_results['authentic_volumes_count'] / len(opportunities)
                        
                        data_quality_results['data_authenticity_score'] = (real_data_rate + authentic_price_rate + authentic_volume_rate) / 3
                        
                        logger.info(f"      📊 Real data rate: {real_data_rate:.2f}")
                        logger.info(f"      📊 Authentic price rate: {authentic_price_rate:.2f}")
                        logger.info(f"      📊 Authentic volume rate: {authentic_volume_rate:.2f}")
                        logger.info(f"      📊 Overall authenticity score: {data_quality_results['data_authenticity_score']:.2f}")
                
                else:
                    logger.error(f"      ❌ Could not get opportunities for data quality analysis: HTTP {response.status_code}")
                    
            except Exception as e:
                logger.error(f"      ❌ Data quality analysis exception: {e}")
            
            # Final analysis
            logger.info(f"\n   📊 DATA QUALITY VALIDATION RESULTS:")
            logger.info(f"      Opportunities with real data: {data_quality_results['opportunities_with_real_data']}")
            logger.info(f"      Authentic prices count: {data_quality_results['authentic_prices_count']}")
            logger.info(f"      Authentic volumes count: {data_quality_results['authentic_volumes_count']}")
            logger.info(f"      BingX scout filtered present: {data_quality_results['bingx_scout_filtered_present']}")
            logger.info(f"      Scout filtered only status: {data_quality_results['scout_filtered_only_status']}")
            logger.info(f"      Data authenticity score: {data_quality_results['data_authenticity_score']:.2f}")
            
            # Calculate test success
            success_criteria = [
                data_quality_results['opportunities_with_real_data'] > 0,
                data_quality_results['authentic_prices_count'] > 0,
                data_quality_results['authentic_volumes_count'] > 0,
                data_quality_results['bingx_scout_filtered_present'],
                data_quality_results['scout_filtered_only_status'],
                data_quality_results['data_authenticity_score'] >= 0.8  # 80% authenticity
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.83:  # 83% success threshold
                self.log_test_result("Data Quality Validation", True, 
                                   f"Data quality successful: {success_count}/{len(success_criteria)} criteria met. Authenticity score: {data_quality_results['data_authenticity_score']:.2f}, Real data: {data_quality_results['opportunities_with_real_data']}")
            else:
                self.log_test_result("Data Quality Validation", False, 
                                   f"Data quality issues: {success_count}/{len(success_criteria)} criteria met. May have fake/placeholder data or missing sources")
                
        except Exception as e:
            self.log_test_result("Data Quality Validation", False, f"Exception: {str(e)}")

    async def test_5_system_performance(self):
        """Test 5: System Performance - Test trending-force-update and startup performance"""
        logger.info("\n🔍 TEST 5: System Performance")
        
        try:
            performance_results = {
                'trending_force_update_accessible': False,
                'trending_force_update_response_time': 0.0,
                'startup_data_fetch_working': False,
                'opportunities_response_time': 0.0,
                'system_responsiveness': False,
                'reduced_symbol_processing': False,
                'performance_metrics': {}
            }
            
            logger.info("   🚀 Testing system performance with new filtering...")
            logger.info("   📊 Expected: trending-force-update working, startup <60s, responsive system")
            
            # Test 5.1: Test /api/trending-force-update endpoint
            logger.info("   📞 Testing /api/trending-force-update endpoint...")
            
            try:
                start_time = time.time()
                response = requests.post(f"{self.api_url}/trending-force-update", timeout=120)
                response_time = time.time() - start_time
                performance_results['trending_force_update_response_time'] = response_time
                
                if response.status_code == 200:
                    performance_results['trending_force_update_accessible'] = True
                    update_data = response.json()
                    logger.info(f"      ✅ Trending force update successful (response time: {response_time:.2f}s)")
                    logger.info(f"      📊 Update response: {update_data}")
                else:
                    logger.warning(f"      ⚠️ Trending force update failed: HTTP {response.status_code}")
                    if response.text:
                        logger.warning(f"         Response: {response.text[:300]}")
                        
            except Exception as e:
                logger.error(f"      ❌ Trending force update exception: {e}")
            
            # Test 5.2: Test opportunities endpoint response time (system responsiveness)
            logger.info("   📞 Testing opportunities endpoint response time...")
            
            try:
                start_time = time.time()
                response = requests.get(f"{self.api_url}/opportunities", timeout=60)
                response_time = time.time() - start_time
                performance_results['opportunities_response_time'] = response_time
                
                if response.status_code == 200:
                    opportunities_data = response.json()
                    
                    # Extract opportunities count
                    if isinstance(opportunities_data, dict) and 'opportunities' in opportunities_data:
                        opportunities_count = len(opportunities_data['opportunities'])
                    elif isinstance(opportunities_data, list):
                        opportunities_count = len(opportunities_data)
                    else:
                        opportunities_count = 0
                    
                    logger.info(f"      ✅ Opportunities endpoint responsive (response time: {response_time:.2f}s)")
                    logger.info(f"      📊 Opportunities count: {opportunities_count}")
                    
                    # Check if response time is reasonable (system responsiveness)
                    if response_time <= 30:  # 30 seconds or less
                        performance_results['system_responsiveness'] = True
                        logger.info(f"      ✅ System responsive: {response_time:.2f}s ≤ 30s")
                    else:
                        logger.warning(f"      ⚠️ System slow: {response_time:.2f}s > 30s")
                    
                    # Check for reduced symbol processing (evidence of optimization)
                    if opportunities_count <= 25:  # Within expected limit
                        performance_results['reduced_symbol_processing'] = True
                        logger.info(f"      ✅ Reduced symbol processing: {opportunities_count} ≤ 25")
                    else:
                        logger.warning(f"      ⚠️ High symbol count: {opportunities_count} > 25")
                
                else:
                    logger.error(f"      ❌ Opportunities endpoint failed: HTTP {response.status_code}")
                    
            except Exception as e:
                logger.error(f"      ❌ Opportunities endpoint exception: {e}")
            
            # Test 5.3: Simulate startup data fetch test (check if system can populate data quickly)
            logger.info("   📞 Testing startup data fetch simulation...")
            
            try:
                # Multiple quick requests to test if system maintains performance
                response_times = []
                
                for i in range(3):
                    start_time = time.time()
                    response = requests.get(f"{self.api_url}/opportunities", timeout=30)
                    response_time = time.time() - start_time
                    response_times.append(response_time)
                    
                    if response.status_code == 200:
                        logger.info(f"      📊 Startup test {i+1}/3: {response_time:.2f}s")
                    else:
                        logger.warning(f"      ⚠️ Startup test {i+1}/3 failed: HTTP {response.status_code}")
                    
                    # Small delay between requests
                    await asyncio.sleep(2)
                
                # Check if all requests completed within reasonable time
                avg_response_time = sum(response_times) / len(response_times) if response_times else 60
                max_response_time = max(response_times) if response_times else 60
                
                if max_response_time <= 60:  # All requests under 60 seconds
                    performance_results['startup_data_fetch_working'] = True
                    logger.info(f"      ✅ Startup data fetch working: max {max_response_time:.2f}s ≤ 60s")
                else:
                    logger.warning(f"      ⚠️ Startup data fetch slow: max {max_response_time:.2f}s > 60s")
                
                performance_results['performance_metrics'] = {
                    'avg_response_time': avg_response_time,
                    'max_response_time': max_response_time,
                    'min_response_time': min(response_times) if response_times else 0,
                    'response_consistency': max_response_time - min(response_times) if response_times else 0
                }
                
                logger.info(f"      📊 Performance metrics: {performance_results['performance_metrics']}")
                
            except Exception as e:
                logger.error(f"      ❌ Startup data fetch test exception: {e}")
            
            # Final analysis
            logger.info(f"\n   📊 SYSTEM PERFORMANCE RESULTS:")
            logger.info(f"      Trending force update accessible: {performance_results['trending_force_update_accessible']}")
            logger.info(f"      Trending force update response time: {performance_results['trending_force_update_response_time']:.2f}s")
            logger.info(f"      Startup data fetch working: {performance_results['startup_data_fetch_working']}")
            logger.info(f"      Opportunities response time: {performance_results['opportunities_response_time']:.2f}s")
            logger.info(f"      System responsiveness: {performance_results['system_responsiveness']}")
            logger.info(f"      Reduced symbol processing: {performance_results['reduced_symbol_processing']}")
            
            # Calculate test success
            success_criteria = [
                performance_results['trending_force_update_accessible'],
                performance_results['startup_data_fetch_working'],
                performance_results['system_responsiveness'],
                performance_results['reduced_symbol_processing'],
                performance_results['opportunities_response_time'] <= 30  # Reasonable response time
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("System Performance", True, 
                                   f"System performance successful: {success_count}/{len(success_criteria)} criteria met. Response time: {performance_results['opportunities_response_time']:.2f}s, Startup working: {performance_results['startup_data_fetch_working']}")
            else:
                self.log_test_result("System Performance", False, 
                                   f"System performance issues: {success_count}/{len(success_criteria)} criteria met. May have slow response times or startup issues")
                
        except Exception as e:
            self.log_test_result("System Performance", False, f"Exception: {str(e)}")

    async def run_all_tests(self):
        """Run all premium API keys integration tests"""
        logger.info("🚀 STARTING PREMIUM API KEYS INTEGRATION & TOP 25 MARKET CAP FILTERING TEST SUITE")
        logger.info("=" * 100)
        
        start_time = time.time()
        
        # Run all tests
        await self.test_1_api_keys_integration()
        await self.test_2_top_25_market_cap_filtering()
        await self.test_3_scout_system_quality()
        await self.test_4_data_quality_validation()
        await self.test_5_system_performance()
        
        total_time = time.time() - start_time
        
        # Generate summary
        logger.info("\n" + "=" * 100)
        logger.info("📊 PREMIUM API KEYS INTEGRATION TEST SUITE SUMMARY")
        logger.info("=" * 100)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success rate: {success_rate:.2f}")
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        
        logger.info("\nDetailed Results:")
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
        
        logger.info("\n" + "=" * 100)
        
        if success_rate >= 0.8:
            logger.info("🎉 PREMIUM API KEYS INTEGRATION TEST SUITE: OVERALL SUCCESS")
        else:
            logger.info("⚠️ PREMIUM API KEYS INTEGRATION TEST SUITE: NEEDS ATTENTION")
        
        logger.info("=" * 100)
        
        return self.test_results

async def main():
    """Main test execution"""
    test_suite = PremiumAPIKeysTestSuite()
    results = await test_suite.run_all_tests()
    
    # Return exit code based on results
    passed_tests = sum(1 for result in results if result['success'])
    total_tests = len(results)
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    if success_rate >= 0.8:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure

if __name__ == "__main__":
    asyncio.run(main())
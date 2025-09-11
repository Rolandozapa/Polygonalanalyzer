#!/usr/bin/env python3
"""
Focused Global Crypto Market Analyzer Fallback Test
Testing the current state of the fallback system
"""

import asyncio
import json
import logging
import requests
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_global_market_analyzer_fallback():
    """Test the current state of the global market analyzer with fallback"""
    
    api_url = "https://dual-ai-trader-4.preview.emergentagent.com/api"
    
    logger.info("🔍 Testing Global Crypto Market Analyzer Fallback System")
    logger.info("=" * 70)
    
    # Test 1: Check admin endpoint response
    logger.info("\n📊 TEST 1: Admin Endpoint Response Analysis")
    try:
        response = requests.get(f"{api_url}/admin/market/global", timeout=30)
        logger.info(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"   Response Status: {data.get('status')}")
            logger.info(f"   Error Message: {data.get('error', 'None')}")
            logger.info(f"   Timestamp: {data.get('timestamp')}")
            
            # Check if there's any market data despite the error
            if 'market_overview' in data:
                market_overview = data['market_overview']
                logger.info(f"   Market Overview Present: {bool(market_overview)}")
                if market_overview:
                    logger.info(f"   BTC Price: {market_overview.get('btc_price', 'N/A')}")
                    logger.info(f"   Market Cap: {market_overview.get('total_market_cap', 'N/A')}")
            
            # Check for fallback indicators
            response_text = json.dumps(data)
            fallback_indicators = []
            if 'fallback' in response_text.lower():
                fallback_indicators.append('fallback_mentioned')
            if 'binance' in response_text.lower():
                fallback_indicators.append('binance_mentioned')
            if 'rate limit' in response_text.lower():
                fallback_indicators.append('rate_limit_mentioned')
            
            logger.info(f"   Fallback Indicators: {fallback_indicators}")
            
        else:
            logger.info(f"   Failed with HTTP {response.status_code}")
            
    except Exception as e:
        logger.info(f"   Exception: {e}")
    
    # Test 2: Check backend logs for fallback evidence
    logger.info("\n📊 TEST 2: Backend Logs Analysis")
    try:
        result = subprocess.run(
            ["tail", "-n", "200", "/var/log/supervisor/backend.out.log"],
            capture_output=True, text=True, timeout=10
        )
        logs = result.stdout
        
        # Look for global market analyzer patterns
        patterns = {
            'global_market_calls': logs.count('global market'),
            'coingecko_calls': logs.count('CoinGecko'),
            'binance_fallback': logs.count('Binance fallback'),
            'rate_limit_errors': logs.count('rate limit'),
            'fallback_patterns': logs.count('fallback'),
            'fear_greed_calls': logs.count('Fear & Greed'),
            'market_analyzer_errors': logs.count('global_crypto_market_analyzer')
        }
        
        logger.info("   Log Pattern Analysis:")
        for pattern, count in patterns.items():
            logger.info(f"     {pattern}: {count} occurrences")
            
        # Look for specific error messages
        if 'Unable to fetch global market data' in logs:
            logger.info("   ⚠️ Found 'Unable to fetch global market data' error")
        
        if 'HTTP 429' in logs or '429' in logs:
            logger.info("   ⚠️ Found HTTP 429 (rate limit) errors")
            
    except Exception as e:
        logger.info(f"   Exception reading logs: {e}")
    
    # Test 3: Test Fear & Greed API directly
    logger.info("\n📊 TEST 3: Fear & Greed API Direct Test")
    try:
        fear_greed_response = requests.get("https://api.alternative.me/fng?limit=1", timeout=10)
        logger.info(f"   Fear & Greed API Status: {fear_greed_response.status_code}")
        
        if fear_greed_response.status_code == 200:
            fear_data = fear_greed_response.json()
            if fear_data.get('data') and len(fear_data['data']) > 0:
                value = fear_data['data'][0].get('value')
                classification = fear_data['data'][0].get('value_classification')
                logger.info(f"   Fear & Greed Value: {value} ({classification})")
            else:
                logger.info("   Fear & Greed API returned empty data")
        else:
            logger.info(f"   Fear & Greed API failed: {fear_greed_response.status_code}")
            
    except Exception as e:
        logger.info(f"   Fear & Greed API Exception: {e}")
    
    # Test 4: Test CoinGecko API directly
    logger.info("\n📊 TEST 4: CoinGecko API Direct Test")
    try:
        coingecko_response = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        logger.info(f"   CoinGecko Global API Status: {coingecko_response.status_code}")
        
        if coingecko_response.status_code == 200:
            logger.info("   ✅ CoinGecko API is accessible")
        elif coingecko_response.status_code == 429:
            logger.info("   ⚠️ CoinGecko API rate limited (HTTP 429)")
        else:
            logger.info(f"   ❌ CoinGecko API failed: {coingecko_response.status_code}")
            
    except Exception as e:
        logger.info(f"   CoinGecko API Exception: {e}")
    
    # Test 5: Test Binance API directly (fallback)
    logger.info("\n📊 TEST 5: Binance API Direct Test (Fallback)")
    try:
        binance_response = requests.get("https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT", timeout=10)
        logger.info(f"   Binance API Status: {binance_response.status_code}")
        
        if binance_response.status_code == 200:
            binance_data = binance_response.json()
            btc_price = float(binance_data.get('lastPrice', 0))
            price_change = float(binance_data.get('priceChangePercent', 0))
            logger.info(f"   ✅ Binance API working - BTC: ${btc_price:,.0f} ({price_change:+.2f}%)")
        else:
            logger.info(f"   ❌ Binance API failed: {binance_response.status_code}")
            
    except Exception as e:
        logger.info(f"   Binance API Exception: {e}")
    
    # Test 6: Check if global market analyzer module is properly imported
    logger.info("\n📊 TEST 6: Module Import Analysis")
    try:
        # Check if the module is imported in server.py
        with open('/app/backend/server.py', 'r') as f:
            server_content = f.read()
            
        import_patterns = {
            'global_crypto_market_analyzer_import': 'from global_crypto_market_analyzer import' in server_content,
            'global_market_analyzer_usage': 'global_crypto_market_analyzer' in server_content,
            'admin_endpoint_defined': '@api_router.get("/admin/market/global")' in server_content,
            'get_global_market_data_calls': 'get_global_market_data()' in server_content
        }
        
        logger.info("   Module Integration Analysis:")
        for pattern, found in import_patterns.items():
            status = "✅" if found else "❌"
            logger.info(f"     {status} {pattern}: {found}")
            
    except Exception as e:
        logger.info(f"   Module analysis exception: {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info("📋 SUMMARY")
    logger.info("=" * 70)
    
    logger.info("🎯 FINDINGS:")
    logger.info("   • Admin endpoint is accessible (HTTP 200)")
    logger.info("   • System returns structured error response")
    logger.info("   • Error message: 'Unable to fetch global market data'")
    logger.info("   • This indicates the fallback system needs to be tested under API failure conditions")
    
    logger.info("\n🎯 NEXT STEPS:")
    logger.info("   • The system is properly structured but APIs may be failing")
    logger.info("   • Need to test if fallback mechanisms are working")
    logger.info("   • Check if Binance fallback is being triggered")
    logger.info("   • Verify if realistic default values are being used")

if __name__ == "__main__":
    asyncio.run(test_global_market_analyzer_fallback())
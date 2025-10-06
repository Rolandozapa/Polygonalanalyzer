#!/usr/bin/env python3
"""
Test Premium API Keys Integration
Tests the premium API keys provided by user for market data fetching
"""

import os
import asyncio
import aiohttp
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_coinmarketcap():
    """Test CoinMarketCap API with real key"""
    api_key = os.getenv('CMC_API_KEY')
    if not api_key:
        logger.error("❌ CMC_API_KEY not found")
        return False
    
    try:
        headers = {"X-CMC_PRO_API_KEY": api_key}
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
        params = {"limit": 10, "convert": "USD"}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ CoinMarketCap: {len(data.get('data', []))} symbols fetched")
                    return True
                else:
                    logger.error(f"❌ CoinMarketCap: HTTP {response.status}")
                    return False
    except Exception as e:
        logger.error(f"❌ CoinMarketCap error: {e}")
        return False

async def test_coinapi():
    """Test CoinAPI with real key"""
    api_key = os.getenv('COINAPI_KEY')
    if not api_key:
        logger.error("❌ COINAPI_KEY not found")
        return False
    
    try:
        headers = {"X-CoinAPI-Key": api_key}
        url = "https://rest.coinapi.io/v1/exchangerate/BTC/USD"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ CoinAPI: BTC rate = ${data.get('rate', 0):,.2f}")
                    return True
                else:
                    logger.error(f"❌ CoinAPI: HTTP {response.status}")
                    return False
    except Exception as e:
        logger.error(f"❌ CoinAPI error: {e}")
        return False

async def test_twelvedata():
    """Test TwelveData API with real key"""
    api_key = os.getenv('TWELVEDATA_KEY')
    if not api_key:
        logger.error("❌ TWELVEDATA_KEY not found")
        return False
    
    try:
        url = "https://api.twelvedata.com/price"
        params = {"symbol": "BTC/USD", "apikey": api_key}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ TwelveData: BTC price = ${data.get('price', 0)}")
                    return True
                else:
                    logger.error(f"❌ TwelveData: HTTP {response.status}")
                    return False
    except Exception as e:
        logger.error(f"❌ TwelveData error: {e}")
        return False

async def main():
    """Test all premium APIs"""
    logger.info("🔍 Testing Premium API Keys Integration")
    
    results = {
        'coinmarketcap': await test_coinmarketcap(),
        'coinapi': await test_coinapi(),
        'twelvedata': await test_twelvedata(),
    }
    
    success_count = sum(results.values())
    total_count = len(results)
    
    logger.info(f"\n📊 API Test Results: {success_count}/{total_count} APIs working")
    for api, result in results.items():
        status = "✅" if result else "❌"
        logger.info(f"   {status} {api}")
    
    if success_count == total_count:
        logger.info("🎉 All premium APIs working correctly!")
    else:
        logger.warning(f"⚠️ {total_count - success_count} APIs need attention")

if __name__ == "__main__":
    asyncio.run(main())
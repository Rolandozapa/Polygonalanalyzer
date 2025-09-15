#!/usr/bin/env python3
"""
Debug premium APIs that aren't working yet
"""
import asyncio
import aiohttp
import logging
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def debug_coinmarketcap_dex():
    """Debug CoinMarketCap DEX API"""
    print("\n🔍 DEBUGGING CoinMarketCap DEX API")
    
    api_key = '70046baa-e887-42ee-a909-03c6b6afab67'
    symbol = 'BTC'
    
    # Try different CMC endpoints
    endpoints = [
        {
            'url': 'https://pro-api.coinmarketcap.com/v4/dex/pairs/ohlcv/historical',
            'params': {
                'symbol': f'{symbol}-USDT',
                'interval': 'daily',
                'count': 10
            }
        },
        {
            'url': 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical',
            'params': {
                'symbol': 'BTC',
                'time_period': 'daily',
                'count': 10
            }
        },
        {
            'url': 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/historical',
            'params': {
                'symbol': 'BTC',
                'interval': '1d',
                'count': 10
            }
        }
    ]
    
    headers = {
        'X-CMC_PRO_API_KEY': api_key,
        'Accept': 'application/json'
    }
    
    for i, endpoint in enumerate(endpoints):
        print(f"\nTrying endpoint {i+1}: {endpoint['url']}")
        print(f"Params: {endpoint['params']}")
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(endpoint['url'], headers=headers, params=endpoint['params']) as response:
                    print(f"Status: {response.status}")
                    print(f"Headers: {dict(response.headers)}")
                    
                    if response.status == 200:
                        data = await response.json()
                        print(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                        print(f"Response sample: {str(data)[:300]}...")
                    else:
                        text = await response.text()
                        print(f"Error response: {text[:300]}...")
                        
        except Exception as e:
            print(f"Exception: {e}")

async def debug_twelvedata():
    """Debug TwelveData API"""
    print("\n🔍 DEBUGGING TwelveData API")
    
    api_key = 'd95a7424cbdd428297a058d8b74b'
    
    # Try different TwelveData endpoints
    endpoints = [
        {
            'url': 'https://api.twelvedata.com/time_series',
            'params': {
                'symbol': 'BTC/USD',
                'interval': '1day',
                'outputsize': '10',
                'apikey': api_key
            }
        },
        {
            'url': 'https://api.twelvedata.com/time_series',
            'params': {
                'symbol': 'BTCUSD',
                'interval': '1day', 
                'outputsize': '10',
                'apikey': api_key
            }
        },
        {
            'url': 'https://api.twelvedata.com/cryptocurrencies',
            'params': {
                'symbol': 'BTC',
                'exchange': 'Binance',
                'apikey': api_key
            }
        }
    ]
    
    for i, endpoint in enumerate(endpoints):
        print(f"\nTrying endpoint {i+1}: {endpoint['url']}")
        print(f"Params: {endpoint['params']}")
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(endpoint['url'], params=endpoint['params']) as response:
                    print(f"Status: {response.status}")
                    print(f"Headers: {dict(response.headers)}")
                    
                    if response.status == 200:
                        data = await response.json()
                        print(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                        print(f"Response sample: {str(data)[:300]}...")
                    else:
                        text = await response.text()
                        print(f"Error response: {text[:300]}...")
                        
        except Exception as e:
            print(f"Exception: {e}")

async def debug_coinapi():
    """Debug CoinAPI"""
    print("\n🔍 DEBUGGING CoinAPI")
    
    api_keys = [
        '30484334-1a7c-49a0-ae01-63922e3e542a',
        'bdb7e19a-0e6a-4954-8ba7-fd460b82d57e'
    ]
    
    # Try different CoinAPI endpoints
    endpoints = [
        'https://rest.coinapi.io/v1/ohlcv/BTC_USDT/history',
        'https://rest.coinapi.io/v1/ohlcv/BINANCE_SPOT_BTC_USDT/history',
        'https://rest.coinapi.io/v1/ohlcv/COINBASE_SPOT_BTC_USD/history'
    ]
    
    for key_idx, api_key in enumerate(api_keys):
        print(f"\n--- Using API Key {key_idx + 1} ---")
        
        for endpoint_idx, url in enumerate(endpoints):
            print(f"\nTrying endpoint {endpoint_idx+1}: {url}")
            
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=10)
            
            params = {
                "period_id": "1DAY",
                "time_start": start_time.isoformat(),
                "time_end": end_time.isoformat(),
                "limit": 10
            }
            headers = {"X-CoinAPI-Key": api_key}
            
            print(f"Params: {params}")
            
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                    async with session.get(url, params=params, headers=headers) as response:
                        print(f"Status: {response.status}")
                        print(f"Headers: {dict(response.headers)}")
                        
                        if response.status == 200:
                            data = await response.json()
                            print(f"Response length: {len(data) if isinstance(data, list) else 'Not a list'}")
                            if isinstance(data, list) and data:
                                print(f"First item: {data[0]}")
                        else:
                            text = await response.text()
                            print(f"Error response: {text[:300]}...")
                            
            except Exception as e:
                print(f"Exception: {e}")

async def main():
    """Run all debug tests"""
    print("🚀 Starting Premium API Debug")
    
    await debug_coinmarketcap_dex()
    await debug_twelvedata()
    await debug_coinapi()

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Debug script for OHLCV data sources
"""
import asyncio
import sys
import os
import logging
import aiohttp

# Add backend to path
sys.path.append('/app/backend')

from enhanced_ohlcv_fetcher import EnhancedOHLCVFetcher

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def debug_bingx(symbol='BTCUSDT'):
    """Debug BingX API calls"""
    print(f"\n🔍 DEBUGGING BingX API for {symbol}")
    
    bingx_base_url = 'https://open-api.bingx.com'
    # BingX futures format requires -USDT not USDT
    if symbol.endswith('USDT'):
        bingx_symbol = symbol.replace('USDT', '-USDT')
    else:
        bingx_symbol = f"{symbol}-USDT"
    
    url = f"{bingx_base_url}/openApi/swap/v2/quote/klines"
    params = {
        "symbol": bingx_symbol,
        "interval": "1d",
        "limit": 10
    }
    
    print(f"URL: {url}")
    print(f"Params: {params}")
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            async with session.get(url, params=params) as response:
                print(f"Status: {response.status}")
                print(f"Headers: {dict(response.headers)}")
                
                if response.status == 200:
                    data = await response.json()
                    print(f"Response: {data}")
                else:
                    text = await response.text()
                    print(f"Error response: {text}")
                    
    except Exception as e:
        print(f"Exception: {e}")

async def debug_binance(symbol='BTCUSDT'):
    """Debug Binance API calls"""
    print(f"\n🔍 DEBUGGING Binance API for {symbol}")
    
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1d", 
        "limit": 10
    }
    
    print(f"URL: {url}")
    print(f"Params: {params}")
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(url, params=params) as response:
                print(f"Status: {response.status}")
                print(f"Headers: {dict(response.headers)}")
                
                if response.status == 200:
                    data = await response.json()
                    print(f"Response length: {len(data)}")
                    if data:
                        print(f"First item: {data[0]}")
                else:
                    text = await response.text()
                    print(f"Error response: {text}")
                    
    except Exception as e:
        print(f"Exception: {e}")

async def debug_coindesk(symbol='BTCUSDT'):
    """Debug CoinDesk API calls"""
    print(f"\n🔍 DEBUGGING CoinDesk API for {symbol}")
    
    base_symbol = symbol.replace('USDT', '').replace('USD', '').lower()
    
    # Try different endpoints
    endpoints = [
        f"https://production-api.coindesk.com/v2/price/values/{base_symbol}",
        f"https://api.coindesk.com/v1/bpi/historical/close.json",
        f"https://production-api.coindesk.com/v2/tb/price/{base_symbol}?convert=USD"
    ]
    
    for url in endpoints:
        print(f"\nTrying URL: {url}")
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
                async with session.get(url) as response:
                    print(f"Status: {response.status}")
                    print(f"Headers: {dict(response.headers)}")
                    
                    if response.status == 200:
                        data = await response.json()
                        print(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                        print(f"Response sample: {str(data)[:200]}...")
                    else:
                        text = await response.text()
                        print(f"Error response: {text[:200]}...")
                        
        except Exception as e:
            print(f"Exception: {e}")

async def debug_yahoo(symbol='BTCUSDT'):
    """Debug Yahoo Finance"""
    print(f"\n🔍 DEBUGGING Yahoo Finance for {symbol}")
    
    import yfinance as yf
    
    yahoo_symbol = symbol.replace('USDT', '-USD')
    print(f"Yahoo symbol: {yahoo_symbol}")
    
    try:
        ticker = yf.Ticker(yahoo_symbol)
        print(f"Ticker info: {ticker.info.get('symbol', 'No symbol')} - {ticker.info.get('longName', 'No name')}")
        
        hist = ticker.history(period="10d")
        print(f"History shape: {hist.shape}")
        print(f"History columns: {list(hist.columns)}")
        print(f"History index: {hist.index}")
        
        if len(hist) > 0:
            print(f"Latest data: {hist.iloc[-1]}")
        
    except Exception as e:
        print(f"Exception: {e}")

async def debug_coingecko(symbol='BTCUSDT'):
    """Debug CoinGecko API"""
    print(f"\n🔍 DEBUGGING CoinGecko API for {symbol}")
    
    # First, try to find the coin ID
    base_symbol = symbol.replace('USDT', '').lower()
    
    search_url = "https://api.coingecko.com/api/v3/search"
    params = {"query": base_symbol}
    
    print(f"Search URL: {search_url}")
    print(f"Search params: {params}")
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            async with session.get(search_url, params=params) as response:
                print(f"Search status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    coins = data.get('coins', [])
                    print(f"Found {len(coins)} coins")
                    
                    # Find exact match
                    coin_id = None
                    for coin in coins[:5]:
                        print(f"Coin: {coin.get('id')} - {coin.get('symbol')} - {coin.get('name')}")
                        if coin.get('symbol', '').upper() == base_symbol.upper():
                            coin_id = coin['id']
                            break
                    
                    if coin_id:
                        print(f"Using coin ID: {coin_id}")
                        
                        # Now try to get OHLC data
                        ohlc_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
                        ohlc_params = {"vs_currency": "usd", "days": "10"}
                        
                        async with session.get(ohlc_url, params=ohlc_params) as ohlc_response:
                            print(f"OHLC status: {ohlc_response.status}")
                            
                            if ohlc_response.status == 200:
                                ohlc_data = await ohlc_response.json()
                                print(f"OHLC data length: {len(ohlc_data)}")
                                if ohlc_data:
                                    print(f"First OHLC: {ohlc_data[0]}")
                            else:
                                ohlc_text = await ohlc_response.text()
                                print(f"OHLC error: {ohlc_text[:200]}...")
                    else:
                        print("No matching coin found")
                else:
                    text = await response.text()
                    print(f"Search error: {text[:200]}...")
                    
    except Exception as e:
        print(f"Exception: {e}")

async def main():
    """Run all debug tests"""
    print("🚀 Starting OHLCV Sources Debug")
    
    symbol = 'BTCUSDT'
    
    await debug_bingx(symbol)
    await debug_binance(symbol)
    await debug_coindesk(symbol)
    await debug_yahoo(symbol)
    await debug_coingecko(symbol)

if __name__ == "__main__":
    asyncio.run(main())
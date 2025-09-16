#!/usr/bin/env python3
"""
Test script for BingX official futures page scraping
"""

import asyncio
import sys
import os
sys.path.append('/app/backend')

from bingx_symbol_fetcher import BingXFuturesFetcher

async def test_official_scraping():
    """Test the new official BingX scraping"""
    fetcher = BingXFuturesFetcher()
    
    print("🔍 Testing BingX official futures page scraping...")
    print(f"🌐 URL: {fetcher.futures_page_url}")
    print()
    
    # Test official scraping
    symbols = await fetcher.get_official_futures_symbols()
    
    if symbols:
        print(f"✅ SUCCESS: Found {len(symbols)} symbols")
        print(f"📋 First 20 symbols: {symbols[:20]}")
        print()
        
        # Check for common symbols
        common_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
        found = [s for s in common_symbols if s in symbols]
        missing = [s for s in common_symbols if s not in symbols]
        
        print(f"✅ Common symbols found: {found}")
        if missing:
            print(f"❌ Common symbols missing: {missing}")
        
        # Check if XTZUSDT is correctly excluded
        if 'XTZUSDT' in symbols:
            print("❌ PROBLEM: XTZUSDT found in symbols (should be excluded)")
        else:
            print("✅ GOOD: XTZUSDT correctly excluded")
            
    else:
        print("❌ FAILED: No symbols found")
    
    print()
    print("🔄 Testing sync wrapper...")
    sync_symbols = fetcher.get_tradable_symbols(force_update=True)
    print(f"✅ Sync method: Found {len(sync_symbols)} symbols")

if __name__ == "__main__":
    asyncio.run(test_official_scraping())
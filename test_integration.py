#!/usr/bin/env python3
"""
Test integration of enhanced OHLCV with the main system
"""
import sys
import asyncio
sys.path.append('/app/backend')

from enhanced_ohlcv_fetcher import enhanced_ohlcv_fetcher

async def test_integration():
    print('🔍 Testing enhanced OHLCV integration...')
    try:
        data = await enhanced_ohlcv_fetcher.get_enhanced_ohlcv_data('BTCUSDT')
        if data is not None and len(data) > 0:
            print(f'✅ SUCCESS: Got {len(data)} days of BTCUSDT data')
            print(f'💰 Latest Close: ${data["Close"].iloc[-1]:.2f}')
            attrs = getattr(data, 'attrs', {})
            print(f'🏆 Primary: {attrs.get("primary_source", "Unknown")}')
            print(f'🥈 Secondary: {attrs.get("secondary_source", "None")}')
            return True
        else:
            print('❌ FAILED: No data returned')
            return False
    except Exception as e:
        print(f'❌ ERROR: {e}')
        return False

if __name__ == "__main__":
    result = asyncio.run(test_integration())
    print(f'Integration test: {"PASSED" if result else "FAILED"}')
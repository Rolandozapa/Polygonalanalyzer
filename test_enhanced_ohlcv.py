#!/usr/bin/env python3
"""
Test script for Enhanced OHLCV Fetcher with new data sources
"""
import asyncio
import sys
import os
import logging

# Add backend to path
sys.path.append('/app/backend')

from enhanced_ohlcv_fetcher import EnhancedOHLCVFetcher

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_single_source(fetcher, symbol, source_name, fetch_method):
    """Test a single data source"""
    try:
        logger.info(f"🔍 Testing {source_name} for {symbol}")
        data = await fetch_method(symbol)
        
        if data is not None and len(data) > 0:
            logger.info(f"✅ {source_name} SUCCESS: {len(data)} days of data for {symbol}")
            logger.info(f"   📊 Latest Close: ${data['Close'].iloc[-1]:.4f}")
            logger.info(f"   📅 Date Range: {data.index[0].date()} to {data.index[-1].date()}")
            return True
        else:
            logger.warning(f"❌ {source_name} FAILED: No data returned for {symbol}")
            return False
            
    except Exception as e:
        logger.error(f"❌ {source_name} ERROR for {symbol}: {e}")
        return False

async def test_enhanced_ohlcv():
    """Test the enhanced OHLCV fetcher with multiple sources"""
    logger.info("🚀 Starting Enhanced OHLCV Fetcher Test")
    
    # Initialize fetcher
    fetcher = EnhancedOHLCVFetcher()
    
    # Test symbols
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    # Test individual sources
    test_sources = [
        ('BingX Enhanced', fetcher._fetch_bingx_enhanced),
        ('Binance Enhanced', fetcher._fetch_binance_enhanced),
        ('CoinDesk Enhanced', fetcher._fetch_coindesk_enhanced),
        ('Kraken Enhanced', fetcher._fetch_kraken_enhanced),
        ('CoinGecko Enhanced', fetcher._fetch_coingecko_enhanced),
        ('Yahoo Finance Enhanced', fetcher._fetch_yahoo_enhanced)
    ]
    
    results = {}
    
    for symbol in test_symbols:
        logger.info(f"\n🎯 Testing sources for {symbol}")
        results[symbol] = {}
        
        for source_name, fetch_method in test_sources:
            success = await test_single_source(fetcher, symbol, source_name, fetch_method)
            results[symbol][source_name] = success
    
    # Test the main enhanced method (multi-source)
    logger.info(f"\n🔄 Testing Multi-Source Enhanced Method")
    for symbol in test_symbols:
        try:
            logger.info(f"🔍 Testing enhanced multi-source for {symbol}")
            data = await fetcher.get_enhanced_ohlcv_data(symbol)
            
            if data is not None and len(data) > 0:
                attrs = getattr(data, 'attrs', {})
                primary_source = attrs.get('primary_source', 'Unknown')
                secondary_source = attrs.get('secondary_source', 'None')
                validation_rate = attrs.get('validation_rate', 0)
                
                logger.info(f"✅ MULTI-SOURCE SUCCESS for {symbol}:")
                logger.info(f"   📊 {len(data)} days of data")
                logger.info(f"   🏆 Primary: {primary_source}")
                logger.info(f"   🥈 Secondary: {secondary_source}")
                logger.info(f"   ✅ Validation: {validation_rate*100:.1f}%")
                logger.info(f"   💰 Latest Close: ${data['Close'].iloc[-1]:.4f}")
                
                results[symbol]['Multi-Source Enhanced'] = True
            else:
                logger.warning(f"❌ MULTI-SOURCE FAILED for {symbol}")
                results[symbol]['Multi-Source Enhanced'] = False
                
        except Exception as e:
            logger.error(f"❌ MULTI-SOURCE ERROR for {symbol}: {e}")
            results[symbol]['Multi-Source Enhanced'] = False
    
    # Print summary
    logger.info(f"\n📋 TEST SUMMARY")
    logger.info("=" * 60)
    
    for symbol in test_symbols:
        logger.info(f"\n{symbol}:")
        for source, success in results[symbol].items():
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"  {source}: {status}")
    
    # Calculate overall success rates
    for source_name, _ in test_sources + [('Multi-Source Enhanced', None)]:
        total = len(test_symbols)
        passed = sum(1 for symbol in test_symbols if results[symbol].get(source_name, False))
        rate = (passed / total) * 100
        logger.info(f"\n{source_name}: {passed}/{total} ({rate:.1f}%)")

if __name__ == "__main__":
    asyncio.run(test_enhanced_ohlcv())
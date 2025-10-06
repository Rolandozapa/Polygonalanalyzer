#!/usr/bin/env python3
"""
🚀 COMPREHENSIVE API TESTING - ALL ENHANCED OHLCV SOURCES
Test all available APIs with provided premium keys
"""
import asyncio
import sys
import os
import logging

# Add backend to path
sys.path.append('/app/backend')

from enhanced_ohlcv_fetcher import EnhancedOHLCVFetcher

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_api_source(fetcher, symbol, source_name, fetch_method):
    """Test a single API source with detailed reporting"""
    try:
        logger.info(f"🔍 Testing {source_name} for {symbol}")
        data = await fetch_method(symbol)
        
        if data is not None and len(data) > 0:
            latest_close = data['Close'].iloc[-1]
            date_range = f"{data.index[0].date()} to {data.index[-1].date()}"
            
            # Check data quality
            has_volume = data['Volume'].sum() > 0
            has_realistic_prices = latest_close > 0.01  # Not fallback prices
            price_variation = (data['High'].max() - data['Low'].min()) / data['Close'].mean()
            
            quality_score = 0
            quality_indicators = []
            
            if has_volume:
                quality_score += 25
                quality_indicators.append("✅ Real Volume")
            else:
                quality_indicators.append("⚠️ Synthetic Volume")
                
            if has_realistic_prices:
                quality_score += 25
                quality_indicators.append("✅ Realistic Prices")
            else:
                quality_indicators.append("❌ Fallback Prices")
                
            if price_variation > 0.01:  # At least 1% price variation
                quality_score += 25
                quality_indicators.append("✅ Price Variation")
            else:
                quality_indicators.append("⚠️ Low Variation")
                
            if len(data) >= 7:  # At least a week of data
                quality_score += 25
                quality_indicators.append("✅ Sufficient History")
            else:
                quality_indicators.append("⚠️ Limited History")
            
            logger.info(f"✅ {source_name} SUCCESS:")
            logger.info(f"   📊 {len(data)} days of data | Quality: {quality_score}/100")
            logger.info(f"   💰 Latest Close: ${latest_close:.4f} | Range: {date_range}")
            logger.info(f"   🔍 Quality: {', '.join(quality_indicators)}")
            
            return {
                'success': True,
                'days': len(data),
                'price': latest_close,
                'quality_score': quality_score,
                'has_volume': has_volume,
                'has_realistic_prices': has_realistic_prices
            }
        else:
            logger.warning(f"❌ {source_name} FAILED: No data returned for {symbol}")
            return {'success': False, 'error': 'No data returned'}
            
    except Exception as e:
        logger.error(f"❌ {source_name} ERROR for {symbol}: {e}")
        return {'success': False, 'error': str(e)}

async def test_comprehensive_apis():
    """Test all enhanced API sources comprehensively"""
    logger.info("🚀 Starting Comprehensive Enhanced OHLCV API Testing")
    
    # Initialize fetcher
    fetcher = EnhancedOHLCVFetcher()
    
    # Test symbols
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    # All enhanced API sources to test
    api_sources = [
        ('BingX Enhanced', fetcher._fetch_bingx_enhanced),
        ('CoinMarketCap DEX Enhanced', fetcher._fetch_cmc_dex_enhanced),
        ('TwelveData Enhanced', fetcher._fetch_twelvedata_enhanced),
        ('CoinAPI Enhanced', fetcher._fetch_coinapi_enhanced),
        ('Kraken Enhanced', fetcher._fetch_kraken_enhanced),
        ('Bitfinex Enhanced', fetcher._fetch_bitfinex_enhanced),
        ('CoinGecko Enhanced', fetcher._fetch_coingecko_enhanced),
        ('CryptoCompare Enhanced', fetcher._fetch_cryptocompare_enhanced),
        ('Yahoo Finance Enhanced', fetcher._fetch_yahoo_enhanced)
    ]
    
    results = {}
    summary_stats = {}
    
    # Test each API source
    for symbol in test_symbols:
        logger.info(f"\n🎯 TESTING ALL SOURCES FOR {symbol}")
        logger.info("=" * 60)
        results[symbol] = {}
        
        for source_name, fetch_method in api_sources:
            result = await test_api_source(fetcher, symbol, source_name, fetch_method)
            results[symbol][source_name] = result
            
            # Update summary stats
            if source_name not in summary_stats:
                summary_stats[source_name] = {
                    'total_tests': 0,
                    'successes': 0,
                    'total_quality': 0,
                    'premium_data_sources': 0
                }
            
            summary_stats[source_name]['total_tests'] += 1
            if result['success']:
                summary_stats[source_name]['successes'] += 1
                summary_stats[source_name]['total_quality'] += result.get('quality_score', 0)
                if result.get('has_volume') and result.get('has_realistic_prices'):
                    summary_stats[source_name]['premium_data_sources'] += 1
    
    # Test multi-source enhanced method
    logger.info(f"\n🔄 TESTING MULTI-SOURCE ENHANCED METHOD")
    logger.info("=" * 60)
    
    for symbol in test_symbols:
        try:
            logger.info(f"🔍 Testing multi-source enhanced for {symbol}")
            data = await fetcher.get_enhanced_ohlcv_data(symbol)
            
            if data is not None and len(data) > 0:
                attrs = getattr(data, 'attrs', {})
                primary_source = attrs.get('primary_source', 'Unknown')
                secondary_source = attrs.get('secondary_source', 'None')
                validation_rate = attrs.get('validation_rate', 0)
                sources_count = attrs.get('sources_count', 1)
                
                logger.info(f"✅ MULTI-SOURCE SUCCESS for {symbol}:")
                logger.info(f"   📊 {len(data)} days | Sources: {sources_count}")
                logger.info(f"   🏆 Primary: {primary_source} | 🥈 Secondary: {secondary_source}")
                logger.info(f"   ✅ Validation: {validation_rate*100:.1f}% | 💰 Close: ${data['Close'].iloc[-1]:.4f}")
                
                results[symbol]['Multi-Source Enhanced'] = {
                    'success': True,
                    'primary_source': primary_source,
                    'sources_count': sources_count,
                    'validation_rate': validation_rate
                }
            else:
                logger.warning(f"❌ MULTI-SOURCE FAILED for {symbol}")
                results[symbol]['Multi-Source Enhanced'] = {'success': False}
                
        except Exception as e:
            logger.error(f"❌ MULTI-SOURCE ERROR for {symbol}: {e}")
            results[symbol]['Multi-Source Enhanced'] = {'success': False, 'error': str(e)}
    
    # Print comprehensive summary
    logger.info(f"\n📋 COMPREHENSIVE API TEST SUMMARY")
    logger.info("=" * 80)
    
    # API Source Performance
    logger.info(f"\n🏆 API SOURCE PERFORMANCE RANKINGS:")
    sorted_sources = sorted(summary_stats.items(), 
                          key=lambda x: (x[1]['successes']/x[1]['total_tests'], 
                                       x[1]['total_quality']/max(x[1]['total_tests'], 1)), 
                          reverse=True)
    
    for i, (source_name, stats) in enumerate(sorted_sources, 1):
        success_rate = (stats['successes'] / stats['total_tests']) * 100
        avg_quality = stats['total_quality'] / max(stats['total_tests'], 1)
        premium_rate = (stats['premium_data_sources'] / max(stats['successes'], 1)) * 100
        
        rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
        
        logger.info(f"{rank_emoji} {i:2d}. {source_name:<30} | "
                   f"Success: {success_rate:5.1f}% | "
                   f"Quality: {avg_quality:5.1f}/100 | "
                   f"Premium: {premium_rate:5.1f}%")
    
    # Detailed results by symbol
    logger.info(f"\n📊 DETAILED RESULTS BY SYMBOL:")
    for symbol in test_symbols:
        logger.info(f"\n{symbol}:")
        successful_sources = []
        failed_sources = []
        
        for source, result in results[symbol].items():
            if result.get('success', False):
                quality = result.get('quality_score', 0)
                successful_sources.append(f"  ✅ {source:<30} (Quality: {quality}/100)")
            else:
                error = result.get('error', 'Unknown error')[:50]
                failed_sources.append(f"  ❌ {source:<30} ({error})")
        
        for source in successful_sources:
            logger.info(source)
        for source in failed_sources:
            logger.info(source)
    
    # Final statistics
    total_tests = sum(stats['total_tests'] for stats in summary_stats.values())
    total_successes = sum(stats['successes'] for stats in summary_stats.values())
    overall_success_rate = (total_successes / total_tests) * 100 if total_tests > 0 else 0
    
    logger.info(f"\n🎯 FINAL STATISTICS:")
    logger.info(f"   Total API Tests: {total_tests}")
    logger.info(f"   Successful Tests: {total_successes}")
    logger.info(f"   Overall Success Rate: {overall_success_rate:.1f}%")
    logger.info(f"   Working Sources: {len([s for s in summary_stats if summary_stats[s]['successes'] > 0])}/{len(summary_stats)}")

if __name__ == "__main__":
    asyncio.run(test_comprehensive_apis())
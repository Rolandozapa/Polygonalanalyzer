#!/usr/bin/env python3
"""
Test script for the new TALib indicators system
Phase 1: Complete & Test TALib Indicators System
"""

import sys
import os
sys.path.append('/app/backend')
sys.path.append('/app/core/indicators')

import pandas as pd
import numpy as np
import logging
from talib_indicators import TALibIndicators, get_talib_indicators

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(periods=100, symbol="BTCUSDT"):
    """Create realistic test OHLCV data"""
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic price data
    base_price = 50000.0
    prices = []
    volumes = []
    
    for i in range(periods):
        # Random walk with trend
        change = np.random.normal(0, 0.02)  # 2% volatility
        if i > 0:
            base_price *= (1 + change)
        else:
            base_price = 50000.0
        
        # OHLC around base price
        high = base_price * (1 + abs(np.random.normal(0, 0.01)))
        low = base_price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = base_price + np.random.normal(0, base_price * 0.005)
        close = base_price
        
        # Volume
        volume = np.random.uniform(1000000, 5000000)
        
        prices.append([open_price, high, low, close])
        volumes.append(volume)
    
    df = pd.DataFrame(prices, columns=['open', 'high', 'low', 'close'])
    df['volume'] = volumes
    
    logger.info(f"Created test data: {len(df)} rows for {symbol}")
    logger.info(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    logger.info(f"Current price: ${df['close'].iloc[-1]:.2f}")
    
    return df

def test_talib_integration():
    """Test the TALib indicators integration"""
    
    logger.info("🚀 TESTING TALIB INDICATORS SYSTEM - PHASE 1")
    logger.info("=" * 60)
    
    # Create test data
    test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    for symbol in test_symbols:
        logger.info(f"\n🔬 Testing {symbol}")
        logger.info("-" * 40)
        
        # Create test data
        df = create_test_data(periods=100, symbol=symbol)
        
        try:
            # Initialize TALib indicators
            talib_indicators = TALibIndicators()
            
            # Test data validation
            logger.info(f"📊 Validating data for {symbol}...")
            is_valid = talib_indicators.validate_data(df)
            logger.info(f"   ✅ Data validation: {'PASSED' if is_valid else 'FAILED'}")
            
            # Test minimum periods
            min_periods = talib_indicators.get_min_periods()
            logger.info(f"   📏 Minimum periods required: {min_periods}")
            logger.info(f"   📊 Data available: {len(df)} periods")
            
            if len(df) >= min_periods:
                logger.info(f"   ✅ Sufficient data for full analysis")
            else:
                logger.info(f"   ⚠️ Limited data - will use adaptive calculations")
            
            # Calculate all indicators
            logger.info(f"\n🧮 Calculating ALL TALib indicators for {symbol}...")
            
            analysis = talib_indicators.calculate_all_indicators(df, symbol)
            
            # Test results
            logger.info(f"\n📊 RESULTS FOR {symbol}:")
            logger.info(f"   🎯 Regime: {analysis.regime} (Confidence: {analysis.confidence:.1%})")
            logger.info(f"   📈 RSI: {analysis.rsi_14:.1f} [{analysis.rsi_zone}]")
            logger.info(f"   📊 MACD: Line={analysis.macd_line:.6f}, Signal={analysis.macd_signal:.6f}, Hist={analysis.macd_histogram:.6f}")
            logger.info(f"   💪 ADX: {analysis.adx:.1f} [{analysis.adx_strength}]")
            logger.info(f"   🔔 Bollinger: Position={analysis.bb_position:.3f}, Squeeze={analysis.bb_squeeze}")
            logger.info(f"   💰 MFI: {analysis.mfi:.1f} [{analysis.mfi_signal}]")
            logger.info(f"   📊 VWAP: Distance={analysis.vwap_distance:+.2f}%, Above={analysis.above_vwap}")
            logger.info(f"   📈 Volume: Ratio={analysis.volume_ratio:.2f}, Surge={analysis.volume_surge}")
            logger.info(f"   🏆 Confluence: {analysis.confluence_grade} (Score: {analysis.confluence_score}, Should Trade: {analysis.should_trade})")
            
            # Verify all required fields are present and not None/NaN
            required_fields = [
                'regime', 'confidence', 'rsi_14', 'macd_line', 'macd_signal', 'macd_histogram',
                'adx', 'bb_position', 'mfi', 'vwap', 'volume_ratio', 'confluence_grade'
            ]
            
            missing_fields = []
            for field in required_fields:
                value = getattr(analysis, field, None)
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    missing_fields.append(field)
            
            if missing_fields:
                logger.error(f"   ❌ Missing or invalid fields: {missing_fields}")
                return False
            else:
                logger.info(f"   ✅ All required fields present and valid")
            
        except Exception as e:
            logger.error(f"❌ Error testing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    logger.info("\n✅ TALib indicators system test COMPLETED successfully!")
    return True

def test_global_instance():
    """Test the global TALib indicators instance"""
    
    logger.info(f"\n🔧 Testing global TALib indicators instance...")
    
    try:
        # Get global instance
        global_indicators = get_talib_indicators()
        
        # Test with simple data
        df = create_test_data(periods=50, symbol="TEST")
        
        # Calculate indicators
        analysis = global_indicators.calculate_all_indicators(df, "TEST")
        
        logger.info(f"   ✅ Global instance working: {analysis.confluence_grade} grade")
        return True
        
    except Exception as e:
        logger.error(f"❌ Global instance test failed: {e}")
        return False

def main():
    """Main test function"""
    
    logger.info("🎯 PHASE 1: COMPLETE & TEST TALIB INDICATORS SYSTEM")
    logger.info("=" * 80)
    
    # Test 1: TALib Integration
    test1_result = test_talib_integration()
    
    # Test 2: Global Instance
    test2_result = test_global_instance()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST SUMMARY:")
    logger.info(f"   TALib Integration Test: {'✅ PASSED' if test1_result else '❌ FAILED'}")
    logger.info(f"   Global Instance Test: {'✅ PASSED' if test2_result else '❌ FAILED'}")
    
    overall_success = test1_result and test2_result
    
    if overall_success:
        logger.info("\n🎉 ALL TESTS PASSED - TALib Indicators System is READY!")
        logger.info("   Next: Integration with server.py")
    else:
        logger.error("\n❌ SOME TESTS FAILED - Need to fix issues before integration")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
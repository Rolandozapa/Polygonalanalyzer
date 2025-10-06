#!/usr/bin/env python3
"""
Test TALib integration with server.py
Test the integrated TALib indicators system
"""

import sys
import os
sys.path.append('/app/backend')
sys.path.append('/app/core')

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_ohlcv_data():
    """Create test OHLCV data with capitalized columns (like enhanced_ohlcv_fetcher)"""
    
    np.random.seed(42)
    periods = 100
    base_price = 50000.0
    
    dates = pd.date_range(start=datetime.now(), periods=periods, freq='1D')
    data = []
    
    for i in range(periods):
        if i > 0:
            base_price *= (1 + np.random.normal(0, 0.02))
        
        high = base_price * (1 + abs(np.random.normal(0, 0.01)))
        low = base_price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = base_price + np.random.normal(0, base_price * 0.005)
        close = base_price
        volume = np.random.uniform(1000000, 5000000)
        
        data.append({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    
    logger.info(f"Created test OHLCV data: {len(df)} rows")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    
    return df

def test_talib_integration():
    """Test TALib integration like server.py would use it"""
    
    logger.info("🚀 TESTING TALIB INTEGRATION WITH SERVER SETUP")
    logger.info("=" * 60)
    
    try:
        # Import exactly like server.py does
        import sys
        sys.path.insert(0, '/app/core')
        from indicators.talib_indicators import TALibIndicators, get_talib_indicators
        
        logger.info("✅ TALib indicators import successful")
        
        # Create test data with capitalized columns (like enhanced_ohlcv_fetcher)
        historical_data = create_test_ohlcv_data()
        
        # Test the integration like server.py does
        logger.info("\n🔬 Testing TALib integration with capitalized columns...")
        
        # Use the global instance like server.py
        talib_indicators = get_talib_indicators()
        talib_analysis = talib_indicators.calculate_all_indicators(historical_data, "BTCUSDT")
        
        # Test results like server.py extracts them
        rsi = talib_analysis.rsi_14
        macd_signal = talib_analysis.macd_signal
        macd_line = talib_analysis.macd_line
        macd_histogram = talib_analysis.macd_histogram
        stochastic_k = talib_analysis.stoch_k
        stochastic_d = talib_analysis.stoch_d
        bb_position = talib_analysis.bb_position
        adx = talib_analysis.adx
        atr = talib_analysis.atr
        vwap = talib_analysis.vwap
        vwap_position = talib_analysis.vwap_distance
        volume_ratio = talib_analysis.volume_ratio
        mfi = talib_analysis.mfi
        
        logger.info("\n📊 EXTRACTED INDICATORS (like server.py):")
        logger.info(f"   RSI: {rsi:.2f}")
        logger.info(f"   MACD: Line={macd_line:.6f}, Signal={macd_signal:.6f}, Hist={macd_histogram:.6f}")
        logger.info(f"   Stochastic: K={stochastic_k:.2f}, D={stochastic_d:.2f}")
        logger.info(f"   Bollinger Position: {bb_position:.3f}")
        logger.info(f"   ADX: {adx:.2f}")
        logger.info(f"   ATR: {atr:.6f}")
        logger.info(f"   VWAP: {vwap:.2f}")
        logger.info(f"   VWAP Distance: {vwap_position:+.2f}%")
        logger.info(f"   Volume Ratio: {volume_ratio:.2f}")
        logger.info(f"   MFI: {mfi:.2f}")
        
        # Test confluence and regime detection
        logger.info(f"\n🎯 ANALYSIS RESULTS:")
        logger.info(f"   Regime: {talib_analysis.regime}")
        logger.info(f"   Confidence: {talib_analysis.confidence:.1%}")
        logger.info(f"   Confluence Grade: {talib_analysis.confluence_grade}")
        logger.info(f"   Confluence Score: {talib_analysis.confluence_score}")
        logger.info(f"   Should Trade: {talib_analysis.should_trade}")
        logger.info(f"   Conviction Level: {talib_analysis.conviction_level}")
        
        # Verify all values are valid (not None or NaN)
        critical_indicators = [
            ('RSI', rsi), ('MACD Line', macd_line), ('MACD Signal', macd_signal), 
            ('MACD Histogram', macd_histogram), ('ADX', adx), ('ATR', atr),
            ('VWAP', vwap), ('MFI', mfi), ('Volume Ratio', volume_ratio)
        ]
        
        all_valid = True
        for name, value in critical_indicators:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                logger.error(f"❌ Invalid {name}: {value}")
                all_valid = False
            else:
                logger.debug(f"✅ Valid {name}: {value}")
        
        if all_valid:
            logger.info("\n✅ ALL INDICATORS VALID - TALib integration working correctly!")
            logger.info("   Ready for production use with server.py")
            return True
        else:
            logger.error("\n❌ Some indicators are invalid - needs fixing")
            return False
            
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    
    logger.info("🎯 TALib Integration Test with Server Setup")
    logger.info("=" * 80)
    
    success = test_talib_integration()
    
    logger.info("\n" + "=" * 80)
    if success:
        logger.info("🎉 TALib INTEGRATION TEST PASSED!")
        logger.info("   The new TALib system is ready for use in server.py")
        logger.info("   It correctly handles capitalized OHLCV columns from enhanced_ohlcv_fetcher")
    else:
        logger.error("❌ TALib integration test FAILED")
        logger.error("   Need to fix issues before using in server.py")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
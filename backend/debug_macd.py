#!/usr/bin/env python3
"""
Debug MACD calculation issue
"""
import pandas as pd
import numpy as np
import asyncio
import logging
from macd_calculator import calculate_macd_optimized, macd_calculator
from enhanced_ohlcv_fetcher import enhanced_ohlcv_fetcher

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def debug_macd_issue():
    """Debug why MACD returns 0.000000"""
    
    # Test 1: Simple synthetic data
    logger.info("=== TEST 1: SYNTHETIC DATA ===")
    synthetic_prices = pd.Series([
        100, 101, 102, 101, 103, 104, 102, 105, 106, 104, 
        107, 108, 106, 109, 110, 108, 111, 112, 110, 113,
        114, 112, 115, 116, 114, 117, 118, 116, 119, 120,
        118, 121, 122, 120, 123, 124, 122, 125, 126, 124
    ])
    
    macd_line, macd_signal, macd_histogram = calculate_macd_optimized(synthetic_prices)
    logger.info(f"Synthetic data MACD: Line={macd_line:.8f}, Signal={macd_signal:.8f}, Histogram={macd_histogram:.8f}")
    
    # Test 2: Real OHLCV data
    logger.info("\n=== TEST 2: REAL OHLCV DATA ===")
    try:
        # Fetch real data for BTCUSDT
        symbol = "BTCUSDT"
        real_data = await enhanced_ohlcv_fetcher.get_enhanced_ohlcv_data(symbol)
        
        if real_data is not None and len(real_data) > 0:
            logger.info(f"Real data shape: {real_data.shape}")
            logger.info(f"Real data columns: {real_data.columns.tolist()}")
            logger.info(f"Real data sample:\n{real_data.head()}")
            
            # Find Close column
            close_col = None
            for col in ['Close', 'close']:
                if col in real_data.columns:
                    close_col = col
                    break
            
            if close_col:
                close_prices = real_data[close_col]
                logger.info(f"Close prices shape: {len(close_prices)}")
                logger.info(f"Close prices sample: {close_prices.head()}")
                logger.info(f"Close prices stats: min={close_prices.min():.2f}, max={close_prices.max():.2f}, mean={close_prices.mean():.2f}")
                
                # Test MACD calculator
                macd_line, macd_signal, macd_histogram = calculate_macd_optimized(close_prices)
                logger.info(f"Real data MACD: Line={macd_line:.8f}, Signal={macd_signal:.8f}, Histogram={macd_histogram:.8f}")
                
                # Direct test with MACD calculator class
                result = macd_calculator.calculate(close_prices)
                logger.info(f"Direct MACD result: {result}")
                
            else:
                logger.error("No Close column found in real data")
        else:
            logger.error("No real data fetched")
            
    except Exception as e:
        logger.error(f"Error testing real data: {e}")
    
    # Test 3: Edge cases
    logger.info("\n=== TEST 3: EDGE CASES ===")
    
    # Empty data
    empty_data = pd.Series([])
    macd_line, macd_signal, macd_histogram = calculate_macd_optimized(empty_data)
    logger.info(f"Empty data MACD: Line={macd_line:.8f}, Signal={macd_signal:.8f}, Histogram={macd_histogram:.8f}")
    
    # Insufficient data
    small_data = pd.Series([100, 101, 102])
    macd_line, macd_signal, macd_histogram = calculate_macd_optimized(small_data)
    logger.info(f"Small data MACD: Line={macd_line:.8f}, Signal={macd_signal:.8f}, Histogram={macd_histogram:.8f}")

if __name__ == "__main__":
    asyncio.run(debug_macd_issue())
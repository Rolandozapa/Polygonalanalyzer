#!/usr/bin/env python3
"""
Detailed debug of MACD calculation steps
"""
import pandas as pd
import numpy as np
import asyncio
import logging
from macd_calculator import MACDCalculator
from enhanced_ohlcv_fetcher import enhanced_ohlcv_fetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_macd_steps():
    """Debug each step of MACD calculation"""
    
    # Get real data
    symbol = "BTCUSDT"
    real_data = await enhanced_ohlcv_fetcher.get_enhanced_ohlcv_data(symbol)
    
    if real_data is None or len(real_data) < 35:
        logger.error("Cannot get sufficient data")
        return
    
    close_prices = real_data['Close']
    logger.info(f"Close prices length: {len(close_prices)}")
    logger.info(f"Close prices range: {close_prices.min():.2f} - {close_prices.max():.2f}")
    
    # Manual MACD calculation step by step
    calculator = MACDCalculator()
    
    # Step 1: Calculate fast EMA (12)
    fast_ema = calculator._calculate_ema(close_prices, 12)
    logger.info(f"Fast EMA (12): Length={len(fast_ema)}, Last value={fast_ema.iloc[-1]:.2f}")
    
    # Step 2: Calculate slow EMA (26)
    slow_ema = calculator._calculate_ema(close_prices, 26)
    logger.info(f"Slow EMA (26): Length={len(slow_ema)}, Last value={slow_ema.iloc[-1]:.2f}")
    
    # Step 3: Calculate MACD line
    macd_line = fast_ema - slow_ema
    macd_line_clean = macd_line.dropna()
    logger.info(f"MACD line: Length={len(macd_line_clean)}, Range={macd_line_clean.min():.2f} - {macd_line_clean.max():.2f}")
    logger.info(f"MACD line last 5 values: {macd_line_clean.tail().tolist()}")
    
    # Step 4: Calculate signal line (9-period EMA of MACD)
    signal_line = calculator._calculate_ema(macd_line, 9)
    signal_line_clean = signal_line.dropna()
    logger.info(f"Signal line: Length={len(signal_line_clean)}, Last value={signal_line_clean.iloc[-1]:.2f}")
    
    # Final calculation
    result = calculator.calculate(close_prices)
    logger.info(f"Final MACD result: {result}")

if __name__ == "__main__":
    asyncio.run(debug_macd_steps())
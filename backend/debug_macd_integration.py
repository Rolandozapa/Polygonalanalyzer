#!/usr/bin/env python3
"""
Debug MACD integration between advanced_technical_indicators and optimized calculator
"""
import pandas as pd
import numpy as np
import asyncio
import logging
from enhanced_ohlcv_fetcher import enhanced_ohlcv_fetcher
from advanced_technical_indicators import AdvancedTechnicalIndicators
from macd_calculator import calculate_macd_optimized

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_macd_integration():
    """Compare MACD from both systems"""
    
    # Get real data
    symbol = "BTCUSDT"
    real_data = await enhanced_ohlcv_fetcher.get_enhanced_ohlcv_data(symbol)
    
    if real_data is None or len(real_data) < 35:
        logger.error("Cannot get sufficient data")
        return
    
    logger.info(f"Data shape: {real_data.shape}")
    logger.info(f"Data columns: {real_data.columns.tolist()}")
    logger.info(f"Data sample:\n{real_data.head()}")
    
    # Test 1: Advanced Technical Indicators system
    logger.info("\n=== ADVANCED TECHNICAL INDICATORS SYSTEM ===")
    try:
        advanced_indicators = AdvancedTechnicalIndicators()
        df_with_indicators = advanced_indicators.calculate_all_indicators(real_data)
        current_indicators = advanced_indicators.get_current_indicators(df_with_indicators)
        
        logger.info(f"Advanced Indicators MACD:")
        logger.info(f"  MACD Line: {current_indicators.macd_line:.8f}")
        logger.info(f"  MACD Signal: {current_indicators.macd_signal:.8f}")
        logger.info(f"  MACD Histogram: {current_indicators.macd_histogram:.8f}")
        
        # Check if MACD columns exist in DataFrame
        macd_columns = [col for col in df_with_indicators.columns if 'macd' in col.lower()]
        logger.info(f"MACD columns in DataFrame: {macd_columns}")
        
        if macd_columns:
            last_row = df_with_indicators.iloc[-1]
            for col in macd_columns:
                logger.info(f"  {col}: {last_row[col]:.8f}")
        
    except Exception as e:
        logger.error(f"Error with advanced indicators: {e}")
    
    # Test 2: Optimized MACD Calculator
    logger.info("\n=== OPTIMIZED MACD CALCULATOR ===")
    try:
        close_prices = real_data['Close']
        macd_line, macd_signal, macd_histogram = calculate_macd_optimized(close_prices)
        logger.info(f"Optimized Calculator MACD:")
        logger.info(f"  MACD Line: {macd_line:.8f}")
        logger.info(f"  MACD Signal: {macd_signal:.8f}")
        logger.info(f"  MACD Histogram: {macd_histogram:.8f}")
        
    except Exception as e:
        logger.error(f"Error with optimized calculator: {e}")
    
    # Test 3: Compare both methods side-by-side
    logger.info("\n=== DIRECT COMPARISON ===")
    try:
        # Manual MACD calculation like advanced_indicators
        ema_12 = real_data['Close'].ewm(span=12).mean()
        ema_26 = real_data['Close'].ewm(span=26).mean()
        macd_line_manual = ema_12 - ema_26
        macd_signal_manual = macd_line_manual.ewm(span=9).mean()
        macd_histogram_manual = macd_line_manual - macd_signal_manual
        
        logger.info(f"Manual calculation (like advanced_indicators):")
        logger.info(f"  MACD Line: {macd_line_manual.iloc[-1]:.8f}")
        logger.info(f"  MACD Signal: {macd_signal_manual.iloc[-1]:.8f}")
        logger.info(f"  MACD Histogram: {macd_histogram_manual.iloc[-1]:.8f}")
        
        logger.info(f"Data stats:")
        logger.info(f"  Close prices count: {len(real_data['Close'])}")
        logger.info(f"  Close prices range: {real_data['Close'].min():.2f} - {real_data['Close'].max():.2f}")
        logger.info(f"  EMA12 last value: {ema_12.iloc[-1]:.2f}")
        logger.info(f"  EMA26 last value: {ema_26.iloc[-1]:.2f}")
        
    except Exception as e:
        logger.error(f"Error with manual calculation: {e}")

if __name__ == "__main__":
    asyncio.run(debug_macd_integration())
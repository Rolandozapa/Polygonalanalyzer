#!/usr/bin/env python3
"""
Debug script to test the specific datetime issue that occurs in the IA1 analysis flow
"""
import sys
sys.path.append('/app/backend')

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import logging
import asyncio

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    try:
        from enhanced_ohlcv_fetcher import enhanced_ohlcv_fetcher
        from advanced_technical_indicators import AdvancedTechnicalIndicators
        
        print("Testing OHLCV fetching and technical indicator calculation...")
        
        # Test with a real symbol from BingX
        symbol = "SHIBUSDT"
        
        print(f"\n1. Fetching OHLCV data for {symbol}...")
        historical_data = await enhanced_ohlcv_fetcher.get_enhanced_ohlcv_data(symbol)
        
        if historical_data is None:
            print(f"❌ No OHLCV data retrieved for {symbol}")
            return
            
        print(f"✅ Retrieved {len(historical_data)} periods of OHLCV data")
        print(f"   Index type: {type(historical_data.index)}")
        print(f"   Index timezone info: {historical_data.index.tz}")
        print(f"   Sample index values: {historical_data.index[:3].tolist()}")
        
        # Check for datetime-related issues in the data
        print(f"\n2. Checking for datetime issues in the data...")
        
        # Check if index has mixed timezone awareness
        if hasattr(historical_data.index, 'tz'):
            if historical_data.index.tz is None:
                print("   Index is timezone-naive")
            else:
                print(f"   Index is timezone-aware: {historical_data.index.tz}")
        
        # Check for any datetime operations that might cause issues
        print(f"\n3. Testing technical indicators calculation...")
        
        try:
            indicators_calc = AdvancedTechnicalIndicators()
            df_with_indicators = indicators_calc.calculate_all_indicators(historical_data)
            print(f"✅ Technical indicators calculated successfully")
            print(f"   Result shape: {df_with_indicators.shape}")
            
            # Extract current indicators
            current_indicators = indicators_calc.get_current_indicators(df_with_indicators)
            print(f"\n4. Current indicators extracted:")
            print(f"   RSI: {current_indicators.rsi_14}")
            print(f"   MACD Signal: {current_indicators.macd_signal}")
            print(f"   MFI: {current_indicators.mfi}")
            print(f"   VWAP: {current_indicators.vwap}")
            
        except Exception as e:
            print(f"❌ Error in technical indicators calculation: {e}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Error in test setup: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
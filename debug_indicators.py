#!/usr/bin/env python3
"""
Debug script to test technical indicators calculation and see what columns are actually created
"""
import sys
sys.path.append('/app/backend')

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from advanced_technical_indicators import AdvancedTechnicalIndicators

# Create sample OHLCV data
dates = pd.date_range(start='2025-01-01', periods=50, freq='D')
np.random.seed(42)

# Generate realistic price data
base_price = 100
prices = []
for i in range(50):
    if i == 0:
        prices.append(base_price)
    else:
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        prices.append(prices[-1] * (1 + change))

# Create OHLCV data
data = []
for i, price in enumerate(prices):
    high = price * (1 + abs(np.random.normal(0, 0.01)))
    low = price * (1 - abs(np.random.normal(0, 0.01)))
    volume = np.random.randint(1000, 10000)
    
    data.append({
        'Open': price,
        'High': high,
        'Low': low,
        'Close': price,
        'Volume': volume
    })

df = pd.DataFrame(data, index=dates)
print("Original DataFrame shape:", df.shape)
print("Original columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Test technical indicators
try:
    indicators_calc = AdvancedTechnicalIndicators()
    df_with_indicators = indicators_calc.calculate_all_indicators(df)
    
    print("\n" + "="*50)
    print("AFTER CALCULATION:")
    print("DataFrame shape:", df_with_indicators.shape)
    print("All columns:", df_with_indicators.columns.tolist())
    
    # Check for specific indicator columns
    expected_indicators = ['rsi_14', 'macd_signal', 'macd_histogram', 'stoch_k', 'stoch_d', 
                          'bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'mfi', 'vwap']
    
    print("\nExpected vs Actual indicators:")
    for indicator in expected_indicators:
        exists = indicator in df_with_indicators.columns
        print(f"  {indicator}: {'✅' if exists else '❌'}")
        if exists:
            last_value = df_with_indicators[indicator].iloc[-1]
            print(f"    Last value: {last_value}")
    
    # Test get_current_indicators method
    current_indicators = indicators_calc.get_current_indicators(df_with_indicators)
    print("\n" + "="*50)
    print("CURRENT INDICATORS OBJECT:")
    print(f"RSI: {current_indicators.rsi_14}")
    print(f"MACD Signal: {current_indicators.macd_signal}")
    print(f"MACD Histogram: {current_indicators.macd_histogram}")
    print(f"Stochastic K: {current_indicators.stoch_k}")
    print(f"BB Position: {current_indicators.bb_position}")
    print(f"MFI: {current_indicators.mfi}")
    print(f"VWAP: {current_indicators.vwap}")
    
    # Check for any NaN or None values in the last row
    last_row = df_with_indicators.iloc[-1]
    print("\n" + "="*50)
    print("LAST ROW VALUES:")
    for col in df_with_indicators.columns:
        value = last_row[col]
        if pd.isna(value):
            print(f"  {col}: NaN")
        elif value is None:
            print(f"  {col}: None")
        else:
            print(f"  {col}: {value}")
            
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
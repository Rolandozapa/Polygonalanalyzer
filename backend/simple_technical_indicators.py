"""
SIMPLE TECHNICAL INDICATORS FOR IA1
Direct, functional calculations that actually work
No complex classes, just working functions
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate RSI - Simple and working"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 0.001)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])
    except Exception as e:
        logger.error(f"RSI calculation error: {e}")
        return None

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
    """Calculate MACD - Simple and working"""
    try:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        
        return (
            float(macd_line.iloc[-1]),
            float(macd_signal.iloc[-1]),
            float(macd_histogram.iloc[-1])
        )
    except Exception as e:
        logger.error(f"MACD calculation error: {e}")
        return None, None, None

def calculate_stochastic(df: pd.DataFrame, k_period: int = 14) -> Tuple[float, float]:
    """Calculate Stochastic - Simple and working"""
    try:
        high = df['high']
        low = df['low']
        close = df['close']
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k_smooth = k_percent.rolling(window=3).mean()
        d_percent = k_smooth.rolling(window=3).mean()
        
        return float(k_smooth.iloc[-1]), float(d_percent.iloc[-1])
    except Exception as e:
        logger.error(f"Stochastic calculation error: {e}")
        return None, None

def calculate_bollinger_bands(prices: pd.Series, period: int = 20) -> Dict:
    """Calculate Bollinger Bands and position - Simple and working"""
    try:
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        
        current_price = prices.iloc[-1]
        bb_position = ((current_price - lower.iloc[-1]) / 
                      (upper.iloc[-1] - lower.iloc[-1])) if upper.iloc[-1] != lower.iloc[-1] else 0.5
        
        return {
            'upper': float(upper.iloc[-1]),
            'middle': float(sma.iloc[-1]),
            'lower': float(lower.iloc[-1]),
            'position': float(bb_position)
        }
    except Exception as e:
        logger.error(f"Bollinger Bands calculation error: {e}")
        return {'upper': None, 'middle': None, 'lower': None, 'position': None}

def calculate_vwap(df: pd.DataFrame) -> float:
    """Calculate VWAP - Simple and working"""
    try:
        if 'volume' not in df.columns:
            return float(df['close'].iloc[-1])
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).sum() / df['volume'].sum()
        return float(vwap)
    except Exception as e:
        logger.error(f"VWAP calculation error: {e}")
        return None

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ATR - Simple and working"""
    try:
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return float(atr.iloc[-1])
    except Exception as e:
        logger.error(f"ATR calculation error: {e}")
        return None

def calculate_volume_analysis(df: pd.DataFrame, period: int = 20) -> Dict:
    """Calculate volume analysis - Simple and working"""
    try:
        if 'volume' not in df.columns:
            return {'ratio': 1.0, 'trend': 'NEUTRAL', 'surge': False}
        
        volume_sma = df['volume'].rolling(period).mean()
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1.0
        
        volume_trend = "INCREASING" if volume_ratio > 1.2 else "DECREASING" if volume_ratio < 0.8 else "NEUTRAL"
        volume_surge = volume_ratio > 2.0
        
        return {
            'ratio': float(volume_ratio),
            'trend': volume_trend,
            'surge': volume_surge
        }
    except Exception as e:
        logger.error(f"Volume analysis error: {e}")
        return {'ratio': 1.0, 'trend': 'NEUTRAL', 'surge': False}

def calculate_all_simple_indicators(df: pd.DataFrame) -> Dict:
    """
    Calculate ALL indicators for IA1 - Simple, working approach
    Returns a clean dictionary with all technical indicators
    """
    try:
        if len(df) < 50:
            logger.warning("Insufficient data for indicator calculation")
            return {}
        
        logger.info(f"Calculating simple indicators for {len(df)} data points")
        
        close = df['close']
        
        # RSI
        rsi = calculate_rsi(close, 14)
        
        # MACD  
        macd_line, macd_signal, macd_histogram = calculate_macd(close)
        
        # Stochastic
        stoch_k, stoch_d = calculate_stochastic(df)
        
        # Bollinger Bands
        bb_data = calculate_bollinger_bands(close)
        
        # VWAP
        vwap = calculate_vwap(df)
        vwap_distance = ((close.iloc[-1] - vwap) / vwap * 100) if vwap and vwap > 0 else 0.0
        
        # ATR
        atr = calculate_atr(df)
        atr_pct = (atr / close.iloc[-1] * 100) if atr and close.iloc[-1] > 0 else 0.0
        
        # Volume
        volume_data = calculate_volume_analysis(df)
        
        # Simple EMAs
        ema_9 = close.ewm(span=9).mean().iloc[-1] if len(close) >= 9 else close.iloc[-1]
        ema_21 = close.ewm(span=21).mean().iloc[-1] if len(close) >= 21 else close.iloc[-1]
        ema_50 = close.ewm(span=50).mean().iloc[-1] if len(close) >= 50 else close.iloc[-1]
        
        # Create comprehensive results dictionary
        indicators = {
            # RSI
            'rsi': rsi,
            'rsi_overbought': rsi > 70 if rsi else False,
            'rsi_oversold': rsi < 30 if rsi else False,
            
            # MACD
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram,
            'macd_bullish': macd_histogram > 0 if macd_histogram else False,
            
            # Stochastic
            'stochastic_k': stoch_k,
            'stochastic_d': stoch_d,
            'stoch_overbought': stoch_k > 80 if stoch_k else False,
            'stoch_oversold': stoch_k < 20 if stoch_k else False,
            
            # Bollinger Bands
            'bb_upper': bb_data['upper'],
            'bb_middle': bb_data['middle'],
            'bb_lower': bb_data['lower'],
            'bb_position': bb_data['position'],
            
            # VWAP
            'vwap': vwap,
            'vwap_distance': vwap_distance,
            'above_vwap': close.iloc[-1] > vwap if vwap else False,
            
            # ATR
            'atr': atr,
            'atr_percentage': atr_pct,
            
            # Volume
            'volume_ratio': volume_data['ratio'],
            'volume_trend': volume_data['trend'],
            'volume_surge': volume_data['surge'],
            
            # EMAs
            'ema_9': float(ema_9),
            'ema_21': float(ema_21),
            'ema_50': float(ema_50),
            
            # Trend Analysis
            'trend_bullish': ema_9 > ema_21 > ema_50 if all([ema_9, ema_21, ema_50]) else False,
            'trend_bearish': ema_9 < ema_21 < ema_50 if all([ema_9, ema_21, ema_50]) else False,
            'price_above_emas': close.iloc[-1] > ema_21 if ema_21 else False,
        }
        
        # Log results
        logger.info(f"✅ Simple indicators calculated successfully:")
        logger.info(f"   RSI: {rsi:.2f if rsi else 'N/A'}")
        logger.info(f"   MACD H: {macd_histogram:.6f if macd_histogram else 'N/A'}")
        logger.info(f"   Stoch K: {stoch_k:.2f if stoch_k else 'N/A'}")
        logger.info(f"   BB Pos: {bb_data['position']:.2f if bb_data['position'] else 'N/A'}")
        logger.info(f"   VWAP Dist: {vwap_distance:.2f if vwap_distance else 'N/A'}%")
        
        return indicators
        
    except Exception as e:
        logger.error(f"Error calculating simple indicators: {e}")
        import traceback
        traceback.print_exc()
        return {}

# Global instance
simple_indicators = None

def get_simple_indicators():
    """Get global simple indicators instance"""
    global simple_indicators
    if simple_indicators is None:
        simple_indicators = True  # Just a flag, functions are standalone
    return simple_indicators
#!/usr/bin/env python3
"""
Fibonacci Retracement Calculator for Crypto Trading
Provides comprehensive Fibonacci analysis integrated into IA1 decision making
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FibonacciLevels:
    """Fibonacci retracement levels and analysis"""
    high_price: float
    low_price: float
    current_price: float
    trend_direction: str  # "bullish", "bearish", "neutral"
    
    # Standard Fibonacci levels
    level_0: float      # 0% (high for bearish, low for bullish)
    level_236: float    # 23.6%
    level_382: float    # 38.2%
    level_500: float    # 50%
    level_618: float    # 61.8%
    level_786: float    # 78.6%
    level_1000: float   # 100% (low for bearish, high for bullish)
    
    # Extended levels for breakouts
    level_1272: float   # 127.2%
    level_1618: float   # 161.8%
    
    # Current position analysis
    current_level_percentage: float
    nearest_level: str
    nearest_level_price: float
    distance_to_nearest: float
    
    # Support/Resistance analysis
    support_levels: List[float]
    resistance_levels: List[float]
    
    # Trading signals
    signal_strength: float  # 0-1 strength of Fibonacci signal
    signal_direction: str   # "bullish", "bearish", "neutral"
    key_level_proximity: bool  # Is price near key Fibonacci level?

class FibonacciCalculator:
    """
    Advanced Fibonacci retracement calculator for crypto trading
    """
    
    def __init__(self):
        self.fib_ratios = {
            "0%": 0.0,
            "23.6%": 0.236,
            "38.2%": 0.382,
            "50%": 0.5,
            "61.8%": 0.618,
            "78.6%": 0.786,
            "100%": 1.0,
            "127.2%": 1.272,
            "161.8%": 1.618
        }
        
        self.key_levels = ["23.6%", "38.2%", "50%", "61.8%", "78.6%"]
        
    def calculate_fibonacci_levels(self, df: pd.DataFrame, lookback_period: int = 20) -> FibonacciLevels:
        """
        Calculate comprehensive Fibonacci retracement levels
        
        Args:
            df: OHLCV DataFrame with High, Low, Close columns
            lookback_period: Period to determine high/low for Fibonacci calculation
            
        Returns:
            FibonacciLevels object with complete analysis
        """
        try:
            if len(df) < lookback_period:
                logger.warning(f"Insufficient data for Fibonacci calculation: {len(df)} < {lookback_period}")
                close_col = 'Close' if 'Close' in df.columns else 'close'
                return self._create_fallback_fibonacci(df[close_col].iloc[-1] if len(df) > 0 else 100.0)
            
            # Get recent high and low for Fibonacci calculation
            recent_data = df.tail(lookback_period)
            
            # 🔧 NORMALIZE COLUMN NAMES (handle both uppercase and lowercase)
            high_col = 'High' if 'High' in df.columns else 'high'
            low_col = 'Low' if 'Low' in df.columns else 'low'
            close_col = 'Close' if 'Close' in df.columns else 'close'
            
            high_price = recent_data[high_col].max()
            low_price = recent_data[low_col].min()
            current_price = df[close_col].iloc[-1]
            
            # Determine trend direction
            trend_direction = self._determine_trend_direction(df, lookback_period)
            
            # Calculate Fibonacci levels
            price_range = high_price - low_price
            
            if trend_direction == "bullish":
                # In bullish trend, levels are calculated from low to high
                base_price = low_price
                level_0 = low_price
                level_1000 = high_price
            else:
                # In bearish trend, levels are calculated from high to low
                base_price = high_price
                level_0 = high_price
                level_1000 = low_price
                price_range = -price_range  # Negative for bearish calculation
            
            # Calculate standard Fibonacci levels
            level_236 = base_price + (price_range * 0.236)
            level_382 = base_price + (price_range * 0.382)
            level_500 = base_price + (price_range * 0.5)
            level_618 = base_price + (price_range * 0.618)
            level_786 = base_price + (price_range * 0.786)
            
            # Calculate extended levels
            level_1272 = base_price + (price_range * 1.272)
            level_1618 = base_price + (price_range * 1.618)
            
            # Analyze current position
            current_position_analysis = self._analyze_current_position(
                current_price, high_price, low_price, trend_direction,
                level_236, level_382, level_500, level_618, level_786
            )
            
            # Determine support and resistance levels
            support_levels, resistance_levels = self._determine_support_resistance(
                current_price, [level_0, level_236, level_382, level_500, level_618, level_786, level_1000]
            )
            
            # Calculate trading signals
            signal_analysis = self._calculate_fibonacci_signals(
                current_price, current_position_analysis, trend_direction
            )
            
            return FibonacciLevels(
                high_price=high_price,
                low_price=low_price,
                current_price=current_price,
                trend_direction=trend_direction,
                level_0=level_0,
                level_236=level_236,
                level_382=level_382,
                level_500=level_500,
                level_618=level_618,
                level_786=level_786,
                level_1000=level_1000,
                level_1272=level_1272,
                level_1618=level_1618,
                current_level_percentage=current_position_analysis["percentage"],
                nearest_level=current_position_analysis["nearest_level"],
                nearest_level_price=current_position_analysis["nearest_price"],
                distance_to_nearest=current_position_analysis["distance"],
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                signal_strength=signal_analysis["strength"],
                signal_direction=signal_analysis["direction"],
                key_level_proximity=signal_analysis["key_level_proximity"]
            )
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {e}")
            return self._create_fallback_fibonacci(df['Close'].iloc[-1] if len(df) > 0 else 100.0)
    
    def _determine_trend_direction(self, df: pd.DataFrame, lookback_period: int) -> str:
        """Determine overall trend direction for Fibonacci calculation"""
        try:
            recent_data = df.tail(lookback_period)
            
            # 🔧 NORMALIZE COLUMN NAMES (handle both uppercase and lowercase)
            close_col = 'Close' if 'Close' in df.columns else 'close'
            
            # Use simple trend analysis
            first_close = recent_data[close_col].iloc[0]
            last_close = recent_data[close_col].iloc[-1]
            
            # Also consider EMA for trend confirmation
            ema_short = recent_data[close_col].ewm(span=5).mean().iloc[-1]
            ema_long = recent_data[close_col].ewm(span=10).mean().iloc[-1]
            
            price_trend = "bullish" if last_close > first_close else "bearish"
            ema_trend = "bullish" if ema_short > ema_long else "bearish"
            
            # Combine signals
            if price_trend == ema_trend:
                return price_trend
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error determining trend direction: {e}")
            return "neutral"
    
    def _analyze_current_position(self, current_price: float, high_price: float, low_price: float, 
                                 trend_direction: str, level_236: float, level_382: float, 
                                 level_500: float, level_618: float, level_786: float) -> Dict:
        """Analyze where current price sits relative to Fibonacci levels"""
        
        price_range = high_price - low_price
        if price_range == 0:
            return {
                "percentage": 50.0,
                "nearest_level": "50.0%",
                "nearest_price": current_price,
                "distance": 0.0
            }
        
        # Calculate current position as percentage
        if trend_direction == "bullish":
            current_percentage = ((current_price - low_price) / price_range) * 100
        else:
            current_percentage = ((high_price - current_price) / price_range) * 100
        
        # Find nearest Fibonacci level
        levels = {
            "0.0%": low_price if trend_direction == "bullish" else high_price,
            "23.6%": level_236,
            "38.2%": level_382,
            "50.0%": level_500,
            "61.8%": level_618,
            "78.6%": level_786,
            "100.0%": high_price if trend_direction == "bullish" else low_price
        }
        
        nearest_level = "50.0%"
        nearest_price = level_500
        min_distance = abs(current_price - level_500)
        
        for level_name, level_price in levels.items():
            distance = abs(current_price - level_price)
            if distance < min_distance:
                min_distance = distance
                nearest_level = level_name
                nearest_price = level_price
        
        return {
            "percentage": current_percentage,
            "nearest_level": nearest_level,
            "nearest_price": nearest_price,
            "distance": min_distance
        }
    
    def _determine_support_resistance(self, current_price: float, fib_levels: List[float]) -> Tuple[List[float], List[float]]:
        """Determine which Fibonacci levels act as support and resistance"""
        
        support_levels = [level for level in fib_levels if level < current_price]
        resistance_levels = [level for level in fib_levels if level > current_price]
        
        # Sort and limit to most relevant levels
        support_levels = sorted(support_levels, reverse=True)[:3]  # Top 3 support levels
        resistance_levels = sorted(resistance_levels)[:3]  # Top 3 resistance levels
        
        return support_levels, resistance_levels
    
    def _calculate_fibonacci_signals(self, current_price: float, position_analysis: Dict, trend_direction: str) -> Dict:
        """Calculate trading signals based on Fibonacci analysis"""
        
        nearest_level = position_analysis["nearest_level"]
        distance_percentage = (position_analysis["distance"] / current_price) * 100
        
        # Check if near key Fibonacci level (within 2%)
        key_level_proximity = distance_percentage < 2.0 and nearest_level in ["23.6%", "38.2%", "50.0%", "61.8%", "78.6%"]
        
        # Calculate signal strength (0-1)
        if key_level_proximity:
            # Strong signal if near key level
            signal_strength = 0.8 if nearest_level in ["38.2%", "61.8%"] else 0.6
        else:
            # Weaker signal if not near key level
            signal_strength = 0.3
        
        # Determine signal direction
        current_percentage = position_analysis["percentage"]
        
        if trend_direction == "bullish":
            if current_percentage < 38.2:
                signal_direction = "bullish"  # Near strong support, expect bounce
            elif current_percentage > 61.8:
                signal_direction = "bearish"  # Near resistance, expect pullback
            else:
                signal_direction = "neutral"
        elif trend_direction == "bearish":
            if current_percentage < 38.2:
                signal_direction = "bearish"  # Continuing downtrend
            elif current_percentage > 61.8:
                signal_direction = "bullish"  # Near strong support, expect bounce
            else:
                signal_direction = "neutral"
        else:
            signal_direction = "neutral"
        
        return {
            "strength": signal_strength,
            "direction": signal_direction,
            "key_level_proximity": key_level_proximity
        }
    
    def _create_fallback_fibonacci(self, current_price: float) -> FibonacciLevels:
        """Create fallback Fibonacci levels when calculation fails"""
        return FibonacciLevels(
            high_price=current_price * 1.05,
            low_price=current_price * 0.95,
            current_price=current_price,
            trend_direction="neutral",
            level_0=current_price * 0.95,
            level_236=current_price * 0.9736,
            level_382=current_price * 0.9882,
            level_500=current_price,
            level_618=current_price * 1.0118,
            level_786=current_price * 1.0264,
            level_1000=current_price * 1.05,
            level_1272=current_price * 1.0772,
            level_1618=current_price * 1.1118,
            current_level_percentage=50.0,
            nearest_level="50.0%",
            nearest_level_price=current_price,
            distance_to_nearest=0.0,
            support_levels=[current_price * 0.95],
            resistance_levels=[current_price * 1.05],
            signal_strength=0.0,
            signal_direction="neutral",
            key_level_proximity=False
        )
    
    def get_fibonacci_for_prompt(self, fibonacci_levels: FibonacciLevels) -> str:
        """Format Fibonacci analysis for IA1 prompt"""
        
        prompt_text = f"""
🔢 FIBONACCI RETRACEMENT ANALYSIS:
📊 Trend Direction: {fibonacci_levels.trend_direction.upper()}
📈 Price Range: ${fibonacci_levels.low_price:.6f} - ${fibonacci_levels.high_price:.6f}
📍 Current Position: {fibonacci_levels.current_level_percentage:.1f}% (nearest: {fibonacci_levels.nearest_level})

🎯 KEY FIBONACCI LEVELS:
• 23.6%: ${fibonacci_levels.level_236:.6f}
• 38.2%: ${fibonacci_levels.level_382:.6f} 
• 50.0%: ${fibonacci_levels.level_500:.6f}
• 61.8%: ${fibonacci_levels.level_618:.6f}
• 78.6%: ${fibonacci_levels.level_786:.6f}

🛡️ SUPPORT LEVELS: {[f"${level:.6f}" for level in fibonacci_levels.support_levels]}
🚧 RESISTANCE LEVELS: {[f"${level:.6f}" for level in fibonacci_levels.resistance_levels]}

🎯 FIBONACCI SIGNAL: {fibonacci_levels.signal_direction.upper()} (Strength: {fibonacci_levels.signal_strength:.1%})
⚡ Key Level Proximity: {'YES' if fibonacci_levels.key_level_proximity else 'NO'}
"""
        
        return prompt_text.strip()

# Global instance
fibonacci_calculator = FibonacciCalculator()

def calculate_fibonacci_retracements(df: pd.DataFrame, lookback_period: int = 20) -> FibonacciLevels:
    """
    Convenience function to calculate Fibonacci retracements
    
    Args:
        df: OHLCV DataFrame
        lookback_period: Period for high/low calculation
        
    Returns:
        FibonacciLevels object
    """
    return fibonacci_calculator.calculate_fibonacci_levels(df, lookback_period)
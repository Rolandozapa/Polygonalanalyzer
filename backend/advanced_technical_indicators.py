"""
ADVANCED TECHNICAL INDICATORS SYSTEM
Calcul avancé des indicateurs techniques pour l'entraînement IA
RSI, MACD, Stochastic, Bollinger Bands + indicateurs composites
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class TechnicalIndicators:
    """Structure complète des indicateurs techniques"""
    # RSI
    rsi_14: float = 0.0
    rsi_9: float = 0.0  # RSI plus réactif
    rsi_21: float = 0.0  # RSI plus lisse
    rsi_divergence: bool = False
    rsi_overbought: bool = False  # RSI > 70
    rsi_oversold: bool = False    # RSI < 30
    
    # MACD
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    macd_bullish_crossover: bool = False
    macd_bearish_crossover: bool = False
    macd_above_zero: bool = False
    
    # Stochastic
    stoch_k: float = 0.0
    stoch_d: float = 0.0
    stoch_overbought: bool = False  # %K > 80
    stoch_oversold: bool = False    # %K < 20
    stoch_bullish_crossover: bool = False  # %K crosses above %D
    stoch_bearish_crossover: bool = False  # %K crosses below %D
    
    # Bollinger Bands
    bb_upper: float = 0.0
    bb_middle: float = 0.0  # SMA 20
    bb_lower: float = 0.0
    bb_width: float = 0.0   # (Upper - Lower) / Middle
    bb_position: float = 0.0  # (Price - Lower) / (Upper - Lower)
    bb_squeeze: bool = False  # BB width < historical average
    bb_expansion: bool = False  # BB width > historical average
    
    # Moving Averages
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    
    # Volume Indicators
    volume_sma: float = 0.0
    volume_ratio: float = 0.0  # Current volume / Average volume
    obv: float = 0.0  # On Balance Volume
    
    # VWAP & Deviation Bands
    vwap: float = 0.0
    vwap_upper_1std: float = 0.0  # VWAP + 1 standard deviation
    vwap_lower_1std: float = 0.0  # VWAP - 1 standard deviation
    vwap_upper_2std: float = 0.0  # VWAP + 2 standard deviations
    vwap_lower_2std: float = 0.0  # VWAP - 2 standard deviations
    vwap_position: float = 0.0    # (Price - VWAP) / VWAP * 100 (percentage above/below VWAP)
    vwap_overbought: bool = False  # Price > VWAP + 1 std dev
    vwap_oversold: bool = False    # Price < VWAP - 1 std dev
    vwap_extreme_overbought: bool = False  # Price > VWAP + 2 std dev
    vwap_extreme_oversold: bool = False    # Price < VWAP - 2 std dev
    vwap_trend: str = "neutral"    # "bullish", "bearish", "neutral" based on price vs VWAP
    
    # Composite Indicators
    trend_strength: float = 0.0  # 0-1 score based on multiple indicators
    momentum_score: float = 0.0  # 0-1 score for momentum
    volatility_score: float = 0.0  # 0-1 score for volatility
    signal_confidence: float = 0.0  # Overall confidence 0-1

@dataclass
class IndicatorSignal:
    """Signal généré par les indicateurs techniques"""
    signal_type: str  # "bullish", "bearish", "neutral"
    strength: float   # 0-1
    confidence: float # 0-1
    timeframe: str    # "short", "medium", "long"
    supporting_indicators: List[str]
    conflicting_indicators: List[str]
    reasoning: str

class AdvancedTechnicalIndicators:
    """Calculateur avancé d'indicateurs techniques"""
    
    def __init__(self):
        self.lookback_periods = {
            'short': 14,
            'medium': 50,
            'long': 200
        }
        logger.info("Advanced Technical Indicators system initialized")
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule tous les indicateurs techniques sur un DataFrame OHLCV"""
        try:
            df = df.copy()
            
            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns: {missing_cols}")
                return df
            
            # RSI (Multiple periods)
            df = self._calculate_rsi(df)
            
            # MACD
            df = self._calculate_macd(df)
            
            # Stochastic
            df = self._calculate_stochastic(df)
            
            # Bollinger Bands
            df = self._calculate_bollinger_bands(df)
            
            # Moving Averages
            df = self._calculate_moving_averages(df)
            
            # Volume Indicators
            df = self._calculate_volume_indicators(df)
            
            # Composite Indicators
            df = self._calculate_composite_indicators(df)
            
            # Signal Detection
            df = self._detect_technical_signals(df)
            
            logger.info(f"Calculated technical indicators for {len(df)} periods")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule RSI avec plusieurs périodes"""
        for period in [9, 14, 21]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # RSI signals
        df['rsi_overbought'] = df['rsi_14'] > 70
        df['rsi_oversold'] = df['rsi_14'] < 30
        
        # RSI Divergence (simplified)
        df['rsi_divergence'] = self._detect_rsi_divergence(df)
        
        return df
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule MACD complet"""
        # MACD Line
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd_line'] = ema_12 - ema_26
        
        # Signal Line
        df['macd_signal'] = df['macd_line'].ewm(span=9).mean()
        
        # Histogram
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        
        # MACD Signals
        df['macd_bullish_crossover'] = (
            (df['macd_line'] > df['macd_signal']) & 
            (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))
        )
        
        df['macd_bearish_crossover'] = (
            (df['macd_line'] < df['macd_signal']) & 
            (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))
        )
        
        df['macd_above_zero'] = df['macd_line'] > 0
        
        return df
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Calcule Stochastic Oscillator"""
        # %K calculation
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        df['stoch_k'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
        
        # %D calculation (SMA of %K)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        # Stochastic signals
        df['stoch_overbought'] = df['stoch_k'] > 80
        df['stoch_oversold'] = df['stoch_k'] < 20
        
        # Crossover signals
        df['stoch_bullish_crossover'] = (
            (df['stoch_k'] > df['stoch_d']) & 
            (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
        )
        
        df['stoch_bearish_crossover'] = (
            (df['stoch_k'] < df['stoch_d']) & 
            (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
        )
        
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Calcule Bollinger Bands complets"""
        # Middle Band (SMA)
        df['bb_middle'] = df['Close'].rolling(window=period).mean()
        
        # Standard deviation
        bb_std = df['Close'].rolling(window=period).std()
        
        # Upper and Lower Bands
        df['bb_upper'] = df['bb_middle'] + (bb_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (bb_std * std_dev)
        
        # Bollinger Band Width
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Price position within bands
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Bollinger Band Squeeze/Expansion
        bb_width_sma = df['bb_width'].rolling(window=20).mean()
        df['bb_squeeze'] = df['bb_width'] < bb_width_sma * 0.8
        df['bb_expansion'] = df['bb_width'] > bb_width_sma * 1.2
        
        return df
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule moyennes mobiles"""
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['sma_200'] = df['Close'].rolling(window=200).mean()
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule indicateurs de volume"""
        # Volume SMA
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # On Balance Volume (OBV)
        df['obv'] = 0.0
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1] - df['Volume'].iloc[i]
            else:
                df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1]
        
        return df
    
    def _calculate_composite_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule indicateurs composites"""
        # Trend Strength (0-1)
        trend_signals = []
        
        # Price vs MA trend signals
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            trend_signals.append((df['Close'] > df['sma_20']).astype(int))
            trend_signals.append((df['sma_20'] > df['sma_50']).astype(int))
            if 'sma_200' in df.columns:
                trend_signals.append((df['sma_50'] > df['sma_200']).astype(int))
        
        # MACD trend signal
        if 'macd_above_zero' in df.columns:
            trend_signals.append(df['macd_above_zero'].astype(int))
        
        if trend_signals:
            df['trend_strength'] = np.mean(trend_signals, axis=0)
        else:
            df['trend_strength'] = 0.5
        
        # Momentum Score (0-1)
        momentum_signals = []
        
        if 'rsi_14' in df.columns:
            # RSI momentum (50 = neutral, >50 = bullish momentum)
            momentum_signals.append((df['rsi_14'] - 50) / 50)
        
        if 'macd_histogram' in df.columns:
            # MACD histogram momentum
            macd_norm = df['macd_histogram'] / df['macd_histogram'].rolling(20).std()
            momentum_signals.append(np.tanh(macd_norm))  # Normalize to -1,1
        
        if 'stoch_k' in df.columns:
            # Stochastic momentum
            momentum_signals.append((df['stoch_k'] - 50) / 50)
        
        if momentum_signals:
            df['momentum_score'] = np.mean(momentum_signals, axis=0)
            df['momentum_score'] = (df['momentum_score'] + 1) / 2  # Convert to 0-1
        else:
            df['momentum_score'] = 0.5
        
        # Volatility Score (0-1)
        if 'bb_width' in df.columns:
            bb_width_percentile = df['bb_width'].rolling(50).rank(pct=True)
            df['volatility_score'] = bb_width_percentile
        else:
            # Fallback: use price volatility
            price_volatility = df['Close'].pct_change().rolling(20).std()
            vol_percentile = price_volatility.rolling(50).rank(pct=True)
            df['volatility_score'] = vol_percentile
        
        return df
    
    def _detect_rsi_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Détecte les divergences RSI (version simplifiée)"""
        divergence = pd.Series(False, index=df.index)
        
        if len(df) < 20:
            return divergence
        
        # Look for divergence in last 10 periods
        for i in range(10, len(df)):
            # Bullish divergence: price makes lower low, RSI makes higher low
            price_window = df['Close'].iloc[i-10:i+1]
            rsi_window = df['rsi_14'].iloc[i-10:i+1]
            
            if len(price_window) > 5 and len(rsi_window) > 5:
                price_min_idx = price_window.idxmin()
                rsi_min_idx = rsi_window.idxmin()
                
                # Simple divergence check
                if (price_window.iloc[-1] < price_window.iloc[0] and 
                    rsi_window.iloc[-1] > rsi_window.iloc[0]):
                    divergence.iloc[i] = True
        
        return divergence
    
    def _detect_technical_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Détecte les signaux techniques composites"""
        df['signal_confidence'] = 0.5  # Default neutral
        
        # Calculate signal confidence based on indicator alignment
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        
        # RSI signals
        if 'rsi_14' in df.columns:
            total_signals += 1
            rsi_signal = (df['rsi_14'] - 50) / 50  # -1 to 1
            bullish_signals += (rsi_signal > 0).astype(int)
            bearish_signals += (rsi_signal < 0).astype(int)
        
        # MACD signals
        if 'macd_histogram' in df.columns:
            total_signals += 1
            bullish_signals += (df['macd_histogram'] > 0).astype(int)
            bearish_signals += (df['macd_histogram'] < 0).astype(int)
        
        # Stochastic signals
        if 'stoch_k' in df.columns:
            total_signals += 1
            stoch_signal = (df['stoch_k'] - 50) / 50
            bullish_signals += (stoch_signal > 0).astype(int)
            bearish_signals += (stoch_signal < 0).astype(int)
        
        # Bollinger Bands signals
        if 'bb_position' in df.columns:
            total_signals += 1
            bb_signal = (df['bb_position'] - 0.5) * 2  # -1 to 1
            bullish_signals += (bb_signal > 0).astype(int)
            bearish_signals += (bb_signal < 0).astype(int)
        
        if total_signals > 0:
            # Calculate confidence based on signal alignment
            signal_alignment = abs(bullish_signals - bearish_signals) / total_signals
            df['signal_confidence'] = 0.5 + (signal_alignment * 0.5)
        
        return df
    
    def get_current_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Extrait les indicateurs actuels (dernière ligne)"""
        if len(df) == 0:
            return TechnicalIndicators()
        
        last_row = df.iloc[-1]
        
        return TechnicalIndicators(
            # RSI
            rsi_14=last_row.get('rsi_14', 50.0),
            rsi_9=last_row.get('rsi_9', 50.0),
            rsi_21=last_row.get('rsi_21', 50.0),
            rsi_divergence=last_row.get('rsi_divergence', False),
            rsi_overbought=last_row.get('rsi_overbought', False),
            rsi_oversold=last_row.get('rsi_oversold', False),
            
            # MACD
            macd_line=last_row.get('macd_line', 0.0),
            macd_signal=last_row.get('macd_signal', 0.0),
            macd_histogram=last_row.get('macd_histogram', 0.0),
            macd_bullish_crossover=last_row.get('macd_bullish_crossover', False),
            macd_bearish_crossover=last_row.get('macd_bearish_crossover', False),
            macd_above_zero=last_row.get('macd_above_zero', False),
            
            # Stochastic
            stoch_k=last_row.get('stoch_k', 50.0),
            stoch_d=last_row.get('stoch_d', 50.0),
            stoch_overbought=last_row.get('stoch_overbought', False),
            stoch_oversold=last_row.get('stoch_oversold', False),
            stoch_bullish_crossover=last_row.get('stoch_bullish_crossover', False),
            stoch_bearish_crossover=last_row.get('stoch_bearish_crossover', False),
            
            # Bollinger Bands
            bb_upper=last_row.get('bb_upper', 0.0),
            bb_middle=last_row.get('bb_middle', 0.0),
            bb_lower=last_row.get('bb_lower', 0.0),
            bb_width=last_row.get('bb_width', 0.0),
            bb_position=last_row.get('bb_position', 0.5),
            bb_squeeze=last_row.get('bb_squeeze', False),
            bb_expansion=last_row.get('bb_expansion', False),
            
            # Moving Averages
            sma_20=last_row.get('sma_20', 0.0),
            sma_50=last_row.get('sma_50', 0.0),
            sma_200=last_row.get('sma_200', 0.0),
            ema_12=last_row.get('ema_12', 0.0),
            ema_26=last_row.get('ema_26', 0.0),
            
            # Volume
            volume_sma=last_row.get('volume_sma', 0.0),
            volume_ratio=last_row.get('volume_ratio', 1.0),
            obv=last_row.get('obv', 0.0),
            
            # Composite
            trend_strength=last_row.get('trend_strength', 0.5),
            momentum_score=last_row.get('momentum_score', 0.5),
            volatility_score=last_row.get('volatility_score', 0.5),
            signal_confidence=last_row.get('signal_confidence', 0.5)
        )
    
    def generate_trading_signal(self, indicators: TechnicalIndicators) -> IndicatorSignal:
        """Génère un signal de trading basé sur les indicateurs"""
        supporting_indicators = []
        conflicting_indicators = []
        bullish_score = 0
        bearish_score = 0
        confidence_factors = []
        
        # RSI Analysis
        if indicators.rsi_oversold:
            bullish_score += 2
            supporting_indicators.append("RSI Oversold")
            confidence_factors.append(0.8)
        elif indicators.rsi_overbought:
            bearish_score += 2
            supporting_indicators.append("RSI Overbought")
            confidence_factors.append(0.8)
        elif indicators.rsi_14 > 55:
            bullish_score += 1
            supporting_indicators.append("RSI Bullish")
            confidence_factors.append(0.6)
        elif indicators.rsi_14 < 45:
            bearish_score += 1
            supporting_indicators.append("RSI Bearish")
            confidence_factors.append(0.6)
        
        # MACD Analysis
        if indicators.macd_bullish_crossover:
            bullish_score += 2
            supporting_indicators.append("MACD Bullish Crossover")
            confidence_factors.append(0.8)
        elif indicators.macd_bearish_crossover:
            bearish_score += 2
            supporting_indicators.append("MACD Bearish Crossover")
            confidence_factors.append(0.8)
        elif indicators.macd_histogram > 0:
            bullish_score += 1
            supporting_indicators.append("MACD Positive")
            confidence_factors.append(0.6)
        elif indicators.macd_histogram < 0:
            bearish_score += 1
            supporting_indicators.append("MACD Negative")
            confidence_factors.append(0.6)
        
        # Stochastic Analysis
        if indicators.stoch_bullish_crossover and indicators.stoch_oversold:
            bullish_score += 2
            supporting_indicators.append("Stoch Bullish from Oversold")
            confidence_factors.append(0.9)
        elif indicators.stoch_bearish_crossover and indicators.stoch_overbought:
            bearish_score += 2
            supporting_indicators.append("Stoch Bearish from Overbought")
            confidence_factors.append(0.9)
        
        # Bollinger Bands Analysis
        if indicators.bb_position < 0.1:
            bullish_score += 1
            supporting_indicators.append("BB Oversold")
            confidence_factors.append(0.7)
        elif indicators.bb_position > 0.9:
            bearish_score += 1
            supporting_indicators.append("BB Overbought")
            confidence_factors.append(0.7)
        
        # Trend and Momentum
        if indicators.trend_strength > 0.7:
            bullish_score += 1
            supporting_indicators.append("Strong Uptrend")
            confidence_factors.append(0.8)
        elif indicators.trend_strength < 0.3:
            bearish_score += 1
            supporting_indicators.append("Strong Downtrend")
            confidence_factors.append(0.8)
        
        # Determine signal
        total_score = bullish_score + bearish_score
        if total_score == 0:
            signal_type = "neutral"
            strength = 0.5
        elif bullish_score > bearish_score:
            signal_type = "bullish"
            strength = min(bullish_score / (total_score + 2), 1.0)
        else:
            signal_type = "bearish"
            strength = min(bearish_score / (total_score + 2), 1.0)
        
        # Calculate confidence
        confidence = np.mean(confidence_factors) if confidence_factors else 0.5
        confidence *= indicators.signal_confidence  # Adjust by composite confidence
        
        # Determine timeframe
        if indicators.volatility_score > 0.7:
            timeframe = "short"
        elif indicators.volatility_score < 0.3:
            timeframe = "long"
        else:
            timeframe = "medium"
        
        # Generate reasoning
        reasoning = f"Signal {signal_type} avec {len(supporting_indicators)} indicateurs confirmant. "
        reasoning += f"Force: {strength:.1%}, Confiance: {confidence:.1%}. "
        reasoning += f"Indicateurs: {', '.join(supporting_indicators[:3])}"
        
        return IndicatorSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            timeframe=timeframe,
            supporting_indicators=supporting_indicators,
            conflicting_indicators=conflicting_indicators,
            reasoning=reasoning
        )

# Instance globale
advanced_technical_indicators = AdvancedTechnicalIndicators()
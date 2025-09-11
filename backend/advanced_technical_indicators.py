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
    
    # Money Flow Index (MFI) - The Volume-Weighted RSI Beast 🔥
    mfi: float = 50.0             # Money Flow Index (0-100)
    mfi_overbought: bool = False  # MFI > 80 (distribution phase)
    mfi_oversold: bool = False    # MFI < 20 (accumulation phase)
    mfi_extreme_overbought: bool = False  # MFI > 90 (extreme distribution)
    mfi_extreme_oversold: bool = False    # MFI < 10 (extreme accumulation)
    mfi_trend: str = "neutral"    # "bullish", "bearish", "neutral"
    mfi_divergence: bool = False  # Price vs MFI divergence (powerful reversal signal)
    money_flow_ratio: float = 1.0 # Positive/Negative money flow ratio
    institutional_activity: str = "neutral"  # "accumulation", "distribution", "neutral"
    
    # Multi EMA/SMA Trend Hierarchy - THE MISSING PIECE 🎯
    ema_9: float = 0.0            # Fast trend (9-period EMA)
    ema_21: float = 0.0           # Medium trend (21-period EMA)
    sma_50: float = 0.0           # Slow trend (50-period SMA)
    ema_200: float = 0.0          # Long-term trend (200-period EMA)
    trend_hierarchy: str = "neutral"  # "strong_bull", "weak_bull", "neutral", "weak_bear", "strong_bear"
    trend_momentum: str = "neutral"   # "accelerating", "steady", "decelerating"
    price_vs_emas: str = "mixed"      # "above_all", "above_fast", "below_fast", "below_all"
    ema_cross_signal: str = "neutral" # "golden_cross", "death_cross", "neutral"
    trend_strength_score: float = 0.5 # 0-1 score based on EMA hierarchy alignment
    
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
    """Calculateur avancé d'indicateurs techniques avec support MULTI-TIMEFRAME 🚀"""
    
    def __init__(self):
        self.lookback_periods = {
            'short': 14,
            'medium': 50,
            'long': 200
        }
        
        # 🔥 TIMEFRAMES POUR ANALYSE PROFESSIONNELLE 🔥
        self.timeframes = {
            'now': 1,           # Valeurs actuelles
            '5min': 5,          # 5 minutes ago  
            '1h': 60,           # 1 heure ago (60 min)
            '4h': 240,          # 4 heures ago (240 min)
            '1d': 1440,         # 1 jour ago (1440 min)
            '5d': 7200,         # 5 jours ago (7200 min)
            '14d': 20160        # 14 jours ago (20160 min)
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
            
            # Multi EMA/SMA Trend Hierarchy - THE CONFLUENCE BEAST FINAL PIECE! 🚀
            df = self._calculate_multi_ema_sma(df)
            
            # Volume Indicators
            df = self._calculate_volume_indicators(df)
            
            # VWAP & Deviation Bands
            df = self._calculate_vwap_bands(df)
            
            # Money Flow Index (MFI) - Volume-weighted RSI for institutional detection
            df = self._calculate_mfi(df)
            
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
    
    def _calculate_multi_ema_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🚀 CALCULE MULTI EMA/SMA TREND HIERARCHY - THE CONFLUENCE BEAST FINAL PIECE! 🚀
        Professional-grade trend analysis with multiple EMAs and SMAs for ULTIMATE precision
        """
        try:
            # === FAST TREND EMAs (Ultra-reactive for momentum detection) ===
            df['ema_9'] = df['Close'].ewm(span=9).mean()       # Lightning-fast trend changes
            df['ema_21'] = df['Close'].ewm(span=21).mean()     # Medium-term momentum
            
            # === SLOW TREND & STRUCTURAL LEVELS ===
            df['sma_50'] = df['Close'].rolling(window=50).mean()     # Institutional level
            df['ema_200'] = df['Close'].ewm(span=200).mean()         # Major trend determinant
            
            # === TREND HIERARCHY ANALYSIS ===
            # Perfect hierarchy: Price > EMA9 > EMA21 > SMA50 > EMA200 (STRONG BULL)
            # Inverse: Price < EMA9 < EMA21 < SMA50 < EMA200 (STRONG BEAR)
            
            df['trend_hierarchy'] = 'neutral'
            df['trend_momentum'] = 'neutral'
            df['price_vs_emas'] = 'mixed'
            df['ema_cross_signal'] = 'neutral'
            df['trend_strength_score'] = 0.5
            
            for i in range(len(df)):
                # Need enough data for EMA200 - use dynamic check
                if i < min(200, len(df) * 0.8):  # Use 80% of data or 200, whichever is smaller
                    continue
                    
                price = df['Close'].iloc[i]
                ema9 = df['ema_9'].iloc[i]
                ema21 = df['ema_21'].iloc[i]
                sma50 = df['sma_50'].iloc[i]
                ema200 = df['ema_200'].iloc[i]
                
                # Skip if any EMA is NaN
                if pd.isna(ema9) or pd.isna(ema21) or pd.isna(sma50) or pd.isna(ema200):
                    continue
                
                # === PRICE VS EMAs CLASSIFICATION ===
                emas_above_price = sum([
                    price > ema9,
                    price > ema21, 
                    price > sma50,
                    price > ema200
                ])
                
                if emas_above_price == 4:
                    df.loc[df.index[i], 'price_vs_emas'] = 'above_all'
                elif emas_above_price >= 2:
                    df.loc[df.index[i], 'price_vs_emas'] = 'above_fast'
                elif emas_above_price == 1:
                    df.loc[df.index[i], 'price_vs_emas'] = 'below_fast'
                else:
                    df.loc[df.index[i], 'price_vs_emas'] = 'below_all'
                
                # === TREND HIERARCHY STRENGTH ===
                # Perfect bull alignment: EMA9 > EMA21 > SMA50 > EMA200
                bull_alignment_score = 0
                if ema9 > ema21: bull_alignment_score += 0.25
                if ema21 > sma50: bull_alignment_score += 0.25  
                if sma50 > ema200: bull_alignment_score += 0.25
                if price > ema9: bull_alignment_score += 0.25
                
                df.loc[df.index[i], 'trend_strength_score'] = bull_alignment_score
                
                # === TREND HIERARCHY CLASSIFICATION ===
                if bull_alignment_score >= 0.8:
                    df.loc[df.index[i], 'trend_hierarchy'] = 'strong_bull'
                elif bull_alignment_score >= 0.6:
                    df.loc[df.index[i], 'trend_hierarchy'] = 'weak_bull'
                elif bull_alignment_score <= 0.2:
                    df.loc[df.index[i], 'trend_hierarchy'] = 'strong_bear'
                elif bull_alignment_score <= 0.4:
                    df.loc[df.index[i], 'trend_hierarchy'] = 'weak_bear'
                else:
                    df.loc[df.index[i], 'trend_hierarchy'] = 'neutral'
            
            # === MOMENTUM ANALYSIS (Rate of Change in EMAs) ===
            df['ema9_slope'] = df['ema_9'].pct_change(periods=3) * 100  # 3-period slope
            df['ema21_slope'] = df['ema_21'].pct_change(periods=5) * 100  # 5-period slope
            
            # Momentum classification based on EMA slopes
            momentum_conditions = (
                (df['ema9_slope'] > 0.1) & (df['ema21_slope'] > 0.05)  # Both EMAs rising
            )
            deceleration_conditions = (
                (df['ema9_slope'] < -0.1) & (df['ema21_slope'] < -0.05)  # Both EMAs falling
            )
            
            df.loc[momentum_conditions, 'trend_momentum'] = 'accelerating'
            df.loc[deceleration_conditions, 'trend_momentum'] = 'decelerating'
            # Default 'neutral' for others
            
            # === GOLDEN CROSS / DEATH CROSS DETECTION ===
            # Golden Cross: EMA9 crosses ABOVE EMA21 (Bullish)
            # Death Cross: EMA9 crosses BELOW EMA21 (Bearish)
            golden_cross = (df['ema_9'] > df['ema_21']) & (df['ema_9'].shift(1) <= df['ema_21'].shift(1))
            death_cross = (df['ema_9'] < df['ema_21']) & (df['ema_9'].shift(1) >= df['ema_21'].shift(1))
            
            df.loc[golden_cross & (df['trend_strength_score'] > 0.5), 'ema_cross_signal'] = 'golden_cross'
            df.loc[death_cross & (df['trend_strength_score'] < 0.5), 'ema_cross_signal'] = 'death_cross'
            
            # === DYNAMIC SUPPORT/RESISTANCE LEVELS ===
            # EMAs act as dynamic S/R - closer EMAs = stronger S/R
            df['ema_support_resistance'] = df[['ema_9', 'ema_21', 'sma_50']].mean(axis=1)
            
            # Clean up temporary columns
            df.drop(['ema9_slope', 'ema21_slope', 'ema_support_resistance'], axis=1, inplace=True, errors='ignore')
            
            logger.info("✅ MULTI EMA/SMA TREND HIERARCHY calculated - CONFLUENCE BEAST IS NOW COMPLETE! 🚀")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating Multi EMA/SMA: {e}")
            # Set safe defaults
            for col in ['ema_9', 'ema_21', 'sma_50', 'ema_200']:
                df[col] = df.get('Close', 0)
            
            df['trend_hierarchy'] = 'neutral'
            df['trend_momentum'] = 'neutral' 
            df['price_vs_emas'] = 'mixed'
            df['ema_cross_signal'] = 'neutral'
            df['trend_strength_score'] = 0.5
            
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
    
    def _calculate_vwap_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule VWAP avec bandes de déviation standard pour améliorer la précision des signaux"""
        try:
            # Calculate typical price (HLC/3)
            df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
            
            # Calculate VWAP (Volume Weighted Average Price)
            # VWAP = Sum(Typical Price * Volume) / Sum(Volume)
            df['price_volume'] = df['typical_price'] * df['Volume']
            df['cumulative_price_volume'] = df['price_volume'].expanding().sum()
            df['cumulative_volume'] = df['Volume'].expanding().sum()
            df['vwap'] = df['cumulative_price_volume'] / df['cumulative_volume']
            
            # Calculate rolling VWAP for better responsiveness (20-period)
            df['vwap_20'] = (df['typical_price'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
            
            # Use the rolling VWAP as main VWAP for better signals
            df['vwap'] = df['vwap_20'].fillna(df['vwap'])
            
            # Calculate standard deviation of (typical_price - vwap)
            df['vwap_deviation'] = df['typical_price'] - df['vwap']
            df['vwap_std'] = df['vwap_deviation'].rolling(window=20).std()
            
            # Calculate VWAP bands (1 and 2 standard deviations)
            df['vwap_upper_1std'] = df['vwap'] + df['vwap_std']
            df['vwap_lower_1std'] = df['vwap'] - df['vwap_std']
            df['vwap_upper_2std'] = df['vwap'] + (2 * df['vwap_std'])
            df['vwap_lower_2std'] = df['vwap'] - (2 * df['vwap_std'])
            
            # Calculate position relative to VWAP (percentage)
            df['vwap_position'] = ((df['Close'] - df['vwap']) / df['vwap']) * 100
            
            # VWAP signals for overbought/oversold conditions
            df['vwap_overbought'] = df['Close'] > df['vwap_upper_1std']
            df['vwap_oversold'] = df['Close'] < df['vwap_lower_1std']
            df['vwap_extreme_overbought'] = df['Close'] > df['vwap_upper_2std']
            df['vwap_extreme_oversold'] = df['Close'] < df['vwap_lower_2std']
            
            # VWAP trend determination
            df['vwap_trend'] = 'neutral'
            df.loc[df['Close'] > df['vwap'] * 1.002, 'vwap_trend'] = 'bullish'  # 0.2% above VWAP
            df.loc[df['Close'] < df['vwap'] * 0.998, 'vwap_trend'] = 'bearish'  # 0.2% below VWAP
            
            # Clean up temporary columns
            df.drop(['typical_price', 'price_volume', 'cumulative_price_volume', 
                    'cumulative_volume', 'vwap_20', 'vwap_deviation', 'vwap_std'], 
                   axis=1, inplace=True, errors='ignore')
            
            logger.info("✅ VWAP with standard deviation bands calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating VWAP bands: {e}")
            # Set default values if calculation fails
            for col in ['vwap', 'vwap_upper_1std', 'vwap_lower_1std', 'vwap_upper_2std', 
                       'vwap_lower_2std', 'vwap_position']:
                df[col] = df.get('Close', 0)
            for col in ['vwap_overbought', 'vwap_oversold', 'vwap_extreme_overbought', 'vwap_extreme_oversold']:
                df[col] = False
            df['vwap_trend'] = 'neutral'
            return df
    
    def _calculate_mfi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule Money Flow Index (MFI) - RSI pondéré par le volume
        🔥 DÉTECTION D'ACCUMULATION/DISTRIBUTION INSTITUTIONNELLE 🔥
        """
        try:
            # Typical Price (HLC/3) - Prix représentatif pour chaque période
            df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
            
            # Raw Money Flow = Typical Price * Volume (flux monétaire brut)
            df['raw_money_flow'] = df['typical_price'] * df['Volume']
            
            # Identifier les périodes de hausse/baisse des prix
            df['price_direction'] = df['typical_price'].diff()
            
            # Money Flow Positif (quand prix monte) et Négatif (quand prix baisse)
            df['positive_money_flow'] = np.where(df['price_direction'] > 0, df['raw_money_flow'], 0)
            df['negative_money_flow'] = np.where(df['price_direction'] < 0, df['raw_money_flow'], 0)
            
            # Calculer les sommes sur période glissante (14 périodes par défaut)
            period = 14
            df['positive_mf_sum'] = df['positive_money_flow'].rolling(window=period).sum()
            df['negative_mf_sum'] = df['negative_money_flow'].rolling(window=period).sum()
            
            # Money Flow Ratio = Positive MF / Negative MF
            df['money_flow_ratio'] = df['positive_mf_sum'] / (df['negative_mf_sum'] + 1e-10)  # Éviter division par 0
            
            # Money Flow Index = 100 - (100 / (1 + Money Flow Ratio))
            df['mfi'] = 100 - (100 / (1 + df['money_flow_ratio']))
            
            # MFI signals et zones critiques
            df['mfi_overbought'] = df['mfi'] > 80       # Distribution zone
            df['mfi_oversold'] = df['mfi'] < 20         # Accumulation zone  
            df['mfi_extreme_overbought'] = df['mfi'] > 90  # Extreme distribution
            df['mfi_extreme_oversold'] = df['mfi'] < 10    # Extreme accumulation
            
            # MFI trend determination
            df['mfi_trend'] = 'neutral'
            df.loc[df['mfi'] > 55, 'mfi_trend'] = 'bullish'    # Au-dessus de 55 = trend haussier
            df.loc[df['mfi'] < 45, 'mfi_trend'] = 'bearish'    # En-dessous de 45 = trend baissier
            
            # Détection d'activité institutionnelle basée sur MFI + Volume
            df['institutional_activity'] = 'neutral'
            
            # Accumulation institutionnelle (MFI bas + volume élevé)
            high_volume_threshold = df['Volume'].rolling(20).quantile(0.8)
            accumulation_conditions = (df['mfi'] < 30) & (df['Volume'] > high_volume_threshold)
            df.loc[accumulation_conditions, 'institutional_activity'] = 'accumulation'
            
            # Distribution institutionnelle (MFI haut + volume élevé)  
            distribution_conditions = (df['mfi'] > 70) & (df['Volume'] > high_volume_threshold)
            df.loc[distribution_conditions, 'institutional_activity'] = 'distribution'
            
            # Détection de divergences MFI/Prix (signal de retournement puissant)
            df['mfi_divergence'] = self._detect_mfi_divergence(df)
            
            # Nettoyage des colonnes temporaires
            df.drop(['typical_price', 'raw_money_flow', 'price_direction', 
                    'positive_money_flow', 'negative_money_flow', 
                    'positive_mf_sum', 'negative_mf_sum'], 
                   axis=1, inplace=True, errors='ignore')
            
            logger.info("✅ MFI (Money Flow Index) calculated successfully - Institutional activity detection active! 🔥")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating MFI: {e}")
            # Set safe defaults if calculation fails
            df['mfi'] = 50.0
            df['money_flow_ratio'] = 1.0
            for col in ['mfi_overbought', 'mfi_oversold', 'mfi_extreme_overbought', 
                       'mfi_extreme_oversold', 'mfi_divergence']:
                df[col] = False
            df['mfi_trend'] = 'neutral'
            df['institutional_activity'] = 'neutral'
            return df
    
    def _detect_mfi_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Détecte les divergences entre MFI et prix - signaux de retournement ultra-puissants"""
        divergence = pd.Series(False, index=df.index)
        
        if len(df) < 20:
            return divergence
        
        # Recherche de divergences sur les 15 dernières périodes
        for i in range(15, len(df)):
            price_window = df['Close'].iloc[i-15:i+1]
            mfi_window = df['mfi'].iloc[i-15:i+1]
            
            if len(price_window) > 10 and len(mfi_window) > 10:
                # Divergence haussière: Prix fait un plus bas mais MFI fait un plus haut
                # (Institutions accumulent malgré baisse des prix)
                if (price_window.iloc[-1] < price_window.iloc[0] and 
                    mfi_window.iloc[-1] > mfi_window.iloc[0]):
                    divergence.iloc[i] = True
                
                # Divergence baissière: Prix fait un plus haut mais MFI fait un plus bas  
                # (Institutions distribuent malgré hausse des prix)
                elif (price_window.iloc[-1] > price_window.iloc[0] and 
                      mfi_window.iloc[-1] < mfi_window.iloc[0]):
                    divergence.iloc[i] = True
        
        return divergence
    
    def _calculate_composite_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule indicateurs composites avec MULTI EMA/SMA CONFLUENCE! 🚀"""
        # Trend Strength (0-1) - NOW WITH EMA/SMA HIERARCHY POWER!
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
        
        # VWAP trend signal (enhanced precision)
        if 'vwap_trend' in df.columns:
            vwap_bullish = (df['vwap_trend'] == 'bullish').astype(int)
            trend_signals.append(vwap_bullish)
        
        # MFI trend signal (volume-weighted institutional sentiment) 🔥
        if 'mfi_trend' in df.columns:
            mfi_bullish = (df['mfi_trend'] == 'bullish').astype(int)
            trend_signals.append(mfi_bullish * 1.2)  # Higher weight for MFI (institutional money detection)
        
        # 🚀 MULTI EMA/SMA TREND HIERARCHY SIGNALS (THE CONFLUENCE BEAST!) 🚀
        if 'trend_hierarchy' in df.columns:
            # Convert trend hierarchy to numerical score
            hierarchy_scores = df['trend_hierarchy'].map({
                'strong_bull': 1.0,
                'weak_bull': 0.75,
                'neutral': 0.5,
                'weak_bear': 0.25,
                'strong_bear': 0.0
            }).fillna(0.5)
            trend_signals.append(hierarchy_scores * 1.3)  # HIGHEST WEIGHT - EMA hierarchy is KING!
        
        # EMA Golden/Death Cross signals (POWERFUL momentum shifts)
        if 'ema_cross_signal' in df.columns:
            golden_cross_signal = (df['ema_cross_signal'] == 'golden_cross').astype(int) * 1.0
            death_cross_signal = (df['ema_cross_signal'] == 'death_cross').astype(int) * -1.0
            cross_signal = golden_cross_signal + death_cross_signal + 0.5  # Convert to 0-1 scale
            trend_signals.append(cross_signal * 1.1)  # High weight for cross signals
        
        # Price vs EMAs positioning (Multi-level confirmation)
        if 'price_vs_emas' in df.columns:
            price_ema_scores = df['price_vs_emas'].map({
                'above_all': 1.0,     # Price above all EMAs = STRONG BULL
                'above_fast': 0.7,    # Price above fast EMAs = MODERATE BULL
                'below_fast': 0.3,    # Price below fast EMAs = MODERATE BEAR
                'below_all': 0.0,     # Price below all EMAs = STRONG BEAR
                'mixed': 0.5          # Mixed positioning = NEUTRAL
            }).fillna(0.5)
            trend_signals.append(price_ema_scores * 1.0)  # Strong weight for price positioning
        
        if trend_signals:
            df['trend_strength'] = np.mean(trend_signals, axis=0)
        else:
            df['trend_strength'] = 0.5
        
        # Momentum Score (0-1) - ENHANCED WITH EMA MOMENTUM!
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
        
        # VWAP momentum signal (position relative to VWAP)
        if 'vwap_position' in df.columns:
            # Normalize VWAP position to -1,1 range (clamp extreme values)
            vwap_momentum = np.clip(df['vwap_position'] / 3.0, -1, 1)  # 3% = full momentum
            momentum_signals.append(vwap_momentum)
        
        # MFI momentum signal (institutional money flow momentum) 🚀
        if 'mfi' in df.columns:
            # Convert MFI (0-100) to momentum (-1 to 1)
            mfi_momentum = (df['mfi'] - 50) / 50  # 50 = neutral, >50 = bullish, <50 = bearish
            momentum_signals.append(mfi_momentum * 1.3)  # Higher weight - institutions move first!
        
        # 🎯 EMA TREND MOMENTUM SIGNALS (Rate of change in trend!) 🎯
        if 'trend_momentum' in df.columns:
            # Convert trend momentum to numerical score
            momentum_scores = df['trend_momentum'].map({
                'accelerating': 0.8,   # Strong positive momentum
                'steady': 0.0,         # Neutral momentum
                'decelerating': -0.8,  # Strong negative momentum
                'neutral': 0.0         # Default neutral
            }).fillna(0.0)
            momentum_signals.append(momentum_scores * 1.2)  # High weight for EMA momentum
        
        # EMA Trend Strength Score boost (Direct trend strength)
        if 'trend_strength_score' in df.columns:
            # Convert 0-1 trend strength to -1,1 momentum (centered at 0.5)
            ema_trend_momentum = (df['trend_strength_score'] - 0.5) * 2  # Convert to -1,1
            momentum_signals.append(ema_trend_momentum * 1.1)  # Strong weight for pure trend strength
        
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
        """Détecte les signaux techniques composites AVEC CONFLUENCE EMA/SMA! 🚀"""
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
        
        # VWAP signals (enhanced precision for entries/exits)
        if 'vwap_position' in df.columns and 'vwap_overbought' in df.columns:
            total_signals += 2  # Weight VWAP more heavily
            
            # VWAP trend signal
            vwap_bullish = (df['vwap_trend'] == 'bullish').astype(int)
            vwap_bearish = (df['vwap_trend'] == 'bearish').astype(int)
            bullish_signals += vwap_bullish
            bearish_signals += vwap_bearish
            
            # VWAP mean reversion signal (contrarian when extreme)
            extreme_oversold = df['vwap_extreme_oversold'].astype(int)
            extreme_overbought = df['vwap_extreme_overbought'].astype(int)
            bullish_signals += extreme_oversold  # Buy when extremely oversold
            bearish_signals += extreme_overbought  # Sell when extremely overbought
        
        # MFI signals (INSTITUTIONAL MONEY DETECTION) 🔥💰
        if 'mfi' in df.columns and 'mfi_overbought' in df.columns:
            total_signals += 3  # Weight MFI HEAVILY - institutions move markets!
            
            # MFI overbought/oversold (stronger than regular RSI due to volume weighting)
            mfi_oversold_signal = df['mfi_oversold'].astype(int)
            mfi_overbought_signal = df['mfi_overbought'].astype(int)
            bullish_signals += mfi_oversold_signal * 1.5  # Higher weight
            bearish_signals += mfi_overbought_signal * 1.5
            
            # MFI extreme levels (ULTRA-STRONG signals)
            mfi_extreme_oversold = df['mfi_extreme_oversold'].astype(int)
            mfi_extreme_overbought = df['mfi_extreme_overbought'].astype(int)
            bullish_signals += mfi_extreme_oversold * 2.0  # MAXIMUM weight - institutions loading up!
            bearish_signals += mfi_extreme_overbought * 2.0  # MAXIMUM weight - institutions dumping!
            
            # MFI divergence (HOLY GRAIL signal - price vs institutional money flow)
            if 'mfi_divergence' in df.columns:
                mfi_divergence_signal = df['mfi_divergence'].astype(int)
                # Divergence can be bullish or bearish - add complexity based on MFI level
                bullish_divergence = mfi_divergence_signal & (df['mfi'] < 50)  # Divergence + MFI < 50 = bullish
                bearish_divergence = mfi_divergence_signal & (df['mfi'] > 50)  # Divergence + MFI > 50 = bearish
                bullish_signals += bullish_divergence.astype(int) * 2.5  # HOLY GRAIL weight
                bearish_signals += bearish_divergence.astype(int) * 2.5
        
        # 🚀🚀🚀 MULTI EMA/SMA SIGNALS - THE CONFLUENCE BEAST FINAL BOSS! 🚀🚀🚀
        if 'trend_hierarchy' in df.columns:
            total_signals += 4  # MAXIMUM WEIGHT - EMA hierarchy is the KING of trend detection!
            
            # Trend Hierarchy Signals (STRONGEST trend confirmation possible)
            strong_bull_signal = (df['trend_hierarchy'] == 'strong_bull').astype(int)
            weak_bull_signal = (df['trend_hierarchy'] == 'weak_bull').astype(int)
            strong_bear_signal = (df['trend_hierarchy'] == 'strong_bear').astype(int)
            weak_bear_signal = (df['trend_hierarchy'] == 'weak_bear').astype(int)
            
            bullish_signals += strong_bull_signal * 3.0   # NUCLEAR bullish weight
            bullish_signals += weak_bull_signal * 1.5     # Strong bullish weight
            bearish_signals += strong_bear_signal * 3.0   # NUCLEAR bearish weight
            bearish_signals += weak_bear_signal * 1.5     # Strong bearish weight
            
            # Golden Cross / Death Cross signals (LEGENDARY momentum shift indicators)
            if 'ema_cross_signal' in df.columns:
                golden_cross_signal = (df['ema_cross_signal'] == 'golden_cross').astype(int)
                death_cross_signal = (df['ema_cross_signal'] == 'death_cross').astype(int)
                bullish_signals += golden_cross_signal * 2.0  # LEGENDARY bullish momentum
                bearish_signals += death_cross_signal * 2.0   # LEGENDARY bearish momentum
            
            # Price vs EMAs positioning (Multi-layer trend confirmation)
            if 'price_vs_emas' in df.columns:
                above_all_emas = (df['price_vs_emas'] == 'above_all').astype(int)
                above_fast_emas = (df['price_vs_emas'] == 'above_fast').astype(int)
                below_fast_emas = (df['price_vs_emas'] == 'below_fast').astype(int)
                below_all_emas = (df['price_vs_emas'] == 'below_all').astype(int)
                
                bullish_signals += above_all_emas * 2.0      # STRONG bullish positioning
                bullish_signals += above_fast_emas * 1.0     # Moderate bullish positioning
                bearish_signals += below_fast_emas * 1.0     # Moderate bearish positioning
                bearish_signals += below_all_emas * 2.0      # STRONG bearish positioning
            
            # EMA Trend Momentum signals (Rate of change confirmation)
            if 'trend_momentum' in df.columns:
                accelerating_signal = (df['trend_momentum'] == 'accelerating').astype(int)
                decelerating_signal = (df['trend_momentum'] == 'decelerating').astype(int)
                bullish_signals += accelerating_signal * 1.5  # Acceleration = bullish momentum
                bearish_signals += decelerating_signal * 1.5  # Deceleration = bearish momentum
        
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
            
            # VWAP & Deviation Bands
            vwap=last_row.get('vwap', 0.0),
            vwap_upper_1std=last_row.get('vwap_upper_1std', 0.0),
            vwap_lower_1std=last_row.get('vwap_lower_1std', 0.0),
            vwap_upper_2std=last_row.get('vwap_upper_2std', 0.0),
            vwap_lower_2std=last_row.get('vwap_lower_2std', 0.0),
            vwap_position=last_row.get('vwap_position', 0.0),
            vwap_overbought=last_row.get('vwap_overbought', False),
            vwap_oversold=last_row.get('vwap_oversold', False),
            vwap_extreme_overbought=last_row.get('vwap_extreme_overbought', False),
            vwap_extreme_oversold=last_row.get('vwap_extreme_oversold', False),
            vwap_trend=last_row.get('vwap_trend', 'neutral'),
            
            # Money Flow Index (MFI) - Institutional Money Detection 🔥
            mfi=last_row.get('mfi', 50.0),
            mfi_overbought=last_row.get('mfi_overbought', False),
            mfi_oversold=last_row.get('mfi_oversold', False),
            mfi_extreme_overbought=last_row.get('mfi_extreme_overbought', False),
            mfi_extreme_oversold=last_row.get('mfi_extreme_oversold', False),
            mfi_trend=last_row.get('mfi_trend', 'neutral'),
            mfi_divergence=last_row.get('mfi_divergence', False),
            money_flow_ratio=last_row.get('money_flow_ratio', 1.0),
            institutional_activity=last_row.get('institutional_activity', 'neutral'),
            
            # Multi EMA/SMA Trend Hierarchy - THE MISSING PIECE 🎯
            ema_9=last_row.get('ema_9', 0.0),
            ema_21=last_row.get('ema_21', 0.0),
            ema_200=last_row.get('ema_200', 0.0),
            trend_hierarchy=last_row.get('trend_hierarchy', 'neutral'),
            trend_momentum=last_row.get('trend_momentum', 'neutral'),
            price_vs_emas=last_row.get('price_vs_emas', 'mixed'),
            ema_cross_signal=last_row.get('ema_cross_signal', 'neutral'),
            trend_strength_score=last_row.get('trend_strength_score', 0.5),
            
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
    
    def get_vwap_enhanced_signal(self, indicators: TechnicalIndicators, current_price: float) -> Dict[str, Any]:
        """
        Génère des signaux VWAP améliorés pour une meilleure précision des entrées/sorties
        Utilisable par IA1 et IA2 pour des signaux plus sensibles et précis
        """
        vwap_signals = {
            'signal_type': 'neutral',
            'strength': 0.0,
            'confidence': 0.0,
            'entry_precision': 'low',
            'risk_reward_potential': 1.0,
            'mean_reversion_opportunity': False,
            'trend_continuation_opportunity': False,
            'support_resistance_level': indicators.vwap,
            'recommended_action': 'hold',
            'reasoning': []
        }
        
        reasoning = []
        strength_factors = []
        
        # 1. VWAP Trend Analysis (Higher Weight)
        if indicators.vwap_trend == 'bullish':
            vwap_signals['trend_continuation_opportunity'] = True
            strength_factors.append(0.3)  # Strong positive factor
            reasoning.append(f"Prix {indicators.vwap_position:.1f}% au-dessus VWAP - tendance haussière")
        elif indicators.vwap_trend == 'bearish':
            vwap_signals['trend_continuation_opportunity'] = True
            strength_factors.append(-0.3)  # Strong negative factor
            reasoning.append(f"Prix {indicators.vwap_position:.1f}% en-dessous VWAP - tendance baissière")
        
        # 2. Mean Reversion Analysis (Extreme Levels)
        if indicators.vwap_extreme_oversold:
            vwap_signals['mean_reversion_opportunity'] = True
            vwap_signals['signal_type'] = 'bullish'
            vwap_signals['recommended_action'] = 'buy'
            vwap_signals['entry_precision'] = 'high'
            vwap_signals['risk_reward_potential'] = 2.5  # High RR potential
            strength_factors.append(0.5)  # Very strong reversal signal
            reasoning.append("EXTREME OVERSOLD vs VWAP (>2σ) - fort potentiel de rebond")
        elif indicators.vwap_extreme_overbought:
            vwap_signals['mean_reversion_opportunity'] = True
            vwap_signals['signal_type'] = 'bearish'
            vwap_signals['recommended_action'] = 'sell'
            vwap_signals['entry_precision'] = 'high'
            vwap_signals['risk_reward_potential'] = 2.5  # High RR potential
            strength_factors.append(-0.5)  # Very strong reversal signal
            reasoning.append("EXTREME OVERBOUGHT vs VWAP (>2σ) - fort potentiel de correction")
        
        # 3. Standard Deviation Band Analysis
        elif indicators.vwap_oversold and not indicators.vwap_extreme_oversold:
            vwap_signals['signal_type'] = 'bullish'
            vwap_signals['entry_precision'] = 'medium'
            vwap_signals['risk_reward_potential'] = 1.8
            strength_factors.append(0.25)
            reasoning.append("Oversold vs VWAP (1σ) - opportunité d'achat modérée")
        elif indicators.vwap_overbought and not indicators.vwap_extreme_overbought:
            vwap_signals['signal_type'] = 'bearish'
            vwap_signals['entry_precision'] = 'medium'
            vwap_signals['risk_reward_potential'] = 1.8
            strength_factors.append(-0.25)
            reasoning.append("Overbought vs VWAP (1σ) - opportunité de vente modérée")
        
        # 4. VWAP as Dynamic Support/Resistance
        price_vwap_distance = abs(indicators.vwap_position)
        if price_vwap_distance < 0.5:  # Very close to VWAP
            vwap_signals['support_resistance_level'] = indicators.vwap
            if indicators.vwap_trend != 'neutral':
                vwap_signals['entry_precision'] = 'high'
                reasoning.append(f"Prix près du VWAP ({price_vwap_distance:.2f}%) - niveau clé de S/R")
        
        # 5. Volume-Price Analysis Enhancement
        if hasattr(indicators, 'volume_ratio') and indicators.volume_ratio > 1.5:
            # High volume enhances VWAP signal reliability
            volume_boost = min(0.2, (indicators.volume_ratio - 1) * 0.1)
            if strength_factors:
                if strength_factors[-1] > 0:
                    strength_factors.append(volume_boost)
                else:
                    strength_factors.append(-volume_boost)
                reasoning.append(f"Volume élevé ({indicators.volume_ratio:.1f}x) confirme signal VWAP")
        
        # Calculate final strength and confidence
        if strength_factors:
            total_strength = np.mean(strength_factors)
            vwap_signals['strength'] = min(1.0, abs(total_strength))
            vwap_signals['confidence'] = min(0.95, vwap_signals['strength'] * 1.2)
            
            if total_strength > 0.1:
                vwap_signals['signal_type'] = 'bullish'
                if total_strength > 0.3:
                    vwap_signals['recommended_action'] = 'strong_buy'
                else:
                    vwap_signals['recommended_action'] = 'buy'
            elif total_strength < -0.1:
                vwap_signals['signal_type'] = 'bearish'
                if total_strength < -0.3:
                    vwap_signals['recommended_action'] = 'strong_sell'
                else:
                    vwap_signals['recommended_action'] = 'sell'
        
        vwap_signals['reasoning'] = reasoning
        
        return vwap_signals
    
    def get_mfi_vwap_combo_signal(self, indicators: TechnicalIndicators, current_price: float) -> Dict[str, Any]:
        """
        🔥 COMBO ULTRA-PUISSANT: MFI + VWAP = DÉTECTION INSTITUTIONNELLE + PRECISION 🔥
        Cette méthode combine la détection d'activité institutionnelle (MFI) avec la précision d'entrée (VWAP)
        pour générer des signaux de TRADING ABSOLUMENT DEVASTATEURS ! 💰
        """
        combo_signals = {
            'signal_type': 'neutral',
            'strength': 0.0,
            'confidence': 0.0,
            'institutional_confirmation': False,
            'precision_entry': False,
            'risk_reward_potential': 1.0,
            'signal_quality': 'low',  # low, medium, high, GODLIKE
            'recommended_action': 'hold',
            'entry_precision_level': 'VWAP',  # VWAP level to watch
            'institutional_signal': 'neutral',
            'combo_strength': 'weak',  # weak, medium, strong, NUCLEAR
            'reasoning': []
        }
        
        reasoning = []
        strength_factors = []
        
        # === 1. DÉTECTION INSTITUTIONNELLE (MFI) === 
        institutional_signal = 'neutral'
        institutional_strength = 0.0
        
        if indicators.mfi_extreme_oversold:
            institutional_signal = 'STRONG_BUY'
            institutional_strength = 0.8
            reasoning.append(f"🚨 INSTITUTIONS ACCUMULENT MASSIVEMENT (MFI: {indicators.mfi:.1f})")
        elif indicators.mfi_extreme_overbought:
            institutional_signal = 'STRONG_SELL'
            institutional_strength = -0.8
            reasoning.append(f"🚨 INSTITUTIONS DISTRIBUENT MASSIVEMENT (MFI: {indicators.mfi:.1f})")
        elif indicators.mfi_oversold:
            institutional_signal = 'BUY'
            institutional_strength = 0.5
            reasoning.append(f"💰 Accumulation institutionnelle (MFI: {indicators.mfi:.1f})")
        elif indicators.mfi_overbought:
            institutional_signal = 'SELL' 
            institutional_strength = -0.5
            reasoning.append(f"💸 Distribution institutionnelle (MFI: {indicators.mfi:.1f})")
        
        # === 2. PRECISION D'ENTRÉE (VWAP) ===
        vwap_precision = 'low'
        vwap_strength = 0.0
        
        if indicators.vwap_extreme_oversold:
            vwap_precision = 'GODLIKE'
            vwap_strength = 0.7
            reasoning.append(f"⚡ ENTRY PRECISION MAXIMALE - Prix {indicators.vwap_position:.1f}% sous VWAP (>2σ)")
        elif indicators.vwap_extreme_overbought:
            vwap_precision = 'GODLIKE'
            vwap_strength = -0.7
            reasoning.append(f"⚡ EXIT PRECISION MAXIMALE - Prix {indicators.vwap_position:.1f}% sur VWAP (>2σ)")
        elif indicators.vwap_oversold:
            vwap_precision = 'high'
            vwap_strength = 0.4
            reasoning.append(f"🎯 Entrée précise proche VWAP support ({indicators.vwap_position:.1f}%)")
        elif indicators.vwap_overbought:
            vwap_precision = 'high'
            vwap_strength = -0.4
            reasoning.append(f"🎯 Exit précis proche VWAP résistance ({indicators.vwap_position:.1f}%)")
        
        # === 3. COMBO MAGIQUE - CONFLUENCE INSTITUTIONNELLE + PRECISION === 
        if institutional_strength != 0 and vwap_strength != 0:
            # Les signaux vont dans la même direction = CONFLUENCE PUISSANTE
            if (institutional_strength > 0 and vwap_strength > 0) or (institutional_strength < 0 and vwap_strength < 0):
                combo_signals['institutional_confirmation'] = True
                combo_signals['precision_entry'] = True
                
                # Calcul de la force combinée (multiplicateur de confluence)
                combined_strength = (abs(institutional_strength) + abs(vwap_strength)) * 1.3  # Bonus confluence
                final_strength = combined_strength if institutional_strength > 0 else -combined_strength
                
                # Détermination de la qualité du signal
                if combined_strength > 1.0:
                    combo_signals['signal_quality'] = 'GODLIKE'
                    combo_signals['combo_strength'] = 'NUCLEAR'
                    combo_signals['risk_reward_potential'] = 3.5
                    reasoning.append("🚀 CONFLUENCE NUCLEAIRE: Institutions + VWAP = SIGNAL DEVASTATEUR!")
                elif combined_strength > 0.7:
                    combo_signals['signal_quality'] = 'high'
                    combo_signals['combo_strength'] = 'strong'
                    combo_signals['risk_reward_potential'] = 2.8
                    reasoning.append("💎 CONFLUENCE FORTE: Signal institutionnel confirmé par VWAP")
                else:
                    combo_signals['signal_quality'] = 'medium'
                    combo_signals['combo_strength'] = 'medium'
                    combo_signals['risk_reward_potential'] = 2.2
                
                # Direction du signal
                if final_strength > 0:
                    combo_signals['signal_type'] = 'bullish'
                    combo_signals['recommended_action'] = 'strong_buy' if combined_strength > 1.0 else 'buy'
                else:
                    combo_signals['signal_type'] = 'bearish'
                    combo_signals['recommended_action'] = 'strong_sell' if combined_strength > 1.0 else 'sell'
                
                combo_signals['strength'] = min(1.0, combined_strength)
                combo_signals['confidence'] = min(0.98, combined_strength * 0.85)
                
        # === 4. SIGNAUX SPÉCIAUX ===
        
        # Divergence MFI (HOLY GRAIL) + VWAP support/resistance
        if indicators.mfi_divergence:
            divergence_boost = 0.3
            if indicators.mfi < 50 and indicators.vwap_position < 0:  # Divergence haussière + sous VWAP
                strength_factors.append(divergence_boost)
                reasoning.append("👑 HOLY GRAIL: Divergence MFI haussière + VWAP support")
            elif indicators.mfi > 50 and indicators.vwap_position > 0:  # Divergence baissière + sur VWAP
                strength_factors.append(-divergence_boost)
                reasoning.append("👑 HOLY GRAIL: Divergence MFI baissière + VWAP résistance")
        
        # Activité institutionnelle massive avec volume
        if indicators.institutional_activity == 'accumulation' and indicators.vwap_trend == 'bullish':
            strength_factors.append(0.4)
            reasoning.append("🏦 INSTITUTIONS ACCUMULENT + VWAP haussier = BUY THE DIP")
        elif indicators.institutional_activity == 'distribution' and indicators.vwap_trend == 'bearish':
            strength_factors.append(-0.4)
            reasoning.append("🏦 INSTITUTIONS DISTRIBUENT + VWAP baissier = SELL THE RALLY")
        
        # === 5. FINALISATION DU SIGNAL ===
        if strength_factors and not combo_signals['institutional_confirmation']:
            # Signaux moins forts mais valides
            total_strength = np.mean(strength_factors)
            combo_signals['strength'] = min(1.0, abs(total_strength))
            combo_signals['confidence'] = min(0.85, combo_signals['strength'] * 0.9)
            
            if total_strength > 0.2:
                combo_signals['signal_type'] = 'bullish'
                combo_signals['recommended_action'] = 'buy'
            elif total_strength < -0.2:
                combo_signals['signal_type'] = 'bearish'
                combo_signals['recommended_action'] = 'sell'
        
        combo_signals['institutional_signal'] = institutional_signal
        combo_signals['entry_precision_level'] = indicators.vwap
        combo_signals['reasoning'] = reasoning
        
        return combo_signals
    
    def detect_ema_market_regime(self, indicators: TechnicalIndicators, current_price: float) -> Dict[str, Any]:
        """
        🎯 DÉTECTION RÉGIME MARCHÉ EMA/SMA - META-INDICATEUR POUR BIAIS DIRECTIONNEL
        Cette méthode détermine le RÉGIME GLOBAL (buy/sell) qui doit influencer toute l'analyse IA
        """
        regime_analysis = {
            'market_regime': 'neutral',
            'regime_strength': 0.5,
            'regime_bias': 'none',
            'regime_confidence': 0.5,
            'directional_filter': 'both',  # 'long_only', 'short_only', 'both'
            'regime_factors': [],
            'regime_warnings': [],
            'suggested_strategy': 'neutral'
        }
        
        regime_factors = []
        regime_score = 0.5  # Start neutral
        
        # === 1. TREND HIERARCHY REGIME ===
        if indicators.trend_hierarchy == 'strong_bull':
            regime_score += 0.3
            regime_factors.append("Perfect bullish EMA hierarchy")
            regime_analysis['directional_filter'] = 'long_preferred'
        elif indicators.trend_hierarchy == 'weak_bull':
            regime_score += 0.15
            regime_factors.append("Moderate bullish EMA structure")
            regime_analysis['directional_filter'] = 'long_bias'
        elif indicators.trend_hierarchy == 'strong_bear':
            regime_score -= 0.3
            regime_factors.append("Perfect bearish EMA hierarchy")
            regime_analysis['directional_filter'] = 'short_preferred'
        elif indicators.trend_hierarchy == 'weak_bear':
            regime_score -= 0.15
            regime_factors.append("Moderate bearish EMA structure")
            regime_analysis['directional_filter'] = 'short_bias'
        
        # === 2. GOLDEN/DEATH CROSS MOMENTUM REGIME ===
        if indicators.ema_cross_signal == 'golden_cross':
            regime_score += 0.2
            regime_factors.append("Golden Cross momentum shift")
            if regime_analysis['directional_filter'] == 'both':
                regime_analysis['directional_filter'] = 'long_momentum'
        elif indicators.ema_cross_signal == 'death_cross':
            regime_score -= 0.2
            regime_factors.append("Death Cross momentum shift")
            if regime_analysis['directional_filter'] == 'both':
                regime_analysis['directional_filter'] = 'short_momentum'
        
        # === 3. PRICE POSITIONING REGIME ===
        if indicators.price_vs_emas == 'above_all':
            regime_score += 0.15
            regime_factors.append("Price above all EMAs (institutional support)")
        elif indicators.price_vs_emas == 'above_fast':
            regime_score += 0.1
            regime_factors.append("Price above fast EMAs (short-term support)")
        elif indicators.price_vs_emas == 'below_fast':
            regime_score -= 0.1
            regime_factors.append("Price below fast EMAs (short-term resistance)")
        elif indicators.price_vs_emas == 'below_all':
            regime_score -= 0.15
            regime_factors.append("Price below all EMAs (institutional resistance)")
        
        # === 4. RECENT MOMENTUM OVERRIDE ===
        # Strong recent performance can override neutral EMA positioning
        # This captures emerging trends not yet reflected in slower EMAs
        if hasattr(indicators, 'recent_momentum'):
            recent_performance = getattr(indicators, 'recent_momentum', 0)
        else:
            # Calculate recent momentum as proxy (price vs EMA9 + EMA21)
            if indicators.ema_9 > 0 and indicators.ema_21 > 0:
                price_vs_fast_emas = (current_price / ((indicators.ema_9 + indicators.ema_21) / 2) - 1) * 100
                if price_vs_fast_emas > 3.0:  # More than 3% above fast EMAs
                    regime_score += 0.15
                    regime_factors.append(f"Strong recent momentum (+{price_vs_fast_emas:.1f}% vs fast EMAs)")
                elif price_vs_fast_emas < -3.0:
                    regime_score -= 0.15
                    regime_factors.append(f"Weak recent momentum ({price_vs_fast_emas:.1f}% vs fast EMAs)")
        
        # === 5. TREND STRENGTH MULTIPLIER ===
        strength_multiplier = indicators.trend_strength_score
        regime_score = 0.5 + (regime_score - 0.5) * strength_multiplier
        
        # === 6. CLASSIFY MARKET REGIME (ADJUSTED THRESHOLDS) ===
        regime_analysis['regime_strength'] = abs(regime_score - 0.5) * 2  # Convert to 0-1
        
        if regime_score >= 0.75:  # Lowered from 0.8
            regime_analysis['market_regime'] = 'strong_buy'
            regime_analysis['regime_bias'] = 'bullish'
            regime_analysis['suggested_strategy'] = 'buy_dips'
            regime_analysis['directional_filter'] = 'long_only'
        elif regime_score >= 0.58:  # Lowered from 0.65
            regime_analysis['market_regime'] = 'buy'
            regime_analysis['regime_bias'] = 'bullish'
            regime_analysis['suggested_strategy'] = 'long_bias'
            regime_analysis['directional_filter'] = 'long_preferred'
        elif regime_score <= 0.25:  # Lowered from 0.2
            regime_analysis['market_regime'] = 'strong_sell'
            regime_analysis['regime_bias'] = 'bearish'
            regime_analysis['suggested_strategy'] = 'sell_rallies'
            regime_analysis['directional_filter'] = 'short_only'
        elif regime_score <= 0.42:  # Lowered from 0.35
            regime_analysis['market_regime'] = 'sell'
            regime_analysis['regime_bias'] = 'bearish'
            regime_analysis['suggested_strategy'] = 'short_bias'
            regime_analysis['directional_filter'] = 'short_preferred'
        else:
            regime_analysis['market_regime'] = 'neutral'
            regime_analysis['regime_bias'] = 'mixed'
            regime_analysis['suggested_strategy'] = 'wait'
            regime_analysis['directional_filter'] = 'both'
        
        # === 6. CONFIDENCE BASED ON CLARITY ===
        regime_analysis['regime_confidence'] = min(0.95, regime_analysis['regime_strength'] + 0.3)
        
        regime_analysis['regime_factors'] = regime_factors
        
        return regime_analysis

    def get_ema_sma_trend_analysis(self, indicators: TechnicalIndicators, current_price: float) -> Dict[str, Any]:
        """
        🎯 ANALYSE COMPLÈTE EMA/SMA TREND HIERARCHY - CONFLUENCE MATRIX FINAL PIECE! 🎯
        Fournit une analyse détaillée de la hiérarchie des tendances EMA/SMA pour IA1 et IA2
        """
        ema_analysis = {
            'trend_direction': 'neutral',
            'trend_strength': 'weak',
            'momentum_state': 'neutral',
            'price_positioning': 'mixed',
            'cross_signals': 'none',
            'support_resistance_levels': [],
            'confidence_score': 0.5,
            'risk_reward_potential': 1.0,
            'trend_probability': 0.5,
            'recommended_strategy': 'wait',
            'key_levels': {},
            'trend_analysis': [],
            'confluence_factors': []
        }
        
        analysis_points = []
        confluence_factors = []
        confidence_factors = []
        
        # === 1. TREND HIERARCHY ANALYSIS ===
        if indicators.trend_hierarchy != 'neutral':
            if indicators.trend_hierarchy == 'strong_bull':
                ema_analysis['trend_direction'] = 'strong_bullish'
                ema_analysis['trend_strength'] = 'very_strong'
                ema_analysis['confidence_score'] = 0.9
                ema_analysis['trend_probability'] = 0.85
                ema_analysis['recommended_strategy'] = 'buy_dips'
                analysis_points.append("🚀 STRONG BULL HIERARCHY: Price > EMA9 > EMA21 > SMA50 > EMA200")
                confluence_factors.append("Perfect bull hierarchy alignment")
            elif indicators.trend_hierarchy == 'weak_bull':
                ema_analysis['trend_direction'] = 'bullish'
                ema_analysis['trend_strength'] = 'moderate'
                ema_analysis['confidence_score'] = 0.7
                ema_analysis['trend_probability'] = 0.68
                ema_analysis['recommended_strategy'] = 'cautious_buy'
                analysis_points.append("📈 WEAK BULL HIERARCHY: Partial bullish EMA alignment")
                confluence_factors.append("Moderate bullish EMA structure")
            elif indicators.trend_hierarchy == 'strong_bear':
                ema_analysis['trend_direction'] = 'strong_bearish'
                ema_analysis['trend_strength'] = 'very_strong'
                ema_analysis['confidence_score'] = 0.9
                ema_analysis['trend_probability'] = 0.15
                ema_analysis['recommended_strategy'] = 'sell_rallies'
                analysis_points.append("💥 STRONG BEAR HIERARCHY: Price < EMA9 < EMA21 < SMA50 < EMA200")
                confluence_factors.append("Perfect bear hierarchy alignment")
            elif indicators.trend_hierarchy == 'weak_bear':
                ema_analysis['trend_direction'] = 'bearish'
                ema_analysis['trend_strength'] = 'moderate'
                ema_analysis['confidence_score'] = 0.7
                ema_analysis['trend_probability'] = 0.32
                ema_analysis['recommended_strategy'] = 'cautious_sell'
                analysis_points.append("📉 WEAK BEAR HIERARCHY: Partial bearish EMA alignment")
                confluence_factors.append("Moderate bearish EMA structure")
        
        # === 2. PRICE POSITIONING ANALYSIS ===
        positioning_analysis = {
            'above_all': ('🟢 MAXIMUM BULLISH POSITIONING', 'Above all EMAs = institutional support', 0.9),
            'above_fast': ('🟡 MODERATE BULLISH POSITIONING', 'Above fast EMAs = short-term bullish', 0.7),
            'below_fast': ('🟠 MODERATE BEARISH POSITIONING', 'Below fast EMAs = short-term bearish', 0.3),
            'below_all': ('🔴 MAXIMUM BEARISH POSITIONING', 'Below all EMAs = institutional resistance', 0.1),
            'mixed': ('⚪ MIXED POSITIONING', 'Conflicting EMA signals', 0.5)
        }
        
        if indicators.price_vs_emas in positioning_analysis:
            description, explanation, prob_adjustment = positioning_analysis[indicators.price_vs_emas]
            ema_analysis['price_positioning'] = indicators.price_vs_emas
            ema_analysis['trend_probability'] *= prob_adjustment
            analysis_points.append(f"{description}: {explanation}")
            confluence_factors.append(f"Price positioning: {indicators.price_vs_emas}")
        
        # === 3. GOLDEN/DEATH CROSS ANALYSIS ===
        if indicators.ema_cross_signal != 'neutral':
            if indicators.ema_cross_signal == 'golden_cross':
                ema_analysis['cross_signals'] = 'golden_cross'
                ema_analysis['momentum_state'] = 'accelerating_bull'
                ema_analysis['risk_reward_potential'] = 2.5
                analysis_points.append("⚡ GOLDEN CROSS: EMA9 crossed above EMA21 - MOMENTUM SHIFT!")
                confluence_factors.append("Golden Cross momentum signal")
                confidence_factors.append(0.8)
            elif indicators.ema_cross_signal == 'death_cross':
                ema_analysis['cross_signals'] = 'death_cross'
                ema_analysis['momentum_state'] = 'accelerating_bear'
                ema_analysis['risk_reward_potential'] = 2.5
                analysis_points.append("💥 DEATH CROSS: EMA9 crossed below EMA21 - MOMENTUM SHIFT!")
                confluence_factors.append("Death Cross momentum signal")
                confidence_factors.append(0.8)
        
        # === 4. TREND MOMENTUM ANALYSIS ===
        if indicators.trend_momentum != 'neutral':
            momentum_analysis = {
                'accelerating': ('🚀 ACCELERATING', 'EMAs spreading apart = increasing momentum', 0.8),
                'decelerating': ('🛑 DECELERATING', 'EMAs converging = decreasing momentum', 0.3),
            }
            
            if indicators.trend_momentum in momentum_analysis:
                description, explanation, momentum_confidence = momentum_analysis[indicators.trend_momentum]
                ema_analysis['momentum_state'] = indicators.trend_momentum
                analysis_points.append(f"{description}: {explanation}")
                confluence_factors.append(f"Trend momentum: {indicators.trend_momentum}")
                confidence_factors.append(momentum_confidence)
        
        # === 5. DYNAMIC SUPPORT/RESISTANCE LEVELS ===
        key_levels = {}
        support_resistance = []
        
        if indicators.ema_9 > 0:
            key_levels['ema_9'] = indicators.ema_9
            support_resistance.append(f"EMA9: ${indicators.ema_9:.4f} (Ultra-fast S/R)")
        
        if indicators.ema_21 > 0:
            key_levels['ema_21'] = indicators.ema_21
            support_resistance.append(f"EMA21: ${indicators.ema_21:.4f} (Fast S/R)")
        
        if indicators.sma_50 > 0:
            key_levels['sma_50'] = indicators.sma_50
            support_resistance.append(f"SMA50: ${indicators.sma_50:.4f} (Institutional S/R)")
        
        if indicators.ema_200 > 0:
            key_levels['ema_200'] = indicators.ema_200
            support_resistance.append(f"EMA200: ${indicators.ema_200:.4f} (Major trend S/R)")
        
        ema_analysis['key_levels'] = key_levels
        ema_analysis['support_resistance_levels'] = support_resistance
        
        # === 6. TREND STRENGTH SCORE INTEGRATION ===
        if indicators.trend_strength_score != 0.5:
            strength_percentage = indicators.trend_strength_score * 100
            if strength_percentage > 80:
                ema_analysis['trend_strength'] = 'very_strong'
                analysis_points.append(f"💪 VERY STRONG TREND: {strength_percentage:.0f}% EMA hierarchy alignment")
            elif strength_percentage > 60:
                ema_analysis['trend_strength'] = 'strong'
                analysis_points.append(f"💪 STRONG TREND: {strength_percentage:.0f}% EMA hierarchy alignment")
            elif strength_percentage < 20:
                ema_analysis['trend_strength'] = 'very_weak'
                analysis_points.append(f"📉 VERY WEAK TREND: {strength_percentage:.0f}% EMA hierarchy alignment")
            elif strength_percentage < 40:
                ema_analysis['trend_strength'] = 'weak'
                analysis_points.append(f"📉 WEAK TREND: {strength_percentage:.0f}% EMA hierarchy alignment")
        
        # === 7. CONFLUENCE CALCULATION ===
        if confidence_factors:
            ema_analysis['confidence_score'] = min(0.95, np.mean(confidence_factors))
        
        # Risk/Reward Potential based on trend clarity
        if ema_analysis['trend_strength'] in ['very_strong', 'strong'] and ema_analysis['cross_signals'] != 'none':
            ema_analysis['risk_reward_potential'] = 3.0
        elif ema_analysis['trend_direction'] in ['strong_bullish', 'strong_bearish']:
            ema_analysis['risk_reward_potential'] = 2.5
        elif ema_analysis['cross_signals'] != 'none':
            ema_analysis['risk_reward_potential'] = 2.0
        
        ema_analysis['trend_analysis'] = analysis_points
        ema_analysis['confluence_factors'] = confluence_factors
        
        return ema_analysis
    
    def get_latest_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """
        Extrait les indicateurs les plus récents du DataFrame calculé
        Alias pour get_current_indicators pour compatibilité multi-timeframe
        """
        return self.get_current_indicators(df)

    async def get_multi_timeframe_indicators_real(self, symbol: str) -> Dict[str, TechnicalIndicators]:
        """
        🚀 VRAIS INDICATEURS MULTI-TIMEFRAME - DONNÉES OHLCV ACTUELLES
        Récupère et analyse les données de DIFFÉRENTS timeframes au moment de la requête
        """
        multi_tf_indicators = {}
        
        try:
            from enhanced_ohlcv_fetcher import enhanced_ohlcv_fetcher
            
            # Récupérer les vraies données multi-timeframe
            logger.info(f"🔍 MULTI-TIMEFRAME REAL DATA: Fetching actual OHLCV for {symbol}")
            multi_tf_data = await enhanced_ohlcv_fetcher.get_multi_timeframe_ohlcv_data(symbol)
            
            if not multi_tf_data:
                logger.warning(f"⚠️ No multi-timeframe data available for {symbol}")
                return {}
            
            # Calculer les indicateurs pour chaque timeframe
            for tf_name, tf_data in multi_tf_data.items():
                try:
                    if len(tf_data) < 50:  # Minimum pour calculer les indicateurs
                        logger.warning(f"⚠️ {tf_name}: Not enough data ({len(tf_data)} points)")
                        continue
                    
                    # Calculer tous les indicateurs pour ce timeframe
                    logger.info(f"🔧 Calculating indicators for {tf_name} ({len(tf_data)} points)")
                    tf_df_with_indicators = self.calculate_all_indicators(tf_data.copy())
                    
                    if len(tf_df_with_indicators) > 0:
                        # Extraire les dernières valeurs (plus récentes)
                        tf_indicators = self.get_latest_indicators(tf_df_with_indicators)
                        multi_tf_indicators[tf_name] = tf_indicators
                        
                        # Log détaillé pour debug
                        logger.info(f"✅ {tf_name}: RSI={tf_indicators.rsi_14:.1f}, "
                                  f"Trend={tf_indicators.trend_hierarchy}, "
                                  f"EMA9=${tf_indicators.ema_9:.6f}")
                    
                except Exception as e:
                    logger.error(f"❌ Error calculating {tf_name} indicators: {e}")
                    continue
            
            logger.info(f"🎯 MULTI-TIMEFRAME REAL SUCCESS: {len(multi_tf_indicators)} timeframes calculated")
            return multi_tf_indicators
            
        except Exception as e:
            logger.error(f"❌ Error in real multi-timeframe calculation: {e}")
            return {}

    def get_multi_timeframe_indicators(self, df: pd.DataFrame) -> Dict[str, TechnicalIndicators]:
        """
        🚀 RÉVOLUTION MULTI-TIMEFRAME: Calcule tous les indicateurs sur TOUS les timeframes
        Comme un trader professionnel qui regarde Daily, 4H, 1H, 5min pour ses décisions !
        """
        multi_tf_indicators = {}
        
        try:
            # S'assurer qu'on a assez de données pour tous les timeframes
            if len(df) < 300:  # Au moins 300 périodes pour avoir 14 jours de données
                logger.warning(f"⚠️ Pas assez de données pour multi-timeframe: {len(df)} périodes")
                # Fallback sur timeframes courts seulement
                available_timeframes = {k: v for k, v in self.timeframes.items() if v <= len(df) // 2}
            else:
                available_timeframes = self.timeframes
            
            logger.info(f"🔍 MULTI-TIMEFRAME ANALYSIS: Calculating indicators for {len(available_timeframes)} timeframes")
            
            for tf_name, periods_back in available_timeframes.items():
                try:
                    if tf_name == 'now':
                        # Timeframe actuel - utilise toute la data
                        tf_df = df.copy()
                    else:
                        # Timeframes historiques - décale les données
                        if periods_back >= len(df):
                            continue  # Skip si pas assez de données
                        
                        # Prendre un snapshot des données à ce moment dans le passé
                        end_idx = len(df) - periods_back
                        tf_df = df.iloc[:end_idx + 50].copy()  # +50 pour avoir assez de données pour les calculs
                        
                        if len(tf_df) < 50:  # Minimum pour calculer les indicateurs
                            continue
                    
                    # Calculer tous les indicateurs pour ce timeframe
                    tf_df_with_indicators = self.calculate_all_indicators(tf_df)
                    
                    if len(tf_df_with_indicators) > 0:
                        # Extraire les dernières valeurs (plus récentes pour ce timeframe)
                        tf_indicators = self.get_latest_indicators(tf_df_with_indicators)
                        multi_tf_indicators[tf_name] = tf_indicators
                        
                        logger.debug(f"✅ {tf_name}: RSI={tf_indicators.rsi_14:.1f}, MFI={tf_indicators.mfi:.1f}, VWAP=${tf_indicators.vwap:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error calculating {tf_name} indicators: {e}")
                    continue
            
            logger.info(f"🎯 MULTI-TIMEFRAME SUCCESS: {len(multi_tf_indicators)} timeframes calculated")
            return multi_tf_indicators
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe calculation: {e}")
            # Fallback to current timeframe only
            try:
                current_indicators = self.get_latest_indicators(self.calculate_all_indicators(df))
                return {'now': current_indicators}
            except:
                # Ultimate fallback - empty dict
                return {}
    
    def format_multi_timeframe_for_prompt(self, multi_tf_indicators: Dict[str, TechnicalIndicators]) -> str:
        """
        🎯 Formate les indicateurs multi-timeframe pour le prompt IA1
        Structure hiérarchique: Long terme → Court terme (comme un trader pro)
        """
        try:
            if not multi_tf_indicators:
                return "⚠️ Multi-timeframe data not available"
            
            # Ordre hiérarchique pour analyse professionnelle
            timeframe_order = ['14d', '5d', '1d', '4h', '1h', '5min', 'now']
            available_tfs = [tf for tf in timeframe_order if tf in multi_tf_indicators]
            
            if not available_tfs:
                return "⚠️ No timeframe data available"
            
            formatted_lines = []
            formatted_lines.append("🔍 MULTI-TIMEFRAME PROFESSIONAL ANALYSIS:")
            
            for tf in available_tfs:
                indicators = multi_tf_indicators[tf]
                
                # Nom du timeframe plus lisible
                tf_names = {
                    '14d': '📅 14-DAY TREND',
                    '5d': '📈 5-DAY MOMENTUM', 
                    '1d': '📊 DAILY STRUCTURE',
                    '4h': '⏰ 4H INTERMEDIATE',
                    '1h': '🕐 1H SHORT-TERM',
                    '5min': '⚡ 5MIN PRECISION',
                    'now': '🎯 CURRENT LEVELS'
                }
                
                # Indicateurs clés formatés
                rsi_status = "OVERSOLD" if indicators.rsi_14 < 30 else "OVERBOUGHT" if indicators.rsi_14 > 70 else "NEUTRAL"
                mfi_status = "ACCUMULATION" if indicators.mfi < 30 else "DISTRIBUTION" if indicators.mfi > 70 else "NEUTRAL"
                vwap_status = "ABOVE" if indicators.vwap_position > 0 else "BELOW"
                
                # Ligne formatée pour ce timeframe
                tf_line = f"{tf_names.get(tf, tf.upper())}: RSI {indicators.rsi_14:.1f}({rsi_status}) | MFI {indicators.mfi:.1f}({mfi_status}) | VWAP {vwap_status}({indicators.vwap_position:+.1f}%)"
                
                # Ajouter des signaux spéciaux pour timeframes clés
                if tf in ['14d', '5d'] and (indicators.mfi < 20 or indicators.mfi > 80):
                    tf_line += f" | 🚨 INSTITUTIONAL {'LOADING' if indicators.mfi < 20 else 'DUMPING'}"
                elif tf == 'now' and (indicators.vwap_extreme_oversold or indicators.vwap_extreme_overbought):
                    tf_line += f" | 🎯 PRECISION {'ENTRY' if indicators.vwap_extreme_oversold else 'EXIT'}"
                
                formatted_lines.append(tf_line)
            
            return '\n            '.join(formatted_lines)
            
        except Exception as e:
            logger.error(f"Error formatting multi-timeframe data: {e}")
            return "⚠️ Error formatting multi-timeframe analysis"

# Instance globale
advanced_technical_indicators = AdvancedTechnicalIndicators()
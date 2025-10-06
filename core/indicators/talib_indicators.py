"""
Professional TALib Indicators System v6.0
Complete implementation of all technical indicators with exact formulas
Based on the detailed specification provided
"""

import pandas as pd
import numpy as np
import talib
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from .base_indicator import TechnicalAnalysisComplete, IndicatorResult

logger = logging.getLogger(__name__)

@dataclass
class RegimeDetectionResult:
    """Résultat complet de la détection de régime"""
    regime: str = "CONSOLIDATION"
    confidence: float = 0.5
    base_confidence: float = 0.5
    technical_consistency: float = 0.5
    combined_confidence: float = 0.5
    regime_persistence: int = 0
    stability_score: float = 0.5
    regime_transition_alert: str = "STABLE"
    fresh_regime: bool = False
    indicators: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.indicators is None:
            self.indicators = {}

class TALibIndicators:
    """
    Professional Technical Indicators using TA-Lib
    Implements all indicators from the detailed specification
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        
        # Load configuration from file or use defaults
        self.config = config or self._load_default_config()
        
        # Paramètres configurables
        self.adx_period = self.config.get('adx_period', 14)
        self.rsi_period = self.config.get('rsi_period', 14)
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2)
        self.atr_period = self.config.get('atr_period', 14)
        
        # Seuils importants
        self.adx_weak = self.config.get('adx_weak', 20)
        self.adx_strong = self.config.get('adx_strong', 25)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.bb_squeeze_threshold = self.config.get('bb_squeeze_threshold', 2.0)
        
        logger.info(f"✅ TALibIndicators initialized with config: {len(self.config)} parameters")
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate input data format and requirements
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check minimum length
            min_length = self.get_min_periods()
            if len(df) < min_length:
                logger.warning(f"TALibIndicators: Insufficient data {len(df)} < {min_length}")
                return False
            
            # Check required columns
            required_cols = self.get_required_columns()
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"TALibIndicators: Missing columns {missing_cols}")
                return False
            
            # Check for NaN values in critical columns
            for col in required_cols:
                if df[col].isna().any():
                    logger.warning(f"TALibIndicators: NaN values found in {col}")
            
            return True
            
        except Exception as e:
            logger.error(f"TALibIndicators: Data validation error: {e}")
            return False
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'adx_period': 14,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14,
            'adx_weak': 20,
            'adx_strong': 25,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'bb_squeeze_threshold': 2.0
        }
    
    def get_min_periods(self) -> int:
        """Minimum periods required for calculations with smart fallback"""
        # Smart minimum: try to get optimal, but work with available
        optimal_minimum = max(20, self.macd_slow)  # 26 for MACD
        return min(optimal_minimum, 20)  # Never require more than 20 days - use fallback completion
    
    def get_required_columns(self) -> List[str]:
        """Required DataFrame columns"""
        return ['open', 'high', 'low', 'close', 'volume']
    
    def calculate_all_indicators(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> TechnicalAnalysisComplete:
        """
        Calculate ALL technical indicators from specification
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol for logging
            
        Returns:
            TechnicalAnalysisComplete with all calculated indicators
        """
        try:
            logger.info(f"🔬 Calculating ALL TALib indicators for {symbol} with {len(df)} bars")
            logger.info(f"   📊 Original columns: {list(df.columns)}")
            
            # Normalize DataFrame to standard OHLCV format
            from .base_indicator import IndicatorUtils
            normalized_df = IndicatorUtils.normalize_data(df)
            
            if normalized_df is None or len(normalized_df) == 0:
                logger.error(f"❌ Data normalization failed for {symbol}")
                return self._create_minimal_analysis(df, symbol)
            
            logger.info(f"   ✅ Normalized columns: {list(normalized_df.columns)}")
            
            # Validate normalized data - but try to calculate even with limited data
            is_valid_data = self.validate_data(normalized_df)
            if not is_valid_data:
                logger.warning(f"⚠️ Limited data for {symbol} ({len(normalized_df)} bars), attempting calculation anyway...")
                # Continue with calculations - many indicators can work with less data
            
            # Extract OHLCV arrays from normalized data
            open_prices = normalized_df['open'].values.astype(np.float64)
            high_prices = normalized_df['high'].values.astype(np.float64)
            low_prices = normalized_df['low'].values.astype(np.float64)
            close_prices = normalized_df['close'].values.astype(np.float64)
            volume = normalized_df['volume'].values.astype(np.float64)
            
            current_price = float(close_prices[-1])
            
            # 🔥 1. INDICATEURS DE TENDANCE
            trend_indicators = self._calculate_trend_indicators(high_prices, low_prices, close_prices)
            
            # 🔥 2. INDICATEURS DE VOLATILITÉ  
            volatility_indicators = self._calculate_volatility_indicators(high_prices, low_prices, close_prices)
            
            # 🔥 3. BOLLINGER BANDS
            bb_indicators = self._calculate_bollinger_bands(close_prices)
            
            # 🔥 4. INDICATEURS DE MOMENTUM
            momentum_indicators = self._calculate_momentum_indicators(close_prices)
            
            # 🔥 5. INDICATEURS DE VOLUME
            volume_indicators = self._calculate_volume_indicators(volume)
            
            # 🔥 6. STOCHASTIC
            stoch_indicators = self._calculate_stochastic_indicators(high_prices, low_prices, close_prices)
            
            # 🔥 7. MFI (Money Flow Index)
            mfi_indicators = self._calculate_mfi_indicators(high_prices, low_prices, close_prices, volume)
            
            # 🔥 8. VWAP
            vwap_indicators = self._calculate_vwap_indicators(high_prices, low_prices, close_prices, volume, current_price)
            
            # 🔥 9. MOVING AVERAGES
            ma_indicators = self._calculate_moving_averages(close_prices, current_price)
            
            # 🔥 10. DÉTECTION DE RÉGIME
            regime_result = self._detect_market_regime(
                trend_indicators, volatility_indicators, bb_indicators,
                momentum_indicators, volume_indicators, ma_indicators
            )
            
            # 🔥 11. CONFLUENCE CALCULATION
            confluence_result = self._calculate_confluence_grade(
                regime_result, trend_indicators, momentum_indicators,
                bb_indicators, volume_indicators
            )
            
            # Construire l'analyse technique complète
            analysis = TechnicalAnalysisComplete(
                # Régime
                regime=regime_result.regime,
                confidence=regime_result.combined_confidence,
                base_confidence=regime_result.base_confidence,
                technical_consistency=regime_result.technical_consistency,
                combined_confidence=regime_result.combined_confidence,
                
                # RSI
                rsi_14=momentum_indicators['rsi'],
                rsi_9=momentum_indicators.get('rsi_9', momentum_indicators['rsi']),
                rsi_21=momentum_indicators.get('rsi_21', momentum_indicators['rsi']),
                rsi_zone=momentum_indicators['rsi_zone'],
                
                # MACD
                macd_line=momentum_indicators['macd_line'],
                macd_signal=momentum_indicators['macd_signal_line'],
                macd_histogram=momentum_indicators['macd_histogram'],
                macd_trend=momentum_indicators['macd_trend'],
                
                # ADX
                adx=trend_indicators['adx'],
                plus_di=trend_indicators['plus_di'],
                minus_di=trend_indicators['minus_di'],
                adx_strength=trend_indicators['adx_strength'],
                
                # Bollinger Bands
                bb_upper=bb_indicators['bb_upper'],
                bb_middle=bb_indicators['bb_middle'],
                bb_lower=bb_indicators['bb_lower'],
                bb_position=bb_indicators['bb_position'],
                bb_squeeze=bb_indicators['bb_squeeze'],
                squeeze_intensity=bb_indicators['squeeze_intensity'],
                
                # Stochastic
                stoch_k=stoch_indicators['stoch_k'],
                stoch_d=stoch_indicators['stoch_d'],
                
                # MFI
                mfi=mfi_indicators['mfi'],
                mfi_signal=mfi_indicators['mfi_signal'],
                
                # ATR & Volatility
                atr=volatility_indicators['atr'],
                atr_pct=volatility_indicators['atr_pct'],
                
                # Volume
                volume_ratio=volume_indicators['volume_ratio'],
                volume_trend=volume_indicators['volume_trend'],
                volume_surge=volume_indicators['volume_surge'],
                
                # Moving Averages
                sma_20=ma_indicators['sma_20'],
                sma_50=ma_indicators['sma_50'],
                ema_9=ma_indicators['ema_9'],
                ema_21=ma_indicators['ema_21'],
                ema_50=ma_indicators['ema_50'],
                ema_200=ma_indicators['ema_200'],
                
                # Trend Analysis
                trend_hierarchy=ma_indicators['trend_hierarchy'],
                trend_strength_score=ma_indicators['trend_strength_score'],
                price_vs_emas=ma_indicators['price_vs_emas'],
                
                # VWAP
                vwap=vwap_indicators['vwap'],
                vwap_distance=vwap_indicators['vwap_distance'],
                above_vwap=vwap_indicators['above_vwap'],
                
                # Confluence
                confluence_grade=confluence_result['grade'],
                confluence_score=confluence_result['score'],
                should_trade=confluence_result['should_trade'],
                conviction_level=confluence_result['conviction_level']
            )
            
            logger.info(f"✅ TALib analysis complete for {symbol}: {confluence_result['grade']} grade, Regime: {regime_result.regime}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error calculating TALib indicators for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return self._create_minimal_analysis(df, symbol)
    
    def _calculate_trend_indicators(self, high: np.array, low: np.array, close: np.array) -> Dict[str, Any]:
        """Calculate trend indicators (ADX, slopes, positions)"""
        try:
            # ADX with correct Wilder method
            adx = talib.ADX(high, low, close, timeperiod=self.adx_period)
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=self.adx_period)
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=self.adx_period)
            
            # Get latest values
            adx_value = float(adx[-1]) if not np.isnan(adx[-1]) else 25.0
            plus_di_value = float(plus_di[-1]) if not np.isnan(plus_di[-1]) else 25.0
            minus_di_value = float(minus_di[-1]) if not np.isnan(minus_di[-1]) else 25.0
            
            # ADX Strength classification
            if adx_value >= 50:
                adx_strength = "VERY_STRONG"
            elif adx_value >= self.adx_strong:
                adx_strength = "STRONG"
            elif adx_value >= self.adx_weak:
                adx_strength = "MODERATE"
            else:
                adx_strength = "WEAK"
            
            # SMA calculations for slopes
            close_series = pd.Series(close)
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            
            # Calculate slopes
            sma_20_slope = self._calculate_slope(sma_20[-10:]) if len(sma_20) >= 10 else 0.0
            sma_50_slope = self._calculate_slope(sma_50[-20:]) if len(sma_50) >= 20 else 0.0
            
            # Position relative to SMAs
            current_price = close[-1]
            above_sma_20 = current_price > sma_20[-1] if not np.isnan(sma_20[-1]) else True
            above_sma_50 = current_price > sma_50[-1] if not np.isnan(sma_50[-1]) else True
            
            # Distance to SMAs (%)
            distance_sma_20 = ((current_price / sma_20[-1]) - 1) * 100 if not np.isnan(sma_20[-1]) else 0.0
            distance_sma_50 = ((current_price / sma_50[-1]) - 1) * 100 if not np.isnan(sma_50[-1]) else 0.0
            
            return {
                'adx': adx_value,
                'plus_di': plus_di_value,
                'minus_di': minus_di_value,
                'adx_strength': adx_strength,
                'sma_20_slope': sma_20_slope,
                'sma_50_slope': sma_50_slope,
                'above_sma_20': above_sma_20,
                'above_sma_50': above_sma_50,
                'distance_sma_20': distance_sma_20,
                'distance_sma_50': distance_sma_50
            }
            
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}")
            return {
                'adx': 25.0, 'plus_di': 25.0, 'minus_di': 25.0, 'adx_strength': 'MODERATE',
                'sma_20_slope': 0.0, 'sma_50_slope': 0.0, 'above_sma_20': True, 'above_sma_50': True,
                'distance_sma_20': 0.0, 'distance_sma_50': 0.0
            }
    
    def _calculate_volatility_indicators(self, high: np.array, low: np.array, close: np.array) -> Dict[str, Any]:
        """Calculate volatility indicators (ATR, ratios)"""
        try:
            # ATR calculations
            atr_14 = talib.ATR(high, low, close, timeperiod=14)
            atr_50 = talib.ATR(high, low, close, timeperiod=50)
            
            atr_value = float(atr_14[-1]) if not np.isnan(atr_14[-1]) else 0.02
            atr_50_value = float(atr_50[-1]) if not np.isnan(atr_50[-1]) else atr_value
            
            # ATR percentage
            current_price = close[-1]
            atr_pct = (atr_value / current_price * 100) if current_price > 0 else 2.0
            
            # Volatility ratio
            volatility_ratio = atr_value / atr_50_value if atr_50_value > 0 else 1.0
            
            return {
                'atr': atr_value,
                'atr_pct': atr_pct,
                'volatility_ratio': volatility_ratio
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {e}")
            return {'atr': 0.02, 'atr_pct': 2.0, 'volatility_ratio': 1.0}
    
    def _calculate_bollinger_bands(self, close: np.array) -> Dict[str, Any]:
        """Calculate Bollinger Bands indicators"""
        try:
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                close, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std
            )
            
            # Get latest values
            upper = float(bb_upper[-1]) if not np.isnan(bb_upper[-1]) else close[-1] * 1.02
            middle = float(bb_middle[-1]) if not np.isnan(bb_middle[-1]) else close[-1]
            lower = float(bb_lower[-1]) if not np.isnan(bb_lower[-1]) else close[-1] * 0.98
            
            # BB Position (0-1)
            current_price = close[-1]
            bb_position = ((current_price - lower) / (upper - lower)) if (upper - lower) > 0 else 0.5
            
            # BB Width percentage
            bb_width_pct = ((upper - lower) / middle * 100) if middle > 0 else 4.0
            
            # Squeeze detection
            bb_squeeze = bb_width_pct < self.bb_squeeze_threshold
            
            # Squeeze intensity
            if bb_squeeze:
                squeeze_intensity = 'EXTREME' if bb_width_pct < 1.0 else 'TIGHT'
            else:
                squeeze_intensity = 'NONE'
            
            return {
                'bb_upper': upper,
                'bb_middle': middle,
                'bb_lower': lower,
                'bb_position': bb_position,
                'bb_width_pct': bb_width_pct,
                'bb_squeeze': bb_squeeze,
                'squeeze_intensity': squeeze_intensity
            }
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            current_price = close[-1]
            return {
                'bb_upper': current_price * 1.02,
                'bb_middle': current_price,
                'bb_lower': current_price * 0.98,
                'bb_position': 0.5,
                'bb_width_pct': 4.0,
                'bb_squeeze': False,
                'squeeze_intensity': 'NONE'
            }
    
    def _calculate_momentum_indicators(self, close: np.array) -> Dict[str, Any]:
        """Calculate momentum indicators (RSI, MACD)"""
        try:
            # RSI
            rsi = talib.RSI(close, timeperiod=self.rsi_period)
            rsi_value = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0
            
            # RSI Zone
            if rsi_value > self.rsi_overbought:
                rsi_zone = "OVERBOUGHT"
            elif rsi_value < self.rsi_oversold:
                rsi_zone = "OVERSOLD"
            elif 40 <= rsi_value <= 60:
                rsi_zone = "NEUTRAL"
            else:
                rsi_zone = "NORMAL"
            
            # RSI Trend (slope)
            rsi_trend = self._calculate_slope(rsi[-10:]) if len(rsi) >= 10 else 0.0
            
            # MACD
            macd_line, macd_signal_line, macd_histogram = talib.MACD(
                close, fastperiod=self.macd_fast, slowperiod=self.macd_slow, signalperiod=self.macd_signal
            )
            
            macd_line_value = float(macd_line[-1]) if not np.isnan(macd_line[-1]) else 0.0
            macd_signal_value = float(macd_signal_line[-1]) if not np.isnan(macd_signal_line[-1]) else 0.0
            macd_histogram_value = float(macd_histogram[-1]) if not np.isnan(macd_histogram[-1]) else 0.0
            
            # MACD Trend (histogram slope)
            macd_trend = self._calculate_slope(macd_histogram[-10:]) if len(macd_histogram) >= 10 else 0.0
            
            # MACD signal classification
            if macd_histogram_value > 0:
                macd_signal_type = "BULLISH"
            elif macd_histogram_value < 0:
                macd_signal_type = "BEARISH"
            else:
                macd_signal_type = "NEUTRAL"
            
            return {
                'rsi': rsi_value,
                'rsi_zone': rsi_zone,
                'rsi_trend': rsi_trend,
                'macd_line': macd_line_value,
                'macd_signal_line': macd_signal_value,
                'macd_histogram': macd_histogram_value,
                'macd_trend': macd_signal_type,
                'macd_slope': macd_trend
            }
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            return {
                'rsi': 50.0, 'rsi_zone': 'NEUTRAL', 'rsi_trend': 0.0,
                'macd_line': 0.0, 'macd_signal_line': 0.0, 'macd_histogram': 0.0,
                'macd_trend': 'NEUTRAL', 'macd_slope': 0.0
            }
    
    def _calculate_stochastic_indicators(self, high: np.array, low: np.array, close: np.array) -> Dict[str, Any]:
        """Calculate Stochastic indicators"""
        try:
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(
                high, low, close, 
                fastk_period=14, slowk_period=3, slowk_matype=0, 
                slowd_period=3, slowd_matype=0
            )
            
            stoch_k_value = float(stoch_k[-1]) if not np.isnan(stoch_k[-1]) else 50.0
            stoch_d_value = float(stoch_d[-1]) if not np.isnan(stoch_d[-1]) else 50.0
            
            return {
                'stoch_k': stoch_k_value,
                'stoch_d': stoch_d_value
            }
            
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return {'stoch_k': 50.0, 'stoch_d': 50.0}
    
    def _calculate_mfi_indicators(self, high: np.array, low: np.array, close: np.array, volume: np.array) -> Dict[str, Any]:
        """Calculate Money Flow Index indicators"""
        try:
            # MFI (Money Flow Index)
            mfi = talib.MFI(high, low, close, volume, timeperiod=14)
            mfi_value = float(mfi[-1]) if not np.isnan(mfi[-1]) else 50.0
            
            # MFI Signal
            if mfi_value > 80:
                mfi_signal = "DISTRIBUTION"
            elif mfi_value < 20:
                mfi_signal = "ACCUMULATION"
            else:
                mfi_signal = "NEUTRAL"
            
            return {
                'mfi': mfi_value,
                'mfi_signal': mfi_signal
            }
            
        except Exception as e:
            logger.error(f"Error calculating MFI: {e}")
            return {'mfi': 50.0, 'mfi_signal': 'NEUTRAL'}
    
    def _calculate_vwap_indicators(self, high: np.array, low: np.array, close: np.array, volume: np.array, current_price: float) -> Dict[str, Any]:
        """Calculate VWAP indicators"""
        try:
            # VWAP calculation
            typical_price = (high + low + close) / 3
            vwap = np.sum(typical_price * volume) / np.sum(volume) if np.sum(volume) > 0 else current_price
            
            # VWAP distance (%)
            vwap_distance = ((current_price - vwap) / vwap * 100) if vwap > 0 else 0.0
            
            # Above VWAP
            above_vwap = current_price > vwap
            
            return {
                'vwap': float(vwap),
                'vwap_distance': vwap_distance,
                'above_vwap': above_vwap
            }
            
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
            return {'vwap': current_price, 'vwap_distance': 0.0, 'above_vwap': True}
    
    def _calculate_volume_indicators(self, volume: np.array) -> Dict[str, Any]:
        """Calculate volume indicators"""
        try:
            # Volume SMA
            volume_sma = talib.SMA(volume, timeperiod=20)
            volume_sma_value = float(volume_sma[-1]) if not np.isnan(volume_sma[-1]) else volume[-1]
            
            # Volume ratio
            current_volume = volume[-1]
            volume_ratio = current_volume / volume_sma_value if volume_sma_value > 0 else 1.0
            
            # Volume trend (slope)
            volume_trend = self._calculate_slope(volume[-10:]) if len(volume) >= 10 else 0.0
            
            # Volume surge
            volume_surge = volume_ratio > 2.0
            
            return {
                'volume_ratio': volume_ratio,
                'volume_trend': volume_trend,
                'volume_surge': volume_surge
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            return {'volume_ratio': 1.0, 'volume_trend': 0.0, 'volume_surge': False}
    
    def _calculate_moving_averages(self, close: np.array, current_price: float) -> Dict[str, Any]:
        """Calculate moving averages and trend analysis"""
        try:
            # Moving averages
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            ema_9 = talib.EMA(close, timeperiod=9)
            ema_21 = talib.EMA(close, timeperiod=21)
            ema_50 = talib.EMA(close, timeperiod=50)
            ema_200 = talib.EMA(close, timeperiod=200)
            
            # Get latest values
            sma_20_val = float(sma_20[-1]) if not np.isnan(sma_20[-1]) else current_price
            sma_50_val = float(sma_50[-1]) if not np.isnan(sma_50[-1]) else current_price
            ema_9_val = float(ema_9[-1]) if not np.isnan(ema_9[-1]) else current_price
            ema_21_val = float(ema_21[-1]) if not np.isnan(ema_21[-1]) else current_price
            ema_50_val = float(ema_50[-1]) if not np.isnan(ema_50[-1]) else current_price
            ema_200_val = float(ema_200[-1]) if not np.isnan(ema_200[-1]) else current_price
            
            # Trend hierarchy
            if ema_9_val > ema_21_val > ema_50_val:
                trend_hierarchy = "BULLISH"
                trend_strength_score = 0.8
            elif ema_9_val < ema_21_val < ema_50_val:
                trend_hierarchy = "BEARISH"
                trend_strength_score = 0.2
            else:
                trend_hierarchy = "NEUTRAL"
                trend_strength_score = 0.5
            
            # Price vs EMAs
            if current_price > ema_21_val:
                price_vs_emas = "ABOVE"
            elif current_price < ema_21_val:
                price_vs_emas = "BELOW"
            else:
                price_vs_emas = "NEUTRAL"
            
            return {
                'sma_20': sma_20_val,
                'sma_50': sma_50_val,
                'ema_9': ema_9_val,
                'ema_21': ema_21_val,
                'ema_50': ema_50_val,
                'ema_200': ema_200_val,
                'trend_hierarchy': trend_hierarchy,
                'trend_strength_score': trend_strength_score,
                'price_vs_emas': price_vs_emas
            }
            
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
            return {
                'sma_20': current_price, 'sma_50': current_price,
                'ema_9': current_price, 'ema_21': current_price, 'ema_50': current_price, 'ema_200': current_price,
                'trend_hierarchy': 'NEUTRAL', 'trend_strength_score': 0.5, 'price_vs_emas': 'NEUTRAL'
            }
    
    def _detect_market_regime(self, trend_indicators: Dict, volatility_indicators: Dict, 
                            bb_indicators: Dict, momentum_indicators: Dict, 
                            volume_indicators: Dict, ma_indicators: Dict) -> RegimeDetectionResult:
        """Detect market regime using all indicators"""
        
        # Score each regime
        regime_scores = {
            'TRENDING_UP_STRONG': 0.0,
            'TRENDING_UP_MODERATE': 0.0,
            'BREAKOUT_BULLISH': 0.0,
            'CONSOLIDATION': 0.0,
            'RANGING_TIGHT': 0.0,
            'RANGING_WIDE': 0.0,
            'VOLATILE': 0.0,
            'TRENDING_DOWN_MODERATE': 0.0,
            'TRENDING_DOWN_STRONG': 0.0,
            'BREAKOUT_BEARISH': 0.0
        }
        
        adx = trend_indicators['adx']
        rsi = momentum_indicators['rsi']
        macd_hist = momentum_indicators['macd_histogram']
        bb_squeeze = bb_indicators['bb_squeeze']
        sma_slope = trend_indicators['sma_20_slope']
        volume_ratio = volume_indicators['volume_ratio']
        atr_pct = volatility_indicators['atr_pct']
        trend_hierarchy = ma_indicators['trend_hierarchy']
        
        # TRENDING_UP_STRONG
        if (adx > 25 and trend_hierarchy == "BULLISH" and 
            macd_hist > 0 and sma_slope > 0.003 and rsi > 50):
            regime_scores['TRENDING_UP_STRONG'] += 0.8
        
        # BREAKOUT_BULLISH
        if (bb_squeeze and bb_indicators['squeeze_intensity'] in ["EXTREME", "TIGHT"] and
            volume_ratio > 1.5 and macd_hist > 0):
            regime_scores['BREAKOUT_BULLISH'] += 0.85
        
        # CONSOLIDATION
        if (adx < 20 and abs(sma_slope) < 0.001 and
            40 < rsi < 60 and not bb_squeeze):
            regime_scores['CONSOLIDATION'] += 0.7
        
        # VOLATILE
        if atr_pct > 3.0 and volume_ratio > 1.8:
            regime_scores['VOLATILE'] += 0.6
        
        # TRENDING_DOWN_STRONG
        if (adx > 25 and trend_hierarchy == "BEARISH" and
            macd_hist < 0 and sma_slope < -0.003 and rsi < 50):
            regime_scores['TRENDING_DOWN_STRONG'] += 0.8
        
        # Select best regime
        best_regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[best_regime]
        
        # Default if no strong signal
        if confidence < 0.5:
            best_regime = "CONSOLIDATION"
            confidence = 0.5
        
        # Calculate technical consistency
        technical_consistency = self._calculate_technical_consistency(
            rsi, macd_hist, trend_hierarchy, volume_ratio, atr_pct, 
            sma_slope, trend_indicators['above_sma_20'], adx
        )
        
        combined_confidence = 0.7 * confidence + 0.3 * technical_consistency
        
        return RegimeDetectionResult(
            regime=best_regime,
            confidence=combined_confidence,
            base_confidence=confidence,
            technical_consistency=technical_consistency,
            combined_confidence=combined_confidence,
            regime_persistence=15,  # Placeholder
            stability_score=0.8,    # Placeholder
            regime_transition_alert="STABLE",
            fresh_regime=True,
            indicators={
                **trend_indicators,
                **volatility_indicators,
                **bb_indicators,
                **momentum_indicators,
                **volume_indicators,
                **ma_indicators
            }
        )
    
    def _calculate_technical_consistency(self, rsi: float, macd_hist: float, trend_hierarchy: str,
                                       volume_ratio: float, atr_pct: float, sma_slope: float,
                                       above_sma_20: bool, adx: float) -> float:
        """Calculate technical consistency score"""
        
        consistency_score = 0.0
        
        # Trend Coherence (35%)
        trend_score = 0.0
        if trend_hierarchy in ["BULLISH", "BEARISH"]:
            trend_score += 0.3
        if abs(sma_slope) > 0.001:
            trend_score += 0.2
        if adx > 18:
            trend_score += 0.3
        if above_sma_20 == (trend_hierarchy == "BULLISH"):
            trend_score += 0.2
        
        consistency_score += 0.35 * trend_score
        
        # Momentum Coherence (35%)
        momentum_score = 0.0
        if ((rsi > 50) == (macd_hist > 0)):
            momentum_score += 0.5
        if 35 < rsi < 70:
            momentum_score += 0.3
        if abs(macd_hist) > 0.001:
            momentum_score += 0.2
        
        consistency_score += 0.35 * momentum_score
        
        # Volume Coherence (15%)
        volume_score = 0.0
        if volume_ratio > 1.0:
            volume_score += 0.6
        if 0.8 < volume_ratio < 2.5:
            volume_score += 0.4
        
        consistency_score += 0.15 * volume_score
        
        # Volatility Coherence (15%)
        volatility_score = 0.0
        if 0.5 < atr_pct < 4.0:
            volatility_score += 0.6
        if atr_pct < 3.0:
            volatility_score += 0.4
        
        consistency_score += 0.15 * volatility_score
        
        # Debug logging for consistency calculation
        logger.debug(f"🔬 Technical Consistency Breakdown: Trend={trend_score:.2f} (35%), Momentum={momentum_score:.2f} (35%), Volume={volume_score:.2f} (15%), Volatility={volatility_score:.2f} (15%) → Total={consistency_score:.2f}")
        
        return min(1.0, max(0.0, consistency_score))
    
    def _calculate_confluence_grade(self, regime_result: RegimeDetectionResult, 
                                  trend_indicators: Dict, momentum_indicators: Dict,
                                  bb_indicators: Dict, volume_indicators: Dict) -> Dict[str, Any]:
        """Calculate confluence grade A++ to D"""
        
        score = 0
        combined_confidence = regime_result.combined_confidence
        adx = trend_indicators['adx']
        rsi = momentum_indicators['rsi']
        macd_hist = momentum_indicators['macd_histogram']
        bb_squeeze = bb_indicators['bb_squeeze']
        volume_ratio = volume_indicators['volume_ratio']
        technical_consistency = regime_result.technical_consistency
        
        # Mandatory requirements check
        mandatory_met = (
            combined_confidence > 0.65 and
            (adx > 18 or bb_squeeze) and
            volume_ratio > 1.0
        )
        
        if not mandatory_met:
            return {
                'grade': 'D',
                'score': 0,
                'should_trade': False,
                'conviction_level': 'ÉVITER'
            }
        
        # Base score from confidence
        score += combined_confidence * 40
        
        # Momentum conditions
        momentum_conditions = 0
        if 40 <= rsi <= 65:
            momentum_conditions += 1
            score += 5
        if abs(macd_hist) > 0.001:
            momentum_conditions += 1
            score += 5
        if bb_squeeze:
            momentum_conditions += 1
            score += 8
        if abs(trend_indicators['sma_20_slope']) > 0.001:
            momentum_conditions += 1
            score += 5
        if volume_ratio > 1.2:
            momentum_conditions += 1
            score += 5
        if trend_indicators['above_sma_20']:
            momentum_conditions += 1
            score += 5
        
        if momentum_conditions < 2:
            return {
                'grade': 'C',
                'score': max(50, int(score)),
                'should_trade': False,
                'conviction_level': 'ATTENDRE'
            }
        
        # High conviction triggers
        if bb_squeeze and combined_confidence > 0.75 and volume_ratio > 1.8:
            score += 15
        if adx > 25 and combined_confidence > 0.8:
            score += 15
        if volume_ratio > 2.0:
            score += 10
        
        score += technical_consistency * 10
        score = min(100, int(score))
        
        # Determine grade
        if score >= 90:
            return {'grade': 'A++', 'score': score, 'should_trade': True, 'conviction_level': 'TRÈS HAUTE'}
        elif score >= 80:
            return {'grade': 'A+', 'score': score, 'should_trade': True, 'conviction_level': 'TRÈS HAUTE'}
        elif score >= 75:
            return {'grade': 'A', 'score': score, 'should_trade': True, 'conviction_level': 'HAUTE'}
        elif score >= 70:
            return {'grade': 'B+', 'score': score, 'should_trade': True, 'conviction_level': 'HAUTE'}
        elif score >= 65:
            return {'grade': 'B', 'score': score, 'should_trade': True, 'conviction_level': 'MOYENNE'}
        else:
            return {'grade': 'C', 'score': score, 'should_trade': False, 'conviction_level': 'FAIBLE'}
    
    def _calculate_slope(self, series: np.array) -> float:
        """Calculate slope of a time series"""
        try:
            if len(series) < 2:
                return 0.0
            
            # Remove NaN values
            clean_series = series[~np.isnan(series)]
            if len(clean_series) < 2:
                return 0.0
            
            x = np.arange(len(clean_series))
            slope = np.polyfit(x, clean_series, 1)[0]
            return float(slope)
            
        except Exception as e:
            logger.debug(f"Error calculating slope: {e}")
            return 0.0
    
    def _create_minimal_analysis(self, df: pd.DataFrame, symbol: str) -> TechnicalAnalysisComplete:
        """Create minimal analysis for insufficient data"""
        
        current_price = float(df['close'].iloc[-1]) if len(df) > 0 else 100.0
        
        return TechnicalAnalysisComplete(
            # Regime
            regime="CONSOLIDATION",
            confidence=0.5,
            technical_consistency=0.5,
            
            # RSI
            rsi_14=50.0,
            rsi_9=50.0,
            rsi_21=50.0,
            rsi_zone="NEUTRAL",
            
            # MACD
            macd_line=0.0,
            macd_signal=0.0,
            macd_histogram=0.0,
            macd_trend="NEUTRAL",
            
            # ADX
            adx=25.0,
            plus_di=25.0,
            minus_di=25.0,
            adx_strength="MODERATE",
            
            # Bollinger Bands
            bb_upper=current_price * 1.02,
            bb_middle=current_price,
            bb_lower=current_price * 0.98,
            bb_position=0.5,
            bb_squeeze=False,
            squeeze_intensity="NONE",
            
            # Stochastic
            stoch_k=50.0,
            stoch_d=50.0,
            
            # MFI
            mfi=50.0,
            mfi_signal="NEUTRAL",
            
            # ATR & Volatility
            atr=0.02,
            atr_pct=2.0,
            
            # Volume
            volume_ratio=1.0,
            volume_trend=0.0,
            volume_surge=False,
            
            # Moving Averages
            sma_20=current_price,
            sma_50=current_price,
            ema_9=current_price,
            ema_21=current_price,
            ema_50=current_price,
            ema_200=current_price,
            
            # Trend Analysis
            trend_hierarchy="NEUTRAL",
            trend_strength_score=0.5,
            price_vs_emas="NEUTRAL",
            
            # VWAP
            vwap=current_price,
            vwap_distance=0.0,
            above_vwap=True,
            
            # Confluence
            confluence_grade='C',
            confluence_score=50,
            should_trade=False,
            conviction_level='FAIBLE'
        )

# Global instance
talib_indicators = TALibIndicators()

def get_talib_indicators() -> TALibIndicators:
    """Get global TALib indicators instance"""
    return talib_indicators
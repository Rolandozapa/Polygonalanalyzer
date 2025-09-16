#!/usr/bin/env python3
"""
MACD Calculator - Module optimisé pour l'Ultra Professional Trading Bot
Calculs MACD robustes avec validation avancée et gestion d'erreurs
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MACDResult:
    """Résultat structuré du calcul MACD"""
    macd_line: float
    signal_line: float  
    histogram: float
    is_valid: bool
    data_points: int
    message: str

class MACDCalculator:
    """
    Calculateur MACD optimisé pour trading cryptocurrencies
    Conçu pour fonctionner avec des données OHLCV limitées et volatiles
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_period = fast_period
        self.slow_period = slow_period  
        self.signal_period = signal_period
        self.min_data_points = max(slow_period + signal_period, 35)  # 26+9 = 35 minimum for MACD
        
    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calcul EMA robuste - optimisé pour toutes les gammes de prix crypto
        """
        try:
            if len(prices) < period:
                logger.debug(f"EMA: Insufficient data ({len(prices)} < {period})")
                return pd.Series(dtype=float)
            
            # Nettoyage des données
            clean_prices = prices.dropna()
            if len(clean_prices) < period:
                logger.debug(f"EMA: Insufficient clean data ({len(clean_prices)} < {period})")
                return pd.Series(dtype=float)
            
            # 🚀 FIXED: Removed problematic micro-price scaling logic
            # Direct EMA calculation works for all price ranges
            ema = clean_prices.ewm(span=period, adjust=False).mean()
            logger.debug(f"EMA: Standard calculation for period {period}, median price: {clean_prices.median():.2f}")
            
            return ema
            
        except Exception as e:
            logger.error(f"EMA calculation error for period {period}: {e}")
            return pd.Series(dtype=float)
    
    def _validate_data(self, prices: pd.Series) -> Tuple[bool, str]:
        """Validation complète des données d'entrée"""
        try:
            if prices is None or len(prices) == 0:
                return False, "Empty price data"
            
            if len(prices) < self.min_data_points:
                return False, f"Insufficient data: {len(prices)} < {self.min_data_points} required"
            
            # Vérification des valeurs nulles/infinies
            clean_prices = prices.dropna()
            if len(clean_prices) < self.min_data_points:
                return False, f"Too many null values: {len(clean_prices)} clean points"
            
            # Vérification des valeurs négatives/zéro
            if (clean_prices <= 0).any():
                return False, "Contains zero or negative prices"
            
            # Vérification de la variation (éviter les prix constants)
            price_std = clean_prices.std()
            price_mean = clean_prices.mean()
            if price_mean > 0 and (price_std / price_mean) < 0.0001:  # Variation < 0.01%
                return False, f"Insufficient price variation: {price_std/price_mean:.6f}"
            
            return True, f"Valid data: {len(clean_prices)} points, variation {price_std/price_mean:.4f}"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def calculate(self, prices: pd.Series) -> MACDResult:
        """
        Calcul MACD principal avec validation complète
        
        Args:
            prices: Série des prix (généralement Close prices)
            
        Returns:
            MACDResult avec tous les détails du calcul
        """
        try:
            # Étape 1: Validation des données
            is_valid, validation_msg = self._validate_data(prices)
            if not is_valid:
                logger.debug(f"MACD validation failed: {validation_msg}")
                return MACDResult(0.0, 0.0, 0.0, False, len(prices), validation_msg)
            
            logger.debug(f"MACD validation passed: {validation_msg}")
            
            # Étape 2: Calcul des EMAs
            ema_fast = self._calculate_ema(prices, self.fast_period)
            ema_slow = self._calculate_ema(prices, self.slow_period)
            
            if ema_fast.empty or ema_slow.empty:
                return MACDResult(0.0, 0.0, 0.0, False, len(prices), "EMA calculation failed")
            
            # Étape 3: Calcul MACD Line
            macd_line_series = ema_fast - ema_slow
            if macd_line_series.empty:
                return MACDResult(0.0, 0.0, 0.0, False, len(prices), "MACD line calculation failed")
            
            # Étape 4: Calcul Signal Line (EMA du MACD)
            signal_line_series = self._calculate_ema(macd_line_series, self.signal_period)
            if signal_line_series.empty:
                return MACDResult(0.0, 0.0, 0.0, False, len(prices), "Signal line calculation failed")
            
            # Étape 5: Valeurs finales (derniers points)
            macd_line = float(macd_line_series.iloc[-1])
            signal_line = float(signal_line_series.iloc[-1])
            histogram = macd_line - signal_line
            
            # Étape 6: Validation des résultats
            if any(np.isnan([macd_line, signal_line, histogram])) or any(np.isinf([macd_line, signal_line, histogram])):
                return MACDResult(0.0, 0.0, 0.0, False, len(prices), "Invalid calculation results (NaN/Inf)")
            
            # Étape 7: Succès
            success_msg = f"MACD calculated successfully: Line={macd_line:.8f}, Signal={signal_line:.8f}"
            logger.debug(success_msg)
            
            return MACDResult(macd_line, signal_line, histogram, True, len(prices), success_msg)
            
        except Exception as e:
            error_msg = f"MACD calculation error: {e}"
            logger.error(error_msg)
            return MACDResult(0.0, 0.0, 0.0, False, len(prices), error_msg)
    
    def get_signal(self, result: MACDResult) -> str:
        """
        Interprétation du signal MACD
        
        Returns:
            'bullish', 'bearish', 'neutral'
        """
        if not result.is_valid:
            return 'neutral'
        
        try:
            # Signal basé sur MACD Line vs Signal Line
            if result.macd_line > result.signal_line:
                if result.histogram > 0:
                    return 'bullish'
                else:
                    return 'neutral'  # Transition
            else:
                if result.histogram < 0:
                    return 'bearish'
                else:
                    return 'neutral'  # Transition
                    
        except Exception as e:
            logger.error(f"MACD signal interpretation error: {e}")
            return 'neutral'

# Instance globale pour utilisation dans l'application
macd_calculator = MACDCalculator()

def calculate_macd_optimized(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
    """
    Interface de compatibilité avec le code existant
    
    Returns:
        Tuple[macd_line, signal_line, histogram]
    """
    try:
        calculator = MACDCalculator(fast, slow, signal)
        result = calculator.calculate(prices)
        
        if result.is_valid:
            logger.info(f"✅ MACD optimized: Line={result.macd_line:.8f}, Signal={result.signal_line:.8f}, Histogram={result.histogram:.8f}")
            return result.macd_line, result.signal_line, result.histogram
        else:
            logger.warning(f"⚠️ MACD failed: {result.message}")
            return 0.0, 0.0, 0.0
            
    except Exception as e:
        logger.error(f"❌ MACD optimized error: {e}")
        return 0.0, 0.0, 0.0

# Test rapide du module
if __name__ == "__main__":
    # Test avec des données simulées
    test_prices = pd.Series([100, 101, 102, 101, 103, 104, 102, 105, 106, 104, 107, 108, 106, 109, 110] * 3)
    result = macd_calculator.calculate(test_prices)
    print(f"Test MACD: {result}")
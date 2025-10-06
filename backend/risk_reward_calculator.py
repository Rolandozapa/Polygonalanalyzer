"""
Risk-Reward Calculator Optimisé pour Trading Bot
Méthode "Niveaux Proches" pour calculs rapides et efficaces
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class RiskRewardCalculator:
    """Calculateur Risk-Reward optimisé avec méthode des niveaux proches"""
    
    def __init__(self, atr_period: int = 14, min_rr: float = 1.5):
        self.atr_period = atr_period
        self.min_rr = min_rr
    
    def calculate_rr_levels(self, current_price: float, price_history: List[float], lookback_period: int = 50) -> Tuple[List[float], List[float]]:
        """
        Calcule les 2-3 niveaux de support/résistance LES PLUS PROCHES
        pour déterminer stop-loss et take-profit rapidement
        """
        try:
            # Extraire les highs/lows récents
            recent_data = price_history[-lookback_period:] if len(price_history) >= lookback_period else price_history
            
            if len(recent_data) < 3:
                logger.warning(f"Insufficient price data: {len(recent_data)} points")
                return [current_price * 0.98], [current_price * 1.02]
            
            high = max(recent_data)
            low = min(recent_data)
            
            # Méthode Pivot Points + Niveaux psychologiques
            pivot = (high + low + recent_data[-1]) / 3
            
            # Supports et résistances proches (pour RR)
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high
            
            # Ajouter les highs/lows récents comme niveaux clés
            resistance_levels = sorted(list(set([r1, high, current_price * 1.01, current_price * 1.02])))
            support_levels = sorted(list(set([s1, low, current_price * 0.99, current_price * 0.98])))
            
            # Nettoyer les niveaux trop proches (< 0.5% de différence)
            resistance_levels = self._clean_levels(resistance_levels, current_price, min_distance_pct=0.005)
            support_levels = self._clean_levels(support_levels, current_price, min_distance_pct=0.005)
            
            return support_levels, resistance_levels
            
        except Exception as e:
            logger.error(f"Error calculating RR levels: {e}")
            # Fallback levels
            return [current_price * 0.98], [current_price * 1.02]
    
    def _clean_levels(self, levels: List[float], reference_price: float, min_distance_pct: float = 0.005) -> List[float]:
        """Remove levels too close to each other"""
        if not levels:
            return levels
        
        cleaned = [levels[0]]
        for level in levels[1:]:
            if abs(level - cleaned[-1]) / reference_price > min_distance_pct:
                cleaned.append(level)
        
        return cleaned[:3]  # Keep max 3 levels
    
    def calculate_atr(self, high: List[float], low: List[float], close: List[float]) -> float:
        """Calcule l'Average True Range pour le stop-loss dynamique"""
        try:
            if len(high) < self.atr_period or len(low) < self.atr_period or len(close) < self.atr_period:
                # Fallback pour données insuffisantes
                return np.mean(np.array(high[-5:]) - np.array(low[-5:])) if len(high) >= 5 else 0.01
            
            high_arr = np.array(high[-self.atr_period:])
            low_arr = np.array(low[-self.atr_period:])
            close_arr = np.array(close[-self.atr_period:])
            
            tr1 = high_arr - low_arr
            tr2 = np.abs(high_arr - np.roll(close_arr, 1))
            tr3 = np.abs(low_arr - np.roll(close_arr, 1))
            
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            return float(np.mean(tr[1:]))  # Skip first element due to roll
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0.01  # Fallback ATR
    
    def find_key_levels(self, prices: List[float], current_price: float) -> Tuple[List[float], List[float]]:
        """Trouve les niveaux clés les plus proches"""
        try:
            # Niveaux de prix ronds (psychologiques)
            round_levels = []
            for i in range(3):
                rounded = round(current_price, max(0, 1 - i))
                round_levels.append(rounded)
            
            # Highs/Lows récents
            recent_high = max(prices[-20:]) if len(prices) >= 20 else max(prices)
            recent_low = min(prices[-20:]) if len(prices) >= 20 else min(prices)
            
            # Combinaison des niveaux
            resistances = sorted([recent_high] + [lvl for lvl in round_levels if lvl > current_price])
            supports = sorted([recent_low] + [lvl for lvl in round_levels if lvl < current_price], reverse=True)
            
            return supports, resistances
            
        except Exception as e:
            logger.error(f"Error finding key levels: {e}")
            return [current_price * 0.98], [current_price * 1.02]
    
    def calculate_rr_ratio(self, entry_price: float, direction: str, support_levels: List[float], resistance_levels: List[float]) -> Dict:
        """
        Calcule le risk-reward basé sur les niveaux les plus proches
        """
        try:
            if direction.upper() == "LONG":
                # Stop-loss au support le plus proche sous l'entrée
                below_entry = [s for s in support_levels if s < entry_price]
                stop_loss = max(below_entry) if below_entry else entry_price * 0.99
                
                # Take-profit à la résistance la plus proche au-dessus
                above_entry = [r for r in resistance_levels if r > entry_price]
                take_profit = min(above_entry) if above_entry else entry_price * 1.02
                
            else:  # SHORT
                # Stop-loss à la résistance la plus proche au-dessus
                above_entry = [r for r in resistance_levels if r > entry_price]
                stop_loss = min(above_entry) if above_entry else entry_price * 1.01
                
                # Take-profit au support le plus proche sous l'entrée
                below_entry = [s for s in support_levels if s < entry_price]
                take_profit = max(below_entry) if below_entry else entry_price * 0.98
            
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            
            if risk > 0:
                rr_ratio = reward / risk
            else:
                rr_ratio = 1.0
            
            return {
                'rr_ratio': round(rr_ratio, 2),
                'stop_loss': round(stop_loss, 6),
                'take_profit': round(take_profit, 6),
                'risk': round(risk, 6),
                'reward': round(reward, 6),
                'entry_price': round(entry_price, 6)
            }
            
        except Exception as e:
            logger.error(f"Error calculating RR ratio: {e}")
            return {
                'rr_ratio': 1.0,
                'stop_loss': entry_price * 0.99,
                'take_profit': entry_price * 1.01,
                'risk': entry_price * 0.01,
                'reward': entry_price * 0.01,
                'entry_price': entry_price
            }
    
    def optimal_rr_setup(self, entry_price: float, direction: str, price_history: List[float], 
                        high_history: List[float] = None, low_history: List[float] = None, 
                        close_history: List[float] = None) -> Dict:
        """
        Calcule le setup optimal de risk-reward avec ATR et niveaux multiples
        """
        try:
            # Use price_history for all if individual arrays not provided
            high_hist = high_history or price_history
            low_hist = low_history or price_history  
            close_hist = close_history or price_history
            
            atr = self.calculate_atr(high_hist, low_hist, close_hist)
            supports, resistances = self.find_key_levels(price_history, entry_price)
            
            if direction.upper() == "LONG":
                # Stop-loss basé sur ATR ou support le plus proche
                sl_candidate1 = entry_price - (atr * 1.5)
                sl_candidate2 = supports[0] if supports else entry_price * 0.98
                stop_loss = max(sl_candidate1, sl_candidate2)  # Plus conservateur
                
                # Take-profit multiples (R1, R2, R3)
                tp_levels = []
                for i, res in enumerate(resistances[:3]):  # 3 premières résistances
                    if res > entry_price:
                        rr = (res - entry_price) / (entry_price - stop_loss)
                        if rr >= self.min_rr:
                            tp_levels.append({
                                'price': round(res, 6), 
                                'rr_ratio': round(rr, 2), 
                                'level': i+1
                            })
                
            else:  # SHORT
                # Stop-loss basé sur ATR ou résistance
                sl_candidate1 = entry_price + (atr * 1.5)
                sl_candidate2 = resistances[0] if resistances else entry_price * 1.02
                stop_loss = min(sl_candidate1, sl_candidate2)  # Plus conservateur
                
                # Take-profit multiples
                tp_levels = []
                for i, sup in enumerate(supports[:3]):
                    if sup < entry_price:
                        rr = (entry_price - sup) / (stop_loss - entry_price)
                        if rr >= self.min_rr:
                            tp_levels.append({
                                'price': round(sup, 6), 
                                'rr_ratio': round(rr, 2), 
                                'level': i+1
                            })
            
            # Sélectionner le meilleur TP (RR le plus élevé)
            best_tp = max(tp_levels, key=lambda x: x['rr_ratio']) if tp_levels else None
            
            result = {
                'stop_loss': round(stop_loss, 6),
                'take_profits': tp_levels,
                'best_take_profit': best_tp['price'] if best_tp else entry_price * (1.02 if direction.upper() == "LONG" else 0.98),
                'best_rr_ratio': best_tp['rr_ratio'] if best_tp else 1.0,
                'atr_value': round(atr, 6),
                'direction': direction.upper(),
                'supports': supports,
                'resistances': resistances
            }
            
            logger.info(f"🎯 RR Setup {direction}: Entry={entry_price:.6f}, SL={result['stop_loss']:.6f}, TP={result['best_take_profit']:.6f}, RR={result['best_rr_ratio']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in optimal_rr_setup: {e}")
            # Fallback simple
            return {
                'stop_loss': entry_price * (0.99 if direction.upper() == "LONG" else 1.01),
                'best_take_profit': entry_price * (1.02 if direction.upper() == "LONG" else 0.98),
                'best_rr_ratio': 1.0,
                'direction': direction.upper()
            }

# Configuration pour différents types de trading
DAY_TRADING_CONFIG = {
    'atr_period': 10,
    'min_rr': 1.2,
    'atr_multiplier': 1.5,
    'lookback': 20
}

SWING_TRADING_CONFIG = {
    'atr_period': 14,
    'min_rr': 2.0,
    'atr_multiplier': 2.0,
    'lookback': 50
}

def create_rr_calculator(trading_style: str = 'swing') -> RiskRewardCalculator:
    """Factory function pour créer le calculateur selon le style de trading"""
    if trading_style.lower() == 'day':
        config = DAY_TRADING_CONFIG
    else:
        config = SWING_TRADING_CONFIG
    
    return RiskRewardCalculator(
        atr_period=config['atr_period'],
        min_rr=config['min_rr']
    )
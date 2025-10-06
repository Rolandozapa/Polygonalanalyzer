"""
POSITION SIZING CALCULATOR
ML-based position sizing with regime, momentum, and Bollinger multipliers
From v4 advanced_technical_indicators
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PositionSizingCalculator:
    """
    Calculate optimal position size based on ML regime detection
    Formula: capital × base_risk × regime_mult × ml_confidence_mult × momentum_mult × bb_mult
    """
    
    def __init__(self, base_capital: float = 10000.0, base_risk_pct: float = 1.0):
        """
        Args:
            base_capital: Total trading capital in dollars
            base_risk_pct: Base risk per trade as percentage (default 1%)
        """
        self.base_capital = base_capital
        self.base_risk_pct = base_risk_pct
        logger.info(f"Position Sizing Calculator initialized: ${base_capital:,.0f} capital, {base_risk_pct}% base risk")
    
    def calculate_position_size(self, 
                               regime_info: Dict,
                               entry_price: float,
                               stop_loss_price: float,
                               capital: Optional[float] = None) -> Dict:
        """
        Calculate optimal position size with all multipliers
        
        Args:
            regime_info: Complete regime detection output
            entry_price: Entry price
            stop_loss_price: Stop loss price
            capital: Override capital (optional)
        
        Returns:
            Dict with position size, risk, multipliers
        """
        
        # Use provided capital or default
        capital = capital or self.base_capital
        
        # Base risk in dollars
        risk_per_trade = capital * (self.base_risk_pct / 100)
        
        # Get all multipliers
        regime_mult = regime_info.get('regime_multiplier', 1.0)
        ml_confidence_mult = regime_info.get('ml_confidence_multiplier', 1.0)
        momentum_mult = self._calculate_momentum_multiplier(regime_info.get('indicators', {}))
        bb_mult = self._calculate_bb_multiplier(regime_info)
        
        # Combined multiplier
        combined_multiplier = regime_mult * ml_confidence_mult * momentum_mult * bb_mult
        
        # Adjusted risk
        adjusted_risk = risk_per_trade * combined_multiplier
        
        # Position size calculation
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            logger.warning("Zero price risk detected, returning zero position size")
            return self._get_zero_position()
        
        position_size_units = adjusted_risk / price_risk
        position_size_dollars = position_size_units * entry_price
        position_size_pct = (position_size_dollars / capital) * 100
        
        # Risk-reward calculation
        rr_ratio = self._calculate_risk_reward(entry_price, stop_loss_price, regime_info)
        
        return {
            'position_size_units': round(position_size_units, 4),
            'position_size_dollars': round(position_size_dollars, 2),
            'position_size_pct': round(position_size_pct, 2),
            'risk_dollars': round(adjusted_risk, 2),
            'risk_pct': round((adjusted_risk / capital) * 100, 2),
            'regime_multiplier': round(regime_mult, 2),
            'ml_confidence_multiplier': round(ml_confidence_mult, 2),
            'momentum_multiplier': round(momentum_mult, 2),
            'bb_multiplier': round(bb_mult, 2),
            'combined_multiplier': round(combined_multiplier, 2),
            'price_risk': round(price_risk, 4),
            'risk_reward_ratio': round(rr_ratio, 2),
            'recommendation': self._get_recommendation(combined_multiplier, rr_ratio)
        }
    
    def _calculate_momentum_multiplier(self, indicators: Dict) -> float:
        """
        Calculate momentum multiplier based on RSI, MACD, and BB quality
        Returns: 0.6 to 1.3
        """
        
        # RSI Quality
        rsi = indicators.get('rsi', 50)
        rsi_trend = indicators.get('rsi_trend', 0)
        
        if 45 <= rsi <= 60 and rsi_trend > 0:
            rsi_quality = 1.0
        elif 40 <= rsi <= 65:
            rsi_quality = 0.8
        elif 30 <= rsi <= 70:
            rsi_quality = 0.6
        elif 70 <= rsi <= 80 and rsi_trend > 0:
            rsi_quality = 0.4
        else:
            rsi_quality = 0.2
        
        # MACD Quality
        macd_hist = indicators.get('macd_histogram', 0)
        macd_trend = indicators.get('macd_trend', 0)
        
        if macd_hist > 0 and macd_trend > 0:
            macd_quality = 1.0
        elif macd_hist > 0:
            macd_quality = 0.8
        elif abs(macd_hist) < 0.1:
            macd_quality = 0.6
        elif macd_hist < 0 and macd_trend > 0:
            macd_quality = 0.4
        else:
            macd_quality = 0.2
        
        # BB Quality
        bb_squeeze = indicators.get('bb_squeeze', False)
        bb_width = indicators.get('bb_width', 0.03)
        
        if bb_squeeze and bb_width < 0.015:
            bb_quality = 1.0  # Extreme squeeze
        elif bb_squeeze:
            bb_quality = 0.8  # Tight squeeze
        elif bb_width < 0.025:
            bb_quality = 0.6
        elif bb_width < 0.05:
            bb_quality = 0.4
        else:
            bb_quality = 0.2
        
        # Weighted average
        momentum_mult = 0.33 * (rsi_quality + macd_quality + bb_quality)
        
        # Normalize between 0.6 and 1.3
        return 0.6 + (momentum_mult * 0.7)
    
    def _calculate_bb_multiplier(self, regime_info: Dict) -> float:
        """
        Calculate Bollinger Bands multiplier
        Squeeze setups get higher multipliers
        Returns: 0.8 to 1.4
        """
        
        indicators = regime_info.get('indicators', {})
        bb_squeeze = indicators.get('bb_squeeze', False)
        bb_width = indicators.get('bb_width', 0.03)
        confidence = regime_info.get('confidence', 0.5)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        persistence = regime_info.get('regime_persistence', 0)
        sma_20_slope = indicators.get('sma_20_slope', 0)
        above_sma_20 = indicators.get('above_sma_20', False)
        
        # ML_EXTREME_SQUEEZE
        if bb_squeeze and bb_width < 0.015 and confidence > 0.85 and volume_ratio > 2.0:
            return 1.4
        
        # ML_TIGHT_SQUEEZE
        if bb_squeeze and confidence > 0.7:
            return 1.2
        
        # BAND_WALK_ML_CONFIRMED (trending with fresh regime)
        if abs(sma_20_slope) > 0.003 and above_sma_20 and persistence < 20:
            return 1.1
        
        # Normal BB conditions
        if 0.02 <= bb_width <= 0.04:
            return 1.0
        
        # Wide bands (high volatility)
        if bb_width > 0.06:
            return 0.8
        
        return 1.0
    
    def _calculate_risk_reward(self, entry_price: float, stop_loss_price: float, 
                              regime_info: Dict) -> float:
        """Calculate risk-reward ratio"""
        
        indicators = regime_info.get('indicators', {})
        
        # Get target based on regime
        regime = regime_info.get('regime', 'RANGING_TIGHT')
        
        # Estimate target multiplier based on regime
        if 'STRONG' in regime:
            target_mult = 3.0
        elif 'BREAKOUT' in regime:
            target_mult = 2.5
        elif 'MODERATE' in regime:
            target_mult = 2.0
        else:
            target_mult = 1.5
        
        risk = abs(entry_price - stop_loss_price)
        reward = risk * target_mult
        
        return reward / risk if risk > 0 else 0
    
    def _get_recommendation(self, combined_multiplier: float, rr_ratio: float) -> str:
        """Generate position sizing recommendation"""
        
        if combined_multiplier > 1.5 and rr_ratio > 2.5:
            return "STRONG_POSITION - Excellent setup with high multipliers"
        elif combined_multiplier > 1.2 and rr_ratio > 2.0:
            return "GOOD_POSITION - Above average setup"
        elif combined_multiplier > 0.8 and rr_ratio > 1.5:
            return "MODERATE_POSITION - Standard setup"
        elif combined_multiplier > 0.5:
            return "REDUCED_POSITION - Below average setup"
        else:
            return "MINIMAL_POSITION - Poor setup, consider waiting"
    
    def _get_zero_position(self) -> Dict:
        """Return zero position structure"""
        return {
            'position_size_units': 0,
            'position_size_dollars': 0,
            'position_size_pct': 0,
            'risk_dollars': 0,
            'risk_pct': 0,
            'regime_multiplier': 0,
            'ml_confidence_multiplier': 0,
            'momentum_multiplier': 0,
            'bb_multiplier': 0,
            'combined_multiplier': 0,
            'price_risk': 0,
            'risk_reward_ratio': 0,
            'recommendation': 'ZERO_POSITION - Invalid parameters'
        }


# Global instance
position_sizing_calculator = PositionSizingCalculator()

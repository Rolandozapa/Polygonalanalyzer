"""
🎯 ENHANCED MARKET CONDITION SCORING SYSTEM
Advanced scoring logic that adjusts IA1 base confidence using market conditions,
liquidity factors, and token-specific metrics.
"""
import math
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

class EnhancedMarketScoring:
    """
    Enhanced market condition scoring system that adjusts IA1 base confidence
    using multiple market factors and token-specific metrics.
    """
    
    def __init__(self):
        # Default amplitude and market cap multipliers
        self.default_amplitude = 20.0
        self.mc_multipliers = {
            'micro': 0.8,    # < $100M
            'small': 0.9,    # $100M - $1B  
            'mid': 1.0,      # $1B - $10B
            'large': 1.1,    # $10B - $100B
            'mega': 1.2      # > $100B
        }
        
        # Default weights for various market factors
        self.default_weights = {
            'var_cap': 0.20,        # Market cap variation 24h
            'var_vol': 0.25,        # Volume variation 24h
            'fg_index': 0.15,       # Fear & Greed contrarian factor
            'volcap_ratio': 0.20,   # Volume/Cap ratio (liquidity)
            'var_price': 0.15,      # Price volatility penalty
            'trend_strength': 0.05  # Trend continuation factor
        }
    
    @staticmethod
    def tanh_norm(x: float, s: float = 1.0) -> float:
        """Normalize x to [-1,1] via tanh with sensitivity s."""
        return math.tanh(x / s)
    
    @staticmethod
    def sigmoid_norm(x: float, s: float = 1.0) -> float:
        """Normalize x to [-1,1] via modified sigmoid."""
        return 2.0 / (1.0 + math.exp(-x / s)) - 1.0
    
    @staticmethod
    def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
        """Clamp value between lo and hi."""
        return max(lo, min(hi, x))
    
    def get_market_cap_bucket(self, market_cap: float) -> str:
        """Determine market cap bucket for multiplier calculation."""
        if market_cap < 100_000_000:  # < $100M
            return 'micro'
        elif market_cap < 1_000_000_000:  # < $1B
            return 'small'
        elif market_cap < 10_000_000_000:  # < $10B
            return 'mid'
        elif market_cap < 100_000_000_000:  # < $100B
            return 'large'
        else:
            return 'mega'
    
    def create_normalization_functions(self) -> Dict[str, Callable]:
        """Create normalization functions for each market factor."""
        return {
            'var_cap': lambda x: self.tanh_norm(x, s=5.0),  # Market cap change sensitivity
            'var_vol': lambda x: self.tanh_norm(x, s=30.0),  # Volume change sensitivity
            'fg_index': lambda x: ((50.0 - x) / 50.0) * -1.0,  # Contrarian F&G (50=neutral)
            'volcap_ratio': lambda x: self.tanh_norm((x - 0.02) * 25, s=5.0),  # Vol/Cap ratio
            'var_price': lambda x: -self.tanh_norm(x, s=10.0),  # Penalize high volatility
            'trend_strength': lambda x: self.tanh_norm((x - 0.5) * 2, s=1.0)  # Trend strength bonus
        }
    
    def extract_market_factors(self, opportunity: Any, analysis: Any) -> Dict[str, float]:
        """
        Extract market factors from MarketOpportunity and TechnicalAnalysis objects.
        
        Args:
            opportunity: MarketOpportunity object with market data
            analysis: TechnicalAnalysis object with technical indicators
            
        Returns:
            Dict with market factor values
        """
        try:
            # Extract basic market data
            current_price = getattr(opportunity, 'current_price', 0.0)
            volume_24h = getattr(opportunity, 'volume_24h', 0.0)
            market_cap = getattr(opportunity, 'market_cap', current_price * 1_000_000)  # Fallback estimate
            price_change_24h = getattr(opportunity, 'price_change_24h', 0.0)
            
            # Calculate derived metrics
            volcap_ratio = (volume_24h / market_cap) if market_cap > 0 else 0.0
            
            # Extract technical strength from analysis
            trend_strength = 0.5  # Default neutral
            if hasattr(analysis, 'trend_strength_score'):
                trend_strength = getattr(analysis, 'trend_strength_score', 50) / 100.0
            elif hasattr(analysis, 'analysis_confidence'):
                trend_strength = getattr(analysis, 'analysis_confidence', 0.5)
            
            # Estimate volume change (simplified - in production, use historical comparison)
            var_vol = min(abs(price_change_24h) * 2, 50.0)  # Rough correlation estimate
            
            # Fear & Greed Index (simplified - in production, fetch from API)
            # For now, derive from price action and volatility
            volatility = abs(price_change_24h)
            if volatility > 10:
                fg_index = 25  # High volatility = fear
            elif volatility > 5:
                fg_index = 40  # Moderate volatility = neutral-fear
            elif price_change_24h > 3:
                fg_index = 65  # Strong positive = greed
            else:
                fg_index = 50  # Neutral
            
            return {
                'var_cap': abs(price_change_24h),  # Using price change as cap change proxy
                'var_vol': var_vol,
                'fg_index': fg_index,
                'volcap_ratio': volcap_ratio,
                'var_price': volatility,
                'trend_strength': trend_strength
            }
            
        except Exception as e:
            logger.warning(f"Error extracting market factors: {e}")
            # Return neutral values
            return {
                'var_cap': 0.0,
                'var_vol': 0.0,
                'fg_index': 50.0,
                'volcap_ratio': 0.02,
                'var_price': 0.0,
                'trend_strength': 0.5
            }
    
    def compute_enhanced_score(self, 
                             note_base: float,
                             factor_scores: Dict[str, float],
                             norm_funcs: Optional[Dict[str, Callable]] = None,
                             weights: Optional[Dict[str, float]] = None,
                             amplitude: float = 20.0,
                             mc_mult: float = 1.0) -> Dict[str, Any]:
        """
        Compute enhanced score using market condition factors.
        
        Args:
            note_base: Base confidence score (0-100)
            factor_scores: Raw market factor values
            norm_funcs: Normalization functions for each factor
            weights: Weights for each factor
            amplitude: Maximum adjustment points
            mc_mult: Market cap multiplier
            
        Returns:
            Dict with scoring breakdown and final score
        """
        if not norm_funcs:
            norm_funcs = self.create_normalization_functions()
        
        if not weights:
            weights = self.default_weights.copy()
        
        # 1) Normalize each factor to [-1,1]
        normalized = {}
        for k, raw in factor_scores.items():
            norm_fn = norm_funcs.get(k, self.tanh_norm)  # Fallback to tanh
            normalized[k] = norm_fn(raw)
        
        # 2) Calculate weighted sum (normalize weights if needed)
        total_weight = sum(weights.values()) if weights else 1.0
        weighted_sum = 0.0
        
        for k, w in weights.items():
            score = normalized.get(k, 0.0)
            weighted_sum += (w / total_weight) * score
        
        # 3) Apply final adjustment
        adjustment = weighted_sum * amplitude * mc_mult
        note_final = self.clamp(note_base + adjustment, 0.0, 100.0)
        
        return {
            'note_base': note_base,
            'factor_scores': factor_scores,
            'normalized': normalized,
            'weighted_sum': weighted_sum,
            'adjustment': adjustment,
            'mc_mult': mc_mult,
            'note_final': note_final,
            'improvement': note_final - note_base
        }
    
    def enhance_ia1_confidence(self, 
                              base_confidence: float,
                              opportunity: Any,
                              analysis: Any,
                              custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Main method to enhance IA1 confidence using market conditions.
        
        Args:
            base_confidence: Original IA1 confidence (0.0-1.0)
            opportunity: MarketOpportunity object
            analysis: TechnicalAnalysis object
            custom_weights: Optional custom weights for factors
            
        Returns:
            Enhanced scoring result
        """
        try:
            # Convert confidence to 0-100 scale
            note_base = base_confidence * 100.0
            
            # Extract market factors
            factor_scores = self.extract_market_factors(opportunity, analysis)
            
            # Determine market cap multiplier
            market_cap = getattr(opportunity, 'market_cap', 0.0)
            if market_cap == 0.0:
                # Estimate market cap from price and assume reasonable supply
                current_price = getattr(opportunity, 'current_price', 1.0)
                market_cap = current_price * 100_000_000  # 100M token estimate
            
            mc_bucket = self.get_market_cap_bucket(market_cap)
            mc_mult = self.mc_multipliers.get(mc_bucket, 1.0)
            
            # Use custom weights if provided
            weights = custom_weights if custom_weights else self.default_weights
            
            # Compute enhanced score
            result = self.compute_enhanced_score(
                note_base=note_base,
                factor_scores=factor_scores,
                weights=weights,
                amplitude=self.default_amplitude,
                mc_mult=mc_mult
            )
            
            # Add metadata
            result['mc_bucket'] = mc_bucket
            result['symbol'] = getattr(opportunity, 'symbol', 'UNKNOWN')
            result['enhanced_confidence'] = result['note_final'] / 100.0  # Back to 0-1 scale
            
            # Log the enhancement
            improvement = result['improvement']
            symbol = result['symbol']
            logger.info(f"🎯 Enhanced scoring for {symbol}: "
                       f"{base_confidence:.1%} → {result['enhanced_confidence']:.1%} "
                       f"({improvement:+.1f} points, {mc_bucket} cap)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced scoring: {e}")
            # Return original confidence if enhancement fails
            return {
                'note_base': base_confidence * 100.0,
                'note_final': base_confidence * 100.0,
                'enhanced_confidence': base_confidence,
                'improvement': 0.0,
                'error': str(e)
            }
    
    def create_market_condition_summary(self, scoring_result: Dict[str, Any]) -> str:
        """
        Create a human-readable summary of market condition adjustments.
        
        Args:
            scoring_result: Result from enhance_ia1_confidence
            
        Returns:
            Formatted summary string
        """
        try:
            symbol = scoring_result.get('symbol', 'UNKNOWN')
            improvement = scoring_result.get('improvement', 0.0)
            mc_bucket = scoring_result.get('mc_bucket', 'unknown')
            factor_scores = scoring_result.get('factor_scores', {})
            normalized = scoring_result.get('normalized', {})
            
            # Determine primary factors
            positive_factors = []
            negative_factors = []
            
            for factor, norm_score in normalized.items():
                if norm_score > 0.1:
                    positive_factors.append(f"{factor}(+{norm_score:.2f})")
                elif norm_score < -0.1:
                    negative_factors.append(f"{factor}({norm_score:.2f})")
            
            summary = f"📊 **{symbol} Market Condition Analysis:**\n"
            summary += f"   • Base → Enhanced: {improvement:+.1f} points\n"
            summary += f"   • Market Cap: {mc_bucket.title()} cap bucket\n"
            
            if positive_factors:
                summary += f"   • ✅ Positive factors: {', '.join(positive_factors)}\n"
            
            if negative_factors:
                summary += f"   • ⚠️ Negative factors: {', '.join(negative_factors)}\n"
            
            # Key metrics
            volcap = factor_scores.get('volcap_ratio', 0.0)
            fg_index = factor_scores.get('fg_index', 50.0)
            var_price = factor_scores.get('var_price', 0.0)
            
            summary += f"   • 💧 Liquidity (Vol/Cap): {volcap:.4f}\n"
            summary += f"   • 😱 Fear/Greed: {fg_index:.0f}/100\n"
            summary += f"   • 📈 Price Volatility: {var_price:.1f}%\n"
            
            return summary
            
        except Exception as e:
            return f"📊 Market condition summary error: {e}"

# Global instance
enhanced_market_scoring = EnhancedMarketScoring()
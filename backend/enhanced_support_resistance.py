"""
ENHANCED SUPPORT & RESISTANCE CALCULATOR
Multi-timeframe detection, clustering, volume validation, and context-aware RR calculation
Based on professional trading techniques with ADX, RSI, and volatility integration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SupportResistanceLevel:
    """Individual support or resistance level with metadata"""
    price: float
    strength: float  # 0-1 score based on touches and volume
    touches: int
    type: str  # "support" or "resistance"
    timeframe: str  # "short", "medium", "long"
    last_touch_age: int  # Bars since last touch
    volume_confirmed: bool


@dataclass
class EnhancedRRAnalysis:
    """Complete RR analysis with context"""
    entry_price: float
    stop_loss: float
    take_profit: float
    risk: float
    reward: float
    rr_ratio: float
    confidence_score: float  # 0-100%
    min_rr_required: float
    trade_valid: bool
    market_context: Dict
    indicators: Dict
    atr_multiple_used: float
    recommendation: str
    reasoning: str


class EnhancedSupportResistanceCalculator:
    """
    Advanced S/R calculator with multi-timeframe detection, clustering, and context awareness
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'timeframes': [3, 8, 21],  # Short, medium, long term windows
            'max_clusters': 5,
            'min_touches': 2,
            'touch_tolerance': 0.005,  # 0.5% tolerance
            'decay_factor': 0.9,  # Temporal decay for recent levels
            'atr_period': 14,
            'adx_period': 14,
            'rsi_period': 14,
            'default_atr_multiple': 2.0,
            'trend_strength_threshold': 25,  # ADX threshold for strong trend
            'overbought_threshold': 70,
            'oversold_threshold': 30,
            'confidence_weights': {
                'trend_strength': 20,
                'trend_alignment': 15,
                'momentum': 10,
                'volatility': 5
            }
        }
        logger.info("Enhanced S/R Calculator initialized with multi-timeframe detection")
    
    def detect_multi_timeframe_swings(self, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """
        Detect swing highs and lows across multiple timeframes
        Returns: (all_supports, all_resistances)
        """
        all_supports = []
        all_resistances = []
        
        current_price = df['close'].iloc[-1] if 'close' in df.columns else df['Close'].iloc[-1]
        
        for window in self.config['timeframes']:
            try:
                # Use appropriate column names (handle both lowercase and uppercase)
                high_col = 'high' if 'high' in df.columns else 'High'
                low_col = 'low' if 'low' in df.columns else 'Low'
                
                # Rolling window to find local extremums
                highs = df[high_col].rolling(window=window, center=True).max()
                lows = df[low_col].rolling(window=window, center=True).min()
                
                # Find peaks (resistance) - where high equals rolling max
                for i in range(window, len(df) - window):
                    if df[high_col].iloc[i] == highs.iloc[i] and df[high_col].iloc[i] > current_price:
                        all_resistances.append({
                            'price': float(df[high_col].iloc[i]),
                            'index': i,
                            'timeframe': self._get_timeframe_name(window),
                            'age': len(df) - i
                        })
                
                # Find troughs (support) - where low equals rolling min
                for i in range(window, len(df) - window):
                    if df[low_col].iloc[i] == lows.iloc[i] and df[low_col].iloc[i] < current_price:
                        all_supports.append({
                            'price': float(df[low_col].iloc[i]),
                            'index': i,
                            'timeframe': self._get_timeframe_name(window),
                            'age': len(df) - i
                        })
                        
            except Exception as e:
                logger.warning(f"Error detecting swings for window {window}: {e}")
                continue
        
        logger.info(f"Multi-timeframe detection: {len(all_supports)} supports, {len(all_resistances)} resistances")
        return all_supports, all_resistances
    
    def _get_timeframe_name(self, window: int) -> str:
        """Convert window size to timeframe name"""
        if window <= 5:
            return "short"
        elif window <= 15:
            return "medium"
        else:
            return "long"
    
    def weighted_clustering(self, levels: List[Dict], current_price: float) -> List[float]:
        """
        Cluster nearby levels with temporal weighting (recent levels matter more)
        """
        if not levels:
            return []
        
        # Extract prices and apply temporal decay weighting
        weighted_prices = []
        for level in levels:
            age_weight = self.config['decay_factor'] ** level['age']
            # Add price multiple times based on weight (simulates weight in clustering)
            count = max(1, int(age_weight * 10))
            weighted_prices.extend([level['price']] * count)
        
        if not weighted_prices:
            return []
        
        # Use simple clustering (grouping nearby prices)
        unique_prices = sorted(set(weighted_prices))
        if len(unique_prices) <= 1:
            return unique_prices
        
        # Dynamic clustering based on price range
        price_range = max(unique_prices) - min(unique_prices)
        tolerance = max(current_price * 0.01, price_range * 0.05)  # 1% or 5% of range
        
        clusters = []
        current_cluster = [unique_prices[0]]
        
        for price in unique_prices[1:]:
            if price - current_cluster[-1] <= tolerance:
                current_cluster.append(price)
            else:
                # Calculate cluster center (weighted average)
                cluster_center = np.mean(current_cluster)
                clusters.append(cluster_center)
                current_cluster = [price]
        
        # Don't forget last cluster
        if current_cluster:
            clusters.append(np.mean(current_cluster))
        
        # Limit to max clusters (keep strongest/most recent)
        if len(clusters) > self.config['max_clusters']:
            # Calculate distance to current price and prefer closer levels
            clusters_with_distance = [(c, abs(c - current_price)) for c in clusters]
            clusters_with_distance.sort(key=lambda x: x[1])
            clusters = [c[0] for c in clusters_with_distance[:self.config['max_clusters']]]
        
        logger.info(f"Clustered {len(weighted_prices)} levels into {len(clusters)} strong levels")
        return sorted(clusters)
    
    def validate_with_volume(self, levels: List[float], df: pd.DataFrame) -> List[SupportResistanceLevel]:
        """
        Validate S/R levels by checking how many times they were tested with volume
        """
        validated_levels = []
        
        close_col = 'close' if 'close' in df.columns else 'Close'
        volume_col = 'volume' if 'volume' in df.columns else 'Volume'
        
        prices = df[close_col].values
        volumes = df[volume_col].values if volume_col in df.columns else np.ones(len(df))
        avg_volume = np.mean(volumes)
        
        for level in levels:
            tolerance = level * self.config['touch_tolerance']
            touches = 0
            high_volume_touches = 0
            last_touch_age = len(df)
            
            # Count touches of this level
            for i in range(len(prices)):
                if abs(prices[i] - level) <= tolerance:
                    touches += 1
                    last_touch_age = min(last_touch_age, len(df) - i)
                    
                    # Check if touch had significant volume
                    if volumes[i] > avg_volume * 1.2:
                        high_volume_touches += 1
            
            # Determine if level is valid
            if touches >= self.config['min_touches']:
                # Calculate strength score
                strength = min(1.0, (touches / 5.0) * 0.6 + (high_volume_touches / max(1, touches)) * 0.4)
                
                # Determine type
                current_price = prices[-1]
                level_type = "resistance" if level > current_price else "support"
                
                validated_level = SupportResistanceLevel(
                    price=level,
                    strength=strength,
                    touches=touches,
                    type=level_type,
                    timeframe="mixed",
                    last_touch_age=last_touch_age,
                    volume_confirmed=high_volume_touches > 0
                )
                validated_levels.append(validated_level)
        
        logger.info(f"Volume validation: {len(validated_levels)}/{len(levels)} levels confirmed")
        return validated_levels
    
    def calculate_enhanced_support_resistance(self, df: pd.DataFrame) -> Dict:
        """
        Main method: Calculate enhanced S/R levels with multi-timeframe + clustering + volume
        """
        try:
            current_price = df['close'].iloc[-1] if 'close' in df.columns else df['Close'].iloc[-1]
            
            # Step 1: Multi-timeframe detection
            support_data, resistance_data = self.detect_multi_timeframe_swings(df)
            
            # Step 2: Weighted clustering
            clustered_supports = self.weighted_clustering(support_data, current_price)
            clustered_resistances = self.weighted_clustering(resistance_data, current_price)
            
            # Step 3: Volume validation
            validated_supports = self.validate_with_volume(clustered_supports, df)
            validated_resistances = self.validate_with_volume(clustered_resistances, df)
            
            # Sort by strength and proximity to current price
            validated_supports.sort(key=lambda x: (x.strength, -abs(x.price - current_price)), reverse=True)
            validated_resistances.sort(key=lambda x: (x.strength, -abs(x.price - current_price)), reverse=True)
            
            return {
                'supports': validated_supports[:5],  # Top 5 supports
                'resistances': validated_resistances[:5],  # Top 5 resistances
                'current_price': current_price,
                'primary_support': validated_supports[0].price if validated_supports else current_price * 0.95,
                'primary_resistance': validated_resistances[0].price if validated_resistances else current_price * 1.05,
                'support_strength': validated_supports[0].strength if validated_supports else 0.5,
                'resistance_strength': validated_resistances[0].strength if validated_resistances else 0.5
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced S/R calculation: {e}")
            # Fallback to simple levels
            current_price = df['close'].iloc[-1] if 'close' in df.columns else df['Close'].iloc[-1]
            return {
                'supports': [],
                'resistances': [],
                'current_price': current_price,
                'primary_support': current_price * 0.95,
                'primary_resistance': current_price * 1.05,
                'support_strength': 0.5,
                'resistance_strength': 0.5
            }
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators for context-aware RR"""
        try:
            high_col = 'high' if 'high' in df.columns else 'High'
            low_col = 'low' if 'low' in df.columns else 'Low'
            close_col = 'close' if 'close' in df.columns else 'Close'
            
            high = df[high_col].values
            low = df[low_col].values
            close = df[close_col].values
            
            # ATR (Average True Range)
            atr = self._calculate_atr(df, self.config['atr_period'])
            
            # ADX (Average Directional Index) - Trend strength
            adx, plus_di, minus_di = self._calculate_adx(df, self.config['adx_period'])
            
            # RSI (Relative Strength Index)
            rsi = self._calculate_rsi(close, self.config['rsi_period'])
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close, 20, 2)
            
            return {
                'atr': atr,
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'rsi': rsi,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'bb_width': (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {
                'atr': 0, 'adx': 0, 'plus_di': 0, 'minus_di': 0,
                'rsi': 50, 'bb_upper': 0, 'bb_middle': 0, 'bb_lower': 0, 'bb_width': 0
            }
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> float:
        """Calculate Average True Range"""
        high_col = 'high' if 'high' in df.columns else 'High'
        low_col = 'low' if 'low' in df.columns else 'Low'
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        high = df[high_col]
        low = df[low_col]
        close = df[close_col]
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return float(atr) if not pd.isna(atr) else 0.0
    
    def _calculate_adx(self, df: pd.DataFrame, period: int) -> Tuple[float, float, float]:
        """Calculate ADX and directional indicators"""
        high_col = 'high' if 'high' in df.columns else 'High'
        low_col = 'low' if 'low' in df.columns else 'Low'
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        high = df[high_col]
        low = df[low_col]
        close = df[close_col]
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        
        # Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        dm_plus[dm_plus < 0] = 0
        dm_minus[dm_minus < 0] = 0
        dm_plus[(dm_plus < dm_minus)] = 0
        dm_minus[(dm_minus < dm_plus)] = 0
        
        # Smoothed DM
        di_plus = 100 * dm_plus.ewm(span=period, adjust=False).mean() / atr
        di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr
        
        # DX and ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return (float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0,
                float(di_plus.iloc[-1]) if not pd.isna(di_plus.iloc[-1]) else 0.0,
                float(di_minus.iloc[-1]) if not pd.isna(di_minus.iloc[-1]) else 0.0)
    
    def _calculate_rsi(self, close: pd.Series, period: int) -> float:
        """Calculate RSI"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_bollinger_bands(self, close: pd.Series, period: int, std_dev: float) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        middle = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return (float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else close.iloc[-1],
                float(middle.iloc[-1]) if not pd.isna(middle.iloc[-1]) else close.iloc[-1],
                float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else close.iloc[-1])
    
    def analyze_market_context(self, indicators: Dict, direction: str) -> Dict:
        """Analyze market context for context-aware decisions"""
        context = {
            'trend_strength': 'weak',
            'trend_direction': 'neutral',
            'momentum': 'neutral',
            'volatility': 'normal',
            'market_condition': 'range'
        }
        
        # Trend strength (ADX)
        if indicators['adx'] > self.config['trend_strength_threshold']:
            context['trend_strength'] = 'strong'
        elif indicators['adx'] > 20:
            context['trend_strength'] = 'medium'
        
        # Trend direction (DI+ vs DI-)
        if indicators['plus_di'] > indicators['minus_di']:
            context['trend_direction'] = 'bullish'
        else:
            context['trend_direction'] = 'bearish'
        
        # Momentum (RSI)
        if indicators['rsi'] > self.config['overbought_threshold']:
            context['momentum'] = 'overbought'
        elif indicators['rsi'] < self.config['oversold_threshold']:
            context['momentum'] = 'oversold'
        
        # Volatility (Bollinger Band width)
        if indicators['bb_width'] > 0.05:
            context['volatility'] = 'high'
        elif indicators['bb_width'] < 0.02:
            context['volatility'] = 'low'
        
        # Market condition
        if context['trend_strength'] == 'strong':
            context['market_condition'] = 'trending'
        elif context['volatility'] == 'high':
            context['market_condition'] = 'volatile'
        
        return context
    
    def calculate_confidence_score(self, context: Dict, direction: str) -> float:
        """Calculate trade confidence score (0-100%)"""
        score = 50.0  # Base score
        
        weights = self.config['confidence_weights']
        
        # Trend strength bonus
        if context['trend_strength'] == 'strong':
            score += weights['trend_strength']
        elif context['trend_strength'] == 'medium':
            score += weights['trend_strength'] / 2
        
        # Trend alignment bonus
        if (direction == 'long' and context['trend_direction'] == 'bullish') or \
           (direction == 'short' and context['trend_direction'] == 'bearish'):
            score += weights['trend_alignment']
        
        # Momentum bonus
        if (direction == 'long' and context['momentum'] == 'oversold') or \
           (direction == 'short' and context['momentum'] == 'overbought'):
            score += weights['momentum']
        elif context['momentum'] == 'neutral':
            score += weights['momentum'] / 2
        
        # Volatility adjustment
        if context['volatility'] == 'normal':
            score += weights['volatility']
        elif context['volatility'] == 'high' and context['trend_strength'] == 'strong':
            score += weights['volatility'] * 2
        
        return min(100.0, max(0.0, score))
    
    def calculate_enhanced_rr(self, df: pd.DataFrame, direction: str, 
                            entry_price: Optional[float] = None,
                            sr_levels: Optional[Dict] = None) -> EnhancedRRAnalysis:
        """
        Calculate context-aware Risk-Reward with ADX, RSI, and dynamic adjustments
        """
        if entry_price is None:
            entry_price = df['close'].iloc[-1] if 'close' in df.columns else df['Close'].iloc[-1]
        
        # Get S/R levels if not provided
        if sr_levels is None:
            sr_levels = self.calculate_enhanced_support_resistance(df)
        
        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(df)
        
        # Analyze market context
        context = self.analyze_market_context(indicators, direction)
        
        # Calculate confidence score
        confidence_score = self.calculate_confidence_score(context, direction)
        
        # Dynamic ATR multiplier based on context
        atr_multiple = self._calculate_dynamic_atr_multiple(context, indicators)
        
        # Calculate stop-loss and take-profit
        stop_loss, take_profit = self._calculate_dynamic_sl_tp(
            entry_price, direction, indicators, context, sr_levels, atr_multiple
        )
        
        # Calculate risk and reward
        if direction == 'long':
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        
        rr_ratio = reward / risk if risk > 0 else 0.0
        
        # Dynamic minimum RR based on confidence
        min_rr_required = self._calculate_dynamic_min_rr(confidence_score)
        
        # Validate trade
        trade_valid = rr_ratio >= min_rr_required and confidence_score >= 40
        
        # Generate recommendation
        recommendation = self._generate_recommendation(confidence_score, rr_ratio, context, direction)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(context, indicators, confidence_score, rr_ratio, direction)
        
        return EnhancedRRAnalysis(
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk=risk,
            reward=reward,
            rr_ratio=rr_ratio,
            confidence_score=confidence_score,
            min_rr_required=min_rr_required,
            trade_valid=trade_valid,
            market_context=context,
            indicators=indicators,
            atr_multiple_used=atr_multiple,
            recommendation=recommendation,
            reasoning=reasoning
        )
    
    def _calculate_dynamic_atr_multiple(self, context: Dict, indicators: Dict) -> float:
        """Calculate dynamic ATR multiplier based on market conditions"""
        base_multiple = self.config['default_atr_multiple']
        
        # Strong trend: wider targets
        if context['trend_strength'] == 'strong':
            base_multiple *= 1.5
        elif context['trend_strength'] == 'weak':
            base_multiple *= 0.8
        
        # High volatility: wider targets
        if context['volatility'] == 'high':
            base_multiple *= 1.2
        elif context['volatility'] == 'low':
            base_multiple *= 0.9
        
        return base_multiple
    
    def _calculate_dynamic_sl_tp(self, entry_price: float, direction: str, 
                                 indicators: Dict, context: Dict, 
                                 sr_levels: Dict, atr_multiple: float) -> Tuple[float, float]:
        """Calculate dynamic stop-loss and take-profit levels"""
        atr = indicators['atr']
        
        # Stop-Loss calculation
        if context['trend_strength'] == 'strong':
            sl_distance = atr * 1.5  # Tighter SL in strong trend
        elif context['volatility'] == 'high':
            sl_distance = atr * 2.5  # Wider SL in high volatility
        else:
            sl_distance = atr * 2.0  # Normal
        
        # Adjust SL for momentum extremes
        if (direction == 'long' and context['momentum'] == 'oversold') or \
           (direction == 'short' and context['momentum'] == 'overbought'):
            sl_distance *= 0.8  # Tighter SL at reversal points
        
        # Calculate SL and TP
        if direction == 'long':
            # Use support level or ATR-based SL
            support = sr_levels.get('primary_support', entry_price * 0.95)
            stop_loss = max(support, entry_price - sl_distance)
            
            # Use resistance or ATR-based TP (prefer Bollinger upper band)
            resistance = sr_levels.get('primary_resistance', entry_price * 1.05)
            atr_tp = entry_price + (atr * atr_multiple)
            bb_tp = indicators.get('bb_upper', atr_tp)
            take_profit = min(resistance, max(atr_tp, bb_tp))
            
        else:  # short
            # Use resistance level or ATR-based SL
            resistance = sr_levels.get('primary_resistance', entry_price * 1.05)
            stop_loss = min(resistance, entry_price + sl_distance)
            
            # Use support or ATR-based TP (prefer Bollinger lower band)
            support = sr_levels.get('primary_support', entry_price * 0.95)
            atr_tp = entry_price - (atr * atr_multiple)
            bb_tp = indicators.get('bb_lower', atr_tp)
            take_profit = max(support, min(atr_tp, bb_tp))
        
        return stop_loss, take_profit
    
    def _calculate_dynamic_min_rr(self, confidence_score: float) -> float:
        """Calculate minimum required RR based on confidence"""
        if confidence_score >= 70:
            return 1.0  # High confidence: lower RR requirement
        elif confidence_score >= 50:
            return 1.2  # Medium confidence
        else:
            return 1.5  # Low confidence: higher RR requirement
    
    def _generate_recommendation(self, confidence_score: float, rr_ratio: float, 
                                context: Dict, direction: str) -> str:
        """Generate trade recommendation"""
        if confidence_score >= 70 and rr_ratio >= 1.5:
            return f"STRONG_{direction.upper()}"
        elif confidence_score >= 60 and rr_ratio >= 1.2:
            return direction.upper()
        elif confidence_score >= 40 and rr_ratio >= 1.0:
            return f"WEAK_{direction.upper()}"
        else:
            return "HOLD"
    
    def _generate_reasoning(self, context: Dict, indicators: Dict, 
                          confidence_score: float, rr_ratio: float, direction: str) -> str:
        """Generate detailed reasoning for the trade"""
        reasoning = f"📊 Enhanced Analysis ({direction.upper()}):\n"
        reasoning += f"Confidence: {confidence_score:.1f}% | RR: {rr_ratio:.2f}:1\n\n"
        reasoning += f"Market Context:\n"
        reasoning += f"- Trend: {context['trend_strength']} {context['trend_direction']} (ADX: {indicators['adx']:.1f})\n"
        reasoning += f"- Momentum: {context['momentum']} (RSI: {indicators['rsi']:.1f})\n"
        reasoning += f"- Volatility: {context['volatility']} (BB Width: {indicators['bb_width']*100:.2f}%)\n"
        reasoning += f"- Condition: {context['market_condition']}\n"
        
        return reasoning


# Global instance for easy import
enhanced_sr_calculator = EnhancedSupportResistanceCalculator()

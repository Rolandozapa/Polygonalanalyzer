"""
Détecteur de Figures Latérales - Ultra Professional Trading System
Module pour identifier et filtrer les patterns latéraux sans vraie tendance
"""

import logging
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TrendType(Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    LATERAL = "lateral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"

@dataclass
class PatternAnalysis:
    trend_type: TrendType
    trend_strength: float  # 0.0 - 1.0
    is_lateral: bool
    price_volatility: float
    volume_consistency: float
    confidence: float  # 0.0 - 1.0
    reasoning: str

class LateralPatternDetector:
    """
    Détecteur sophistiqué de figures latérales pour filtrer les opportunités
    sans vraie tendance directionnelle
    """
    
    def __init__(self):
        self.lateral_threshold_price = 1.0  # +/- 1% considéré comme latéral
        self.lateral_threshold_volume = 0.3  # Variation volume < 30% = latéral
        self.min_trend_strength = 0.6  # Minimum pour considérer une vraie tendance
        
    def analyze_trend_pattern(
        self,
        symbol: str,
        price_change_pct: float,
        volume: float,
        additional_data: Optional[Dict] = None
    ) -> PatternAnalysis:
        """
        Analyse le pattern de trend pour détecter les figures latérales
        
        Args:
            symbol: Symbol crypto (e.g., "BTCUSDT")
            price_change_pct: Changement prix 24h en %
            volume: Volume 24h
            additional_data: Données additionnelles (optionnel)
            
        Returns:
            PatternAnalysis avec classification du trend
        """
        try:
            # 1. Classification basique basée sur price change
            trend_type = self._classify_basic_trend(price_change_pct)
            
            # 2. Calcul de la force du trend
            trend_strength = self._calculate_trend_strength(price_change_pct, volume)
            
            # 3. Détection figure latérale
            is_lateral = self._detect_lateral_pattern(price_change_pct, volume, trend_strength)
            
            # 4. Analyse volatilité prix
            price_volatility = abs(price_change_pct) / 100.0
            
            # 5. Consistance volume (estimation basée sur volume absolu)
            volume_consistency = self._estimate_volume_consistency(volume)
            
            # 6. Calcul confidence globale
            confidence = self._calculate_confidence(
                trend_strength, price_volatility, volume_consistency, is_lateral
            )
            
            # 7. Génération reasoning
            reasoning = self._generate_reasoning(
                trend_type, trend_strength, is_lateral, price_change_pct, volume
            )
            
            analysis = PatternAnalysis(
                trend_type=trend_type,
                trend_strength=trend_strength,
                is_lateral=is_lateral,
                price_volatility=price_volatility,
                volume_consistency=volume_consistency,
                confidence=confidence,
                reasoning=reasoning
            )
            
            logger.debug(f"Pattern analysis for {symbol}: {trend_type.value}, "
                        f"lateral={is_lateral}, strength={trend_strength:.2f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing pattern for {symbol}: {e}")
            # Fallback conservatif
            return PatternAnalysis(
                trend_type=TrendType.LATERAL,
                trend_strength=0.0,
                is_lateral=True,
                price_volatility=0.0,
                volume_consistency=0.0,
                confidence=0.0,
                reasoning=f"Error in analysis: {e}"
            )
    
    def _classify_basic_trend(self, price_change_pct: float) -> TrendType:
        """Classification basique du trend basée sur price change"""
        abs_change = abs(price_change_pct)
        
        if price_change_pct > 5.0:
            return TrendType.STRONG_BULLISH
        elif price_change_pct > 2.0:
            return TrendType.BULLISH
        elif price_change_pct < -5.0:
            return TrendType.STRONG_BEARISH
        elif price_change_pct < -2.0:
            return TrendType.BEARISH
        else:
            return TrendType.LATERAL
    
    def _calculate_trend_strength(self, price_change_pct: float, volume: float) -> float:
        """
        Calcul de la force du trend combinant price change et volume
        
        Returns:
            Float 0.0-1.0 représentant la force du trend
        """
        # Composante prix (0-1)
        price_component = min(abs(price_change_pct) / 10.0, 1.0)  # Normalize to 10%
        
        # Composante volume (estimation logarithmique)
        # Volume élevé = trend plus fort
        if volume <= 0:
            volume_component = 0.0
        else:
            # Normalisation logarithmique du volume
            log_volume = math.log10(max(volume, 1))
            volume_component = min(log_volume / 8.0, 1.0)  # Normalize to 10^8
        
        # Moyenne pondérée: prix 70%, volume 30%
        trend_strength = (price_component * 0.7) + (volume_component * 0.3)
        
        return min(trend_strength, 1.0)
    
    def _detect_lateral_pattern(
        self, 
        price_change_pct: float, 
        volume: float, 
        trend_strength: float
    ) -> bool:
        """
        Détection des figures latérales basée sur plusieurs critères
        
        Returns:
            True si pattern latéral détecté
        """
        # Critère 1: Changement prix faible
        price_lateral = abs(price_change_pct) < self.lateral_threshold_price
        
        # Critère 2: Force trend faible
        strength_lateral = trend_strength < self.min_trend_strength
        
        # Critère 3: Volume inconsistant (très faible ou très élevé peut indiquer manipulation)
        volume_suspicious = volume < 100000 or volume > 50000000  # Seuils arbitraires
        
        # Pattern latéral si au moins 2 critères sur 3
        lateral_indicators = sum([price_lateral, strength_lateral, volume_suspicious])
        
        return lateral_indicators >= 2
    
    def _estimate_volume_consistency(self, volume: float) -> float:
        """
        Estimation de la consistance du volume
        
        Returns:
            Float 0.0-1.0 (1.0 = très consistant)
        """
        if volume <= 0:
            return 0.0
        
        # Estimation basée sur la position du volume dans une fourchette "normale"
        # Volume "normal" entre 500K et 10M
        if 500000 <= volume <= 10000000:
            # Volume dans la fourchette normale
            consistency = 0.8 + (min(volume / 10000000, 1.0) * 0.2)
        elif volume < 500000:
            # Volume trop faible
            consistency = volume / 500000 * 0.5
        else:
            # Volume très élevé (possiblement manipulation)
            consistency = max(0.3, 1.0 - ((volume - 10000000) / 40000000))
        
        return min(max(consistency, 0.0), 1.0)
    
    def _calculate_confidence(
        self,
        trend_strength: float,
        price_volatility: float,
        volume_consistency: float,
        is_lateral: bool
    ) -> float:
        """Calcul de la confidence globale de l'analyse"""
        
        # Base confidence sur trend strength et volume consistency
        base_confidence = (trend_strength * 0.6) + (volume_consistency * 0.4)
        
        # Ajustement basé sur volatilité prix
        if price_volatility < 0.005:  # < 0.5%
            volatility_penalty = 0.3  # Très faible volatilité = moins fiable
        elif price_volatility > 0.15:  # > 15%
            volatility_penalty = 0.2  # Très haute volatilité = moins fiable
        else:
            volatility_penalty = 0.0
        
        # Pénalité si pattern latéral détecté
        lateral_penalty = 0.4 if is_lateral else 0.0
        
        final_confidence = base_confidence - volatility_penalty - lateral_penalty
        
        return min(max(final_confidence, 0.0), 1.0)
    
    def _generate_reasoning(
        self,
        trend_type: TrendType,
        trend_strength: float,
        is_lateral: bool,
        price_change_pct: float,
        volume: float
    ) -> str:
        """Génère une explication textuelle de l'analyse"""
        
        reasoning_parts = []
        
        # Trend classification
        reasoning_parts.append(f"Trend: {trend_type.value.replace('_', ' ').title()}")
        
        # Strength analysis
        if trend_strength > 0.8:
            reasoning_parts.append("très forte conviction")
        elif trend_strength > 0.6:
            reasoning_parts.append("conviction modérée")
        elif trend_strength > 0.3:
            reasoning_parts.append("conviction faible")
        else:
            reasoning_parts.append("conviction très faible")
        
        # Price movement
        reasoning_parts.append(f"mouvement prix {price_change_pct:+.1f}%")
        
        # Volume assessment
        if volume > 10000000:
            reasoning_parts.append("volume très élevé")
        elif volume > 1000000:
            reasoning_parts.append("volume normal")
        elif volume > 100000:
            reasoning_parts.append("volume faible")
        else:
            reasoning_parts.append("volume très faible")
        
        # Lateral pattern
        if is_lateral:
            reasoning_parts.append("⚠️ PATTERN LATÉRAL détecté - pas de vraie tendance")
        else:
            reasoning_parts.append("✅ Vraie tendance directionnelle")
        
        return " | ".join(reasoning_parts)
    
    def should_filter_opportunity(self, analysis: PatternAnalysis) -> bool:
        """
        Détermine si une opportunité doit être filtrée (exclue)
        
        Returns:
            True si l'opportunité doit être filtrée (exclue)
        """
        # Filtrer si:
        # 1. Pattern latéral détecté
        # 2. Trend strength très faible
        # 3. Confidence très faible
        
        return (
            analysis.is_lateral or 
            analysis.trend_strength < 0.3 or 
            analysis.confidence < 0.4
        )

# Instance globale
lateral_pattern_detector = LateralPatternDetector()
"""
Multi-Timeframe Technical Analysis Engine
Analyse professionnelle sur plusieurs timeframes avec confluence intelligente
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd

from multi_timeframe_config import MULTI_TIMEFRAME_CONFIG, get_confluence_requirements

logger = logging.getLogger(__name__)

@dataclass
class TimeframeAnalysis:
    """Résultat d'analyse pour un timeframe donné"""
    timeframe: str
    indicators: Dict[str, Any] = field(default_factory=dict)
    signals: Dict[str, str] = field(default_factory=dict)  # BULLISH/BEARISH/NEUTRAL
    confidence: float = 0.5
    key_levels: Dict[str, float] = field(default_factory=dict)
    summary: str = ""
    
@dataclass 
class MultiTimeframeResult:
    """Résultat d'analyse multi-timeframe complète"""
    symbol: str
    timestamp: datetime
    timeframe_analyses: Dict[str, TimeframeAnalysis] = field(default_factory=dict)
    confluence_score: float = 0.5
    primary_signal: str = "NEUTRAL"  # BULLISH/BEARISH/NEUTRAL
    recommended_action: str = "HOLD"  # LONG/SHORT/HOLD
    confidence_grade: str = "C"  # A+ to D
    risk_reward_ratio: float = 1.0
    key_insights: List[str] = field(default_factory=list)

class MultiTimeframeAnalyzer:
    """Moteur d'analyse multi-timeframe professionnel"""
    
    def __init__(self):
        self.timeframes = ["15m", "1h", "4h", "1d"]
        self.config = MULTI_TIMEFRAME_CONFIG
        
    async def analyze_symbol(self, symbol: str, data_fetcher, talib_calculator) -> MultiTimeframeResult:
        """Analyse complète multi-timeframe d'un symbole"""
        
        logger.info(f"🔍 Starting multi-timeframe analysis for {symbol}")
        
        result = MultiTimeframeResult(
            symbol=symbol,
            timestamp=datetime.now()
        )
        
        # ÉTAPE 1: Analyser chaque timeframe individuellement
        for timeframe in self.timeframes:
            try:
                timeframe_result = await self._analyze_timeframe(
                    symbol, timeframe, data_fetcher, talib_calculator
                )
                result.timeframe_analyses[timeframe] = timeframe_result
                logger.info(f"✅ {timeframe} analysis completed for {symbol}")
                
            except Exception as e:
                logger.error(f"❌ Error analyzing {timeframe} for {symbol}: {e}")
                # Créer un résultat par défaut
                result.timeframe_analyses[timeframe] = TimeframeAnalysis(
                    timeframe=timeframe,
                    summary=f"Error: {str(e)[:100]}"
                )
        
        # ÉTAPE 2: Calculer confluence inter-timeframes  
        result = self._calculate_confluence(result)
        
        # ÉTAPE 3: Générer recommandation finale
        result = self._generate_recommendation(result)
        
        logger.info(f"🎯 Multi-timeframe analysis completed for {symbol}: {result.primary_signal} (Grade: {result.confidence_grade})")
        
        return result
    
    async def _analyze_timeframe(self, symbol: str, timeframe: str, data_fetcher, talib_calculator) -> TimeframeAnalysis:
        """Analyse d'un timeframe spécifique"""
        
        config = self.config.get(timeframe, {})
        required_indicators = config.get("indicators", [])
        
        logger.info(f"📊 Analyzing {timeframe} for {symbol} (indicators: {len(required_indicators)})")
        
        # Récupérer données pour ce timeframe
        historical_data = await data_fetcher(symbol, timeframe=timeframe)
        
        if historical_data is None or len(historical_data) < 50:
            raise ValueError(f"Insufficient data for {timeframe}: {len(historical_data) if historical_data else 0} periods")
        
        # Calculer indicateurs TALib pour ce timeframe
        talib_analysis = await talib_calculator(historical_data)
        
        # Extraire signaux spécifiques au timeframe
        analysis = TimeframeAnalysis(timeframe=timeframe)
        analysis = self._extract_timeframe_signals(analysis, talib_analysis, required_indicators)
        analysis = self._calculate_timeframe_confidence(analysis)
        analysis = self._generate_timeframe_summary(analysis)
        
        return analysis
    
    def _extract_timeframe_signals(self, analysis: TimeframeAnalysis, talib_data, indicators: List[str]) -> TimeframeAnalysis:
        """Extrait les signaux spécifiques pour le timeframe"""
        
        for indicator in indicators:
            try:
                if indicator == "macd":
                    macd_hist = getattr(talib_data, 'macd_histogram', 0.0)
                    analysis.indicators['macd_histogram'] = macd_hist
                    analysis.signals['macd'] = "BULLISH" if macd_hist > 0 else "BEARISH" if macd_hist < 0 else "NEUTRAL"
                    
                elif indicator == "rsi":
                    rsi = getattr(talib_data, 'rsi', 50.0)
                    analysis.indicators['rsi'] = rsi
                    analysis.signals['rsi'] = "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "NEUTRAL"
                    
                elif indicator == "ema_21":
                    ema21 = getattr(talib_data, 'ema_21', 0.0)
                    current_price = getattr(talib_data, 'current_price', 0.0)
                    analysis.indicators['ema_21'] = ema21
                    analysis.signals['ema_21'] = "BULLISH" if current_price > ema21 else "BEARISH"
                    
                elif indicator == "adx":
                    adx = getattr(talib_data, 'adx', 25.0)
                    plus_di = getattr(talib_data, 'plus_di', 25.0)
                    minus_di = getattr(talib_data, 'minus_di', 25.0)
                    analysis.indicators['adx'] = adx
                    analysis.indicators['plus_di'] = plus_di
                    analysis.indicators['minus_di'] = minus_di
                    
                    if adx > 25:
                        trend_signal = "BULLISH" if plus_di > minus_di else "BEARISH"
                    else:
                        trend_signal = "CONSOLIDATION"
                    analysis.signals['adx'] = trend_signal
                    
                elif indicator == "volume_ratio":
                    vol_ratio = getattr(talib_data, 'volume_ratio', 1.0)
                    analysis.indicators['volume_ratio'] = vol_ratio
                    analysis.signals['volume'] = "HIGH" if vol_ratio > 1.5 else "NORMAL" if vol_ratio > 0.7 else "LOW"
                    
                # Ajouter d'autres indicateurs selon besoins...
                    
            except Exception as e:
                logger.warning(f"Error extracting {indicator} for {analysis.timeframe}: {e}")
                
        return analysis
    
    def _calculate_timeframe_confidence(self, analysis: TimeframeAnalysis) -> TimeframeAnalysis:
        """Calcule le niveau de confiance pour ce timeframe"""
        
        # Compter signaux cohérents
        bullish_signals = sum(1 for signal in analysis.signals.values() if "BULLISH" in signal or signal == "OVERSOLD")
        bearish_signals = sum(1 for signal in analysis.signals.values() if "BEARISH" in signal or signal == "OVERBOUGHT") 
        total_signals = len(analysis.signals)
        
        if total_signals == 0:
            analysis.confidence = 0.5
        else:
            # Confiance basée sur alignement des signaux
            max_aligned = max(bullish_signals, bearish_signals)
            analysis.confidence = (max_aligned / total_signals) * 0.8 + 0.2  # Min 20%, Max 100%
            
        return analysis
    
    def _generate_timeframe_summary(self, analysis: TimeframeAnalysis) -> TimeframeAnalysis:
        """Génère un résumé textuel pour ce timeframe"""
        
        tf = analysis.timeframe
        config = self.config.get(tf, {})
        primary_use = config.get("primary_use", "analysis")
        
        # Identifier signal dominant
        bullish_count = sum(1 for s in analysis.signals.values() if "BULLISH" in s)
        bearish_count = sum(1 for s in analysis.signals.values() if "BEARISH" in s)
        
        if bullish_count > bearish_count:
            dominant = "BULLISH"
        elif bearish_count > bullish_count:
            dominant = "BEARISH"  
        else:
            dominant = "NEUTRAL"
            
        analysis.summary = f"{tf.upper()}: {dominant} bias ({primary_use}) - Confidence: {analysis.confidence:.0%}"
        
        return analysis
    
    def _calculate_confluence(self, result: MultiTimeframeResult) -> MultiTimeframeResult:
        """Calcule la confluence entre timeframes"""
        
        timeframes = result.timeframe_analyses
        
        # Compter alignements directionnels
        bullish_timeframes = 0
        bearish_timeframes = 0
        neutral_timeframes = 0
        
        total_confidence = 0
        
        for tf, analysis in timeframes.items():
            if not analysis.signals:
                continue
                
            weight = self.config.get(tf, {}).get("weight", 0.25)
            
            # Compter signaux par direction
            tf_bullish = sum(1 for s in analysis.signals.values() if "BULLISH" in s or s == "OVERSOLD")
            tf_bearish = sum(1 for s in analysis.signals.values() if "BEARISH" in s or s == "OVERBOUGHT")
            
            if tf_bullish > tf_bearish:
                bullish_timeframes += weight
            elif tf_bearish > tf_bullish:
                bearish_timeframes += weight
            else:
                neutral_timeframes += weight
                
            total_confidence += analysis.confidence * weight
        
        # Score confluence final
        max_direction = max(bullish_timeframes, bearish_timeframes, neutral_timeframes)
        result.confluence_score = max_direction
        
        # Signal primaire
        if bullish_timeframes > bearish_timeframes and bullish_timeframes > neutral_timeframes:
            result.primary_signal = "BULLISH"
        elif bearish_timeframes > bullish_timeframes and bearish_timeframes > neutral_timeframes:
            result.primary_signal = "BEARISH"
        else:
            result.primary_signal = "NEUTRAL"
            
        return result
    
    def _generate_recommendation(self, result: MultiTimeframeResult) -> MultiTimeframeResult:
        """Génère la recommandation finale et le grade de confiance"""
        
        # Grade basé sur confluence
        if result.confluence_score >= 0.8:
            result.confidence_grade = "A+"
        elif result.confluence_score >= 0.7:
            result.confidence_grade = "A"
        elif result.confluence_score >= 0.6:
            result.confidence_grade = "B"
        elif result.confluence_score >= 0.5:
            result.confidence_grade = "C"
        else:
            result.confidence_grade = "D"
            
        # Recommandation d'action
        if result.primary_signal == "BULLISH" and result.confluence_score > 0.6:
            result.recommended_action = "LONG"
        elif result.primary_signal == "BEARISH" and result.confluence_score > 0.6:
            result.recommended_action = "SHORT" 
        else:
            result.recommended_action = "HOLD"
            
        # Risk/Reward basé sur confluence et timeframes
        result.risk_reward_ratio = 1.0 + (result.confluence_score - 0.5) * 4  # RR entre 1:1 et 3:1
        
        # Insights clés
        result.key_insights = self._generate_key_insights(result)
        
        return result
    
    def _generate_key_insights(self, result: MultiTimeframeResult) -> List[str]:
        """Génère les insights clés de l'analyse"""
        
        insights = []
        
        # Analyse confluence
        if result.confluence_score > 0.7:
            insights.append(f"🎯 Strong {result.primary_signal.lower()} confluence across timeframes ({result.confluence_score:.0%})")
        elif result.confluence_score < 0.4:
            insights.append(f"⚠️ Mixed signals - low confluence ({result.confluence_score:.0%})")
            
        # Analyse par timeframe
        for tf, analysis in result.timeframe_analyses.items():
            config = self.config.get(tf, {})
            primary_use = config.get("primary_use", "")
            
            if analysis.confidence > 0.75:
                insights.append(f"✅ {tf.upper()}: Strong {primary_use} signals (confidence: {analysis.confidence:.0%})")
            elif analysis.confidence < 0.4:
                insights.append(f"❌ {tf.upper()}: Weak {primary_use} signals (confidence: {analysis.confidence:.0%})")
                
        return insights[:5]  # Limiter à 5 insights maximum
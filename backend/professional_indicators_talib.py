"""
PROFESSIONAL TECHNICAL INDICATORS WITH TALIB
Système complet d'indicateurs techniques pour IA1 v6.0
Basé sur TA-Lib pour la précision maximale
"""

import pandas as pd
import numpy as np
import talib
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class CompleteTechnicalAnalysis:
    """Structure complète d'analyse technique pour IA1 v6.0"""
    
    # === REGIME DETECTION ===
    regime: str = "NEUTRAL"
    confidence: float = 0.5
    base_confidence: float = 0.5
    technical_consistency: float = 0.5
    combined_confidence: float = 0.5
    regime_persistence: int = 0
    stability_score: float = 0.5
    regime_transition_alert: str = "STABLE"
    fresh_regime: bool = False
    
    # === RSI INDICATORS ===
    rsi_14: float = 50.0
    rsi_9: float = 50.0
    rsi_21: float = 50.0
    rsi_zone: str = "NEUTRAL"
    rsi_overbought: bool = False
    rsi_oversold: bool = False
    rsi_divergence: bool = False
    
    # === MACD INDICATORS ===
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    macd_trend: str = "NEUTRAL"
    macd_bullish: bool = False
    macd_bearish: bool = False
    
    # === ADX (CORRECTED WILDER METHOD) ===
    adx: float = 25.0
    adx_strength: str = "MODERATE"
    plus_di: float = 25.0
    minus_di: float = 25.0
    dx: float = 0.0
    
    # === BOLLINGER BANDS ===
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_position: float = 0.5
    bb_squeeze: bool = False
    squeeze_intensity: str = "NONE"
    bb_bandwidth: float = 0.04
    
    # === STOCHASTIC ===
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    stoch_overbought: bool = False
    stoch_oversold: bool = False
    
    # === MFI (MONEY FLOW INDEX) ===
    mfi: float = 50.0
    mfi_overbought: bool = False
    mfi_oversold: bool = False
    mfi_signal: str = "NEUTRAL"
    
    # === ATR & VOLATILITY ===
    atr: float = 0.02
    atr_pct: float = 2.0
    volatility_ratio: float = 1.0
    
    # === VOLUME ANALYSIS ===
    volume_ratio: float = 1.0
    volume_trend: float = 0.0
    volume_surge: bool = False
    
    # === MOVING AVERAGES ===
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    ema_9: float = 0.0
    ema_21: float = 0.0
    ema_50: float = 0.0
    ema_200: float = 0.0
    sma_20_slope: float = 0.0
    
    # === TREND ANALYSIS ===
    trend_hierarchy: str = "NEUTRAL"
    trend_strength_score: float = 0.5
    price_vs_emas: str = "NEUTRAL"
    above_sma_20: bool = True
    
    # === VWAP ===
    vwap: float = 0.0
    vwap_distance: float = 0.0
    above_vwap: bool = True
    
    # === CONFLUENCE GRADING ===
    confluence_grade: str = "C"
    confluence_score: int = 50
    should_trade: bool = False
    conviction_level: str = "FAIBLE"
    
    # === POSITION SIZING ===
    regime_multiplier: float = 1.0
    ml_confidence_multiplier: float = 1.0
    momentum_multiplier: float = 1.0
    bb_multiplier: float = 1.0
    combined_multiplier: float = 1.0
    position_size_pct: float = 1.0
    risk_pct: float = 1.0

class ProfessionalIndicatorsTALib:
    """
    Système professionnel d'indicateurs techniques avec TA-Lib
    Implémente EXACTEMENT les indicateurs requis par IA1 v6.0
    """
    
    def __init__(self):
        self.regime_history = deque(maxlen=50)  # Historique des régimes
        logger.info("🚀 ProfessionalIndicatorsTALib initialized with TA-Lib")
    
    def calculate_all_indicators(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> CompleteTechnicalAnalysis:
        """
        Calcule TOUS les indicateurs techniques avec TA-Lib
        Retourne une analyse complète pour IA1 v6.0
        """
        try:
            if len(df) < 50:
                logger.warning(f"⚠️ Insufficient data for {symbol}: {len(df)} < 50 bars")
                return self._create_minimal_analysis(df, symbol)
            
            logger.info(f"🔬 Calculating professional indicators for {symbol} with {len(df)} bars")
            
            # 🔧 ADAPTIVE COLUMN MAPPING - Handle different OHLCV column names
            logger.info(f"   📊 DataFrame columns: {df.columns.tolist()}")
            
            # Auto-detect column names (case insensitive)
            column_mapping = {}
            for standard_col in ['high', 'low', 'close', 'open', 'volume']:
                for df_col in df.columns:
                    if standard_col.lower() in df_col.lower():
                        column_mapping[standard_col] = df_col
                        break
                
                # Fallback patterns
                if standard_col not in column_mapping:
                    if standard_col == 'close' and 'Close' in df.columns:
                        column_mapping['close'] = 'Close'
                    elif standard_col == 'high' and 'High' in df.columns:
                        column_mapping['high'] = 'High'
                    elif standard_col == 'low' and 'Low' in df.columns:
                        column_mapping['low'] = 'Low'
                    elif standard_col == 'volume' and 'Volume' in df.columns:
                        column_mapping['volume'] = 'Volume'
            
            logger.info(f"   🔧 Column mapping detected: {column_mapping}")
            
            # Extract data with proper column mapping
            try:
                high = df[column_mapping.get('high', df.columns[1])].values.astype(np.float64)  # Fallback to 2nd column
                low = df[column_mapping.get('low', df.columns[2])].values.astype(np.float64)   # Fallback to 3rd column  
                close = df[column_mapping.get('close', df.columns[3])].values.astype(np.float64) # Fallback to 4th column
                
                if 'volume' in column_mapping:
                    volume = df[column_mapping['volume']].values.astype(np.float64)
                else:
                    volume = np.array([100000] * len(df), dtype=np.float64)  # Default volume
                    logger.warning(f"   ⚠️ No volume column found, using default 100K volume")
                    
            except Exception as col_error:
                logger.error(f"   ❌ Column extraction failed: {col_error}")
                logger.error(f"   📊 Available columns: {df.columns.tolist()}")
                raise col_error
            
            current_price = float(close[-1])
            
            # === 1. RSI INDICATORS ===
            rsi_14 = talib.RSI(close, timeperiod=14)[-1]
            rsi_9 = talib.RSI(close, timeperiod=9)[-1]
            rsi_21 = talib.RSI(close, timeperiod=21)[-1]
            
            # Déterminer la zone RSI
            if rsi_14 > 70:
                rsi_zone = "OVERBOUGHT"
                rsi_overbought = True
                rsi_oversold = False
            elif rsi_14 < 30:
                rsi_zone = "OVERSOLD"
                rsi_overbought = False
                rsi_oversold = True
            elif 40 <= rsi_14 <= 60:
                rsi_zone = "NEUTRAL"
                rsi_overbought = False
                rsi_oversold = False
            else:
                rsi_zone = "NORMAL"
                rsi_overbought = False
                rsi_oversold = False
            
            logger.info(f"   📊 RSI: 14={rsi_14:.1f}, 9={rsi_9:.1f}, 21={rsi_21:.1f} [{rsi_zone}]")
            
            # === 2. MACD INDICATORS ===
            macd_line_array, macd_signal_array, macd_hist_array = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            
            macd_line = macd_line_array[-1]
            macd_signal = macd_signal_array[-1]
            macd_histogram = macd_hist_array[-1]
            
            # Déterminer la tendance MACD
            if macd_histogram > 0:
                if len(macd_hist_array) > 1 and macd_histogram > macd_hist_array[-2]:
                    macd_trend = "BULLISH_RISING"
                    macd_bullish = True
                else:
                    macd_trend = "BULLISH_STABLE"
                    macd_bullish = True
                macd_bearish = False
            elif macd_histogram < 0:
                if len(macd_hist_array) > 1 and macd_histogram < macd_hist_array[-2]:
                    macd_trend = "BEARISH_FALLING"
                    macd_bearish = True
                else:
                    macd_trend = "BEARISH_STABLE"
                    macd_bearish = True
                macd_bullish = False
            else:
                macd_trend = "NEUTRAL"
                macd_bullish = False
                macd_bearish = False
            
            logger.info(f"   📊 MACD: Line={macd_line:.6f}, Signal={macd_signal:.6f}, Hist={macd_histogram:.6f} [{macd_trend}]")
            
            # === 3. ADX (CORRECTED WILDER METHOD) ===
            adx = talib.ADX(high, low, close, timeperiod=14)[-1]
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)[-1]
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)[-1]
            
            # Calculer DX manuellement
            if plus_di + minus_di != 0:
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            else:
                dx = 0
            
            # Déterminer la force ADX
            if adx >= 50:
                adx_strength = "VERY_STRONG"
            elif adx >= 25:
                adx_strength = "STRONG"
            elif adx >= 20:
                adx_strength = "MODERATE"
            else:
                adx_strength = "WEAK"
            
            logger.info(f"   📊 ADX: {adx:.1f} [{adx_strength}], +DI={plus_di:.1f}, -DI={minus_di:.1f}")
            
            # === 4. BOLLINGER BANDS ===
            bb_upper_array, bb_middle_array, bb_lower_array = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            
            bb_upper = bb_upper_array[-1]
            bb_middle = bb_middle_array[-1]
            bb_lower = bb_lower_array[-1]
            
            # Position dans les bandes
            if bb_upper != bb_lower:
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            else:
                bb_position = 0.5
            
            # Calculer bandwidth pour squeeze
            bb_bandwidth = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0.04
            
            # Détection squeeze
            if bb_bandwidth < 0.02:
                bb_squeeze = True
                squeeze_intensity = "EXTREME"
            elif bb_bandwidth < 0.035:
                bb_squeeze = True
                squeeze_intensity = "TIGHT"
            else:
                bb_squeeze = False
                squeeze_intensity = "NONE"
            
            logger.info(f"   📊 BB: Upper={bb_upper:.4f}, Lower={bb_lower:.4f}, Pos={bb_position:.2f}, Squeeze={bb_squeeze} [{squeeze_intensity}]")
            
            # === 5. STOCHASTIC ===
            stoch_k_array, stoch_d_array = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            
            stoch_k = stoch_k_array[-1]
            stoch_d = stoch_d_array[-1]
            
            stoch_overbought = stoch_k > 80
            stoch_oversold = stoch_k < 20
            
            logger.info(f"   📊 Stochastic: K={stoch_k:.1f}, D={stoch_d:.1f}")
            
            # === 6. MFI (MONEY FLOW INDEX) ===
            mfi = talib.MFI(high, low, close, volume, timeperiod=14)[-1]
            
            mfi_overbought = mfi > 80
            mfi_oversold = mfi < 20
            
            if mfi > 80:
                mfi_signal = "DISTRIBUTION"
            elif mfi < 20:
                mfi_signal = "ACCUMULATION"
            else:
                mfi_signal = "NEUTRAL"
            
            logger.info(f"   📊 MFI: {mfi:.1f} [{mfi_signal}]")
            
            # === 7. ATR & VOLATILITY ===
            atr = talib.ATR(high, low, close, timeperiod=14)[-1]
            atr_pct = (atr / current_price * 100) if current_price > 0 else 2.0
            
            # Calculer volatility ratio (ATR actuel vs moyenne)
            if len(df) >= 50:
                atr_50_avg = talib.ATR(high, low, close, timeperiod=14)[-50:-1].mean()
                volatility_ratio = atr / atr_50_avg if atr_50_avg > 0 else 1.0
            else:
                volatility_ratio = 1.0
            
            logger.info(f"   📊 ATR: {atr:.6f} ({atr_pct:.2f}%), Vol Ratio={volatility_ratio:.2f}")
            
            # === 8. MOVING AVERAGES ===
            sma_20 = talib.SMA(close, timeperiod=20)[-1]
            sma_50 = talib.SMA(close, timeperiod=50)[-1] if len(close) >= 50 else sma_20
            sma_200 = talib.SMA(close, timeperiod=200)[-1] if len(close) >= 200 else sma_20
            
            ema_9 = talib.EMA(close, timeperiod=9)[-1]
            ema_21 = talib.EMA(close, timeperiod=21)[-1]
            ema_50 = talib.EMA(close, timeperiod=50)[-1] if len(close) >= 50 else ema_21
            ema_200 = talib.EMA(close, timeperiod=200)[-1] if len(close) >= 200 else ema_21
            
            # Calculer slope SMA 20
            if len(df) >= 25:
                sma_20_prev = talib.SMA(close, timeperiod=20)[-5]
                sma_20_slope = (sma_20 - sma_20_prev) / sma_20_prev if sma_20_prev > 0 else 0
            else:
                sma_20_slope = 0.0
            
            # Déterminer hiérarchie de tendance
            above_sma_20 = current_price > sma_20
            
            if ema_9 > ema_21 > ema_50:
                trend_hierarchy = "BULLISH"
                trend_strength_score = 0.8
                price_vs_emas = "ABOVE" if current_price > ema_21 else "BELOW"
            elif ema_9 < ema_21 < ema_50:
                trend_hierarchy = "BEARISH"
                trend_strength_score = 0.2
                price_vs_emas = "BELOW" if current_price < ema_21 else "ABOVE"
            else:
                trend_hierarchy = "NEUTRAL"
                trend_strength_score = 0.5
                price_vs_emas = "NEUTRAL"
            
            logger.info(f"   📊 EMAs: 9={ema_9:.4f}, 21={ema_21:.4f}, 50={ema_50:.4f} [{trend_hierarchy}]")
            
            # === 9. VWAP ===
            # Calculer VWAP simple
            typical_price = (high + low + close) / 3
            vwap = np.sum(typical_price * volume) / np.sum(volume) if np.sum(volume) > 0 else current_price
            vwap_distance = ((current_price - vwap) / vwap * 100) if vwap > 0 else 0.0
            above_vwap = current_price > vwap
            
            logger.info(f"   📊 VWAP: {vwap:.4f}, Distance={vwap_distance:+.2f}%")
            
            # === 10. VOLUME ANALYSIS ===
            if len(volume) >= 20:
                volume_sma = talib.SMA(volume.astype(np.float64), timeperiod=20)[-1]
                volume_ratio = volume[-1] / volume_sma if volume_sma > 0 else 1.0
                
                # Calculer tendance volume sur 10 barres
                if len(volume) >= 10:
                    volume_trend = np.polyfit(range(10), volume[-10:], 1)[0] / volume[-1] if volume[-1] > 0 else 0
                else:
                    volume_trend = 0.0
                
                volume_surge = volume_ratio > 2.0
            else:
                volume_ratio = 1.0
                volume_trend = 0.0
                volume_surge = False
            
            logger.info(f"   📊 Volume: Ratio={volume_ratio:.2f}, Trend={volume_trend:.4f}, Surge={volume_surge}")
            
            # === 11. REGIME DETECTION ===
            regime_info = self._detect_market_regime(
                adx, rsi_14, macd_histogram, bb_squeeze, sma_20_slope, 
                trend_hierarchy, volume_ratio, atr_pct, squeeze_intensity
            )
            
            # === 12. TECHNICAL CONSISTENCY ===
            technical_consistency = self._calculate_technical_consistency(
                rsi_14, macd_histogram, trend_hierarchy, volume_ratio, 
                atr_pct, sma_20_slope, above_sma_20, adx
            )
            
            # === 13. COMBINED CONFIDENCE ===
            base_confidence = regime_info['confidence']
            combined_confidence = 0.7 * base_confidence + 0.3 * technical_consistency
            
            # === 14. CONFLUENCE GRADING ===
            confluence_result = self._calculate_confluence_grade(
                regime_info, combined_confidence, rsi_14, macd_histogram, 
                bb_squeeze, volume_ratio, adx, technical_consistency
            )
            
            # === 15. POSITION SIZING MULTIPLIERS ===
            multipliers = self._calculate_position_multipliers(
                regime_info['regime'], combined_confidence, rsi_14, 
                macd_histogram, bb_squeeze, squeeze_intensity, volume_ratio
            )
            
            # Créer l'analyse complète
            analysis = CompleteTechnicalAnalysis(
                # Regime
                regime=regime_info['regime'],
                confidence=combined_confidence,
                base_confidence=base_confidence,
                technical_consistency=technical_consistency,
                combined_confidence=combined_confidence,
                regime_persistence=regime_info['persistence'],
                stability_score=regime_info['stability_score'],
                regime_transition_alert=regime_info['transition_alert'],
                fresh_regime=regime_info['fresh_regime'],
                
                # RSI
                rsi_14=rsi_14,
                rsi_9=rsi_9,
                rsi_21=rsi_21,
                rsi_zone=rsi_zone,
                rsi_overbought=rsi_overbought,
                rsi_oversold=rsi_oversold,
                rsi_divergence=False,  # TODO: Implement divergence detection
                
                # MACD
                macd_line=macd_line,
                macd_signal=macd_signal,
                macd_histogram=macd_histogram,
                macd_trend=macd_trend,
                macd_bullish=macd_bullish,
                macd_bearish=macd_bearish,
                
                # ADX
                adx=adx,
                adx_strength=adx_strength,
                plus_di=plus_di,
                minus_di=minus_di,
                dx=dx,
                
                # Bollinger Bands
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                bb_position=bb_position,
                bb_squeeze=bb_squeeze,
                squeeze_intensity=squeeze_intensity,
                bb_bandwidth=bb_bandwidth,
                
                # Stochastic
                stoch_k=stoch_k,
                stoch_d=stoch_d,
                stoch_overbought=stoch_overbought,
                stoch_oversold=stoch_oversold,
                
                # MFI
                mfi=mfi,
                mfi_overbought=mfi_overbought,
                mfi_oversold=mfi_oversold,
                mfi_signal=mfi_signal,
                
                # ATR & Volatility
                atr=atr,
                atr_pct=atr_pct,
                volatility_ratio=volatility_ratio,
                
                # Volume
                volume_ratio=volume_ratio,
                volume_trend=volume_trend,
                volume_surge=volume_surge,
                
                # Moving Averages
                sma_20=sma_20,
                sma_50=sma_50,
                sma_200=sma_200,
                ema_9=ema_9,
                ema_21=ema_21,
                ema_50=ema_50,
                ema_200=ema_200,
                sma_20_slope=sma_20_slope,
                
                # Trend
                trend_hierarchy=trend_hierarchy,
                trend_strength_score=trend_strength_score,
                price_vs_emas=price_vs_emas,
                above_sma_20=above_sma_20,
                
                # VWAP
                vwap=vwap,
                vwap_distance=vwap_distance,
                above_vwap=above_vwap,
                
                # Confluence
                confluence_grade=confluence_result['grade'],
                confluence_score=confluence_result['score'],
                should_trade=confluence_result['should_trade'],
                conviction_level=confluence_result['conviction_level'],
                
                # Position Sizing
                regime_multiplier=multipliers['regime_multiplier'],
                ml_confidence_multiplier=multipliers['ml_confidence_multiplier'],
                momentum_multiplier=multipliers['momentum_multiplier'],
                bb_multiplier=multipliers['bb_multiplier'],
                combined_multiplier=multipliers['combined_multiplier'],
                position_size_pct=multipliers['position_size_pct'],
                risk_pct=multipliers['risk_pct']
            )
            
            logger.info(f"✅ Professional analysis complete for {symbol}: {confluence_result['grade']} grade, {combined_confidence:.1%} confidence")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error in professional indicators calculation for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return self._create_minimal_analysis(df, symbol)
    
    def _detect_market_regime(self, adx, rsi, macd_hist, bb_squeeze, sma_slope, 
                            trend_hierarchy, volume_ratio, atr_pct, squeeze_intensity) -> Dict:
        """Détecte le régime de marché selon les critères IA1 v6.0"""
        
        # Score de base pour chaque régime
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
        
        # TRENDING_UP_STRONG
        if (adx > 25 and trend_hierarchy == "BULLISH" and 
            macd_hist > 0 and sma_slope > 0.003 and rsi > 50):
            regime_scores['TRENDING_UP_STRONG'] += 0.8
            
        # BREAKOUT_BULLISH  
        if (bb_squeeze and squeeze_intensity in ["EXTREME", "TIGHT"] and 
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
            
        # Sélectionner le régime avec le score le plus élevé
        best_regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[best_regime]
        
        # Si aucun régime n'a un score élevé, utiliser CONSOLIDATION par défaut
        if confidence < 0.5:
            best_regime = "CONSOLIDATION"
            confidence = 0.5
        
        # Persistance et transition (simulé pour l'instant)
        self.regime_history.append(best_regime)
        persistence = len([r for r in self.regime_history if r == best_regime])
        
        # Détection de transition
        if len(self.regime_history) >= 5:
            recent_regimes = list(self.regime_history)[-5:]
            if len(set(recent_regimes)) == 1:
                transition_alert = "STABLE"
            elif len(set(recent_regimes[-3:])) == 3:
                transition_alert = "IMMINENT_CHANGE"
            else:
                transition_alert = "EARLY_WARNING"
        else:
            transition_alert = "STABLE"
        
        fresh_regime = persistence < 15
        stability_score = min(1.0, persistence / 40.0)
        
        return {
            'regime': best_regime,
            'confidence': confidence,
            'persistence': persistence,
            'stability_score': stability_score,
            'transition_alert': transition_alert,
            'fresh_regime': fresh_regime
        }
    
    def _calculate_technical_consistency(self, rsi, macd_hist, trend_hierarchy, 
                                       volume_ratio, atr_pct, sma_slope, above_sma_20, adx) -> float:
        """Calcule la cohérence technique selon IA1 v6.0"""
        
        consistency_score = 0.0
        
        # Trend Coherence (35%)
        trend_score = 0.0
        if trend_hierarchy in ["BULLISH", "BEARISH"]:
            trend_score += 0.3  # SMA alignment
        if abs(sma_slope) > 0.001:
            trend_score += 0.2  # Meaningful slope
        if adx > 18:
            trend_score += 0.3  # ADX confirms trend
        if above_sma_20 == (trend_hierarchy == "BULLISH"):
            trend_score += 0.2  # Price-trend alignment
        
        consistency_score += 0.35 * trend_score
        
        # Momentum Coherence (35%)
        momentum_score = 0.0
        if ((rsi > 50) == (macd_hist > 0)):
            momentum_score += 0.5  # RSI-MACD alignment
        if 35 < rsi < 70:
            momentum_score += 0.3  # RSI not extreme
        if abs(macd_hist) > 0.001:
            momentum_score += 0.2  # MACD has direction
        
        consistency_score += 0.35 * momentum_score
        
        # Volume Coherence (15%)
        volume_score = 0.0
        if volume_ratio > 1.0:
            volume_score += 0.6  # Volume confirms
        if 0.8 < volume_ratio < 2.5:
            volume_score += 0.4  # Volume stable range
        
        consistency_score += 0.15 * volume_score
        
        # Volatility Coherence (15%)
        volatility_score = 0.0
        if 0.5 < atr_pct < 4.0:
            volatility_score += 0.6  # Appropriate volatility
        if atr_pct < 3.0:
            volatility_score += 0.4  # Not excessive volatility
        
        consistency_score += 0.15 * volatility_score
        
        return min(1.0, max(0.0, consistency_score))
    
    def _calculate_confluence_grade(self, regime_info, combined_confidence, rsi, 
                                  macd_hist, bb_squeeze, volume_ratio, adx, technical_consistency) -> Dict:
        """Calcule le grade de confluence selon IA1 v6.0"""
        
        score = 0
        
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
        
        # Momentum conditions (check minimum 2 of 6)
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
        if abs(regime_info.get('sma_slope', 0)) > 0.001:
            momentum_conditions += 1
            score += 5
        if volume_ratio > 1.2:
            momentum_conditions += 1
            score += 5
        if regime_info.get('above_sma_20', True):
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
            score += 15  # ML_BREAKOUT_SQUEEZE
        if adx > 25 and combined_confidence > 0.8:
            score += 15  # ML_TREND_ACCELERATION
        if regime_info.get('fresh_regime', False) and combined_confidence > 0.85:
            score += 20  # ML_FRESH_REGIME
        if volume_ratio > 2.0:
            score += 10  # ML_VOLUME_SURGE
        
        # Technical consistency bonus
        score += technical_consistency * 10
        
        # Determine grade
        score = min(100, int(score))
        
        if score >= 90:
            grade = 'A++'
            should_trade = True
            conviction = 'TRÈS HAUTE'
        elif score >= 80:
            grade = 'A+'
            should_trade = True
            conviction = 'TRÈS HAUTE'
        elif score >= 75:
            grade = 'A'
            should_trade = True
            conviction = 'HAUTE'
        elif score >= 70:
            grade = 'B+'
            should_trade = True
            conviction = 'HAUTE'
        elif score >= 65:
            grade = 'B'
            should_trade = True
            conviction = 'MOYENNE'
        else:
            grade = 'C'
            should_trade = False
            conviction = 'FAIBLE'
        
        return {
            'grade': grade,
            'score': score,
            'should_trade': should_trade,
            'conviction_level': conviction
        }
    
    def _calculate_position_multipliers(self, regime, confidence, rsi, macd_hist, 
                                      bb_squeeze, squeeze_intensity, volume_ratio) -> Dict:
        """Calcule les multiplicateurs de position selon IA1 v6.0"""
        
        # Regime multipliers
        regime_multipliers = {
            'TRENDING_UP_STRONG': 1.3,
            'BREAKOUT_BULLISH': 1.25,
            'TRENDING_UP_MODERATE': 1.15,
            'CONSOLIDATION': 0.9,
            'RANGING_TIGHT': 0.8,
            'RANGING_WIDE': 0.7,
            'VOLATILE': 0.6,
            'TRENDING_DOWN_MODERATE': 0.5,
            'TRENDING_DOWN_STRONG': 0.3,
            'BREAKOUT_BEARISH': 0.3
        }
        
        regime_mult = regime_multipliers.get(regime, 1.0)
        
        # ML confidence multipliers
        if confidence >= 0.9:
            ml_mult = 1.3
        elif confidence >= 0.8:
            ml_mult = 1.15
        elif confidence >= 0.7:
            ml_mult = 1.0
        elif confidence >= 0.6:
            ml_mult = 0.8
        else:
            ml_mult = 0.5
        
        # Momentum multiplier (simplified)
        momentum_mult = 1.0
        if 45 <= rsi <= 60 and macd_hist > 0:
            momentum_mult = 1.2
        elif rsi > 70 or rsi < 30:
            momentum_mult = 0.7
        
        # BB squeeze multipliers
        if squeeze_intensity == "EXTREME" and confidence > 0.85 and volume_ratio > 2.0:
            bb_mult = 1.4
        elif squeeze_intensity == "TIGHT" and confidence > 0.7:
            bb_mult = 1.2
        elif bb_squeeze:
            bb_mult = 1.1
        else:
            bb_mult = 1.0
        
        combined_mult = regime_mult * ml_mult * momentum_mult * bb_mult
        position_size_pct = combined_mult * 1.0  # Base 1%
        risk_pct = min(1.5, position_size_pct)  # Cap at 1.5%
        
        return {
            'regime_multiplier': regime_mult,
            'ml_confidence_multiplier': ml_mult,
            'momentum_multiplier': momentum_mult,
            'bb_multiplier': bb_mult,
            'combined_multiplier': combined_mult,
            'position_size_pct': position_size_pct,
            'risk_pct': risk_pct
        }
    
    def _create_minimal_analysis(self, df: pd.DataFrame, symbol: str) -> CompleteTechnicalAnalysis:
        """Crée une analyse minimale pour données insuffisantes"""
        
        current_price = float(df['close'].iloc[-1])
        
        return CompleteTechnicalAnalysis(
            regime="CONSOLIDATION",
            confidence=0.5,
            base_confidence=0.5,
            technical_consistency=0.5,
            combined_confidence=0.5,
            
            rsi_14=50.0,
            rsi_9=50.0,
            rsi_21=50.0,
            
            bb_upper=current_price * 1.02,
            bb_middle=current_price,
            bb_lower=current_price * 0.98,
            
            sma_20=current_price,
            ema_9=current_price,
            ema_21=current_price,
            
            vwap=current_price,
            
            confluence_grade='C',
            confluence_score=50,
            should_trade=False,
            conviction_level='FAIBLE'
        )

# Global instance
professional_indicators_talib = ProfessionalIndicatorsTALib()

def get_professional_indicators():
    """Retourne l'instance globale des indicateurs professionnels"""
    return professional_indicators_talib
import pandas as pd
import numpy as np
import logging
import aiohttp
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import yfinance as yf
import os
from enum import Enum

logger = logging.getLogger(__name__)

class PatternType(Enum):
    BULLISH_BREAKOUT = "bullish_breakout"
    BEARISH_BREAKDOWN = "bearish_breakdown"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    GOLDEN_CROSS = "golden_cross"
    DEATH_CROSS = "death_cross"
    VOLUME_SPIKE = "volume_spike"
    RSI_DIVERGENCE = "rsi_divergence"
    MACD_BULLISH = "macd_bullish"
    SUPPORT_BOUNCE = "support_bounce"
    RESISTANCE_BREAK = "resistance_break"
    
    # Nouvelles tendances soutenues pour positions long/short
    SUSTAINED_BULLISH_TREND = "sustained_bullish_trend"
    SUSTAINED_BEARISH_TREND = "sustained_bearish_trend"
    BULLISH_CHANNEL = "bullish_channel"
    BEARISH_CHANNEL = "bearish_channel"
    DYNAMIC_SUPPORT_UPTREND = "dynamic_support_uptrend"
    DYNAMIC_RESISTANCE_DOWNTREND = "dynamic_resistance_downtrend"
    BULLISH_MOMENTUM_CONTINUATION = "bullish_momentum_continuation"
    BEARISH_MOMENTUM_CONTINUATION = "bearish_momentum_continuation"
    MULTIPLE_MA_BULLISH_ALIGNMENT = "multiple_ma_bullish_alignment"
    MULTIPLE_MA_BEARISH_ALIGNMENT = "multiple_ma_bearish_alignment"

@dataclass
class TechnicalPattern:
    symbol: str
    pattern_type: PatternType
    confidence: float
    strength: float  # 0-1, force du signal
    entry_price: float
    target_price: float
    stop_loss: float
    volume_confirmation: bool
    trading_direction: str = "long"  # "long", "short", "neutral"
    trend_duration_days: int = 0  # Durée de la tendance détectée
    trend_strength_score: float = 0.0  # Score de force de tendance (0-1)
    timeframe: str = "1d"
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    additional_data: Dict[str, Any] = field(default_factory=dict)

class TechnicalPatternDetector:
    """
    Détecteur de patterns techniques OHLCV pour pré-filtrer avant IA1
    """
    
    def __init__(self):
        self.coinapi_key = os.getenv('COINAPI_KEY')
        self.twelvedata_key = os.getenv('TWELVEDATA_KEY')
        self.min_pattern_strength = 0.6  # Seuil minimum pour déclencher IA1
        self.lookback_days = 30  # Période d'analyse
        
        logger.info("TechnicalPatternDetector initialized - Pre-filtering for IA1")
    
    async def should_analyze_with_ia1(self, symbol: str) -> Tuple[bool, Optional[TechnicalPattern]]:
        """
        Détermine si un crypto doit être analysé par IA1 basé sur des patterns techniques
        """
        try:
            # Récupère les données OHLCV
            ohlcv_data = await self._get_ohlcv_data(symbol)
            
            if ohlcv_data is None or len(ohlcv_data) < 20:
                logger.debug(f"Insufficient OHLCV data for {symbol}")
                return False, None
            
            # Détecte les patterns techniques
            patterns = self._detect_all_patterns(symbol, ohlcv_data)
            
            # Filtre les patterns significatifs
            strong_patterns = [p for p in patterns if p.strength >= self.min_pattern_strength]
            
            if strong_patterns:
                # Retourne le pattern le plus fort
                best_pattern = max(strong_patterns, key=lambda x: x.strength)
                logger.info(f"🎯 TECHNICAL FILTER: {symbol} - {best_pattern.pattern_type.value} (strength: {best_pattern.strength:.2f}) -> SENDING TO IA1")
                return True, best_pattern
            else:
                logger.debug(f"⚪ TECHNICAL FILTER: {symbol} - No strong patterns detected -> SKIPPING IA1")
                return False, None
                
        except Exception as e:
            logger.error(f"Error in technical pattern detection for {symbol}: {e}")
            # En cas d'erreur, on laisse passer pour éviter de bloquer
            return True, None
    
    async def _get_ohlcv_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Récupère les données OHLCV de multiples sources avec fallback
        """
        # Essaie CoinAPI en premier (meilleure qualité)
        data = await self._fetch_coinapi_ohlcv(symbol)
        if data is not None:
            return data
        
        # Fallback vers TwelveData
        data = await self._fetch_twelvedata_ohlcv(symbol)
        if data is not None:
            return data
        
        # Dernier recours : Yahoo Finance
        data = await self._fetch_yahoo_ohlcv(symbol)
        if data is not None:
            return data
        
        logger.warning(f"Failed to get OHLCV data for {symbol} from all sources")
        return None
    
    async def _fetch_coinapi_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        """Récupère OHLCV depuis CoinAPI"""
        if not self.coinapi_key:
            return None
        
        try:
            # Format du symbole pour CoinAPI
            coinapi_symbol = f"{symbol.replace('USDT', '')}_USDT"
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=self.lookback_days)
            
            url = f"https://rest.coinapi.io/v1/ohlcv/{coinapi_symbol}/history"
            params = {
                "period_id": "1DAY",
                "time_start": start_time.isoformat(),
                "time_end": end_time.isoformat(),
                "limit": self.lookback_days
            }
            headers = {"X-CoinAPI-Key": self.coinapi_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_coinapi_ohlcv(data)
                    else:
                        logger.debug(f"CoinAPI OHLCV failed for {symbol}: {response.status}")
                        
        except Exception as e:
            logger.debug(f"CoinAPI OHLCV error for {symbol}: {e}")
        
        return None
    
    async def _fetch_twelvedata_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        """Récupère OHLCV depuis TwelveData"""
        if not self.twelvedata_key:
            return None
        
        try:
            # Format du symbole pour TwelveData
            td_symbol = symbol.replace('USDT', '/USDT')
            
            url = "https://api.twelvedata.com/time_series"
            params = {
                "symbol": td_symbol,
                "interval": "1day",
                "outputsize": str(self.lookback_days),
                "apikey": self.twelvedata_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_twelvedata_ohlcv(data)
                    else:
                        logger.debug(f"TwelveData OHLCV failed for {symbol}: {response.status}")
                        
        except Exception as e:
            logger.debug(f"TwelveData OHLCV error for {symbol}: {e}")
        
        return None
    
    async def _fetch_yahoo_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        """Récupère OHLCV depuis Yahoo Finance"""
        try:
            loop = asyncio.get_event_loop()
            yf_symbol = symbol.replace('USDT', '-USD')
            
            ticker = await loop.run_in_executor(None, yf.Ticker, yf_symbol)
            hist = await loop.run_in_executor(None, lambda: ticker.history(period=f"{self.lookback_days}d"))
            
            if not hist.empty:
                # Normalise les colonnes
                hist.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                return hist
                
        except Exception as e:
            logger.debug(f"Yahoo Finance OHLCV error for {symbol}: {e}")
        
        return None
    
    def _parse_coinapi_ohlcv(self, data: List[Dict]) -> pd.DataFrame:
        """Parse les données OHLCV de CoinAPI"""
        df_data = []
        for item in data:
            df_data.append({
                'timestamp': pd.to_datetime(item['time_period_start']),
                'Open': float(item['price_open']),
                'High': float(item['price_high']),
                'Low': float(item['price_low']),
                'Close': float(item['price_close']),
                'Volume': float(item['volume_traded'])
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def _parse_twelvedata_ohlcv(self, data: Dict) -> pd.DataFrame:
        """Parse les données OHLCV de TwelveData"""
        if 'values' not in data:
            return pd.DataFrame()
        
        df_data = []
        for item in data['values']:
            df_data.append({
                'timestamp': pd.to_datetime(item['datetime']),
                'Open': float(item['open']),
                'High': float(item['high']),
                'Low': float(item['low']),
                'Close': float(item['close']),
                'Volume': float(item['volume'])
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        return df.sort_index()
    
    def _detect_all_patterns(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte tous les patterns techniques incluant les tendances soutenues"""
        patterns = []
        
        # 1. Détection de tendances soutenues (nouvelles méthodes)
        patterns.extend(self._detect_sustained_trends(symbol, df))
        
        # 2. Détection d'alignements de moyennes mobiles multiples
        patterns.extend(self._detect_multiple_ma_alignment(symbol, df))
        
        # 3. Détection de canaux directionnels
        patterns.extend(self._detect_directional_channels(symbol, df))
        
        # 4. Détection de momentum continuation patterns
        patterns.extend(self._detect_momentum_continuation(symbol, df))
        
        # 5. Détection de tendances (Golden/Death Cross) - existant
        patterns.extend(self._detect_moving_average_patterns(symbol, df))
        
        # 6. Détection de breakouts - existant
        patterns.extend(self._detect_breakout_patterns(symbol, df))
        
        # 7. Détection de figures chartistes - existant
        patterns.extend(self._detect_chart_patterns(symbol, df))
        
        # 8. Détection de signaux volume - existant
        patterns.extend(self._detect_volume_patterns(symbol, df))
        
        # 9. Détection RSI/MACD - existant
        patterns.extend(self._detect_oscillator_patterns(symbol, df))
        
        return patterns
    
    def _detect_moving_average_patterns(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte les patterns de moyennes mobiles"""
        patterns = []
        
        # Calcule les moyennes mobiles
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        
        if len(df) < 50:
            return patterns
        
        # Golden Cross (MA20 croise au-dessus de MA50)
        ma20_above_ma50 = df['MA20'] > df['MA50']
        previous_below = df['MA20'].shift(1) <= df['MA50'].shift(1)
        golden_cross = ma20_above_ma50 & previous_below
        
        if golden_cross.iloc[-5:].any():  # Dans les 5 derniers jours
            strength = min((df['MA20'].iloc[-1] - df['MA50'].iloc[-1]) / df['Close'].iloc[-1] * 100, 1.0)
            patterns.append(TechnicalPattern(
                symbol=symbol,
                pattern_type=PatternType.GOLDEN_CROSS,
                confidence=0.8,
                strength=max(strength, 0.6),
                entry_price=df['Close'].iloc[-1],
                target_price=df['Close'].iloc[-1] * 1.05,
                stop_loss=df['MA20'].iloc[-1] * 0.95,
                volume_confirmation=self._check_volume_increase(df),
                trading_direction="long",  # Position LONG pour Golden Cross
                trend_duration_days=self._calculate_trend_duration(df, "bullish"),
                trend_strength_score=max(strength, 0.6),
                additional_data={'ma20': df['MA20'].iloc[-1], 'ma50': df['MA50'].iloc[-1]}
            ))
        
        # Death Cross (MA20 croise en-dessous de MA50)
        ma20_below_ma50 = df['MA20'] < df['MA50']
        previous_above = df['MA20'].shift(1) >= df['MA50'].shift(1)
        death_cross = ma20_below_ma50 & previous_above
        
        if death_cross.iloc[-5:].any():
            strength = min(abs(df['MA20'].iloc[-1] - df['MA50'].iloc[-1]) / df['Close'].iloc[-1] * 100, 1.0)
            patterns.append(TechnicalPattern(
                symbol=symbol,
                pattern_type=PatternType.DEATH_CROSS,
                confidence=0.8,
                strength=max(strength, 0.6),
                entry_price=df['Close'].iloc[-1],
                target_price=df['Close'].iloc[-1] * 0.95,
                stop_loss=df['MA20'].iloc[-1] * 1.05,
                volume_confirmation=self._check_volume_increase(df),
                additional_data={'ma20': df['MA20'].iloc[-1], 'ma50': df['MA50'].iloc[-1]}
            ))
        
        return patterns
    
    def _detect_breakout_patterns(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte les patterns de breakout"""
        patterns = []
        
        # Calcule résistance et support des 20 derniers jours
        lookback = 20
        if len(df) < lookback:
            return patterns
        
        recent_high = df['High'].iloc[-lookback:].max()
        recent_low = df['Low'].iloc[-lookback:].min()
        current_price = df['Close'].iloc[-1]
        current_volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].iloc[-10:].mean()
        
        # Breakout haussier (cassure de résistance)
        if current_price > recent_high * 1.01 and current_volume > avg_volume * 1.5:
            strength = min((current_price - recent_high) / recent_high * 10, 1.0)
            patterns.append(TechnicalPattern(
                symbol=symbol,
                pattern_type=PatternType.BULLISH_BREAKOUT,
                confidence=0.85,
                strength=max(strength, 0.7),
                entry_price=current_price,
                target_price=current_price * 1.08,
                stop_loss=recent_high * 0.98,
                volume_confirmation=True,
                additional_data={'resistance_level': recent_high, 'volume_ratio': current_volume/avg_volume}
            ))
        
        # Breakdown baissier (cassure de support)
        elif current_price < recent_low * 0.99 and current_volume > avg_volume * 1.5:
            strength = min((recent_low - current_price) / recent_low * 10, 1.0)
            patterns.append(TechnicalPattern(
                symbol=symbol,
                pattern_type=PatternType.BEARISH_BREAKDOWN,
                confidence=0.85,
                strength=max(strength, 0.7),
                entry_price=current_price,
                target_price=current_price * 0.92,
                stop_loss=recent_low * 1.02,
                volume_confirmation=True,
                additional_data={'support_level': recent_low, 'volume_ratio': current_volume/avg_volume}
            ))
        
        return patterns
    
    def _detect_chart_patterns(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte les figures chartistes"""
        patterns = []
        
        if len(df) < 15:
            return patterns
        
        # Triangle ascendant (résistance horizontale, support montant)
        recent_highs = df['High'].iloc[-15:]
        recent_lows = df['Low'].iloc[-15:]
        
        # Calcule les pentes
        high_slope = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
        low_slope = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
        
        # Triangle ascendant : résistance plate, support montant
        if abs(high_slope) < recent_highs.mean() * 0.001 and low_slope > 0:
            strength = min(low_slope / recent_lows.mean() * 1000, 1.0)
            if strength > 0.3:
                patterns.append(TechnicalPattern(
                    symbol=symbol,
                    pattern_type=PatternType.ASCENDING_TRIANGLE,
                    confidence=0.75,
                    strength=strength,
                    entry_price=df['Close'].iloc[-1],
                    target_price=recent_highs.max() * 1.03,
                    stop_loss=recent_lows.iloc[-5:].min() * 0.98,
                    volume_confirmation=self._check_volume_increase(df),
                    additional_data={'resistance': recent_highs.max(), 'support_slope': low_slope}
                ))
        
        return patterns
    
    def _detect_volume_patterns(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte les patterns de volume"""
        patterns = []
        
        if len(df) < 10:
            return patterns
        
        current_volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].iloc[-10:].mean()
        price_change = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]
        
        # Spike de volume avec mouvement de prix
        if current_volume > avg_volume * 2 and abs(price_change) > 0.03:
            strength = min(current_volume / avg_volume / 5, 1.0)
            patterns.append(TechnicalPattern(
                symbol=symbol,
                pattern_type=PatternType.VOLUME_SPIKE,
                confidence=0.7,
                strength=strength,
                entry_price=df['Close'].iloc[-1],
                target_price=df['Close'].iloc[-1] * (1 + price_change * 2),
                stop_loss=df['Close'].iloc[-1] * (1 - abs(price_change)),
                volume_confirmation=True,
                additional_data={'volume_ratio': current_volume/avg_volume, 'price_change': price_change}
            ))
        
        return patterns
    
    def _detect_oscillator_patterns(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte les patterns RSI/MACD"""
        patterns = []
        
        if len(df) < 26:
            return patterns
        
        # Calcule RSI
        rsi = self._calculate_rsi(df['Close'])
        
        # Calcule MACD
        macd_line, macd_signal, macd_histogram = self._calculate_macd(df['Close'])
        
        # Signal MACD bullish (croisement haussier)
        if len(macd_histogram) >= 2:
            if macd_histogram.iloc[-1] > 0 and macd_histogram.iloc[-2] <= 0:
                strength = min(abs(macd_histogram.iloc[-1]) * 1000, 1.0)
                patterns.append(TechnicalPattern(
                    symbol=symbol,
                    pattern_type=PatternType.MACD_BULLISH,
                    confidence=0.75,
                    strength=max(strength, 0.5),
                    entry_price=df['Close'].iloc[-1],
                    target_price=df['Close'].iloc[-1] * 1.05,
                    stop_loss=df['Close'].iloc[-1] * 0.97,
                    volume_confirmation=self._check_volume_increase(df),
                    additional_data={'macd_histogram': macd_histogram.iloc[-1], 'rsi': rsi.iloc[-1]}
                ))
        
        return patterns
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcule RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calcule MACD"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd_line = exp1 - exp2
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    
    def _check_volume_increase(self, df: pd.DataFrame) -> bool:
        """Vérifie si le volume augmente"""
        if len(df) < 5:
            return False
        
        recent_volume = df['Volume'].iloc[-3:].mean()
        previous_volume = df['Volume'].iloc[-10:-3].mean()
        
        return recent_volume > previous_volume * 1.2

    def _detect_sustained_trends(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte les tendances soutenues pour positions long/short"""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        # Calcule les moyennes mobiles pour analyse de tendance
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        
        # Analyse la pente des moyennes mobiles
        ma5_slope = self._calculate_slope(df['MA5'].iloc[-10:])
        ma10_slope = self._calculate_slope(df['MA10'].iloc[-10:])
        ma20_slope = self._calculate_slope(df['MA20'].iloc[-15:])
        
        current_price = df['Close'].iloc[-1]
        
        # Tendance haussière soutenue (LONG)
        bullish_conditions = [
            ma5_slope > 0,  # MA5 en hausse
            ma10_slope > 0,  # MA10 en hausse
            ma20_slope > 0,  # MA20 en hausse
            current_price > df['MA5'].iloc[-1],  # Prix au-dessus MA5
            df['MA5'].iloc[-1] > df['MA10'].iloc[-1],  # MA5 > MA10
            df['MA10'].iloc[-1] > df['MA20'].iloc[-1],  # MA10 > MA20
        ]
        
        if sum(bullish_conditions) >= 5:  # Au moins 5 conditions sur 6
            # Calcule la force de la tendance
            trend_strength = min((ma5_slope + ma10_slope + ma20_slope) * 1000, 1.0)
            trend_duration = self._calculate_trend_duration(df, "bullish")
            
            patterns.append(TechnicalPattern(
                symbol=symbol,
                pattern_type=PatternType.SUSTAINED_BULLISH_TREND,
                confidence=0.85,
                strength=max(trend_strength, 0.7),
                entry_price=current_price,
                target_price=current_price * 1.08,  # 8% target
                stop_loss=df['MA10'].iloc[-1] * 0.97,  # Stop sous MA10
                volume_confirmation=self._check_volume_increase(df),
                trading_direction="long",
                trend_duration_days=trend_duration,
                trend_strength_score=trend_strength,
                additional_data={
                    'ma5_slope': ma5_slope,
                    'ma10_slope': ma10_slope,
                    'ma20_slope': ma20_slope,
                    'ma_alignment': 'bullish'
                }
            ))
        
        # Tendance baissière soutenue (SHORT)
        bearish_conditions = [
            ma5_slope < 0,  # MA5 en baisse
            ma10_slope < 0,  # MA10 en baisse
            ma20_slope < 0,  # MA20 en baisse
            current_price < df['MA5'].iloc[-1],  # Prix en-dessous MA5
            df['MA5'].iloc[-1] < df['MA10'].iloc[-1],  # MA5 < MA10
            df['MA10'].iloc[-1] < df['MA20'].iloc[-1],  # MA10 < MA20
        ]
        
        if sum(bearish_conditions) >= 5:  # Au moins 5 conditions sur 6
            # Calcule la force de la tendance baissière
            trend_strength = min(abs(ma5_slope + ma10_slope + ma20_slope) * 1000, 1.0)
            trend_duration = self._calculate_trend_duration(df, "bearish")
            
            patterns.append(TechnicalPattern(
                symbol=symbol,
                pattern_type=PatternType.SUSTAINED_BEARISH_TREND,
                confidence=0.85,
                strength=max(trend_strength, 0.7),
                entry_price=current_price,
                target_price=current_price * 0.92,  # -8% target
                stop_loss=df['MA10'].iloc[-1] * 1.03,  # Stop au-dessus MA10
                volume_confirmation=self._check_volume_increase(df),
                trading_direction="short",
                trend_duration_days=trend_duration,
                trend_strength_score=trend_strength,
                additional_data={
                    'ma5_slope': ma5_slope,
                    'ma10_slope': ma10_slope,
                    'ma20_slope': ma20_slope,
                    'ma_alignment': 'bearish'
                }
            ))
        
        return patterns

    def _detect_multiple_ma_alignment(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte l'alignement de multiples moyennes mobiles"""
        patterns = []
        
        if len(df) < 50:
            return patterns
        
        # Calcule plusieurs moyennes mobiles
        df['MA8'] = df['Close'].rolling(8).mean()
        df['MA13'] = df['Close'].rolling(13).mean()
        df['MA21'] = df['Close'].rolling(21).mean()
        df['MA34'] = df['Close'].rolling(34).mean()
        
        current_price = df['Close'].iloc[-1]
        
        # Alignement bullish (toutes les MA en ordre croissant)
        bullish_alignment = (
            current_price > df['MA8'].iloc[-1] > df['MA13'].iloc[-1] > 
            df['MA21'].iloc[-1] > df['MA34'].iloc[-1]
        )
        
        if bullish_alignment:
            # Calcule l'écartement des MA (plus grand = plus fort)
            ma_spread = (df['MA8'].iloc[-1] - df['MA34'].iloc[-1]) / df['MA34'].iloc[-1]
            strength = min(ma_spread * 20, 1.0)
            
            patterns.append(TechnicalPattern(
                symbol=symbol,
                pattern_type=PatternType.MULTIPLE_MA_BULLISH_ALIGNMENT,
                confidence=0.9,
                strength=max(strength, 0.75),
                entry_price=current_price,
                target_price=current_price * 1.06,
                stop_loss=df['MA13'].iloc[-1] * 0.98,
                volume_confirmation=self._check_volume_increase(df),
                trading_direction="long",
                trend_duration_days=self._calculate_trend_duration(df, "bullish"),
                trend_strength_score=strength,
                additional_data={'ma_spread': ma_spread, 'alignment': 'perfect_bullish'}
            ))
        
        # Alignement bearish (toutes les MA en ordre décroissant)
        bearish_alignment = (
            current_price < df['MA8'].iloc[-1] < df['MA13'].iloc[-1] < 
            df['MA21'].iloc[-1] < df['MA34'].iloc[-1]
        )
        
        if bearish_alignment:
            # Calcule l'écartement des MA
            ma_spread = (df['MA34'].iloc[-1] - df['MA8'].iloc[-1]) / df['MA8'].iloc[-1]
            strength = min(ma_spread * 20, 1.0)
            
            patterns.append(TechnicalPattern(
                symbol=symbol,
                pattern_type=PatternType.MULTIPLE_MA_BEARISH_ALIGNMENT,
                confidence=0.9,
                strength=max(strength, 0.75),
                entry_price=current_price,
                target_price=current_price * 0.94,
                stop_loss=df['MA13'].iloc[-1] * 1.02,
                volume_confirmation=self._check_volume_increase(df),
                trading_direction="short",
                trend_duration_days=self._calculate_trend_duration(df, "bearish"),
                trend_strength_score=strength,
                additional_data={'ma_spread': ma_spread, 'alignment': 'perfect_bearish'}
            ))
        
        return patterns

    def _detect_directional_channels(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte les canaux directionnels (bullish/bearish channels)"""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        # Analyse les 20 derniers jours
        recent_highs = df['High'].iloc[-20:]
        recent_lows = df['Low'].iloc[-20:]
        current_price = df['Close'].iloc[-1]
        
        # Calcule les lignes de tendance
        high_slope = self._calculate_slope(recent_highs)
        low_slope = self._calculate_slope(recent_lows)
        
        # Canal haussier : support et résistance en hausse
        if high_slope > 0 and low_slope > 0 and abs(high_slope - low_slope) < high_slope * 0.5:
            channel_strength = min((high_slope + low_slope) * 1000, 1.0)
            
            patterns.append(TechnicalPattern(
                symbol=symbol,
                pattern_type=PatternType.BULLISH_CHANNEL,
                confidence=0.8,
                strength=max(channel_strength, 0.6),
                entry_price=current_price,
                target_price=recent_highs.max() + (recent_highs.max() - recent_lows.min()) * 0.3,
                stop_loss=recent_lows.iloc[-5:].min() * 0.98,
                volume_confirmation=self._check_volume_increase(df),
                trading_direction="long",
                trend_duration_days=self._calculate_trend_duration(df, "bullish"),
                trend_strength_score=channel_strength,
                additional_data={
                    'high_slope': high_slope,
                    'low_slope': low_slope,
                    'channel_width': recent_highs.max() - recent_lows.min()
                }
            ))
        
        # Canal baissier : support et résistance en baisse
        elif high_slope < 0 and low_slope < 0 and abs(high_slope - low_slope) < abs(high_slope) * 0.5:
            channel_strength = min(abs(high_slope + low_slope) * 1000, 1.0)
            
            patterns.append(TechnicalPattern(
                symbol=symbol,
                pattern_type=PatternType.BEARISH_CHANNEL,
                confidence=0.8,
                strength=max(channel_strength, 0.6),
                entry_price=current_price,
                target_price=recent_lows.min() - (recent_highs.max() - recent_lows.min()) * 0.3,
                stop_loss=recent_highs.iloc[-5:].max() * 1.02,
                volume_confirmation=self._check_volume_increase(df),
                trading_direction="short",
                trend_duration_days=self._calculate_trend_duration(df, "bearish"),
                trend_strength_score=channel_strength,
                additional_data={
                    'high_slope': high_slope,
                    'low_slope': low_slope,
                    'channel_width': recent_highs.max() - recent_lows.min()
                }
            ))
        
        return patterns

    def _detect_momentum_continuation(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte les patterns de continuation de momentum"""
        patterns = []
        
        if len(df) < 15:
            return patterns
        
        # Calcule le momentum sur différentes périodes
        momentum_5d = (df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6]
        momentum_10d = (df['Close'].iloc[-1] - df['Close'].iloc[-11]) / df['Close'].iloc[-11]
        
        current_price = df['Close'].iloc[-1]
        volume_ratio = df['Volume'].iloc[-3:].mean() / df['Volume'].iloc[-10:-3].mean()
        
        # Momentum bullish continuation
        if momentum_5d > 0.02 and momentum_10d > 0.03 and volume_ratio > 1.2:
            momentum_strength = min((momentum_5d + momentum_10d) * 10, 1.0)
            
            patterns.append(TechnicalPattern(
                symbol=symbol,
                pattern_type=PatternType.BULLISH_MOMENTUM_CONTINUATION,
                confidence=0.75,
                strength=max(momentum_strength, 0.65),
                entry_price=current_price,
                target_price=current_price * (1 + momentum_10d * 0.8),
                stop_loss=current_price * (1 - momentum_10d * 0.3),
                volume_confirmation=True,
                trading_direction="long",
                trend_duration_days=self._calculate_trend_duration(df, "bullish"),
                trend_strength_score=momentum_strength,
                additional_data={
                    'momentum_5d': momentum_5d,
                    'momentum_10d': momentum_10d,
                    'volume_ratio': volume_ratio
                }
            ))
        
        # Momentum bearish continuation
        elif momentum_5d < -0.02 and momentum_10d < -0.03 and volume_ratio > 1.2:
            momentum_strength = min(abs(momentum_5d + momentum_10d) * 10, 1.0)
            
            patterns.append(TechnicalPattern(
                symbol=symbol,
                pattern_type=PatternType.BEARISH_MOMENTUM_CONTINUATION,
                confidence=0.75,
                strength=max(momentum_strength, 0.65),
                entry_price=current_price,
                target_price=current_price * (1 + momentum_10d * 0.8),  # momentum_10d est négatif
                stop_loss=current_price * (1 - momentum_10d * 0.3),
                volume_confirmation=True,
                trading_direction="short",
                trend_duration_days=self._calculate_trend_duration(df, "bearish"),
                trend_strength_score=momentum_strength,
                additional_data={
                    'momentum_5d': momentum_5d,
                    'momentum_10d': momentum_10d,
                    'volume_ratio': volume_ratio
                }
            ))
        
        return patterns

    def _calculate_slope(self, series: pd.Series) -> float:
        """Calcule la pente d'une série temporelle"""
        if len(series) < 2:
            return 0.0
        
        x = np.arange(len(series))
        y = series.values
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalise par rapport à la valeur moyenne
        return slope / series.mean() if series.mean() != 0 else 0.0

    def _calculate_trend_duration(self, df: pd.DataFrame, trend_type: str) -> int:
        """Calcule la durée de la tendance en jours"""
        if len(df) < 10:
            return 0
        
        df['MA10'] = df['Close'].rolling(10).mean()
        
        duration = 0
        if trend_type == "bullish":
            # Compte les jours consécutifs où prix > MA10 et MA10 en hausse
            for i in range(len(df) - 1, 0, -1):
                if (df['Close'].iloc[i] > df['MA10'].iloc[i] and 
                    df['MA10'].iloc[i] > df['MA10'].iloc[i-1]):
                    duration += 1
                else:
                    break
        elif trend_type == "bearish":
            # Compte les jours consécutifs où prix < MA10 et MA10 en baisse
            for i in range(len(df) - 1, 0, -1):
                if (df['Close'].iloc[i] < df['MA10'].iloc[i] and 
                    df['MA10'].iloc[i] < df['MA10'].iloc[i-1]):
                    duration += 1
                else:
                    break
        
        return duration

# Global instance
technical_pattern_detector = TechnicalPatternDetector()
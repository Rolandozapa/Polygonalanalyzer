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
    # Patterns existants
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
    
    # Tendances soutenues
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
    
    # NOUVELLES FIGURES CHARTISTES CLASSIQUES
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    FLAG_BULLISH = "flag_bullish"
    FLAG_BEARISH = "flag_bearish"
    PENNANT_BULLISH = "pennant_bullish"
    PENNANT_BEARISH = "pennant_bearish"
    CUP_AND_HANDLE = "cup_and_handle"
    RISING_WEDGE = "rising_wedge"
    FALLING_WEDGE = "falling_wedge"
    RECTANGLE_CONSOLIDATION = "rectangle_consolidation"
    ROUNDING_BOTTOM = "rounding_bottom"
    ROUNDING_TOP = "rounding_top"
    
    # PATTERNS HARMONIQUES AVANCÉS
    GARTLEY_BULLISH = "gartley_bullish"
    GARTLEY_BEARISH = "gartley_bearish"
    BAT_BULLISH = "bat_bullish"
    BAT_BEARISH = "bat_bearish"
    BUTTERFLY_BULLISH = "butterfly_bullish"
    BUTTERFLY_BEARISH = "butterfly_bearish"
    CRAB_BULLISH = "crab_bullish"
    CRAB_BEARISH = "crab_bearish"
    
    # PATTERNS DE VOLATILITÉ
    DIAMOND_TOP = "diamond_top"
    DIAMOND_BOTTOM = "diamond_bottom"
    EXPANDING_WEDGE = "expanding_wedge"
    CONTRACTING_WEDGE = "contracting_wedge"
    HORN_TOP = "horn_top"
    HORN_BOTTOM = "horn_bottom"

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
        self.min_pattern_strength = 0.25  # Seuil encore plus bas pour détecter les figures chartistes subtiles
        self.lookback_days = 30  # Période d'analyse
        
        # Compteurs pour APIs gratuites (optimisation selon générosité)
        self.daily_counters = {
            'binance': 0,      # 1,200 req/min - ULTRA GÉNÉREUX
            'coingecko': 0,    # 10,000 req/mois - TRÈS GÉNÉREUX  
            'cryptocompare': 0, # 100,000 req/mois - GÉNÉREUX
            'yahoo': 0         # Pas de limite connue
        }
        
        logger.info("TechnicalPatternDetector initialized - Optimized for most generous APIs")
        logger.info("Priority: Binance (1200/min) > CoinGecko (10k/month) > CryptoCompare (100k/month)")
    
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
        Récupère les données OHLCV des APIs les plus généreuses en priorité
        Ordre optimisé selon générosité des limites : Binance > CoinGecko > CryptoCompare > CoinAPI > TwelveData
        """
        # 1. BINANCE API (1,200 req/min - ULTRA GÉNÉREUX!)
        data = await self._fetch_binance_ohlcv(symbol)
        if data is not None:
            return data
        
        # 2. CoinGecko (10,000 appels/mois, 10 req/sec - TRÈS GÉNÉREUX)
        data = await self._fetch_coingecko_ohlcv(symbol)
        if data is not None:
            return data
        
        # 3. CryptoCompare (100,000 appels/mois mais limite req/minute)
        data = await self._fetch_cryptocompare_ohlcv(symbol)
        if data is not None:
            return data
        
        # 4. CoinAPI (payant mais fiable - en backup)
        data = await self._fetch_coinapi_ohlcv(symbol)
        if data is not None:
            return data
        
        # 5. TwelveData (payant - en backup)
        data = await self._fetch_twelvedata_ohlcv(symbol)
        if data is not None:
            return data
        
        # 6. Yahoo Finance (gratuit - dernier recours)
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
        """Détecte tous les patterns techniques disponibles"""
        all_patterns = []
        
        try:
            # Patterns de moyennes mobiles
            all_patterns.extend(self._detect_moving_average_patterns(symbol, df))
            
            # Patterns de breakout/breakdown
            all_patterns.extend(self._detect_breakout_patterns(symbol, df))
            
            # Patterns de volume
            all_patterns.extend(self._detect_volume_patterns(symbol, df))
            
            # Patterns d'oscillateurs
            all_patterns.extend(self._detect_oscillator_patterns(symbol, df))
            
            # Patterns de tendances soutenues
            all_patterns.extend(self._detect_sustained_trends(symbol, df))
            
            # Alignement des moyennes mobiles multiples
            all_patterns.extend(self._detect_multiple_ma_alignment(symbol, df))
            
            # Canaux directionnels
            all_patterns.extend(self._detect_directional_channels(symbol, df))
            
            # Continuation de momentum
            all_patterns.extend(self._detect_momentum_continuation(symbol, df))
            
            # PATTERNS CHARTISTES CLASSIQUES
            all_patterns.extend(self._detect_classic_chart_patterns(symbol, df))
            
            # PATTERNS DE RETOURNEMENT (Head & Shoulders, Double Top/Bottom, Triple)
            all_patterns.extend(self._detect_reversal_patterns(symbol, df))
            
            # PATTERNS TRIANGULAIRES ET WEDGES
            all_patterns.extend(self._detect_triangle_wedge_patterns(symbol, df))
            
            # PATTERNS DE CONTINUATION (Flags, Pennants)
            all_patterns.extend(self._detect_flag_pennant_patterns(symbol, df))
            
            # PATTERNS HARMONIQUES AVANCÉS (Gartley, Bat, Butterfly, Crab)
            all_patterns.extend(self._detect_harmonic_patterns(symbol, df))
            
            # PATTERNS DE VOLATILITÉ (Diamond, Expanding Wedge)
            all_patterns.extend(self._detect_diamond_patterns(symbol, df))
            all_patterns.extend(self._detect_expanding_wedge_patterns(symbol, df))
            
            # Filtrer par force minimale et éviter les doublons
            filtered_patterns = []
            seen_types = set()
            
            for pattern in sorted(all_patterns, key=lambda x: x.strength, reverse=True):
                # Garder seulement les patterns uniques par type avec la plus haute force
                if pattern.pattern_type not in seen_types and pattern.strength >= self.min_pattern_strength:
                    filtered_patterns.append(pattern)
                    seen_types.add(pattern.pattern_type)
            
            logger.info(f"Detected {len(filtered_patterns)} strong patterns for {symbol}: {[p.pattern_type.value for p in filtered_patterns]}")
            
        except Exception as e:
            logger.error(f"Error in pattern detection for {symbol}: {e}")
            
        return filtered_patterns[:5]  # Retourner max 5 patterns les plus forts
    
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
                trading_direction="short",  # Position SHORT pour Death Cross
                trend_duration_days=self._calculate_trend_duration(df, "bearish"),
                trend_strength_score=max(strength, 0.6),
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
                trading_direction="long",  # Position LONG pour breakout haussier
                trend_duration_days=self._calculate_trend_duration(df, "bullish"),
                trend_strength_score=max(strength, 0.7),
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
                trading_direction="short",  # Position SHORT pour breakdown baissier
                trend_duration_days=self._calculate_trend_duration(df, "bearish"),
                trend_strength_score=max(strength, 0.7),
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
        
        # Calcule le momentum sur différentes périodes - ROBUST: Prevent all mathematical errors
        if len(df) < 12:  # Need at least 12 data points
            return patterns
            
        close_6_ago = df['Close'].iloc[-6] if len(df) > 6 else df['Close'].iloc[0]
        close_11_ago = df['Close'].iloc[-11] if len(df) > 11 else df['Close'].iloc[0]
        current_close = df['Close'].iloc[-1]
        
        # Validate all values are finite and positive
        if not pd.notna([current_close, close_6_ago, close_11_ago]).all() or \
           not all(val > 0 for val in [current_close, close_6_ago, close_11_ago]):
            return patterns
        
        momentum_5d = (current_close - close_6_ago) / close_6_ago if close_6_ago > 0 else 0
        momentum_10d = (current_close - close_11_ago) / close_11_ago if close_11_ago > 0 else 0
        
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

    def _detect_classic_chart_patterns(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte les figures chartistes classiques"""
        patterns = []
        
        if len(df) < 30:
            return patterns
        
        # Cup and Handle pattern
        patterns.extend(self._detect_cup_and_handle(symbol, df))
        
        # Rectangle consolidation
        patterns.extend(self._detect_rectangle_consolidation(symbol, df))
        
        # Rounding patterns
        patterns.extend(self._detect_rounding_patterns(symbol, df))
        
        return patterns

    def _detect_reversal_patterns(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte les patterns de retournement (Head & Shoulders, Double Top/Bottom)"""
        patterns = []
        
        if len(df) < 25:
            return patterns
        
        # Head and Shoulders pattern
        patterns.extend(self._detect_head_and_shoulders(symbol, df))
        
        # Double Top/Bottom patterns
        patterns.extend(self._detect_double_patterns(symbol, df))
        
        # Triple Top/Bottom patterns
        patterns.extend(self._detect_triple_patterns(symbol, df))
        
        return patterns

    def _detect_triangle_wedge_patterns(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte les triangles et wedges"""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        # Triangles symétriques
        patterns.extend(self._detect_symmetrical_triangle(symbol, df))
        
        # Triangles ascendants/descendants (améliorer l'existant)
        patterns.extend(self._detect_enhanced_triangles(symbol, df))
        
        # Rising/Falling Wedges
        patterns.extend(self._detect_wedge_patterns(symbol, df))
        
        return patterns

    def _detect_flag_pennant_patterns(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte les flags et pennants (patterns de continuation)"""
        patterns = []
        
        if len(df) < 15:
            return patterns
        
        # Flag patterns
        patterns.extend(self._detect_flag_patterns(symbol, df))
        
        # Pennant patterns
        patterns.extend(self._detect_pennant_patterns(symbol, df))
        
        return patterns

    def _detect_head_and_shoulders(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte Head and Shoulders et Inverse Head and Shoulders"""
        patterns = []
        
        try:
            highs = df['High'].rolling(3, center=True).max() == df['High']
            lows = df['Low'].rolling(3, center=True).min() == df['Low']
            
            current_price = df['Close'].iloc[-1]
            
            # Head and Shoulders (bearish reversal)
            high_peaks = df[highs]['High'].iloc[-15:]  # Derniers 15 pics
            if len(high_peaks) >= 3:
                # Cherche pattern: épaule gauche < tête > épaule droite
                for i in range(len(high_peaks) - 2):
                    left_shoulder = high_peaks.iloc[i]
                    head = high_peaks.iloc[i + 1]
                    right_shoulder = high_peaks.iloc[i + 2]
                    
                    if (left_shoulder < head > right_shoulder and 
                        abs(left_shoulder - right_shoulder) / head < 0.03):  # Épaules similaires
                        
                        neckline = min(left_shoulder, right_shoulder) * 0.98
                        strength = min((head - neckline) / neckline * 5, 1.0)
                        
                        if strength > 0.4:
                            patterns.append(TechnicalPattern(
                                symbol=symbol,
                                pattern_type=PatternType.HEAD_AND_SHOULDERS,
                                confidence=0.85,
                                strength=strength,
                                entry_price=current_price,
                                target_price=neckline * 0.95,
                                stop_loss=head * 1.02,
                                volume_confirmation=self._check_volume_increase(df),
                                trading_direction="short",
                                trend_duration_days=self._calculate_trend_duration(df, "bearish"),
                                trend_strength_score=strength,
                                additional_data={
                                    'head_price': head,
                                    'left_shoulder': left_shoulder,
                                    'right_shoulder': right_shoulder,
                                    'neckline': neckline
                                }
                            ))

            # Inverse Head and Shoulders (bullish reversal)
            low_troughs = df[lows]['Low'].iloc[-15:]  # Derniers 15 creux
            if len(low_troughs) >= 3:
                for i in range(len(low_troughs) - 2):
                    left_shoulder = low_troughs.iloc[i]
                    head = low_troughs.iloc[i + 1]
                    right_shoulder = low_troughs.iloc[i + 2]
                    
                    if (left_shoulder > head < right_shoulder and 
                        abs(left_shoulder - right_shoulder) / head < 0.03):
                        
                        neckline = max(left_shoulder, right_shoulder) * 1.02
                        strength = min((neckline - head) / head * 5, 1.0)
                        
                        if strength > 0.4:
                            patterns.append(TechnicalPattern(
                                symbol=symbol,
                                pattern_type=PatternType.INVERSE_HEAD_AND_SHOULDERS,
                                confidence=0.85,
                                strength=strength,
                                entry_price=current_price,
                                target_price=neckline * 1.05,
                                stop_loss=head * 0.98,
                                volume_confirmation=self._check_volume_increase(df),
                                trading_direction="long",
                                trend_duration_days=self._calculate_trend_duration(df, "bullish"),
                                trend_strength_score=strength,
                                additional_data={
                                    'head_price': head,
                                    'left_shoulder': left_shoulder,
                                    'right_shoulder': right_shoulder,
                                    'neckline': neckline
                                }
                            ))

        except Exception as e:
            logger.debug(f"Head and Shoulders detection error for {symbol}: {e}")

        return patterns

    def _detect_double_patterns(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte Double Top et Double Bottom"""
        patterns = []
        
        try:
            current_price = df['Close'].iloc[-1]
            
            # Double Top (bearish reversal)
            highs = df['High'].rolling(5, center=True).max() == df['High']
            high_peaks = df[highs]['High'].iloc[-10:]
            
            if len(high_peaks) >= 2:
                for i in range(len(high_peaks) - 1):
                    peak1 = high_peaks.iloc[i]
                    peak2 = high_peaks.iloc[i + 1]
                    
                    # Double top : deux pics similaires - ROBUST: Prevent all mathematical errors
                    if not pd.notna([peak1, peak2]).all() or peak1 <= 0 or peak2 <= 0:
                        continue
                        
                    peak_max = max(peak1, peak2)
                    if peak_max > 0 and abs(peak1 - peak2) / peak_max < 0.02:  # Différence < 2%
                        valley = df['Low'][df.index > high_peaks.index[i]].min()
                        
                        # Validate valley value
                        if not pd.notna(valley) or valley <= 0:
                            continue
                            
                        strength = min((max(peak1, peak2) - valley) / valley * 3, 1.0)
                        
                        if strength > 0.5:
                            patterns.append(TechnicalPattern(
                                symbol=symbol,
                                pattern_type=PatternType.DOUBLE_TOP,
                                confidence=0.8,
                                strength=strength,
                                entry_price=current_price,
                                target_price=valley * 0.97,
                                stop_loss=max(peak1, peak2) * 1.02,
                                volume_confirmation=self._check_volume_increase(df),
                                trading_direction="short",
                                trend_duration_days=self._calculate_trend_duration(df, "bearish"),
                                trend_strength_score=strength,
                                additional_data={'peak1': peak1, 'peak2': peak2, 'valley': valley}
                            ))

            # Double Bottom (bullish reversal)
            lows = df['Low'].rolling(5, center=True).min() == df['Low']
            low_troughs = df[lows]['Low'].iloc[-10:]
            
            if len(low_troughs) >= 2:
                for i in range(len(low_troughs) - 1):
                    trough1 = low_troughs.iloc[i]
                    trough2 = low_troughs.iloc[i + 1]
                    
                    # ROBUST: Prevent all mathematical errors for double bottom
                    if not pd.notna([trough1, trough2]).all() or trough1 <= 0 or trough2 <= 0:
                        continue
                        
                    trough_min = min(trough1, trough2)
                    if trough_min > 0 and abs(trough1 - trough2) / trough_min < 0.02:
                        peak = df['High'][df.index > low_troughs.index[i]].max()
                        
                        # Validate peak value
                        if not pd.notna(peak) or peak <= 0:
                            continue
                            
                        strength = min((peak - min(trough1, trough2)) / min(trough1, trough2) * 3, 1.0)
                        
                        if strength > 0.5:
                            patterns.append(TechnicalPattern(
                                symbol=symbol,
                                pattern_type=PatternType.DOUBLE_BOTTOM,
                                confidence=0.8,
                                strength=strength,
                                entry_price=current_price,
                                target_price=peak * 1.03,
                                stop_loss=min(trough1, trough2) * 0.98,
                                volume_confirmation=self._check_volume_increase(df),
                                trading_direction="long",
                                trend_duration_days=self._calculate_trend_duration(df, "bullish"),
                                trend_strength_score=strength,
                                additional_data={'trough1': trough1, 'trough2': trough2, 'peak': peak}
                            ))

        except Exception as e:
            logger.debug(f"Double pattern detection error for {symbol}: {e}")

        return patterns

    def _detect_flag_patterns(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte les patterns Flag (continuation)"""
        patterns = []
        
        try:
            if len(df) < 15:
                return patterns
            
            current_price = df['Close'].iloc[-1]
            
            # Flag pattern : forte hausse/baisse suivie d'une consolidation rectangulaire
            price_change_pre = (df['Close'].iloc[-10] - df['Close'].iloc[-15]) / df['Close'].iloc[-15]
            
            # Consolidation récente (5 derniers jours)
            recent_highs = df['High'].iloc[-5:].max()
            recent_lows = df['Low'].iloc[-5:].min()
            consolidation_range = (recent_highs - recent_lows) / recent_lows
            
            # Bullish Flag : forte hausse puis consolidation plate
            if price_change_pre > 0.05 and consolidation_range < 0.03:  # Hausse 5%+ puis consolidation 3%-
                strength = min(price_change_pre * 10, 1.0)
                patterns.append(TechnicalPattern(
                    symbol=symbol,
                    pattern_type=PatternType.FLAG_BULLISH,
                    confidence=0.75,
                    strength=strength,
                    entry_price=current_price,
                    target_price=current_price * (1 + price_change_pre * 0.8),
                    stop_loss=recent_lows * 0.98,
                    volume_confirmation=self._check_volume_increase(df),
                    trading_direction="long",
                    trend_duration_days=self._calculate_trend_duration(df, "bullish"),
                    trend_strength_score=strength,
                    additional_data={
                        'pre_move': price_change_pre,
                        'consolidation_range': consolidation_range,
                        'flag_high': recent_highs,
                        'flag_low': recent_lows
                    }
                ))
            
            # Bearish Flag : forte baisse puis consolidation plate
            elif price_change_pre < -0.05 and consolidation_range < 0.03:
                strength = min(abs(price_change_pre) * 10, 1.0)
                patterns.append(TechnicalPattern(
                    symbol=symbol,
                    pattern_type=PatternType.FLAG_BEARISH,
                    confidence=0.75,
                    strength=strength,
                    entry_price=current_price,
                    target_price=current_price * (1 + price_change_pre * 0.8),  # price_change_pre négatif
                    stop_loss=recent_highs * 1.02,
                    volume_confirmation=self._check_volume_increase(df),
                    trading_direction="short",
                    trend_duration_days=self._calculate_trend_duration(df, "bearish"),
                    trend_strength_score=strength,
                    additional_data={
                        'pre_move': price_change_pre,
                        'consolidation_range': consolidation_range,
                        'flag_high': recent_highs,
                        'flag_low': recent_lows
                    }
                ))

        except Exception as e:
            logger.debug(f"Flag pattern detection error for {symbol}: {e}")

        return patterns

    def _detect_cup_and_handle(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte le pattern Cup and Handle (bullish)"""
        patterns = []
        
        try:
            if len(df) < 30:
                return patterns
            
            current_price = df['Close'].iloc[-1]
            prices = df['Close'].iloc[-30:]
            
            # Cup : U-shape sur 20-25 jours, Handle : petite consolidation sur 5-10 jours
            # ROBUST FIX: Check for valid data before calculations
            if len(prices) < 30:
                return patterns
                
            cup_start = prices.iloc[0]
            cup_low = prices.iloc[5:25].min()
            cup_end = prices.iloc[25] if len(prices) > 25 else prices.iloc[-1]
            handle_low = prices.iloc[25:].min() if len(prices) > 25 else cup_low
            
            # Validate all values are finite and positive
            if not all(pd.notna([cup_start, cup_low, cup_end, handle_low])) or \
               not all(val > 0 for val in [cup_start, cup_low, cup_end, handle_low]):
                return patterns
            
            # Conditions Cup and Handle - ROBUST: Prevent all mathematical errors
            cup_depth = (cup_start - cup_low) / cup_start if cup_start > 0 else 0
            handle_depth = (cup_end - handle_low) / cup_end if cup_end > 0 else 0
            
            if (0.1 < cup_depth < 0.3 and  # Cup depth 10-30%
                handle_depth < cup_depth * 0.3 and  # Handle shallow vs cup
                current_price > cup_start * 0.95):  # Prix proche du cup rim
                
                strength = min(cup_depth * 3, 1.0)
                patterns.append(TechnicalPattern(
                    symbol=symbol,
                    pattern_type=PatternType.CUP_AND_HANDLE,
                    confidence=0.8,
                    strength=strength,
                    entry_price=current_price,
                    target_price=cup_start * (1 + cup_depth),  # Measured move
                    stop_loss=handle_low * 0.98,
                    volume_confirmation=self._check_volume_increase(df),
                    trading_direction="long",
                    trend_duration_days=self._calculate_trend_duration(df, "bullish"),
                    trend_strength_score=strength,
                    additional_data={
                        'cup_start': cup_start,
                        'cup_low': cup_low,
                        'cup_depth': cup_depth,
                        'handle_low': handle_low,
                        'handle_depth': handle_depth
                    }
                ))

        except Exception as e:
            logger.debug(f"Cup and Handle detection error for {symbol}: {e}")

        return patterns

    def _detect_wedge_patterns(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte Rising et Falling Wedges"""
        patterns = []
        
        try:
            if len(df) < 20:
                return patterns
            
            current_price = df['Close'].iloc[-1]
            recent_highs = df['High'].iloc[-15:]
            recent_lows = df['Low'].iloc[-15:]
            
            # Calcule les tendances des highs et lows
            high_slope = self._calculate_slope(recent_highs)
            low_slope = self._calculate_slope(recent_lows)
            
            # Rising Wedge : highs et lows montent, mais highs slope < lows slope (bearish)
            if high_slope > 0 and low_slope > 0 and low_slope > high_slope * 1.2:
                strength = min(abs(high_slope - low_slope) * 1000, 1.0)
                patterns.append(TechnicalPattern(
                    symbol=symbol,
                    pattern_type=PatternType.RISING_WEDGE,
                    confidence=0.75,
                    strength=strength,
                    entry_price=current_price,
                    target_price=recent_lows.iloc[0] * 0.95,  # Retour vers support initial
                    stop_loss=recent_highs.max() * 1.02,
                    volume_confirmation=self._check_volume_increase(df),
                    trading_direction="short",
                    trend_duration_days=self._calculate_trend_duration(df, "bearish"),
                    trend_strength_score=strength,
                    additional_data={'high_slope': high_slope, 'low_slope': low_slope}
                ))
            
            # Falling Wedge : highs et lows descendent, mais lows slope < highs slope (bullish)
            elif high_slope < 0 and low_slope < 0 and abs(low_slope) > abs(high_slope) * 1.2:
                strength = min(abs(high_slope - low_slope) * 1000, 1.0)
                patterns.append(TechnicalPattern(
                    symbol=symbol,
                    pattern_type=PatternType.FALLING_WEDGE,
                    confidence=0.75,
                    strength=strength,
                    entry_price=current_price,
                    target_price=recent_highs.iloc[0] * 1.05,  # Retour vers résistance initiale
                    stop_loss=recent_lows.min() * 0.98,
                    volume_confirmation=self._check_volume_increase(df),
                    trading_direction="long",
                    trend_duration_days=self._calculate_trend_duration(df, "bullish"),
                    trend_strength_score=strength,
                    additional_data={'high_slope': high_slope, 'low_slope': low_slope}
                ))

        except Exception as e:
            logger.debug(f"Wedge pattern detection error for {symbol}: {e}")

        return patterns

    def _detect_triple_patterns(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte Triple Top et Triple Bottom - implémentation complète"""
        patterns = []
        
        try:
            if len(df) < 30:
                return patterns
            
            highs = df['High'].rolling(3, center=True).max() == df['High']
            lows = df['Low'].rolling(3, center=True).min() == df['Low']
            current_price = df['Close'].iloc[-1]
            
            # Triple Top (bearish reversal)
            high_peaks = df[highs]['High'].iloc[-20:]  # Derniers 20 pics
            if len(high_peaks) >= 3:
                for i in range(len(high_peaks) - 2):
                    peak1 = high_peaks.iloc[i]
                    peak2 = high_peaks.iloc[i + 1] 
                    peak3 = high_peaks.iloc[i + 2]
                    
                    # Les 3 pics doivent être à des niveaux similaires (±2%) - FIXED: Prevent division by zero
                    avg_peak = (peak1 + peak2 + peak3) / 3
                    if (avg_peak > 0 and 
                        abs(peak1 - avg_peak) / avg_peak < 0.02 and 
                        abs(peak2 - avg_peak) / avg_peak < 0.02 and
                        abs(peak3 - avg_peak) / avg_peak < 0.02):
                        
                        # Support line (neckline)
                        support_level = avg_peak * 0.96
                        strength = min((avg_peak - support_level) / support_level * 4, 1.0)
                        
                        if strength > 0.5:
                            patterns.append(TechnicalPattern(
                                symbol=symbol,
                                pattern_type=PatternType.TRIPLE_TOP,
                                confidence=0.88,
                                strength=strength,
                                entry_price=current_price,
                                target_price=support_level * 0.94,
                                stop_loss=avg_peak * 1.03,
                                volume_confirmation=self._check_volume_decrease_on_peaks(df),
                                trading_direction="short",
                                trend_duration_days=self._calculate_trend_duration(df, "bearish"),
                                trend_strength_score=strength,
                                additional_data={
                                    'peak1': peak1, 'peak2': peak2, 'peak3': peak3,
                                    'support_level': support_level, 'avg_peak': avg_peak
                                }
                            ))

            # Triple Bottom (bullish reversal)
            low_troughs = df[lows]['Low'].iloc[-20:]
            if len(low_troughs) >= 3:
                for i in range(len(low_troughs) - 2):
                    trough1 = low_troughs.iloc[i]
                    trough2 = low_troughs.iloc[i + 1]
                    trough3 = low_troughs.iloc[i + 2]
                    
                    avg_trough = (trough1 + trough2 + trough3) / 3
                    if (avg_trough > 0 and 
                        abs(trough1 - avg_trough) / avg_trough < 0.02 and
                        abs(trough2 - avg_trough) / avg_trough < 0.02 and
                        abs(trough3 - avg_trough) / avg_trough < 0.02):
                        
                        resistance_level = avg_trough * 1.04
                        strength = min((resistance_level - avg_trough) / avg_trough * 4, 1.0)
                        
                        if strength > 0.5:
                            patterns.append(TechnicalPattern(
                                symbol=symbol,
                                pattern_type=PatternType.TRIPLE_BOTTOM,
                                confidence=0.88,
                                strength=strength,
                                entry_price=current_price,
                                target_price=resistance_level * 1.06,
                                stop_loss=avg_trough * 0.97,
                                volume_confirmation=self._check_volume_increase_on_breakout(df),
                                trading_direction="long",
                                trend_duration_days=self._calculate_trend_duration(df, "bullish"),
                                trend_strength_score=strength,
                                additional_data={
                                    'trough1': trough1, 'trough2': trough2, 'trough3': trough3,
                                    'resistance_level': resistance_level, 'avg_trough': avg_trough
                                }
                            ))
                            
        except Exception as e:
            logger.debug(f"Triple patterns detection error for {symbol}: {e}")
            
        return patterns

    def _detect_symmetrical_triangle(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte Triangle Symétrique - implémentation complète"""
        patterns = []
        
        try:
            if len(df) < 25:
                return patterns
                
            current_price = df['Close'].iloc[-1]
            
            # Détection des lignes de tendance convergentes
            highs = df['High'].rolling(3, center=True).max() == df['High']
            lows = df['Low'].rolling(3, center=True).min() == df['Low']
            
            high_peaks = df[highs]['High'].iloc[-15:]
            low_troughs = df[lows]['Low'].iloc[-15:]
            
            if len(high_peaks) >= 3 and len(low_troughs) >= 3:
                # Calculer les pentes des lignes de support et résistance
                high_slope = self._calculate_trend_line_slope(high_peaks.values)
                low_slope = self._calculate_trend_line_slope(low_troughs.values)
                
                # Triangle symétrique: pente résistance négative, pente support positive
                if high_slope < -0.001 and low_slope > 0.001:
                    # Point de convergence approximatif
                    range_start = max(high_peaks.max(), low_troughs.max())
                    range_end = min(high_peaks.min(), low_troughs.min())
                    range_compression = (range_start - range_end) / range_start
                    
                    if 0.05 < range_compression < 0.25:  # Compression significative
                        strength = min(range_compression * 3, 1.0)
                        
                        # Direction dépend de la tendance précédente
                        prev_trend = self._detect_previous_trend(df)
                        direction = prev_trend if prev_trend in ["long", "short"] else "long"
                        
                        target_multiplier = 1.04 if direction == "long" else 0.96
                        stop_multiplier = 0.97 if direction == "long" else 1.03
                        
                        patterns.append(TechnicalPattern(
                            symbol=symbol,
                            pattern_type=PatternType.SYMMETRICAL_TRIANGLE,
                            confidence=0.82,
                            strength=strength,
                            entry_price=current_price,
                            target_price=current_price * target_multiplier,
                            stop_loss=current_price * stop_multiplier,
                            volume_confirmation=self._check_volume_contraction(df),
                            trading_direction=direction,
                            trend_duration_days=15,
                            trend_strength_score=strength,
                            additional_data={
                                'high_slope': high_slope,
                                'low_slope': low_slope,
                                'range_compression': range_compression,
                                'convergence_zone': (range_start + range_end) / 2
                            }
                        ))
                        
        except Exception as e:
            logger.debug(f"Symmetrical triangle detection error for {symbol}: {e}")
            
        return patterns

    def _detect_enhanced_triangles(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Améliore la détection existante des triangles ascendants/descendants"""
        patterns = []
        
        try:
            if len(df) < 20:
                return patterns
                
            current_price = df['Close'].iloc[-1]
            highs = df['High'].rolling(3, center=True).max() == df['High']
            lows = df['Low'].rolling(3, center=True).min() == df['Low']
            
            high_peaks = df[highs]['High'].iloc[-12:]
            low_troughs = df[lows]['Low'].iloc[-12:]
            
            if len(high_peaks) >= 3 and len(low_troughs) >= 3:
                high_slope = self._calculate_trend_line_slope(high_peaks.values)
                low_slope = self._calculate_trend_line_slope(low_troughs.values)
                
                # Triangle Ascendant: résistance horizontale, support ascendant
                if abs(high_slope) < 0.0005 and low_slope > 0.002:
                    resistance_level = high_peaks.max()
                    strength = min(low_slope * 1000, 1.0)
                    
                    patterns.append(TechnicalPattern(
                        symbol=symbol,
                        pattern_type=PatternType.ASCENDING_TRIANGLE,
                        confidence=0.85,
                        strength=strength,
                        entry_price=current_price,
                        target_price=resistance_level * 1.05,
                        stop_loss=low_troughs.iloc[-1] * 0.98,
                        volume_confirmation=self._check_volume_increase_on_breakout(df),
                        trading_direction="long",
                        trend_duration_days=12,
                        trend_strength_score=strength,
                        additional_data={
                            'resistance_level': resistance_level,
                            'support_slope': low_slope,
                            'breakout_target': resistance_level * 1.05
                        }
                    ))
                
                # Triangle Descendant: support horizontal, résistance descendante  
                elif abs(low_slope) < 0.0005 and high_slope < -0.002:
                    support_level = low_troughs.min()
                    strength = min(abs(high_slope) * 1000, 1.0)
                    
                    patterns.append(TechnicalPattern(
                        symbol=symbol,
                        pattern_type=PatternType.DESCENDING_TRIANGLE,
                        confidence=0.85,
                        strength=strength,
                        entry_price=current_price,
                        target_price=support_level * 0.95,
                        stop_loss=high_peaks.iloc[-1] * 1.02,
                        volume_confirmation=self._check_volume_increase_on_breakdown(df),
                        trading_direction="short",
                        trend_duration_days=12,
                        trend_strength_score=strength,
                        additional_data={
                            'support_level': support_level,
                            'resistance_slope': high_slope,
                            'breakdown_target': support_level * 0.95
                        }
                    ))
                    
        except Exception as e:
            logger.debug(f"Enhanced triangles detection error for {symbol}: {e}")
            
        return patterns

    def _detect_pennant_patterns(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte les Pennants - implémentation complète"""
        patterns = []
        
        try:
            if len(df) < 15:
                return patterns
                
            current_price = df['Close'].iloc[-1]
            
            # Pennant: petite consolidation triangulaire après mouvement fort
            # 1. Détecter mouvement initial fort (flagpole)
            price_change_10d = (df['Close'].iloc[-1] / df['Close'].iloc[-11] - 1) * 100
            
            if abs(price_change_10d) > 5:  # Mouvement d'au moins 5% en 10 jours
                # 2. Détecter consolidation triangulaire (5-8 jours)
                recent_df = df.iloc[-8:]
                if len(recent_df) >= 5:
                    high_range = recent_df['High'].max() - recent_df['High'].min()
                    low_range = recent_df['Low'].max() - recent_df['Low'].min()
                    avg_range = (high_range + low_range) / 2
                    
                    # Contraction du range (caractéristique du pennant)
                    range_contraction = avg_range / current_price
                    
                    if 0.01 < range_contraction < 0.05:  # 1-5% de contraction
                        direction = "long" if price_change_10d > 0 else "short"
                        strength = min(abs(price_change_10d) / 10, 1.0)
                        
                        # Le pennant continue généralement la tendance initiale
                        if direction == "long":
                            target_price = current_price * (1 + abs(price_change_10d) / 200)
                            stop_loss = recent_df['Low'].min() * 0.99
                            pattern_type = PatternType.PENNANT_BULLISH
                        else:
                            target_price = current_price * (1 - abs(price_change_10d) / 200)
                            stop_loss = recent_df['High'].max() * 1.01
                            pattern_type = PatternType.PENNANT_BEARISH
                        
                        patterns.append(TechnicalPattern(
                            symbol=symbol,
                            pattern_type=pattern_type,
                            confidence=0.83,
                            strength=strength,
                            entry_price=current_price,
                            target_price=target_price,
                            stop_loss=stop_loss,
                            volume_confirmation=self._check_volume_contraction(df),
                            trading_direction=direction,
                            trend_duration_days=8,
                            trend_strength_score=strength,
                            additional_data={
                                'flagpole_move': price_change_10d,
                                'range_contraction': range_contraction,
                                'consolidation_days': len(recent_df)
                            }
                        ))
                        
        except Exception as e:
            logger.debug(f"Pennant patterns detection error for {symbol}: {e}")
            
        return patterns

    def _detect_rectangle_consolidation(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte les consolidations rectangulaires"""
        patterns = []
        
        try:
            if len(df) < 20:
                return patterns
                
            current_price = df['Close'].iloc[-1]
            
            # Analyser les 15 derniers jours pour détecter un rectangle
            recent_df = df.iloc[-15:]
            
            # Identifier les niveaux de support et résistance horizontaux
            resistance_level = recent_df['High'].max()
            support_level = recent_df['Low'].min()
            range_size = (resistance_level - support_level) / current_price
            
            # Rectangle valide: range entre 2-8%, prix oscille entre support/résistance
            if 0.02 < range_size < 0.08:
                # Vérifier que le prix teste les niveaux plusieurs fois
                resistance_tests = (recent_df['High'] > resistance_level * 0.99).sum()
                support_tests = (recent_df['Low'] < support_level * 1.01).sum()
                
                if resistance_tests >= 2 and support_tests >= 2:
                    # Position dans le rectangle
                    position_in_range = (current_price - support_level) / (resistance_level - support_level)
                    
                    # Détecter la direction probable du breakout
                    trend_before = self._detect_previous_trend(df.iloc[:-15])
                    
                    if trend_before == "long":
                        direction = "long"
                        target_price = resistance_level * (1 + range_size)
                        stop_loss = support_level * 0.99
                    elif trend_before == "short":
                        direction = "short"  
                        target_price = support_level * (1 - range_size)
                        stop_loss = resistance_level * 1.01
                    else:
                        # Neutral - attendre le breakout
                        direction = "long" if position_in_range < 0.5 else "short"
                        target_price = resistance_level * 1.03 if direction == "long" else support_level * 0.97
                        stop_loss = support_level * 0.99 if direction == "long" else resistance_level * 1.01
                    
                    strength = min(range_size * 10 + (resistance_tests + support_tests) * 0.1, 1.0)
                    
                    patterns.append(TechnicalPattern(
                        symbol=symbol,
                        pattern_type=PatternType.RECTANGLE_CONSOLIDATION,
                        confidence=0.80,
                        strength=strength, 
                        entry_price=current_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        volume_confirmation=self._check_volume_contraction(df),
                        trading_direction=direction,
                        trend_duration_days=15,
                        trend_strength_score=strength,
                        additional_data={
                            'resistance_level': resistance_level,
                            'support_level': support_level,
                            'range_size': range_size,
                            'resistance_tests': resistance_tests,
                            'support_tests': support_tests,
                            'position_in_range': position_in_range
                        }
                    ))
                    
        except Exception as e:
            logger.debug(f"Rectangle consolidation detection error for {symbol}: {e}")
            
        return patterns

    def _detect_rounding_patterns(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte Rounding Top et Bottom"""
        patterns = []
        
        try:
            if len(df) < 30:
                return patterns
                
            current_price = df['Close'].iloc[-1]
            
            # Analyser les 25 derniers jours pour forme arrondie
            recent_df = df.iloc[-25:]
            prices = recent_df['Close'].values
            
            # Fitting polynômial de degré 2 pour détecter la courbure
            x = np.arange(len(prices))
            coeffs = np.polyfit(x, prices, 2)
            
            # Coefficient du terme quadratique indique la courbure
            curvature = coeffs[0]
            r_squared = self._calculate_r_squared(prices, np.polyval(coeffs, x))
            
            # Rounding Bottom: courbure positive (forme de U)
            if curvature > 0 and r_squared > 0.7:
                lowest_point = prices.min()
                current_position = len(prices) - 1
                
                # Vérifier que nous sommes dans la partie droite de la courbe
                if current_position > len(prices) * 0.6 and current_price > lowest_point * 1.02:
                    strength = min(curvature * 10000 + r_squared, 1.0)
                    
                    patterns.append(TechnicalPattern(
                        symbol=symbol,
                        pattern_type=PatternType.ROUNDING_BOTTOM,
                        confidence=0.85,
                        strength=strength,
                        entry_price=current_price,
                        target_price=lowest_point * 1.15,  # 15% au-dessus du creux
                        stop_loss=lowest_point * 0.95,
                        volume_confirmation=self._check_volume_expansion(df),
                        trading_direction="long",
                        trend_duration_days=25,
                        trend_strength_score=strength,
                        additional_data={
                            'curvature': curvature,
                            'r_squared': r_squared,
                            'lowest_point': lowest_point,
                            'polynomial_coeffs': coeffs.tolist()
                        }
                    ))
            
            # Rounding Top: courbure négative (forme de ∩)
            elif curvature < 0 and r_squared > 0.7:
                highest_point = prices.max()
                current_position = len(prices) - 1
                
                if current_position > len(prices) * 0.6 and current_price < highest_point * 0.98:
                    strength = min(abs(curvature) * 10000 + r_squared, 1.0)
                    
                    patterns.append(TechnicalPattern(
                        symbol=symbol,
                        pattern_type=PatternType.ROUNDING_TOP,
                        confidence=0.85,
                        strength=strength,
                        entry_price=current_price,
                        target_price=highest_point * 0.85,  # 15% en-dessous du pic
                        stop_loss=highest_point * 1.05,
                        volume_confirmation=self._check_volume_decrease_on_peaks(df),
                        trading_direction="short",
                        trend_duration_days=25,
                        trend_strength_score=strength,
                        additional_data={
                            'curvature': curvature,
                            'r_squared': r_squared,
                            'highest_point': highest_point,
                            'polynomial_coeffs': coeffs.tolist()
                        }
                    ))
                    
        except Exception as e:
            logger.debug(f"Rounding patterns detection error for {symbol}: {e}")
            
        return patterns

    # Méthodes utilitaires pour les nouveaux patterns
    def _calculate_trend_line_slope(self, prices):
        """Calcule la pente d'une ligne de tendance avec protection contre les erreurs"""
        try:
            if len(prices) < 2:
                return 0.0
            
            # Nettoyer les données NaN
            clean_prices = [p for p in prices if not np.isnan(p) and p > 0]
            if len(clean_prices) < 2:
                return 0.0
                
            x = np.arange(len(clean_prices))
            
            # Protection contre les erreurs de calcul
            if np.any(np.isnan(clean_prices)) or np.any(np.isinf(clean_prices)):
                return 0.0
                
            slope, _ = np.polyfit(x, clean_prices, 1)
            
            # Vérifier que le résultat est valide
            if np.isnan(slope) or np.isinf(slope):
                return 0.0
                
            return float(slope)
            
        except Exception as e:
            logger.debug(f"Error calculating trend line slope: {e}")
            return 0.0
    
    def _detect_previous_trend(self, df):
        """Détecte la tendance précédente avec protection d'erreurs"""
        try:
            if len(df) < 10:
                return "neutral"
            
            # Vérifier que les données sont valides
            if df['Close'].empty or df['Close'].isna().all():
                return "neutral"
            
            # Calculer le changement avec protection
            start_price = df['Close'].iloc[-10]
            end_price = df['Close'].iloc[-1]
            
            if start_price <= 0 or end_price <= 0 or np.isnan(start_price) or np.isnan(end_price):
                return "neutral"
            
            recent_change = (end_price / start_price - 1) * 100
            
            if np.isnan(recent_change) or np.isinf(recent_change):
                return "neutral"
                
            if recent_change > 3:
                return "long"
            elif recent_change < -3:
                return "short"
            return "neutral"
            
        except Exception as e:
            logger.debug(f"Error detecting previous trend: {e}")
            return "neutral"
    
    def _check_volume_contraction(self, df):
        """Vérifie la contraction du volume avec protection d'erreurs"""
        try:
            if len(df) < 10 or 'Volume' not in df.columns:
                return False
            
            # Vérifier que les colonnes Volume existent et ont des données
            if df['Volume'].empty or df['Volume'].isna().all():
                return False
            
            recent_volume = df['Volume'].iloc[-5:].mean()
            previous_volume = df['Volume'].iloc[-10:-5].mean()
            
            # Protection contre NaN et valeurs nulles
            if (np.isnan(recent_volume) or np.isnan(previous_volume) or 
                recent_volume <= 0 or previous_volume <= 0):
                return False
            
            return recent_volume < previous_volume * 0.8
            
        except Exception as e:
            logger.debug(f"Error checking volume contraction: {e}")
            return False
    
    def _check_volume_expansion(self, df):
        """Vérifie l'expansion du volume avec protection d'erreurs"""
        try:
            if len(df) < 10 or 'Volume' not in df.columns:
                return False
            
            if df['Volume'].empty or df['Volume'].isna().all():
                return False
            
            recent_volume = df['Volume'].iloc[-5:].mean()
            previous_volume = df['Volume'].iloc[-10:-5].mean()
            
            if (np.isnan(recent_volume) or np.isnan(previous_volume) or 
                recent_volume <= 0 or previous_volume <= 0):
                return False
            
            return recent_volume > previous_volume * 1.2
            
        except Exception as e:
            logger.debug(f"Error checking volume expansion: {e}")
            return False
    
    def _check_volume_decrease_on_peaks(self, df):
        """Vérifie la diminution du volume sur les pics avec protection d'erreurs"""
        try:
            if len(df) < 8 or 'Volume' not in df.columns:
                return False
            
            if df['Volume'].empty or df['Volume'].isna().all():
                return False
            
            recent_high_volume = df.nlargest(3, 'High')['Volume'].mean()
            avg_volume = df['Volume'].mean()
            
            if np.isnan(recent_high_volume) or np.isnan(avg_volume) or avg_volume <= 0:
                return False
            
            return recent_high_volume < avg_volume * 0.9
            
        except Exception as e:
            logger.debug(f"Error checking volume decrease on peaks: {e}")
            return False
    
    def _check_volume_increase_on_breakout(self, df):
        """Vérifie l'augmentation du volume sur breakout avec protection d'erreurs"""
        try:
            if len(df) < 5 or 'Volume' not in df.columns:
                return False
            
            if df['Volume'].empty or df['Volume'].isna().all():
                return False
            
            recent_volume = df['Volume'].iloc[-3:].mean()
            avg_volume = df['Volume'].iloc[-10:-3].mean()
            
            if (np.isnan(recent_volume) or np.isnan(avg_volume) or 
                recent_volume <= 0 or avg_volume <= 0):
                return False
            
            return recent_volume > avg_volume * 1.3
            
        except Exception as e:
            logger.debug(f"Error checking volume increase on breakout: {e}")
            return False
    
    def _check_volume_increase_on_breakdown(self, df):
        """Vérifie l'augmentation du volume sur breakdown (bearish)"""
        return self._check_volume_increase_on_breakout(df)  # Même logique
    
    def _calculate_r_squared(self, actual, predicted):
        """Calcule le R² pour évaluer la qualité d'un fit avec protection d'erreurs"""
        try:
            if len(actual) != len(predicted) or len(actual) == 0:
                return 0.0
                
            # Nettoyer les données
            actual = np.array(actual)
            predicted = np.array(predicted)
            
            # Vérifier NaN et inf
            if np.any(np.isnan(actual)) or np.any(np.isnan(predicted)) or \
               np.any(np.isinf(actual)) or np.any(np.isinf(predicted)):
                return 0.0
            
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            
            # Protection contre division par zéro
            if ss_tot == 0 or np.isnan(ss_tot) or np.isinf(ss_tot):
                return 0.0
                
            r_squared = 1 - (ss_res / ss_tot)
            
            # Vérifier résultat valide
            if np.isnan(r_squared) or np.isinf(r_squared):
                return 0.0
                
            return max(0.0, min(1.0, float(r_squared)))  # Limiter entre 0 et 1
            
        except Exception as e:
            logger.debug(f"Error calculating R²: {e}")
            return 0.0
    
    def _detect_harmonic_patterns(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte les patterns harmoniques (Gartley, Bat, Butterfly, Crab)"""
        patterns = []
        
        try:
            if len(df) < 50:
                return patterns
            
            # Identifier les points pivots significatifs
            pivots = self._find_pivot_points(df, window=5)
            
            if len(pivots) >= 5:  # Besoin de 5 points pour patterns harmoniques
                # Analyser les dernières formations de 5 points (X-A-B-C-D)
                for i in range(len(pivots) - 4):
                    points = pivots[i:i+5]
                    X, A, B, C, D = [p['price'] for p in points]
                    
                    # Calculer les ratios de Fibonacci avec protection
                    XA = abs(A - X)
                    AB = abs(B - A) 
                    BC = abs(C - B)
                    CD = abs(D - C)
                    
                    # Protection contre division par zéro
                    if XA <= 0 or AB <= 0 or BC <= 0 or CD <= 0:
                        continue
                    
                    # Vérifier que toutes les valeurs sont finies
                    if not all(np.isfinite([XA, AB, BC, CD])):
                        continue
                    
                    AB_XA = AB / XA
                    BC_AB = BC / AB
                    CD_BC = CD / BC if BC > 0 else 0
                    
                    # Vérifier que les ratios sont valides
                    if not all(np.isfinite([AB_XA, BC_AB, CD_BC])):
                        continue
                        
                        # Pattern Gartley (0.618, 0.382, 1.272)
                        if (0.58 <= AB_XA <= 0.68 and 0.35 <= BC_AB <= 0.42):
                            direction = "long" if D < A else "short"
                            confidence = 0.75 + (1 - abs(AB_XA - 0.618)) * 0.15
                            
                            if direction == "long":
                                patterns.append(TechnicalPattern(
                                    symbol=symbol,
                                    pattern_type=PatternType.GARTLEY_BULLISH,
                                    confidence=confidence,
                                    strength=0.8,
                                    entry_price=D,
                                    target_price=D * 1.05,
                                    stop_loss=D * 0.97,
                                    volume_confirmation=True,
                                    trading_direction="long",
                                    additional_data={
                                        'points': {'X': X, 'A': A, 'B': B, 'C': C, 'D': D},
                                        'ratios': {'AB_XA': AB_XA, 'BC_AB': BC_AB}
                                    }
                                ))
                            else:
                                patterns.append(TechnicalPattern(
                                    symbol=symbol,
                                    pattern_type=PatternType.GARTLEY_BEARISH,
                                    confidence=confidence,
                                    strength=0.8,
                                    entry_price=D,
                                    target_price=D * 0.95,
                                    stop_loss=D * 1.03,
                                    volume_confirmation=True,
                                    trading_direction="short",
                                    additional_data={
                                        'points': {'X': X, 'A': A, 'B': B, 'C': C, 'D': D},
                                        'ratios': {'AB_XA': AB_XA, 'BC_AB': BC_AB}
                                    }
                                ))
                        
                        # Pattern Bat (0.382, 0.382, 0.886)
                        elif (0.35 <= AB_XA <= 0.42 and 0.35 <= BC_AB <= 0.42):
                            direction = "long" if D < A else "short"
                            confidence = 0.78
                            
                            pattern_type = PatternType.BAT_BULLISH if direction == "long" else PatternType.BAT_BEARISH
                            target_mult = 1.06 if direction == "long" else 0.94
                            stop_mult = 0.96 if direction == "long" else 1.04
                            
                            patterns.append(TechnicalPattern(
                                symbol=symbol,
                                pattern_type=pattern_type,
                                confidence=confidence,
                                strength=0.82,
                                entry_price=D,
                                target_price=D * target_mult,
                                stop_loss=D * stop_mult,
                                volume_confirmation=True,
                                trading_direction=direction,
                                additional_data={
                                    'points': {'X': X, 'A': A, 'B': B, 'C': C, 'D': D},
                                    'ratios': {'AB_XA': AB_XA, 'BC_AB': BC_AB}
                                }
                            ))
                        
                        # Pattern Butterfly (0.786, 0.382, 1.618)
                        elif (0.75 <= AB_XA <= 0.82 and 0.35 <= BC_AB <= 0.42):
                            direction = "long" if D < A else "short"
                            confidence = 0.73
                            
                            pattern_type = PatternType.BUTTERFLY_BULLISH if direction == "long" else PatternType.BUTTERFLY_BEARISH
                            target_mult = 1.08 if direction == "long" else 0.92
                            stop_mult = 0.95 if direction == "long" else 1.05
                            
                            patterns.append(TechnicalPattern(
                                symbol=symbol,
                                pattern_type=pattern_type,
                                confidence=confidence,
                                strength=0.78,
                                entry_price=D,
                                target_price=D * target_mult,
                                stop_loss=D * stop_mult,
                                volume_confirmation=True,
                                trading_direction=direction,
                                additional_data={
                                    'points': {'X': X, 'A': A, 'B': B, 'C': C, 'D': D},
                                    'ratios': {'AB_XA': AB_XA, 'BC_AB': BC_AB}
                                }
                            ))
                            
        except Exception as e:
            logger.debug(f"Harmonic patterns detection error for {symbol}: {e}")
            
        return patterns
    
    def _find_pivot_points(self, df: pd.DataFrame, window: int = 5) -> List[Dict]:
        """Trouve les points pivots (maxima et minima locaux)"""
        pivots = []
        
        for i in range(window, len(df) - window):
            # Maximum local
            if all(df['High'].iloc[i] >= df['High'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['High'].iloc[i] >= df['High'].iloc[i+j] for j in range(1, window+1)):
                pivots.append({
                    'index': i,
                    'price': df['High'].iloc[i],
                    'type': 'high',
                    'date': df.index[i] if hasattr(df.index[i], 'date') else i
                })
            
            # Minimum local  
            elif all(df['Low'].iloc[i] <= df['Low'].iloc[i-j] for j in range(1, window+1)) and \
                 all(df['Low'].iloc[i] <= df['Low'].iloc[i+j] for j in range(1, window+1)):
                pivots.append({
                    'index': i,
                    'price': df['Low'].iloc[i],
                    'type': 'low',
                    'date': df.index[i] if hasattr(df.index[i], 'date') else i
                })
        
        return sorted(pivots, key=lambda x: x['index'])
    
    def _detect_diamond_patterns(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte les patterns Diamant (Diamond Top/Bottom)"""
        patterns = []
        
        try:
            if len(df) < 30:
                return patterns
            
            current_price = df['Close'].iloc[-1]
            
            # Analyser volatilité et structure en diamant
            recent_df = df.iloc[-25:]
            
            # Calculer la volatilité sur période glissante
            volatility = recent_df['Close'].rolling(5).std()
            
            # Pattern diamant: volatilité croissante puis décroissante
            first_half_vol = volatility.iloc[:12].mean()
            second_half_vol = volatility.iloc[-12:].mean()
            
            if first_half_vol > 0 and second_half_vol > 0:
                vol_ratio = first_half_vol / second_half_vol
                
                # Diamant Top: volatilité décroissante après expansion
                if vol_ratio > 1.5:
                    highest_point = recent_df['High'].max()
                    if current_price < highest_point * 0.98:
                        patterns.append(TechnicalPattern(
                            symbol=symbol,
                            pattern_type=PatternType.DIAMOND_TOP,
                            confidence=0.75,
                            strength=min(vol_ratio / 2, 1.0),
                            entry_price=current_price,
                            target_price=highest_point * 0.85,
                            stop_loss=highest_point * 1.02,
                            volume_confirmation=self._check_volume_decrease_on_peaks(df),
                            trading_direction="short",
                            trend_duration_days=25,
                            additional_data={
                                'volatility_ratio': vol_ratio,
                                'highest_point': highest_point,
                                'first_half_vol': first_half_vol,
                                'second_half_vol': second_half_vol
                            }
                        ))
                
                # Diamant Bottom: volatilité décroissante après contraction
                elif vol_ratio < 0.7:
                    lowest_point = recent_df['Low'].min()
                    if current_price > lowest_point * 1.02:
                        patterns.append(TechnicalPattern(
                            symbol=symbol,
                            pattern_type=PatternType.DIAMOND_BOTTOM,
                            confidence=0.75,
                            strength=min(1 / vol_ratio, 1.0),
                            entry_price=current_price,
                            target_price=lowest_point * 1.15,
                            stop_loss=lowest_point * 0.98,
                            volume_confirmation=self._check_volume_expansion(df),
                            trading_direction="long",
                            trend_duration_days=25,
                            additional_data={
                                'volatility_ratio': vol_ratio,
                                'lowest_point': lowest_point,
                                'first_half_vol': first_half_vol,
                                'second_half_vol': second_half_vol
                            }
                        ))
                        
        except Exception as e:
            logger.debug(f"Diamond patterns detection error for {symbol}: {e}")
            
        return patterns
    
    def _detect_expanding_wedge_patterns(self, symbol: str, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Détecte les biseaux d'élargissement (Expanding Wedges)"""
        patterns = []
        
        try:
            if len(df) < 20:
                return patterns
            
            current_price = df['Close'].iloc[-1]
            
            # Analyser l'expansion des ranges
            recent_df = df.iloc[-15:]
            
            # Calculer l'évolution du range (High - Low)
            ranges = recent_df['High'] - recent_df['Low']
            early_ranges = ranges.iloc[:5].mean()
            late_ranges = ranges.iloc[-5:].mean()
            
            if early_ranges > 0:
                expansion_ratio = late_ranges / early_ranges
                
                # Expanding Wedge: ranges croissants (instabilité croissante)
                if expansion_ratio > 1.4:
                    # Déterminer la direction basée sur la tendance
                    price_trend = (recent_df['Close'].iloc[-1] / recent_df['Close'].iloc[0] - 1) * 100
                    
                    strength = min((expansion_ratio - 1) * 2, 1.0)
                    
                    patterns.append(TechnicalPattern(
                        symbol=symbol,
                        pattern_type=PatternType.EXPANDING_WEDGE,
                        confidence=0.70,
                        strength=strength,
                        entry_price=current_price,
                        target_price=current_price * (1.05 if price_trend > 0 else 0.95),
                        stop_loss=current_price * (0.96 if price_trend > 0 else 1.04),
                        volume_confirmation=self._check_volume_expansion(df),
                        trading_direction="long" if price_trend > 0 else "short",
                        trend_duration_days=15,
                        additional_data={
                            'expansion_ratio': expansion_ratio,
                            'early_ranges': early_ranges,
                            'late_ranges': late_ranges,
                            'price_trend': price_trend
                        }
                    ))
                    
        except Exception as e:
            logger.debug(f"Expanding wedge detection error for {symbol}: {e}")
            
        return patterns

    async def _fetch_binance_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        """Récupère OHLCV depuis Binance API (1,200 req/min - ULTRA GÉNÉREUX!)"""
        try:
            # Format du symbole pour Binance
            binance_symbol = symbol.replace('USDT', 'USDT')  # Déjà au bon format
            
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol": binance_symbol,
                "interval": "1d",
                "limit": self.lookback_days
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_binance_ohlcv(data)
                    else:
                        logger.debug(f"Binance OHLCV failed for {symbol}: {response.status}")
                        
        except Exception as e:
            logger.debug(f"Binance OHLCV error for {symbol}: {e}")
        
        return None

    async def _fetch_coingecko_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        """Récupère OHLCV depuis CoinGecko API (10,000 appels/mois, 10 req/sec)"""
        try:
            # Convertit le symbole pour CoinGecko (nécessite l'ID de la coin)
            # Pour simplifier, on utilise les principales cryptos
            symbol_map = {
                'BTCUSDT': 'bitcoin',
                'ETHUSDT': 'ethereum', 
                'BNBUSDT': 'binancecoin',
                'SOLUSDT': 'solana',
                'XRPUSDT': 'ripple',
                'ADAUSDT': 'cardano',
                'DOGEUSDT': 'dogecoin',
                'AVAXUSDT': 'avalanche-2',
                'DOTUSDT': 'polkadot',
                'MATICUSDT': 'matic-network'
            }
            
            if symbol not in symbol_map:
                return None
            
            coin_id = symbol_map[symbol]
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
            params = {
                "vs_currency": "usd",
                "days": str(self.lookback_days)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_coingecko_ohlcv(data)
                    else:
                        logger.debug(f"CoinGecko OHLCV failed for {symbol}: {response.status}")
                        
        except Exception as e:
            logger.debug(f"CoinGecko OHLCV error for {symbol}: {e}")
        
        return None

    async def _fetch_cryptocompare_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        """Récupère OHLCV depuis CryptoCompare API (100,000 appels/mois)"""
        try:
            # Format pour CryptoCompare
            base_symbol = symbol.replace('USDT', '')
            
            url = "https://min-api.cryptocompare.com/data/v2/histoday"
            params = {
                "fsym": base_symbol,
                "tsym": "USD",
                "limit": self.lookback_days
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_cryptocompare_ohlcv_historical(data)
                    else:
                        logger.debug(f"CryptoCompare OHLCV failed for {symbol}: {response.status}")
                        
        except Exception as e:
            logger.debug(f"CryptoCompare OHLCV error for {symbol}: {e}")
        
        return None

    def _parse_binance_ohlcv(self, data: List) -> pd.DataFrame:
        """Parse les données OHLCV de Binance"""
        df_data = []
        for item in data:
            df_data.append({
                'timestamp': pd.to_datetime(item[0], unit='ms'),
                'Open': float(item[1]),
                'High': float(item[2]),
                'Low': float(item[3]),
                'Close': float(item[4]),
                'Volume': float(item[5])
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        return df

    def _parse_coingecko_ohlcv(self, data: List) -> pd.DataFrame:
        """Parse les données OHLCV de CoinGecko"""
        df_data = []
        for item in data:
            df_data.append({
                'timestamp': pd.to_datetime(item[0], unit='ms'),
                'Open': float(item[1]),
                'High': float(item[2]),
                'Low': float(item[3]),
                'Close': float(item[4])
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        # CoinGecko OHLC n'inclut pas le volume, on l'estime
        df['Volume'] = 1000000  # Volume par défaut
        return df

    def _parse_cryptocompare_ohlcv_historical(self, data: Dict) -> pd.DataFrame:
        """Parse les données OHLCV historiques de CryptoCompare"""
        if 'Data' not in data or 'Data' not in data['Data']:
            return pd.DataFrame()
        
        df_data = []
        for item in data['Data']['Data']:
            df_data.append({
                'timestamp': pd.to_datetime(item['time'], unit='s'),
                'Open': float(item['open']),
                'High': float(item['high']),
                'Low': float(item['low']),
                'Close': float(item['close']),
                'Volume': float(item['volumeto'])  # Volume en USD
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        return df.sort_index()

# Global instance
technical_pattern_detector = TechnicalPatternDetector()
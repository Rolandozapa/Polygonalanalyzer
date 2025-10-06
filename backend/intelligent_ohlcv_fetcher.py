"""
INTELLIGENT OHLCV FETCHER - Module de récupération OHLCV multi-sources intelligent
🎯 Objectif: Compléter les données OHLCV existantes avec des timeframes fins (5min)
           depuis des sources API différentes pour éviter la redondance

Fonctionnalités:
- Métadonnées de tracking des sources utilisées
- Complétion intelligente avec source différente de IA1
- Support/Résistance haute précision (5min, 1h, 4h)
- RR recalculé dynamique basé sur niveaux fins
"""

import pandas as pd
import numpy as np
import aiohttp
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple, Any
import os
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import json

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class OHLCVMetadata:
    """Métadonnées pour traquer les sources de données utilisées"""
    primary_source: str              # Source utilisée par IA1 (ex: "binance", "coingecko")
    primary_timeframe: str           # Timeframe primaire (ex: "1d")
    primary_data_quality: float      # Score qualité 0-1
    primary_data_count: int          # Nombre de points de données
    
    # Informations pour éviter redondance
    sources_used: List[str]          # Liste des sources déjà utilisées
    avoided_sources: List[str]       # Sources évitées pour diversification
    
    # Métadonnées pour complétion
    completion_needed: bool = True   # Si on a besoin de complétion
    preferred_completion_sources: List[str] = None  # Sources préférées pour complétion
    
    def __post_init__(self):
        if self.preferred_completion_sources is None:
            self.preferred_completion_sources = []

@dataclass  
class HighFrequencyData:
    """Données haute fréquence avec métadonnées"""
    symbol: str
    timeframe: str                   # "5m", "15m", "1h", "4h"
    data: pd.DataFrame              # OHLCV DataFrame
    source: str                     # Source API utilisée
    quality_score: float            # Score qualité 0-1
    data_count: int                 # Nombre de points
    fetch_timestamp: datetime       # Moment de récupération
    
    # Support/Résistance calculés depuis ces données
    micro_support_levels: List[float] = None    # S/R dernières 4-6h
    micro_resistance_levels: List[float] = None
    intraday_support_levels: List[float] = None  # S/R dernières 24h
    intraday_resistance_levels: List[float] = None
    
    def __post_init__(self):
        if self.micro_support_levels is None:
            self.micro_support_levels = []
        if self.micro_resistance_levels is None:
            self.micro_resistance_levels = []
        if self.intraday_support_levels is None:
            self.intraday_support_levels = []
        if self.intraday_resistance_levels is None:
            self.intraday_resistance_levels = []

@dataclass
class EnhancedSupportResistance:
    """Niveaux S/R améliorés avec précision haute fréquence"""
    symbol: str
    
    # Niveaux Daily (contexte long terme)
    daily_support: float
    daily_resistance: float
    
    # Niveaux Intraday (24h precision)
    intraday_support: float
    intraday_resistance: float
    
    # Niveaux Micro (4-6h haute précision) 
    micro_support: float
    micro_resistance: float
    
    # Métadonnées
    confidence_daily: float          # Confiance niveaux daily
    confidence_intraday: float       # Confiance niveaux intraday  
    confidence_micro: float          # Confiance niveaux micro
    
    # Source et timing
    calculation_source: str          # Source utilisée pour calcul
    calculation_timestamp: datetime  # Moment du calcul
    
@dataclass
class DynamicRiskReward:
    """RR calculé dynamiquement avec différents niveaux de précision"""
    symbol: str
    signal_type: str                 # "LONG", "SHORT", "HOLD"
    entry_price: float
    
    # RR basés sur différents timeframes
    rr_micro: float                  # RR basé sur niveaux 5m (4-6h)
    rr_intraday: float              # RR basé sur niveaux 1h (24h)  
    rr_daily: float                 # RR basé sur niveaux daily
    
    # RR final et logique de sélection
    rr_final: float                 # RR retenu pour décision
    rr_selection_logic: str         # Comment le RR final a été choisi
    
    # Détails des niveaux utilisés
    selected_support: float         # Support utilisé pour calcul final
    selected_resistance: float      # Résistance utilisée pour calcul final  
    selected_source: str            # Source des niveaux sélectionnés
    
    # Métadonnées
    calculation_timestamp: datetime
    confidence_score: float         # Confiance dans le calcul RR

class IntelligentOHLCVFetcher:
    """Fetcher OHLCV intelligent avec complétion multi-sources"""
    
    def __init__(self):
        # Configuration APIs
        self.binance_enabled = True
        self.coinbase_enabled = True  
        self.kraken_enabled = True
        self.bingx_enabled = True     # Déjà intégré dans le système
        
        # APIs keys (optionnelles selon source)
        self.alpha_vantage_key = os.environ.get('ALPHA_VANTAGE_KEY')
        self.twelvedata_key = os.environ.get('TWELVEDATA_KEY') 
        self.coinapi_key = os.environ.get('COINAPI_KEY')
        
        # Configuration timeframes haute fréquence
        self.high_frequency_timeframes = {
            '5m': {'minutes': 5, 'periods_4h': 48, 'periods_24h': 288},    # 48 points = 4h, 288 = 24h
            '15m': {'minutes': 15, 'periods_4h': 16, 'periods_24h': 96},   # 16 points = 4h, 96 = 24h  
            '1h': {'minutes': 60, 'periods_4h': 4, 'periods_24h': 24},     # 4 points = 4h, 24 = 24h
            '4h': {'minutes': 240, 'periods_4h': 1, 'periods_24h': 6}      # 1 point = 4h, 6 = 24h
        }
        
        # Mapping priorité sources par diversification
        self.source_priority_matrix = {
            # Si IA1 a utilisé ces sources → Utiliser ces sources pour complétion
            'binance': ['coinbase', 'kraken', 'bingx', 'twelvedata'],
            'coingecko': ['binance', 'coinbase', 'alpha_vantage'], 
            'yahoo_finance': ['binance', 'coinapi', 'twelvedata'],
            'coinapi': ['binance', 'coingecko', 'coinbase'],
            'twelvedata': ['binance', 'coinbase', 'coinapi']
        }
        
        logger.info("Intelligent OHLCV Fetcher initialized - Multi-source completion ready")
    
    async def complete_ohlcv_data(self, 
                                  symbol: str, 
                                  existing_metadata: OHLCVMetadata,
                                  target_timeframe: str = '5m',
                                  hours_back: int = 24) -> Optional[HighFrequencyData]:
        """
        🎯 FONCTION PRINCIPALE: Compléter les données OHLCV avec haute fréquence
        
        Args:
            symbol: Symbole crypto (ex: BTCUSDT)
            existing_metadata: Métadonnées des données IA1 existantes
            target_timeframe: Timeframe souhaité ('5m', '15m', '1h', '4h')
            hours_back: Nombre d'heures à récupérer (24h par défaut)
        
        Returns:
            HighFrequencyData avec données complétées ou None si échec
        """
        
        logger.info(f"🎯 Starting intelligent OHLCV completion for {symbol}")
        logger.info(f"   Primary source was: {existing_metadata.primary_source}")
        logger.info(f"   Target timeframe: {target_timeframe}, Hours back: {hours_back}")
        
        # 1. Déterminer les sources à éviter (diversification)
        avoided_sources = [existing_metadata.primary_source] + existing_metadata.sources_used
        preferred_sources = self.source_priority_matrix.get(
            existing_metadata.primary_source, 
            ['binance', 'coinbase', 'bingx']
        )
        
        # Filtrer les sources préférées pour éviter redondance
        completion_sources = [src for src in preferred_sources if src not in avoided_sources]
        logger.info(f"   Avoided sources: {avoided_sources}")
        logger.info(f"   Completion sources to try: {completion_sources}")
        
        # 2. Essayer chaque source de complétion  
        for source in completion_sources:
            try:
                logger.info(f"🔍 Trying completion source: {source}")
                hf_data = await self._fetch_high_frequency_data(
                    symbol=symbol,
                    source=source, 
                    timeframe=target_timeframe,
                    hours_back=hours_back
                )
                
                if hf_data and len(hf_data.data) >= 20:  # Minimum de données
                    logger.info(f"✅ Success with {source}: {len(hf_data.data)} data points")
                    
                    # 3. Calculer les niveaux S/R haute précision
                    enhanced_hf_data = await self._calculate_high_precision_sr_levels(hf_data)
                    
                    return enhanced_hf_data
                    
                else:
                    logger.warning(f"⚠️ {source} provided insufficient data: {len(hf_data.data) if hf_data else 0} points")
                    
            except Exception as e:
                logger.warning(f"❌ {source} failed: {str(e)}")
                continue
        
        # 4. Si tous les sources préférées échouent, essayer sources de fallback
        logger.warning(f"🚨 All preferred sources failed for {symbol}, trying fallback sources")
        fallback_sources = ['alpha_vantage', 'coinapi', 'twelvedata']
        
        for source in fallback_sources:
            if source not in avoided_sources:
                try:
                    logger.info(f"🔄 Trying fallback source: {source}")
                    hf_data = await self._fetch_high_frequency_data(
                        symbol=symbol,
                        source=source,
                        timeframe=target_timeframe, 
                        hours_back=hours_back
                    )
                    
                    if hf_data and len(hf_data.data) >= 10:  # Standard plus relaxé pour fallback
                        logger.info(f"🆘 Fallback success with {source}: {len(hf_data.data)} data points")
                        enhanced_hf_data = await self._calculate_high_precision_sr_levels(hf_data)
                        return enhanced_hf_data
                        
                except Exception as e:
                    logger.warning(f"❌ Fallback {source} failed: {str(e)}")
                    continue
        
        logger.error(f"🚨 Complete failure: No high-frequency data available for {symbol}")
        return None
    
    async def _fetch_high_frequency_data(self, 
                                        symbol: str, 
                                        source: str, 
                                        timeframe: str,
                                        hours_back: int) -> Optional[HighFrequencyData]:
        """Récupérer données haute fréquence depuis une source spécifique"""
        
        fetch_functions = {
            'binance': self._fetch_binance_hf,
            'coinbase': self._fetch_coinbase_hf,
            'kraken': self._fetch_kraken_hf, 
            'bingx': self._fetch_bingx_hf,
            'alpha_vantage': self._fetch_alphavantage_hf,
            'twelvedata': self._fetch_twelvedata_hf,
            'coinapi': self._fetch_coinapi_hf
        }
        
        fetch_func = fetch_functions.get(source)
        if not fetch_func:
            logger.error(f"Unknown source: {source}")
            return None
            
        try:
            df = await fetch_func(symbol, timeframe, hours_back)
            if df is None or len(df) < 10:
                return None
                
            # Calculer score qualité basé sur complétude et cohérence des données
            quality_score = self._calculate_data_quality_score(df)
            
            return HighFrequencyData(
                symbol=symbol,
                timeframe=timeframe,
                data=df,
                source=source,
                quality_score=quality_score,
                data_count=len(df),
                fetch_timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Error fetching {source} HF data for {symbol}: {e}")
            return None
    
    async def _fetch_binance_hf(self, symbol: str, timeframe: str, hours_back: int) -> Optional[pd.DataFrame]:
        """Récupérer données haute fréquence depuis Binance"""
        try:
            # Mapping timeframes Binance
            binance_intervals = {
                '5m': '5m', '15m': '15m', '1h': '1h', '4h': '4h'
            }
            interval = binance_intervals.get(timeframe)
            if not interval:
                return None
                
            # Calculer nombre de klines nécessaires
            minutes_per_interval = self.high_frequency_timeframes[timeframe]['minutes']
            total_minutes = hours_back * 60
            limit = min(int(total_minutes / minutes_per_interval), 1000)  # Max 1000 Binance limit
            
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_binance_hf_data(data, symbol)
                    else:
                        logger.warning(f"Binance HF API returned {response.status} for {symbol}")
                        return None
                        
        except Exception as e:
            logger.error(f"Binance HF fetch error for {symbol}: {e}")
            return None
    
    async def _fetch_coinbase_hf(self, symbol: str, timeframe: str, hours_back: int) -> Optional[pd.DataFrame]:
        """Récupérer données haute fréquence depuis Coinbase Pro"""
        try:
            # Coinbase utilise un format différent (BTC-USD au lieu de BTCUSDT)
            cb_symbol = symbol.replace('USDT', '-USD')
            
            # Mapping granularity Coinbase (en secondes)
            granularity_map = {
                '5m': 300, '15m': 900, '1h': 3600, '4h': 14400
            }
            granularity = granularity_map.get(timeframe)
            if not granularity:
                return None
            
            # Calculer start/end times
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours_back)
            
            url = f"https://api.exchange.coinbase.com/products/{cb_symbol}/candles"
            params = {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(), 
                "granularity": granularity
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_coinbase_hf_data(data, symbol)
                    else:
                        logger.warning(f"Coinbase HF API returned {response.status} for {symbol}")
                        return None
                        
        except Exception as e:
            logger.error(f"Coinbase HF fetch error for {symbol}: {e}")
            return None
    
    async def _fetch_kraken_hf(self, symbol: str, timeframe: str, hours_back: int) -> Optional[pd.DataFrame]:
        """Récupérer données haute fréquence depuis Kraken"""
        try:
            # Format symbole Kraken
            base_symbol = symbol.replace('USDT', '')
            kraken_symbol = f"{base_symbol}USD"  # Kraken utilise USD pas USDT pour certains
            
            # Mapping interval Kraken 
            interval_map = {
                '5m': 5, '15m': 15, '1h': 60, '4h': 240
            }
            interval = interval_map.get(timeframe)
            if not interval:
                return None
            
            # Calculer since (timestamp UNIX)
            since_timestamp = int((datetime.now(timezone.utc) - timedelta(hours=hours_back)).timestamp())
            
            url = "https://api.kraken.com/0/public/OHLC"
            params = {
                "pair": kraken_symbol,
                "interval": interval,
                "since": since_timestamp
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_kraken_hf_data(data, symbol, kraken_symbol)
                    else:
                        logger.warning(f"Kraken HF API returned {response.status} for {symbol}")
                        return None
                        
        except Exception as e:
            logger.error(f"Kraken HF fetch error for {symbol}: {e}")
            return None
    
    async def _fetch_bingx_hf(self, symbol: str, timeframe: str, hours_back: int) -> Optional[pd.DataFrame]:
        """Récupérer données haute fréquency depuis BingX (déjà intégré)"""
        try:
            # BingX interval mapping
            bingx_intervals = {
                '5m': '5m', '15m': '15m', '1h': '1h', '4h': '4h'
            }
            interval = bingx_intervals.get(timeframe)
            if not interval:
                return None
            
            # Importer le BingX engine existant
            from bingx_official_engine import bingx_official_engine
            
            # Utiliser l'engine existant pour récupérer OHLCV
            # Note: Adapter selon l'API BingX disponible dans bingx_official_engine
            klines_data = await bingx_official_engine.get_kline_data(
                symbol=symbol,
                interval=interval,
                limit=min(int(hours_back * 60 / self.high_frequency_timeframes[timeframe]['minutes']), 1000)
            )
            
            if klines_data:
                return self._parse_bingx_hf_data(klines_data, symbol)
            else:
                return None
                
        except Exception as e:
            logger.error(f"BingX HF fetch error for {symbol}: {e}")
            return None
    
    async def _fetch_alphavantage_hf(self, symbol: str, timeframe: str, hours_back: int) -> Optional[pd.DataFrame]:
        """Récupérer données haute fréquence depuis Alpha Vantage"""
        if not self.alpha_vantage_key:
            return None
            
        try:
            # Alpha Vantage a des limitations sur intraday crypto
            # Utiliser TIME_SERIES_INTRADAY pour crypto si disponible
            base_symbol = symbol.replace('USDT', '')
            
            # Mapping interval Alpha Vantage
            interval_map = {
                '5m': '5min', '15m': '15min', '1h': '60min'
            }
            interval = interval_map.get(timeframe)
            if not interval:
                return None  # Alpha Vantage ne supporte pas 4h intraday
                
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": f"{base_symbol}USD",  # Format Alpha Vantage  
                "interval": interval,
                "apikey": self.alpha_vantage_key,
                "outputsize": "full"  # Pour avoir plus de données historiques
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_alphavantage_hf_data(data, symbol, interval)
                    else:
                        logger.warning(f"Alpha Vantage HF API returned {response.status} for {symbol}")
                        return None
                        
        except Exception as e:
            logger.error(f"Alpha Vantage HF fetch error for {symbol}: {e}")
            return None
    
    async def _fetch_twelvedata_hf(self, symbol: str, timeframe: str, hours_back: int) -> Optional[pd.DataFrame]:
        """Récupérer données haute fréquence depuis TwelveData"""
        if not self.twelvedata_key:
            return None
            
        try:
            # Format symbole TwelveData
            td_symbol = symbol.replace('USDT', '/USD')
            
            # Mapping interval TwelveData
            interval_map = {
                '5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h'  
            }
            interval = interval_map.get(timeframe)
            if not interval:
                return None
            
            # Calculer outputsize nécessaire
            minutes_per_interval = self.high_frequency_timeframes[timeframe]['minutes'] 
            total_points = int((hours_back * 60) / minutes_per_interval)
            outputsize = min(total_points, 5000)  # Limit TwelveData
            
            url = "https://api.twelvedata.com/time_series"
            params = {
                "symbol": td_symbol,
                "interval": interval,
                "outputsize": str(outputsize),
                "apikey": self.twelvedata_key
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_twelvedata_hf_data(data, symbol)
                    else:
                        logger.warning(f"TwelveData HF API returned {response.status} for {symbol}")
                        return None
                        
        except Exception as e:
            logger.error(f"TwelveData HF fetch error for {symbol}: {e}")
            return None
    
    async def _fetch_coinapi_hf(self, symbol: str, timeframe: str, hours_back: int) -> Optional[pd.DataFrame]:
        """Récupérer données haute fréquence depuis CoinAPI"""
        if not self.coinapi_key:
            return None
            
        try:
            # Format symbole CoinAPI
            coinapi_symbol = f"{symbol.replace('USDT', '')}_USDT"
            
            # Mapping period CoinAPI
            period_map = {
                '5m': '5MIN', '15m': '15MIN', '1h': '1HRS', '4h': '4HRS'
            }
            period = period_map.get(timeframe)
            if not period:
                return None
                
            # Calculer timestamps
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours_back)
            
            url = f"https://rest.coinapi.io/v1/ohlcv/{coinapi_symbol}/history"
            params = {
                "period_id": period,
                "time_start": start_time.isoformat(),
                "time_end": end_time.isoformat(),
                "limit": min(int(hours_back * 60 / self.high_frequency_timeframes[timeframe]['minutes']), 10000)
            }
            headers = {"X-CoinAPI-Key": self.coinapi_key}
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_coinapi_hf_data(data, symbol)
                    else:
                        logger.warning(f"CoinAPI HF API returned {response.status} for {symbol}")
                        return None
                        
        except Exception as e:
            logger.error(f"CoinAPI HF fetch error for {symbol}: {e}")
            return None
    
    def _parse_binance_hf_data(self, data: List, symbol: str) -> Optional[pd.DataFrame]:
        """Parser les données haute fréquence de Binance"""
        try:
            if not data or len(data) < 5:
                return None
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Conversion format standard
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Conversion numérique
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            return df.sort_index()
            
        except Exception as e:
            logger.error(f"Error parsing Binance HF data for {symbol}: {e}")
            return None
    
    def _parse_coinbase_hf_data(self, data: List, symbol: str) -> Optional[pd.DataFrame]:
        """Parser les données haute fréquence de Coinbase"""
        try:
            if not data or len(data) < 5:
                return None
            
            # Coinbase format: [timestamp, low, high, open, close, volume]
            records = []
            for item in data:
                if len(item) >= 6:
                    records.append({
                        'timestamp': pd.to_datetime(item[0], unit='s'),
                        'Open': float(item[3]),
                        'High': float(item[2]),
                        'Low': float(item[1]),
                        'Close': float(item[4]),
                        'Volume': float(item[5])
                    })
            
            if not records:
                return None
                
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            return df.sort_index()
            
        except Exception as e:
            logger.error(f"Error parsing Coinbase HF data for {symbol}: {e}")
            return None
    
    def _parse_kraken_hf_data(self, data: Dict, symbol: str, kraken_symbol: str) -> Optional[pd.DataFrame]:
        """Parser les données haute fréquence de Kraken"""
        try:
            if 'result' not in data or not data['result']:
                return None
            
            # Kraken retourne les données sous clé du symbole
            ohlc_data = None
            for key in data['result'].keys():
                if key != 'last':  # 'last' est metadata, pas OHLC
                    ohlc_data = data['result'][key]
                    break
            
            if not ohlc_data or len(ohlc_data) < 5:
                return None
            
            # Format Kraken: [timestamp, open, high, low, close, vwap, volume, count]
            records = []
            for item in ohlc_data:
                if len(item) >= 7:
                    records.append({
                        'timestamp': pd.to_datetime(float(item[0]), unit='s'),
                        'Open': float(item[1]),
                        'High': float(item[2]),
                        'Low': float(item[3]),
                        'Close': float(item[4]),
                        'Volume': float(item[6])
                    })
            
            if not records:
                return None
                
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            return df.sort_index()
            
        except Exception as e:
            logger.error(f"Error parsing Kraken HF data for {symbol}: {e}")
            return None
    
    def _parse_bingx_hf_data(self, data: List, symbol: str) -> Optional[pd.DataFrame]:
        """Parser les données haute fréquence de BingX"""
        try:
            if not data or len(data) < 5:
                return None
            
            # Format BingX similaire à Binance généralement
            # Adapter selon le format exact retourné par bingx_official_engine
            records = []
            for item in data:
                # Supposer format [timestamp, open, high, low, close, volume]
                if len(item) >= 6:
                    records.append({
                        'timestamp': pd.to_datetime(item[0], unit='ms'),
                        'Open': float(item[1]),
                        'High': float(item[2]),
                        'Low': float(item[3]),
                        'Close': float(item[4]),
                        'Volume': float(item[5])
                    })
            
            if not records:
                return None
                
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            return df.sort_index()
            
        except Exception as e:
            logger.error(f"Error parsing BingX HF data for {symbol}: {e}")
            return None
    
    def _parse_alphavantage_hf_data(self, data: Dict, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Parser les données haute fréquence d'Alpha Vantage"""
        try:
            # Alpha Vantage format: "Time Series (5min)" etc.
            time_series_key = f"Time Series ({interval})"
            time_series = data.get(time_series_key, {})
            
            if not time_series:
                return None
            
            records = []
            for timestamp_str, values in time_series.items():
                records.append({
                    'timestamp': pd.to_datetime(timestamp_str),
                    'Open': float(values.get('1. open', 0)),
                    'High': float(values.get('2. high', 0)),
                    'Low': float(values.get('3. low', 0)),
                    'Close': float(values.get('4. close', 0)),
                    'Volume': float(values.get('5. volume', 0))
                })
            
            if not records:
                return None
                
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            return df.sort_index()
            
        except Exception as e:
            logger.error(f"Error parsing Alpha Vantage HF data for {symbol}: {e}")
            return None
    
    def _parse_twelvedata_hf_data(self, data: Dict, symbol: str) -> Optional[pd.DataFrame]:
        """Parser les données haute fréquence de TwelveData"""
        try:
            if 'values' not in data or not data['values']:
                return None
            
            records = []
            for item in data['values']:
                records.append({
                    'timestamp': pd.to_datetime(item['datetime']),
                    'Open': float(item['open']),
                    'High': float(item['high']),
                    'Low': float(item['low']),
                    'Close': float(item['close']),
                    'Volume': float(item.get('volume', 0))
                })
            
            if not records:
                return None
                
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            return df.sort_index()
            
        except Exception as e:
            logger.error(f"Error parsing TwelveData HF data for {symbol}: {e}")
            return None
    
    def _parse_coinapi_hf_data(self, data: List, symbol: str) -> Optional[pd.DataFrame]:
        """Parser les données haute fréquence de CoinAPI"""
        try:
            if not data or len(data) < 5:
                return None
            
            records = []
            for item in data:
                records.append({
                    'timestamp': pd.to_datetime(item['time_period_start']),
                    'Open': float(item['price_open']),
                    'High': float(item['price_high']),
                    'Low': float(item['price_low']),
                    'Close': float(item['price_close']),
                    'Volume': float(item.get('volume_traded', 0))
                })
            
            if not records:
                return None
                
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            return df.sort_index()
            
        except Exception as e:
            logger.error(f"Error parsing CoinAPI HF data for {symbol}: {e}")
            return None
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculer un score de qualité des données (0-1)"""
        try:
            if df is None or len(df) == 0:
                return 0.0
            
            quality_factors = []
            
            # 1. Complétude des données (pas de NaN)
            completeness = 1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
            quality_factors.append(completeness)
            
            # 2. Cohérence OHLC (High >= Low, Close entre High/Low)
            ohlc_coherence = 0.0
            if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                valid_ohlc = (
                    (df['High'] >= df['Low']) & 
                    (df['Close'] >= df['Low']) & 
                    (df['Close'] <= df['High']) &
                    (df['Open'] >= df['Low']) & 
                    (df['Open'] <= df['High'])
                ).sum()
                ohlc_coherence = valid_ohlc / len(df)
            quality_factors.append(ohlc_coherence)
            
            # 3. Continuité temporelle (pas de gaps importants)
            if len(df) > 1:
                time_diffs = df.index.to_series().diff().dropna()
                median_diff = time_diffs.median()
                # Calculer proportion de gaps "normaux" (<=2x median)
                normal_gaps = (time_diffs <= median_diff * 2).sum()
                time_continuity = normal_gaps / len(time_diffs) if len(time_diffs) > 0 else 1.0
            else:
                time_continuity = 1.0
            quality_factors.append(time_continuity)
            
            # 4. Volume cohérence (pas de valeurs aberrantes)
            volume_coherence = 1.0
            if 'Volume' in df.columns and df['Volume'].sum() > 0:
                # Détecter outliers volume (> 5x médiane ou < 5% médiane)
                median_volume = df['Volume'].median()
                if median_volume > 0:
                    normal_volume = ((df['Volume'] >= median_volume * 0.05) & 
                                   (df['Volume'] <= median_volume * 5)).sum()
                    volume_coherence = normal_volume / len(df)
            quality_factors.append(volume_coherence)
            
            # Score final: moyenne pondérée
            weights = [0.3, 0.4, 0.2, 0.1]  # OHLC coherence plus important
            final_score = sum(factor * weight for factor, weight in zip(quality_factors, weights))
            
            return min(max(final_score, 0.0), 1.0)  # Clamp entre 0-1
            
        except Exception as e:
            logger.error(f"Error calculating data quality score: {e}")
            return 0.5  # Score moyen en cas d'erreur
    
    async def _calculate_high_precision_sr_levels(self, hf_data: HighFrequencyData) -> HighFrequencyData:
        """
        🎯 Calculer les niveaux Support/Résistance haute précision depuis données HF
        
        Utilise différentes méthodes:
        - Pivot Points sur données 5m/15m (4-6h) → Niveaux micro
        - Support/Résistance testés sur 1h/4h (24h) → Niveaux intraday  
        """
        
        try:
            df = hf_data.data
            logger.info(f"🧮 Calculating high-precision S/R levels for {hf_data.symbol}")
            logger.info(f"   Data: {len(df)} points from {hf_data.source} ({hf_data.timeframe})")
            
            # 1. Niveaux MICRO (4-6 dernières heures) 
            # Utiliser pivot points + zones de congestion
            if hf_data.timeframe in ['5m', '15m']:
                micro_support, micro_resistance = self._calculate_micro_sr_levels(df)
                hf_data.micro_support_levels = micro_support
                hf_data.micro_resistance_levels = micro_resistance
                logger.info(f"   📍 Micro levels: S={micro_support} R={micro_resistance}")
            
            # 2. Niveaux INTRADAY (24 dernières heures)
            # Utiliser support/résistance testés + swing highs/lows
            if hf_data.timeframe in ['1h', '4h'] or len(df) >= 24:
                intraday_support, intraday_resistance = self._calculate_intraday_sr_levels(df)
                hf_data.intraday_support_levels = intraday_support
                hf_data.intraday_resistance_levels = intraday_resistance
                logger.info(f"   📍 Intraday levels: S={intraday_support} R={intraday_resistance}")
            
            logger.info(f"✅ S/R calculation completed for {hf_data.symbol}")
            return hf_data
            
        except Exception as e:
            logger.error(f"Error calculating S/R levels for {hf_data.symbol}: {e}")
            return hf_data  # Retourner même sans S/R si erreur
    
    def _calculate_micro_sr_levels(self, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Calculer niveaux S/R micro (4-6h) basés sur pivot points et zones de congestion"""
        try:
            support_levels = []
            resistance_levels = []
            
            if len(df) < 20:  # Minimum de données
                return support_levels, resistance_levels
            
            # 1. Pivot Points classiques
            # Prendre dernières 4-6h de données (selon timeframe)
            recent_periods = min(len(df), 72 if '5m' in str(df.index.freq) else 24)  # 6h pour 5m, 24h pour 15m
            recent_df = df.tail(recent_periods)
            
            high_prices = recent_df['High'].values
            low_prices = recent_df['Low'].values
            close_prices = recent_df['Close'].values
            
            # Détecter swing highs et lows (pics locaux)
            from scipy.signal import find_peaks
            
            # Pics résistance (swing highs) 
            peaks, _ = find_peaks(high_prices, distance=5, prominence=high_prices.std()*0.5)
            for peak_idx in peaks:
                resistance_levels.append(high_prices[peak_idx])
            
            # Creux support (swing lows)
            valleys, _ = find_peaks(-low_prices, distance=5, prominence=low_prices.std()*0.5)  # Invert for valleys
            for valley_idx in valleys:
                support_levels.append(low_prices[valley_idx])
            
            # 2. Zones de congestion (prix testés plusieurs fois)
            # Grouper les prix par bins et identifier ceux les plus "visités"
            price_range = high_prices.max() - low_prices.min()
            num_bins = min(int(price_range / (close_prices[-1] * 0.001)), 50)  # Bins de 0.1%
            
            if num_bins > 5:
                hist, bin_edges = np.histogram(close_prices, bins=num_bins)
                # Prendre les bins les plus fréquentés (top 20%)
                top_bins = np.argsort(hist)[-max(1, num_bins//5):]  # Top 20%
                
                for bin_idx in top_bins:
                    bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
                    current_price = close_prices[-1]
                    
                    # Classer comme support ou résistance selon position vs prix actuel
                    if bin_center < current_price:
                        support_levels.append(bin_center)
                    else:
                        resistance_levels.append(bin_center)
            
            # 3. Nettoyer et trier les niveaux
            current_price = close_prices[-1]
            
            # Support: garder niveaux below current price, trier décroissant
            support_levels = [s for s in support_levels if s < current_price and s > current_price * 0.95]  # Max 5% en dessous
            support_levels = sorted(list(set(support_levels)), reverse=True)[:5]  # Top 5
            
            # Résistance: garder niveaux above current price, trier croissant  
            resistance_levels = [r for r in resistance_levels if r > current_price and r < current_price * 1.05]  # Max 5% au dessus
            resistance_levels = sorted(list(set(resistance_levels)))[:5]  # Top 5
            
            logger.debug(f"Micro S/R calculated: {len(support_levels)} supports, {len(resistance_levels)} resistances")
            return support_levels, resistance_levels
            
        except Exception as e:
            logger.error(f"Error calculating micro S/R levels: {e}")
            return [], []
    
    def _calculate_intraday_sr_levels(self, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Calculer niveaux S/R intraday (24h) basés sur support/résistance testés"""
        try:
            support_levels = []
            resistance_levels = []
            
            if len(df) < 10:
                return support_levels, resistance_levels
            
            # Prendre jusqu'à 24h de données (selon timeframe)
            lookback_periods = min(len(df), 24 if 'H' in str(df.index.freq) else 144)  # 24h pour hourly, 144 pour 10min
            intraday_df = df.tail(lookback_periods)
            
            high_prices = intraday_df['High'].values
            low_prices = intraday_df['Low'].values
            close_prices = intraday_df['Close'].values
            volumes = intraday_df['Volume'].values if 'Volume' in intraday_df.columns else None
            
            # 1. Support/Résistance par fractales
            def is_fractal_high(data, idx, window=2):
                """Check if point is a fractal high (local maximum)"""
                if idx < window or idx >= len(data) - window:
                    return False
                current = data[idx]
                return all(current >= data[i] for i in range(idx-window, idx+window+1) if i != idx)
            
            def is_fractal_low(data, idx, window=2):
                """Check if point is a fractal low (local minimum)"""  
                if idx < window or idx >= len(data) - window:
                    return False
                current = data[idx]
                return all(current <= data[i] for i in range(idx-window, idx+window+1) if i != idx)
            
            # Identifier fractales
            for i in range(len(high_prices)):
                if is_fractal_high(high_prices, i):
                    resistance_levels.append(high_prices[i])
                if is_fractal_low(low_prices, i):
                    support_levels.append(low_prices[i])
            
            # 2. Niveaux de volume élevé (si volume disponible)
            if volumes is not None and volumes.sum() > 0:
                # VWAP et écarts-types comme niveaux
                vwap = np.average(close_prices, weights=volumes)
                price_variance = np.average((close_prices - vwap)**2, weights=volumes)
                vwap_std = np.sqrt(price_variance)
                
                current_price = close_prices[-1]
                
                # VWAP bands comme S/R
                vwap_upper = vwap + vwap_std
                vwap_lower = vwap - vwap_std
                
                if vwap_upper > current_price:
                    resistance_levels.append(vwap_upper)
                if vwap_lower < current_price:
                    support_levels.append(vwap_lower)
                
                # VWAP lui-même peut être S ou R
                if abs(vwap - current_price) > current_price * 0.005:  # Si >0.5% d'écart
                    if vwap > current_price:
                        resistance_levels.append(vwap)
                    else:
                        support_levels.append(vwap)
            
            # 3. Niveaux psychologiques (round numbers)
            current_price = close_prices[-1]
            price_magnitude = 10 ** (len(str(int(current_price))) - 1)  # Ordre de grandeur
            
            # Chercher niveaux ronds proches
            for multiplier in [1, 1.5, 2, 2.5, 3, 5]:
                round_level = round(current_price / (price_magnitude * multiplier)) * (price_magnitude * multiplier)
                if abs(round_level - current_price) / current_price < 0.05:  # Dans 5%
                    if round_level > current_price:
                        resistance_levels.append(round_level)
                    elif round_level < current_price:
                        support_levels.append(round_level)
            
            # 4. Nettoyer et trier
            current_price = close_prices[-1]
            
            # Support: garder below current, trier décroissant, éliminer doublons
            support_levels = [s for s in support_levels if s < current_price and s > current_price * 0.90]
            support_levels = sorted(list(set(support_levels)), reverse=True)[:7]  # Top 7
            
            # Résistance: garder above current, trier croissant, éliminer doublons
            resistance_levels = [r for r in resistance_levels if r > current_price and r < current_price * 1.10]
            resistance_levels = sorted(list(set(resistance_levels)))[:7]  # Top 7
            
            logger.debug(f"Intraday S/R calculated: {len(support_levels)} supports, {len(resistance_levels)} resistances")
            return support_levels, resistance_levels
            
        except Exception as e:
            logger.error(f"Error calculating intraday S/R levels: {e}")
            return [], []
    
    async def calculate_enhanced_sr_levels(self, 
                                          symbol: str, 
                                          hf_data: HighFrequencyData,
                                          daily_support: float, 
                                          daily_resistance: float) -> EnhancedSupportResistance:
        """
        🎯 Calculer Enhanced S/R combinant niveaux daily + haute fréquence
        """
        try:
            current_price = hf_data.data['Close'].iloc[-1]
            
            # 1. Sélectionner meilleurs niveaux de chaque timeframe
            
            # Micro (plus proche du prix actuel)
            micro_support = None
            micro_resistance = None
            
            if hf_data.micro_support_levels:
                # Prendre le support le plus proche (mais en dessous)
                valid_supports = [s for s in hf_data.micro_support_levels if s < current_price]
                micro_support = max(valid_supports) if valid_supports else daily_support
            else:
                micro_support = daily_support
                
            if hf_data.micro_resistance_levels:
                # Prendre la résistance la plus proche (mais au dessus)
                valid_resistances = [r for r in hf_data.micro_resistance_levels if r > current_price]
                micro_resistance = min(valid_resistances) if valid_resistances else daily_resistance
            else:
                micro_resistance = daily_resistance
            
            # Intraday (niveaux intermédiaires)
            intraday_support = None
            intraday_resistance = None
            
            if hf_data.intraday_support_levels:
                valid_supports = [s for s in hf_data.intraday_support_levels if s < current_price]
                intraday_support = max(valid_supports) if valid_supports else daily_support
            else:
                intraday_support = daily_support
                
            if hf_data.intraday_resistance_levels:
                valid_resistances = [r for r in hf_data.intraday_resistance_levels if r > current_price]
                intraday_resistance = min(valid_resistances) if valid_resistances else daily_resistance
            else:
                intraday_resistance = daily_resistance
            
            # 2. Calculer confiance basée sur qualité données et proximité niveaux
            confidence_micro = min(hf_data.quality_score * 1.2, 1.0) if hf_data.timeframe in ['5m', '15m'] else 0.7
            confidence_intraday = min(hf_data.quality_score * 1.1, 1.0) if hf_data.timeframe in ['1h', '4h'] else 0.8
            confidence_daily = 0.9  # Daily data généralement fiable
            
            enhanced_sr = EnhancedSupportResistance(
                symbol=symbol,
                daily_support=daily_support,
                daily_resistance=daily_resistance,
                intraday_support=intraday_support,
                intraday_resistance=intraday_resistance,
                micro_support=micro_support,
                micro_resistance=micro_resistance,
                confidence_daily=confidence_daily,
                confidence_intraday=confidence_intraday,
                confidence_micro=confidence_micro,
                calculation_source=hf_data.source,
                calculation_timestamp=datetime.now(timezone.utc)
            )
            
            logger.info(f"✅ Enhanced S/R calculated for {symbol}:")
            logger.info(f"   Daily: S={daily_support:.6f} R={daily_resistance:.6f}")
            logger.info(f"   Intraday: S={intraday_support:.6f} R={intraday_resistance:.6f}")
            logger.info(f"   Micro: S={micro_support:.6f} R={micro_resistance:.6f}")
            
            return enhanced_sr
            
        except Exception as e:
            logger.error(f"Error calculating enhanced S/R for {symbol}: {e}")
            # Return fallback avec niveaux daily
            return EnhancedSupportResistance(
                symbol=symbol,
                daily_support=daily_support,
                daily_resistance=daily_resistance,
                intraday_support=daily_support,
                intraday_resistance=daily_resistance,
                micro_support=daily_support,
                micro_resistance=daily_resistance,
                confidence_daily=0.8,
                confidence_intraday=0.6,
                confidence_micro=0.5,
                calculation_source="fallback",
                calculation_timestamp=datetime.now(timezone.utc)
            )
    
    async def calculate_dynamic_risk_reward(self,
                                           symbol: str,
                                           signal_type: str,
                                           entry_price: float,
                                           enhanced_sr: EnhancedSupportResistance) -> DynamicRiskReward:
        """
        🎯 CALCULER RR DYNAMIQUE avec niveaux S/R haute précision
        
        Logique:
        - RR Micro: Basé sur niveaux 5m (trading à court terme)
        - RR Intraday: Basé sur niveaux 1h (swing trading)  
        - RR Daily: Basé sur niveaux daily (position trading)
        - RR Final: Sélection intelligente selon contexte
        """
        try:
            logger.info(f"🧮 Calculating dynamic RR for {symbol} {signal_type} @ {entry_price:.6f}")
            
            # 1. Calculer RR pour chaque timeframe
            if signal_type.upper() == "LONG":
                # LONG: RR = (TP - Entry) / (Entry - SL)
                
                # Micro RR (niveaux haute fréquence)
                micro_risk = entry_price - enhanced_sr.micro_support if enhanced_sr.micro_support else 0
                micro_reward = enhanced_sr.micro_resistance - entry_price if enhanced_sr.micro_resistance else 0
                rr_micro = micro_reward / micro_risk if micro_risk > 0 else 0
                
                # Intraday RR 
                intraday_risk = entry_price - enhanced_sr.intraday_support if enhanced_sr.intraday_support else 0
                intraday_reward = enhanced_sr.intraday_resistance - entry_price if enhanced_sr.intraday_resistance else 0
                rr_intraday = intraday_reward / intraday_risk if intraday_risk > 0 else 0
                
                # Daily RR
                daily_risk = entry_price - enhanced_sr.daily_support if enhanced_sr.daily_support else 0
                daily_reward = enhanced_sr.daily_resistance - entry_price if enhanced_sr.daily_resistance else 0
                rr_daily = daily_reward / daily_risk if daily_risk > 0 else 0
                
            elif signal_type.upper() == "SHORT":
                # SHORT: RR = (Entry - TP) / (SL - Entry)
                
                # Micro RR
                micro_reward = entry_price - enhanced_sr.micro_support if enhanced_sr.micro_support else 0
                micro_risk = enhanced_sr.micro_resistance - entry_price if enhanced_sr.micro_resistance else 0
                rr_micro = micro_reward / micro_risk if micro_risk > 0 else 0
                
                # Intraday RR
                intraday_reward = entry_price - enhanced_sr.intraday_support if enhanced_sr.intraday_support else 0
                intraday_risk = enhanced_sr.intraday_resistance - entry_price if enhanced_sr.intraday_resistance else 0
                rr_intraday = intraday_reward / intraday_risk if intraday_risk > 0 else 0
                
                # Daily RR
                daily_reward = entry_price - enhanced_sr.daily_support if enhanced_sr.daily_support else 0
                daily_risk = enhanced_sr.daily_resistance - entry_price if enhanced_sr.daily_resistance else 0
                rr_daily = daily_reward / daily_risk if daily_risk > 0 else 0
                
            else:  # HOLD
                # Pour HOLD, utiliser RR neutre/conservateur
                rr_micro = 1.0
                rr_intraday = 1.0
                rr_daily = 1.0
            
            # 2. Sélection RR final basée sur logique intelligente
            rr_final, selection_logic, selected_support, selected_resistance, selected_source = self._select_optimal_rr(
                signal_type, entry_price, enhanced_sr,
                rr_micro, rr_intraday, rr_daily
            )
            
            # 3. Calculer confidence score basé sur cohérence des RR
            confidence_score = self._calculate_rr_confidence(
                rr_micro, rr_intraday, rr_daily, enhanced_sr
            )
            
            dynamic_rr = DynamicRiskReward(
                symbol=symbol,
                signal_type=signal_type,
                entry_price=entry_price,
                rr_micro=rr_micro,
                rr_intraday=rr_intraday,
                rr_daily=rr_daily,
                rr_final=rr_final,
                rr_selection_logic=selection_logic,
                selected_support=selected_support,
                selected_resistance=selected_resistance,
                selected_source=selected_source,
                calculation_timestamp=datetime.now(timezone.utc),
                confidence_score=confidence_score
            )
            
            logger.info(f"✅ Dynamic RR calculated for {symbol}:")
            logger.info(f"   Micro: {rr_micro:.2f}:1 | Intraday: {rr_intraday:.2f}:1 | Daily: {rr_daily:.2f}:1")
            logger.info(f"   🎯 FINAL RR: {rr_final:.2f}:1 ({selection_logic})")
            logger.info(f"   Confidence: {confidence_score:.2f}")
            
            return dynamic_rr
            
        except Exception as e:
            logger.error(f"Error calculating dynamic RR for {symbol}: {e}")
            # Fallback RR conservateur
            return DynamicRiskReward(
                symbol=symbol,
                signal_type=signal_type,
                entry_price=entry_price,
                rr_micro=1.0,
                rr_intraday=1.0,
                rr_daily=1.0,
                rr_final=1.0,
                rr_selection_logic="fallback_conservative",
                selected_support=entry_price * 0.98,
                selected_resistance=entry_price * 1.02,
                selected_source="fallback",
                calculation_timestamp=datetime.now(timezone.utc),
                confidence_score=0.5
            )
    
    def _select_optimal_rr(self, 
                          signal_type: str, 
                          entry_price: float,
                          enhanced_sr: EnhancedSupportResistance,
                          rr_micro: float, 
                          rr_intraday: float, 
                          rr_daily: float) -> Tuple[float, str, float, float, str]:
        """Sélectionner RR optimal basé sur logique intelligente"""
        
        try:
            # Logique de sélection RR:
            # 1. Si RR micro > 2.0 et confidence haute → Utiliser micro (trade rapide)
            # 2. Si RR intraday > 1.5 et cohérent avec daily → Utiliser intraday  
            # 3. Si RR daily > 1.2 → Utiliser daily (conservateur)
            # 4. Sinon prendre le plus conservateur > 1.0
            
            valid_rrs = [(rr_micro, "micro", enhanced_sr.micro_support, enhanced_sr.micro_resistance, enhanced_sr.calculation_source),
                        (rr_intraday, "intraday", enhanced_sr.intraday_support, enhanced_sr.intraday_resistance, enhanced_sr.calculation_source),
                        (rr_daily, "daily", enhanced_sr.daily_support, enhanced_sr.daily_resistance, "daily_data")]
            
            # Filtrer RR > 0.8 (minimum viable)
            valid_rrs = [(rr, timeframe, supp, res, src) for rr, timeframe, supp, res, src in valid_rrs if rr >= 0.8]
            
            if not valid_rrs:
                # Aucun RR viable, utiliser daily comme fallback
                return 1.0, "no_viable_rr_fallback", enhanced_sr.daily_support, enhanced_sr.daily_resistance, "daily_fallback"
            
            # 1. Priorité micro si excellent RR et confiance
            if rr_micro >= 2.0 and enhanced_sr.confidence_micro >= 0.8:
                return rr_micro, "micro_excellent_rr_high_confidence", enhanced_sr.micro_support, enhanced_sr.micro_resistance, enhanced_sr.calculation_source
            
            # 2. Priorité intraday si bon RR et cohérent
            if rr_intraday >= 1.5:
                # Vérifier cohérence avec daily (écart < 50%)
                if rr_daily > 0 and abs(rr_intraday - rr_daily) / max(rr_intraday, rr_daily) <= 0.5:
                    return rr_intraday, "intraday_good_rr_coherent_with_daily", enhanced_sr.intraday_support, enhanced_sr.intraday_resistance, enhanced_sr.calculation_source
                elif enhanced_sr.confidence_intraday >= 0.8:
                    return rr_intraday, "intraday_good_rr_high_confidence", enhanced_sr.intraday_support, enhanced_sr.intraday_resistance, enhanced_sr.calculation_source
            
            # 3. Daily si convenable
            if rr_daily >= 1.2:
                return rr_daily, "daily_conservative_reliable", enhanced_sr.daily_support, enhanced_sr.daily_resistance, "daily_data"
            
            # 4. Prendre le meilleur RR disponible > 1.0
            best_rr_above_1 = [(rr, timeframe, supp, res, src) for rr, timeframe, supp, res, src in valid_rrs if rr >= 1.0]
            if best_rr_above_1:
                best = max(best_rr_above_1, key=lambda x: x[0])  # Max RR
                return best[0], f"best_available_rr_{best[1]}", best[2], best[3], best[4]
            
            # 5. Dernier recours: meilleur RR même si < 1.0
            best = max(valid_rrs, key=lambda x: x[0])
            return best[0], f"last_resort_best_rr_{best[1]}", best[2], best[3], best[4]
            
        except Exception as e:
            logger.error(f"Error selecting optimal RR: {e}")
            return 1.0, "error_fallback", enhanced_sr.daily_support, enhanced_sr.daily_resistance, "error_fallback"
    
    def _calculate_rr_confidence(self, 
                                rr_micro: float, 
                                rr_intraday: float, 
                                rr_daily: float,
                                enhanced_sr: EnhancedSupportResistance) -> float:
        """Calculer confidence score basé sur cohérence des RR et qualité S/R"""
        try:
            confidence_factors = []
            
            # 1. Cohérence entre RR timeframes
            rr_values = [rr for rr in [rr_micro, rr_intraday, rr_daily] if rr > 0]
            if len(rr_values) >= 2:
                rr_std = np.std(rr_values)
                rr_mean = np.mean(rr_values)
                # Coefficient de variation (plus bas = plus cohérent)
                coherence_score = max(0, 1.0 - (rr_std / rr_mean if rr_mean > 0 else 1))
                confidence_factors.append(coherence_score)
            else:
                confidence_factors.append(0.5)  # Score moyen si pas assez de données
            
            # 2. Qualité des données sources
            sr_confidence_avg = np.mean([
                enhanced_sr.confidence_daily,
                enhanced_sr.confidence_intraday, 
                enhanced_sr.confidence_micro
            ])
            confidence_factors.append(sr_confidence_avg)
            
            # 3. Réalisme des RR (ni trop bas ni trop élevés)
            rr_realism = 0.0
            for rr in rr_values:
                if 0.8 <= rr <= 5.0:  # Range réaliste
                    rr_realism += 1.0
                elif 0.5 <= rr < 0.8 or 5.0 < rr <= 10.0:  # Acceptable
                    rr_realism += 0.7
                else:  # Peu réaliste
                    rr_realism += 0.3
            
            rr_realism_score = rr_realism / len(rr_values) if rr_values else 0.5
            confidence_factors.append(rr_realism_score)
            
            # Score final: moyenne pondérée
            weights = [0.4, 0.4, 0.2]  # Cohérence et qualité plus importants
            final_confidence = sum(factor * weight for factor, weight in zip(confidence_factors, weights))
            
            return min(max(final_confidence, 0.0), 1.0)  # Clamp 0-1
            
        except Exception as e:
            logger.error(f"Error calculating RR confidence: {e}")
            return 0.6  # Score moyen par défaut
    
    async def get_ohlcv_metadata_from_existing_data(self, existing_data: pd.DataFrame, primary_source: str = "unknown") -> OHLCVMetadata:
        """
        🎯 Créer métadonnées OHLCV depuis données existantes (utilisées par IA1)
        
        Cette fonction analyse les données existantes pour déterminer leur qualité
        et préparer la logique de complétion intelligente
        """
        
        try:
            # Calculer qualité des données existantes
            data_quality = self._calculate_data_quality_score(existing_data)
            data_count = len(existing_data)
            
            # Inférer timeframe des données existantes
            if len(existing_data) > 1:
                time_diff = (existing_data.index[-1] - existing_data.index[-2])
                if time_diff.total_seconds() <= 300:  # <= 5min
                    primary_timeframe = "5m"
                elif time_diff.total_seconds() <= 3600:  # <= 1h
                    primary_timeframe = "1h" 
                elif time_diff.total_seconds() <= 86400:  # <= 1d
                    primary_timeframe = "1d"
                else:
                    primary_timeframe = "unknown"
            else:
                primary_timeframe = "1d"  # Default assumption
            
            # Sources probablement utilisées (basé sur attributs du DataFrame si disponible)
            sources_used = [primary_source]
            if hasattr(existing_data, 'attrs') and 'sources_used' in existing_data.attrs:
                sources_used.extend(existing_data.attrs['sources_used'])
            
            # Déterminer sources préférées pour complétion
            preferred_completion_sources = self.source_priority_matrix.get(
                primary_source, 
                ['binance', 'coinbase', 'bingx']
            )
            
            metadata = OHLCVMetadata(
                primary_source=primary_source,
                primary_timeframe=primary_timeframe,
                primary_data_quality=data_quality,
                primary_data_count=data_count,
                sources_used=sources_used,
                avoided_sources=sources_used.copy(),  # Éviter redondance
                completion_needed=True,  # Par défaut on veut complétion
                preferred_completion_sources=preferred_completion_sources
            )
            
            logger.info(f"📊 OHLCV Metadata created:")
            logger.info(f"   Primary: {primary_source} ({primary_timeframe}) - Quality: {data_quality:.2f}")
            logger.info(f"   Data count: {data_count}")
            logger.info(f"   Preferred completion: {preferred_completion_sources[:3]}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error creating OHLCV metadata: {e}")
            # Fallback metadata 
            return OHLCVMetadata(
                primary_source=primary_source,
                primary_timeframe="1d",
                primary_data_quality=0.7,
                primary_data_count=len(existing_data) if existing_data is not None else 0,
                sources_used=[primary_source],
                avoided_sources=[primary_source],
                completion_needed=True,
                preferred_completion_sources=['binance', 'coinbase', 'bingx']
            )

# Instance globale
intelligent_ohlcv_fetcher = IntelligentOHLCVFetcher()
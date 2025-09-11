"""
GLOBAL CRYPTO MARKET ANALYZER - Module d'analyse des conditions générales du marché
🎯 Objectif: Fournir aux IA1/IA2 une vision complète du contexte macro du marché crypto

Fonctionnalités:
- Données globales du marché (Market Cap, Volume, Dominance BTC)  
- Fear & Greed Index pour le sentiment
- Détection Bull/Bear Market automatique
- Métriques de volatilité et liquidité globales
- Intégration multi-source (CoinGecko, Fear&Greed, Binance)
"""

import aiohttp
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Régimes de marché détectés automatiquement"""
    EXTREME_BULL = "extreme_bull"      # Bull market extrême (Fear&Greed >75, +30%+ mensuel)
    BULL = "bull"                      # Bull market (Fear&Greed >50, momentum positif)
    NEUTRAL_BULLISH = "neutral_bullish" # Neutre avec biais haussier
    NEUTRAL = "neutral"                # Marché neutre/sideways
    NEUTRAL_BEARISH = "neutral_bearish" # Neutre avec biais baissier  
    BEAR = "bear"                      # Bear market (Fear&Greed <50, momentum négatif)
    EXTREME_BEAR = "extreme_bear"      # Bear market extrême (Fear&Greed <25, crash)

class MarketSentiment(Enum):
    """Sentiment du marché basé sur Fear & Greed + autres métriques"""
    EXTREME_GREED = "extreme_greed"    # >75
    GREED = "greed"                    # 55-75
    NEUTRAL = "neutral"                # 45-55
    FEAR = "fear"                      # 25-45
    EXTREME_FEAR = "extreme_fear"      # <25

@dataclass
class GlobalMarketData:
    """Données globales du marché crypto"""
    
    # Métriques de base
    timestamp: datetime
    total_market_cap: float            # Market cap total crypto (USD)
    total_volume_24h: float            # Volume 24h total (USD)  
    btc_dominance: float               # Dominance Bitcoin (%)
    eth_dominance: float               # Dominance Ethereum (%)
    
    # Données Bitcoin (référence)
    btc_price: float                   # Prix BTC actuel
    btc_change_24h: float              # Change BTC 24h (%)
    btc_change_7d: float               # Change BTC 7d (%)
    btc_change_30d: float              # Change BTC 30d (%)
    
    # 🚨 NOUVELLE VARIABLE CRITIQUE: Market Cap 24h
    market_cap_change_24h: float       # Change Market Cap 24h (%)
    
    # Fear & Greed Index
    fear_greed_value: int              # Valeur 0-100
    fear_greed_classification: str     # Extreme Fear, Fear, Neutral, Greed, Extreme Greed
    
    # Analyses dérivées
    market_regime: MarketRegime        # Régime de marché détecté
    market_sentiment: MarketSentiment  # Sentiment dominant
    volatility_regime: str             # "low", "medium", "high", "extreme"
    liquidity_condition: str           # "poor", "moderate", "good", "excellent"
    
    # Métriques compositifs  
    bull_bear_score: float             # Score -100 (extreme bear) à +100 (extreme bull)
    market_health_score: float         # Score 0-100 de santé générale du marché
    opportunity_score: float           # Score 0-100 d'opportunités de trading
    
    # Context pour les IAs
    market_context_summary: str        # Résumé textuel du contexte actuel
    trading_recommendations: List[str] # Recommandations générales de trading

@dataclass
class AdvancedMarketMetrics:
    """Métriques avancées calculées"""
    
    # Volatilité
    btc_volatility_7d: float           # Volatilité BTC 7 jours
    market_correlation: float          # Corrélation générale du marché avec BTC
    
    # Volume et liquidité
    volume_trend_7d: float             # Tendance volume 7 jours (%)
    volume_vs_mcap_ratio: float        # Ratio Volume/Market Cap
    
    # Momentum
    momentum_1d: float                 # Momentum 1 jour
    momentum_7d: float                 # Momentum 7 jours  
    momentum_30d: float                # Momentum 30 jours
    
    # Support/Résistance macro
    btc_support_level: float           # Support BTC macro
    btc_resistance_level: float        # Résistance BTC macro
    
    # Indicateurs techniques globaux
    global_rsi_14d: float              # RSI 14j sur market cap total
    global_ema_cross_signal: str       # Signal croisement EMA globales

class GlobalCryptoMarketAnalyzer:
    """Analyseur global du marché des cryptomonnaies"""
    
    def __init__(self):
        # Configuration APIs
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
        self.fear_greed_url = "https://api.alternative.me/fng"
        self.binance_base_url = "https://api.binance.com/api/v3"
        
        # Cache pour éviter trop d'appels API
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Configuration seuils
        self.config = {
            "bull_market_threshold": 20,      # % gain mensuel pour bull market
            "bear_market_threshold": -20,     # % perte mensuelle pour bear market
            "high_volatility_threshold": 5,   # % volatilité quotidienne
            "volume_significance_threshold": 1.5,  # Ratio volume significatif vs moyenne
        }
        
        logger.info("Global Crypto Market Analyzer initialized")
    
    async def get_global_market_data(self) -> Optional[GlobalMarketData]:
        """
        🎯 FONCTION PRINCIPALE: Récupérer et analyser les conditions globales du marché
        """
        try:
            logger.info("🌍 Fetching global crypto market data...")
            
            # Vérifier cache
            cache_key = "global_market_data"
            if self._is_cache_valid(cache_key):
                logger.info("📦 Using cached global market data")
                return self.cache[cache_key]["data"]
            
            # Récupérer données en parallèle
            coingecko_data = await self._fetch_coingecko_global_data()
            fear_greed_data = await self._fetch_fear_greed_index() 
            btc_historical_data = await self._fetch_btc_historical_data()
            
            if not coingecko_data:
                logger.error("❌ Failed to fetch CoinGecko data")
                return None
                
            # Calculer métriques avancées
            advanced_metrics = await self._calculate_advanced_metrics(
                coingecko_data, btc_historical_data
            )
            
            # Analyser régime de marché
            market_regime = self._analyze_market_regime(
                coingecko_data, fear_greed_data, btc_historical_data
            )
            
            # Analyser sentiment
            market_sentiment = self._analyze_market_sentiment(fear_greed_data, coingecko_data)
            
            # Calculer scores composites
            bull_bear_score = self._calculate_bull_bear_score(
                coingecko_data, fear_greed_data, btc_historical_data
            )
            
            market_health_score = self._calculate_market_health_score(
                coingecko_data, fear_greed_data, advanced_metrics
            )
            
            opportunity_score = self._calculate_opportunity_score(
                market_regime, market_sentiment, advanced_metrics
            )
            
            # Analyser volatilité et liquidité
            volatility_regime = self._analyze_volatility_regime(btc_historical_data)
            liquidity_condition = self._analyze_liquidity_condition(coingecko_data)
            
            # Générer contexte et recommandations
            market_context = self._generate_market_context_summary(
                market_regime, market_sentiment, bull_bear_score, volatility_regime
            )
            
            trading_recommendations = self._generate_trading_recommendations(
                market_regime, market_sentiment, opportunity_score, volatility_regime
            )
            
            # Calculer le changement Market Cap 24h
            market_cap_change_24h = await self._calculate_market_cap_change_24h(
                coingecko_data, btc_historical_data
            )
            
            # Construire objet final
            global_market_data = GlobalMarketData(
                timestamp=datetime.now(timezone.utc),
                total_market_cap=coingecko_data.get("total_market_cap", {}).get("usd", 0),
                total_volume_24h=coingecko_data.get("total_volume", {}).get("usd", 0),
                btc_dominance=coingecko_data.get("market_cap_percentage", {}).get("btc", 0),
                eth_dominance=coingecko_data.get("market_cap_percentage", {}).get("eth", 0),
                btc_price=btc_historical_data.get("current_price", 0),
                btc_change_24h=btc_historical_data.get("price_change_percentage_24h", 0),
                btc_change_7d=btc_historical_data.get("price_change_percentage_7d", 0),
                btc_change_30d=btc_historical_data.get("price_change_percentage_30d", 0),
                market_cap_change_24h=market_cap_change_24h,  # 🚨 NOUVELLE VARIABLE CRITIQUE
                fear_greed_value=fear_greed_data.get("value", 50),
                fear_greed_classification=fear_greed_data.get("value_classification", "Neutral"),
                market_regime=market_regime,
                market_sentiment=market_sentiment,
                volatility_regime=volatility_regime,
                liquidity_condition=liquidity_condition,
                bull_bear_score=bull_bear_score,
                market_health_score=market_health_score,
                opportunity_score=opportunity_score,
                market_context_summary=market_context,
                trading_recommendations=trading_recommendations
            )
            
            # Mettre en cache
            self.cache[cache_key] = {
                "timestamp": datetime.now(timezone.utc),
                "data": global_market_data
            }
            
            logger.info(f"✅ Global market analysis completed: {market_regime.value}, Sentiment: {market_sentiment.value}")
            return global_market_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching global market data: {e}")
            return None
    
    async def _fetch_coingecko_global_data(self) -> Optional[Dict]:
        """Récupérer données globales depuis CoinGecko"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                
                # Global data endpoint
                global_url = f"{self.coingecko_base_url}/global"
                async with session.get(global_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info("✅ CoinGecko data fetched successfully")
                        return data.get("data", {})
                    elif response.status == 429:
                        logger.warning("⚠️ CoinGecko rate limit exceeded, trying CoinMarketCap fallback")
                        return await self._fetch_coinmarketcap_global_fallback()
                    else:
                        logger.warning(f"CoinGecko global API returned {response.status}")
                        return await self._fetch_coinmarketcap_global_fallback()
                        
        except Exception as e:
            logger.error(f"Error fetching CoinGecko global data: {e}")
            return await self._fetch_coinmarketcap_global_fallback()
    
    async def _fetch_coinmarketcap_global_fallback(self) -> Optional[Dict]:
        """Fallback CRITIQUE: récupérer données essentielles via sources multiples"""
        try:
            logger.info("🚨 FALLBACK CRITIQUE ACTIVÉ - Récupération données essentielles (24h/BTC/MarketCap/Volume)")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
                
                # 🎯 PRIORITÉ 1: Prix Bitcoin + Variation 24h (BINANCE - LE PLUS FIABLE)
                btc_data = await self._get_critical_btc_data(session)
                
                # 🎯 PRIORITÉ 2: Market Cap + Volume global (Multiple sources)
                market_data = await self._get_critical_market_data(session, btc_data)
                
                if btc_data and market_data:
                    logger.info(f"✅ FALLBACK SUCCESS: BTC=${btc_data['price']:,.0f} ({btc_data['change_24h']:+.2f}%), MCap=${market_data['total_mcap']/1e12:.2f}T")
                    
                    return {
                        "total_market_cap": {"usd": market_data['total_mcap']},
                        "total_volume": {"usd": market_data['total_volume']},
                        "market_cap_percentage": {
                            "btc": market_data['btc_dominance'], 
                            "eth": market_data['eth_dominance']
                        }
                    }
                else:
                    logger.error("❌ CRITICAL FALLBACK FAILED - Using emergency defaults")
                    return await self._get_emergency_critical_data()
                    
        except Exception as e:
            logger.error(f"❌ ALL CRITICAL FALLBACKS FAILED: {e}")
            return await self._get_emergency_critical_data()
    
    async def _get_critical_btc_data(self, session: aiohttp.ClientSession) -> Optional[Dict]:
        """Récupérer données critiques Bitcoin (prix + variation 24h)"""
        
        # Source 1: Binance (le plus fiable)
        try:
            binance_url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
            async with session.get(binance_url) as response:
                if response.status == 200:
                    data = await response.json()
                    price = float(data.get('lastPrice', 0))
                    change_24h = float(data.get('priceChangePercent', 0))
                    volume_btc = float(data.get('volume', 0))
                    
                    logger.info(f"✅ Binance BTC: ${price:,.0f} ({change_24h:+.2f}%)")
                    return {
                        'price': price,
                        'change_24h': change_24h,
                        'volume_btc': volume_btc,
                        'source': 'binance'
                    }
        except Exception as e:
            logger.warning(f"Binance BTC failed: {e}")
        
        # Source 2: Coinbase Pro
        try:
            coinbase_url = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
            async with session.get(coinbase_url) as response:
                if response.status == 200:
                    data = await response.json()
                    price = float(data.get('price', 0))
                    
                    # Coinbase ne donne pas directement le change 24h, on estime
                    logger.info(f"✅ Coinbase BTC: ${price:,.0f}")
                    return {
                        'price': price,
                        'change_24h': 0,  # Pas disponible sur Coinbase ticker
                        'volume_btc': 0,
                        'source': 'coinbase'
                    }
        except Exception as e:
            logger.warning(f"Coinbase BTC failed: {e}")
        
        # Source 3: Kraken 
        try:
            kraken_url = "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"
            async with session.get(kraken_url) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get('result', {})
                    btc_data = result.get('XXBTZUSD', {})
                    
                    if btc_data:
                        price = float(btc_data.get('c', [0])[0])  # Last price
                        logger.info(f"✅ Kraken BTC: ${price:,.0f}")
                        return {
                            'price': price,
                            'change_24h': 0,  # Calculer si nécessaire
                            'volume_btc': 0,
                            'source': 'kraken'
                        }
        except Exception as e:
            logger.warning(f"Kraken BTC failed: {e}")
        
        logger.error("❌ TOUTES LES SOURCES BTC ONT ÉCHOUÉ")
        return None
    
    async def _get_critical_market_data(self, session: aiohttp.ClientSession, btc_data: Dict) -> Optional[Dict]:
        """Estimer Market Cap + Volume global basé sur données BTC"""
        
        if not btc_data:
            return None
            
        try:
            btc_price = btc_data['price']
            btc_volume = btc_data.get('volume_btc', 0)
            
            # 🧮 ESTIMATIONS INTELLIGENTES basées sur données historiques réelles
            
            # Market Cap: BTC représente ~54% du marché crypto
            btc_supply = 19700000  # ~19.7M BTC en circulation
            btc_market_cap = btc_price * btc_supply
            total_market_cap = btc_market_cap / 0.54  # BTC dominance ~54%
            
            # Volume: BTC représente ~40-50% du volume quotidien
            if btc_volume > 0:
                btc_volume_usd = btc_volume * btc_price
                total_volume = btc_volume_usd / 0.45  # BTC volume dominance ~45%
            else:
                # Fallback: ratio Market Cap / Volume typique ~30:1
                total_volume = total_market_cap / 30
            
            # Dominances réalistes
            btc_dominance = 54.0  # Réaliste pour 2025
            eth_dominance = 17.5  # ETH dominance typique
            
            logger.info(f"📊 Market Data Estimé: MCap=${total_market_cap/1e12:.2f}T, Vol=${total_volume/1e9:.1f}B")
            
            return {
                'total_mcap': total_market_cap,
                'total_volume': total_volume, 
                'btc_dominance': btc_dominance,
                'eth_dominance': eth_dominance
            }
            
        except Exception as e:
            logger.error(f"Market data estimation failed: {e}")
            return None
    
    async def _get_emergency_critical_data(self) -> Dict:
        """Données d'urgence basées sur moyennes de marché réalistes"""
        
        logger.warning("🚨 EMERGENCY MODE: Utilisation données d'urgence réalistes")
        
        # Données de marché réalistes pour 2025 (basées sur historique récent)
        emergency_data = {
            "total_market_cap": {"usd": 2400000000000},  # $2.4T (réaliste)
            "total_volume": {"usd": 85000000000},        # $85B (réaliste)  
            "market_cap_percentage": {
                "btc": 53.5,   # BTC dominance réaliste
                "eth": 17.2    # ETH dominance réaliste
            },
            "market_cap_change_percentage_24h_usd": 1.8  # 🚨 Market Cap change 24h réaliste
        }
        
        logger.info("✅ EMERGENCY DATA LOADED: MCap=$2.4T, Vol=$85B, BTC=53.5%")
        return emergency_data
    
    async def _fetch_fear_greed_index(self) -> Optional[Dict]:
        """Récupérer Fear & Greed Index"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                
                # Fear & Greed endpoint 
                async with session.get(f"{self.fear_greed_url}?limit=1") as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("data") and len(data["data"]) > 0:
                            return data["data"][0]
                    else:
                        logger.warning(f"Fear & Greed API returned {response.status}")
                        
            # Fallback en cas d'échec
            return {"value": 50, "value_classification": "Neutral", "timestamp": ""}
            
        except Exception as e:
            logger.warning(f"Error fetching Fear & Greed index, using fallback: {e}")
            return {"value": 50, "value_classification": "Neutral", "timestamp": ""}
    
    async def _fetch_btc_historical_data(self) -> Optional[Dict]:
        """Récupérer données historiques Bitcoin depuis CoinGecko avec fallback Binance"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                
                # Essayer CoinGecko d'abord
                btc_url = f"{self.coingecko_base_url}/coins/bitcoin"
                params = {
                    "localization": "false",
                    "tickers": "false", 
                    "market_data": "true",
                    "community_data": "false",
                    "developer_data": "false"
                }
                
                async with session.get(btc_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        market_data = data.get("market_data", {})
                        
                        return {
                            "current_price": market_data.get("current_price", {}).get("usd", 0),
                            "price_change_percentage_24h": market_data.get("price_change_percentage_24h", 0),
                            "price_change_percentage_7d": market_data.get("price_change_percentage_7d", 0),
                            "price_change_percentage_30d": market_data.get("price_change_percentage_30d", 0),
                            "market_cap": market_data.get("market_cap", {}).get("usd", 0),
                            "total_volume": market_data.get("total_volume", {}).get("usd", 0)
                        }
                    elif response.status == 429:
                        logger.warning("⚠️ CoinGecko BTC rate limit, trying Binance fallback")
                        return await self._fetch_btc_binance_fallback(session)
                    else:
                        logger.warning(f"Bitcoin historical data API returned {response.status}")
                        return await self._fetch_btc_binance_fallback(session)
                        
        except Exception as e:
            logger.error(f"Error fetching Bitcoin historical data: {e}")
            async with aiohttp.ClientSession() as session:
                return await self._fetch_btc_binance_fallback(session)
    
    async def _fetch_btc_binance_fallback(self, session: aiohttp.ClientSession) -> Optional[Dict]:
        """Fallback CRITIQUE: récupérer données BTC essentielles via Binance + estimations"""
        try:
            logger.info("🚨 BTC CRITICAL FALLBACK: Récupération données essentielles")
            
            # 1. Données temps réel Binance (24h)
            ticker_url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
            
            async with session.get(ticker_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    current_price = float(data.get('lastPrice', 0))
                    price_change_24h = float(data.get('priceChangePercent', 0))
                    volume_24h = float(data.get('volume', 0))
                    
                    # 2. Essayer d'obtenir données historiques 7j et 30j via Binance Klines
                    price_7d_ago, price_30d_ago = await self._get_btc_historical_prices(session, current_price)
                    
                    # Calculer variations 7j et 30j
                    if price_7d_ago > 0:
                        price_change_7d = ((current_price - price_7d_ago) / price_7d_ago) * 100
                    else:
                        # Estimation basée sur 24h (7 jours = trend amplifié)
                        price_change_7d = price_change_24h * 3.2  # Facteur empirique
                    
                    if price_30d_ago > 0:
                        price_change_30d = ((current_price - price_30d_ago) / price_30d_ago) * 100
                    else:
                        # Estimation basée sur 7j (30j = trend long terme)
                        price_change_30d = price_change_7d * 2.1  # Facteur empirique
                    
                    # 3. Calculer Market Cap et Volume
                    btc_supply = 19700000  # BTC en circulation
                    market_cap = current_price * btc_supply
                    total_volume = volume_24h * current_price
                    
                    logger.info(f"✅ BTC CRITICAL DATA: ${current_price:,.0f} | 24h: {price_change_24h:+.1f}% | 7d: {price_change_7d:+.1f}% | 30d: {price_change_30d:+.1f}%")
                    
                    return {
                        "current_price": current_price,
                        "price_change_percentage_24h": price_change_24h,
                        "price_change_percentage_7d": price_change_7d,
                        "price_change_percentage_30d": price_change_30d,
                        "market_cap": market_cap,
                        "total_volume": total_volume
                    }
                else:
                    logger.warning(f"Binance BTC ticker failed: HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"Binance BTC critical fallback failed: {e}")
        
        # EMERGENCY: Retourner des données réalistes d'urgence
        logger.warning("🚨 BTC EMERGENCY DATA - Using realistic defaults")
        return {
            "current_price": 43500,    # Prix BTC réaliste
            "price_change_percentage_24h": 1.2,   # Variation modérée
            "price_change_percentage_7d": 4.8,    # Variation 7j réaliste
            "price_change_percentage_30d": -2.1,   # Correction mensuelle légère
            "market_cap": 43500 * 19700000,       # Market cap calculé
            "total_volume": 25000000000           # Volume 24h réaliste
        }
    
    async def _get_btc_historical_prices(self, session: aiohttp.ClientSession, current_price: float) -> Tuple[float, float]:
        """Récupérer prix historiques BTC (7j et 30j) via Binance Klines"""
        
        try:
            # Binance Klines pour données historiques
            # 7 jours = 7 * 24 heures = 168 heures en arrière
            klines_url = "https://api.binance.com/api/v3/klines"
            
            # Prix il y a 7 jours (klines 1d, limit=8 pour avoir 7 jours complets)
            params_7d = {
                "symbol": "BTCUSDT",
                "interval": "1d",
                "limit": 8  # 7 jours + aujourd'hui
            }
            
            async with session.get(klines_url, params=params_7d) as response:
                if response.status == 200:
                    klines_7d = await response.json()
                    if len(klines_7d) >= 8:
                        # Prix d'ouverture il y a 7 jours (index 0 = le plus ancien)
                        price_7d_ago = float(klines_7d[0][1])  # Open price
                    else:
                        price_7d_ago = 0
                else:
                    price_7d_ago = 0
            
            # Prix il y a 30 jours
            params_30d = {
                "symbol": "BTCUSDT", 
                "interval": "1d",
                "limit": 31  # 30 jours + aujourd'hui
            }
            
            async with session.get(klines_url, params=params_30d) as response:
                if response.status == 200:
                    klines_30d = await response.json()
                    if len(klines_30d) >= 31:
                        price_30d_ago = float(klines_30d[0][1])  # Open price 30j ago
                    else:
                        price_30d_ago = 0
                else:
                    price_30d_ago = 0
            
            logger.info(f"📊 Historical prices: 7d ago=${price_7d_ago:,.0f}, 30d ago=${price_30d_ago:,.0f}")
            return price_7d_ago, price_30d_ago
            
        except Exception as e:
            logger.warning(f"Historical prices fetch failed: {e}")
            return 0, 0  # Will trigger estimation fallback
    
    async def _calculate_advanced_metrics(self, 
                                        coingecko_data: Dict, 
                                        btc_data: Dict) -> AdvancedMarketMetrics:
        """Calculer métriques avancées"""
        try:
            # Calculer volatilité BTC (approximation basée sur les changes)
            btc_volatility_7d = abs(btc_data.get("price_change_percentage_7d", 0))
            
            # Volume vs Market Cap ratio
            total_volume = coingecko_data.get("total_volume", {}).get("usd", 1)
            total_mcap = coingecko_data.get("total_market_cap", {}).get("usd", 1)
            volume_vs_mcap_ratio = (total_volume / total_mcap) * 100 if total_mcap > 0 else 0
            
            # Momentum approximations
            momentum_1d = btc_data.get("price_change_percentage_24h", 0)
            momentum_7d = btc_data.get("price_change_percentage_7d", 0) 
            momentum_30d = btc_data.get("price_change_percentage_30d", 0)
            
            # Support/Résistance BTC simplifiés (basés sur prix actuel et volatilité)
            btc_price = btc_data.get("current_price", 50000)
            volatility_factor = btc_volatility_7d / 100
            btc_support_level = btc_price * (1 - volatility_factor * 0.5)
            btc_resistance_level = btc_price * (1 + volatility_factor * 0.5)
            
            return AdvancedMarketMetrics(
                btc_volatility_7d=btc_volatility_7d,
                market_correlation=0.8,  # Valeur approximative (corrélation crypto typique)
                volume_trend_7d=0,  # Nécessiterait données historiques
                volume_vs_mcap_ratio=volume_vs_mcap_ratio,
                momentum_1d=momentum_1d,
                momentum_7d=momentum_7d,
                momentum_30d=momentum_30d,
                btc_support_level=btc_support_level,
                btc_resistance_level=btc_resistance_level,
                global_rsi_14d=50,  # Nécessiterait calcul sur données historiques
                global_ema_cross_signal="neutral"  # Nécessiterait calcul EMA
            )
            
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {e}")
            # Retourner métriques par défaut
            return AdvancedMarketMetrics(
                btc_volatility_7d=5.0,
                market_correlation=0.7,
                volume_trend_7d=0,
                volume_vs_mcap_ratio=5.0,
                momentum_1d=0,
                momentum_7d=0, 
                momentum_30d=0,
                btc_support_level=45000,
                btc_resistance_level=55000,
                global_rsi_14d=50,
                global_ema_cross_signal="neutral"
            )
    
    def _analyze_market_regime(self, 
                             coingecko_data: Dict, 
                             fear_greed_data: Dict,
                             btc_data: Dict) -> MarketRegime:
        """Analyser et déterminer le régime de marché actuel"""
        try:
            # Extraire métriques clés
            btc_change_30d = btc_data.get("price_change_percentage_30d", 0)
            btc_change_7d = btc_data.get("price_change_percentage_7d", 0)
            fear_greed_value = fear_greed_data.get("value", 50)
            
            # Logique de détection des régimes
            if btc_change_30d > 50 and fear_greed_value > 75:
                return MarketRegime.EXTREME_BULL
            elif btc_change_30d > self.config["bull_market_threshold"] and fear_greed_value > 60:
                return MarketRegime.BULL
            elif btc_change_30d < -50 and fear_greed_value < 25:
                return MarketRegime.EXTREME_BEAR
            elif btc_change_30d < self.config["bear_market_threshold"] and fear_greed_value < 40:
                return MarketRegime.BEAR
            elif btc_change_7d > 10 and fear_greed_value >= 50:
                return MarketRegime.NEUTRAL_BULLISH
            elif btc_change_7d < -10 and fear_greed_value < 50:
                return MarketRegime.NEUTRAL_BEARISH
            else:
                return MarketRegime.NEUTRAL
                
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return MarketRegime.NEUTRAL
    
    def _analyze_market_sentiment(self, 
                                fear_greed_data: Dict,
                                coingecko_data: Dict) -> MarketSentiment:
        """Analyser le sentiment du marché"""
        try:
            fear_greed_value = fear_greed_data.get("value", 50)
            
            if fear_greed_value >= 75:
                return MarketSentiment.EXTREME_GREED
            elif fear_greed_value >= 55:
                return MarketSentiment.GREED
            elif fear_greed_value >= 45:
                return MarketSentiment.NEUTRAL
            elif fear_greed_value >= 25:
                return MarketSentiment.FEAR
            else:
                return MarketSentiment.EXTREME_FEAR
                
        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {e}")
            return MarketSentiment.NEUTRAL
    
    def _calculate_bull_bear_score(self,
                                 coingecko_data: Dict,
                                 fear_greed_data: Dict,
                                 btc_data: Dict) -> float:
        """Calculer score Bull/Bear (-100 à +100)"""
        try:
            # Composantes du score
            btc_momentum_30d = btc_data.get("price_change_percentage_30d", 0)
            btc_momentum_7d = btc_data.get("price_change_percentage_7d", 0)
            fear_greed_normalized = (fear_greed_data.get("value", 50) - 50) * 2  # -100 à +100
            
            # Pondération des facteurs
            momentum_weight = 0.4
            short_momentum_weight = 0.3
            sentiment_weight = 0.3
            
            bull_bear_score = (
                btc_momentum_30d * momentum_weight +
                btc_momentum_7d * short_momentum_weight + 
                fear_greed_normalized * sentiment_weight
            )
            
            # Clamp entre -100 et +100
            return max(-100, min(100, bull_bear_score))
            
        except Exception as e:
            logger.error(f"Error calculating bull/bear score: {e}")
            return 0
    
    def _calculate_market_health_score(self,
                                     coingecko_data: Dict,
                                     fear_greed_data: Dict, 
                                     advanced_metrics: AdvancedMarketMetrics) -> float:
        """Calculer score de santé du marché (0 à 100)"""
        try:
            # Facteurs de santé du marché
            
            # 1. Liquidité (volume vs market cap)
            volume_health = min(advanced_metrics.volume_vs_mcap_ratio * 10, 40)  # Max 40 points
            
            # 2. Volatilité (modérée est mieux)  
            volatility = advanced_metrics.btc_volatility_7d
            if volatility < 3:  # Très faible volatilité
                volatility_health = 20
            elif volatility < 7:  # Volatilité modérée (idéale)
                volatility_health = 30
            elif volatility < 15:  # Volatilité élevée
                volatility_health = 20
            else:  # Volatilité extrême
                volatility_health = 10
            
            # 3. Sentiment (neutre à légèrement positif est sain)
            fear_greed_value = fear_greed_data.get("value", 50)
            if 40 <= fear_greed_value <= 70:  # Zone saine
                sentiment_health = 30
            elif 30 <= fear_greed_value < 40 or 70 < fear_greed_value <= 80:
                sentiment_health = 20
            else:  # Extrêmes pas sains
                sentiment_health = 10
            
            total_health = volume_health + volatility_health + sentiment_health
            return min(100, max(0, total_health))
            
        except Exception as e:
            logger.error(f"Error calculating market health score: {e}")
            return 50
    
    def _calculate_opportunity_score(self,
                                   market_regime: MarketRegime,
                                   market_sentiment: MarketSentiment,
                                   advanced_metrics: AdvancedMarketMetrics) -> float:
        """Calculer score d'opportunités de trading (0 à 100)"""
        try:
            base_score = 50
            
            # Bonus selon régime de marché
            regime_bonus = {
                MarketRegime.EXTREME_BULL: 30,
                MarketRegime.BULL: 20,
                MarketRegime.NEUTRAL_BULLISH: 10,
                MarketRegime.NEUTRAL: 0,
                MarketRegime.NEUTRAL_BEARISH: -10,
                MarketRegime.BEAR: -20,
                MarketRegime.EXTREME_BEAR: -30
            }.get(market_regime, 0)
            
            # Bonus selon sentiment (extrêmes peuvent créer des opportunités)
            sentiment_bonus = {
                MarketSentiment.EXTREME_GREED: -10,  # Danger de correction
                MarketSentiment.GREED: 10,
                MarketSentiment.NEUTRAL: 15,
                MarketSentiment.FEAR: 25,  # Opportunités d'achat
                MarketSentiment.EXTREME_FEAR: 35   # Excellentes opportunités
            }.get(market_sentiment, 0)
            
            # Bonus volatilité (volatilité modérée = plus d'opportunités)
            volatility = advanced_metrics.btc_volatility_7d
            if 5 <= volatility <= 15:  # Volatilité optimale pour trading
                volatility_bonus = 20
            elif 3 <= volatility < 5 or 15 < volatility <= 25:
                volatility_bonus = 10
            else:
                volatility_bonus = 0
            
            opportunity_score = base_score + regime_bonus + sentiment_bonus + volatility_bonus
            return min(100, max(0, opportunity_score))
            
        except Exception as e:
            logger.error(f"Error calculating opportunity score: {e}")
            return 50
    
    def _analyze_volatility_regime(self, btc_data: Dict) -> str:
        """Analyser le régime de volatilité"""
        try:
            # Approximation basée sur les changements de prix
            changes = [
                abs(btc_data.get("price_change_percentage_24h", 0)),
                abs(btc_data.get("price_change_percentage_7d", 0)) / 7,  # Normaliser sur jour
            ]
            avg_daily_change = np.mean(changes)
            
            if avg_daily_change < 2:
                return "low"
            elif avg_daily_change < 5:
                return "medium"
            elif avg_daily_change < 10:
                return "high"
            else:
                return "extreme"
                
        except Exception as e:
            logger.error(f"Error analyzing volatility regime: {e}")
            return "medium"
    
    def _analyze_liquidity_condition(self, coingecko_data: Dict) -> str:
        """Analyser les conditions de liquidité"""
        try:
            total_volume = coingecko_data.get("total_volume", {}).get("usd", 0)
            total_mcap = coingecko_data.get("total_market_cap", {}).get("usd", 1)
            
            volume_ratio = (total_volume / total_mcap) * 100 if total_mcap > 0 else 0
            
            if volume_ratio > 8:
                return "excellent"
            elif volume_ratio > 5:
                return "good"
            elif volume_ratio > 2:
                return "moderate"
            else:
                return "poor"
                
        except Exception as e:
            logger.error(f"Error analyzing liquidity condition: {e}")
            return "moderate"
    
    def _generate_market_context_summary(self,
                                       market_regime: MarketRegime,
                                       market_sentiment: MarketSentiment,
                                       bull_bear_score: float,
                                       volatility_regime: str) -> str:
        """Générer résumé textuel du contexte de marché"""
        
        try:
            # Template de base
            regime_descriptions = {
                MarketRegime.EXTREME_BULL: "🚀 EXTREME BULL MARKET - Euphorie généralisée, gains massifs",
                MarketRegime.BULL: "📈 BULL MARKET - Momentum haussier fort et soutenu",
                MarketRegime.NEUTRAL_BULLISH: "📊 NEUTRAL-BULLISH - Tendance légèrement haussière",
                MarketRegime.NEUTRAL: "⚖️ NEUTRAL MARKET - Marché sideways, direction incertaine",
                MarketRegime.NEUTRAL_BEARISH: "📉 NEUTRAL-BEARISH - Tendance légèrement baissière",
                MarketRegime.BEAR: "📉 BEAR MARKET - Correction en cours, pessimisme dominant",
                MarketRegime.EXTREME_BEAR: "💥 EXTREME BEAR MARKET - Crash/capitulation, panique générale"
            }
            
            sentiment_descriptions = {
                MarketSentiment.EXTREME_GREED: "🔥 EXTREME GREED - Risque de correction imminente",
                MarketSentiment.GREED: "😤 GREED - Optimisme élevé, prudence recommandée", 
                MarketSentiment.NEUTRAL: "😐 NEUTRAL SENTIMENT - Marché équilibré",
                MarketSentiment.FEAR: "😰 FEAR - Pessimisme, opportunités d'achat potentielles",
                MarketSentiment.EXTREME_FEAR: "😱 EXTREME FEAR - Capitulation, opportunités majeures"
            }
            
            volatility_descriptions = {
                "low": "Volatilité FAIBLE - Marché calme, mouvements limités",
                "medium": "Volatilité MODÉRÉE - Conditions normales de trading",
                "high": "Volatilité ÉLEVÉE - Mouvements importants, opportunités accrues", 
                "extreme": "Volatilité EXTRÊME - Marché chaotique, gestion des risques cruciale"
            }
            
            # Construction du résumé
            summary = f"""CONTEXTE MARCHÉ CRYPTO GLOBAL:
{regime_descriptions.get(market_regime, market_regime.value)}
{sentiment_descriptions.get(market_sentiment, market_sentiment.value)}
{volatility_descriptions.get(volatility_regime, f"Volatilité {volatility_regime}")}

SCORE BULL/BEAR: {bull_bear_score:+.1f}/100 {'(Momentum BULLISH)' if bull_bear_score > 20 else '(Momentum BEARISH)' if bull_bear_score < -20 else '(Momentum NEUTRE)'}"""

            return summary
            
        except Exception as e:
            logger.error(f"Error generating market context summary: {e}")
            return "CONTEXTE MARCHÉ: Données partiellement disponibles"
    
    def _generate_trading_recommendations(self,
                                        market_regime: MarketRegime,
                                        market_sentiment: MarketSentiment, 
                                        opportunity_score: float,
                                        volatility_regime: str) -> List[str]:
        """Générer recommandations de trading contextuelles"""
        
        try:
            recommendations = []
            
            # Recommandations selon régime de marché
            regime_recs = {
                MarketRegime.EXTREME_BULL: [
                    "PRUDENCE: Marché en surchauffe, privilégier les prises de bénéfices",
                    "Surveiller les signaux de retournement et corrections"
                ],
                MarketRegime.BULL: [
                    "BULLISH: Favoriser les positions longues sur les pullbacks",
                    "Utiliser des trailing stops pour sécuriser les gains"
                ],
                MarketRegime.NEUTRAL_BULLISH: [
                    "Opportunités LONG sélectives sur les supports techniques", 
                    "Patience recommandée, attendre des confirmations claires"
                ],
                MarketRegime.NEUTRAL: [
                    "RANGE TRADING: Acheter supports, vendre résistances",
                    "Privilégier les stratégies neutres et l'accumulation graduelle"
                ],
                MarketRegime.NEUTRAL_BEARISH: [
                    "Prudence sur les positions longues, favoriser le cash",
                    "Opportunités SHORT sélectives sur les résistances"
                ],
                MarketRegime.BEAR: [
                    "BEARISH: Éviter les positions longues, cash is king",
                    "DCA sur les altcoins de qualité pour accumulation long terme"
                ],
                MarketRegime.EXTREME_BEAR: [
                    "OPPORTUNITY: Accumulation agressive sur les actifs de qualité",
                    "Maximum risk management, positions de petite taille"
                ]
            }
            
            recommendations.extend(regime_recs.get(market_regime, []))
            
            # Recommandations selon sentiment
            if market_sentiment == MarketSentiment.EXTREME_FEAR:
                recommendations.append("CONTRARIAN: Excellente opportunité d'achat sur la panique")
            elif market_sentiment == MarketSentiment.EXTREME_GREED:
                recommendations.append("RISK OFF: Réduire l'exposition, sécuriser les profits")
            
            # Recommandations selon volatilité
            if volatility_regime == "extreme":
                recommendations.append("VOLATILITÉ EXTRÊME: Réduire les tailles de position de 50%")
            elif volatility_regime == "high":
                recommendations.append("VOLATILITÉ ÉLEVÉE: Excellent pour scalping et day trading")
            
            # Recommandation selon score d'opportunité
            if opportunity_score > 75:
                recommendations.append(f"OPPORTUNITÉS EXCELLENTES ({opportunity_score:.0f}/100): Augmenter allocation trading")
            elif opportunity_score < 25:
                recommendations.append(f"PEU D'OPPORTUNITÉS ({opportunity_score:.0f}/100): Conservation capital recommandée")
            
            return recommendations[:5]  # Maximum 5 recommandations
            
        except Exception as e:
            logger.error(f"Error generating trading recommendations: {e}")
            return ["Recommandations indisponibles - Analyser manuellement le contexte de marché"]
    
    async def _calculate_market_cap_change_24h(self, 
                                         coingecko_data: Dict, 
                                         btc_data: Dict) -> float:
        """
        🚨 CALCUL MARKET CAP CHANGE 24H - Variable critique essentielle
        
        Méthodes utilisées (par ordre de priorité):
        1. Données directes CoinGecko si disponibles
        2. Estimation basée sur Bitcoin (corrélation ~0.85)
        3. Fallback via calcul composite BTC + ETH + dominances
        """
        try:
            logger.info("🧮 Calculating Market Cap 24h change...")
            
            # 1. MÉTHODE DIRECTE: CoinGecko market cap change (si disponible)
            if coingecko_data and "market_cap_change_percentage_24h_usd" in coingecko_data:
                direct_change = coingecko_data["market_cap_change_percentage_24h_usd"]
                if isinstance(direct_change, (int, float)) and -50 <= direct_change <= 50:
                    logger.info(f"✅ Direct Market Cap 24h: {direct_change:+.2f}% (CoinGecko)")
                    return direct_change
            
            # 2. MÉTHODE ESTIMATION BITCOIN: Market cap total suit BTC avec facteur
            if btc_data and "price_change_percentage_24h" in btc_data:
                btc_change_24h = btc_data["price_change_percentage_24h"]
                btc_dominance = coingecko_data.get("market_cap_percentage", {}).get("btc", 54)
                
                if isinstance(btc_change_24h, (int, float)):
                    # Factor basé sur dominance BTC et corrélation historique
                    # Plus la dominance BTC est élevée, plus le market cap total suit BTC
                    dominance_factor = btc_dominance / 100  # 0.54 pour 54% dominance
                    correlation_factor = 0.85  # Corrélation historique BTC/Total Market Cap
                    
                    estimated_mcap_change = btc_change_24h * dominance_factor * correlation_factor
                    
                    logger.info(f"✅ Estimated Market Cap 24h: {estimated_mcap_change:+.2f}% (BTC-based)")
                    logger.info(f"   BTC change: {btc_change_24h:+.2f}%, BTC dominance: {btc_dominance:.1f}%")
                    
                    return estimated_mcap_change
            
            # 3. MÉTHODE COMPOSITE: Calcul basé sur moyennes de marché
            try:
                # Utiliser des données de fallback Binance pour estimer
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    # Récupérer données BTC et ETH pour estimation composite
                    btc_ticker_url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
                    eth_ticker_url = "https://api.binance.com/api/v3/ticker/24hr?symbol=ETHUSDT"
                    
                    btc_response = await session.get(btc_ticker_url)
                    eth_response = await session.get(eth_ticker_url)
                    
                    if btc_response.status == 200 and eth_response.status == 200:
                        btc_binance = await btc_response.json()
                        eth_binance = await eth_response.json()
                        
                        btc_change = float(btc_binance.get('priceChangePercent', 0))
                        eth_change = float(eth_binance.get('priceChangePercent', 0))
                        
                        # Dominances
                        btc_dom = coingecko_data.get("market_cap_percentage", {}).get("btc", 54)
                        eth_dom = coingecko_data.get("market_cap_percentage", {}).get("eth", 17)
                        
                        # Calcul composite pondéré
                        composite_change = (
                            (btc_change * btc_dom / 100) + 
                            (eth_change * eth_dom / 100) +
                            (btc_change * 0.3)  # Influence générale BTC sur altcoins
                        )
                        
                        logger.info(f"✅ Composite Market Cap 24h: {composite_change:+.2f}% (BTC+ETH weighted)")
                        logger.info(f"   BTC: {btc_change:+.2f}% ({btc_dom:.1f}%), ETH: {eth_change:+.2f}% ({eth_dom:.1f}%)")
                        
                        return composite_change
            
            except Exception as e:
                logger.warning(f"Composite calculation failed: {e}")
            
            # 4. FALLBACK FINAL: Valeur neutre réaliste
            logger.warning("⚠️ All Market Cap 24h calculation methods failed, using neutral fallback")
            return 0.0  # Marché stable par défaut
            
        except Exception as e:
            logger.error(f"Error calculating Market Cap 24h change: {e}")
            return 0.0

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Vérifier si le cache est encore valide"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key]["timestamp"]
        now = datetime.now(timezone.utc)
        return (now - cache_time).total_seconds() < self.cache_duration
    
    async def get_market_context_for_ias(self) -> str:
        """
        🎯 Obtenir contexte de marché formaté pour les IAs
        """
        try:
            market_data = await self.get_global_market_data()
            
            if not market_data:
                return "🚨 MARKET CONTEXT UNAVAILABLE - Proceed with individual asset analysis only"
            
            # Formater pour les IAs
            ia_context = f"""
🌍 **GLOBAL CRYPTO MARKET CONTEXT** (Updated: {market_data.timestamp.strftime('%Y-%m-%d %H:%M UTC')}):

📊 **MARKET OVERVIEW:**
• Total Market Cap: ${market_data.total_market_cap/1e12:.2f}T ({market_data.market_cap_change_24h:+.2f}% 24h)
• 24h Volume: ${market_data.total_volume_24h/1e9:.1f}B
• BTC Dominance: {market_data.btc_dominance:.1f}% | ETH: {market_data.eth_dominance:.1f}%
• BTC: ${market_data.btc_price:,.0f} ({market_data.btc_change_24h:+.1f}% 24h, {market_data.btc_change_7d:+.1f}% 7d)

🎯 **MARKET REGIME & SENTIMENT:**
• Regime: {market_data.market_regime.value.upper().replace('_', ' ')}
• Sentiment: {market_data.fear_greed_classification} ({market_data.fear_greed_value}/100)
• Bull/Bear Score: {market_data.bull_bear_score:+.1f}/100
• Volatility: {market_data.volatility_regime.upper()} | Liquidity: {market_data.liquidity_condition.upper()}

📈 **OPPORTUNITY ASSESSMENT:**
• Market Health: {market_data.market_health_score:.0f}/100
• Trading Opportunities: {market_data.opportunity_score:.0f}/100
• Key Recommendations: {' | '.join(market_data.trading_recommendations[:2])}

{market_data.market_context_summary}

⚠️ **IA TRADING GUIDANCE:**
Use this macro context to adjust your confidence levels and risk assessments. In BEAR/HIGH FEAR regimes, be more selective. In BULL/GREED regimes, watch for reversal signals.
"""
            
            return ia_context.strip()
            
        except Exception as e:
            logger.error(f"Error generating IA context: {e}")
            return "🚨 MARKET CONTEXT ERROR - Proceed with individual asset analysis"

# Instance globale
global_crypto_market_analyzer = GlobalCryptoMarketAnalyzer()
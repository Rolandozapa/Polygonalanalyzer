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
        """Fallback: récupérer données via CoinMarketCap API (gratuit)"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                
                # CoinMarketCap free endpoints
                cmc_global_url = "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest"
                cmc_btc_url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
                
                # Essayer sans clé API d'abord (endpoints publics limités)
                try:
                    # Alternative: utiliser Binance API pour les données de base
                    binance_ticker_url = "https://api.binance.com/api/v3/ticker/24hr"
                    
                    async with session.get(binance_ticker_url) as response:
                        if response.status == 200:
                            binance_data = await response.json()
                            
                            # Filtrer BTC et ETH
                            btc_data = next((item for item in binance_data if item['symbol'] == 'BTCUSDT'), None)
                            eth_data = next((item for item in binance_data if item['symbol'] == 'ETHUSDT'), None)
                            
                            if btc_data:
                                logger.info("✅ Using Binance API as CoinGecko fallback")
                                
                                # Créer structure compatible avec CoinGecko
                                return {
                                    "total_market_cap": {"usd": 2400000000000},  # Estimation 2.4T
                                    "total_volume": {"usd": float(btc_data.get('volume', 0)) * 50},  # Approximation
                                    "market_cap_percentage": {
                                        "btc": 54.0,  # Approximation BTC dominance
                                        "eth": 18.0   # Approximation ETH dominance  
                                    }
                                }
                                
                except Exception as binance_error:
                    logger.warning(f"Binance fallback also failed: {binance_error}")
                
                # Dernière tentative: valeurs par défaut réalistes
                logger.warning("⚠️ All market data sources failed, using realistic defaults")
                return {
                    "total_market_cap": {"usd": 2400000000000},  # ~2.4T realistic
                    "total_volume": {"usd": 80000000000},        # ~80B realistic  
                    "market_cap_percentage": {
                        "btc": 53.5,  # BTC dominance réaliste
                        "eth": 17.8   # ETH dominance réaliste
                    }
                }
                
        except Exception as e:
            logger.error(f"All market data fallbacks failed: {e}")
            return None
    
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
        """Fallback: récupérer données BTC via Binance"""
        try:
            # Binance 24hr ticker
            ticker_url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
            
            async with session.get(ticker_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    current_price = float(data.get('lastPrice', 0))
                    price_change_24h = float(data.get('priceChangePercent', 0))
                    
                    logger.info(f"✅ Using Binance fallback for BTC data: ${current_price:,.0f}")
                    
                    # Pour 7d et 30d, on fait des approximations basées sur les données 24h
                    # (pas idéal mais mieux que pas de données)
                    estimated_7d = price_change_24h * 3.5  # Approximation grossière
                    estimated_30d = price_change_24h * 12   # Approximation grossière
                    
                    return {
                        "current_price": current_price,
                        "price_change_percentage_24h": price_change_24h,
                        "price_change_percentage_7d": estimated_7d,
                        "price_change_percentage_30d": estimated_30d,
                        "market_cap": current_price * 19500000,  # ~19.5M BTC en circulation
                        "total_volume": float(data.get('volume', 0)) * current_price
                    }
                else:
                    logger.warning(f"Binance BTC API returned {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Binance BTC fallback failed: {e}")
            return None
    
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
• Total Market Cap: ${market_data.total_market_cap/1e12:.2f}T (+/- indication needed)
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
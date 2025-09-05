import os
import aiohttp
import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
import ccxt
from dotenv import load_dotenv
import hashlib
from collections import defaultdict, deque

load_dotenv()

logger = logging.getLogger(__name__)

class APIStatus(Enum):
    ACTIVE = "active"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    TIMEOUT = "timeout"
    DISABLED = "disabled"

@dataclass
class APIEndpoint:
    name: str
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    rate_limit: int = 60  # requests per minute
    timeout: int = 30
    priority: int = 1  # 1 = highest priority
    status: APIStatus = APIStatus.ACTIVE
    last_request_time: float = 0
    request_count: int = 0
    error_count: int = 0
    last_error_time: float = 0
    success_rate: float = 1.0

@dataclass
class MarketDataResponse:
    symbol: str
    price: float
    volume_24h: float
    price_change_24h: float
    volatility: float = 0.02  # Default volatility estimate
    market_cap: Optional[float] = None
    market_cap_rank: Optional[int] = None
    source: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = 1.0  # Data quality confidence
    additional_data: Dict[str, Any] = field(default_factory=dict)

class AdvancedMarketAggregator:
    """
    Système de fallback robuste et parallélisé pour agrégation de données de marché
    """
    
    def __init__(self):
        # API Keys
        self.cmc_api_key = os.getenv('CMC_API_KEY')
        self.coinapi_key = os.getenv('COINAPI_KEY')
        self.twelvedata_key = os.getenv('TWELVEDATA_KEY')
        self.binance_key = os.getenv('BINANCE_KEY')
        
        # Rate limiting et caching
        self.request_history = defaultdict(deque)  # Track requests per API
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache
        
        # Performance monitoring
        self.api_performance = {}
        self.total_requests = 0
        self.successful_requests = 0
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=20)
        
        # Initialize API endpoints
        self.api_endpoints = self._initialize_api_endpoints()
        
        # Initialize CCXT exchanges
        self.exchanges = self._initialize_exchanges()
        
        logger.info("AdvancedMarketAggregator initialized with multi-threaded fallback system")
    
    def _initialize_api_endpoints(self) -> List[APIEndpoint]:
        """Initialize all API endpoints with configurations"""
        endpoints = []
        
        # CoinMarketCap - Highest priority (most reliable)
        if self.cmc_api_key:
            endpoints.extend([
                APIEndpoint(
                    name="cmc_listings",
                    url="https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest",
                    headers={"X-CMC_PRO_API_KEY": self.cmc_api_key, "Accept": "application/json"},
                    rate_limit=333,  # 10000 credits per month ≈ 333 per day
                    priority=1
                ),
                APIEndpoint(
                    name="cmc_quotes",
                    url="https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest",
                    headers={"X-CMC_PRO_API_KEY": self.cmc_api_key, "Accept": "application/json"},
                    rate_limit=500,
                    priority=1
                ),
                # DEX endpoints spéciaux - Comprehensive DEX Coverage
                APIEndpoint(
                    name="cmc_dex_listings",
                    url="https://pro-api.coinmarketcap.com/v4/dex/listings/quotes",
                    headers={"X-CMC_PRO_API_KEY": self.cmc_api_key, "Accept": "application/json"},
                    rate_limit=200,
                    priority=2
                ),
                APIEndpoint(
                    name="cmc_dex_info",
                    url="https://pro-api.coinmarketcap.com/v4/dex/listings/info",
                    headers={"X-CMC_PRO_API_KEY": self.cmc_api_key, "Accept": "application/json"},
                    rate_limit=150,
                    priority=2
                ),
                APIEndpoint(
                    name="cmc_dex_networks",
                    url="https://pro-api.coinmarketcap.com/v4/dex/networks/list",
                    headers={"X-CMC_PRO_API_KEY": self.cmc_api_key, "Accept": "application/json"},
                    rate_limit=100,
                    priority=3
                ),
                APIEndpoint(
                    name="cmc_dex_ohlcv",
                    url="https://pro-api.coinmarketcap.com/v4/dex/pairs/ohlcv/historical",
                    headers={"X-CMC_PRO_API_KEY": self.cmc_api_key, "Accept": "application/json"},
                    rate_limit=100,
                    priority=3
                ),
                APIEndpoint(
                    name="cmc_dex_trades",
                    url="https://pro-api.coinmarketcap.com/v4/dex/pairs/trade/latest",
                    headers={"X-CMC_PRO_API_KEY": self.cmc_api_key, "Accept": "application/json"},
                    rate_limit=100,
                    priority=3
                )
            ])
        
        # CoinAPI - High priority (professional data)
        if self.coinapi_key:
            endpoints.extend([
                APIEndpoint(
                    name="coinapi_exchanges",
                    url="https://rest.coinapi.io/v1/exchanges",
                    headers={"X-CoinAPI-Key": self.coinapi_key},
                    rate_limit=100,
                    priority=2
                ),
                APIEndpoint(
                    name="coinapi_symbols",
                    url="https://rest.coinapi.io/v1/symbols",
                    headers={"X-CoinAPI-Key": self.coinapi_key},
                    rate_limit=100,
                    priority=2
                ),
                APIEndpoint(
                    name="coinapi_quotes",
                    url="https://rest.coinapi.io/v1/quotes/current",
                    headers={"X-CoinAPI-Key": self.coinapi_key},
                    rate_limit=100,
                    priority=2
                )
            ])
        
        # CoinGecko - Medium priority (free, reliable)
        endpoints.extend([
            APIEndpoint(
                name="coingecko_markets",
                url="https://api.coingecko.com/api/v3/coins/markets",
                rate_limit=50,  # Conservative for free tier
                priority=3
            ),
            APIEndpoint(
                name="coingecko_simple_price",
                url="https://api.coingecko.com/api/v3/simple/price",
                rate_limit=50,
                priority=3
            ),
            APIEndpoint(
                name="coingecko_coins_list",
                url="https://api.coingecko.com/api/v3/coins/list",
                rate_limit=30,
                priority=4
            ),
            # CoinGecko trending (free)
            APIEndpoint(
                name="coingecko_trending",
                url="https://api.coingecko.com/api/v3/search/trending",
                rate_limit=30,
                priority=4
            ),
            # CoinGecko global data (free)
            APIEndpoint(
                name="coingecko_global",
                url="https://api.coingecko.com/api/v3/global",
                rate_limit=30,
                priority=5
            )
        ])
        
        # CoinCap - Free alternative API
        endpoints.extend([
            APIEndpoint(
                name="coincap_assets",
                url="https://api.coincap.io/v2/assets",
                rate_limit=100,
                priority=4
            ),
            APIEndpoint(
                name="coincap_markets",
                url="https://api.coincap.io/v2/markets",
                rate_limit=100, 
                priority=4
            )
        ])
        
        # CryptoCompare - Free tier
        endpoints.extend([
            APIEndpoint(
                name="cryptocompare_top",
                url="https://min-api.cryptocompare.com/data/top/mktcapfull",
                params={"limit": 100, "tsym": "USD"},
                rate_limit=100,
                priority=4
            ),
            APIEndpoint(
                name="cryptocompare_price",
                url="https://min-api.cryptocompare.com/data/pricemultifull",
                rate_limit=100,
                priority=4
            )
        ])
        
        # Twelve Data - Medium priority (financial data)
        if self.twelvedata_key:
            endpoints.extend([
                APIEndpoint(
                    name="twelvedata_crypto",
                    url="https://api.twelvedata.com/cryptocurrencies",
                    params={"apikey": self.twelvedata_key},
                    rate_limit=100,
                    priority=3
                ),
                APIEndpoint(
                    name="twelvedata_price",
                    url="https://api.twelvedata.com/price",
                    params={"apikey": self.twelvedata_key},
                    rate_limit=100,
                    priority=3
                )
            ])
        
        return sorted(endpoints, key=lambda x: x.priority)
    
    def _initialize_exchanges(self) -> Dict[str, Any]:
        """Initialize CCXT exchanges"""
        exchanges = {}
        
        try:
            # Binance
            if self.binance_key:
                exchanges['binance'] = ccxt.binance({
                    'apiKey': self.binance_key,
                    'sandbox': False,
                    'enableRateLimit': True,
                    'timeout': 30000
                })
            
            # Public exchanges (no API key needed)
            exchanges['bitfinex'] = ccxt.bitfinex({'enableRateLimit': True})
            exchanges['kraken'] = ccxt.kraken({'enableRateLimit': True})
            exchanges['coinbase'] = ccxt.coinbasepro({'enableRateLimit': True})
            
        except Exception as e:
            logger.warning(f"Error initializing exchanges: {e}")
        
        return exchanges
    
    async def get_comprehensive_market_data(self, 
                                          symbols: Optional[List[str]] = None, 
                                          limit: int = 500,
                                          include_dex: bool = True) -> List[MarketDataResponse]:
        """
        Récupère des données de marché complètes en utilisant tous les endpoints en parallèle
        """
        logger.info(f"Starting comprehensive market data aggregation for {limit} symbols")
        
        # Préparer les tâches parallèles
        tasks = []
        
        # 1. CoinMarketCap listings (priority 1)
        if self._can_make_request("cmc_listings"):
            tasks.append(self._fetch_cmc_listings(limit))
        
        # 2. CoinGecko markets (backup priority)
        if self._can_make_request("coingecko_markets"):
            tasks.append(self._fetch_coingecko_markets(limit))
        
        # 3. CoinAPI data (if available)
        if self._can_make_request("coinapi_exchanges"):
            tasks.append(self._fetch_coinapi_data())
        
        # 4. DEX data from CoinMarketCap
        if include_dex and self._can_make_request("cmc_dex_listings"):
            tasks.append(self._fetch_cmc_dex_data())
        
        # 5. Exchange data from CCXT
        tasks.append(self._fetch_exchange_data())
        
        # 6. Yahoo Finance for major cryptos
        tasks.append(self._fetch_yahoo_finance_crypto())
        
        # 7. CoinCap data (free alternative)
        tasks.append(self._fetch_coincap_data())
        
        # 8. CryptoCompare data (free tier)
        tasks.append(self._fetch_cryptocompare_data())
        
        # 9. CoinGecko trending (additional data)
        if self._can_make_request("coingecko_trending"):
            tasks.append(self._fetch_coingecko_trending())
        
        # Exécuter toutes les tâches en parallèle
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Agréger tous les résultats
            all_data = []
            for result in results:
                if isinstance(result, list):
                    all_data.extend(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Task failed: {result}")
            
            # Déduplication et fusion intelligente
            merged_data = self._merge_and_deduplicate(all_data)
            
            # Tri par qualité et market cap
            sorted_data = self._sort_by_quality_and_ranking(merged_data)
            
            logger.info(f"Aggregated {len(sorted_data)} unique market data points from {len([r for r in results if not isinstance(r, Exception)])} sources")
            
            return sorted_data[:limit]
            
        except Exception as e:
            logger.error(f"Error in comprehensive market data aggregation: {e}")
            return await self._fallback_data_fetch(limit)
    
    async def _fetch_cmc_listings(self, limit: int) -> List[MarketDataResponse]:
        """Fetch data from CoinMarketCap listings"""
        endpoint = next((ep for ep in self.api_endpoints if ep.name == "cmc_listings"), None)
        if not endpoint or not self._can_make_request("cmc_listings"):
            return []
        
        try:
            params = {
                "start": 1,
                "limit": min(limit, 5000),
                "convert": "USD",
                "sort": "market_cap",
                "sort_dir": "desc"
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=endpoint.timeout)) as session:
                start_time = time.time()
                async with session.get(endpoint.url, headers=endpoint.headers, params=params) as response:
                    self._update_request_stats("cmc_listings", time.time() - start_time, response.status == 200)
                    
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_cmc_listings(data)
                    else:
                        logger.warning(f"CMC listings API returned {response.status}")
                        return []
                        
        except Exception as e:
            self._update_request_stats("cmc_listings", 30, False)
            logger.error(f"Error fetching CMC listings: {e}")
            return []
    
    async def _fetch_coingecko_markets(self, limit: int) -> List[MarketDataResponse]:
        """Fetch data from CoinGecko markets"""
        try:
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": min(limit, 250),
                "page": 1,
                "sparkline": "false",
                "price_change_percentage": "24h"
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                start_time = time.time()
                async with session.get("https://api.coingecko.com/api/v3/coins/markets", params=params) as response:
                    self._update_request_stats("coingecko_markets", time.time() - start_time, response.status == 200)
                    
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_coingecko_markets(data)
                    else:
                        return []
                        
        except Exception as e:
            self._update_request_stats("coingecko_markets", 30, False)
            logger.error(f"Error fetching CoinGecko markets: {e}")
            return []
    
    async def _fetch_coinapi_data(self) -> List[MarketDataResponse]:
        """Fetch data from CoinAPI"""
        if not self.coinapi_key:
            return []
        
        try:
            headers = {"X-CoinAPI-Key": self.coinapi_key}
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                # Get popular trading pairs
                start_time = time.time()
                async with session.get("https://rest.coinapi.io/v1/quotes/current", 
                                     headers=headers,
                                     params={"filter_asset_id": "BTC,ETH,BNB,XRP,SOL,ADA,DOGE,TRX,AVAX,MATIC"}) as response:
                    self._update_request_stats("coinapi_quotes", time.time() - start_time, response.status == 200)
                    
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_coinapi_quotes(data)
                    else:
                        return []
                        
        except Exception as e:
            self._update_request_stats("coinapi_quotes", 30, False)
            logger.error(f"Error fetching CoinAPI data: {e}")
            return []
    
    async def _fetch_cmc_dex_data(self) -> List[MarketDataResponse]:
        """Fetch DEX data from CoinMarketCap"""
        if not self.cmc_api_key:
            return []
        
        try:
            headers = {"X-CMC_PRO_API_KEY": self.cmc_api_key, "Accept": "application/json"}
            params = {"limit": 100, "sort": "volume_24h", "sort_dir": "desc"}
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                start_time = time.time()
                async with session.get("https://pro-api.coinmarketcap.com/v4/dex/listings/quotes", 
                                     headers=headers, params=params) as response:
                    self._update_request_stats("cmc_dex_listings", time.time() - start_time, response.status == 200)
                    
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_cmc_dex_data(data)
                    else:
                        return []
                        
        except Exception as e:
            self._update_request_stats("cmc_dex_listings", 30, False)
            logger.error(f"Error fetching CMC DEX data: {e}")
            return []
    
    async def _fetch_exchange_data(self) -> List[MarketDataResponse]:
        """Fetch data from CCXT exchanges in parallel"""
        if not self.exchanges:
            return []
        
        tasks = []
        for exchange_name, exchange in self.exchanges.items():
            if exchange_name == 'binance' and not self.binance_key:
                continue
            tasks.append(self._fetch_single_exchange_data(exchange_name, exchange))
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            all_data = []
            for result in results:
                if isinstance(result, list):
                    all_data.extend(result)
            return all_data
        except Exception as e:
            logger.error(f"Error fetching exchange data: {e}")
            return []
    
    async def _fetch_single_exchange_data(self, exchange_name: str, exchange) -> List[MarketDataResponse]:
        """Fetch data from a single exchange"""
        try:
            loop = asyncio.get_event_loop()
            
            # Run in thread pool to avoid blocking
            tickers = await loop.run_in_executor(
                self.thread_pool, 
                lambda: exchange.fetch_tickers() if hasattr(exchange, 'fetch_tickers') else {}
            )
            
            data = []
            for symbol, ticker in tickers.items():
                if '/USDT' in symbol or '/USD' in symbol:
                    try:
                        base_symbol = symbol.split('/')[0]
                        data.append(MarketDataResponse(
                            symbol=f"{base_symbol}USDT",
                            price=ticker.get('close', 0) or 0,
                            volume_24h=ticker.get('quoteVolume', 0) or 0,
                            price_change_24h=ticker.get('percentage', 0) or 0,
                            volatility=abs(ticker.get('percentage', 0) or 0) / 100.0,  # Estimate volatility from 24h change
                            source=f"{exchange_name}_ccxt",
                            confidence=0.8  # CCXT data is generally reliable
                        ))
                    except Exception as e:
                        continue
            
            logger.info(f"Fetched {len(data)} tickers from {exchange_name}")
            return data[:50]  # Limit per exchange
            
        except Exception as e:
            logger.warning(f"Error fetching data from {exchange_name}: {e}")
            return []
    
    async def _fetch_yahoo_finance_crypto(self) -> List[MarketDataResponse]:
        """Fetch major crypto data from Yahoo Finance"""
        try:
            loop = asyncio.get_event_loop()
            
            # Major crypto symbols on Yahoo Finance
            symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'SOL-USD', 
                      'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'DOT-USD', 'MATIC-USD']
            
            async def fetch_yf_data(symbol):
                try:
                    ticker = await loop.run_in_executor(self.thread_pool, yf.Ticker, symbol)
                    info = await loop.run_in_executor(self.thread_pool, lambda: ticker.info)
                    hist = await loop.run_in_executor(self.thread_pool, lambda: ticker.history(period='2d'))
                    
                    if not hist.empty and info:
                        current_price = hist['Close'].iloc[-1]
                        prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                        price_change = ((current_price - prev_price) / prev_price) * 100
                        
                        return MarketDataResponse(
                            symbol=symbol.replace('-USD', 'USDT'),
                            price=current_price,
                            volume_24h=hist['Volume'].iloc[-1] * current_price,
                            price_change_24h=price_change,
                            volatility=abs(price_change) / 100.0,  # Estimate volatility from 24h change
                            market_cap=info.get('marketCap'),
                            source="yahoo_finance",
                            confidence=0.9  # Yahoo Finance is very reliable
                        )
                except Exception as e:
                    logger.debug(f"Error fetching Yahoo Finance data for {symbol}: {e}")
                    return None
            
            tasks = [fetch_yf_data(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return [r for r in results if isinstance(r, MarketDataResponse)]
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance crypto data: {e}")
            return []
    
    def _parse_cmc_listings(self, data: Dict) -> List[MarketDataResponse]:
        """Parse CoinMarketCap listings response"""
        parsed_data = []
        
        for crypto in data.get('data', []):
            try:
                quote = crypto.get('quote', {}).get('USD', {})
                parsed_data.append(MarketDataResponse(
                    symbol=f"{crypto.get('symbol')}USDT",
                    price=quote.get('price', 0),
                    volume_24h=quote.get('volume_24h', 0),
                    price_change_24h=quote.get('percent_change_24h', 0),
                    volatility=abs(quote.get('percent_change_24h', 0)) / 100.0,  # Estimate volatility from 24h change
                    market_cap=quote.get('market_cap', 0),
                    market_cap_rank=crypto.get('cmc_rank'),
                    source="coinmarketcap",
                    confidence=0.95,  # CMC is highly reliable
                    additional_data={
                        'name': crypto.get('name'),
                        'circulating_supply': crypto.get('circulating_supply'),
                        'total_supply': crypto.get('total_supply')
                    }
                ))
            except Exception as e:
                logger.debug(f"Error parsing CMC crypto data: {e}")
                continue
        
        return parsed_data
    
    def _parse_coingecko_markets(self, data: List) -> List[MarketDataResponse]:
        """Parse CoinGecko markets response"""
        parsed_data = []
        
        for crypto in data:
            try:
                parsed_data.append(MarketDataResponse(
                    symbol=f"{crypto.get('symbol', '').upper()}USDT",
                    price=crypto.get('current_price', 0),
                    volume_24h=crypto.get('total_volume', 0),
                    price_change_24h=crypto.get('price_change_percentage_24h', 0),
                    volatility=abs(crypto.get('price_change_percentage_24h', 0)) / 100.0,  # Estimate volatility from 24h change
                    market_cap=crypto.get('market_cap', 0),
                    market_cap_rank=crypto.get('market_cap_rank'),
                    source="coingecko",
                    confidence=0.9,  # CoinGecko is reliable
                    additional_data={
                        'name': crypto.get('name'),
                        'high_24h': crypto.get('high_24h'),
                        'low_24h': crypto.get('low_24h')
                    }
                ))
            except Exception as e:
                logger.debug(f"Error parsing CoinGecko crypto data: {e}")
                continue
        
        return parsed_data
    
    def _parse_coinapi_quotes(self, data: List) -> List[MarketDataResponse]:
        """Parse CoinAPI quotes response"""
        parsed_data = []
        
        for quote in data:
            try:
                if quote.get('asset_id_quote') == 'USD':
                    parsed_data.append(MarketDataResponse(
                        symbol=f"{quote.get('asset_id_base')}USDT",
                        price=quote.get('rate', 0),
                        volume_24h=0,  # Not available in quotes endpoint
                        price_change_24h=0,  # Would need historical data
                        volatility=0.02,  # Default volatility since no historical data
                        source="coinapi",
                        confidence=0.85
                    ))
            except Exception as e:
                logger.debug(f"Error parsing CoinAPI quote data: {e}")
                continue
        
        return parsed_data
    
    def _parse_cmc_dex_data(self, data: Dict) -> List[MarketDataResponse]:
        """Parse CoinMarketCap DEX data response"""
        parsed_data = []
        
        for pair in data.get('data', []):
            try:
                quote = pair.get('quote', {}).get('USD', {})
                base_symbol = pair.get('base_currency_symbol', '')
                
                if base_symbol:
                    parsed_data.append(MarketDataResponse(
                        symbol=f"{base_symbol}USDT",
                        price=quote.get('price', 0),
                        volume_24h=quote.get('volume_24h', 0),
                        price_change_24h=quote.get('percent_change_24h', 0),
                        volatility=abs(quote.get('percent_change_24h', 0)) / 100.0,  # Estimate volatility from 24h change
                        source="coinmarketcap_dex",
                        confidence=0.8,  # DEX data slightly less reliable
                        additional_data={
                            'dex_name': pair.get('dex_name'),
                            'pair_address': pair.get('pair_address')
                        }
                    ))
            except Exception as e:
                logger.debug(f"Error parsing CMC DEX data: {e}")
                continue
        
        return parsed_data
    
    def _merge_and_deduplicate(self, data: List[MarketDataResponse]) -> List[MarketDataResponse]:
        """Merge data from multiple sources and deduplicate intelligently"""
        symbol_groups = defaultdict(list)
        
        # Group by symbol
        for item in data:
            if item.price > 0:  # Only valid price data
                symbol_groups[item.symbol].append(item)
        
        merged_data = []
        
        for symbol, items in symbol_groups.items():
            if len(items) == 1:
                merged_data.append(items[0])
            else:
                # Merge multiple sources for the same symbol
                merged_item = self._merge_symbol_data(items)
                merged_data.append(merged_item)
        
        return merged_data
    
    def _merge_symbol_data(self, items: List[MarketDataResponse]) -> MarketDataResponse:
        """Merge data from multiple sources for the same symbol"""
        # Sort by confidence and recency
        items.sort(key=lambda x: (x.confidence, x.timestamp), reverse=True)
        
        # Use highest confidence source as base
        base_item = items[0]
        
        # Average price from high-confidence sources
        high_conf_items = [item for item in items if item.confidence >= 0.8]
        if len(high_conf_items) > 1:
            avg_price = np.mean([item.price for item in high_conf_items])
            base_item.price = avg_price
        
        # Use highest volume and market cap available
        base_item.volume_24h = max(item.volume_24h for item in items)
        base_item.market_cap = max((item.market_cap for item in items if item.market_cap), default=base_item.market_cap)
        
        # Combine sources
        base_item.source = f"merged_{len(items)}_sources"
        base_item.confidence = min(base_item.confidence + 0.05, 1.0)  # Slight confidence boost for merged data
        
        return base_item
    
    def _sort_by_quality_and_ranking(self, data: List[MarketDataResponse]) -> List[MarketDataResponse]:
        """Sort data by quality and market cap ranking"""
        def sort_key(item):
            # Primary: market cap rank (lower is better, 0 if None)
            rank = item.market_cap_rank or 999999
            # Secondary: confidence (higher is better)
            confidence = item.confidence
            # Tertiary: volume (higher is better)
            volume = item.volume_24h or 0
            
            return (rank, -confidence, -volume)
        
        return sorted(data, key=sort_key)
    
    async def _fallback_data_fetch(self, limit: int) -> List[MarketDataResponse]:
        """Fallback method when all primary sources fail"""
        logger.warning("Using fallback data fetch method")
        
        # Try simple CoinGecko request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.coingecko.com/api/v3/coins/markets",
                                     params={"vs_currency": "usd", "per_page": limit, "page": 1}) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_coingecko_markets(data)
        except Exception as e:
            logger.error(f"Fallback CoinGecko request failed: {e}")
        
        # Last resort: return empty list
        return []
    
    def _can_make_request(self, api_name: str) -> bool:
        """Check if we can make a request to the API (rate limiting)"""
        now = time.time()
        endpoint = next((ep for ep in self.api_endpoints if ep.name == api_name), None)
        
        if not endpoint or endpoint.status != APIStatus.ACTIVE:
            return False
        
        # Check rate limiting
        requests_in_last_minute = len([t for t in self.request_history[api_name] 
                                     if now - t < 60])
        
        if requests_in_last_minute >= endpoint.rate_limit:
            return False
        
        # Check if recently failed
        if (endpoint.last_error_time > 0 and 
            now - endpoint.last_error_time < 300 and  # 5 minutes
            endpoint.error_count > 3):
            return False
        
        return True
    
    def _update_request_stats(self, api_name: str, response_time: float, success: bool):
        """Update request statistics for an API"""
        now = time.time()
        self.request_history[api_name].append(now)
        
        # Keep only last hour of requests
        self.request_history[api_name] = deque([
            t for t in self.request_history[api_name] if now - t < 3600
        ])
        
        endpoint = next((ep for ep in self.api_endpoints if ep.name == api_name), None)
        if endpoint:
            endpoint.last_request_time = now
            endpoint.request_count += 1
            
            if success:
                endpoint.error_count = max(0, endpoint.error_count - 1)
                endpoint.success_rate = (endpoint.success_rate * 0.9) + (1.0 * 0.1)
            else:
                endpoint.error_count += 1
                endpoint.last_error_time = now
                endpoint.success_rate = endpoint.success_rate * 0.9
        
        self.total_requests += 1
        if success:
            self.successful_requests += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all APIs"""
        stats = {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": self.successful_requests / self.total_requests if self.total_requests > 0 else 0,
            "api_endpoints": []
        }
        
        for endpoint in self.api_endpoints:
            stats["api_endpoints"].append({
                "name": endpoint.name,
                "status": endpoint.status.value,
                "requests": endpoint.request_count,
                "success_rate": endpoint.success_rate,
                "error_count": endpoint.error_count,
                "last_request": endpoint.last_request_time,
                "priority": endpoint.priority
            })
        
        return stats

    async def _fetch_coincap_data(self) -> List[MarketDataResponse]:
        """Fetch data from CoinCap API (free alternative)"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                start_time = time.time()
                async with session.get("https://api.coincap.io/v2/assets", 
                                     params={"limit": 100}) as response:
                    self._update_request_stats("coincap_assets", time.time() - start_time, response.status == 200)
                    
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_coincap_data(data)
                    else:
                        return []
                        
        except Exception as e:
            self._update_request_stats("coincap_assets", 30, False)
            logger.error(f"Error fetching CoinCap data: {e}")
            return []

    async def _fetch_cryptocompare_data(self) -> List[MarketDataResponse]:
        """Fetch data from CryptoCompare API (free tier)"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                start_time = time.time()
                async with session.get("https://min-api.cryptocompare.com/data/top/mktcapfull", 
                                     params={"limit": 50, "tsym": "USD"}) as response:
                    self._update_request_stats("cryptocompare_top", time.time() - start_time, response.status == 200)
                    
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_cryptocompare_data(data)
                    else:
                        return []
                        
        except Exception as e:
            self._update_request_stats("cryptocompare_top", 30, False)
            logger.error(f"Error fetching CryptoCompare data: {e}")
            return []

    async def _fetch_coingecko_trending(self) -> List[MarketDataResponse]:
        """Fetch trending data from CoinGecko"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                start_time = time.time()
                async with session.get("https://api.coingecko.com/api/v3/search/trending") as response:
                    self._update_request_stats("coingecko_trending", time.time() - start_time, response.status == 200)
                    
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_coingecko_trending_data(data)
                    else:
                        return []
                        
        except Exception as e:
            self._update_request_stats("coingecko_trending", 30, False)
            logger.error(f"Error fetching CoinGecko trending data: {e}")
            return []

    def _parse_coincap_data(self, data: Dict) -> List[MarketDataResponse]:
        """Parse CoinCap response"""
        parsed_data = []
        
        for asset in data.get('data', []):
            try:
                price_change = float(asset.get('changePercent24Hr', 0))
                parsed_data.append(MarketDataResponse(
                    symbol=f"{asset.get('symbol')}USDT",
                    price=float(asset.get('priceUsd', 0)),
                    volume_24h=float(asset.get('volumeUsd24Hr', 0)),
                    price_change_24h=price_change,
                    volatility=abs(price_change) / 100.0,
                    market_cap=float(asset.get('marketCapUsd', 0)),
                    market_cap_rank=int(asset.get('rank', 999)),
                    source="coincap",
                    confidence=0.85,
                    additional_data={
                        'name': asset.get('name'),
                        'supply': asset.get('supply')
                    }
                ))
            except Exception as e:
                logger.debug(f"Error parsing CoinCap asset data: {e}")
                continue
        
        return parsed_data

    def _parse_cryptocompare_data(self, data: Dict) -> List[MarketDataResponse]:
        """Parse CryptoCompare response"""
        parsed_data = []
        
        for crypto in data.get('Data', []):
            try:
                coin_info = crypto.get('CoinInfo', {})
                raw_data = crypto.get('RAW', {}).get('USD', {})
                
                if raw_data:
                    parsed_data.append(MarketDataResponse(
                        symbol=f"{coin_info.get('Name')}USDT",
                        price=float(raw_data.get('PRICE', 0)),
                        volume_24h=float(raw_data.get('VOLUME24HOURTO', 0)),
                        price_change_24h=float(raw_data.get('CHANGEPCT24HOUR', 0)),
                        volatility=abs(float(raw_data.get('CHANGEPCT24HOUR', 0))) / 100.0,
                        market_cap=float(raw_data.get('MKTCAP', 0)),
                        source="cryptocompare",
                        confidence=0.85,
                        additional_data={
                            'name': coin_info.get('FullName'),
                            'algorithm': coin_info.get('Algorithm')
                        }
                    ))
            except Exception as e:
                logger.debug(f"Error parsing CryptoCompare crypto data: {e}")
                continue
        
        return parsed_data

    def _parse_coingecko_trending_data(self, data: Dict) -> List[MarketDataResponse]:
        """Parse CoinGecko trending response"""
        parsed_data = []
        
        for coin in data.get('coins', []):
            try:
                coin_data = coin.get('item', {})
                # Note: trending endpoint doesn't provide price data, so we'll use placeholder values
                parsed_data.append(MarketDataResponse(
                    symbol=f"{coin_data.get('symbol', '').upper()}USDT",
                    price=0.0,  # Not available in trending endpoint
                    volume_24h=0.0,  # Not available in trending endpoint
                    price_change_24h=0.0,  # Not available in trending endpoint
                    volatility=0.05,  # Default trending volatility
                    market_cap_rank=coin_data.get('market_cap_rank'),
                    source="coingecko_trending",
                    confidence=0.7,  # Lower confidence as no price data
                    additional_data={
                        'name': coin_data.get('name'),
                        'score': coin_data.get('score'),
                        'trending_rank': coin_data.get('score')
                    }
                ))
            except Exception as e:
                logger.debug(f"Error parsing CoinGecko trending data: {e}")
                continue
        
        return parsed_data

# Global instance
advanced_market_aggregator = AdvancedMarketAggregator()
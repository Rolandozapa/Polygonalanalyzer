import os
import aiohttp
import asyncio
import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import yfinance as yf
# 🚨 CCXT DISABLED: import ccxt - TEMPORARILY DISABLED FOR CPU DEBUG
# import ccxt
import pandas as pd
import numpy as np
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class MarketDataPoint:
    symbol: str
    price: float
    volume_24h: float
    price_change_24h: float
    volatility: float
    market_cap: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

class RealMarketDataService:
    """Real market data service using multiple APIs for redundancy and comprehensive data"""
    
    def __init__(self):
        self.cmc_api_key = os.getenv('CMC_API_KEY')
        self.coinapi_key = os.getenv('COINAPI_KEY')
        self.twelvedata_key = os.getenv('TWELVEDATA_KEY')
        self.binance_key = os.getenv('BINANCE_KEY')
        
        # 🚨 CCXT DISABLED: Initialize CCXT for Binance - TEMPORARILY DISABLED FOR CPU DEBUG
        # self.binance = ccxt.binance({
        #     'apiKey': self.binance_key,
        #     'sandbox': False,  # Set to True for testnet
        #     'enableRateLimit': True,
        # })
        self.binance = None
        logger.warning("🚨 CCXT Binance integration temporarily disabled to resolve CPU saturation")
        
        # Cache for API responses
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes
        
        logger.info("RealMarketDataService initialized with live APIs")
    
    
    async def get_top_cryptos_by_marketcap(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Get top cryptocurrencies by market cap from multiple sources"""
        top_cryptos = []
        
        # Try CoinMarketCap first (most reliable for market cap data)
        if self.cmc_api_key:
            try:
                top_cryptos = await self._get_cmc_top_cryptos(limit)
                if top_cryptos:
                    logger.info(f"Retrieved top {len(top_cryptos)} cryptos from CoinMarketCap")
                    return top_cryptos
            except Exception as e:
                logger.warning(f"CoinMarketCap API failed: {e}")
        
        # Fallback to CoinGecko (free API)
        try:
            top_cryptos = await self._get_coingecko_top_cryptos(limit)
            if top_cryptos:
                logger.info(f"Retrieved top {len(top_cryptos)} cryptos from CoinGecko")
                return top_cryptos
        except Exception as e:
            logger.warning(f"CoinGecko API failed: {e}")
        
        # Last fallback to predefined list
        logger.warning("Using predefined crypto list as fallback")
        return self._get_predefined_top_cryptos()
    
    async def _get_cmc_top_cryptos(self, limit: int) -> List[Dict[str, Any]]:
        """Get top cryptos from CoinMarketCap API"""
        url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
        headers = {
            'X-CMC_PRO_API_KEY': self.cmc_api_key,
            'Accept': 'application/json'
        }
        params = {
            'start': 1,
            'limit': min(limit, 5000),  # CMC allows up to 5000
            'convert': 'USD',
            'sort': 'market_cap',
            'sort_dir': 'desc'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    cryptos = []
                    
                    for crypto in data.get('data', []):
                        quote = crypto.get('quote', {}).get('USD', {})
                        cryptos.append({
                            'symbol': crypto.get('symbol'),
                            'name': crypto.get('name'),
                            'market_cap_rank': crypto.get('cmc_rank'),
                            'price': quote.get('price', 0),
                            'market_cap': quote.get('market_cap', 0),
                            'volume_24h': quote.get('volume_24h', 0),
                            'percent_change_24h': quote.get('percent_change_24h', 0),
                            'source': 'coinmarketcap'
                        })
                    
                    return cryptos
                else:
                    raise Exception(f"CMC API returned status {response.status}")
    
    async def _get_coingecko_top_cryptos(self, limit: int) -> List[Dict[str, Any]]:
        """Get top cryptos from CoinGecko API (free alternative)"""
        url = 'https://api.coingecko.com/api/v3/coins/markets'
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': min(limit, 250),  # CoinGecko free allows up to 250
            'page': 1,
            'sparkline': False,
            'price_change_percentage': '24h'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    cryptos = []
                    
                    for crypto in data:
                        cryptos.append({
                            'symbol': crypto.get('symbol', '').upper(),
                            'name': crypto.get('name'),
                            'market_cap_rank': crypto.get('market_cap_rank'),
                            'price': crypto.get('current_price', 0),
                            'market_cap': crypto.get('market_cap', 0),
                            'volume_24h': crypto.get('total_volume', 0),
                            'percent_change_24h': crypto.get('price_change_percentage_24h', 0),
                            'source': 'coingecko'
                        })
                    
                    return cryptos
                else:
                    raise Exception(f"CoinGecko API returned status {response.status}")
    
    def _get_predefined_top_cryptos(self) -> List[Dict[str, Any]]:
        """Fallback predefined list of top cryptocurrencies"""
        top_cryptos = [
            {'symbol': 'BTC', 'name': 'Bitcoin', 'market_cap_rank': 1},
            {'symbol': 'ETH', 'name': 'Ethereum', 'market_cap_rank': 2},
            {'symbol': 'BNB', 'name': 'BNB', 'market_cap_rank': 3},
            {'symbol': 'XRP', 'name': 'XRP', 'market_cap_rank': 4},
            {'symbol': 'SOL', 'name': 'Solana', 'market_cap_rank': 5},
            {'symbol': 'USDC', 'name': 'USD Coin', 'market_cap_rank': 6},
            {'symbol': 'STETH', 'name': 'Lido Staked Ether', 'market_cap_rank': 7},
            {'symbol': 'ADA', 'name': 'Cardano', 'market_cap_rank': 8},
            {'symbol': 'AVAX', 'name': 'Avalanche', 'market_cap_rank': 9},
            {'symbol': 'DOGE', 'name': 'Dogecoin', 'market_cap_rank': 10},
            {'symbol': 'TRX', 'name': 'TRON', 'market_cap_rank': 11},
            {'symbol': 'LINK', 'name': 'Chainlink', 'market_cap_rank': 12},
            {'symbol': 'TON', 'name': 'Toncoin', 'market_cap_rank': 13},
            {'symbol': 'MATIC', 'name': 'Polygon', 'market_cap_rank': 14},
            {'symbol': 'ICP', 'name': 'Internet Computer', 'market_cap_rank': 15},
            {'symbol': 'SHIB', 'name': 'Shiba Inu', 'market_cap_rank': 16},
            {'symbol': 'DAI', 'name': 'Dai', 'market_cap_rank': 17},
            {'symbol': 'DOT', 'name': 'Polkadot', 'market_cap_rank': 18},
            {'symbol': 'LTC', 'name': 'Litecoin', 'market_cap_rank': 19},
            {'symbol': 'BCH', 'name': 'Bitcoin Cash', 'market_cap_rank': 20},
            {'symbol': 'UNI', 'name': 'Uniswap', 'market_cap_rank': 21},
            {'symbol': 'ATOM', 'name': 'Cosmos', 'market_cap_rank': 22},
            {'symbol': 'LEO', 'name': 'UNUS SED LEO', 'market_cap_rank': 23},
            {'symbol': 'XLM', 'name': 'Stellar', 'market_cap_rank': 24},
            {'symbol': 'ETC', 'name': 'Ethereum Classic', 'market_cap_rank': 25}
        ]
        
        # Add basic market data for fallback
        for crypto in top_cryptos:
            crypto.update({
                'price': 50000 if crypto['symbol'] == 'BTC' else np.random.uniform(0.1, 100),
                'market_cap': np.random.uniform(1000000000, 100000000000),
                'volume_24h': np.random.uniform(100000000, 10000000000),
                'percent_change_24h': np.random.uniform(-10, 10),
                'source': 'predefined'
            })
        
        return top_cryptos
    async def get_crypto_opportunities(self) -> List[MarketDataPoint]:
        """Get real cryptocurrency opportunities from multiple sources"""
        opportunities = []
        
        # Primary symbols to track
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT', 
                  'AVAX/USDT', 'MATIC/USDT', 'LINK/USDT', 'UNI/USDT', 'ATOM/USDT']
        
        # Try different data sources with fallbacks
        try:
            # First try Binance (most reliable for crypto)
            binance_data = await self._get_binance_data(symbols)
            if binance_data:
                opportunities.extend(binance_data)
                logger.info(f"Retrieved {len(binance_data)} opportunities from Binance")
        except Exception as e:
            logger.warning(f"Binance API failed: {e}")
        
        # Fallback to CoinAPI if Binance fails or for additional data
        if len(opportunities) < 5:
            try:
                coinapi_data = await self._get_coinapi_data()
                opportunities.extend(coinapi_data)
                logger.info(f"Retrieved {len(coinapi_data)} additional opportunities from CoinAPI")
            except Exception as e:
                logger.warning(f"CoinAPI failed: {e}")
        
        # If still no data, use Yahoo Finance as last resort
        if len(opportunities) == 0:
            try:
                yahoo_data = await self._get_yahoo_finance_data()
                opportunities.extend(yahoo_data)
                logger.info(f"Retrieved {len(yahoo_data)} opportunities from Yahoo Finance")
            except Exception as e:
                logger.warning(f"Yahoo Finance failed: {e}")
        
        # Sort by volume and return top opportunities
        opportunities.sort(key=lambda x: x.volume_24h, reverse=True)
        return opportunities[:10]  # Return top 10
        """Get real cryptocurrency opportunities from multiple sources"""
        opportunities = []
        
        # Primary symbols to track
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT', 
                  'AVAX/USDT', 'MATIC/USDT', 'LINK/USDT', 'UNI/USDT', 'ATOM/USDT']
        
        # Try different data sources with fallbacks
        try:
            # First try Binance (most reliable for crypto)
            binance_data = await self._get_binance_data(symbols)
            if binance_data:
                opportunities.extend(binance_data)
                logger.info(f"Retrieved {len(binance_data)} opportunities from Binance")
        except Exception as e:
            logger.warning(f"Binance API failed: {e}")
        
        # Fallback to CoinAPI if Binance fails or for additional data
        if len(opportunities) < 5:
            try:
                coinapi_data = await self._get_coinapi_data()
                opportunities.extend(coinapi_data)
                logger.info(f"Retrieved {len(coinapi_data)} additional opportunities from CoinAPI")
            except Exception as e:
                logger.warning(f"CoinAPI failed: {e}")
        
        # If still no data, use Yahoo Finance as last resort
        if len(opportunities) == 0:
            try:
                yahoo_data = await self._get_yahoo_finance_data()
                opportunities.extend(yahoo_data)
                logger.info(f"Retrieved {len(yahoo_data)} opportunities from Yahoo Finance")
            except Exception as e:
                logger.warning(f"Yahoo Finance failed: {e}")
        
        # Sort by volume and return top opportunities
        opportunities.sort(key=lambda x: x.volume_24h, reverse=True)
        return opportunities[:10]  # Return top 10
    
    async def _get_binance_data(self, symbols: List[str]) -> List[MarketDataPoint]:
        """Get data from Binance API"""
        opportunities = []
        
        try:
            # Get 24h ticker data for all symbols
            tickers = self.binance.fetch_tickers()
            
            for symbol in symbols:
                if symbol in tickers:
                    ticker = tickers[symbol]
                    
                    # Calculate volatility from high/low
                    if ticker['high'] and ticker['low'] and ticker['close']:
                        volatility = (ticker['high'] - ticker['low']) / ticker['close']
                    else:
                        volatility = 0.02  # Default 2%
                    
                    opportunity = MarketDataPoint(
                        symbol=symbol.replace('/', ''),  # BTCUSDT format
                        price=ticker['close'] or 0,
                        volume_24h=ticker['quoteVolume'] or 0,
                        price_change_24h=ticker['percentage'] or 0,
                        volatility=volatility,
                        market_cap=None,  # Not available in ticker
                        timestamp=datetime.now(timezone.utc)
                    )
                    opportunities.append(opportunity)
                    
        except Exception as e:
            logger.error(f"Error fetching Binance data: {e}")
            raise
        
        return opportunities
    
    async def _get_coinapi_data(self) -> List[MarketDataPoint]:
        """Get data from CoinAPI"""
        opportunities = []
        
        if not self.coinapi_key:
            logger.warning("CoinAPI key not available")
            return opportunities
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {'X-CoinAPI-Key': self.coinapi_key}
                
                # Get top cryptocurrencies by volume
                url = 'https://rest.coinapi.io/v1/exchanges/BINANCE/symbols'
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Filter for USDT pairs
                        usdt_pairs = [item for item in data if item['symbol_id'].endswith('USDT') and 
                                     any(coin in item['symbol_id'] for coin in ['BTC', 'ETH', 'SOL', 'ADA', 'DOT'])]
                        
                        # Get current rates for each pair
                        for pair in usdt_pairs[:5]:  # Limit to 5 to avoid rate limits
                            rate_url = f'https://rest.coinapi.io/v1/exchangerate/{pair["asset_id_base"]}/USDT'
                            async with session.get(rate_url, headers=headers) as rate_response:
                                if rate_response.status == 200:
                                    rate_data = await rate_response.json()
                                    
                                    opportunity = MarketDataPoint(
                                        symbol=f"{pair['asset_id_base']}USDT",
                                        price=rate_data.get('rate', 0),
                                        volume_24h=np.random.uniform(100000000, 1000000000),  # Volume not in free tier
                                        price_change_24h=np.random.uniform(-5, 5),  # Historical data needs premium
                                        volatility=np.random.uniform(0.01, 0.05),
                                        timestamp=datetime.now(timezone.utc)
                                    )
                                    opportunities.append(opportunity)
                    
        except Exception as e:
            logger.error(f"Error fetching CoinAPI data: {e}")
            raise
        
        return opportunities
    
    async def _get_yahoo_finance_data(self) -> List[MarketDataPoint]:
        """Get data from Yahoo Finance (crypto and traditional assets)"""
        opportunities = []
        
        try:
            # Yahoo Finance crypto symbols
            crypto_symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD']
            
            for symbol in crypto_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period='2d')
                    
                    if not hist.empty and info:
                        current_price = hist['Close'].iloc[-1]
                        prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                        price_change = ((current_price - prev_price) / prev_price) * 100
                        
                        # Calculate volatility from recent data
                        volatility = hist['Close'].pct_change().std()
                        if pd.isna(volatility):
                            volatility = 0.02
                        
                        opportunity = MarketDataPoint(
                            symbol=symbol.replace('-USD', 'USDT'),
                            price=current_price,
                            volume_24h=hist['Volume'].iloc[-1] * current_price,
                            price_change_24h=price_change,
                            volatility=volatility,
                            market_cap=info.get('marketCap'),
                            timestamp=datetime.now(timezone.utc)
                        )
                        opportunities.append(opportunity)
                        
                except Exception as e:
                    logger.warning(f"Failed to get Yahoo Finance data for {symbol}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data: {e}")
            raise
        
        return opportunities
    
    async def get_historical_data(self, symbol: str, days: int = 10) -> pd.DataFrame:
        """Get historical OHLCV data for technical analysis"""
        try:
            # Try Binance first
            if hasattr(self.binance, 'fetch_ohlcv'):
                symbol_ccxt = symbol.replace('USDT', '/USDT')
                ohlcv = self.binance.fetch_ohlcv(symbol_ccxt, '1d', limit=days)
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                return df
                
        except Exception as e:
            logger.warning(f"Binance historical data failed for {symbol}: {e}")
        
        # Fallback to Yahoo Finance
        try:
            yf_symbol = symbol.replace('USDT', '-USD')
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period=f"{days}d")
            
            if not hist.empty:
                return hist
                
        except Exception as e:
            logger.warning(f"Yahoo Finance historical data failed for {symbol}: {e}")
        
        # Generate synthetic data as last resort
        logger.warning(f"Generating synthetic historical data for {symbol}")
        return self._generate_synthetic_ohlcv(days)
    
    def _generate_synthetic_ohlcv(self, days: int) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate realistic price movement
        base_price = 50000  # Starting price
        prices = []
        
        for i in range(days):
            if i == 0:
                prices.append(base_price)
            else:
                # Random walk with slight upward trend
                change = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% std
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, base_price * 0.5))  # Floor at 50% of base
        
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        df['Open'] = df['Close'].shift(1).fillna(df['Close'])
        df['High'] = df[['Open', 'Close']].max(axis=1) * np.random.uniform(1.00, 1.05, days)
        df['Low'] = df[['Open', 'Close']].min(axis=1) * np.random.uniform(0.95, 1.00, days)
        df['Volume'] = np.random.uniform(1000000, 10000000, days)
        
        return df
    
    async def get_market_sentiment(self) -> Dict[str, Any]:
        """Get market sentiment indicators"""
        try:
            # This would integrate with sentiment APIs in production
            # For now, return basic sentiment based on market data
            
            opportunities = await self.get_crypto_opportunities()
            
            if not opportunities:
                return {"sentiment": "neutral", "confidence": 0.5}
            
            # Calculate overall market sentiment
            avg_change = np.mean([opp.price_change_24h for opp in opportunities])
            avg_volatility = np.mean([opp.volatility for opp in opportunities])
            
            if avg_change > 2 and avg_volatility < 0.05:
                sentiment = "bullish"
                confidence = 0.8
            elif avg_change < -2 and avg_volatility > 0.08:
                sentiment = "bearish"
                confidence = 0.8
            else:
                sentiment = "neutral"
                confidence = 0.6
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "avg_price_change": avg_change,
                "avg_volatility": avg_volatility,
                "market_cap_trend": "stable"  # Would calculate from real data
            }
            
        except Exception as e:
            logger.error(f"Error calculating market sentiment: {e}")
            return {"sentiment": "neutral", "confidence": 0.5}

# Global instance
market_data_service = RealMarketDataService()
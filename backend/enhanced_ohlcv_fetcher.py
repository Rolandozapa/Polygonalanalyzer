import pandas as pd
import numpy as np
import aiohttp
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class EnhancedOHLCVFetcher:
    """Enhanced OHLCV data fetcher with improved symbol resolution and multiple sources"""
    
    def __init__(self):
        self.lookback_days = 100  # Increased for better technical indicators
        self.coinapi_key = os.environ.get('COINAPI_KEY')
        self.coingecko_key = os.environ.get('COINGECKO_API_KEY') 
        self.twelvedata_key = os.environ.get('TWELVEDATA_KEY')
        
        # Enhanced symbol mapping
        self.symbol_mappings = {
            # Binance format
            'binance': {
                'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'BNB': 'BNBUSDT',
                'SOL': 'SOLUSDT', 'XRP': 'XRPUSDT', 'ADA': 'ADAUSDT', 
                'DOGE': 'DOGEUSDT', 'AVAX': 'AVAXUSDT', 'DOT': 'DOTUSDT',
                'MATIC': 'MATICUSDT', 'LINK': 'LINKUSDT', 'UNI': 'UNIUSDT',
                'LTC': 'LTCUSDT', 'BCH': 'BCHUSDT', 'ATOM': 'ATOMUSDT',
                'FIL': 'FILUSDT', 'TRX': 'TRXUSDT', 'ETC': 'ETCUSDT',
                'NEAR': 'NEARUSDT', 'ALGO': 'ALGOUSDT', 'VET': 'VETUSDT'
            },
            # CoinGecko format
            'coingecko': {
                'BTCUSDT': 'bitcoin', 'ETHUSDT': 'ethereum', 'BNBUSDT': 'binancecoin',
                'SOLUSDT': 'solana', 'XRPUSDT': 'ripple', 'ADAUSDT': 'cardano',
                'DOGEUSDT': 'dogecoin', 'AVAXUSDT': 'avalanche-2', 'DOTUSDT': 'polkadot',
                'MATICUSDT': 'matic-network', 'LINKUSDT': 'chainlink', 'UNIUSDT': 'uniswap',
                'LTCUSDT': 'litecoin', 'BCHUSDT': 'bitcoin-cash', 'ATOMUSDT': 'cosmos',
                'FILUSDT': 'filecoin', 'TRXUSDT': 'tron', 'ETCUSDT': 'ethereum-classic',
                'NEARUSDT': 'near', 'ALGOUSDT': 'algorand', 'VETUSDT': 'vechain'
            }
        }
        
    async def get_enhanced_ohlcv_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Enhanced OHLCV data fetching with better symbol resolution"""
        # Normalize symbol
        normalized_symbol = self._normalize_symbol(symbol)
        
        logger.info(f"🔍 Fetching enhanced OHLCV for {symbol} (normalized: {normalized_symbol})")
        
        # Try multiple sources with enhanced logic
        sources = [
            ('Binance Enhanced', self._fetch_binance_enhanced),
            ('CoinGecko Enhanced', self._fetch_coingecko_enhanced), 
            ('TwelveData Enhanced', self._fetch_twelvedata_enhanced),
            ('CoinAPI Enhanced', self._fetch_coinapi_enhanced),
            ('Yahoo Finance Enhanced', self._fetch_yahoo_enhanced)
        ]
        
        for source_name, fetch_func in sources:
            try:
                data = await fetch_func(normalized_symbol)
                if data is not None and len(data) >= 50:  # Minimum for good MACD
                    logger.info(f"✅ {source_name} provided {len(data)} days of data for {symbol}")
                    return self._validate_and_clean_data(data)
                elif data is not None:
                    logger.debug(f"⚠️ {source_name} provided insufficient data for {symbol}: {len(data)} days")
            except Exception as e:
                logger.debug(f"❌ {source_name} failed for {symbol}: {e}")
                
        logger.warning(f"❌ All enhanced sources failed for {symbol}")
        return None
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to standard format"""
        symbol = symbol.upper().strip()
        
        # Handle different formats
        if symbol.endswith('USDT'):
            return symbol
        elif symbol.endswith('USD'):
            return symbol.replace('USD', 'USDT')
        else:
            # Try to map base symbol to USDT pair
            if symbol in self.symbol_mappings['binance']:
                return self.symbol_mappings['binance'][symbol]
            else:
                return f"{symbol}USDT"
    
    async def _fetch_binance_enhanced(self, symbol: str) -> Optional[pd.DataFrame]:
        """Enhanced Binance fetching with better error handling"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": "1d", 
                "limit": self.lookback_days
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_binance_data(data, symbol)
                    else:
                        logger.debug(f"Binance API returned {response.status} for {symbol}")
                        
        except Exception as e:
            logger.debug(f"Binance enhanced fetch error for {symbol}: {e}")
        
        return None
    
    async def _fetch_coingecko_enhanced(self, symbol: str) -> Optional[pd.DataFrame]:
        """Enhanced CoinGecko fetching with ID resolution"""
        try:
            # Get coin ID from mapping or try to resolve
            coin_id = self.symbol_mappings['coingecko'].get(symbol)
            if not coin_id:
                coin_id = await self._resolve_coingecko_id(symbol)
                
            if not coin_id:
                return None
                
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
            params = {"vs_currency": "usd", "days": str(self.lookback_days)}
            
            headers = {}
            if self.coingecko_key:
                headers["x-cg-demo-api-key"] = self.coingecko_key
                
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_coingecko_data(data, symbol)
                    else:
                        logger.debug(f"CoinGecko API returned {response.status} for {symbol}")
                        
        except Exception as e:
            logger.debug(f"CoinGecko enhanced fetch error for {symbol}: {e}")
        
        return None
    
    async def _resolve_coingecko_id(self, symbol: str) -> Optional[str]:
        """Try to resolve CoinGecko coin ID from symbol"""
        try:
            # Remove USDT suffix for search
            search_term = symbol.replace('USDT', '').lower()
            
            url = "https://api.coingecko.com/api/v3/search"
            params = {"query": search_term}
            
            headers = {}
            if self.coingecko_key:
                headers["x-cg-demo-api-key"] = self.coingecko_key
                
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        coins = data.get('coins', [])
                        
                        # Look for exact symbol match
                        for coin in coins[:5]:  # Check top 5 results
                            if coin.get('symbol', '').upper() == search_term.upper():
                                logger.debug(f"Resolved {symbol} to CoinGecko ID: {coin['id']}")
                                return coin['id']
                                
        except Exception as e:
            logger.debug(f"CoinGecko ID resolution error for {symbol}: {e}")
        
        return None
    
    async def _fetch_twelvedata_enhanced(self, symbol: str) -> Optional[pd.DataFrame]:
        """Enhanced TwelveData fetching"""
        if not self.twelvedata_key:
            return None
            
        try:
            # Format symbol for TwelveData
            td_symbol = symbol.replace('USDT', '/USD')
            
            url = "https://api.twelvedata.com/time_series"
            params = {
                "symbol": td_symbol,
                "interval": "1day",
                "outputsize": str(self.lookback_days),
                "apikey": self.twelvedata_key
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_twelvedata_data(data, symbol)
                    else:
                        logger.debug(f"TwelveData API returned {response.status} for {symbol}")
                        
        except Exception as e:
            logger.debug(f"TwelveData enhanced fetch error for {symbol}: {e}")
        
        return None
    
    async def _fetch_coinapi_enhanced(self, symbol: str) -> Optional[pd.DataFrame]:
        """Enhanced CoinAPI fetching"""
        if not self.coinapi_key:
            return None
            
        try:
            # Format symbol for CoinAPI
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
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_coinapi_data(data, symbol)
                    else:
                        logger.debug(f"CoinAPI returned {response.status} for {symbol}")
                        
        except Exception as e:
            logger.debug(f"CoinAPI enhanced fetch error for {symbol}: {e}")
        
        return None
    
    async def _fetch_yahoo_enhanced(self, symbol: str) -> Optional[pd.DataFrame]:
        """Enhanced Yahoo Finance fetching as fallback"""
        try:
            import yfinance as yf
            
            # Convert to Yahoo format
            yahoo_symbol = symbol.replace('USDT', '-USD')
            
            # Use yfinance to get data
            ticker = yf.Ticker(yahoo_symbol)
            hist = ticker.history(period=f"{self.lookback_days}d")
            
            if len(hist) > 20:  # Minimum reasonable data
                # Convert to our standard format
                df = pd.DataFrame({
                    'Open': hist['Open'],
                    'High': hist['High'], 
                    'Low': hist['Low'],
                    'Close': hist['Close'],
                    'Volume': hist['Volume']
                })
                df.index = pd.to_datetime(df.index)
                return df
                
        except Exception as e:
            logger.debug(f"Yahoo Finance enhanced fetch error for {symbol}: {e}")
        
        return None
    
    def _parse_binance_data(self, data: List, symbol: str) -> Optional[pd.DataFrame]:
        """Parse Binance klines data"""
        try:
            if not data or len(data) < 10:
                return None
                
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to standard format
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Rename columns to standard format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            return df
            
        except Exception as e:
            logger.debug(f"Error parsing Binance data for {symbol}: {e}")
            return None
    
    def _parse_coingecko_data(self, data: List, symbol: str) -> Optional[pd.DataFrame]:
        """Parse CoinGecko OHLC data"""
        try:
            if not data or len(data) < 10:
                return None
                
            df = pd.DataFrame(data, columns=['timestamp', 'Open', 'High', 'Low', 'Close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add synthetic volume (CoinGecko OHLC doesn't include volume)
            df['Volume'] = 1000000  # Synthetic volume for calculation purposes
            
            return df
            
        except Exception as e:
            logger.debug(f"Error parsing CoinGecko data for {symbol}: {e}")
            return None
    
    def _parse_twelvedata_data(self, data: Dict, symbol: str) -> Optional[pd.DataFrame]:
        """Parse TwelveData response"""
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
                    'Volume': float(item.get('volume', 1000000))
                })
            
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()  # Ensure chronological order
            
            return df
            
        except Exception as e:
            logger.debug(f"Error parsing TwelveData data for {symbol}: {e}")
            return None
    
    def _parse_coinapi_data(self, data: List, symbol: str) -> Optional[pd.DataFrame]:
        """Parse CoinAPI OHLCV data"""
        try:
            if not data or len(data) < 10:
                return None
                
            records = []
            for item in data:
                records.append({
                    'timestamp': pd.to_datetime(item['time_period_start']),
                    'Open': float(item['price_open']),
                    'High': float(item['price_high']),
                    'Low': float(item['price_low']),
                    'Close': float(item['price_close']),
                    'Volume': float(item.get('volume_traded', 1000000))
                })
            
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            logger.debug(f"Error parsing CoinAPI data for {symbol}: {e}")
            return None
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean OHLCV data"""
        try:
            # Remove any rows with NaN values
            df = df.dropna()
            
            # Ensure all prices are positive
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                df = df[df[col] > 0]
            
            # Ensure High >= Low
            df = df[df['High'] >= df['Low']]
            
            # Ensure Volume is non-negative
            df = df[df['Volume'] >= 0]
            
            # Sort by date
            df = df.sort_index()
            
            logger.debug(f"Data validation completed: {len(df)} valid records")
            return df
            
        except Exception as e:
            logger.warning(f"Error validating OHLCV data: {e}")
            return df

# Global instance
enhanced_ohlcv_fetcher = EnhancedOHLCVFetcher()
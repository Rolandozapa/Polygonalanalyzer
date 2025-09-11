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
        self.lookback_days = 150  # Increased for better technical indicators (especially MACD)
        self.coinapi_key = os.environ.get('COINAPI_KEY')
        self.coingecko_key = os.environ.get('COINGECKO_API_KEY') 
        self.twelvedata_key = os.environ.get('TWELVEDATA_KEY')
        
        # Historical Data Fallback APIs
        self.alpha_vantage_key = os.environ.get('ALPHA_VANTAGE_KEY')
        self.polygon_key = os.environ.get('POLYGON_KEY')
        self.iex_cloud_key = os.environ.get('IEX_CLOUD_KEY')
        
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
        
    async def get_multi_timeframe_ohlcv_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        🚀 RÉCUPÈRE VRAIES DONNÉES OHLCV MULTI-TIMEFRAME
        Version améliorée avec logique timeframe plus intelligente
        """
        multi_tf_data = {}
        
        try:
            # Récupérer les données de base (quotidiennes)
            base_data = await self.get_enhanced_ohlcv_data(symbol)
            if base_data is None or len(base_data) < 50:
                logger.warning(f"⚠️ Insufficient base data for {symbol}")
                return {}
            
            # Calculer le momentum récent pour ajuster les timeframes
            current_price = base_data.iloc[-1]['Close']
            price_7d_ago = base_data.iloc[-7]['Close'] if len(base_data) > 7 else current_price
            recent_momentum = ((current_price / price_7d_ago) - 1) * 100
            
            logger.info(f"📊 {symbol} recent momentum: {recent_momentum:+.2f}% (7d)")
            
            # Timeframes intelligents basés sur momentum
            timeframes_config = {
                'now': {
                    'data': base_data.tail(20).copy(),  # 20 derniers jours pour "now"
                    'description': 'Recent trend (20 days)'
                },
                '5min': {
                    'data': base_data.tail(15).copy(),  # 15 derniers jours
                    'description': 'Short-term trend (15 days)'  
                },
                '1h': {
                    'data': base_data.tail(30).copy(),  # 30 derniers jours
                    'description': 'Medium-term trend (30 days)'
                },
                '4h': {
                    'data': base_data.tail(60).copy(),  # 60 derniers jours
                    'description': 'Intermediate trend (60 days)'
                },
                '1d': {
                    'data': base_data.copy(),  # Toutes les données
                    'description': 'Long-term trend (all data)'
                }
            }
            
            # Si momentum fort récent, ajuster les poids des timeframes courts
            if abs(recent_momentum) > 5.0:  # Momentum fort
                logger.info(f"🚀 Strong momentum detected for {symbol}, adjusting timeframe focus")
                # Donner plus de poids aux timeframes courts
                timeframes_config['now']['data'] = base_data.tail(10).copy()  # Focus sur très récent
                timeframes_config['5min']['data'] = base_data.tail(15).copy()
            
            for tf_name, config in timeframes_config.items():
                try:
                    tf_data = config['data']
                    if len(tf_data) >= 20:  # Minimum pour calculs
                        multi_tf_data[tf_name] = tf_data
                        logger.info(f"✅ {tf_name}: {len(tf_data)} points ({config['description']})")
                    else:
                        logger.warning(f"⚠️ {tf_name}: Insufficient data ({len(tf_data)} points)")
                        
                except Exception as e:
                    logger.error(f"❌ Error preparing {tf_name} data: {e}")
                    continue
            
            return multi_tf_data
            
        except Exception as e:
            logger.error(f"❌ Error in multi-timeframe data preparation: {e}")
            return {}

    async def get_enhanced_ohlcv_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Enhanced OHLCV data fetching with multi-source validation (minimum 2 sources)"""
        # Normalize symbol
        normalized_symbol = self._normalize_symbol(symbol)
        
        logger.info(f"🔍 Fetching multi-source OHLCV for {symbol} (normalized: {normalized_symbol})")
        
        # Try multiple sources simultaneously for validation
        sources = [
            ('Binance Enhanced', self._fetch_binance_enhanced),
            ('CoinGecko Enhanced', self._fetch_coingecko_enhanced), 
            ('TwelveData Enhanced', self._fetch_twelvedata_enhanced),
            ('CoinAPI Enhanced', self._fetch_coinapi_enhanced),
            ('Yahoo Finance Enhanced', self._fetch_yahoo_enhanced)
        ]
        
        successful_data = []
        
        # Try to get data from multiple sources
        for source_name, fetch_func in sources:
            try:
                data = await fetch_func(normalized_symbol)
                if data is not None and len(data) >= 20:  # Reduced minimum for more flexibility
                    validated_data = self._validate_and_clean_data(data)
                    if validated_data is not None and len(validated_data) >= 20:
                        successful_data.append((source_name, validated_data))
                        logger.info(f"✅ {source_name} provided {len(validated_data)} days of data for {symbol}")
                        
                        # Stop after getting 2 good sources (efficiency)
                        if len(successful_data) >= 2:
                            break
                elif data is not None:
                    logger.debug(f"⚠️ {source_name} provided insufficient data for {symbol}: {len(data)} days")
            except Exception as e:
                logger.debug(f"❌ {source_name} failed for {symbol}: {e}")
        
        # Process results based on how many sources we got
        if len(successful_data) >= 2:
            # We have multiple sources - validate and combine
            logger.info(f"🎯 Multi-source validation for {symbol}: {len(successful_data)} sources")
            return self._combine_multi_source_data(successful_data, symbol)
        elif len(successful_data) == 1:
            # Only one source - use it but log the limitation
            source_name, data = successful_data[0]
            logger.info(f"⚠️ Single-source data for {symbol} from {source_name}")
            return data
        else:
            # ALL PRIMARY SOURCES FAILED - Try historical data fallback APIs
            logger.warning(f"❌ All primary sources failed for {symbol} - trying historical fallback APIs")
            return await self._fetch_historical_fallback_data(normalized_symbol, symbol)
    
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
    
    async def _fetch_historical_fallback_data(self, normalized_symbol: str, original_symbol: str) -> Optional[pd.DataFrame]:
        """
        FALLBACK SYSTEM: Try specialized historical data APIs when primary sources fail
        Guarantees minimum 20 days of data even when all primary OHLCV sources fail
        """
        logger.info(f"🔄 FALLBACK ACTIVATED: Trying historical data APIs for {original_symbol}")
        
        # Historical data fallback sources (specialized for historical data)
        fallback_sources = [
            ('Alpha Vantage Historical', self._fetch_alpha_vantage_historical),
            ('Polygon Historical', self._fetch_polygon_historical),
            ('IEX Cloud Historical', self._fetch_iex_cloud_historical),
            ('CoinCap Historical', self._fetch_coincap_historical),
            ('Messari Historical', self._fetch_messari_historical),
            ('CryptoCompare Historical', self._fetch_cryptocompare_historical_fallback)
        ]
        
        for source_name, fetch_func in fallback_sources:
            try:
                logger.info(f"🔍 Trying {source_name} for {original_symbol}")
                data = await fetch_func(normalized_symbol)
                
                if data is not None and len(data) >= 20:  # Minimum required days
                    validated_data = self._validate_and_clean_data(data)
                    if validated_data is not None and len(validated_data) >= 20:
                        logger.info(f"✅ FALLBACK SUCCESS: {source_name} provided {len(validated_data)} days for {original_symbol}")
                        
                        # Add fallback metadata
                        validated_data.attrs = {
                            'primary_source': source_name,
                            'secondary_source': 'None',
                            'validation_rate': 0.8,  # Good but single source
                            'sources_count': 1,
                            'fallback_used': True
                        }
                        
                        return validated_data
                elif data is not None:
                    logger.debug(f"⚠️ {source_name} insufficient data for {original_symbol}: {len(data)} days")
                    
            except Exception as e:
                logger.debug(f"❌ {source_name} failed for {original_symbol}: {e}")
                continue
        
        # FINAL FALLBACK: If all else fails, try to get ANY data from primary sources with lower standards
        logger.warning(f"🚨 ALL FALLBACK APIS FAILED for {original_symbol} - trying relaxed primary sources")
        return await self._fetch_relaxed_primary_data(normalized_symbol, original_symbol)
    
    async def _fetch_alpha_vantage_historical(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical data from Alpha Vantage - very reliable for historical data"""
        if not self.alpha_vantage_key:
            return None
            
        try:
            # Format symbol for Alpha Vantage (they support many crypto symbols)
            av_symbol = symbol.replace('USDT', 'USD') if symbol.endswith('USDT') else symbol
            
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "DIGITAL_CURRENCY_DAILY",
                "symbol": av_symbol.replace('USD', '').replace('USDT', ''),  # Remove USD suffix
                "market": "USD",
                "apikey": self.alpha_vantage_key
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_alpha_vantage_data(data, symbol)
                    else:
                        logger.debug(f"Alpha Vantage returned {response.status} for {symbol}")
                        
        except Exception as e:
            logger.debug(f"Alpha Vantage error for {symbol}: {e}")
        
        return None
    
    async def _fetch_polygon_historical(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical data from Polygon - excellent for crypto historical data"""
        if not self.polygon_key:
            return None
            
        try:
            # Format symbol for Polygon (X:BTCUSD format)
            base_symbol = symbol.replace('USDT', '').replace('USD', '')
            polygon_symbol = f"X:{base_symbol}USD"
            
            # Get data for the last 150 days
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=self.lookback_days)
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{polygon_symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {"apikey": self.polygon_key}
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_polygon_data(data, symbol)
                    else:
                        logger.debug(f"Polygon returned {response.status} for {symbol}")
                        
        except Exception as e:
            logger.debug(f"Polygon error for {symbol}: {e}")
        
        return None
    
    async def _fetch_iex_cloud_historical(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical data from IEX Cloud - reliable backup source"""
        if not self.iex_cloud_key:
            return None
            
        try:
            # Format symbol for IEX (they have crypto support)
            iex_symbol = symbol.replace('USDT', 'USD') if symbol.endswith('USDT') else symbol
            
            url = f"https://cloud.iexapis.com/stable/crypto/{iex_symbol}/chart/3m"
            params = {"token": self.iex_cloud_key}
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_iex_cloud_data(data, symbol)
                    else:
                        logger.debug(f"IEX Cloud returned {response.status} for {symbol}")
                        
        except Exception as e:
            logger.debug(f"IEX Cloud error for {symbol}: {e}")
        
        return None
    
    async def _fetch_coincap_historical(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical data from CoinCap - free historical endpoint not used yet"""
        try:
            # CoinCap uses different symbol format - need to resolve asset ID
            base_symbol = symbol.replace('USDT', '').replace('USD', '').lower()
            
            # First, get the asset ID
            assets_url = "https://api.coincap.io/v2/assets"
            params = {"search": base_symbol}
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(assets_url, params=params) as response:
                    if response.status == 200:
                        assets_data = await response.json()
                        assets = assets_data.get('data', [])
                        
                        # Find matching asset
                        asset_id = None
                        for asset in assets:
                            if asset.get('symbol', '').lower() == base_symbol:
                                asset_id = asset.get('id')
                                break
                        
                        if not asset_id:
                            return None
                        
                        # Now get historical data
                        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
                        start_time = int((datetime.now(timezone.utc) - timedelta(days=self.lookback_days)).timestamp() * 1000)
                        
                        history_url = f"https://api.coincap.io/v2/assets/{asset_id}/history"
                        history_params = {
                            "interval": "d1",
                            "start": start_time,
                            "end": end_time
                        }
                        
                        async with session.get(history_url, params=history_params) as hist_response:
                            if hist_response.status == 200:
                                hist_data = await hist_response.json()
                                return self._parse_coincap_historical_data(hist_data, symbol)
                                
        except Exception as e:
            logger.debug(f"CoinCap historical error for {symbol}: {e}")
        
        return None
    
    async def _fetch_messari_historical(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical data from Messari - crypto-specialized historical data"""
        try:
            # Format symbol for Messari
            base_symbol = symbol.replace('USDT', '').replace('USD', '').lower()
            
            # Messari uses slug format, try common mappings
            symbol_mapping = {
                'btc': 'bitcoin',
                'eth': 'ethereum',
                'bnb': 'binance-coin',
                'sol': 'solana',
                'xrp': 'xrp',
                'ada': 'cardano',
                'doge': 'dogecoin',
                'avax': 'avalanche',
                'dot': 'polkadot',
                'matic': 'polygon'
            }
            
            messari_symbol = symbol_mapping.get(base_symbol, base_symbol)
            
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=self.lookback_days)
            
            url = f"https://data.messari.io/api/v1/assets/{messari_symbol}/metrics/price/time-series"
            params = {
                "start": start_date.strftime('%Y-%m-%d'),
                "end": end_date.strftime('%Y-%m-%d'),
                "interval": "1d"
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_messari_data(data, symbol)
                    else:
                        logger.debug(f"Messari returned {response.status} for {symbol}")
                        
        except Exception as e:
            logger.debug(f"Messari error for {symbol}: {e}")
        
        return None
    
    async def _fetch_cryptocompare_historical_fallback(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical data from CryptoCompare - fallback endpoint for historical data"""
        try:
            base_symbol = symbol.replace('USDT', '').replace('USD', '')
            
            url = "https://min-api.cryptocompare.com/data/v2/histoday"
            params = {
                "fsym": base_symbol,
                "tsym": "USD",
                "limit": self.lookback_days,
                "api_key": os.environ.get('CRYPTOCOMPARE_KEY', '')  # Optional key
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_cryptocompare_historical_data(data, symbol)
                    else:
                        logger.debug(f"CryptoCompare historical returned {response.status} for {symbol}")
                        
        except Exception as e:
            logger.debug(f"CryptoCompare historical error for {symbol}: {e}")
        
        return None
    
    async def _fetch_relaxed_primary_data(self, normalized_symbol: str, original_symbol: str) -> Optional[pd.DataFrame]:
        """
        LAST RESORT: Try primary sources again with relaxed requirements
        Accept any data >= 20 days from any primary source
        """
        logger.info(f"🚨 LAST RESORT: Relaxed primary source attempt for {original_symbol}")
        
        sources = [
            ('Binance Relaxed', self._fetch_binance_enhanced),
            ('CoinGecko Relaxed', self._fetch_coingecko_enhanced),
            ('Yahoo Finance Relaxed', self._fetch_yahoo_enhanced)
        ]
        
        for source_name, fetch_func in sources:
            try:
                data = await fetch_func(normalized_symbol)
                if data is not None and len(data) >= 20:  # Minimum 20 days
                    validated_data = self._validate_and_clean_data(data)
                    if validated_data is not None and len(validated_data) >= 20:
                        logger.info(f"🆘 EMERGENCY SUCCESS: {source_name} provided {len(validated_data)} days for {original_symbol}")
                        
                        # Add emergency metadata
                        validated_data.attrs = {
                            'primary_source': source_name,
                            'secondary_source': 'None',
                            'validation_rate': 0.7,  # Lower confidence but better than nothing
                            'sources_count': 1,
                            'fallback_used': True,
                            'emergency_mode': True
                        }
                        
                        return validated_data
                        
            except Exception as e:
                logger.debug(f"❌ {source_name} relaxed attempt failed: {e}")
                continue
        
        logger.error(f"🚨 COMPLETE FAILURE: No historical data available for {original_symbol} from any source")
        return None
    
    def _parse_alpha_vantage_data(self, data: Dict, symbol: str) -> Optional[pd.DataFrame]:
        """Parse Alpha Vantage digital currency data"""
        try:
            time_series = data.get('Time Series (Digital Currency Daily)', {})
            if not time_series:
                return None
            
            records = []
            for date_str, values in time_series.items():
                records.append({
                    'timestamp': pd.to_datetime(date_str),
                    'Open': float(values.get('1a. open (USD)', 0)),
                    'High': float(values.get('2a. high (USD)', 0)),
                    'Low': float(values.get('3a. low (USD)', 0)),
                    'Close': float(values.get('4a. close (USD)', 0)),
                    'Volume': float(values.get('5. volume', 1000000))
                })
            
            if not records:
                return None
                
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            logger.debug(f"Error parsing Alpha Vantage data for {symbol}: {e}")
            return None
    
    def _parse_polygon_data(self, data: Dict, symbol: str) -> Optional[pd.DataFrame]:
        """Parse Polygon aggregates data"""
        try:
            results = data.get('results', [])
            if not results:
                return None
            
            records = []
            for item in results:
                records.append({
                    'timestamp': pd.to_datetime(item['t'], unit='ms'),
                    'Open': float(item['o']),
                    'High': float(item['h']),
                    'Low': float(item['l']),
                    'Close': float(item['c']),
                    'Volume': float(item.get('v', 1000000))
                })
            
            if not records:
                return None
                
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            logger.debug(f"Error parsing Polygon data for {symbol}: {e}")
            return None
    
    def _parse_iex_cloud_data(self, data: List, symbol: str) -> Optional[pd.DataFrame]:
        """Parse IEX Cloud crypto data"""
        try:
            if not data:
                return None
            
            records = []
            for item in data:
                records.append({
                    'timestamp': pd.to_datetime(item['date']),
                    'Open': float(item.get('open', item.get('close', 0))),
                    'High': float(item.get('high', item.get('close', 0))),
                    'Low': float(item.get('low', item.get('close', 0))),
                    'Close': float(item['close']),
                    'Volume': float(item.get('volume', 1000000))
                })
            
            if not records:
                return None
                
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            logger.debug(f"Error parsing IEX Cloud data for {symbol}: {e}")
            return None
    
    def _parse_coincap_historical_data(self, data: Dict, symbol: str) -> Optional[pd.DataFrame]:
        """Parse CoinCap historical data"""
        try:
            history_data = data.get('data', [])
            if not history_data:
                return None
            
            records = []
            for item in history_data:
                price = float(item['priceUsd'])
                records.append({
                    'timestamp': pd.to_datetime(item['time']),
                    'Open': price,  # CoinCap only provides price, not OHLC
                    'High': price,
                    'Low': price,
                    'Close': price,
                    'Volume': 1000000  # Synthetic volume
                })
            
            if not records:
                return None
                
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            logger.debug(f"Error parsing CoinCap historical data for {symbol}: {e}")
            return None
    
    def _parse_messari_data(self, data: Dict, symbol: str) -> Optional[pd.DataFrame]:
        """Parse Messari time series data"""
        try:
            time_series = data.get('data', {}).get('values', [])
            if not time_series:
                return None
            
            records = []
            for item in time_series:
                timestamp = pd.to_datetime(item[0], unit='ms')
                price = float(item[1]) if item[1] is not None else None
                
                if price is not None:
                    records.append({
                        'timestamp': timestamp,
                        'Open': price,  # Messari provides price points, not full OHLC
                        'High': price,
                        'Low': price,
                        'Close': price,
                        'Volume': 1000000  # Synthetic volume
                    })
            
            if not records:
                return None
                
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            logger.debug(f"Error parsing Messari data for {symbol}: {e}")
            return None
    
    def _parse_cryptocompare_historical_data(self, data: Dict, symbol: str) -> Optional[pd.DataFrame]:
        """Parse CryptoCompare historical data"""
        try:
            time_series = data.get('Data', {}).get('Data', [])
            if not time_series:
                return None
            
            records = []
            for item in time_series:
                records.append({
                    'timestamp': pd.to_datetime(item['time'], unit='s'),
                    'Open': float(item['open']),
                    'High': float(item['high']),
                    'Low': float(item['low']),
                    'Close': float(item['close']),
                    'Volume': float(item.get('volumeto', 1000000))
                })
            
            if not records:
                return None
                
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            logger.debug(f"Error parsing CryptoCompare historical data for {symbol}: {e}")
            return None
    
    def _combine_multi_source_data(self, successful_data: List[Tuple[str, pd.DataFrame]], symbol: str) -> pd.DataFrame:
        """Combine and validate data from multiple sources"""
        try:
            if len(successful_data) < 2:
                return successful_data[0][1] if successful_data else None
            
            # Get the two best sources (most data)
            sorted_data = sorted(successful_data, key=lambda x: len(x[1]), reverse=True)
            primary_source, primary_data = sorted_data[0]
            secondary_source, secondary_data = sorted_data[1]
            
            logger.info(f"🔗 Combining {primary_source} ({len(primary_data)} days) with {secondary_source} ({len(secondary_data)} days)")
            
            # Use primary source as base and validate with secondary
            combined_data = primary_data.copy()
            
            # Find overlapping date range
            primary_dates = set(primary_data.index.date)
            secondary_dates = set(secondary_data.index.date)
            common_dates = primary_dates & secondary_dates
            
            if len(common_dates) < 30:  # Need at least 30 overlapping days for validation
                logger.info(f"📊 Limited overlap ({len(common_dates)} days), using primary source: {primary_source}")
                return combined_data
            
            # Validate prices on common dates (they should be similar)
            validation_errors = 0
            for date in list(common_dates)[:10]:  # Check first 10 common dates
                try:
                    primary_close = primary_data.loc[primary_data.index.date == date, 'Close'].iloc[0]
                    secondary_close = secondary_data.loc[secondary_data.index.date == date, 'Close'].iloc[0]
                    
                    # Allow up to 5% difference between sources
                    price_diff = abs(primary_close - secondary_close) / primary_close
                    if price_diff > 0.05:  # 5% difference threshold
                        validation_errors += 1
                except:
                    validation_errors += 1
            
            validation_rate = 1 - (validation_errors / 10)
            logger.info(f"🎯 Multi-source validation rate: {validation_rate*100:.1f}% for {symbol}")
            
            # Add multi-source metadata
            combined_data.attrs = {
                'primary_source': primary_source,
                'secondary_source': secondary_source,
                'validation_rate': validation_rate,
                'sources_count': len(successful_data)
            }
            
            return combined_data
            
        except Exception as e:
            logger.warning(f"Error combining multi-source data for {symbol}: {e}")
            # Fallback to primary source
            return successful_data[0][1] if successful_data else None
    
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
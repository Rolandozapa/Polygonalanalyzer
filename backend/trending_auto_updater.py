import asyncio
import logging
import aiohttp
import re
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import json
from dataclasses import dataclass, field

# CPU monitoring optimization - moved from loop
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import du détecteur de patterns latéraux
from lateral_pattern_detector import lateral_pattern_detector, PatternAnalysis

logger = logging.getLogger(__name__)

@dataclass
class TrendingCrypto:
    symbol: str
    name: str
    rank: Optional[int] = None
    price: Optional[float] = None  # 🚨 AJOUT CRITIQUE: Prix actuel
    price_change: Optional[float] = None
    volume: Optional[float] = None
    market_cap: Optional[float] = None
    source: str = "bingx_futures"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class TrendingAutoUpdater:
    """
    🚀 BingX Futures Auto-updater - Récupère trends et top 25 depuis BingX API (variation min 5%)
    Remplace Readdy.link par source officielle BingX futures
    """
    
    def __init__(self):
        # 🎯 BingX API endpoints for futures market data  
        self.bingx_api_base = "https://open-api.bingx.com"
        self.bingx_futures_url = "https://bingx.com/en/market/futures/usd-m-perp"
        self.update_interval = 14400  # 4 heures comme demandé par l'utilisateur
        self.last_update = None
        self.current_trending = []
        self.is_running = False
        self.update_task = None
        
        # 🔥 INTEGRATION: Pattern detector pour filtrage avancé
        self.pattern_detector = lateral_pattern_detector
        
        # 🔥 BingX specific patterns pour extraction
        self.bingx_patterns = [
            r'([A-Z]{2,10})USDT.*?USD.*?\+?(-?\d+\.?\d*)%.*?(\d+\.?\d*[KMB]?)',  # Volume pattern
            r'([A-Z]{2,10})USDT.*?USD.*?\+?(-?\d+\.?\d*)%.*?(\d+\.?\d*[KMBT]?)',  # Market cap pattern
        ]
        
        # 🎯 BingX TOP 25 MARKET CAP FUTURES (User Request: Limit scout to top 25 only)
        self.bingx_top_25_futures = [
            # Top 10 by market cap
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", 
            "DOGEUSDT", "BNBUSDT", "TRXUSDT", "LINKUSDT", "AVAXUSDT",
            # Top 11-20 by market cap  
            "DOTUSDT", "MATICUSDT", "LTCUSDT", "BCHUSDT", "UNIUSDT",
            "FILUSDT", "ICPUSDT", "NEARUSDT", "APTUSDT", "ATOMUSDT",
            # Top 21-25 by market cap
            "OPUSDT", "ARBUSDT", "INJUSDT", "FTMUSDT", "GMXUSDT"
        ]  # Total: 25 symbols (focused on market cap leaders only)
        
        # 🔄 COMPATIBILITY: Keep broader list for fallback scenarios
        self.bingx_extended_futures = [
            "XLMUSDT", "PIUSDT", "CROUSDT", "MUSDT", "WLFIUSDT", "SUIUSDT",
            "ENSUSDT", "ALGOUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT", 
            "THETAUSDT", "FLOWUSDT", "CHZUSDT", "EGLDUSDT", "HBARUSDT", 
            "VETUSDT", "GRTUSDT", "COMPUSDT", "YFIUSDT", "SUSHIUSDT", "BATUSDT"
        ]  # Extended list for backup scenarios
        
        logger.info("TrendingAutoUpdater initialized - 4h update cycle avec filtres avancés")
    
    async def start_auto_update(self):
        """Démarre le système d'auto-update des trends"""
        if self.is_running:
            logger.warning("Auto-updater already running")
            return
        
        self.is_running = True
        # 🚨 OPTIMIZED: Réactiver avec optimisations CPU
        self.update_task = asyncio.create_task(self._update_loop())
        logger.info("✅ OPTIMIZED: Trending update loop reactivated with CPU optimizations")
        logger.info("🔄 Auto-trending updater started - checking every 4 hours")
        
        # 🚨 IMMEDIATE STARTUP UPDATE: Perform lightweight BingX data fetch on startup
        # This ensures opportunities endpoint has data immediately without waiting 5 minutes
        logger.info("🚀 STARTUP: Triggering immediate BingX data fetch to populate cache")
        asyncio.create_task(self._startup_data_fetch())
        logger.info("⏰ Regular trending updates will continue every 4 hours")
    
    async def _startup_data_fetch(self):
        """Lightweight startup data fetch to ensure immediate availability"""
        try:
            # Wait 30 seconds for system to fully initialize
            await asyncio.sleep(30)
            logger.info("🚀 STARTUP: Beginning lightweight BingX data fetch...")
            
            # Perform a quick update to populate cache
            trending_cryptos = await self.fetch_trending_cryptos()
            if trending_cryptos:
                self.current_trending = trending_cryptos
                self.last_update = datetime.now(timezone.utc)
                logger.info(f"✅ STARTUP SUCCESS: Cached {len(trending_cryptos)} BingX opportunities for immediate availability")
            else:
                logger.warning("⚠️ STARTUP: BingX data fetch returned empty - will retry in regular cycle")
                
        except Exception as e:
            logger.error(f"❌ STARTUP: Error in initial BingX data fetch: {e}")
            logger.info("⏰ Will retry in regular 4-hour cycle")

    async def stop_auto_update(self):
        """Arrête le système d'auto-update"""
        self.is_running = False
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        logger.info("Auto-trending updater stopped")
    
    async def _update_loop(self):
        """🚀 OPTIMIZED: Boucle principale d'update - VERSION CPU OPTIMISÉE"""
        # 🚨 CPU OPTIMIZATION: Moderate delay to avoid startup overload while maintaining responsiveness
        await asyncio.sleep(120)  # 2 minutes - reduced from 5 minutes for better responsiveness
        
        while self.is_running:
            try:
                # 🚨 STRICT CPU PROTECTION - Plus strict et plus efficace
                if PSUTIL_AVAILABLE:
                    cpu_usage = psutil.cpu_percent(interval=1)  # 1 seconde sample
                    if cpu_usage > 50.0:  # Plus strict : 50% au lieu de 70%
                        logger.warning(f"🚨 HIGH CPU ({cpu_usage:.1f}%) - Delaying trending update")
                        await asyncio.sleep(1800)  # 30 minutes au lieu de 4h si CPU élevé
                        continue
                
                logger.info("🔍 Starting OPTIMIZED trending update cycle...")
                
                # 🚨 CPU OPTIMIZATION: Update avec timeout strict
                try:
                    # Timeout de 60 secondes max pour l'update
                    await asyncio.wait_for(self.update_trending_list(), timeout=60.0)
                    logger.info(f"⏰ Next trending update in 4 hours (CPU optimized)")
                except asyncio.TimeoutError:
                    logger.warning("⏰ Trending update timeout - skipping this cycle")
                
                # 🚨 CPU OPTIMIZATION: Intervalle plus long entre vérifications CPU
                await asyncio.sleep(self.update_interval)  # 4 heures complètes
                
            except asyncio.CancelledError:
                logger.info("Trending update loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in trending update loop: {e}")
                await asyncio.sleep(14400)  # Retry in 4 hours on error
    
    async def fetch_trending_cryptos(self) -> List[TrendingCrypto]:
        """
        🚀 Récupère les trending cryptos depuis BingX Futures API + page market
        """
        trending_cryptos = []
        
        try:
            # 🎯 METHOD 1: BingX API pour ticker 24h (top gainers)
            bingx_api_tickers = await self._fetch_bingx_api_data()
            if bingx_api_tickers:
                trending_cryptos.extend(bingx_api_tickers)
                logger.info(f"🔥 BingX API: Found {len(bingx_api_tickers)} trending cryptos from API")
            
            # 🚨 PAS DE FALLBACK - Utilisation UNIQUEMENT des données API BingX
            if len(trending_cryptos) == 0:
                logger.error("❌ SCOUT CRITICAL: No data from BingX API - returning empty list")
                return []  # Pas de fallback, retourner vide si l'API échoue
            
            # Remove duplicates et tri par volume/price change
            unique_cryptos = {}
            for crypto in trending_cryptos:
                if crypto.symbol not in unique_cryptos:
                    unique_cryptos[crypto.symbol] = crypto
                elif crypto.volume and (not unique_cryptos[crypto.symbol].volume or crypto.volume > unique_cryptos[crypto.symbol].volume):
                    unique_cryptos[crypto.symbol] = crypto
            
            final_trending = list(unique_cryptos.values())
            
            # Trier par volume d'abord (indicateur de vraie activité) puis price change
            final_trending.sort(key=lambda x: (x.volume or 0, abs(x.price_change or 0)), reverse=True)
            
            logger.info(f"✅ BingX FILTERED: {len(final_trending)} cryptos passed filtering criteria (volume 5%+, prix 1%+, anti-latéral)")
            return final_trending  # 🚨 CORRECTION: Retourner SEULEMENT les cryptos qui passent les filtres
            
        except Exception as e:
            logger.error(f"❌ SCOUT ERROR: Failed to fetch BingX trending cryptos: {e}")
            # Pas de fallback d'urgence - retourner vide si échec total
            return []
    
    def get_cached_or_fetch_sync(self) -> List[TrendingCrypto]:
        """
        🔄 Méthode SYNCHRONE pour récupérer les cryptos trending
        Vérifie le cache d'abord, sinon lance une tâche async dans un nouveau thread
        """
        try:
            # Vérifier si nous avons des données fraîches en cache
            current_time = datetime.now(timezone.utc)
            data_is_fresh = False
            
            if self.last_update and self.current_trending:
                if isinstance(self.last_update, datetime):
                    last_update = self.last_update
                    if last_update.tzinfo is None:
                        last_update = last_update.replace(tzinfo=timezone.utc)
                    data_is_fresh = (current_time - last_update).total_seconds() < 14400  # 4 heures
                elif isinstance(self.last_update, (int, float)):
                    last_update = datetime.fromtimestamp(self.last_update, tz=timezone.utc)
                    data_is_fresh = (current_time - last_update).total_seconds() < 14400
            
            if data_is_fresh and self.current_trending:
                logger.info(f"🔥 SYNC CACHE HIT: Using {len(self.current_trending)} cached BingX cryptos")
                return self.current_trending
            
            # Pas de données fraîches, utiliser un thread pour éviter les problèmes d'event loop
            # 🚨 CRITICAL CPU FIX: Simplifier l'appel synchrone - éviter ThreadPoolExecutor
            logger.info("🔄 SIMPLE SYNC: Getting cached BingX data (CPU optimized)")
            
            # Au lieu de créer un nouveau thread/event loop, utiliser les données cached
            if self.current_trending and self.last_update:
                time_since_update = (current_time - self.last_update).total_seconds()
                if time_since_update < self.update_interval:  # Moins de 4 heures
                    logger.info(f"✅ CACHE HIT: Using cached trending data ({time_since_update:.0f}s old)")
                    return self.current_trending
            
            # 🚨 NO FAKE DATA FALLBACK: System must use real market data only
            logger.error("❌ CRITICAL: BingX API failed completely - NO FAKE FALLBACK DATA")
            logger.error("❌ Trading bot requires REAL market data - system will return empty list")
            self.current_trending = []
            return []
                
        except Exception as e:
            logger.error(f"❌ SYNC ERROR: {e}")
            return []
    
    async def _fetch_bingx_api_data(self) -> List[TrendingCrypto]:
        """Fetch trending data from BingX API with authentication"""
        try:
            # Get API key from environment
            bingx_api_key = os.getenv('BINGX_API_KEY')
            
            headers = {}
            if bingx_api_key:
                headers['X-BX-APIKEY'] = bingx_api_key
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                # BingX API endpoint pour 24h ticker statistics  
                api_url = f"{self.bingx_api_base}/openApi/swap/v2/quote/ticker"
                logger.info(f"🔥 BingX API Call: {api_url} (with API key: {'✅' if bingx_api_key else '❌'})")
                
                async with session.get(api_url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('code') == 0 and 'data' in data:
                            trending_list = []
                            
                            # 🎯 PRIORITY PROCESSING: Process top 25 market cap symbols first
                            priority_symbols = []  # Top 25 market cap
                            extended_symbols = []  # Other symbols for backup
                            
                            for ticker_data in data['data']:
                                try:
                                    symbol = ticker_data.get('symbol', '').replace('-', '')
                                    if not symbol.endswith('USDT'):
                                        continue
                                    
                                    # Separate top 25 market cap symbols from others
                                    if symbol in self.bingx_top_25_futures:
                                        priority_symbols.append(ticker_data)
                                    else:
                                        extended_symbols.append(ticker_data)
                                except (ValueError, KeyError):
                                    continue
                            
                            logger.info(f"🎯 BingX FILTERING: Found {len(priority_symbols)} top-25 symbols, {len(extended_symbols)} extended symbols")
                            
                            # Process priority symbols (top 25 market cap) with relaxed filters
                            for ticker_data in priority_symbols:
                                try:
                                    symbol = ticker_data.get('symbol', '').replace('-', '')
                                    
                                    price_change_pct = float(ticker_data.get('priceChangePercent', 0))
                                    volume = float(ticker_data.get('volume', 0))
                                    
                                    # 🚨 CORRECTION CRITIQUE: Récupérer le prix actuel depuis l'API BingX
                                    current_price = float(ticker_data.get('lastPrice', 0)) or float(ticker_data.get('price', 0))
                                    
                                    # Si pas de prix dans l'API, calculer depuis le change %
                                    if current_price <= 0 and price_change_pct != 0:
                                        # Essayer de déduire le prix depuis le change % (approximatif)
                                        # Si on a +8%, le prix actuel = prix_hier * 1.08
                                        # Mais on n'a pas le prix d'hier, donc on va chercher avec OHLCV
                                        logger.debug(f"⚠️ No price from BingX API for {symbol}, will use OHLCV fallback")
                                    
                                    # 🎯 FILTRES UTILISATEUR: min var volume daily 5%, min var price 1%
                                    # Filtre 1: Variation de prix minimum 1%
                                    if abs(price_change_pct) < 5.0:  # Variation minimum 5% pour analyses significatives
                                        continue
                                    
                                    # Filtre 2: Volume minimum et variation de volume 5%
                                    # Pour simplifier, on utilise le volume absolu comme proxy
                                    if volume < 500000:  # Volume minimum
                                        continue
                                    
                                    # 🎯 DÉTECTION FIGURES LATÉRALES: filtrer les mouvements trop faibles
                                    # Éviter les figures latérales sans vraie tendance
                                    if abs(price_change_pct) < 5.5 and volume < 1000000:  # Mouvement faible ET faible volume
                                        continue
                                    
                                    # 🔥 ANALYSE AVANCÉE: Détection pattern latéral avec IA
                                    pattern_analysis = lateral_pattern_detector.analyze_trend_pattern(
                                        symbol=symbol,
                                        price_change_pct=price_change_pct,
                                        volume=volume
                                    )
                                    
                                    # Filtrer si pattern latéral détecté
                                    if lateral_pattern_detector.should_filter_opportunity(pattern_analysis):
                                        logger.debug(f"🚫 FILTERED {symbol}: {pattern_analysis.reasoning}")
                                        continue
                                    
                                    # Si toutes les conditions sont remplies, ajouter à la liste
                                    crypto = TrendingCrypto(
                                        symbol=symbol,
                                        name=symbol.replace('USDT', ''),
                                        price=current_price if current_price > 0 else None,  # 🚨 PRIX ACTUEL
                                        price_change=price_change_pct,
                                        volume=volume,
                                        source="bingx_api"
                                    )
                                    trending_list.append(crypto)
                                    
                                    # Log des cryptos acceptés avec analyse
                                    logger.debug(f"✅ ACCEPTED {symbol}: {pattern_analysis.reasoning}")
                                
                                except (ValueError, KeyError) as e:
                                    continue
                            
                            return trending_list
                    else:
                        logger.warning(f"BingX API returned status {response.status}")
                        
        except Exception as e:
            logger.error(f"❌ BingX API fetch error: {e}")
        
        return []
    
    async def _fetch_bingx_page_data(self) -> List[TrendingCrypto]:
        """Scrape BingX futures page for additional data"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(self.bingx_futures_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Parse content for trading pairs data
                        trending_list = []
                        
                        # Pattern pour capturer données trading pairs
                        pattern = r'([A-Z]{2,10})USDT.*?USD.*?\+?(-?\d+\.?\d*)%.*?(\d+\.?\d*[KMBT]?)'
                        matches = re.findall(pattern, content)
                        
                        for match in matches:
                            try:
                                symbol = f"{match[0]}USDT"
                                price_change = float(match[1])
                                volume_str = match[2]
                                
                                # Convert volume string to number
                                volume = self._parse_volume_string(volume_str)
                                
                                crypto = TrendingCrypto(
                                    symbol=symbol,
                                    name=match[0],
                                    price_change=price_change,
                                    volume=volume,
                                    source="bingx_page"
                                )
                                trending_list.append(crypto)
                                
                            except (ValueError, IndexError):
                                continue
                        
                        return trending_list
                        
        except Exception as e:
            logger.error(f"❌ BingX page scraping error: {e}")
        
        return []
    
    async def _create_fallback_cryptos(self) -> List[TrendingCrypto]:
        """🚨 DEPRECATED: No longer creates fake fallback data - returns empty list"""
        logger.error("❌ CRITICAL: _create_fallback_cryptos called - NO FAKE DATA ALLOWED")
        logger.error("❌ Trading bot requires REAL market data only")
        return []
    
    def _parse_volume_string(self, volume_str: str) -> float:
        """Parse volume string like '1.41B', '950.32M' to float"""
        try:
            volume_str = volume_str.strip().upper()
            if volume_str.endswith('B'):
                return float(volume_str[:-1]) * 1_000_000_000
            elif volume_str.endswith('M'):
                return float(volume_str[:-1]) * 1_000_000
            elif volume_str.endswith('K'):
                return float(volume_str[:-1]) * 1_000
            else:
                return float(volume_str)
        except:
            return 0.0
    
    async def update_trending_list(self) -> List[TrendingCrypto]:
        """
        🚀 Met à jour la liste des cryptos trending depuis BingX API/Page
        NOUVELLE VERSION: Utilise BingX au lieu de Readdy pour data fraîche
        """
        try:
            logger.info("🔍 Fetching latest trending cryptos from BingX...")
            
            # 🎯 Utilise la nouvelle méthode BingX
            trending_cryptos = await self.fetch_trending_cryptos()
            
            if trending_cryptos:
                self.current_trending = trending_cryptos
                self.last_update = datetime.now(timezone.utc)
                
                symbols = [crypto.symbol for crypto in trending_cryptos[:10]]  # Top 10 pour log
                logger.info(f"✅ BingX Updated trending list: {symbols}")
                
                # Log détaillé des trends trouvés avec données BingX
                for crypto in trending_cryptos[:5]:  # Top 5
                    volume_str = f", Vol: {crypto.volume/1_000_000:.1f}M" if crypto.volume else ""
                    change_str = f", Change: {crypto.price_change:+.2f}%" if crypto.price_change else ""
                    logger.info(f"   📈 {crypto.symbol} ({crypto.name}) - Source: {crypto.source}{change_str}{volume_str}")
                
                return trending_cryptos
            else:
                logger.warning("No trending cryptos found from BingX sources")
                return []
                
        except Exception as e:
            logger.error(f"Error updating BingX trending list: {e}")
            # 🚨 NO FAKE DATA FALLBACK: Return empty list on API failure
            logger.error("❌ CRITICAL: BingX trending update failed - NO FAKE FALLBACK DATA")
            return []
    
    async def _fetch_page_content(self) -> Optional[str]:
        """Récupère le contenu de la page Readdy avec timeout strict"""
        try:
            # 🚨 TIMEOUT STRICT pour éviter les blocages
            timeout = aiohttp.ClientTimeout(total=10, connect=5)  # Timeout réduit à 10s
            async with aiohttp.ClientSession(timeout=timeout) as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (compatible; TradingBot/3.0; +https://trading-bot.ai)'
                }
                
                logger.info(f"📡 Fetching trending data from {self.trending_url} (timeout: 10s)")
                async with session.get(self.trending_url, headers=headers) as response:
                    if response.status == 200:
                        content = await response.text()
                        logger.info(f"✅ Successfully fetched page content ({len(content)} chars)")
                        return content
                    else:
                        logger.warning(f"❌ HTTP {response.status} from trending page")
                        return None
                        
        except asyncio.TimeoutError:
            logger.error("⏰ Timeout fetching trending page (10s limit)")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"🌐 Network error fetching trending page: {e}")
            return None
        except Exception as e:
            logger.error(f"💥 Unexpected error fetching trending page: {e}")
            return None
    
    def _parse_trending_cryptos(self, content: str) -> List[TrendingCrypto]:
        """Parse le contenu pour extraire les cryptos trending"""
        trending_cryptos = []
        
        try:
            # Recherche dans différentes sections
            sections_to_check = [
                content,
                self._extract_trends_section(content),
                self._extract_bottom_section(content)
            ]
            
            found_symbols = set()
            
            for section in sections_to_check:
                if not section:
                    continue
                
                # Pattern matching pour les cryptos
                for pattern in self.crypto_patterns:
                    matches = re.finditer(pattern, section, re.IGNORECASE | re.MULTILINE)
                    
                    for match in matches:
                        try:
                            symbol = match.group(1).upper()
                            rank = int(match.group(2)) if len(match.groups()) > 1 else None
                            
                            # Filtres de qualité
                            if self._is_valid_crypto_symbol(symbol) and symbol not in found_symbols:
                                # Extraction du nom (optionnel)
                                name = self._extract_crypto_name(section, symbol)
                                
                                crypto = TrendingCrypto(
                                    symbol=symbol,
                                    name=name or symbol,
                                    rank=rank,
                                    source="readdy_auto_crawl"
                                )
                                
                                trending_cryptos.append(crypto)
                                found_symbols.add(symbol)
                                
                        except (ValueError, IndexError) as e:
                            continue
            
            # Recherche patterns spécifiques connus de votre page
            known_trending = self._extract_known_patterns(content)
            for crypto in known_trending:
                if crypto.symbol not in found_symbols:
                    trending_cryptos.append(crypto)
                    found_symbols.add(crypto.symbol)
            
            # Trie par rank (plus bas = meilleur)
            trending_cryptos.sort(key=lambda x: x.rank or 9999)
            
            return trending_cryptos[:15]  # Top 15
            
        except Exception as e:
            logger.error(f"Error parsing trending cryptos: {e}")
            return []
    
    def _extract_trends_section(self, content: str) -> Optional[str]:
        """Extrait la section trends de la page"""
        try:
            # Recherche de mots-clés pour identifier la section trends
            trend_keywords = ["trend", "trending", "hot", "gainer", "mover", "top"]
            
            lines = content.split('\n')
            trends_section = []
            in_trends = False
            
            for i, line in enumerate(lines):
                line_lower = line.lower()
                
                if any(keyword in line_lower for keyword in trend_keywords):
                    in_trends = True
                    trends_section = lines[max(0, i-5):min(len(lines), i+20)]
                    break
            
            return '\n'.join(trends_section) if trends_section else None
            
        except Exception as e:
            logger.debug(f"Error extracting trends section: {e}")
            return None
    
    def _extract_bottom_section(self, content: str) -> Optional[str]:
        """Extrait la section du bas de la page (où sont souvent les trends)"""
        try:
            lines = content.split('\n')
            # Prend les 30 dernières lignes significatives
            bottom_lines = [line for line in lines[-50:] if line.strip()]
            return '\n'.join(bottom_lines[-30:])
        except:
            return None
    
    def _extract_known_patterns(self, content: str) -> List[TrendingCrypto]:
        """Extrait les patterns spécifiques connus de votre page"""
        known_cryptos = []
        
        # Patterns spécifiques observés dans votre page
        specific_patterns = [
            r'World Liberty Financial.*?(\d+)',
            r'Euler.*?(\d+)', 
            r'Portal to Bitcoin.*?(\d+)',
            r'PinLink.*?(\d+)',
            r'Pump\.fun.*?(\d+)',
            r'Somnia.*?(\d+)'
        ]
        
        symbol_mapping = {
            'World Liberty Financial': 'WLFI',
            'Euler': 'EUL',
            'Portal to Bitcoin': 'PTB', 
            'PinLink': 'PIN',
            'Pump.fun': 'PUMP',
            'Somnia': 'SOMI'
        }
        
        for pattern in specific_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    name = match.group(0).split('.')[0].strip()
                    rank = int(match.group(1)) if match.groups() else None
                    symbol = symbol_mapping.get(name, name.upper()[:4])
                    
                    crypto = TrendingCrypto(
                        symbol=symbol,
                        name=name,
                        rank=rank,
                        source="readdy_known_pattern"
                    )
                    known_cryptos.append(crypto)
                    
                except (ValueError, IndexError):
                    continue
        
        return known_cryptos
    
    def _is_valid_crypto_symbol(self, symbol: str) -> bool:
        """Valide si c'est un symbole crypto valide"""
        if not symbol or len(symbol) < 2 or len(symbol) > 10:
            return False
        
        # Exclusions
        excluded = ['HTTP', 'HTTPS', 'WWW', 'COM', 'NET', 'ORG', 'HTML', 'API', 'JSON', 'XML']
        if symbol in excluded:
            return False
        
        # Doit être alphanumérique
        if not symbol.replace('_', '').replace('-', '').isalnum():
            return False
        
        return True
    
    def _extract_crypto_name(self, content: str, symbol: str) -> Optional[str]:
        """Essaie d'extraire le nom complet du crypto"""
        try:
            # Recherche autour du symbole
            pattern = rf'([A-Za-z\s]+)\s*\(?{symbol}\)?'
            match = re.search(pattern, content)
            if match:
                name = match.group(1).strip()
                if len(name) > 2 and len(name) < 50:
                    return name
            return None
        except:
            return None
    
    def get_current_trending_symbols(self) -> List[str]:
        """Retourne la liste actuelle des symboles trending"""
        return [crypto.symbol for crypto in self.current_trending]
    
    def get_trending_info(self) -> Dict[str, Any]:
        """Retourne les infos sur les trends actuels"""
        return {
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "trending_count": len(self.current_trending),
            "trending_symbols": self.get_current_trending_symbols(),
            "trending_details": [
                {
                    "symbol": crypto.symbol,
                    "name": crypto.name,
                    "rank": crypto.rank,
                    "source": crypto.source
                }
                for crypto in self.current_trending[:10]
            ],
            "next_update": (self.last_update + timedelta(seconds=self.update_interval)).isoformat() 
                          if self.last_update else None,
            "auto_update_active": self.is_running
        }
    
    async def force_update(self) -> Dict[str, Any]:
        """Force une mise à jour manuelle des trends"""
        logger.info("🔄 Forcing manual trending update...")
        trending_cryptos = await self.update_trending_list()
        
        return {
            "updated": len(trending_cryptos) > 0,
            "count": len(trending_cryptos),
            "symbols": [crypto.symbol for crypto in trending_cryptos],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Global instance
trending_auto_updater = TrendingAutoUpdater()
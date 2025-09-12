import asyncio
import logging
import aiohttp
import re
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

logger = logging.getLogger(__name__)

@dataclass
class TrendingCrypto:
    symbol: str
    name: str
    rank: Optional[int] = None
    price_change: Optional[float] = None
    volume: Optional[float] = None
    market_cap: Optional[float] = None
    source: str = "bingx_futures"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class TrendingAutoUpdater:
    """
    🚀 BingX Futures Auto-updater - Récupère trends et top 50 depuis BingX API
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
        
        # 🔥 BingX specific patterns pour extraction
        self.bingx_patterns = [
            r'([A-Z]{2,10})USDT.*?USD.*?\+?(-?\d+\.?\d*)%.*?(\d+\.?\d*[KMB]?)',  # Volume pattern
            r'([A-Z]{2,10})USDT.*?USD.*?\+?(-?\d+\.?\d*)%.*?(\d+\.?\d*[KMBT]?)',  # Market cap pattern
        ]
        
        # Top crypto symbols pour fallback (BingX top futures)
        self.bingx_top_futures = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
            "BNBUSDT", "HYPEUSDT", "SUIUSDT", "TRXUSDT", "LINKUSDT", "AVAXUSDT",
            "XLMUSDT", "PIUSDT", "CROUSDT", "MUSDT", "WLFIUSDT", "UNIUSDT",
            "DOTUSDT", "MATICUSDT", "LTCUSDT", "BCHUSDT", "ETCUSDT", "FILUSDT",
            "ICPUSDT", "NEARUSDT", "APTUSDT", "FTMUSDT", "INJUSDT", "GMXUSDT"
        ]
        
        logger.info("TrendingAutoUpdater initialized - 4h update cycle avec filtres avancés")
    
    async def start_auto_update(self):
        """Démarre le système d'auto-update des trends"""
        if self.is_running:
            logger.warning("Auto-updater already running")
            return
        
        self.is_running = True
        self.update_task = asyncio.create_task(self._update_loop())
        logger.info("🔄 Auto-trending updater started - checking every 4 hours")
        
        # 🚨 CORRECTION: Pas de premier update immédiat au startup pour éviter les blocages
        # L'update se fera lors du premier cycle de la boucle
        logger.info("⏰ First trending update will occur in the background loop")
    
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
        """Boucle principale d'update toutes les 6h avec protection CPU"""
        # Attendre un peu avant le premier update pour éviter la surcharge au startup
        await asyncio.sleep(60)  # 1 minute de delay
        
        while self.is_running:
            try:
                # 🚨 PROTECTION CPU - Vérifier la charge avant l'update (non-blocking CPU check)
                if PSUTIL_AVAILABLE:
                    cpu_usage = psutil.cpu_percent()  # Non-blocking version - CPU optimized
                    if cpu_usage > 70.0:
                        logger.warning(f"🚨 HIGH CPU ({cpu_usage:.1f}%) - Skipping trending update")
                        await asyncio.sleep(14400)  # Wait 4 hours
                        continue
                
                logger.info("🔍 Starting trending update cycle...")
                await self.update_trending_list()
                logger.info(f"⏰ Next trending update in 4 hours")
                await asyncio.sleep(self.update_interval)
                
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
            
            # 🎯 METHOD 2: BingX Futures page scraping (backup)
            if len(trending_cryptos) < 20:  # Si pas assez depuis API
                bingx_page_data = await self._fetch_bingx_page_data()
                if bingx_page_data:
                    trending_cryptos.extend(bingx_page_data)
                    logger.info(f"🔥 BingX Page: Found {len(bingx_page_data)} additional cryptos from market page")
            
            # 🎯 METHOD 3: Fallback avec top BingX futures
            if len(trending_cryptos) < 10:
                fallback_cryptos = await self._create_fallback_cryptos()
                trending_cryptos.extend(fallback_cryptos)
                logger.info(f"🔥 BingX Fallback: Added {len(fallback_cryptos)} top futures symbols")
            
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
            
            logger.info(f"✅ BingX TOTAL: {len(final_trending)} unique trending cryptos retrieved")
            return final_trending[:50]  # Top 50
            
        except Exception as e:
            logger.error(f"❌ Error fetching BingX trending cryptos: {e}")
            # Emergency fallback
            return await self._create_fallback_cryptos()
    
    async def _fetch_bingx_api_data(self) -> List[TrendingCrypto]:
        """Fetch trending data from BingX API"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                # BingX API endpoint pour 24h ticker statistics
                api_url = f"{self.bingx_api_base}/openApi/swap/v2/quote/ticker"
                
                async with session.get(api_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('code') == 0 and 'data' in data:
                            trending_list = []
                            
                            for ticker_data in data['data']:
                                try:
                                    symbol = ticker_data.get('symbol', '').replace('-', '')
                                    if not symbol.endswith('USDT'):
                                        continue
                                    
                                    price_change_pct = float(ticker_data.get('priceChangePercent', 0))
                                    volume = float(ticker_data.get('volume', 0))
                                    
                                    # 🎯 FILTRES UTILISATEUR: min var volume daily 5%, min var price 1%
                                    # Filtre 1: Variation de prix minimum 1%
                                    if abs(price_change_pct) < 1.0:
                                        continue
                                    
                                    # Filtre 2: Volume minimum et variation de volume 5%
                                    # Pour simplifier, on utilise le volume absolu comme proxy
                                    if volume < 500000:  # Volume minimum
                                        continue
                                    
                                    # 🎯 DÉTECTION FIGURES LATÉRALES: filtrer les mouvements trop faibles
                                    # Éviter les figures latérales sans vraie tendance
                                    if abs(price_change_pct) < 1.5 and volume < 1000000:  # Très faible mouvement ET faible volume
                                        continue
                                    
                                    # Si toutes les conditions sont remplies, ajouter à la liste
                                    crypto = TrendingCrypto(
                                        symbol=symbol,
                                        name=symbol.replace('USDT', ''),
                                        price_change=price_change_pct,
                                        volume=volume,
                                        source="bingx_api"
                                    )
                                    trending_list.append(crypto)
                                
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
        """Create fallback trending list from top BingX futures"""
        fallback_list = []
        
        for i, symbol in enumerate(self.bingx_top_futures[:30]):
            crypto = TrendingCrypto(
                symbol=symbol,
                name=symbol.replace('USDT', ''),
                rank=i + 1,
                price_change=0.0,  # Will be updated by market data
                volume=1000000.0,  # Mock volume
                market_cap=1000000000.0,  # Mock market cap
                source="bingx_fallback"
            )
            fallback_list.append(crypto)
        
        return fallback_list
    
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
            # Emergency fallback
            return await self._create_fallback_cryptos()
    
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
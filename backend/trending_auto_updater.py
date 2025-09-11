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
    source: str = "readdy_trends"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class TrendingAutoUpdater:
    """
    Auto-updater qui récupère les trends crypto toutes les 6h depuis Readdy
    """
    
    def __init__(self):
        self.trending_url = "https://readdy.link/preview/917833d5-a5d5-4425-867f-4fe110fa36f2/1956022"
        self.update_interval = 21600  # 6 heures en secondes
        self.last_update = None
        self.current_trending = []
        self.is_running = False
        self.update_task = None
        
        # Patterns de détection des cryptos trending
        self.crypto_patterns = [
            r'([A-Z]{2,10})\s*-?\s*.*?Rank\s*#(\d+)',  # Pattern principal
            r'([A-Z]{2,10})\s*\([^)]+\)\s*.*?#(\d+)',   # Pattern avec parenthèses
            r'([A-Z]{2,10})\s*.*?#(\d+)',               # Pattern simple
            r'([A-Z]{2,10}USDT?)',                      # Pattern direct
        ]
        
        logger.info("TrendingAutoUpdater initialized - 6h update cycle")
    
    async def start_auto_update(self):
        """Démarre le système d'auto-update des trends"""
        if self.is_running:
            logger.warning("Auto-updater already running")
            return
        
        self.is_running = True
        self.update_task = asyncio.create_task(self._update_loop())
        logger.info("🔄 Auto-trending updater started - checking every 6 hours")
        
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
                # 🚨 PROTECTION CPU - Vérifier la charge avant l'update
                try:
                    import psutil
                    cpu_usage = psutil.cpu_percent(interval=1)
                    if cpu_usage > 70.0:
                        logger.warning(f"🚨 HIGH CPU ({cpu_usage:.1f}%) - Skipping trending update")
                        await asyncio.sleep(3600)  # Wait 1 hour
                        continue
                except ImportError:
                    pass  # Si psutil pas disponible, continuer normalement
                
                logger.info("🔍 Starting trending update cycle...")
                await self.update_trending_list()
                logger.info(f"⏰ Next trending update in 6 hours")
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                logger.info("Trending update loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in trending update loop: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour on error
    
    async def update_trending_list(self) -> List[TrendingCrypto]:
        """Met à jour la liste des cryptos trending depuis Readdy"""
        try:
            logger.info("🔍 Fetching latest trending cryptos from Readdy...")
            
            # Récupère le contenu de la page
            page_content = await self._fetch_page_content()
            
            if not page_content:
                logger.error("Failed to fetch page content")
                return []
            
            # Parse les cryptos trending
            trending_cryptos = self._parse_trending_cryptos(page_content)
            
            if trending_cryptos:
                self.current_trending = trending_cryptos
                self.last_update = datetime.now(timezone.utc)
                
                symbols = [crypto.symbol for crypto in trending_cryptos]
                logger.info(f"✅ Updated trending list: {symbols}")
                
                # Log détaillé des trends trouvés
                for crypto in trending_cryptos[:5]:  # Top 5
                    logger.info(f"   📈 {crypto.symbol} ({crypto.name}) - Rank #{crypto.rank}")
                
                return trending_cryptos
            else:
                logger.warning("No trending cryptos found in page content")
                return []
                
        except Exception as e:
            logger.error(f"Error updating trending list: {e}")
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
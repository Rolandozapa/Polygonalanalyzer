import requests
import json
import os
import time
import logging
import re
import aiohttp
import asyncio
from typing import List, Dict, Set
from datetime import datetime
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class BingXFuturesFetcher:
    """Récupère et filtre les symboles futures disponibles sur BingX"""
    
    def __init__(self):
        self.base_url = "https://open-api.bingx.com"
        self.futures_page_url = "https://bingx.com/en/market/futures/usd-m-perp"  # Official futures page
        self.allowed_quotes = {"USDT"}  # Seuls les USDT pairs
        self.excluded_keywords = {
            "TEST", "BEAR", "BULL", "UP", "DOWN", "LEVERAGE", 
            "SHORT", "LONG", "3L", "3S", "5L", "5S"
        }  # Tokens à éviter
        self.cache_file = "/app/backend/bingx_tradable_symbols.json"
        self.cache_time_file = "/app/backend/bingx_cache_time.txt"
        
    def get_available_symbols(self) -> List[Dict]:
        """Récupère tous les symboles futures disponibles sur BingX"""
        endpoint = "/openApi/swap/v2/quote/contracts"
        url = self.base_url + endpoint
        
        try:
            logger.info(f"🔍 Récupération symboles BingX depuis {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('code') == 0:
                contracts = data.get('data', [])
                logger.info(f"✅ {len(contracts)} contrats récupérés depuis BingX")
                return contracts
            else:
                logger.error(f"❌ Erreur API BingX: {data.get('msg', 'Unknown error')}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Exception récupération symboles BingX: {e}")
            return []
    
    def filter_symbols(self, symbols: List[Dict]) -> List[str]:
        """Filtre les symboles selon les critères définis"""
        filtered = []
        excluded_count = 0
        
        for symbol_info in symbols:
            symbol = symbol_info.get('symbol', '')
            status = symbol_info.get('status', 0)
            
            # Vérifier que le symbole est actif (status = 1)
            if status != 1:
                excluded_count += 1
                continue
                
            # Exclure les symboles avec des mots-clés interdits
            if any(keyword in symbol.upper() for keyword in self.excluded_keywords):
                excluded_count += 1
                continue
                
            # Filtrer par devise de cotation (USDT seulement)
            if not any(symbol.endswith(quote) for quote in self.allowed_quotes):
                excluded_count += 1
                continue
                
            filtered.append(symbol)
            
        logger.info(f"📊 FILTRAGE BingX: {len(filtered)} symboles gardés, {excluded_count} exclus")
        return sorted(filtered)
    
    def save_to_cache(self, symbols: List[str]):
        """Sauvegarde les symboles filtrés dans le cache"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            cache_data = {
                "symbols": symbols,
                "count": len(symbols),
                "updated_at": datetime.now().isoformat(),
                "source": "BingX Futures API"
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            with open(self.cache_time_file, 'w') as f:
                f.write(str(time.time()))
                
            logger.info(f"💾 Cache BingX mis à jour: {len(symbols)} symboles sauvés")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde cache BingX: {e}")
    
    def load_from_cache(self) -> List[str]:
        """Charge les symboles depuis le cache"""
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
                
            symbols = cache_data.get('symbols', [])
            updated_at = cache_data.get('updated_at', 'Unknown')
            
            logger.info(f"📂 Cache BingX chargé: {len(symbols)} symboles (updated: {updated_at})")
            return symbols
            
        except FileNotFoundError:
            logger.warning(f"⚠️ Cache BingX non trouvé: {self.cache_file}")
            return []
        except Exception as e:
            logger.error(f"❌ Erreur lecture cache BingX: {e}")
            return []
    
    async def get_official_futures_symbols(self) -> List[str]:
        """🎯 NEW: Scrape official BingX futures page for real tradeable symbols"""
        try:
            logger.info(f"🔍 Fetching OFFICIAL BingX futures from {self.futures_page_url}")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                async with session.get(self.futures_page_url, headers=headers) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        
                        # Parse HTML to extract trading pairs
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # Look for trading pairs in various possible selectors
                        trading_pairs = set()
                        
                        # Method 1: Look for elements containing USDT symbols
                        for element in soup.find_all(text=re.compile(r'[A-Z]+USDT')):
                            matches = re.findall(r'([A-Z]{2,10}USDT)', str(element))
                            trading_pairs.update(matches)
                        
                        # Method 2: Look for data attributes or JSON data
                        for script in soup.find_all('script'):
                            if script.string:
                                matches = re.findall(r'"([A-Z]{2,10}USDT)"', script.string)
                                trading_pairs.update(matches)
                        
                        # Method 3: Look for specific classes/IDs (common in trading interfaces)
                        for selector in ['.symbol', '.trading-pair', '[data-symbol]', '.pair-name']:
                            elements = soup.select(selector)
                            for elem in elements:
                                text = elem.get_text() or elem.get('data-symbol', '')
                                if 'USDT' in text:
                                    match = re.search(r'([A-Z]{2,10}USDT)', text)
                                    if match:
                                        trading_pairs.add(match.group(1))
                        
                        # Filter and clean the symbols
                        filtered_symbols = []
                        for symbol in trading_pairs:
                            if self._is_valid_symbol(symbol):
                                filtered_symbols.append(symbol)
                        
                        logger.info(f"✅ OFFICIAL BingX: Found {len(filtered_symbols)} valid futures symbols")
                        logger.info(f"📋 Sample symbols: {sorted(filtered_symbols)[:10]}")
                        
                        return sorted(filtered_symbols)
                    
                    else:
                        logger.error(f"❌ Failed to fetch BingX page: HTTP {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"❌ Error scraping BingX official page: {e}")
            return []
    
    def _is_valid_symbol(self, symbol: str) -> bool:
        """Validate if a symbol should be included"""
        if not symbol.endswith('USDT'):
            return False
            
        # Remove USDT to get base symbol
        base = symbol[:-4]
        
        # Check excluded keywords
        for keyword in self.excluded_keywords:
            if keyword in base.upper():
                return False
        
        # Must be reasonable length
        if len(base) < 2 or len(base) > 10:
            return False
            
        return True
    
    def is_cache_valid(self, max_age_hours: int = 6) -> bool:
        """Vérifie si le cache est encore valide"""
        try:
            with open(self.cache_time_file, 'r') as f:
                cache_time = float(f.read())
            
            age_hours = (time.time() - cache_time) / 3600
            is_valid = age_hours < max_age_hours
            
            logger.info(f"📅 Cache BingX age: {age_hours:.1f}h ({'valide' if is_valid else 'expiré'})")
            return is_valid
            
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.error(f"❌ Erreur vérification cache: {e}")
            return False
    
    async def get_tradable_symbols_async(self, force_update: bool = False) -> List[str]:
        """
        🎯 NEW: Récupère symboles depuis la page officielle BingX avec système de cache
        force_update: Force la mise à jour même si cache valide
        """
        # Utiliser le cache si valide et pas de force update
        if not force_update and self.is_cache_valid():
            cached_symbols = self.load_from_cache()
            if cached_symbols:
                logger.info(f"📂 Using cached BingX symbols: {len(cached_symbols)} symbols")
                return cached_symbols
        
        # 🎯 PRIMARY: Récupérer depuis la page officielle BingX
        logger.info("🔄 Fetching from OFFICIAL BingX futures page...")
        official_symbols = await self.get_official_futures_symbols()
        
        if official_symbols and len(official_symbols) > 10:  # Minimum reasonable number
            logger.info(f"✅ OFFICIAL SUCCESS: {len(official_symbols)} symbols from BingX page")
            self.save_to_cache(official_symbols)
            return official_symbols
        
        # FALLBACK: Utiliser l'ancienne API si le scraping échoue
        logger.warning("⚠️ Official page scraping failed, trying API fallback...")
        api_symbols = self.get_available_symbols()
        
        if api_symbols:
            tradable_symbols = self.filter_symbols(api_symbols)
            logger.info(f"✅ API FALLBACK: {len(tradable_symbols)} symbols from API")
            self.save_to_cache(tradable_symbols)
            return tradable_symbols
        
        # LAST RESORT: Utiliser cache expiré si tout échoue
        logger.error("❌ Both official page and API failed, using expired cache")
        return self.load_from_cache()
    
    def get_tradable_symbols(self, force_update: bool = False) -> List[str]:
        """
        Synchronous wrapper for async symbol fetching
        """
        try:
            # Try to use existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create a task
                import threading
                result = [None]
                exception = [None]
                
                def run_async():
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        result[0] = new_loop.run_until_complete(self.get_tradable_symbols_async(force_update))
                        new_loop.close()
                    except Exception as e:
                        exception[0] = e
                
                thread = threading.Thread(target=run_async)
                thread.start()
                thread.join(timeout=60)  # 60 second timeout
                
                if exception[0]:
                    raise exception[0]
                return result[0] if result[0] else []
            else:
                return loop.run_until_complete(self.get_tradable_symbols_async(force_update))
        except Exception as e:
            logger.error(f"❌ Error in sync wrapper: {e}")
            # Fallback to old method
            return self._fallback_sync_method(force_update)
    
    def _fallback_sync_method(self, force_update: bool = False) -> List[str]:
        """Fallback synchronous method using old API approach"""
        if not force_update and self.is_cache_valid():
            cached_symbols = self.load_from_cache()
            if cached_symbols:
                return cached_symbols
        
        # Use old API method
        all_symbols = self.get_available_symbols()
        if all_symbols:
            tradable_symbols = self.filter_symbols(all_symbols)
            self.save_to_cache(tradable_symbols)
            return tradable_symbols
        
        return self.load_from_cache()
    
    def is_symbol_tradable(self, symbol: str) -> bool:
        """Vérifie si un symbole spécifique est tradable sur BingX - Format flexible"""
        tradable_symbols = self.get_tradable_symbols()
        
        # Test direct
        if symbol in tradable_symbols:
            return True
        
        # Test avec tiret : WLDUSDT → WLD-USDT
        if symbol.endswith('USDT') and '-' not in symbol:
            base = symbol[:-4]  # Enlever USDT
            dash_format = f"{base}-USDT"
            if dash_format in tradable_symbols:
                return True
        
        # Test sans tiret : WLD-USDT → WLDUSDT  
        if '-USDT' in symbol:
            no_dash_format = symbol.replace('-', '')
            if no_dash_format in tradable_symbols:
                return True
        
        return False

# Instance globale
bingx_fetcher = BingXFuturesFetcher()

def get_bingx_tradable_symbols() -> List[str]:
    """Fonction helper pour récupérer les symboles BingX"""
    return bingx_fetcher.get_tradable_symbols()

def is_bingx_tradable(symbol: str) -> bool:
    """Fonction helper pour vérifier si un symbole est tradable"""
    return bingx_fetcher.is_symbol_tradable(symbol)
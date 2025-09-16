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
    
    def get_tradable_symbols(self, force_update: bool = False) -> List[str]:
        """
        Récupère les symboles tradables avec système de cache intelligent
        force_update: Force la mise à jour depuis l'API même si cache valide
        """
        # Utiliser le cache si valide et pas de force update
        if not force_update and self.is_cache_valid():
            cached_symbols = self.load_from_cache()
            if cached_symbols:
                return cached_symbols
        
        # Récupérer depuis l'API BingX
        logger.info("🔄 Mise à jour symboles BingX depuis API...")
        all_symbols = self.get_available_symbols()
        
        if not all_symbols:
            # Fallback sur cache même expiré si API échoue
            logger.warning("⚠️ API BingX échouée, utilisation cache expiré")
            return self.load_from_cache()
        
        # Filtrer et sauvegarder
        tradable_symbols = self.filter_symbols(all_symbols)
        self.save_to_cache(tradable_symbols)
        
        return tradable_symbols
    
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
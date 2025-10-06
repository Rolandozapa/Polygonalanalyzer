#!/usr/bin/env python3
"""
TRENDING PARSER DEBUG SCRIPT
Debug the Readdy.link content parsing to understand why "No trending cryptos found"
"""

import asyncio
import aiohttp
import re
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrendingParserDebug:
    def __init__(self):
        self.trending_url = "https://readdy.link/preview/917833d5-a5d5-4425-867f-4fe110fa36f2/1956022"
        
        # Patterns from trending_auto_updater.py
        self.crypto_patterns = [
            r'([A-Z]{2,10})\s*-?\s*.*?Rank\s*#(\d+)',  # Pattern principal
            r'([A-Z]{2,10})\s*\([^)]+\)\s*.*?#(\d+)',   # Pattern avec parenthèses
            r'([A-Z]{2,10})\s*.*?#(\d+)',               # Pattern simple
            r'([A-Z]{2,10}USDT?)',                      # Pattern direct
        ]
    
    async def debug_trending_parsing(self):
        """Debug the trending parsing process step by step"""
        logger.info("🔍 Starting Trending Parser Debug")
        
        # Step 1: Fetch content
        logger.info("📡 Step 1: Fetching Readdy.link content...")
        content = await self._fetch_page_content()
        
        if not content:
            logger.error("❌ Failed to fetch content")
            return
        
        logger.info(f"✅ Content fetched: {len(content)} characters")
        
        # Step 2: Analyze content structure
        logger.info("\n📊 Step 2: Analyzing content structure...")
        self._analyze_content_structure(content)
        
        # Step 3: Test pattern matching
        logger.info("\n🔍 Step 3: Testing pattern matching...")
        found_cryptos = self._test_pattern_matching(content)
        
        # Step 4: Look for specific crypto indicators
        logger.info("\n🔍 Step 4: Looking for crypto indicators...")
        self._find_crypto_indicators(content)
        
        # Step 5: Try alternative parsing approaches
        logger.info("\n🔍 Step 5: Testing alternative parsing approaches...")
        self._test_alternative_parsing(content)
        
        logger.info(f"\n🎯 FINAL RESULT: Found {len(found_cryptos)} trending cryptos")
        for crypto in found_cryptos:
            logger.info(f"   📈 {crypto}")
    
    async def _fetch_page_content(self):
        """Fetch page content with detailed logging"""
        try:
            timeout = aiohttp.ClientTimeout(total=15, connect=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                async with session.get(self.trending_url, headers=headers) as response:
                    logger.info(f"   Response status: {response.status}")
                    logger.info(f"   Response headers: {dict(response.headers)}")
                    
                    if response.status == 200:
                        content = await response.text()
                        return content
                    else:
                        logger.error(f"   HTTP {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"   Exception: {e}")
            return None
    
    def _analyze_content_structure(self, content):
        """Analyze the structure of the content"""
        lines = content.split('\n')
        logger.info(f"   Total lines: {len(lines)}")
        
        # Look for HTML structure
        html_tags = re.findall(r'<(\w+)', content)
        unique_tags = list(set(html_tags))
        logger.info(f"   HTML tags found: {unique_tags[:20]}")  # First 20 tags
        
        # Look for crypto-related keywords
        crypto_keywords = ['crypto', 'bitcoin', 'ethereum', 'BTC', 'ETH', 'USDT', 'trending', 'rank', 'price', 'market']
        keyword_counts = {}
        
        for keyword in crypto_keywords:
            count = len(re.findall(keyword, content, re.IGNORECASE))
            if count > 0:
                keyword_counts[keyword] = count
        
        logger.info(f"   Crypto keyword counts: {keyword_counts}")
        
        # Look for potential data sections
        if 'json' in content.lower():
            logger.info("   ✅ JSON data detected in content")
        if 'script' in content.lower():
            logger.info("   ✅ JavaScript sections detected")
        if 'data-' in content.lower():
            logger.info("   ✅ Data attributes detected")
    
    def _test_pattern_matching(self, content):
        """Test each pattern against the content"""
        found_cryptos = []
        
        for i, pattern in enumerate(self.crypto_patterns):
            logger.info(f"   Testing pattern {i+1}: {pattern}")
            
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            pattern_matches = []
            
            for match in matches:
                try:
                    symbol = match.group(1).upper()
                    rank = int(match.group(2)) if len(match.groups()) > 1 else None
                    
                    if self._is_valid_crypto_symbol(symbol):
                        pattern_matches.append(f"{symbol} (rank: {rank})")
                        found_cryptos.append(symbol)
                        
                except (ValueError, IndexError):
                    continue
            
            logger.info(f"      Matches: {pattern_matches[:10]}")  # First 10 matches
        
        return list(set(found_cryptos))  # Remove duplicates
    
    def _find_crypto_indicators(self, content):
        """Look for specific crypto indicators in the content"""
        
        # Look for common crypto symbols
        common_symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'XRP', 'DOGE', 'AVAX', 'DOT', 'MATIC', 'LINK']
        found_symbols = []
        
        for symbol in common_symbols:
            if symbol in content:
                found_symbols.append(symbol)
        
        logger.info(f"   Common crypto symbols found: {found_symbols}")
        
        # Look for USDT pairs
        usdt_pattern = r'([A-Z]{2,10})USDT'
        usdt_matches = re.findall(usdt_pattern, content)
        unique_usdt = list(set(usdt_matches))
        logger.info(f"   USDT pairs found: {unique_usdt[:20]}")  # First 20
        
        # Look for price patterns
        price_patterns = [
            r'\$[\d,]+\.?\d*',  # Dollar prices
            r'[\d,]+\.?\d*\s*USD',  # USD prices
            r'[\d,]+\.?\d*\s*%'  # Percentage changes
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, content)
            if matches:
                logger.info(f"   Price pattern '{pattern}' found {len(matches)} times: {matches[:5]}")
    
    def _test_alternative_parsing(self, content):
        """Test alternative parsing approaches"""
        
        # Approach 1: Look for JSON data
        logger.info("   Testing JSON extraction...")
        json_pattern = r'\{[^{}]*"[^"]*"[^{}]*\}'
        json_matches = re.findall(json_pattern, content)
        logger.info(f"      JSON-like structures found: {len(json_matches)}")
        
        # Approach 2: Look for table-like structures
        logger.info("   Testing table structure extraction...")
        table_patterns = [
            r'<tr[^>]*>.*?</tr>',
            r'<td[^>]*>.*?</td>',
            r'<th[^>]*>.*?</th>'
        ]
        
        for pattern in table_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                logger.info(f"      Table pattern '{pattern}' found {len(matches)} times")
        
        # Approach 3: Look for specific text patterns
        logger.info("   Testing text pattern extraction...")
        text_patterns = [
            r'Rank\s*#?\s*(\d+)',
            r'#(\d+)\s*[A-Z]{2,10}',
            r'([A-Z]{2,10})\s*[#\-]\s*(\d+)',
            r'(\d+)\.\s*([A-Z]{2,10})'
        ]
        
        for pattern in text_patterns:
            matches = re.findall(pattern, content)
            if matches:
                logger.info(f"      Text pattern '{pattern}' found {len(matches)} times: {matches[:5]}")
        
        # Approach 4: Look for script tags with data
        logger.info("   Testing script tag extraction...")
        script_pattern = r'<script[^>]*>(.*?)</script>'
        script_matches = re.findall(script_pattern, content, re.DOTALL)
        logger.info(f"      Script tags found: {len(script_matches)}")
        
        for i, script in enumerate(script_matches[:3]):  # Check first 3 scripts
            if any(keyword in script.lower() for keyword in ['crypto', 'bitcoin', 'rank', 'symbol']):
                logger.info(f"      Script {i+1} contains crypto-related content: {script[:200]}...")
    
    def _is_valid_crypto_symbol(self, symbol):
        """Check if symbol looks like a valid crypto symbol"""
        if not symbol or len(symbol) < 2 or len(symbol) > 10:
            return False
        
        # Filter out common false positives
        false_positives = [
            'HTML', 'HTTP', 'HTTPS', 'JSON', 'XML', 'CSS', 'JS', 'API',
            'URL', 'URI', 'UTF', 'ASCII', 'PNG', 'JPG', 'GIF', 'PDF',
            'ZIP', 'RAR', 'TAR', 'GZ', 'SQL', 'PHP', 'ASP', 'JSP'
        ]
        
        return symbol not in false_positives

async def main():
    debug = TrendingParserDebug()
    await debug.debug_trending_parsing()

if __name__ == "__main__":
    asyncio.run(main())
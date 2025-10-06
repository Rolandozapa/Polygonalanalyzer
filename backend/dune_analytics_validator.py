"""
🔮 DUNE ANALYTICS VALIDATOR - IA2 Institutional Validation Layer
Provides on-chain volume, liquidity and DEX activity data for IA2 strategic decisions
"""

import aiohttp
import asyncio
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class DuneValidationData:
    """Dune Analytics validation data for IA2"""
    symbol: str
    dex_volume_24h: float = 0.0
    dex_liquidity: float = 0.0  
    dex_transactions: int = 0
    price_impact_1k: float = 0.0  # Price impact for $1k trade
    liquidity_score: float = 0.0  # 0-100 liquidity health score
    volume_trend_7d: float = 0.0  # Volume trend over 7 days
    institutional_flow: str = "neutral"  # large_inflow, large_outflow, neutral
    dex_dominance: float = 0.0  # DEX volume / Total volume ratio
    validation_timestamp: datetime = None

class DuneAnalyticsValidator:
    """🔮 Dune Analytics Institutional Validator for IA2"""
    
    def __init__(self):
        self.api_key = os.environ.get('DUNE_API_KEY', '2K3F0FhNZ53UxijCdgbdmtFfdeUWjvTd')
        self.base_url = "https://api.dune.com/api/v1"
        
        # Token address mappings for Dune queries
        self.token_mappings = {
            'BTCUSDT': {'address': '0x2260fac5e5542a773aa44fbcfedf7c193bc2c599', 'symbol': 'WBTC'},  # Wrapped BTC
            'ETHUSDT': {'address': '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2', 'symbol': 'WETH'},  # Wrapped ETH
            'UNIUSDT': {'address': '0x1f9840a85d5af5bf1d1762f925bdaddc4201f984', 'symbol': 'UNI'},
            'LINKUSDT': {'address': '0x514910771af9ca656af840dff83e8264ecf986ca', 'symbol': 'LINK'},
            'AAVEUSDT': {'address': '0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9', 'symbol': 'AAVE'},
            'SUSHIUSDT': {'address': '0x6b3595068778dd592e39a122f4f5a5cf09c90fe2', 'symbol': 'SUSHI'},
            'COMPUSDT': {'address': '0xc00e94cb662c3520282e6f5717214004a7f26888', 'symbol': 'COMP'},
            'CRVUSDT': {'address': '0xd533a949740bb3306d119cc777fa900ba034cd52', 'symbol': 'CRV'},
            'SNXUSDT': {'address': '0xc011a73ee8576fb46f5e1c5751ca3b9fe0af2a6f', 'symbol': 'SNX'},
            'MKRUSDT': {'address': '0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2', 'symbol': 'MKR'}
        }
        
        # Pre-built Dune queries for institutional validation
        self.queries = {
            'dex_volume_liquidity': 3420234,  # Custom query for DEX volume and liquidity
            'large_transactions': 3420235,    # Custom query for whale movements  
            'price_impact': 3420236,         # Custom query for liquidity depth
            'volume_trend': 3420237          # Custom query for volume trends
        }

    async def get_institutional_validation(self, symbol: str) -> Optional[DuneValidationData]:
        """🏛️ Get comprehensive institutional validation data from Dune Analytics"""
        try:
            if symbol not in self.token_mappings:
                logger.warning(f"⚠️ Dune: No token mapping for {symbol}")
                return None
            
            token_info = self.token_mappings[symbol]
            token_address = token_info['address']
            
            # Fetch multiple data points in parallel
            tasks = [
                self._get_dex_volume_liquidity(token_address),
                self._get_large_transactions(token_address),
                self._get_price_impact(token_address),
                self._get_volume_trend(token_address)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results into validation data
            validation_data = DuneValidationData(
                symbol=symbol,
                validation_timestamp=datetime.now(timezone.utc)
            )
            
            # Process each result
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"⚠️ Dune query {i} failed: {result}")
                    continue
                    
                if i == 0 and result:  # DEX volume & liquidity
                    validation_data.dex_volume_24h = result.get('volume_24h', 0.0)
                    validation_data.dex_liquidity = result.get('liquidity', 0.0)
                    validation_data.dex_transactions = result.get('transactions', 0)
                    
                elif i == 1 and result:  # Large transactions
                    validation_data.institutional_flow = result.get('flow_direction', 'neutral')
                    
                elif i == 2 and result:  # Price impact
                    validation_data.price_impact_1k = result.get('price_impact_1k', 0.0)
                    validation_data.liquidity_score = result.get('liquidity_score', 0.0)
                    
                elif i == 3 and result:  # Volume trend
                    validation_data.volume_trend_7d = result.get('volume_trend_7d', 0.0)
                    validation_data.dex_dominance = result.get('dex_dominance', 0.0)
            
            logger.info(f"✅ Dune validation for {symbol}: Volume=${validation_data.dex_volume_24h:,.0f}, Liquidity=${validation_data.dex_liquidity:,.0f}")
            return validation_data
            
        except Exception as e:
            logger.error(f"❌ Dune Analytics validation error for {symbol}: {e}")
            return None

    async def _get_dex_volume_liquidity(self, token_address: str) -> Optional[Dict]:
        """Get DEX volume and liquidity data"""
        try:
            # Use a pre-built query or execute custom SQL
            query_sql = f"""
            SELECT 
                SUM(volume_usd) as volume_24h,
                AVG(liquidity_usd) as liquidity,
                COUNT(*) as transactions
            FROM dex.trades 
            WHERE token_a_address = '{token_address}' 
            OR token_b_address = '{token_address}'
            AND block_time >= NOW() - INTERVAL '24 hours'
            """
            
            result = await self._execute_query(query_sql)
            return result
            
        except Exception as e:
            logger.warning(f"❌ Dune DEX volume/liquidity error: {e}")
            return None

    async def _get_large_transactions(self, token_address: str) -> Optional[Dict]:
        """Detect large institutional transactions (whale movements)"""
        try:
            query_sql = f"""
            SELECT 
                SUM(CASE WHEN amount_usd > 100000 THEN amount_usd ELSE 0 END) as large_inflows,
                SUM(CASE WHEN amount_usd > 100000 THEN -amount_usd ELSE 0 END) as large_outflows,
                COUNT(*) as large_tx_count
            FROM erc20_ethereum.evt_transfer 
            WHERE contract_address = '{token_address}'
            AND block_time >= NOW() - INTERVAL '24 hours'
            """
            
            result = await self._execute_query(query_sql)
            
            if result:
                large_inflows = result.get('large_inflows', 0) or 0
                large_outflows = abs(result.get('large_outflows', 0) or 0)
                
                if large_inflows > large_outflows * 1.5:
                    flow_direction = "large_inflow"
                elif large_outflows > large_inflows * 1.5:
                    flow_direction = "large_outflow"  
                else:
                    flow_direction = "neutral"
                    
                return {'flow_direction': flow_direction}
            
            return None
            
        except Exception as e:
            logger.warning(f"❌ Dune large transactions error: {e}")
            return None

    async def _get_price_impact(self, token_address: str) -> Optional[Dict]:
        """Calculate price impact and liquidity depth"""
        try:
            # Simplified liquidity scoring based on recent DEX activity
            query_sql = f"""
            SELECT 
                AVG(amount_usd) as avg_trade_size,
                STDDEV(amount_usd) as trade_volatility,
                COUNT(*) as trade_count
            FROM dex.trades
            WHERE token_a_address = '{token_address}' 
            OR token_b_address = '{token_address}'
            AND block_time >= NOW() - INTERVAL '1 hour'
            """
            
            result = await self._execute_query(query_sql)
            
            if result:
                avg_trade = result.get('avg_trade_size', 0) or 0
                trade_count = result.get('trade_count', 0) or 0
                
                # Simple liquidity scoring (0-100)
                liquidity_score = min(100, (avg_trade / 1000) * (trade_count / 10))
                price_impact_1k = max(0.1, min(5.0, 1000 / max(avg_trade, 100)))
                
                return {
                    'liquidity_score': liquidity_score,
                    'price_impact_1k': price_impact_1k
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"❌ Dune price impact error: {e}")  
            return None

    async def _get_volume_trend(self, token_address: str) -> Optional[Dict]:
        """Get volume trend and DEX dominance"""
        try:
            query_sql = f"""
            SELECT 
                SUM(CASE WHEN block_time >= NOW() - INTERVAL '1 day' THEN volume_usd ELSE 0 END) as volume_24h,
                SUM(CASE WHEN block_time >= NOW() - INTERVAL '7 days' THEN volume_usd ELSE 0 END) as volume_7d
            FROM dex.trades
            WHERE token_a_address = '{token_address}' 
            OR token_b_address = '{token_address}'
            AND block_time >= NOW() - INTERVAL '7 days'
            """
            
            result = await self._execute_query(query_sql)
            
            if result:
                volume_24h = result.get('volume_24h', 0) or 0
                volume_7d = result.get('volume_7d', 0) or 0
                
                # Calculate 7-day trend
                volume_trend_7d = 0.0
                if volume_7d > 0:
                    daily_avg = volume_7d / 7
                    volume_trend_7d = ((volume_24h - daily_avg) / daily_avg) * 100
                
                return {
                    'volume_trend_7d': volume_trend_7d,
                    'dex_dominance': 0.8  # Placeholder - would need CEX data to calculate
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"❌ Dune volume trend error: {e}")
            return None

    async def _execute_query(self, sql_query: str) -> Optional[Dict]:
        """Execute a SQL query on Dune Analytics"""
        try:
            headers = {
                'X-Dune-API-Key': self.api_key,
                'Content-Type': 'application/json'
            }
            
            # Submit query
            submit_data = {
                'query_sql': sql_query,
                'is_private': False
            }
            
            async with aiohttp.ClientSession() as session:
                # Submit query for execution
                async with session.post(
                    f"{self.base_url}/query/execute", 
                    headers=headers, 
                    json=submit_data,
                    timeout=30
                ) as response:
                    
                    if response.status == 200:
                        submit_result = await response.json()
                        execution_id = submit_result.get('execution_id')
                        
                        if execution_id:
                            # Poll for results
                            for attempt in range(10):  # Max 10 attempts
                                await asyncio.sleep(2)  # Wait 2 seconds
                                
                                async with session.get(
                                    f"{self.base_url}/execution/{execution_id}/results",
                                    headers=headers,
                                    timeout=15
                                ) as result_response:
                                    
                                    if result_response.status == 200:
                                        result_data = await result_response.json()
                                        
                                        if result_data.get('state') == 'QUERY_STATE_COMPLETED':
                                            rows = result_data.get('result', {}).get('rows', [])
                                            if rows:
                                                return rows[0]  # Return first row
                                        
                                        elif result_data.get('state') == 'QUERY_STATE_FAILED':
                                            logger.warning("⚠️ Dune query failed")
                                            break
                    
                    else:
                        logger.warning(f"⚠️ Dune API error: {response.status}")
            
            return None
            
        except Exception as e:
            logger.warning(f"❌ Dune query execution error: {e}")
            return None

# Global instance
dune_validator = DuneAnalyticsValidator()
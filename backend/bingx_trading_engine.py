import os
import hmac
import hashlib
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import aiohttp
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class BingXOrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class BingXOrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"
    TRAILING_STOP_MARKET = "TRAILING_STOP_MARKET"

class BingXPositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"

class BingXOrderStatus(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    PENDING_CANCEL = "PENDING_CANCEL"
    REJECTED = "REJECTED"

@dataclass
class BingXBalance:
    asset: str
    balance: float
    available: float
    locked: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class BingXPosition:
    symbol: str
    position_side: str
    position_amount: float
    entry_price: float
    mark_price: float
    pnl: float
    pnl_ratio: float
    margin: float
    isolated_margin: float
    leverage: int
    margin_type: str  # "isolated" or "cross"
    liquidation_price: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class BingXOrder:
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    executed_qty: float
    status: str
    time_in_force: str
    position_side: str
    stop_price: Optional[float] = None
    working_type: Optional[str] = None
    created_time: Optional[datetime] = None
    updated_time: Optional[datetime] = None

@dataclass
class BingXTicker:
    symbol: str
    price_change: float
    price_change_percent: float
    last_price: float
    bid_price: float
    ask_price: float
    volume: float
    quote_volume: float
    high_price: float
    low_price: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class BingXTradingEngine:
    """
    BingX Trading Engine - Interface complète pour trading automatisé
    """
    
    def __init__(self):
        self.api_key = os.getenv('BINGX_API_KEY')
        self.secret_key = os.getenv('BINGX_SECRET_KEY')
        self.base_url = os.getenv('BINGX_BASE_URL', 'https://open-api.bingx.com')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("BingX API credentials are required")
        
        # Trading configuration
        self.default_leverage = 10
        self.max_position_size = 0.1  # 10% of portfolio max
        self.trading_enabled = True
        self.demo_mode = False  # Set to True for paper trading
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.last_request_time = 0
        
        logger.info("BingX Trading Engine initialized - LIVE TRADING MODE")
    
    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC SHA256 signature for BingX API"""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _get_headers(self, signature: str) -> Dict[str, str]:
        """Get headers for BingX API requests"""
        return {
            'X-BX-APIKEY': self.api_key,
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (compatible; BingX-TradingBot/3.0)',
            'X-BX-SIGNATURE': signature
        }
    
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict[str, Any]:
        """Make authenticated request to BingX API"""
        if params is None:
            params = {}
        
        # Add timestamp
        timestamp = int(time.time() * 1000)
        params['timestamp'] = timestamp
        
        # Create query string
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        
        # Generate signature
        signature = self._generate_signature(query_string)
        
        # Prepare request
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers(signature)
        
        try:
            self.total_requests += 1
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                if method.upper() == 'GET':
                    async with session.get(url, headers=headers, params=params, timeout=30) as response:
                        result = await response.json()
                elif method.upper() == 'POST':
                    async with session.post(url, headers=headers, params=params, json=data, timeout=30) as response:
                        result = await response.json()
                elif method.upper() == 'DELETE':
                    async with session.delete(url, headers=headers, params=params, timeout=30) as response:
                        result = await response.json()
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Track performance
            self.last_request_time = time.time() - start_time
            if response.status == 200 and result.get('code') == 0:
                self.successful_requests += 1
            
            # Check for API errors
            if result.get('code') != 0:
                logger.error(f"BingX API error: {result.get('msg', 'Unknown error')} (Code: {result.get('code')})")
                raise Exception(f"BingX API error: {result.get('msg', 'Unknown error')}")
            
            return result.get('data', result)
            
        except Exception as e:
            logger.error(f"BingX API request failed: {e}")
            raise
    
    # Market Data Methods
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange trading rules and symbol information"""
        try:
            result = await self._make_request('GET', '/openApi/swap/v2/quote/contracts')
            return result
        except Exception as e:
            logger.error(f"Failed to get exchange info: {e}")
            return {}
    
    async def get_ticker_24hr(self, symbol: Optional[str] = None) -> List[BingXTicker]:
        """Get 24hr ticker price change statistics"""
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol
            
            result = await self._make_request('GET', '/openApi/swap/v2/quote/ticker', params)
            
            tickers = []
            if isinstance(result, list):
                ticker_data = result
            else:
                ticker_data = [result]
            
            for ticker in ticker_data:
                tickers.append(BingXTicker(
                    symbol=ticker.get('symbol', ''),
                    price_change=float(ticker.get('priceChange', 0)),
                    price_change_percent=float(ticker.get('priceChangePercent', 0)),
                    last_price=float(ticker.get('lastPrice', 0)),
                    bid_price=float(ticker.get('bidPrice', 0)),
                    ask_price=float(ticker.get('askPrice', 0)),
                    volume=float(ticker.get('volume', 0)),
                    quote_volume=float(ticker.get('quoteVolume', 0)),
                    high_price=float(ticker.get('highPrice', 0)),
                    low_price=float(ticker.get('lowPrice', 0))
                ))
            
            return tickers
            
        except Exception as e:
            logger.error(f"Failed to get ticker data: {e}")
            return []
    
    async def get_klines(self, symbol: str, interval: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Get K-line/candlestick data"""
        try:
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            result = await self._make_request('GET', '/openApi/swap/v3/quote/klines', params)
            
            if not result:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(result, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            
            # Convert types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get klines for {symbol}: {e}")
            return pd.DataFrame()
    
    # Account Methods
    async def get_account_balance(self) -> List[BingXBalance]:
        """Get account balance"""
        try:
            result = await self._make_request('GET', '/openApi/swap/v3/user/balance')
            
            balances = []
            if isinstance(result, dict) and 'balance' in result:
                balance_data = result['balance']
            else:
                balance_data = result if isinstance(result, list) else []
            
            for balance in balance_data:
                balances.append(BingXBalance(
                    asset=balance.get('asset', ''),
                    balance=float(balance.get('balance', 0)),
                    available=float(balance.get('availableMargin', 0)),
                    locked=float(balance.get('frozenMargin', 0))
                ))
            
            return balances
            
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return []
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[BingXPosition]:
        """Get current positions"""
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol
            
            result = await self._make_request('GET', '/openApi/swap/v2/user/positions', params)
            
            positions = []
            position_data = result if isinstance(result, list) else []
            
            for pos in position_data:
                if float(pos.get('positionAmt', 0)) != 0:  # Only non-zero positions
                    positions.append(BingXPosition(
                        symbol=pos.get('symbol', ''),
                        position_side=pos.get('positionSide', ''),
                        position_amount=float(pos.get('positionAmt', 0)),
                        entry_price=float(pos.get('entryPrice', 0)),
                        mark_price=float(pos.get('markPrice', 0)),
                        pnl=float(pos.get('unRealizedProfit', 0)),
                        pnl_ratio=float(pos.get('percentage', 0)),
                        margin=float(pos.get('isolatedMargin', 0)),
                        isolated_margin=float(pos.get('isolatedMargin', 0)),
                        leverage=int(pos.get('leverage', 1)),
                        margin_type=pos.get('marginType', 'cross'),
                        liquidation_price=float(pos.get('liquidationPrice', 0)) if pos.get('liquidationPrice') else None
                    ))
            
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    # Trading Methods
    async def place_order(self, 
                         symbol: str,
                         side: BingXOrderSide,
                         order_type: BingXOrderType,
                         quantity: float,
                         price: Optional[float] = None,
                         position_side: BingXPositionSide = BingXPositionSide.BOTH,
                         stop_price: Optional[float] = None,
                         time_in_force: str = 'GTC',
                         client_order_id: Optional[str] = None) -> Optional[BingXOrder]:
        """Place a new order"""
        
        if self.demo_mode:
            logger.info(f"DEMO MODE - Would place order: {side} {quantity} {symbol} at {price}")
            return None
        
        try:
            params = {
                'symbol': symbol,
                'side': side.value,
                'type': order_type.value,
                'quantity': str(quantity),
                'positionSide': position_side.value,
                'timeInForce': time_in_force
            }
            
            if price:
                params['price'] = str(price)
            
            if stop_price:
                params['stopPrice'] = str(stop_price)
            
            if client_order_id:
                params['clientOrderId'] = client_order_id
            
            result = await self._make_request('POST', '/openApi/swap/v2/trade/order', params)
            
            if result:
                order = BingXOrder(
                    order_id=result.get('orderId', ''),
                    client_order_id=result.get('clientOrderId', ''),
                    symbol=symbol,
                    side=side.value,
                    order_type=order_type.value,
                    quantity=quantity,
                    price=price,
                    executed_qty=float(result.get('executedQty', 0)),
                    status=result.get('status', 'NEW'),
                    time_in_force=time_in_force,
                    position_side=position_side.value,
                    stop_price=stop_price,
                    created_time=datetime.now(timezone.utc)
                )
                
                logger.info(f"Order placed successfully: {order.order_id} - {side} {quantity} {symbol}")
                return order
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an existing order"""
        try:
            params = {
                'symbol': symbol,
                'orderId': order_id
            }
            
            result = await self._make_request('DELETE', '/openApi/swap/v2/trade/order', params)
            
            if result:
                logger.info(f"Order cancelled successfully: {order_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[BingXOrder]:
        """Get all open orders"""
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol
            
            result = await self._make_request('GET', '/openApi/swap/v2/trade/openOrders', params)
            
            orders = []
            order_data = result if isinstance(result, list) else []
            
            for order in order_data:
                orders.append(BingXOrder(
                    order_id=order.get('orderId', ''),
                    client_order_id=order.get('clientOrderId', ''),
                    symbol=order.get('symbol', ''),
                    side=order.get('side', ''),
                    order_type=order.get('type', ''),
                    quantity=float(order.get('origQty', 0)),
                    price=float(order.get('price', 0)) if order.get('price') else None,
                    executed_qty=float(order.get('executedQty', 0)),
                    status=order.get('status', ''),
                    time_in_force=order.get('timeInForce', ''),
                    position_side=order.get('positionSide', ''),
                    stop_price=float(order.get('stopPrice', 0)) if order.get('stopPrice') else None,
                    working_type=order.get('workingType'),
                    created_time=datetime.fromtimestamp(order.get('time', 0) / 1000, timezone.utc) if order.get('time') else None,
                    updated_time=datetime.fromtimestamp(order.get('updateTime', 0) / 1000, timezone.utc) if order.get('updateTime') else None
                ))
            
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []
    
    # Advanced Trading Methods
    async def place_stop_loss_take_profit_order(self, 
                                               symbol: str,
                                               side: BingXOrderSide,
                                               quantity: float,
                                               stop_loss_price: Optional[float] = None,
                                               take_profit_price: Optional[float] = None,
                                               position_side: BingXPositionSide = BingXPositionSide.BOTH) -> Dict[str, Optional[BingXOrder]]:
        """Place stop-loss and take-profit orders"""
        
        results = {'stop_loss': None, 'take_profit': None}
        
        # Place stop-loss order
        if stop_loss_price:
            try:
                sl_order = await self.place_order(
                    symbol=symbol,
                    side=side,
                    order_type=BingXOrderType.STOP_MARKET,
                    quantity=quantity,
                    stop_price=stop_loss_price,
                    position_side=position_side
                )
                results['stop_loss'] = sl_order
            except Exception as e:
                logger.error(f"Failed to place stop-loss order: {e}")
        
        # Place take-profit order
        if take_profit_price:
            try:
                tp_order = await self.place_order(
                    symbol=symbol,
                    side=side,
                    order_type=BingXOrderType.TAKE_PROFIT_MARKET,
                    quantity=quantity,
                    stop_price=take_profit_price,
                    position_side=position_side
                )
                results['take_profit'] = tp_order
            except Exception as e:
                logger.error(f"Failed to place take-profit order: {e}")
        
        return results
    
    async def set_leverage(self, symbol: str, leverage: int, margin_type: str = 'ISOLATED') -> bool:
        """Set leverage for a symbol"""
        try:
            params = {
                'symbol': symbol,
                'leverage': leverage,
                'side': 'BOTH'
            }
            
            result = await self._make_request('POST', '/openApi/swap/v2/trade/leverage', params)
            
            if result:
                logger.info(f"Leverage set to {leverage}x for {symbol}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to set leverage for {symbol}: {e}")
            return False
    
    async def close_position(self, symbol: str, position_side: BingXPositionSide = BingXPositionSide.BOTH) -> bool:
        """Close all positions for a symbol"""
        try:
            positions = await self.get_positions(symbol)
            
            for position in positions:
                if abs(position.position_amount) > 0:
                    # Determine order side (opposite of position)
                    side = BingXOrderSide.SELL if position.position_amount > 0 else BingXOrderSide.BUY
                    
                    # Place market order to close position
                    order = await self.place_order(
                        symbol=symbol,
                        side=side,
                        order_type=BingXOrderType.MARKET,
                        quantity=abs(position.position_amount),
                        position_side=BingXPositionSide(position.position_side)
                    )
                    
                    if order:
                        logger.info(f"Position closed: {symbol} - {position.position_side}")
                    else:
                        logger.error(f"Failed to close position: {symbol} - {position.position_side}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to close positions for {symbol}: {e}")
            return False
    
    # Risk Management
    def calculate_position_size(self, account_balance: float, risk_percentage: float, entry_price: float, stop_loss_price: float) -> float:
        """Calculate position size based on risk management"""
        if entry_price <= 0 or stop_loss_price <= 0 or account_balance <= 0:
            return 0
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        # Calculate total risk amount
        total_risk = account_balance * (risk_percentage / 100)
        
        # Calculate position size
        position_size = total_risk / risk_per_unit
        
        # Apply maximum position size limit
        max_position_value = account_balance * self.max_position_size
        max_quantity = max_position_value / entry_price
        
        return min(position_size, max_quantity)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get trading engine performance statistics"""
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'success_rate': success_rate,
            'last_request_time': self.last_request_time,
            'trading_enabled': self.trading_enabled,
            'demo_mode': self.demo_mode,
            'default_leverage': self.default_leverage,
            'max_position_size': self.max_position_size
        }
    
    # Health Check
    async def test_connectivity(self) -> bool:
        """Test API connectivity and authentication"""
        try:
            result = await self.get_account_balance()
            if result:
                logger.info("BingX API connectivity test successful")
                return True
            else:
                logger.error("BingX API connectivity test failed - no balance data")
                return False
        except Exception as e:
            logger.error(f"BingX API connectivity test failed: {e}")
            return False

# Global instance
bingx_trading_engine = BingXTradingEngine()
"""
BingX Integration System for Dual AI Trading Bot
Implements sophisticated order placement, position management, and real-time monitoring
for automated cryptocurrency trading with BingX Futures API.
"""

import os
import asyncio
import hmac
import hashlib
import time
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
import pandas as pd
from pydantic import BaseModel

# Setup logging
logger = logging.getLogger(__name__)

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"

class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class OrderStatus(Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

@dataclass
class TradingPosition:
    symbol: str
    side: str  # LONG or SHORT
    quantity: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: int = 5
    position_id: Optional[str] = None

@dataclass
class OrderState:
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: Optional[float]
    status: str
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    timestamp: datetime = None

@dataclass
class RiskParameters:
    max_position_size: float = 0.1  # Maximum 10% of balance per position
    max_total_exposure: float = 0.5  # Maximum 50% total exposure
    max_leverage: int = 10  # Maximum leverage allowed
    stop_loss_percentage: float = 0.02  # 2% stop loss
    max_drawdown: float = 0.1  # 10% maximum drawdown
    daily_loss_limit: float = 0.05  # 5% daily loss limit

class BingXAuthenticator:
    """Handles BingX API authentication with HMAC-SHA256 signature generation"""
    
    def __init__(self):
        self.api_key = os.getenv('BINGX_API_KEY')
        self.secret_key = os.getenv('BINGX_SECRET_KEY')
        self.base_url = os.getenv('BINGX_BASE_URL', 'https://open-api.bingx.com')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("BingX API credentials not found in environment variables")
    
    def generate_signature(self, method: str, path: str, params: str = "") -> Dict[str, str]:
        """Generate authentication headers for BingX API requests"""
        timestamp = str(int(time.time() * 1000))
        
        # Create signature string according to BingX documentation
        # Format: METHOD + PATH + QUERY_STRING + TIMESTAMP + BODY
        sign_string = f"{method.upper()}{path}{params}{timestamp}"
        
        # Generate HMAC-SHA256 signature
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            sign_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return {
            'X-BX-APIKEY': self.api_key,
            'X-BX-TIMESTAMP': timestamp,
            'X-BX-SIGNATURE': signature,
            'Content-Type': 'application/json'
        }

class RateLimiter:
    """Rate limiter to prevent API throttling - BingX allows 20 requests per second"""
    
    def __init__(self, max_requests: int = 18, time_window: int = 1):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire rate limit token"""
        async with self.lock:
            now = time.time()
            
            # Remove old requests outside the time window
            self.requests = [req_time for req_time in self.requests if req_time > now - self.time_window]
            
            # If we're at the limit, wait
            if len(self.requests) >= self.max_requests:
                sleep_time = self.time_window - (now - self.requests[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    return await self.acquire()
            
            # Add current request
            self.requests.append(now)

class BingXTradingClient:
    """Main BingX trading client with comprehensive trading functionality"""
    
    def __init__(self):
        self.authenticator = BingXAuthenticator()
        self.rate_limiter = RateLimiter()
        self.active_positions: Dict[str, TradingPosition] = {}
        self.pending_orders: Dict[str, OrderState] = {}
        self.order_history: List[OrderState] = []
        self.session_start_balance = 0.0
        self.emergency_stop_triggered = False
        self.risk_params = RiskParameters()
        
        # HTTP client for API calls
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.client.aclose()
    
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        """Make authenticated request to BingX API with rate limiting - Fixed BingX Official Format"""
        await self.rate_limiter.acquire()
        
        url = f"{self.authenticator.base_url}{endpoint}"
        
        # BingX Authentication: Different approach for GET vs POST
        if method.upper() == "GET":
            # GET requests: All parameters in query string
            all_params = {}
            if params:
                all_params.update(params)
            if data:
                all_params.update(data)
            
            # Add timestamp
            all_params['timestamp'] = int(time.time() * 1000)
            
            # Sort parameters alphabetically by key (BingX requirement for consistent signature)
            sorted_params = sorted(all_params.items(), key=lambda x: x[0])
            
            # Create query string for signature (BingX official format)
            query_string = '&'.join([f"{k}={v}" for k, v in sorted_params])
            
            # Generate signature
            signature = hmac.new(
                self.authenticator.secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Build final URL with signature
            final_query_string = f"{query_string}&signature={signature}"
            final_url = f"{url}?{final_query_string}"
            
            # Headers for GET
            headers = {
                'X-BX-APIKEY': self.authenticator.api_key,
                'Content-Type': 'application/json'
            }
            
            # Make GET request
            response = await self.client.get(final_url, headers=headers)
            
        elif method.upper() == "POST":
            # POST requests: Parameters in body, signature based on body params
            request_body = {}
            if params:
                request_body.update(params)
            if data:
                request_body.update(data)
            
            # Add timestamp to body
            request_body['timestamp'] = int(time.time() * 1000)
            
            # Sort parameters alphabetically for signature generation
            sorted_params = sorted(request_body.items(), key=lambda x: x[0])
            
            # Create parameter string for signature (from body parameters)
            params_string = '&'.join([f"{k}={v}" for k, v in sorted_params])
            
            # Generate signature using body parameters
            signature = hmac.new(
                self.authenticator.secret_key.encode('utf-8'),
                params_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Build URL with only signature in query string
            final_url = f"{url}?signature={signature}"
            
            # Headers for POST
            headers = {
                'X-BX-APIKEY': self.authenticator.api_key,
                'Content-Type': 'application/json'
            }
            
            logger.info(f"🚀 BINGX POST REQUEST: URL={final_url}")
            logger.info(f"🚀 BINGX POST BODY: {request_body}")
            
            # Make POST request with JSON body
            response = await self.client.post(final_url, headers=headers, json=request_body)
            
        else:
            # Other methods (PUT, DELETE, etc.)
            raise Exception(f"Unsupported HTTP method: {method}")
        
        try:
            response.raise_for_status()
            result = response.json()
            
            # Check for BingX API specific errors
            if result.get('code') != 0:
                error_msg = result.get('msg', 'Unknown BingX API error')
                logger.error(f"BingX API Error - Code: {result.get('code')}, Message: {error_msg}")
                raise Exception(f"BingX API Error [{result.get('code')}]: {error_msg}")
            
            return result
        
        except httpx.HTTPError as e:
            logger.error(f"HTTP error making request to {endpoint}: {e}")
            # Log more details for debugging
            logger.error(f"URL: {final_url}")
            logger.error(f"Headers: {headers}")
            raise
        except Exception as e:
            logger.error(f"Error making request to {endpoint}: {e}")
            raise
    
    async def get_account_balance(self) -> Dict[str, Any]:
        """Retrieve current account balance and margin information"""
        try:
            result = await self._make_request("GET", "/openApi/swap/v2/user/balance", {})
            
            if 'data' in result and 'balance' in result['data']:
                balance_data = result['data']['balance']
                return {
                    'balance': float(balance_data.get('balance', 0)),
                    'equity': float(balance_data.get('equity', 0)),
                    'used_margin': float(balance_data.get('usedMargin', 0)),
                    'available_margin': float(balance_data.get('availableMargin', 0)),
                    'unrealized_pnl': float(balance_data.get('unrealizedProfit', 0))
                }
            
            return {
                'balance': 0.0,
                'equity': 0.0,
                'used_margin': 0.0,
                'available_margin': 0.0,
                'unrealized_pnl': 0.0
            }
        
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")
            raise
    
    async def get_market_price(self, symbol: str) -> float:
        """Get current market price for a symbol"""
        try:
            result = await self._make_request("GET", "/openApi/swap/v2/quote/ticker", {"symbol": symbol})
            
            if 'data' in result:
                return float(result['data']['lastPrice'])
            else:
                raise Exception(f"No price data for {symbol}")
        
        except Exception as e:
            logger.error(f"Error fetching market price for {symbol}: {e}")
            raise
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a trading pair - Fixed BingX parameters"""
        try:
            data = {
                "symbol": symbol,
                "side": "BOTH",  # Required by BingX for futures leverage
                "leverage": leverage
            }
            
            logger.info(f"🚀 SETTING BINGX LEVERAGE: {symbol} = {leverage}x")
            result = await self._make_request("POST", "/openApi/swap/v2/trade/leverage", data=data)
            return result.get('code') == 0
        
        except Exception as e:
            logger.error(f"Error setting leverage for {symbol}: {e}")
            return False
    
    async def place_market_order(self, position: TradingPosition) -> Dict[str, Any]:
        """Place a market order for the specified position - Fixed BingX Parameter Format"""
        try:
            # Set leverage first
            await self.set_leverage(position.symbol, position.leverage)
            
            # Determine order side based on position direction
            order_side = OrderSide.BUY.value if position.side == "LONG" else OrderSide.SELL.value
            position_side = PositionSide.LONG.value if position.side == "LONG" else PositionSide.SHORT.value
            
            # Create order data with correct BingX parameter names
            order_data = {
                "symbol": position.symbol,
                "side": order_side,
                "positionSide": position_side,
                "type": OrderType.MARKET.value,
                "quantity": str(position.quantity),  # BingX might expect string format
                "timestamp": int(time.time() * 1000)  # Add timestamp as required by BingX
            }
            
            logger.info(f"🚀 PLACING BINGX ORDER: {order_data}")
            
            result = await self._make_request("POST", "/openApi/swap/v2/trade/order", data=order_data)
            
            if 'data' in result:
                order_response = result['data']
                order_id = order_response['orderId']
                
                logger.info(f"✅ BINGX ORDER PLACED: {order_id} for {position.symbol}")
                
                # Track order state
                order_state = OrderState(
                    order_id=order_id,
                    symbol=position.symbol,
                    side=position.side,
                    quantity=position.quantity,
                    price=None,  # Market order, no specific price
                    status="PENDING",
                    remaining_quantity=position.quantity,
                    timestamp=datetime.now()
                )
                
                self.pending_orders[order_id] = order_state
                
                # Start order monitoring
                asyncio.create_task(self.monitor_order(order_id))
                
                # Set stop loss and take profit if specified
                if position.stop_loss or position.take_profit:
                    await self.set_stop_orders(position, order_id)
                
                return order_response
            else:
                raise Exception(f"No order data in BingX response: {result}")
        
        except Exception as e:
            logger.error(f"Error placing market order on BingX: {e}")
            raise
    
    async def set_stop_orders(self, position: TradingPosition, order_id: str):
        """Set stop loss and take profit orders"""
        try:
            if position.stop_loss:
                stop_side = OrderSide.SELL.value if position.side == "LONG" else OrderSide.BUY.value
                position_side = PositionSide.LONG.value if position.side == "LONG" else PositionSide.SHORT.value
                
                stop_data = {
                    "symbol": position.symbol,
                    "side": stop_side,
                    "positionSide": position_side,
                    "type": OrderType.STOP_MARKET.value,
                    "quantity": position.quantity,
                    "stopPrice": position.stop_loss,
                    "workingType": "MARK_PRICE"
                }
                
                await self._make_request("POST", "/openApi/swap/v2/trade/order", data=stop_data)
            
            if position.take_profit:
                tp_side = OrderSide.SELL.value if position.side == "LONG" else OrderSide.BUY.value
                position_side = PositionSide.LONG.value if position.side == "LONG" else PositionSide.SHORT.value
                
                tp_data = {
                    "symbol": position.symbol,
                    "side": tp_side,
                    "positionSide": position_side,
                    "type": OrderType.TAKE_PROFIT_MARKET.value,
                    "quantity": position.quantity,
                    "stopPrice": position.take_profit,
                    "workingType": "MARK_PRICE"
                }
                
                await self._make_request("POST", "/openApi/swap/v2/trade/order", data=tp_data)
        
        except Exception as e:
            logger.error(f"Error setting stop orders: {e}")
    
    async def monitor_order(self, order_id: str):
        """Monitor order status and update position when filled"""
        max_checks = 60  # Monitor for up to 60 iterations (5 minutes)
        check_interval = 5  # Check every 5 seconds
        
        for _ in range(max_checks):
            try:
                if order_id not in self.pending_orders:
                    break
                
                # Check order status
                result = await self._make_request("GET", "/openApi/swap/v2/trade/order", {"orderId": order_id})
                
                if 'data' in result:
                    order_data = result['data']
                    status = order_data['status']
                    filled_qty = float(order_data.get('executedQty', 0))
                    
                    order_state = self.pending_orders[order_id]
                    order_state.status = status
                    order_state.filled_quantity = filled_qty
                    order_state.remaining_quantity = order_state.quantity - filled_qty
                    
                    if status in [OrderStatus.FILLED.value, OrderStatus.PARTIALLY_FILLED.value]:
                        await self.update_position(order_state)
                    
                    if status in [OrderStatus.FILLED.value, OrderStatus.CANCELED.value, OrderStatus.REJECTED.value]:
                        # Move to history
                        self.order_history.append(order_state)
                        del self.pending_orders[order_id]
                        break
                
                await asyncio.sleep(check_interval)
            
            except Exception as e:
                logger.error(f"Error monitoring order {order_id}: {e}")
                await asyncio.sleep(check_interval)
    
    async def update_position(self, order_state: OrderState):
        """Update position based on filled order"""
        symbol = order_state.symbol
        filled_qty = order_state.filled_quantity
        
        # Adjust quantity based on side
        if order_state.side == "LONG":
            position_change = filled_qty
        else:
            position_change = -filled_qty
        
        # Update or create position
        if symbol in self.active_positions:
            current_position = self.active_positions[symbol]
            # Update existing position logic would go here
        else:
            # Create new position
            position = TradingPosition(
                symbol=symbol,
                side=order_state.side,
                quantity=filled_qty,
                position_id=order_state.order_id
            )
            self.active_positions[symbol] = position
        
        logger.info(f"Position updated: {symbol} - {order_state.side} {filled_qty}")
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Retrieve all open positions"""
        try:
            result = await self._make_request("GET", "/openApi/swap/v2/user/positions", {})
            
            if 'data' in result:
                positions = []
                for pos in result['data']:
                    if float(pos['positionAmt']) != 0:
                        positions.append({
                            'symbol': pos['symbol'],
                            'size': float(pos['positionAmt']),
                            'side': 'LONG' if float(pos['positionAmt']) > 0 else 'SHORT',
                            'entry_price': float(pos['entryPrice']),
                            'mark_price': float(pos['markPrice']),
                            'unrealized_pnl': float(pos['unRealizedProfit']),
                            'percentage': float(pos.get('percentage', 0))
                        })
                return positions
            
            return []
        
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            raise
    
    async def close_position(self, symbol: str, position_side: str) -> Dict[str, Any]:
        """Close an existing position"""
        try:
            # Get current position size
            positions = await self.get_positions()
            target_position = next((p for p in positions if p['symbol'] == symbol), None)
            
            if not target_position:
                raise Exception("Position not found")
            
            # Close position with opposite order
            close_side = OrderSide.SELL.value if target_position['side'] == "LONG" else OrderSide.BUY.value
            quantity = abs(target_position['size'])
            
            order_data = {
                "symbol": symbol,
                "side": close_side,
                "positionSide": position_side,
                "type": OrderType.MARKET.value,
                "quantity": quantity
            }
            
            result = await self._make_request("POST", "/openApi/swap/v2/trade/order", data=order_data)
            return result.get('data', {})
        
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            raise
    
    async def get_trading_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trading history"""
        try:
            params = {"limit": limit}
            result = await self._make_request("GET", "/openApi/swap/v2/trade/allOrders", params)
            
            if 'data' in result:
                return result['data']
            
            return []
        
        except Exception as e:
            logger.error(f"Error fetching trading history: {e}")
            raise

class RiskManager:
    """Advanced risk management system for position sizing and safety controls"""
    
    def __init__(self, trading_client: BingXTradingClient):
        self.trading_client = trading_client
        self.risk_params = RiskParameters()
        self.daily_pnl = 0.0
        self.session_start_balance = 0.0
        self.emergency_stop_triggered = False
    
    async def initialize_session(self):
        """Initialize risk management session"""
        balance = await self.trading_client.get_account_balance()
        self.session_start_balance = balance['balance']
        self.daily_pnl = 0.0
        self.emergency_stop_triggered = False
        logger.info(f"Risk management session initialized with balance: {self.session_start_balance}")
    
    async def validate_position(self, position: TradingPosition) -> Dict[str, Any]:
        """Validate position against risk parameters"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'adjusted_position': None
        }
        
        # Check emergency stop
        if self.emergency_stop_triggered:
            validation_result['valid'] = False
            validation_result['errors'].append("Emergency stop is active")
            return validation_result
        
        # Get current account state
        balance = await self.trading_client.get_account_balance()
        current_positions = await self.trading_client.get_positions()
        
        # Calculate position value
        market_price = await self.trading_client.get_market_price(position.symbol)
        position_value = position.quantity * market_price * position.leverage
        
        # Check position size limit
        max_position_value = balance['balance'] * self.risk_params.max_position_size
        if position_value > max_position_value:
            # Adjust position size
            adjusted_quantity = max_position_value / (market_price * position.leverage)
            validation_result['warnings'].append(
                f"Position size reduced from {position.quantity} to {adjusted_quantity:.6f} due to risk limits"
            )
            position.quantity = adjusted_quantity
            validation_result['adjusted_position'] = position
        
        # Check total exposure limit
        total_exposure = sum(abs(pos['size']) * pos['mark_price'] for pos in current_positions)
        new_total_exposure = total_exposure + position_value
        max_total_value = balance['balance'] * self.risk_params.max_total_exposure
        
        if new_total_exposure > max_total_value:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"Total exposure would exceed limit: {new_total_exposure:.2f} > {max_total_value:.2f}"
            )
        
        # Check leverage limit
        if position.leverage > self.risk_params.max_leverage:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"Leverage {position.leverage} exceeds maximum {self.risk_params.max_leverage}"
            )
        
        # Check daily loss limit
        current_drawdown = (self.session_start_balance - balance['balance']) / self.session_start_balance
        if current_drawdown > self.risk_params.daily_loss_limit:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"Daily loss limit exceeded: {current_drawdown:.2%} > {self.risk_params.daily_loss_limit:.2%}"
            )
            await self.trigger_emergency_stop("Daily loss limit exceeded")
        
        # Set automatic stop loss if not specified
        if not position.stop_loss:
            if position.side == "LONG":
                position.stop_loss = market_price * (1 - self.risk_params.stop_loss_percentage)
            else:
                position.stop_loss = market_price * (1 + self.risk_params.stop_loss_percentage)
            
            validation_result['warnings'].append(
                f"Automatic stop loss set at {position.stop_loss:.6f}"
            )
            validation_result['adjusted_position'] = position
        
        return validation_result
    
    async def calculate_position_size(self, symbol: str, risk_percentage: float = None) -> float:
        """Calculate optimal position size based on risk parameters"""
        if risk_percentage is None:
            risk_percentage = self.risk_params.max_position_size
        
        balance = await self.trading_client.get_account_balance()
        market_price = await self.trading_client.get_market_price(symbol)
        
        # Calculate position size based on available balance and risk percentage
        risk_amount = balance['available_margin'] * risk_percentage
        position_size = risk_amount / market_price
        
        return position_size
    
    async def trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop to halt all trading"""
        self.emergency_stop_triggered = True
        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
        
        # Close all positions
        positions = await self.trading_client.get_positions()
        for position in positions:
            try:
                await self.trading_client.close_position(position['symbol'], position['side'])
                logger.info(f"Emergency closure of position: {position['symbol']}")
            except Exception as e:
                logger.error(f"Error closing position during emergency stop: {e}")
        
        return {
            'emergency_stop': True,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }

class BingXIntegrationManager:
    """Main integration manager combining all BingX trading functionality"""
    
    def __init__(self):
        self.trading_client = None
        self.risk_manager = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the BingX integration system"""
        if self._initialized:
            return
        
        try:
            self.trading_client = BingXTradingClient()
            self.risk_manager = RiskManager(self.trading_client)
            
            # Test API connectivity
            await self.trading_client.get_account_balance()
            logger.info("BingX API connectivity verified")
            
            # Initialize risk management
            await self.risk_manager.initialize_session()
            
            self._initialized = True
            logger.info("BingX Integration Manager initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing BingX Integration Manager: {e}")
            raise
    
    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """Convert symbol format from BTCUSDT to BTC-USDT for BingX API"""
        if '-' in symbol:
            return symbol  # Already in correct format
        
        # Handle common patterns
        if symbol.endswith('USDT'):
            base = symbol[:-4]  # Remove 'USDT'
            return f"{base}-USDT"
        elif symbol.endswith('USDC'):
            base = symbol[:-4]  # Remove 'USDC'
            return f"{base}-USDC"
        elif symbol.endswith('BTC'):
            base = symbol[:-3]  # Remove 'BTC'
            return f"{base}-BTC"
        elif symbol.endswith('ETH'):
            base = symbol[:-3]  # Remove 'ETH'
            return f"{base}-ETH"
        
        # Default case - assume USDT if no recognized suffix
        return f"{symbol}-USDT"

    async def execute_ia2_trade(self, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade based on IA2 decision data"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Extract trade parameters from IA2 decision
            symbol = decision_data.get('symbol')
            signal = decision_data.get('signal')
            confidence = decision_data.get('confidence', 0)
            position_size = decision_data.get('position_size', 0)
            
            # Normalize symbol format for BingX (BTCUSDT -> BTC-USDT)
            bingx_symbol = self.normalize_symbol(symbol)
            
            # Skip if HOLD signal or zero position size
            if signal == 'HOLD' or position_size == 0:
                return {
                    'status': 'skipped',
                    'reason': 'HOLD signal or zero position size',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Calculate position parameters
            balance = await self.trading_client.get_account_balance()
            market_price = await self.trading_client.get_market_price(bingx_symbol)
            
            # Calculate quantity based on position size percentage
            risk_amount = balance['balance'] * (position_size / 100)
            leverage = min(decision_data.get('leverage', 5), 10)  # Cap at 10x
            quantity = risk_amount / market_price
            
            # Set stop loss and take profit
            if signal == 'LONG':
                stop_loss = market_price * 0.95  # 5% stop loss for LONG
                take_profit = market_price * 1.10  # 10% take profit for LONG
            else:  # SHORT
                stop_loss = market_price * 1.05  # 5% stop loss for SHORT
                take_profit = market_price * 0.90  # 10% take profit for SHORT
            
            # Create trading position
            position = TradingPosition(
                symbol=bingx_symbol,  # Use normalized symbol
                side=signal,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=leverage
            )
            
            # Validate position with risk management
            validation = await self.risk_manager.validate_position(position)
            
            if not validation['valid']:
                return {
                    'status': 'rejected',
                    'errors': validation['errors'],
                    'timestamp': datetime.now().isoformat()
                }
            
            # Use adjusted position if available
            final_position = validation['adjusted_position'] or position
            
            # Execute the trade
            order_result = await self.trading_client.place_market_order(final_position)
            
            return {
                'status': 'executed',
                'order_id': order_result.get('orderId'),
                'symbol': symbol,
                'side': signal,
                'quantity': final_position.quantity,
                'leverage': leverage,
                'risk_validation': validation,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error executing IA2 trade: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            if not self._initialized:
                return {
                    'status': 'not_initialized',
                    'timestamp': datetime.now().isoformat()
                }
            
            balance = await self.trading_client.get_account_balance()
            positions = await self.trading_client.get_positions()
            
            return {
                'status': 'operational',
                'api_connected': True,
                'balance': balance,
                'active_positions': len(positions),
                'positions': positions,
                'emergency_stop': self.risk_manager.emergency_stop_triggered,
                'session_pnl': balance['balance'] - self.risk_manager.session_start_balance,
                'pending_orders': len(self.trading_client.pending_orders),
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def close_all_positions(self) -> Dict[str, Any]:
        """Close all open positions (emergency function)"""
        try:
            if not self._initialized:
                await self.initialize()
            
            positions = await self.trading_client.get_positions()
            closed_positions = []
            
            for position in positions:
                try:
                    result = await self.trading_client.close_position(
                        position['symbol'], 
                        position['side']
                    )
                    closed_positions.append({
                        'symbol': position['symbol'],
                        'side': position['side'],
                        'size': position['size'],
                        'order_id': result.get('orderId')
                    })
                except Exception as e:
                    logger.error(f"Error closing position {position['symbol']}: {e}")
            
            return {
                'status': 'completed',
                'closed_positions': closed_positions,
                'total_closed': len(closed_positions),
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Global instance
bingx_manager = BingXIntegrationManager()

# Export main classes and functions
__all__ = [
    'BingXIntegrationManager',
    'BingXTradingClient', 
    'RiskManager',
    'TradingPosition',
    'bingx_manager'
]
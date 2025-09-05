import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import asyncio
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Import the official BingX library
from bingx_py.asyncio import BingXAsyncClient

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

class BingXPositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"

@dataclass
class BingXBalance:
    asset: str
    balance: float
    available: float
    locked: float

@dataclass
class BingXOrder:
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    status: str
    executed_qty: float
    avg_price: Optional[float]
    client_order_id: Optional[str]
    time_in_force: Optional[str]
    create_time: int
    update_time: int

class BingXOfficialTradingEngine:
    def __init__(self):
        self.api_key = os.environ.get('BINGX_API_KEY')
        self.secret_key = os.environ.get('BINGX_SECRET_KEY')
        self.base_url = os.environ.get('BINGX_BASE_URL', 'https://open-api.bingx.com')
        
        if not self.api_key or not self.secret_key:
            logger.error("BingX API credentials not found in environment variables")
            raise ValueError("BingX API credentials are required")
        
        # Initialize the official BingX client
        self.client = BingXAsyncClient(
            api_key=self.api_key,
            api_secret=self.secret_key,
            demo_trading=False  # Set to True for demo trading
        )
        
        logger.info("BingX Official Trading Engine initialized")

    async def get_account_balance(self) -> List[BingXBalance]:
        """Get account balance using official BingX API"""
        try:
            logger.info("Fetching account balance from BingX (futures account)")
            
            # Use SwapPerpetualAPI for futures trading balance
            balance_response = await self.client.swap.account.get_balance()
            
            balances = []
            if balance_response and hasattr(balance_response, 'data') and balance_response.data:
                for balance_item in balance_response.data:
                    if hasattr(balance_item, 'asset'):
                        balances.append(BingXBalance(
                            asset=balance_item.asset,
                            balance=float(getattr(balance_item, 'balance', 0)),
                            available=float(getattr(balance_item, 'availableMargin', 0)),
                            locked=float(getattr(balance_item, 'frozenMargin', 0))
                        ))
                        logger.info(f"Balance found: {balance_item.asset} = {getattr(balance_item, 'balance', 0)}")
            else:
                logger.warning("No balance data received from BingX API")
            
            return balances
            
        except Exception as e:
            logger.error(f"Failed to get account balance from BingX: {e}")
            # Try spot account as fallback
            try:
                logger.info("Trying spot account balance as fallback")
                spot_balance = await self.client.spot.account.query_assets()
                
                balances = []
                if spot_balance and hasattr(spot_balance, 'data') and spot_balance.data:
                    for balance_item in spot_balance.data:
                        if hasattr(balance_item, 'coin'):
                            balances.append(BingXBalance(
                                asset=balance_item.coin,
                                balance=float(getattr(balance_item, 'free', 0)) + float(getattr(balance_item, 'locked', 0)),
                                available=float(getattr(balance_item, 'free', 0)),
                                locked=float(getattr(balance_item, 'locked', 0))
                            ))
                            logger.info(f"Spot balance found: {balance_item.coin} = {getattr(balance_item, 'free', 0)}")
                
                return balances
                
            except Exception as spot_error:
                logger.error(f"Spot balance fallback also failed: {spot_error}")
                return []

    async def place_order(self, symbol: str, side: BingXOrderSide, order_type: BingXOrderType, 
                         quantity: float, price: Optional[float] = None, 
                         position_side: BingXPositionSide = BingXPositionSide.BOTH) -> Optional[BingXOrder]:
        """Place an order using official BingX API"""
        try:
            logger.info(f"Placing {side} {order_type} order for {quantity} {symbol}")
            
            # Use SwapPerpetualAPI for futures trading
            from bingx_py.models.swap.trades import OrderRequest
            from bingx_py.models.general import OrderSide, TimeInForce
            
            # Convert our enums to BingX library enums
            bingx_side = OrderSide.BUY if side == BingXOrderSide.BUY else OrderSide.SELL
            
            order_request = OrderRequest(
                symbol=symbol,
                side=bingx_side,
                position_side=position_side.value,
                order_type=order_type.value,
                quantity=quantity,
                price=price if order_type == BingXOrderType.LIMIT else None,
                time_in_force=TimeInForce.GTC if order_type == BingXOrderType.LIMIT else None
            )
            
            order_response = await self.client.swap.trades.place_order(order_request)
            
            if order_response and hasattr(order_response, 'data'):
                order_data = order_response.data
                return BingXOrder(
                    order_id=str(order_data.orderId),
                    symbol=symbol,
                    side=side.value,
                    order_type=order_type.value,
                    quantity=quantity,
                    price=price,
                    status=getattr(order_data, 'status', 'NEW'),
                    executed_qty=float(getattr(order_data, 'executedQty', 0)),
                    avg_price=float(getattr(order_data, 'avgPrice', 0)) if getattr(order_data, 'avgPrice', 0) else None,
                    client_order_id=getattr(order_data, 'clientOrderId', None),
                    time_in_force=getattr(order_data, 'timeInForce', None),
                    create_time=int(getattr(order_data, 'time', datetime.now().timestamp() * 1000)),
                    update_time=int(getattr(order_data, 'updateTime', datetime.now().timestamp() * 1000))
                )
            else:
                logger.error("No order data received from BingX API")
                return None
                
        except Exception as e:
            logger.error(f"Failed to place order on BingX: {e}")
            return None

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol"""
        try:
            logger.info(f"Setting leverage for {symbol} to {leverage}x")
            
            leverage_response = await self.client.swap.trades.set_leverage(
                symbol=symbol,
                leverage=leverage
            )
            
            return leverage_response is not None
            
        except Exception as e:
            logger.error(f"Failed to set leverage for {symbol}: {e}")
            return False

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        try:
            logger.info("Fetching current positions from BingX")
            
            positions_response = await self.client.swap.account.get_position()
            
            positions = []
            if positions_response and hasattr(positions_response, 'data'):
                for position in positions_response.data:
                    positions.append({
                        'symbol': getattr(position, 'symbol', ''),
                        'position_side': getattr(position, 'positionSide', ''),
                        'position_amount': float(getattr(position, 'positionAmt', 0)),
                        'entry_price': float(getattr(position, 'entryPrice', 0)),
                        'mark_price': float(getattr(position, 'markPrice', 0)),
                        'pnl': float(getattr(position, 'unRealizedProfit', 0)),
                        'pnl_percentage': float(getattr(position, 'percentage', 0)),
                        'margin': float(getattr(position, 'isolatedMargin', 0)),
                        'leverage': int(getattr(position, 'leverage', 1))
                    })
            
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions from BingX: {e}")
            return []

    async def test_connectivity(self) -> bool:
        """Test API connectivity"""
        try:
            logger.info("Testing BingX API connectivity")
            
            # Test by querying API key permissions
            permissions_response = await self.client.query_api_key_permissions()
            
            if permissions_response:
                logger.info("BingX API connectivity test successful")
                return True
            else:
                logger.error("BingX API connectivity test failed - no response")
                return False
                
        except Exception as e:
            logger.error(f"BingX API connectivity test failed: {e}")
            return False

    async def close(self):
        """Close the client connection"""
        try:
            if hasattr(self.client, 'close'):
                await self.client.close()
            logger.info("BingX client connection closed")
        except Exception as e:
            logger.error(f"Error closing BingX client: {e}")

# Create global instance
bingx_official_engine = BingXOfficialTradingEngine()
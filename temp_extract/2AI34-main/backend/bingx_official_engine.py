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
        """Get account balance from BingX FUTURES account - FIXED PARSING"""
        try:
            logger.info("Fetching account balance from BingX FUTURES account with CORRECT parsing")
            
            # Use async context manager for proper session handling
            async with self.client as client:
                # Get FUTURES account data - the API call works, we just need to parse it correctly
                balance_response = await client.swap.query_account_data()
                
                balances = []
                if balance_response and hasattr(balance_response, 'data'):
                    data = balance_response.data
                    logger.info(f"Raw account data received: {data}")
                    
                    # The data is a LIST of AccountData objects, not a single object!
                    if isinstance(data, list):
                        for account_item in data:
                            logger.info(f"Processing account item: {account_item}")
                            
                            # Extract fields from AccountData object
                            asset = getattr(account_item, 'asset', 'UNKNOWN')
                            balance = float(getattr(account_item, 'balance', 0))
                            equity = float(getattr(account_item, 'equity', 0))
                            available_margin = float(getattr(account_item, 'available_margin', 0))
                            used_margin = float(getattr(account_item, 'used_margin', 0))
                            unrealized_profit = float(getattr(account_item, 'unrealized_profit', 0))
                            
                            # Only include assets with non-zero balance
                            if balance > 0:
                                balances.append(BingXBalance(
                                    asset=asset,
                                    balance=balance,
                                    available=available_margin,
                                    locked=used_margin
                                ))
                                logger.info(f"💰 FOUND {asset}: Balance=${balance}, Available=${available_margin}, Used=${used_margin}")
                    
                    elif hasattr(data, 'balance'):
                        # Single AccountData object
                        asset = getattr(data, 'asset', 'USDT')
                        balance = float(getattr(data, 'balance', 0))
                        available_margin = float(getattr(data, 'available_margin', 0))
                        used_margin = float(getattr(data, 'used_margin', 0))
                        
                        if balance > 0:
                            balances.append(BingXBalance(
                                asset=asset,
                                balance=balance,
                                available=available_margin,
                                locked=used_margin
                            ))
                            logger.info(f"💰 SINGLE ACCOUNT FOUND {asset}: Balance=${balance}")
                
                total_balance = sum(b.balance for b in balances)
                logger.info(f"✅ TOTAL FUTURES BALANCE FOUND: ${total_balance} across {len(balances)} assets")
                
                return balances
            
        except Exception as e:
            logger.error(f"Failed to get FUTURES account balance: {e}")
            
            # Still try spot as fallback
            try:
                logger.info("Trying spot account balance as fallback")
                async with self.client as client:
                    spot_balance = await client.spot.query_assets()
                    
                    balances = []
                    if spot_balance and hasattr(spot_balance, 'data'):
                        for balance_item in spot_balance.data:
                            if hasattr(balance_item, 'coin'):
                                total_balance = float(getattr(balance_item, 'free', 0)) + float(getattr(balance_item, 'locked', 0))
                                if total_balance > 0:
                                    balances.append(BingXBalance(
                                        asset=balance_item.coin,
                                        balance=total_balance,
                                        available=float(getattr(balance_item, 'free', 0)),
                                        locked=float(getattr(balance_item, 'locked', 0))
                                    ))
                                    logger.info(f"Spot fallback balance: {balance_item.coin} = {total_balance}")
                    
                    return balances
                    
            except Exception as spot_error:
                logger.error(f"Spot balance fallback also failed: {spot_error}")
                
                # Final fallback - return empty instead of mock
                logger.warning("All balance methods failed - check API configuration")
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
"""
Active Position Manager - Real-time trading execution and position monitoring
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import uuid

from bingx_trading_engine import BingXTradingEngine, BingXOrderSide, BingXOrderType, BingXPositionSide
from bingx_official_engine import BingXOfficialTradingEngine

logger = logging.getLogger(__name__)

class PositionStatus(str, Enum):
    OPENING = "OPENING"
    ACTIVE = "ACTIVE"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    ERROR = "ERROR"

class TradeExecutionMode(str, Enum):
    LIVE = "LIVE"
    SIMULATION = "SIMULATION"

@dataclass
class ActivePosition:
    """Represents an active trading position with dynamic management"""
    id: str
    symbol: str
    signal: str  # LONG/SHORT
    entry_price: float
    quantity: float
    current_price: float
    
    # Position sizing
    account_balance: float
    risk_percentage: float  # 2% default
    position_size_usd: float
    leverage: float
    
    # Take Profit Strategy (Probabilistic)
    tp_levels: List[Dict[str, Any]]  # [{"level": 1, "price": 1.234, "percentage": 35, "filled": False}]
    tp_total_levels: int
    
    # Stop Loss & Trailing
    initial_stop_loss: float
    current_stop_loss: float
    
    # Fields with defaults
    tp_filled_levels: int = 0
    trailing_sl_active: bool = False
    trailing_sl_percentage: float = 3.0  # Default 3%
    tp1_activated: bool = False
    
    # P&L Tracking
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    pnl_percentage: float = 0.0
    
    # Status & Timing
    status: PositionStatus = PositionStatus.OPENING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # BingX Integration
    bingx_order_id: Optional[str] = None
    bingx_position_id: Optional[str] = None
    tp_order_ids: List[str] = field(default_factory=list)
    sl_order_id: Optional[str] = None
    
    # Risk Management
    max_loss_usd: float = 0.0
    risk_reward_ratio: float = 0.0

@dataclass
class TradeExecutionResult:
    """Result of trade execution attempt"""
    success: bool
    position_id: Optional[str] = None
    error_message: Optional[str] = None
    bingx_order_id: Optional[str] = None
    execution_details: Dict[str, Any] = field(default_factory=dict)

class ActivePositionManager:
    """Manages active trading positions with real-time monitoring and dynamic trailing stops"""
    
    def __init__(self, execution_mode: TradeExecutionMode = TradeExecutionMode.SIMULATION):
        self.execution_mode = execution_mode
        self.active_positions: Dict[str, ActivePosition] = {}
        
        # BingX Integration
        self.bingx_engine = BingXTradingEngine() if execution_mode == TradeExecutionMode.LIVE else None
        self.bingx_official = BingXOfficialTradingEngine() if execution_mode == TradeExecutionMode.LIVE else None
        
        # Risk Management Configuration
        self.max_positions = 5
        self.max_account_risk = 0.10  # 10% max account risk
        self.default_leverage = 3.0
        
        # Monitoring - CPU optimized (increased from 5s to 15s)
        self.monitoring_active = False
        self.update_interval = 15.0  # 15 seconds (CPU optimized)
        
        logger.info(f"Active Position Manager initialized in {execution_mode} mode")
    
    async def execute_trade_from_ia2_decision(self, decision_data: Dict[str, Any]) -> TradeExecutionResult:
        """Execute trade based on IA2 decision with probabilistic TP system"""
        try:
            logger.info(f"🚀 Executing trade from IA2 decision: {decision_data.get('symbol')} {decision_data.get('signal')}")
            
            # Extract decision details
            symbol = decision_data.get('symbol')
            signal = decision_data.get('signal', '').upper()
            entry_price = float(decision_data.get('entry_price', 0))
            confidence = float(decision_data.get('confidence', 0))
            ia2_position_size_percentage = float(decision_data.get('position_size_percentage', 0))
            
            # Skip HOLD signals
            if signal == 'HOLD':
                logger.info(f"📝 Skipping HOLD signal for {symbol}")
                return TradeExecutionResult(success=True, error_message="HOLD signal - no trade executed")
            
            # Skip if IA2 determined 0% position size
            if ia2_position_size_percentage <= 0:
                logger.info(f"⏭️ Skipping execution for {symbol}: IA2 determined 0% position size")
                return TradeExecutionResult(success=True, error_message="IA2 position size 0% - no trade executed")
            
            # Extract probabilistic TP strategy
            take_profit_strategy = decision_data.get('take_profit_strategy', {})
            tp_levels = take_profit_strategy.get('tp_levels', [])
            
            if not tp_levels:
                logger.warning(f"⚠️ No probabilistic TP levels found for {symbol}, using legacy format")
                # Fallback to legacy TP levels
                tp_levels = self._convert_legacy_tp_format(decision_data)
            
            # Get account balance for position sizing
            account_balance = await self._get_account_balance()
            
            # Use IA2's exact position size calculation
            stop_loss_price = float(decision_data.get('stop_loss', entry_price * 0.95))
            position_size_usd = account_balance * ia2_position_size_percentage
            quantity = position_size_usd / entry_price
            
            logger.info(f"💰 Using IA2 position sizing: {ia2_position_size_percentage:.1%} = ${position_size_usd:.2f} (Quantity: {quantity:.6f})")
            
            # Create active position
            position = ActivePosition(
                id=str(uuid.uuid4()),
                symbol=symbol,
                signal=signal,
                entry_price=entry_price,
                quantity=quantity,
                current_price=entry_price,
                account_balance=account_balance,
                risk_percentage=ia2_position_size_percentage * 100,  # Convert to percentage for display
                position_size_usd=position_size_usd,
                leverage=self.default_leverage,
                tp_levels=self._format_tp_levels(tp_levels, entry_price, signal),
                tp_total_levels=len(tp_levels),
                initial_stop_loss=stop_loss_price,
                current_stop_loss=stop_loss_price,
                max_loss_usd=abs(entry_price - stop_loss_price) * quantity,  # Actual risk amount
                risk_reward_ratio=float(decision_data.get('risk_reward_ratio', 1.5))
            )
            
            # Execute trade
            if self.execution_mode == TradeExecutionMode.LIVE:
                execution_result = await self._execute_live_trade(position)
            else:
                execution_result = await self._execute_simulation_trade(position)
            
            if execution_result.success:
                # Add to active positions
                self.active_positions[position.id] = position
                logger.info(f"✅ Trade executed successfully: {symbol} {signal} - Position ID: {position.id} (Size: {ia2_position_size_percentage:.1%})")
                
                # Start monitoring if not already active  
                if not self.monitoring_active:
                    asyncio.create_task(self._start_position_monitoring())
                
                return TradeExecutionResult(
                    success=True,
                    position_id=position.id,
                    bingx_order_id=execution_result.bingx_order_id,
                    execution_details={
                        'symbol': symbol,
                        'signal': signal,
                        'quantity': position.quantity,
                        'position_size_usd': position.position_size_usd,
                        'position_size_percentage': ia2_position_size_percentage,
                        'tp_levels': len(position.tp_levels),
                        'leverage': position.leverage
                    }
                )
            else:
                logger.error(f"❌ Trade execution failed for {symbol}: {execution_result.error_message}")
                return execution_result
            
        except Exception as e:
            logger.error(f"❌ Error executing trade from IA2 decision: {e}")
            return TradeExecutionResult(
                success=False,
                error_message=f"Trade execution error: {str(e)}"
            )
    
    def _format_tp_levels(self, tp_levels: List[Dict], entry_price: float, signal: str) -> List[Dict[str, Any]]:
        """Format TP levels for position management"""
        formatted_levels = []
        
        for tp_level in tp_levels:
            level_num = tp_level.get('level', 1)
            percentage_from_entry = tp_level.get('percentage_from_entry', 0)
            position_distribution = tp_level.get('position_distribution', 25)
            probability_reasoning = tp_level.get('probability_reasoning', '')
            
            # Calculate TP price
            if signal == 'LONG':
                tp_price = entry_price * (1 + percentage_from_entry / 100)
            else:  # SHORT
                tp_price = entry_price * (1 - percentage_from_entry / 100)
            
            formatted_levels.append({
                'level': level_num,
                'price': round(tp_price, 6),
                'percentage_from_entry': percentage_from_entry,
                'position_distribution': position_distribution,
                'probability_reasoning': probability_reasoning,
                'filled': False,
                'order_id': None,
                'filled_at': None
            })
        
        return formatted_levels
    
    def _convert_legacy_tp_format(self, decision_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert legacy TP format to probabilistic format"""
        legacy_levels = [
            {'level': 1, 'percentage_from_entry': 1.5, 'position_distribution': 35, 'probability_reasoning': 'Conservative first target'},
            {'level': 2, 'percentage_from_entry': 3.0, 'position_distribution': 40, 'probability_reasoning': 'Primary profit target'},
            {'level': 3, 'percentage_from_entry': 5.0, 'position_distribution': 25, 'probability_reasoning': 'Extended profit target'}
        ]
        return legacy_levels
    
    async def _execute_live_trade(self, position: ActivePosition) -> TradeExecutionResult:
        """Execute live trade on BingX"""
        try:
            logger.info(f"🔴 LIVE TRADING: Executing {position.signal} for {position.symbol}")
            
            # Set leverage
            await self.bingx_engine.set_leverage(position.symbol, int(position.leverage))
            
            # Determine order side and position side
            if position.signal == 'LONG':
                side = BingXOrderSide.BUY
                position_side = BingXPositionSide.LONG
            else:
                side = BingXOrderSide.SELL
                position_side = BingXPositionSide.SHORT
            
            # Place market entry order
            order = await self.bingx_engine.place_order(
                symbol=position.symbol,
                side=side,
                order_type=BingXOrderType.MARKET,
                quantity=position.quantity,
                position_side=position_side
            )
            
            if order:
                position.bingx_order_id = order.order_id
                position.status = PositionStatus.ACTIVE
                
                # Place initial stop-loss order
                await self._place_stop_loss_order(position)
                
                # Place TP orders based on probabilistic strategy
                await self._place_probabilistic_tp_orders(position)
                
                logger.info(f"✅ Live trade executed: {position.symbol} - Order ID: {order.order_id}")
                return TradeExecutionResult(
                    success=True,
                    bingx_order_id=order.order_id,
                    execution_details={'live_trading': True}
                )
            else:
                return TradeExecutionResult(
                    success=False,
                    error_message="Failed to place entry order on BingX"
                )
            
        except Exception as e:
            logger.error(f"❌ Live trade execution error: {e}")
            return TradeExecutionResult(
                success=False,
                error_message=f"Live execution error: {str(e)}"
            )
    
    async def _execute_simulation_trade(self, position: ActivePosition) -> TradeExecutionResult:
        """Execute simulated trade for testing"""
        logger.info(f"🟡 SIMULATION: Executing {position.signal} for {position.symbol}")
        
        # Simulate successful execution
        position.bingx_order_id = f"SIM_{uuid.uuid4().hex[:8]}"
        position.status = PositionStatus.ACTIVE
        
        return TradeExecutionResult(
            success=True,
            bingx_order_id=position.bingx_order_id,
            execution_details={'simulation': True}
        )
    
    async def _place_probabilistic_tp_orders(self, position: ActivePosition):
        """Place multiple TP orders based on probabilistic distribution"""
        try:
            for tp_level in position.tp_levels:
                # Calculate quantity for this TP level
                tp_quantity = position.quantity * (tp_level['position_distribution'] / 100)
                
                # Determine order side (opposite of position)
                if position.signal == 'LONG':
                    side = BingXOrderSide.SELL
                    position_side = BingXPositionSide.LONG
                else:
                    side = BingXOrderSide.BUY
                    position_side = BingXPositionSide.SHORT
                
                if self.execution_mode == TradeExecutionMode.LIVE:
                    # Place TP limit order
                    tp_order = await self.bingx_engine.place_order(
                        symbol=position.symbol,
                        side=side,
                        order_type=BingXOrderType.TAKE_PROFIT_LIMIT,
                        quantity=tp_quantity,
                        price=tp_level['price'],
                        position_side=position_side
                    )
                    
                    if tp_order:
                        tp_level['order_id'] = tp_order.order_id
                        position.tp_order_ids.append(tp_order.order_id)
                        logger.info(f"📈 TP{tp_level['level']} order placed: {tp_quantity} @ ${tp_level['price']:.6f}")
                else:
                    # Simulation mode
                    tp_level['order_id'] = f"TP_SIM_{uuid.uuid4().hex[:6]}"
                    logger.info(f"📈 TP{tp_level['level']} simulated: {tp_quantity} @ ${tp_level['price']:.6f}")
            
        except Exception as e:
            logger.error(f"❌ Error placing TP orders: {e}")
    
    async def _place_stop_loss_order(self, position: ActivePosition):
        """Place initial stop-loss order"""
        try:
            # Determine order side (opposite of position)
            if position.signal == 'LONG':
                side = BingXOrderSide.SELL
                position_side = BingXPositionSide.LONG
            else:
                side = BingXOrderSide.BUY
                position_side = BingXPositionSide.SHORT
            
            if self.execution_mode == TradeExecutionMode.LIVE:
                sl_order = await self.bingx_engine.place_order(
                    symbol=position.symbol,
                    side=side,
                    order_type=BingXOrderType.STOP_MARKET,
                    quantity=position.quantity,
                    stop_price=position.current_stop_loss,
                    position_side=position_side
                )
                
                if sl_order:
                    position.sl_order_id = sl_order.order_id
                    logger.info(f"🛡️ Stop-loss placed: ${position.current_stop_loss:.6f}")
            else:
                # Simulation mode
                position.sl_order_id = f"SL_SIM_{uuid.uuid4().hex[:6]}"
                logger.info(f"🛡️ Stop-loss simulated: ${position.current_stop_loss:.6f}")
                
        except Exception as e:
            logger.error(f"❌ Error placing stop-loss: {e}")
    
    async def _get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            if self.execution_mode == TradeExecutionMode.LIVE and self.bingx_engine:
                balances = await self.bingx_engine.get_account_balance()
                usdt_balance = next((b for b in balances if b.asset == 'USDT'), None)
                return usdt_balance.available if usdt_balance else 250.0
            else:
                # Simulation balance
                return 250.0
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return 250.0  # Fallback balance
    
    async def _start_position_monitoring(self):
        """Start real-time position monitoring"""
        self.monitoring_active = True
        logger.info("🔄 Starting active position monitoring")
        
        while self.monitoring_active and self.active_positions:
            try:
                await self._update_all_positions()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"❌ Position monitoring error: {e}")
                await asyncio.sleep(self.update_interval)
        
        self.monitoring_active = False
        logger.info("⏹️ Position monitoring stopped")
    
    async def _update_all_positions(self):
        """Update all active positions with current market prices"""
        for position_id, position in list(self.active_positions.items()):
            try:
                # Get current price (simulation for now)
                current_price = await self._get_current_price(position.symbol)
                position.current_price = current_price
                position.updated_at = datetime.now(timezone.utc)
                
                # Update P&L
                self._calculate_pnl(position)
                
                # Check TP levels
                await self._check_tp_levels(position)
                
                # Handle trailing stop logic
                await self._update_trailing_stop(position)
                
                # Check if position should be closed
                if position.status == PositionStatus.CLOSING:
                    await self._close_position(position)
                
            except Exception as e:
                logger.error(f"❌ Error updating position {position.symbol}: {e}")
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price (simulation for now)"""
        # For simulation, return a slightly varied price
        # In live mode, this would fetch from BingX API
        import random
        base_price = 1.0  # This would be fetched from market
        variation = random.uniform(-0.02, 0.02)  # ±2% random variation
        return base_price * (1 + variation)
    
    def _calculate_pnl(self, position: ActivePosition):
        """Calculate current P&L for position"""
        if position.signal == 'LONG':
            pnl = (position.current_price - position.entry_price) * position.quantity
        else:  # SHORT
            pnl = (position.entry_price - position.current_price) * position.quantity
        
        position.unrealized_pnl = pnl
        position.pnl_percentage = (pnl / position.position_size_usd) * 100 if position.position_size_usd > 0 else 0
    
    async def _check_tp_levels(self, position: ActivePosition):
        """Check if any TP levels have been hit"""
        for tp_level in position.tp_levels:
            if tp_level['filled']:
                continue
            
            hit = False
            if position.signal == 'LONG' and position.current_price >= tp_level['price']:
                hit = True
            elif position.signal == 'SHORT' and position.current_price <= tp_level['price']:
                hit = True
            
            if hit:
                tp_level['filled'] = True
                tp_level['filled_at'] = datetime.now(timezone.utc)
                position.tp_filled_levels += 1
                
                logger.info(f"🎯 TP{tp_level['level']} HIT for {position.symbol}: ${tp_level['price']:.6f}")
                
                # Activate trailing stop at TP1
                if tp_level['level'] == 1 and not position.tp1_activated:
                    position.tp1_activated = True
                    position.trailing_sl_active = True
                    logger.info(f"🚀 TP1 activated - Trailing SL now ACTIVE for {position.symbol}")
    
    async def _update_trailing_stop(self, position: ActivePosition):
        """Update trailing stop loss when active"""
        if not position.trailing_sl_active or not position.tp1_activated:
            return
        
        # Calculate new trailing SL based on current price
        trailing_distance = position.current_price * (position.trailing_sl_percentage / 100)
        
        if position.signal == 'LONG':
            new_sl = position.current_price - trailing_distance
            # Only move SL up (never down)
            if new_sl > position.current_stop_loss:
                old_sl = position.current_stop_loss
                position.current_stop_loss = new_sl
                logger.info(f"📈 Trailing SL updated for {position.symbol}: ${old_sl:.6f} → ${new_sl:.6f}")
                await self._update_stop_loss_order(position)
        else:  # SHORT
            new_sl = position.current_price + trailing_distance
            # Only move SL down (never up) for short positions
            if new_sl < position.current_stop_loss:
                old_sl = position.current_stop_loss
                position.current_stop_loss = new_sl
                logger.info(f"📉 Trailing SL updated for {position.symbol}: ${old_sl:.6f} → ${new_sl:.6f}")
                await self._update_stop_loss_order(position)
    
    async def _update_stop_loss_order(self, position: ActivePosition):
        """Update stop-loss order on exchange"""
        if self.execution_mode == TradeExecutionMode.LIVE:
            try:
                # Cancel existing SL order
                if position.sl_order_id:
                    await self.bingx_engine.cancel_order(position.symbol, position.sl_order_id)
                
                # Place new SL order
                await self._place_stop_loss_order(position)
                
            except Exception as e:
                logger.error(f"❌ Error updating stop-loss order: {e}")
    
    async def _close_position(self, position: ActivePosition):
        """Close position completely"""
        try:
            logger.info(f"🔒 Closing position: {position.symbol}")
            
            if self.execution_mode == TradeExecutionMode.LIVE:
                success = await self.bingx_engine.close_position(position.symbol)
                if success:
                    position.status = PositionStatus.CLOSED
                else:
                    logger.error(f"❌ Failed to close position: {position.symbol}")
            else:
                # Simulation close
                position.status = PositionStatus.CLOSED
            
            # Remove from active positions
            if position.id in self.active_positions:
                del self.active_positions[position.id]
                
            logger.info(f"✅ Position closed: {position.symbol} - Final P&L: ${position.unrealized_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error closing position {position.symbol}: {e}")
    
    def get_active_positions_summary(self) -> Dict[str, Any]:
        """Get summary of all active positions"""
        positions_data = []
        total_pnl = 0.0
        total_value = 0.0
        
        for position in self.active_positions.values():
            positions_data.append({
                'id': position.id,
                'symbol': position.symbol,
                'signal': position.signal,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'quantity': position.quantity,
                'position_size_usd': position.position_size_usd,
                'unrealized_pnl': position.unrealized_pnl,
                'pnl_percentage': position.pnl_percentage,
                'leverage': position.leverage,
                'tp_levels': position.tp_levels,
                'tp_filled_levels': position.tp_filled_levels,
                'tp_total_levels': position.tp_total_levels,
                'current_stop_loss': position.current_stop_loss,
                'trailing_sl_active': position.trailing_sl_active,
                'tp1_activated': position.tp1_activated,
                'status': position.status.value,
                'created_at': position.created_at.isoformat(),
                'updated_at': position.updated_at.isoformat()
            })
            
            total_pnl += position.unrealized_pnl
            total_value += position.position_size_usd
        
        return {
            'active_positions': positions_data,
            'total_positions': len(self.active_positions),
            'total_unrealized_pnl': round(total_pnl, 2),
            'total_position_value': round(total_value, 2),
            'execution_mode': self.execution_mode.value,
            'monitoring_active': self.monitoring_active
        }

# Global instance
active_position_manager = ActivePositionManager(execution_mode=TradeExecutionMode.SIMULATION)
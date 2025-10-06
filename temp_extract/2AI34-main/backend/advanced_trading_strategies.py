import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

from bingx_trading_engine import bingx_trading_engine, BingXOrderSide, BingXOrderType, BingXPositionSide

logger = logging.getLogger(__name__)

class PositionDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"

class TakeProfitStatus(str, Enum):
    PENDING = "PENDING"
    PARTIAL_FILLED = "PARTIAL_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"

@dataclass
class TakeProfitLevel:
    level: int
    price: float
    quantity_percentage: float  # Percentage of total position
    status: TakeProfitStatus
    order_id: Optional[str] = None
    filled_quantity: float = 0.0
    description: str = ""

@dataclass
class AdvancedTradingStrategy:
    symbol: str
    direction: PositionDirection
    entry_price: float
    total_quantity: float
    stop_loss: float
    take_profit_levels: List[TakeProfitLevel]
    confidence: float
    created_at: datetime
    updated_at: datetime
    ia1_analysis_id: str
    reasoning: str
    
    # Position tracking
    current_position_quantity: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Strategy status
    is_active: bool = True
    is_inverted: bool = False
    original_strategy_id: Optional[str] = None

class AdvancedTradingStrategyManager:
    """Gestionnaire de stratégies de trading avancées avec TP multiples et inversion automatique"""
    
    def __init__(self):
        self.active_strategies: Dict[str, AdvancedTradingStrategy] = {}
        self.position_inversion_threshold = 0.10  # 10% de confiance supérieure minimum pour inverser
        
    async def create_advanced_strategy(self, 
                                     symbol: str, 
                                     direction: PositionDirection,
                                     entry_price: float,
                                     quantity: float,
                                     confidence: float,
                                     ia1_analysis_id: str,
                                     reasoning: str) -> AdvancedTradingStrategy:
        """Créer une stratégie avancée avec TP échelonnés"""
        
        # Calculer stop loss et take profits multiples
        risk_reward_ratio = max(2.0, confidence * 3)  # Plus de confiance = meilleur R:R
        
        if direction == PositionDirection.LONG:
            # Pour LONG
            stop_loss = entry_price * (1 - 0.03)  # 3% stop loss
            tp_base = entry_price * (1 + 0.03 * risk_reward_ratio)
            
            take_profit_levels = [
                TakeProfitLevel(1, entry_price * 1.015, 25.0, TakeProfitStatus.PENDING, 
                               description="Premier TP - Sécuriser 25% des gains"),
                TakeProfitLevel(2, entry_price * 1.03, 30.0, TakeProfitStatus.PENDING,
                               description="Deuxième TP - Récupérer mise initiale"),
                TakeProfitLevel(3, entry_price * 1.05, 25.0, TakeProfitStatus.PENDING,
                               description="Troisième TP - Profits intermédiaires"),
                TakeProfitLevel(4, tp_base, 20.0, TakeProfitStatus.PENDING,
                               description="TP Final - Objectif maximum")
            ]
        else:  # SHORT
            stop_loss = entry_price * (1 + 0.03)  # 3% stop loss
            tp_base = entry_price * (1 - 0.03 * risk_reward_ratio)
            
            take_profit_levels = [
                TakeProfitLevel(1, entry_price * 0.985, 25.0, TakeProfitStatus.PENDING,
                               description="Premier TP - Sécuriser 25% des gains"),
                TakeProfitLevel(2, entry_price * 0.97, 30.0, TakeProfitStatus.PENDING,
                               description="Deuxième TP - Récupérer mise initiale"),
                TakeProfitLevel(3, entry_price * 0.95, 25.0, TakeProfitStatus.PENDING,
                               description="Troisième TP - Profits intermédiaires"),
                TakeProfitLevel(4, tp_base, 20.0, TakeProfitStatus.PENDING,
                               description="TP Final - Objectif maximum")
            ]
        
        strategy = AdvancedTradingStrategy(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            total_quantity=quantity,
            stop_loss=stop_loss,
            take_profit_levels=take_profit_levels,
            confidence=confidence,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            ia1_analysis_id=ia1_analysis_id,
            reasoning=reasoning,
            current_position_quantity=quantity
        )
        
        strategy_id = f"{symbol}_{direction}_{int(datetime.now().timestamp())}"
        self.active_strategies[strategy_id] = strategy
        
        logger.info(f"🎯 Stratégie avancée créée pour {symbol}: {len(take_profit_levels)} niveaux TP, confiance {confidence:.2%}")
        
        return strategy
    
    async def execute_strategy(self, strategy: AdvancedTradingStrategy) -> bool:
        """Exécuter la stratégie avec ordres BingX"""
        try:
            symbol = strategy.symbol
            logger.info(f"🚀 Exécution stratégie avancée pour {symbol}")
            
            # 1. Placer l'ordre d'entrée principal
            side = BingXOrderSide.BUY if strategy.direction == PositionDirection.LONG else BingXOrderSide.SELL
            position_side = BingXPositionSide.LONG if strategy.direction == PositionDirection.LONG else BingXPositionSide.SHORT
            
            entry_order = await bingx_trading_engine.place_order(
                symbol=symbol,
                side=side,
                order_type=BingXOrderType.MARKET,
                quantity=strategy.total_quantity,
                position_side=position_side
            )
            
            if not entry_order:
                logger.error(f"❌ Échec ordre d'entrée pour {symbol}")
                return False
            
            logger.info(f"✅ Ordre d'entrée placé pour {symbol}: {entry_order.order_id}")
            
            # 2. Placer les ordres Take Profit échelonnés
            for tp_level in strategy.take_profit_levels:
                await self._place_take_profit_order(strategy, tp_level)
                await asyncio.sleep(0.5)  # Éviter le rate limiting
            
            # 3. Placer l'ordre Stop Loss
            await self._place_stop_loss_order(strategy)
            
            strategy.updated_at = datetime.now(timezone.utc)
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur exécution stratégie pour {strategy.symbol}: {e}")
            return False
    
    async def _place_take_profit_order(self, strategy: AdvancedTradingStrategy, tp_level: TakeProfitLevel):
        """Placer un ordre take profit spécifique"""
        try:
            quantity = (strategy.total_quantity * tp_level.quantity_percentage) / 100
            
            # Côté opposé à l'entrée pour fermer la position
            side = BingXOrderSide.SELL if strategy.direction == PositionDirection.LONG else BingXOrderSide.BUY
            position_side = BingXPositionSide.LONG if strategy.direction == PositionDirection.LONG else BingXPositionSide.SHORT
            
            tp_order = await bingx_trading_engine.place_order(
                symbol=strategy.symbol,
                side=side,
                order_type=BingXOrderType.LIMIT,
                quantity=quantity,
                price=tp_level.price,
                position_side=position_side
            )
            
            if tp_order:
                tp_level.order_id = tp_order.order_id
                logger.info(f"✅ TP Niveau {tp_level.level} placé pour {strategy.symbol}: {tp_level.price:.6f} ({tp_level.quantity_percentage}%)")
            
        except Exception as e:
            logger.error(f"❌ Erreur TP niveau {tp_level.level} pour {strategy.symbol}: {e}")
    
    async def _place_stop_loss_order(self, strategy: AdvancedTradingStrategy):
        """Placer l'ordre stop loss"""
        try:
            side = BingXOrderSide.SELL if strategy.direction == PositionDirection.LONG else BingXOrderSide.BUY
            position_side = BingXPositionSide.LONG if strategy.direction == PositionDirection.LONG else BingXPositionSide.SHORT
            
            sl_order = await bingx_trading_engine.place_order(
                symbol=strategy.symbol,
                side=side,
                order_type=BingXOrderType.STOP_MARKET,
                quantity=strategy.current_position_quantity,
                price=strategy.stop_loss,
                position_side=position_side
            )
            
            if sl_order:
                logger.info(f"🛡️ Stop Loss placé pour {strategy.symbol}: {strategy.stop_loss:.6f}")
            
        except Exception as e:
            logger.error(f"❌ Erreur Stop Loss pour {strategy.symbol}: {e}")
    
    async def check_position_inversion_signal(self, 
                                            symbol: str, 
                                            new_direction: PositionDirection, 
                                            new_confidence: float,
                                            ia1_analysis_id: str,
                                            reasoning: str) -> bool:
        """Vérifier si on doit inverser une position existante"""
        
        current_strategy = None
        for strategy_id, strategy in self.active_strategies.items():
            if strategy.symbol == symbol and strategy.is_active:
                current_strategy = strategy
                break
        
        if not current_strategy:
            logger.debug(f"Aucune position active pour {symbol} - pas d'inversion nécessaire")
            return False
        
        # Vérifier si c'est un signal dans l'autre direction
        if current_strategy.direction == new_direction:
            logger.debug(f"Signal dans la même direction pour {symbol} - pas d'inversion")
            return False
        
        # Vérifier le seuil de confiance pour inversion
        confidence_delta = new_confidence - current_strategy.confidence
        if confidence_delta < self.position_inversion_threshold:
            logger.info(f"⚠️ Signal opposé pour {symbol} mais confiance insuffisante: "
                       f"{new_confidence:.2%} vs {current_strategy.confidence:.2%} "
                       f"(delta: {confidence_delta:.2%}, seuil: {self.position_inversion_threshold:.2%})")
            return False
        
        logger.info(f"🔄 INVERSION DE POSITION DÉCLENCHÉE pour {symbol}: "
                   f"{current_strategy.direction} -> {new_direction} "
                   f"(confiance: {current_strategy.confidence:.2%} -> {new_confidence:.2%})")
        
        # Exécuter l'inversion
        success = await self._execute_position_inversion(
            current_strategy, new_direction, new_confidence, ia1_analysis_id, reasoning
        )
        
        return success
    
    async def _execute_position_inversion(self, 
                                        current_strategy: AdvancedTradingStrategy,
                                        new_direction: PositionDirection,
                                        new_confidence: float,
                                        ia1_analysis_id: str,
                                        reasoning: str) -> bool:
        """Exécuter l'inversion de position"""
        try:
            symbol = current_strategy.symbol
            logger.info(f"🔄 Exécution inversion pour {symbol}")
            
            # 1. Fermer la position actuelle
            await self._close_current_position(current_strategy)
            
            # 2. Marquer la stratégie actuelle comme inactive
            current_strategy.is_active = False
            current_strategy.updated_at = datetime.now(timezone.utc)
            
            # 3. Créer une nouvelle stratégie dans l'autre direction
            current_price = await self._get_current_market_price(symbol)
            if not current_price:
                logger.error(f"❌ Impossible d'obtenir le prix pour {symbol}")
                return False
            
            new_strategy = await self.create_advanced_strategy(
                symbol=symbol,
                direction=new_direction,
                entry_price=current_price,
                quantity=current_strategy.total_quantity,  # Même quantité
                confidence=new_confidence,
                ia1_analysis_id=ia1_analysis_id,
                reasoning=f"INVERSION: {reasoning} (ancienne confiance: {current_strategy.confidence:.2%})"
            )
            
            new_strategy.is_inverted = True
            new_strategy.original_strategy_id = id(current_strategy)
            
            # 4. Exécuter la nouvelle stratégie
            success = await self.execute_strategy(new_strategy)
            
            if success:
                logger.info(f"✅ Inversion réussie pour {symbol}: {current_strategy.direction} -> {new_direction}")
            else:
                logger.error(f"❌ Échec inversion pour {symbol}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Erreur inversion position pour {symbol}: {e}")
            return False
    
    async def _close_current_position(self, strategy: AdvancedTradingStrategy):
        """Fermer la position actuelle (annuler TP et SL, fermer au marché)"""
        try:
            symbol = strategy.symbol
            
            # Annuler tous les ordres Take Profit en attente
            for tp_level in strategy.take_profit_levels:
                if tp_level.order_id and tp_level.status == TakeProfitStatus.PENDING:
                    await bingx_trading_engine.cancel_order(symbol, tp_level.order_id)
                    tp_level.status = TakeProfitStatus.CANCELLED
            
            # Fermer la position au marché
            side = BingXOrderSide.SELL if strategy.direction == PositionDirection.LONG else BingXOrderSide.BUY
            position_side = BingXPositionSide.LONG if strategy.direction == PositionDirection.LONG else BingXPositionSide.SHORT
            
            close_order = await bingx_trading_engine.place_order(
                symbol=symbol,
                side=side,
                order_type=BingXOrderType.MARKET,
                quantity=strategy.current_position_quantity,
                position_side=position_side
            )
            
            if close_order:
                logger.info(f"🔒 Position fermée pour {symbol}: {close_order.order_id}")
            
        except Exception as e:
            logger.error(f"❌ Erreur fermeture position pour {strategy.symbol}: {e}")
    
    async def _get_current_market_price(self, symbol: str) -> Optional[float]:
        """Obtenir le prix actuel du marché"""
        try:
            # Utiliser l'aggregator pour obtenir le prix actuel
            from advanced_market_aggregator import advanced_market_aggregator
            
            market_data = await advanced_market_aggregator.get_comprehensive_market_data(symbol)
            if market_data and market_data.price > 0:
                return market_data.price
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération prix pour {symbol}: {e}")
            return None
    
    async def update_strategy_performance(self):
        """Mettre à jour les performances des stratégies actives"""
        for strategy_id, strategy in self.active_strategies.items():
            if strategy.is_active:
                await self._update_single_strategy_performance(strategy)
    
    async def _update_single_strategy_performance(self, strategy: AdvancedTradingStrategy):
        """Mettre à jour les performances d'une stratégie"""
        try:
            # Obtenir les positions actuelles depuis BingX
            positions = await bingx_trading_engine.get_positions()
            
            symbol_position = None
            for position in positions:
                if position['symbol'] == strategy.symbol:
                    symbol_position = position
                    break
            
            if symbol_position:
                strategy.current_position_quantity = abs(float(symbol_position.get('position_amount', 0)))
                strategy.unrealized_pnl = float(symbol_position.get('pnl', 0))
            
            strategy.updated_at = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"❌ Erreur mise à jour performance pour {strategy.symbol}: {e}")
    
    def get_strategy_summary(self) -> Dict:
        """Obtenir un résumé des stratégies actives"""
        active_count = sum(1 for s in self.active_strategies.values() if s.is_active)
        inverted_count = sum(1 for s in self.active_strategies.values() if s.is_inverted)
        
        total_unrealized_pnl = sum(s.unrealized_pnl for s in self.active_strategies.values() if s.is_active)
        
        return {
            "total_strategies": len(self.active_strategies),
            "active_strategies": active_count,
            "inverted_strategies": inverted_count,
            "total_unrealized_pnl": total_unrealized_pnl,
            "active_symbols": [s.symbol for s in self.active_strategies.values() if s.is_active]
        }

# Instance globale
advanced_strategy_manager = AdvancedTradingStrategyManager()
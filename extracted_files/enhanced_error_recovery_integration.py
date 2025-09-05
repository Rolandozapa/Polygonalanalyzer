# enhanced_error_recovery_integration.py
"""
Integration of enhanced error recovery and resilience patterns into the trading system
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from error_recovery import (
    ErrorRecoveryManager, 
    ErrorSeverity, 
    SystemState,
    CircuitBreakerAdvanced,
    RetryManager,
    ResilientTradingComponent
)

class ResilientAIServiceClient(ResilientTradingComponent):
    """Resilient AI service client with built-in error recovery"""
    
    def __init__(self, service_url: str, service_name: str):
        super().__init__(f"ai_service_{service_name}")
        self.service_url = service_url
        self.service_name = service_name
    
    async def call_service(self, payload: Dict) -> Optional[Dict]:
        """Make a resilient call to the AI service"""
        
        async def _make_request():
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.service_url, 
                    json=payload, 
                    timeout=30
                ) as response:
                    response.raise_for_status()
                    return await response.json()
        
        try:
            return await self.resilient_call(
                func=_make_request,
                error_type="ai_service",
                severity=ErrorSeverity.HIGH,
                max_retries=3
            )
        except Exception as e:
            self.logger.error(f"AI service call failed after retries: {e}")
            return None

class ResilientBingXAPIClient(ResilientTradingComponent):
    """Resilient BingX API client with built-in error recovery"""
    
    def __init__(self, api_client):
        super().__init__("bingx_api")
        self.api_client = api_client
    
    async def fetch_open_positions(self) -> List[Dict]:
        """Resiliently fetch open positions"""
        try:
            return await self.resilient_call(
                func=self.api_client.fetch_open_positions,
                error_type="api_call",
                severity=ErrorSeverity.MEDIUM,
                max_retries=2
            )
        except Exception as e:
            self.logger.error(f"Failed to fetch open positions: {e}")
            return []
    
    async def place_order(self, **kwargs) -> Optional[Dict]:
        """Resiliently place an order"""
        try:
            return await self.resilient_call(
                func=lambda: self.api_client.place_order(**kwargs),
                error_type="order_execution",
                severity=ErrorSeverity.HIGH,
                max_retries=3
            )
        except Exception as e:
            self.logger.error(f"Order placement failed: {e}")
            return None

class EnhancedTradingOrchestratorWithRecovery:
    """Enhanced trading orchestrator with integrated error recovery"""
    
    def __init__(self, scouting_bot, bingx_api_client, trading_config):
        self.scouting_bot = scouting_bot
        self.bingx_api = ResilientBingXAPIClient(bingx_api_client)
        self.config = trading_config
        
        # Initialize error recovery manager
        self.error_manager = ErrorRecoveryManager()
        
        # Initialize resilient AI clients
        self.main_ai_client = ResilientAIServiceClient(
            self.config.ia_strategy_url, "main_strategy"
        )
        self.specialized_ai_client = ResilientAIServiceClient(
            self.config.ia_specialized_url, "specialized"
        )
        self.bingx_ai_client = ResilientAIServiceClient(
            self.config.bingx_ia_url, "bingx"
        )
        
        self.logger = logging.getLogger(__name__)
    
    async def run_complete_trading_cycle(self) -> List[Dict]:
        """Execute complete trading cycle with enhanced error recovery"""
        try:
            # 1. Scouting opportunities with error handling
            opportunities = await self._scan_opportunities_with_recovery()
            if not opportunities:
                return []
            
            # 2. AI consultations with error recovery
            recommendations = await self._process_opportunities_with_recovery(opportunities)
            if not recommendations:
                return []
            
            # 3. Execute trades with error recovery
            executed_trades = await self._execute_trades_with_recovery(recommendations)
            
            # 4. Monitor positions with error recovery
            await self._monitor_positions_with_recovery()
            
            return executed_trades
            
        except Exception as e:
            self.error_manager.record_error(
                component="orchestrator",
                error_type="trading_cycle",
                severity=ErrorSeverity.CRITICAL,
                message=f"Complete trading cycle failed: {e}",
                exception=e
            )
            return []
    
    async def _scan_opportunities_with_recovery(self) -> Dict:
        """Scan opportunities with error recovery"""
        try:
            # Use retry manager for scanning
            retry_manager = RetryManager()
            return await retry_manager.retry_with_backoff(
                func=self.scouting_bot.scan_and_filter_opportunities,
                max_attempts=3,
                base_delay=1.0,
                max_delay=10.0,
                backoff_factor=2.0,
                retry_on=(Exception,)
            )
        except Exception as e:
            self.error_manager.record_error(
                component="scouting_bot",
                error_type="opportunity_scanning",
                severity=ErrorSeverity.HIGH,
                message="Failed to scan opportunities",
                exception=e
            )
            return {}
    
    async def _process_opportunities_with_recovery(self, opportunities: Dict) -> List[Dict]:
        """Process opportunities with error recovery"""
        recommendations = []
        
        for symbol, data in opportunities.items():
            try:
                # Consult main AI with recovery
                main_rec = await self.main_ai_client.call_service({
                    "symbol": symbol,
                    "data": data
                })
                
                if not main_rec:
                    continue
                
                # Consult specialized AI with recovery
                specialized_advice = await self.specialized_ai_client.call_service({
                    "symbol": symbol,
                    "opportunity_data": data,
                    "main_recommendation": main_rec
                })
                
                if specialized_advice and specialized_advice.get('confirmed', False):
                    # Merge recommendations
                    final_rec = self._merge_recommendations(main_rec, specialized_advice)
                    recommendations.append(final_rec)
                    
            except Exception as e:
                self.error_manager.record_error(
                    component="ai_processing",
                    error_type="ai_consultation",
                    severity=ErrorSeverity.MEDIUM,
                    message=f"Failed to process opportunity for {symbol}",
                    exception=e,
                    context={"symbol": symbol}
                )
                continue
        
        return recommendations
    
    async def _execute_trades_with_recovery(self, recommendations: List[Dict]) -> List[Dict]:
        """Execute trades with error recovery"""
        executed_trades = []
        
        for recommendation in recommendations:
            try:
                # Validate recommendation
                if not self._validate_recommendation(recommendation):
                    continue
                
                # Calculate position size with risk management
                position_size = self._calculate_position_size(recommendation)
                
                # Execute trade with recovery
                trade_result = await self.bingx_api.place_order(
                    symbol=recommendation['symbol'],
                    side=recommendation['action'],
                    quantity=position_size
                )
                
                if trade_result:
                    executed_trades.append({
                        **trade_result,
                        "recommendation": recommendation
                    })
                    
            except Exception as e:
                self.error_manager.record_error(
                    component="trade_execution",
                    error_type="order_placement",
                    severity=ErrorSeverity.HIGH,
                    message=f"Failed to execute trade for {recommendation.get('symbol', 'unknown')}",
                    exception=e,
                    context={"recommendation": recommendation}
                )
                continue
        
        return executed_trades
    
    async def _monitor_positions_with_recovery(self):
        """Monitor positions with error recovery"""
        try:
            positions = await self.bingx_api.fetch_open_positions()
            
            for position in positions:
                try:
                    # Consult BingX AI for position management
                    management_advice = await self.bingx_ai_client.call_service({
                        "position": position,
                        "market_conditions": self._get_market_conditions(position['symbol'])
                    })
                    
                    if management_advice and management_advice.get('action_required', False):
                        await self._execute_position_management(position, management_advice)
                        
                except Exception as e:
                    self.error_manager.record_error(
                        component="position_management",
                        error_type="position_monitoring",
                        severity=ErrorSeverity.MEDIUM,
                        message=f"Failed to monitor position for {position['symbol']}",
                        exception=e,
                        context={"position": position}
                    )
                    continue
                    
        except Exception as e:
            self.error_manager.record_error(
                component="position_management",
                error_type="position_fetch",
                severity=ErrorSeverity.MEDIUM,
                message="Failed to fetch positions for monitoring",
                exception=e
            )
    
    async def _execute_position_management(self, position: Dict, advice: Dict):
        """Execute position management with error recovery"""
        try:
            symbol = position['symbol']
            
            if advice.get('adjust_stop_loss', False):
                await self.bingx_api.set_stop_loss(symbol, advice['new_stop_loss'])
            
            if advice.get('adjust_take_profit', False):
                await self.bingx_api.set_take_profit(symbol, advice['new_take_profit'])
            
            if advice.get('enable_trailing_stop', False):
                await self.bingx_api.set_trailing_stop(symbol, advice['trailing_params'])
            
            if advice.get('close_position', False):
                await self.bingx_api.close_position(symbol)
                
        except Exception as e:
            self.error_manager.record_error(
                component="position_management",
                error_type="position_adjustment",
                severity=ErrorSeverity.HIGH,
                message=f"Failed to execute position management for {position['symbol']}",
                exception=e,
                context={"position": position, "advice": advice}
            )
    
    def _merge_recommendations(self, main_rec: Dict, specialized_advice: Dict) -> Dict:
        """Merge recommendations from different AI services"""
        merged = main_rec.copy()
        
        if 'adjusted_parameters' in specialized_advice:
            merged.update(specialized_advice['adjusted_parameters'])
        
        # Calculate combined confidence
        main_confidence = main_rec.get('confidence_score', 0.5)
        specialized_confidence = specialized_advice.get('confidence_score', 0.5)
        merged['combined_confidence'] = (main_confidence + specialized_confidence) / 2
        
        return merged
    
    def _validate_recommendation(self, recommendation: Dict) -> bool:
        """Validate trading recommendation"""
        required_fields = ['symbol', 'action', 'combined_confidence']
        
        for field in required_fields:
            if field not in recommendation:
                self.logger.warning(f"Recommendation missing required field: {field}")
                return False
        
        if recommendation['combined_confidence'] < 0.6:
            self.logger.warning(f"Low confidence recommendation for {recommendation['symbol']}")
            return False
        
        return True
    
    def _calculate_position_size(self, recommendation: Dict) -> float:
        """Calculate position size based on risk management"""
        base_size = recommendation.get('position_size', self.config.max_position_size)
        confidence = recommendation['combined_confidence']
        
        # Adjust position size based on confidence
        adjusted_size = base_size * confidence
        
        # Apply risk per trade limit
        risk_adjusted_size = min(adjusted_size, self.config.risk_per_trade)
        
        return risk_adjusted_size
    
    def _get_market_conditions(self, symbol: str) -> Dict:
        """Get current market conditions for a symbol"""
        # This would typically fetch real market data
        # For now, return mock data
        return {
            "volatility": 0.02,
            "volume": 1000,
            "trend": "bullish",
            "liquidity": "high"
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status including error recovery metrics"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "orchestrator_status": "operational",
            "error_recovery": self.error_manager.get_system_health_summary(),
            "component_health": {}
        }
        
        # Add health status for each component
        for component in ["main_ai", "specialized_ai", "bingx_ai", "bingx_api"]:
            health_status["component_health"][component] = {
                "status": "healthy",
                "last_error": None,
                "error_count": 0
            }
        
        return health_status

# Integration with main trading system
def integrate_error_recovery(system):
    """Integrate error recovery into the main trading system"""
    
    # Create enhanced orchestrator with error recovery
    enhanced_orchestrator = EnhancedTradingOrchestratorWithRecovery(
        system.scouting_bot,
        system.bingx_api,
        system.config
    )
    
    # Replace the original orchestrator
    system.orchestrator = enhanced_orchestrator
    
    # Add error recovery manager to system for access elsewhere
    system.error_manager = enhanced_orchestrator.error_manager
    
    # Set up periodic health reporting
    asyncio.create_task(_periodic_health_report(system))
    
    return system

async def _periodic_health_report(system):
    """Periodically report system health"""
    while True:
        try:
            health_status = system.error_manager.get_system_health_summary()
            system.logger.info(f"System health status: {health_status}")
            
            # Alert on critical status
            if health_status['overall_state'] == 'critical':
                system.logger.critical(
                    f"System in critical state! Components: {health_status['critical_components']}"
                )
            
            await asyncio.sleep(300)  # Report every 5 minutes
            
        except Exception as e:
            system.logger.error(f"Health reporting failed: {e}")
            await asyncio.sleep(60)  # Retry after 1 minute on error

# Usage in main.py
"""
In your main.py, replace the orchestrator initialization with:

# Initialize enhanced orchestrator with error recovery
from enhanced_error_recovery_integration import EnhancedTradingOrchestratorWithRecovery

trading_config = TradingConfig(
    ia_strategy_url=getattr(config, 'IA_STRATEGY_URL', 'http://localhost:8001/api/analyze'),
    ia_specialized_url=getattr(config, 'IA_SPECIALIZED_URL', 'http://localhost:8002/api/consult'),
    bingx_ia_url=getattr(config, 'BINGX_IA_URL', 'http://localhost:8003/api/position'),
    max_retry_attempts=getattr(config, 'MAX_RETRY_ATTEMPTS', 3),
    request_timeout=getattr(config, 'REQUEST_TIMEOUT', 30),
    max_position_size=getattr(config, 'MAX_POSITION_SIZE', 0.1),
    risk_per_trade=getattr(config, 'RISK_PER_TRADE', 0.02)
)

system.orchestrator = EnhancedTradingOrchestratorWithRecovery(
    system.scouting_bot, 
    system.bingx_api,
    trading_config
)

# Add error manager to system
system.error_manager = system.orchestrator.error_manager
"""
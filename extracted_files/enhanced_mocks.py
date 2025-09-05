# enhanced_mocks.py
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

class EnhancedMockScoutingBot:
    """Enhanced mock scouting bot with realistic data simulation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.failure_rate = 0.05  # 5% failure rate for testing
        
    def scan_and_filter_opportunities(self) -> Dict[str, Any]:
        # Simulate occasional failures
        if random.random() < self.failure_rate:
            raise ConnectionError("Mock network failure in scouting")
        
        # Generate realistic market data
        symbols = ["BTC-USDT", "ETH-USDT", "ADA-USDT", "SOL-USDT"]
        opportunities = {}
        
        for symbol in symbols[:random.randint(0, len(symbols))]:
            base_price = {"BTC-USDT": 50000, "ETH-USDT": 3000, 
                         "ADA-USDT": 0.5, "SOL-USDT": 100}[symbol]
            
            # Generate realistic price history with some trend
            price_history = []
            current_price = base_price
            
            for i in range(50):
                # Add some random walk with slight trend
                change_pct = random.gauss(0, 0.02)  # 2% standard deviation
                current_price *= (1 + change_pct)
                price_history.append(round(current_price, 2))
            
            # Generate volume history
            base_volume = random.uniform(1000, 10000)
            volume_history = []
            for i in range(50):
                volume_multiplier = random.uniform(0.5, 2.0)
                volume_history.append(round(base_volume * volume_multiplier, 0))
            
            opportunities[symbol] = {
                "price_history": price_history,
                "volume_history": volume_history,
                "timestamp": datetime.now().isoformat(),
                "confidence_indicator": random.uniform(0.4, 0.9),
                "market_cap_rank": random.randint(1, 100)
            }
        
        return opportunities

class EnhancedMockBingXAPIClient:
    """Enhanced mock BingX API client with realistic behavior"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.failure_rate = 0.03  # 3% failure rate
        self.positions = []
        self.order_counter = 1000
        
    def _simulate_failure(self, operation: str):
        """Simulate random API failures"""
        if random.random() < self.failure_rate:
            error_types = [
                "API rate limit exceeded",
                "Insufficient balance",
                "Market closed",
                "Network timeout",
                "Invalid symbol"
            ]
            raise Exception(f"Mock {operation} failure: {random.choice(error_types)}")
    
    def fetch_open_positions(self) -> List[Dict]:
        self._simulate_failure("fetch_positions")
        
        # Return mock positions with realistic data
        mock_positions = []
        for i, symbol in enumerate(["BTC-USDT", "ETH-USDT"][:random.randint(0, 3)]):
            entry_price = random.uniform(45000, 55000) if "BTC" in symbol else random.uniform(2800, 3200)
            current_price = entry_price * random.uniform(0.95, 1.05)  # +/- 5% from entry
            
            position = {
                "symbol": symbol,
                "entryPrice": str(entry_price),
                "currentPrice": str(current_price),
                "positionAmt": str(random.uniform(-2, 2)),  # Can be negative for short
                "unRealizedProfit": str((current_price - entry_price) * 0.1),
                "leverage": str(random.randint(5, 20)),
                "liquidationPrice": str(entry_price * 0.8),  # Simplified calculation
                "updateTime": int((datetime.now() - timedelta(hours=random.randint(1, 48))).timestamp() * 1000)
            }
            mock_positions.append(position)
        
        self.positions = mock_positions
        return mock_positions
    
    def fetch_order_book(self, symbol: str) -> Dict:
        self._simulate_failure("fetch_orderbook")
        
        mid_price = random.uniform(45000, 55000) if "BTC" in symbol else random.uniform(2800, 3200)
        spread = mid_price * 0.001  # 0.1% spread
        
        # Generate realistic order book
        bids = []
        asks = []
        
        for i in range(10):
            bid_price = mid_price - spread/2 - (i * spread * 0.1)
            ask_price = mid_price + spread/2 + (i * spread * 0.1)
            
            bids.append([bid_price, random.uniform(0.1, 5.0)])
            asks.append([ask_price, random.uniform(0.1, 5.0)])
        
        return {
            'bids': bids,
            'asks': asks,
            'timestamp': int(time.time() * 1000)
        }
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> List[List]:
        self._simulate_failure("fetch_ohlcv")
        
        base_price = random.uniform(45000, 55000) if "BTC" in symbol else random.uniform(2800, 3200)
        current_time = int(time.time())
        
        ohlcv_data = []
        current_price = base_price
        
        for i in range(limit):
            timestamp = current_time - (i * 3600)  # Hourly data
            
            # Generate OHLCV with some realism
            open_price = current_price
            high_price = open_price * random.uniform(1.0, 1.02)
            low_price = open_price * random.uniform(0.98, 1.0)
            close_price = random.uniform(low_price, high_price)
            volume = random.uniform(100, 10000)
            
            ohlcv_data.insert(0, [timestamp, open_price, high_price, low_price, close_price, volume])
            current_price = close_price
        
        return ohlcv_data
    
    def place_order(self, **kwargs) -> Dict:
        self._simulate_failure("place_order")
        
        # Validate required parameters
        required_params = ['symbol', 'side', 'quantity']
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Simulate order placement
        self.order_counter += 1
        
        return {
            'status': 'success',
            'orderId': str(self.order_counter),
            'symbol': kwargs['symbol'],
            'side': kwargs['side'],
            'quantity': kwargs['quantity'],
            'timestamp': int(time.time() * 1000)
        }
    
    def set_stop_loss(self, symbol: str, price: float) -> Dict:
        self._simulate_failure("set_stop_loss")
        return {'status': 'success', 'stopPrice': price}
    
    def set_take_profit(self, symbol: str, price: float) -> Dict:
        self._simulate_failure("set_take_profit")
        return {'status': 'success', 'takeProfitPrice': price}
    
    def set_trailing_stop(self, symbol: str, params: Dict) -> Dict:
        self._simulate_failure("set_trailing_stop")
        return {'status': 'success', 'trailingStop': params}
    
    def close_position(self, symbol: str) -> Dict:
        self._simulate_failure("close_position")
        # Remove position from mock positions
        self.positions = [p for p in self.positions if p['symbol'] != symbol]
        return {'status': 'success', 'closed_symbol': symbol}

class MockAIServices:
    """Mock AI services for testing the orchestrator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.failure_rate = 0.02  # 2% failure rate
        
    def _simulate_ai_failure(self):
        if random.random() < self.failure_rate:
            raise Exception("Mock AI service temporarily unavailable")
    
    def get_main_strategy_response(self, symbol: str, data: Dict) -> Dict:
        self._simulate_ai_failure()
        
        confidence = random.uniform(0.3, 0.95)
        action = random.choice(['buy', 'sell', 'long', 'short', 'hold'])
        
        return {
            "symbol": symbol,
            "action": action,
            "confidence_score": confidence,
            "rationale": f"Mock AI analysis for {symbol}: {action} signal detected",
            "position_size": min(0.1, confidence * 0.15),  # Scale with confidence
            "stop_loss": data.get('price_history', [50000])[-1] * random.uniform(0.95, 0.98),
            "take_profit": data.get('price_history', [50000])[-1] * random.uniform(1.02, 1.05),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_specialized_response(self, symbol: str, data: Dict, main_rec: Dict) -> Dict:
        self._simulate_ai_failure()
        
        # Specialized AI sometimes disagrees or requests clarification
        needs_clarification = random.random() < 0.2  # 20% chance
        confirmed = random.random() > 0.3 if not needs_clarification else True
        
        response = {
            "confirmed": confirmed,
            "confidence_score": random.uniform(0.4, 0.9),
            "needs_clarification": needs_clarification,
            "rationale": f"Specialized analysis for {symbol}",
            "timestamp": datetime.now().isoformat()
        }
        
        if needs_clarification:
            response["clarification_questions"] = random.sample([
                "What is the current volatility level?",
                "How does the volume compare to historical averages?",
                "What is the correlation with BTC?",
                "What are the current liquidity conditions?"
            ], random.randint(1, 3))
        
        if confirmed:
            # Provide some adjustments to the main recommendation
            response["adjusted_parameters"] = {
                "position_size": main_rec.get("position_size", 0.05) * random.uniform(0.8, 1.2),
                "stop_loss": main_rec.get("stop_loss", 50000) * random.uniform(0.98, 1.02),
                "take_profit": main_rec.get("take_profit", 52000) * random.uniform(0.98, 1.02)
            }
        
        return response
    
    def get_bingx_response(self, position: Dict, metrics: Dict = None) -> Dict:
        self._simulate_ai_failure()
        
        # BingX AI for position management
        action_required = random.random() > 0.7  # 30% chance of action needed
        
        response = {
            "action_required": action_required,
            "timestamp": datetime.now().isoformat()
        }
        
        if action_required:
            actions = random.sample([
                "adjust_stop_loss",
                "adjust_take_profit",
                "enable_trailing_stop",
                "close_position"
            ], random.randint(1, 2))
            
            for action in actions:
                response[action] = True
                
                if action == "adjust_stop_loss":
                    current_price = float(position["currentPrice"])
                    response["new_stop_loss"] = current_price * random.uniform(0.95, 0.98)
                elif action == "adjust_take_profit":
                    current_price = float(position["currentPrice"])
                    response["new_take_profit"] = current_price * random.uniform(1.02, 1.05)
                elif action == "enable_trailing_stop":
                    response["trailing_params"] = {
                        "activation_price": float(position["currentPrice"]) * 1.02,
                        "callback_rate": random.uniform(0.01, 0.03)
                    }
                elif action == "close_position":
                    response["close_reason"] = random.choice([
                        "Take profit target reached",
                        "Risk management triggered",
                        "Market conditions changed"
                    ])
        
        return response

# Integration helpers
class MockServiceRegistry:
    """Registry for mock services to simulate external dependencies"""
    
    def __init__(self):
        self.ai_services = MockAIServices()
        self.response_delay = 0.1  # Simulate network delay
        
    async def call_service(self, url: str, payload: Dict) -> Dict:
        """Simulate async service calls with realistic delays and responses"""
        await asyncio.sleep(self.response_delay)
        
        if "strategy" in url:
            symbol = payload.get("symbol", "UNKNOWN")
            data = payload.get("data", {})
            return self.ai_services.get_main_strategy_response(symbol, data)
        elif "specialized" in url:
            symbol = payload.get("symbol", "UNKNOWN")
            data = payload.get("opportunity_data", {})
            main_rec = payload.get("main_recommendation", {})
            return self.ai_services.get_specialized_response(symbol, data, main_rec)
        elif "bingx" in url:
            # This would be position management
            return self.ai_services.get_bingx_response({}, {})
        else:
            raise ValueError(f"Unknown service URL: {url}")

# Usage example for testing
def create_test_environment():
    """Create a complete test environment with enhanced mocks"""
    return {
        "scouting_bot": EnhancedMockScoutingBot(),
        "bingx_api": EnhancedMockBingXAPIClient(),
        "ai_services": MockAIServices(),
        "service_registry": MockServiceRegistry()
    }

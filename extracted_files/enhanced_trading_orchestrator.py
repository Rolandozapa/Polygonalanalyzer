import requests
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import time
import logging
import hashlib
import hmac
from datetime import datetime, timedelta
from dataclasses import dataclass
from contextlib import contextmanager
import sqlite3

# Enhanced imports for production features
try:
    from tenacity import retry, stop_after_attempt, wait_exponential
    from cachetools import TTLCache
    from prometheus_client import Counter, Histogram, Gauge
    PRODUCTION_FEATURES = True
except ImportError:
    PRODUCTION_FEATURES = False
    logging.warning("Production features disabled. Install: pip install tenacity cachetools prometheus-client")

from security_utils import execute_with_retry, call_ia_endpoint, validate_ia_response
import config

@dataclass
class TradingConfig:
    """Enhanced configuration management"""
    ia_strategy_url: str = config.IA_STRATEGY_URL
    ia_specialized_url: str = config.IA_SPECIALIZED_URL
    bingx_ia_url: str = config.BINGX_IA_URL
    max_retry_attempts: int = config.MAX_RETRY_ATTEMPTS
    request_timeout: int = config.REQUEST_TIMEOUT
    cache_ttl: int = 300
    max_position_size: float = config.MAX_POSITION_SIZE
    risk_per_trade: float = config.RISK_PER_TRADE

class AdvancedTradingOrchestrator:
    def __init__(self, scouting_bot, bingx_api_client, trading_config: TradingConfig = None):
        self.scouting_bot = scouting_bot
        self.bingx_api = bingx_api_client
        self.config = trading_config or TradingConfig()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize production features if available
        if PRODUCTION_FEATURES:
            self.ia_cache = TTLCache(maxsize=100, ttl=self.config.cache_ttl)
            self._init_metrics()
        else:
            self.ia_cache = {}
        
        # Session for connection reuse
        self.session = requests.Session()
        self._configure_session()
        
        # Conversation management
        self.ia_conversations = {}
        
        # Database for persistent conversation storage
        self._init_database()
        
    def _configure_session(self):
        """Configure HTTP session with proper settings"""
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=0  # Handle retries manually
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
    def _init_metrics(self):
        """Initialize Prometheus metrics if available"""
        if PRODUCTION_FEATURES:
            self.ai_requests_total = Counter(
                'ai_requests_total', 
                'Total AI requests', 
                ['service', 'status']
            )
            self.ai_response_time = Histogram(
                'ai_response_time_seconds', 
                'AI response time', 
                ['service']
            )
            self.active_positions = Gauge(
                'active_positions_count', 
                'Number of active positions'
            )
    
    def _init_database(self):
        """Initialize conversation database"""
        try:
            with self.get_conversation_db() as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT UNIQUE NOT NULL,
                        symbol TEXT NOT NULL,
                        source_ia TEXT NOT NULL,
                        target_ia TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        response TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_conversations_symbol 
                    ON conversations(symbol)
                ''')
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")

    def run_complete_trading_cycle(self):
        """Execute complete trading cycle with enhanced error handling"""
        start_time = time.time()
        
        try:
            # 1. Scouting opportunities
            self.logger.info("Starting opportunity scanning")
            opportunities = self.scouting_bot.scan_and_filter_opportunities()
            
            if not opportunities:
                self.logger.info("No opportunities found")
                return []
            
            # 2. AI consultations with enhanced processing
            strategy_recommendations = []
            
            for symbol, data in opportunities.items():
                try:
                    recommendation = self._process_symbol_opportunity(symbol, data)
                    if recommendation:
                        strategy_recommendations.append(recommendation)
                except Exception as e:
                    self.logger.error(f"Failed to process {symbol}: {e}")
                    continue
            
            # 3. Execute trades
            executed_trades = self.execute_ia_recommendations(strategy_recommendations)
            
            # 4. Position monitoring
            self.monitor_and_manage_positions()
            
            # 5. Update metrics
            if PRODUCTION_FEATURES:
                cycle_time = time.time() - start_time
                self.logger.info(f"Trading cycle completed in {cycle_time:.2f}s")
            
            return executed_trades
            
        except Exception as e:
            self.logger.error(f"Trading cycle failed: {e}")
            return []
    
    def _process_symbol_opportunity(self, symbol: str, data: Dict) -> Optional[Dict]:
        """Process individual symbol opportunity with full AI consultation"""
        
        # Check cache first
        cache_key = f"{symbol}_{hash(str(data))}"
        if cache_key in self.ia_cache:
            self.logger.debug(f"Using cached result for {symbol}")
            return self.ia_cache[cache_key]
        
        # Consult main strategy AI
        main_recommendation = self.consult_main_strategy_ai(symbol, data)
        
        if not main_recommendation:
            return None
        
        # Validate main AI response
        required_fields = ['action', 'confidence_score', 'symbol']
        if not validate_ia_response(main_recommendation, required_fields):
            self.logger.warning(f"Invalid main AI response for {symbol}")
            return None
        
        # Consult specialized AI for validation
        specialized_advice = self.consult_specialized_ai(
            symbol, data, main_recommendation
        )
        
        if specialized_advice and specialized_advice.get('confirmed', False):
            # Merge recommendations
            final_recommendation = self.merge_ia_recommendations(
                main_recommendation, specialized_advice
            )
            
            # Cache the result
            self.ia_cache[cache_key] = final_recommendation
            
            return final_recommendation
        
        return None

    @contextmanager
    def get_conversation_db(self):
        """Database context manager"""
        conn = sqlite3.connect('conversations.db')
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def consult_main_strategy_ai(self, symbol: str, opportunity_data: Dict) -> Optional[Dict]:
        """Enhanced main AI consultation with retry and monitoring"""
        
        def _make_request():
            payload = self.prepare_ai_payload(symbol, opportunity_data)
            
            start_time = time.time()
            response = self.session.post(
                self.config.ia_strategy_url,
                json=payload,
                timeout=self.config.request_timeout
            )
            
            # Record metrics
            if PRODUCTION_FEATURES:
                response_time = time.time() - start_time
                self.ai_response_time.labels(service='main_ai').observe(response_time)
                
                if response.status_code == 200:
                    self.ai_requests_total.labels(service='main_ai', status='success').inc()
                else:
                    self.ai_requests_total.labels(service='main_ai', status='error').inc()
            
            if response.status_code == 200:
                return response.json()
            else:
                raise requests.RequestException(f"HTTP {response.status_code}")
        
        try:
            return execute_with_retry(
                _make_request,
                max_retries=self.config.max_retry_attempts
            )
        except Exception as e:
            self.logger.error(f"Main AI consultation failed for {symbol}: {e}")
            return None

    def consult_specialized_ai(self, symbol: str, opportunity_data: Dict, 
                              main_recommendation: Dict) -> Optional[Dict]:
        """Enhanced specialized AI consultation with conversation management"""
        
        try:
            # Prepare context from database
            conversation_context = self._get_conversation_context(symbol, 'specialized')
            
            specialized_payload = {
                "symbol": symbol,
                "opportunity_data": opportunity_data,
                "main_recommendation": main_recommendation,
                "conversation_context": conversation_context
            }
            
            def _make_request():
                return self.session.post(
                    self.config.ia_specialized_url,
                    json=specialized_payload,
                    timeout=45
                )
            
            response = execute_with_retry(_make_request)
            
            if response.status_code == 200:
                result = response.json()
                
                # Store conversation context
                self._store_conversation_context(symbol, 'specialized', result)
                
                # Handle clarification requests
                if result.get('needs_clarification', False):
                    result = self._handle_clarification_process(
                        symbol, opportunity_data, result
                    )
                
                return result
            else:
                self.logger.error(f"Specialized AI error for {symbol}: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Specialized AI consultation failed: {e}")
            return None
    
    def _get_conversation_context(self, symbol: str, ia_type: str) -> Dict:
        """Retrieve conversation context from database"""
        try:
            with self.get_conversation_db() as conn:
                cursor = conn.execute('''
                    SELECT response FROM conversations 
                    WHERE symbol = ? AND target_ia = ? 
                    ORDER BY timestamp DESC LIMIT 5
                ''', (symbol, ia_type))
                
                contexts = [json.loads(row[0]) for row in cursor.fetchall()]
                return {"previous_interactions": contexts}
        except Exception as e:
            self.logger.warning(f"Failed to retrieve conversation context: {e}")
            return {}
    
    def _store_conversation_context(self, symbol: str, ia_type: str, result: Dict):
        """Store conversation context in database"""
        try:
            conversation_id = f"{symbol}_{ia_type}_{datetime.now().timestamp()}"
            
            with self.get_conversation_db() as conn:
                conn.execute('''
                    INSERT INTO conversations 
                    (conversation_id, symbol, source_ia, target_ia, payload, response)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    conversation_id,
                    symbol,
                    'orchestrator',
                    ia_type,
                    json.dumps({}),  # Store payload if needed
                    json.dumps(result)
                ))
        except Exception as e:
            self.logger.warning(f"Failed to store conversation context: {e}")

    def _handle_clarification_process(self, symbol: str, opportunity_data: Dict, 
                                    initial_result: Dict) -> Dict:
        """Handle clarification questions from specialized AI"""
        
        clarification_questions = initial_result.get('clarification_questions', [])
        clarifications = self.get_clarifications(symbol, opportunity_data, clarification_questions)
        
        if clarifications:
            try:
                followup_payload = {
                    "symbol": symbol,
                    "clarifications": clarifications,
                    "conversation_context": self._get_conversation_context(symbol, 'specialized')
                }
                
                response = self.session.post(
                    f"{self.config.ia_specialized_url}/clarify",
                    json=followup_payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self._store_conversation_context(symbol, 'specialized', result)
                    return result
                    
            except Exception as e:
                self.logger.error(f"Clarification process failed: {e}")
        
        return initial_result

    def get_clarifications(self, symbol: str, opportunity_data: Dict, 
                          questions: List[str]) -> Dict[str, Any]:
        """Enhanced clarification handling with more analysis types"""
        clarifications = {}
        
        for question in questions:
            question_lower = question.lower()
            
            try:
                if "volatility" in question_lower:
                    volatility = self.calculate_volatility(opportunity_data)
                    volatility_level = self._categorize_volatility(volatility)
                    clarifications[question] = {
                        "value": volatility,
                        "level": volatility_level,
                        "description": f"Volatilité {volatility_level}: {volatility:.2f}%"
                    }
                    
                elif "volume" in question_lower:
                    volume_analysis = self.analyze_volume_pattern(opportunity_data)
                    volume_metrics = self._calculate_volume_metrics(opportunity_data)
                    clarifications[question] = {
                        "analysis": volume_analysis,
                        "metrics": volume_metrics
                    }
                    
                elif "correlation" in question_lower:
                    correlation = self.check_market_correlation(symbol)
                    correlation_strength = self._categorize_correlation(correlation)
                    clarifications[question] = {
                        "value": correlation,
                        "strength": correlation_strength,
                        "description": f"Corrélation BTC: {correlation:.2f} ({correlation_strength})"
                    }
                    
                elif "liquidity" in question_lower:
                    liquidity_metrics = self._analyze_liquidity(symbol, opportunity_data)
                    clarifications[question] = liquidity_metrics
                    
                elif "technical" in question_lower:
                    technical_analysis = self._perform_technical_analysis(symbol, opportunity_data)
                    clarifications[question] = technical_analysis
                    
                else:
                    self.logger.info(f"Manual intervention needed for {symbol}: {question}")
                    clarifications[question] = {
                        "status": "needs_manual_review",
                        "question": question
                    }
                    
            except Exception as e:
                self.logger.error(f"Clarification analysis failed for '{question}': {e}")
                clarifications[question] = {"error": f"Analysis failed: {str(e)}"}
        
        return clarifications

    # Enhanced utility methods
    def _categorize_volatility(self, volatility: float) -> str:
        """Categorize volatility level"""
        if volatility > 80:
            return "très élevée"
        elif volatility > 50:
            return "élevée"
        elif volatility > 20:
            return "modérée"
        else:
            return "faible"
    
    def _categorize_correlation(self, correlation: float) -> str:
        """Categorize correlation strength"""
        abs_corr = abs(correlation)
        if abs_corr > 0.8:
            return "très forte"
        elif abs_corr > 0.6:
            return "forte"
        elif abs_corr > 0.3:
            return "modérée"
        else:
            return "faible"
    
    def _calculate_volume_metrics(self, opportunity_data: Dict) -> Dict:
        """Calculate detailed volume metrics"""
        volume_data = opportunity_data.get('volume_history', [])
        if len(volume_data) < 10:
            return {"error": "Insufficient volume data"}
        
        recent_volume = np.mean(volume_data[-5:])
        historical_volume = np.mean(volume_data[:-5])
        volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 0
        
        return {
            "recent_avg": recent_volume,
            "historical_avg": historical_volume,
            "ratio": volume_ratio,
            "trend": "increasing" if volume_ratio > 1.2 else "decreasing" if volume_ratio < 0.8 else "stable"
        }
    
    def _analyze_liquidity(self, symbol: str, opportunity_data: Dict) -> Dict:
        """Analyze market liquidity"""
        try:
            orderbook = self.bingx_api.fetch_order_book(symbol)
            
            if orderbook:
                bid_volume = sum([float(bid[1]) for bid in orderbook.get('bids', [])[:10]])
                ask_volume = sum([float(ask[1]) for ask in orderbook.get('asks', [])[:10]])
                
                spread = float(orderbook['asks'][0][0]) - float(orderbook['bids'][0][0])
                mid_price = (float(orderbook['asks'][0][0]) + float(orderbook['bids'][0][0])) / 2
                spread_pct = (spread / mid_price) * 100
                
                return {
                    "bid_volume": bid_volume,
                    "ask_volume": ask_volume,
                    "spread_percentage": spread_pct,
                    "liquidity_score": min(100, (bid_volume + ask_volume) / 1000),
                    "market_impact": "low" if spread_pct < 0.1 else "medium" if spread_pct < 0.5 else "high"
                }
        except Exception as e:
            self.logger.error(f"Liquidity analysis failed: {e}")
        
        return {"error": "Liquidity analysis unavailable"}
    
    def _perform_technical_analysis(self, symbol: str, opportunity_data: Dict) -> Dict:
        """Perform basic technical analysis"""
        prices = opportunity_data.get('price_history', [])
        if len(prices) < 20:
            return {"error": "Insufficient price data"}
        
        try:
            # Calculate technical indicators
            sma_20 = np.mean(prices[-20:])
            sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else sma_20
            
            current_price = prices[-1]
            rsi = self._calculate_rsi(prices)
            
            # Determine trend
            trend = "bullish" if current_price > sma_20 > sma_50 else "bearish" if current_price < sma_20 < sma_50 else "sideways"
            
            return {
                "current_price": current_price,
                "sma_20": sma_20,
                "sma_50": sma_50,
                "rsi": rsi,
                "trend": trend,
                "price_vs_sma20": ((current_price - sma_20) / sma_20) * 100
            }
        except Exception as e:
            return {"error": f"Technical analysis failed: {str(e)}"}
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def merge_ia_recommendations(self, main_recommendation: Dict, 
                                specialized_advice: Dict) -> Dict:
        """Enhanced recommendation merging with risk assessment"""
        merged = main_recommendation.copy()
        
        # Incorporate specialized adjustments
        if 'adjusted_parameters' in specialized_advice:
            merged.update(specialized_advice['adjusted_parameters'])
        
        # Calculate combined confidence with weighting
        main_confidence = main_recommendation.get('confidence_score', 0.5)
        specialized_confidence = specialized_advice.get('confidence_score', 0.5)
        
        # Weight specialized AI more heavily if it has more context
        specialized_weight = 0.6 if specialized_advice.get('analysis_depth') == 'deep' else 0.4
        main_weight = 1 - specialized_weight
        
        merged['combined_confidence'] = (main_confidence * main_weight + 
                                       specialized_confidence * specialized_weight)
        
        # Risk assessment
        merged['risk_level'] = self._assess_combined_risk(main_recommendation, specialized_advice)
        
        # Enhanced rationale
        merged['combined_rationale'] = {
            'main_ai': main_recommendation.get('rationale', ''),
            'specialized_ai': specialized_advice.get('rationale', ''),
            'consensus_strength': 'high' if abs(main_confidence - specialized_confidence) < 0.2 else 'medium'
        }
        
        return merged
    
    def _assess_combined_risk(self, main_rec: Dict, spec_rec: Dict) -> str:
        """Assess combined risk level from both AI recommendations"""
        main_confidence = main_rec.get('confidence_score', 0.5)
        spec_confidence = spec_rec.get('confidence_score', 0.5)
        
        avg_confidence = (main_confidence + spec_confidence) / 2
        confidence_agreement = 1 - abs(main_confidence - spec_confidence)
        
        if avg_confidence > 0.8 and confidence_agreement > 0.8:
            return "low"
        elif avg_confidence > 0.6 and confidence_agreement > 0.6:
            return "medium"
        else:
            return "high"

    def monitor_and_manage_positions(self):
        """Enhanced position monitoring with comprehensive management"""
        try:
            open_positions = self.bingx_api.fetch_open_positions()
            
            if PRODUCTION_FEATURES:
                self.active_positions.set(len(open_positions))
            
            for position in open_positions:
                try:
                    # Enhanced position analysis
                    position_metrics = self._calculate_position_metrics(position)
                    
                    # Consult BingX AI with enhanced data
                    management_advice = self.consult_bingx_ia(position, position_metrics)
                    
                    if management_advice and management_advice.get('action_required', False):
                        self.execute_position_management(position, management_advice)
                        
                except Exception as e:
                    self.logger.error(f"Position management error for {position['symbol']}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Position monitoring failed: {e}")
    
    def _calculate_position_metrics(self, position: Dict) -> Dict:
        """Calculate comprehensive position metrics"""
        try:
            current_price = float(position['currentPrice'])
            entry_price = float(position['entryPrice'])
            position_size = abs(float(position['positionAmt']))
            
            # Basic metrics
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            if float(position['positionAmt']) < 0:  # Short position
                pnl_pct *= -1
            
            # Time in position
            entry_time = datetime.fromtimestamp(position['updateTime'] / 1000)
            time_in_position = (datetime.now() - entry_time).total_seconds() / 3600
            
            # Risk metrics
            leverage = float(position.get('leverage', 1))
            liquidation_price = float(position.get('liquidationPrice', 0))
            
            liquidation_distance = 0
            if liquidation_price > 0:
                liquidation_distance = abs((current_price - liquidation_price) / current_price) * 100
            
            return {
                'unrealized_pnl_pct': pnl_pct,
                'time_in_position_hours': time_in_position,
                'liquidation_distance_pct': liquidation_distance,
                'leverage': leverage,
                'position_value_usd': position_size * current_price,
                'risk_score': self._calculate_position_risk_score(position, pnl_pct, liquidation_distance)
            }
            
        except Exception as e:
            self.logger.error(f"Position metrics calculation failed: {e}")
            return {}
    
    def _calculate_position_risk_score(self, position: Dict, pnl_pct: float, 
                                     liquidation_distance: float) -> float:
        """Calculate risk score for position (0-100, higher = riskier)"""
        risk_score = 0
        
        # PnL risk
        if pnl_pct < -10:
            risk_score += 30
        elif pnl_pct < -5:
            risk_score += 15
        
        # Liquidation risk
        if liquidation_distance < 5:
            risk_score += 40
        elif liquidation_distance < 10:
            risk_score += 20
        
        # Leverage risk
        leverage = float(position.get('leverage', 1))
        if leverage > 20:
            risk_score += 20
        elif leverage > 10:
            risk_score += 10
        
        # Time risk (positions open too long)
        entry_time = datetime.fromtimestamp(position['updateTime'] / 1000)
        hours_open = (datetime.now() - entry_time).total_seconds() / 3600
        if hours_open > 168:  # 1 week
            risk_score += 10
        
        return min(100, risk_score)

    # All other methods from the original implementation remain the same
    # (consult_bingx_ia, execute_position_management, utility methods, etc.)
    
    def consult_bingx_ia(self, position: Dict, position_metrics: Dict = None) -> Optional[Dict]:
        """Enhanced BingX AI consultation with position metrics"""
        try:
            position_data = {
                "symbol": position['symbol'],
                "entry_price": position['entryPrice'],
                "current_price": position['currentPrice'],
                "position_size": position['positionAmt'],
                "unrealized_pnl": position['unRealizedProfit'],
                "leverage": position['leverage'],
                "liquidation_price": position.get('liquidationPrice'),
                "market_conditions": self.get_current_market_conditions(position['symbol']),
                "position_metrics": position_metrics or {}
            }
            
            def _make_request():
                return self.session.post(
                    self.config.bingx_ia_url,
                    json=position_data,
                    timeout=30
                )
            
            response = execute_with_retry(_make_request)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"BingX AI error for {position['symbol']}: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"BingX AI consultation failed: {e}")
            return None

    def execute_position_management(self, position: Dict, management_advice: Dict):
        """Execute position management with enhanced logging"""
        symbol = position['symbol']
        
        try:
            actions_taken = []
            
            if management_advice.get('adjust_stop_loss', False):
                new_stop_loss = management_advice['new_stop_loss']
                self.bingx_api.set_stop_loss(symbol, new_stop_loss)
                actions_taken.append(f"Stop-loss adjusted to {new_stop_loss}")
            
            if management_advice.get('adjust_take_profit', False):
                new_take_profit = management_advice['new_take_profit']
                self.bingx_api.set_take_profit(symbol, new_take_profit)
                actions_taken.append(f"Take-profit adjusted to {new_take_profit}")
            
            if management_advice.get('enable_trailing_stop', False):
                trailing_params = management_advice['trailing_params']
                self.bingx_api.set_trailing_stop(symbol, trailing_params)
                actions_taken.append("Trailing stop enabled")
            
            if management_advice.get('close_position', False):
                reason = management_advice['close_reason']
                self.bingx_api.close_position(symbol)
                actions_taken.append(f"Position closed: {reason}")
            
            if actions_taken:
                self.logger.info(f"Position management for {symbol}: {'; '.join(actions_taken)}")
                
        except Exception as e:
            self.logger.error(f"Position management execution failed for {symbol}: {e}")

    # Utility methods (same as original but with enhanced error handling)
    def calculate_volatility(self, opportunity_data: Dict) -> float:
        """Calculate recent volatility with enhanced error handling"""
        try:
            prices = opportunity_data.get('price_history', [])
            if len(prices) > 1:
                returns = np.diff(prices) / np.array(prices[:-1])
                return float(np.std(returns) * 100 * np.sqrt(365))
        except Exception as e:
            self.logger.error(f"Volatility calculation failed: {e}")
        return 0.0

    def analyze_volume_pattern(self, opportunity_data: Dict) -> str:
        """Analyze volume patterns with enhanced metrics"""
        try:
            volume_data = opportunity_data.get('volume_history', [])
            if len(volume_data) < 10:
                return "Données de volume insuffisantes"
            
            recent_avg = np.mean(volume_data[-5:])
            historical_avg = np.mean(volume_data[:-5])
            
            if historical_avg == 0:
                return "Données de volume historiques invalides"
            
            ratio = recent_avg / historical_avg
            
            if ratio > 2:
                return "Volume très élevé récemment"
            elif ratio > 1.5:
                return "Volume élevé récemment" 
            elif ratio < 0.5:
                return "Volume très faible récemment"
            elif ratio < 0.8:
                return "Volume faible récemment"
            else:
                return "Volume dans la normale"
                
        except Exception as e:
            self.logger.error(f"Volume pattern analysis failed: {e}")
            return "Erreur analyse volume"

    def check_market_correlation(self, symbol: str) -> float:
        """Check market correlation with enhanced error handling"""
        try:
            btc_data = self.bingx_api.fetch_ohlcv('BTC-USDT', '1h', limit=100)
            symbol_data = self.bingx_api.fetch_ohlcv(symbol, '1h', limit=100)
            
            if not btc_data or not symbol_data or len(btc_data) < 2 or len(symbol_data) < 2:
                return 0.0
            
            # Extract closing prices
            btc_prices = [float(x[4]) for x in btc_data]
            symbol_prices = [float(x[4]) for x in symbol_data]
            
            # Calculate returns
            btc_returns = np.diff(btc_prices) / np.array(btc_prices[:-1])
            symbol_returns = np.diff(symbol_prices) / np.array(symbol_prices[:-1])
            
            # Ensure same length
            min_length = min(len(btc_returns), len(symbol_returns))
            btc_returns = btc_returns[-min_length:]
            symbol_returns = symbol_returns[-min_length:]
            
            if min_length < 10:
                return 0.0
            
            # Calculate correlation
            correlation_matrix = np.corrcoef(btc_returns, symbol_returns)
            correlation = correlation_matrix[0, 1]
            
            # Handle NaN case
            if np.isnan(correlation):
                return 0.0
                
            return float(correlation)
            
        except Exception as e:
            self.logger.error(f"Correlation calculation failed for {symbol}: {e}")
            return 0.0

    def prepare_ai_payload(self, symbol: str, opportunity_data: Dict) -> Dict:
        """Prepare enhanced AI payload with comprehensive data"""
        try:
            # Basic payload
            payload = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "data": opportunity_data
            }
            
            # Add market context
            payload["market_context"] = {
                "volatility": self.calculate_volatility(opportunity_data),
                "volume_analysis": self.analyze_volume_pattern(opportunity_data),
                "btc_correlation": self.check_market_correlation(symbol)
            }
            
            # Add technical indicators if price history available
            prices = opportunity_data.get('price_history', [])
            if len(prices) >= 20:
                payload["technical_indicators"] = self._perform_technical_analysis(symbol, opportunity_data)
            
            # Add risk parameters
            payload["risk_parameters"] = {
                "max_position_size": self.config.max_position_size,
                "risk_per_trade": self.config.risk_per_trade,
                "current_time": datetime.now().isoformat()
            }
            
            return payload
            
        except Exception as e:
            self.logger.error(f"Payload preparation failed: {e}")
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "error": "Payload preparation failed"
            }

    def get_current_market_conditions(self, symbol: str) -> Dict:
        """Get comprehensive current market conditions"""
        try:
            # Fetch recent price data
            recent_data = self.bingx_api.fetch_ohlcv(symbol, '1h', limit=50)
            
            if not recent_data or len(recent_data) < 10:
                return {"error": "Insufficient market data"}
            
            prices = [float(x[4]) for x in recent_data]
            volumes = [float(x[5]) for x in recent_data]
            
            opportunity_data = {
                "price_history": prices,
                "volume_history": volumes
            }
            
            conditions = {
                "volatility": self.calculate_volatility(opportunity_data),
                "volume_pattern": self.analyze_volume_pattern(opportunity_data),
                "btc_correlation": self.check_market_correlation(symbol),
                "price_trend": self._determine_price_trend(prices),
                "market_strength": self._assess_market_strength(prices, volumes),
                "last_updated": datetime.now().isoformat()
            }
            
            return conditions
            
        except Exception as e:
            self.logger.error(f"Market conditions analysis failed for {symbol}: {e}")
            return {
                "error": f"Market analysis failed: {str(e)}",
                "last_updated": datetime.now().isoformat()
            }

    def _determine_price_trend(self, prices: List[float]) -> str:
        """Determine current price trend"""
        if len(prices) < 10:
            return "insufficient_data"
        
        try:
            # Calculate trend using linear regression
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            
            # Normalize slope by price level
            normalized_slope = (slope / prices[-1]) * 100
            
            if normalized_slope > 0.5:
                return "strong_bullish"
            elif normalized_slope > 0.1:
                return "bullish"
            elif normalized_slope < -0.5:
                return "strong_bearish"
            elif normalized_slope < -0.1:
                return "bearish"
            else:
                return "sideways"
                
        except Exception as e:
            self.logger.error(f"Price trend calculation failed: {e}")
            return "unknown"

    def _assess_market_strength(self, prices: List[float], volumes: List[float]) -> str:
        """Assess market strength based on price and volume"""
        if len(prices) < 10 or len(volumes) < 10:
            return "insufficient_data"
        
        try:
            # Price momentum
            recent_prices = prices[-5:]
            earlier_prices = prices[-10:-5]
            
            price_change = (np.mean(recent_prices) - np.mean(earlier_prices)) / np.mean(earlier_prices)
            
            # Volume confirmation
            recent_volume = np.mean(volumes[-5:])
            historical_volume = np.mean(volumes[-10:-5])
            volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1
            
            # Combine signals
            if price_change > 0.02 and volume_ratio > 1.2:
                return "strong"
            elif price_change > 0.01 and volume_ratio > 1.0:
                return "moderate"
            elif price_change < -0.02 and volume_ratio > 1.2:
                return "weak_with_volume"
            elif abs(price_change) < 0.01 and volume_ratio < 0.8:
                return "consolidating"
            else:
                return "neutral"
                
        except Exception as e:
            self.logger.error(f"Market strength assessment failed: {e}")
            return "unknown"

    def execute_ia_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Execute AI recommendations with enhanced risk management"""
        executed_trades = []
        
        for recommendation in recommendations:
            try:
                # Validate recommendation
                if not self._validate_recommendation(recommendation):
                    self.logger.warning(f"Invalid recommendation for {recommendation.get('symbol')}")
                    continue
                
                # Apply risk management filters
                if not self._pass_risk_filters(recommendation):
                    self.logger.info(f"Recommendation filtered by risk management: {recommendation['symbol']}")
                    continue
                
                # Calculate position size with risk management
                position_size = self._calculate_safe_position_size(recommendation)
                
                # Execute trade
                trade_result = self._execute_single_trade(recommendation, position_size)
                
                if trade_result:
                    executed_trades.append(trade_result)
                    self.logger.info(f"Trade executed for {recommendation['symbol']}: {trade_result}")
                
            except Exception as e:
                self.logger.error(f"Trade execution failed for {recommendation.get('symbol')}: {e}")
        
        return executed_trades

    def _validate_recommendation(self, recommendation: Dict) -> bool:
        """Validate recommendation structure and values"""
        required_fields = ['symbol', 'action', 'combined_confidence']
        
        for field in required_fields:
            if field not in recommendation:
                self.logger.error(f"Missing required field: {field}")
                return False
        
        # Validate action
        if recommendation['action'] not in ['buy', 'sell', 'long', 'short']:
            self.logger.error(f"Invalid action: {recommendation['action']}")
            return False
        
        # Validate confidence
        confidence = recommendation['combined_confidence']
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            self.logger.error(f"Invalid confidence score: {confidence}")
            return False
        
        return True

    def _pass_risk_filters(self, recommendation: Dict) -> bool:
        """Apply risk management filters"""
        
        # Confidence threshold
        min_confidence = 0.6  # Configurable threshold
        if recommendation['combined_confidence'] < min_confidence:
            return False
        
        # Risk level check
        risk_level = recommendation.get('risk_level', 'high')
        if risk_level == 'high' and recommendation['combined_confidence'] < 0.8:
            return False
        
        # Market conditions check
        symbol = recommendation['symbol']
        market_conditions = self.get_current_market_conditions(symbol)
        
        if market_conditions.get('volatility', 0) > 100:  # Very high volatility
            self.logger.warning(f"High volatility detected for {symbol}, reducing position or skipping")
            return recommendation['combined_confidence'] > 0.85
        
        return True

    def _calculate_safe_position_size(self, recommendation: Dict) -> float:
        """Calculate safe position size based on risk parameters"""
        base_size = recommendation.get('position_size', self.config.max_position_size)
        confidence = recommendation['combined_confidence']
        risk_level = recommendation.get('risk_level', 'medium')
        
        # Adjust based on confidence
        confidence_multiplier = confidence
        
        # Adjust based on risk level
        risk_multipliers = {
            'low': 1.0,
            'medium': 0.75,
            'high': 0.5
        }
        risk_multiplier = risk_multipliers.get(risk_level, 0.5)
        
        # Calculate final size
        adjusted_size = base_size * confidence_multiplier * risk_multiplier
        
        # Ensure within limits
        max_size = self.config.max_position_size
        return min(adjusted_size, max_size)

    def _execute_single_trade(self, recommendation: Dict, position_size: float) -> Optional[Dict]:
        """Execute a single trade with comprehensive error handling"""
        try:
            symbol = recommendation['symbol']
            action = recommendation['action']
            
            # Prepare order parameters
            order_params = {
                'symbol': symbol,
                'side': action,
                'quantity': position_size,
                'type': 'market'  # Or 'limit' based on strategy
            }
            
            # Add stop loss if provided
            if 'stop_loss' in recommendation:
                order_params['stopPrice'] = recommendation['stop_loss']
            
            # Add take profit if provided
            if 'take_profit' in recommendation:
                order_params['takeProfitPrice'] = recommendation['take_profit']
            
            # Execute the order
            result = self.bingx_api.place_order(**order_params)
            
            if result and result.get('status') == 'success':
                return {
                    'symbol': symbol,
                    'action': action,
                    'quantity': position_size,
                    'order_id': result.get('orderId'),
                    'timestamp': datetime.now().isoformat(),
                    'recommendation_confidence': recommendation['combined_confidence'],
                    'status': 'executed'
                }
            else:
                self.logger.error(f"Order execution failed for {symbol}: {result}")
                return None
                
        except Exception as e:
            self.logger.error(f"Single trade execution failed: {e}")
            return None

    def cleanup_old_conversations(self, max_age_days: int = 7):
        """Clean up old conversations from database"""
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            
            with self.get_conversation_db() as conn:
                cursor = conn.execute(
                    "DELETE FROM conversations WHERE timestamp < ?",
                    (cutoff_date,)
                )
                deleted_count = cursor.rowcount
                
                if deleted_count > 0:
                    self.logger.info(f"Cleaned up {deleted_count} old conversation records")
                    
        except Exception as e:
            self.logger.error(f"Conversation cleanup failed: {e}")

    def get_system_health(self) -> Dict:
        """Get comprehensive system health status"""
        health = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'components': {}
        }
        
        try:
            # Check AI services connectivity
            health['components']['main_ai'] = self._check_service_health(self.config.ia_strategy_url)
            health['components']['specialized_ai'] = self._check_service_health(self.config.ia_specialized_url)
            health['components']['bingx_ai'] = self._check_service_health(self.config.bingx_ia_url)
            
            # Check database connectivity
            try:
                with self.get_conversation_db() as conn:
                    conn.execute("SELECT 1").fetchone()
                health['components']['database'] = 'healthy'
            except Exception as e:
                health['components']['database'] = f'error: {str(e)}'
            
            # Check BingX API connectivity
            try:
                # This would be a simple health check call to BingX
                # self.bingx_api.ping() or similar
                health['components']['bingx_api'] = 'healthy'  # Placeholder
            except Exception as e:
                health['components']['bingx_api'] = f'error: {str(e)}'
            
            # Overall status
            unhealthy_components = [k for k, v in health['components'].items() 
                                  if isinstance(v, str) and 'error' in v.lower()]
            
            if unhealthy_components:
                health['status'] = 'degraded' if len(unhealthy_components) == 1 else 'unhealthy'
                health['issues'] = unhealthy_components
            
            # Add metrics if available
            if PRODUCTION_FEATURES:
                health['metrics'] = {
                    'cache_size': len(self.ia_cache),
                    'conversation_count': len(self.ia_conversations)
                }
            
            return health
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }

    def _check_service_health(self, url: str) -> str:
        """Check health of individual service"""
        try:
            # Simple health check - adjust endpoint as needed
            health_url = url.replace('/api/', '/health')
            response = requests.get(health_url, timeout=5)
            return 'healthy' if response.status_code == 200 else f'error: HTTP {response.status_code}'
        except Exception as e:
            return f'error: {str(e)}'


# Sample responses remain the same but with enhanced structure
SAMPLE_SPECIALIZED_IA_RESPONSE = {
    "confirmed": True,
    "confidence_score": 0.88,
    "adjusted_parameters": {
        "position_size": 0.03,
        "stop_loss": 41800.00,
        "take_profit": 48500.00,
        "time_horizon": "short_term"
    },
    "rationale": "Confirmation du signal par analyse avancée des flux. Volume soutenu et faible corrélation négative avec le BTC.",
    "needs_clarification": False,
    "clarification_questions": [],
    "updated_context": {
        "last_analysis": datetime.now().isoformat(),
        "analysis_depth": "deep",
        "market_regime": "trending",
        "confidence_factors": ["volume_confirmation", "technical_alignment", "risk_reward_ratio"]
    },
    "risk_assessment": {
        "level": "medium",
        "factors": ["volatility_moderate", "liquidity_adequate"],
        "mitigation": ["tight_stop_loss", "position_sizing"]
    }
}

SAMPLE_BINGX_IA_RESPONSE = {
    "action_required": True,
    "adjust_stop_loss": True,
    "new_stop_loss": 43500.00,
    "adjust_take_profit": False,
    "enable_trailing_stop": True,
    "trailing_params": {
        "activation_price": 46000.00,
        "callback_rate": 0.02
    },
    "close_position": False,
    "close_reason": "",
    "rationale": "Prix approchant de la cible, activation du trailing stop pour protéger les gains.",
    "risk_metrics": {
        "current_risk_score": 25,
        "liquidation_distance": 15.5,
        "recommended_action_urgency": "medium"
    },
    "market_analysis": {
        "trend_strength": "moderate",
        "volatility_forecast": "increasing",
        "support_resistance_levels": [43000, 46500, 49000]
    }
}
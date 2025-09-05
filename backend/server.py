from fastapi import FastAPI, APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import asyncio
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, timezone
from enum import Enum
import pandas as pd
import numpy as np
from emergentintegrations.llm.chat import LlmChat, UserMessage

# Import our advanced market aggregator, BingX trading engine, trending auto-updater, and technical pattern detector
from advanced_market_aggregator import advanced_market_aggregator, MarketDataResponse
from bingx_trading_engine import bingx_trading_engine, BingXOrderSide, BingXOrderType, BingXPositionSide
from trending_auto_updater import trending_auto_updater
from technical_pattern_detector import technical_pattern_detector, TechnicalPattern

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="Dual AI Trading Bot System - Ultra Professional Edition", version="3.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enums
class SignalType(str, Enum):
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"

class TradingStatus(str, Enum):
    ANALYZING = "analyzing"
    PENDING = "pending"
    EXECUTED = "executed"
    REJECTED = "rejected"
    CLOSED = "closed"

# Data Models
class MarketOpportunity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    current_price: float
    volume_24h: float
    price_change_24h: float
    volatility: float
    market_cap: Optional[float] = None
    market_cap_rank: Optional[int] = None
    data_sources: List[str] = []
    data_confidence: float = 1.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TechnicalAnalysis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    rsi: float
    macd_signal: float
    bollinger_position: float
    fibonacci_level: float
    support_levels: List[float]
    resistance_levels: List[float]
    patterns_detected: List[str]
    analysis_confidence: float
    ia1_reasoning: str
    market_sentiment: str = "neutral"
    data_sources: List[str] = []
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TradingDecision(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    signal: SignalType
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    position_size: float
    risk_reward_ratio: float
    ia1_analysis_id: str
    ia2_reasoning: str
    status: TradingStatus = TradingStatus.PENDING
    # BingX integration fields
    bingx_order_id: Optional[str] = None
    bingx_position_id: Optional[str] = None
    actual_entry_price: Optional[float] = None
    actual_quantity: Optional[float] = None
    bingx_status: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class LiveTradingPosition(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    position_side: str
    position_amount: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_percentage: float
    margin_used: float
    leverage: int
    stop_loss_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None
    decision_id: str
    status: str = "OPEN"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class BingXAccountInfo(BaseModel):
    total_balance: float
    available_balance: float
    used_margin: float
    unrealized_pnl: float
    total_positions: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AIConversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    decision_id: str
    ia1_message: str
    ia2_response: str
    conversation_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TradingPerformance(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    decision_id: str
    symbol: str
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    duration_minutes: Optional[int] = None
    outcome: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                pass

manager = ConnectionManager()

# AI Chat instances
def get_ia1_chat():
    return LlmChat(
        api_key=os.environ.get('EMERGENT_LLM_KEY'),
        session_id="ia1-fast-technical-analyst",
        system_message="""You are IA1, a FAST technical analyst for cryptocurrency trading.

Your role:
- Quick technical analysis with key indicators (RSI, MACD, Bollinger)
- Identify main chart patterns 
- Provide confidence score and recommendation
- Keep analysis CONCISE but ACCURATE

Respond in JSON format:
{
    "analysis": "concise technical summary",
    "rsi_signal": "oversold/neutral/overbought",
    "macd_trend": "bullish/bearish/neutral", 
    "patterns": ["key patterns only"],
    "support": [key_support_level],
    "resistance": [key_resistance_level],
    "confidence": 0.85,
    "recommendation": "long/short/hold",
    "reasoning": "brief explanation"
}"""
    ).with_model("openai", "gpt-4o")  # Use GPT-4o for speed

def get_ia2_chat():
    return LlmChat(
        api_key=os.environ.get('EMERGENT_LLM_KEY'),
        session_id="ia2-ultra-decision-agent",
        system_message="""You are IA2, an ultra-intelligent trading decision agent with advanced risk management and multi-source data validation.

Your capabilities:
- Process technical analysis from multiple validated data sources
- Advanced risk management with dynamic position sizing
- Multi-source data confidence assessment
- Autonomous decision making with intelligent clarification protocols
- Real-time market condition evaluation

Decision Framework:
- **Ultra High Confidence (>0.9)**: Execute with full position size, multiple data sources confirm
- **High Confidence (0.8-0.9)**: Execute with standard position size, strong signal confirmation
- **Medium Confidence (0.6-0.8)**: Execute with reduced size OR request specific clarification
- **Low Confidence (<0.6)**: Reject OR ask detailed questions about data discrepancies

Advanced Risk Management:
1. **Multi-Source Validation**: Require 2+ sources for major decisions
2. **Dynamic Position Sizing**: Based on data confidence and market volatility
3. **Intelligent Stop-Loss**: At validated technical levels from multiple sources
4. **Adaptive Take-Profits**: Based on cross-source resistance/support confirmation
5. **Correlation Risk**: Consider existing positions and market regime

Respond in JSON format:
{
    "decision": "execute/clarify/reject",
    "signal": "long/short/hold",
    "confidence": 0.85,
    "data_confidence_assessment": "evaluation of multi-source data quality",
    "position_size": 0.02,
    "stop_loss": 45000,
    "take_profit_levels": [46000, 47000, 48000],
    "risk_reward_ratio": 3.0,
    "max_drawdown_risk": 0.02,
    "market_conditions": "favorable/neutral/unfavorable",
    "data_sources_weight": {"source": "confidence_weight"},
    "reasoning": "detailed multi-source analysis with risk assessment",
    "questions_for_ia1": ["specific questions if clarification needed"],
    "execution_notes": "multi-source validated execution instructions"
}"""
    ).with_model("openai", "gpt-5")

# Ultra Professional Trading System Classes
class UltraProfessionalCryptoScout:
    def __init__(self):
        self.market_aggregator = advanced_market_aggregator
        self.trending_updater = trending_auto_updater
        self.max_cryptos_to_analyze = 15  # Réduit mais focus sur trending
        self.min_market_cap = 1_000_000    # $1M minimum (plus bas pour trending coins)
        self.min_volume_24h = 100_000      # $100K minimum (plus accessible)
        self.require_multiple_sources = True
        self.min_data_confidence = 0.7
        
        # Focus trending configuration
        self.trending_symbols = [
            # TOP 25 cryptomonnaies par market cap pour analyse technique complète
            'BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOGE', 'AVAX', 'DOT', 'MATIC', 
            'LINK', 'LTC', 'BCH', 'UNI', 'ATOM', 'FIL', 'APT', 'NEAR', 'VET', 'ICP',
            'HBAR', 'ALGO', 'ETC', 'MANA', 'SAND'
        ]  # Top 25 pour analyse patterns techniques
        self.focus_trending = True
        self.min_price_change_threshold = 3.0  # Focus sur les mouvements >3%
        self.volume_spike_multiplier = 2.0     # Volume >2x moyenne
        self.auto_update_trending = True       # Auto-update depuis Readdy
    
    async def initialize_trending_system(self):
        """Initialise le système de trending auto-update"""
        if self.auto_update_trending:
            logger.info("🔄 Starting trending auto-updater (6h cycle)")
            await self.trending_updater.start_auto_update()
            
            # Met à jour immédiatement les symboles trending
            await self._sync_trending_symbols()
    
    async def _sync_trending_symbols(self):
        """Synchronise les symboles trending avec l'auto-updater"""
        try:
            current_symbols = self.trending_updater.get_current_trending_symbols()
            if current_symbols:
                self.trending_symbols = current_symbols
                logger.info(f"📈 Trending symbols updated from crawler: {current_symbols}")
            else:
                # Fallback vers TOP 25 cryptos par market cap pour analyse technique complète
                top25_trending = [
                    'BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOGE', 'AVAX', 'DOT', 'MATIC',
                    'LINK', 'LTC', 'BCH', 'UNI', 'ATOM', 'FIL', 'APT', 'NEAR', 'VET', 'ICP', 
                    'HBAR', 'ALGO', 'ETC', 'MANA', 'SAND'
                ]
                self.trending_symbols = top25_trending
                logger.info(f"📈 Using TOP 25 crypto symbols for technical analysis: {top25_trending}")
        except Exception as e:
            logger.error(f"Error syncing trending symbols: {e}")
            # Fallback final vers top 10
            self.trending_symbols = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOGE', 'AVAX', 'DOT', 'MATIC']
            logger.info(f"📈 Using fallback top 10 symbols: {self.trending_symbols}")
    
    async def scan_opportunities(self) -> List[MarketOpportunity]:
        """Ultra professional trend-focused market scanning with auto-updated trends"""
        try:
            # Sync trending symbols if auto-update is enabled
            if self.auto_update_trending:
                await self._sync_trending_symbols()
            
            logger.info(f"Starting TREND-FOCUSED scan with symbols: {self.trending_symbols}")
            
            if self.focus_trending:
                # Get trending opportunities first
                trending_opportunities = await self._scan_trending_opportunities()
                logger.info(f"Found {len(trending_opportunities)} trending opportunities")
                
                # Get high-momentum opportunities
                momentum_opportunities = await self._scan_momentum_opportunities()
                logger.info(f"Found {len(momentum_opportunities)} momentum opportunities")
                
                # Combine and deduplicate
                all_opportunities = trending_opportunities + momentum_opportunities
                unique_opportunities = self._deduplicate_opportunities(all_opportunities)
                
            else:
                # Fallback to comprehensive scan
                market_responses = await self.market_aggregator.get_comprehensive_market_data(
                    limit=100,  # Reduced from 500
                    include_dex=True
                )
                unique_opportunities = self._convert_responses_to_opportunities(market_responses)
            
            # Sort by trending score
            sorted_opportunities = self._sort_by_trending_score(unique_opportunities)
            
            # Limit results for focused analysis
            final_opportunities = sorted_opportunities[:self.max_cryptos_to_analyze]
            
            logger.info(f"TREND-FOCUSED scan complete: {len(final_opportunities)} high-potential opportunities selected")
            
            return final_opportunities
            
        except Exception as e:
            logger.error(f"Error in trend-focused market scan: {e}")
            return []
    
    async def _scan_trending_opportunities(self) -> List[MarketOpportunity]:
        """Scan specifically for trending cryptocurrencies"""
        opportunities = []
        
        try:
            # Get CoinGecko trending
            market_responses = await self.market_aggregator.get_comprehensive_market_data(limit=100)
            
            for response in market_responses:
                symbol_base = response.symbol.replace('USDT', '').replace('USD', '')
                
                # Check if it's in our trending list
                is_trending = symbol_base.upper() in [s.upper() for s in self.trending_symbols]
                
                # Check for trending characteristics
                has_high_volatility = response.volatility > 0.05  # >5% volatility
                has_significant_move = abs(response.price_change_24h) > self.min_price_change_threshold
                has_volume_spike = response.volume_24h > 1_000_000  # Good volume
                
                if is_trending or (has_high_volatility and has_significant_move and has_volume_spike):
                    if self._passes_trending_filters(response):
                        opportunity = self._convert_response_to_opportunity(response)
                        # Boost trending score
                        opportunity.data_confidence = min(opportunity.data_confidence + 0.1, 1.0)
                        opportunities.append(opportunity)
                        
                        logger.info(f"TRENDING: {symbol_base} - {response.price_change_24h:.2f}% change, vol: ${response.volume_24h:,.0f}")
            
        except Exception as e:
            logger.error(f"Error scanning trending opportunities: {e}")
        
        return opportunities
    
    async def _scan_momentum_opportunities(self) -> List[MarketOpportunity]:
        """Scan for high-momentum opportunities (big movers)"""
        opportunities = []
        
        try:
            # Get market data focused on momentum
            market_responses = await self.market_aggregator.get_comprehensive_market_data(limit=200)
            
            # Sort by price change (both positive and negative momentum)
            sorted_responses = sorted(market_responses, 
                                    key=lambda x: abs(x.price_change_24h), 
                                    reverse=True)
            
            # Take top movers
            top_movers = sorted_responses[:20]
            
            for response in top_movers:
                if self._passes_momentum_filters(response):
                    opportunity = self._convert_response_to_opportunity(response)
                    opportunities.append(opportunity)
                    
                    logger.info(f"MOMENTUM: {response.symbol} - {response.price_change_24h:.2f}% change")
        
        except Exception as e:
            logger.error(f"Error scanning momentum opportunities: {e}")
        
        return opportunities
    
    def _passes_trending_filters(self, response: MarketDataResponse) -> bool:
        """Apply trending-specific filters"""
        # More lenient filters for trending coins
        if response.price <= 0:
            return False
        
        # Lower market cap threshold for trending
        if response.market_cap and response.market_cap < self.min_market_cap:
            return False
        
        # Minimum volume (lower for trending)
        if response.volume_24h < self.min_volume_24h:
            return False
        
        # Data confidence
        if response.confidence < 0.6:  # Lower threshold for trending
            return False
        
        # Skip obvious stablecoins
        symbol = response.symbol.upper()
        stablecoins = ['USDT', 'USDC', 'BUSD', 'DAI', 'TUSD']
        if any(stable in symbol for stable in stablecoins):
            return False
        
        return True
    
    def _passes_momentum_filters(self, response: MarketDataResponse) -> bool:
        """Apply momentum-specific filters"""
        # Must have significant price movement
        if abs(response.price_change_24h) < self.min_price_change_threshold:
            return False
        
        # Must have decent volume
        if response.volume_24h < 500_000:  # $500K minimum for momentum
            return False
        
        # Basic quality filters
        if response.price <= 0:
            return False
        
        # Confidence threshold
        if response.confidence < 0.7:
            return False
        
        return True
    
    def _convert_response_to_opportunity(self, response: MarketDataResponse) -> MarketOpportunity:
        """Convert MarketDataResponse to MarketOpportunity"""
        return MarketOpportunity(
            symbol=response.symbol,
            current_price=response.price,
            volume_24h=response.volume_24h,
            price_change_24h=response.price_change_24h,
            volatility=self._calculate_volatility(response.price_change_24h),
            market_cap=response.market_cap,
            market_cap_rank=response.market_cap_rank,
            data_sources=[response.source],
            data_confidence=response.confidence,
            timestamp=response.timestamp
        )
    
    def _convert_responses_to_opportunities(self, responses: List[MarketDataResponse]) -> List[MarketOpportunity]:
        """Convert multiple responses to opportunities"""
        opportunities = []
        for response in responses:
            if self._passes_professional_filters(response):
                opportunity = self._convert_response_to_opportunity(response)
                opportunities.append(opportunity)
        return opportunities
    
    def _deduplicate_opportunities(self, opportunities: List[MarketOpportunity]) -> List[MarketOpportunity]:
        """Remove duplicate opportunities by symbol"""
        seen_symbols = set()
        unique_opportunities = []
        
        for opp in opportunities:
            if opp.symbol not in seen_symbols:
                seen_symbols.add(opp.symbol)
                unique_opportunities.append(opp)
        
        return unique_opportunities
    
    def _sort_by_trending_score(self, opportunities: List[MarketOpportunity]) -> List[MarketOpportunity]:
        """Sort opportunities by trending score"""
        def trending_score(opp):
            score = 0
            
            # Price movement score (both directions valuable)
            score += abs(opp.price_change_24h) * 0.3
            
            # Volume score
            volume_score = min(opp.volume_24h / 10_000_000, 10) * 0.2  # Cap at 10
            score += volume_score
            
            # Volatility score (but not too much)
            volatility_score = min(opp.volatility * 100, 15) * 0.2  # Cap at 15%
            score += volatility_score
            
            # Data confidence score
            score += opp.data_confidence * 0.3
            
            # Trending symbol bonus
            symbol_base = opp.symbol.replace('USDT', '').replace('USD', '')
            if symbol_base.upper() in [s.upper() for s in self.trending_symbols]:
                score += 2.0  # Big bonus for trending symbols
            
            return score
        
        return sorted(opportunities, key=trending_score, reverse=True)
    
    def _passes_professional_filters(self, response: MarketDataResponse) -> bool:
        """Apply professional-grade filters to market data (same as before)"""
        # Price validation
        if response.price <= 0:
            return False
        
        # Market cap filter (more lenient for trending)
        if response.market_cap and response.market_cap < 500_000:  # $500K minimum
            return False
        
        # Volume filter
        if response.volume_24h < 50_000:  # $50K minimum for trending
            return False
        
        # Data confidence filter
        if response.confidence < 0.6:  # Lower for trending
            return False
        
        # Skip obvious stablecoins
        symbol = response.symbol.upper()
        stablecoins = ['USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'FRAX', 'LUSD']
        if any(stable in symbol for stable in stablecoins):
            return False
        
        return True
    
    def _calculate_volatility(self, price_change_24h: float) -> float:
        """Calculate volatility estimate from 24h price change"""
        return abs(price_change_24h) / 100.0

class UltraProfessionalIA1TechnicalAnalyst:
    def __init__(self):
        self.chat = get_ia1_chat()
        self.market_aggregator = advanced_market_aggregator
    
    async def analyze_opportunity(self, opportunity: MarketOpportunity) -> Optional[TechnicalAnalysis]:
        """Ultra professional technical analysis with OHLCV pattern pre-filtering"""
        try:
            logger.info(f"🔍 TECHNICAL PRE-FILTER: Checking {opportunity.symbol} for chartist patterns...")
            
            # Pré-filtrage technique avec OHLCV
            should_analyze, detected_pattern = await technical_pattern_detector.should_analyze_with_ia1(opportunity.symbol)
            
            if not should_analyze:
                logger.info(f"⚪ SKIPPED IA1: {opportunity.symbol} - No significant technical patterns detected")
                return None
            
            if detected_pattern:
                logger.info(f"✅ PATTERN DETECTED: {opportunity.symbol} - {detected_pattern.pattern_type.value} (strength: {detected_pattern.strength:.2f})")
            
            logger.info(f"🚀 IA1 analyzing {opportunity.symbol} - Technical filter PASSED")
            
            # Get additional historical data if needed - VRAIES données uniquement
            historical_data = await self._get_enhanced_historical_data(opportunity.symbol)
            
            if historical_data is None:
                logger.info(f"❌ SKIPPING IA1 for {opportunity.symbol} - No real OHLCV data available")
                return None  # Pas d'analyse sans vraies données
            
            # Calculate advanced technical indicators avec données étendues
            rsi = self._calculate_rsi(historical_data['Close'])
            macd_line, macd_signal, macd_histogram = self._calculate_macd(historical_data['Close'])
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(historical_data['Close'])
            
            # Debug logging pour vérifier les vraies valeurs calculées
            logger.info(f"🔢 {opportunity.symbol} - RSI: {rsi:.2f}, MACD: {macd_signal:.6f}, Data points: {len(historical_data)}")
            
            # Calculate Bollinger Band position
            current_price = opportunity.current_price
            if bb_upper > bb_lower:
                bb_position = (current_price - bb_middle) / (bb_upper - bb_middle)
            else:
                bb_position = 0
            
            # Get market sentiment from aggregator
            performance_stats = self.market_aggregator.get_performance_stats()
            
            # Create ultra professional analysis prompt
            market_cap_str = f"${opportunity.market_cap:,.0f}" if opportunity.market_cap else "N/A"
            
            prompt = f"""
            FAST TECHNICAL ANALYSIS - {opportunity.symbol}
            
            Price: ${opportunity.current_price:,.2f} | 24h: {opportunity.price_change_24h:.2f}% | Vol: ${opportunity.volume_24h:,.0f}
            Market Cap: {market_cap_str} | Rank: #{opportunity.market_cap_rank or 'N/A'}
            
            TECHNICAL INDICATORS (10-day):
            RSI: {rsi:.1f} | MACD: {macd_histogram:.4f} | BB Position: {bb_position:.2f}
            Support: ${self._find_support_levels(historical_data, current_price)[0] if self._find_support_levels(historical_data, current_price) else current_price * 0.95:.2f} | Resistance: ${self._find_resistance_levels(historical_data, current_price)[0] if self._find_resistance_levels(historical_data, current_price) else current_price * 1.05:.2f}
            
            Recent 5-day Close: {historical_data['Close'].tail().tolist()}
            
            Provide concise technical analysis with confidence score.
            """
            
            response = await self.chat.send_message(UserMessage(text=prompt))
            
            # Enrichir le raisonnement avec le pattern technique détecté
            reasoning = response[:1100] if response else "Ultra professional analysis with multi-source validation"
            if detected_pattern:
                direction_emoji = "📈" if detected_pattern.trading_direction == "long" else "📉" if detected_pattern.trading_direction == "short" else "⚖️"
                reasoning += f"\n\n🎯 TECHNICAL PATTERN: {detected_pattern.pattern_type.value}"
                reasoning += f"\n{direction_emoji} Direction: {detected_pattern.trading_direction.upper()} (strength: {detected_pattern.strength:.2f})"
                reasoning += f"\nTrend Duration: {detected_pattern.trend_duration_days} days"
                reasoning += f"\nEntry: ${detected_pattern.entry_price:.2f} → Target: ${detected_pattern.target_price:.2f}"
            
            # Create ultra professional analysis avec validation JSON
            analysis_data = {
                "rsi": rsi,
                "macd_signal": macd_signal,
                "bollinger_position": bb_position,
                "fibonacci_level": self._calculate_fibonacci_retracement(historical_data),
                "support_levels": self._find_support_levels(historical_data, current_price),
                "resistance_levels": self._find_resistance_levels(historical_data, current_price),
                "patterns_detected": self._detect_advanced_patterns(historical_data),
                "analysis_confidence": self._calculate_analysis_confidence(
                    rsi, macd_histogram, bb_position, opportunity.volatility, opportunity.data_confidence
                ),
                "ia1_reasoning": reasoning,
                "market_sentiment": self._determine_market_sentiment(opportunity),
                "data_sources": opportunity.data_sources
            }
            
            # Valide et nettoie les données pour éviter les erreurs JSON
            validated_data = self._validate_analysis_data(analysis_data)
            
            # Ajuster la confiance basée sur le pattern technique
            if detected_pattern and detected_pattern.strength > 0.7:
                validated_data["analysis_confidence"] = min(validated_data["analysis_confidence"] + 0.1, 0.98)
                if detected_pattern.pattern_type.value not in validated_data["patterns_detected"]:
                    validated_data["patterns_detected"].insert(0, detected_pattern.pattern_type.value)
            
            return TechnicalAnalysis(
                symbol=opportunity.symbol,
                **validated_data
            )
            
        except Exception as e:
            logger.error(f"IA1 ultra analysis error for {opportunity.symbol}: {e}")
            return self._create_fallback_analysis(opportunity)
    
    async def _get_enhanced_historical_data(self, symbol: str, days: int = 50) -> Optional[pd.DataFrame]:
        """Get enhanced historical data - VRAIES données seulement, pas de synthétique"""
        try:
            # Utilise le système OHLCV du technical_pattern_detector (Binance, CoinGecko, etc.)
            real_data = await technical_pattern_detector._get_ohlcv_data(symbol)
            
            if real_data is not None and len(real_data) >= 26:  # Minimum pour MACD
                logger.info(f"✅ IA1 using REAL OHLCV data for {symbol}: {len(real_data)} days")
                return real_data.tail(days)  # Garde les X derniers jours
            else:
                logger.warning(f"❌ IA1 REJECTING {symbol} - insufficient real data: {len(real_data) if real_data is not None else 0} days")
                return None  # Pas de données synthétiques
                
        except Exception as e:
            logger.warning(f"❌ IA1 REJECTING {symbol} - OHLCV API error: {e}")
            return None  # Pas de fallback synthétique
    
    # Note: Synthetic data generation removed - using REAL OHLCV data only
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            if len(prices) < period + 1:
                return 50.0  # Neutral RSI
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            rsi_value = float(rsi.iloc[-1])
            
            # Ensure RSI is within valid range
            if pd.isna(rsi_value) or not (0 <= rsi_value <= 100):
                return 50.0
            
            return round(rsi_value, 2)
        except:
            return 50.0
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD indicator"""
        try:
            if len(prices) < slow + signal:
                return 0.0, 0.0, 0.0  # Neutral MACD
            
            exp1 = prices.ewm(span=fast).mean()
            exp2 = prices.ewm(span=slow).mean()
            macd_line = exp1 - exp2
            macd_signal = macd_line.ewm(span=signal).mean()
            macd_histogram = macd_line - macd_signal
            
            line_val = float(macd_line.iloc[-1])
            signal_val = float(macd_signal.iloc[-1])
            hist_val = float(macd_histogram.iloc[-1])
            
            # Ensure values are valid
            line_val = line_val if not pd.isna(line_val) else 0.0
            signal_val = signal_val if not pd.isna(signal_val) else 0.0
            hist_val = hist_val if not pd.isna(hist_val) else 0.0
            
            return round(line_val, 6), round(signal_val, 6), round(hist_val, 6)
        except:
            return 0.0, 0.0, 0.0
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2):
        """Calculate Bollinger Bands"""
        try:
            if len(prices) < period:
                current = float(prices.iloc[-1]) if len(prices) > 0 else 100.0
                return current * 1.02, current, current * 0.98  # Default bands
            
            middle = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            
            upper_val = float(upper.iloc[-1])
            middle_val = float(middle.iloc[-1])
            lower_val = float(lower.iloc[-1])
            
            # Ensure values are valid
            if pd.isna(upper_val) or pd.isna(middle_val) or pd.isna(lower_val):
                current = float(prices.iloc[-1])
                return current * 1.02, current, current * 0.98
            
            return round(upper_val, 2), round(middle_val, 2), round(lower_val, 2)
        except:
            current_price = float(prices.iloc[-1]) if len(prices) > 0 else 100.0
            return current_price * 1.02, current_price, current_price * 0.98
    
    def _ensure_json_safe(self, value, default=0.0):
        """S'assure qu'une valeur est safe pour la sérialisation JSON"""
        try:
            if value is None:
                return default
            if isinstance(value, (list, tuple)):
                return [self._ensure_json_safe(v, default) for v in value]
            if isinstance(value, dict):
                return {k: self._ensure_json_safe(v, default) for k, v in value.items()}
            if isinstance(value, str):
                return value
            
            # Vérifie les valeurs numériques
            if pd.isna(value) or not pd.notna(value):
                return default
            if abs(value) == float('inf') or abs(value) > 1e10:
                return default
            if not isinstance(value, (int, float)):
                return default
                
            return float(value)
        except:
            return default

    def _validate_analysis_data(self, analysis_data: dict) -> dict:
        """Valide et nettoie toutes les données d'analyse pour JSON"""
        try:
            cleaned_data = {}
            
            # Validation des champs numériques
            cleaned_data["rsi"] = self._ensure_json_safe(analysis_data.get("rsi"), 50.0)
            cleaned_data["macd_signal"] = self._ensure_json_safe(analysis_data.get("macd_signal"), 0.0)
            cleaned_data["bollinger_position"] = self._ensure_json_safe(analysis_data.get("bollinger_position"), 0.0)
            cleaned_data["fibonacci_level"] = self._ensure_json_safe(analysis_data.get("fibonacci_level"), 0.618)
            cleaned_data["analysis_confidence"] = self._ensure_json_safe(analysis_data.get("analysis_confidence"), 0.5)
            
            # Validation des listes
            cleaned_data["support_levels"] = self._ensure_json_safe(analysis_data.get("support_levels", []), [])
            cleaned_data["resistance_levels"] = self._ensure_json_safe(analysis_data.get("resistance_levels", []), [])
            cleaned_data["patterns_detected"] = analysis_data.get("patterns_detected", ["No patterns detected"])
            
            # Validation des strings
            cleaned_data["ia1_reasoning"] = str(analysis_data.get("ia1_reasoning", "Analysis completed"))
            cleaned_data["market_sentiment"] = str(analysis_data.get("market_sentiment", "neutral"))
            cleaned_data["data_sources"] = analysis_data.get("data_sources", ["internal"])
            
            return cleaned_data
        except Exception as e:
            logger.error(f"Error validating analysis data: {e}")
            return {
                "rsi": 50.0,
                "macd_signal": 0.0,
                "bollinger_position": 0.0,
                "fibonacci_level": 0.618,
                "support_levels": [],
                "resistance_levels": [],
                "patterns_detected": ["Analysis validation error"],
                "analysis_confidence": 0.5,
                "ia1_reasoning": "Analysis completed with data validation",
                "market_sentiment": "neutral",
                "data_sources": ["internal"]
            }

    def _calculate_analysis_confidence(self, rsi: float, macd_histogram: float, bb_position: float, volatility: float, data_confidence: float) -> float:
        """Calcule la confiance de l'analyse technique"""
        try:
            # Initialise la confiance de base
            confidence = 0.5
            
            # RSI dans des zones significatives
            rsi_safe = self._ensure_json_safe(rsi, 50.0)
            if rsi_safe < 30 or rsi_safe > 70:
                confidence += 0.15
            elif 35 < rsi_safe < 65:
                confidence += 0.1
            
            # MACD histogram strength
            macd_safe = self._ensure_json_safe(macd_histogram, 0.0)
            if abs(macd_safe) > 0.01:
                confidence += 0.1
            
            # Bollinger bands position
            bb_safe = self._ensure_json_safe(bb_position, 0.0)
            if abs(bb_safe) > 0.7:  # Near bands
                confidence += 0.1
            
            # Volatilité appropriée
            vol_safe = self._ensure_json_safe(volatility, 0.02)
            if 0.01 < vol_safe < 0.05:  # Sweet spot volatility
                confidence += 0.1
            
            # Data confidence from aggregator
            data_conf_safe = self._ensure_json_safe(data_confidence, 0.5)
            confidence += data_conf_safe * 0.2
            
            # Ensure confidence is within bounds
            return self._ensure_json_safe(confidence, 0.5)
            
        except Exception as e:
            logger.debug(f"Error calculating analysis confidence: {e}")
            return 0.5

    def _determine_market_sentiment(self, opportunity) -> str:
        """Détermine le sentiment du marché"""
        try:
            change = self._ensure_json_safe(opportunity.price_change_24h, 0.0)
            
            if change > 5:
                return "very_bullish"
            elif change > 2:
                return "bullish"
            elif change > -2:
                return "neutral"
            elif change > -5:
                return "bearish"
            else:
                return "very_bearish"
        except:
            return "neutral"

# Global instances

    def _calculate_fibonacci_retracement(self, historical_data: pd.DataFrame) -> float:
        """Calcule le niveau de retracement Fibonacci actuel"""
        try:
            if len(historical_data) < 20:
                return 0.618  # Niveau par défaut
            
            high = historical_data['High'].max()
            low = historical_data['Low'].min()
            current = historical_data['Close'].iloc[-1]
            
            if high == low:  # Évite division par zéro
                return 0.618
            
            fib_level = (current - low) / (high - low)
            
            # S'assure que la valeur est valide pour JSON
            if not (0 <= fib_level <= 2):  # Valeur raisonnable
                return 0.618
            
            return round(fib_level, 3)
        except:
            return 0.618
    
    def _find_support_levels(self, df: pd.DataFrame, current_price: float) -> List[float]:
        """Trouve les niveaux de support clés"""
        try:
            if len(df) < 10:
                return [current_price * 0.95, current_price * 0.90]
            
            lows = df['Low'].rolling(5).min().dropna().unique()
            supports = [float(low) for low in lows if low < current_price and low > 0]
            supports = sorted(supports, reverse=True)
            
            # Limite à 3 niveaux et s'assure qu'ils sont valides
            valid_supports = []
            for support in supports[:3]:
                if support > 0 and support < current_price * 1.5:  # Valeurs raisonnables
                    valid_supports.append(round(support, 2))
            
            return valid_supports if valid_supports else [current_price * 0.95]
        except:
            return [current_price * 0.95]
    
    def _find_resistance_levels(self, df: pd.DataFrame, current_price: float) -> List[float]:
        """Trouve les niveaux de résistance clés"""
        try:
            if len(df) < 10:
                return [current_price * 1.05, current_price * 1.10]
            
            highs = df['High'].rolling(5).max().dropna().unique()
            resistances = [float(high) for high in highs if high > current_price and high > 0]
            resistances = sorted(resistances)
            
            # Limite à 3 niveaux et s'assure qu'ils sont valides
            valid_resistances = []
            for resistance in resistances[:3]:
                if resistance > current_price and resistance < current_price * 2:  # Valeurs raisonnables
                    valid_resistances.append(round(resistance, 2))
            
            return valid_resistances if valid_resistances else [current_price * 1.05]
        except:
            return [current_price * 1.05]
    
    def _detect_advanced_patterns(self, historical_data: pd.DataFrame) -> List[str]:
        """Detect advanced chart patterns"""
        patterns = []
        try:
            prices = historical_data['Close']
            
            if len(prices) >= 20:
                # Trend analysis
                short_ma = prices.rolling(5).mean()
                long_ma = prices.rolling(20).mean()
                
                if short_ma.iloc[-1] > long_ma.iloc[-1]:
                    if short_ma.iloc[-5] <= long_ma.iloc[-5]:
                        patterns.append("Golden Cross Formation")
                    else:
                        patterns.append("Bullish Trend Continuation")
                else:
                    if short_ma.iloc[-5] >= long_ma.iloc[-5]:
                        patterns.append("Death Cross Formation")
                    else:
                        patterns.append("Bearish Trend Continuation")
                
                # Volatility patterns
                volatility = prices.pct_change().rolling(10).std()
                if volatility.iloc[-1] > volatility.quantile(0.8):
                    patterns.append("High Volatility Breakout")
                elif volatility.iloc[-1] < volatility.quantile(0.2):
                    patterns.append("Low Volatility Consolidation")
                
                # Volume-price analysis
                if 'Volume' in historical_data.columns:
                    volume = historical_data['Volume']
                    if volume.iloc[-1] > volume.rolling(20).mean().iloc[-1] * 1.5:
                        patterns.append("Volume Spike Confirmation")
            
        except Exception as e:
            logger.debug(f"Pattern detection error: {e}")
            patterns = ["Advanced Pattern Analysis"]
        
        return patterns
    
    def _calculate_analysis_confidence(self, rsi: float, macd_histogram: float, 
                                     bb_position: float, volatility: float, 
                                     data_confidence: float) -> float:
        """Calculate comprehensive analysis confidence"""
        confidence = 0.5  # Base confidence
        
        # Data quality boost
        confidence += data_confidence * 0.2
        
        # Technical indicator alignment
        signal_strength = 0
        
        # RSI signals
        if rsi < 25 or rsi > 75:  # Strong oversold/overbought
            signal_strength += 0.15
        elif rsi < 35 or rsi > 65:  # Moderate levels
            signal_strength += 0.1
        
        # MACD momentum
        if abs(macd_histogram) > 0.002:  # Strong momentum
            signal_strength += 0.15
        elif abs(macd_histogram) > 0.001:  # Moderate momentum
            signal_strength += 0.1
        
        # Bollinger Band position
        if abs(bb_position) > 0.7:  # Near bands
            signal_strength += 0.1
        
        # Volatility consideration
        if 0.02 <= volatility <= 0.06:  # Optimal volatility range
            signal_strength += 0.05
        elif volatility > 0.1:  # Too volatile
            signal_strength -= 0.05
        
        confidence += signal_strength
        
        return min(confidence, 0.98)  # Cap at 98%
    
    def _determine_market_sentiment(self, opportunity: MarketOpportunity) -> str:
        """Determine market sentiment based on opportunity data"""
        if opportunity.price_change_24h > 5:
            return "bullish"
        elif opportunity.price_change_24h < -5:
            return "bearish"
        elif opportunity.volatility > 0.08:
            return "volatile"
        else:
            return "neutral"
    
    def _create_fallback_analysis(self, opportunity: MarketOpportunity) -> TechnicalAnalysis:
        """Create fallback analysis when AI fails"""
        return TechnicalAnalysis(
            symbol=opportunity.symbol,
            rsi=50.0,
            macd_signal=0.0,
            bollinger_position=0.0,
            fibonacci_level=0.5,
            support_levels=[opportunity.current_price * 0.95],
            resistance_levels=[opportunity.current_price * 1.05],
            patterns_detected=["Ultra Professional Analysis Pending"],
            analysis_confidence=0.7,
            ia1_reasoning=f"Fallback ultra professional analysis for {opportunity.symbol}",
            market_sentiment="neutral",
            data_sources=opportunity.data_sources
        )

class UltraProfessionalIA2DecisionAgent:
    def __init__(self):
        self.chat = get_ia2_chat()
        self.market_aggregator = advanced_market_aggregator
        self.bingx_engine = bingx_trading_engine
        self.live_trading_enabled = True  # Set to False for simulation only
        self.max_risk_per_trade = 0.02  # 2% risk per trade
    
    async def make_decision(self, opportunity: MarketOpportunity, analysis: TechnicalAnalysis) -> TradingDecision:
        """Ultra professional trading decision with BingX live trading integration"""
        try:
            logger.info(f"IA2 making ultra professional decision for {opportunity.symbol}")
            
            # Get BingX account info for risk management
            account_balance = await self._get_account_balance()
            
            # Get aggregator performance stats
            perf_stats = self.market_aggregator.get_performance_stats()
            
            # Format market cap safely
            market_cap_str = f"${opportunity.market_cap:,.0f}" if opportunity.market_cap else "N/A"
            
            prompt = f"""
            ULTRA PROFESSIONAL TRADING DECISION - BingX Live Trading Integration
            
            LIVE TRADING ACCOUNT STATUS:
            - Available Balance: ${account_balance:,.2f} USDT
            - Live Trading: {'ENABLED' if self.live_trading_enabled else 'SIMULATION ONLY'}
            - Max Risk Per Trade: {self.max_risk_per_trade * 100}% ({self.max_risk_per_trade * account_balance:,.2f} USDT)
            
            MARKET DATA VALIDATION:
            Symbol: {opportunity.symbol}
            Current Price: ${opportunity.current_price:,.2f}
            Market Cap: {market_cap_str} (Rank #{opportunity.market_cap_rank or 'N/A'})
            24h Volume: ${opportunity.volume_24h:,.0f}
            24h Change: {opportunity.price_change_24h:.2f}%
            Volatility: {opportunity.volatility:.4f}
            
            DATA SOURCE VALIDATION:
            - Primary Sources: {', '.join(opportunity.data_sources)}
            - Data Confidence: {opportunity.data_confidence:.2f}
            - Analysis Sources: {', '.join(analysis.data_sources)}
            - Market Aggregator Success Rate: {perf_stats.get('success_rate', 0):.2f}
            - Active API Endpoints: {len(perf_stats.get('api_endpoints', []))}
            
            ULTRA PROFESSIONAL IA1 ANALYSIS:
            - RSI: {analysis.rsi:.2f}
            - MACD Signal: {analysis.macd_signal:.6f}
            - Bollinger Position: {analysis.bollinger_position:.2f}
            - Fibonacci Level: {analysis.fibonacci_level:.3f}
            - Support Levels: {analysis.support_levels}
            - Resistance Levels: {analysis.resistance_levels}
            - Patterns: {', '.join(analysis.patterns_detected)}
            - IA1 Confidence: {analysis.analysis_confidence:.2f}
            - Market Sentiment: {analysis.market_sentiment}
            - IA1 Reasoning: {analysis.ia1_reasoning}
            
            LIVE TRADING REQUIREMENTS:
            - Minimum position size: $10 USDT
            - Maximum position size: {self.max_risk_per_trade * account_balance:,.2f} USDT
            - Required confidence for live execution: >0.75
            - Multi-source validation required for trades >$100
            
            Make ultra professional trading decision for LIVE EXECUTION on BingX.
            """
            
            response = await self.chat.send_message(UserMessage(text=prompt))
            
            # Parse LLM JSON response
            llm_decision = await self._parse_llm_response(response)
            
            # Generate ultra professional decision with live trading considerations
            decision_logic = await self._evaluate_live_trading_decision(opportunity, analysis, perf_stats, account_balance, llm_decision)
            
            # Create trading decision
            decision = TradingDecision(
                symbol=opportunity.symbol,
                signal=decision_logic["signal"],
                confidence=decision_logic["confidence"],
                entry_price=opportunity.current_price,
                stop_loss=decision_logic["stop_loss"],
                take_profit_1=decision_logic["tp1"],
                take_profit_2=decision_logic["tp2"],
                take_profit_3=decision_logic["tp3"],
                position_size=decision_logic["position_size"],
                risk_reward_ratio=decision_logic["risk_reward"],
                ia1_analysis_id=analysis.id,
                ia2_reasoning=decision_logic["reasoning"][:1500] if decision_logic["reasoning"] else "IA2 analysis completed",
                status=TradingStatus.PENDING
            )
            
            # Execute live trading if enabled and signal is not HOLD
            if self.live_trading_enabled and decision.signal != SignalType.HOLD:
                await self._execute_live_trade(decision)
            
            logger.info(f"IA2 ultra professional decision for {opportunity.symbol}: {decision.signal} (confidence: {decision.confidence:.2f})")
            return decision
            
        except Exception as e:
            logger.error(f"IA2 ultra decision error for {opportunity.symbol}: {e}")
            return self._create_fallback_decision(opportunity, analysis)
    
    async def _get_account_balance(self) -> float:
        """Get current account balance from BingX"""
        try:
            balances = await self.bingx_engine.get_account_balance()
            usdt_balance = next((balance for balance in balances if balance.asset == 'USDT'), None)
            return usdt_balance.available if usdt_balance else 0.0
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return 1000.0  # Fallback balance
    
    async def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse IA2 LLM JSON response with fallback"""
        if not response:
            return {}
            
        try:
            # Try to parse JSON response
            import json
            # Clean response - sometimes LLM adds extra text
            response_clean = response.strip()
            if response_clean.startswith('```json'):
                response_clean = response_clean.replace('```json', '').replace('```', '').strip()
            elif response_clean.startswith('```'):
                response_clean = response_clean.replace('```', '').strip()
            
            # Find JSON in response if embedded in text
            start_idx = response_clean.find('{')
            end_idx = response_clean.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                response_clean = response_clean[start_idx:end_idx]
            
            parsed = json.loads(response_clean)
            
            # Validate expected fields
            if not isinstance(parsed, dict):
                logger.warning("IA2 LLM response is not a dict, using fallback")
                return {}
                
            return parsed
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse IA2 LLM response: {e}, using raw response")
            return {"reasoning": response[:1000] if response else "IA2 LLM response parsing failed"}
    
    async def _evaluate_live_trading_decision(self, 
                                            opportunity: MarketOpportunity, 
                                            analysis: TechnicalAnalysis, 
                                            perf_stats: Dict,
                                            account_balance: float,
                                            llm_decision: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate trading decision with live trading risk management"""
        
        signal = SignalType.HOLD
        base_confidence = (analysis.analysis_confidence + opportunity.data_confidence) / 2
        confidence = max(base_confidence, 0.5)  # Start with minimum 50% base confidence
        
        # Start with LLM reasoning if available
        llm_reasoning = ""
        llm_confidence_boost = 0.0
        if llm_decision:
            llm_reasoning = llm_decision.get("reasoning", "")
            llm_confidence = llm_decision.get("confidence", 0.0)
            if 0.5 <= llm_confidence <= 1.0:  # Valid LLM confidence
                llm_confidence_boost = min((llm_confidence - 0.5) * 0.3, 0.25)  # Up to 0.25 boost
                confidence += llm_confidence_boost
                
        reasoning = f"IA2 Decision Analysis: {llm_reasoning[:500]} " if llm_reasoning else "Ultra professional live trading analysis: "
        
        # LIVE TRADING GATES (More reasonable approach)
        confidence_penalties = 0
        
        # Minimum balance gate
        if account_balance < 50:  # Minimum $50 USDT
            reasoning += "Insufficient account balance for live trading. "
            confidence = max(confidence * 0.6, 0.5)  # Maintain 50% minimum even with penalty
            return self._create_hold_decision(reasoning, confidence, opportunity.current_price)
        
        # Data quality assessment (adjust rather than penalize heavily)
        if opportunity.data_confidence < 0.6:  # Very low data confidence
            reasoning += "Low data confidence - conservative approach. "
            confidence_penalties += 0.1
        elif opportunity.data_confidence < 0.75:  # Moderate data confidence
            reasoning += "Moderate data confidence. "
            confidence_penalties += 0.05
        
        if analysis.analysis_confidence < 0.6:  # Very low analysis confidence
            reasoning += "Low analysis confidence - conservative approach. "
            confidence_penalties += 0.1
        elif analysis.analysis_confidence < 0.75:  # Moderate analysis confidence  
            reasoning += "Moderate analysis confidence. "
            confidence_penalties += 0.05
        
        # Apply confidence penalties (subtract rather than multiply)
        confidence = max(confidence - confidence_penalties, 0.5)  # Maintain 50% minimum
        
        # Multi-source validation bonus (reward rather than penalize)
        if len(opportunity.data_sources) >= 3:
            confidence += 0.05
            reasoning += "Multiple data sources confirmed. "
        elif len(opportunity.data_sources) >= 2:
            confidence += 0.03
            reasoning += "Dual source validation. "
        
        # Market conditions assessment
        if opportunity.volatility > 0.25:  # Very high volatility
            reasoning += "Extreme market volatility - high risk/reward. "
            confidence -= 0.05
        elif opportunity.volatility > 0.15:  # High volatility
            reasoning += "High market volatility. "
            confidence -= 0.02
        
        # CRITICAL: Enforce 50% minimum confidence after ALL penalties
        confidence = max(confidence, 0.5)
        
        # Enhanced signal scoring for live trading
        bullish_signals = 0
        bearish_signals = 0
        signal_strength = 0
        
        # RSI analysis (more conservative for live trading)
        if analysis.rsi < 20:  # Extremely oversold
            bullish_signals += 4
            signal_strength += 0.4
            reasoning += "RSI extremely oversold - strong live buy signal. "
        elif analysis.rsi < 30:
            bullish_signals += 2
            signal_strength += 0.25
            reasoning += "RSI oversold - live buy signal. "
        elif analysis.rsi > 80:  # Extremely overbought
            bearish_signals += 4
            signal_strength += 0.4
            reasoning += "RSI extremely overbought - strong live sell signal. "
        elif analysis.rsi > 70:
            bearish_signals += 2
            signal_strength += 0.25
            reasoning += "RSI overbought - live sell signal. "
        
        # MACD analysis (enhanced for live trading)
        if analysis.macd_signal > 0.01:  # Strong bullish momentum
            bullish_signals += 3
            signal_strength += 0.3
            reasoning += "Strong MACD bullish momentum - live trading confirmed. "
        elif analysis.macd_signal > 0:
            bullish_signals += 1
            signal_strength += 0.15
            reasoning += "MACD bullish momentum. "
        elif analysis.macd_signal < -0.01:  # Strong bearish momentum
            bearish_signals += 3
            signal_strength += 0.3
            reasoning += "Strong MACD bearish momentum - live short confirmed. "
        elif analysis.macd_signal < 0:
            bearish_signals += 1
            signal_strength += 0.15
            reasoning += "MACD bearish momentum. "
        
        # Volume validation (critical for live trading)
        if opportunity.volume_24h > 50_000_000:  # High volume for safety
            signal_strength += 0.2
            reasoning += "High volume validation for live trading. "
        elif opportunity.volume_24h < 1_000_000:  # Too low volume
            signal_strength -= 0.3
            reasoning += "Low volume - risky for live trading. "
        
        # Pattern confirmation
        bullish_patterns = ["Golden Cross", "Bullish", "Breakout", "Support", "Bounce"]
        bearish_patterns = ["Death Cross", "Bearish", "Breakdown", "Resistance", "Rejection"]
        
        for pattern in analysis.patterns_detected:
            if any(bp in pattern for bp in bullish_patterns):
                bullish_signals += 2
                signal_strength += 0.15
                reasoning += f"Bullish pattern confirmed: {pattern}. "
            elif any(bp in pattern for bp in bearish_patterns):
                bearish_signals += 2
                signal_strength += 0.15
                reasoning += f"Bearish pattern confirmed: {pattern}. "
        
        # Live trading decision logic (more balanced thresholds)
        net_signals = bullish_signals - bearish_signals
        
        # Incorporate LLM decision if available
        llm_signal_boost = 0
        if llm_decision:
            llm_signal = llm_decision.get("signal", "").lower()
            if llm_signal in ["long", "buy"]:
                llm_signal_boost = 2
                reasoning += "LLM recommends LONG position. "
            elif llm_signal in ["short", "sell"]:
                llm_signal_boost = -2
                reasoning += "LLM recommends SHORT position. "
        
        net_signals += llm_signal_boost
        
        # More reasonable trading thresholds based on industry standards
        if net_signals >= 4 and confidence > 0.65 and signal_strength > 0.5:  # Strong signals
            signal = SignalType.LONG
            confidence = min(confidence + 0.1, 0.95)
            reasoning += "LIVE LONG: Strong bullish signals confirmed for live execution. "
        elif net_signals >= 2 and confidence > 0.55 and signal_strength > 0.35:  # Moderate signals
            signal = SignalType.LONG
            confidence = min(confidence + 0.05, 0.80)
            reasoning += "LIVE LONG: Moderate bullish signals for live execution. "
        elif net_signals <= -4 and confidence > 0.65 and signal_strength > 0.5:  # Strong bearish
            signal = SignalType.SHORT
            confidence = min(confidence + 0.1, 0.95)
            reasoning += "LIVE SHORT: Strong bearish signals confirmed for live execution. "
        elif net_signals <= -2 and confidence > 0.55 and signal_strength > 0.35:  # Moderate bearish
            signal = SignalType.SHORT
            confidence = min(confidence + 0.05, 0.80)
            reasoning += "LIVE SHORT: Moderate bearish signals for live execution. "
        else:
            signal = SignalType.HOLD
            reasoning += f"LIVE HOLD: Signals below threshold for live trading (net: {net_signals}, strength: {signal_strength:.2f}, conf: {confidence:.2f}). "
        
        # Calculate live trading levels with enhanced risk management
        current_price = opportunity.current_price
        atr_estimate = current_price * max(opportunity.volatility, 0.02)
        
        if signal == SignalType.LONG:
            # Conservative stop-loss for live trading
            stop_loss_distance = max(atr_estimate * 2, current_price * 0.03)  # Min 3% stop
            stop_loss = current_price - stop_loss_distance
            
            # Conservative take-profits
            tp1 = current_price + (stop_loss_distance * 2)  # 2:1 R:R minimum
            tp2 = current_price + (stop_loss_distance * 3)  # 3:1 R:R
            tp3 = current_price + (stop_loss_distance * 4)  # 4:1 R:R
            
        elif signal == SignalType.SHORT:
            # Conservative stop-loss for live trading
            stop_loss_distance = max(atr_estimate * 2, current_price * 0.03)  # Min 3% stop
            stop_loss = current_price + stop_loss_distance
            
            # Conservative take-profits
            tp1 = current_price - (stop_loss_distance * 2)  # 2:1 R:R minimum
            tp2 = current_price - (stop_loss_distance * 3)  # 3:1 R:R
            tp3 = current_price - (stop_loss_distance * 4)  # 4:1 R:R
            
        else:
            stop_loss = current_price
            tp1 = tp2 = tp3 = current_price
        
        # Live trading risk-reward calculation
        if signal != SignalType.HOLD:
            risk = abs(current_price - stop_loss)
            reward = abs(tp1 - current_price)
            risk_reward = reward / risk if risk > 0 else 1.0
            
            # Minimum 1.2:1 R:R for live trading (more accessible)
            if risk_reward < 1.2:
                signal = SignalType.HOLD
                reasoning += "Risk-reward ratio too low for live trading (min 1.2:1 required). "
                confidence = max(confidence * 0.9, 0.5)  # Maintain 50% minimum
        else:
            risk_reward = 1.0
        
        # Live trading position sizing (more conservative)
        if signal != SignalType.HOLD:
            # Calculate position size based on risk
            risk_amount = account_balance * self.max_risk_per_trade
            stop_distance = abs(current_price - stop_loss)
            calculated_quantity = risk_amount / stop_distance
            
            # Apply additional safety limits
            max_position_value = account_balance * 0.3  # Max 30% of balance
            max_quantity = max_position_value / current_price
            
            position_size_percentage = min(
                (calculated_quantity * current_price) / account_balance,
                0.05  # Max 5% for live trading
            )
        else:
            position_size_percentage = 0.0
        
        return {
            "signal": signal,
            "confidence": confidence,
            "stop_loss": stop_loss,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "position_size": position_size_percentage,
            "risk_reward": risk_reward,
            "reasoning": reasoning,
            "signal_strength": signal_strength,
            "net_signals": net_signals,
            "live_trading_ready": signal != SignalType.HOLD and confidence > 0.75
        }
    
    def _create_hold_decision(self, reasoning: str, confidence: float, current_price: float) -> Dict[str, Any]:
        """Create a HOLD decision"""
        return {
            "signal": SignalType.HOLD,
            "confidence": confidence,
            "stop_loss": current_price,
            "tp1": current_price,
            "tp2": current_price,
            "tp3": current_price,
            "position_size": 0.0,
            "risk_reward": 1.0,
            "reasoning": reasoning,
            "signal_strength": 0.0,
            "net_signals": 0,
            "live_trading_ready": False
        }
    
    async def _execute_live_trade(self, decision: TradingDecision):
        """Execute live trade on BingX"""
        try:
            if decision.signal == SignalType.HOLD:
                return
            
            logger.info(f"Executing LIVE TRADE on BingX: {decision.signal} {decision.symbol}")
            
            # Get account balance for position sizing
            account_balance = await self._get_account_balance()
            
            # Calculate actual quantity
            position_value = account_balance * decision.position_size
            quantity = position_value / decision.entry_price
            
            # Minimum quantity check
            if quantity < 0.001:  # Minimum 0.001 for most futures
                logger.warning(f"Position size too small for live trading: {quantity}")
                decision.bingx_status = "REJECTED_MIN_SIZE"
                return
            
            # Set leverage (default 10x for futures)
            await self.bingx_engine.set_leverage(decision.symbol, 10)
            
            # Determine order side
            side = BingXOrderSide.BUY if decision.signal == SignalType.LONG else BingXOrderSide.SELL
            position_side = BingXPositionSide.LONG if decision.signal == SignalType.LONG else BingXPositionSide.SHORT
            
            # Place market order
            order = await self.bingx_engine.place_order(
                symbol=decision.symbol,
                side=side,
                order_type=BingXOrderType.MARKET,
                quantity=quantity,
                position_side=position_side
            )
            
            if order:
                # Update decision with BingX order info
                decision.bingx_order_id = order.order_id
                decision.actual_entry_price = order.price or decision.entry_price
                decision.actual_quantity = order.executed_qty
                decision.bingx_status = order.status
                decision.status = TradingStatus.EXECUTED
                
                logger.info(f"LIVE ORDER EXECUTED: {order.order_id} - {side} {quantity:.6f} {decision.symbol}")
                
                # Place stop-loss and take-profit orders
                await self._place_stop_orders(decision, quantity, position_side)
                
            else:
                decision.bingx_status = "ORDER_FAILED"
                decision.status = TradingStatus.REJECTED
                logger.error(f"Failed to execute live order for {decision.symbol}")
                
        except Exception as e:
            logger.error(f"Live trading execution error: {e}")
            decision.bingx_status = f"ERROR: {str(e)}"
            decision.status = TradingStatus.REJECTED
    
    async def _place_stop_orders(self, decision: TradingDecision, quantity: float, position_side: BingXPositionSide):
        """Place stop-loss and take-profit orders"""
        try:
            # Determine opposite side for closing orders
            close_side = BingXOrderSide.SELL if decision.signal == SignalType.LONG else BingXOrderSide.BUY
            
            # Place stop-loss order
            if decision.stop_loss != decision.entry_price:
                sl_order = await self.bingx_engine.place_order(
                    symbol=decision.symbol,
                    side=close_side,
                    order_type=BingXOrderType.STOP_MARKET,
                    quantity=quantity,
                    stop_price=decision.stop_loss,
                    position_side=position_side
                )
                
                if sl_order:
                    logger.info(f"Stop-loss order placed: {sl_order.order_id} at {decision.stop_loss}")
            
            # Place take-profit order (first level)
            if decision.take_profit_1 != decision.entry_price:
                tp_order = await self.bingx_engine.place_order(
                    symbol=decision.symbol,
                    side=close_side,
                    order_type=BingXOrderType.TAKE_PROFIT_MARKET,
                    quantity=quantity * 0.5,  # Close 50% at first TP
                    stop_price=decision.take_profit_1,
                    position_side=position_side
                )
                
                if tp_order:
                    logger.info(f"Take-profit order placed: {tp_order.order_id} at {decision.take_profit_1}")
                    
        except Exception as e:
            logger.error(f"Failed to place stop orders: {e}")
    
    def _create_fallback_decision(self, opportunity: MarketOpportunity, analysis: TechnicalAnalysis) -> TradingDecision:
        """Create ultra professional fallback decision"""
        return TradingDecision(
            symbol=opportunity.symbol,
            signal=SignalType.HOLD,
            confidence=0.6,
            entry_price=opportunity.current_price,
            stop_loss=opportunity.current_price,
            take_profit_1=opportunity.current_price,
            take_profit_2=opportunity.current_price,
            take_profit_3=opportunity.current_price,
            position_size=0.0,
            risk_reward_ratio=1.0,
            ia1_analysis_id=analysis.id,
            ia2_reasoning=f"Ultra professional fallback decision for {opportunity.symbol} - live trading system temporarily unavailable",
            bingx_status="FALLBACK"
        )
    
    def _evaluate_ultra_professional_decision(self, opportunity: MarketOpportunity, 
                                            analysis: TechnicalAnalysis, 
                                            perf_stats: Dict) -> Dict[str, Any]:
        """Ultra professional decision evaluation with multi-source validation"""
        
        signal = SignalType.HOLD
        confidence = (analysis.analysis_confidence + opportunity.data_confidence) / 2
        reasoning = "Ultra professional multi-source analysis: "
        
        # Data quality gates
        if opportunity.data_confidence < 0.7:
            reasoning += "Insufficient data confidence. "
            confidence *= 0.8
        
        if analysis.analysis_confidence < 0.6:
            reasoning += "Low analysis confidence. "
            confidence *= 0.9
        
        # Multi-source validation bonus
        if len(opportunity.data_sources) >= 2:
            confidence = min(confidence + 0.05, 0.98)
            reasoning += f"Multi-source validation ({len(opportunity.data_sources)} sources). "
        
        # Advanced signal scoring
        bullish_signals = 0
        bearish_signals = 0
        signal_strength = 0
        
        # RSI analysis (enhanced)
        if analysis.rsi < 25:
            bullish_signals += 3
            signal_strength += 0.3
            reasoning += "RSI extremely oversold (strong buy signal). "
        elif analysis.rsi < 35:
            bullish_signals += 2
            signal_strength += 0.2
            reasoning += "RSI oversold (buy signal). "
        elif analysis.rsi > 75:
            bearish_signals += 3
            signal_strength += 0.3
            reasoning += "RSI extremely overbought (strong sell signal). "
        elif analysis.rsi > 65:
            bearish_signals += 2
            signal_strength += 0.2
            reasoning += "RSI overbought (sell signal). "
        
        # MACD analysis (enhanced)
        if analysis.macd_signal > 0.005:
            bullish_signals += 2
            signal_strength += 0.25
            reasoning += "Strong MACD bullish momentum. "
        elif analysis.macd_signal > 0:
            bullish_signals += 1
            signal_strength += 0.15
            reasoning += "MACD bullish momentum. "
        elif analysis.macd_signal < -0.005:
            bearish_signals += 2
            signal_strength += 0.25
            reasoning += "Strong MACD bearish momentum. "
        elif analysis.macd_signal < 0:
            bearish_signals += 1
            signal_strength += 0.15
            reasoning += "MACD bearish momentum. "
        
        # Bollinger Bands analysis (enhanced)
        if analysis.bollinger_position < -0.8:
            bullish_signals += 2
            signal_strength += 0.2
            reasoning += "Price at lower Bollinger Band (oversold bounce expected). "
        elif analysis.bollinger_position > 0.8:
            bearish_signals += 2
            signal_strength += 0.2
            reasoning += "Price at upper Bollinger Band (overbought rejection expected). "
        
        # Volume and market cap validation
        if opportunity.volume_24h > 10_000_000:  # High volume
            signal_strength += 0.1
            reasoning += "High volume validation. "
        
        if opportunity.market_cap and opportunity.market_cap > 1_000_000_000:  # Large cap
            signal_strength += 0.05
            reasoning += "Large cap stability. "
        
        # Pattern analysis bonus
        bullish_patterns = ["Golden Cross", "Bullish", "Breakout", "Support"]
        bearish_patterns = ["Death Cross", "Bearish", "Breakdown", "Resistance"]
        
        for pattern in analysis.patterns_detected:
            if any(bp in pattern for bp in bullish_patterns):
                bullish_signals += 1
                signal_strength += 0.1
            elif any(bp in pattern for bp in bearish_patterns):
                bearish_signals += 1
                signal_strength += 0.1
        
        # Market sentiment consideration
        if analysis.market_sentiment == "bullish":
            bullish_signals += 1
            reasoning += "Bullish market sentiment. "
        elif analysis.market_sentiment == "bearish":
            bearish_signals += 1
            reasoning += "Bearish market sentiment. "
        
        # Decision logic with enhanced thresholds
        net_signals = bullish_signals - bearish_signals
        
        if net_signals >= 3 and confidence > 0.8 and signal_strength > 0.5:
            signal = SignalType.LONG
            confidence = min(confidence + 0.1, 0.98)
            reasoning += "ULTRA BULLISH: Multiple strong signals confirmed. "
        elif net_signals >= 2 and confidence > 0.75 and signal_strength > 0.4:
            signal = SignalType.LONG
            confidence = min(confidence + 0.05, 0.95)
            reasoning += "BULLISH: Strong signals with good confidence. "
        elif net_signals <= -3 and confidence > 0.8 and signal_strength > 0.5:
            signal = SignalType.SHORT
            confidence = min(confidence + 0.1, 0.98)
            reasoning += "ULTRA BEARISH: Multiple strong signals confirmed. "
        elif net_signals <= -2 and confidence > 0.75 and signal_strength > 0.4:
            signal = SignalType.SHORT
            confidence = min(confidence + 0.05, 0.95)
            reasoning += "BEARISH: Strong signals with good confidence. "
        else:
            signal = SignalType.HOLD
            reasoning += f"HOLD: Insufficient signal strength (net: {net_signals}, strength: {signal_strength:.2f}). "
        
        # Calculate ultra professional levels
        current_price = opportunity.current_price
        atr_estimate = current_price * max(opportunity.volatility, 0.02)
        
        if signal == SignalType.LONG:
            # Use validated support/resistance levels
            stop_loss = min(analysis.support_levels) if analysis.support_levels else current_price * 0.97
            stop_loss = max(stop_loss, current_price - (2.5 * atr_estimate))  # ATR-based minimum
            
            tp1 = min(analysis.resistance_levels) if analysis.resistance_levels else current_price * 1.03
            tp1 = min(tp1, current_price + (2 * atr_estimate))  # ATR-based maximum
            tp2 = current_price + (4 * atr_estimate)
            tp3 = current_price + (6 * atr_estimate)
            
        elif signal == SignalType.SHORT:
            # Use validated support/resistance levels  
            stop_loss = max(analysis.resistance_levels) if analysis.resistance_levels else current_price * 1.03
            stop_loss = min(stop_loss, current_price + (2.5 * atr_estimate))  # ATR-based minimum
            
            tp1 = max(analysis.support_levels) if analysis.support_levels else current_price * 0.97
            tp1 = max(tp1, current_price - (2 * atr_estimate))  # ATR-based maximum
            tp2 = current_price - (4 * atr_estimate)
            tp3 = current_price - (6 * atr_estimate)
            
        else:
            stop_loss = current_price
            tp1 = tp2 = tp3 = current_price
        
        # Ultra professional risk-reward calculation
        if signal != SignalType.HOLD:
            risk = abs(current_price - stop_loss)
            reward = abs(tp1 - current_price)
            risk_reward = reward / risk if risk > 0 else 1.0
            
            # Minimum risk-reward filter
            if risk_reward < 1.5:
                signal = SignalType.HOLD
                reasoning += "Insufficient risk-reward ratio. "
                confidence *= 0.8
        else:
            risk_reward = 1.0
        
        # Ultra professional position sizing
        base_size = 0.02  # 2% base risk
        
        if signal != SignalType.HOLD:
            # Adjust based on confidence and data quality
            confidence_multiplier = confidence
            data_quality_multiplier = opportunity.data_confidence
            
            position_size = base_size * confidence_multiplier * data_quality_multiplier
            position_size = min(position_size, 0.05)  # Max 5% position
            position_size = max(position_size, 0.005)  # Min 0.5% position
        else:
            position_size = 0.0
        
        return {
            "signal": signal,
            "confidence": confidence,
            "stop_loss": stop_loss,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "position_size": position_size,
            "risk_reward": risk_reward,
            "reasoning": reasoning,
            "signal_strength": signal_strength,
            "net_signals": net_signals
        }
    
    def _create_fallback_decision(self, opportunity: MarketOpportunity, analysis: TechnicalAnalysis) -> TradingDecision:
        """Create ultra professional fallback decision"""
        return TradingDecision(
            symbol=opportunity.symbol,
            signal=SignalType.HOLD,
            confidence=0.6,
            entry_price=opportunity.current_price,
            stop_loss=opportunity.current_price,
            take_profit_1=opportunity.current_price,
            take_profit_2=opportunity.current_price,
            take_profit_3=opportunity.current_price,
            position_size=0.0,
            risk_reward_ratio=1.0,
            ia1_analysis_id=analysis.id,
            ia2_reasoning=f"Ultra professional fallback decision for {opportunity.symbol} - multi-source validation pending"
        )

# Ultra Professional Trading Orchestrator
class UltraProfessionalTradingOrchestrator:
    def __init__(self):
        self.scout = UltraProfessionalCryptoScout()
        self.ia1 = UltraProfessionalIA1TechnicalAnalyst()  
        self.ia2 = UltraProfessionalIA2DecisionAgent()
        self.is_running = False
        self.cycle_count = 0
        self._initialized = False
    
    async def initialize(self):
        """Initialize the trading orchestrator with trending system"""
        if not self._initialized:
            logger.info("🚀 Initializing Ultra Professional Trading Orchestrator...")
            await self.scout.initialize_trending_system()
            self._initialized = True
            logger.info("✅ Trading orchestrator initialized with auto-trending system")
    
    async def run_trading_cycle(self):
        """Execute ultra professional trading cycle with auto-updated trends"""
        try:
            # Ensure system is initialized
            if not self._initialized:
                await self.initialize()
            
            self.cycle_count += 1
            logger.info(f"Starting ultra professional trading cycle #{self.cycle_count} with auto-trending")
            
            # 1. Ultra professional market scan with auto-updated trends
            opportunities = await self.scout.scan_opportunities()
            logger.info(f"Ultra scan found {len(opportunities)} high-quality trending opportunities")
            
            if not opportunities:
                logger.warning("No opportunities found in ultra professional trending scan")
                return 0
            
            # Broadcast to frontend
            await manager.broadcast({
                "type": "opportunities_found", 
                "data": [opp.dict() for opp in opportunities],
                "cycle": self.cycle_count,
                "ultra_professional": True,
                "trending_auto_updated": True
            })
            
            # 2. Ultra professional IA1 analysis (parallel processing for top opportunities)
            top_opportunities = opportunities[:10]  # Analyze top 10 for performance
            analysis_tasks = []
            
            for opportunity in top_opportunities:
                analysis_tasks.append(self.ia1.analyze_opportunity(opportunity))
            
            # Execute analyses in parallel
            analyses = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            valid_analyses = []
            filtered_count = 0
            rejected_no_data_count = 0
            
            for i, analysis in enumerate(analyses):
                if isinstance(analysis, TechnicalAnalysis):
                    valid_analyses.append((top_opportunities[i], analysis))
                    
                    # Store analysis
                    await db.technical_analyses.insert_one(analysis.dict())
                    
                    # Broadcast analysis
                    await manager.broadcast({
                        "type": "technical_analysis",
                        "data": analysis.dict(),
                        "ultra_professional": True,
                        "trending_focused": True
                    })
                else:
                    if analysis is None:
                        # Vérifier si c'est un rejet pour pattern ou pour données
                        symbol = top_opportunities[i].symbol
                        should_analyze, detected_pattern = await technical_pattern_detector.should_analyze_with_ia1(symbol)
                        
                        if should_analyze:
                            rejected_no_data_count += 1
                            logger.debug(f"❌ DATA REJECTION: {symbol} - Pattern detected but no real OHLCV data")
                        else:
                            filtered_count += 1
                            logger.debug(f"⚪ PATTERN FILTER: {symbol} - No strong patterns, skipped IA1")
                    else:
                        logger.warning(f"Analysis failed for {top_opportunities[i].symbol}: {analysis}")
            
            logger.info(f"📊 IA1 PROCESSING RESULTS: {len(valid_analyses)} analyzed, {filtered_count} pattern-filtered, {rejected_no_data_count} data-rejected, {len(top_opportunities) - len(valid_analyses) - filtered_count - rejected_no_data_count} errors")
            logger.info(f"Completed {len(valid_analyses)} ultra professional analyses with REAL OHLCV data only")
            
            # 3. Ultra professional IA2 decisions (parallel processing)
            decision_tasks = []
            for opportunity, analysis in valid_analyses:
                decision_tasks.append(self.ia2.make_decision(opportunity, analysis))
            
            # Execute decisions in parallel
            decisions = await asyncio.gather(*decision_tasks, return_exceptions=True)
            
            valid_decisions = 0
            for i, decision in enumerate(decisions):
                if isinstance(decision, TradingDecision):
                    valid_decisions += 1
                    
                    # Store decision
                    await db.trading_decisions.insert_one(decision.dict())
                    
                    # Broadcast decision
                    await manager.broadcast({
                        "type": "trading_decision",
                        "data": decision.dict(),
                        "ultra_professional": True,
                        "trending_focused": True
                    })
                    
                    # Store opportunity
                    opportunity = valid_analyses[i][0]
                    await db.market_opportunities.insert_one(opportunity.dict())
                    
                    logger.info(f"Ultra professional trending decision for {opportunity.symbol}: {decision.signal} (confidence: {decision.confidence:.2f})")
                else:
                    logger.warning(f"Decision failed: {decision}")
            
            logger.info(f"Ultra professional trending cycle #{self.cycle_count} complete: {valid_decisions} decisions generated")
            return len(opportunities)
            
        except Exception as e:
            logger.error(f"Ultra professional trending cycle error: {e}")
            return 0

# Global orchestrator instance
orchestrator = UltraProfessionalTradingOrchestrator()

# Enhanced API Endpoints
@api_router.get("/")
async def root():
    return {
        "message": "Dual AI Trading Bot System - Ultra Professional Edition", 
        "status": "active", 
        "version": "3.0.0",
        "features": [
            "Multi-source data aggregation",
            "7+ API endpoints with intelligent fallback",
            "Advanced technical analysis with GPT-5",
            "Ultra professional risk management",
            "Real-time multi-threaded data processing"
        ]
    }

@api_router.post("/start-trading")
async def start_trading():
    """Start the ultra professional trading system"""
    if orchestrator.is_running:
        return {"message": "Ultra professional trading system already running", "version": "3.0.0"}
    
    orchestrator.is_running = True
    asyncio.create_task(ultra_professional_trading_loop())
    return {"message": "Ultra professional trading system started with multi-source data aggregation", "version": "3.0.0"}

@api_router.post("/stop-trading")  
async def stop_trading():
    """Stop the ultra professional trading system"""
    orchestrator.is_running = False
    return {"message": "Ultra professional trading system stopped", "version": "3.0.0"}

@api_router.get("/opportunities")
async def get_opportunities():
    """Get recent market opportunities"""
    opportunities = await db.market_opportunities.find().sort("timestamp", -1).limit(50).to_list(50)
    for opp in opportunities:
        opp.pop('_id', None)
    return {"opportunities": opportunities, "ultra_professional": True}

@api_router.get("/status")
async def get_status():
    """Get system status for frontend connectivity"""
    return {
        "status": "connected",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "3.0.0",
        "message": "Ultra Professional Trading System Active"
    }

@api_router.get("/analyses-debug")
async def get_analyses_debug():
    """Debug endpoint pour identifier les problèmes JSON"""
    try:
        analyses = await db.technical_analyses.find().sort("timestamp", -1).limit(5).to_list(5)
        
        debug_info = {
            "count": len(analyses),
            "analyses": [],
            "errors": []
        }
        
        for i, analysis in enumerate(analyses):
            analysis.pop('_id', None)
            symbol = analysis.get('symbol', f'unknown_{i}')
            
            # Test individual fields
            field_status = {}
            for key, value in analysis.items():
                try:
                    json.dumps(value)
                    field_status[key] = "OK"
                except Exception as e:
                    field_status[key] = f"ERROR: {str(e)} (value: {repr(value)[:100]})"
                    debug_info["errors"].append(f"{symbol}.{key}: {repr(value)[:100]}")
            
            debug_info["analyses"].append({
                "symbol": symbol,
                "field_status": field_status
            })
        
        return debug_info
        
    except Exception as e:
        return {"error": str(e), "type": str(type(e))}

@api_router.get("/analyses-simple")
async def get_analyses_simple():
    """Version simplifiée des analyses pour debug"""
    try:
        analyses = await db.technical_analyses.find().sort("timestamp", -1).limit(10).to_list(10)
        
        simple_analyses = []
        for analysis in analyses:
            simple_analyses.append({
                "symbol": analysis.get('symbol', 'UNKNOWN'),
                "rsi": float(analysis.get('rsi', 50.0)) if analysis.get('rsi') is not None else 50.0,
                "confidence": float(analysis.get('analysis_confidence', 0.5)) if analysis.get('analysis_confidence') is not None else 0.5,
                "patterns": analysis.get('patterns_detected', [])[:3],  # Max 3 patterns
                "timestamp": analysis.get('timestamp').isoformat() if isinstance(analysis.get('timestamp'), datetime) else str(analysis.get('timestamp', ''))
            })
        
        return {"analyses": simple_analyses, "count": len(simple_analyses)}
        
    except Exception as e:
        return {"error": str(e), "analyses": []}

@api_router.get("/analyses")
async def get_analyses():
    """Get recent technical analyses - VRAIES valeurs IA1 avec validation JSON"""
    try:
        real_analyses = await db.technical_analyses.find().sort("timestamp", -1).limit(10).to_list(10)
        
        if not real_analyses:
            return {"analyses": [], "ultra_professional": True, "note": "No analyses found"}
        
        validated_analyses = []
        for analysis in real_analyses:
            try:
                # Remove MongoDB _id
                analysis.pop('_id', None)
                
                # Fix timestamp issue (seul problème JSON identifié)
                if 'timestamp' in analysis and isinstance(analysis['timestamp'], datetime):
                    analysis['timestamp'] = analysis['timestamp'].isoformat()
                elif 'timestamp' in analysis:
                    analysis['timestamp'] = str(analysis['timestamp'])
                
                # Validation sécurisée des valeurs numériques (garder les vraies valeurs IA1)
                numeric_fields = ['rsi', 'macd_signal', 'bollinger_position', 'fibonacci_level', 'analysis_confidence']
                for field in numeric_fields:
                    if field in analysis:
                        val = analysis[field]
                        if val is None or pd.isna(val) or not pd.notna(val) or abs(val) > 1e6:
                            # Remplace seulement les valeurs invalides, garde les vraies valeurs IA1
                            if field == 'rsi':
                                analysis[field] = 50.0
                            elif field == 'analysis_confidence':
                                analysis[field] = 0.5
                            elif field == 'fibonacci_level':
                                analysis[field] = 0.618
                            else:
                                analysis[field] = 0.0
                        else:
                            # Garde les vraies valeurs IA1 calculées
                            analysis[field] = float(val)
                
                # Valide les listes (support/resistance)
                for list_field in ['support_levels', 'resistance_levels']:
                    if list_field in analysis:
                        if not isinstance(analysis[list_field], list):
                            analysis[list_field] = []
                        else:
                            # Nettoie les valeurs invalides dans les listes
                            clean_list = []
                            for val in analysis[list_field]:
                                try:
                                    if pd.notna(val) and abs(float(val)) < 1e6:
                                        clean_list.append(float(val))
                                except:
                                    pass
                            analysis[list_field] = clean_list[:5]  # Max 5 niveaux
                
                # Valide les strings
                string_fields = ['ia1_reasoning', 'market_sentiment']
                for field in string_fields:
                    if field in analysis:
                        analysis[field] = str(analysis[field]) if analysis[field] is not None else ""
                
                # Valide patterns_detected
                if 'patterns_detected' not in analysis or not isinstance(analysis['patterns_detected'], list):
                    analysis['patterns_detected'] = ["No patterns detected"]
                
                validated_analyses.append(analysis)
                
            except Exception as e:
                logger.error(f"Error validating analysis for {analysis.get('symbol', 'unknown')}: {e}")
                continue
        
        return {
            "analyses": validated_analyses, 
            "ultra_professional": True,
            "note": f"Real IA1 analyses with validated RSI, MACD, BB, Fibonacci values"
        }
        
    except Exception as e:
        logger.error(f"Error fetching real analyses: {e}")
        return {"analyses": [], "ultra_professional": True, "error": str(e)}

@api_router.get("/decisions")
async def get_decisions():
    """Get recent trading decisions"""
    decisions = await db.trading_decisions.find().sort("timestamp", -1).limit(30).to_list(30)
    for decision in decisions:
        decision.pop('_id', None)
    return {"decisions": decisions, "ultra_professional": True}

@api_router.get("/market-aggregator-stats")
async def get_market_aggregator_stats():
    """Get market aggregator performance statistics"""
    try:
        stats = advanced_market_aggregator.get_performance_stats()
        return {
            "aggregator_stats": stats,
            "ultra_professional": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/performance")
async def get_performance():
    """Get ultra professional trading performance metrics"""
    try:
        decisions = await db.trading_decisions.find().to_list(200)
        opportunities = await db.market_opportunities.find().to_list(200)
        analyses = await db.technical_analyses.find().to_list(200)
        
        total_trades = len([d for d in decisions if d.get('status') == 'executed'])
        profitable_trades = len([d for d in decisions if d.get('status') == 'executed' and d.get('signal') != 'hold'])
        
        # Enhanced performance metrics
        high_confidence_decisions = len([d for d in decisions if d.get('confidence', 0) > 0.8])
        multi_source_opportunities = len([o for o in opportunities if len(o.get('data_sources', [])) > 1])
        
        performance = {
            "total_opportunities": len(opportunities),
            "multi_source_opportunities": multi_source_opportunities,
            "total_analyses": len(analyses),
            "total_decisions": len(decisions),
            "executed_trades": total_trades,
            "high_confidence_decisions": high_confidence_decisions,
            "win_rate": (profitable_trades / total_trades * 100) if total_trades > 0 else 0,
            "avg_confidence": sum([d.get('confidence', 0) for d in decisions]) / len(decisions) if decisions else 0,
            "avg_data_confidence": sum([o.get('data_confidence', 0) for o in opportunities]) / len(opportunities) if opportunities else 0,
            "data_source_diversity": len(set([src for opp in opportunities for src in opp.get('data_sources', [])])),
            "ultra_professional": True,
            "version": "3.0.0",
            "last_update": datetime.now(timezone.utc).isoformat()
        }
        
        return {"performance": performance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/bingx-status")
async def get_bingx_status():
    """Get BingX exchange status and account info"""
    try:
        # Test connectivity
        connectivity = await bingx_trading_engine.test_connectivity()
        
        # Get account balance
        balances = await bingx_trading_engine.get_account_balance()
        
        # Get open positions
        positions = await bingx_trading_engine.get_positions()
        
        # Get performance stats
        perf_stats = bingx_trading_engine.get_performance_stats()
        
        return {
            "connectivity": connectivity,
            "account_balances": [balance.__dict__ for balance in balances],
            "active_positions": [pos.__dict__ for pos in positions],
            "performance_stats": perf_stats,
            "live_trading_enabled": orchestrator.ia2.live_trading_enabled,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/live-positions")
async def get_live_positions():
    """Get current live trading positions from database"""
    try:
        positions = await db.live_positions.find().sort("timestamp", -1).limit(20).to_list(20)
        for pos in positions:
            pos.pop('_id', None)
        return {"positions": positions, "live_trading": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/bingx-orders")
async def get_bingx_orders(symbol: Optional[str] = None):
    """Get current open orders on BingX"""
    try:
        orders = await bingx_trading_engine.get_open_orders(symbol)
        return {
            "orders": [order.__dict__ for order in orders],
            "total_orders": len(orders),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/close-position/{symbol}")
async def close_position(symbol: str):
    """Manually close a position on BingX"""
    try:
        result = await bingx_trading_engine.close_position(symbol)
        return {
            "success": result,
            "message": f"Position closure {'successful' if result else 'failed'} for {symbol}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/toggle-live-trading")
async def toggle_live_trading(enabled: bool):
    """Enable or disable live trading"""
    try:
        orchestrator.ia2.live_trading_enabled = enabled
        return {
            "live_trading_enabled": enabled,
            "message": f"Live trading {'enabled' if enabled else 'disabled'}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/trading-performance-live")
async def get_live_trading_performance():
    """Get live trading performance metrics"""
    try:
        # Get decisions with BingX integration
        decisions = await db.trading_decisions.find(
            {"bingx_order_id": {"$exists": True}}
        ).sort("timestamp", -1).limit(100).to_list(100)
        
        # Calculate live trading stats
        total_live_trades = len(decisions)
        executed_trades = len([d for d in decisions if d.get('bingx_status') == 'FILLED'])
        successful_orders = len([d for d in decisions if d.get('bingx_order_id')])
        
        # Get BingX performance
        bingx_stats = bingx_trading_engine.get_performance_stats()
        
        performance = {
            "total_live_trades": total_live_trades,
            "executed_trades": executed_trades,
            "successful_orders": successful_orders,
            "order_success_rate": (successful_orders / total_live_trades * 100) if total_live_trades > 0 else 0,
            "bingx_api_success_rate": bingx_stats.get('success_rate', 0),
            "live_trading_enabled": orchestrator.ia2.live_trading_enabled,
            "demo_mode": bingx_stats.get('demo_mode', False),
            "last_api_response_time": bingx_stats.get('last_request_time', 0),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return {"performance": performance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/market-status")
async def get_market_status():
    """Get ultra professional market status with BingX integration"""
    try:
        aggregator_stats = advanced_market_aggregator.get_performance_stats()
        bingx_stats = bingx_trading_engine.get_performance_stats()
        
        return {
            "market_aggregator": {
                "total_requests": aggregator_stats.get('total_requests', 0),
                "success_rate": aggregator_stats.get('success_rate', 0),
                "active_endpoints": len([ep for ep in aggregator_stats.get('api_endpoints', []) if ep.get('status') == 'active'])
            },
            "bingx_exchange": {
                "connectivity": "active",
                "live_trading_enabled": orchestrator.ia2.live_trading_enabled,
                "demo_mode": bingx_stats.get('demo_mode', False),
                "api_success_rate": bingx_stats.get('success_rate', 0),
                "total_requests": bingx_stats.get('total_requests', 0)
            },
            "api_status": {
                "coinmarketcap": "ultra_professional",
                "coingecko": "active",
                "coinapi": "active",
                "yahoo_finance": "active",
                "binance": "ccxt_integration",
                "bitfinex": "ccxt_integration",
                "bingx": "live_trading_ready",
                "dex_data": "coinmarketcap_v4"
            },
            "system_status": "ultra_professional_live_trading",
            "version": "3.0.0",
            "features": [
                "Multi-source aggregation",
                "Parallel processing", 
                "Intelligent fallback",
                "Advanced risk management",
                "BingX live trading integration",
                "Real-time position monitoring"
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}

@api_router.get("/trending-auto-status")
async def get_trending_auto_status():
    """Get trending auto-updater status and current trends"""
    try:
        trending_info = trending_auto_updater.get_trending_info()
        return {
            "trending_auto_updater": trending_info,
            "scout_trending_symbols": orchestrator.scout.trending_symbols,
            "auto_update_enabled": orchestrator.scout.auto_update_trending,
            "readdy_url": trending_auto_updater.trending_url,
            "update_interval_hours": trending_auto_updater.update_interval / 3600,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/trending-force-update")
async def force_trending_update():
    """Force manual update of trending cryptos from Readdy"""
    try:
        result = await trending_auto_updater.force_update()
        
        # Update scout symbols if successful
        if result.get("updated"):
            await orchestrator.scout._sync_trending_symbols()
        
        return {
            "force_update_result": result,
            "scout_symbols_updated": orchestrator.scout.trending_symbols,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/trending-auto-toggle")
async def toggle_trending_auto_update(enabled: bool):
    """Enable or disable trending auto-update"""
    try:
        if enabled:
            if not trending_auto_updater.is_running:
                await trending_auto_updater.start_auto_update()
            orchestrator.scout.auto_update_trending = True
        else:
            if trending_auto_updater.is_running:
                await trending_auto_updater.stop_auto_update()  
            orchestrator.scout.auto_update_trending = False
        
        return {
            "auto_update_enabled": enabled,
            "trending_updater_running": trending_auto_updater.is_running,
            "message": f"Trending auto-update {'enabled' if enabled else 'disabled'}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/scout-trending-config")
async def get_scout_trending_config():
    """Get current trending-focused scout configuration"""
    scout = orchestrator.scout
    return {
        "trending_focus_enabled": scout.focus_trending,
        "trending_symbols": scout.trending_symbols,
        "max_cryptos_analyzed": scout.max_cryptos_to_analyze,
        "min_price_change_threshold": scout.min_price_change_threshold,
        "volume_spike_multiplier": scout.volume_spike_multiplier,
        "min_market_cap": scout.min_market_cap,
        "min_volume_24h": scout.min_volume_24h
    }

@api_router.post("/scout-trending-config")
async def update_scout_trending_config(config: dict):
    """Update trending-focused scout configuration"""
    try:
        scout = orchestrator.scout
        
        if "trending_focus_enabled" in config:
            scout.focus_trending = config["trending_focus_enabled"]
        
        if "trending_symbols" in config:
            scout.trending_symbols = config["trending_symbols"]
        
        if "max_cryptos_analyzed" in config:
            scout.max_cryptos_to_analyze = min(max(config["max_cryptos_analyzed"], 5), 50)
        
        if "min_price_change_threshold" in config:
            scout.min_price_change_threshold = max(config["min_price_change_threshold"], 1.0)
        
        return {
            "message": "Trending scout configuration updated successfully",
            "config": {
                "trending_focus_enabled": scout.focus_trending,
                "trending_symbols": scout.trending_symbols,
                "max_cryptos_analyzed": scout.max_cryptos_to_analyze,
                "min_price_change_threshold": scout.min_price_change_threshold
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.get("/scout-config")
async def get_scout_config():
    """Get current scout configuration (legacy endpoint)"""
    scout = orchestrator.scout
    return {
        "trending_focus_enabled": scout.focus_trending,
        "trending_symbols": scout.trending_symbols,
        "max_cryptos_analyzed": scout.max_cryptos_to_analyze,
        "min_price_change_threshold": scout.min_price_change_threshold,
        "volume_spike_multiplier": scout.volume_spike_multiplier,
        "min_market_cap": scout.min_market_cap,
        "min_volume_24h": scout.min_volume_24h
    }

@api_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(json.dumps({
                "type": "pong", 
                "message": "Connected to Ultra Professional Trading System v3.0.0",
                "features": ["Multi-source data", "Advanced AI analysis", "Professional risk management"]
            }))
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Ultra professional background trading loop with trending auto-update
async def ultra_professional_trading_loop():
    """Ultra professional continuous trading loop with trending auto-update"""
    # Initialize the orchestrator
    await orchestrator.initialize()
    
    while orchestrator.is_running:
        try:
            cycle_start = datetime.now()
            opportunities_processed = await orchestrator.run_trading_cycle()
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            logger.info(f"Ultra professional trending cycle #{orchestrator.cycle_count} completed in {cycle_duration:.2f}s, processed {opportunities_processed} opportunities")
            
            # Broadcast cycle completion with trending info
            await manager.broadcast({
                "type": "cycle_complete",
                "cycle": orchestrator.cycle_count,
                "duration": cycle_duration,
                "opportunities_processed": opportunities_processed,
                "ultra_professional": True,
                "trending_auto_updated": True,
                "trending_symbols": orchestrator.scout.trending_symbols,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Ultra professional cycle timing - every 3 minutes for comprehensive analysis
            await asyncio.sleep(180)
            
        except Exception as e:
            logger.error(f"Ultra professional trending trading loop error: {e}")
            await asyncio.sleep(120)  # Wait 2 minutes on error

# WebSocket endpoint for real-time updates
@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time trading updates"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "status": "connected",
            "message": "Ultra Professional Trading System Connected",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Keep connection alive and send updates
        while True:
            try:
                # Send periodic updates every 30 seconds
                await asyncio.sleep(30)
                
                # Get current system status
                opportunities_count = await db.market_opportunities.count_documents({})
                analyses_count = await db.technical_analyses.count_documents({})
                decisions_count = await db.trading_decisions.count_documents({})
                
                update_data = {
                    "type": "update",
                    "data": {
                        "opportunities_count": opportunities_count,
                        "analyses_count": analyses_count,
                        "decisions_count": decisions_count,
                        "system_status": "active",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                }
                
                await websocket.send_json(update_data)
                
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket endpoint error: {e}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
    # Shutdown thread pool
    if hasattr(orchestrator.scout.market_aggregator, 'thread_pool'):
        orchestrator.scout.market_aggregator.thread_pool.shutdown(wait=True)
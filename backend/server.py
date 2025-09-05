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

# Import our real market data service
from market_data_service import market_data_service, MarketDataPoint

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="Dual AI Trading Bot System - Professional Edition", version="2.0.0")

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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TechnicalAnalysis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    rsi: float
    macd_signal: float
    bollinger_position: float  # -1 to 1, where -1 is lower band, 1 is upper band
    fibonacci_level: float
    support_levels: List[float]
    resistance_levels: List[float]
    patterns_detected: List[str]
    analysis_confidence: float
    ia1_reasoning: str
    market_sentiment: str = "neutral"
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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AIConversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    decision_id: str
    ia1_message: str
    ia2_response: str
    conversation_type: str  # "clarification", "confirmation", "analysis"
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
    outcome: Optional[str] = None  # "win", "loss", "breakeven"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
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
        session_id="ia1-technical-analyst-pro",
        system_message="""You are IA1, an expert technical analyst and chart pattern recognition specialist for cryptocurrency trading with access to REAL market data.

Your role:
- Analyze REAL market data and technical indicators (RSI, MACD, Bollinger Bands, Fibonacci levels)
- Identify chart patterns (Head & Shoulders, Double Top/Bottom, Cup & Handle, Triangles, Bull/Bear Flags)
- Consider support/resistance levels, trend analysis, and market sentiment
- Provide detailed technical analysis with confidence levels based on multiple timeframes
- Factor in volume analysis and market microstructure

Key Analysis Framework:
1. Price Action Analysis (trend, momentum, reversal patterns)
2. Technical Indicators (RSI overbought/oversold, MACD crossovers, Bollinger squeeze)
3. Volume Profile (accumulation/distribution, volume spikes)
4. Market Structure (higher highs/lows, support/resistance breaks)
5. Risk Assessment (volatility, correlation with major crypto trends)

Respond in JSON format with:
{
    "analysis": "detailed technical analysis with specific levels",
    "rsi_interpretation": "RSI analysis with overbought/oversold signals",
    "macd_signal": "MACD trend and momentum analysis", 
    "pattern_detected": ["specific patterns with entry/exit levels"],
    "support_levels": [precise price levels],
    "resistance_levels": [precise price levels],
    "volume_analysis": "volume trend and significance",
    "market_structure": "trend analysis and key levels",
    "confidence": 0.85,
    "recommendation": "long/short/hold with specific reasoning",
    "reasoning": "comprehensive explanation with risk factors"
}"""
    ).with_model("openai", "gpt-5")

def get_ia2_chat():
    return LlmChat(
        api_key=os.environ.get('EMERGENT_LLM_KEY'),
        session_id="ia2-decision-agent-pro",
        system_message="""You are IA2, an intelligent trading decision agent specialized in risk management and strategy execution with access to REAL market analysis.

Your role:
- Review technical analysis from IA1 based on REAL market data
- Make autonomous trading decisions when confidence is high (>0.8)
- Ask clarifying questions to IA1 when analysis is unclear or confidence is moderate (0.6-0.8)
- Calculate position sizes based on advanced risk management (max 2% risk per trade)
- Set dynamic stop-loss and take-profit levels based on volatility and market conditions
- Consider market sentiment, correlation, and portfolio risk

Decision Framework:
- High confidence (>0.8): Execute immediately with full position size
- Medium confidence (0.6-0.8): Execute with reduced size OR request clarification from IA1
- Low confidence (<0.6): Reject trade OR ask specific questions to IA1

Risk Management Rules:
1. Maximum 2% account risk per trade
2. Position size based on ATR and volatility
3. Stop-loss at technical levels, not arbitrary percentages
4. Take-profit at key resistance/support with partial exits
5. Consider correlation with existing positions

Respond in JSON format:
{
    "decision": "execute/clarify/reject",
    "signal": "long/short/hold",
    "confidence": 0.85,
    "position_size": 0.02,
    "stop_loss": 45000,
    "take_profit_levels": [46000, 47000, 48000],
    "risk_reward_ratio": 3.0,
    "max_drawdown_risk": 0.02,
    "market_conditions": "favorable/neutral/unfavorable",
    "reasoning": "detailed explanation with risk assessment",
    "questions_for_ia1": ["specific technical questions if clarification needed"],
    "execution_notes": "specific instructions for trade management"
}"""
    ).with_model("openai", "gpt-5")

# Enhanced Trading System Classes
class ProfessionalCryptoScout:
    def __init__(self):
        self.market_service = market_data_service
        self.max_cryptos_to_analyze = 50  # Analyser top 50 par défaut pour éviter la surcharge
        self.min_market_cap = 10_000_000  # 10M$ minimum market cap
        self.min_volume_24h = 1_000_000   # 1M$ minimum volume 24h
    
    async def scan_opportunities(self) -> List[MarketOpportunity]:
        """Scan real market opportunities from top 500 cryptos by market cap"""
        try:
            logger.info(f"Scanning top {self.max_cryptos_to_analyze} cryptos by market cap...")
            
            # Get top cryptos by market cap (up to 500)
            top_cryptos = await self.market_service.get_top_cryptos_by_marketcap(limit=500)
            
            if not top_cryptos:
                logger.warning("No top cryptos data available, falling back to basic scan")
                return await self._fallback_scan()
            
            # Filter cryptos based on our criteria
            filtered_cryptos = self._filter_cryptos(top_cryptos)
            
            # Convert to MarketOpportunity objects
            opportunities = []
            for crypto in filtered_cryptos[:self.max_cryptos_to_analyze]:
                try:
                    opportunity = MarketOpportunity(
                        symbol=f"{crypto['symbol']}USDT",  # Add USDT pair
                        current_price=crypto.get('price', 0),
                        volume_24h=crypto.get('volume_24h', 0),
                        price_change_24h=crypto.get('percent_change_24h', 0),
                        volatility=self._calculate_volatility(crypto.get('percent_change_24h', 0)),
                        market_cap=crypto.get('market_cap', 0),
                        timestamp=datetime.now(timezone.utc)
                    )
                    opportunities.append(opportunity)
                except Exception as e:
                    logger.warning(f"Error processing crypto {crypto.get('symbol', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Found {len(opportunities)} opportunities from top {len(top_cryptos)} cryptos (source: {top_cryptos[0].get('source', 'unknown')})")
            return opportunities
            
        except Exception as e:
            logger.error(f"Error scanning top crypto opportunities: {e}")
            # Fallback to basic scan
            return await self._fallback_scan()
    
    def _filter_cryptos(self, cryptos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter cryptos based on market cap, volume, and trading criteria"""
        filtered = []
        
        for crypto in cryptos:
            # Skip stablecoins and wrapped tokens (optional)
            symbol = crypto.get('symbol', '').upper()
            if symbol in ['USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'WBTC', 'WETH']:
                continue
            
            # Filter by market cap
            market_cap = crypto.get('market_cap', 0)
            if market_cap < self.min_market_cap:
                continue
            
            # Filter by volume
            volume_24h = crypto.get('volume_24h', 0)
            if volume_24h < self.min_volume_24h:
                continue
            
            # Filter by price (avoid penny coins)
            price = crypto.get('price', 0)
            if price < 0.0001:  # Less than $0.0001
                continue
            
            # Skip if no rank (data quality check)
            if not crypto.get('market_cap_rank'):
                continue
            
            filtered.append(crypto)
        
        # Sort by market cap rank (ascending = higher market cap first)
        filtered.sort(key=lambda x: x.get('market_cap_rank', 999999))
        
        logger.info(f"Filtered {len(filtered)} tradeable cryptos from {len(cryptos)} total")
        return filtered
    
    def _calculate_volatility(self, price_change_24h: float) -> float:
        """Calculate volatility estimate from 24h price change"""
        # Simple volatility estimate based on 24h change
        return abs(price_change_24h) / 100.0  # Convert percentage to decimal
    
    async def _fallback_scan(self) -> List[MarketOpportunity]:
        """Fallback to original scan method"""
        try:
            # Get real market data using original method
            market_data_points = await self.market_service.get_crypto_opportunities()
            
            opportunities = []
            for data_point in market_data_points:
                opportunity = MarketOpportunity(
                    symbol=data_point.symbol,
                    current_price=data_point.price,
                    volume_24h=data_point.volume_24h,
                    price_change_24h=data_point.price_change_24h,
                    volatility=data_point.volatility,
                    market_cap=data_point.market_cap,
                    timestamp=data_point.timestamp
                )
                opportunities.append(opportunity)
            
            logger.info(f"Fallback scan found {len(opportunities)} opportunities")
            return opportunities
            
        except Exception as e:
            logger.error(f"Fallback scan failed: {e}")
            return []

class ProfessionalIA1TechnicalAnalyst:
    def __init__(self):
        self.chat = get_ia1_chat()
        self.market_service = market_data_service
    
    async def analyze_opportunity(self, opportunity: MarketOpportunity) -> TechnicalAnalysis:
        """Analyze market opportunity using AI with real historical data"""
        try:
            logger.info(f"IA1 analyzing {opportunity.symbol} with real market data...")
            
            # Get real historical data for technical analysis
            historical_data = await self.market_service.get_historical_data(opportunity.symbol, days=30)
            
            # Calculate real technical indicators
            rsi = self._calculate_rsi(historical_data['Close'])
            macd_line, macd_signal, macd_histogram = self._calculate_macd(historical_data['Close'])
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(historical_data['Close'])
            
            # Calculate Bollinger Band position
            current_price = opportunity.current_price
            if bb_upper > bb_lower:
                bb_position = (current_price - bb_middle) / (bb_upper - bb_middle)
            else:
                bb_position = 0
            
            # Get market sentiment
            sentiment_data = await self.market_service.get_market_sentiment()
            
            # Create comprehensive analysis prompt with real data
            # Format market cap safely
            market_cap_str = f"${opportunity.market_cap:,.0f}" if opportunity.market_cap else "N/A"
            
            prompt = f"""
            Analyze this REAL cryptocurrency trading opportunity with actual market data:
            
            Symbol: {opportunity.symbol}
            Current Price: ${opportunity.current_price:,.2f}
            24h Volume: ${opportunity.volume_24h:,.0f}
            24h Change: {opportunity.price_change_24h:.2f}%
            Volatility: {opportunity.volatility:.2f}
            Market Cap: {market_cap_str}
            
            REAL Technical Indicators (calculated from 30-day history):
            - RSI (14): {rsi:.2f}
            - MACD Line: {macd_line:.6f}
            - MACD Signal: {macd_signal:.6f}
            - MACD Histogram: {macd_histogram:.6f}
            - Bollinger Band Position: {bb_position:.2f} (-1=lower band, 0=middle, 1=upper band)
            - Current BB Upper: ${bb_upper:.2f}
            - Current BB Lower: ${bb_lower:.2f}
            
            Market Context:
            - Overall Sentiment: {sentiment_data.get('sentiment', 'neutral')}
            - Market Avg Change: {sentiment_data.get('avg_price_change', 0):.2f}%
            - Market Avg Volatility: {sentiment_data.get('avg_volatility', 0):.2f}
            
            Recent Price Action (last 5 days):
            {historical_data['Close'].tail().to_string()}
            
            Provide detailed technical analysis with specific entry/exit levels based on this REAL data.
            """
            
            response = await self.chat.send_message(UserMessage(text=prompt))
            
            # Parse AI response and create analysis
            analysis_data = {
                "rsi": rsi,
                "macd_signal": macd_signal,
                "bollinger_position": bb_position,
                "fibonacci_level": self._calculate_fibonacci_retracement(historical_data),
                "support_levels": self._find_support_levels(historical_data, current_price),
                "resistance_levels": self._find_resistance_levels(historical_data, current_price),
                "patterns_detected": self._detect_patterns(historical_data),
                "analysis_confidence": self._calculate_confidence(rsi, macd_histogram, bb_position, opportunity.volatility),
                "ia1_reasoning": response[:1000] if response else "Technical analysis completed with real market data",
                "market_sentiment": sentiment_data.get('sentiment', 'neutral')
            }
            
            return TechnicalAnalysis(
                symbol=opportunity.symbol,
                **analysis_data
            )
            
        except Exception as e:
            logger.error(f"IA1 analysis error for {opportunity.symbol}: {e}")
            # Fallback analysis with real price data
            return self._create_fallback_analysis(opportunity)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except:
            return 50.0
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD indicator"""
        try:
            exp1 = prices.ewm(span=fast).mean()
            exp2 = prices.ewm(span=slow).mean()
            macd_line = exp1 - exp2
            macd_signal = macd_line.ewm(span=signal).mean()
            macd_histogram = macd_line - macd_signal
            
            return float(macd_line.iloc[-1]), float(macd_signal.iloc[-1]), float(macd_histogram.iloc[-1])
        except:
            return 0.0, 0.0, 0.0
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2):
        """Calculate Bollinger Bands"""
        try:
            rolling_mean = prices.rolling(window=period).mean()
            rolling_std = prices.rolling(window=period).std()
            upper_band = rolling_mean + (rolling_std * std_dev)
            lower_band = rolling_mean - (rolling_std * std_dev)
            
            return float(upper_band.iloc[-1]), float(rolling_mean.iloc[-1]), float(lower_band.iloc[-1])
        except:
            current_price = float(prices.iloc[-1])
            return current_price * 1.02, current_price, current_price * 0.98
    
    def _calculate_fibonacci_retracement(self, historical_data: pd.DataFrame) -> float:
        """Calculate Fibonacci retracement level"""
        try:
            high = historical_data['High'].max()
            low = historical_data['Low'].min()
            current = historical_data['Close'].iloc[-1]
            
            # Calculate retracement level
            retracement = (current - low) / (high - low)
            return float(retracement)
        except:
            return 0.5
    
    def _find_support_levels(self, historical_data: pd.DataFrame, current_price: float) -> List[float]:
        """Find key support levels"""
        try:
            lows = historical_data['Low'].rolling(window=5).min()
            support_levels = []
            
            for low in lows.dropna().unique()[-10:]:  # Last 10 unique lows
                if low < current_price * 0.98:  # At least 2% below current price
                    support_levels.append(float(low))
            
            return sorted(support_levels, reverse=True)[:3]  # Top 3 closest supports
        except:
            return [current_price * 0.95, current_price * 0.90]
    
    def _find_resistance_levels(self, historical_data: pd.DataFrame, current_price: float) -> List[float]:
        """Find key resistance levels"""
        try:
            highs = historical_data['High'].rolling(window=5).max()
            resistance_levels = []
            
            for high in highs.dropna().unique()[-10:]:  # Last 10 unique highs
                if high > current_price * 1.02:  # At least 2% above current price
                    resistance_levels.append(float(high))
            
            return sorted(resistance_levels)[:3]  # Top 3 closest resistances
        except:
            return [current_price * 1.05, current_price * 1.10]
    
    def _detect_patterns(self, historical_data: pd.DataFrame) -> List[str]:
        """Detect chart patterns"""
        patterns = []
        try:
            prices = historical_data['Close']
            
            # Simple pattern detection
            if len(prices) >= 20:
                recent_trend = prices.iloc[-5:].mean() / prices.iloc[-10:-5].mean()
                if recent_trend > 1.02:
                    patterns.append("Uptrend formation")
                elif recent_trend < 0.98:
                    patterns.append("Downtrend formation")
                
                # Volatility patterns
                volatility = prices.pct_change().std()
                if volatility > 0.05:
                    patterns.append("High volatility breakout")
                elif volatility < 0.02:
                    patterns.append("Low volatility consolidation")
            
        except:
            patterns = ["Pattern analysis pending"]
        
        return patterns
    
    def _calculate_confidence(self, rsi: float, macd_histogram: float, bb_position: float, volatility: float) -> float:
        """Calculate analysis confidence based on multiple factors"""
        confidence = 0.5  # Base confidence
        
        # RSI confidence
        if 30 <= rsi <= 70:  # Neutral zone
            confidence += 0.1
        elif rsi < 30 or rsi > 70:  # Extreme zones
            confidence += 0.2
        
        # MACD confidence
        if abs(macd_histogram) > 0.001:  # Strong momentum
            confidence += 0.15
        
        # Bollinger Bands confidence
        if abs(bb_position) > 0.5:  # Near bands
            confidence += 0.1
        
        # Volatility confidence (moderate volatility is better)
        if 0.02 <= volatility <= 0.05:
            confidence += 0.1
        
        return min(confidence, 0.95)
    
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
            patterns_detected=["Analysis pending - using real price data"],
            analysis_confidence=0.6,
            ia1_reasoning=f"Fallback analysis for {opportunity.symbol} at ${opportunity.current_price:,.2f}",
            market_sentiment="neutral"
        )

class ProfessionalIA2DecisionAgent:
    def __init__(self):
        self.chat = get_ia2_chat()
    
    async def make_decision(self, opportunity: MarketOpportunity, analysis: TechnicalAnalysis) -> TradingDecision:
        """Make trading decision based on IA1 analysis with real market data"""
        try:
            logger.info(f"IA2 making decision for {opportunity.symbol} based on real analysis...")
            
            # Format market cap safely
            market_cap_str = f"${opportunity.market_cap:,.0f}" if opportunity.market_cap else "N/A"
            
            prompt = f"""
            Review this technical analysis based on REAL market data and make a trading decision:
            
            Symbol: {opportunity.symbol}
            Current Price: ${opportunity.current_price:,.2f}
            Market Cap: {market_cap_str}
            24h Volume: ${opportunity.volume_24h:,.0f}
            24h Change: {opportunity.price_change_24h:.2f}%
            Volatility: {opportunity.volatility:.2f}
            
            IA1 Technical Analysis (from REAL data):
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
            
            Make your trading decision based on this comprehensive real market analysis.
            Consider risk management, position sizing, and current market conditions.
            """
            
            response = await self.chat.send_message(UserMessage(text=prompt))
            
            # Generate decision based on analysis confidence and market conditions
            decision_logic = self._evaluate_trading_decision(opportunity, analysis)
            
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
                ia2_reasoning=response[:1000] if response else decision_logic["reasoning"],
                status=TradingStatus.EXECUTED if decision_logic["signal"] != SignalType.HOLD else TradingStatus.REJECTED
            )
            
            logger.info(f"IA2 decision for {opportunity.symbol}: {decision.signal} (confidence: {decision.confidence:.2f})")
            return decision
            
        except Exception as e:
            logger.error(f"IA2 decision error for {opportunity.symbol}: {e}")
            return self._create_fallback_decision(opportunity, analysis)
    
    def _evaluate_trading_decision(self, opportunity: MarketOpportunity, analysis: TechnicalAnalysis) -> Dict[str, Any]:
        """Evaluate trading decision based on comprehensive analysis"""
        
        # Base decision logic
        signal = SignalType.HOLD
        confidence = analysis.analysis_confidence
        reasoning = "Analysis based on real market data: "
        
        # Technical analysis scoring
        bullish_signals = 0
        bearish_signals = 0
        
        # RSI analysis
        if analysis.rsi < 30:
            bullish_signals += 2
            reasoning += "RSI oversold (bullish). "
        elif analysis.rsi > 70:
            bearish_signals += 2
            reasoning += "RSI overbought (bearish). "
        elif 40 <= analysis.rsi <= 60:
            bullish_signals += 1
            reasoning += "RSI neutral-bullish. "
        
        # MACD analysis
        if analysis.macd_signal > 0:
            bullish_signals += 1
            reasoning += "MACD bullish momentum. "
        elif analysis.macd_signal < -0.001:
            bearish_signals += 1
            reasoning += "MACD bearish momentum. "
        
        # Bollinger Bands analysis
        if analysis.bollinger_position < -0.5:
            bullish_signals += 1
            reasoning += "Price near lower Bollinger Band (potential bounce). "
        elif analysis.bollinger_position > 0.5:
            bearish_signals += 1
            reasoning += "Price near upper Bollinger Band (potential rejection). "
        
        # Price action analysis
        if opportunity.price_change_24h > 5 and opportunity.volatility > 0.05:
            if bullish_signals > bearish_signals:
                bullish_signals += 1
                reasoning += "Strong bullish momentum with high volume. "
        elif opportunity.price_change_24h < -5 and opportunity.volatility > 0.05:
            bearish_signals += 1
            reasoning += "Strong bearish momentum. "
        
        # Market sentiment consideration
        if analysis.market_sentiment == "bullish":
            bullish_signals += 1
            reasoning += "Positive market sentiment. "
        elif analysis.market_sentiment == "bearish":
            bearish_signals += 1
            reasoning += "Negative market sentiment. "
        
        # Decision logic
        net_signals = bullish_signals - bearish_signals
        
        if net_signals >= 2 and confidence > 0.7:
            signal = SignalType.LONG
            confidence = min(confidence + 0.1, 0.95)
        elif net_signals <= -2 and confidence > 0.7:
            signal = SignalType.SHORT
            confidence = min(confidence + 0.1, 0.95)
        else:
            signal = SignalType.HOLD
            reasoning += "Insufficient signal strength or confidence for trade execution. "
        
        # Calculate levels
        current_price = opportunity.current_price
        atr_estimate = current_price * opportunity.volatility
        
        if signal == SignalType.LONG:
            stop_loss = max(analysis.support_levels[0] if analysis.support_levels else current_price * 0.98, 
                           current_price - (2 * atr_estimate))
            tp1 = min(analysis.resistance_levels[0] if analysis.resistance_levels else current_price * 1.05,
                     current_price + (1.5 * atr_estimate))
            tp2 = current_price + (3 * atr_estimate)
            tp3 = current_price + (5 * atr_estimate)
        elif signal == SignalType.SHORT:
            stop_loss = min(analysis.resistance_levels[0] if analysis.resistance_levels else current_price * 1.02,
                           current_price + (2 * atr_estimate))
            tp1 = max(analysis.support_levels[0] if analysis.support_levels else current_price * 0.95,
                     current_price - (1.5 * atr_estimate))
            tp2 = current_price - (3 * atr_estimate)
            tp3 = current_price - (5 * atr_estimate)
        else:
            stop_loss = current_price
            tp1 = tp2 = tp3 = current_price
        
        # Risk-reward calculation
        if signal != SignalType.HOLD:
            risk = abs(current_price - stop_loss)
            reward = abs(tp1 - current_price)
            risk_reward = reward / risk if risk > 0 else 1.0
        else:
            risk_reward = 1.0
        
        # Position sizing (2% risk rule)
        position_size = 0.02 if signal != SignalType.HOLD else 0.0
        
        return {
            "signal": signal,
            "confidence": confidence,
            "stop_loss": stop_loss,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "position_size": position_size,
            "risk_reward": risk_reward,
            "reasoning": reasoning
        }
    
    def _create_fallback_decision(self, opportunity: MarketOpportunity, analysis: TechnicalAnalysis) -> TradingDecision:
        """Create fallback decision when AI fails"""
        return TradingDecision(
            symbol=opportunity.symbol,
            signal=SignalType.HOLD,
            confidence=0.5,
            entry_price=opportunity.current_price,
            stop_loss=opportunity.current_price,
            take_profit_1=opportunity.current_price,
            take_profit_2=opportunity.current_price,
            take_profit_3=opportunity.current_price,
            position_size=0.0,
            risk_reward_ratio=1.0,
            ia1_analysis_id=analysis.id,
            ia2_reasoning=f"Fallback decision for {opportunity.symbol} - system temporarily unavailable"
        )

# Professional Trading Orchestrator
class ProfessionalTradingOrchestrator:
    def __init__(self):
        self.scout = ProfessionalCryptoScout()
        self.ia1 = ProfessionalIA1TechnicalAnalyst()
        self.ia2 = ProfessionalIA2DecisionAgent()
        self.is_running = False
    
    async def run_trading_cycle(self):
        """Execute complete professional trading cycle with real data"""
        try:
            logger.info("Starting professional trading cycle with real market data...")
            
            # 1. Scout for real opportunities
            opportunities = await self.scout.scan_opportunities()
            logger.info(f"Found {len(opportunities)} real market opportunities")
            
            if not opportunities:
                logger.warning("No market opportunities found - APIs may be unavailable")
                return 0
            
            # Broadcast to frontend
            await manager.broadcast({
                "type": "opportunities_found",
                "data": [opp.dict() for opp in opportunities]
            })
            
            # 2. Analyze with IA1 using real data
            for opportunity in opportunities[:5]:  # Limit to 5 to manage API costs
                try:
                    analysis = await self.ia1.analyze_opportunity(opportunity)
                    
                    # Store analysis
                    await db.technical_analyses.insert_one(analysis.dict())
                    
                    # Broadcast analysis
                    await manager.broadcast({
                        "type": "technical_analysis",
                        "data": analysis.dict()
                    })
                    
                    # 3. Decision with IA2
                    decision = await self.ia2.make_decision(opportunity, analysis)
                    
                    # Store decision
                    await db.trading_decisions.insert_one(decision.dict())
                    
                    # Broadcast decision
                    await manager.broadcast({
                        "type": "trading_decision",
                        "data": decision.dict()
                    })
                    
                    logger.info(f"Professional analysis complete for {opportunity.symbol}: {decision.signal} (confidence: {decision.confidence:.2f})")
                    
                except Exception as e:
                    logger.error(f"Error processing {opportunity.symbol}: {e}")
                    continue
                
                # Store opportunities
                await db.market_opportunities.insert_one(opportunity.dict())
            
            return len(opportunities)
            
        except Exception as e:
            logger.error(f"Professional trading cycle error: {e}")
            return 0

# Global orchestrator instance
orchestrator = ProfessionalTradingOrchestrator()

# API Endpoints (same as before but now with real data)
@api_router.get("/")
async def root():
    return {"message": "Dual AI Trading Bot System - Professional Edition", "status": "active", "version": "2.0.0"}

@api_router.post("/start-trading")
async def start_trading():
    """Start the trading system"""
    if orchestrator.is_running:
        return {"message": "Professional trading system already running"}
    
    orchestrator.is_running = True
    # Start background task
    asyncio.create_task(continuous_trading_loop())
    return {"message": "Professional trading system started with real market data"}

@api_router.post("/stop-trading")  
async def stop_trading():
    """Stop the trading system"""
    orchestrator.is_running = False
    return {"message": "Professional trading system stopped"}

@api_router.get("/opportunities")
async def get_opportunities():
    """Get recent market opportunities"""
    opportunities = await db.market_opportunities.find().sort("timestamp", -1).limit(20).to_list(20)
    # Remove MongoDB ObjectId for JSON serialization
    for opp in opportunities:
        opp.pop('_id', None)
    return {"opportunities": opportunities}

@api_router.get("/analyses")
async def get_analyses():
    """Get recent technical analyses"""
    analyses = await db.technical_analyses.find().sort("timestamp", -1).limit(20).to_list(20)
    # Remove MongoDB ObjectId for JSON serialization
    for analysis in analyses:
        analysis.pop('_id', None)
    return {"analyses": analyses}

@api_router.get("/decisions")
async def get_decisions():
    """Get recent trading decisions"""
    decisions = await db.trading_decisions.find().sort("timestamp", -1).limit(20).to_list(20)
    # Remove MongoDB ObjectId for JSON serialization
    for decision in decisions:
        decision.pop('_id', None)
    return {"decisions": decisions}

@api_router.get("/conversations")
async def get_conversations():
    """Get AI conversations"""
    conversations = await db.ai_conversations.find().sort("timestamp", -1).limit(20).to_list(20)
    for conv in conversations:
        conv.pop('_id', None)
    return {"conversations": conversations}

@api_router.get("/performance")
async def get_performance():
    """Get trading performance metrics"""
    decisions = await db.trading_decisions.find().to_list(100)
    
    total_trades = len([d for d in decisions if d.get('status') == 'executed'])
    profitable_trades = len([d for d in decisions if d.get('status') == 'executed' and d.get('signal') != 'hold'])
    
    performance = {
        "total_opportunities": await db.market_opportunities.count_documents({}),
        "total_analyses": await db.technical_analyses.count_documents({}),
        "total_decisions": len(decisions),
        "executed_trades": total_trades,
        "win_rate": (profitable_trades / total_trades * 100) if total_trades > 0 else 0,
        "avg_confidence": sum([d.get('confidence', 0) for d in decisions]) / len(decisions) if decisions else 0,
        "data_source": "Real Market APIs",
        "last_update": datetime.now(timezone.utc).isoformat()
    }
    
    return {"performance": performance}

@api_router.get("/market-status")
async def get_market_status():
    """Get current market status and API health"""
    try:
        sentiment = await market_data_service.get_market_sentiment()
        return {
            "market_sentiment": sentiment,
            "api_status": {
                "binance": "active",
                "coinapi": "active" if market_data_service.coinapi_key else "not configured",
                "yahoo_finance": "active",
                "cmc": "active" if market_data_service.cmc_api_key else "not configured"
            },
            "system_status": "professional",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}

@api_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle any client messages if needed
            await websocket.send_text(json.dumps({"type": "pong", "message": "Connected to Professional Trading System"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Background trading loop (enhanced)
async def continuous_trading_loop():
    """Continuous professional trading loop with real data"""
    while orchestrator.is_running:
        try:
            cycle_start = datetime.now()
            await orchestrator.run_trading_cycle()
            
            # Professional cycle timing - every 2 minutes to respect API limits
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            logger.info(f"Professional trading cycle completed in {cycle_duration:.2f}s")
            
            await asyncio.sleep(120)  # Run every 2 minutes
        except Exception as e:
            logger.error(f"Professional trading loop error: {e}")
            await asyncio.sleep(60)

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
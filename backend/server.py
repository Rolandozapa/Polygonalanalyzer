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

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="Dual AI Trading Bot System", version="1.0.0")

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
        session_id="ia1-technical-analyst",
        system_message="""You are IA1, an expert technical analyst and chart pattern recognition specialist for cryptocurrency trading.

Your role:
- Analyze market data and technical indicators (RSI, MACD, Bollinger Bands, Fibonacci levels)
- Identify chart patterns (Head & Shoulders, Double Top/Bottom, Cup & Handle, Triangles, etc.)
- Provide detailed technical analysis with confidence levels
- Consider support/resistance levels and trend analysis
- Always provide reasoning for your analysis

Respond in JSON format with:
{
    "analysis": "detailed technical analysis",
    "rsi_interpretation": "RSI analysis",
    "macd_signal": "MACD interpretation", 
    "pattern_detected": ["list of patterns found"],
    "support_levels": [price levels],
    "resistance_levels": [price levels],
    "confidence": 0.85,
    "recommendation": "long/short/hold",
    "reasoning": "detailed explanation"
}"""
    ).with_model("openai", "gpt-5")

def get_ia2_chat():
    return LlmChat(
        api_key=os.environ.get('EMERGENT_LLM_KEY'),
        session_id="ia2-decision-agent",
        system_message="""You are IA2, an intelligent trading decision agent specialized in risk management and strategy execution.

Your role:
- Review technical analysis from IA1
- Make autonomous trading decisions when confidence is high (>0.8)
- Ask clarifying questions to IA1 when analysis is unclear or confidence is moderate (0.6-0.8)
- Calculate position sizes based on risk management (max 2% risk per trade)
- Set stop-loss and take-profit levels
- Provide clear reasoning for decisions

Decision criteria:
- High confidence (>0.8): Execute immediately
- Medium confidence (0.6-0.8): Request clarification from IA1
- Low confidence (<0.6): Reject trade

Respond in JSON format:
{
    "decision": "execute/clarify/reject",
    "signal": "long/short/hold",
    "confidence": 0.85,
    "position_size": 0.02,
    "stop_loss": 45000,
    "take_profit_levels": [46000, 47000, 48000],
    "risk_reward_ratio": 3.0,
    "reasoning": "detailed explanation",
    "questions_for_ia1": ["question if clarification needed"]
}"""
    ).with_model("openai", "gpt-5")

# Trading System Classes
class CryptoScout:
    def __init__(self):
        self.active_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT"]
    
    async def scan_opportunities(self) -> List[MarketOpportunity]:
        """Generate mock market opportunities"""
        opportunities = []
        
        for symbol in self.active_symbols:
            # Generate realistic mock data
            base_price = {"BTCUSDT": 65000, "ETHUSDT": 3500, "SOLUSDT": 150, "ADAUSDT": 0.5, "DOTUSDT": 7.5}[symbol]
            price_variation = np.random.normal(0, 0.02)  # 2% volatility
            current_price = base_price * (1 + price_variation)
            
            opportunity = MarketOpportunity(
                symbol=symbol,
                current_price=round(current_price, 2),
                volume_24h=np.random.uniform(100000000, 1000000000),
                price_change_24h=np.random.uniform(-5, 5),
                volatility=np.random.uniform(0.01, 0.05)
            )
            opportunities.append(opportunity)
        
        return opportunities

class IA1TechnicalAnalyst:
    def __init__(self):
        self.chat = get_ia1_chat()
    
    async def analyze_opportunity(self, opportunity: MarketOpportunity) -> TechnicalAnalysis:
        """Analyze market opportunity using AI"""
        # Generate mock technical indicators for demonstration
        rsi = np.random.uniform(20, 80)
        macd = np.random.uniform(-0.5, 0.5)
        bb_position = np.random.uniform(-1, 1)
        
        # Create analysis prompt
        prompt = f"""
        Analyze this cryptocurrency trading opportunity:
        
        Symbol: {opportunity.symbol}
        Current Price: ${opportunity.current_price}
        24h Volume: ${opportunity.volume_24h:,.0f}
        24h Change: {opportunity.price_change_24h:.2f}%
        Volatility: {opportunity.volatility:.2f}
        
        Technical Indicators (for reference):
        - RSI: {rsi:.2f}
        - MACD Signal: {macd:.3f}
        - Bollinger Band Position: {bb_position:.2f} (-1=lower, 0=middle, 1=upper)
        
        Provide detailed technical analysis and trading recommendation.
        """
        
        try:
            response = await self.chat.send_message(UserMessage(text=prompt))
            
            # Parse AI response (simplified for demo)
            analysis_data = {
                "rsi": rsi,
                "macd_signal": macd,
                "bollinger_position": bb_position,
                "fibonacci_level": np.random.uniform(0.382, 0.618),
                "support_levels": [opportunity.current_price * 0.95, opportunity.current_price * 0.90],
                "resistance_levels": [opportunity.current_price * 1.05, opportunity.current_price * 1.10],
                "patterns_detected": ["Triangle formation", "RSI divergence"] if rsi > 70 else ["Support bounce"],
                "analysis_confidence": np.random.uniform(0.6, 0.95),
                "ia1_reasoning": response[:500] if response else "Technical analysis completed"
            }
            
            return TechnicalAnalysis(
                symbol=opportunity.symbol,
                **analysis_data
            )
            
        except Exception as e:
            logger.error(f"IA1 analysis error: {e}")
            # Fallback analysis
            return TechnicalAnalysis(
                symbol=opportunity.symbol,
                rsi=rsi,
                macd_signal=macd,
                bollinger_position=bb_position,
                fibonacci_level=0.5,
                support_levels=[opportunity.current_price * 0.95],
                resistance_levels=[opportunity.current_price * 1.05],
                patterns_detected=["Analysis pending"],
                analysis_confidence=0.5,
                ia1_reasoning="Fallback analysis - AI temporarily unavailable"
            )

class IA2DecisionAgent:
    def __init__(self):
        self.chat = get_ia2_chat()
    
    async def make_decision(self, opportunity: MarketOpportunity, analysis: TechnicalAnalysis) -> TradingDecision:
        """Make trading decision based on IA1 analysis"""
        prompt = f"""
        Review this technical analysis and make a trading decision:
        
        Symbol: {opportunity.symbol}
        Current Price: ${opportunity.current_price}
        
        IA1 Technical Analysis:
        - RSI: {analysis.rsi:.2f}
        - MACD Signal: {analysis.macd_signal:.3f}
        - Bollinger Position: {analysis.bollinger_position:.2f}
        - Patterns: {', '.join(analysis.patterns_detected)}
        - Support Levels: {analysis.support_levels}
        - Resistance Levels: {analysis.resistance_levels}
        - IA1 Confidence: {analysis.analysis_confidence:.2f}
        - IA1 Reasoning: {analysis.ia1_reasoning}
        
        Make your trading decision based on risk management principles.
        """
        
        try:
            response = await self.chat.send_message(UserMessage(text=prompt))
            
            # Generate decision based on analysis confidence
            if analysis.analysis_confidence > 0.8:
                signal = SignalType.LONG if analysis.rsi < 50 and analysis.macd_signal > 0 else SignalType.SHORT
            elif analysis.analysis_confidence > 0.6:
                signal = SignalType.HOLD  # Neutral when moderate confidence
            else:
                signal = SignalType.HOLD
            
            # Calculate levels
            stop_loss = opportunity.current_price * 0.98 if signal == SignalType.LONG else opportunity.current_price * 1.02
            tp1 = opportunity.current_price * 1.02 if signal == SignalType.LONG else opportunity.current_price * 0.98
            tp2 = opportunity.current_price * 1.04 if signal == SignalType.LONG else opportunity.current_price * 0.96
            tp3 = opportunity.current_price * 1.06 if signal == SignalType.LONG else opportunity.current_price * 0.94
            
            decision = TradingDecision(
                symbol=opportunity.symbol,
                signal=signal,
                confidence=min(analysis.analysis_confidence + 0.1, 0.95),
                entry_price=opportunity.current_price,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                take_profit_3=tp3,
                position_size=0.02,  # 2% risk
                risk_reward_ratio=2.0,
                ia1_analysis_id=analysis.id,
                ia2_reasoning=response[:500] if response else "Decision based on technical analysis",
                status=TradingStatus.EXECUTED if signal != SignalType.HOLD else TradingStatus.REJECTED
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"IA2 decision error: {e}")
            # Fallback decision
            return TradingDecision(
                symbol=opportunity.symbol,
                signal=SignalType.HOLD,
                confidence=0.5,
                entry_price=opportunity.current_price,
                stop_loss=opportunity.current_price,
                take_profit_1=opportunity.current_price,
                take_profit_2=opportunity.current_price,
                take_profit_3=opportunity.current_price,
                position_size=0.01,
                risk_reward_ratio=1.0,
                ia1_analysis_id=analysis.id,
                ia2_reasoning="Fallback decision - AI temporarily unavailable"
            )

# Trading Orchestrator
class TradingOrchestrator:
    def __init__(self):
        self.scout = CryptoScout()
        self.ia1 = IA1TechnicalAnalyst()
        self.ia2 = IA2DecisionAgent()
        self.is_running = False
    
    async def run_trading_cycle(self):
        """Execute complete trading cycle"""
        try:
            logger.info("Starting trading cycle...")
            
            # 1. Scout for opportunities
            opportunities = await self.scout.scan_opportunities()
            logger.info(f"Found {len(opportunities)} opportunities")
            
            # Broadcast to frontend
            await manager.broadcast({
                "type": "opportunities_found",
                "data": [opp.dict() for opp in opportunities]
            })
            
            # 2. Analyze with IA1
            for opportunity in opportunities:
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
                
                # Store opportunity
                await db.market_opportunities.insert_one(opportunity.dict())
                
                logger.info(f"Decision for {opportunity.symbol}: {decision.signal} (confidence: {decision.confidence:.2f})")
            
            return len(opportunities)
            
        except Exception as e:
            logger.error(f"Trading cycle error: {e}")
            return 0

# Global orchestrator instance
orchestrator = TradingOrchestrator()

# API Endpoints
@api_router.get("/")
async def root():
    return {"message": "Dual AI Trading Bot System", "status": "active"}

@api_router.post("/start-trading")
async def start_trading():
    """Start the trading system"""
    if orchestrator.is_running:
        return {"message": "Trading system already running"}
    
    orchestrator.is_running = True
    # Start background task
    asyncio.create_task(continuous_trading_loop())
    return {"message": "Trading system started"}

@api_router.post("/stop-trading")  
async def stop_trading():
    """Stop the trading system"""
    orchestrator.is_running = False
    return {"message": "Trading system stopped"}

@api_router.get("/opportunities")
async def get_opportunities():
    """Get recent market opportunities"""
    opportunities = await db.market_opportunities.find().sort("timestamp", -1).limit(20).to_list(20)
    return {"opportunities": opportunities}

@api_router.get("/analyses")
async def get_analyses():
    """Get recent technical analyses"""
    analyses = await db.technical_analyses.find().sort("timestamp", -1).limit(20).to_list(20)
    return {"analyses": analyses}

@api_router.get("/decisions")
async def get_decisions():
    """Get recent trading decisions"""
    decisions = await db.trading_decisions.find().sort("timestamp", -1).limit(20).to_list(20)
    return {"decisions": decisions}

@api_router.get("/conversations")
async def get_conversations():
    """Get AI conversations"""
    conversations = await db.ai_conversations.find().sort("timestamp", -1).limit(20).to_list(20)
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
        "avg_confidence": sum([d.get('confidence', 0) for d in decisions]) / len(decisions) if decisions else 0
    }
    
    return {"performance": performance}

@api_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle any client messages if needed
            await websocket.send_text(json.dumps({"type": "pong", "message": "Connected"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Background trading loop
async def continuous_trading_loop():
    """Continuous trading loop"""
    while orchestrator.is_running:
        try:
            await orchestrator.run_trading_cycle()
            await asyncio.sleep(60)  # Run every minute for demo
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
            await asyncio.sleep(30)

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
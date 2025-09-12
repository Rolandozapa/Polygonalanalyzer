"""
Common data models used across the trading system
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
from enum import Enum
import pytz

# Configuration du fuseau horaire de Paris
PARIS_TZ = pytz.timezone('Europe/Paris')

def get_paris_time():
    """Obtenir l'heure actuelle en heure de Paris"""
    return datetime.now(PARIS_TZ)

def generate_position_id():
    """Generate unique position ID for IA1→IA2 tracking"""
    return f"POS_{uuid.uuid4().hex[:12].upper()}"

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
    timestamp: datetime = Field(default_factory=get_paris_time)

class TechnicalAnalysis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    position_id: str = Field(default_factory=generate_position_id)  # 🆕 Tracking ID for IA1→IA2
    symbol: str
    rsi: float
    macd_signal: float
    stochastic: float = 50.0  # Stochastic %K
    stochastic_d: float = 50.0  # Stochastic %D  
    bollinger_position: float
    fibonacci_level: float
    fibonacci_nearest_level: str = "61.8"  # Niveau Fibonacci le plus proche
    fibonacci_trend_direction: str = "neutral"  # bullish, bearish, neutral
    support_levels: List[float]
    resistance_levels: List[float]
    patterns_detected: List[str]
    analysis_confidence: float
    ia1_reasoning: str
    ia1_signal: str = "hold"  # NOUVEAU: Signal IA1 (long/short/hold) pour filtrage IA2
    market_sentiment: str = "neutral"
    data_sources: List[str] = []
    timestamp: datetime = Field(default_factory=get_paris_time)
    # NOUVEAUX CHAMPS RISK-REWARD IA1
    risk_reward_ratio: float = 0.0
    entry_price: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    risk_amount: float = 0.0
    reward_amount: float = 0.0
    rr_reasoning: str = ""

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
    timestamp: datetime = Field(default_factory=get_paris_time)
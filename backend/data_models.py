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
    macd_line: Optional[float] = 0.0  # MACD line value
    macd_histogram: Optional[float] = 0.0  # MACD histogram
    stochastic: float = 50.0  # Stochastic %K
    stochastic_d: float = 50.0  # Stochastic %D  
    bollinger_position: float
    # 🚀 BASIC TECHNICAL INDICATOR SIGNALS
    rsi_signal: Optional[str] = "neutral"  # extreme_overbought, overbought, oversold, extreme_oversold, neutral
    macd_trend: Optional[str] = "neutral"  # strong_bullish, bullish, bearish, strong_bearish, neutral
    stochastic_signal: Optional[str] = "neutral"  # extreme_overbought, overbought, oversold, extreme_oversold, neutral
    fibonacci_level: float
    fibonacci_nearest_level: str = "61.8"  # Niveau Fibonacci le plus proche
    fibonacci_trend_direction: str = "neutral"  # bullish, bearish, neutral
    fibonacci_signal_strength: Optional[float] = 0.0  # Fibonacci signal strength (0-1)
    fibonacci_signal_direction: Optional[str] = "neutral"  # Fibonacci signal direction
    fibonacci_key_level_proximity: Optional[bool] = False  # Near key Fibonacci level
    support_levels: List[float]
    resistance_levels: List[float]
    patterns_detected: List[str]
    analysis_confidence: float
    ia1_reasoning: str
    ia1_signal: str = "hold"  # NOUVEAU: Signal IA1 (long/short/hold) pour filtrage IA2
    market_sentiment: str = "neutral"
    data_sources: List[str] = []
    # 🚀 ADVANCED TECHNICAL INDICATORS FOR IA2
    mfi_value: Optional[float] = 50.0  # Money Flow Index
    mfi_signal: Optional[str] = "neutral"  # overbought, oversold, neutral
    mfi_institution: Optional[str] = "neutral"  # Institution activity
    vwap_price: Optional[float] = 0.0  # VWAP price
    vwap_position: Optional[float] = 0.0  # Price position vs VWAP %
    vwap_signal: Optional[str] = "neutral"  # overbought, oversold, neutral
    vwap_trend: Optional[str] = "neutral"  # bullish, bearish, neutral
    ema_hierarchy: Optional[str] = "neutral"  # STRONG_BULL, WEAK_BULL, STRONG_BEAR, etc.
    ema_position: Optional[str] = "neutral"  # above_all, mixed, below_all
    ema_cross_signal: Optional[str] = "neutral"  # bullish, bearish, neutral
    ema_strength: Optional[float] = 0.0  # EMA trend strength %
    multi_timeframe_dominant: Optional[str] = "DAILY"  # Dominant timeframe
    multi_timeframe_pattern: Optional[str] = "NEUTRAL"  # Pattern detected
    multi_timeframe_confidence: Optional[float] = 0.5  # Hierarchy confidence
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
    ia1_position_id: Optional[str] = None  # 🆕 Reference to IA1 analysis position_id
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
    # 🚀 NEW IA2 STRATEGIC FIELDS
    strategic_reasoning: Optional[str] = "No strategic reasoning provided"
    market_regime_assessment: Optional[str] = "neutral"
    position_size_recommendation: Optional[float] = 2.0
    execution_priority: Optional[str] = "immediate"
    calculated_rr: Optional[float] = 1.0
    rr_reasoning: Optional[str] = "No RR reasoning provided"
    risk_level: Optional[str] = "medium"
    strategy_type: Optional[str] = "strategic_analysis"  # For frontend Strategy field
    # 🎯 NEW IA2 TECHNICAL LEVELS & AUTO-EXECUTION
    ia2_entry_price: Optional[float] = 0.0
    ia2_stop_loss: Optional[float] = 0.0
    ia2_take_profit_1: Optional[float] = 0.0
    ia2_take_profit_2: Optional[float] = 0.0
    ia2_take_profit_3: Optional[float] = 0.0
    ia2_calculated_rr: Optional[float] = 0.0
    trade_execution_ready: Optional[bool] = False
    auto_execution_triggered: Optional[bool] = False
    status: TradingStatus = TradingStatus.PENDING
    # BingX integration fields
    bingx_order_id: Optional[str] = None
    bingx_position_id: Optional[str] = None
    actual_entry_price: Optional[float] = None
    actual_quantity: Optional[float] = None
    bingx_status: Optional[str] = None
    timestamp: datetime = Field(default_factory=get_paris_time)

class TradingPerformance(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    entry_price: float
    exit_price: Optional[float] = None
    position_size: float
    pnl: Optional[float] = None
    duration_minutes: Optional[int] = None
    outcome: Optional[str] = None
    timestamp: datetime = Field(default_factory=get_paris_time)

class PositionTracking(BaseModel):
    """Track IA1→IA2 position flow for resilience and avoid duplicates"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    position_id: str  # Unique position ID from IA1
    symbol: str
    ia1_analysis_id: str  # Reference to TechnicalAnalysis.id
    ia1_timestamp: datetime
    ia1_confidence: float
    ia1_signal: str
    
    # IA2 tracking
    ia2_decision_id: Optional[str] = None  # Reference to TradingDecision.id if processed
    ia2_timestamp: Optional[datetime] = None
    ia2_status: str = "pending"  # pending, processed, failed, skipped
    
    # VOIE tracking
    voie_used: Optional[int] = None  # 1, 2, or 3
    voie_3_eligible: bool = False  # >95% confidence
    
    # Resilience tracking
    processing_attempts: int = 0
    last_attempt: Optional[datetime] = None
    error_message: Optional[str] = None
    
    timestamp: datetime = Field(default_factory=get_paris_time)
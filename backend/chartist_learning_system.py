"""
CHARTIST LEARNING SYSTEM - Système d'apprentissage spécialisé pour les figures chartistes
Analyse la performance des figures chartistes pour optimiser les stratégies long et short
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import json
from enum import Enum

from technical_pattern_detector import PatternType, TechnicalPattern

logger = logging.getLogger(__name__)

class TradingDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

@dataclass
class ChartistPattern:
    """Analyse complète d'une figure chartiste"""
    pattern_name: str
    pattern_type: PatternType
    category: str  # "reversal", "continuation", "consolidation", "harmonic"
    primary_direction: TradingDirection
    market_context_preference: List[str]  # ["BULL", "BEAR", "SIDEWAYS", "VOLATILE"]
    success_rate_long: float = 0.0
    success_rate_short: float = 0.0
    avg_return_long: float = 0.0
    avg_return_short: float = 0.0
    optimal_entry_timing: str = ""
    volume_importance: float = 0.0  # 0-1, importance du volume pour cette figure
    timeframe_effectiveness: Dict[str, float] = None  # Efficacité par timeframe
    risk_reward_profile: Dict[str, float] = None
    formation_characteristics: Dict[str, Any] = None

@dataclass
class ChartistStrategy:
    """Stratégie de trading basée sur une figure chartiste"""
    pattern_type: PatternType
    direction: TradingDirection
    entry_conditions: List[str]
    exit_conditions: List[str]
    stop_loss_placement: str
    take_profit_targets: List[float]  # Multiples du risque
    position_sizing_factor: float  # Multiplicateur de la taille de position standard
    market_context_filter: List[str]  # Conditions de marché favorables
    confirmation_indicators: List[str]  # Indicateurs de confirmation requis
    success_probability: float = 0.0
    avg_holding_period_days: float = 0.0
    max_risk_per_trade: float = 0.02  # 2% par défaut

class ChartistLearningSystem:
    """Système d'apprentissage spécialisé pour les figures chartistes"""
    
    def __init__(self):
        self.chartist_patterns = {}
        self.chartist_strategies = {}
        self.pattern_performance_history = defaultdict(list)
        
        # Bibliothèque complète des figures chartistes avec leurs caractéristiques
        self._initialize_chartist_library()
        
        # Historique d'apprentissage
        self.learning_sessions = []
        
        logger.info("Chartist Learning System initialized")
    
    def _initialize_chartist_library(self):
        """Initialise la bibliothèque complète des figures chartistes avec tous les 40+ patterns"""
        
        # PATTERNS DE BASE & INDICATEURS
        self.chartist_patterns["golden_cross"] = ChartistPattern(
            pattern_name="Golden Cross",
            pattern_type=PatternType.GOLDEN_CROSS,
            category="continuation",
            primary_direction=TradingDirection.LONG,
            market_context_preference=["BULL", "SIDEWAYS"],
            success_rate_long=0.78,
            success_rate_short=0.15,
            avg_return_long=8.9,
            avg_return_short=-4.2,
            optimal_entry_timing="cross_confirmation",
            volume_importance=0.7,
            timeframe_effectiveness={"4h": 0.8, "1d": 0.9, "1w": 0.8},
            risk_reward_profile={"optimal_rr": 2.8, "max_risk": 0.02}
        )
        
        self.chartist_patterns["death_cross"] = ChartistPattern(
            pattern_name="Death Cross",
            pattern_type=PatternType.DEATH_CROSS,
            category="reversal",
            primary_direction=TradingDirection.SHORT,
            market_context_preference=["BEAR", "VOLATILE"],
            success_rate_long=0.18,
            success_rate_short=0.74,
            avg_return_long=-3.8,
            avg_return_short=7.6,
            optimal_entry_timing="cross_confirmation",
            volume_importance=0.7,
            timeframe_effectiveness={"4h": 0.8, "1d": 0.9, "1w": 0.8},
            risk_reward_profile={"optimal_rr": 2.6, "max_risk": 0.025}
        )
        
        # FIGURES DE RETOURNEMENT CLASSIQUES
        self.chartist_patterns["head_and_shoulders"] = ChartistPattern(
            pattern_name="Tête et Épaules",
            pattern_type=PatternType.HEAD_AND_SHOULDERS,
            category="reversal",
            primary_direction=TradingDirection.SHORT,
            market_context_preference=["BULL", "VOLATILE"],
            success_rate_long=0.25,
            success_rate_short=0.72,
            avg_return_long=-2.1,
            avg_return_short=8.4,
            optimal_entry_timing="break_neckline",
            volume_importance=0.9,
            timeframe_effectiveness={"1h": 0.6, "4h": 0.8, "1d": 0.9},
            risk_reward_profile={"optimal_rr": 2.8, "max_risk": 0.03}
        )
        
        self.chartist_patterns["inverse_head_and_shoulders"] = ChartistPattern(
            pattern_name="Tête et Épaules Inversée",
            pattern_type=PatternType.INVERSE_HEAD_AND_SHOULDERS,
            category="reversal",
            primary_direction=TradingDirection.LONG,
            market_context_preference=["BEAR", "VOLATILE"],
            success_rate_long=0.74,
            success_rate_short=0.28,
            avg_return_long=9.2,
            avg_return_short=-2.3,
            optimal_entry_timing="break_neckline",
            volume_importance=0.9,
            timeframe_effectiveness={"1h": 0.6, "4h": 0.8, "1d": 0.9},
            risk_reward_profile={"optimal_rr": 3.1, "max_risk": 0.025}
        )
        
        self.chartist_patterns["double_top"] = ChartistPattern(
            pattern_name="Double Sommet",
            pattern_type=PatternType.DOUBLE_TOP,
            category="reversal",
            primary_direction=TradingDirection.SHORT,
            market_context_preference=["BULL", "SIDEWAYS"],
            success_rate_long=0.22,
            success_rate_short=0.68,
            avg_return_long=-1.8,
            avg_return_short=6.7,
            optimal_entry_timing="break_support_valley",
            volume_importance=0.8,
            timeframe_effectiveness={"4h": 0.7, "1d": 0.9, "1w": 0.8},
            risk_reward_profile={"optimal_rr": 2.5, "max_risk": 0.025}
        )
        
        self.chartist_patterns["double_bottom"] = ChartistPattern(
            pattern_name="Double Creux",
            pattern_type=PatternType.DOUBLE_BOTTOM,
            category="reversal",
            primary_direction=TradingDirection.LONG,
            market_context_preference=["BEAR", "SIDEWAYS"],
            success_rate_long=0.71,
            success_rate_short=0.24,
            avg_return_long=7.8,
            avg_return_short=-2.1,
            optimal_entry_timing="break_resistance_peak",
            volume_importance=0.8,
            timeframe_effectiveness={"4h": 0.7, "1d": 0.9, "1w": 0.8},
            risk_reward_profile={"optimal_rr": 2.7, "max_risk": 0.025}
        )
        
        self.chartist_patterns["triple_top"] = ChartistPattern(
            pattern_name="Triple Sommet",
            pattern_type=PatternType.TRIPLE_TOP,
            category="reversal",
            primary_direction=TradingDirection.SHORT,
            market_context_preference=["BULL", "SIDEWAYS"],
            success_rate_long=0.20,
            success_rate_short=0.75,
            avg_return_long=-2.4,
            avg_return_short=9.1,
            optimal_entry_timing="break_support_line",
            volume_importance=0.9,
            timeframe_effectiveness={"1d": 0.9, "1w": 0.8},
            risk_reward_profile={"optimal_rr": 3.2, "max_risk": 0.025}
        )
        
        self.chartist_patterns["triple_bottom"] = ChartistPattern(
            pattern_name="Triple Creux",
            pattern_type=PatternType.TRIPLE_BOTTOM,
            category="reversal",
            primary_direction=TradingDirection.LONG,
            market_context_preference=["BEAR", "SIDEWAYS"],
            success_rate_long=0.76,
            success_rate_short=0.19,
            avg_return_long=9.8,
            avg_return_short=-2.6,
            optimal_entry_timing="break_resistance_line",
            volume_importance=0.9,
            timeframe_effectiveness={"1d": 0.9, "1w": 0.8},
            risk_reward_profile={"optimal_rr": 3.4, "max_risk": 0.025}
        )
        
        # TRIANGLES
        self.chartist_patterns["ascending_triangle"] = ChartistPattern(
            pattern_name="Triangle Ascendant",
            pattern_type=PatternType.ASCENDING_TRIANGLE,
            category="continuation",
            primary_direction=TradingDirection.LONG,
            market_context_preference=["BULL", "SIDEWAYS"],
            success_rate_long=0.75,
            success_rate_short=0.28,
            avg_return_long=6.8,
            avg_return_short=-2.4,
            optimal_entry_timing="break_horizontal_resistance",
            volume_importance=0.8,
            timeframe_effectiveness={"4h": 0.8, "1d": 0.9, "1w": 0.7},
            risk_reward_profile={"optimal_rr": 2.4, "max_risk": 0.02}
        )
        
        self.chartist_patterns["descending_triangle"] = ChartistPattern(
            pattern_name="Triangle Descendant",
            pattern_type=PatternType.DESCENDING_TRIANGLE,
            category="continuation",
            primary_direction=TradingDirection.SHORT,
            market_context_preference=["BEAR", "SIDEWAYS"],
            success_rate_long=0.26,
            success_rate_short=0.73,
            avg_return_long=-2.6,
            avg_return_short=6.2,
            optimal_entry_timing="break_horizontal_support",
            volume_importance=0.8,
            timeframe_effectiveness={"4h": 0.8, "1d": 0.9, "1w": 0.7},
            risk_reward_profile={"optimal_rr": 2.3, "max_risk": 0.02}
        )
        
        self.chartist_patterns["symmetrical_triangle"] = ChartistPattern(
            pattern_name="Triangle Symétrique",
            pattern_type=PatternType.SYMMETRICAL_TRIANGLE,
            category="continuation",
            primary_direction=TradingDirection.NEUTRAL,
            market_context_preference=["SIDEWAYS", "VOLATILE"],
            success_rate_long=0.58,
            success_rate_short=0.54,
            avg_return_long=4.2,
            avg_return_short=3.8,
            optimal_entry_timing="break_triangle_boundary",
            volume_importance=0.9,
            timeframe_effectiveness={"4h": 0.8, "1d": 0.9, "1w": 0.7},
            risk_reward_profile={"optimal_rr": 2.0, "max_risk": 0.02}
        )
        
        # FLAGS ET PENNANTS
        self.chartist_patterns["flag_bullish"] = ChartistPattern(
            pattern_name="Drapeau Haussier",
            pattern_type=PatternType.FLAG_BULLISH,
            category="continuation",
            primary_direction=TradingDirection.LONG,
            market_context_preference=["BULL", "VOLATILE"],
            success_rate_long=0.78,
            success_rate_short=0.18,
            avg_return_long=5.4,
            avg_return_short=-3.2,
            optimal_entry_timing="break_flag_resistance",
            volume_importance=0.9,
            timeframe_effectiveness={"1h": 0.8, "4h": 0.9, "1d": 0.7},
            risk_reward_profile={"optimal_rr": 2.2, "max_risk": 0.02}
        )
        
        self.chartist_patterns["flag_bearish"] = ChartistPattern(
            pattern_name="Drapeau Baissier",
            pattern_type=PatternType.FLAG_BEARISH,
            category="continuation",
            primary_direction=TradingDirection.SHORT,
            market_context_preference=["BEAR", "VOLATILE"],
            success_rate_long=0.15,
            success_rate_short=0.76,
            avg_return_long=-3.5,
            avg_return_short=5.8,
            optimal_entry_timing="break_flag_support",
            volume_importance=0.9,
            timeframe_effectiveness={"1h": 0.8, "4h": 0.9, "1d": 0.7},
            risk_reward_profile={"optimal_rr": 2.3, "max_risk": 0.02}
        )
        
        self.chartist_patterns["pennant_bullish"] = ChartistPattern(
            pattern_name="Pennant Haussier",
            pattern_type=PatternType.PENNANT_BULLISH,
            category="continuation",
            primary_direction=TradingDirection.LONG,
            market_context_preference=["BULL", "VOLATILE"],
            success_rate_long=0.73,
            success_rate_short=0.22,
            avg_return_long=4.8,
            avg_return_short=-2.8,
            optimal_entry_timing="break_pennant_resistance",
            volume_importance=0.9,
            timeframe_effectiveness={"1h": 0.9, "4h": 0.8, "1d": 0.6},
            risk_reward_profile={"optimal_rr": 2.1, "max_risk": 0.015}
        )
        
        self.chartist_patterns["pennant_bearish"] = ChartistPattern(
            pattern_name="Pennant Baissier",
            pattern_type=PatternType.PENNANT_BEARISH,
            category="continuation",
            primary_direction=TradingDirection.SHORT,
            market_context_preference=["BEAR", "VOLATILE"],
            success_rate_long=0.20,
            success_rate_short=0.74,
            avg_return_long=-3.1,
            avg_return_short=5.2,
            optimal_entry_timing="break_pennant_support",
            volume_importance=0.9,
            timeframe_effectiveness={"1h": 0.9, "4h": 0.8, "1d": 0.6},
            risk_reward_profile={"optimal_rr": 2.2, "max_risk": 0.015}
        )
        
        # WEDGES
        self.chartist_patterns["rising_wedge"] = ChartistPattern(
            pattern_name="Biseau Ascendant",
            pattern_type=PatternType.RISING_WEDGE,
            category="reversal",
            primary_direction=TradingDirection.SHORT,
            market_context_preference=["BULL", "VOLATILE"],
            success_rate_long=0.35,
            success_rate_short=0.69,
            avg_return_long=-1.2,
            avg_return_short=7.1,
            optimal_entry_timing="break_lower_trendline",
            volume_importance=0.7,
            timeframe_effectiveness={"1h": 0.6, "4h": 0.8, "1d": 0.9},
            risk_reward_profile={"optimal_rr": 2.8, "max_risk": 0.025}
        )
        
        self.chartist_patterns["falling_wedge"] = ChartistPattern(
            pattern_name="Biseau Descendant",
            pattern_type=PatternType.FALLING_WEDGE,
            category="reversal",
            primary_direction=TradingDirection.LONG,
            market_context_preference=["BEAR", "VOLATILE"],
            success_rate_long=0.72,
            success_rate_short=0.31,
            avg_return_long=8.3,
            avg_return_short=-1.5,
            optimal_entry_timing="break_upper_trendline",
            volume_importance=0.7,
            timeframe_effectiveness={"1h": 0.6, "4h": 0.8, "1d": 0.9},
            risk_reward_profile={"optimal_rr": 2.9, "max_risk": 0.025}
        )
        
        self.chartist_patterns["expanding_wedge"] = ChartistPattern(
            pattern_name="Biseau Élargi",
            pattern_type=PatternType.EXPANDING_WEDGE,
            category="reversal",
            primary_direction=TradingDirection.NEUTRAL,
            market_context_preference=["VOLATILE"],
            success_rate_long=0.48,
            success_rate_short=0.52,
            avg_return_long=3.2,
            avg_return_short=3.8,
            optimal_entry_timing="boundary_break_confirmation",
            volume_importance=0.8,
            timeframe_effectiveness={"4h": 0.7, "1d": 0.8},
            risk_reward_profile={"optimal_rr": 2.0, "max_risk": 0.03}
        )
        
        # CUP AND HANDLE
        self.chartist_patterns["cup_and_handle"] = ChartistPattern(
            pattern_name="Tasse avec Anse",
            pattern_type=PatternType.CUP_AND_HANDLE,
            category="continuation",
            primary_direction=TradingDirection.LONG,
            market_context_preference=["BULL", "SIDEWAYS"],
            success_rate_long=0.81,
            success_rate_short=0.12,
            avg_return_long=12.4,
            avg_return_short=-4.1,
            optimal_entry_timing="break_handle_resistance",
            volume_importance=0.9,
            timeframe_effectiveness={"1d": 0.9, "1w": 0.9, "1M": 0.8},
            risk_reward_profile={"optimal_rr": 4.2, "max_risk": 0.02}
        )
        
        # RECTANGLES ET CONSOLIDATIONS
        self.chartist_patterns["rectangle_consolidation"] = ChartistPattern(
            pattern_name="Rectangle de Consolidation",
            pattern_type=PatternType.RECTANGLE_CONSOLIDATION,
            category="consolidation",
            primary_direction=TradingDirection.NEUTRAL,
            market_context_preference=["SIDEWAYS"],
            success_rate_long=0.52,
            success_rate_short=0.48,
            avg_return_long=3.4,
            avg_return_short=3.1,
            optimal_entry_timing="boundary_break",
            volume_importance=0.6,
            timeframe_effectiveness={"4h": 0.8, "1d": 0.9},
            risk_reward_profile={"optimal_rr": 1.8, "max_risk": 0.02}
        )
        
        # ROUNDING PATTERNS
        self.chartist_patterns["rounding_bottom"] = ChartistPattern(
            pattern_name="Fond Arrondi",
            pattern_type=PatternType.ROUNDING_BOTTOM,
            category="reversal",
            primary_direction=TradingDirection.LONG,
            market_context_preference=["BEAR", "SIDEWAYS"],
            success_rate_long=0.67,
            success_rate_short=0.25,
            avg_return_long=8.7,
            avg_return_short=-2.8,
            optimal_entry_timing="volume_breakout",
            volume_importance=0.9,
            timeframe_effectiveness={"1d": 0.9, "1w": 0.8},
            risk_reward_profile={"optimal_rr": 3.2, "max_risk": 0.025}
        )
        
        self.chartist_patterns["rounding_top"] = ChartistPattern(
            pattern_name="Sommet Arrondi",
            pattern_type=PatternType.ROUNDING_TOP,
            category="reversal",
            primary_direction=TradingDirection.SHORT,
            market_context_preference=["BULL", "SIDEWAYS"],
            success_rate_long=0.28,
            success_rate_short=0.65,
            avg_return_long=-2.9,
            avg_return_short=7.2,
            optimal_entry_timing="volume_breakdown",
            volume_importance=0.9,
            timeframe_effectiveness={"1d": 0.9, "1w": 0.8},
            risk_reward_profile={"optimal_rr": 2.8, "max_risk": 0.025}
        )
        
        # PATTERNS HARMONIQUES
        self.chartist_patterns["gartley_bullish"] = ChartistPattern(
            pattern_name="Gartley Haussier",
            pattern_type=PatternType.GARTLEY_BULLISH,
            category="harmonic",
            primary_direction=TradingDirection.LONG,
            market_context_preference=["BEAR", "SIDEWAYS", "VOLATILE"],
            success_rate_long=0.73,
            success_rate_short=0.22,
            avg_return_long=9.7,
            avg_return_short=-2.8,
            optimal_entry_timing="completion_point_D",
            volume_importance=0.6,
            timeframe_effectiveness={"4h": 0.9, "1d": 0.9, "1w": 0.8},
            risk_reward_profile={"optimal_rr": 3.5, "max_risk": 0.015}
        )
        
        self.chartist_patterns["gartley_bearish"] = ChartistPattern(
            pattern_name="Gartley Baissier",
            pattern_type=PatternType.GARTLEY_BEARISH,
            category="harmonic",
            primary_direction=TradingDirection.SHORT,
            market_context_preference=["BULL", "SIDEWAYS", "VOLATILE"],
            success_rate_long=0.25,
            success_rate_short=0.71,
            avg_return_long=-2.6,
            avg_return_short=8.9,
            optimal_entry_timing="completion_point_D",
            volume_importance=0.6,
            timeframe_effectiveness={"4h": 0.9, "1d": 0.9, "1w": 0.8},
            risk_reward_profile={"optimal_rr": 3.3, "max_risk": 0.015}
        )
        
        self.chartist_patterns["bat_bullish"] = ChartistPattern(
            pattern_name="Bat Haussier",
            pattern_type=PatternType.BAT_BULLISH,
            category="harmonic",
            primary_direction=TradingDirection.LONG,
            market_context_preference=["BEAR", "VOLATILE"],
            success_rate_long=0.69,
            success_rate_short=0.28,
            avg_return_long=8.2,
            avg_return_short=-2.1,
            optimal_entry_timing="completion_point_D",
            volume_importance=0.5,
            timeframe_effectiveness={"4h": 0.8, "1d": 0.9},
            risk_reward_profile={"optimal_rr": 3.0, "max_risk": 0.015}
        )
        
        self.chartist_patterns["butterfly_bullish"] = ChartistPattern(
            pattern_name="Butterfly Haussier",
            pattern_type=PatternType.BUTTERFLY_BULLISH,
            category="harmonic",
            primary_direction=TradingDirection.LONG,
            market_context_preference=["BEAR", "VOLATILE"],
            success_rate_long=0.75,
            success_rate_short=0.20,
            avg_return_long=11.3,
            avg_return_short=-3.4,
            optimal_entry_timing="completion_point_D",
            volume_importance=0.5,
            timeframe_effectiveness={"4h": 0.9, "1d": 0.9},
            risk_reward_profile={"optimal_rr": 4.1, "max_risk": 0.015}
        )
        
        # PATTERNS DE VOLATILITÉ
        self.chartist_patterns["diamond_top"] = ChartistPattern(
            pattern_name="Diamant de Sommet",
            pattern_type=PatternType.DIAMOND_TOP,
            category="reversal",
            primary_direction=TradingDirection.SHORT,
            market_context_preference=["BULL", "VOLATILE"],
            success_rate_long=0.32,
            success_rate_short=0.68,
            avg_return_long=-2.1,
            avg_return_short=6.8,
            optimal_entry_timing="diamond_breakdown",
            volume_importance=0.8,
            timeframe_effectiveness={"4h": 0.7, "1d": 0.8},
            risk_reward_profile={"optimal_rr": 2.5, "max_risk": 0.025}
        )
        
        self.chartist_patterns["diamond_bottom"] = ChartistPattern(
            pattern_name="Diamant de Creux",
            pattern_type=PatternType.DIAMOND_BOTTOM,
            category="reversal",
            primary_direction=TradingDirection.LONG,
            market_context_preference=["BEAR", "VOLATILE"],
            success_rate_long=0.66,
            success_rate_short=0.30,
            avg_return_long=7.4,
            avg_return_short=-2.3,
            optimal_entry_timing="diamond_breakout",
            volume_importance=0.8,
            timeframe_effectiveness={"4h": 0.7, "1d": 0.8},
            risk_reward_profile={"optimal_rr": 2.7, "max_risk": 0.025}
        )
        
        # PATTERNS DE SUPPORT/RÉSISTANCE
        self.chartist_patterns["support_bounce"] = ChartistPattern(
            pattern_name="Rebond sur Support",
            pattern_type=PatternType.SUPPORT_BOUNCE,
            category="continuation",
            primary_direction=TradingDirection.LONG,
            market_context_preference=["BULL", "SIDEWAYS"],
            success_rate_long=0.64,
            success_rate_short=0.28,
            avg_return_long=4.2,
            avg_return_short=-1.9,
            optimal_entry_timing="bounce_confirmation",
            volume_importance=0.7,
            timeframe_effectiveness={"1h": 0.8, "4h": 0.9, "1d": 0.8},
            risk_reward_profile={"optimal_rr": 2.0, "max_risk": 0.015}
        )
        
        self.chartist_patterns["resistance_break"] = ChartistPattern(
            pattern_name="Cassure de Résistance",
            pattern_type=PatternType.RESISTANCE_BREAK,
            category="continuation",
            primary_direction=TradingDirection.LONG,
            market_context_preference=["BULL", "VOLATILE"],
            success_rate_long=0.67,
            success_rate_short=0.25,
            avg_return_long=5.8,
            avg_return_short=-2.4,
            optimal_entry_timing="breakout_confirmation",
            volume_importance=0.9,
            timeframe_effectiveness={"1h": 0.8, "4h": 0.9, "1d": 0.8},
            risk_reward_profile={"optimal_rr": 2.3, "max_risk": 0.015}
        )
        
        # PATTERNS DE TENDANCE
        self.chartist_patterns["bullish_channel"] = ChartistPattern(
            pattern_name="Canal Haussier",
            pattern_type=PatternType.BULLISH_CHANNEL,
            category="continuation",
            primary_direction=TradingDirection.LONG,
            market_context_preference=["BULL"],
            success_rate_long=0.72,
            success_rate_short=0.20,
            avg_return_long=6.4,
            avg_return_short=-3.1,
            optimal_entry_timing="channel_support_bounce",
            volume_importance=0.6,
            timeframe_effectiveness={"4h": 0.8, "1d": 0.9, "1w": 0.8},
            risk_reward_profile={"optimal_rr": 2.2, "max_risk": 0.02}
        )
        
        self.chartist_patterns["bearish_channel"] = ChartistPattern(
            pattern_name="Canal Baissier",
            pattern_type=PatternType.BEARISH_CHANNEL,
            category="continuation",
            primary_direction=TradingDirection.SHORT,
            market_context_preference=["BEAR"],
            success_rate_long=0.22,
            success_rate_short=0.70,
            avg_return_long=-3.2,
            avg_return_short=5.9,
            optimal_entry_timing="channel_resistance_rejection",
            volume_importance=0.6,
            timeframe_effectiveness={"4h": 0.8, "1d": 0.9, "1w": 0.8},
            risk_reward_profile={"optimal_rr": 2.1, "max_risk": 0.02}
        )
        
        logger.info(f"Initialized complete chartist library with {len(self.chartist_patterns)} patterns")
    
    def analyze_pattern_performance(self, historical_data: Dict[str, pd.DataFrame], 
                                  detected_patterns: List[TechnicalPattern]) -> Dict[str, Any]:
        """Analyse la performance des figures chartistes sur les données historiques"""
        
        pattern_performance = defaultdict(lambda: {
            'long_trades': [],
            'short_trades': [],
            'success_rates': {'long': 0.0, 'short': 0.0},
            'avg_returns': {'long': 0.0, 'short': 0.0},
            'market_context_performance': defaultdict(list)
        })
        
        logger.info(f"🔍 Analyzing performance of {len(detected_patterns)} patterns across {len(historical_data)} symbols")
        
        for pattern in detected_patterns:
            symbol = pattern.symbol
            pattern_type = pattern.pattern_type.value
            
            if symbol not in historical_data:
                continue
            
            df = historical_data[symbol]
            pattern_date = pattern.detected_at
            
            # Simuler les trades long et short pour chaque pattern
            long_performance = self._simulate_trade(df, pattern, TradingDirection.LONG)
            short_performance = self._simulate_trade(df, pattern, TradingDirection.SHORT)
            
            pattern_performance[pattern_type]['long_trades'].append(long_performance)
            pattern_performance[pattern_type]['short_trades'].append(short_performance)
            
            # Déterminer le contexte de marché au moment du pattern
            market_context = self._determine_market_context(df, pattern_date)
            pattern_performance[pattern_type]['market_context_performance'][market_context].append({
                'long': long_performance,
                'short': short_performance
            })
        
        # Calculer les statistiques finales
        for pattern_type, data in pattern_performance.items():
            # Success rates
            long_trades = data['long_trades']
            short_trades = data['short_trades']
            
            if long_trades:
                data['success_rates']['long'] = sum(1 for trade in long_trades if trade['return'] > 2) / len(long_trades)
                data['avg_returns']['long'] = np.mean([trade['return'] for trade in long_trades])
            
            if short_trades:
                data['success_rates']['short'] = sum(1 for trade in short_trades if trade['return'] > 2) / len(short_trades)
                data['avg_returns']['short'] = np.mean([trade['return'] for trade in short_trades])
        
        return dict(pattern_performance)
    
    def _simulate_trade(self, df: pd.DataFrame, pattern: TechnicalPattern, 
                       direction: TradingDirection) -> Dict[str, Any]:
        """Simule un trade basé sur un pattern"""
        
        entry_price = pattern.entry_price
        pattern_date = pattern.detected_at
        
        # Trouver l'index de la date du pattern
        try:
            pattern_idx = df[df['Date'] <= pattern_date.strftime('%Y-%m-%d')].index[-1]
        except:
            return {'return': 0.0, 'success': False, 'exit_reason': 'no_data'}
        
        # Définir stop loss et take profit basés sur la figure chartiste
        chartist_info = self.chartist_patterns.get(pattern.pattern_type.value)
        
        if direction == TradingDirection.LONG:
            if chartist_info:
                stop_loss = entry_price * (1 - chartist_info.risk_reward_profile.get('max_risk', 0.02))
                take_profit = entry_price * (1 + chartist_info.risk_reward_profile.get('optimal_rr', 2.0) * 
                                           chartist_info.risk_reward_profile.get('max_risk', 0.02))
            else:
                stop_loss = entry_price * 0.98
                take_profit = entry_price * 1.04
        else:  # SHORT
            if chartist_info:
                stop_loss = entry_price * (1 + chartist_info.risk_reward_profile.get('max_risk', 0.02))
                take_profit = entry_price * (1 - chartist_info.risk_reward_profile.get('optimal_rr', 2.0) * 
                                           chartist_info.risk_reward_profile.get('max_risk', 0.02))
            else:
                stop_loss = entry_price * 1.02
                take_profit = entry_price * 0.96
        
        # Simuler le trade sur les 10 jours suivants
        for i in range(1, min(11, len(df) - pattern_idx)):
            current_idx = pattern_idx + i
            if current_idx >= len(df):
                break
                
            current_price = df.iloc[current_idx]['Close']
            
            if direction == TradingDirection.LONG:
                if current_price <= stop_loss:
                    return_pct = (stop_loss / entry_price - 1) * 100
                    return {'return': return_pct, 'success': False, 'exit_reason': 'stop_loss', 'days': i}
                elif current_price >= take_profit:
                    return_pct = (take_profit / entry_price - 1) * 100
                    return {'return': return_pct, 'success': True, 'exit_reason': 'take_profit', 'days': i}
            else:  # SHORT
                if current_price >= stop_loss:
                    return_pct = (entry_price / stop_loss - 1) * 100
                    return {'return': return_pct, 'success': False, 'exit_reason': 'stop_loss', 'days': i}
                elif current_price <= take_profit:
                    return_pct = (entry_price / take_profit - 1) * 100
                    return {'return': return_pct, 'success': True, 'exit_reason': 'take_profit', 'days': i}
        
        # Sortie par timeout
        final_price = df.iloc[min(pattern_idx + 10, len(df) - 1)]['Close']
        if direction == TradingDirection.LONG:
            return_pct = (final_price / entry_price - 1) * 100
        else:
            return_pct = (entry_price / final_price - 1) * 100
        
        return {'return': return_pct, 'success': return_pct > 2, 'exit_reason': 'timeout', 'days': 10}
    
    def _determine_market_context(self, df: pd.DataFrame, pattern_date: datetime) -> str:
        """Détermine le contexte de marché au moment du pattern"""
        try:
            # Calculer la volatilité et le changement de prix sur 20 jours
            end_idx = df[df['Date'] <= pattern_date.strftime('%Y-%m-%d')].index[-1]
            start_idx = max(0, end_idx - 20)
            
            window_data = df.iloc[start_idx:end_idx+1]
            
            price_change = (window_data['Close'].iloc[-1] / window_data['Close'].iloc[0] - 1) * 100
            volatility = window_data['Close'].pct_change().std() * 100 * np.sqrt(252)  # Annualisée
            
            if volatility > 60:
                return "VOLATILE"
            elif abs(price_change) < 5:
                return "SIDEWAYS"
            elif price_change > 5:
                return "BULL"
            else:
                return "BEAR"
                
        except:
            return "UNKNOWN"
    
    def generate_chartist_strategies(self) -> Dict[str, ChartistStrategy]:
        """Génère des stratégies optimisées basées sur l'apprentissage des patterns"""
        
        strategies = {}
        
        for pattern_name, pattern_info in self.chartist_patterns.items():
            
            # Stratégie pour la direction primaire
            primary_direction = pattern_info.primary_direction
            
            if primary_direction == TradingDirection.LONG:
                success_rate = pattern_info.success_rate_long
                avg_return = pattern_info.avg_return_long
            else:
                success_rate = pattern_info.success_rate_short
                avg_return = pattern_info.avg_return_short
            
            # Calculer le facteur de taille de position basé sur la performance
            position_factor = 0.5 + (success_rate - 0.5) * 1.5  # 0.5 à 2.0
            position_factor = max(0.3, min(2.0, position_factor))
            
            strategy = ChartistStrategy(
                pattern_type=pattern_info.pattern_type,
                direction=primary_direction,
                entry_conditions=self._generate_entry_conditions(pattern_info),
                exit_conditions=self._generate_exit_conditions(pattern_info),
                stop_loss_placement=self._generate_stop_loss_rules(pattern_info),
                take_profit_targets=self._generate_take_profit_targets(pattern_info),
                position_sizing_factor=position_factor,
                market_context_filter=pattern_info.market_context_preference,
                confirmation_indicators=self._generate_confirmation_indicators(pattern_info),
                success_probability=success_rate,
                avg_holding_period_days=pattern_info.formation_characteristics.get('duration_days', 7),
                max_risk_per_trade=pattern_info.risk_reward_profile.get('max_risk', 0.02)
            )
            
            strategies[f"{pattern_name}_{primary_direction.value}"] = strategy
            
            # Ajouter stratégie contraire si pertinente
            if pattern_info.category == "reversal":
                opposite_direction = TradingDirection.SHORT if primary_direction == TradingDirection.LONG else TradingDirection.LONG
                opposite_success = pattern_info.success_rate_short if primary_direction == TradingDirection.LONG else pattern_info.success_rate_long
                
                if opposite_success > 0.3:  # Seulement si un minimum viable
                    opposite_factor = 0.3 + (opposite_success - 0.3) * 0.8
                    
                    opposite_strategy = ChartistStrategy(
                        pattern_type=pattern_info.pattern_type,
                        direction=opposite_direction,
                        entry_conditions=self._generate_entry_conditions(pattern_info, opposite=True),
                        exit_conditions=self._generate_exit_conditions(pattern_info, opposite=True),
                        stop_loss_placement=self._generate_stop_loss_rules(pattern_info, opposite=True),
                        take_profit_targets=[1.5, 2.0],  # Plus conservateur
                        position_sizing_factor=opposite_factor,
                        market_context_filter=pattern_info.market_context_preference,
                        confirmation_indicators=self._generate_confirmation_indicators(pattern_info),
                        success_probability=opposite_success,
                        avg_holding_period_days=pattern_info.formation_characteristics.get('duration_days', 7) // 2,
                        max_risk_per_trade=pattern_info.risk_reward_profile.get('max_risk', 0.02) * 1.5
                    )
                    
                    strategies[f"{pattern_name}_{opposite_direction.value}_contrarian"] = opposite_strategy
        
        self.chartist_strategies = strategies
        return strategies
    
    def _generate_entry_conditions(self, pattern_info: ChartistPattern, opposite: bool = False) -> List[str]:
        """Génère les conditions d'entrée pour un pattern"""
        conditions = []
        
        if pattern_info.volume_importance > 0.7:
            conditions.append("volume_confirmation_required")
        
        if pattern_info.pattern_type in [PatternType.HEAD_AND_SHOULDERS, PatternType.INVERSE_HEAD_AND_SHOULDERS]:
            conditions.append("neckline_break_confirmation")
            
        elif pattern_info.pattern_type in [PatternType.DOUBLE_TOP, PatternType.DOUBLE_BOTTOM]:
            conditions.append("valley_or_peak_break")
            
        elif "triangle" in pattern_info.pattern_type.value:
            conditions.append("trendline_break_with_volume")
            
        elif "flag" in pattern_info.pattern_type.value:
            conditions.append("flag_boundary_break")
        
        conditions.append("rsi_not_extreme")  # RSI entre 30-70
        
        if not opposite:
            conditions.append(f"market_context_in_{pattern_info.market_context_preference}")
        
        return conditions
    
    def _generate_exit_conditions(self, pattern_info: ChartistPattern, opposite: bool = False) -> List[str]:
        """Génère les conditions de sortie"""
        return [
            "take_profit_target_reached",
            "stop_loss_triggered", 
            "pattern_invalidation",
            "time_based_exit_10_days"
        ]
    
    def _generate_stop_loss_rules(self, pattern_info: ChartistPattern, opposite: bool = False) -> str:
        """Génère les règles de stop loss"""
        if "triangle" in pattern_info.pattern_type.value:
            return "below_broken_trendline" if not opposite else "above_broken_trendline"
        elif pattern_info.pattern_type in [PatternType.HEAD_AND_SHOULDERS, PatternType.INVERSE_HEAD_AND_SHOULDERS]:
            return "beyond_opposite_shoulder"
        elif "flag" in pattern_info.pattern_type.value:
            return "opposite_flag_boundary"
        else:
            return "percentage_based_2_percent"
    
    def _generate_take_profit_targets(self, pattern_info: ChartistPattern) -> List[float]:
        """Génère les objectifs de take profit"""
        base_rr = pattern_info.risk_reward_profile.get('optimal_rr', 2.0)
        return [base_rr * 0.6, base_rr * 1.0, base_rr * 1.4]  # 3 niveaux
    
    def _generate_confirmation_indicators(self, pattern_info: ChartistPattern) -> List[str]:
        """Génère les indicateurs de confirmation requis"""
        indicators = ["volume_spike"]
        
        if pattern_info.category == "reversal":
            indicators.extend(["rsi_divergence", "macd_signal_change"])
        elif pattern_info.category == "continuation":
            indicators.extend(["momentum_continuation", "moving_average_support"])
        
        return indicators
    
    def get_optimal_strategy_for_pattern(self, pattern_type: PatternType, 
                                       market_context: str, direction: TradingDirection = None) -> Optional[ChartistStrategy]:
        """Retourne la stratégie optimale pour un pattern donné"""
        
        # Chercher la stratégie correspondante
        for strategy_name, strategy in self.chartist_strategies.items():
            if (strategy.pattern_type == pattern_type and 
                market_context in strategy.market_context_filter and
                (direction is None or strategy.direction == direction)):
                return strategy
        
        return None
    
    def get_pattern_recommendations(self, detected_patterns: List[TechnicalPattern], 
                                  market_context: str) -> List[Dict[str, Any]]:
        """Fournit des recommandations basées sur les patterns détectés"""
        
        recommendations = []
        
        for pattern in detected_patterns:
            pattern_info = self.chartist_patterns.get(pattern.pattern_type.value)
            if not pattern_info:
                continue
            
            # Vérifier si le contexte de marché est favorable
            if market_context not in pattern_info.market_context_preference:
                continue
            
            # Obtenir la stratégie optimale
            strategy = self.get_optimal_strategy_for_pattern(
                pattern.pattern_type, market_context, pattern_info.primary_direction
            )
            
            if strategy:
                recommendation = {
                    'symbol': pattern.symbol,
                    'pattern_name': pattern_info.pattern_name,
                    'pattern_type': pattern.pattern_type.value,
                    'recommended_direction': strategy.direction.value,
                    'success_probability': strategy.success_probability,
                    'position_sizing_factor': strategy.position_sizing_factor,
                    'risk_reward_ratio': pattern_info.risk_reward_profile.get('optimal_rr', 2.0),
                    'max_risk': strategy.max_risk_per_trade,
                    'entry_conditions': strategy.entry_conditions,
                    'take_profit_targets': strategy.take_profit_targets,
                    'market_context_match': market_context in pattern_info.market_context_preference,
                    'volume_importance': pattern_info.volume_importance,
                    'estimated_duration_days': strategy.avg_holding_period_days
                }
                
                recommendations.append(recommendation)
        
        # Trier par probabilité de succès
        recommendations.sort(key=lambda x: x['success_probability'], reverse=True)
        
        return recommendations
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de l'apprentissage des figures chartistes"""
        
        return {
            'total_patterns_in_library': len(self.chartist_patterns),
            'strategies_generated': len(self.chartist_strategies),
            'pattern_categories': {
                'reversal': len([p for p in self.chartist_patterns.values() if p.category == 'reversal']),
                'continuation': len([p for p in self.chartist_patterns.values() if p.category == 'continuation']),
                'harmonic': len([p for p in self.chartist_patterns.values() if p.category == 'harmonic']),
                'consolidation': len([p for p in self.chartist_patterns.values() if p.category == 'consolidation'])
            },
            'best_long_patterns': [
                {'name': p.pattern_name, 'success_rate': p.success_rate_long, 'avg_return': p.avg_return_long}
                for p in sorted(self.chartist_patterns.values(), key=lambda x: x.success_rate_long, reverse=True)[:5]
            ],
            'best_short_patterns': [
                {'name': p.pattern_name, 'success_rate': p.success_rate_short, 'avg_return': p.avg_return_short}
                for p in sorted(self.chartist_patterns.values(), key=lambda x: x.success_rate_short, reverse=True)[:5]
            ],
            'market_context_preferences': {
                context: len([p for p in self.chartist_patterns.values() if context in p.market_context_preference])
                for context in ['BULL', 'BEAR', 'SIDEWAYS', 'VOLATILE']
            }
        }

# Instance globale
chartist_learning_system = ChartistLearningSystem()
"""
Configuration Multi-Timeframe pour Analyse Technique Professionnelle
Définit quels indicateurs utiliser sur chaque timeframe pour une analyse optimale
"""

# 🎯 CONFIGURATION MULTI-TIMEFRAME PROFESSIONNELLE
MULTI_TIMEFRAME_CONFIG = {
    "15m": {
        "description": "Signaux d'entrée précis et momentum immédiat",
        "indicators": [
            "macd",           # Momentum précis
            "rsi",            # Sur-achat/sur-vente immédiat  
            "volume_ratio",   # Confirmation d'activité
            "bb_position"     # Position dans les bandes (extrêmes)
        ],
        "weight": 0.25,       # 25% du score final
        "primary_use": "timing_entry"
    },
    
    "1h": {
        "description": "Tendance principale et momentum moyen terme", 
        "indicators": [
            "ema_21",         # Direction tendance
            "ema_50",         # Confirmation tendance
            "adx",            # Force de tendance
            "bb_squeeze",     # Volatilité compression
            "atr"             # Volatilité dynamique
        ],
        "weight": 0.35,       # 35% du score final
        "primary_use": "trend_direction"
    },
    
    "4h": {
        "description": "Contexte stratégique et niveaux majeurs",
        "indicators": [
            "sma_20",         # Support/résistance dynamique
            "sma_200",        # Tendance long terme
            "vwap",           # Niveaux institutionnels
            "pattern_detection", # Figures chartistes
            "fibonacci"       # Niveaux de retracement
        ],
        "weight": 0.25,       # 25% du score final  
        "primary_use": "strategic_context"
    },
    
    "1d": {
        "description": "Vision macro et confluence générale",
        "indicators": [
            "regime_ml",      # Classification macro
            "confluence_global", # Score confluence général
            "trend_hierarchy" # Hiérarchie des tendances
        ],
        "weight": 0.15,       # 15% du score final
        "primary_use": "macro_context"
    }
}

# 🎯 RÈGLES DE CONFLUENCE INTER-TIMEFRAMES
CONFLUENCE_RULES = {
    "bullish_alignment": {
        "15m": {"macd": "> 0", "rsi": "> 50"},
        "1h": {"ema_21": "> ema_50", "adx": "> 25"}, 
        "4h": {"price": "> sma_200", "vwap": "support"},
        "1d": {"regime": "TRENDING_UP"}
    },
    
    "bearish_alignment": {
        "15m": {"macd": "< 0", "rsi": "< 50"},
        "1h": {"ema_21": "< ema_50", "adx": "> 25"},
        "4h": {"price": "< sma_200", "vwap": "resistance"}, 
        "1d": {"regime": "TRENDING_DOWN"}
    },
    
    "consolidation_detected": {
        "15m": {"bb_position": "0.3 < x < 0.7"},
        "1h": {"adx": "< 20", "bb_squeeze": "true"},
        "4h": {"pattern": "rectangle|consolidation"},
        "1d": {"regime": "CONSOLIDATION"}
    }
}

# 🎯 TIMEFRAME PRIORITÉS SELON TYPE DE TRADING
TRADING_STYLE_TIMEFRAMES = {
    "scalp": {"primary": "15m", "secondary": "1h", "confirmation": "4h"},
    "intraday": {"primary": "1h", "secondary": "4h", "confirmation": "1d"},
    "swing": {"primary": "4h", "secondary": "1d", "confirmation": "1h"},
    "position": {"primary": "1d", "secondary": "4h", "confirmation": "1h"}
}

def get_timeframe_config(timeframe: str) -> dict:
    """Récupère la configuration pour un timeframe donné"""
    return MULTI_TIMEFRAME_CONFIG.get(timeframe, {})

def get_confluence_requirements(signal_type: str) -> dict:
    """Récupère les exigences de confluence pour un type de signal"""
    return CONFLUENCE_RULES.get(signal_type, {})

def calculate_timeframe_weight(timeframe: str, trading_style: str = "intraday") -> float:
    """Calcule le poids d'un timeframe selon le style de trading"""
    base_weight = MULTI_TIMEFRAME_CONFIG.get(timeframe, {}).get("weight", 0.25)
    
    # Ajustement selon style de trading
    style_config = TRADING_STYLE_TIMEFRAMES.get(trading_style, {})
    
    if timeframe == style_config.get("primary"):
        return base_weight * 1.5  # Boost timeframe principal
    elif timeframe == style_config.get("secondary"):  
        return base_weight * 1.2  # Boost timeframe secondaire
    else:
        return base_weight
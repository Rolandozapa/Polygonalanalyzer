"""
Base indicator system for modular technical analysis
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class IndicatorResult:
    """Base result structure for technical indicators"""
    value: Optional[float] = None
    signal: str = "NEUTRAL"
    confidence: float = 0.5
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass 
class TechnicalAnalysisComplete:
    """Complete technical analysis structure"""
    
    # Core indicators
    rsi_14: float = 50.0
    rsi_9: float = 50.0  
    rsi_21: float = 50.0
    rsi_zone: str = "NEUTRAL"
    
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    macd_trend: str = "NEUTRAL"
    
    adx: float = 25.0
    plus_di: float = 25.0
    minus_di: float = 25.0
    adx_strength: str = "MODERATE"
    
    bb_upper: float = 0.0
    bb_middle: float = 0.0  
    bb_lower: float = 0.0
    bb_position: float = 0.5
    bb_squeeze: bool = False
    squeeze_intensity: str = "NONE"
    
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    
    mfi: float = 50.0
    mfi_signal: str = "NEUTRAL"
    
    atr: float = 0.02
    atr_pct: float = 2.0
    
    vwap: float = 0.0
    vwap_distance: float = 0.0
    above_vwap: bool = True
    
    # Moving averages
    ema_9: float = 0.0
    ema_21: float = 0.0
    ema_50: float = 0.0
    ema_200: float = 0.0
    sma_20: float = 0.0
    sma_50: float = 0.0
    
    # Trend analysis
    trend_hierarchy: str = "NEUTRAL"
    trend_strength_score: float = 0.5
    price_vs_emas: str = "NEUTRAL"
    
    # Volume analysis
    volume_ratio: float = 1.0
    volume_trend: str = "NEUTRAL"
    volume_surge: bool = False
    
    # Market regime
    regime: str = "CONSOLIDATION"
    confidence: float = 0.5
    technical_consistency: float = 0.5
    
    # Confluence
    confluence_grade: str = "C"
    confluence_score: int = 50
    should_trade: bool = False
    conviction_level: str = "FAIBLE"

class BaseIndicator(ABC):
    """
    Abstract base class for all technical indicators
    Provides common functionality and interface
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate the indicator value
        
        Args:
            data: OHLCV DataFrame with columns [high, low, close, volume]
            
        Returns:
            IndicatorResult with value, signal, and metadata
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data format and requirements
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check minimum length
            min_length = self.get_min_periods()
            if len(data) < min_length:
                logger.warning(f"{self.name}: Insufficient data {len(data)} < {min_length}")
                return False
            
            # Check required columns
            required_cols = self.get_required_columns()
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.error(f"{self.name}: Missing columns {missing_cols}")
                return False
            
            # Check for NaN values in critical columns
            for col in required_cols:
                if data[col].isna().any():
                    logger.warning(f"{self.name}: NaN values found in {col}")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Data validation error: {e}")
            return False
    
    def get_min_periods(self) -> int:
        """Get minimum number of periods required for calculation"""
        return 14  # Default minimum
    
    def get_required_columns(self) -> List[str]:
        """Get required DataFrame columns"""
        return ['close']  # Default requirement
    
    def safe_calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Safe calculation wrapper with error handling
        
        Args:
            data: Input DataFrame
            
        Returns:
            IndicatorResult (may contain error information)
        """
        try:
            if not self.validate_data(data):
                return IndicatorResult(
                    value=None,
                    signal="ERROR",
                    confidence=0.0,
                    metadata={"error": "Data validation failed"}
                )
            
            return self.calculate(data)
            
        except Exception as e:
            logger.error(f"{self.name}: Calculation error: {e}")
            return IndicatorResult(
                value=None,
                signal="ERROR", 
                confidence=0.0,
                metadata={"error": str(e)}
            )

class IndicatorUtils:
    """Utility functions for indicator calculations"""
    
    @staticmethod
    def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
        """
        Auto-detect OHLCV column names from DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping standard names to actual column names
        """
        column_mapping = {}
        
        # Standard patterns to look for
        patterns = {
            'open': ['open', 'Open', 'OPEN', 'o'],
            'high': ['high', 'High', 'HIGH', 'h'], 
            'low': ['low', 'Low', 'LOW', 'l'],
            'close': ['close', 'Close', 'CLOSE', 'c'],
            'volume': ['volume', 'Volume', 'VOLUME', 'vol', 'Vol']
        }
        
        for standard_name, variations in patterns.items():
            for variation in variations:
                if variation in df.columns:
                    column_mapping[standard_name] = variation
                    break
            
            # If not found, try substring matching
            if standard_name not in column_mapping:
                for col in df.columns:
                    if standard_name.lower() in col.lower():
                        column_mapping[standard_name] = col
                        break
        
        # Fallback to positional mapping if needed
        if not column_mapping and len(df.columns) >= 4:
            logger.warning("Using positional column mapping (assuming OHLCV order)")
            column_mapping = {
                'open': df.columns[0],
                'high': df.columns[1],
                'low': df.columns[2], 
                'close': df.columns[3],
                'volume': df.columns[4] if len(df.columns) > 4 else None
            }
        
        return column_mapping
    
    @staticmethod
    def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame to standard OHLCV format
        
        Args:
            df: Input DataFrame with various column formats
            
        Returns:
            DataFrame with standardized columns [open, high, low, close, volume]
        """
        try:
            column_mapping = IndicatorUtils.detect_columns(df)
            
            normalized_df = pd.DataFrame()
            
            # Map columns to standard names
            for standard_name in ['open', 'high', 'low', 'close', 'volume']:
                if standard_name in column_mapping and column_mapping[standard_name]:
                    normalized_df[standard_name] = df[column_mapping[standard_name]]
                elif standard_name == 'volume':
                    # Default volume if not available
                    normalized_df['volume'] = 100000.0
                else:
                    logger.error(f"Required column {standard_name} not found")
                    return None
            
            # Ensure numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in normalized_df:
                    normalized_df[col] = pd.to_numeric(normalized_df[col], errors='coerce')
            
            # Remove rows with NaN values in critical columns
            normalized_df = normalized_df.dropna(subset=['close'])
            
            logger.debug(f"Data normalized: {len(normalized_df)} rows, columns: {list(normalized_df.columns)}")
            
            return normalized_df
            
        except Exception as e:
            logger.error(f"Data normalization error: {e}")
            return None
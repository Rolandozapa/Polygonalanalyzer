#!/usr/bin/env python3
"""
Test MACD calculation in IA1 analysis flow
"""
import asyncio
import logging
from enhanced_ohlcv_fetcher import enhanced_ohlcv_fetcher
from advanced_technical_indicators import AdvancedTechnicalIndicators
from data_models import MarketOpportunity
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_macd_in_ia1_flow():
    """Test MACD calculation as it would happen in IA1 analysis"""
    
    # Create test opportunity
    opportunity = MarketOpportunity(
        symbol="BTCUSDT",
        current_price=113000.0,
        volume_24h=1000000.0,
        price_change_24h=2.5,
        volatility=0.05,
        timestamp=datetime.utcnow()
    )
    
    logger.info(f"🔍 Testing MACD calculation flow for {opportunity.symbol}")
    
    # Step 1: Get historical data (same as IA1 analysis)
    logger.info("📊 Fetching OHLCV data...")
    historical_data = await enhanced_ohlcv_fetcher.get_enhanced_ohlcv_data(opportunity.symbol)
    
    if historical_data is None or len(historical_data) < 20:
        logger.error(f"❌ Insufficient data: {len(historical_data) if historical_data is not None else 0}")
        return
    
    logger.info(f"✅ Got {len(historical_data)} days of OHLCV data")
    logger.info(f"Data columns: {historical_data.columns.tolist()}")
    logger.info(f"Data sample:\n{historical_data.head()}")
    
    # Step 2: Calculate advanced technical indicators (same as IA1 analysis)
    logger.info("📈 Calculating advanced technical indicators...")
    advanced_indicators = AdvancedTechnicalIndicators()
    df_with_indicators = advanced_indicators.calculate_all_indicators(historical_data)
    indicators = advanced_indicators.get_current_indicators(df_with_indicators)
    
    # Step 3: Check MACD values (same as IA1 analysis)
    logger.info(f"🔍 MACD Results from Advanced Indicators:")
    logger.info(f"   MACD Line: {indicators.macd_line:.8f}")
    logger.info(f"   MACD Signal: {indicators.macd_signal:.8f}")
    logger.info(f"   MACD Histogram: {indicators.macd_histogram:.8f}")
    
    # Step 4: Check if MACD columns exist in DataFrame
    macd_columns = [col for col in df_with_indicators.columns if 'macd' in col.lower()]
    logger.info(f"MACD columns in DataFrame: {macd_columns}")
    
    if macd_columns:
        last_row = df_with_indicators.iloc[-1]
        for col in macd_columns:
            logger.info(f"  {col}: {last_row[col]:.8f}")
    
    # Step 5: Simulate analysis_data update
    logger.info("📋 Simulating analysis_data update (as in IA1 analysis):")
    analysis_data = {
        "symbol": opportunity.symbol,
        "macd_signal": indicators.macd_signal,
        "macd_line": indicators.macd_line,
        "macd_histogram": indicators.macd_histogram,
        "macd_trend": ("bullish" if indicators.macd_histogram > 0 else "bearish" if indicators.macd_histogram < 0 else "neutral"),
    }
    
    logger.info(f"Analysis data MACD: {analysis_data}")
    
    # Check if values are zero
    if (indicators.macd_signal == 0.0 and indicators.macd_line == 0.0 and indicators.macd_histogram == 0.0):
        logger.error("❌ ALL MACD VALUES ARE ZERO - This is the bug!")
        logger.info("🔍 Investigating why Advanced Technical Indicators returns zeros...")
        
        # Check data quality
        logger.info(f"Data quality check:")
        logger.info(f"  Close prices: min={historical_data['Close'].min():.2f}, max={historical_data['Close'].max():.2f}")
        logger.info(f"  Data length: {len(historical_data)}")
        logger.info(f"  Any NaN values: {historical_data.isnull().any().any()}")
    else:
        logger.info("✅ MACD values are non-zero - calculation working!")

if __name__ == "__main__":
    asyncio.run(test_macd_in_ia1_flow())
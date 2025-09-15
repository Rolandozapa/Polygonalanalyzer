#!/usr/bin/env python3
"""
🎯 TEST ENHANCED MARKET CONDITION SCORING SYSTEM
Demonstrate the sophisticated market analysis scoring for IA1 confidence enhancement.
"""
import sys
import asyncio
import logging

# Add backend to path
sys.path.append('/app/backend')

from enhanced_market_scoring import enhanced_market_scoring
from data_models import MarketOpportunity

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_opportunity(symbol: str, current_price: float, price_change_24h: float, 
                          volume_24h: float = 1000000, market_cap: float = None) -> MarketOpportunity:
    """Create a test market opportunity"""
    if market_cap is None:
        market_cap = current_price * 1_000_000  # Estimate
    
    return MarketOpportunity(
        symbol=symbol,
        current_price=current_price,
        price_change_24h=price_change_24h,
        volume_24h=volume_24h,
        market_cap=market_cap,
        volatility=abs(price_change_24h) / 100.0,
        data_confidence=0.9,
        source="test",
        last_updated="2025-09-15T20:00:00Z"
    )

def create_test_analysis(confidence: float, trend_strength: float = 75.0):
    """Create a test analysis object"""
    return type('TestAnalysis', (), {
        'trend_strength_score': trend_strength,
        'analysis_confidence': confidence
    })()

async def test_enhanced_scoring_scenarios():
    """Test various market scenarios with enhanced scoring"""
    logger.info("🚀 Starting Enhanced Market Condition Scoring Tests")
    
    test_scenarios = [
        {
            'name': 'BTC Bull Market - Large Cap',
            'symbol': 'BTCUSDT',
            'price': 115000.0,
            'price_change': 8.5,  # Strong positive momentum
            'volume': 50_000_000_000,
            'market_cap': 2_200_000_000_000,  # $2.2T (mega cap)
            'base_confidence': 0.72,
            'trend_strength': 85.0
        },
        {
            'name': 'ETH Correction - Large Cap',
            'symbol': 'ETHUSDT', 
            'price': 3200.0,
            'price_change': -4.2,  # Bearish momentum
            'volume': 15_000_000_000,
            'market_cap': 400_000_000_000,  # $400B (large cap)
            'base_confidence': 0.68,
            'trend_strength': 45.0
        },
        {
            'name': 'SOL High Volatility - Mid Cap',
            'symbol': 'SOLUSDT',
            'price': 145.0,
            'price_change': 12.3,  # Very high volatility
            'volume': 2_000_000_000,
            'market_cap': 65_000_000_000,  # $65B (mid cap)
            'base_confidence': 0.78,
            'trend_strength': 92.0
        },
        {
            'name': 'PEPE Meme Pump - Small Cap',
            'symbol': 'PEPEUSDT',
            'price': 0.000012,
            'price_change': 25.8,  # Extreme pump
            'volume': 800_000_000,
            'market_cap': 5_000_000_000,  # $5B (small cap)
            'base_confidence': 0.65,
            'trend_strength': 95.0
        },
        {
            'name': 'Unknown Altcoin - Micro Cap',
            'symbol': 'XYZUSDT',
            'price': 0.024,
            'price_change': 3.1,  # Moderate movement
            'volume': 50_000_000,
            'market_cap': 80_000_000,  # $80M (micro cap)
            'base_confidence': 0.71,
            'trend_strength': 60.0
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        logger.info(f"\n🎯 Testing: {scenario['name']}")
        logger.info("=" * 60)
        
        # Create test opportunity
        opportunity = create_test_opportunity(
            symbol=scenario['symbol'],
            current_price=scenario['price'],
            price_change_24h=scenario['price_change'],
            volume_24h=scenario['volume'],
            market_cap=scenario['market_cap']
        )
        
        # Create test analysis
        analysis = create_test_analysis(
            confidence=scenario['base_confidence'],
            trend_strength=scenario['trend_strength']
        )
        
        # Apply enhanced scoring
        scoring_result = enhanced_market_scoring.enhance_ia1_confidence(
            base_confidence=scenario['base_confidence'],
            opportunity=opportunity,
            analysis=analysis
        )
        
        # Create market summary
        market_summary = enhanced_market_scoring.create_market_condition_summary(scoring_result)
        
        # Log results
        logger.info(f"💰 Price: ${scenario['price']:,.6f} ({scenario['price_change']:+.1f}%)")
        logger.info(f"💧 Volume: ${scenario['volume']:,.0f}")
        logger.info(f"🏦 Market Cap: ${scenario['market_cap']:,.0f}")
        logger.info(f"🧠 Base Confidence: {scenario['base_confidence']:.1%}")
        logger.info(f"📊 Enhanced Confidence: {scoring_result['enhanced_confidence']:.1%}")
        logger.info(f"📈 Improvement: {scoring_result['improvement']:+.1f} points")
        logger.info(f"🏆 Market Cap Bucket: {scoring_result['mc_bucket']}")
        logger.info(market_summary)
        
        # Store results for summary
        results.append({
            'name': scenario['name'],
            'symbol': scenario['symbol'],
            'base_confidence': scenario['base_confidence'],
            'enhanced_confidence': scoring_result['enhanced_confidence'],
            'improvement': scoring_result['improvement'],
            'mc_bucket': scoring_result['mc_bucket']
        })
    
    # Print comprehensive summary
    logger.info(f"\n📋 ENHANCED SCORING RESULTS SUMMARY")
    logger.info("=" * 80)
    
    for result in results:
        improvement_emoji = "🚀" if result['improvement'] > 5 else "📈" if result['improvement'] > 0 else "📉" if result['improvement'] < -5 else "➡️"
        logger.info(f"{improvement_emoji} {result['name']:<30} | "
                   f"Base: {result['base_confidence']:5.1%} → "
                   f"Enhanced: {result['enhanced_confidence']:5.1%} "
                   f"({result['improvement']:+4.1f}pts) | "
                   f"{result['mc_bucket'].title()}")
    
    # Calculate statistics
    improvements = [r['improvement'] for r in results]
    avg_improvement = sum(improvements) / len(improvements)
    positive_improvements = sum(1 for i in improvements if i > 0)
    
    logger.info(f"\n🎯 STATISTICS:")
    logger.info(f"   Average Improvement: {avg_improvement:+.1f} points")
    logger.info(f"   Positive Enhancements: {positive_improvements}/{len(results)} ({positive_improvements/len(results)*100:.1f}%)")
    logger.info(f"   Range: {min(improvements):+.1f} to {max(improvements):+.1f} points")

async def test_market_factors_extraction():
    """Test the market factors extraction logic"""
    logger.info(f"\n🔍 Testing Market Factors Extraction")
    logger.info("=" * 60)
    
    # Create a test opportunity
    opportunity = create_test_opportunity(
        symbol='TESTUSDT',
        current_price=1.50,
        price_change_24h=7.2,
        volume_24h=5_000_000,
        market_cap=150_000_000
    )
    
    analysis = create_test_analysis(confidence=0.75, trend_strength=80.0)
    
    # Extract market factors
    factors = enhanced_market_scoring.extract_market_factors(opportunity, analysis)
    
    logger.info(f"📊 Extracted Market Factors:")
    for factor, value in factors.items():
        logger.info(f"   • {factor}: {value:.4f}")
    
    # Test normalization functions
    norm_funcs = enhanced_market_scoring.create_normalization_functions()
    logger.info(f"\n🔄 Normalized Factors:")
    for factor, value in factors.items():
        if factor in norm_funcs:
            normalized = norm_funcs[factor](value)
            logger.info(f"   • {factor}: {value:.4f} → {normalized:.4f}")

if __name__ == "__main__":
    asyncio.run(test_enhanced_scoring_scenarios())
    asyncio.run(test_market_factors_extraction())
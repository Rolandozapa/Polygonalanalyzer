#!/usr/bin/env python3
"""
Direct IA2 Test to Trigger String Indices Error
"""

import asyncio
import sys
import os
sys.path.append('/app/backend')

from pymongo import MongoClient
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_direct_ia2_execution():
    """Test IA2 execution directly"""
    try:
        # Import server components
        from server import get_ia2_chat, db
        from data_models import TechnicalAnalysis, MarketOpportunity
        
        logger.info("🎯 Testing direct IA2 execution to trigger string indices error")
        
        # Get a recent high-confidence IA1 analysis from database
        mongo_client = MongoClient("mongodb://localhost:27017")
        database = mongo_client["myapp"]
        
        # Find high confidence IA1 analysis that should trigger IA2
        high_conf_analysis = database.technical_analyses.find_one({
            "analysis_confidence": {"$gte": 0.7},
            "ia1_signal": {"$in": ["long", "short"]}
        }, sort=[("timestamp", -1)])
        
        if not high_conf_analysis:
            logger.error("❌ No high confidence IA1 analysis found")
            return False
        
        logger.info(f"✅ Found high confidence analysis: {high_conf_analysis['symbol']} - {high_conf_analysis['analysis_confidence']:.1%} confidence, {high_conf_analysis['ia1_signal']} signal")
        
        # Create TechnicalAnalysis object
        analysis = TechnicalAnalysis(**high_conf_analysis)
        
        # Get IA2 chat instance
        ia2_chat = get_ia2_chat()
        logger.info("✅ IA2 chat instance created")
        
        # Create a mock opportunity for IA2
        opportunity = MarketOpportunity(
            symbol=analysis.symbol,
            current_price=analysis.entry_price or 100.0,
            price_change_24h=2.5,
            volume_24h=1000000,
            market_cap=1000000000,
            volatility=0.05,
            data_sources=["test"]
        )
        
        logger.info("🚀 Attempting to trigger IA2 make_decision method...")
        
        # Import the IA2 class and try to trigger the error
        from server import UltraProfessionalOrchestrator
        
        # Create orchestrator instance
        orchestrator = UltraProfessionalOrchestrator()
        await orchestrator.initialize()
        
        logger.info("✅ Orchestrator initialized")
        
        # Try to trigger IA2 decision making
        logger.info("🎯 Calling IA2 make_decision method...")
        
        # Create perf_stats as required by make_decision
        perf_stats = {"api_calls": 0, "success_rate": 0.8, "avg_response_time": 0.5}
        
        try:
            decision = await orchestrator.ia2.make_decision(opportunity, analysis, perf_stats)
            logger.info(f"✅ IA2 decision completed: {decision.signal if decision else 'None'}")
            return True
            
        except Exception as ia2_error:
            logger.error(f"🚨 IA2 ERROR CAUGHT: {str(ia2_error)}")
            logger.error(f"🚨 ERROR TYPE: {type(ia2_error).__name__}")
            
            # Check if this is the string indices error
            if "string indices must be integers" in str(ia2_error).lower():
                logger.error("🎯 STRING INDICES ERROR DETECTED!")
                logger.error(f"🎯 Full error: {str(ia2_error)}")
                
                # Try to get traceback
                import traceback
                logger.error("🎯 TRACEBACK:")
                logger.error(traceback.format_exc())
                
                return True  # Success in finding the error
            else:
                logger.error(f"🔍 Different error found: {str(ia2_error)}")
                return False
        
    except Exception as e:
        logger.error(f"❌ Test setup error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def main():
    """Main test function"""
    logger.info("🚀 Starting Direct IA2 String Indices Error Test")
    
    success = await test_direct_ia2_execution()
    
    if success:
        logger.info("✅ Test completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Test failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
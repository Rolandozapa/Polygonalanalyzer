#!/usr/bin/env python3

import asyncio
import sys
import os
sys.path.append('/app')

from backend_test import DualAITradingBotTester

async def main():
    """Test decision cache clearing and fresh IA2 decision generation"""
    tester = DualAITradingBotTester()
    
    print("🗑️ TESTING DECISION CACHE CLEARING AND FRESH IA2 GENERATION")
    print("=" * 80)
    
    # Run the comprehensive cache clearing and fresh generation tests
    status, results = await tester.run_decision_cache_and_fresh_generation_tests()
    
    print(f"\n🎯 FINAL RESULT: {status}")
    print(f"📊 Results: {results}")
    
    return status == "SUCCESS"

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
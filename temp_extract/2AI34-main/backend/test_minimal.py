#!/usr/bin/env python3
"""
Test minimal pour identifier la source du problème multiprocessing
"""
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_basic_imports():
    """Test des imports de base"""
    try:
        logger.info("Testing basic imports...")
        
        # Test FastAPI
        from fastapi import FastAPI
        logger.info("✅ FastAPI OK")
        
        # Test pandas/numpy
        import pandas as pd
        import numpy as np
        logger.info("✅ pandas/numpy OK")
        
        # Test asyncio
        await asyncio.sleep(0.1)
        logger.info("✅ AsyncIO OK")
        
        logger.info("🎯 Basic imports test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error in basic imports: {e}")

if __name__ == "__main__":
    asyncio.run(test_basic_imports())
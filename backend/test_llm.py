#!/usr/bin/env python3
"""
Test LLM import pour voir si c'est la source du multiprocessing
"""
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_llm_import():
    """Test LLM import spécifiquement"""
    try:
        logger.info("Testing LLM import...")
        
        # Test emergentintegrations
        from emergentintegrations.llm.chat import LlmChat, UserMessage
        logger.info("✅ emergentintegrations import OK")
        
        # Test d'initialisation LLM
        chat = LlmChat.gpt4o()
        logger.info("✅ LLM initialization OK")
        
        logger.info("🎯 LLM test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error in LLM test: {e}")

if __name__ == "__main__":
    asyncio.run(test_llm_import())